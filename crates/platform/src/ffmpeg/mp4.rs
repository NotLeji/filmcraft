//! Minimal fragmented-MP4 reader for the ffmpeg child process.
//!
//! ffmpeg runs with `-f mp4 -movflags frag_keyframe+empty_moov+default_base_moof`, so its
//! stdout is: `ftyp`, `moov` (carrying the `avcC` / `hvcC` sample entry and the track
//! timescale we asked for), then one `moof` + `mdat` pair per GOP. The fragment sample tables
//! (`trun`) give exact per-sample sizes, sync flags, durations and composition offsets, so
//! B-frames stay correct without any timestamp guessing, and the `mdat` payloads are already
//! length-prefixed NAL units, exactly what the MP4 muxer wants.
//!
//! Only the boxes ffmpeg writes for this output are understood; anything else is skipped
//! (top level) or a clear error (inside `moof` / `moov`), never a panic.

use std::io::{BufReader, Read};

use filmcraft_export::{EncodedPacket, ExportError, Result};
use filmcraft_isobmff::{AvcConfig, HevcConfig, SampleEntry};

/// Sanity caps: a corrupt box size must not balloon an allocation.
const MAX_MOOV: u64 = 64 << 20;
const MAX_MOOF: u64 = 64 << 20;
const MAX_MDAT: u64 = 1 << 30;

/// The parsed init segment.
pub struct InitInfo {
    /// MP4 sample entry from the `avcC` / `hvcC` box.
    pub entry: SampleEntry,
    /// Whether the stream is HEVC (else H.264).
    pub hevc: bool,
}

struct BoxHeader {
    typ: [u8; 4],
    /// Total box size including the header.
    size: u64,
    /// Header length in bytes: 8, or 16 with a 64-bit largesize.
    header_len: u64,
    /// Stream offset of the box's first byte.
    start: u64,
}

/// Defaults from `trex`, for sample fields a `trun` / `tfhd` does not carry.
#[derive(Clone, Copy, Default)]
struct Trex {
    duration: u32,
    size: u32,
    flags: u32,
}

/// The streaming reader: init segment first, then one GOP per [`FragReader::next_fragment`].
pub struct FragReader<R: Read> {
    inner: BufReader<R>,
    /// Stream offset of the next byte to read.
    pos: u64,
    trex: Trex,
    /// Composition offset of the first sample of the first fragment (the B-frame delay, for
    /// the edit list); `None` until the first fragment is parsed.
    first_cto: Option<i64>,
}

impl<R: Read> FragReader<R> {
    pub fn new(r: R) -> Self {
        FragReader { inner: BufReader::with_capacity(64 << 10, r), pos: 0, trex: Trex::default(), first_cto: None }
    }

    /// Composition offset of the first sample of the first fragment, once parsed.
    pub fn first_cto(&self) -> Option<i64> {
        self.first_cto
    }

    fn read_exact_vec(&mut self, n: u64, cap: u64, what: &str) -> Result<Vec<u8>> {
        if n > cap {
            return Err(ExportError::Encode(format!("ffmpeg: {what} box of {n} bytes exceeds the {cap}-byte cap")));
        }
        let n = usize::try_from(n).map_err(|_| ExportError::Encode(format!("ffmpeg: {what} box size {n} does not fit")))?;
        let mut b = vec![0u8; n];
        self.inner.read_exact(&mut b).map_err(|e| ExportError::Encode(format!("ffmpeg: truncated {what}: {e}")))?;
        self.pos += n as u64;
        Ok(b)
    }

    /// The next top-level box header, or `None` on a clean EOF at a box boundary.
    fn read_box_header(&mut self) -> Result<Option<BoxHeader>> {
        let start = self.pos;
        let mut first = [0u8; 1];
        let n = self.inner.read(&mut first).map_err(|e| ExportError::Encode(format!("ffmpeg: stdout read: {e}")))?;
        if n == 0 {
            return Ok(None);
        }
        let mut h = [0u8; 8];
        h[0] = first[0];
        self.inner.read_exact(&mut h[1..]).map_err(|e| ExportError::Encode(format!("ffmpeg: truncated box header: {e}")))?;
        self.pos += 8;
        let mut size = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as u64;
        let typ: [u8; 4] = [h[4], h[5], h[6], h[7]];
        let mut header_len = 8u64;
        if size == 1 {
            let mut x = [0u8; 8];
            self.inner.read_exact(&mut x).map_err(|e| ExportError::Encode(format!("ffmpeg: truncated large box size: {e}")))?;
            self.pos += 8;
            size = u64::from_be_bytes(x);
            header_len = 16;
        }
        if size < header_len {
            return Err(ExportError::Encode(format!("ffmpeg: box {} smaller than its header", typ_str(&typ))));
        }
        Ok(Some(BoxHeader { typ, size, header_len, start }))
    }

    fn read_box_payload(&mut self, h: &BoxHeader, cap: u64, what: &str) -> Result<Vec<u8>> {
        self.read_exact_vec(h.size - h.header_len, cap, what)
    }

    fn skip_box(&mut self, h: &BoxHeader, cap: u64) -> Result<()> {
        self.read_box_payload(h, cap, &format!("{} payload", typ_str(&h.typ))).map(|_| ())
    }

    /// Read boxes until the init segment (`moov`) is parsed: the sample entry, the codec, and
    /// confirmation that the track timescale is the one we asked ffmpeg for.
    pub fn read_init(&mut self, w: u16, h: u16, timescale: u32) -> Result<InitInfo> {
        loop {
            let Some(header) = self.read_box_header()? else {
                return Err(ExportError::Encode("ffmpeg: end of stream before the MP4 init segment".into()));
            };
            match &header.typ {
                b"moov" => {
                    let payload = self.read_box_payload(&header, MAX_MOOV, "moov")?;
                    return parse_moov(&payload, w, h, timescale, &mut self.trex);
                }
                // `ftyp`, a stray `sidx`: not needed, but skipping keeps the reader robust
                _ => self.skip_box(&header, MAX_MOOV)?,
            }
        }
    }

    /// The next GOP as packets, or `None` on a clean EOF (after `flush` drained the child).
    pub fn next_fragment(&mut self) -> Result<Option<Vec<EncodedPacket>>> {
        loop {
            let Some(header) = self.read_box_header()? else { return Ok(None) };
            if &header.typ != b"moof" {
                self.skip_box(&header, MAX_MOOF)?;
                continue;
            }
            let moof_start = header.start;
            let moof = self.read_box_payload(&header, MAX_MOOF, "moof")?;
            let samples = parse_moof(&moof, moof_start, &self.trex)?;
            // the media data follows its fragment header
            let Some(mdat) = self.read_box_header()? else {
                return Err(ExportError::Encode("ffmpeg: end of stream between moof and mdat".into()));
            };
            if &mdat.typ != b"mdat" {
                return Err(ExportError::Encode(format!("ffmpeg: expected mdat after moof, found {}", typ_str(&mdat.typ))));
            }
            let mdat_payload_start = mdat.start + mdat.header_len;
            let mdat_payload = self.read_box_payload(&mdat, MAX_MDAT, "mdat")?;
            let mut packets = Vec::with_capacity(samples.len());
            for s in &samples {
                let begin = s
                    .file_offset
                    .checked_sub(mdat_payload_start)
                    .and_then(|o| usize::try_from(o).ok())
                    .filter(|o| o.checked_add(s.size).is_some_and(|end| end <= mdat_payload.len()))
                    .ok_or_else(|| ExportError::Encode("ffmpeg: sample offset outside its mdat".into()))?;
                packets.push(EncodedPacket { data: mdat_payload[begin..begin + s.size].to_vec(), key: s.key, duration: s.duration, composition_offset: s.cto });
            }
            if self.first_cto.is_none() {
                self.first_cto = samples.first().map(|s| i64::from(s.cto));
            }
            return Ok(Some(packets));
        }
    }
}

fn typ_str(t: &[u8; 4]) -> String {
    String::from_utf8_lossy(t).into_owned()
}

/// Cursor over an in-memory box payload.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cur { b, p: 0 }
    }
    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.p)
    }
    fn u8(&mut self) -> Result<u8> {
        self.bytes(1).map(|b| b[0])
    }
    fn u32(&mut self) -> Result<u32> {
        self.bytes(4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        self.bytes(8).map(|b| u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn i32(&mut self) -> Result<i32> {
        self.u32().map(|v| v as i32)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(ExportError::Encode("ffmpeg: truncated box payload".into()));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }
    /// The child boxes of this payload: (type, payload).
    fn children(&mut self) -> Result<Vec<([u8; 4], &'a [u8])>> {
        let mut out = Vec::new();
        while self.remaining() > 0 {
            if self.remaining() < 8 {
                return Err(ExportError::Encode("ffmpeg: truncated child box header".into()));
            }
            let mut size = u64::from(self.u32()?);
            let typ: [u8; 4] = self.bytes(4)?.try_into().map_err(|_| ExportError::Encode("ffmpeg: box type".into()))?;
            if size == 1 {
                size = self.u64()?;
            }
            if size < 8 {
                return Err(ExportError::Encode(format!("ffmpeg: child box {} smaller than its header", typ_str(&typ))));
            }
            let payload = self.bytes((size - 8) as usize)?;
            out.push((typ, payload));
        }
        Ok(out)
    }
    fn child<'b>(boxes: &'b [([u8; 4], &'a [u8])], typ: &[u8; 4]) -> Option<&'b [u8]> {
        boxes.iter().find(|(t, _)| t == typ).map(|(_, p)| *p)
    }
}

fn parse_moov(moov: &[u8], w: u16, h: u16, timescale: u32, trex: &mut Trex) -> Result<InitInfo> {
    let mut cur = Cur::new(moov);
    let mut init: Option<InitInfo> = None;
    for (typ, payload) in cur.children()? {
        match &typ {
            b"trak" => {
                if init.is_none() {
                    init = parse_trak(payload, w, h, timescale)?;
                }
            }
            b"mvex" => {
                let mut c = Cur::new(payload);
                for (t, p) in c.children()? {
                    if &t == b"trex" {
                        let mut tc = Cur::new(p);
                        tc.skip(8)?; // version + flags + track_ID
                        tc.skip(4)?; // default_sample_description_index
                        *trex = Trex { duration: tc.u32()?, size: tc.u32()?, flags: tc.u32()? };
                    }
                }
            }
            _ => {}
        }
    }
    init.ok_or_else(|| ExportError::Encode("ffmpeg: no video track in the MP4 init segment".into()))
}

fn parse_trak(trak: &[u8], w: u16, h: u16, timescale: u32) -> Result<Option<InitInfo>> {
    let mut cur = Cur::new(trak);
    let boxes = cur.children()?;
    let Some(mdia) = Cur::child(&boxes, b"mdia") else { return Ok(None) };
    let mut mdia = Cur::new(mdia);
    let mdia_boxes = mdia.children()?;
    let Some(mdhd) = Cur::child(&mdia_boxes, b"mdhd") else { return Ok(None) };
    let mut c = Cur::new(mdhd);
    let version = c.u8()?;
    c.skip(3)?;
    if version == 1 {
        c.skip(16)?;
    } else {
        c.skip(8)?;
    }
    let ts = c.u32()?;
    if ts != timescale {
        return Err(ExportError::Encode(format!("ffmpeg: track timescale {ts}, expected {timescale}")));
    }
    let Some(hdlr) = Cur::child(&mdia_boxes, b"hdlr") else { return Ok(None) };
    let mut c = Cur::new(hdlr);
    c.skip(8)?;
    let handler: [u8; 4] = c.bytes(4)?.try_into().map_err(|_| ExportError::Encode("ffmpeg: hdlr".into()))?;
    if &handler != b"vide" {
        return Ok(None);
    }
    let Some(minf) = Cur::child(&mdia_boxes, b"minf") else { return Ok(None) };
    let mut minf = Cur::new(minf);
    let Some(stbl) = Cur::child(&minf.children()?, b"stbl") else { return Ok(None) };
    let mut stbl = Cur::new(stbl);
    let Some(stsd) = Cur::child(&stbl.children()?, b"stsd") else { return Ok(None) };
    let mut stsd = Cur::new(stsd);
    stsd.skip(8)?; // version + flags + entry_count
    let entry_boxes = stsd.children()?;
    let (entry_typ, entry) = entry_boxes.first().ok_or_else(|| ExportError::Encode("ffmpeg: empty stsd".into()))?;
    // a visual sample entry is 78 fixed bytes, then its child boxes (avcC / hvcC)
    let mut ec = Cur::new(entry);
    ec.skip(78)?;
    let entry_children = ec.children()?;
    let (entry, hevc) = match entry_typ {
        t if t == b"avc1" || t == b"avc2" => {
            let avcc = Cur::child(&entry_children, b"avcC").ok_or_else(|| ExportError::Encode("ffmpeg: avc1 without avcC".into()))?;
            let cfg = AvcConfig::parse(avcc).map_err(|e| ExportError::Encode(format!("ffmpeg: unreadable avcC: {e:?}")))?;
            (SampleEntry::avc(cfg, w, h), false)
        }
        t if t == b"hev1" || t == b"hvc1" => {
            let hvcc = Cur::child(&entry_children, b"hvcC").ok_or_else(|| ExportError::Encode("ffmpeg: hvc1 without hvcC".into()))?;
            let cfg = HevcConfig::parse(hvcc).map_err(|e| ExportError::Encode(format!("ffmpeg: unreadable hvcC: {e:?}")))?;
            (SampleEntry::hevc(cfg, w, h), true)
        }
        _ => return Err(ExportError::Encode(format!("ffmpeg: unexpected sample entry {}", typ_str(entry_typ)))),
    };
    Ok(Some(InitInfo { entry, hevc }))
}

/// One parsed sample: absolute file offset and length in `mdat`, timing, sync flag.
struct Sample {
    file_offset: u64,
    size: usize,
    key: bool,
    duration: u32,
    cto: i32,
}

fn parse_moof(moof: &[u8], moof_start: u64, trex: &Trex) -> Result<Vec<Sample>> {
    let mut cur = Cur::new(moof);
    let mut samples = Vec::new();
    for (typ, payload) in cur.children()? {
        if &typ == b"traf" {
            samples.extend(parse_traf(payload, moof_start, trex)?);
        }
    }
    if samples.is_empty() {
        return Err(ExportError::Encode("ffmpeg: moof without samples".into()));
    }
    Ok(samples)
}

fn parse_traf(traf: &[u8], moof_start: u64, trex: &Trex) -> Result<Vec<Sample>> {
    let mut cur = Cur::new(traf);
    let boxes = cur.children()?;
    let Some(tfhd) = Cur::child(&boxes, b"tfhd") else {
        return Err(ExportError::Encode("ffmpeg: traf without tfhd".into()));
    };
    let Some(tfdt) = Cur::child(&boxes, b"tfdt") else {
        return Err(ExportError::Encode("ffmpeg: traf without tfdt".into()));
    };
    let Some(trun) = Cur::child(&boxes, b"trun") else {
        return Err(ExportError::Encode("ffmpeg: traf without trun".into()));
    };

    let mut c = Cur::new(tfhd);
    c.skip(1)?; // version
    let flags = (u32::from(c.u8()?) << 16) | (u32::from(c.u8()?) << 8) | u32::from(c.u8()?);
    c.skip(4)?; // track_ID
    let mut base_offset: Option<u64> = None;
    if flags & 0x0000_0001 != 0 {
        base_offset = Some(c.u64()?);
    }
    if flags & 0x0000_0002 != 0 {
        c.skip(4)?; // sample_description_index
    }
    let mut def_duration = trex.duration;
    let mut def_size = trex.size;
    let mut def_flags = trex.flags;
    if flags & 0x0000_0008 != 0 {
        def_duration = c.u32()?;
    }
    if flags & 0x0000_0010 != 0 {
        def_size = c.u32()?;
    }
    if flags & 0x0000_0020 != 0 {
        def_flags = c.u32()?;
    }
    // with `default_base_moof` (what we ask ffmpeg for) sample offsets are relative to the
    // fragment header; an explicit base-data-offset wins where present
    let base = base_offset.unwrap_or(moof_start);

    let mut c = Cur::new(tfdt);
    let _version = c.u8()?;
    c.skip(3)?;
    // decode time is not needed per sample (durations accumulate from it), only that it parses
    if _version == 1 {
        c.u64()?;
    } else {
        c.u32()?;
    }

    let mut c = Cur::new(trun);
    let trun_version = c.u8()?;
    let tflags = (u32::from(c.u8()?) << 16) | (u32::from(c.u8()?) << 8) | u32::from(c.u8()?);
    let sample_count = c.u32()?;
    let mut data_offset: i64 = 0;
    if tflags & 0x0000_0001 != 0 {
        data_offset = i64::from(c.i32()?);
    }
    let mut first_flags: Option<u32> = None;
    if tflags & 0x0000_0004 != 0 {
        first_flags = Some(c.u32()?);
    }
    let mut samples = Vec::with_capacity(sample_count.min(1 << 20) as usize);
    let mut file_offset = i64::try_from(base).map_err(|_| ExportError::Encode("ffmpeg: moof past i64".into()))? + data_offset;
    for i in 0..sample_count {
        let duration = if tflags & 0x0000_0100 != 0 { c.u32()? } else { def_duration };
        if duration == 0 {
            return Err(ExportError::Encode("ffmpeg: sample without duration".into()));
        }
        let size = if tflags & 0x0000_0200 != 0 { c.u32()? } else { def_size };
        let size = usize::try_from(size).map_err(|_| ExportError::Encode("ffmpeg: sample size does not fit".into()))?;
        if size == 0 {
            return Err(ExportError::Encode("ffmpeg: sample without size".into()));
        }
        let sflags = if i == 0 {
            if tflags & 0x0000_0400 != 0 { c.u32()? } else { first_flags.unwrap_or(def_flags) }
        } else if tflags & 0x0000_0400 != 0 {
            c.u32()?
        } else {
            def_flags
        };
        let cto = if tflags & 0x0000_0800 != 0 { if trun_version == 1 { c.i32()? } else { c.u32()? as i32 } } else { 0 };
        if file_offset < 0 {
            return Err(ExportError::Encode("ffmpeg: sample before its moof".into()));
        }
        samples.push(Sample {
            file_offset: file_offset as u64,
            size,
            // a sync sample unless the non-sync bit is set
            key: sflags & 0x0001_0000 == 0,
            duration,
            cto,
        });
        file_offset = file_offset.saturating_add(size as i64);
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal box builder for synthetic streams: size + type + payload.
    fn bx(typ: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + payload.len());
        b.extend_from_slice(&(8 + payload.len() as u32).to_be_bytes());
        b.extend_from_slice(typ);
        b.extend_from_slice(payload);
        b
    }

    /// A minimal avcC (the parser only stores the parameter sets).
    fn avcc() -> Vec<u8> {
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0xab, 0x40];
        let pps = [0x68u8, 0xce, 0x06, 0xe2];
        let mut b = vec![0x01, sps[1], sps[2], sps[3], 0xff, 0xe1];
        b.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        b.extend_from_slice(&sps);
        b.push(0x01);
        b.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        b.extend_from_slice(&pps);
        b
    }

    /// moov with one video track: mdhd timescale, avc1/avcC, trex defaults.
    fn moov(timescale: u32) -> Vec<u8> {
        let mdhd = {
            let mut p = vec![0u8; 4]; // version + flags
            p.extend_from_slice(&0u32.to_be_bytes()); // creation
            p.extend_from_slice(&0u32.to_be_bytes()); // modification
            p.extend_from_slice(&timescale.to_be_bytes());
            p.extend_from_slice(&0u32.to_be_bytes()); // duration
            p.extend_from_slice(&0x55c4u16.to_be_bytes()); // language
            p.extend_from_slice(&0u16.to_be_bytes()); // quality
            bx(b"mdhd", &p)
        };
        let mut hdlr_p = vec![0u8; 8];
        hdlr_p.extend_from_slice(b"vide");
        hdlr_p.extend_from_slice(b"VideoHandler\0");
        let hdlr = bx(b"hdlr", &hdlr_p);
        let avc1 = {
            let mut p = vec![0u8; 78]; // visual sample entry fixed fields
            p.extend_from_slice(&bx(b"avcC", &avcc()));
            bx(b"avc1", &p)
        };
        let mut stsd_p = vec![0u8; 8]; // version + flags + entry_count
        stsd_p[7] = 1;
        stsd_p.extend_from_slice(&avc1);
        let stbl = bx(b"stbl", &bx(b"stsd", &stsd_p));
        let minf = bx(b"minf", &stbl);
        let mut mdia_p = Vec::new();
        mdia_p.extend_from_slice(&mdhd);
        mdia_p.extend_from_slice(&hdlr);
        mdia_p.extend_from_slice(&minf);
        let mdia = bx(b"mdia", &mdia_p);
        let trak = bx(b"trak", &mdia);
        let mut trex_p = vec![0u8; 4];
        trex_p.extend_from_slice(&1u32.to_be_bytes()); // track_ID
        trex_p.extend_from_slice(&1u32.to_be_bytes()); // desc index
        trex_p.extend_from_slice(&1u32.to_be_bytes()); // default duration
        trex_p.extend_from_slice(&0u32.to_be_bytes()); // default size
        trex_p.extend_from_slice(&0u32.to_be_bytes()); // default flags
        let mvex = bx(b"mvex", &bx(b"trex", &trex_p));
        let mut moov_p = Vec::new();
        moov_p.extend_from_slice(&trak);
        moov_p.extend_from_slice(&mvex);
        bx(b"moov", &moov_p)
    }

    /// moof + mdat for two samples: an IDR (sync) then a P frame (non-sync), durations of 1,
    /// composition offsets 2 and 0 (one frame of reordering delay).
    fn fragment() -> (Vec<u8>, Vec<u8>, usize) {
        // tfhd: version 0, flags default-base-is-moof | default-sample-duration
        let mut tfhd_p = vec![0u8, 0x02, 0x00, 0x08];
        tfhd_p.extend_from_slice(&1u32.to_be_bytes()); // track_ID
        tfhd_p.extend_from_slice(&1u32.to_be_bytes()); // default duration
        let tfhd = bx(b"tfhd", &tfhd_p);
        // tfdt version 0, base 0
        let mut tfdt_p = vec![0u8, 0, 0, 0];
        tfdt_p.extend_from_slice(&0u32.to_be_bytes());
        let tfdt = bx(b"tfdt", &tfdt_p);
        // trun v1: data-offset | sample-duration | sample-size | sample-flags | cto
        let mut trun_p = vec![1u8, 0x00, 0x0f, 0x01];
        trun_p.extend_from_slice(&2u32.to_be_bytes()); // sample_count
        let data_offset_at = trun_p.len();
        trun_p.extend_from_slice(&0u32.to_be_bytes()); // data_offset (patched below)
        // sample 0: dur 1, size 4, flags sync, cto 2
        trun_p.extend_from_slice(&1u32.to_be_bytes());
        trun_p.extend_from_slice(&4u32.to_be_bytes());
        trun_p.extend_from_slice(&0x0200_0000u32.to_be_bytes());
        trun_p.extend_from_slice(&2i32.to_be_bytes());
        // sample 1: dur 1, size 4, flags non-sync (bit 16), cto 0
        trun_p.extend_from_slice(&1u32.to_be_bytes());
        trun_p.extend_from_slice(&4u32.to_be_bytes());
        trun_p.extend_from_slice(&0x0101_0000u32.to_be_bytes());
        trun_p.extend_from_slice(&0i32.to_be_bytes());
        let trun = bx(b"trun", &trun_p);
        let mut traf_p = Vec::new();
        traf_p.extend_from_slice(&tfhd);
        traf_p.extend_from_slice(&tfdt);
        traf_p.extend_from_slice(&trun);
        let traf = bx(b"traf", &traf_p);
        let mut mfhd_p = vec![0u8; 4];
        mfhd_p.extend_from_slice(&1u32.to_be_bytes());
        let mut moof_p = Vec::new();
        moof_p.extend_from_slice(&bx(b"mfhd", &mfhd_p));
        moof_p.extend_from_slice(&traf);
        let mut moof = bx(b"moof", &moof_p);
        // data_offset is relative to the moof start (default-base-is-moof): patch it now that
        // the moof size is known. Its position: moof header + mfhd box + traf header + tfhd
        // box + tfdt box + trun header + version/flags + sample_count.
        let at = 8 + mfhd_len() + 8 + tfhd.len() + tfdt.len() + 8 + 8;
        let data_offset = (moof.len() + 8) as u32; // + mdat header
        moof[at..at + 4].copy_from_slice(&data_offset.to_be_bytes());
        let mdat = bx(b"mdat", &[0x00, 0x00, 0x00, 0x04, 0x11, 0x22, 0x33, 0x44]);
        return (moof, mdat, data_offset_at);
        fn mfhd_len() -> usize {
            16
        }
    }

    #[test]
    fn init_segment_yields_an_avc_sample_entry() {
        let ftyp = bx(b"ftyp", b"isom");
        let mut stream = Vec::new();
        stream.extend_from_slice(&ftyp);
        stream.extend_from_slice(&moov(30));
        let mut r = FragReader::new(&stream[..]);
        let init = r.read_init(64, 64, 30).unwrap();
        assert!(!init.hevc);
        // a bogus timescale is rejected: timestamps would silently go wrong
        let mut r = FragReader::new(&stream[..]);
        assert!(r.read_init(64, 64, 25).is_err());
    }

    #[test]
    fn fragment_samples_come_out_with_exact_timing_and_sync_flags() {
        let ftyp = bx(b"ftyp", b"isom");
        let (moof, mdat, _) = fragment();
        let mut stream = Vec::new();
        stream.extend_from_slice(&ftyp);
        stream.extend_from_slice(&moov(30));
        stream.extend_from_slice(&moof);
        stream.extend_from_slice(&mdat);
        let mut r = FragReader::new(&stream[..]);
        r.read_init(64, 64, 30).unwrap();
        let packets = r.next_fragment().unwrap().unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!((packets[0].key, packets[0].duration, packets[0].composition_offset), (true, 1, 2));
        assert_eq!((packets[1].key, packets[1].duration, packets[1].composition_offset), (false, 1, 0));
        assert_eq!(packets[0].data, [0x00, 0x00, 0x00, 0x04]);
        assert_eq!(packets[1].data, [0x11, 0x22, 0x33, 0x44]);
        // the edit-list offset is the first sample's composition offset
        assert_eq!(r.first_cto(), Some(2));
        // clean EOF afterwards
        assert!(r.next_fragment().unwrap().is_none());
    }

    #[test]
    fn truncated_streams_are_errors_not_panics() {
        let ftyp = bx(b"ftyp", b"isom");
        let (moof, _) = fragment();
        let mut stream = Vec::new();
        stream.extend_from_slice(&ftyp);
        stream.extend_from_slice(&moov(30));
        stream.extend_from_slice(&moof);
        // mdat missing: truncated
        let mut r = FragReader::new(&stream[..]);
        r.read_init(64, 64, 30).unwrap();
        assert!(r.next_fragment().is_err());
        // garbage instead of boxes
        let mut r = FragReader::new(&b"\xff\xfe\x00garbage-data-here......."[..]);
        assert!(r.read_init(64, 64, 30).is_err());
    }
}
