//! The external-ffmpeg export encoder: [`factory`] is registered behind the native
//! hardware encoders and in front of the built-in software ones
//! (`filmcraft_export::register_encoder`).
//!
//! - H.264: takes an export only when Export ▸ Hardware encoding is Auto, the native
//!   encoder declined it (or there is none), *and* ffmpeg offers a hardware encoder
//!   (QSV/VAAPI/NVENC/AMF/VideoToolbox). When ffmpeg would only use its software encoder
//!   (libx264) the factory returns `None` and the built-in software encoder runs,
//!   exactly as before — no behavior change on machines without HW acceleration.
//! - H.265 (`Format::Hevc`): there is no software encoder, so the Hardware encoding toggle
//!   does not apply and choosing the format is the opt-in. What ffmpeg cannot do (interlaced,
//!   two-pass, non-square pixels, sizes outside the MP4 sample entry, ...) ends the export
//!   with an error that says why; when this machine has no ffmpeg H.265 encoder at all,
//!   [`filmcraft_export::available`] says so and the factory returns `None` (the export's own
//!   "encoder not available" error).
//!
//! Pictures go in as `yuv420p` on ffmpeg's stdin; a fragmented MP4 comes back on its stdout
//! ([`super::mp4`]) with exact sample timing, so B-frames stay correct. 8-bit SDR only: an
//! HDR export is declined, and an HDR sequence is tone-mapped to SDR as before. A missing
//! ffmpeg binary is never a crash: the factory declines and the export falls back (H.264) or
//! reports "encoder not available" (H.265), naming the binary it looked for.
//!
//! The ffmpeg child is supervised: its stdout is drained on a reader thread (no pipe
//! deadlock), its stderr tail is captured for error reports, and a wedged child is killed
//! instead of hanging the export.

use std::io::{BufReader, Read as _, Write as _};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use filmcraft_export::{
    BitrateMode, EncodedPacket, EncoderFrame, ExportError, ExportSettings, FieldOrder, Format, H264Pass, HardwareEncoding, Result, VideoEncoder,
};
use filmcraft_isobmff::SampleEntry;
use filmcraft_time::FrameRate;

use super::mp4::FragReader;
use super::probe::{self, Codec};

/// How long [`open`] waits for ffmpeg's init segment before giving up on a wedged child.
const INIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Rate control, resolved from [`ExportSettings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RateControl {
    Cbr { kbps: u32, max_kbps: u32 },
    Vbr { target_kbps: u32, max_kbps: u32 },
    Cqp { qp: u8 },
}

/// The encoder's resolved configuration, or why the export is not taken.
struct FfmpegConfig {
    encoder: String,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    rate: RateControl,
    keyint: u32,
    /// Whether the picked encoder is hardware (what the export hardware-encoder counters count).
    hw: bool,
}

/// Why ffmpeg does not take this export (the next encoder does, for H.264), or the encoder's
/// configuration.
fn config(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings, encoder: &str) -> std::result::Result<FfmpegConfig, String> {
    let hevc = format == Format::Hevc;
    if !(format == Format::H264 || hevc) || s.format.is_mxf() {
        return Err("only MP4 / MOV H.264 and H.265 exports".into());
    }
    // HEVC has no software encoder: choosing the format is the opt-in, the toggle does not apply
    if !hevc && s.hardware_encoding != HardwareEncoding::Auto {
        return Err("hardware encoding is off".into());
    }
    if s.signal.is_hdr() {
        return Err("HDR (this encoder writes 8-bit SDR)".into());
    }
    if !matches!(s.h264_pass, H264Pass::Single) || s.bitrate_mode == BitrateMode::Vbr2Pass {
        return Err("two-pass VBR".into());
    }
    if s.bitrate_mode == BitrateMode::Crf {
        // the built-in H.264 encoder takes CRF; declining lets it
        return Err("CRF (constant quality)".into());
    }
    if s.pixel_aspect.is_some_and(|(n, d)| n != d) {
        return Err("non-square pixels".into());
    }
    if s.field_order != FieldOrder::Progressive {
        return Err("interlaced output".into());
    }
    if w == 0 || h == 0 || !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        return Err("ffmpeg needs nonzero even dimensions".into());
    }
    if w > u32::from(u16::MAX) || h > u32::from(u16::MAX) {
        return Err(format!("{w}x{h} does not fit an MP4 sample entry"));
    }
    let (Ok(fps_num), Ok(fps_den)) = (u32::try_from(rate.num), u32::try_from(rate.den)) else {
        return Err("frame rate".into());
    };
    if fps_num == 0 || fps_den == 0 {
        return Err("frame rate".into());
    }
    let kbps = s.bitrate_kbps.max(100);
    let max_kbps = s.max_bitrate_kbps.filter(|m| *m >= kbps).unwrap_or_else(|| kbps.saturating_add(kbps / 2));
    let rate = match s.bitrate_mode {
        BitrateMode::Cbr => RateControl::Cbr { kbps, max_kbps },
        BitrateMode::Vbr1Pass => RateControl::Vbr { target_kbps: kbps, max_kbps },
        BitrateMode::Cqp => RateControl::Cqp { qp: s.cqp.min(51) },
        BitrateMode::Vbr2Pass | BitrateMode::Crf => return Err("two-pass VBR / CRF".into()),
    };
    // CQP support is per encoder (VideoToolbox has no constant-quantizer mode): fail the
    // configuration here, with the encoder named, rather than at the first frame
    if let RateControl::Cqp { .. } = rate {
        rate_args(encoder, &rate, &mut Vec::new()).map_err(|why| format!("{encoder}: {why}"))?;
    }
    let keyint = s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (f64::from(fps_num) / f64::from(fps_den) * 2.0).round().max(1.0) as u32);
    Ok(FfmpegConfig { encoder: encoder.to_string(), width: w, height: h, fps_num, fps_den, rate, keyint, hw: probe::is_hardware_encoder(encoder) })
}

/// The ffmpeg arguments for one rate-control mode on one encoder family. Hardware encoders
/// each spell constant quality differently; unknown encoders are an error, not a guess.
fn rate_args(encoder: &str, rate: &RateControl, out: &mut Vec<String>) -> std::result::Result<(), String> {
    if encoder.ends_with("_qsv") {
        match rate {
            RateControl::Cbr { kbps, max_kbps } | RateControl::Vbr { target_kbps: kbps, max_kbps } => {
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            // Intel's constant-quality mode (ICQ): the closest Quick Sync Video has to CQP
            RateControl::Cqp { qp } => out.extend(["-global_quality".to_string(), qp.to_string()]),
        }
    } else if encoder.ends_with("_nvenc") {
        match rate {
            RateControl::Cbr { kbps, max_kbps } => {
                out.extend(["-rc".to_string(), "cbr".to_string()]);
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            RateControl::Vbr { target_kbps: kbps, max_kbps } => {
                out.extend(["-rc".to_string(), "vbr".to_string()]);
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            RateControl::Cqp { qp } => out.extend(["-rc".to_string(), "constqp".to_string(), "-qp".to_string(), qp.to_string()]),
        }
    } else if encoder.ends_with("_vaapi") {
        match rate {
            RateControl::Cbr { kbps, max_kbps } | RateControl::Vbr { target_kbps: kbps, max_kbps } => {
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            RateControl::Cqp { qp } => out.extend(["-qp".to_string(), qp.to_string()]),
        }
    } else if encoder.ends_with("_amf") {
        match rate {
            RateControl::Cbr { kbps, max_kbps } => {
                out.extend(["-rc".to_string(), "cbr".to_string()]);
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            RateControl::Vbr { target_kbps: kbps, max_kbps } => {
                out.extend(["-rc".to_string(), "vbr".to_string()]);
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            RateControl::Cqp { qp } => {
                out.extend(["-rc".to_string(), "cqp".to_string(), "-qp_i".to_string(), qp.to_string(), "-qp_p".to_string(), qp.to_string()]);
            }
        }
    } else if encoder.ends_with("_videotoolbox") {
        match rate {
            RateControl::Cbr { kbps, .. } | RateControl::Vbr { target_kbps: kbps, .. } => {
                bitrate_args("b:v", *kbps, out);
            }
            RateControl::Cqp { .. } => return Err("no constant-quantizer mode".into()),
        }
    } else if encoder == "libx264" || encoder == "libx265" {
        match rate {
            RateControl::Cbr { kbps, max_kbps } => {
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
                bitrate_args("bufsize", *max_kbps, out);
            }
            RateControl::Vbr { target_kbps: kbps, max_kbps } => {
                bitrate_args("b:v", *kbps, out);
                bitrate_args("maxrate", *max_kbps, out);
            }
            RateControl::Cqp { qp } => out.extend(["-qp".to_string(), qp.to_string()]),
        }
    } else {
        return Err(format!("unknown ffmpeg encoder {encoder}"));
    }
    Ok(())
}

fn bitrate_args(flag: &str, kbps: u32, out: &mut Vec<String>) {
    out.push(format!("-{flag}"));
    out.push(format!("{kbps}k"));
}

/// The ffmpeg command line for `cfg` (without the binary itself).
fn argv(cfg: &FfmpegConfig) -> std::result::Result<Vec<String>, String> {
    let mut a = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-nostdin".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-s".to_string(),
        format!("{width}x{height}", width = cfg.width, height = cfg.height),
        "-framerate".to_string(),
        format!("{num}/{den}", num = cfg.fps_num, den = cfg.fps_den),
        "-i".to_string(),
        "pipe:0".to_string(),
        "-an".to_string(),
        "-c:v".to_string(),
        cfg.encoder.clone(),
    ];
    let mut rate = Vec::new();
    rate_args(&cfg.encoder, &cfg.rate, &mut rate)?;
    a.extend(rate);
    a.extend([
        "-g".to_string(),
        cfg.keyint.to_string(),
        // the reader parses the track timescale; forcing it keeps every timestamp in the
        // encoder's own units
        "-video_track_timescale".to_string(),
        cfg.fps_num.to_string(),
        "-f".to_string(),
        "mp4".to_string(),
        "-movflags".to_string(),
        "frag_keyframe+empty_moov+default_base_moof".to_string(),
        "pipe:1".to_string(),
    ]);
    Ok(a)
}

/// Messages from the stdout reader thread to the encoder.
enum WorkerMsg {
    Init(Box<SampleEntry>),
    Packets { packets: Vec<EncodedPacket>, first_cto: Option<i64> },
    Error(String),
    Done,
}

/// ffmpeg's last stderr words, trimmed, parenthesised for error reports (empty when silent).
fn tail_msg(tail: &str) -> String {
    let tail = tail.trim();
    if tail.is_empty() { String::new() } else { format!(" ({tail})") }
}

/// Spawn ffmpeg for `cfg` and wait for its init segment (which arrives before any frame is
/// needed). A wedged child becomes a clear error, never a hang.
fn open(cfg: &FfmpegConfig) -> std::result::Result<FfmpegEncoder, String> {
    let bin = probe::ffmpeg_binary().ok_or_else(|| format!("no ffmpeg binary ({var} or PATH)", var = probe::ENV_BINARY))?;
    let args = argv(cfg)?;
    let mut child = std::process::Command::new(&bin)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {bin}: {e}", bin = bin.display()))?;
    let stdin = child.stdin.take().ok_or_else(|| "ffmpeg: no stdin".to_string())?;
    let stdout = child.stdout.take().ok_or_else(|| "ffmpeg: no stdout".to_string())?;
    let stderr = child.stderr.take().ok_or_else(|| "ffmpeg: no stderr".to_string())?;

    let (tx, rx) = channel::<WorkerMsg>();
    let (err_tx, err_rx) = channel::<String>();

    // stderr drainer: keeps the last 32 KB for error reports, and keeps draining past that so
    // a chatty ffmpeg can never block on a full pipe
    std::thread::Builder::new()
        .name("filmcraft-ffmpeg-stderr".into())
        .spawn(move || {
            let mut tail = Vec::new();
            let mut reader = BufReader::new(stderr);
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        tail.extend_from_slice(&buf[..n]);
                        if tail.len() > 32768 {
                            tail.drain(..tail.len() - 32768);
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = err_tx.send(String::from_utf8_lossy(&tail).into_owned());
        })
        .map_err(|e| format!("cannot spawn the ffmpeg stderr thread: {e}"))?;

    // stdout reader: parses the fragmented MP4 into packets; the encoder never blocks on a
    // full stdout pipe because this thread always drains it
    let (w16, h16) = (cfg.width as u16, cfg.height as u16); // validated ≤ u16::MAX in config()
    let timescale = cfg.fps_num;
    std::thread::Builder::new()
        .name("filmcraft-ffmpeg-reader".into())
        .spawn(move || {
            let mut reader = FragReader::new(stdout);
            match reader.read_init(w16, h16, timescale) {
                Ok(init) => {
                    if tx.send(WorkerMsg::Init(Box::new(init.entry))).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Error(e.to_string()));
                    return;
                }
            }
            loop {
                match reader.next_fragment() {
                    Ok(Some(packets)) => {
                        let first_cto = reader.first_cto();
                        if tx.send(WorkerMsg::Packets { packets, first_cto }).is_err() {
                            return;
                        }
                    }
                    Ok(None) => {
                        let _ = tx.send(WorkerMsg::Done);
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Error(e.to_string()));
                        return;
                    }
                }
            }
        })
        .map_err(|e| format!("cannot spawn the ffmpeg reader thread: {e}"))?;

    let entry = match rx.recv_timeout(INIT_TIMEOUT) {
        Ok(WorkerMsg::Init(entry)) => *entry,
        Ok(WorkerMsg::Error(e)) => {
            kill(&mut child);
            return Err(e);
        }
        Ok(_) => {
            kill(&mut child);
            return Err("ffmpeg: unexpected reader message".into());
        }
        Err(_) => {
            kill(&mut child);
            return Err("ffmpeg: timed out waiting for the MP4 init segment".into());
        }
    };
    Ok(FfmpegEncoder {
        child,
        stdin: Some(stdin),
        rx,
        err_rx,
        y: Vec::new(),
        u: Vec::new(),
        v: Vec::new(),
        w: cfg.width,
        h: cfg.height,
        timescale: cfg.fps_num,
        sample_entry: entry,
        media_start: None,
        hw: cfg.hw,
        dead: false,
        saw_done: false,
    })
}

fn kill(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// The ffmpeg-backed [`VideoEncoder`].
pub struct FfmpegEncoder {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<WorkerMsg>,
    err_rx: Receiver<String>,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    w: u32,
    h: u32,
    timescale: u32,
    sample_entry: SampleEntry,
    /// Edit-list offset (B-frame delay), once the first fragment's timing is known.
    media_start: Option<i64>,
    hw: bool,
    /// The child is gone (failed or finished early): further calls are errors, not hangs.
    dead: bool,
    /// The reader reported EOF before `flush`: the child ended mid-export.
    saw_done: bool,
}

impl FfmpegEncoder {
    /// Kill the child and collect ffmpeg's last words; after this the encoder is dead.
    fn shutdown(&mut self) -> String {
        self.dead = true;
        kill(&mut self.child);
        // drain anything already sent (ignored: the export already failed)
        for msg in self.rx.try_iter() {
            if let WorkerMsg::Packets { first_cto, .. } = msg
                && self.media_start.is_none()
            {
                self.media_start = first_cto.filter(|c| *c > 0);
            }
        }
        self.err_rx.try_recv().unwrap_or_default()
    }

    /// Move newly arrived packets into `out`; a reader failure kills the child and becomes
    /// an error carrying ffmpeg's stderr tail.
    fn drain(&mut self, out: &mut Vec<EncodedPacket>) -> Result<()> {
        for msg in self.rx.try_iter() {
            match msg {
                WorkerMsg::Init(_) => {}
                WorkerMsg::Packets { packets, first_cto } => {
                    if self.media_start.is_none() {
                        self.media_start = first_cto.filter(|c| *c > 0);
                    }
                    out.extend(packets);
                }
                WorkerMsg::Error(e) => {
                    let tail = self.shutdown();
                    return Err(ExportError::Encode(format!("ffmpeg: {e}{tail}", tail = tail_msg(&tail))));
                }
                WorkerMsg::Done => self.saw_done = true,
            }
        }
        Ok(())
    }
}

impl Drop for FfmpegEncoder {
    fn drop(&mut self) {
        if !self.dead {
            // the reader thread is blocked on stdout; killing closes the pipes so it exits
            kill(&mut self.child);
        }
    }
}

impl VideoEncoder for FfmpegEncoder {
    fn sample_entry(&self) -> SampleEntry {
        self.sample_entry.clone()
    }

    fn timescale(&self) -> u32 {
        self.timescale
    }

    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        if self.stdin.is_none() || self.dead {
            return Err(ExportError::Encode("ffmpeg: encode after the encoder ended".into()));
        }
        let mut out = Vec::new();
        self.drain(&mut out)?;
        if self.saw_done {
            return Err(ExportError::Encode("ffmpeg: the encoder ended before all frames were written".into()));
        }
        if f.width != self.w || f.height != self.h {
            return Err(ExportError::Encode(format!(
                "ffmpeg: a {fw}x{fh} picture for a {sw}x{sh} encoder",
                fw = f.width,
                fh = f.height,
                sw = self.w,
                sh = self.h
            )));
        }
        if f.hdr.is_some() {
            return Err(ExportError::Unsupported("ffmpeg: HDR pictures are not supported (8-bit SDR only)".into()));
        }
        let (w, h) = (self.w as usize, self.h as usize);
        if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Err(ExportError::Encode("ffmpeg: the RGBA picture is smaller than its size".into()));
        }
        filmcraft_export::timed(filmcraft_export::Stage::Convert, || filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut self.y, &mut self.u, &mut self.v));
        let write_result = {
            let Some(stdin) = self.stdin.as_mut() else {
                return Err(ExportError::Encode("ffmpeg: encode after the encoder ended".into()));
            };
            stdin.write_all(&self.y).and_then(|()| stdin.write_all(&self.u)).and_then(|()| stdin.write_all(&self.v))
        };
        if let Err(e) = write_result {
            let tail = self.shutdown();
            return Err(ExportError::Encode(format!("ffmpeg: lost while writing a frame: {e}{tail}", tail = tail_msg(&tail))));
        }
        if self.hw {
            filmcraft_export::note_hw_encode_frame();
        }
        Ok(out)
    }

    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        if self.dead {
            return Err(ExportError::Encode("ffmpeg: the encoder already failed".into()));
        }
        // closing stdin is ffmpeg's end-of-stream: it drains its encoder and exits
        drop(self.stdin.take());
        let mut out = Vec::new();
        loop {
            match self.rx.recv() {
                Ok(WorkerMsg::Init(_)) => {}
                Ok(WorkerMsg::Packets { packets, first_cto }) => {
                    if self.media_start.is_none() {
                        self.media_start = first_cto.filter(|c| *c > 0);
                    }
                    out.extend(packets);
                }
                Ok(WorkerMsg::Error(e)) => {
                    let tail = self.shutdown();
                    return Err(ExportError::Encode(format!("ffmpeg: {e}{tail}", tail = tail_msg(&tail))));
                }
                Ok(WorkerMsg::Done) | Err(_) => break,
            }
        }
        let status = self.child.wait().map_err(|e| ExportError::Encode(format!("ffmpeg: {e}")))?;
        self.dead = true;
        let tail = self.err_rx.recv().unwrap_or_default();
        if !status.success() {
            return Err(ExportError::Encode(format!("ffmpeg: exited with {status}{tail}", tail = tail_msg(&tail))));
        }
        Ok(out)
    }

    fn media_start(&self) -> Option<i64> {
        self.media_start
    }
}

/// The Export encoder factory (see the module documentation).
pub fn factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    let encoder = select_encoder_for(format);
    // The ffmpeg H.264 path exists for hardware acceleration. When ffmpeg would only use
    // its software encoder (libx264), decline and let the built-in software encoder handle
    // it: there is no speed or quality win in shelling out to libx264, and the built-in's
    // output is what the export workflow tests expect. (HEVC has no built-in encoder, so
    // ffmpeg is always the path there, hardware or software.)
    if format != Format::Hevc
        && let Some(name) = encoder
        && !probe::is_hardware_encoder(name)
    {
        log::info!("ffmpeg encoding declined: H.264 would be software-only ({name})");
        filmcraft_export::note_hw_encode_declined();
        return None;
    }
    factory_with(format, w, h, rate, s, encoder, open)
}

/// [`factory`] with the encoder picked by the caller (tests, debugging): `None` declines
/// without touching the machine, exactly like a missing ffmpeg binary.
pub fn factory_with_encoder(
    format: Format,
    w: u32,
    h: u32,
    rate: FrameRate,
    s: &ExportSettings,
    encoder: Option<&str>,
) -> Option<Result<Box<dyn VideoEncoder>>> {
    factory_with(format, w, h, rate, s, encoder, open)
}

fn select_encoder_for(format: Format) -> Option<&'static str> {
    probe::select_encoder(if format == Format::Hevc { Codec::Hevc } else { Codec::H264 })
}

/// The factory with its two machine-touching steps passed in, so tests can decline and fail
/// without an ffmpeg binary.
fn factory_with(
    format: Format,
    w: u32,
    h: u32,
    rate: FrameRate,
    s: &ExportSettings,
    encoder: Option<&str>,
    open: impl FnOnce(&FfmpegConfig) -> std::result::Result<FfmpegEncoder, String>,
) -> Option<Result<Box<dyn VideoEncoder>>> {
    let hevc = format == Format::Hevc;
    let declined = |why: &str| {
        log::info!("ffmpeg encoding declined: {why}");
        filmcraft_export::note_hw_encode_declined();
        None
    };
    // no ffmpeg encoder on this machine: H.264 falls through to the next factory, H.265 to
    // the export's own "encoder not available"
    let Some(encoder) = encoder else { return declined("no ffmpeg encoder") };
    let cfg = match config(format, w, h, rate, s, encoder) {
        Ok(c) => c,
        Err(why) => return declined(&why),
    };
    match open(&cfg) {
        Ok(enc) => {
            if cfg.hw {
                filmcraft_export::note_hw_encode_session();
            }
            Some(Ok(Box::new(enc)))
        }
        // the encoder was probed but the spawn failed: for H.264 the software encoder takes
        // over; for H.265 nothing else can, so the export fails naming ffmpeg (loud, not a
        // silent "encoder not available")
        Err(why) => {
            log::info!("ffmpeg encoding failed: {why}");
            filmcraft_export::note_hw_encode_declined();
            if hevc { Some(Err(ExportError::Unsupported(format!("H.265 export with ffmpeg: {why}")))) } else { None }
        }
    }
}

/// Whether an H.265 export through ffmpeg is possible on this machine (cached).
pub fn hevc_available() -> bool {
    probe::hevc_available()
}

/// Ask [`hevc_available`] ahead of time, off the UI thread.
pub fn warm_hevc_probe() {
    probe::warm_hevc_probe()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(mode: BitrateMode) -> ExportSettings {
        ExportSettings { hardware_encoding: HardwareEncoding::Auto, bitrate_mode: mode, bitrate_kbps: 4000, cqp: 20, ..Default::default() }
    }

    fn cfg_for(format: Format, mode: BitrateMode, encoder: &str) -> std::result::Result<FfmpegConfig, String> {
        config(format, 640, 360, FrameRate::FPS_24, &settings(mode), encoder)
    }

    #[test]
    fn off_and_unsupported_settings_decline_without_spawning() {
        let declined = |format, w, h, s: &ExportSettings, encoder: Option<&str>| {
            assert!(factory_with(format, w, h, FrameRate::FPS_24, s, encoder, |_| panic!("must not spawn ffmpeg")).is_none(), "{format:?} should decline");
        };
        // no encoder at all: declines without touching the machine
        let s = settings(BitrateMode::Cqp);
        declined(Format::H264, 640, 360, &s, None);
        declined(Format::Hevc, 640, 360, &s, None);
        // hardware encoding off: the built-in encoder's turn
        let off = ExportSettings { hardware_encoding: HardwareEncoding::Off, ..settings(BitrateMode::Vbr1Pass) };
        declined(Format::H264, 640, 360, &off, Some("libx264"));
        // the toggle does not apply to HEVC (no software encoder)
        assert!(cfg_for(Format::Hevc, BitrateMode::Vbr1Pass, "libx265").is_ok());
        // the rest is not ffmpeg's
        declined(Format::ProRes, 640, 360, &s, Some("libx264"));
        let mxf = ExportSettings { format: Format::MxfOp1a, ..settings(BitrateMode::Vbr1Pass) };
        declined(Format::H264, 640, 360, &mxf, Some("libx264"));
        declined(Format::H264, 641, 360, &s, Some("libx264"));
        declined(Format::H264, 0, 360, &s, Some("libx264"));
        let hdr = ExportSettings { signal: filmcraft_export::ColorSignal::PQ, ..s.clone() };
        declined(Format::Hevc, 640, 360, &hdr, Some("libx265"));
        declined(Format::H264, 640, 360, &settings(BitrateMode::Vbr2Pass), Some("libx264"));
        declined(Format::H264, 640, 360, &settings(BitrateMode::Crf), Some("libx264"));
        let interlaced = ExportSettings { field_order: FieldOrder::UpperFirst, ..s.clone() };
        declined(Format::H264, 640, 360, &interlaced, Some("libx264"));
        let sar = ExportSettings { pixel_aspect: Some((4, 3)), ..s.clone() };
        declined(Format::H264, 640, 360, &sar, Some("libx264"));
        // hostile sizes and frame rates
        assert!(config(Format::H264, 70_000, 64, FrameRate::FPS_24, &s, "libx264").is_err());
        assert!(config(Format::H264, 64, 70_000, FrameRate::FPS_24, &s, "libx264").is_err());
        for rate in [FrameRate { num: 0, den: 1 }, FrameRate { num: 24, den: 0 }, FrameRate { num: -24, den: 1 }] {
            assert!(config(Format::H264, 640, 360, rate, &s, "libx264").is_err(), "{rate:?}");
        }
    }

    #[test]
    fn config_resolves_rate_control_and_keyint() {
        let c = cfg_for(Format::H264, BitrateMode::Cqp, "h264_qsv").unwrap();
        assert_eq!(c.rate, RateControl::Cqp { qp: 20 });
        assert!(c.hw);
        // 2 seconds at 24 fps by default
        assert_eq!(c.keyint, 48);
        let c = cfg_for(Format::Hevc, BitrateMode::Cbr, "libx265").unwrap();
        assert_eq!(c.rate, RateControl::Cbr { kbps: 4000, max_kbps: 6000 });
        assert!(!c.hw);
        assert_eq!(c.encoder, "libx265");
        // the quantizer is clamped into the 0..=51 range the encoders take
        let mut s = settings(BitrateMode::Cqp);
        s.cqp = 200;
        assert_eq!(config(Format::H264, 640, 360, FrameRate::FPS_24, &s, "libx264").unwrap().rate, RateControl::Cqp { qp: 51 });
    }

    #[test]
    fn argv_spells_every_encoder_family() {
        let args = |encoder: &str, mode: BitrateMode| {
            let cfg = cfg_for(if encoder.starts_with("hevc") { Format::Hevc } else { Format::H264 }, mode, encoder).unwrap();
            argv(&cfg).unwrap().join(" ")
        };
        // Quick Sync Video: constant quality is -global_quality (ICQ)
        let qsv = args("h264_qsv", BitrateMode::Cqp);
        assert!(qsv.contains("-c:v h264_qsv -global_quality 20"), "{qsv}");
        // NVENC: explicit rate-control selection
        let nvenc = args("hevc_nvenc", BitrateMode::Cqp);
        assert!(nvenc.contains("-c:v hevc_nvenc -rc constqp -qp 20"), "{nvenc}");
        let nvenc_cbr = args("h264_nvenc", BitrateMode::Cbr);
        assert!(nvenc_cbr.contains("-rc cbr -b:v 4000k -maxrate 6000k"), "{nvenc_cbr}");
        // VA-API: plain -qp
        assert!(args("h264_vaapi", BitrateMode::Cqp).contains("-qp 20"));
        // AMF: per-picture-type QPs
        let amf = args("hevc_amf", BitrateMode::Cqp);
        assert!(amf.contains("-rc cqp -qp_i 20 -qp_p 20"), "{amf}");
        // software fallback: -qp, and CBR gets a VBV
        let sw = args("libx264", BitrateMode::Cqp);
        assert!(sw.contains("-c:v libx264 -qp 20"), "{sw}");
        let sw_cbr = args("libx264", BitrateMode::Cbr);
        assert!(sw_cbr.contains("-b:v 4000k -maxrate 6000k -bufsize 6000k"), "{sw_cbr}");
        // every command line pipes raw video in and fragmented MP4 out
        for a in [&qsv, &nvenc, &sw] {
            assert!(a.contains("-f rawvideo -pix_fmt yuv420p -s 640x360 -framerate 24/1 -i pipe:0"), "{a}");
            assert!(a.contains("-an -f mp4 -movflags frag_keyframe+empty_moov+default_base_moof pipe:1"), "{a}");
            assert!(a.contains("-video_track_timescale 24"), "{a}");
        }
        // VideoToolbox has no constant-quantizer mode: a clear error, not a guess
        assert!(cfg_for(Format::H264, BitrateMode::Cqp, "h264_videotoolbox").is_err());
        assert!(cfg_for(Format::H264, BitrateMode::Vbr1Pass, "h264_videotoolbox").is_ok());
    }

    #[test]
    fn a_failing_open_is_an_error_for_hevc_and_a_fallback_for_h264() {
        let err = |format| factory_with(format, 640, 360, FrameRate::FPS_24, &settings(BitrateMode::Vbr1Pass), Some("libx264"), |_| Err("boom".to_string()));
        // H.264: the software encoder takes it
        assert!(err(Format::H264).is_none());
        // HEVC: nothing else can take it, so the export fails naming ffmpeg
        let e = match err(Format::Hevc) {
            Some(Err(e)) => e,
            _ => panic!("expected the HEVC export to fail"),
        };
        assert!(e.to_string().contains("ffmpeg"), "{e}");
    }
}
