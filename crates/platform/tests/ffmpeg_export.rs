//! End-to-end test of the external-ffmpeg encoder through a real ffmpeg binary
//! (`libx264` / `libx265`, forced explicitly so the test does not depend on what hardware
//! the machine has). Skipped gracefully when no suitable ffmpeg runs here; the argument
//! building, the fragmented-MP4 parser and the decline logic are covered by unit tests.

use filmcraft_export::{BitrateMode, EncodedPacket, EncoderFrame, ExportSettings, Format, HardwareEncoding, VideoEncoder};
use filmcraft_platform::ffmpeg;
use filmcraft_time::FrameRate;

fn ffmpeg_has(encoder: &str) -> bool {
    let Some(bin) = ffmpeg::ffmpeg_binary() else { return false };
    std::process::Command::new(bin).arg("-hide_banner").arg("-h").arg(format!("encoder={encoder}")).output().map(|o| o.status.success()).unwrap_or(false)
}

fn test_frame(w: u32, h: u32, i: u64) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x.wrapping_add(y).wrapping_add(i as u32 * 7) % 256) as u8;
            px.extend_from_slice(&[v, 255 - v, v / 2, 255]);
        }
    }
    px
}

fn roundtrip(format: Format, encoder: &str, frames: u64) -> Vec<EncodedPacket> {
    let mut s = ExportSettings::default();
    s.format = format;
    s.hardware_encoding = HardwareEncoding::Auto;
    s.bitrate_mode = BitrateMode::Cqp;
    s.cqp = 30;
    let mut enc = ffmpeg::export::factory_with_encoder(format, 64, 64, FrameRate::FPS_24, &s, Some(encoder))
        .expect("the ffmpeg factory takes the export")
        .expect("the encoder opens");
    assert_eq!(enc.timescale(), 24);
    let mut packets = Vec::new();
    for i in 0..frames {
        let rgba = test_frame(64, 64, i);
        let frame = EncoderFrame { width: 64, height: 64, rgba: &rgba, hdr: None, index: i };
        packets.extend(enc.encode(&frame).expect("encode"));
    }
    packets.extend(enc.flush().expect("flush"));
    // the sample entry parses (avcC / hvcC from the init segment)
    let _ = enc.sample_entry();
    packets
}

#[test]
fn h264_cqp_roundtrip_through_libx264() {
    if !ffmpeg_has("libx264") {
        eprintln!("skipped: no ffmpeg with libx264 on PATH");
        return;
    }
    let packets = roundtrip(Format::H264, "libx264", 6);
    assert!(packets.len() >= 6, "every frame produced a packet, got {n}", n = packets.len());
    assert!(packets.iter().any(|p| p.key), "at least one keyframe");
    assert!(packets.iter().all(|p| p.duration == 1), "one frame per packet at 24 fps");
    assert!(packets.iter().all(|p| p.composition_offset >= 0), "sane composition offsets");
    assert!(packets.iter().all(|p| !p.data.is_empty()), "no empty packets");
}

#[test]
fn hevc_cqp_roundtrip_through_libx265() {
    if !ffmpeg_has("libx265") {
        eprintln!("skipped: no ffmpeg with libx265 on PATH");
        return;
    }
    let packets = roundtrip(Format::Hevc, "libx265", 4);
    assert!(packets.len() >= 4, "every frame produced a packet, got {n}", n = packets.len());
    assert!(packets.iter().any(|p| p.key), "at least one keyframe");
}

#[test]
fn declines_without_an_encoder_or_with_hardware_off() {
    let mut s = ExportSettings::default();
    s.format = Format::H264;
    s.hardware_encoding = HardwareEncoding::Auto;
    // no encoder picked (no ffmpeg): declines without touching the machine
    assert!(ffmpeg::export::factory_with_encoder(Format::H264, 64, 64, FrameRate::FPS_24, &s, None).is_none());
    assert!(ffmpeg::export::factory_with_encoder(Format::Hevc, 64, 64, FrameRate::FPS_24, &s, None).is_none());
    // hardware encoding off: the built-in encoder's turn
    s.hardware_encoding = HardwareEncoding::Off;
    assert!(ffmpeg::export::factory_with_encoder(Format::H264, 64, 64, FrameRate::FPS_24, &s, Some("libx264")).is_none());
}
