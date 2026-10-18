//! Finding the external `ffmpeg` binary and picking its best H.264 / HEVC encoder.
//!
//! The binary is `FILMCRAFT_FFMPEG` when set, else `ffmpeg` on `PATH`; both are verified by
//! running `ffmpeg -version`, so a stale path degrades to "no ffmpeg" instead of a confusing
//! spawn failure later. Encoder choice is probed once per process (`ffmpeg -h encoder=<name>`)
//! and cached: hardware first per OS, then the software fallback, so machines without any
//! hardware encoder still export (and CI, which has no GPU, exercises the whole pipeline
//! through `libx264` / `libx265`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Environment override for the ffmpeg binary (`ffmpeg` on `PATH` otherwise).
pub const ENV_BINARY: &str = "FILMCRAFT_FFMPEG";
/// Environment override forcing the encoder (tests, debugging): one ffmpeg encoder name
/// (`libx264`), or `h264:<name>,hevc:<name>` to pick per codec.
pub const ENV_ENCODER: &str = "FILMCRAFT_FFMPEG_ENCODER";

/// The codec being exported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

#[cfg(target_os = "windows")]
const H264_CANDIDATES: &[&str] = &["h264_nvenc", "h264_qsv", "h264_amf", "libx264"];
#[cfg(target_os = "macos")]
const H264_CANDIDATES: &[&str] = &["h264_videotoolbox", "libx264"];
#[cfg(target_os = "linux")]
const H264_CANDIDATES: &[&str] = &["h264_qsv", "h264_vaapi", "h264_nvenc", "libx264"];
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
const H264_CANDIDATES: &[&str] = &["libx264"];

#[cfg(target_os = "windows")]
const HEVC_CANDIDATES: &[&str] = &["hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"];
#[cfg(target_os = "macos")]
const HEVC_CANDIDATES: &[&str] = &["hevc_videotoolbox", "libx265"];
#[cfg(target_os = "linux")]
const HEVC_CANDIDATES: &[&str] = &["hevc_qsv", "hevc_vaapi", "hevc_nvenc", "libx265"];
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
const HEVC_CANDIDATES: &[&str] = &["libx265"];

fn candidates(codec: Codec) -> &'static [&'static str] {
    match codec {
        Codec::H264 => H264_CANDIDATES,
        Codec::Hevc => HEVC_CANDIDATES,
    }
}

/// Whether `bin` starts and reports its version (a cheap liveness check, not a version gate).
fn runs(bin: &Path) -> bool {
    Command::new(bin).arg("-hide_banner").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn find_binary() -> Option<PathBuf> {
    if let Ok(p) = std::env::var(ENV_BINARY) {
        let p = PathBuf::from(p);
        if runs(&p) {
            return Some(p);
        }
        log::warn!("{ENV_BINARY} points at an ffmpeg that does not run; falling back to PATH");
    }
    let path = PathBuf::from("ffmpeg");
    runs(&path).then_some(path)
}

/// The external ffmpeg binary, if one runs on this machine (cached).
pub fn ffmpeg_binary() -> Option<PathBuf> {
    static BINARY: OnceLock<Option<PathBuf>> = OnceLock::new();
    BINARY.get_or_init(find_binary).clone()
}

/// Whether this ffmpeg build ships `name` (`ffmpeg -h encoder=<name>` exits 0).
fn encoder_runs(bin: &Path, name: &str) -> bool {
    Command::new(bin).arg("-hide_banner").arg("-h").arg(format!("encoder={name}")).output().map(|o| o.status.success()).unwrap_or(false)
}

/// Parse [`ENV_ENCODER`]'s value for `codec`: `Some(name)` from `h264:<name>,hevc:<name>`
/// pairs or a bare name applying to both codecs.
fn parse_forced(v: &str, codec: Codec) -> Option<&str> {
    if v.contains(':') {
        let want = match codec {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
        };
        for pair in v.split(',') {
            let (k, name) = pair.split_once(':')?;
            if k.trim() == want && !name.trim().is_empty() {
                return Some(name.trim());
            }
        }
        return None;
    }
    let name = v.trim();
    (!name.is_empty()).then_some(name)
}

/// The [`ENV_ENCODER`] override for `codec`. The name is leaked once on purpose: selection is
/// cached for the life of the process, and the override is a debugging aid, not hot-path.
fn forced_encoder(codec: Codec) -> Option<&'static str> {
    let v = std::env::var(ENV_ENCODER).ok()?;
    parse_forced(&v, codec).map(|name| Box::leak(name.to_string().into_boxed_str()) as &'static str)
}

/// The ffmpeg encoder to use for `codec`: the [`ENV_ENCODER`] override, else the first
/// candidate the local ffmpeg build ships (cached per codec).
pub fn select_encoder(codec: Codec) -> Option<&'static str> {
    static H264: OnceLock<Option<&'static str>> = OnceLock::new();
    static HEVC: OnceLock<Option<&'static str>> = OnceLock::new();
    let (slot, cands) = match codec {
        Codec::H264 => (&H264, H264_CANDIDATES),
        Codec::Hevc => (&HEVC, HEVC_CANDIDATES),
    };
    *slot.get_or_init(|| {
        if let Some(forced) = forced_encoder(codec) {
            log::info!("{ENV_ENCODER} forces the ffmpeg encoder to {forced}");
            return Some(forced);
        }
        let bin = ffmpeg_binary()?;
        let pick = cands.iter().find(|name| encoder_runs(bin.as_path(), name)).copied();
        match pick {
            Some(name) => log::info!("ffmpeg encoder for {codec:?}: {name}"),
            None => log::info!("no ffmpeg encoder for {codec:?} in {}", bin.display()),
        }
        pick
    })
}

/// Whether the picked encoder is a hardware one (as opposed to the `libx264` / `libx265`
/// software fallback): what the export hardware-encoder counters count.
pub fn is_hardware_encoder(name: &str) -> bool {
    !name.starts_with("lib")
}

/// Whether an H.265 export through ffmpeg is possible on this machine (cached): what offers
/// `Format::Hevc` in the export dialog on systems without a native HEVC encoder.
pub fn hevc_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| select_encoder(Codec::Hevc).is_some())
}

/// Ask [`hevc_available`] now, on a thread of its own, so the first draw of the format list
/// does not spend two ffmpeg probes on the UI thread. Mirrors the native probes' warming.
pub fn warm_hevc_probe() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("filmcraft-ffmpeg-probe".into()).spawn(|| {
            let _ = std::panic::catch_unwind(hevc_available);
        });
        if let Err(why) = spawned {
            log::info!("ffmpeg HEVC probe not started ahead of time: {why}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_encoder_parses_pairs_and_bare_names() {
        assert_eq!(parse_forced("libx264", Codec::H264), Some("libx264"));
        assert_eq!(parse_forced("libx264", Codec::Hevc), Some("libx264"));
        assert_eq!(parse_forced("h264:libx264,hevc:libx265", Codec::H264), Some("libx264"));
        assert_eq!(parse_forced("h264:libx264,hevc:libx265", Codec::Hevc), Some("libx265"));
        assert_eq!(parse_forced("h264:libx264", Codec::Hevc), None);
        assert_eq!(parse_forced("h264: ,hevc:libx265", Codec::H264), None);
        assert_eq!(parse_forced("", Codec::H264), None);
        assert_eq!(parse_forced("   ", Codec::Hevc), None);
    }

    #[test]
    fn every_candidate_list_ends_in_a_software_fallback() {
        for list in [H264_CANDIDATES, HEVC_CANDIDATES] {
            assert!(!list.is_empty());
            let last = list.last().unwrap();
            assert!(last.starts_with("lib"), "last candidate should be software, got {last}");
        }
    }

    #[test]
    fn hardware_detection_covers_the_candidate_lists() {
        for name in H264_CANDIDATES.iter().chain(HEVC_CANDIDATES) {
            assert_eq!(is_hardware_encoder(name), !name.starts_with("lib"));
        }
    }
}
