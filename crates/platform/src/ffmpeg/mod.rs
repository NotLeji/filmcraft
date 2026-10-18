//! H.264 / H.265 export through an external `ffmpeg` binary.
//!
//! This module exists only in this fork (branch `lezy/ffmpeg-encoder`): upstream rejects any
//! ffmpeg dependency ("ffmpeg never linked or shipped"), so it must never be merged into `main`
//! or proposed upstream. It is license-safe all the same: the binary is found on the user's
//! system at run time and only spoken to through pipes (mere aggregation, no linking), so the
//! `MIT OR Apache-2.0` license is unaffected.
//!
//! What it does: [`export::factory`] implements [`filmcraft_export::VideoEncoder`] by piping raw
//! `yuv420p` frames into `ffmpeg`'s stdin and reading a fragmented MP4 back from its stdout
//! ([`mp4`]). The fragment sample tables give exact per-sample sizes, sync flags, durations and
//! composition offsets, so B-frames stay correct. The encoder is picked by [`probe`] per machine:
//! QSV / NVENC / VA-API / AMF / VideoToolbox where the local ffmpeg build has them, `libx264` /
//! `libx265` as the software fallback, which is also what makes H.265 exports possible on machines
//! whose native encoders cannot do it.
//!
//! [`export::factory`] is registered behind the native hardware encoders (NVENC, VideoToolbox),
//! so those stay preferred where they exist, and in front of the built-in software encoders:
//!
//! - H.264 takes an export only when Export ▸ Hardware encoding is Auto and the native encoder
//!   declined it (or there is none); otherwise the built-in encoder runs, exactly as before.
//! - H.265 (`Format::Hevc`) has no built-in encoder, so choosing the format is the opt-in, and
//!   [`hevc_available`] (a cached probe, warmed off the UI thread) is what offers the format.
//!
//! Added export option: [`filmcraft_export::BitrateMode::Cqp`] (constant quantizer, 0–51), the
//! native quality mode of hardware encoders, mapped per ffmpeg encoder (`-qp`, `-cq`,
//! `-global_quality`, …). A missing ffmpeg binary is never a crash: the factory declines and the
//! export falls back (H.264) or reports "encoder not available" (H.265), naming the binary it
//! looked for.

pub mod export;
pub mod mp4;
pub mod probe;

pub use export::{factory, hevc_available, warm_hevc_probe};
pub use probe::{Codec, ffmpeg_binary, select_encoder};
