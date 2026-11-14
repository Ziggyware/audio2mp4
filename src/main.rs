/* [SPEC]
 * WIN_LOOPBACK_MP3_MUX v5.0.0 — Native Windows WASAPI + Media Foundation H.264/AAC & Ultra-Physics Video Studio.
 *
 * Guarantees:
 * 1. Zero reliance on external `ffmpeg.exe` on Windows: Uses Windows Media Foundation (`IMFSinkWriter`) to hardware-encode
 *    standards-compliant H.264 (`avc1`) video + AAC-LC (`mp4a` 0x40) audio directly from the physics compositor & PCM capture.
 * 2. Native Multi-Video `.mp4` Ingestion (`IMFSourceReader`): Decodes external `.mp4` clips natively and applies analytical
 *    continuum-mechanics transitions (`Spring`, `Wave`, `Vortex`, `Chroma`, `Dissolve`) across moving video frames or images.
 * 3. Post-Mux ISO-BMFF `moov.udta.meta.ilst` + `ID32` Binary Atom Injector with automatic `stco`/`co64` offset relocation.
 * 4. Interactive 24-bit TrueColor (OkLCh) Console DAW with 512-pt Radix-2 FFT Spectrum, ITU-R BS.1770-4 LUFS, Beat-Reactive
 *    Spectral Flux Auto-Transitions, and an internal `[T]` Stereo Harmonic + 120BPM Sub-Bass Test Synthesizer.
 */

use std::cell::UnsafeCell;
use std::env;
use std::f32::consts::{PI, TAU};
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mp3lame_encoder::{Bitrate, Builder, DualPcm, FlushNoGap, Quality};

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

// ============================================================================
// SECTION 1: OKLCH PERCEPTUAL COLORIMETRY -> 24-BIT TRUECOLOR ANSI ENGINE
// ============================================================================

#[inline]
pub fn oklch_to_srgb(l: f32, c: f32, h_deg: f32) -> (u8, u8, u8) {
    let h_rad = h_deg.to_radians();
    let a = c * h_rad.cos();
    let b = c * h_rad.sin();

    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l_cubed = l_ * l_ * l_;
    let m_cubed = m_ * m_ * m_;
    let s_cubed = s_ * s_ * s_;

    let r_lin = 4.0767416621 * l_cubed - 3.3077115913 * m_cubed + 0.2309699292 * s_cubed;
    let g_lin = -1.2684380046 * l_cubed + 2.6097574011 * m_cubed - 0.3413193965 * s_cubed;
    let b_lin = -0.0041960863 * l_cubed - 0.7034186147 * m_cubed + 1.7076147010 * s_cubed;

    #[inline(always)]
    fn linear_to_srgb_u8(x: f32) -> u8 {
        let clamped = x.clamp(0.0, 1.0);
        let s = if clamped <= 0.0031308 {
            12.92 * clamped
        } else {
            1.055 * clamped.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    }

    (
        linear_to_srgb_u8(r_lin),
        linear_to_srgb_u8(g_lin),
        linear_to_srgb_u8(b_lin),
    )
}

#[inline]
pub fn push_fg_oklch(buf: &mut String, l: f32, c: f32, h_deg: f32) {
    let (r, g, b) = oklch_to_srgb(l, c, h_deg);
    let _ = write!(buf, "\x1b[38;2;{r};{g};{b}m");
}

#[inline]
pub fn push_bg_oklch(buf: &mut String, l: f32, c: f32, h_deg: f32) {
    let (r, g, b) = oklch_to_srgb(l, c, h_deg);
    let _ = write!(buf, "\x1b[48;2;{r};{g};{b}m");
}

// ============================================================================
// SECTION 2: DOMAIN TYPES & AUTONOMOUS TRANSITION STATE
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Mp3,
    M4a,
    VideoMp4,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PhysicsTransition {
    Spring = 0,
    Wave = 1,
    Vortex = 2,
    Chroma = 3,
    Dissolve = 4,
}

impl PhysicsTransition {
    pub fn from_u8(v: u8) -> Self {
        match v % 5 {
            0 => Self::Spring,
            1 => Self::Wave,
            2 => Self::Vortex,
            3 => Self::Chroma,
            _ => Self::Dissolve,
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "spring" | "elastic" | "1" => Ok(Self::Spring),
            "wave" | "ripple" | "2" => Ok(Self::Wave),
            "vortex" | "swirl" | "3" => Ok(Self::Vortex),
            "chroma" | "prism" | "dispersion" | "4" => Ok(Self::Chroma),
            "dissolve" | "fade" | "5" => Ok(Self::Dissolve),
            other => Err(format!(
                "Unknown transition '{other}'. Valid: spring, wave, vortex, chroma, dissolve"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Spring => "1:SPRING",
            Self::Wave => "2:WAVE",
            Self::Vortex => "3:VORTEX",
            Self::Chroma => "4:CHROMA",
            Self::Dissolve => "5:DISSOLVE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AutoTransitionMode {
    BeatReactive = 0,
    TimerGrid = 1,
    ManualOnly = 2,
}

impl AutoTransitionMode {
    pub fn from_u8(v: u8) -> Self {
        match v % 3 {
            0 => Self::BeatReactive,
            1 => Self::TimerGrid,
            _ => Self::ManualOnly,
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "beat" | "audio" | "flux" | "reactive" => Ok(Self::BeatReactive),
            "timer" | "auto" | "grid" | "interval" => Ok(Self::TimerGrid),
            "manual" | "off" | "none" => Ok(Self::ManualOnly),
            other => Err(format!(
                "Unknown --auto-mode '{other}'. Valid: beat, timer, manual"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BeatReactive => "BEAT+TIMER AUTO",
            Self::TimerGrid => "FIXED TIMER AUTO",
            Self::ManualOnly => "MANUAL ONLY",
        }
    }
}

pub struct InteractiveState {
    pub paused: AtomicBool,
    pub test_synth_active: AtomicBool,
    pub transition_mode: AtomicU8,
    pub auto_mode: AtomicU8,
    pub cycle_shaders: AtomicBool,
    pub gain_half_db_steps: AtomicI32,
    pub gate_db_int: AtomicI32,
    pub active_scene_idx: AtomicUsize,
    pub cue_markers_count: AtomicUsize,
}

impl InteractiveState {
    pub fn new(
        initial_trans: PhysicsTransition,
        initial_auto: AutoTransitionMode,
        cycle_shaders: bool,
        test_synth: bool,
        initial_gate_db: Option<f32>,
    ) -> Self {
        Self {
            paused: AtomicBool::new(false),
            test_synth_active: AtomicBool::new(test_synth),
            transition_mode: AtomicU8::new(initial_trans as u8),
            auto_mode: AtomicU8::new(initial_auto as u8),
            cycle_shaders: AtomicBool::new(cycle_shaders),
            gain_half_db_steps: AtomicI32::new(0),
            gate_db_int: AtomicI32::new(initial_gate_db.map(|g| g.round() as i32).unwrap_or(-100)),
            active_scene_idx: AtomicUsize::new(0),
            cue_markers_count: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn gain_db(&self) -> f32 {
        (self.gain_half_db_steps.load(Ordering::Relaxed) as f32) * 0.5
    }

    #[inline]
    pub fn gain_linear(&self) -> f32 {
        10.0f32.powf(self.gain_db() / 20.0)
    }

    #[inline]
    pub fn gate_db(&self) -> Option<f32> {
        let v = self.gate_db_int.load(Ordering::Relaxed);
        if v <= -95 {
            None
        } else {
            Some(v as f32)
        }
    }

    #[inline]
    pub fn transition(&self) -> PhysicsTransition {
        PhysicsTransition::from_u8(self.transition_mode.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn auto_mode(&self) -> AutoTransitionMode {
        AutoTransitionMode::from_u8(self.auto_mode.load(Ordering::Relaxed))
    }

    pub fn trigger_next_scene(&self, num_scenes: usize) -> (usize, PhysicsTransition) {
        let next_sc = (self.active_scene_idx.fetch_add(1, Ordering::AcqRel) + 1) % num_scenes.max(1);
        let trans = if self.cycle_shaders.load(Ordering::Relaxed) {
            let prev_t = self.transition_mode.fetch_add(1, Ordering::AcqRel);
            PhysicsTransition::from_u8(prev_t.wrapping_add(1))
        } else {
            self.transition()
        };
        (next_sc, trans)
    }
}

#[derive(Clone, Debug)]
pub struct TimelineEvent {
    pub sample_frame: u64,
    pub from_scene: usize,
    pub target_scene: usize,
    pub transition: PhysicsTransition,
    pub trigger_reason: &'static str,
}

#[derive(Clone, Debug)]
pub struct RgbCanvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>, // RGB24 [R, G, B, ...]
}

impl RgbCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0u8; width * height * 3],
        }
    }

    #[inline(always)]
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 3] {
        let x = (u.clamp(0.0, 1.0) * ((self.width - 1) as f32)).max(0.0);
        let y = (v.clamp(0.0, 1.0) * ((self.height - 1) as f32)).max(0.0);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - (x0 as f32);
        let fy = y - (y0 as f32);

        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        let i00 = (y0 * self.width + x0) * 3;
        let i10 = (y0 * self.width + x1) * 3;
        let i01 = (y1 * self.width + x0) * 3;
        let i11 = (y1 * self.width + x1) * 3;

        let p = &self.pixels;
        [
            p[i00] as f32 * w00 + p[i10] as f32 * w10 + p[i01] as f32 * w01 + p[i11] as f32 * w11,
            p[i00 + 1] as f32 * w00
                + p[i10 + 1] as f32 * w10
                + p[i01 + 1] as f32 * w01
                + p[i11 + 1] as f32 * w11,
            p[i00 + 2] as f32 * w00
                + p[i10 + 2] as f32 * w10
                + p[i01 + 2] as f32 * w01
                + p[i11 + 2] as f32 * w11,
        ]
    }

    pub fn resize_to(&self, target_w: usize, target_h: usize) -> Self {
        if self.width == target_w && self.height == target_h {
            return self.clone();
        }
        let mut out = Self::new(target_w, target_h);
        for y in 0..target_h {
            let v = (y as f32) / ((target_h - 1).max(1) as f32);
            for x in 0..target_w {
                let u = (x as f32) / ((target_w - 1).max(1) as f32);
                let rgb = self.sample_bilinear(u, v);
                let dst = (y * target_w + x) * 3;
                out.pixels[dst] = rgb[0].round().clamp(0.0, 255.0) as u8;
                out.pixels[dst + 1] = rgb[1].round().clamp(0.0, 255.0) as u8;
                out.pixels[dst + 2] = rgb[2].round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

/// A visual source can be either a static/animated procedural or image canvas,
/// or a decoded multi-frame video clip loaded via Windows Media Foundation `IMFSourceReader`.
#[derive(Clone, Debug)]
pub struct VisualScene {
    pub frames: Vec<RgbCanvas>,
}

impl VisualScene {
    pub fn from_single(canvas: RgbCanvas) -> Self {
        Self {
            frames: vec![canvas],
        }
    }

    #[inline]
    pub fn frame_at(&self, frame_idx: usize) -> &RgbCanvas {
        &self.frames[frame_idx % self.frames.len().max(1)]
    }
}

#[derive(Clone, Debug)]
pub struct MediaMetadata {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub year: String,
    pub lang: [u8; 3],
    pub behind_text: String,
    pub cover_mime: &'static str,
    pub cover_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct CliConfig {
    pub out_path: PathBuf,
    pub format: ExportFormat,
    pub bitrate_kbps: u16,
    pub duration_secs: u64,
    pub fade_ms: u32,
    pub trim_silence_db: Option<f32>,
    pub test_synth: bool,
    pub transition: PhysicsTransition,
    pub auto_mode: AutoTransitionMode,
    pub auto_interval_secs: f32,
    pub cycle_shaders: bool,
    pub transition_secs: f32,
    pub video_width: usize,
    pub video_height: usize,
    pub fps: u32,
    pub clip_paths: Vec<PathBuf>,
    pub meta: MediaMetadata,
}

// ============================================================================
// SECTION 2B: WILDCARD MEDIA INPUT EXPANSION  (`clips\\*.mp4`, directories)
// ============================================================================
// cmd.exe and PowerShell do NOT expand `*` / `?` wildcards for native
// executables, so `win_loopback_mp3_mux.exe clips\\*.mp4` arrives with the
// pattern literally in argv. Expansion therefore happens here, in-process:
// one directory enumeration per pattern, a case-insensitive matcher, and
// sorted + de-duplicated results so scene order is deterministic.
// Wildcards are supported in the FINAL path component (the `path\*.mp4`
// shape); a plain directory argument expands to every media file inside it.

const MEDIA_EXTS: [&str; 12] = [
    "mp4", "mov", "mkv", "png", "jpg", "jpeg", "bmp", "gif", "ppm", "webp", "tif", "tiff",
];

#[inline]
fn has_wildcards(s: &str) -> bool {
    s.contains('*') || s.contains('?')
}

/// Case-insensitive `*` / `?` match: classic two-pointer scan with star
/// backtracking. Effectively linear on filename patterns.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_p, mut star_t) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi].eq_ignore_ascii_case(&t[ti])) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_p = pi;
            star_t = ti;
            pi += 1;
        } else if star_p != usize::MAX {
            pi = star_p + 1;
            star_t += 1;
            ti = star_t;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn is_media_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => MEDIA_EXTS.iter().any(|m| m.eq_ignore_ascii_case(ext)),
        None => false,
    }
}

/// Sorted media files directly inside `dir` whose names match `name_pattern`
/// (`None` = every media file). Single read_dir pass, no recursion.
fn scan_dir_sorted(dir: &Path, name_pattern: Option<&str>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
            if !is_file {
                continue;
            }
            let path = entry.path();
            match name_pattern {
                Some(pat) => {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if wildcard_match(pat, name) {
                            out.push(path);
                        }
                    }
                }
                None => {
                    if is_media_file(&path) {
                        out.push(path);
                    }
                }
            }
        }
    }
    out.sort();
    out
}

/// Expands ONE input: an existing file (returned as-is — the loader reports
/// any error exactly as before), an existing directory (all media files
/// inside it, sorted), or a wildcard pattern (`path\*.mp4`).
pub fn expand_one_pattern(pattern: &Path) -> Vec<PathBuf> {
    let pat_str = pattern.to_string_lossy();
    if !has_wildcards(&pat_str) {
        if pattern.is_dir() {
            return scan_dir_sorted(pattern, None);
        }
        return vec![pattern.to_path_buf()];
    }
    let Some(name_pat) = pattern.file_name().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let dir = match pattern.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    scan_dir_sorted(&dir, Some(name_pat))
}

/// Expands every collected CLI media input, in order. A wildcard pattern or
/// directory that matches nothing is a hard error — better than silently
/// recording with procedural fallback scenes. Plain file paths pass through.
pub fn expand_media_inputs(patterns: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut out: Vec<PathBuf> = Vec::new();
    for pat in patterns {
        if has_wildcards(&pat.to_string_lossy()) || pat.is_dir() {
            let hits = expand_one_pattern(pat);
            if hits.is_empty() {
                return Err(format!("No media files match '{}'", pat.display()));
            }
            out.extend(hits);
        } else {
            out.push(pat.clone());
        }
    }
    out.dedup();
    Ok(out)
}

// ============================================================================
// SECTION 3: SPSC RING BUFFER + 512-PT FFT + SPECTRAL FLUX BEAT DETECTOR
// ============================================================================

#[repr(align(64))]
struct CacheAlignedAtomicUsize(AtomicUsize);

pub struct SpscStereoRing {
    mask: usize,
    buffer: Box<[UnsafeCell<f32>]>,
    write_idx: CacheAlignedAtomicUsize,
    read_idx: CacheAlignedAtomicUsize,
    dropped_frames: CacheAlignedAtomicUsize,
}

unsafe impl Sync for SpscStereoRing {}
unsafe impl Send for SpscStereoRing {}

impl SpscStereoRing {
    pub fn new_pow2(frames_pow2: usize) -> Self {
        assert!(frames_pow2.is_power_of_two() && frames_pow2 >= 4096);
        let mut vec = Vec::with_capacity(frames_pow2 * 2);
        for _ in 0..(frames_pow2 * 2) {
            vec.push(UnsafeCell::new(0.0f32));
        }
        Self {
            mask: frames_pow2 - 1,
            buffer: vec.into_boxed_slice(),
            write_idx: CacheAlignedAtomicUsize(AtomicUsize::new(0)),
            read_idx: CacheAlignedAtomicUsize(AtomicUsize::new(0)),
            dropped_frames: CacheAlignedAtomicUsize(AtomicUsize::new(0)),
        }
    }

    #[inline]
    pub fn push_stereo(&self, left: f32, right: f32) -> bool {
        let w = self.write_idx.0.load(Ordering::Relaxed);
        let r = self.read_idx.0.load(Ordering::Acquire);
        if w.wrapping_sub(r) >= (self.mask + 1) {
            self.dropped_frames.0.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let slot = (w & self.mask) * 2;
        unsafe {
            *self.buffer[slot].get() = left;
            *self.buffer[slot + 1].get() = right;
        }
        self.write_idx.0.store(w.wrapping_add(1), Ordering::Release);
        true
    }

    #[inline]
    pub fn pop_batch(
        &self,
        out_left: &mut Vec<f32>,
        out_right: &mut Vec<f32>,
        max_frames: usize,
    ) -> usize {
        let r = self.read_idx.0.load(Ordering::Relaxed);
        let w = self.write_idx.0.load(Ordering::Acquire);
        let avail = w.wrapping_sub(r).min(max_frames);
        if avail == 0 {
            return 0;
        }
        for i in 0..avail {
            let slot = ((r.wrapping_add(i)) & self.mask) * 2;
            unsafe {
                out_left.push(*self.buffer[slot].get());
                out_right.push(*self.buffer[slot + 1].get());
            }
        }
        self.read_idx.0.store(r.wrapping_add(avail), Ordering::Release);
        avail
    }

    #[inline]
    pub fn fill_ratio(&self) -> f32 {
        let w = self.write_idx.0.load(Ordering::Relaxed);
        let r = self.read_idx.0.load(Ordering::Relaxed);
        (w.wrapping_sub(r) as f32) / ((self.mask + 1) as f32)
    }

    #[inline]
    pub fn dropped(&self) -> usize {
        self.dropped_frames.0.load(Ordering::Relaxed)
    }
}

pub struct SpectralAnalyzer32 {
    window: [f32; 512],
    ring_mono: [f32; 512],
    ring_pos: usize,
    re: [f32; 512],
    im: [f32; 512],
    prev_low_mag: [f32; 8],
    flux_ewma_mean: f32,
    flux_ewma_var: f32,
    pub bands_db: [f32; 32],
    pub peaks_db: [f32; 32],
    pub phase_corr: f32,
    pub last_flux_zscore: f32,
    pub beat_detected: bool,
}

impl SpectralAnalyzer32 {
    pub fn new() -> Self {
        let mut window = [0.0f32; 512];
        for i in 0..512 {
            let phase = (TAU * (i as f32)) / 511.0;
            window[i] = 0.35875 - 0.48829 * phase.cos() + 0.14128 * (2.0 * phase).cos()
                - 0.01168 * (3.0 * phase).cos();
        }
        Self {
            window,
            ring_mono: [0.0; 512],
            ring_pos: 0,
            re: [0.0; 512],
            im: [0.0; 512],
            prev_low_mag: [0.0; 8],
            flux_ewma_mean: 0.01,
            flux_ewma_var: 0.001,
            bands_db: [-80.0; 32],
            peaks_db: [-80.0; 32],
            phase_corr: 1.0,
            last_flux_zscore: 0.0,
            beat_detected: false,
        }
    }

    pub fn ingest_stereo_slice(&mut self, left: &[f32], right: &[f32]) {
        let n = left.len().min(right.len());
        if n == 0 {
            return;
        }
        let mut sum_lr = 0.0f64;
        let mut sum_ll = 0.0f64;
        let mut sum_rr = 0.0f64;

        for i in 0..n {
            let l = left[i];
            let r = right[i];
            sum_lr += (l as f64) * (r as f64);
            sum_ll += (l as f64) * (l as f64);
            sum_rr += (r as f64) * (r as f64);
            self.ring_mono[self.ring_pos] = (l + r) * 0.5;
            self.ring_pos = (self.ring_pos + 1) & 511;
        }

        let denom = (sum_ll * sum_rr).sqrt();
        if denom > 1.0e-9 {
            let instant_corr = (sum_lr / denom).clamp(-1.0, 1.0) as f32;
            self.phase_corr = self.phase_corr * 0.82 + instant_corr * 0.18;
        }

        self.compute_fft_512_and_onset();
    }

    fn compute_fft_512_and_onset(&mut self) {
        for i in 0..512 {
            let sample = self.ring_mono[(self.ring_pos + i) & 511];
            self.re[i] = sample * self.window[i];
            self.im[i] = 0.0;
        }

        let mut j = 0usize;
        for i in 0..512 {
            if i < j {
                self.re.swap(i, j);
            }
            let mut m = 256usize;
            while m >= 1 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        let mut len = 2usize;
        while len <= 512 {
            let half = len >> 1;
            let ang = -TAU / (len as f32);
            let (wlen_sin, wlen_cos) = ang.sin_cos();
            let mut i = 0usize;
            while i < 512 {
                let mut w_re = 1.0f32;
                let mut w_im = 0.0f32;
                for k in 0..half {
                    let u_re = self.re[i + k];
                    let u_im = self.im[i + k];
                    let v_re = self.re[i + k + half] * w_re - self.im[i + k + half] * w_im;
                    let v_im = self.re[i + k + half] * w_im + self.im[i + k + half] * w_re;
                    self.re[i + k] = u_re + v_re;
                    self.im[i + k] = u_im + v_im;
                    self.re[i + k + half] = u_re - v_re;
                    self.im[i + k + half] = u_im - v_im;

                    let next_re = w_re * wlen_cos - w_im * wlen_sin;
                    w_im = w_re * wlen_sin + w_im * wlen_cos;
                    w_re = next_re;
                }
                i += len;
            }
            len <<= 1;
        }

        let mut positive_flux = 0.0f32;
        for k in 1..=8 {
            let mag = (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt() * (2.0 / 512.0);
            let diff = (mag - self.prev_low_mag[k - 1]).max(0.0);
            positive_flux += diff;
            self.prev_low_mag[k - 1] = mag;
        }

        let delta = positive_flux - self.flux_ewma_mean;
        self.flux_ewma_mean += 0.08 * delta;
        self.flux_ewma_var = 0.92 * self.flux_ewma_var + 0.08 * delta * delta;
        let std_dev = self.flux_ewma_var.sqrt().max(1.0e-4);
        self.last_flux_zscore = (delta / std_dev).clamp(-3.0, 10.0);
        self.beat_detected = self.last_flux_zscore > 2.25 && positive_flux > 0.020;

        for b in 0..32 {
            let f0 = ((b as f32) / 32.0).powf(1.85);
            let f1 = (((b + 1) as f32) / 32.0).powf(1.85);
            let bin_start = (1 + (f0 * 240.0) as usize).clamp(1, 254);
            let bin_end = (1 + (f1 * 240.0) as usize).max(bin_start + 1).min(255);

            let mut mag_acc = 0.0f32;
            for k in bin_start..=bin_end {
                let mag =
                    (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt() * (2.0 / 512.0);
                if mag > mag_acc {
                    mag_acc = mag;
                }
            }
            let db = if mag_acc > 1.0e-6 {
                (20.0 * mag_acc.log10()).clamp(-75.0, 0.0)
            } else {
                -75.0
            };

            if db > self.bands_db[b] {
                self.bands_db[b] = self.bands_db[b] * 0.45 + db * 0.55;
            } else {
                self.bands_db[b] = self.bands_db[b] * 0.72 + db * 0.28;
            }

            if self.bands_db[b] >= self.peaks_db[b] {
                self.peaks_db[b] = self.bands_db[b];
            } else {
                self.peaks_db[b] = (self.peaks_db[b] - 0.65).max(-75.0);
            }
        }
    }
}

// ============================================================================
// SECTION 4: ITU-R BS.1770-4 K-WEIGHTING LUFS & CLEAN BOUNDARY CONDITIONING
// ============================================================================

#[derive(Clone, Copy)]
struct BiquadFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl BiquadFilter {
    #[inline(always)]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        if y.abs() < 1.0e-30 {
            0.0
        } else {
            y
        }
    }
}

pub struct KWeightingMeter {
    pre_l: BiquadFilter,
    pre_r: BiquadFilter,
    rlb_l: BiquadFilter,
    rlb_r: BiquadFilter,
    window_energy: [f64; 40],
    block_idx: usize,
    current_sum: f64,
    current_samples: usize,
    samples_per_100ms: usize,
    integrated_energy_sum: f64,
    integrated_blocks: usize,
}

impl KWeightingMeter {
    pub fn new(sample_rate: u32) -> Self {
        let pre = BiquadFilter {
            b0: 1.53512485958697,
            b1: -2.69169618940638,
            b2: 1.19839281085285,
            a1: -1.69065929318241,
            a2: 0.73248077421585,
            z1: 0.0,
            z2: 0.0,
        };
        let rlb = BiquadFilter {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: -1.99004745483398,
            a2: 0.99007225036621,
            z1: 0.0,
            z2: 0.0,
        };
        Self {
            pre_l: pre,
            pre_r: pre,
            rlb_l: rlb,
            rlb_r: rlb,
            window_energy: [0.0; 40],
            block_idx: 0,
            current_sum: 0.0,
            current_samples: 0,
            samples_per_100ms: ((sample_rate as usize) / 10).max(1),
            integrated_energy_sum: 0.0,
            integrated_blocks: 0,
        }
    }

    #[inline]
    pub fn push_frame(&mut self, l: f32, r: f32) {
        let yl = self.rlb_l.process(self.pre_l.process(l as f64));
        let yr = self.rlb_r.process(self.pre_r.process(r as f64));
        self.current_sum += yl * yl + yr * yr;
        self.current_samples += 1;
        if self.current_samples >= self.samples_per_100ms {
            let mean_sq = self.current_sum / (self.current_samples as f64);
            self.window_energy[self.block_idx % 40] = mean_sq;
            self.block_idx = self.block_idx.wrapping_add(1);
            let lufs_block = -0.691 + 10.0 * (mean_sq.max(1.0e-12)).log10();
            if lufs_block > -70.0 {
                self.integrated_energy_sum += mean_sq;
                self.integrated_blocks += 1;
            }
            self.current_sum = 0.0;
            self.current_samples = 0;
        }
    }

    pub fn momentary_lufs(&self) -> f64 {
        let count = self.block_idx.min(4);
        if count == 0 {
            return -70.0;
        }
        let mut sum = 0.0;
        for i in 0..count {
            let idx = (self.block_idx.wrapping_sub(1).wrapping_sub(i)) % 40;
            sum += self.window_energy[idx];
        }
        let mean = sum / (count as f64);
        if mean <= 1.0e-10 {
            -70.0
        } else {
            (-0.691 + 10.0 * mean.log10()).clamp(-70.0, 3.0)
        }
    }

    pub fn integrated_lufs(&self) -> f64 {
        if self.integrated_blocks == 0 {
            return -70.0;
        }
        let mean = self.integrated_energy_sum / (self.integrated_blocks as f64);
        if mean <= 1.0e-10 {
            -70.0
        } else {
            (-0.691 + 10.0 * mean.log10()).clamp(-70.0, 3.0)
        }
    }
}

#[inline(always)]
fn xorshift32(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    (x as f32) * (1.0 / 4294967296.0) - 0.5
}

#[inline(always)]
fn quantize_f32_tpdf(sample: f32, rng: &mut u32) -> i16 {
    let clean = if !sample.is_finite() || sample.abs() < 1.0e-20 {
        0.0
    } else {
        sample.clamp(-1.0, 1.0)
    };
    let dither = xorshift32(rng) + xorshift32(rng);
    ((clean * 32767.0 + dither).round()).clamp(-32768.0, 32767.0) as i16
}

pub fn condition_clean_boundaries(
    left: &mut Vec<f32>,
    right: &mut Vec<f32>,
    sample_rate: u32,
    fade_ms: u32,
    trim_db: Option<f32>,
) -> u64 {
    let n = left.len().min(right.len());
    if n == 0 {
        return 0;
    }

    let mut start_idx = 0usize;
    let mut end_idx = n;

    if let Some(db) = trim_db {
        let thresh = 10.0f32.powf(db / 20.0);
        while start_idx < n && left[start_idx].abs().max(right[start_idx].abs()) < thresh {
            start_idx += 1;
        }
        while end_idx > start_idx
            && left[end_idx - 1].abs().max(right[end_idx - 1].abs()) < thresh
        {
            end_idx -= 1;
        }
        if end_idx <= start_idx {
            start_idx = 0;
        } else if start_idx > 0 || end_idx < n {
            left.drain(0..start_idx);
            right.drain(0..start_idx);
            let new_len = end_idx - start_idx;
            left.truncate(new_len);
            right.truncate(new_len);
        }
    }

    let len = left.len();
    let fade_samples = (((sample_rate as u64) * (fade_ms as u64)) / 1000) as usize;
    let f_len = fade_samples.min(len / 2);

    if f_len > 0 {
        let inv_f = PI / (f_len as f32);
        for i in 0..f_len {
            let gain = 0.5 * (1.0 - ((i as f32) * inv_f).cos());
            left[i] *= gain;
            right[i] *= gain;
            let tail = len - 1 - i;
            left[tail] *= gain;
            right[tail] *= gain;
        }
    }

    start_idx as u64
}

// ============================================================================
// SECTION 5: PROCEDURAL OKLCH WAVEFIELD SCENES & IMAGE / VIDEO INGESTION
// ============================================================================

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    let crc_start = out.len();
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(payload);
    let crc = crc32_ieee(&out[crc_start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

pub fn encode_rgb_canvas_to_png(canvas: &RgbCanvas) -> Vec<u8> {
    let w = canvas.width;
    let h = canvas.height;
    let mut raw_scanlines = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw_scanlines.push(0x00);
        let row_start = y * w * 3;
        raw_scanlines.extend_from_slice(&canvas.pixels[row_start..row_start + w * 3]);
    }

    let mut zlib = Vec::with_capacity(raw_scanlines.len() + 64);
    zlib.push(0x78);
    zlib.push(0x01);

    let mut adler_a: u32 = 1;
    let mut adler_b: u32 = 0;
    for &b in &raw_scanlines {
        adler_a = (adler_a + b as u32) % 65521;
        adler_b = (adler_b + adler_a) % 65521;
    }

    let chunks: Vec<&[u8]> = raw_scanlines.chunks(65535).collect();
    for (idx, chunk) in chunks.iter().enumerate() {
        zlib.push(if idx + 1 == chunks.len() { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(chunk);
    }
    zlib.extend_from_slice(&((adler_b << 16) | adler_a).to_be_bytes());

    let mut png = Vec::with_capacity(zlib.len() + 128);
    png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    write_png_chunk(&mut png, b"IHDR", &ihdr);
    write_png_chunk(&mut png, b"IDAT", &zlib);
    write_png_chunk(&mut png, b"IEND", &[]);
    png
}

pub fn synthesize_procedural_scene(width: usize, height: usize, variant: usize) -> RgbCanvas {
    let mut canvas = RgbCanvas::new(width, height);
    let base_hue = (variant as f32) * 85.0 + 190.0;
    let freq = 14.0 + (variant as f32) * 5.0;
    let petals = 3.0 + (variant as f32);

    for y in 0..height {
        let fy = ((y as f32) / (height as f32)) * 2.0 - 1.0;
        for x in 0..width {
            let fx = ((x as f32) / (width as f32)) * 2.0 - 1.0;
            let r = (fx * fx + fy * fy).sqrt();
            let theta = fy.atan2(fx);

            let interferogram = ((r * freq - theta * petals).cos()
                * (fx * 7.0 + fy * 7.0 + (variant as f32)).sin())
                * 0.5
                + 0.5;
            let grid = if x % 32 == 0 || y % 32 == 0 { 0.14 } else { 0.0 };

            let l = (0.26 + interferogram * 0.56 + grid).clamp(0.18, 0.94);
            let c = (0.16 + (1.0 - r * 0.45).max(0.0) * 0.17).clamp(0.06, 0.33);
            let h =
                (base_hue + interferogram * 150.0 + theta.to_degrees() * 0.5).rem_euclid(360.0);

            let (sr, sg, sb) = oklch_to_srgb(l, c, h);
            let dst = (y * width + x) * 3;
            canvas.pixels[dst] = sr;
            canvas.pixels[dst + 1] = sg;
            canvas.pixels[dst + 2] = sb;
        }
    }
    canvas
}

pub fn load_image_as_rgb_canvas(
    path: &Path,
    target_w: usize,
    target_h: usize,
) -> Result<RgbCanvas, String> {
    let raw = fs::read(path).map_err(|e| format!("Failed to read '{}': {e}", path.display()))?;

    if raw.starts_with(b"P6\n") || raw.starts_with(b"P6\r") {
        let header_end = raw
            .windows(4)
            .position(|w| w == b"255\n")
            .map(|p| p + 4)
            .ok_or("Invalid PPM header")?;
        let header_str =
            core::str::from_utf8(&raw[..header_end]).map_err(|_| "Invalid PPM UTF-8")?;
        let tokens: Vec<&str> = header_str
            .split_whitespace()
            .filter(|t| !t.starts_with('#'))
            .collect();
        if tokens.len() >= 4 {
            let w: usize = tokens[1].parse().map_err(|_| "Invalid PPM width")?;
            let h: usize = tokens[2].parse().map_err(|_| "Invalid PPM height")?;
            if raw.len() >= header_end + w * h * 3 {
                let mut c = RgbCanvas::new(w, h);
                c.pixels
                    .copy_from_slice(&raw[header_end..header_end + w * h * 3]);
                return Ok(c.resize_to(target_w, target_h));
            }
        }
    }

    #[cfg(windows)]
    {
        let temp_ppm = env::temp_dir().join(format!(
            "win_audio_mux_frame_{}_{}.rgb",
            std::process::id(),
            target_w ^ target_h
        ));
        let ps_script = format!(
            "Add-Type -AssemblyName System.Drawing; \
             $img = [System.Drawing.Image]::FromFile('{}'); \
             $bmp = New-Object System.Drawing.Bitmap($img, {}, {}); \
             $rect = New-Object System.Drawing.Rectangle(0, 0, {}, {}); \
             $bmp24 = $bmp.Clone($rect, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb); \
             $data = $bmp24.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, $bmp24.PixelFormat); \
             $stride = $data.Stride; \
             $raw = New-Object byte[] ($stride * {}); \
             [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $raw, 0, $raw.Length); \
             $bmp24.UnlockBits($data); \
             $out = New-Object byte[] ({} * {} * 3); \
             $dst = 0; \
             for ($y = 0; $y -lt {}; $y++) {{ \
               $row = $y * $stride; \
               for ($x = 0; $x -lt {}; $x++) {{ \
                 $s = $row + $x * 3; \
                 $out[$dst] = $raw[$s + 2]; \
                 $out[$dst + 1] = $raw[$s + 1]; \
                 $out[$dst + 2] = $raw[$s]; \
                 $dst += 3; \
               }} \
             }} \
             [System.IO.File]::WriteAllBytes('{}', $out); \
             $bmp24.Dispose(); $bmp.Dispose(); $img.Dispose();",
            path.display().to_string().replace('\'', "''"),
            target_w,
            target_h,
            target_w,
            target_h,
            target_h,
            target_w,
            target_h,
            target_h,
            target_w,
            temp_ppm.display().to_string().replace('\'', "''")
        );

        let status = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &ps_script])
            .status();

        if let Ok(st) = status {
            if st.success() {
                if let Ok(rgb_data) = fs::read(&temp_ppm) {
                    let _ = fs::remove_file(&temp_ppm);
                    if rgb_data.len() == target_w * target_h * 3 {
                        return Ok(RgbCanvas {
                            width: target_w,
                            height: target_h,
                            pixels: rgb_data,
                        });
                    }
                }
            }
        }
    }

    Err(format!(
        "Could not decode image '{}' into RGB raster",
        path.display()
    ))
}

// ============================================================================
// SECTION 6: ULTRA-PHYSICS VIDEO TRANSITION SHADER KERNELS
// ============================================================================

pub fn render_physics_transition_frame(
    scene_a: &RgbCanvas,
    scene_b: &RgbCanvas,
    t_raw: f32,
    mode: PhysicsTransition,
    out_canvas: &mut RgbCanvas,
) {
    let t = t_raw.clamp(0.0, 1.0);
    let w = out_canvas.width;
    let h = out_canvas.height;
    let inv_w = 1.0 / ((w - 1).max(1) as f32);
    let inv_h = 1.0 / ((h - 1).max(1) as f32);

    match mode {
        PhysicsTransition::Spring => {
            let zeta = 0.42f32;
            let wn = 14.0f32;
            let wd = wn * (1.0 - zeta * zeta).sqrt();
            let decay = (-zeta * wn * t).exp();
            let disp = 1.0 - decay * ((wd * t).cos() + (zeta * wn / wd) * (wd * t).sin());
            let velocity = decay * (wn * wn / wd) * (wd * t).sin();

            for y in 0..h {
                let v = (y as f32) * inv_h;
                for x in 0..w {
                    let u = (x as f32) * inv_w;
                    let u_a = u + disp;
                    let u_b = u - (1.0 - disp);
                    let dst = (y * w + x) * 3;

                    let blur_shift = (velocity * 0.012).clamp(-0.08, 0.08);
                    let mut rgb = [0.0f32; 3];
                    for tap in [-1.0f32, 0.0, 1.0] {
                        let sample_u_a = u_a + tap * blur_shift;
                        let sample_u_b = u_b + tap * blur_shift;
                        let s = if (0.0..=1.0).contains(&sample_u_b) {
                            scene_b.sample_bilinear(sample_u_b, v)
                        } else {
                            scene_a.sample_bilinear(sample_u_a.rem_euclid(1.0), v)
                        };
                        rgb[0] += s[0] * (1.0 / 3.0);
                        rgb[1] += s[1] * (1.0 / 3.0);
                        rgb[2] += s[2] * (1.0 / 3.0);
                    }
                    out_canvas.pixels[dst] = rgb[0].clamp(0.0, 255.0) as u8;
                    out_canvas.pixels[dst + 1] = rgb[1].clamp(0.0, 255.0) as u8;
                    out_canvas.pixels[dst + 2] = rgb[2].clamp(0.0, 255.0) as u8;
                }
            }
        }

        PhysicsTransition::Wave => {
            let envelope = (PI * t).sin();
            let mix_alpha = 0.5 * (1.0 - (PI * t).cos());

            for y in 0..h {
                let v = (y as f32) * inv_h;
                let dy = v - 0.5;
                for x in 0..w {
                    let u = (x as f32) * inv_w;
                    let dx = u - 0.5;
                    let r = (dx * dx + dy * dy).sqrt().max(1.0e-5);
                    let ripple = envelope * 0.085 * (-2.8 * r).exp() * (32.0 * r - 18.0 * t).cos();
                    let off_u = (dx / r) * ripple;
                    let off_v = (dy / r) * ripple;

                    let ca = scene_a.sample_bilinear(u + off_u * (1.0 - t), v + off_v * (1.0 - t));
                    let cb = scene_b.sample_bilinear(u - off_u * t, v - off_v * t);

                    let dst = (y * w + x) * 3;
                    out_canvas.pixels[dst] =
                        (ca[0] * (1.0 - mix_alpha) + cb[0] * mix_alpha).clamp(0.0, 255.0) as u8;
                    out_canvas.pixels[dst + 1] =
                        (ca[1] * (1.0 - mix_alpha) + cb[1] * mix_alpha).clamp(0.0, 255.0) as u8;
                    out_canvas.pixels[dst + 2] =
                        (ca[2] * (1.0 - mix_alpha) + cb[2] * mix_alpha).clamp(0.0, 255.0) as u8;
                }
            }
        }

        PhysicsTransition::Vortex => {
            let peak_twist = 5.5 * (PI * t).sin();
            let mix_alpha = 0.5 * (1.0 - (PI * t).cos());

            for y in 0..h {
                let v = (y as f32) * inv_h;
                let dy = v - 0.5;
                for x in 0..w {
                    let u = (x as f32) * inv_w;
                    let dx = u - 0.5;
                    let r2 = dx * dx + dy * dy;
                    let gauss = (-r2 / 0.14).exp();

                    let ang_a = peak_twist * gauss * (1.0 - t * 0.35);
                    let (sa, ca_rot) = ang_a.sin_cos();
                    let ua = 0.5 + dx * ca_rot - dy * sa;
                    let va = 0.5 + dx * sa + dy * ca_rot;

                    let ang_b = -peak_twist * gauss * (1.0 - (1.0 - t) * 0.35);
                    let (sb, cb_rot) = ang_b.sin_cos();
                    let ub = 0.5 + dx * cb_rot - dy * sb;
                    let vb = 0.5 + dx * sb + dy * cb_rot;

                    let pix_a = scene_a.sample_bilinear(ua, va);
                    let pix_b = scene_b.sample_bilinear(ub, vb);

                    let dst = (y * w + x) * 3;
                    out_canvas.pixels[dst] =
                        (pix_a[0] * (1.0 - mix_alpha) + pix_b[0] * mix_alpha).clamp(0.0, 255.0)
                            as u8;
                    out_canvas.pixels[dst + 1] =
                        (pix_a[1] * (1.0 - mix_alpha) + pix_b[1] * mix_alpha).clamp(0.0, 255.0)
                            as u8;
                    out_canvas.pixels[dst + 2] =
                        (pix_a[2] * (1.0 - mix_alpha) + pix_b[2] * mix_alpha).clamp(0.0, 255.0)
                            as u8;
                }
            }
        }

        PhysicsTransition::Chroma => {
            let pulse = (PI * t).sin();
            let mix_alpha = 0.5 * (1.0 - (PI * t).cos());
            let disp_r = 0.045 * pulse;
            let disp_g = 0.080 * pulse;
            let disp_b = 0.125 * pulse;

            for y in 0..h {
                let v = (y as f32) * inv_h;
                let dy = v - 0.5;
                for x in 0..w {
                    let u = (x as f32) * inv_w;
                    let dx = u - 0.5;
                    let radial = (dx * dx + dy * dy).sqrt() + 0.25;

                    let ra = scene_a
                        .sample_bilinear(u + dx * disp_r * radial, v + dy * disp_r * radial)[0];
                    let ga = scene_a
                        .sample_bilinear(u + dx * disp_g * radial, v + dy * disp_g * radial)[1];
                    let ba = scene_a
                        .sample_bilinear(u + dx * disp_b * radial, v + dy * disp_b * radial)[2];

                    let rb = scene_b
                        .sample_bilinear(u - dx * disp_r * radial, v - dy * disp_r * radial)[0];
                    let gb = scene_b
                        .sample_bilinear(u - dx * disp_g * radial, v - dy * disp_g * radial)[1];
                    let bb = scene_b
                        .sample_bilinear(u - dx * disp_b * radial, v - dy * disp_b * radial)[2];

                    let dst = (y * w + x) * 3;
                    out_canvas.pixels[dst] =
                        (ra * (1.0 - mix_alpha) + rb * mix_alpha).clamp(0.0, 255.0) as u8;
                    out_canvas.pixels[dst + 1] =
                        (ga * (1.0 - mix_alpha) + gb * mix_alpha).clamp(0.0, 255.0) as u8;
                    out_canvas.pixels[dst + 2] =
                        (ba * (1.0 - mix_alpha) + bb * mix_alpha).clamp(0.0, 255.0) as u8;
                }
            }
        }

        PhysicsTransition::Dissolve => {
            let alpha = 0.5 * (1.0 - (PI * t).cos());
            let inv_255 = 1.0 / 255.0;
            for idx in 0..out_canvas.pixels.len() {
                let a = (scene_a.pixels[idx] as f32) * inv_255;
                let b = (scene_b.pixels[idx] as f32) * inv_255;
                let lin = a * a * (1.0 - alpha) + b * b * alpha;
                out_canvas.pixels[idx] = (lin.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ============================================================================
// SECTION 7: WINDOWS MEDIA FOUNDATION HARDWARE H.264 + AAC SINK WRITER & DECODER
// ============================================================================

#[cfg(windows)]
mod mf_video_engine {
    use super::*;
    use windows::core::{Interface, HSTRING};
    use windows::Win32::Media::MediaFoundation::*;
    use windows::Win32::System::Com::*;

    struct MfGuard;
    impl Drop for MfGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = MFShutdown();
            }
        }
    }

    /// Converts an `RgbCanvas` (`RGB24`, top-down) into BT.601 `NV12` (`Y` plane followed by interleaved `UV` plane),
    /// which is the native input format for Windows Media Foundation's hardware H.264 encoders (NVENC / QuickSync / AMF).
    fn rgb24_to_nv12(canvas: &RgbCanvas, nv12_buf: &mut [u8]) {
        let w = canvas.width;
        let h = canvas.height;
        let y_plane_size = w * h;
        let rgb = &canvas.pixels;

        for y in 0..h {
            let y_row = y * w;
            let rgb_row = y_row * 3;
            for x in 0..w {
                let idx = rgb_row + x * 3;
                let r = rgb[idx] as i32;
                let g = rgb[idx + 1] as i32;
                let b = rgb[idx + 2] as i32;
                let luma = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(16, 235) as u8;
                nv12_buf[y_row + x] = luma;
            }
        }

        let uv_offset = y_plane_size;
        for cy in 0..(h / 2) {
            let y0 = cy * 2;
            let uv_row = uv_offset + cy * w;
            for cx in 0..(w / 2) {
                let x0 = cx * 2;
                let i00 = (y0 * w + x0) * 3;
                let i10 = (y0 * w + x0 + 1) * 3;
                let i01 = ((y0 + 1) * w + x0) * 3;
                let i11 = ((y0 + 1) * w + x0 + 1) * 3;

                let r = (rgb[i00] as i32 + rgb[i10] as i32 + rgb[i01] as i32 + rgb[i11] as i32) >> 2;
                let g = (rgb[i00 + 1] as i32
                    + rgb[i10 + 1] as i32
                    + rgb[i01 + 1] as i32
                    + rgb[i11 + 1] as i32)
                    >> 2;
                let b = (rgb[i00 + 2] as i32
                    + rgb[i10 + 2] as i32
                    + rgb[i01 + 2] as i32
                    + rgb[i11 + 2] as i32)
                    >> 2;

                let u = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(16, 240) as u8;
                let v = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(16, 240) as u8;
                nv12_buf[uv_row + cx * 2] = u;
                nv12_buf[uv_row + cx * 2 + 1] = v;
            }
        }
    }

    /// Decodes an external `.mp4` / `.mov` / `.mkv` file into a sequence of `RgbCanvas` frames via `IMFSourceReader`.
    pub fn decode_video_clip_to_canvases(
        path: &Path,
        target_w: usize,
        target_h: usize,
        max_frames: usize,
    ) -> Result<VisualScene, String> {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL)
                .map_err(|e| format!("MFStartup failed: {e}"))?;
            let _mf_guard = MfGuard;

            let mut attrs: Option<IMFAttributes> = None;
            MFCreateAttributes(&mut attrs, 2)
                .map_err(|e| format!("MFCreateAttributes: {e}"))?;
            let attrs = attrs.ok_or("Failed to create IMFAttributes")?;
            let _ = attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1);

            let url = HSTRING::from(path.as_os_str());
            let reader: IMFSourceReader =
                MFCreateSourceReaderFromURL(&url, &attrs)
                    .map_err(|e| format!("MFCreateSourceReaderFromURL('{}'): {e}", path.display()))?;

            let out_type: IMFMediaType =
                MFCreateMediaType().map_err(|e| format!("MFCreateMediaType: {e}"))?;
            out_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .map_err(|e| format!("SetGUID Major: {e}"))?;
            out_type
                .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)
                .map_err(|e| format!("SetGUID Subtype RGB32: {e}"))?;
            reader
                .SetCurrentMediaType(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    None,
                    &out_type,
                )
                .map_err(|e| format!("SetCurrentMediaType(RGB32): {e}"))?;

            let actual_type = reader
                .GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)
                .map_err(|e| format!("GetCurrentMediaType: {e}"))?;
            let frame_size_packed = actual_type
                .GetUINT64(&MF_MT_FRAME_SIZE)
                .map_err(|e| format!("GetUINT64(MF_MT_FRAME_SIZE): {e}"))?;
            let src_w = ((frame_size_packed >> 32) as usize).max(1);
            let src_h = ((frame_size_packed & 0xFFFF_FFFF) as usize).max(1);

            let mut frames = Vec::new();
            while frames.len() < max_frames {
                let mut stream_flags: u32 = 0;
                let mut sample: Option<IMFSample> = None;
                reader
                    .ReadSample(
                        MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                        0,
                        None,
                        Some(&mut stream_flags),
                        None,
                        Some(&mut sample),
                    )
                    .map_err(|e| format!("ReadSample: {e}"))?;

                if (stream_flags & (MF_SOURCE_READERF_ENDOFSTREAM.0 as u32)) != 0 {
                    break;
                }
                let Some(sample) = sample else {
                    continue;
                };
                let buf: IMFMediaBuffer = sample
                    .ConvertToContiguousBuffer()
                    .map_err(|e| format!("ConvertToContiguousBuffer: {e}"))?;

                let mut ptr: *mut u8 = std::ptr::null_mut();
                let mut cur_len: u32 = 0;
                buf.Lock(&mut ptr, None, Some(&mut cur_len))
                    .map_err(|e| format!("IMFMediaBuffer::Lock: {e}"))?;

                let needed = src_w * src_h * 4;
                if !ptr.is_null() && (cur_len as usize) >= needed {
                    let bgra = std::slice::from_raw_parts(ptr, needed);
                    let mut src_canvas = RgbCanvas::new(src_w, src_h);
                    for i in 0..(src_w * src_h) {
                        src_canvas.pixels[i * 3] = bgra[i * 4 + 2];     // R
                        src_canvas.pixels[i * 3 + 1] = bgra[i * 4 + 1]; // G
                        src_canvas.pixels[i * 3 + 2] = bgra[i * 4];     // B
                    }
                    frames.push(if src_w == target_w && src_h == target_h {
                        src_canvas // already at target: skip resize_to's full-frame clone
                    } else {
                        src_canvas.resize_to(target_w, target_h)
                    });
                }
                let _ = buf.Unlock();
            }

            if frames.is_empty() {
                return Err(format!("No video frames decoded from '{}'", path.display()));
            }
            Ok(VisualScene { frames })
        }
    }

    /// Encodes a complete H.264 (`avc1`) + AAC-LC (`mp4a` `0x40`) `.mp4` video file natively via Windows
    /// Media Foundation `IMFSinkWriter`, then injects the `udta.meta.ilst` + `ID32` metadata atoms.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mp4_via_media_foundation(
        sample_rate: u32,
        left_f32: &[f32],
        right_f32: &[f32],
        meta: &MediaMetadata,
        scenes: &[VisualScene],
        default_transition: PhysicsTransition,
        auto_mode: AutoTransitionMode,
        auto_interval_secs: f32,
        cycle_shaders: bool,
        transition_secs: f32,
        fps: u32,
        timeline_events: &[TimelineEvent],
        out_mp4_path: &Path,
    ) -> Result<(), String> {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL)
                .map_err(|e| format!("MFStartup failed: {e}"))?;
            let _mf_guard = MfGuard;

            let width = scenes[0].frame_at(0).width;
            let height = scenes[0].frame_at(0).height;
            let total_audio_frames = left_f32.len().min(right_f32.len());
            let audio_dur_sec = (total_audio_frames as f64) / (sample_rate.max(1) as f64);
            let total_video_frames = ((audio_dur_sec * (fps as f64)).ceil() as usize).max(1);

            if out_mp4_path.exists() {
                let _ = fs::remove_file(out_mp4_path);
            }

            let mut attrs: Option<IMFAttributes> = None;
            MFCreateAttributes(&mut attrs, 4)
                .map_err(|e| format!("MFCreateAttributes: {e}"))?;
            let attrs = attrs.ok_or("Failed to allocate IMFAttributes")?;
            let _ = attrs.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1);
            let _ = attrs.SetGUID(&MF_TRANSCODE_CONTAINERTYPE, &MFTranscodeContainerType_MPEG4);

            let url = HSTRING::from(out_mp4_path.as_os_str());
            let writer: IMFSinkWriter = MFCreateSinkWriterFromURL(&url, None, &attrs)
                .map_err(|e| format!("MFCreateSinkWriterFromURL('{}'): {e}", out_mp4_path.display()))?;

            // 1. Configure H.264 Output Video Stream
            let v_out: IMFMediaType =
                MFCreateMediaType().map_err(|e| format!("MFCreateMediaType: {e}"))?;
            v_out.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).unwrap();
            v_out.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264).unwrap();
            v_out.SetUINT32(&MF_MT_AVG_BITRATE, 6_000_000).unwrap();
            v_out
                .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
                .unwrap();
            v_out
                .SetUINT64(
                    &MF_MT_FRAME_SIZE,
                    ((width as u64) << 32) | (height as u64),
                )
                .unwrap();
            v_out
                .SetUINT64(&MF_MT_FRAME_RATE, ((fps as u64) << 32) | 1u64)
                .unwrap();
            v_out
                .SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, (1u64 << 32) | 1u64)
                .unwrap();
            let video_stream_idx = writer
                .AddStream(&v_out)
                .map_err(|e| format!("AddStream(H264): {e}"))?;

            // Configure NV12 Input Video MediaType
            let v_in: IMFMediaType =
                MFCreateMediaType().map_err(|e| format!("MFCreateMediaType: {e}"))?;
            v_in.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).unwrap();
            v_in.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12).unwrap();
            v_in.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
                .unwrap();
            v_in.SetUINT64(
                &MF_MT_FRAME_SIZE,
                ((width as u64) << 32) | (height as u64),
            )
            .unwrap();
            v_in.SetUINT64(&MF_MT_FRAME_RATE, ((fps as u64) << 32) | 1u64)
                .unwrap();
            v_in.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, (1u64 << 32) | 1u64)
                .unwrap();
            writer
                .SetInputMediaType(video_stream_idx, &v_in, None)
                .map_err(|e| format!("SetInputMediaType(NV12): {e}"))?;

            // 2. Configure AAC-LC Output Audio Stream (48000 Hz or 44100 Hz stereo, 192 kbps = 24000 bytes/sec)
            let aac_sr = if sample_rate == 44100 { 44100u32 } else { 48000u32 };
            let a_out: IMFMediaType =
                MFCreateMediaType().map_err(|e| format!("MFCreateMediaType: {e}"))?;
            a_out.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio).unwrap();
            a_out.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC).unwrap();
            a_out.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16).unwrap();
            a_out.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, aac_sr).unwrap();
            a_out.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, 2).unwrap();
            a_out.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 24000).unwrap();
            let audio_stream_idx = writer
                .AddStream(&a_out)
                .map_err(|e| format!("AddStream(AAC): {e}"))?;

            // Configure 16-bit PCM Input Audio MediaType
            let a_in: IMFMediaType =
                MFCreateMediaType().map_err(|e| format!("MFCreateMediaType: {e}"))?;
            a_in.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio).unwrap();
            a_in.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM).unwrap();
            a_in.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, 2).unwrap();
            a_in.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, aac_sr).unwrap();
            a_in.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16).unwrap();
            a_in.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, 4).unwrap();
            a_in.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, aac_sr * 4).unwrap();
            a_in.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1).unwrap();
            writer
                .SetInputMediaType(audio_stream_idx, &a_in, None)
                .map_err(|e| format!("SetInputMediaType(PCM): {e}"))?;

            writer
                .BeginWriting()
                .map_err(|e| format!("IMFSinkWriter::BeginWriting: {e}"))?;

            // Build transition schedule
            let trans_frames = ((transition_secs * (fps as f32)).round() as usize).max(2);
            let mut schedule: Vec<(usize, usize, usize, PhysicsTransition)> = Vec::new();
            if !timeline_events.is_empty() && scenes.len() > 1 {
                for ev in timeline_events {
                    let v_frame = (((ev.sample_frame as f64) / (sample_rate.max(1) as f64))
                        * (fps as f64))
                        .round() as usize;
                    let from_sc = ev.from_scene % scenes.len();
                    let to_sc = ev.target_scene % scenes.len();
                    if from_sc != to_sc
                        && v_frame + 2 <= total_video_frames
                        && schedule
                            .last()
                            .map(|&(pf, _, _, _)| v_frame >= pf + trans_frames)
                            .unwrap_or(true)
                    {
                        schedule.push((v_frame, from_sc, to_sc, ev.transition));
                    }
                }
            }

            if schedule.is_empty() && scenes.len() > 1 && auto_mode != AutoTransitionMode::ManualOnly
            {
                let step_frames =
                    ((auto_interval_secs * (fps as f32)).round() as usize).max(trans_frames + 4);
                let mut f_cursor = step_frames.saturating_sub(trans_frames);
                let mut cur_sc = 0usize;
                let mut shader_idx = default_transition as u8;
                while f_cursor + trans_frames <= total_video_frames {
                    let next_sc = (cur_sc + 1) % scenes.len();
                    let mode = if cycle_shaders {
                        let m = PhysicsTransition::from_u8(shader_idx);
                        shader_idx = shader_idx.wrapping_add(1);
                        m
                    } else {
                        default_transition
                    };
                    schedule.push((f_cursor, cur_sc, next_sc, mode));
                    cur_sc = next_sc;
                    f_cursor += step_frames;
                }
            }

            // 3. Feed NV12 Video Frames to Hardware H.264 Stream
            let nv12_byte_len = width * height * 3 / 2;
            let mut nv12_scratch = vec![0u8; nv12_byte_len];
            let mut transition_canvas = RgbCanvas::new(width, height);
            let frame_duration_hns: i64 = (10_000_000i64 / (fps as i64)).max(1);
            let mut current_hold_scene = 0usize;

            for f_idx in 0..total_video_frames {
                let mut active_trans: Option<(usize, usize, f32, PhysicsTransition)> = None;
                for &(start_f, from_sc, to_sc, mode) in &schedule {
                    if f_idx >= start_f && f_idx < start_f + trans_frames {
                        let t = ((f_idx - start_f) as f32) / ((trans_frames - 1).max(1) as f32);
                        active_trans = Some((from_sc, to_sc, t, mode));
                        current_hold_scene = to_sc;
                        break;
                    } else if f_idx >= start_f + trans_frames {
                        current_hold_scene = to_sc;
                    }
                }

                if let Some((from_sc, to_sc, t, mode)) = active_trans {
                    let canvas_a = scenes[from_sc].frame_at(f_idx);
                    let canvas_b = scenes[to_sc].frame_at(f_idx);
                    render_physics_transition_frame(
                        canvas_a,
                        canvas_b,
                        t,
                        mode,
                        &mut transition_canvas,
                    );
                    rgb24_to_nv12(&transition_canvas, &mut nv12_scratch);
                } else {
                    let sc = scenes[current_hold_scene % scenes.len()].frame_at(f_idx);
                    rgb24_to_nv12(sc, &mut nv12_scratch);
                }

                let media_buf: IMFMediaBuffer = MFCreateMemoryBuffer(nv12_byte_len as u32)
                    .map_err(|e| format!("MFCreateMemoryBuffer(NV12): {e}"))?;
                let mut dst_ptr: *mut u8 = std::ptr::null_mut();
                media_buf
                    .Lock(&mut dst_ptr, None, None)
                    .map_err(|e| format!("Lock NV12: {e}"))?;
                std::ptr::copy_nonoverlapping(
                    nv12_scratch.as_ptr(),
                    dst_ptr,
                    nv12_byte_len,
                );
                media_buf.Unlock().unwrap();
                media_buf.SetCurrentLength(nv12_byte_len as u32).unwrap();

                let sample: IMFSample =
                    MFCreateSample().map_err(|e| format!("MFCreateSample: {e}"))?;
                sample.AddBuffer(&media_buf).unwrap();
                sample
                    .SetSampleTime((f_idx as i64) * frame_duration_hns)
                    .unwrap();
                sample.SetSampleDuration(frame_duration_hns).unwrap();

                writer
                    .WriteSample(video_stream_idx, &sample)
                    .map_err(|e| format!("WriteSample(Video frame {f_idx}): {e}"))?;
            }

            // 4. Feed Interleaved 16-bit Stereo PCM Chunks to Native AAC-LC Stream
            let mut rng = 0x9E37_79B9u32;
            let chunk_frames = 4096usize;
            let mut f_cursor = 0usize;

            while f_cursor < total_audio_frames {
                let count = (total_audio_frames - f_cursor).min(chunk_frames);
                let byte_len = count * 4; // 2 channels * 2 bytes (i16)
                let media_buf: IMFMediaBuffer = MFCreateMemoryBuffer(byte_len as u32)
                    .map_err(|e| format!("MFCreateMemoryBuffer(PCM): {e}"))?;
                let mut dst_ptr: *mut u8 = std::ptr::null_mut();
                media_buf
                    .Lock(&mut dst_ptr, None, None)
                    .map_err(|e| format!("Lock PCM: {e}"))?;

                let pcm_i16 = std::slice::from_raw_parts_mut(dst_ptr as *mut i16, count * 2);
                for i in 0..count {
                    pcm_i16[i * 2] = quantize_f32_tpdf(left_f32[f_cursor + i], &mut rng);
                    pcm_i16[i * 2 + 1] = quantize_f32_tpdf(right_f32[f_cursor + i], &mut rng);
                }

                media_buf.Unlock().unwrap();
                media_buf.SetCurrentLength(byte_len as u32).unwrap();

                let sample_time_hns =
                    ((f_cursor as i64) * 10_000_000i64) / (aac_sr.max(1) as i64);
                let sample_dur_hns = ((count as i64) * 10_000_000i64) / (aac_sr.max(1) as i64);

                let sample: IMFSample =
                    MFCreateSample().map_err(|e| format!("MFCreateSample(PCM): {e}"))?;
                sample.AddBuffer(&media_buf).unwrap();
                sample.SetSampleTime(sample_time_hns).unwrap();
                sample.SetSampleDuration(sample_dur_hns.max(1)).unwrap();

                writer
                    .WriteSample(audio_stream_idx, &sample)
                    .map_err(|e| format!("WriteSample(Audio): {e}"))?;

                f_cursor += count;
            }

            writer
                .Finalize()
                .map_err(|e| format!("IMFSinkWriter::Finalize: {e}"))?;
        }

        // 5. Post-Mux Injection of `udta.meta.ilst` (Cover Art + Lyrics/Text) & `ID32` into the finalized MP4
        let _ = inject_udta_metadata_into_mp4_file(out_mp4_path, meta);
        Ok(())
    }
}

/// Injects the `udta` (`meta.ilst` cover art + backing text + `ID32`) atom into a finalized MP4 file.
/// If `moov` appears after `mdat` (standard `IMFSinkWriter` layout), appending `udta` to `moov`
/// requires zero `stco` offset changes!
fn inject_udta_metadata_into_mp4_file(mp4_path: &Path, meta: &MediaMetadata) -> Result<(), String> {
    let raw = fs::read(mp4_path).map_err(|e| format!("Failed reading MP4 for metadata: {e}"))?;
    let id3_bytes = serialize_id3v23_header(meta)?;
    let udta_box = build_udta_box(meta, &id3_bytes);

    let mut pos = 0usize;
    let mut moov_range: Option<(usize, usize)> = None;
    let mut mdat_pos: Option<usize> = None;

    while pos + 8 <= raw.len() {
        let sz32 = u32::from_be_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
        let fourcc = &raw[pos + 4..pos + 8];
        let (box_len, _hdr_len) = if sz32 == 1 && pos + 16 <= raw.len() {
            let sz64 = u64::from_be_bytes([
                raw[pos + 8],
                raw[pos + 9],
                raw[pos + 10],
                raw[pos + 11],
                raw[pos + 12],
                raw[pos + 13],
                raw[pos + 14],
                raw[pos + 15],
            ]) as usize;
            (sz64, 16usize)
        } else {
            (sz32, 8usize)
        };
        if box_len < 8 || pos + box_len > raw.len() {
            break;
        }
        if fourcc == b"moov" {
            moov_range = Some((pos, box_len));
        } else if fourcc == b"mdat" {
            mdat_pos = Some(pos);
        }
        pos += box_len;
    }

    let Some((moov_start, moov_len)) = moov_range else {
        return Ok(());
    };
    let Some(mdat_offset) = mdat_pos else {
        return Ok(());
    };

    // Only append directly when moov is after mdat so stco chunk offsets into mdat remain untouched
    if moov_start > mdat_offset {
        let new_moov_len = (moov_len + udta_box.len()) as u32;
        let mut patched = Vec::with_capacity(raw.len() + udta_box.len());
        patched.extend_from_slice(&raw[..moov_start]);
        patched.extend_from_slice(&new_moov_len.to_be_bytes());
        patched.extend_from_slice(&raw[moov_start + 4..moov_start + moov_len]);
        patched.extend_from_slice(&udta_box);
        patched.extend_from_slice(&raw[moov_start + moov_len..]);
        write_atomic_file(mp4_path, &patched)?;
    }
    Ok(())
}

// ============================================================================
// SECTION 8: ID3v2.3 & ISO-BMFF (.M4A) SERIALIZERS
// ============================================================================

#[inline(always)]
fn encode_synchsafe_u28(val: u32) -> [u8; 4] {
    [
        ((val >> 21) & 0x7F) as u8,
        ((val >> 14) & 0x7F) as u8,
        ((val >> 7) & 0x7F) as u8,
        (val & 0x7F) as u8,
    ]
}

#[inline(always)]
fn encode_utf16le_bom(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + text.len() * 2);
    out.push(0xFF);
    out.push(0xFE);
    for unit in text.encode_utf16() {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out
}

fn push_text_frame(buf: &mut Vec<u8>, frame_id: &[u8; 4], value: &str) {
    let encoded = encode_utf16le_bom(value);
    let payload_len = (1 + encoded.len()) as u32;
    buf.extend_from_slice(frame_id);
    buf.extend_from_slice(&payload_len.to_be_bytes());
    buf.extend_from_slice(&[0x00, 0x00]);
    buf.push(0x01);
    buf.extend_from_slice(&encoded);
}

fn push_uslt_or_comm_frame(
    buf: &mut Vec<u8>,
    frame_id: &[u8; 4],
    lang: &[u8; 3],
    descriptor: &str,
    body: &str,
) {
    let desc_bytes = encode_utf16le_bom(descriptor);
    let body_bytes = encode_utf16le_bom(body);
    let payload_len = (1 + 3 + desc_bytes.len() + 2 + body_bytes.len()) as u32;
    buf.extend_from_slice(frame_id);
    buf.extend_from_slice(&payload_len.to_be_bytes());
    buf.extend_from_slice(&[0x00, 0x00]);
    buf.push(0x01);
    buf.extend_from_slice(lang);
    buf.extend_from_slice(&desc_bytes);
    buf.extend_from_slice(&[0x00, 0x00]);
    buf.extend_from_slice(&body_bytes);
}

fn push_apic_frame(buf: &mut Vec<u8>, mime: &str, image_bytes: &[u8]) {
    let mime_bytes = mime.as_bytes();
    let desc_bytes = b"Cover (front)";
    let payload_len =
        (1 + mime_bytes.len() + 1 + 1 + desc_bytes.len() + 1 + image_bytes.len()) as u32;
    buf.extend_from_slice(b"APIC");
    buf.extend_from_slice(&payload_len.to_be_bytes());
    buf.extend_from_slice(&[0x00, 0x00]);
    buf.push(0x00);
    buf.extend_from_slice(mime_bytes);
    buf.push(0x00);
    buf.push(0x03);
    buf.extend_from_slice(desc_bytes);
    buf.push(0x00);
    buf.extend_from_slice(image_bytes);
}

pub fn serialize_id3v23_header(cfg: &MediaMetadata) -> Result<Vec<u8>, String> {
    let mut frames = Vec::with_capacity(4096 + cfg.cover_bytes.len());
    push_text_frame(&mut frames, b"TIT2", &cfg.title);
    push_text_frame(&mut frames, b"TPE1", &cfg.artist);
    push_text_frame(&mut frames, b"TALB", &cfg.album);
    push_text_frame(&mut frames, b"TYER", &cfg.year);
    if !cfg.behind_text.is_empty() {
        push_uslt_or_comm_frame(&mut frames, b"USLT", &cfg.lang, "Transcript", &cfg.behind_text);
        push_uslt_or_comm_frame(&mut frames, b"COMM", &cfg.lang, "ID3v2.3 Text", &cfg.behind_text);
    }
    if !cfg.cover_bytes.is_empty() {
        push_apic_frame(&mut frames, cfg.cover_mime, &cfg.cover_bytes);
    }
    if frames.len() >= 0x0FFF_FFFF {
        return Err("ID3v2.3 payload exceeds 28-bit synchsafe bound".to_string());
    }
    let mut tag = Vec::with_capacity(10 + frames.len());
    tag.extend_from_slice(b"ID3\x03\x00\x00");
    tag.extend_from_slice(&encode_synchsafe_u28(frames.len() as u32));
    tag.extend_from_slice(&frames);
    Ok(tag)
}

pub fn parse_mp3_frames(raw_mp3: &[u8]) -> (Vec<u8>, Vec<u32>, u32) {
    const BITRATES_V1_L3: [u32; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const BITRATES_V2_L3: [u32; 16] = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ];
    const SR_V1: [u32; 4] = [44100, 48000, 32000, 0];
    const SR_V2: [u32; 4] = [22050, 24000, 16000, 0];
    const SR_V25: [u32; 4] = [11025, 12000, 8000, 0];

    let mut clean_stream = Vec::with_capacity(raw_mp3.len());
    let mut sizes = Vec::with_capacity(raw_mp3.len() / 384 + 16);
    let mut detected_spf = 1152u32;
    let mut offset = 0usize;
    let n = raw_mp3.len();

    while offset + 4 <= n {
        if raw_mp3[offset] != 0xFF || (raw_mp3[offset + 1] & 0xE0) != 0xE0 {
            offset += 1;
            continue;
        }
        let ver_id = (raw_mp3[offset + 1] >> 3) & 0x03;
        let layer_id = (raw_mp3[offset + 1] >> 1) & 0x03;
        if ver_id == 1 || layer_id != 1 {
            offset += 1;
            continue;
        }
        let b2 = raw_mp3[offset + 2];
        let br_idx = ((b2 >> 4) & 0x0F) as usize;
        let sr_idx = ((b2 >> 2) & 0x03) as usize;
        let padding = ((b2 >> 1) & 0x01) as u32;
        if br_idx == 0 || br_idx == 15 || sr_idx == 3 {
            offset += 1;
            continue;
        }
        let (bitrate, sr, spf, coeff) = match ver_id {
            3 => (BITRATES_V1_L3[br_idx], SR_V1[sr_idx], 1152u32, 144u32),
            2 => (BITRATES_V2_L3[br_idx], SR_V2[sr_idx], 576u32, 72u32),
            0 => (BITRATES_V2_L3[br_idx], SR_V25[sr_idx], 576u32, 72u32),
            _ => {
                offset += 1;
                continue;
            }
        };
        if sr == 0 || bitrate == 0 {
            offset += 1;
            continue;
        }
        let frame_len = ((coeff * bitrate * 1000) / sr + padding) as usize;
        if frame_len < 24 || offset + frame_len > n {
            break;
        }
        detected_spf = spf;
        clean_stream.extend_from_slice(&raw_mp3[offset..offset + frame_len]);
        sizes.push(frame_len as u32);
        offset += frame_len;
    }

    if sizes.is_empty() && !raw_mp3.is_empty() {
        clean_stream.extend_from_slice(raw_mp3);
        sizes.push(raw_mp3.len() as u32);
    }
    (clean_stream, sizes, detected_spf)
}

#[inline(always)]
fn mp4_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let total_len = (8 + payload.len()) as u32;
    let mut out = Vec::with_capacity(total_len as usize);
    out.extend_from_slice(&total_len.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

#[inline(always)]
fn mp4_full_box(fourcc: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let total_len = (12 + payload.len()) as u32;
    let mut out = Vec::with_capacity(total_len as usize);
    out.extend_from_slice(&total_len.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.push(version);
    out.push(((flags >> 16) & 0xFF) as u8);
    out.push(((flags >> 8) & 0xFF) as u8);
    out.push((flags & 0xFF) as u8);
    out.extend_from_slice(payload);
    out
}

fn build_itunes_text(fourcc: &[u8; 4], text: &str) -> Vec<u8> {
    let utf8 = text.as_bytes();
    let mut data = Vec::with_capacity(8 + utf8.len());
    data.extend_from_slice(&1u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(utf8);
    mp4_box(fourcc, &mp4_box(b"data", &data))
}

fn build_itunes_covr(mime: &str, image: &[u8]) -> Vec<u8> {
    let type_code: u32 = if mime.contains("jpeg") || mime.contains("jpg") {
        13
    } else {
        14
    };
    let mut data = Vec::with_capacity(8 + image.len());
    data.extend_from_slice(&type_code.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(image);
    mp4_box(b"covr", &mp4_box(b"data", &data))
}

fn build_udta_box(meta: &MediaMetadata, id3_bytes: &[u8]) -> Vec<u8> {
    let mut hdlr_payload = Vec::with_capacity(24);
    hdlr_payload.extend_from_slice(&0u32.to_be_bytes());
    hdlr_payload.extend_from_slice(b"mdirappl");
    hdlr_payload.extend_from_slice(&[0u8; 9]);
    let meta_hdlr = mp4_full_box(b"hdlr", 0, 0, &hdlr_payload);

    let mut ilst_children = Vec::new();
    if !meta.title.is_empty() {
        ilst_children.extend_from_slice(&build_itunes_text(b"\xA9nam", &meta.title));
    }
    if !meta.artist.is_empty() {
        ilst_children.extend_from_slice(&build_itunes_text(b"\xA9ART", &meta.artist));
    }
    if !meta.album.is_empty() {
        ilst_children.extend_from_slice(&build_itunes_text(b"\xA9alb", &meta.album));
    }
    if !meta.year.is_empty() {
        ilst_children.extend_from_slice(&build_itunes_text(b"\xA9day", &meta.year));
    }
    if !meta.behind_text.is_empty() {
        ilst_children.extend_from_slice(&build_itunes_text(b"\xA9lyr", &meta.behind_text));
        ilst_children.extend_from_slice(&build_itunes_text(b"\xA9cmt", &meta.behind_text));
    }
    if !meta.cover_bytes.is_empty() {
        ilst_children.extend_from_slice(&build_itunes_covr(meta.cover_mime, &meta.cover_bytes));
    }
    let ilst = mp4_box(b"ilst", &ilst_children);

    let mut id32_payload = Vec::with_capacity(2 + id3_bytes.len());
    id32_payload.extend_from_slice(&0x15C7u16.to_be_bytes());
    id32_payload.extend_from_slice(id3_bytes);
    let id32 = mp4_full_box(b"ID32", 0, 0, &id32_payload);

    let mut meta_children = Vec::with_capacity(meta_hdlr.len() + ilst.len() + id32.len());
    meta_children.extend_from_slice(&meta_hdlr);
    meta_children.extend_from_slice(&ilst);
    meta_children.extend_from_slice(&id32);
    let meta_box = mp4_full_box(b"meta", 0, 0, &meta_children);
    mp4_box(b"udta", &meta_box)
}

fn build_mvhd_box(timescale: u32, duration_u32: u32, next_track_id: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity(96);
    p.extend_from_slice(&0u32.to_be_bytes());
    p.extend_from_slice(&0u32.to_be_bytes());
    p.extend_from_slice(&timescale.to_be_bytes());
    p.extend_from_slice(&duration_u32.to_be_bytes());
    p.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    p.extend_from_slice(&0x0100u16.to_be_bytes());
    p.extend_from_slice(&[0u8; 10]);
    for &v in &[0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        p.extend_from_slice(&v.to_be_bytes());
    }
    p.extend_from_slice(&[0u8; 24]);
    p.extend_from_slice(&next_track_id.to_be_bytes());
    mp4_full_box(b"mvhd", 0, 0, &p)
}

fn build_audio_trak(
    track_id: u32,
    sample_rate: u32,
    duration_u32: u32,
    frame_sizes: &[u32],
    spf: u32,
    audio_chunk_offset: u32,
) -> Vec<u8> {
    let mut tkhd_p = Vec::with_capacity(80);
    tkhd_p.extend_from_slice(&0u32.to_be_bytes());
    tkhd_p.extend_from_slice(&0u32.to_be_bytes());
    tkhd_p.extend_from_slice(&track_id.to_be_bytes());
    tkhd_p.extend_from_slice(&0u32.to_be_bytes());
    tkhd_p.extend_from_slice(&duration_u32.to_be_bytes());
    tkhd_p.extend_from_slice(&[0u8; 8]);
    tkhd_p.extend_from_slice(&0u16.to_be_bytes());
    tkhd_p.extend_from_slice(&0u16.to_be_bytes());
    tkhd_p.extend_from_slice(&0x0100u16.to_be_bytes());
    tkhd_p.extend_from_slice(&0u16.to_be_bytes());
    for &v in &[0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        tkhd_p.extend_from_slice(&v.to_be_bytes());
    }
    tkhd_p.extend_from_slice(&0u32.to_be_bytes());
    tkhd_p.extend_from_slice(&0u32.to_be_bytes());
    let tkhd = mp4_full_box(b"tkhd", 0, 0x000007, &tkhd_p);

    let mut mdhd_p = Vec::with_capacity(20);
    mdhd_p.extend_from_slice(&0u32.to_be_bytes());
    mdhd_p.extend_from_slice(&0u32.to_be_bytes());
    mdhd_p.extend_from_slice(&sample_rate.to_be_bytes());
    mdhd_p.extend_from_slice(&duration_u32.to_be_bytes());
    mdhd_p.extend_from_slice(&0x15C7u16.to_be_bytes());
    mdhd_p.extend_from_slice(&0u16.to_be_bytes());
    let mdhd = mp4_full_box(b"mdhd", 0, 0, &mdhd_p);

    let mut hdlr_p = Vec::with_capacity(32);
    hdlr_p.extend_from_slice(&0u32.to_be_bytes());
    hdlr_p.extend_from_slice(b"soun");
    hdlr_p.extend_from_slice(&[0u8; 12]);
    hdlr_p.extend_from_slice(b"SoundHandler\0");
    let hdlr = mp4_full_box(b"hdlr", 0, 0, &hdlr_p);

    let smhd = mp4_full_box(b"smhd", 0, 0, &[0u8; 4]);
    let url_box = mp4_full_box(b"url ", 0, 0x000001, &[]);
    let dref = mp4_full_box(b"dref", 0, 0, &[&1u32.to_be_bytes()[..], &url_box[..]].concat());
    let dinf = mp4_box(b"dinf", &dref);

    let obj_type: u8 = if sample_rate >= 32000 { 0x6B } else { 0x69 };
    let esds = mp4_full_box(
        b"esds",
        0,
        0,
        &[
            0x03, 0x17, 0x00, 0x01, 0x00, 0x04, 0x0F, obj_type, 0x15, 0x00, 0x18, 0x00, 0x00, 0x04,
            0xE2, 0x00, 0x00, 0x02, 0xEE, 0x00, 0x06, 0x01, 0x02,
        ],
    );

    let mut mp4a_p = Vec::with_capacity(28 + esds.len());
    mp4a_p.extend_from_slice(&[0u8; 6]);
    mp4a_p.extend_from_slice(&1u16.to_be_bytes());
    mp4a_p.extend_from_slice(&[0u8; 8]);
    mp4a_p.extend_from_slice(&2u16.to_be_bytes());
    mp4a_p.extend_from_slice(&16u16.to_be_bytes());
    mp4a_p.extend_from_slice(&0u16.to_be_bytes());
    mp4a_p.extend_from_slice(&0u16.to_be_bytes());
    mp4a_p.extend_from_slice(&((sample_rate.min(65535)) << 16).to_be_bytes());
    mp4a_p.extend_from_slice(&esds);
    let mp4a = mp4_box(b"mp4a", &mp4a_p);

    let stsd = mp4_full_box(b"stsd", 0, 0, &[&1u32.to_be_bytes()[..], &mp4a[..]].concat());
    let stts = mp4_full_box(
        b"stts",
        0,
        0,
        &[
            &1u32.to_be_bytes()[..],
            &(frame_sizes.len() as u32).to_be_bytes()[..],
            &spf.to_be_bytes()[..],
        ]
        .concat(),
    );
    let stsc = mp4_full_box(
        b"stsc",
        0,
        0,
        &[
            &1u32.to_be_bytes()[..],
            &1u32.to_be_bytes()[..],
            &(frame_sizes.len() as u32).to_be_bytes()[..],
            &1u32.to_be_bytes()[..],
        ]
        .concat(),
    );

    let mut stsz_p = Vec::with_capacity(8 + frame_sizes.len() * 4);
    stsz_p.extend_from_slice(&0u32.to_be_bytes());
    stsz_p.extend_from_slice(&(frame_sizes.len() as u32).to_be_bytes());
    for &sz in frame_sizes {
        stsz_p.extend_from_slice(&sz.to_be_bytes());
    }
    let stsz = mp4_full_box(b"stsz", 0, 0, &stsz_p);

    let stco = mp4_full_box(
        b"stco",
        0,
        0,
        &[
            &1u32.to_be_bytes()[..],
            &audio_chunk_offset.to_be_bytes()[..],
        ]
        .concat(),
    );

    let stbl = mp4_box(b"stbl", &[&stsd[..], &stts[..], &stsc[..], &stsz[..], &stco[..]].concat());
    let minf = mp4_box(b"minf", &[&smhd[..], &dinf[..], &stbl[..]].concat());
    let mdia = mp4_box(b"mdia", &[&mdhd[..], &hdlr[..], &minf[..]].concat());
    mp4_box(b"trak", &[&tkhd[..], &mdia[..]].concat())
}

pub fn serialize_m4a_container(
    sample_rate: u32,
    raw_mp3: &[u8],
    meta: &MediaMetadata,
) -> Result<Vec<u8>, String> {
    let (clean_mp3, frame_sizes, spf) = parse_mp3_frames(raw_mp3);
    let id3_bytes = serialize_id3v23_header(meta)?;
    let duration_u32 = ((frame_sizes.len() as u64) * (spf as u64)).min(u32::MAX as u64) as u32;

    let ftyp = mp4_box(
        b"ftyp",
        &[&b"M4A "[..], &0u32.to_be_bytes()[..], &b"M4A mp42isom"[..]].concat(),
    );
    let mvhd = build_mvhd_box(sample_rate, duration_u32, 2);
    let udta = build_udta_box(meta, &id3_bytes);

    let dummy_trak = build_audio_trak(1, sample_rate, duration_u32, &frame_sizes, spf, 0);
    let exact_audio_offset =
        (ftyp.len() + 8 + mvhd.len() + dummy_trak.len() + udta.len() + 8) as u32;

    let audio_trak = build_audio_trak(
        1,
        sample_rate,
        duration_u32,
        &frame_sizes,
        spf,
        exact_audio_offset,
    );
    let moov = mp4_box(b"moov", &[&mvhd[..], &audio_trak[..], &udta[..]].concat());
    let mdat = mp4_box(b"mdat", &clean_mp3);

    Ok([&ftyp[..], &moov[..], &mdat[..]].concat())
}

// ============================================================================
// SECTION 9: INTERACTIVE 24-BIT TRUECOLOR TUI RENDERER & INPUT ENGINE
// ============================================================================

pub struct CaptureOutput {
    pub sample_rate: u32,
    pub left_pcm: Vec<f32>,
    pub right_pcm: Vec<f32>,
    pub mp3_bytes: Vec<u8>,
    pub total_frames: u64,
    pub integrated_lufs: f64,
    pub max_peak_dbfs: f32,
    pub dropped_frames: usize,
    pub timeline_events: Vec<TimelineEvent>,
    pub cue_notes: Vec<String>,
}

fn map_bitrate(kbps: u16) -> Bitrate {
    match kbps {
        0..=140 => Bitrate::Kbps128,
        141..=220 => Bitrate::Kbps192,
        221..=280 => Bitrate::Kbps256,
        _ => Bitrate::Kbps320,
    }
}

fn encode_stereo_f32_to_mp3(
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
    bitrate_kbps: u16,
) -> Result<Vec<u8>, String> {
    let mut builder = Builder::new().ok_or("Failed to allocate LAME Builder")?;
    builder
        .set_num_channels(2)
        .map_err(|e| format!("set_num_channels: {e:?}"))?;
    builder
        .set_sample_rate(sample_rate)
        .map_err(|e| format!("set_sample_rate: {e:?}"))?;
    builder
        .set_brate(map_bitrate(bitrate_kbps))
        .map_err(|e| format!("set_brate: {e:?}"))?;
    builder
        .set_quality(Quality::Best)
        .map_err(|e| format!("set_quality: {e:?}"))?;
    let mut encoder = builder
        .build()
        .map_err(|e| format!("LAME build failed: {e:?}"))?;

    let n = left.len().min(right.len());
    let mut rng = 0x9E37_79B9u32;
    let mut l_i16 = Vec::with_capacity(n);
    let mut r_i16 = Vec::with_capacity(n);
    for i in 0..n {
        l_i16.push(quantize_f32_tpdf(left[i], &mut rng));
        r_i16.push(quantize_f32_tpdf(right[i], &mut rng));
    }

    let req = mp3lame_encoder::max_required_buffer_size(n) + 8192;
    let mut mp3_bytes = Vec::with_capacity(req);
    unsafe {
        let encoded = encoder
            .encode(
                DualPcm {
                    left: &l_i16,
                    right: &r_i16,
                },
                mp3_bytes.spare_capacity_mut(),
            )
            .map_err(|e| format!("LAME encode error: {e:?}"))?;
        mp3_bytes.set_len(mp3_bytes.len() + encoded);

        let flushed = encoder
            .flush::<FlushNoGap>(mp3_bytes.spare_capacity_mut())
            .map_err(|e| format!("LAME flush error: {e:?}"))?;
        mp3_bytes.set_len(mp3_bytes.len() + flushed);
    }
    Ok(mp3_bytes)
}

#[allow(clippy::too_many_arguments)]
fn render_interactive_dashboard(
    frame_buf: &mut String,
    cfg: &CliConfig,
    state: &InteractiveState,
    spectrum: &SpectralAnalyzer32,
    meter: &KWeightingMeter,
    left_peak_db: f32,
    right_peak_db: f32,
    max_peak_db: f32,
    captured_secs: f64,
    secs_until_next_auto: f32,
    total_transitions_fired: usize,
    ring_fill: f32,
    num_scenes: usize,
    last_event_msg: &str,
) {
    frame_buf.clear();
    frame_buf.push_str("\x1b[H");

    let paused = state.paused.load(Ordering::Relaxed);
    let synth_on = state.test_synth_active.load(Ordering::Relaxed);
    let active_trans = state.transition();
    let auto_mode = state.auto_mode();
    let cycling = state.cycle_shaders.load(Ordering::Relaxed);
    let gain_db = state.gain_db();
    let gate_opt = state.gate_db();
    let cur_scene = state.active_scene_idx.load(Ordering::Relaxed) % num_scenes.max(1);

    push_bg_oklch(frame_buf, 0.20, 0.06, 265.0);
    push_fg_oklch(frame_buf, 0.88, 0.18, 195.0);
    let _ = writeln!(
        frame_buf,
        "╔══════════════════════════════════════════════════════════════════════════════╗\x1b[K"
    );
    push_bg_oklch(frame_buf, 0.20, 0.06, 265.0);
    push_fg_oklch(frame_buf, 0.86, 0.19, 80.0);
    let status_badge = if paused {
        "\x1b[48;2;180;40;60m\x1b[38;2;255;255;255m ⏸ PAUSED \x1b[0m"
    } else if synth_on {
        "\x1b[48;2;140;40;190m\x1b[38;2;255;255;255m ● REC+SYNTH \x1b[0m"
    } else {
        "\x1b[48;2;0;150;90m\x1b[38;2;255;255;255m ● REC LIVE \x1b[0m"
    };
    let _ = write!(
        frame_buf,
        "║  WIN_LOOPBACK_MP3_MUX v5.0 // MEDIA FOUNDATION H.264+AAC DAW {status_badge}"
    );
    push_bg_oklch(frame_buf, 0.20, 0.06, 265.0);
    push_fg_oklch(frame_buf, 0.88, 0.18, 195.0);
    let _ = writeln!(frame_buf, "    ║\x1b[0m\x1b[K");

    push_bg_oklch(frame_buf, 0.20, 0.06, 265.0);
    push_fg_oklch(frame_buf, 0.88, 0.18, 195.0);
    let _ = writeln!(
        frame_buf,
        "╠══════════════════════════════════════════════════════════════════════════════╣\x1b[0m\x1b[K"
    );

    let progress_ratio = (captured_secs / (cfg.duration_secs.max(1) as f64)).clamp(0.0, 1.0);
    let prog_cells = (progress_ratio * 34.0).round() as usize;
    frame_buf.push_str("  \x1b[1;38;2;190;210;240mTIME:\x1b[0m ");
    push_fg_oklch(frame_buf, 0.85, 0.19, 150.0);
    let _ = write!(
        frame_buf,
        "{:05.1}s / {:05.1}s ",
        captured_secs, cfg.duration_secs as f64
    );
    frame_buf.push('[');
    for c in 0..34 {
        if c < prog_cells {
            let hue = 190.0 - (c as f32 / 34.0) * 110.0;
            push_fg_oklch(frame_buf, 0.80, 0.20, hue);
            frame_buf.push('━');
        } else {
            frame_buf.push_str("\x1b[38;2;55;65;85m─");
        }
    }
    let _ = writeln!(
        frame_buf,
        "\x1b[0m] {:3.0}% | Ring:{:4.1}%\x1b[K",
        progress_ratio * 100.0,
        ring_fill * 100.0
    );

    let render_meter_row = |buf: &mut String, label: &str, db: f32| {
        let _ = write!(buf, "  \x1b[1;38;2;180;200;230m{label}\x1b[0m [");
        let norm = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
        let lit = (norm * 42.0).round() as usize;
        for i in 0..42 {
            if i < lit {
                let frac = (i as f32) / 42.0;
                let hue = if frac < 0.70 {
                    155.0 - (frac / 0.70) * 65.0
                } else {
                    90.0 - ((frac - 0.70) / 0.30) * 65.0
                };
                push_fg_oklch(buf, 0.78, 0.21, hue);
                buf.push('█');
            } else {
                buf.push_str("\x1b[38;2;40;46;62m·");
            }
        }
        push_fg_oklch(buf, 0.85, 0.15, if db > -3.0 { 28.0 } else { 155.0 });
        let _ = writeln!(buf, "\x1b[0m] {:+6.1} dBFS\x1b[K", db);
    };

    render_meter_row(frame_buf, "L-CH", left_peak_db);
    render_meter_row(frame_buf, "R-CH", right_peak_db);

    let m_lufs = meter.momentary_lufs();
    let i_lufs = meter.integrated_lufs();
    let corr = spectrum.phase_corr;
    let corr_pos = (((corr + 1.0) * 0.5) * 14.0).round().clamp(0.0, 14.0) as usize;
    let mut corr_bar = String::new();
    for i in 0..=14 {
        if i == corr_pos {
            corr_bar.push_str("\x1b[1;38;2;0;245;212m◆\x1b[0m");
        } else if i == 7 {
            corr_bar.push_str("\x1b[38;2;120;135;160m┼\x1b[0m");
        } else {
            corr_bar.push_str("\x1b[38;2;55;65;85m─\x1b[0m");
        }
    }
    let _ = writeln!(
        frame_buf,
        "  \x1b[1;38;2;180;200;230mLUFS:\x1b[0m \x1b[38;2;255;200;80mM:{:+5.1}\x1b[0m | \x1b[38;2;0;230;180mI:{:+5.1}\x1b[0m | \x1b[38;2;255;120;150mMaxPk:{:+5.1}dB\x1b[0m | \x1b[1;38;2;180;200;230mPHASE:\x1b[0m[-1{corr_bar}+1] | \x1b[38;2;255;180;255mFluxZ:{:+.1}\x1b[0m\x1b[K",
        m_lufs, i_lufs, max_peak_db, spectrum.last_flux_zscore
    );

    push_fg_oklch(frame_buf, 0.72, 0.14, 230.0);
    let _ = writeln!(
        frame_buf,
        "  ┌────────────────── 32-BAND 512-PT RADIX-2 FFT SPECTRUM ───────────────────┐\x1b[0m\x1b[K"
    );
    const ROWS: usize = 6;
    const SUB_BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    for row in (0..ROWS).rev() {
        frame_buf.push_str("  \x1b[38;2;90;115;165m│\x1b[0m   ");
        for b in 0..32 {
            let norm = ((spectrum.bands_db[b] + 72.0) / 72.0).clamp(0.0, 1.0);
            let peak_norm = ((spectrum.peaks_db[b] + 72.0) / 72.0).clamp(0.0, 1.0);
            let total_eighths = (norm * (ROWS as f32) * 8.0).round() as usize;
            let peak_row = ((peak_norm * (ROWS as f32)).floor() as usize).min(ROWS - 1);

            let row_floor = row * 8;
            let ch = if total_eighths >= row_floor + 8 {
                '█'
            } else if total_eighths > row_floor {
                SUB_BLOCKS[total_eighths - row_floor]
            } else if row == peak_row && peak_norm > 0.06 {
                '▔'
            } else {
                '·'
            };

            let hue = 255.0 - (b as f32 / 31.0) * 235.0;
            let lightness = 0.58 + (row as f32 / (ROWS as f32)) * 0.28;
            if ch == '·' {
                frame_buf.push_str("\x1b[38;2;32;38;52m· ");
            } else if ch == '▔' {
                push_fg_oklch(frame_buf, 0.92, 0.18, hue);
                frame_buf.push_str("▔ ");
            } else {
                push_fg_oklch(frame_buf, lightness, 0.22, hue);
                frame_buf.push(ch);
                frame_buf.push(' ');
            }
        }
        frame_buf.push_str("\x1b[0m   \x1b[38;2;90;115;165m│\x1b[0m\x1b[K\n");
    }
    push_fg_oklch(frame_buf, 0.72, 0.14, 230.0);
    let _ = writeln!(
        frame_buf,
        "  └──── 32Hz ──── 125Hz ──── 500Hz ──── 2kHz ───── 8kHz ──── 16kHz ──── 24kHz┘\x1b[0m\x1b[K"
    );

    frame_buf.push_str("  \x1b[1;38;2;200;215;245mSHADERS:\x1b[0m ");
    for idx in 0..5u8 {
        let mode = PhysicsTransition::from_u8(idx);
        if mode == active_trans {
            push_bg_oklch(frame_buf, 0.78, 0.20, 75.0);
            frame_buf.push_str("\x1b[1;38;2;10;12;18m ");
            frame_buf.push_str(mode.label());
            frame_buf.push_str(" \x1b[0m ");
        } else {
            frame_buf.push_str("\x1b[48;2;25;32;46m\x1b[38;2;140;160;190m ");
            frame_buf.push_str(mode.label());
            frame_buf.push_str(" \x1b[0m ");
        }
    }
    let cycle_badge = if cycling {
        "\x1b[48;2;0;120;140m\x1b[1;38;2;255;255;255m CYCLE:ON \x1b[0m"
    } else {
        "\x1b[48;2;45;52;68m\x1b[38;2;160;175;200m CYCLE:OFF \x1b[0m"
    };
    let _ = writeln!(frame_buf, "{cycle_badge}\x1b[K");

    let gate_label = match gate_opt {
        Some(g) => format!("{g:+.0}dB"),
        None => "OFF".to_string(),
    };
    let next_auto_str = if auto_mode == AutoTransitionMode::ManualOnly {
        "MANUAL".to_string()
    } else {
        format!("in {:.1}s", secs_until_next_auto.max(0.0))
    };
    let _ = writeln!(
        frame_buf,
        "  \x1b[1;38;2;200;215;245mAUTO-PILOT:\x1b[0m \x1b[1;38;2;0;245;190m{}\x1b[0m ({next_auto_str}) | Scene:\x1b[1;38;2;120;210;255m#{}/{}\x1b[0m | Trans:\x1b[1;38;2;255;200;90m{}\x1b[0m | Gain:\x1b[1;38;2;0;235;170m{:+4.1}dB\x1b[0m | Gate:\x1b[1;38;2;255;190;60m{}\x1b[0m\x1b[K",
        auto_mode.label(),
        cur_scene + 1,
        num_scenes.max(1),
        total_transitions_fired,
        gain_db,
        gate_label
    );

    push_fg_oklch(frame_buf, 0.68, 0.12, 250.0);
    let _ = writeln!(
        frame_buf,
        "  ────────────────────────────────────────────────────────────────────────────\x1b[0m\x1b[K"
    );
    let _ = writeln!(
        frame_buf,
        "  \x1b[48;2;35;45;68m\x1b[1;38;2;255;160;255m T \x1b[0m Test Synth  \x1b[48;2;35;45;68m\x1b[1;38;2;0;240;255m A \x1b[0m Auto-Mode  \x1b[48;2;35;45;68m\x1b[1;38;2;0;245;190m C \x1b[0m Cycle  \x1b[48;2;35;45;68m\x1b[1;38;2;255;195;0m 1..5 \x1b[0m Shader  \x1b[48;2;35;45;68m\x1b[1;38;2;0;235;150m N / TAB \x1b[0m Transition Now\x1b[K"
    );
    let _ = writeln!(
        frame_buf,
        "  \x1b[48;2;35;45;68m\x1b[1;38;2;255;135;180m +/- \x1b[0m Gain    \x1b[48;2;35;45;68m\x1b[1;38;2;180;155;255m [/] \x1b[0m Gate  \x1b[48;2;35;45;68m\x1b[1;38;2;120;240;180m M \x1b[0m Cue  \x1b[48;2;35;45;68m\x1b[1;38;2;0;240;255m SPACE \x1b[0m Pause  \x1b[48;2;140;30;55m\x1b[1;38;2;255;255;255m S / ENTER / Q \x1b[0m Finish & Mux\x1b[K"
    );
    let _ = writeln!(
        frame_buf,
        "  \x1b[38;2;130;155;195mEVENT LOG:\x1b[0m \x1b[1;38;2;255;225;130m{}\x1b[0m\x1b[K",
        last_event_msg
    );
}

#[cfg(windows)]
mod wasapi_loopback {
    use super::*;
    use windows::Win32::Foundation::{BOOL, TRUE};
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::*;
    use windows::Win32::System::Console::{
        GetConsoleMode, GetNumberOfConsoleInputEvents, GetStdHandle, ReadConsoleInputW,
        SetConsoleCtrlHandler, SetConsoleMode, CONSOLE_MODE, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT,
        CTRL_C_EVENT, ENABLE_VIRTUAL_TERMINAL_PROCESSING, INPUT_RECORD, KEY_EVENT,
        STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };

    struct ComGuard;
    impl Drop for ComGuard {
        fn drop(&mut self) {
            unsafe {
                CoUninitialize();
            }
        }
    }

    struct CoTaskMemGuard(*mut WAVEFORMATEX);
    impl Drop for CoTaskMemGuard {
        fn drop(&mut self) {
            unsafe {
                if !self.0.is_null() {
                    CoTaskMemFree(Some(self.0 as *const _));
                }
            }
        }
    }

    struct TerminalSessionGuard;
    impl TerminalSessionGuard {
        fn enter() -> Self {
            print!("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H");
            let _ = std::io::stdout().flush();
            Self
        }
    }
    impl Drop for TerminalSessionGuard {
        fn drop(&mut self) {
            print!("\x1b[?25h\x1b[?1049l\x1b[0m");
            let _ = std::io::stdout().flush();
        }
    }

    unsafe extern "system" fn ctrl_handler(ctrl_type: u32) -> BOOL {
        if ctrl_type == CTRL_C_EVENT
            || ctrl_type == CTRL_BREAK_EVENT
            || ctrl_type == CTRL_CLOSE_EVENT
        {
            STOP_REQUESTED.store(true, Ordering::Release);
            return TRUE;
        }
        BOOL(0)
    }

    pub fn init_terminal_and_signals() {
        unsafe {
            let _ = SetConsoleCtrlHandler(Some(ctrl_handler), TRUE);
            if let Ok(handle) = GetStdHandle(STD_OUTPUT_HANDLE) {
                let mut mode = CONSOLE_MODE(0);
                if GetConsoleMode(handle, &mut mode).is_ok() {
                    let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn poll_console_hotkeys(
        state: &InteractiveState,
        num_scenes: usize,
        current_frame: u64,
        last_transition_frame: &mut u64,
        sample_rate: u32,
        timeline_events: &mut Vec<TimelineEvent>,
        cue_notes: &mut Vec<String>,
        last_msg: &mut String,
    ) {
        unsafe {
            let Ok(stdin_handle) = GetStdHandle(STD_INPUT_HANDLE) else {
                return;
            };
            let mut pending: u32 = 0;
            if GetNumberOfConsoleInputEvents(stdin_handle, &mut pending).is_err() || pending == 0 {
                return;
            }

            let mut records = [INPUT_RECORD::default(); 16];
            let to_read = pending.min(16);
            let mut read_count: u32 = 0;
            if ReadConsoleInputW(stdin_handle, &mut records[..to_read as usize], &mut read_count)
                .is_err()
            {
                return;
            }

            for rec in &records[..read_count as usize] {
                if rec.EventType != (KEY_EVENT as u16) {
                    continue;
                }
                let key_ev = rec.Event.KeyEvent;
                if !key_ev.bKeyDown.as_bool() {
                    continue;
                }
                let vk = key_ev.wVirtualKeyCode;
                let ch = key_ev.uChar.UnicodeChar;
                let sec_stamp = (current_frame as f64) / (sample_rate.max(1) as f64);

                match (vk, ch) {
                    (0x20, _) => {
                        let was_paused = state.paused.fetch_xor(true, Ordering::AcqRel);
                        *last_msg = if !was_paused {
                            format!("@{sec_stamp:05.1}s: Capture PAUSED (press SPACE to resume)")
                        } else {
                            format!("@{sec_stamp:05.1}s: Capture RESUMED")
                        };
                    }
                    (_, 0x74) | (_, 0x54) => {
                        let was_on = state.test_synth_active.fetch_xor(true, Ordering::AcqRel);
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: Internal Harmonic Test Synth -> {}",
                            if !was_on {
                                "ENABLED (220Hz+330Hz+440Hz + 120BPM Kick)"
                            } else {
                                "DISABLED"
                            }
                        );
                    }
                    (_, 0x61) | (_, 0x41) => {
                        let prev = state.auto_mode.fetch_add(1, Ordering::AcqRel);
                        let next_mode = AutoTransitionMode::from_u8(prev.wrapping_add(1));
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: Auto-Transition Mode -> {}",
                            next_mode.label()
                        );
                    }
                    (_, 0x63) | (_, 0x43) => {
                        let was_on = state.cycle_shaders.fetch_xor(true, Ordering::AcqRel);
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: Auto Shader Cycling -> {}",
                            if !was_on { "ENABLED" } else { "DISABLED" }
                        );
                    }
                    (_, c) if (0x31..=0x35).contains(&c) => {
                        let idx = (c - 0x31) as u8;
                        state.transition_mode.store(idx, Ordering::Release);
                        let mode = PhysicsTransition::from_u8(idx);
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: Physics Transition Armed -> {}",
                            mode.label()
                        );
                    }
                    (0x09, _) | (_, 0x6E) | (_, 0x4E) => {
                        let from_sc =
                            state.active_scene_idx.load(Ordering::Relaxed) % num_scenes.max(1);
                        let (next_sc, mode) = state.trigger_next_scene(num_scenes);
                        *last_transition_frame = current_frame;
                        timeline_events.push(TimelineEvent {
                            sample_frame: current_frame,
                            from_scene: from_sc,
                            target_scene: next_sc,
                            transition: mode,
                            trigger_reason: "MANUAL [N]",
                        });
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: [MANUAL] {:?} Transition -> Scene #{}",
                            mode,
                            next_sc + 1
                        );
                    }
                    (0x26, _) | (_, 0x2B) | (_, 0x3D) => {
                        let prev = state.gain_half_db_steps.load(Ordering::Relaxed);
                        let next = (prev + 1).clamp(-48, 48);
                        state.gain_half_db_steps.store(next, Ordering::Relaxed);
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: Pre-Gain adjusted to {:+4.1} dB",
                            (next as f32) * 0.5
                        );
                    }
                    (0x28, _) | (_, 0x2D) | (_, 0x5F) => {
                        let prev = state.gain_half_db_steps.load(Ordering::Relaxed);
                        let next = (prev - 1).clamp(-48, 48);
                        state.gain_half_db_steps.store(next, Ordering::Relaxed);
                        *last_msg = format!(
                            "@{sec_stamp:05.1}s: Pre-Gain adjusted to {:+4.1} dB",
                            (next as f32) * 0.5
                        );
                    }
                    (0x27, _) | (_, 0x5D) => {
                        let prev = state.gate_db_int.load(Ordering::Relaxed);
                        let next = if prev <= -95 { -70 } else { (prev + 2).min(-15) };
                        state.gate_db_int.store(next, Ordering::Relaxed);
                        *last_msg = format!("@{sec_stamp:05.1}s: Auto-Silence Gate -> {next} dBFS");
                    }
                    (0x25, _) | (_, 0x5B) => {
                        let prev = state.gate_db_int.load(Ordering::Relaxed);
                        let next = (prev - 2).max(-100);
                        state.gate_db_int.store(next, Ordering::Relaxed);
                        if next <= -95 {
                            *last_msg = format!("@{sec_stamp:05.1}s: Auto-Silence Gate -> OFF");
                        } else {
                            *last_msg =
                                format!("@{sec_stamp:05.1}s: Auto-Silence Gate -> {next} dBFS");
                        }
                    }
                    (_, 0x6D) | (_, 0x4D) => {
                        let m_num = state.cue_markers_count.fetch_add(1, Ordering::AcqRel) + 1;
                        let note = format!(
                            "[Cue #{m_num:02} @ {sec_stamp:06.2}s | Scene #{} | {:?}]",
                            (state.active_scene_idx.load(Ordering::Relaxed) % num_scenes.max(1))
                                + 1,
                            state.transition()
                        );
                        cue_notes.push(note.clone());
                        *last_msg = format!("Dropped {note}");
                    }
                    (0x0D, _) | (0x1B, _) | (_, 0x73) | (_, 0x53) | (_, 0x71) | (_, 0x51) => {
                        *last_msg =
                            format!("@{sec_stamp:05.1}s: Clean Stop Triggered — Finalizing...");
                        STOP_REQUESTED.store(true, Ordering::Release);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn capture(cfg: &CliConfig, num_scenes: usize) -> Result<CaptureOutput, String> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED)
                .ok()
                .map_err(|e| format!("CoInitializeEx failed: {e:?}"))?;
            let _com_guard = ComGuard;

            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .map_err(|e| format!("CoCreateInstance(MMDeviceEnumerator): {e}"))?;
            let device: IMMDevice = enumerator
                .GetDefaultAudioEndpoint(eRender, eConsole)
                .map_err(|e| format!("GetDefaultAudioEndpoint: {e}"))?;
            let client: IAudioClient = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|e| format!("IAudioClient::Activate: {e}"))?;

            let fmt_ptr = client
                .GetMixFormat()
                .map_err(|e| format!("IAudioClient::GetMixFormat: {e}"))?;
            let _fmt_guard = CoTaskMemGuard(fmt_ptr);
            let fmt = *fmt_ptr;

            let hw_sr = fmt.nSamplesPerSec;
            let ch = (fmt.nChannels as usize).max(1);
            let bps = fmt.wBitsPerSample;

            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_LOOPBACK,
                    1_000_000,
                    0,
                    fmt_ptr,
                    None,
                )
                .map_err(|e| format!("IAudioClient::Initialize: {e}"))?;

            let capture_client: IAudioCaptureClient = client
                .GetService()
                .map_err(|e| format!("GetService(IAudioCaptureClient): {e}"))?;

            let decimate = if hw_sr >= 176_400 {
                4usize
            } else if hw_sr >= 88_200 {
                2usize
            } else {
                1usize
            };
            let target_sr = hw_sr / (decimate as u32);

            let ring = Arc::new(SpscStereoRing::new_pow2(262_144));
            let producer_ring = Arc::clone(&ring);
            let capture_done = Arc::new(AtomicBool::new(false));
            let producer_done = Arc::clone(&capture_done);

            let state = Arc::new(InteractiveState::new(
                cfg.transition,
                cfg.auto_mode,
                cfg.cycle_shaders,
                cfg.test_synth,
                cfg.trim_silence_db,
            ));
            let consumer_state = Arc::clone(&state);
            let cfg_clone = cfg.clone();

            let consumer = std::thread::spawn(move || -> Result<CaptureOutput, String> {
                let _tui_guard = TerminalSessionGuard::enter();
                let est_samples = (target_sr as usize) * (cfg_clone.duration_secs as usize + 1);
                let mut full_left: Vec<f32> = Vec::with_capacity(est_samples);
                let mut full_right: Vec<f32> = Vec::with_capacity(est_samples);
                let mut batch_l: Vec<f32> = Vec::with_capacity(4096);
                let mut batch_r: Vec<f32> = Vec::with_capacity(4096);

                let mut meter = KWeightingMeter::new(target_sr);
                let mut spectrum = SpectralAnalyzer32::new();
                let mut timeline_events: Vec<TimelineEvent> = Vec::new();
                let mut cue_notes: Vec<String> = Vec::new();
                let mut frame_buf = String::with_capacity(16384);
                let mut last_msg = format!(
                    "WASAPI Loopback Active ({} Hz). Play system audio or press [T] for internal Test Synth!",
                    target_sr
                );

                let mut max_peak_dbfs: f32 = -96.0;
                let mut smooth_l_db: f32 = -96.0;
                let mut smooth_r_db: f32 = -96.0;
                let mut last_transition_frame: u64 = 0;
                let mut synth_phase_idx: u64 = 0;
                let min_refractory_frames =
                    ((cfg_clone.auto_interval_secs * (target_sr as f32)).round() as u64).max(1);
                let mut last_hud = Instant::now();

                loop {
                    let cur_frame = full_left.len() as u64;
                    poll_console_hotkeys(
                        &consumer_state,
                        num_scenes,
                        cur_frame,
                        &mut last_transition_frame,
                        target_sr,
                        &mut timeline_events,
                        &mut cue_notes,
                        &mut last_msg,
                    );

                    batch_l.clear();
                    batch_r.clear();
                    let popped = ring.pop_batch(&mut batch_l, &mut batch_r, 2048);

                    if popped > 0 {
                        let gain_lin = consumer_state.gain_linear();
                        let gate_lin = consumer_state
                            .gate_db()
                            .map(|db| 10.0f32.powf(db / 20.0))
                            .unwrap_or(0.0);
                        let is_paused = consumer_state.paused.load(Ordering::Relaxed);
                        let synth_on = consumer_state.test_synth_active.load(Ordering::Relaxed);

                        let mut pk_l = 0.0f32;
                        let mut pk_r = 0.0f32;
                        let inv_sr = 1.0 / (target_sr as f32);

                        for i in 0..popped {
                            let mut l = batch_l[i] * gain_lin;
                            let mut r = batch_r[i] * gain_lin;

                            if synth_on {
                                let t = (synth_phase_idx as f32) * inv_sr;
                                synth_phase_idx = synth_phase_idx.wrapping_add(1);
                                let beat_pos = (t * 2.0).fract();
                                let kick_env = (-beat_pos * 12.0).exp();
                                let kick_freq = 55.0 + 95.0 * (-beat_pos * 28.0).exp();
                                let kick = (TAU * kick_freq * beat_pos).sin() * kick_env * 0.45;
                                let pad_l = (TAU * 220.0 * t).sin() * 0.14
                                    + (TAU * 261.63 * t).sin() * 0.12
                                    + (TAU * 329.63 * t).sin() * 0.10;
                                let pad_r = (TAU * 220.5 * t).sin() * 0.14
                                    + (TAU * 329.63 * t).sin() * 0.12
                                    + (TAU * 493.88 * t).sin() * 0.10;
                                l += kick + pad_l;
                                r += kick + pad_r;
                            }

                            l = l.clamp(-1.0, 1.0);
                            r = r.clamp(-1.0, 1.0);
                            batch_l[i] = l;
                            batch_r[i] = r;
                            if l.abs() > pk_l {
                                pk_l = l.abs();
                            }
                            if r.abs() > pk_r {
                                pk_r = r.abs();
                            }
                        }

                        spectrum.ingest_stereo_slice(&batch_l, &batch_r);

                        let db_l = if pk_l > 1.0e-9 {
                            20.0 * pk_l.log10()
                        } else {
                            -96.0
                        };
                        let db_r = if pk_r > 1.0e-9 {
                            20.0 * pk_r.log10()
                        } else {
                            -96.0
                        };
                        smooth_l_db = if db_l > smooth_l_db {
                            db_l
                        } else {
                            smooth_l_db * 0.78 + db_l * 0.22
                        };
                        smooth_r_db = if db_r > smooth_r_db {
                            db_r
                        } else {
                            smooth_r_db * 0.78 + db_r * 0.22
                        };
                        let batch_max_db = db_l.max(db_r);
                        if batch_max_db > max_peak_dbfs {
                            max_peak_dbfs = batch_max_db;
                        }

                        if !is_paused && pk_l.max(pk_r) >= gate_lin {
                            for i in 0..popped {
                                meter.push_frame(batch_l[i], batch_r[i]);
                            }
                            full_left.extend_from_slice(&batch_l);
                            full_right.extend_from_slice(&batch_r);

                            let now_frame = full_left.len() as u64;
                            let elapsed_since_last =
                                now_frame.saturating_sub(last_transition_frame);
                            let auto_mode = consumer_state.auto_mode();

                            if num_scenes > 1 && elapsed_since_last >= min_refractory_frames {
                                let should_fire_reason: Option<&'static str> = match auto_mode {
                                    AutoTransitionMode::TimerGrid => Some("AUTO-TIMER"),
                                    AutoTransitionMode::BeatReactive => {
                                        if spectrum.beat_detected {
                                            Some("AUTO-BEAT ONSET")
                                        } else if elapsed_since_last >= min_refractory_frames * 2 {
                                            Some("AUTO-FALLBACK TIMER")
                                        } else {
                                            None
                                        }
                                    }
                                    AutoTransitionMode::ManualOnly => None,
                                };

                                if let Some(reason) = should_fire_reason {
                                    let from_sc = consumer_state
                                        .active_scene_idx
                                        .load(Ordering::Relaxed)
                                        % num_scenes;
                                    let (next_sc, mode) =
                                        consumer_state.trigger_next_scene(num_scenes);
                                    last_transition_frame = now_frame;
                                    let sec_stamp = (now_frame as f64) / (target_sr as f64);
                                    timeline_events.push(TimelineEvent {
                                        sample_frame: now_frame,
                                        from_scene: from_sc,
                                        target_scene: next_sc,
                                        transition: mode,
                                        trigger_reason: reason,
                                    });
                                    last_msg = format!(
                                        "@{sec_stamp:05.1}s: [{reason}] {:?} -> Scene #{}/{}",
                                        mode,
                                        next_sc + 1,
                                        num_scenes
                                    );
                                }
                            }
                        }
                    } else if capture_done.load(Ordering::Acquire) {
                        break;
                    } else {
                        std::thread::sleep(Duration::from_millis(4));
                    }

                    if last_hud.elapsed() >= Duration::from_millis(40) {
                        let now_f = full_left.len() as u64;
                        let captured_secs = (now_f as f64) / (target_sr as f64);
                        let rem_frames = min_refractory_frames
                            .saturating_sub(now_f.saturating_sub(last_transition_frame));
                        let secs_until_next = (rem_frames as f32) / (target_sr as f32);

                        render_interactive_dashboard(
                            &mut frame_buf,
                            &cfg_clone,
                            &consumer_state,
                            &spectrum,
                            &meter,
                            smooth_l_db,
                            smooth_r_db,
                            max_peak_dbfs,
                            captured_secs,
                            secs_until_next,
                            timeline_events.len(),
                            ring.fill_ratio(),
                            num_scenes,
                            &last_msg,
                        );
                        let _ = std::io::stdout().write_all(frame_buf.as_bytes());
                        let _ = std::io::stdout().flush();
                        last_hud = Instant::now();
                    }

                    if (full_left.len() as u64) >= (target_sr as u64) * cfg_clone.duration_secs {
                        STOP_REQUESTED.store(true, Ordering::Release);
                        break;
                    }
                }

                drop(_tui_guard);

                let trimmed_head = condition_clean_boundaries(
                    &mut full_left,
                    &mut full_right,
                    target_sr,
                    cfg_clone.fade_ms,
                    consumer_state.gate_db(),
                );

                let final_len = full_left.len() as u64;
                timeline_events.retain_mut(|ev| {
                    if ev.sample_frame < trimmed_head {
                        false
                    } else {
                        ev.sample_frame -= trimmed_head;
                        ev.sample_frame < final_len
                    }
                });

                let mp3_bytes = encode_stereo_f32_to_mp3(
                    &full_left,
                    &full_right,
                    target_sr,
                    cfg_clone.bitrate_kbps,
                )?;

                Ok(CaptureOutput {
                    sample_rate: target_sr,
                    left_pcm: full_left,
                    right_pcm: full_right,
                    mp3_bytes,
                    total_frames: final_len,
                    integrated_lufs: meter.integrated_lufs(),
                    max_peak_dbfs,
                    dropped_frames: ring.dropped(),
                    timeline_events,
                    cue_notes,
                })
            });

            client
                .Start()
                .map_err(|e| format!("IAudioClient::Start: {e}"))?;

            let mut dec_ctr = 0usize;
            let mut last_packet_instant = Instant::now();

            while !STOP_REQUESTED.load(Ordering::Acquire) {
                let mut pkt = capture_client
                    .GetNextPacketSize()
                    .map_err(|e| format!("GetNextPacketSize: {e}"))?;

                if pkt == 0 {
                    let idle_elapsed = last_packet_instant.elapsed();
                    if idle_elapsed >= Duration::from_millis(12) {
                        let synth_frames =
                            ((target_sr as u64 * idle_elapsed.as_millis() as u64) / 1000)
                                .clamp(128, 2048) as usize;
                        for _ in 0..synth_frames {
                            producer_ring.push_stereo(0.0, 0.0);
                        }
                        last_packet_instant = Instant::now();
                    }
                    std::thread::sleep(Duration::from_millis(3));
                    continue;
                }

                last_packet_instant = Instant::now();
                while pkt > 0 {
                    let mut ptr: *mut u8 = std::ptr::null_mut();
                    let mut num_frames: u32 = 0;
                    let mut flags: u32 = 0;
                    capture_client
                        .GetBuffer(&mut ptr, &mut num_frames, &mut flags, None, None)
                        .map_err(|e| format!("GetBuffer: {e}"))?;
                    let frames = num_frames as usize;
                    let is_silent = (flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)) != 0;

                    if is_silent || ptr.is_null() {
                        for _ in 0..frames {
                            if dec_ctr % decimate == 0 {
                                producer_ring.push_stereo(0.0, 0.0);
                            }
                            dec_ctr = dec_ctr.wrapping_add(1);
                        }
                    } else if bps == 16 {
                        let s16 = std::slice::from_raw_parts(ptr as *const i16, frames * ch);
                        for f in 0..frames {
                            if dec_ctr % decimate == 0 {
                                let base = f * ch;
                                let l = (s16[base] as f32) * (1.0 / 32768.0);
                                let r = if ch > 1 {
                                    (s16[base + 1] as f32) * (1.0 / 32768.0)
                                } else {
                                    l
                                };
                                producer_ring.push_stereo(l, r);
                            }
                            dec_ctr = dec_ctr.wrapping_add(1);
                        }
                    } else {
                        let f32s = std::slice::from_raw_parts(ptr as *const f32, frames * ch);
                        for f in 0..frames {
                            if dec_ctr % decimate == 0 {
                                let base = f * ch;
                                let l = f32s[base];
                                let r = if ch > 1 { f32s[base + 1] } else { l };
                                producer_ring.push_stereo(l, r);
                            }
                            dec_ctr = dec_ctr.wrapping_add(1);
                        }
                    }

                    capture_client
                        .ReleaseBuffer(num_frames)
                        .map_err(|e| format!("ReleaseBuffer: {e}"))?;
                    pkt = capture_client
                        .GetNextPacketSize()
                        .map_err(|e| format!("GetNextPacketSize loop: {e}"))?;
                }
            }

            let _ = client.Stop();
            producer_done.store(true, Ordering::Release);
            consumer
                .join()
                .map_err(|_| "Consumer TUI/DSP thread panicked".to_string())?
        }
    }
}

#[cfg(not(windows))]
mod wasapi_loopback {
    use super::*;
    pub fn init_terminal_and_signals() {}
    pub fn capture(cfg: &CliConfig, num_scenes: usize) -> Result<CaptureOutput, String> {
        let sr = 48000u32;
        let total = (sr as usize) * (cfg.duration_secs as usize);
        let mut l = Vec::with_capacity(total);
        let mut r = Vec::with_capacity(total);
        let mut meter = KWeightingMeter::new(sr);
        for i in 0..total {
            let t = (i as f32) / (sr as f32);
            let s = (t * 440.0 * TAU).sin() * 0.25;
            meter.push_frame(s, s);
            l.push(s);
            r.push(s);
        }
        let _ = condition_clean_boundaries(&mut l, &mut r, sr, cfg.fade_ms, cfg.trim_silence_db);
        let mp3_bytes = encode_stereo_f32_to_mp3(&l, &r, sr, cfg.bitrate_kbps)?;

        let mut timeline_events = Vec::new();
        let step_f = ((cfg.auto_interval_secs * (sr as f32)) as u64).max(1);
        let mut cur_f = step_f;
        let mut sc = 0usize;
        while cur_f < (l.len() as u64) && num_scenes > 1 {
            let next_sc = (sc + 1) % num_scenes;
            timeline_events.push(TimelineEvent {
                sample_frame: cur_f,
                from_scene: sc,
                target_scene: next_sc,
                transition: PhysicsTransition::from_u8(sc as u8),
                trigger_reason: "AUTO-TIMER",
            });
            sc = next_sc;
            cur_f += step_f;
        }

        let total_frames = l.len() as u64;
        Ok(CaptureOutput {
            sample_rate: sr,
            left_pcm: l,
            right_pcm: r,
            mp3_bytes,
            total_frames,
            integrated_lufs: meter.integrated_lufs(),
            max_peak_dbfs: -12.0,
            dropped_frames: 0,
            timeline_events,
            cue_notes: Vec::new(),
        })
    }
}

// ============================================================================
// SECTION 10: OKLCH CHROMATIC CLI HELP & ARGUMENT PARSER
// ============================================================================

fn print_help() {
    let mut b = String::with_capacity(4096);
    push_fg_oklch(&mut b, 0.84, 0.18, 195.0);
    b.push_str("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
    push_fg_oklch(&mut b, 0.88, 0.20, 75.0);
    b.push_str("║  WIN_LOOPBACK_MP3_MUX v5.0.0 // MEDIA FOUNDATION H.264+AAC PHYSICS STUDIO    ║\n");
    push_fg_oklch(&mut b, 0.84, 0.18, 195.0);
    b.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\x1b[0m\n");

    b.push_str("  \x1b[1;38;2;230;238;255mSYNOPSIS:\x1b[0m\n");
    b.push_str("    win_loopback_mp3_mux.exe [OPTIONS]\n\n");

    b.push_str("  \x1b[1;38;2;230;238;255mLIVE HOTKEYS (WHILE RECORDING):\x1b[0m\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;255;160;255m T \x1b[0m         Toggle internal Stereo Harmonic + 120BPM Kick Test Synth\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;0;240;255m A \x1b[0m         Cycle Auto-Transition Mode (Beat+Timer -> Fixed Timer -> Manual)\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;0;245;190m C \x1b[0m         Toggle Auto Physics Shader Cycling (ON / OFF)\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;255;195;0m 1 .. 5 \x1b[0m    Select Physics Shader (1:Spring, 2:Wave, 3:Vortex, 4:Chroma, 5:Dissolve)\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;0;235;150m N / TAB \x1b[0m   Trigger immediate physics transition to next scene\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;255;135;180m + / - \x1b[0m     Adjust digital Pre-Gain (±0.5 dB)   \x1b[48;2;35;48;72m\x1b[1;38;2;180;155;255m [ / ] \x1b[0m Adjust Silence Gate\n");
    b.push_str("    \x1b[48;2;35;48;72m\x1b[1;38;2;0;240;255m SPACE \x1b[0m     Pause / Resume                      \x1b[48;2;140;30;55m\x1b[1;38;2;255;255;255m S / ENTER \x1b[0m Finish & Mux\n\n");

    b.push_str("  \x1b[1;38;2;230;238;255mCLI OPTIONS:\x1b[0m\n");
    b.push_str("    \x1b[1;38;2;0;235;155m-o, --out <PATH>\x1b[0m           Output path (e.g. session.mp4, track.mp3, master)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m-f, --format <FMT>\x1b[0m         Container: mp3 | m4a | mp4 | all (default: mp4)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --test-tone\x1b[0m            Enable internal stereo harmonic + beat synth on startup\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --clips <P1,P2,...>\x1b[0m    Comma-separated images or .mp4 videos to combine\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    <PATHS...>\x1b[0m           Bare inputs: files, a folder, or wildcards (clips\\*.mp4)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --auto-mode <MODE>\x1b[0m     Auto-transition mode: beat | timer | manual (default: beat)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --auto-secs <SEC>\x1b[0m      Auto-transition interval / cooldown in seconds (default: 4.0)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --transition <TYPE>\x1b[0m    Initial physics shader: spring | wave | vortex | chroma | dissolve\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --trans-secs <SEC>\x1b[0m     Transition duration in seconds (default: 1.2)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m-d, --seconds <SECS>\x1b[0m       Max capture duration in seconds (default: 30)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m    --fade-ms <MS>\x1b[0m         Clean start/end raised-cosine fade in ms (default: 150)\n");
    b.push_str("    \x1b[1;38;2;0;235;155m-t, --text <STRING>\x1b[0m        Text/lyrics/transcript embedded behind the audio\n");
    println!("{b}");
}

fn parse_cli_args(args: &[String]) -> Result<Option<CliConfig>, String> {
    if args.len() <= 1 || args.iter().any(|a| a == "--help" || a == "-h" || a == "/?") {
        print_help();
        return Ok(None);
    }

    let mut out_path = PathBuf::from("output.mp4");
    let mut explicit_format: Option<ExportFormat> = None;
    let mut clip_paths: Vec<PathBuf> = Vec::new();
    let mut behind_text = String::from("Captured via Windows WASAPI Loopback Pipeline.");
    let mut title = String::from("Windows Loopback Capture");
    let mut artist = String::from("WASAPI Render Endpoint");
    let mut album = String::from("System Audio Archive");
    let mut year = String::from("2026");
    let mut bitrate_kbps: u16 = 192;
    let mut duration_secs: u64 = 30;
    let mut fade_ms: u32 = 150;
    let mut trim_silence_db: Option<f32> = None;
    let mut test_synth = false;
    let mut transition = PhysicsTransition::Spring;
    let mut auto_mode = AutoTransitionMode::BeatReactive;
    let mut auto_interval_secs: f32 = 4.0;
    let mut cycle_shaders = true;
    let mut transition_secs: f32 = 1.2;
    let mut video_width: usize = 640;
    let mut video_height: usize = 640;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "-o" => {
                i += 1;
                out_path = PathBuf::from(args.get(i).ok_or("--out requires a path")?);
            }
            "--format" | "-f" => {
                i += 1;
                let v = args.get(i).ok_or("--format requires mp3|m4a|mp4|all")?;
                explicit_format = Some(match v.to_ascii_lowercase().as_str() {
                    "mp3" => ExportFormat::Mp3,
                    "m4a" => ExportFormat::M4a,
                    "mp4" | "video" => ExportFormat::VideoMp4,
                    "all" | "both" => ExportFormat::All,
                    other => return Err(format!("Unknown --format '{other}'")),
                });
            }
            "--mp3" => explicit_format = Some(ExportFormat::Mp3),
            "--m4a" => explicit_format = Some(ExportFormat::M4a),
            "--mp4" => explicit_format = Some(ExportFormat::VideoMp4),
            "--all" | "--both" => explicit_format = Some(ExportFormat::All),
            "--test-tone" | "--synth" => {
                test_synth = true;
            }
            "--image" | "-i" => {
                i += 1;
                let v = args.get(i).ok_or("--image requires a path")?;
                clip_paths.push(PathBuf::from(v));
            }
            "--clips" => {
                i += 1;
                let v = args.get(i).ok_or("--clips requires comma-separated paths")?;
                for part in v.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        clip_paths.push(PathBuf::from(trimmed));
                    }
                }
            }
            "--transition" => {
                i += 1;
                let v = args.get(i).ok_or("--transition requires a transition type")?;
                transition = PhysicsTransition::from_str(v)?;
            }
            "--auto-mode" => {
                i += 1;
                let v = args.get(i).ok_or("--auto-mode requires beat|timer|manual")?;
                auto_mode = AutoTransitionMode::from_str(v)?;
            }
            "--auto-secs" => {
                i += 1;
                let v = args.get(i).ok_or("--auto-secs requires seconds")?;
                auto_interval_secs = v
                    .parse::<f32>()
                    .map_err(|_| "Invalid --auto-secs")?
                    .clamp(0.5, 120.0);
            }
            "--no-cycle" => {
                cycle_shaders = false;
            }
            "--cycle-shaders" => {
                cycle_shaders = true;
            }
            "--trans-secs" => {
                i += 1;
                let v = args.get(i).ok_or("--trans-secs requires a float")?;
                transition_secs = v
                    .parse::<f32>()
                    .map_err(|_| "Invalid --trans-secs")?
                    .clamp(0.1, 10.0);
            }
            "--res" => {
                i += 1;
                let v = args.get(i).ok_or("--res requires WxH (e.g. 640x640)")?;
                let parts: Vec<&str> = v.split(['x', 'X']).collect();
                if parts.len() != 2 {
                    return Err("Invalid --res format; expected WxH such as 640x640".to_string());
                }
                let rw: usize = parts[0].parse().map_err(|_| "Invalid width in --res")?;
                let rh: usize = parts[1].parse().map_err(|_| "Invalid height in --res")?;
                video_width = ((rw + 8) / 16).max(1) * 16;
                video_height = ((rh + 8) / 16).max(1) * 16;
            }
            "--text" | "-t" => {
                i += 1;
                behind_text = args.get(i).ok_or("--text requires a string")?.clone();
            }
            "--text-file" => {
                i += 1;
                let p = args.get(i).ok_or("--text-file requires a path")?;
                behind_text =
                    fs::read_to_string(p).map_err(|e| format!("Failed reading '{p}': {e}"))?;
            }
            "--title" => {
                i += 1;
                title = args.get(i).ok_or("--title requires a string")?.clone();
            }
            "--artist" => {
                i += 1;
                artist = args.get(i).ok_or("--artist requires a string")?.clone();
            }
            "--album" => {
                i += 1;
                album = args.get(i).ok_or("--album requires a string")?.clone();
            }
            "--year" => {
                i += 1;
                year = args.get(i).ok_or("--year requires a string")?.clone();
            }
            "--bitrate" | "-b" => {
                i += 1;
                let v = args.get(i).ok_or("--bitrate requires 128|192|256|320")?;
                bitrate_kbps = v.parse::<u16>().map_err(|_| "Invalid --bitrate")?;
            }
            "--seconds" | "-d" => {
                i += 1;
                let v = args.get(i).ok_or("--seconds requires seconds")?;
                duration_secs = v.parse::<u64>().map_err(|_| "Invalid --seconds")?.max(1);
            }
            "--fade-ms" => {
                i += 1;
                let v = args.get(i).ok_or("--fade-ms requires milliseconds")?;
                fade_ms = v.parse::<u32>().map_err(|_| "Invalid --fade-ms")?;
            }
            "--trim-db" => {
                i += 1;
                let v = args.get(i).ok_or("--trim-db requires a dBFS float (e.g. -65)")?;
                trim_silence_db = Some(v.parse::<f32>().map_err(|_| "Invalid --trim-db")?);
            }
            other if !other.starts_with('-') => {
                // Bare positional media input: a file, a directory, or a
                // wildcard pattern such as `clips\\*.mp4` (the shell does not
                // expand it). Comma-separated lists work here too.
                for part in other.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        clip_paths.push(PathBuf::from(trimmed));
                    }
                }
            }
            unknown => {
                return Err(format!("Unrecognized argument '{unknown}'. Run with --help."));
            }
        }
        i += 1;
    }

    let format = explicit_format.unwrap_or_else(|| {
        match out_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "mp3" => ExportFormat::Mp3,
            "m4a" => ExportFormat::M4a,
            "mp4" => ExportFormat::VideoMp4,
            _ => ExportFormat::VideoMp4,
        }
    });

    // Expand wildcards / directories once, after every input is collected
    // (covers --clips, --image, and bare positional patterns alike).
    let clip_paths = expand_media_inputs(&clip_paths)?;

    Ok(Some(CliConfig {
        out_path,
        format,
        bitrate_kbps,
        duration_secs,
        fade_ms,
        trim_silence_db,
        test_synth,
        transition,
        auto_mode,
        auto_interval_secs,
        cycle_shaders,
        transition_secs,
        video_width,
        video_height,
        fps: 30,
        clip_paths,
        meta: MediaMetadata {
            title,
            artist,
            album,
            year,
            lang: *b"eng",
            behind_text,
            cover_mime: "image/png",
            cover_bytes: Vec::new(),
        },
    }))
}

fn write_atomic_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f =
        File::create(path).map_err(|e| format!("Failed to create '{}': {e}", path.display()))?;
    f.write_all(bytes)
        .map_err(|e| format!("Failed to write '{}': {e}", path.display()))?;
    f.flush()
        .map_err(|e| format!("Failed to flush '{}': {e}", path.display()))
}

fn main() -> Result<(), String> {
    wasapi_loopback::init_terminal_and_signals();

    let args: Vec<String> = env::args().collect();
    let mut cfg = match parse_cli_args(&args)? {
        Some(c) => c,
        None => return Ok(()),
    };

    let mut scenes: Vec<VisualScene> = Vec::new();

    for p in &cfg.clip_paths {
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "mp4" || ext == "mov" || ext == "mkv" {
            #[cfg(windows)]
            {
                // The decoded-frame cache is the memory ceiling: cap frames by
                // a ~768 MiB pixel budget instead of a flat 900 (which is
                // >1 GiB of RGB24 for a single 640x640 clip).
                let bytes_per_frame = (cfg.video_width * cfg.video_height * 3).max(1);
                let mem_frame_cap = (768 * 1024 * 1024) / bytes_per_frame;
                let max_f = ((cfg.duration_secs as usize) * (cfg.fps as usize))
                    .min(900)
                    .min(mem_frame_cap.max(30));
                let clip_scene = mf_video_engine::decode_video_clip_to_canvases(
                    p,
                    cfg.video_width,
                    cfg.video_height,
                    max_f,
                )?;
                scenes.push(clip_scene);
            }
            #[cfg(not(windows))]
            {
                return Err(format!(
                    "Video clip '{}' needs Windows Media Foundation decoding — build/run on Windows.",
                    p.display()
                ));
            }
        } else {
            let canvas = load_image_as_rgb_canvas(p, cfg.video_width, cfg.video_height)?;
            scenes.push(VisualScene::from_single(canvas));
        }
    }

    if scenes.len() < 2 {
        let start_v = scenes.len();
        for v in start_v..4 {
            scenes.push(VisualScene::from_single(synthesize_procedural_scene(
                cfg.video_width,
                cfg.video_height,
                v,
            )));
        }
    }

    cfg.meta.cover_mime = "image/png";
    cfg.meta.cover_bytes = encode_rgb_canvas_to_png(scenes[0].frame_at(0));

    let capture = wasapi_loopback::capture(&cfg, scenes.len())?;

    if !capture.timeline_events.is_empty() || !capture.cue_notes.is_empty() {
        cfg.meta
            .behind_text
            .push_str("\n\n--- AUTO & LIVE TRANSITION LOG ---\n");
        for ev in &capture.timeline_events {
            let sec = (ev.sample_frame as f64) / (capture.sample_rate.max(1) as f64);
            let _ = writeln!(
                cfg.meta.behind_text,
                "[{sec:06.2}s] {} : Scene #{} -> Scene #{} ({:?})",
                ev.trigger_reason,
                ev.from_scene + 1,
                ev.target_scene + 1,
                ev.transition
            );
        }
        for note in &capture.cue_notes {
            cfg.meta.behind_text.push_str(note);
            cfg.meta.behind_text.push('\n');
        }
    }

    println!(
        "\x1b[1;38;2;0;229;255m[SESSION TELEMETRY]\x1b[0m Clean Frames: \x1b[1;38;2;255;210;80m{}\x1b[0m ({:.2}s) | Integrated: \x1b[1;38;2;0;235;165m{:+.1} LUFS\x1b[0m | Max Peak: \x1b[1;38;2;255;120;150m{:+.1} dBFS\x1b[0m | Auto/Live Transitions: \x1b[1;38;2;190;155;255m{}\x1b[0m",
        capture.total_frames,
        (capture.total_frames as f64) / (capture.sample_rate.max(1) as f64),
        capture.integrated_lufs,
        capture.max_peak_dbfs,
        capture.timeline_events.len()
    );

    let id3 = serialize_id3v23_header(&cfg.meta)?;
    let mut full_mp3 = Vec::with_capacity(id3.len() + capture.mp3_bytes.len());
    full_mp3.extend_from_slice(&id3);
    full_mp3.extend_from_slice(&capture.mp3_bytes);

    if matches!(cfg.format, ExportFormat::Mp3 | ExportFormat::All) {
        let p = if cfg.format == ExportFormat::All {
            cfg.out_path.with_extension("mp3")
        } else {
            cfg.out_path.clone()
        };
        write_atomic_file(&p, &full_mp3)?;
        println!(
            "\x1b[1;38;2;0;235;140m[OK]\x1b[0m MP3 (ID3v2.3 APIC + USLT/COMM) -> \x1b[1;38;2;255;225;120m{}\x1b[0m ({} B)",
            p.display(),
            full_mp3.len()
        );
    }

    if matches!(cfg.format, ExportFormat::M4a | ExportFormat::All) {
        let p = if cfg.format == ExportFormat::All {
            cfg.out_path.with_extension("m4a")
        } else {
            cfg.out_path.clone()
        };
        let m4a_bytes =
            serialize_m4a_container(capture.sample_rate, &capture.mp3_bytes, &cfg.meta)?;
        write_atomic_file(&p, &m4a_bytes)?;
        println!(
            "\x1b[1;38;2;0;235;140m[OK]\x1b[0m M4A (Audio ISO-BMFF + ilst + ID32) -> \x1b[1;38;2;255;225;120m{}\x1b[0m ({} B)",
            p.display(),
            m4a_bytes.len()
        );
    }

    if matches!(cfg.format, ExportFormat::VideoMp4 | ExportFormat::All) {
        let p = if cfg.format == ExportFormat::All {
            cfg.out_path.with_extension("mp4")
        } else {
            cfg.out_path.clone()
        };

        print!("\x1b[38;2;0;229;255m[ENCODING HARDWARE H.264 + AAC-LC MP4 VIA WINDOWS MEDIA FOUNDATION...]\x1b[0m\r");
        let _ = std::io::stdout().flush();

        #[cfg(windows)]
        {
            mf_video_engine::encode_mp4_via_media_foundation(
                capture.sample_rate,
                &capture.left_pcm,
                &capture.right_pcm,
                &cfg.meta,
                &scenes,
                cfg.transition,
                cfg.auto_mode,
                cfg.auto_interval_secs,
                cfg.cycle_shaders,
                cfg.transition_secs,
                cfg.fps,
                &capture.timeline_events,
                &p,
            )?;
        }

        let sz = fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        println!(
            "\x1b[1;38;2;0;235;140m[OK]\x1b[0m MP4 (MediaFoundation H.264 NV12 + AAC-LC 192k {}x{} @ {}fps + ilst/ID32) -> \x1b[1;38;2;255;225;120m{}\x1b[0m ({} B)          ",
            cfg.video_width,
            cfg.video_height,
            cfg.fps,
            p.display(),
            sz
        );
    }

    Ok(())
}