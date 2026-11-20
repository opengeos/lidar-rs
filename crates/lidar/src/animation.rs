//! Shaded-relief rendering and animated GIF generation for inundation dynamics.
//!
//! Pure-Rust port of `7_Play_Animation.py` from `opengeos/lidar`'s ArcGIS
//! Hydrology Analyst toolbox. Computes multi-lamp directional hillshade with
//! `gist_earth` RGB blending and encodes `flood_*.tif` time-series rasters into
//! an animated GIF (with optional rising-then-falling ping-pong playback and
//! status banner) with zero C/C++ or GDAL dependencies.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use gif::{Encoder, Frame, Repeat};
use richdem::{Array2D, DemNum, load_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::raster_ops::ensure_no_path_collision;

/// Parameters for directional shaded relief (`hill_shade` in `7_Play_Animation.py`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HillshadeOptions {
    /// Lamp azimuth angle in degrees (default `135.0`).
    pub azimuth_deg: f64,
    /// Lamp elevation angle above horizon in degrees (default `45.0`).
    pub elevation_deg: f64,
    /// Ambient illumination weight (default `1.0`).
    pub ambient_weight: f64,
    /// Directional lamp illumination weight (default `5.0`).
    pub lamp_weight: f64,
    /// Vertical exaggeration applied to terrain gradients (default `1.0`).
    pub z_factor: f64,
}

impl Default for HillshadeOptions {
    fn default() -> Self {
        Self {
            azimuth_deg: 135.0,
            elevation_deg: 45.0,
            ambient_weight: 1.0,
            lamp_weight: 5.0,
            z_factor: 1.0,
        }
    }
}

/// Options controlling [`animate_inundation`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnimateInundationOptions {
    /// Frame delay in milliseconds (default `200`, minimum `20`).
    pub frame_delay_ms: u16,
    /// If `true`, play rising water followed by decreasing water (matching `7_Play_Animation.py`).
    pub ping_pong: bool,
    /// GIF loop count (`0` = loop forever, default `0`).
    pub loops: u16,
    /// Opacity of the inundated water overlay (`0.0..=1.0`, default `0.55`).
    pub water_alpha: f32,
    /// RGB color of the inundated water overlay (default `[30, 120, 230]`).
    pub water_rgb: [u8; 3],
    /// Optional maximum output width/height in pixels (nearest-neighbor downscale if larger).
    pub max_dimension: Option<usize>,
    /// Shaded relief rendering options.
    pub hillshade: HillshadeOptions,
    /// Optional time step in hours for labeling frames (`T = X.Xh`).
    pub time_step_h: Option<f64>,
    /// Optional rainfall intensity in cm/h for the banner title.
    pub rain_intensity_cm_h: Option<f64>,
}

impl Default for AnimateInundationOptions {
    fn default() -> Self {
        Self {
            frame_delay_ms: 200,
            ping_pong: true,
            loops: 0,
            water_alpha: 0.55,
            water_rgb: [30, 120, 230],
            max_dimension: None,
            hillshade: HillshadeOptions::default(),
            time_step_h: Some(1.0),
            rain_intensity_cm_h: None,
        }
    }
}

/// Output metadata produced by [`animate_inundation`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnimateInundationResult {
    /// Path to the generated animated GIF.
    pub gif_path: PathBuf,
    /// Number of frames written to the GIF.
    pub frame_count: usize,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels (including the top status banner).
    pub height: usize,
}

/// Piecewise linear approximation of matplotlib's `gist_earth` colormap (`t` in `[0, 1]`).
fn gist_earth_rgb(t: f64) -> [f64; 3] {
    const STOPS: [(f64, [f64; 3]); 9] = [
        (0.00, [0.00, 0.00, 0.00]),
        (0.12, [0.08, 0.22, 0.45]),
        (0.25, [0.18, 0.44, 0.48]),
        (0.38, [0.24, 0.56, 0.34]),
        (0.52, [0.47, 0.65, 0.31]),
        (0.66, [0.71, 0.68, 0.42]),
        (0.78, [0.76, 0.59, 0.47]),
        (0.90, [0.85, 0.77, 0.75]),
        (1.00, [0.99, 0.98, 0.99]),
    ];
    let t = t.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return [
                c0[0] + f * (c1[0] - c0[0]),
                c0[1] + f * (c1[1] - c0[1]),
                c0[2] + f * (c1[2] - c0[2]),
            ];
        }
    }
    STOPS[STOPS.len() - 1].1
}

/// Computes an RGB shaded-relief buffer (`3 * width * height` bytes) matching
/// `hill_shade` in `7_Play_Animation.py` (`gist_earth` + surface-normal RGB blending).
pub fn compute_hillshade_rgb(dem: &Array2D<f64>, opts: &HillshadeOptions) -> Vec<u8> {
    let w = dem.width();
    let h = dem.height();
    let nd = dem.no_data();
    let data = dem.data();

    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in data {
        if v.is_finite() && !v.rd_eq(nd) {
            if v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
        }
    }
    if !vmin.is_finite() || !vmax.is_finite() {
        return vec![128u8; w * h * 3];
    }
    let span = (vmax - vmin).max(1e-12);

    // Build finite elevation buffer (replacing NoData with vmin for stable central differences).
    let elev: Vec<f64> = data
        .iter()
        .map(|&v| {
            if v.is_finite() && !v.rd_eq(nd) {
                v
            } else {
                vmin
            }
        })
        .collect();

    let az_rad = opts.azimuth_deg.to_radians();
    let el_rad = opts.elevation_deg.to_radians();
    let light_h = el_rad.sin();
    let light_r = el_rad.cos() * az_rad.sin();
    let light_c = el_rad.cos() * az_rad.cos();

    let w_sum = (opts.ambient_weight + opts.lamp_weight).max(1e-12);
    let amb_w = opts.ambient_weight / w_sum;
    let lamp_w = opts.lamp_weight / w_sum;
    let zf = if opts.z_factor.is_finite() && opts.z_factor > 0.0 {
        opts.z_factor
    } else {
        1.0
    };

    let mut rgb = vec![0u8; w * h * 3];
    for r in 0..h {
        let r_prev = r.saturating_sub(1);
        let r_next = (r + 1).min(h.saturating_sub(1));
        let dr_denom = (r_next - r_prev).max(1) as f64;
        for c in 0..w {
            let idx = r * w + c;
            let raw = data[idx];
            if !raw.is_finite() || raw.rd_eq(nd) {
                rgb[idx * 3] = 32;
                rgb[idx * 3 + 1] = 32;
                rgb[idx * 3 + 2] = 32;
                continue;
            }
            let c_prev = c.saturating_sub(1);
            let c_next = (c + 1).min(w.saturating_sub(1));
            let dc_denom = (c_next - c_prev).max(1) as f64;

            let dr = (elev[r_next * w + c] - elev[r_prev * w + c]) / dr_denom * zf;
            let dc = (elev[r * w + c_next] - elev[r * w + c_prev]) / dc_denom * zf;

            // Cross product of vr=(dr, 1, 0) and vc=(dc, 0, 1) -> (1, -dr, -dc).
            let nx = 1.0;
            let ny = -dr;
            let nz = -dc;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let dot = ((nx * light_h + ny * light_r + nz * light_c) / n_len).clamp(0.0, 1.0);
            let intensity = (amb_w + lamp_w * dot).clamp(0.0, 1.0);

            let base = gist_earth_rgb((raw - vmin) / span);
            rgb[idx * 3] = (base[0] * intensity * 255.0).round().clamp(0.0, 255.0) as u8;
            rgb[idx * 3 + 1] = (base[1] * intensity * 255.0).round().clamp(0.0, 255.0) as u8;
            rgb[idx * 3 + 2] = (base[2] * intensity * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

/// Minimal 5×7 uppercase/digit/punctuation bitmap glyph table for zero-dependency HUD labels.
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        ':' => [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        _ => [0x00; 7],
    }
}

fn draw_banner(
    buf: &mut [u8],
    width: usize,
    banner_h: usize,
    text: &str,
    progress: f64,
    accent_rgb: [u8; 3],
) {
    // Dark slate background for banner
    for y in 0..banner_h {
        let row_off = y * width * 3;
        for x in 0..width {
            buf[row_off + x * 3] = 24;
            buf[row_off + x * 3 + 1] = 28;
            buf[row_off + x * 3 + 2] = 36;
        }
    }
    // 2px progress bar along the bottom of the banner
    let bar_w = ((width as f64) * progress.clamp(0.0, 1.0)).round() as usize;
    for y in banner_h.saturating_sub(3)..banner_h {
        let row_off = y * width * 3;
        for x in 0..bar_w.min(width) {
            buf[row_off + x * 3] = accent_rgb[0];
            buf[row_off + x * 3 + 1] = accent_rgb[1];
            buf[row_off + x * 3 + 2] = accent_rgb[2];
        }
    }
    // Draw text centered horizontally
    let char_w = 6usize;
    let total_w = text.len() * char_w;
    let start_x = width.saturating_sub(total_w) / 2;
    let start_y = banner_h.saturating_sub(10) / 2;
    for (ci, ch) in text.chars().enumerate() {
        let glyph = glyph_5x7(ch);
        let gx0 = start_x + ci * char_w;
        for (gy, &bits) in glyph.iter().enumerate() {
            let py = start_y + gy;
            if py >= banner_h.saturating_sub(3) {
                continue;
            }
            for gx in 0..5 {
                if (bits >> (4 - gx)) & 1 == 1 {
                    let px = gx0 + gx;
                    if px < width {
                        let off = (py * width + px) * 3;
                        buf[off] = 242;
                        buf[off + 1] = 246;
                        buf[off + 2] = 250;
                    }
                }
            }
        }
    }
}

fn downscale_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut dst = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        let sy = (y * sh / dh).min(sh - 1);
        for x in 0..dw {
            let sx = (x * sw / dw).min(sw - 1);
            let s = (sy * sw + sx) * 3;
            let d = (y * dw + x) * 3;
            dst[d..d + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    dst
}

fn parse_flood_frame_index(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_str()?;
    let lower = stem.to_ascii_lowercase();
    let suffix = lower.strip_prefix("flood_")?;
    suffix.parse::<usize>().ok()
}

fn discover_flood_frames(in_flood_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut frames: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(in_flood_dir)? {
        let entry = entry?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower.starts_with("flood_") && (lower.ends_with(".tif") || lower.ends_with(".tiff")) {
            frames.push(p);
        }
    }
    frames.sort_by(|a, b| {
        let ka = parse_flood_frame_index(a);
        let kb = parse_flood_frame_index(b);
        ka.cmp(&kb).then_with(|| a.file_name().cmp(&b.file_name()))
    });
    if frames.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "no flood_*.tif frames found in {}",
            in_flood_dir.display()
        )));
    }
    Ok(frames)
}

/// Shared internal GIF renderer used by both [`animate_inundation`] (disk frames -> file)
/// and [`animate_inundation_to_bytes`] (in-memory `fill_time` -> byte buffer).
fn render_inundation_gif<W, F>(
    dem: &Array2D<f64>,
    num_steps: usize,
    options: &AnimateInundationOptions,
    writer: W,
    mut load_wet_mask: F,
) -> Result<(usize, usize, usize)>
where
    W: std::io::Write,
    F: FnMut(usize) -> Result<Vec<bool>>,
{
    if num_steps == 0 || num_steps > 10_000 {
        return Err(Error::InvalidArgument(format!(
            "num_steps must be in 1..=10000, got {num_steps}"
        )));
    }
    if !options.water_alpha.is_finite() || !(0.0..=1.0).contains(&options.water_alpha) {
        return Err(Error::InvalidArgument(format!(
            "water_alpha must be in [0.0, 1.0], got {}",
            options.water_alpha
        )));
    }
    let sw = dem.width();
    let sh = dem.height();
    if sw == 0 || sh == 0 {
        return Err(Error::InvalidArgument("input DEM is empty".into()));
    }

    let (dw, dh) = match options.max_dimension {
        Some(max_dim) if max_dim > 0 && (sw > max_dim || sh > max_dim) => {
            if sw >= sh {
                let nh = ((sh as f64) * (max_dim as f64) / (sw as f64))
                    .round()
                    .max(1.0) as usize;
                (max_dim, nh)
            } else {
                let nw = ((sw as f64) * (max_dim as f64) / (sh as f64))
                    .round()
                    .max(1.0) as usize;
                (nw, max_dim)
            }
        }
        _ => (sw, sh),
    };

    let banner_h = 22usize;
    let total_h = dh + banner_h;
    if dw > u16::MAX as usize || total_h > u16::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "animation frame dimensions ({dw}x{total_h}) exceed GIF u16 limit (65535); set max_dimension"
        )));
    }

    let base_rgb_full = compute_hillshade_rgb(dem, &options.hillshade);
    let base_rgb = downscale_rgb(&base_rgb_full, sw, sh, dw, dh);

    let mut encoder = Encoder::new(writer, dw as u16, total_h as u16, &[])
        .map_err(|e| Error::Algorithm(format!("GIF encoder init failed: {e}")))?;
    let repeat = if options.loops == 0 {
        Repeat::Infinite
    } else {
        Repeat::Finite(options.loops)
    };
    encoder
        .set_repeat(repeat)
        .map_err(|e| Error::Algorithm(format!("GIF repeat header failed: {e}")))?;

    let delay_cs = (options.frame_delay_ms.max(20) / 10).max(2);
    let alpha = options.water_alpha;
    let inv_alpha = 1.0 - alpha;
    let [wr, wg, wb] = options.water_rgb;

    // Build playback sequence: rising (0..N), and if ping_pong, decreasing (N-2..=0).
    let mut sequence: Vec<(usize, bool)> = (0..num_steps).map(|i| (i, true)).collect();
    if options.ping_pong && num_steps > 1 {
        for i in (0..num_steps - 1).rev() {
            sequence.push((i, false));
        }
    }

    let mut frame_rgb = vec![0u8; dw * total_h * 3];
    for &(src_idx, rising) in &sequence {
        let wet = load_wet_mask(src_idx)?;
        if wet.len() != sw * sh {
            return Err(Error::DimensionMismatch(format!(
                "wet mask length ({}) does not match DEM ({sw}x{sh})",
                wet.len()
            )));
        }

        let progress = (src_idx + 1) as f64 / num_steps as f64;
        let time_label = match options.time_step_h {
            Some(dt) => format!("T = {:.1}H", (src_idx + 1) as f64 * dt),
            None => format!("STEP {}/{num_steps}", src_idx + 1),
        };
        let phase = if rising { "RISING" } else { "FALLING" };
        let banner_text = match options.rain_intensity_cm_h {
            Some(r) => format!("{phase}  {time_label}  ({r:.1} CM/H)"),
            None => format!("{phase}  {time_label}"),
        };
        draw_banner(
            &mut frame_rgb[..dw * banner_h * 3],
            dw,
            banner_h,
            &banner_text,
            progress,
            options.water_rgb,
        );

        let body = &mut frame_rgb[dw * banner_h * 3..];
        body.copy_from_slice(&base_rgb);
        for y in 0..dh {
            let sy = (y * sh / dh).min(sh - 1);
            for x in 0..dw {
                let sx = (x * sw / dw).min(sw - 1);
                if wet[sy * sw + sx] {
                    let p = (y * dw + x) * 3;
                    body[p] = (body[p] as f32 * inv_alpha + wr as f32 * alpha).round() as u8;
                    body[p + 1] =
                        (body[p + 1] as f32 * inv_alpha + wg as f32 * alpha).round() as u8;
                    body[p + 2] =
                        (body[p + 2] as f32 * inv_alpha + wb as f32 * alpha).round() as u8;
                }
            }
        }

        let mut gif_frame = Frame::from_rgb_speed(dw as u16, total_h as u16, &frame_rgb, 10);
        gif_frame.delay = delay_cs;
        encoder
            .write_frame(&gif_frame)
            .map_err(|e| Error::Algorithm(format!("GIF frame write failed: {e}")))?;
    }

    Ok((sequence.len(), dw, total_h))
}

/// Renders a DEM shaded relief and time-series `flood_*.tif` inundation overlays
/// into an animated GIF (`out_gif`).
///
/// Pure-Rust port of `7_Play_Animation.py`.
pub fn animate_inundation(
    in_dem: impl AsRef<Path>,
    in_flood_dir: impl AsRef<Path>,
    out_gif: impl AsRef<Path>,
    options: &AnimateInundationOptions,
) -> Result<AnimateInundationResult> {
    let in_dem = in_dem.as_ref();
    let in_flood_dir = in_flood_dir.as_ref();
    let out_gif = out_gif.as_ref();
    ensure_no_path_collision(in_dem, &[out_gif])?;

    let frames = discover_flood_frames(in_flood_dir)?;
    let dem = load_raster::<f64>(in_dem, None)?;
    let sw = dem.width();
    let sh = dem.height();

    // Validate dimensions of every flood frame upfront (dropping each raster immediately)
    // so errors never leave a truncated GIF and peak memory stays O(pixels) rather than
    // O(frames * pixels).
    for fpath in &frames {
        let p_str = fpath.to_str().ok_or_else(|| {
            Error::InvalidArgument(format!("non-UTF-8 path: {}", fpath.display()))
        })?;
        let ds = terrio::open_raster(p_str)?;
        if ds.raster.cols != sw || ds.raster.rows != sh {
            return Err(Error::DimensionMismatch(format!(
                "flood frame {} ({}x{}) does not match DEM ({sw}x{sh})",
                fpath.display(),
                ds.raster.cols,
                ds.raster.rows
            )));
        }
    }

    if let Some(parent) = out_gif.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = BufWriter::new(File::create(out_gif)?);
    let (frame_count, width, height) =
        render_inundation_gif(&dem, frames.len(), options, file, |src_idx| {
            let flood = load_raster::<f64>(&frames[src_idx], None)?;
            let fnd = flood.no_data();
            Ok(flood
                .data()
                .iter()
                .map(|&v| v.is_finite() && !v.rd_eq(fnd) && v > 0.0)
                .collect())
        })?;

    Ok(AnimateInundationResult {
        gif_path: out_gif.to_path_buf(),
        frame_count,
        width,
        height,
    })
}

/// Renders an in-memory animated inundation GIF directly from a DEM and a
/// `fill_time` raster (hours to inundation, as returned by
/// [`crate::simulate_inundation_in_memory`]), with zero filesystem I/O.
///
/// Returns `(gif_bytes, frame_count, width, height)`.
pub fn animate_inundation_to_bytes(
    dem: &Array2D<f64>,
    fill_time: &Array2D<f64>,
    num_steps: usize,
    time_step_h: f64,
    options: &AnimateInundationOptions,
) -> Result<(Vec<u8>, usize, usize, usize)> {
    if !time_step_h.is_finite() || time_step_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "time_step_h must be finite and > 0, got {time_step_h}"
        )));
    }
    if fill_time.width() != dem.width() || fill_time.height() != dem.height() {
        return Err(Error::DimensionMismatch(format!(
            "fill_time ({}x{}) does not match DEM ({}x{})",
            fill_time.width(),
            fill_time.height(),
            dem.width(),
            dem.height()
        )));
    }

    let mut effective_opts = options.clone();
    if effective_opts.time_step_h.is_none() {
        effective_opts.time_step_h = Some(time_step_h);
    }

    let ft_data = fill_time.data();
    let ft_nd = fill_time.no_data();
    let mut gif_bytes = Vec::new();
    let (frame_count, width, height) =
        render_inundation_gif(dem, num_steps, &effective_opts, &mut gif_bytes, |src_idx| {
            let t_hours = (src_idx + 1) as f64 * time_step_h;
            Ok(ft_data
                .iter()
                .map(|&ft| ft.is_finite() && !ft.rd_eq(ft_nd) && ft >= 0.0 && ft <= t_hours)
                .collect())
        })?;

    Ok((gif_bytes, frame_count, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use richdem::save_raster;

    #[test]
    fn generates_valid_animated_gif_from_synthetic_flood_frames() {
        let tmp = tempfile::tempdir().unwrap();
        let dem_path = tmp.path().join("dem.tif");
        let flood_dir = tmp.path().join("flood");
        std::fs::create_dir_all(&flood_dir).unwrap();

        let mut dem_data = vec![0.0; 20 * 20];
        for r in 0..20 {
            for c in 0..20 {
                let d = ((r as f64 - 10.0).powi(2) + (c as f64 - 10.0).powi(2)).sqrt();
                dem_data[r * 20 + c] = 100.0 + d;
            }
        }
        let dem = Array2D::from_vec(20, 20, dem_data).unwrap();
        save_raster(&dem, &dem_path, None, false).unwrap();

        for step in 1..=3 {
            let mut f = vec![0u32; 20 * 20];
            let radius = step as f64 * 2.5;
            for r in 0..20 {
                for c in 0..20 {
                    let d = ((r as f64 - 10.0).powi(2) + (c as f64 - 10.0).powi(2)).sqrt();
                    if d <= radius {
                        f[r * 20 + c] = 1;
                    }
                }
            }
            let mut arr = Array2D::from_vec(20, 20, f).unwrap();
            arr.set_no_data(0);
            save_raster(
                &arr,
                flood_dir.join(format!("flood_0{step}.tif")),
                None,
                false,
            )
            .unwrap();
        }

        let gif_path = tmp.path().join("inundation.gif");
        let res = animate_inundation(
            &dem_path,
            &flood_dir,
            &gif_path,
            &AnimateInundationOptions {
                ping_pong: true,
                rain_intensity_cm_h: Some(5.0),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(res.frame_count, 5); // 3 rising + 2 falling
        assert_eq!(res.width, 20);
        assert_eq!(res.height, 42); // 20 + 22 banner
        let bytes = std::fs::read(&gif_path).unwrap();
        assert!(bytes.starts_with(b"GIF89a"));
    }

    #[test]
    fn dimension_mismatch_leaves_no_partial_gif() {
        let tmp = tempfile::tempdir().unwrap();
        let dem_path = tmp.path().join("dem.tif");
        let flood_dir = tmp.path().join("flood");
        std::fs::create_dir_all(&flood_dir).unwrap();

        let dem = Array2D::from_vec(10, 10, vec![100.0; 100]).unwrap();
        let bad_flood = Array2D::from_vec(5, 5, vec![1u32; 25]).unwrap();
        save_raster(&dem, &dem_path, None, false).unwrap();
        save_raster(&bad_flood, flood_dir.join("flood_01.tif"), None, false).unwrap();

        let gif_path = tmp.path().join("should_not_exist.gif");
        let err = animate_inundation(
            &dem_path,
            &flood_dir,
            &gif_path,
            &AnimateInundationOptions::default(),
        );
        assert!(err.is_err());
        assert!(
            !gif_path.exists(),
            "partial GIF should not be created on error"
        );
    }

    #[test]
    fn animate_inundation_to_bytes_produces_valid_gif() {
        let mut dem_data = vec![0.0; 20 * 20];
        let mut ft_data = vec![-1.0; 20 * 20];
        for r in 0..20 {
            for c in 0..20 {
                let d = ((r as f64 - 10.0).powi(2) + (c as f64 - 10.0).powi(2)).sqrt();
                dem_data[r * 20 + c] = 100.0 + d;
                if d <= 6.0 {
                    ft_data[r * 20 + c] = d * 0.5;
                }
            }
        }
        let dem = Array2D::from_vec(20, 20, dem_data).unwrap();
        let mut ft = Array2D::from_vec(20, 20, ft_data).unwrap();
        ft.set_no_data(-1.0);

        let (bytes, frames, w, h) = animate_inundation_to_bytes(
            &dem,
            &ft,
            4,
            1.0,
            &AnimateInundationOptions {
                ping_pong: true,
                rain_intensity_cm_h: Some(5.0),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(frames, 7); // 4 rising + 3 falling
        assert_eq!(w, 20);
        assert_eq!(h, 42);
        assert!(bytes.starts_with(b"GIF89a"));

        // Decode the generated GIF bytes and verify every frame and dimension.
        let mut decoder = gif::DecodeOptions::new()
            .read_info(std::io::Cursor::new(&bytes))
            .unwrap();
        assert_eq!(decoder.width(), w as u16);
        assert_eq!(decoder.height(), h as u16);
        let mut decoded_frames = 0usize;
        while let Some(frame) = decoder.read_next_frame().unwrap() {
            assert_eq!(frame.width, w as u16);
            assert_eq!(frame.height, h as u16);
            decoded_frames += 1;
        }
        assert_eq!(decoded_frames, frames);
    }
}
