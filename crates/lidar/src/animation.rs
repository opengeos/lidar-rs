//! Shaded-relief rendering and animated GIF generation for inundation dynamics.
//!
//! Pure-Rust port of `7_Play_Animation.py` from `opengeos/lidar`'s ArcGIS
//! Hydrology Analyst toolbox. Computes multi-lamp directional hillshade with
//! `gist_earth` RGB blending and encodes `flood_*.tif` time-series rasters into
//! an animated GIF (with optional rising-then-falling ping-pong playback and
//! status banner) with zero C/C++ or GDAL dependencies.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use gif::{Encoder, Frame, Repeat};
use richdem::{Array2D, DemNum, load_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::inundation::SpillFlowPath;
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
    /// Optional time-varying scaled runoff hyetograph `Vec<(dt_h, intensity_cm_h)>` for per-frame
    /// cumulative runoff banner labels.
    #[cfg_attr(feature = "serde", serde(default))]
    pub scaled_hyetograph: Option<Vec<(f64, f64)>>,
    /// Draw a bottom-left scale bar in map units (default `false`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub show_scale_bar: bool,
    /// Draw a compact bottom-right color swatch legend (default `false`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub show_legend: bool,
    /// Optional override label for the primary legend swatch (defaults to `"WATER"` for inundation
    /// frames and `"DEPTH"` for DEM depth previews).
    #[cfg_attr(feature = "serde", serde(default))]
    pub legend_label: Option<String>,
    /// Optional vector polylines (in source DEM `(col, row)` coordinates) overlaid onto each frame.
    #[cfg_attr(feature = "serde", serde(default))]
    pub overlay_polylines: Vec<Vec<(usize, usize)>>,
    /// RGB color for `overlay_polylines` (default `[12, 60, 140]`).
    #[cfg_attr(feature = "serde", serde(default = "default_overlay_rgb"))]
    pub overlay_rgb: [u8; 3],
}

fn default_overlay_rgb() -> [u8; 3] {
    [12, 60, 140]
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
            scaled_hyetograph: None,
            show_scale_bar: false,
            show_legend: false,
            legend_label: None,
            overlay_polylines: Vec::new(),
            overlay_rgb: default_overlay_rgb(),
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

    // Scale finite differences by full pixel-vector lengths in map units for projected CRSs
    // (supporting rotated geotransforms), defaulting to 1.0 for geographic CRSs or missing
    // geotransforms.
    let (cell_x, cell_y) = if is_geographic_crs(&dem.projection, dem.geotransform) {
        (1.0, 1.0)
    } else {
        let dx = dem.geotransform[1].hypot(dem.geotransform[4]);
        let dy = dem.geotransform[2].hypot(dem.geotransform[5]);
        let cx = if dx.is_finite() && dx > 0.0 { dx } else { 1.0 };
        let cy = if dy.is_finite() && dy > 0.0 { dy } else { 1.0 };
        (cx, cy)
    };

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
        for c in 0..w {
            let idx = r * w + c;
            let raw = data[idx];
            if !raw.is_finite() || raw.rd_eq(nd) {
                rgb[idx * 3] = 32;
                rgb[idx * 3 + 1] = 32;
                rgb[idx * 3 + 2] = 32;
                continue;
            }
            // Substitute the center pixel's elevation for missing or NoData neighbors so
            // raster boundaries and NoData masks never produce artificial cliff halos.
            let up_val = if r > 0 {
                let v = data[(r - 1) * w + c];
                (v.is_finite() && !v.rd_eq(nd)).then_some(v)
            } else {
                None
            };
            let dn_val = if r + 1 < h {
                let v = data[(r + 1) * w + c];
                (v.is_finite() && !v.rd_eq(nd)).then_some(v)
            } else {
                None
            };
            let dr_steps = (usize::from(up_val.is_some()) + usize::from(dn_val.is_some())).max(1);
            let dr =
                (dn_val.unwrap_or(raw) - up_val.unwrap_or(raw)) / (dr_steps as f64 * cell_y) * zf;

            let lt_val = if c > 0 {
                let v = data[r * w + (c - 1)];
                (v.is_finite() && !v.rd_eq(nd)).then_some(v)
            } else {
                None
            };
            let rt_val = if c + 1 < w {
                let v = data[r * w + (c + 1)];
                (v.is_finite() && !v.rd_eq(nd)).then_some(v)
            } else {
                None
            };
            let dc_steps = (usize::from(lt_val.is_some()) + usize::from(rt_val.is_some())).max(1);
            let dc =
                (rt_val.unwrap_or(raw) - lt_val.unwrap_or(raw)) / (dc_steps as f64 * cell_x) * zf;

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
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
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

/// Pixel height of the status banner prepended to animated inundation GIF frames.
pub const ANIMATION_BANNER_HEIGHT: usize = 22;

fn compute_gif_dimensions(
    sw: usize,
    sh: usize,
    options: &AnimateInundationOptions,
) -> Result<(usize, usize, usize)> {
    if !options.water_alpha.is_finite() || !(0.0..=1.0).contains(&options.water_alpha) {
        return Err(Error::InvalidArgument(format!(
            "water_alpha must be in [0.0, 1.0], got {}",
            options.water_alpha
        )));
    }
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
    let total_h = dh + ANIMATION_BANNER_HEIGHT;
    if dw > u16::MAX as usize || total_h > u16::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "animation frame dimensions ({dw}x{total_h}) exceed GIF u16 limit (65535); set max_dimension"
        )));
    }
    Ok((dw, dh, total_h))
}

fn put_pixel_rgb(buf: &mut [u8], w: usize, h: usize, x: isize, y: isize, rgb: [u8; 3]) {
    if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
        return;
    }
    let p = ((y as usize) * w + (x as usize)) * 3;
    buf[p] = rgb[0];
    buf[p + 1] = rgb[1];
    buf[p + 2] = rgb[2];
}

fn draw_thick_segment(
    buf: &mut [u8],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    radius: isize,
    rgb: [u8; 3],
) {
    let (x0, y0) = p0;
    let (x1, y1) = p1;
    let steps = (x1 - x0).hypot(y1 - y0).ceil().max(1.0) as usize;
    for s in 0..=steps {
        let f = s as f64 / steps as f64;
        let cx = (x0 + (x1 - x0) * f).round() as isize;
        let cy = (y0 + (y1 - y0) * f).round() as isize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius + radius {
                    put_pixel_rgb(buf, w, h, cx + dx, cy + dy, rgb);
                }
            }
        }
    }
}

/// Draws active overland spill flow paths (whose upstream depression is full at `t_hours`,
/// before `merge_time_h`, and along each contiguous dry overland portion) onto the RGB frame body.
#[allow(clippy::too_many_arguments)]
fn draw_active_spill_paths(
    body: &mut [u8],
    dw: usize,
    dh: usize,
    sw: usize,
    sh: usize,
    wet: &[bool],
    t_hours: f64,
    spill_paths: &[SpillFlowPath],
) {
    if spill_paths.is_empty() || sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return;
    }
    let sx_scale = dw as f64 / sw as f64;
    let sy_scale = dh as f64 / sh as f64;
    let max_x = (dw - 1) as f64;
    let max_y = (dh - 1) as f64;
    let map_pt = |(c, r): (usize, usize)| -> (f64, f64) {
        (
            ((c as f64 + 0.5) * sx_scale - 0.5).clamp(0.0, max_x),
            ((r as f64 + 0.5) * sy_scale - 0.5).clamp(0.0, max_y),
        )
    };
    const HALO: [u8; 3] = [12, 32, 68];
    const CORE: [u8; 3] = [56, 189, 248];

    for path in spill_paths {
        if t_hours < path.spill_time_h
            || path.merge_time_h.is_some_and(|mt| t_hours >= mt)
            || path.cells.len() < 2
        {
            continue;
        }
        // Draw each contiguous unsubmerged (dry) overland run separately so channels never
        // bridge across submerged intermediate pockets.
        let n_cells = path.cells.len();
        let mut i = 0usize;
        while i < n_cells {
            let (c, r) = path.cells[i];
            if c >= sw || r >= sh || wet[r * sw + c] {
                i += 1;
                continue;
            }
            let run_start = i;
            while i + 1 < n_cells {
                let (nc, nr) = path.cells[i + 1];
                if nc < sw && nr < sh && !wet[nr * sw + nc] {
                    i += 1;
                } else {
                    break;
                }
            }
            let run_end = i;
            i += 1;

            let start = run_start.saturating_sub(1);
            let end = (run_end + 1).min(n_cells - 1);
            if end <= start {
                continue;
            }
            let seg = &path.cells[start..=end];

            // Dark halo pass, then vivid cyan core pass.
            for (radius, color) in [(2isize, HALO), (1isize, CORE)] {
                for pair in seg.windows(2) {
                    draw_thick_segment(
                        body,
                        dw,
                        dh,
                        map_pt(pair[0]),
                        map_pt(pair[1]),
                        radius,
                        color,
                    );
                }
            }

            // Directional arrowhead at downstream shoreline tip.
            let (tx, ty) = map_pt(seg[seg.len() - 1]);
            let (bx, by) = map_pt(seg[seg.len().saturating_sub(5)]);
            let dx = tx - bx;
            let dy = ty - by;
            let len = dx.hypot(dy);
            if len > 1e-3 {
                let ux = dx / len;
                let uy = dy / len;
                let head_len = 5.5;
                let spread = 0.52f64;
                let (sin_s, cos_s) = spread.sin_cos();
                let lx = tx - head_len * (ux * cos_s - uy * sin_s);
                let ly = ty - head_len * (ux * sin_s + uy * cos_s);
                let rx = tx - head_len * (ux * cos_s + uy * sin_s);
                let ry = ty - head_len * (-ux * sin_s + uy * cos_s);
                for (radius, color) in [(2isize, HALO), (1isize, CORE)] {
                    draw_thick_segment(body, dw, dh, (tx, ty), (lx, ly), radius, color);
                    draw_thick_segment(body, dw, dh, (tx, ty), (rx, ry), radius, color);
                }
            }
        }
    }
}

/// Shared internal GIF renderer used by [`animate_inundation`],
/// [`animate_inundation_to_bytes`], and [`animate_inundation_to_bytes_with_spill_paths`].
fn render_inundation_gif<W, F>(
    dem: &Array2D<f64>,
    num_steps: usize,
    options: &AnimateInundationOptions,
    spill_paths: &[SpillFlowPath],
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
    let sw = dem.width();
    let sh = dem.height();
    let (dw, dh, total_h) = compute_gif_dimensions(sw, sh, options)?;
    let banner_h = ANIMATION_BANNER_HEIGHT;

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
        let banner_text = if let (Some(hyeto), Some(dt)) =
            (options.scaled_hyetograph.as_deref(), options.time_step_h)
        {
            let t_hours = (src_idx + 1) as f64 * dt;
            let cum_mm = crate::inundation::cumulative_runoff_mm(
                t_hours,
                options.rain_intensity_cm_h.unwrap_or(0.0),
                Some(hyeto),
            );
            format!("{phase}  {time_label}  ({cum_mm:.0} MM)")
        } else {
            match options.rain_intensity_cm_h {
                Some(r) => format!("{phase}  {time_label}  ({r:.1} CM/H)"),
                None => format!("{phase}  {time_label}"),
            }
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

        if !options.overlay_polylines.is_empty() {
            draw_overlay_polylines(
                body,
                dw,
                dh,
                sw,
                sh,
                &options.overlay_polylines,
                options.overlay_rgb,
            );
        }

        if !spill_paths.is_empty()
            && let Some(dt) = options.time_step_h
        {
            let t_hours = (src_idx + 1) as f64 * dt;
            draw_active_spill_paths(body, dw, dh, sw, sh, &wet, t_hours, spill_paths);
        }

        let legend_label = options.legend_label.as_deref().unwrap_or("WATER");
        let has_spills = !spill_paths.is_empty();
        if options.show_scale_bar && !is_geographic_crs(&dem.projection, dem.geotransform) {
            let cell_m = dem.geotransform[1].hypot(dem.geotransform[4]).max(1e-6);
            let reserve_right = if options.show_legend {
                legend_box_width(legend_label, has_spills) + 12
            } else {
                0
            };
            draw_scale_bar(body, dw, dh, sw, cell_m, &dem.projection, reserve_right);
        }
        if options.show_legend {
            draw_legend_box(body, dw, dh, options.water_rgb, legend_label, has_spills);
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

    // Validate options, frame count, and output dimensions before touching the filesystem
    // so invalid inputs never create an empty/partial output GIF.
    if frames.is_empty() || frames.len() > 10_000 {
        return Err(Error::InvalidArgument(format!(
            "num_steps must be in 1..=10000, got {}",
            frames.len()
        )));
    }
    let _ = compute_gif_dimensions(sw, sh, options)?;

    // Validate dimensions of every flood frame upfront using lazy metadata inspection
    // (zero pixel decoding) so errors never leave a truncated GIF and each frame raster
    // is decoded only once during rendering.
    for fpath in &frames {
        let p_str = fpath.to_str().ok_or_else(|| {
            Error::InvalidArgument(format!("non-UTF-8 path: {}", fpath.display()))
        })?;
        let lazy = terrio::open_raster_lazy(p_str)?;
        if lazy.width() != sw || lazy.height() != sh {
            return Err(Error::DimensionMismatch(format!(
                "flood frame {} ({}x{}) does not match DEM ({sw}x{sh})",
                fpath.display(),
                lazy.width(),
                lazy.height()
            )));
        }
    }

    if let Some(parent) = out_gif.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    // Write the rendered GIF to a unique sibling temporary file and atomically rename into
    // place only after encoding and flushing succeed, so failures never truncate or corrupt
    // an existing `out_gif`.
    let parent_dir = out_gif
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (tmp_path, tmp_file) = create_sibling_temp_file(parent_dir, ".lidar_gif_", "gif")?;
    struct TempFileGuard(Option<PathBuf>);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            if let Some(p) = self.0.take() {
                let _ = std::fs::remove_file(p);
            }
        }
    }
    let mut guard = TempFileGuard(Some(tmp_path.clone()));

    let mut writer = BufWriter::new(tmp_file);
    let (frame_count, width, height) =
        render_inundation_gif(&dem, frames.len(), options, &[], &mut writer, |src_idx| {
            let flood = load_raster::<f64>(&frames[src_idx], None)?;
            let fnd = flood.no_data();
            Ok(flood
                .data()
                .iter()
                .map(|&v| v.is_finite() && !v.rd_eq(fnd) && v > 0.0)
                .collect())
        })?;
    writer.flush()?;
    drop(writer);

    std::fs::rename(&tmp_path, out_gif)?;
    guard.0 = None;

    Ok(AnimateInundationResult {
        gif_path: out_gif.to_path_buf(),
        frame_count,
        width,
        height,
    })
}

fn create_sibling_temp_file(parent: &Path, prefix: &str, ext: &str) -> Result<(PathBuf, File)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    for _ in 0..1024 {
        let seq = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!("{prefix}{pid}_{seq}.{ext}"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(f) => return Ok((candidate, f)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(Error::Io(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "failed to allocate unique temporary file for GIF output",
    )))
}

/// Loads a DEM raster for preview rendering, reading a matching GeoTIFF/COG overview via
/// `terrio::open_raster_lazy` when `allow_overview` is true, `max_dim > 0`, and an overview
/// exists whose largest dimension is still `>= max_dim` and smaller than native resolution.
///
/// When `open_raster_lazy` succeeds but no suitable overview exists, full-resolution band 1
/// is read directly from the already-opened lazy handle rather than reopening and re-decoding
/// the raster from disk a second time. Falls back to [`load_raster`] only when
/// `allow_overview` is false or `open_raster_lazy` cannot open the format (e.g., legacy ESRI
/// ASCII grids).
pub fn load_dem_for_preview(
    path: impl AsRef<Path>,
    max_dim: usize,
    allow_overview: bool,
) -> Result<Array2D<f64>> {
    let path = path.as_ref();
    if allow_overview
        && max_dim > 0
        && let Some(p_str) = path.to_str()
        && let Ok(mut lazy) = terrio::open_raster_lazy(p_str)
    {
        let native_w = lazy.width();
        let native_h = lazy.height();
        let chosen_ov = if native_w.max(native_h) > max_dim {
            lazy.overviews()
                .iter()
                .enumerate()
                .rfind(|&(_, &(ow, oh))| {
                    ow > 0 && oh > 0 && ow.max(oh) >= max_dim && (ow < native_w || oh < native_h)
                })
                .map(|(idx, &dims)| (idx, dims))
        } else {
            None
        };
        let ds_res = if let Some((ov_idx, (ow, oh))) = chosen_ov {
            lazy.read_window(
                terrio::Window::full(ow, oh),
                &terrio::WindowOptions {
                    bands: vec![1],
                    overview: Some(ov_idx),
                },
            )
        } else if native_w > 0 && native_h > 0 {
            lazy.read_window(
                terrio::Window::full(native_w, native_h),
                &terrio::WindowOptions {
                    bands: vec![1],
                    overview: None,
                },
            )
        } else {
            lazy.read()
        };
        if let Ok(ds) = ds_res
            && let Ok(dyn_arr) = richdem::dataset_to_dynamic(ds, p_str)
        {
            return Ok(dyn_arr.to_f64());
        }
    }
    Ok(load_raster::<f64>(path, None)?)
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
    animate_inundation_to_bytes_with_spill_paths(
        dem,
        fill_time,
        num_steps,
        time_step_h,
        &[],
        options,
    )
}

/// Like [`animate_inundation_to_bytes`], additionally rendering active overland
/// [`SpillFlowPath`] channels (with directional arrowheads) once each upstream depression
/// fills and before the connecting saddle submerges.
pub fn animate_inundation_to_bytes_with_spill_paths(
    dem: &Array2D<f64>,
    fill_time: &Array2D<f64>,
    num_steps: usize,
    time_step_h: f64,
    spill_paths: &[SpillFlowPath],
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
    effective_opts.time_step_h = Some(time_step_h);

    let ft_data = fill_time.data();
    let ft_nd = fill_time.no_data();
    let mut gif_bytes = Vec::new();
    let (frame_count, width, height) = render_inundation_gif(
        dem,
        num_steps,
        &effective_opts,
        spill_paths,
        &mut gif_bytes,
        |src_idx| {
            let t_hours = (src_idx + 1) as f64 * time_step_h;
            Ok(ft_data
                .iter()
                .map(|&ft| ft.is_finite() && !ft.rd_eq(ft_nd) && (0.0..=t_hours).contains(&ft))
                .collect())
        },
    )?;

    Ok((gif_bytes, frame_count, width, height))
}

fn draw_text_5x7(
    buf: &mut [u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    text: &str,
    rgb: [u8; 3],
) {
    let char_w = 6usize;
    for (ci, ch) in text.chars().enumerate() {
        let glyph = glyph_5x7(ch);
        let gx0 = x0 + ci * char_w;
        for (gy, row_bits) in glyph.iter().enumerate() {
            let py = y0 + gy;
            if py >= height {
                continue;
            }
            for gx in 0..5usize {
                if (row_bits & (1 << (4 - gx))) != 0 {
                    let px = gx0 + gx;
                    if px < width {
                        let p = (py * width + px) * 3;
                        buf[p] = rgb[0];
                        buf[p + 1] = rgb[1];
                        buf[p + 2] = rgb[2];
                    }
                }
            }
        }
    }
}

fn draw_overlay_polylines(
    body: &mut [u8],
    dw: usize,
    dh: usize,
    sw: usize,
    sh: usize,
    polylines: &[Vec<(usize, usize)>],
    rgb: [u8; 3],
) {
    if dw == 0 || dh == 0 || sw == 0 || sh == 0 {
        return;
    }
    let sx = dw as f64 / sw as f64;
    let sy = dh as f64 / sh as f64;
    let max_x = (dw - 1) as f64;
    let max_y = (dh - 1) as f64;
    let map_x =
        |c: usize| -> isize { ((c as f64 + 0.5) * sx - 0.5).clamp(0.0, max_x).round() as isize };
    let map_y =
        |r: usize| -> isize { ((r as f64 + 0.5) * sy - 0.5).clamp(0.0, max_y).round() as isize };
    for line in polylines {
        for seg in line.windows(2) {
            let x0 = map_x(seg[0].0);
            let y0 = map_y(seg[0].1);
            let x1 = map_x(seg[1].0);
            let y1 = map_y(seg[1].1);
            let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
            for s in 0..=steps {
                let px = x0 + (x1 - x0) * s / steps;
                let py = y0 + (y1 - y0) * s / steps;
                if px >= 0 && py >= 0 && (px as usize) < dw && (py as usize) < dh {
                    let p = (py as usize * dw + px as usize) * 3;
                    body[p] = rgb[0];
                    body[p + 1] = rgb[1];
                    body[p + 2] = rgb[2];
                }
            }
        }
    }
}

fn draw_scale_bar(
    body: &mut [u8],
    dw: usize,
    dh: usize,
    sw: usize,
    cell_units: f64,
    projection: &str,
    reserve_right: usize,
) {
    let avail_w = dw.saturating_sub(reserve_right + 16);
    if dw < 90 || dh < 36 || avail_w < 36 || !cell_units.is_finite() || cell_units <= 0.0 {
        return;
    }
    let p_lower = projection.to_ascii_lowercase();
    let unit_is_us_survey_feet = p_lower.contains("us_survey_foot")
        || p_lower.contains("us survey foot")
        || p_lower.contains("+units=us-ft");
    let unit_is_intl_feet = !unit_is_us_survey_feet
        && (p_lower.contains("foot") || p_lower.contains("feet") || p_lower.contains("+units=ft"));
    let cell_m = if unit_is_us_survey_feet {
        cell_units * (1200.0 / 3937.0)
    } else if unit_is_intl_feet {
        cell_units * 0.3048
    } else {
        cell_units
    };
    let total_m = sw as f64 * cell_m;
    let max_bar_px = (dw as f64 * 0.35).min(avail_w.saturating_sub(12) as f64);
    if max_bar_px < 16.0 {
        return;
    }
    let max_bar_m = (max_bar_px / dw as f64) * total_m;
    let target_m = total_m * 0.22;
    let nice_vals = [
        0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0,
        10_000.0, 25_000.0, 50_000.0,
    ];
    let Some(bar_m) = nice_vals
        .iter()
        .copied()
        .filter(|&v| v <= max_bar_m && ((v / total_m) * dw as f64) >= 16.0)
        .min_by(|a, b| (a - target_m).abs().total_cmp(&(b - target_m).abs()))
    else {
        return;
    };
    let bar_px = ((bar_m / total_m) * dw as f64).round() as usize;
    if bar_px < 16 {
        return;
    }
    let label = if bar_m >= 1000.0 {
        format!("{} KM", (bar_m / 1000.0).round() as u32)
    } else if bar_m >= 1.0 {
        format!("{} M", bar_m.round() as u32)
    } else {
        format!("{bar_m:.1} M")
    };
    let bx0 = 10usize;
    let by0 = dh.saturating_sub(24);
    let pad = 4usize;
    let label_px = label.len() * 6;
    let box_w = (bar_px.max(label_px) + 2 * pad).min(dw.saturating_sub(bx0));
    if reserve_right > 0 && bx0 + box_w + 6 > dw.saturating_sub(reserve_right) {
        return;
    }
    let box_h = 20usize.min(dh.saturating_sub(by0));
    for y in by0..(by0 + box_h) {
        for x in bx0..(bx0 + box_w) {
            let p = (y * dw + x) * 3;
            body[p] = ((body[p] as u16 + 24) / 3) as u8;
            body[p + 1] = ((body[p + 1] as u16 + 28) / 3) as u8;
            body[p + 2] = ((body[p + 2] as u16 + 36) / 3) as u8;
        }
    }
    draw_text_5x7(body, dw, dh, bx0 + pad, by0 + 2, &label, [245, 248, 255]);
    let bar_y = by0 + 12;
    let half = bar_px / 2;
    for y in bar_y..(bar_y + 4).min(dh) {
        for x in 0..bar_px {
            let px = bx0 + pad + x;
            if px < dw {
                let p = (y * dw + px) * 3;
                let c = if x < half { 245 } else { 40 };
                body[p] = c;
                body[p + 1] = c;
                body[p + 2] = c;
            }
        }
    }
}

fn legend_box_width(label: &str, has_spills: bool) -> usize {
    let text_cols = label.chars().count().max(if has_spills { 5 } else { 0 });
    (17 + text_cols * 6 + 6).max(84)
}

fn draw_legend_box(
    body: &mut [u8],
    dw: usize,
    dh: usize,
    water_rgb: [u8; 3],
    label: &str,
    has_spills: bool,
) {
    let box_w = legend_box_width(label, has_spills);
    let box_h = if has_spills { 30usize } else { 18usize };
    if dw < box_w + 16 || dh < box_h + 16 {
        return;
    }
    let lx0 = dw - box_w - 8;
    let ly0 = dh - box_h - 8;
    for y in ly0..(ly0 + box_h) {
        for x in lx0..(lx0 + box_w) {
            let p = (y * dw + x) * 3;
            body[p] = ((body[p] as u16 + 20) / 3) as u8;
            body[p + 1] = ((body[p + 1] as u16 + 24) / 3) as u8;
            body[p + 2] = ((body[p + 2] as u16 + 32) / 3) as u8;
        }
    }
    // Primary swatch
    for y in (ly0 + 5)..(ly0 + 11) {
        for x in (lx0 + 5)..(lx0 + 13) {
            let p = (y * dw + x) * 3;
            body[p] = water_rgb[0];
            body[p + 1] = water_rgb[1];
            body[p + 2] = water_rgb[2];
        }
    }
    draw_text_5x7(body, dw, dh, lx0 + 17, ly0 + 5, label, [245, 248, 255]);
    if has_spills {
        for y in (ly0 + 17)..(ly0 + 23) {
            for x in (lx0 + 5)..(lx0 + 13) {
                let p = (y * dw + x) * 3;
                body[p] = 56;
                body[p + 1] = 189;
                body[p + 2] = 248;
            }
        }
        draw_text_5x7(body, dw, dh, lx0 + 17, ly0 + 17, "SPILL", [245, 248, 255]);
    }
}

/// Encodes an RGB24 buffer (`3 * width * height` bytes) into a pure-Rust PNG byte vector.
pub fn encode_rgb_to_png_bytes(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    if rgb.len() != width * height * 3 {
        return Err(Error::DimensionMismatch(format!(
            "RGB buffer length {} does not match {width}x{height}x3",
            rgb.len()
        )));
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| Error::Algorithm(format!("PNG header error: {e}")))?;
        writer
            .write_image_data(rgb)
            .map_err(|e| Error::Algorithm(format!("PNG encode error: {e}")))?;
    }
    Ok(out)
}

/// Returns `true` when the DEM projection metadata indicates geographic (angular degree)
/// coordinates where a linear metric scale bar would be misleading.
pub(crate) fn is_geographic_crs(projection: &str, geotransform: [f64; 6]) -> bool {
    let trimmed = projection.trim();
    if !trimmed.is_empty() {
        if let Ok(srs) = terrio::crs::SpatialRef::parse(trimmed)
            && !srs.is_unknown()
        {
            return srs.is_geographic();
        }
        let p = trimmed.to_ascii_lowercase();
        let is_projected = p.contains("projcs[")
            || p.contains("projcrs[")
            || p.contains("+proj=utm")
            || p.contains("+proj=aea")
            || p.contains("+proj=lcc")
            || p.contains("+proj=merc");
        if !is_projected
            && (p.contains("geogcs[")
                || p.contains("geogcrs[")
                || p.contains("epsg:4326")
                || p.contains("epsg:4269")
                || p.contains("+proj=longlat"))
        {
            return true;
        }
        return false;
    }
    let cell = geotransform[1].hypot(geotransform[4]);
    cell > 0.0 && cell < 1e-2
}

/// Renders a single shaded-relief inundation frame at `t_hours` into PNG bytes
/// with banner, optional active spill paths, optional scale bar, and optional legend.
pub fn render_inundation_frame_png_bytes(
    dem: &Array2D<f64>,
    fill_time: &Array2D<f64>,
    t_hours: f64,
    progress: f64,
    spill_paths: &[SpillFlowPath],
    options: &AnimateInundationOptions,
) -> Result<Vec<u8>> {
    if !t_hours.is_finite() || t_hours < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "t_hours must be finite and >= 0, got {t_hours}"
        )));
    }
    if !progress.is_finite() || progress < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "progress must be finite and >= 0, got {progress}"
        )));
    }
    let sw = dem.width();
    let sh = dem.height();
    crate::inundation::validate_grid_alignment(dem, fill_time, "fill_time")?;
    let (dw, dh, total_h) = compute_gif_dimensions(sw, sh, options)?;
    let full_bg = compute_hillshade_rgb(dem, &options.hillshade);
    let bg = downscale_rgb(&full_bg, sw, sh, dw, dh);
    let alpha = options.water_alpha.clamp(0.0, 1.0);
    let inv_alpha = 1.0 - alpha;
    let [wr, wg, wb] = options.water_rgb;
    let ft_data = fill_time.data();
    let ft_nd = fill_time.no_data();
    let cell_m = dem.geotransform[1].hypot(dem.geotransform[4]).max(1e-6);

    let wet: Vec<bool> = ft_data
        .iter()
        .map(|&ft| ft.is_finite() && !ft.rd_eq(ft_nd) && (0.0..=t_hours).contains(&ft))
        .collect();

    let mut frame_rgb = vec![0u8; dw * total_h * 3];
    let label = if let Some(hyeto) = options.scaled_hyetograph.as_deref() {
        let cum_mm = crate::inundation::cumulative_runoff_mm(
            t_hours,
            options.rain_intensity_cm_h.unwrap_or(0.0),
            Some(hyeto),
        );
        format!("T = {t_hours:.1}H  ({cum_mm:.0} MM)")
    } else {
        match options.rain_intensity_cm_h {
            Some(r) => format!("T = {t_hours:.1}H  ({:.0} MM)", t_hours * r * 10.0),
            None => format!("T = {t_hours:.1}H"),
        }
    };
    draw_banner(
        &mut frame_rgb,
        dw,
        ANIMATION_BANNER_HEIGHT,
        &label,
        progress.clamp(0.0, 1.0),
        [56, 189, 248],
    );

    let body = &mut frame_rgb[ANIMATION_BANNER_HEIGHT * dw * 3..];
    body.copy_from_slice(&bg);
    for dr in 0..dh {
        let sr = dr * sh / dh;
        for dc in 0..dw {
            let sc = dc * sw / dw;
            if wet[sr * sw + sc] {
                let p = (dr * dw + dc) * 3;
                body[p] = (body[p] as f32 * inv_alpha + wr as f32 * alpha).round() as u8;
                body[p + 1] = (body[p + 1] as f32 * inv_alpha + wg as f32 * alpha).round() as u8;
                body[p + 2] = (body[p + 2] as f32 * inv_alpha + wb as f32 * alpha).round() as u8;
            }
        }
    }
    if !options.overlay_polylines.is_empty() {
        draw_overlay_polylines(
            body,
            dw,
            dh,
            sw,
            sh,
            &options.overlay_polylines,
            options.overlay_rgb,
        );
    }
    if !spill_paths.is_empty() {
        draw_active_spill_paths(body, dw, dh, sw, sh, &wet, t_hours, spill_paths);
    }
    let legend_label = options.legend_label.as_deref().unwrap_or("WATER");
    let has_spills = !spill_paths.is_empty();
    if options.show_scale_bar && !is_geographic_crs(&dem.projection, dem.geotransform) {
        let reserve_right = if options.show_legend {
            legend_box_width(legend_label, has_spills) + 12
        } else {
            0
        };
        draw_scale_bar(body, dw, dh, sw, cell_m, &dem.projection, reserve_right);
    }
    if options.show_legend {
        draw_legend_box(body, dw, dh, options.water_rgb, legend_label, has_spills);
    }
    encode_rgb_to_png_bytes(&frame_rgb, dw, total_h)
}

/// Renders a shaded-relief PNG preview of `dem` with an optional positive-depth overlay
/// (e.g., `depth.tif` or `sink.tif`), optional scale bar, and optional legend.
pub fn render_dem_preview_png_bytes(
    dem: &Array2D<f64>,
    depth_overlay: Option<&Array2D<f64>>,
    options: &AnimateInundationOptions,
) -> Result<Vec<u8>> {
    let sw = dem.width();
    let sh = dem.height();
    if let Some(ov) = depth_overlay {
        crate::inundation::validate_grid_alignment(dem, ov, "depth_overlay")?;
    }
    let (dw, dh, _) = compute_gif_dimensions(sw, sh, options)?;
    let full_bg = compute_hillshade_rgb(dem, &options.hillshade);
    let mut rgb = downscale_rgb(&full_bg, sw, sh, dw, dh);
    let cell_m = dem.geotransform[1].hypot(dem.geotransform[4]).max(1e-6);

    if let Some(ov) = depth_overlay {
        let alpha = options.water_alpha.clamp(0.0, 1.0);
        let inv_alpha = 1.0 - alpha;
        let [wr, wg, wb] = options.water_rgb;
        let ov_data = ov.data();
        let ov_nd = ov.no_data();
        for dr in 0..dh {
            let sr = dr * sh / dh;
            for dc in 0..dw {
                let sc = dc * sw / dw;
                let v = ov_data[sr * sw + sc];
                if v.is_finite() && !v.rd_eq(ov_nd) && v > 0.0 {
                    let p = (dr * dw + dc) * 3;
                    rgb[p] = (rgb[p] as f32 * inv_alpha + wr as f32 * alpha).round() as u8;
                    rgb[p + 1] = (rgb[p + 1] as f32 * inv_alpha + wg as f32 * alpha).round() as u8;
                    rgb[p + 2] = (rgb[p + 2] as f32 * inv_alpha + wb as f32 * alpha).round() as u8;
                }
            }
        }
    }
    let legend_label = options.legend_label.as_deref().unwrap_or("DEPTH");
    if options.show_scale_bar && !is_geographic_crs(&dem.projection, dem.geotransform) {
        let reserve_right = if options.show_legend && depth_overlay.is_some() {
            legend_box_width(legend_label, false) + 12
        } else {
            0
        };
        draw_scale_bar(&mut rgb, dw, dh, sw, cell_m, &dem.projection, reserve_right);
    }
    if options.show_legend && depth_overlay.is_some() {
        draw_legend_box(&mut rgb, dw, dh, options.water_rgb, legend_label, false);
    }
    encode_rgb_to_png_bytes(&rgb, dw, dh)
}

/// Exports individual time-series inundation frames as numbered PNG files (`frame_001.png`, ...)
/// with optional banner, scale bar, legend, spill paths, and vector overlays.
pub fn export_inundation_frames_png(
    dem: &Array2D<f64>,
    fill_time: &Array2D<f64>,
    num_steps: usize,
    time_step_h: f64,
    spill_paths: &[SpillFlowPath],
    out_dir: impl AsRef<Path>,
    options: &AnimateInundationOptions,
) -> Result<Vec<PathBuf>> {
    if !time_step_h.is_finite() || time_step_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "time_step_h must be finite and > 0, got {time_step_h}"
        )));
    }
    if num_steps == 0 || num_steps > 10_000 {
        return Err(Error::InvalidArgument(format!(
            "num_steps must be in 1..=10000, got {num_steps}"
        )));
    }
    let sw = dem.width();
    let sh = dem.height();
    if fill_time.width() != sw || fill_time.height() != sh {
        return Err(Error::DimensionMismatch(format!(
            "fill_time ({}x{}) does not match DEM ({sw}x{sh})",
            fill_time.width(),
            fill_time.height()
        )));
    }
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let (dw, dh, total_h) = compute_gif_dimensions(sw, sh, options)?;
    let full_bg = compute_hillshade_rgb(dem, &options.hillshade);
    let bg = downscale_rgb(&full_bg, sw, sh, dw, dh);
    let alpha = options.water_alpha.clamp(0.0, 1.0);
    let inv_alpha = 1.0 - alpha;
    let [wr, wg, wb] = options.water_rgb;
    let ft_data = fill_time.data();
    let ft_nd = fill_time.no_data();
    let digits = (num_steps as f64).log10().floor() as usize + 1;
    let cell_m = dem.geotransform[1].hypot(dem.geotransform[4]).max(1e-6);
    let draw_scale =
        options.show_scale_bar && !is_geographic_crs(&dem.projection, dem.geotransform);

    let mut frame_rgb = vec![0u8; dw * total_h * 3];
    let mut paths = Vec::with_capacity(num_steps);

    for src_idx in 0..num_steps {
        let t_hours = (src_idx + 1) as f64 * time_step_h;
        let wet: Vec<bool> = ft_data
            .iter()
            .map(|&ft| ft.is_finite() && !ft.rd_eq(ft_nd) && (0.0..=t_hours).contains(&ft))
            .collect();

        let label = if let Some(hyeto) = options.scaled_hyetograph.as_deref() {
            let cum_mm = crate::inundation::cumulative_runoff_mm(
                t_hours,
                options.rain_intensity_cm_h.unwrap_or(0.0),
                Some(hyeto),
            );
            format!("T = {t_hours:.1}H  ({cum_mm:.0} MM)")
        } else {
            match options.rain_intensity_cm_h {
                Some(r) => format!("T = {t_hours:.1}H  ({:.0} MM)", t_hours * r * 10.0),
                None => format!("T = {t_hours:.1}H"),
            }
        };
        let progress = (src_idx + 1) as f64 / num_steps as f64;
        draw_banner(
            &mut frame_rgb,
            dw,
            ANIMATION_BANNER_HEIGHT,
            &label,
            progress,
            [56, 189, 248],
        );

        let body = &mut frame_rgb[ANIMATION_BANNER_HEIGHT * dw * 3..];
        body.copy_from_slice(&bg);
        for dr in 0..dh {
            let sr = dr * sh / dh;
            for dc in 0..dw {
                let sc = dc * sw / dw;
                if wet[sr * sw + sc] {
                    let p = (dr * dw + dc) * 3;
                    body[p] = (body[p] as f32 * inv_alpha + wr as f32 * alpha).round() as u8;
                    body[p + 1] =
                        (body[p + 1] as f32 * inv_alpha + wg as f32 * alpha).round() as u8;
                    body[p + 2] =
                        (body[p + 2] as f32 * inv_alpha + wb as f32 * alpha).round() as u8;
                }
            }
        }
        if !options.overlay_polylines.is_empty() {
            draw_overlay_polylines(
                body,
                dw,
                dh,
                sw,
                sh,
                &options.overlay_polylines,
                options.overlay_rgb,
            );
        }
        if !spill_paths.is_empty() {
            draw_active_spill_paths(body, dw, dh, sw, sh, &wet, t_hours, spill_paths);
        }
        let legend_label = options.legend_label.as_deref().unwrap_or("WATER");
        let has_spills = !spill_paths.is_empty();
        if draw_scale {
            let reserve_right = if options.show_legend {
                legend_box_width(legend_label, has_spills) + 12
            } else {
                0
            };
            draw_scale_bar(body, dw, dh, sw, cell_m, &dem.projection, reserve_right);
        }
        if options.show_legend {
            draw_legend_box(body, dw, dh, options.water_rgb, legend_label, has_spills);
        }

        let png_bytes = encode_rgb_to_png_bytes(&frame_rgb, dw, total_h)?;
        let out_path = out_dir.join(format!("frame_{:0digits$}.png", src_idx + 1));
        std::fs::write(&out_path, png_bytes)?;
        paths.push(out_path);
    }
    Ok(paths)
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

    #[test]
    fn animate_inundation_to_bytes_with_spill_paths_renders_channels() {
        let mut dem_data = vec![0.0; 24 * 24];
        let mut ft_data = vec![-1.0; 24 * 24];
        for r in 0..24 {
            for c in 0..24 {
                dem_data[r * 24 + c] = 100.0 + (r as f64) * 0.1;
                if r <= 4 && c <= 4 {
                    ft_data[r * 24 + c] = 1.5;
                } else if r >= 18 && c >= 18 {
                    ft_data[r * 24 + c] = 2.0;
                }
            }
        }
        let dem = Array2D::from_vec(24, 24, dem_data).unwrap();
        let mut ft = Array2D::from_vec(24, 24, ft_data).unwrap();
        ft.set_no_data(-1.0);

        // Steps are at T = 1.0 h (before spill), T = 2.0 h (active spill), T = 3.0 h (at merge).
        let spill = SpillFlowPath {
            from_dep_id: 1,
            from_level: 1,
            to_dep_id: 2,
            spill_time_h: 1.5,
            merge_time_h: Some(3.0),
            spill_elev: 101.0,
            cells: (4..=18).map(|i| (i, i)).collect(),
        };
        let opts = AnimateInundationOptions {
            ping_pong: false,
            ..Default::default()
        };

        let (plain_bytes, frames0, w0, h0) =
            animate_inundation_to_bytes(&dem, &ft, 3, 1.0, &opts).unwrap();
        let (spill_bytes, frames1, w1, h1) =
            animate_inundation_to_bytes_with_spill_paths(&dem, &ft, 3, 1.0, &[spill], &opts)
                .unwrap();

        assert_eq!((frames0, w0, h0), (frames1, w1, h1));
        assert!(spill_bytes.starts_with(b"GIF89a"));

        // Decode both GIFs to RGB per frame and verify channel timing:
        // frame 0 (T=1.0h < spill_time_h): channel region identical to plain GIF;
        // frame 1 (T=2.0h in [spill_time_h, merge_time_h)): channel visibly painted;
        // frame 2 (T=3.0h >= merge_time_h): channel disappears and matches plain GIF.
        let decode_rgb_frames = |bytes: &[u8]| -> Vec<Vec<u8>> {
            let mut dec_opts = gif::DecodeOptions::new();
            dec_opts.set_color_output(gif::ColorOutput::RGBA);
            let mut decoder = dec_opts.read_info(std::io::Cursor::new(bytes)).unwrap();
            let mut out = Vec::new();
            while let Some(frame) = decoder.read_next_frame().unwrap() {
                out.push(frame.buffer.to_vec());
            }
            out
        };
        let plain_frames = decode_rgb_frames(&plain_bytes);
        let spill_frames = decode_rgb_frames(&spill_bytes);
        assert_eq!(plain_frames.len(), 3);
        assert_eq!(spill_frames.len(), 3);

        let channel_diff_count = |a: &[u8], b: &[u8]| -> usize {
            (6..=16)
                .filter(|&i| {
                    let y = i + ANIMATION_BANNER_HEIGHT;
                    let p = (y * w0 + i) * 4;
                    a[p..p + 3] != b[p..p + 3]
                })
                .count()
        };
        assert_eq!(
            channel_diff_count(&plain_frames[0], &spill_frames[0]),
            0,
            "channel must not render before spill_time_h"
        );
        assert!(
            channel_diff_count(&plain_frames[1], &spill_frames[1]) >= 8,
            "channel must render along active dry path between spill_time_h and merge_time_h"
        );
        assert_eq!(
            channel_diff_count(&plain_frames[2], &spill_frames[2]),
            0,
            "channel must disappear at merge_time_h"
        );
    }

    #[test]
    fn geographic_crs_detects_nad83_epsg_4269_and_projected_utms() {
        assert!(is_geographic_crs(
            "EPSG:4269",
            [0.0, 0.0001, 0.0, 0.0, 0.0, -0.0001]
        ));
        assert!(is_geographic_crs(
            "EPSG:4326",
            [0.0, 0.0001, 0.0, 0.0, 0.0, -0.0001]
        ));
        assert!(!is_geographic_crs(
            "EPSG:26915",
            [500_000.0, 1.0, 0.0, 4_500_000.0, 0.0, -1.0]
        ));
        assert!(!is_geographic_crs(
            "EPSG:5070",
            [0.0, 10.0, 0.0, 0.0, 0.0, -10.0]
        ));
    }

    #[test]
    fn hillshade_avoids_nodata_cliff_halo_and_scales_by_cell_spacing() {
        // 7x7 flat plateau at 500.0 with a NoData border and a single low valid cell (100.0)
        // in the corner so vmin = 100.0 << plateau.
        let mut data = vec![-9999.0; 7 * 7];
        for r in 1..6 {
            for c in 1..6 {
                data[r * 7 + c] = 500.0;
            }
        }
        data[0] = 100.0;
        let mut dem = Array2D::from_vec(7, 7, data).unwrap();
        dem.set_no_data(-9999.0);
        dem.projection = "EPSG:26915".to_string();
        dem.geotransform = [500_000.0, 1.0, 0.0, 4_500_000.0, 0.0, -1.0];

        let rgb = compute_hillshade_rgb(&dem, &HillshadeOptions::default());
        let interior = &rgb[(3 * 7 + 3) * 3..(3 * 7 + 3) * 3 + 3];
        let border_by_nodata = &rgb[(3 * 7 + 5) * 3..(3 * 7 + 5) * 3 + 3];
        assert_eq!(
            interior, border_by_nodata,
            "plateau pixel adjacent to NoData must not suffer a cliff halo"
        );

        // Cell-spacing invariance: same physical 0.5 slope sampled at 1m vs 10m produces identical
        // hillshade brightness for pixels at the same normalized elevation.
        let mut ramp_1m = vec![0.0; 5 * 5];
        let mut ramp_10m = vec![0.0; 5 * 5];
        for r in 0..5 {
            for c in 0..5 {
                ramp_1m[r * 5 + c] = 100.0 + 0.5 * (c as f64);
                ramp_10m[r * 5 + c] = 100.0 + 0.5 * (c as f64 * 10.0);
            }
        }
        let mut d1 = Array2D::from_vec(5, 5, ramp_1m).unwrap();
        d1.projection = "EPSG:26915".to_string();
        d1.geotransform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let mut d10 = Array2D::from_vec(5, 5, ramp_10m).unwrap();
        d10.projection = "EPSG:26915".to_string();
        d10.geotransform = [0.0, 10.0, 0.0, 0.0, 0.0, -10.0];

        let rgb1 = compute_hillshade_rgb(&d1, &HillshadeOptions::default());
        let rgb10 = compute_hillshade_rgb(&d10, &HillshadeOptions::default());
        assert_eq!(rgb1, rgb10);

        // Rotated geotransform with pixel vector length 10m (6, 8) yields identical spacing to 10m.
        let mut d10_rot = d10.clone();
        d10_rot.geotransform = [0.0, 6.0, 8.0, 0.0, 8.0, -6.0];
        let rgb10_rot = compute_hillshade_rgb(&d10_rot, &HillshadeOptions::default());
        assert_eq!(rgb10, rgb10_rot);
    }

    #[test]
    fn render_inundation_frame_validates_time_and_renders_antecedent_water_at_t0() {
        let dem = Array2D::from_vec(12, 12, vec![100.0; 144]).unwrap();
        let mut ft_data = vec![-1.0; 144];
        ft_data[6 * 12 + 6] = 0.0; // antecedent water at t=0
        ft_data[6 * 12 + 7] = 2.0; // inundated later
        let mut ft = Array2D::from_vec(12, 12, ft_data).unwrap();
        ft.set_no_data(-1.0);

        let opts = AnimateInundationOptions::default();
        assert!(render_inundation_frame_png_bytes(&dem, &ft, f64::NAN, 0.0, &[], &opts).is_err());
        assert!(render_inundation_frame_png_bytes(&dem, &ft, -0.5, 0.0, &[], &opts).is_err());
        assert!(render_inundation_frame_png_bytes(&dem, &ft, 0.0, f64::NAN, &[], &opts).is_err());
        assert!(render_inundation_frame_png_bytes(&dem, &ft, 0.0, -0.1, &[], &opts).is_err());

        let png_t0 = render_inundation_frame_png_bytes(&dem, &ft, 0.0, 0.0, &[], &opts).unwrap();
        let mut ft_dry = Array2D::from_vec(12, 12, vec![-1.0; 144]).unwrap();
        ft_dry.set_no_data(-1.0);
        let png_dry =
            render_inundation_frame_png_bytes(&dem, &ft_dry, 0.0, 0.0, &[], &opts).unwrap();
        assert_ne!(
            png_t0, png_dry,
            "t_hours == 0.0 must render cells with fill_time == 0.0"
        );

        // Hyetograph option renders valid GIF bytes.
        let hyeto_opts = AnimateInundationOptions {
            ping_pong: false,
            scaled_hyetograph: Some(vec![(1.0, 2.0), (1.0, 6.0)]),
            show_legend: true,
            ..Default::default()
        };
        let (gif_bytes, frames, _, _) =
            animate_inundation_to_bytes(&dem, &ft, 2, 1.0, &hyeto_opts).unwrap();
        assert_eq!(frames, 2);
        assert!(gif_bytes.starts_with(b"GIF89a"));

        // Depth preview legend uses "DEPTH" by default (distinct from "WATER").
        let depth = Array2D::from_vec(120, 60, vec![1.0; 120 * 60]).unwrap();
        let big_dem = Array2D::from_vec(120, 60, vec![100.0; 120 * 60]).unwrap();
        let depth_default = render_dem_preview_png_bytes(
            &big_dem,
            Some(&depth),
            &AnimateInundationOptions {
                show_legend: true,
                ..Default::default()
            },
        )
        .unwrap();
        let depth_water = render_dem_preview_png_bytes(
            &big_dem,
            Some(&depth),
            &AnimateInundationOptions {
                show_legend: true,
                legend_label: Some("WATER".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_ne!(depth_default, depth_water);
    }

    #[test]
    fn load_dem_for_preview_uses_cog_overview_and_reuses_lazy_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let src_tif = tmp.path().join("dem_1024.tif");
        let cog_tif = tmp.path().join("dem_1024_cog.tif");

        let (w, h) = (1024usize, 1024usize);
        let mut data = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                let dx = c as f64 - 512.0;
                let dy = r as f64 - 512.0;
                data.push(100.0 + 0.02 * (dx * dx + dy * dy).sqrt());
            }
        }
        let mut dem = Array2D::from_vec(w, h, data).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [500_000.0, 1.0, 0.0, 4_001_024.0, 0.0, -1.0];
        richdem::save_raster(&dem, &src_tif, None, false).unwrap();

        crate::raster_ops::image_to_cog(&src_tif, &cog_tif, None).unwrap();

        let lazy = terrio::open_raster_lazy(cog_tif.to_str().unwrap()).unwrap();
        assert!(
            !lazy.overviews().is_empty(),
            "expected COG overviews for 1024x1024 raster"
        );

        let ov_arr = load_dem_for_preview(&cog_tif, 256, true).unwrap();
        assert!(
            ov_arr.width() < w && ov_arr.height() < h,
            "expected overview smaller than native 1024x1024, got {}x{}",
            ov_arr.width(),
            ov_arr.height()
        );
        assert!(
            ov_arr.width().max(ov_arr.height()) >= 256,
            "expected overview >= max_dim (256), got {}x{}",
            ov_arr.width(),
            ov_arr.height()
        );

        // Without overviews on src_tif, lazy handle reads full resolution cleanly.
        let src_arr = load_dem_for_preview(&src_tif, 256, true).unwrap();
        assert_eq!((src_arr.width(), src_arr.height()), (w, h));

        // When allow_overview is false (e.g. overlay alignment), full resolution is loaded.
        let full_arr = load_dem_for_preview(&cog_tif, 256, false).unwrap();
        assert_eq!((full_arr.width(), full_arr.height()), (w, h));
    }
}
