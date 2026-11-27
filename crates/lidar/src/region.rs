//! Connected-component region grouping and geometric/topographic region properties.
//!
//! Pure-Rust replacement for `scipy.ndimage.label` + `regionGroup` and
//! `skimage.measure.regionprops` used throughout `opengeos/lidar`.

use std::collections::HashSet;
use std::f64::consts::{FRAC_PI_4, PI};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::SpatialRef;

/// CRS-aware metric cell dimensions and linear-unit scale factor for a raster grid.
///
/// For projected CRSs (meters, US survey feet, international feet, etc.), `to_meter` is
/// the CRS linear unit scale factor (`1.0` for metric or unknown CRSs) and
/// `cell_width_m`, `cell_height_m`, and `cell_area_m2` are converted directly into SI
/// meters and square meters.
///
/// For geographic CRSs (e.g., `EPSG:4326`, `EPSG:4269`), cell dimensions are computed
/// using WGS 84 ellipsoidal prime-vertical and meridional radii of curvature evaluated at
/// the raster's center latitude, and `to_meter` stores the geometric-mean meters-per-degree
/// scale factor at that latitude.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CrsMetricScale {
    /// `true` when the CRS is geographic (angular degrees), `false` for projected or unknown CRSs.
    pub is_geographic: bool,
    /// Linear-unit conversion factor to meters (`m / map_unit` for projected CRSs; geometric-mean
    /// meters per degree at center latitude for geographic CRSs).
    pub to_meter: f64,
    /// Cell column spacing in meters.
    pub cell_width_m: f64,
    /// Cell row spacing in meters.
    pub cell_height_m: f64,
    /// Cell surface area in square meters.
    pub cell_area_m2: f64,
}

/// Computes CRS-aware metric cell spacing and cell area in SI meters (`m`, `m^2`) from
/// a raster's projection string, GDAL-style 6-element affine geotransform, width, and height.
///
/// - Projected CRSs (meters, US survey feet, international feet): reads `to_meter` from
///   `terrio::crs::SpatialRef::parse(projection)` (defaulting to `1.0` when unknown).
/// - Geographic CRSs (`EPSG:4326`, `EPSG:4269`, etc.): evaluates the WGS 84 ellipsoidal
///   radii of curvature at the raster's center `(width / 2, height / 2)` latitude to convert
///   degree spacings into accurate local geodesic meters.
pub fn crs_metric_scale(
    projection: &str,
    geotransform: &[f64; 6],
    width: usize,
    height: usize,
) -> CrsMetricScale {
    let (_, _, scale) = crs_metric_basis(projection, geotransform, width, height);
    scale
}

/// Computes the metric column basis vector `vx_m` (meters per `+1 col`), metric row basis
/// vector `vy_m` (meters per `+1 row`), and [`CrsMetricScale`] for a raster grid.
pub(crate) fn crs_metric_basis(
    projection: &str,
    geotransform: &[f64; 6],
    width: usize,
    height: usize,
) -> ((f64, f64), (f64, f64), CrsMetricScale) {
    let (vx, vy, raw_area) = affine_basis(geotransform);
    let trimmed = projection.trim();
    if !trimmed.is_empty()
        && let Ok(srs) = SpatialRef::parse(trimmed)
        && !srs.is_unknown()
    {
        if srs.is_geographic() {
            // WGS 84 ellipsoid parameters.
            const WGS84_A: f64 = 6_378_137.0;
            const WGS84_F: f64 = 1.0 / 298.257_223_563;
            const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

            let mid_lat_raw = geotransform[3]
                + 0.5 * (width as f64) * geotransform[4]
                + 0.5 * (height as f64) * geotransform[5];
            if !mid_lat_raw.is_finite() {
                return (
                    (f64::NAN, f64::NAN),
                    (f64::NAN, f64::NAN),
                    CrsMetricScale {
                        is_geographic: true,
                        to_meter: f64::NAN,
                        cell_width_m: f64::NAN,
                        cell_height_m: f64::NAN,
                        cell_area_m2: f64::NAN,
                    },
                );
            }
            let mid_lat_deg = mid_lat_raw.clamp(-89.999_999, 89.999_999);
            let phi = mid_lat_deg.to_radians();
            let sin_phi = phi.sin();
            let cos_phi = phi.cos().abs();
            let w_sq = (1.0 - WGS84_E2 * sin_phi * sin_phi).max(1e-12);
            let w_sqrt = w_sq.sqrt();
            // Prime vertical (east-west) and meridional (north-south) radii of curvature.
            let n_rad = WGS84_A / w_sqrt;
            let m_rad = WGS84_A * (1.0 - WGS84_E2) / (w_sq * w_sqrt);
            let m_per_deg_lon = (PI / 180.0) * n_rad * cos_phi;
            let m_per_deg_lat = (PI / 180.0) * m_rad;

            let vx_m = (vx.0 * m_per_deg_lon, vx.1 * m_per_deg_lat);
            let vy_m = (vy.0 * m_per_deg_lon, vy.1 * m_per_deg_lat);
            let cell_width_m = vx_m.0.hypot(vx_m.1);
            let cell_height_m = vy_m.0.hypot(vy_m.1);
            let cell_area_m2 = (vx_m.0 * vy_m.1 - vy_m.0 * vx_m.1).abs();
            let to_meter = (m_per_deg_lon * m_per_deg_lat).sqrt();
            return (
                vx_m,
                vy_m,
                CrsMetricScale {
                    is_geographic: true,
                    to_meter,
                    cell_width_m,
                    cell_height_m,
                    cell_area_m2,
                },
            );
        }

        let to_meter = srs
            .projection_params()
            .map(|p| p.to_meter)
            .filter(|&v| v.is_finite() && v > 0.0)
            .unwrap_or(1.0);
        let vx_m = (vx.0 * to_meter, vx.1 * to_meter);
        let vy_m = (vy.0 * to_meter, vy.1 * to_meter);
        let cell_width_m = vx_m.0.hypot(vx_m.1);
        let cell_height_m = vy_m.0.hypot(vy_m.1);
        let cell_area_m2 = raw_area * to_meter * to_meter;
        return (
            vx_m,
            vy_m,
            CrsMetricScale {
                is_geographic: false,
                to_meter,
                cell_width_m,
                cell_height_m,
                cell_area_m2,
            },
        );
    }

    let cell_width_m = vx.0.hypot(vx.1);
    let cell_height_m = vy.0.hypot(vy.1);
    (
        vx,
        vy,
        CrsMetricScale {
            is_geographic: false,
            to_meter: 1.0,
            cell_width_m,
            cell_height_m,
            cell_area_m2: raw_area,
        },
    )
}

/// Computes a safe metric basis `(vx_m, vy_m, cell_width_m, cell_height_m, cell_area_m2)`
/// from a raster's CRS projection, geotransform, and dimensions, falling back to native
/// affine basis (or unit basis) when the geotransform or CRS yields non-finite or
/// non-positive cell spacing/area.
pub(crate) fn safe_crs_metric_basis(
    projection: &str,
    geotransform: &[f64; 6],
    width: usize,
    height: usize,
) -> ((f64, f64), (f64, f64), f64, f64, f64) {
    // Reject degenerate/zero-size raw geotransforms before CRS unit conversion so
    // geographic CRSs do not scale fallback unit vectors by meters-per-degree.
    let raw_rx = geotransform[1].hypot(geotransform[4]);
    let raw_ry = geotransform[2].hypot(geotransform[5]);
    let raw_det = (geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]).abs();
    if !raw_rx.is_finite()
        || raw_rx <= 0.0
        || !raw_ry.is_finite()
        || raw_ry <= 0.0
        || !raw_det.is_finite()
        || raw_det <= 0.0
    {
        return ((1.0, 0.0), (0.0, -1.0), 1.0, 1.0, 1.0);
    }

    let (vx_m, vy_m, scale) = crs_metric_basis(projection, geotransform, width, height);
    if scale.cell_width_m.is_finite()
        && scale.cell_width_m > 0.0
        && scale.cell_height_m.is_finite()
        && scale.cell_height_m > 0.0
        && scale.cell_area_m2.is_finite()
        && scale.cell_area_m2 > 0.0
        && vx_m.0.is_finite()
        && vx_m.1.is_finite()
        && vy_m.0.is_finite()
        && vy_m.1.is_finite()
    {
        (
            vx_m,
            vy_m,
            scale.cell_width_m,
            scale.cell_height_m,
            scale.cell_area_m2,
        )
    } else if scale.is_geographic {
        // Geographic CRS with non-finite/degenerate geotransform origin: fall back to unit
        // pixel-scale basis (1.0 per cell) rather than exposing raw decimal degrees as meters.
        ((1.0, 0.0), (0.0, -1.0), 1.0, 1.0, 1.0)
    } else {
        (
            (geotransform[1], geotransform[4]),
            (geotransform[2], geotransform[5]),
            raw_rx,
            raw_ry,
            raw_det,
        )
    }
}

/// Returns `(meters_per_x_unit, meters_per_y_unit)` for converting raw CRS coordinate
/// differences `(dx, dy)` into SI meters at the raster center.
pub(crate) fn crs_axis_meters_per_unit(
    projection: &str,
    geotransform: &[f64; 6],
    width: usize,
    height: usize,
) -> (f64, f64) {
    let unit_gt = [geotransform[0], 1.0, 0.0, geotransform[3], 0.0, -1.0];
    let (vx_m, vy_m, scale) = crs_metric_basis(projection, &unit_gt, 0, 0);
    if scale.is_geographic {
        // Evaluate latitude at the true raster center rather than the top-left corner.
        let mid_lat_raw = geotransform[3]
            + 0.5 * (width as f64) * geotransform[4]
            + 0.5 * (height as f64) * geotransform[5];
        if !mid_lat_raw.is_finite() {
            return (1.0, 1.0);
        }
        let mid_gt = [geotransform[0], 1.0, 0.0, mid_lat_raw, 0.0, -1.0];
        let (mvx, mvy, _) = crs_metric_basis(projection, &mid_gt, 0, 0);
        let sx = mvx.0.abs();
        let sy = mvy.1.abs();
        if sx.is_finite() && sx > 0.0 && sy.is_finite() && sy > 0.0 {
            return (sx, sy);
        }
        return (1.0, 1.0);
    }
    let sx = vx_m.0.abs();
    let sy = vy_m.1.abs();
    if sx.is_finite() && sx > 0.0 && sy.is_finite() && sy > 0.0 {
        (sx, sy)
    } else {
        (1.0, 1.0)
    }
}

/// Geometric and elevation statistics of a single labeled connected component.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegionProps {
    /// 1-based connected component label.
    pub label: u32,
    /// Number of pixels in the region.
    pub count: usize,
    /// Bounding box `(min_row, min_col, max_row, max_col)` with exclusive maxima.
    pub bbox: (usize, usize, usize, usize),
    /// Coordinates `(row, col)` of the first pixel encountered in row-major order.
    pub first_pixel: (usize, usize),
    /// Surface area (`count * resolution^2`).
    pub area: f64,
    /// Depression/feature volume (`mean_depth * count * resolution^2`).
    pub volume: f64,
    /// Mean depth relative to `max_elev`.
    pub mean_depth: f64,
    /// Maximum depth (`max_elev - min_elev`).
    pub max_depth: f64,
    /// Minimum elevation inside the region.
    pub min_elev: f64,
    /// Maximum (spill/boundary) elevation inside the region.
    pub max_elev: f64,
    /// Perimeter in SI meters when CRS metadata is present (or native map units otherwise;
    /// Benkrid & Crookes 4-neighborhood estimator).
    pub perimeter: f64,
    /// Major axis length of the equivalent ellipse in SI meters (or native map units without CRS).
    pub major_axis: f64,
    /// Minor axis length of the equivalent ellipse in SI meters (clamped to >= pixel spacing).
    pub minor_axis: f64,
    /// Ratio `major_axis / minor_axis`.
    pub elongatedness: f64,
    /// Eccentricity of the equivalent ellipse in `[0, 1]`.
    pub eccentricity: f64,
    /// Orientation angle in degrees (`[-90, 90]`).
    pub orientation: f64,
    /// Extent: ratio of pixel count to bounding-box pixel area.
    pub area_bbox_ratio: f64,
}

#[inline]
fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Labels 4-connected foreground regions over any slice `&[T]` and removes any
/// component with pixel count `<= min_size`, returning `(labels, nb_labels)` where
/// `0` is background and regions are numbered `1..=nb_labels` in row-major scan order.
///
/// # Panics
///
/// Panics if `data.len() != width * height` (or if `width * height` overflows `usize`).
pub fn region_group_slice<T: Copy, F: Fn(T) -> bool>(
    data: &[T],
    width: usize,
    height: usize,
    min_size: usize,
    is_fg: F,
) -> (Vec<u32>, usize) {
    let n = width
        .checked_mul(height)
        .expect("region_group dimensions overflow usize");
    assert_eq!(
        data.len(),
        n,
        "region_group data length ({}) must equal width * height ({n})",
        data.len()
    );
    let mut labels = vec![0u32; n];
    if n == 0 {
        return (labels, 0);
    }

    // 1-based union-find; index 0 is unused background sentinel.
    let mut parent: Vec<u32> = vec![0];

    for row in 0..height {
        let row_off = row * width;
        for col in 0..width {
            let i = row_off + col;
            if !is_fg(data[i]) {
                continue;
            }
            let left = if col > 0 { labels[i - 1] } else { 0 };
            let up = if row > 0 { labels[i - width] } else { 0 };
            match (left, up) {
                (0, 0) => {
                    assert!(
                        parent.len() <= u32::MAX as usize,
                        "region_group exceeded maximum label count ({})",
                        u32::MAX
                    );
                    let next = parent.len() as u32;
                    parent.push(next);
                    labels[i] = next;
                }
                (l, 0) => {
                    labels[i] = uf_find(&mut parent, l);
                }
                (0, u) => {
                    labels[i] = uf_find(&mut parent, u);
                }
                (l, u) => {
                    let rl = uf_find(&mut parent, l);
                    let ru = uf_find(&mut parent, u);
                    if rl == ru {
                        labels[i] = rl;
                    } else {
                        let (lo, hi) = if rl < ru { (rl, ru) } else { (ru, rl) };
                        parent[hi as usize] = lo;
                        labels[i] = lo;
                    }
                }
            }
        }
    }

    // Resolve roots and count sizes.
    let mut sizes = vec![0usize; parent.len()];
    for l in &mut labels {
        if *l != 0 {
            let r = uf_find(&mut parent, *l);
            *l = r;
            sizes[r as usize] += 1;
        }
    }

    // Filter out regions with size <= min_size and compact labels in scan order.
    let mut remap = vec![0u32; parent.len()];
    let mut next_label = 0u32;
    for l in &mut labels {
        if *l != 0 {
            let root = *l as usize;
            if sizes[root] > min_size {
                if remap[root] == 0 {
                    next_label += 1;
                    remap[root] = next_label;
                }
                *l = remap[root];
            } else {
                *l = 0;
            }
        }
    }

    (labels, next_label as usize)
}

/// Labels 4-connected foreground regions and removes any component with
/// pixel count `<= min_size`, returning `(labels, nb_labels)` where `0` is
/// background and regions are numbered `1..=nb_labels` in row-major scan order.
///
/// Matches `lidar`'s `regionGroup` (`scipy.ndimage.label` 4-connectivity with
/// `sizes > min_size`).
///
/// # Panics
///
/// Panics if `data.len() != width * height` (or if `width * height` overflows `usize`).
pub fn region_group<F>(
    data: &[f64],
    width: usize,
    height: usize,
    min_size: usize,
    is_foreground: F,
) -> (Vec<u32>, u32)
where
    F: Fn(f64) -> bool,
{
    let (labels, nb_labels) = region_group_slice(data, width, height, min_size, is_foreground);
    (labels, nb_labels as u32)
}

/// Computes `skimage.measure.perimeter` (Benkrid & Crookes 2000, 4-neighborhood)
/// on a binary mask of dimensions `bh × bw` in pixel units.
#[cfg(test)]
pub(crate) fn benkrid_crookes_perimeter(mask: &[bool], bw: usize, bh: usize) -> f64 {
    benkrid_crookes_perimeter_affine(mask, bw, bh, (1.0, 0.0), (0.0, -1.0))
}

/// Anisotropic / affine-aware Benkrid & Crookes (2000) 4-neighborhood perimeter estimator
/// using column basis vector `vx = (gt[1], gt[4])` and row basis vector `vy = (gt[2], gt[5])`.
///
/// Reduces identically to `benkrid_crookes_perimeter * res` for square north-up pixels.
pub(crate) fn benkrid_crookes_perimeter_affine(
    mask: &[bool],
    bw: usize,
    bh: usize,
    vx: (f64, f64),
    vy: (f64, f64),
) -> f64 {
    if bw == 0 || bh == 0 {
        return 0.0;
    }

    // 1. Border pixels: foreground pixels with at least one 4-neighbor == false (outside counts as false).
    let mut border = vec![0u8; bw * bh];
    for r in 0..bh {
        let off = r * bw;
        for c in 0..bw {
            if !mask[off + c] {
                continue;
            }
            let up = r > 0 && mask[(r - 1) * bw + c];
            let down = r + 1 < bh && mask[(r + 1) * bw + c];
            let left = c > 0 && mask[off + c - 1];
            let right = c + 1 < bw && mask[off + c + 1];
            if !(up && down && left && right) {
                border[off + c] = 1;
            }
        }
    }

    benkrid_crookes_perimeter_affine_inner(&border, bw, bh, vx, vy)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn benkrid_crookes_cell_weight(
    horiz: u8,
    vert: u8,
    d_pos: u8,
    d_neg: u8,
    res_x: f64,
    res_y: f64,
    diag_pos: f64,
    diag_neg: f64,
) -> f64 {
    let ortho = horiz + vert;
    let diag = d_pos + d_neg;
    let code = 1u8 + 2 * ortho + 10 * diag;
    let ortho_len = if horiz > vert {
        res_x
    } else if vert > horiz {
        res_y
    } else {
        0.5 * (res_x + res_y)
    };
    let diag_len = if d_pos > d_neg {
        diag_pos
    } else if d_neg > d_pos {
        diag_neg
    } else {
        0.5 * (diag_pos + diag_neg)
    };
    match code {
        5 | 7 | 15 | 17 | 25 | 27 => ortho_len,
        21 | 33 => diag_len,
        13 | 23 => 0.5 * (ortho_len + diag_len),
        _ => 0.0,
    }
}

/// Computes Benkrid & Crookes affine-aware perimeter over a dense binary mask.
fn benkrid_crookes_perimeter_affine_inner(
    border: &[u8],
    bw: usize,
    bh: usize,
    vx: (f64, f64),
    vy: (f64, f64),
) -> f64 {
    let res_x = vx.0.hypot(vx.1);
    let res_y = vy.0.hypot(vy.1);
    let diag_pos = (vx.0 + vy.0).hypot(vx.1 + vy.1);
    let diag_neg = (vx.0 - vy.0).hypot(vx.1 - vy.1);
    let mut perim = 0.0f64;

    let get = |r: isize, c: isize| -> u8 {
        if r < 0 || c < 0 || r >= bh as isize || c >= bw as isize {
            0
        } else {
            border[r as usize * bw + c as usize]
        }
    };

    for r in 0..bh {
        for c in 0..bw {
            if border[r * bw + c] == 0 {
                continue;
            }
            let ri = r as isize;
            let ci = c as isize;
            let horiz = get(ri, ci - 1) + get(ri, ci + 1);
            let vert = get(ri - 1, ci) + get(ri + 1, ci);
            let d_pos = get(ri - 1, ci - 1) + get(ri + 1, ci + 1);
            let d_neg = get(ri - 1, ci + 1) + get(ri + 1, ci - 1);
            perim += benkrid_crookes_cell_weight(
                horiz, vert, d_pos, d_neg, res_x, res_y, diag_pos, diag_neg,
            );
        }
    }

    perim
}

/// Computes region properties for every label `1..=nb_labels` assuming square pixels
/// of side `resolution`, matching `skimage.measure.regionprops` and `lidar`'s `get_dep_props`.
///
/// # Panics
///
/// Panics if `labels.len() != width * height`, `elev.len() != width * height`, or any
/// non-zero label exceeds `nb_labels`.
pub fn region_props(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    resolution: f64,
) -> Vec<RegionProps> {
    region_props_xy(
        labels,
        nb_labels,
        elev,
        width,
        height,
        resolution,
        resolution,
        resolution * resolution,
    )
}

/// Computes region properties for every label `1..=nb_labels` using separate column/row
/// pixel spacings (`res_x`, `res_y`) and `cell_area` (the absolute affine determinant
/// `|gt[1]*gt[5] - gt[2]*gt[4]|`).
///
/// # Panics
///
/// Panics if `labels.len() != width * height`, `elev.len() != width * height`, or any
/// non-zero label exceeds `nb_labels`.
#[allow(clippy::too_many_arguments)]
pub fn region_props_xy(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    res_x: f64,
    res_y: f64,
    cell_area: f64,
) -> Vec<RegionProps> {
    region_props_affine_basis(
        labels,
        nb_labels,
        elev,
        width,
        height,
        (res_x, 0.0),
        (0.0, -res_y),
        cell_area,
    )
}

/// Lightweight per-component summary (count, bbox, first pixel, elevation stats, area, volume)
/// computed in one pass with zero per-component mask/border heap allocations and no perimeter
/// or equivalent-ellipse moment calculations.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegionBasicStats {
    pub label: u32,
    pub count: usize,
    pub bbox: (usize, usize, usize, usize),
    pub first_pixel: (usize, usize),
    pub area: f64,
    pub volume: f64,
    pub mean_depth: f64,
    pub max_depth: f64,
    pub min_elev: f64,
    pub max_elev: f64,
}

#[inline]
pub(crate) fn affine_basis(gt: &[f64; 6]) -> ((f64, f64), (f64, f64), f64) {
    let res_x = gt[1].hypot(gt[4]);
    let res_y = gt[2].hypot(gt[5]);
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    if res_x > 0.0
        && res_x.is_finite()
        && res_y > 0.0
        && res_y.is_finite()
        && det > 0.0
        && det.is_finite()
    {
        ((gt[1], gt[4]), (gt[2], gt[5]), det)
    } else {
        ((1.0, 0.0), (0.0, -1.0), 1.0)
    }
}

/// Computes lightweight elevation and bounding-box statistics for labels `1..=nb_labels`
/// without allocating per-component masks or computing perimeter/ellipse moments.
pub(crate) fn region_basic_stats(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    cell_area: f64,
) -> Vec<RegionBasicStats> {
    let n = width
        .checked_mul(height)
        .expect("region_basic_stats dimensions overflow usize");
    assert_eq!(
        labels.len(),
        n,
        "region_basic_stats labels length ({}) must equal width * height ({n})",
        labels.len()
    );
    assert_eq!(
        elev.len(),
        n,
        "region_basic_stats elev length ({}) must equal width * height ({n})",
        elev.len()
    );
    let k = nb_labels as usize;
    if k == 0 {
        return Vec::new();
    }

    let mut count = vec![0usize; k];
    let mut min_row = vec![usize::MAX; k];
    let mut min_col = vec![usize::MAX; k];
    let mut max_row = vec![0usize; k];
    let mut max_col = vec![0usize; k];
    let mut first_pixel = vec![(usize::MAX, usize::MAX); k];
    let mut valid_z_count = vec![0usize; k];
    let mut z_ref = vec![0.0f64; k];
    let mut min_elev = vec![f64::INFINITY; k];
    let mut max_elev = vec![f64::NEG_INFINITY; k];
    let mut sum_elev_shifted = vec![0.0f64; k];

    for r in 0..height {
        let off = r * width;
        for c in 0..width {
            let lbl = labels[off + c];
            if lbl == 0 {
                continue;
            }
            assert!(
                lbl <= nb_labels,
                "label {lbl} exceeds nb_labels ({nb_labels})"
            );
            let idx = (lbl - 1) as usize;
            let z = elev[off + c];

            if count[idx] == 0 {
                first_pixel[idx] = (r, c);
            }
            count[idx] += 1;
            if r < min_row[idx] {
                min_row[idx] = r;
            }
            if c < min_col[idx] {
                min_col[idx] = c;
            }
            if r + 1 > max_row[idx] {
                max_row[idx] = r + 1;
            }
            if c + 1 > max_col[idx] {
                max_col[idx] = c + 1;
            }

            if z.is_finite() {
                if valid_z_count[idx] == 0 {
                    z_ref[idx] = z;
                }
                valid_z_count[idx] += 1;
                if z < min_elev[idx] {
                    min_elev[idx] = z;
                }
                if z > max_elev[idx] {
                    max_elev[idx] = z;
                }
                sum_elev_shifted[idx] += z - z_ref[idx];
            }
        }
    }

    let mut out = Vec::with_capacity(k);
    for idx in 0..k {
        let n = count[idx];
        if n == 0 {
            continue;
        }
        let nf = n as f64;
        let (min_z, max_z, max_depth, mean_depth) = if valid_z_count[idx] > 0 {
            let z_nf = valid_z_count[idx] as f64;
            let min_z = min_elev[idx];
            let max_z = max_elev[idx];
            let max_depth = (max_z - min_z).max(0.0);
            let mean_depth = ((max_z - z_ref[idx]) * z_nf - sum_elev_shifted[idx]).max(0.0) / z_nf;
            (min_z, max_z, max_depth, mean_depth)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let area = nf * cell_area;
        let volume = mean_depth * area;
        out.push(RegionBasicStats {
            label: (idx + 1) as u32,
            count: n,
            bbox: (min_row[idx], min_col[idx], max_row[idx], max_col[idx]),
            first_pixel: first_pixel[idx],
            area,
            volume,
            mean_depth,
            max_depth,
            min_elev: min_z,
            max_elev: max_z,
        });
    }
    out
}

/// Computes derived morphometric indices and the linear-artifact heuristic shared by
/// [`crate::slicing`] and [`crate::merge_tree`].
///
/// Returns `(circularity, depth_diam_ratio, hypsometric_integral, rim_slope_deg, artifact_flag)`.
pub(crate) fn extended_morphometrics(
    area: f64,
    perimeter: f64,
    avg_depth: f64,
    max_depth: f64,
    elongatedness: f64,
    area_bbox_ratio: f64,
) -> (f64, f64, f64, f64, bool) {
    let circularity = if perimeter > 0.0 && area > 0.0 {
        (4.0 * PI * area / (perimeter * perimeter)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eq_radius = if area > 0.0 { (area / PI).sqrt() } else { 0.0 };
    let depth_diam_ratio = if eq_radius > 0.0 {
        max_depth / (2.0 * eq_radius)
    } else {
        0.0
    };
    let hypsometric_integral = if max_depth > 0.0 {
        (avg_depth / max_depth).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let rim_slope_deg = if eq_radius > 0.0 && max_depth > 0.0 {
        (max_depth / eq_radius).atan().to_degrees()
    } else {
        0.0
    };
    let artifact_flag = elongatedness > 6.0 && area_bbox_ratio < 0.25;
    (
        circularity,
        depth_diam_ratio,
        hypsometric_integral,
        rim_slope_deg,
        artifact_flag,
    )
}

/// Computes Benkrid & Crookes affine-aware perimeter over an arbitrary sparse set of
/// occupied `(row, col)` cells in `O(|cells|)` memory and expected `O(|cells|)` time,
/// matching [`benkrid_crookes_perimeter_affine`] without allocating a dense bbox mask.
/// Computes the exact affine-aware Benkrid & Crookes perimeter directly from a pre-filtered
/// set of 4-neighborhood border cells `(row, col)`.
pub(crate) fn benkrid_crookes_perimeter_from_border_set(
    border: &HashSet<(isize, isize)>,
    vx: (f64, f64),
    vy: (f64, f64),
) -> f64 {
    if border.is_empty() {
        return 0.0;
    }
    let res_x = vx.0.hypot(vx.1);
    let res_y = vy.0.hypot(vy.1);
    let diag_pos = (vx.0 + vy.0).hypot(vx.1 + vy.1);
    let diag_neg = (vx.0 - vy.0).hypot(vx.1 - vy.1);
    let mut perim = 0.0f64;

    let get = |r: isize, c: isize| -> u8 { u8::from(border.contains(&(r, c))) };

    for &(ri, ci) in border {
        let horiz = get(ri, ci - 1) + get(ri, ci + 1);
        let vert = get(ri - 1, ci) + get(ri + 1, ci);
        let d_pos = get(ri - 1, ci - 1) + get(ri + 1, ci + 1);
        let d_neg = get(ri - 1, ci + 1) + get(ri + 1, ci - 1);
        perim += benkrid_crookes_cell_weight(
            horiz, vert, d_pos, d_neg, res_x, res_y, diag_pos, diag_neg,
        );
    }
    perim
}

/// Promotes a single [`RegionBasicStats`] entry to full [`RegionProps`] by computing
/// its affine-aware Benkrid & Crookes perimeter and equivalent-ellipse moments over its
/// bounding box only. Produces results identical to [`region_props_affine`].
#[cfg(test)]
pub(crate) fn region_props_for_single_label(
    labels: &[u32],
    basic: &RegionBasicStats,
    width: usize,
    gt: &[f64; 6],
) -> RegionProps {
    let (vx, vy, cell_area) = affine_basis(gt);
    region_props_for_single_label_basis(labels, basic, width, vx, vy, cell_area)
}

pub(crate) fn region_props_for_single_label_basis(
    labels: &[u32],
    basic: &RegionBasicStats,
    width: usize,
    vx: (f64, f64),
    vy: (f64, f64),
    cell_area: f64,
) -> RegionProps {
    let res_x = vx.0.hypot(vx.1);
    let res_y = vy.0.hypot(vy.1);
    let max_side = res_x.max(res_y);
    let min_res = if max_side > 0.0 && cell_area > 0.0 {
        (cell_area / max_side).min(res_x.min(res_y))
    } else {
        res_x.min(res_y)
    };

    let (r0, c0, r1, c1) = basic.bbox;
    let bh = r1 - r0;
    let bw = c1 - c0;
    let mut mask = vec![false; bw * bh];
    let target = basic.label;
    let (fr, fc) = basic.first_pixel;
    let fr_f = fr as f64;
    let fc_f = fc as f64;

    let mut sum_r = 0.0f64;
    let mut sum_c = 0.0f64;
    let mut sum_rr = 0.0f64;
    let mut sum_cc = 0.0f64;
    let mut sum_rc = 0.0f64;

    for r in r0..r1 {
        let src_off = r * width;
        let dst_off = (r - r0) * bw;
        let dr_pix = r as f64 - fr_f;
        for c in c0..c1 {
            if labels[src_off + c] == target {
                mask[dst_off + (c - c0)] = true;
                let dc_pix = c as f64 - fc_f;
                let cf = dc_pix * vx.0 + dr_pix * vy.0;
                let rf = -(dc_pix * vx.1 + dr_pix * vy.1);
                sum_r += rf;
                sum_c += cf;
                sum_rr += rf * rf;
                sum_cc += cf * cf;
                sum_rc += rf * cf;
            }
        }
    }

    let perimeter = benkrid_crookes_perimeter_affine(&mask, bw, bh, vx, vy);

    let nf = basic.count as f64;
    let r_bar = sum_r / nf;
    let c_bar = sum_c / nf;
    let mu20 = (sum_rr / nf - r_bar * r_bar).max(0.0);
    let mu02 = (sum_cc / nf - c_bar * c_bar).max(0.0);
    let mu11 = sum_rc / nf - r_bar * c_bar;

    let common = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();
    let l1 = (0.5 * (mu20 + mu02 + common)).max(0.0);
    let l2 = (0.5 * (mu20 + mu02 - common)).max(0.0);

    let major_axis = 4.0 * l1.sqrt();
    let minor_axis = (4.0 * l2.sqrt()).max(min_res);
    let elongatedness = if minor_axis > 0.0 {
        major_axis / minor_axis
    } else {
        0.0
    };
    let eccentricity = if l1 > 0.0 {
        (1.0 - l2 / l1).max(0.0).sqrt()
    } else {
        0.0
    };
    let orientation_rad = if (mu20 - mu02).abs() == 0.0 {
        if mu11 < 0.0 { -FRAC_PI_4 } else { FRAC_PI_4 }
    } else {
        0.5 * (2.0 * mu11).atan2(mu20 - mu02)
    };
    let orientation = orientation_rad / PI * 180.0;

    let bbox_area = (bw * bh) as f64;
    let area_bbox_ratio = if bbox_area > 0.0 { nf / bbox_area } else { 0.0 };

    RegionProps {
        label: basic.label,
        count: basic.count,
        bbox: basic.bbox,
        first_pixel: basic.first_pixel,
        area: basic.area,
        volume: basic.volume,
        mean_depth: basic.mean_depth,
        max_depth: basic.max_depth,
        min_elev: basic.min_elev,
        max_elev: basic.max_elev,
        perimeter,
        major_axis,
        minor_axis,
        elongatedness,
        eccentricity,
        orientation,
        area_bbox_ratio,
    }
}

/// Computes region properties using a full 6-element GDAL-style affine geotransform,
/// accurately handling anisotropic pixel spacings, rotation, and shear in both
/// perimeter and second central spatial moments.
#[cfg(test)]
pub(crate) fn region_props_affine(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    gt: &[f64; 6],
) -> Vec<RegionProps> {
    let (vx, vy, cell_area) = affine_basis(gt);
    region_props_affine_basis(labels, nb_labels, elev, width, height, vx, vy, cell_area)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn region_props_affine_basis(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    vx: (f64, f64),
    vy: (f64, f64),
    cell_area: f64,
) -> Vec<RegionProps> {
    let n = width
        .checked_mul(height)
        .expect("region_props dimensions overflow usize");
    assert_eq!(
        labels.len(),
        n,
        "region_props labels length ({}) must equal width * height ({n})",
        labels.len()
    );
    assert_eq!(
        elev.len(),
        n,
        "region_props elev length ({}) must equal width * height ({n})",
        elev.len()
    );
    let k = nb_labels as usize;
    if k == 0 {
        return Vec::new();
    }

    let mut count = vec![0usize; k];
    let mut min_row = vec![usize::MAX; k];
    let mut min_col = vec![usize::MAX; k];
    let mut max_row = vec![0usize; k];
    let mut max_col = vec![0usize; k];
    let mut first_pixel = vec![(usize::MAX, usize::MAX); k];
    let mut sum_r = vec![0.0f64; k];
    let mut sum_c = vec![0.0f64; k];
    let mut sum_rr = vec![0.0f64; k];
    let mut sum_cc = vec![0.0f64; k];
    let mut sum_rc = vec![0.0f64; k];
    let mut valid_z_count = vec![0usize; k];
    let mut z_ref = vec![0.0f64; k];
    let mut min_elev = vec![f64::INFINITY; k];
    let mut max_elev = vec![f64::NEG_INFINITY; k];
    let mut sum_elev_shifted = vec![0.0f64; k];

    for r in 0..height {
        let off = r * width;
        for c in 0..width {
            let lbl = labels[off + c];
            if lbl == 0 {
                continue;
            }
            assert!(
                lbl <= nb_labels,
                "label {lbl} exceeds nb_labels ({nb_labels})"
            );
            let idx = (lbl - 1) as usize;
            let z = elev[off + c];

            if count[idx] == 0 {
                first_pixel[idx] = (r, c);
            }
            count[idx] += 1;
            if r < min_row[idx] {
                min_row[idx] = r;
            }
            if c < min_col[idx] {
                min_col[idx] = c;
            }
            if r + 1 > max_row[idx] {
                max_row[idx] = r + 1;
            }
            if c + 1 > max_col[idx] {
                max_col[idx] = c + 1;
            }
            // Shift coordinates relative to `first_pixel` before accumulating moments
            // to avoid catastrophic cancellation on large raster indices.
            let dr_pix = r as f64 - first_pixel[idx].0 as f64;
            let dc_pix = c as f64 - first_pixel[idx].1 as f64;
            let cf = dc_pix * vx.0 + dr_pix * vy.0;
            let rf = -(dc_pix * vx.1 + dr_pix * vy.1);
            sum_r[idx] += rf;
            sum_c[idx] += cf;
            sum_rr[idx] += rf * rf;
            sum_cc[idx] += cf * cf;
            sum_rc[idx] += rf * cf;

            if z.is_finite() {
                if valid_z_count[idx] == 0 {
                    z_ref[idx] = z;
                }
                valid_z_count[idx] += 1;
                if z < min_elev[idx] {
                    min_elev[idx] = z;
                }
                if z > max_elev[idx] {
                    max_elev[idx] = z;
                }
                sum_elev_shifted[idx] += z - z_ref[idx];
            }
        }
    }

    let res_x = vx.0.hypot(vx.1);
    let res_y = vy.0.hypot(vy.1);
    let max_side = res_x.max(res_y);
    let min_res = if max_side > 0.0 && cell_area > 0.0 {
        (cell_area / max_side).min(res_x.min(res_y))
    } else {
        res_x.min(res_y)
    };
    let mut props = Vec::with_capacity(k);

    for idx in 0..k {
        let n = count[idx];
        if n == 0 {
            continue;
        }
        let nf = n as f64;
        let r0 = min_row[idx];
        let c0 = min_col[idx];
        let r1 = max_row[idx];
        let c1 = max_col[idx];
        let bh = r1 - r0;
        let bw = c1 - c0;

        // Build local binary mask over bbox to compute affine-aware Benkrid & Crookes perimeter.
        let mut mask = vec![false; bw * bh];
        let target = (idx + 1) as u32;
        for r in r0..r1 {
            let src_off = r * width;
            let dst_off = (r - r0) * bw;
            for c in c0..c1 {
                if labels[src_off + c] == target {
                    mask[dst_off + (c - c0)] = true;
                }
            }
        }
        let perimeter = benkrid_crookes_perimeter_affine(&mask, bw, bh, vx, vy);

        // Second central spatial moments in map units (matches skimage when square north-up).
        let r_bar = sum_r[idx] / nf;
        let c_bar = sum_c[idx] / nf;
        let mu20 = (sum_rr[idx] / nf - r_bar * r_bar).max(0.0);
        let mu02 = (sum_cc[idx] / nf - c_bar * c_bar).max(0.0);
        let mu11 = sum_rc[idx] / nf - r_bar * c_bar;

        let common = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();
        let l1 = (0.5 * (mu20 + mu02 + common)).max(0.0);
        let l2 = (0.5 * (mu20 + mu02 - common)).max(0.0);

        let major_axis = 4.0 * l1.sqrt();
        let minor_axis = (4.0 * l2.sqrt()).max(min_res);
        let elongatedness = if minor_axis > 0.0 {
            major_axis / minor_axis
        } else {
            0.0
        };
        let eccentricity = if l1 > 0.0 {
            (1.0 - l2 / l1).max(0.0).sqrt()
        } else {
            0.0
        };
        let orientation_rad = if (mu20 - mu02).abs() == 0.0 {
            if mu11 < 0.0 { -FRAC_PI_4 } else { FRAC_PI_4 }
        } else {
            0.5 * (2.0 * mu11).atan2(mu20 - mu02)
        };
        let orientation = orientation_rad / PI * 180.0;

        let bbox_area = (bw * bh) as f64;
        let area_bbox_ratio = if bbox_area > 0.0 { nf / bbox_area } else { 0.0 };

        let (min_z, max_z, max_depth, mean_depth) = if valid_z_count[idx] > 0 {
            let z_nf = valid_z_count[idx] as f64;
            let min_z = min_elev[idx];
            let max_z = max_elev[idx];
            let max_depth = (max_z - min_z).max(0.0);
            let mean_depth = ((max_z - z_ref[idx]) * z_nf - sum_elev_shifted[idx]).max(0.0) / z_nf;
            (min_z, max_z, max_depth, mean_depth)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let area = nf * cell_area;
        let volume = mean_depth * area;

        props.push(RegionProps {
            label: target,
            count: n,
            bbox: (r0, c0, r1, c1),
            first_pixel: first_pixel[idx],
            area,
            volume,
            mean_depth,
            max_depth,
            min_elev: min_z,
            max_elev: max_z,
            perimeter,
            major_axis,
            minor_axis,
            elongatedness,
            eccentricity,
            orientation,
            area_bbox_ratio,
        });
    }

    props
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_group_filters_by_min_size_and_4_connectivity() {
        #[rustfmt::skip]
        let data = [
            1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        // Diagonal touching between (1,1) and (2,2) stays separate under 4-connectivity.
        let (labels, n) = region_group(&data, 4, 4, 2, |v| v > 0.0);
        assert_eq!(n, 2);
        assert_eq!(labels[0], 1);
        assert_eq!(labels[3], 0); // size 1 <= min_size=2 filtered out
        assert_eq!(labels[10], 2);
    }

    #[test]
    fn benkrid_crookes_square_perimeter_matches_skimage() {
        let mask3 = vec![true; 9];
        assert!((benkrid_crookes_perimeter(&mask3, 3, 3) - 8.0).abs() < 1e-12);
        let mask4 = vec![true; 16];
        assert!((benkrid_crookes_perimeter(&mask4, 4, 4) - 12.0).abs() < 1e-12);
    }

    #[test]
    fn region_props_square_geometry() {
        let data = vec![10.0; 16];
        let (labels, n) = region_group(&data, 4, 4, 1, |v| v > 0.0);
        let props = region_props(&labels, n, &data, 4, 4, 2.0);
        assert_eq!(props.len(), 1);
        let p = &props[0];
        assert_eq!(p.count, 16);
        assert!((p.area - 64.0).abs() < 1e-12);
        assert!((p.perimeter - 24.0).abs() < 1e-12);
        assert!((p.area_bbox_ratio - 1.0).abs() < 1e-12);
        assert!(p.eccentricity.abs() < 1e-12);
        assert!((p.elongatedness - 1.0).abs() < 1e-12);
    }

    #[test]
    fn region_group_slice_works_on_i32_and_matches_region_group() {
        let data_i32 = [1i32, 1, 0, 2, 1, 1, 0, 2, 0, 0, 3, 3, 0, 0, 3, 3];
        let data_f64: Vec<f64> = data_i32.iter().map(|&v| v as f64).collect();
        let (lbl_slice, n_slice) = region_group_slice(&data_i32, 4, 4, 2, |v| v > 0);
        let (lbl_f64, n_f64) = region_group(&data_f64, 4, 4, 2, |v| v > 0.0);
        assert_eq!(n_slice, n_f64 as usize);
        assert_eq!(lbl_slice, lbl_f64);
    }

    #[test]
    fn region_basic_stats_and_single_label_props_match_region_props_affine() {
        let (w, h) = (8usize, 6usize);
        let mut elev = vec![f64::NAN; w * h];
        for r in 0..h {
            for c in 0..w {
                if (r < 3 && c < 4) || (r >= 3 && c >= 4 && !(r == 5 && c == 7)) {
                    elev[r * w + c] = 50.0 + (r as f64) * 0.7 - (c as f64) * 0.3;
                }
            }
        }
        let (labels, nb) = region_group(&elev, w, h, 1, |v| v.is_finite());
        assert_eq!(nb, 2);

        // Anisotropic + slightly rotated geotransform.
        let gt = [100.0, 2.0, 0.3, 200.0, 0.1, -1.5];
        let (_, _, cell_area) = affine_basis(&gt);
        let full = region_props_affine(&labels, nb, &elev, w, h, &gt);
        let basic = region_basic_stats(&labels, nb, &elev, w, h, cell_area);
        assert_eq!(full.len(), basic.len());

        for (expected, b) in full.iter().zip(basic.iter()) {
            let promoted = region_props_for_single_label(&labels, b, w, &gt);
            assert_eq!(promoted.label, expected.label);
            assert_eq!(promoted.count, expected.count);
            assert_eq!(promoted.bbox, expected.bbox);
            assert_eq!(promoted.first_pixel, expected.first_pixel);
            assert!((promoted.area - expected.area).abs() < 1e-12);
            assert!((promoted.volume - expected.volume).abs() < 1e-12);
            assert!((promoted.mean_depth - expected.mean_depth).abs() < 1e-12);
            assert!((promoted.max_depth - expected.max_depth).abs() < 1e-12);
            assert!((promoted.min_elev - expected.min_elev).abs() < 1e-12);
            assert!((promoted.max_elev - expected.max_elev).abs() < 1e-12);
            assert!((promoted.perimeter - expected.perimeter).abs() < 1e-12);
            assert!((promoted.major_axis - expected.major_axis).abs() < 1e-12);
            assert!((promoted.minor_axis - expected.minor_axis).abs() < 1e-12);
            assert!((promoted.elongatedness - expected.elongatedness).abs() < 1e-12);
            assert!((promoted.eccentricity - expected.eccentricity).abs() < 1e-12);
            assert!((promoted.orientation - expected.orientation).abs() < 1e-12);
            assert!((promoted.area_bbox_ratio - expected.area_bbox_ratio).abs() < 1e-12);
        }
    }
}
