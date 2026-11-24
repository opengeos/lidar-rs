//! Connected-component region grouping and geometric/topographic region properties.
//!
//! Pure-Rust replacement for `scipy.ndimage.label` + `regionGroup` and
//! `skimage.measure.regionprops` used throughout `opengeos/lidar`.

use std::f64::consts::{FRAC_PI_4, PI};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
    /// Perimeter in map units (Benkrid & Crookes 4-neighborhood estimator × `resolution`).
    pub perimeter: f64,
    /// Major axis length of the equivalent ellipse in map units.
    pub major_axis: f64,
    /// Minor axis length of the equivalent ellipse in map units (clamped to >= `resolution`).
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
            if !is_foreground(data[i]) {
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

    (labels, next_label)
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

    // 2. Convolve border with [[10, 2, 10], [2, 1, 2], [10, 2, 10]] (constant 0 border)
    // and weight each segment by its true map-unit step length along vx, vy, vx+vy, vx-vy.
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
            let ortho = horiz + vert;
            let d_pos = get(ri - 1, ci - 1) + get(ri + 1, ci + 1);
            let d_neg = get(ri - 1, ci + 1) + get(ri + 1, ci - 1);
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
            perim += match code {
                5 | 7 | 15 | 17 | 25 | 27 => ortho_len,
                21 | 33 => diag_len,
                13 | 23 => 0.5 * (ortho_len + diag_len),
                _ => 0.0,
            };
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
    region_props_affine_inner(
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

/// Computes region properties using a full 6-element GDAL-style affine geotransform,
/// accurately handling anisotropic pixel spacings, rotation, and shear in both
/// perimeter and second central spatial moments.
pub(crate) fn region_props_affine(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    gt: &[f64; 6],
) -> Vec<RegionProps> {
    let res_x = gt[1].hypot(gt[4]);
    let res_y = gt[2].hypot(gt[5]);
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let (vx, vy, cell_area) = if res_x > 0.0
        && res_x.is_finite()
        && res_y > 0.0
        && res_y.is_finite()
        && det > 0.0
        && det.is_finite()
    {
        ((gt[1], gt[4]), (gt[2], gt[5]), det)
    } else {
        ((1.0, 0.0), (0.0, -1.0), 1.0)
    };
    region_props_affine_inner(labels, nb_labels, elev, width, height, vx, vy, cell_area)
}

#[allow(clippy::too_many_arguments)]
fn region_props_affine_inner(
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
        let mut minor_axis = 4.0 * l2.sqrt();
        if minor_axis == 0.0 {
            minor_axis = min_res;
        }
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
}
