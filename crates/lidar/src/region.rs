//! Connected-component region grouping and geometric/topographic region properties.
//!
//! Pure-Rust replacement for `scipy.ndimage.label` + `regionGroup` and
//! `skimage.measure.regionprops` used throughout `opengeos/lidar`.

use std::f64::consts::{FRAC_PI_4, PI, SQRT_2};

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
    let n = width * height;
    debug_assert_eq!(data.len(), n);
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
/// on a binary mask of dimensions `bh × bw`.
pub(crate) fn benkrid_crookes_perimeter(mask: &[bool], bw: usize, bh: usize) -> f64 {
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
    // and sum Benkrid & Crookes weights matching scikit-image.
    let diag_half = 0.5 * (1.0 + SQRT_2);
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
            let ortho = get(ri - 1, ci) + get(ri + 1, ci) + get(ri, ci - 1) + get(ri, ci + 1);
            let diag = get(ri - 1, ci - 1)
                + get(ri - 1, ci + 1)
                + get(ri + 1, ci - 1)
                + get(ri + 1, ci + 1);
            let code = 1u8 + 2 * ortho + 10 * diag;
            perim += match code {
                5 | 7 | 15 | 17 | 25 | 27 => 1.0,
                21 | 33 => SQRT_2,
                13 | 23 => diag_half,
                _ => 0.0,
            };
        }
    }

    perim
}

/// Computes region properties for every label `1..=nb_labels`, matching
/// `skimage.measure.regionprops` and `lidar`'s `get_dep_props`.
pub fn region_props(
    labels: &[u32],
    nb_labels: u32,
    elev: &[f64],
    width: usize,
    height: usize,
    resolution: f64,
) -> Vec<RegionProps> {
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
    let mut min_elev = vec![f64::INFINITY; k];
    let mut max_elev = vec![f64::NEG_INFINITY; k];
    let mut sum_elev = vec![0.0f64; k];

    for r in 0..height {
        let off = r * width;
        let rf = r as f64;
        for c in 0..width {
            let lbl = labels[off + c];
            if lbl == 0 {
                continue;
            }
            let idx = (lbl - 1) as usize;
            let cf = c as f64;
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
            sum_r[idx] += rf;
            sum_c[idx] += cf;
            sum_rr[idx] += rf * rf;
            sum_cc[idx] += cf * cf;
            sum_rc[idx] += rf * cf;
            if z < min_elev[idx] {
                min_elev[idx] = z;
            }
            if z > max_elev[idx] {
                max_elev[idx] = z;
            }
            sum_elev[idx] += z;
        }
    }

    let cell_area = resolution * resolution;
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

        // Build local binary mask over bbox to compute Benkrid & Crookes perimeter.
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
        let perimeter = benkrid_crookes_perimeter(&mask, bw, bh) * resolution;

        // Second central spatial moments (matching skimage.measure.regionprops).
        let r_bar = sum_r[idx] / nf;
        let c_bar = sum_c[idx] / nf;
        let mu20 = (sum_rr[idx] / nf - r_bar * r_bar).max(0.0);
        let mu02 = (sum_cc[idx] / nf - c_bar * c_bar).max(0.0);
        let mu11 = sum_rc[idx] / nf - r_bar * c_bar;

        let common = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();
        let l1 = (0.5 * (mu20 + mu02 + common)).max(0.0);
        let l2 = (0.5 * (mu20 + mu02 - common)).max(0.0);

        let major_axis = 4.0 * l1.sqrt() * resolution;
        let mut minor_axis = 4.0 * l2.sqrt() * resolution;
        if minor_axis == 0.0 {
            minor_axis = resolution;
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

        let min_z = min_elev[idx];
        let max_z = max_elev[idx];
        let max_depth = max_z - min_z;
        let mean_depth = (max_z * nf - sum_elev[idx]) / nf;
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
