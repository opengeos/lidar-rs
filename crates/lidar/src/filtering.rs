//! DEM smoothing filters: mean, median, Gaussian, bilateral, and Lindsay et al. (2019)
//! feature-preserving smoothing.
//!
//! Pure-Rust port and extension of `lidar/filtering.py`, supporting both in-memory
//! [`richdem::Array2D`] buffers and file-to-file GeoTIFF/COG processing via
//! `terrio` (no GDAL or SciPy required).

use std::path::Path;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::region::affine_basis;

/// Reflects an out-of-bounds index `idx` into `0..len` using SciPy's
/// `mode='reflect'` boundary condition (`d c b a | a b c d | d c b a`).
#[inline]
fn reflect_index(mut idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let n = len as isize;
    let period = 2 * n;
    idx = idx.rem_euclid(period);
    if idx >= n {
        (period - 1 - idx) as usize
    } else {
        idx as usize
    }
}

#[inline]
fn is_missing_val(no_data: f64, val: f64) -> bool {
    !val.is_finite() || val.rd_eq(no_data)
}

#[inline]
fn sort2(a: &mut f64, b: &mut f64) {
    if a.total_cmp(b).is_gt() {
        std::mem::swap(a, b);
    }
}

/// Optimal 19-comparator sorting network returning the exact median of 9 finite values.
#[inline]
fn median9(v: [f64; 9]) -> f64 {
    let [
        mut v0,
        mut v1,
        mut v2,
        mut v3,
        mut v4,
        mut v5,
        mut v6,
        mut v7,
        mut v8,
    ] = v;
    // Sort three triples: (0,1,2), (3,4,5), (6,7,8).
    sort2(&mut v1, &mut v2);
    sort2(&mut v4, &mut v5);
    sort2(&mut v7, &mut v8);
    sort2(&mut v0, &mut v1);
    sort2(&mut v3, &mut v4);
    sort2(&mut v6, &mut v7);
    sort2(&mut v1, &mut v2);
    sort2(&mut v4, &mut v5);
    sort2(&mut v7, &mut v8);
    // Max of triple mins -> v6; min of triple maxes -> v2; median of triple mids -> v4.
    sort2(&mut v0, &mut v3);
    sort2(&mut v5, &mut v8);
    sort2(&mut v4, &mut v7);
    sort2(&mut v3, &mut v6);
    sort2(&mut v1, &mut v4);
    sort2(&mut v2, &mut v5);
    sort2(&mut v4, &mut v7);
    // Median of the remaining 3 candidates (v6, v4, v2) -> v4.
    sort2(&mut v4, &mut v2);
    sort2(&mut v6, &mut v4);
    sort2(&mut v4, &mut v2);
    v4
}

#[inline]
fn sort5(v0: &mut f64, v1: &mut f64, v2: &mut f64, v3: &mut f64, v4: &mut f64) {
    // Optimal 9-comparator sorting network for 5 elements under f64::total_cmp.
    sort2(v0, v1);
    sort2(v3, v4);
    sort2(v2, v4);
    sort2(v2, v3);
    sort2(v1, v4);
    sort2(v0, v3);
    sort2(v0, v2);
    sort2(v1, v3);
    sort2(v1, v2);
}

/// Exact stack-allocated median of 25 finite values under `f64::total_cmp`.
///
/// Sorts the 5 rows and 5 columns of the 5x5 grid with 9-comparator networks;
/// the 6 smallest corners (rank <= 10) and 6 largest corners (rank >= 14) are
/// guaranteed <= and >= the true rank-12 median respectively, leaving 13
/// candidates whose exact rank-6 element equals the global median.
#[inline]
fn median25(mut v: [f64; 25]) -> f64 {
    // Sort 5 rows.
    for r in 0..5 {
        let o = r * 5;
        let [mut a, mut b, mut c, mut d, mut e] = [v[o], v[o + 1], v[o + 2], v[o + 3], v[o + 4]];
        sort5(&mut a, &mut b, &mut c, &mut d, &mut e);
        v[o] = a;
        v[o + 1] = b;
        v[o + 2] = c;
        v[o + 3] = d;
        v[o + 4] = e;
    }
    // Sort 5 columns (rows remain sorted afterward).
    for c in 0..5 {
        let [mut a, mut b, mut cc, mut d, mut e] =
            [v[c], v[5 + c], v[10 + c], v[15 + c], v[20 + c]];
        sort5(&mut a, &mut b, &mut cc, &mut d, &mut e);
        v[c] = a;
        v[5 + c] = b;
        v[10 + c] = cc;
        v[15 + c] = d;
        v[20 + c] = e;
    }
    // Remaining 13 candidates after pruning 6 guaranteed <= median and 6 guaranteed >= median.
    let mut rem = [
        v[3], v[4], v[7], v[8], v[9], v[11], v[12], v[13], v[15], v[16], v[17], v[20], v[21],
    ];
    let (_, med, _) = rem.select_nth_unstable_by(6, f64::total_cmp);
    *med
}

#[inline]
fn median5x5_window(vals: [f64; 25], no_data: f64, center: f64) -> f64 {
    let mut all_valid = true;
    for &v in &vals {
        if is_missing_val(no_data, v) {
            all_valid = false;
            break;
        }
    }
    if all_valid {
        return median25(vals);
    }
    let mut valid = [0.0f64; 25];
    let mut n = 0usize;
    for v in vals {
        if !is_missing_val(no_data, v) {
            valid[n] = v;
            n += 1;
        }
    }
    if n == 0 {
        center
    } else {
        median_small_slice(&mut valid[..n])
    }
}

#[inline]
fn median_small_slice(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    buf.sort_unstable_by(f64::total_cmp);
    let mid = n / 2;
    if n % 2 == 1 {
        buf[mid]
    } else {
        0.5 * (buf[mid - 1] + buf[mid])
    }
}

#[inline]
fn median3x3_window(vals: [f64; 9], no_data: f64, center: f64) -> f64 {
    let all_valid = !is_missing_val(no_data, vals[0])
        && !is_missing_val(no_data, vals[1])
        && !is_missing_val(no_data, vals[2])
        && !is_missing_val(no_data, vals[3])
        && !is_missing_val(no_data, vals[4])
        && !is_missing_val(no_data, vals[5])
        && !is_missing_val(no_data, vals[6])
        && !is_missing_val(no_data, vals[7])
        && !is_missing_val(no_data, vals[8]);
    if all_valid {
        return median9(vals);
    }
    let mut valid = [0.0f64; 9];
    let mut n = 0usize;
    for v in vals {
        if !is_missing_val(no_data, v) {
            valid[n] = v;
            n += 1;
        }
    }
    if n == 0 {
        center
    } else {
        median_small_slice(&mut valid[..n])
    }
}

fn make_output_template(dem: &Array2D<f64>, data: Vec<f64>) -> Result<Array2D<f64>> {
    let mut out = Array2D::from_vec(dem.width(), dem.height(), data)?;
    out.copy_metadata_from(dem);
    out.set_no_data(dem.no_data());
    Ok(out)
}

/// Maximum supported square filter kernel size (1025 × 1025) to prevent
/// excessive memory allocation or O(N·k²) denial-of-service on untrusted input.
const MAX_KERNEL_SIZE: usize = 1025;

fn validate_kernel_size(kernel_size: usize) -> Result<()> {
    if kernel_size == 0 || kernel_size > MAX_KERNEL_SIZE {
        return Err(Error::InvalidArgument(format!(
            "kernel_size must be in 1..={MAX_KERNEL_SIZE}, got {kernel_size}"
        )));
    }
    Ok(())
}

/// Output rows processed per horizontal scratch band in [`mean_filter`], bounding scratch
/// memory to `O(min(H, MEAN_BAND_ROWS + k - 1) * W)` rather than `O(H * W)`.
const MEAN_BAND_ROWS: usize = 256;

/// Overflow-safe cold fallback for [`mean_filter`] when horizontal partial sums overflow
/// `f64::MAX` on extreme finite inputs (e.g. `+/-1e308`).
#[cold]
#[allow(clippy::too_many_arguments)]
fn mean_2d_scaled(
    src: &[f64],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    r_neg: isize,
    r_pos: isize,
    no_data: f64,
    cnt: u32,
    center: f64,
) -> f64 {
    if cnt == 0 {
        return center;
    }
    let inv_cnt = 1.0 / (cnt as f64);
    let mut scaled_sum = 0.0f64;
    for dy in -r_neg..=r_pos {
        let ny = reflect_index(y as isize + dy, h);
        let row_off = ny * w;
        for dx in -r_neg..=r_pos {
            let nx = reflect_index(x as isize + dx, w);
            let v = src[row_off + nx];
            if !is_missing_val(no_data, v) {
                scaled_sum += v * inv_cnt;
            }
        }
    }
    scaled_sum
}

/// Applies a `kernel_size x kernel_size` box mean filter to a DEM.
///
/// Uses a banded separable 2-pass formulation carrying valid sums and valid counts across
/// both passes so NoData / non-finite cells are excluded identically to a 2D window while
/// bounding scratch memory to `O(min(H, 256 + k - 1) * W)`. Ordinary interior sliding-window
/// updates run in `O(1)` per pixel in both the horizontal and vertical passes; reflected
/// border pixels, per-band sliding-window initialization, and periodic or cancellation-triggered
/// exact rescans cost up to `O(kernel_size)` per pixel.
///
/// Boundary pixels use symmetric reflection (`scipy.ndimage` `mode='reflect'`).
/// NoData and non-finite cells are preserved untouched and excluded from
/// neighbor averages.
pub fn mean_filter(dem: &Array2D<f64>, kernel_size: usize) -> Result<Array2D<f64>> {
    validate_kernel_size(kernel_size)?;
    let w = dem.width();
    let h = dem.height();
    if w == 0 || h == 0 || kernel_size == 1 {
        return Ok(dem.clone());
    }

    let r_neg_u = kernel_size / 2;
    let r_pos_u = kernel_size - 1 - r_neg_u;
    let r_neg = r_neg_u as isize;
    let r_pos = r_pos_u as isize;
    let no_data = dem.no_data();
    let src = dem.data();

    let interior_start = r_neg_u.min(w);
    let interior_end = w.saturating_sub(r_pos_u).max(interior_start);
    let resync_interval = (kernel_size * 4).max(64);

    // Bounded horizontal scratch reused across bands of output rows.
    let max_band_src_rows = h.min(MEAN_BAND_ROWS + kernel_size - 1);
    let mut horiz = vec![(0.0f64, 0u32); max_band_src_rows * w];
    let mut out = vec![0.0f64; w * h];
    let mut src_rows: Vec<usize> = Vec::with_capacity(max_band_src_rows);

    let compute_h_row = |src_y: usize, h_row: &mut [(f64, u32)]| {
        let src_row = &src[src_y * w..(src_y + 1) * w];
        for (x, slot) in h_row.iter_mut().enumerate().take(interior_start) {
            let mut sum = 0.0f64;
            let mut cnt = 0u32;
            for dx in -r_neg..=r_pos {
                let nx = reflect_index(x as isize + dx, w);
                let v = src_row[nx];
                if !is_missing_val(no_data, v) {
                    sum += v;
                    cnt += 1;
                }
            }
            *slot = (sum, cnt);
        }

        if interior_start < interior_end {
            // Initialize sliding window at `x = interior_start`.
            let mut sum = 0.0f64;
            let mut cnt = 0u32;
            let s0 = interior_start - r_neg_u;
            let e0 = interior_start + r_pos_u;
            for &v in &src_row[s0..=e0] {
                if !is_missing_val(no_data, v) {
                    sum += v;
                    cnt += 1;
                }
            }
            h_row[interior_start] = (sum, cnt);

            let mut steps = 0usize;
            for x in (interior_start + 1)..interior_end {
                let old_v = src_row[x - 1 - r_neg_u];
                let new_v = src_row[x + r_pos_u];
                let old_valid = !is_missing_val(no_data, old_v);
                let new_valid = !is_missing_val(no_data, new_v);
                if old_valid {
                    sum -= old_v;
                    cnt -= 1;
                }
                if new_valid {
                    sum += new_v;
                    cnt += 1;
                }
                if cnt == 0 {
                    sum = 0.0;
                    steps = 0;
                    h_row[x] = (0.0, 0);
                    continue;
                }
                steps += 1;
                // Recompute directly on periodic interval, non-finite sum, or large cancellation.
                let needs_resync = steps >= resync_interval
                    || !sum.is_finite()
                    || (old_valid && old_v.abs() > 1e4 * sum.abs().max(1.0));
                if needs_resync {
                    sum = 0.0;
                    cnt = 0;
                    let start = x - r_neg_u;
                    let end = x + r_pos_u;
                    for &v in &src_row[start..=end] {
                        if !is_missing_val(no_data, v) {
                            sum += v;
                            cnt += 1;
                        }
                    }
                    steps = 0;
                }
                h_row[x] = (sum, cnt);
            }
        }

        for (x, slot) in h_row.iter_mut().enumerate().skip(interior_end) {
            let mut sum = 0.0f64;
            let mut cnt = 0u32;
            for dx in -r_neg..=r_pos {
                let nx = reflect_index(x as isize + dx, w);
                let v = src_row[nx];
                if !is_missing_val(no_data, v) {
                    sum += v;
                    cnt += 1;
                }
            }
            *slot = (sum, cnt);
        }
    };

    // Split rows into top border, interior bands (always eligible for O(1) vertical sliding
    // window), and bottom border.
    let y_int_start = r_neg_u.min(h);
    let y_int_end = h.saturating_sub(r_pos_u).max(y_int_start);
    let mut y_bands: Vec<(usize, usize)> = Vec::new();
    if y_int_start > 0 {
        let mut ys = 0usize;
        while ys < y_int_start {
            let ye = (ys + MEAN_BAND_ROWS).min(y_int_start);
            y_bands.push((ys, ye));
            ys = ye;
        }
    }
    if y_int_start < y_int_end {
        let mut ys = y_int_start;
        while ys < y_int_end {
            let ye = (ys + MEAN_BAND_ROWS).min(y_int_end);
            y_bands.push((ys, ye));
            ys = ye;
        }
    }
    if y_int_end < h {
        let mut ys = y_int_end;
        while ys < h {
            let ye = (ys + MEAN_BAND_ROWS).min(h);
            y_bands.push((ys, ye));
            ys = ye;
        }
    }

    for (y_start, y_end) in y_bands {
        let band_rows = y_end - y_start;
        let band_interior = y_start >= r_neg_u && y_end + r_pos_u <= h;

        let (src_base, band_src_count) = if band_interior {
            let base = y_start - r_neg_u;
            let count = band_rows + kernel_size - 1;
            let horiz_slice = &mut horiz[..count * w];
            #[cfg(feature = "parallel")]
            horiz_slice
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(i, h_row)| compute_h_row(base + i, h_row));
            #[cfg(not(feature = "parallel"))]
            horiz_slice
                .chunks_mut(w)
                .enumerate()
                .for_each(|(i, h_row)| compute_h_row(base + i, h_row));
            (Some(base), count)
        } else {
            src_rows.clear();
            for y in y_start..y_end {
                for dy in -r_neg..=r_pos {
                    src_rows.push(reflect_index(y as isize + dy, h));
                }
            }
            src_rows.sort_unstable();
            src_rows.dedup();
            let count = src_rows.len();
            let horiz_slice = &mut horiz[..count * w];
            let src_rows_ref = &src_rows;
            #[cfg(feature = "parallel")]
            horiz_slice
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(i, h_row)| compute_h_row(src_rows_ref[i], h_row));
            #[cfg(not(feature = "parallel"))]
            horiz_slice
                .chunks_mut(w)
                .enumerate()
                .for_each(|(i, h_row)| compute_h_row(src_rows_ref[i], h_row));
            (None, count)
        };

        let horiz_band = &horiz[..band_src_count * w];
        let src_rows_ref = &src_rows;

        if band_interior && kernel_size > 3 {
            // True O(1) vertical sliding window over column tiles of this interior band.
            // Each tile covers 64 columns so its running accumulators stay hot in L1 cache.
            const COL_TILE: usize = 64;
            let num_tiles = w.div_ceil(COL_TILE);
            let mut tile_bufs: Vec<Vec<f64>> = (0..num_tiles)
                .map(|t| {
                    let c0 = t * COL_TILE;
                    let cw = (w - c0).min(COL_TILE);
                    vec![0.0f64; band_rows * cw]
                })
                .collect();

            let process_tile = |(t, tbuf): (usize, &mut Vec<f64>)| {
                let c0 = t * COL_TILE;
                let cw = (w - c0).min(COL_TILE);
                let mut acc_sum = [0.0f64; COL_TILE];
                let mut acc_cnt = [0u32; COL_TILE];
                let mut acc_steps = [0usize; COL_TILE];

                // Initialize at local_y = 0 from horiz_band rows 0..kernel_size.
                for k in 0..kernel_size {
                    let hrow = &horiz_band[k * w + c0..k * w + c0 + cw];
                    for dx in 0..cw {
                        acc_sum[dx] += hrow[dx].0;
                        acc_cnt[dx] += hrow[dx].1;
                    }
                }
                {
                    let y = y_start;
                    let srow = &src[y * w + c0..y * w + c0 + cw];
                    for dx in 0..cw {
                        let center = srow[dx];
                        if is_missing_val(no_data, center) {
                            tbuf[dx] = center;
                        } else if acc_cnt[dx] > 0 {
                            let avg = acc_sum[dx] / acc_cnt[dx] as f64;
                            tbuf[dx] = if avg.is_finite() {
                                avg
                            } else {
                                mean_2d_scaled(
                                    src,
                                    w,
                                    h,
                                    c0 + dx,
                                    y,
                                    r_neg,
                                    r_pos,
                                    no_data,
                                    acc_cnt[dx],
                                    center,
                                )
                            };
                        } else {
                            tbuf[dx] = center;
                        }
                    }
                }

                // Slide down local_y = 1..band_rows in O(1) per cell.
                for local_y in 1..band_rows {
                    let y = y_start + local_y;
                    let old_row = &horiz_band[(local_y - 1) * w + c0..(local_y - 1) * w + c0 + cw];
                    let new_row = &horiz_band[(local_y - 1 + kernel_size) * w + c0
                        ..(local_y - 1 + kernel_size) * w + c0 + cw];
                    let srow = &src[y * w + c0..y * w + c0 + cw];
                    let dst = &mut tbuf[local_y * cw..(local_y + 1) * cw];

                    for dx in 0..cw {
                        let (old_s, old_c) = old_row[dx];
                        let (new_s, new_c) = new_row[dx];
                        acc_sum[dx] = acc_sum[dx] - old_s + new_s;
                        acc_cnt[dx] = acc_cnt[dx] - old_c + new_c;
                        acc_steps[dx] += 1;

                        let needs_resync = acc_cnt[dx] == 0
                            || acc_steps[dx] >= resync_interval
                            || !acc_sum[dx].is_finite()
                            || (old_c > 0 && old_s.abs() > 1e4 * acc_sum[dx].abs().max(1.0));
                        if needs_resync {
                            let mut s = 0.0f64;
                            let mut c = 0u32;
                            for k in 0..kernel_size {
                                let (hs, hc) = horiz_band[(local_y + k) * w + c0 + dx];
                                s += hs;
                                c += hc;
                            }
                            acc_sum[dx] = s;
                            acc_cnt[dx] = c;
                            acc_steps[dx] = 0;
                        }

                        let center = srow[dx];
                        if is_missing_val(no_data, center) {
                            dst[dx] = center;
                        } else if acc_cnt[dx] > 0 {
                            let avg = acc_sum[dx] / acc_cnt[dx] as f64;
                            dst[dx] = if avg.is_finite() {
                                avg
                            } else {
                                mean_2d_scaled(
                                    src,
                                    w,
                                    h,
                                    c0 + dx,
                                    y,
                                    r_neg,
                                    r_pos,
                                    no_data,
                                    acc_cnt[dx],
                                    center,
                                )
                            };
                        } else {
                            dst[dx] = center;
                        }
                    }
                }
            };

            #[cfg(feature = "parallel")]
            tile_bufs.par_iter_mut().enumerate().for_each(process_tile);
            #[cfg(not(feature = "parallel"))]
            tile_bufs.iter_mut().enumerate().for_each(process_tile);

            // Scatter tile results back into `out[y_start..y_end]`.
            for (t, tbuf) in tile_bufs.iter().enumerate() {
                let c0 = t * COL_TILE;
                let cw = (w - c0).min(COL_TILE);
                for local_y in 0..band_rows {
                    let y = y_start + local_y;
                    out[y * w + c0..y * w + c0 + cw]
                        .copy_from_slice(&tbuf[local_y * cw..(local_y + 1) * cw]);
                }
            }
            continue;
        }

        let v_pass_row = |(local_y, out_row): (usize, &mut [f64])| {
            let y = y_start + local_y;
            let src_row = &src[y * w..(y + 1) * w];
            let mut row_offs = [0usize; MAX_KERNEL_SIZE];
            let offs = &mut row_offs[..kernel_size];
            if let Some(base) = src_base {
                let y_base = y - r_neg_u - base;
                for (k, slot) in offs.iter_mut().enumerate() {
                    *slot = (y_base + k) * w;
                }
            } else {
                for (k, slot) in offs.iter_mut().enumerate() {
                    let dy = k as isize - r_neg;
                    let ny = reflect_index(y as isize + dy, h);
                    let pos = src_rows_ref.binary_search(&ny).unwrap();
                    *slot = pos * w;
                }
            }

            if kernel_size == 3 {
                let r0 = &horiz_band[offs[0]..offs[0] + w];
                let r1 = &horiz_band[offs[1]..offs[1] + w];
                let r2 = &horiz_band[offs[2]..offs[2] + w];
                for x in 0..w {
                    let center = src_row[x];
                    if is_missing_val(no_data, center) {
                        out_row[x] = center;
                        continue;
                    }
                    let sum = r0[x].0 + r1[x].0 + r2[x].0;
                    let cnt = r0[x].1 + r1[x].1 + r2[x].1;
                    out_row[x] = if cnt > 0 {
                        let avg = sum / cnt as f64;
                        if avg.is_finite() {
                            avg
                        } else {
                            mean_2d_scaled(src, w, h, x, y, r_neg, r_pos, no_data, cnt, center)
                        }
                    } else {
                        center
                    };
                }
            } else {
                for x in 0..w {
                    let center = src_row[x];
                    if is_missing_val(no_data, center) {
                        out_row[x] = center;
                        continue;
                    }
                    let mut sum = 0.0f64;
                    let mut cnt = 0u32;
                    for &ro in offs.iter() {
                        let (hs, hc) = horiz_band[ro + x];
                        sum += hs;
                        cnt += hc;
                    }
                    out_row[x] = if cnt > 0 {
                        let avg = sum / cnt as f64;
                        if avg.is_finite() {
                            avg
                        } else {
                            mean_2d_scaled(src, w, h, x, y, r_neg, r_pos, no_data, cnt, center)
                        }
                    } else {
                        center
                    };
                }
            }
        };

        let out_band = &mut out[y_start * w..y_end * w];
        #[cfg(feature = "parallel")]
        out_band.par_chunks_mut(w).enumerate().for_each(v_pass_row);
        #[cfg(not(feature = "parallel"))]
        out_band.chunks_mut(w).enumerate().for_each(v_pass_row);
    }

    make_output_template(dem, out)
}

/// Applies a `kernel_size × kernel_size` moving-window median filter to a DEM.
///
/// Boundary pixels use symmetric reflection (`scipy.ndimage` `mode='reflect'`).
/// NoData and non-finite cells are preserved untouched and excluded from the
/// median window.
pub fn median_filter(dem: &Array2D<f64>, kernel_size: usize) -> Result<Array2D<f64>> {
    validate_kernel_size(kernel_size)?;
    let w = dem.width();
    let h = dem.height();
    if w == 0 || h == 0 || kernel_size == 1 {
        return Ok(dem.clone());
    }

    let no_data = dem.no_data();
    let src = dem.data();
    let mut out = vec![0.0f64; w * h];

    if kernel_size == 3 {
        let compute_row_3x3 = |(y, out_row): (usize, &mut [f64])| {
            let y0 = if y > 0 { y - 1 } else { reflect_index(-1, h) };
            let y1 = y;
            let y2 = if y + 1 < h {
                y + 1
            } else {
                reflect_index(y as isize + 1, h)
            };
            let r0 = &src[y0 * w..(y0 + 1) * w];
            let r1 = &src[y1 * w..(y1 + 1) * w];
            let r2 = &src[y2 * w..(y2 + 1) * w];

            // Left border (x = 0)
            {
                let center = r1[0];
                if is_missing_val(no_data, center) {
                    out_row[0] = center;
                } else {
                    let x0 = reflect_index(-1, w);
                    let x1 = 0;
                    let x2 = if w > 1 { 1 } else { reflect_index(1, w) };
                    out_row[0] = median3x3_window(
                        [
                            r0[x0], r0[x1], r0[x2], r1[x0], r1[x1], r1[x2], r2[x0], r2[x1], r2[x2],
                        ],
                        no_data,
                        center,
                    );
                }
            }

            // Interior pixels (1..w-1): direct indexing, zero reflect_index calls.
            if w > 2 {
                for x in 1..(w - 1) {
                    let center = r1[x];
                    if is_missing_val(no_data, center) {
                        out_row[x] = center;
                        continue;
                    }
                    out_row[x] = median3x3_window(
                        [
                            r0[x - 1],
                            r0[x],
                            r0[x + 1],
                            r1[x - 1],
                            center,
                            r1[x + 1],
                            r2[x - 1],
                            r2[x],
                            r2[x + 1],
                        ],
                        no_data,
                        center,
                    );
                }
            }

            // Right border (x = w - 1) when w > 1
            if w > 1 {
                let x = w - 1;
                let center = r1[x];
                if is_missing_val(no_data, center) {
                    out_row[x] = center;
                } else {
                    let x0 = x - 1;
                    let x1 = x;
                    let x2 = reflect_index(x as isize + 1, w);
                    out_row[x] = median3x3_window(
                        [
                            r0[x0], r0[x1], r0[x2], r1[x0], r1[x1], r1[x2], r2[x0], r2[x1], r2[x2],
                        ],
                        no_data,
                        center,
                    );
                }
            }
        };

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(w).enumerate().for_each(compute_row_3x3);
        #[cfg(not(feature = "parallel"))]
        out.chunks_mut(w).enumerate().for_each(compute_row_3x3);

        return make_output_template(dem, out);
    }

    if kernel_size == 5 {
        let compute_row_5x5 = |(y, out_row): (usize, &mut [f64])| {
            let y0 = if y >= 2 {
                y - 2
            } else {
                reflect_index(y as isize - 2, h)
            };
            let y1 = if y >= 1 {
                y - 1
            } else {
                reflect_index(y as isize - 1, h)
            };
            let y2 = y;
            let y3 = if y + 1 < h {
                y + 1
            } else {
                reflect_index(y as isize + 1, h)
            };
            let y4 = if y + 2 < h {
                y + 2
            } else {
                reflect_index(y as isize + 2, h)
            };
            let r0 = &src[y0 * w..(y0 + 1) * w];
            let r1 = &src[y1 * w..(y1 + 1) * w];
            let r2 = &src[y2 * w..(y2 + 1) * w];
            let r3 = &src[y3 * w..(y3 + 1) * w];
            let r4 = &src[y4 * w..(y4 + 1) * w];

            let int_start = 2usize.min(w);
            let int_end = w.saturating_sub(2).max(int_start);

            let eval_border = |x: usize| -> f64 {
                let center = r2[x];
                if is_missing_val(no_data, center) {
                    return center;
                }
                let xi = x as isize;
                let x0 = reflect_index(xi - 2, w);
                let x1 = reflect_index(xi - 1, w);
                let x2 = x;
                let x3 = reflect_index(xi + 1, w);
                let x4 = reflect_index(xi + 2, w);
                median5x5_window(
                    [
                        r0[x0], r0[x1], r0[x2], r0[x3], r0[x4], r1[x0], r1[x1], r1[x2], r1[x3],
                        r1[x4], r2[x0], r2[x1], center, r2[x3], r2[x4], r3[x0], r3[x1], r3[x2],
                        r3[x3], r3[x4], r4[x0], r4[x1], r4[x2], r4[x3], r4[x4],
                    ],
                    no_data,
                    center,
                )
            };

            for (x, slot) in out_row.iter_mut().enumerate().take(int_start) {
                *slot = eval_border(x);
            }
            for x in int_start..int_end {
                let center = r2[x];
                if is_missing_val(no_data, center) {
                    out_row[x] = center;
                    continue;
                }
                out_row[x] = median5x5_window(
                    [
                        r0[x - 2],
                        r0[x - 1],
                        r0[x],
                        r0[x + 1],
                        r0[x + 2],
                        r1[x - 2],
                        r1[x - 1],
                        r1[x],
                        r1[x + 1],
                        r1[x + 2],
                        r2[x - 2],
                        r2[x - 1],
                        center,
                        r2[x + 1],
                        r2[x + 2],
                        r3[x - 2],
                        r3[x - 1],
                        r3[x],
                        r3[x + 1],
                        r3[x + 2],
                        r4[x - 2],
                        r4[x - 1],
                        r4[x],
                        r4[x + 1],
                        r4[x + 2],
                    ],
                    no_data,
                    center,
                );
            }
            for (x, slot) in out_row.iter_mut().enumerate().skip(int_end) {
                *slot = eval_border(x);
            }
        };

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(w).enumerate().for_each(compute_row_5x5);
        #[cfg(not(feature = "parallel"))]
        out.chunks_mut(w).enumerate().for_each(compute_row_5x5);

        return make_output_template(dem, out);
    }

    let r_neg_u = kernel_size / 2;
    let r_pos_u = kernel_size - 1 - r_neg_u;
    let r_neg = r_neg_u as isize;
    let r_pos = r_pos_u as isize;
    let win_cap = kernel_size.checked_mul(kernel_size).ok_or_else(|| {
        Error::InvalidArgument(format!("kernel_size² overflows usize: {kernel_size}"))
    })?;
    let interior_start = r_neg_u.min(w);
    let interior_end = w.saturating_sub(r_pos_u).max(interior_start);

    let compute_row = |(y, out_row): (usize, &mut [f64])| {
        let mut window = Vec::with_capacity(win_cap);
        let y_interior = y >= r_neg_u && y + r_pos_u < h;
        let mut row_offs = [0usize; MAX_KERNEL_SIZE];
        let offs = &mut row_offs[..kernel_size];
        if y_interior {
            let y_base = y - r_neg_u;
            for (k, slot) in offs.iter_mut().enumerate() {
                *slot = (y_base + k) * w;
            }
        } else {
            for (k, slot) in offs.iter_mut().enumerate() {
                let dy = k as isize - r_neg;
                *slot = reflect_index(y as isize + dy, h) * w;
            }
        }
        let src_row = &src[y * w..(y + 1) * w];

        let finish_median = |window: &mut [f64], center: f64| -> f64 {
            let n = window.len();
            if n == 0 {
                center
            } else {
                let mid = n / 2;
                window.select_nth_unstable_by(mid, f64::total_cmp);
                if n % 2 == 1 {
                    window[mid]
                } else {
                    let lower_max = window[..mid]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    0.5 * (lower_max + window[mid])
                }
            }
        };

        for (x, slot) in out_row.iter_mut().enumerate().take(interior_start) {
            let center = src_row[x];
            if is_missing_val(no_data, center) {
                *slot = center;
                continue;
            }
            window.clear();
            for &row_off in offs.iter() {
                for dx in -r_neg..=r_pos {
                    let nx = reflect_index(x as isize + dx, w);
                    let v = src[row_off + nx];
                    if !is_missing_val(no_data, v) {
                        window.push(v);
                    }
                }
            }
            *slot = finish_median(&mut window, center);
        }

        for (x, slot) in out_row
            .iter_mut()
            .enumerate()
            .take(interior_end)
            .skip(interior_start)
        {
            let center = src_row[x];
            if is_missing_val(no_data, center) {
                *slot = center;
                continue;
            }
            window.clear();
            let x_lo = x - r_neg_u;
            let x_hi = x + r_pos_u;
            for &row_off in offs.iter() {
                for &v in &src[row_off + x_lo..=row_off + x_hi] {
                    if !is_missing_val(no_data, v) {
                        window.push(v);
                    }
                }
            }
            *slot = finish_median(&mut window, center);
        }

        for (x, slot) in out_row.iter_mut().enumerate().skip(interior_end) {
            let center = src_row[x];
            if is_missing_val(no_data, center) {
                *slot = center;
                continue;
            }
            window.clear();
            for &row_off in offs.iter() {
                for dx in -r_neg..=r_pos {
                    let nx = reflect_index(x as isize + dx, w);
                    let v = src[row_off + nx];
                    if !is_missing_val(no_data, v) {
                        window.push(v);
                    }
                }
            }
            *slot = finish_median(&mut window, center);
        }
    };

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(w).enumerate().for_each(compute_row);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w).enumerate().for_each(compute_row);

    make_output_template(dem, out)
}

/// Builds a normalized 1D Gaussian kernel matching `scipy.ndimage.gaussian_filter`
/// (`truncate = 4.0`). Returns an error if the resulting kernel width exceeds
/// `MAX_KERNEL_SIZE`.
fn gaussian_kernel_1d(sigma: f64, truncate: f64) -> Result<Vec<f64>> {
    let max_radius = (MAX_KERNEL_SIZE - 1) / 2;
    let raw_radius = (truncate * sigma + 0.5).floor();
    if !raw_radius.is_finite() || raw_radius > max_radius as f64 {
        return Err(Error::InvalidArgument(format!(
            "sigma={sigma} produces a Gaussian kernel wider than MAX_KERNEL_SIZE ({MAX_KERNEL_SIZE})"
        )));
    }
    let radius = raw_radius as isize;
    if radius == 0 {
        return Ok(vec![1.0]);
    }
    let denom = 2.0 * sigma * sigma;
    if denom == 0.0 || !denom.is_finite() {
        return Ok(vec![1.0]);
    }
    let size = (2 * radius + 1) as usize;
    let mut kernel = Vec::new();
    kernel.try_reserve_exact(size).map_err(|e| {
        Error::Algorithm(format!(
            "failed to allocate Gaussian kernel of size {size}: {e}"
        ))
    })?;
    let mut sum = 0.0f64;
    for i in -radius..=radius {
        let w = (-((i * i) as f64) / denom).exp();
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum;
    }
    Ok(kernel)
}

/// Applies a separable 2D Gaussian filter with standard deviation `sigma` to a DEM.
///
/// Matches `scipy.ndimage.gaussian_filter` (`truncate = 4.0`, `mode = 'reflect'`)
/// while preserving NoData / non-finite cells via exact 2D normalized convolution
/// (carrying weighted values and validity weights across both passes before dividing).
pub fn gaussian_filter(dem: &Array2D<f64>, sigma: f64) -> Result<Array2D<f64>> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "sigma must be finite and >= 0, got {sigma}"
        )));
    }
    let w = dem.width();
    let h = dem.height();
    if w == 0 || h == 0 || sigma == 0.0 {
        return Ok(dem.clone());
    }

    let kernel = gaussian_kernel_1d(sigma, 4.0)?;
    let radius_u = kernel.len() / 2;
    let radius = radius_u as isize;
    let no_data = dem.no_data();
    let src = dem.data();

    let interior_start = radius_u.min(w);
    let interior_end = w.saturating_sub(radius_u).max(interior_start);

    // Horizontal pass: accumulate (weighted_val, validity_weight) for EVERY cell
    // (including missing cells, so diagonal valid neighbors propagate into the vertical pass).
    let mut horiz = vec![(0.0f64, 0.0f64); w * h];
    let h_pass_row = |(y, h_row): (usize, &mut [(f64, f64)])| {
        let src_row = &src[y * w..(y + 1) * w];
        for (x, slot) in h_row.iter_mut().enumerate().take(interior_start) {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for (ki, &kw) in kernel.iter().enumerate() {
                let dx = ki as isize - radius;
                let nx = reflect_index(x as isize + dx, w);
                let v = src_row[nx];
                if !is_missing_val(no_data, v) {
                    sum += v * kw;
                    wsum += kw;
                }
            }
            *slot = (sum, wsum);
        }
        for (x, slot) in h_row
            .iter_mut()
            .enumerate()
            .take(interior_end)
            .skip(interior_start)
        {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            let base = x - radius_u;
            for (ki, &kw) in kernel.iter().enumerate() {
                let v = src_row[base + ki];
                if !is_missing_val(no_data, v) {
                    sum += v * kw;
                    wsum += kw;
                }
            }
            *slot = (sum, wsum);
        }
        for (x, slot) in h_row.iter_mut().enumerate().skip(interior_end) {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for (ki, &kw) in kernel.iter().enumerate() {
                let dx = ki as isize - radius;
                let nx = reflect_index(x as isize + dx, w);
                let v = src_row[nx];
                if !is_missing_val(no_data, v) {
                    sum += v * kw;
                    wsum += kw;
                }
            }
            *slot = (sum, wsum);
        }
    };

    #[cfg(feature = "parallel")]
    horiz.par_chunks_mut(w).enumerate().for_each(h_pass_row);
    #[cfg(not(feature = "parallel"))]
    horiz.chunks_mut(w).enumerate().for_each(h_pass_row);

    // Vertical pass: convolve both weighted values and weights directly into flat output.
    let mut out = vec![0.0f64; w * h];
    let v_pass_row = |(y, out_row): (usize, &mut [f64])| {
        let src_row = &src[y * w..(y + 1) * w];
        let y_interior = y >= radius_u && y + radius_u < h;
        let mut row_offs = [0usize; MAX_KERNEL_SIZE];
        let offs = &mut row_offs[..kernel.len()];
        if y_interior {
            let y_base = y - radius_u;
            for (ki, slot) in offs.iter_mut().enumerate() {
                *slot = (y_base + ki) * w;
            }
        } else {
            for (ki, slot) in offs.iter_mut().enumerate() {
                let dy = ki as isize - radius;
                *slot = reflect_index(y as isize + dy, h) * w;
            }
        }
        for x in 0..w {
            let orig = src_row[x];
            if is_missing_val(no_data, orig) {
                out_row[x] = orig;
                continue;
            }
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for (&ro, &kw) in offs.iter().zip(kernel.iter()) {
                let (hv, hw) = horiz[ro + x];
                sum += hv * kw;
                wsum += hw * kw;
            }
            out_row[x] = if wsum > 0.0 { sum / wsum } else { orig };
        }
    };

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(w).enumerate().for_each(v_pass_row);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w).enumerate().for_each(v_pass_row);

    make_output_template(dem, out)
}

/// Applies [`mean_filter`] to a raster file on disk and optionally saves the result.
pub fn mean_filter_file(
    in_dem: impl AsRef<Path>,
    kernel_size: usize,
    out_file: Option<&Path>,
) -> Result<Array2D<f64>> {
    let dem = load_raster::<f64>(in_dem, None)?;
    let filtered = mean_filter(&dem, kernel_size)?;
    if let Some(out) = out_file {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        save_raster(
            &filtered,
            out,
            Some(&format!("mean_filter(kernel_size={kernel_size})")),
            false,
        )?;
    }
    Ok(filtered)
}

/// Applies [`median_filter`] to a raster file on disk and optionally saves the result.
pub fn median_filter_file(
    in_dem: impl AsRef<Path>,
    kernel_size: usize,
    out_file: Option<&Path>,
) -> Result<Array2D<f64>> {
    let dem = load_raster::<f64>(in_dem, None)?;
    let filtered = median_filter(&dem, kernel_size)?;
    if let Some(out) = out_file {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        save_raster(
            &filtered,
            out,
            Some(&format!("median_filter(kernel_size={kernel_size})")),
            false,
        )?;
    }
    Ok(filtered)
}

/// Applies [`gaussian_filter`] to a raster file on disk and optionally saves the result.
pub fn gaussian_filter_file(
    in_dem: impl AsRef<Path>,
    sigma: f64,
    out_file: Option<&Path>,
) -> Result<Array2D<f64>> {
    let dem = load_raster::<f64>(in_dem, None)?;
    let filtered = gaussian_filter(&dem, sigma)?;
    if let Some(out) = out_file {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        save_raster(
            &filtered,
            out,
            Some(&format!("gaussian_filter(sigma={sigma})")),
            false,
        )?;
    }
    Ok(filtered)
}

/// Applies an edge-preserving 2D bilateral filter to a DEM.
///
/// Combines a spatial Gaussian kernel (`sigma_spatial`, truncated at 3 standard deviations)
/// with an elevation-range Gaussian kernel `exp(-dz^2 / (2 * sigma_range^2))` so planar
/// micro-noise is smoothed while sharp topographic breaklines (road embankments, depression
/// rims, terraces) are preserved. NoData and non-finite cells are preserved untouched and
/// excluded from neighbor averages.
pub fn bilateral_filter(
    dem: &Array2D<f64>,
    sigma_spatial: f64,
    sigma_range: f64,
) -> Result<Array2D<f64>> {
    if !sigma_spatial.is_finite() || sigma_spatial < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "sigma_spatial must be finite and >= 0, got {sigma_spatial}"
        )));
    }
    if !sigma_range.is_finite() || sigma_range <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "sigma_range must be finite and > 0, got {sigma_range}"
        )));
    }
    let w = dem.width();
    let h = dem.height();
    if w == 0 || h == 0 || sigma_spatial == 0.0 {
        return Ok(dem.clone());
    }

    let max_radius = (MAX_KERNEL_SIZE - 1) / 2;
    let raw_radius = (3.0 * sigma_spatial + 0.5).floor();
    if !raw_radius.is_finite() || raw_radius > max_radius as f64 {
        return Err(Error::InvalidArgument(format!(
            "sigma_spatial={sigma_spatial} produces a bilateral kernel wider than MAX_KERNEL_SIZE ({MAX_KERNEL_SIZE})"
        )));
    }
    let radius_u = raw_radius as usize;
    if radius_u == 0 {
        return Ok(dem.clone());
    }
    let radius = radius_u as isize;
    let ksize = 2 * radius_u + 1;

    let inv_two_sigma_s2 = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    let mut spatial_w = Vec::with_capacity(ksize * ksize);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let d2 = (dx * dx + dy * dy) as f64;
            spatial_w.push((-d2 * inv_two_sigma_s2).exp());
        }
    }

    let no_data = dem.no_data();
    let src = dem.data();
    let mut out = vec![0.0f64; w * h];
    let int_start = radius_u.min(w);
    let int_end = w.saturating_sub(radius_u).max(int_start);

    // Compute range weight scaling `dz / sigma_range` before squaring so large finite
    // elevations (e.g. [0, 1e180] with sigma_range=1e200) and tiny positive sigmas never
    // produce NaN weights.
    #[inline]
    fn bilateral_range_weight(dz: f64, sigma_range: f64) -> f64 {
        if dz == 0.0 {
            1.0
        } else {
            let scaled = dz / sigma_range;
            (-0.5 * scaled * scaled).exp()
        }
    }

    let compute_row = |(y, out_row): (usize, &mut [f64])| {
        let src_row = &src[y * w..(y + 1) * w];
        let y_interior = y >= radius_u && y + radius_u < h;
        let mut row_offs = [0usize; MAX_KERNEL_SIZE];
        let offs = &mut row_offs[..ksize];
        if y_interior {
            let y_base = y - radius_u;
            for (ky, slot) in offs.iter_mut().enumerate() {
                *slot = (y_base + ky) * w;
            }
        } else {
            for (ky, slot) in offs.iter_mut().enumerate() {
                let dy = ky as isize - radius;
                *slot = reflect_index(y as isize + dy, h) * w;
            }
        }

        let eval_border = |x: usize| -> f64 {
            let center = src_row[x];
            if is_missing_val(no_data, center) {
                return center;
            }
            // Accumulate weighted deltas relative to `center` to keep constant extreme
            // rasters (e.g. 1e308) exact and finite without unnormalized numerator overflow.
            let mut num_dz = 0.0f64;
            let mut den = 0.0f64;
            let xi = x as isize;
            for (ky, &ro) in offs.iter().enumerate() {
                let w_row = &spatial_w[ky * ksize..(ky + 1) * ksize];
                for (kx, &ws) in w_row.iter().enumerate() {
                    let nx = reflect_index(xi + (kx as isize - radius), w);
                    let v = src[ro + nx];
                    if !is_missing_val(no_data, v) {
                        let dz = v - center;
                        let wt = ws * bilateral_range_weight(dz, sigma_range);
                        if wt > 0.0 {
                            num_dz += wt * dz;
                            den += wt;
                        }
                    }
                }
            }
            if den <= 0.0 {
                return center;
            }
            let res = center + num_dz / den;
            if res.is_finite() {
                return res;
            }
            // Overflow-safe normalized-weight fallback when `wt * dz` overflows f64.
            let inv_den = 1.0 / den;
            let mut scaled = 0.0f64;
            for (ky, &ro) in offs.iter().enumerate() {
                let w_row = &spatial_w[ky * ksize..(ky + 1) * ksize];
                for (kx, &ws) in w_row.iter().enumerate() {
                    let nx = reflect_index(xi + (kx as isize - radius), w);
                    let v = src[ro + nx];
                    if !is_missing_val(no_data, v) {
                        let dz = v - center;
                        let wt = ws * bilateral_range_weight(dz, sigma_range);
                        if wt > 0.0 {
                            scaled += (wt * inv_den) * v;
                        }
                    }
                }
            }
            if scaled.is_finite() { scaled } else { center }
        };

        for (x, slot) in out_row.iter_mut().enumerate().take(int_start) {
            *slot = eval_border(x);
        }
        for x in int_start..int_end {
            let center = src_row[x];
            if is_missing_val(no_data, center) {
                out_row[x] = center;
                continue;
            }
            let mut num_dz = 0.0f64;
            let mut den = 0.0f64;
            let x0 = x - radius_u;
            for (ky, &ro) in offs.iter().enumerate() {
                let w_row = &spatial_w[ky * ksize..(ky + 1) * ksize];
                let s_slice = &src[ro + x0..ro + x0 + ksize];
                for (&v, &ws) in s_slice.iter().zip(w_row.iter()) {
                    if !is_missing_val(no_data, v) {
                        let dz = v - center;
                        let wt = ws * bilateral_range_weight(dz, sigma_range);
                        if wt > 0.0 {
                            num_dz += wt * dz;
                            den += wt;
                        }
                    }
                }
            }
            if den <= 0.0 {
                out_row[x] = center;
                continue;
            }
            let res = center + num_dz / den;
            if res.is_finite() {
                out_row[x] = res;
            } else {
                let inv_den = 1.0 / den;
                let mut scaled = 0.0f64;
                for (ky, &ro) in offs.iter().enumerate() {
                    let w_row = &spatial_w[ky * ksize..(ky + 1) * ksize];
                    let s_slice = &src[ro + x0..ro + x0 + ksize];
                    for (&v, &ws) in s_slice.iter().zip(w_row.iter()) {
                        if !is_missing_val(no_data, v) {
                            let dz = v - center;
                            let wt = ws * bilateral_range_weight(dz, sigma_range);
                            if wt > 0.0 {
                                scaled += (wt * inv_den) * v;
                            }
                        }
                    }
                }
                out_row[x] = if scaled.is_finite() { scaled } else { center };
            }
        }
        for (x, slot) in out_row.iter_mut().enumerate().skip(int_end) {
            *slot = eval_border(x);
        }
    };

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(w).enumerate().for_each(compute_row);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w).enumerate().for_each(compute_row);

    make_output_template(dem, out)
}

/// Applies [`bilateral_filter`] to a raster file on disk and optionally saves the result.
pub fn bilateral_filter_file(
    in_dem: impl AsRef<Path>,
    sigma_spatial: f64,
    sigma_range: f64,
    out_file: Option<&Path>,
) -> Result<Array2D<f64>> {
    let dem = load_raster::<f64>(in_dem, None)?;
    let filtered = bilateral_filter(&dem, sigma_spatial, sigma_range)?;
    if let Some(out) = out_file {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        save_raster(
            &filtered,
            out,
            Some(&format!(
                "bilateral_filter(sigma_spatial={sigma_spatial}, sigma_range={sigma_range})"
            )),
            false,
        )?;
    }
    Ok(filtered)
}

/// Options controlling [`feature_preserving_smoothing`] (Lindsay et al., 2019).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FeaturePreservingOptions {
    /// Odd centered square window size in pixels for normal-vector smoothing (default `11`).
    /// Must be an odd integer in `1..=1025`.
    pub filter_size: usize,
    /// Maximum angle threshold in degrees between neighboring surface normals for
    /// smoothing across a facet (`acos(n_i . n_j) < norm_diff_deg`, default `15.0`).
    pub norm_diff_deg: f64,
    /// Number of elevation-updating iterations to fit the smoothed normal field (default `3`).
    pub num_iter: usize,
    /// Optional maximum absolute elevation change per cell relative to the input DEM.
    pub max_z_change: Option<f64>,
    /// Compress output GeoTIFF with DEFLATE (file wrapper only, default `false`).
    pub compress: bool,
}

impl Default for FeaturePreservingOptions {
    fn default() -> Self {
        Self {
            filter_size: 11,
            norm_diff_deg: 15.0,
            num_iter: 3,
            max_z_change: None,
            compress: false,
        }
    }
}

/// Applies Lindsay et al. (2019) feature-preserving DEM smoothing to reduce micro-topographic
/// noise while preserving sharp breaklines such as road embankments, channel banks, and
/// depression rims.
///
/// Requires an odd `options.filter_size` (centered symmetric window).
///
/// Algorithm stages:
/// 1. Compute per-cell unit surface normals in true map coordinates using the full 2x2
///    affine geotransform Jacobian (accurately handling orthogonal, rotated, and sheared grids).
/// 2. Smooth the normal vector field over the centered `filter_size x filter_size` neighborhood
///    using only neighboring normals within `norm_diff_deg` degrees (`n_i . n_j >= cos(norm_diff_deg)`).
/// 3. Iteratively update cell elevations (`num_iter` passes) to fit the smoothed normal field
///    across compatible neighbors using map-space affine displacements and overflow-safe
///    delta accumulation, optionally clamping per-cell change to `max_z_change`.
pub fn feature_preserving_smoothing(
    dem: &Array2D<f64>,
    options: &FeaturePreservingOptions,
) -> Result<Array2D<f64>> {
    validate_kernel_size(options.filter_size)?;
    if options.filter_size.is_multiple_of(2) {
        return Err(Error::InvalidArgument(format!(
            "filter_size must be odd, got {}",
            options.filter_size
        )));
    }
    if !options.norm_diff_deg.is_finite()
        || options.norm_diff_deg <= 0.0
        || options.norm_diff_deg > 180.0
    {
        return Err(Error::InvalidArgument(format!(
            "norm_diff_deg must be finite in (0, 180], got {}",
            options.norm_diff_deg
        )));
    }
    if let Some(max_dz) = options.max_z_change
        && (!max_dz.is_finite() || max_dz < 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "max_z_change must be finite and >= 0, got {max_dz}"
        )));
    }

    let w = dem.width();
    let h = dem.height();
    if w == 0 || h == 0 || options.filter_size == 1 || options.num_iter == 0 {
        return Ok(dem.clone());
    }

    // Full 2x2 affine basis: col step vx = (a, d), row step vy = (b, e).
    let (vx, vy, _) = affine_basis(&dem.geotransform);
    let (a, d) = vx;
    let (b, e) = vy;
    let det = a * e - b * d;
    let no_data = dem.no_data();
    let src = dem.data();

    // Stage 1: Compute per-cell unit surface normals (nx, ny, nz) in map space.
    // Given grid-space directional derivatives (dz/dc, dz/dr), map-space gradient satisfies
    //   [a d; b e] [gx; gy] = [dz/dc; dz/dr]
    // so gx = (e * dz/dc - d * dz/dr) / det, gy = (-b * dz/dc + a * dz/dr) / det.
    let mut normals = vec![[0.0f64; 3]; w * h];
    let compute_normal_row = |(y, nrow): (usize, &mut [[f64; 3]])| {
        let row_off = y * w;
        for (x, slot) in nrow.iter_mut().enumerate() {
            let zc = src[row_off + x];
            if is_missing_val(no_data, zc) {
                *slot = [0.0, 0.0, 0.0];
                continue;
            }
            let z_w = if x > 0 {
                let v = src[row_off + x - 1];
                (!is_missing_val(no_data, v)).then_some(v)
            } else {
                None
            };
            let z_e = if x + 1 < w {
                let v = src[row_off + x + 1];
                (!is_missing_val(no_data, v)).then_some(v)
            } else {
                None
            };
            let dz_dc = match (z_w, z_e) {
                (Some(zw), Some(ze)) => 0.5 * (ze - zw),
                (Some(zw), None) => zc - zw,
                (None, Some(ze)) => ze - zc,
                (None, None) => 0.0,
            };

            let z_n = if y > 0 {
                let v = src[(y - 1) * w + x];
                (!is_missing_val(no_data, v)).then_some(v)
            } else {
                None
            };
            let z_s = if y + 1 < h {
                let v = src[(y + 1) * w + x];
                (!is_missing_val(no_data, v)).then_some(v)
            } else {
                None
            };
            let dz_dr = match (z_n, z_s) {
                (Some(zn), Some(zs)) => 0.5 * (zs - zn),
                (Some(zn), None) => zc - zn,
                (None, Some(zs)) => zs - zc,
                (None, None) => 0.0,
            };

            let gx = (e * dz_dc - d * dz_dr) / det;
            let gy = (-b * dz_dc + a * dz_dr) / det;
            let inv_len = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
            *slot = if inv_len.is_finite() {
                [-gx * inv_len, -gy * inv_len, inv_len]
            } else {
                [0.0, 0.0, 1.0]
            };
        }
    };

    #[cfg(feature = "parallel")]
    normals
        .par_chunks_mut(w)
        .enumerate()
        .for_each(compute_normal_row);
    #[cfg(not(feature = "parallel"))]
    normals
        .chunks_mut(w)
        .enumerate()
        .for_each(compute_normal_row);

    // Stage 2: Edge-preserving normal vector smoothing over the `filter_size x filter_size` window.
    let radius_u = options.filter_size / 2;
    let cos_thresh = options.norm_diff_deg.to_radians().cos();
    let inv_one_minus_cos = if (1.0 - cos_thresh).abs() > 1e-12 {
        1.0 / (1.0 - cos_thresh)
    } else {
        0.0
    };
    let mut smooth_normals = vec![[0.0f64; 3]; w * h];

    let smooth_normal_row = |(y, srow): (usize, &mut [[f64; 3]])| {
        let y_lo = y.saturating_sub(radius_u);
        let y_hi = (y + radius_u).min(h - 1);
        for (x, slot) in srow.iter_mut().enumerate() {
            let nc = normals[y * w + x];
            if nc[2] == 0.0 {
                *slot = [0.0, 0.0, 0.0];
                continue;
            }
            let x_lo = x.saturating_sub(radius_u);
            let x_hi = (x + radius_u).min(w - 1);
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            for ny in y_lo..=y_hi {
                let n_slice = &normals[ny * w + x_lo..=ny * w + x_hi];
                for &nj in n_slice {
                    if nj[2] == 0.0 {
                        continue;
                    }
                    let dot = (nc[0] * nj[0] + nc[1] * nj[1] + nc[2] * nj[2]).clamp(-1.0, 1.0);
                    if dot >= cos_thresh {
                        let wt = if inv_one_minus_cos > 0.0 {
                            let t = (dot - cos_thresh) * inv_one_minus_cos;
                            t * t
                        } else {
                            1.0
                        };
                        sx += wt * nj[0];
                        sy += wt * nj[1];
                        sz += wt * nj[2];
                    }
                }
            }
            let len = (sx * sx + sy * sy + sz * sz).sqrt();
            *slot = if len > 0.0 {
                [sx / len, sy / len, sz / len]
            } else {
                nc
            };
        }
    };

    #[cfg(feature = "parallel")]
    smooth_normals
        .par_chunks_mut(w)
        .enumerate()
        .for_each(smooth_normal_row);
    #[cfg(not(feature = "parallel"))]
    smooth_normals
        .chunks_mut(w)
        .enumerate()
        .for_each(smooth_normal_row);

    // Stage 3: Iteratively update cell elevations to match the smoothed normal field.
    const NBR8: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let mut cur_z = src.to_vec();
    let mut next_z = vec![0.0f64; w * h];
    let max_dz_opt = options.max_z_change;

    for _ in 0..options.num_iter {
        let cur_ref = &cur_z;
        let sn_ref = &smooth_normals;
        let update_row = |(y, out_row): (usize, &mut [f64])| {
            let row_off = y * w;
            for (x, slot) in out_row.iter_mut().enumerate() {
                let idx = row_off + x;
                let z0 = src[idx];
                if is_missing_val(no_data, z0) {
                    *slot = z0;
                    continue;
                }
                let z_cur = cur_ref[idx];
                let ni = sn_ref[idx];
                if ni[2] <= 1e-12 {
                    *slot = z_cur;
                    continue;
                }
                // Accumulate predicted deltas relative to `z_cur` so flat extreme terrain
                // (e.g. constant 1e308) stays finite and exact without sum overflow.
                let mut sum_dz = 0.0f64;
                let mut sum_w = 0.0f64;
                let mut preds = [(0.0f64, 0.0f64); 8];
                let mut n_preds = 0usize;
                for &(ox, oy) in &NBR8 {
                    let nx_i = x as isize + ox;
                    let ny_i = y as isize + oy;
                    if nx_i < 0 || ny_i < 0 || nx_i >= w as isize || ny_i >= h as isize {
                        continue;
                    }
                    let j = ny_i as usize * w + nx_i as usize;
                    let zj = cur_ref[j];
                    if is_missing_val(no_data, zj) {
                        continue;
                    }
                    let nj = sn_ref[j];
                    if nj[2] <= 1e-12 {
                        continue;
                    }
                    let dot = (ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2]).clamp(-1.0, 1.0);
                    if dot >= cos_thresh {
                        let wt = if inv_one_minus_cos > 0.0 {
                            let t = (dot - cos_thresh) * inv_one_minus_cos;
                            t * t
                        } else {
                            1.0
                        };
                        // Full affine map-space displacement from neighbor j to center i.
                        let dc_ij = -(ox as f64);
                        let dr_ij = -(oy as f64);
                        let dx_ij = dc_ij * a + dr_ij * b;
                        let dy_ij = dc_ij * d + dr_ij * e;
                        let slope_term = (nj[0] * dx_ij + nj[1] * dy_ij) / nj[2];
                        let delta_pred = (zj - z_cur) - slope_term;
                        if delta_pred.is_finite() {
                            sum_dz += wt * delta_pred;
                            sum_w += wt;
                            preds[n_preds] = (wt, delta_pred);
                            n_preds += 1;
                        }
                    }
                }
                let mut z_new = if sum_w > 0.0 {
                    let cand = z_cur + sum_dz / sum_w;
                    if cand.is_finite() {
                        cand
                    } else {
                        let inv_w = 1.0 / sum_w;
                        let mut scaled_dz = 0.0f64;
                        for &(wt, dp) in &preds[..n_preds] {
                            scaled_dz += (wt * inv_w) * dp;
                        }
                        let fb = z_cur + scaled_dz;
                        if fb.is_finite() { fb } else { z_cur }
                    }
                } else {
                    z_cur
                };
                if let Some(max_dz) = max_dz_opt {
                    z_new = z_new.clamp(z0 - max_dz, z0 + max_dz);
                }
                *slot = z_new;
            }
        };

        #[cfg(feature = "parallel")]
        next_z.par_chunks_mut(w).enumerate().for_each(update_row);
        #[cfg(not(feature = "parallel"))]
        next_z.chunks_mut(w).enumerate().for_each(update_row);

        std::mem::swap(&mut cur_z, &mut next_z);
    }

    make_output_template(dem, cur_z)
}

/// Applies [`feature_preserving_smoothing`] to a raster file on disk and optionally saves the result.
pub fn feature_preserving_smoothing_file(
    in_dem: impl AsRef<Path>,
    options: &FeaturePreservingOptions,
    out_file: Option<&Path>,
) -> Result<Array2D<f64>> {
    let dem = load_raster::<f64>(in_dem, None)?;
    let filtered = feature_preserving_smoothing(&dem, options)?;
    if let Some(out) = out_file {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        save_raster(
            &filtered,
            out,
            Some(&format!(
                "feature_preserving_smoothing(filter_size={}, norm_diff_deg={}, num_iter={})",
                options.filter_size, options.norm_diff_deg, options.num_iter
            )),
            options.compress,
        )?;
    }
    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_index_matches_scipy() {
        // For len = 4: [0, 1, 2, 3]
        // reflect: 3 2 1 0 | 0 1 2 3 | 3 2 1 0
        assert_eq!(reflect_index(-1, 4), 0);
        assert_eq!(reflect_index(-2, 4), 1);
        assert_eq!(reflect_index(-4, 4), 3);
        assert_eq!(reflect_index(0, 4), 0);
        assert_eq!(reflect_index(3, 4), 3);
        assert_eq!(reflect_index(4, 4), 3);
        assert_eq!(reflect_index(5, 4), 2);
    }

    #[test]
    fn filters_preserve_constant_and_nodata() {
        let mut dem = Array2D::from_vec(5, 5, vec![10.0; 25]).unwrap();
        dem.set_no_data(-9999.0);
        dem.set(2, 2, -9999.0);

        let m = mean_filter(&dem, 3).unwrap();
        let med = median_filter(&dem, 3).unwrap();
        let g = gaussian_filter(&dem, 1.0).unwrap();

        for arr in [&m, &med, &g] {
            assert_eq!(arr.get(2, 2), -9999.0);
            for y in 0..5 {
                for x in 0..5 {
                    if (x, y) != (2, 2) {
                        assert!((arr.get(x, y) - 10.0).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn median_filter_removes_impulse_spike() {
        let mut dem = Array2D::from_vec(5, 5, vec![5.0; 25]).unwrap();
        dem.set(2, 2, 1000.0);
        let med = median_filter(&dem, 3).unwrap();
        assert_eq!(med.get(2, 2), 5.0);
    }

    #[test]
    fn rejects_invalid_and_oversized_kernel() {
        let dem = Array2D::from_vec(3, 3, vec![1.0; 9]).unwrap();
        assert!(mean_filter(&dem, 0).is_err());
        assert!(median_filter(&dem, 0).is_err());
        assert!(mean_filter(&dem, 10_000).is_err());
        assert!(median_filter(&dem, 10_000).is_err());
        assert!(gaussian_filter(&dem, 1e12).is_err());
        assert!(gaussian_filter(&dem, 1e20).is_err());
    }

    #[test]
    fn gaussian_carries_validity_weights_across_both_passes() {
        let mut dem = Array2D::from_vec(2, 2, vec![0.0, -9999.0, 10.0, 10.0]).unwrap();
        dem.set_no_data(-9999.0);
        let g = gaussian_filter(&dem, 1.0).unwrap();
        assert_eq!(g.get(1, 0), -9999.0);
        // Bottom-right (1,1) receives positive diagonal weight from valid top-left 0.0.
        assert!(g.get(1, 1) < 10.0, "expected < 10.0, got {}", g.get(1, 1));
    }

    #[test]
    fn median9_matches_exhaustive_zero_one_principle() {
        // By the Zero-One Principle, a sorting/selection network is exact for all real
        // inputs if and only if it is exact for all 2^9 = 512 binary vectors.
        for mask in 0u16..512 {
            let mut arr = [0.0f64; 9];
            let mut ones = 0usize;
            for (i, slot) in arr.iter_mut().enumerate() {
                if (mask >> i) & 1 == 1 {
                    *slot = 1.0;
                    ones += 1;
                }
            }
            let expected = if ones >= 5 { 1.0 } else { 0.0 };
            assert_eq!(
                median9(arr),
                expected,
                "median9 failed for binary mask {mask:#011b}"
            );
        }
    }

    #[test]
    fn optimized_filters_match_naive_2d_reference() {
        fn naive_mean(dem: &Array2D<f64>, k: usize) -> Vec<f64> {
            let (w, h) = (dem.width(), dem.height());
            let r_neg = (k / 2) as isize;
            let r_pos = (k as isize) - 1 - r_neg;
            let nd = dem.no_data();
            let src = dem.data();
            let mut out = vec![0.0; w * h];
            for y in 0..h {
                for x in 0..w {
                    let center = src[y * w + x];
                    if is_missing_val(nd, center) {
                        out[y * w + x] = center;
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut cnt = 0usize;
                    for dy in -r_neg..=r_pos {
                        let ny = reflect_index(y as isize + dy, h);
                        for dx in -r_neg..=r_pos {
                            let nx = reflect_index(x as isize + dx, w);
                            let v = src[ny * w + nx];
                            if !is_missing_val(nd, v) {
                                sum += v;
                                cnt += 1;
                            }
                        }
                    }
                    out[y * w + x] = if cnt > 0 { sum / cnt as f64 } else { center };
                }
            }
            out
        }

        fn naive_median(dem: &Array2D<f64>, k: usize) -> Vec<f64> {
            let (w, h) = (dem.width(), dem.height());
            let r_neg = (k / 2) as isize;
            let r_pos = (k as isize) - 1 - r_neg;
            let nd = dem.no_data();
            let src = dem.data();
            let mut out = vec![0.0; w * h];
            let mut win = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let center = src[y * w + x];
                    if is_missing_val(nd, center) {
                        out[y * w + x] = center;
                        continue;
                    }
                    win.clear();
                    for dy in -r_neg..=r_pos {
                        let ny = reflect_index(y as isize + dy, h);
                        for dx in -r_neg..=r_pos {
                            let nx = reflect_index(x as isize + dx, w);
                            let v = src[ny * w + nx];
                            if !is_missing_val(nd, v) {
                                win.push(v);
                            }
                        }
                    }
                    win.sort_by(f64::total_cmp);
                    let n = win.len();
                    out[y * w + x] = if n == 0 {
                        center
                    } else if n % 2 == 1 {
                        win[n / 2]
                    } else {
                        0.5 * (win[n / 2 - 1] + win[n / 2])
                    };
                }
            }
            out
        }

        for &(w, h) in &[(1usize, 5usize), (5, 1), (2, 2), (13, 17)] {
            let mut data = Vec::with_capacity(w * h);
            for i in 0..(w * h) {
                let v = if i % 11 == 3 {
                    -9999.0
                } else if i % 19 == 7 {
                    f64::NAN
                } else {
                    100.0 + ((i * 37) % 97) as f64 * 0.25
                };
                data.push(v);
            }
            let mut dem = Array2D::from_vec(w, h, data).unwrap();
            dem.set_no_data(-9999.0);

            for k in [2usize, 3, 4, 5, 11] {
                let got_mean = mean_filter(&dem, k).unwrap();
                let exp_mean = naive_mean(&dem, k);
                for (a, b) in got_mean.data().iter().zip(exp_mean.iter()) {
                    if b.is_nan() {
                        assert!(a.is_nan());
                    } else {
                        assert!(
                            (a - b).abs() < 1e-12,
                            "mean mismatch ({w}x{h}, k={k}): got {a}, expected {b}"
                        );
                    }
                }

                if k <= 5 {
                    let got_med = median_filter(&dem, k).unwrap();
                    let exp_med = naive_median(&dem, k);
                    for (a, b) in got_med.data().iter().zip(exp_med.iter()) {
                        if b.is_nan() {
                            assert!(a.is_nan());
                        } else {
                            assert!(
                                (a - b).abs() < 1e-12,
                                "median mismatch ({w}x{h}, k={k}): got {a}, expected {b}"
                            );
                        }
                    }
                }
            }

            // Compare Gaussian filter against direct 2D normalized convolution reference.
            for sigma in [0.5, 1.0, 2.5] {
                let g = gaussian_filter(&dem, sigma).unwrap();
                let k1d = gaussian_kernel_1d(sigma, 4.0).unwrap();
                let rad = (k1d.len() / 2) as isize;
                let nd = dem.no_data();
                let src_data = dem.data();
                for y in 0..h {
                    for x in 0..w {
                        let orig = src_data[y * w + x];
                        let out_v = g.data()[y * w + x];
                        if orig.is_nan() {
                            assert!(out_v.is_nan());
                        } else if orig == -9999.0 {
                            assert_eq!(out_v, -9999.0);
                        } else {
                            let mut num = 0.0f64;
                            let mut den = 0.0f64;
                            for (ky, &wy) in k1d.iter().enumerate() {
                                let ny = reflect_index(y as isize + (ky as isize - rad), h);
                                for (kx, &wx) in k1d.iter().enumerate() {
                                    let nx = reflect_index(x as isize + (kx as isize - rad), w);
                                    let v = src_data[ny * w + nx];
                                    if !is_missing_val(nd, v) {
                                        let w2 = wy * wx;
                                        num += v * w2;
                                        den += w2;
                                    }
                                }
                            }
                            let expected = if den > 0.0 { num / den } else { orig };
                            assert!(
                                (out_v - expected).abs() < 1e-12,
                                "gaussian mismatch ({w}x{h}, sigma={sigma}) at ({x},{y}): got {out_v}, expected {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn median9_matches_signed_zero_ordering_of_fallback_path() {
        // 5 entries of -0.0 and 4 entries of +0.0 -> median is -0.0 under total_cmp.
        let vals_neg = [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0];
        let mut fallback_neg = vals_neg;
        assert_eq!(
            median9(vals_neg).to_bits(),
            median_small_slice(&mut fallback_neg).to_bits()
        );
        assert_eq!(median9(vals_neg).to_bits(), (-0.0f64).to_bits());

        // 3x3 grid with -0.0 at center and +0.0 elsewhere -> median is +0.0 under total_cmp.
        let mut dem = Array2D::from_vec(3, 3, vec![0.0f64; 9]).unwrap();
        dem.set_no_data(-9999.0);
        dem.set(1, 1, -0.0);
        let med = median_filter(&dem, 3).unwrap();
        assert_eq!(med.get(1, 1).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn mean_filter_keeps_finite_result_when_horizontal_partial_overflows() {
        #[rustfmt::skip]
        let data = vec![
            -1e308, 0.0,    0.0,
             1e308, 1e308,  0.0,
            -1e308, 0.0,    0.0,
        ];
        let mut dem = Array2D::from_vec(3, 3, data).unwrap();
        dem.set_no_data(-9999.0);
        let mean = mean_filter(&dem, 3).unwrap();
        let center = mean.get(1, 1);
        assert!(
            center.is_finite() && center.abs() < 1e-12,
            "expected finite 0.0 mean for balanced extreme values, got {center}"
        );
    }

    #[test]
    fn median25_matches_sorted_reference_and_signed_zeros() {
        let mut state: u64 = 0xDEAD_BEEF_1337_4242;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        // Test 2,000 diverse binary masks drawn from high LCG bits (not the alternating low bit)
        // plus 2,000 continuous vectors including signed zeros.
        let mut distinct_masks = std::collections::HashSet::new();
        for _ in 0..2000 {
            let mask = (next() >> 39) as u32;
            distinct_masks.insert(mask);
            let mut arr = [0.0f64; 25];
            let mut ones = 0usize;
            for (i, slot) in arr.iter_mut().enumerate() {
                if ((mask >> i) & 1) == 1 {
                    *slot = 1.0;
                    ones += 1;
                }
            }
            let expected = if ones >= 13 { 1.0 } else { 0.0 };
            assert_eq!(median25(arr), expected);
        }
        assert!(distinct_masks.len() > 1900);
        for _ in 0..2000 {
            let mut arr = [0.0f64; 25];
            for slot in &mut arr {
                let r = (next() >> 32) % 11;
                *slot = if r == 0 {
                    -0.0
                } else if r == 1 {
                    0.0
                } else {
                    (((next() >> 32) % 1000) as f64) - 500.0
                };
            }
            let mut sorted = arr;
            sorted.sort_by(f64::total_cmp);
            assert_eq!(median25(arr).to_bits(), sorted[12].to_bits());
        }
    }

    #[test]
    fn feature_preserving_smoothing_rejects_even_filter_size() {
        let dem = Array2D::from_vec(5, 5, vec![10.0; 25]).unwrap();
        for even in [2usize, 4, 10] {
            let err = feature_preserving_smoothing(
                &dem,
                &FeaturePreservingOptions {
                    filter_size: even,
                    ..Default::default()
                },
            );
            assert!(err.is_err(), "expected even filter_size={even} to fail");
        }
    }

    #[test]
    fn bilateral_and_feature_preserving_handle_extreme_finite_values() {
        // Bilateral: large elevations [0, 1e180] with sigma_range=1e200 must blend smoothly without NaN.
        let mut big = Array2D::from_vec(2, 1, vec![0.0, 1e180]).unwrap();
        big.set_no_data(-9999.0);
        let b_big = bilateral_filter(&big, 1.0, 1e200).unwrap();
        assert!(
            b_big.get(0, 0).is_finite() && b_big.get(0, 0) > 0.0,
            "expected finite blended > 0.0, got {}",
            b_big.get(0, 0)
        );
        assert!(
            b_big.get(1, 0).is_finite() && b_big.get(1, 0) < 1e180,
            "expected finite blended < 1e180, got {}",
            b_big.get(1, 0)
        );

        // Bilateral and FPS: constant 3x3 raster of 1e308 must stay finite 1e308.
        let mut flat_max = Array2D::from_vec(3, 3, vec![1e308; 9]).unwrap();
        flat_max.set_no_data(-9999.0);
        let b_max = bilateral_filter(&flat_max, 1.0, 1.0).unwrap();
        let fps_max = feature_preserving_smoothing(
            &flat_max,
            &FeaturePreservingOptions {
                filter_size: 3,
                norm_diff_deg: 15.0,
                num_iter: 3,
                max_z_change: None,
                compress: false,
            },
        )
        .unwrap();
        for (&bv, &fv) in b_max.data().iter().zip(fps_max.data().iter()) {
            assert_eq!(bv, 1e308);
            assert_eq!(fv, 1e308);
        }
    }

    #[test]
    fn feature_preserving_smoothing_uses_full_affine_basis_on_sheared_grid() {
        // Sheared grid with vx = (1.0, 0.0), vy = (0.6, 0.8).
        // 1. Any tilted plane z(c, r) = 10 + 1.5*c - 0.7*r must be an exact fixed point.
        let (w, h) = (13usize, 13usize);
        let mut plane = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                plane.push(10.0 + 1.5 * c as f64 - 0.7 * r as f64);
            }
        }
        let mut plane_dem = Array2D::from_vec(w, h, plane).unwrap();
        plane_dem.set_no_data(-9999.0);
        plane_dem.geotransform = [0.0, 1.0, 0.6, 0.0, 0.0, 0.8];
        let plane_out = feature_preserving_smoothing(
            &plane_dem,
            &FeaturePreservingOptions {
                filter_size: 5,
                norm_diff_deg: 15.0,
                num_iter: 4,
                max_z_change: None,
                compress: false,
            },
        )
        .unwrap();
        for (&orig, &out) in plane_dem.data().iter().zip(plane_out.data().iter()) {
            assert!(
                (orig - out).abs() < 1e-10,
                "sheared affine plane must be a fixed point: orig={orig}, out={out}"
            );
        }

        // 2. Two facets with grid slopes (1, 0) and (0, 1) on sheared basis (1, 0) & (0.6, 0.8)
        // have true map-space normal angle ~88.6 deg (vs 60 deg if shear were ignored).
        // Separated by a 1-column NoData strip at c = 7 so both sides keep exact one-sided
        // facet normals, a 5x5 normal-smoothing window reaches across c = 7: with norm_diff_deg = 75.0,
        // the full affine basis keeps the two facets completely unblended (whereas ignoring shear
        // would blend them because 60 deg < 75 deg).
        let (sw, sh) = (15usize, 15usize);
        let mut ridge = vec![0.0f64; sw * sh];
        for r in 0..sh {
            for c in 0..sw {
                ridge[r * sw + c] = if c < 7 {
                    c as f64
                } else if c == 7 {
                    -9999.0
                } else {
                    r as f64
                };
            }
        }
        let mut ridge_dem = Array2D::from_vec(sw, sh, ridge).unwrap();
        ridge_dem.set_no_data(-9999.0);
        ridge_dem.geotransform = [0.0, 1.0, 0.6, 0.0, 0.0, 0.8];
        let ridge_out = feature_preserving_smoothing(
            &ridge_dem,
            &FeaturePreservingOptions {
                filter_size: 5,
                norm_diff_deg: 75.0,
                num_iter: 2,
                max_z_change: None,
                compress: false,
            },
        )
        .unwrap();
        // Cell (6, 7) sees right-facet normals at c=8 inside its 5x5 window; true angle is 88.6 deg > 75 deg.
        assert!(
            (ridge_out.get(6, 7) - 6.0).abs() < 1e-10,
            "left facet should not blend with right facet at 75 deg cutoff, got {}",
            ridge_out.get(6, 7)
        );
        // Cell (8, 7) sees left-facet normals at c=6 inside its 5x5 window; true angle is 88.6 deg > 75 deg.
        assert!(
            (ridge_out.get(8, 7) - 7.0).abs() < 1e-10,
            "right facet should not blend with left facet at 75 deg cutoff, got {}",
            ridge_out.get(8, 7)
        );
    }

    #[test]
    fn feature_preserving_smoothing_preserves_exact_tilted_plane() {
        let (w, h) = (15usize, 15usize);
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                data.push(100.0 + 0.3 * (x as f64) * 2.0 - 0.15 * (y as f64) * 2.0);
            }
        }
        let mut dem = Array2D::from_vec(w, h, data).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 2.0, 0.0, 30.0, 0.0, -2.0];

        let out = feature_preserving_smoothing(
            &dem,
            &FeaturePreservingOptions {
                filter_size: 5,
                norm_diff_deg: 15.0,
                num_iter: 4,
                max_z_change: None,
                compress: false,
            },
        )
        .unwrap();
        for (&a, &b) in dem.data().iter().zip(out.data().iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "tilted plane should be an exact fixed point: orig={a}, out={b}"
            );
        }
    }
}
