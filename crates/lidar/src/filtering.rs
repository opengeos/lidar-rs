//! DEM smoothing filters: mean, median, and Gaussian.
//!
//! Pure-Rust port of `lidar/filtering.py`, supporting both in-memory
//! [`richdem::Array2D`] buffers and file-to-file GeoTIFF/COG processing via
//! `terrio` (no GDAL or SciPy required).

use std::path::Path;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, load_raster, save_raster};

use crate::error::{Error, Result};

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

/// Applies a `kernel_size × kernel_size` box mean filter to a DEM.
///
/// Uses a banded separable 2-pass formulation carrying valid sums and valid counts across both
/// passes so NoData / non-finite cells are excluded identically to a 2D window while
/// reducing per-pixel work from `O(k^2)` to `O(2k)` and bounding scratch memory to
/// `O(min(H, 256 + k - 1) * W)`.
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
        for (x, slot) in h_row
            .iter_mut()
            .enumerate()
            .take(interior_end)
            .skip(interior_start)
        {
            let mut sum = 0.0f64;
            let mut cnt = 0u32;
            let start = x - r_neg_u;
            let end = x + r_pos_u;
            for &v in &src_row[start..=end] {
                if !is_missing_val(no_data, v) {
                    sum += v;
                    cnt += 1;
                }
            }
            *slot = (sum, cnt);
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

    for y_start in (0..h).step_by(MEAN_BAND_ROWS) {
        let y_end = (y_start + MEAN_BAND_ROWS).min(h);
        let band_interior = y_start >= r_neg_u && y_end + r_pos_u <= h;

        let (src_base, band_src_count) = if band_interior {
            let base = y_start - r_neg_u;
            let count = (y_end - y_start) + kernel_size - 1;
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
}
