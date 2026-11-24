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
fn is_missing(dem: &Array2D<f64>, val: f64) -> bool {
    !val.is_finite() || val.rd_eq(dem.no_data())
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

/// Applies a `kernel_size × kernel_size` box mean filter to a DEM.
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

    let r_neg = (kernel_size / 2) as isize;
    let r_pos = (kernel_size as isize) - 1 - r_neg;
    let src = dem.data();

    let compute_row = |y: usize| -> Vec<f64> {
        let mut row = Vec::with_capacity(w);
        for x in 0..w {
            let center = src[y * w + x];
            if is_missing(dem, center) {
                row.push(center);
                continue;
            }
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for dy in -r_neg..=r_pos {
                let ny = reflect_index(y as isize + dy, h);
                let row_off = ny * w;
                for dx in -r_neg..=r_pos {
                    let nx = reflect_index(x as isize + dx, w);
                    let v = src[row_off + nx];
                    if !is_missing(dem, v) {
                        sum += v;
                        count += 1;
                    }
                }
            }
            row.push(if count > 0 {
                sum / count as f64
            } else {
                center
            });
        }
        row
    };

    #[cfg(feature = "parallel")]
    let rows: Vec<Vec<f64>> = (0..h).into_par_iter().map(compute_row).collect();
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<Vec<f64>> = (0..h).map(compute_row).collect();

    let mut out = Vec::with_capacity(w * h);
    for r in rows {
        out.extend_from_slice(&r);
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

    let r_neg = (kernel_size / 2) as isize;
    let r_pos = (kernel_size as isize) - 1 - r_neg;
    let win_cap = kernel_size.checked_mul(kernel_size).ok_or_else(|| {
        Error::InvalidArgument(format!("kernel_size² overflows usize: {kernel_size}"))
    })?;
    let src = dem.data();

    let compute_row = |y: usize| -> Vec<f64> {
        let mut row = Vec::with_capacity(w);
        let mut window = Vec::with_capacity(win_cap);
        for x in 0..w {
            let center = src[y * w + x];
            if is_missing(dem, center) {
                row.push(center);
                continue;
            }
            window.clear();
            for dy in -r_neg..=r_pos {
                let ny = reflect_index(y as isize + dy, h);
                let row_off = ny * w;
                for dx in -r_neg..=r_pos {
                    let nx = reflect_index(x as isize + dx, w);
                    let v = src[row_off + nx];
                    if !is_missing(dem, v) {
                        window.push(v);
                    }
                }
            }
            let n = window.len();
            if n == 0 {
                row.push(center);
            } else {
                let mid = n / 2;
                window.select_nth_unstable_by(mid, f64::total_cmp);
                let med = if n % 2 == 1 {
                    window[mid]
                } else {
                    let lower_max = window[..mid]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    0.5 * (lower_max + window[mid])
                };
                row.push(med);
            }
        }
        row
    };

    #[cfg(feature = "parallel")]
    let rows: Vec<Vec<f64>> = (0..h).into_par_iter().map(compute_row).collect();
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<Vec<f64>> = (0..h).map(compute_row).collect();

    let mut out = Vec::with_capacity(w * h);
    for r in rows {
        out.extend_from_slice(&r);
    }
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
    let radius = (kernel.len() / 2) as isize;
    let src = dem.data();

    // Horizontal pass: accumulate weighted value and validity weight for EVERY cell
    // (including missing cells, so diagonal valid neighbors propagate into the vertical pass).
    let h_pass_row = |y: usize| -> (Vec<f64>, Vec<f64>) {
        let mut val_row = Vec::with_capacity(w);
        let mut wt_row = Vec::with_capacity(w);
        let row_off = y * w;
        for x in 0..w {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for (ki, &kw) in kernel.iter().enumerate() {
                let dx = ki as isize - radius;
                let nx = reflect_index(x as isize + dx, w);
                let v = src[row_off + nx];
                if !is_missing(dem, v) {
                    sum += v * kw;
                    wsum += kw;
                }
            }
            val_row.push(sum);
            wt_row.push(wsum);
        }
        (val_row, wt_row)
    };

    #[cfg(feature = "parallel")]
    let h_rows: Vec<(Vec<f64>, Vec<f64>)> = (0..h).into_par_iter().map(h_pass_row).collect();
    #[cfg(not(feature = "parallel"))]
    let h_rows: Vec<(Vec<f64>, Vec<f64>)> = (0..h).map(h_pass_row).collect();

    let mut horiz_val = Vec::with_capacity(w * h);
    let mut horiz_wt = Vec::with_capacity(w * h);
    for (vr, wr) in h_rows {
        horiz_val.extend_from_slice(&vr);
        horiz_wt.extend_from_slice(&wr);
    }

    // Vertical pass: convolve both weighted values and weights, then divide once for valid cells.
    let v_pass_row = |y: usize| -> Vec<f64> {
        let mut row = Vec::with_capacity(w);
        let row_off = y * w;
        for x in 0..w {
            let orig = src[row_off + x];
            if is_missing(dem, orig) {
                row.push(orig);
                continue;
            }
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for (ki, &kw) in kernel.iter().enumerate() {
                let dy = ki as isize - radius;
                let ny = reflect_index(y as isize + dy, h);
                let idx = ny * w + x;
                sum += horiz_val[idx] * kw;
                wsum += horiz_wt[idx] * kw;
            }
            row.push(if wsum > 0.0 { sum / wsum } else { orig });
        }
        row
    };

    #[cfg(feature = "parallel")]
    let v_rows: Vec<Vec<f64>> = (0..h).into_par_iter().map(v_pass_row).collect();
    #[cfg(not(feature = "parallel"))]
    let v_rows: Vec<Vec<f64>> = (0..h).map(v_pass_row).collect();

    let mut out = Vec::with_capacity(w * h);
    for r in v_rows {
        out.extend_from_slice(&r);
    }
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
}
