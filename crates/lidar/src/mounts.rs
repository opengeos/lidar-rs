//! Delineating the nested hierarchy of elevated features (mounts / peaks / tree canopies).
//!
//! Pure-Rust port of `lidar/mounts.py`. Flips a DEM/DSM upside-down so that
//! elevated features become surface depressions, then applies [`extract_sinks`]
//! and [`delineate_depressions`].

use std::path::{Path, PathBuf};

use richdem::{Array2D, DemNum, Topology, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::filling::{ExtractSinksOptions, ExtractSinksResult, extract_sinks};
use crate::slicing::{DelineateOptions, DelineateResult, delineate_depressions};

/// Output paths and metadata produced by [`delineate_mounts`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DelineateMountsResult {
    /// Path to `dem_flip.tif` (the inverted DEM/DSM).
    pub flipped_dem_path: PathBuf,
    /// Outputs of the intermediate [`extract_sinks`] step.
    pub sinks: ExtractSinksResult,
    /// Outputs of the [`delineate_depressions`] level-set step on the inverted surface.
    pub mounts: DelineateResult,
}

/// Flips a DEM upside-down: `flipped = -dem + max_elev + delta`, preserving NoData cells.
///
/// Matches `lidar.FlipDEM` (with exact NoData preservation).
pub fn flip_dem(dem: &Array2D<f64>, delta: f64) -> Result<Array2D<f64>> {
    if !delta.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "delta must be finite, got {delta}"
        )));
    }
    let nd = dem.no_data();
    let is_missing = |v: f64| !v.is_finite() || v.rd_eq(nd);

    let mut max_elev = f64::NEG_INFINITY;
    for &v in dem.data() {
        if !is_missing(v) && v > max_elev {
            max_elev = v;
        }
    }
    if !max_elev.is_finite() {
        return Ok(dem.clone());
    }

    let flipped_data: Vec<f64> = dem
        .data()
        .iter()
        .map(|&v| {
            if is_missing(v) {
                v
            } else {
                -v + max_elev + delta
            }
        })
        .collect();

    let mut out = Array2D::from_vec(dem.width(), dem.height(), flipped_data)?;
    out.copy_metadata_from(dem);
    out.set_no_data(nd);
    Ok(out)
}

/// Flips a DEM file on disk and optionally saves the inverted DEM to `out_file`.
pub fn flip_dem_file(
    in_dem: impl AsRef<Path>,
    delta: f64,
    out_file: Option<&Path>,
) -> Result<Array2D<f64>> {
    let dem = load_raster::<f64>(in_dem, None)?;
    let flipped = flip_dem(&dem, delta)?;
    if let Some(out) = out_file {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        save_raster(
            &flipped,
            out,
            Some(&format!("flip_dem(delta={delta})")),
            false,
        )?;
    }
    Ok(flipped)
}

/// Delineates the nested hierarchy of elevated features (mounts, buildings, tree crowns)
/// in a DEM/DSM.
///
/// Mirrors `lidar.DelineateMounts`: flips the DEM (`delta = 100.0`), runs
/// [`extract_sinks`], and runs [`delineate_depressions`].
pub fn delineate_mounts(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &DelineateOptions,
) -> Result<DelineateMountsResult> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let flipped_dem_path = out_dir.join("dem_flip.tif");
    flip_dem_file(in_dem, 100.0, Some(&flipped_dem_path))?;

    let sink_opts = ExtractSinksOptions {
        min_size: options.min_size,
        filled_dem: None,
        topology: Topology::D8,
        keep_files: true,
        compress: options.compress,
    };
    let sinks = extract_sinks(&flipped_dem_path, out_dir, &sink_opts)?;
    let mounts = delineate_depressions(&sinks.sink_path, out_dir, options)?;

    Ok(DelineateMountsResult {
        flipped_dem_path,
        sinks,
        mounts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_dem_inverts_and_preserves_nodata() {
        let mut dem = Array2D::from_vec(3, 1, vec![10.0, 25.0, -9999.0]).unwrap();
        dem.set_no_data(-9999.0);
        let flipped = flip_dem(&dem, 100.0).unwrap();
        assert!((flipped.get(0, 0) - 115.0).abs() < 1e-12);
        assert!((flipped.get(1, 0) - 100.0).abs() < 1e-12);
        assert_eq!(flipped.get(2, 0), -9999.0);
    }
}
