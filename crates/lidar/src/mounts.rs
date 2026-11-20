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
use crate::filling::{
    ExtractSinksOptions, ExtractSinksResult, extract_sinks, extract_sinks_in_memory,
};
use crate::slicing::{
    DelineateOptions, DelineateResult, InMemoryDepressions, delineate_depressions,
    delineate_depressions_in_memory,
};

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
/// If the input `no_data` sentinel collides with any valid flipped elevation, a
/// collision-free output NoData value is chosen automatically and missing cells are
/// remapped to it so valid inverted peaks are never masked out.
///
/// Matches `lidar.FlipDEM` (with collision-free NoData preservation).
pub fn flip_dem(dem: &Array2D<f64>, delta: f64) -> Result<Array2D<f64>> {
    if !delta.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "delta must be finite, got {delta}"
        )));
    }
    let nd = dem.no_data();
    let is_missing = |v: f64| !v.is_finite() || v.rd_eq(nd);

    let mut min_elev = f64::INFINITY;
    let mut max_elev = f64::NEG_INFINITY;
    for &v in dem.data() {
        if !is_missing(v) {
            if v < min_elev {
                min_elev = v;
            }
            if v > max_elev {
                max_elev = v;
            }
        }
    }
    if !max_elev.is_finite() {
        return Ok(dem.clone());
    }

    // Valid flipped elevations lie in [delta, max_elev - min_elev + delta].
    let flipped_min = delta;
    let flipped_max = (max_elev - min_elev) + delta;
    let out_nd = if nd.is_finite() && nd >= flipped_min && nd <= flipped_max {
        if -9999.0 < flipped_min {
            -9999.0
        } else {
            flipped_min - 9999.0
        }
    } else {
        nd
    };

    let flipped_data: Vec<f64> = dem
        .data()
        .iter()
        .map(|&v| {
            if is_missing(v) {
                out_nd
            } else {
                -v + max_elev + delta
            }
        })
        .collect();

    let mut out = Array2D::from_vec(dem.width(), dem.height(), flipped_data)?;
    out.copy_metadata_from(dem);
    out.set_no_data(out_nd);
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
/// completely in memory with zero filesystem I/O.
///
/// Flips the input DSM (`delta = 100.0`), extracts sinks with D8 topology, and runs
/// [`delineate_depressions_in_memory`] on the inverted surface.
pub fn delineate_mounts_in_memory(
    dsm: &Array2D<f64>,
    min_size: usize,
    min_height: f64,
    interval: f64,
) -> Result<InMemoryDepressions> {
    let flipped = flip_dem(dsm, 100.0)?;
    let sinks = extract_sinks_in_memory(&flipped, min_size, Topology::D8)?;
    delineate_depressions_in_memory(&sinks.sink, min_size, min_height, interval)
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
    let in_dem = in_dem.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let flipped_dem_path = out_dir.join("dem_flip.tif");
    let protected_outputs = [
        flipped_dem_path.clone(),
        out_dir.join("dem.tif"),
        out_dir.join("dem_filled.tif"),
        out_dir.join("dem_diff.tif"),
        out_dir.join("depth.tif"),
        out_dir.join("sink.tif"),
        out_dir.join("region.tif"),
        out_dir.join("depression_id.tif"),
        out_dir.join("depression_level.tif"),
    ];
    let protected_refs: Vec<&Path> = protected_outputs.iter().map(PathBuf::as_path).collect();
    crate::raster_ops::ensure_no_path_collision(in_dem, &protected_refs)?;
    flip_dem_file(in_dem, 100.0, Some(&flipped_dem_path))?;

    let sink_opts = ExtractSinksOptions {
        min_size: options.min_size,
        filled_dem: None,
        topology: Topology::D8,
        keep_files: true,
        compress: options.compress,
    };
    let sinks = extract_sinks(&flipped_dem_path, out_dir, &sink_opts)?;
    let sink_path = sinks
        .sink_path
        .as_ref()
        .ok_or_else(|| Error::Algorithm("extract_sinks did not produce sink.tif".into()))?;
    let mounts = delineate_depressions(sink_path, out_dir, options)?;

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

    #[test]
    fn flip_dem_avoids_nodata_collision_with_inverted_peak() {
        let mut dem = Array2D::from_vec(3, 1, vec![100.0, 120.0, 80.0]).unwrap();
        dem.set_no_data(100.0);
        let flipped = flip_dem(&dem, 100.0).unwrap();
        // Valid peak 120 maps to 100.0; NoData must be remapped away from 100.0.
        assert_ne!(flipped.no_data(), 100.0);
        assert_eq!(flipped.get(0, 0), flipped.no_data());
        assert!((flipped.get(1, 0) - 100.0).abs() < 1e-12);
        assert!((flipped.get(2, 0) - 140.0).abs() < 1e-12);
    }
}
