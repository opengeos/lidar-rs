//! Pure-Rust LiDAR point cloud I/O, format conversion, ground classification,
//! and DSM / DTM / CHM gridding via `terrio`.
//!
//! Replaces `laspy` and `WhiteboxTools` (`read_lidar`, `write_lidar`,
//! `convert_lidar`, `lidar_to_dsm`) from `lidar/common.py` with zero C dependencies,
//! and adds ASPRS classification / return filtering, Cloth Simulation Filter (CSF)
//! and Progressive Morphological Filter (PMF) ground classification, bare-earth DTM
//! generation, and Canopy Height Model (CHM) creation.

use std::path::Path;

pub use terrio::pointcloud::{
    Dimension, LasHeader, PointCloud, PointFilter, PointFormat, PointReadOptions,
    PointWriteOptions, ReturnKind, ValueType, Values,
    filters::{CsfOptions, PmfOptions, csf as terrio_csf, pmf as terrio_pmf},
    grid::{GridMethod, GridOptions, GridStat},
};

use richdem::DemNum;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::RasterWriteOptions;

use crate::error::{Error, Result};

fn path_to_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", path.display())))
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Ground classification algorithm for [`lidar_to_dtm`] and [`classify_ground`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GroundFilterMethod {
    /// Use pre-classified ASPRS ground returns (`Classification == 2`) without reclassifying.
    Preclassified,
    /// Cloth Simulation Filter (Zhang et al., 2016) via `terrio`.
    #[default]
    Csf,
    /// Progressive Morphological Filter (Zhang et al., 2003) via `terrio`.
    Pmf,
}

/// Options for [`lidar_to_dsm`].
#[derive(Debug, Clone)]
pub struct LidarToDsmOptions {
    /// Output raster pixel resolution (default `1.0`).
    pub resolution: f64,
    /// Search radius around each cell center; `None` uses `terrio`'s default (`resolution * sqrt(2)`).
    pub radius: Option<f64>,
    /// Optional minimum Z threshold for point inclusion.
    pub min_z: Option<f64>,
    /// Optional maximum Z threshold for point inclusion.
    pub max_z: Option<f64>,
    /// Optional ASPRS classification codes to keep (e.g., `vec![2]` for ground). Empty/None keeps all.
    pub classifications: Option<Vec<u8>>,
    /// Optional pulse return kinds to keep (e.g., `First`, `Last`, `Only`). Empty/None keeps all.
    pub return_kinds: Option<Vec<ReturnKind>>,
    /// Grid statistic to compute (`GridStat::Max` by default for DSM).
    pub stat: GridStat,
    /// Optional interpolation method (`GridMethod::Statistics` by default).
    pub method: GridMethod,
    /// Gap-filling window size in cells (`0` = no gap filling).
    pub window_size: usize,
    /// Output NoData value (`-9999.0` by default).
    pub nodata: f64,
}

impl Default for LidarToDsmOptions {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            radius: None,
            min_z: None,
            max_z: None,
            classifications: None,
            return_kinds: None,
            stat: GridStat::Max,
            method: GridMethod::Statistics,
            window_size: 0,
            nodata: -9999.0,
        }
    }
}

/// Options for [`lidar_to_dtm`] and [`lidar_to_chm`].
#[derive(Debug, Clone)]
pub struct LidarToDtmOptions {
    /// Output raster pixel resolution (default `1.0`).
    pub resolution: f64,
    /// Ground classification strategy (`GroundFilterMethod::Csf` by default).
    pub ground_method: GroundFilterMethod,
    /// ASPRS ground class code (default `2`).
    pub ground_class: u8,
    /// Search radius around each cell center; `None` uses `terrio` default.
    pub radius: Option<f64>,
    /// Interpolation method for bare-earth gridding (`GridMethod::Linear` by default).
    pub method: GridMethod,
    /// Grid statistic when `method == GridMethod::Statistics` (`GridStat::Idw` by default).
    pub stat: GridStat,
    /// Gap-filling window size in cells (default `5`).
    pub window_size: usize,
    /// Output NoData value (`-9999.0` by default).
    pub nodata: f64,
}

impl Default for LidarToDtmOptions {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Csf,
            ground_class: 2,
            radius: None,
            method: GridMethod::Linear,
            stat: GridStat::Idw,
            window_size: 5,
            nodata: -9999.0,
        }
    }
}

/// Reads a LiDAR point cloud file (LAS, LAZ, COPC, PLY, E57, EPT).
///
/// Mirrors `lidar.read_lidar`.
pub fn read_lidar(path: impl AsRef<Path>) -> Result<PointCloud> {
    let p = path_to_str(path.as_ref())?;
    let mut reader = terrio::pointcloud::open_points(p)?;
    Ok(reader.read(&PointReadOptions::default())?)
}

/// Reads a LiDAR point cloud with an explicit [`PointFilter`] (classification, return kind, Z range, bounds).
pub fn read_lidar_filtered(path: impl AsRef<Path>, filter: PointFilter) -> Result<PointCloud> {
    let p = path_to_str(path.as_ref())?;
    let mut reader = terrio::pointcloud::open_points(p)?;
    let opts = PointReadOptions {
        filter,
        ..Default::default()
    };
    Ok(reader.read(&opts)?)
}

/// Writes a [`PointCloud`] to disk (LAS, LAZ, COPC, PLY, or E57 by file extension).
///
/// Mirrors `lidar.write_lidar`.
pub fn write_lidar(
    cloud: &PointCloud,
    destination: impl AsRef<Path>,
    options: &PointWriteOptions,
) -> Result<()> {
    let dst = destination.as_ref();
    ensure_parent_dir(dst)?;
    let dst_str = path_to_str(dst)?;
    terrio::pointcloud::write_points(cloud, dst_str, options)?;
    Ok(())
}

/// Converts a LiDAR point cloud between formats, LAS point formats, or LAS minor versions.
///
/// Mirrors `lidar.convert_lidar`.
pub fn convert_lidar(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    point_format_id: Option<u8>,
    minor_version: Option<u8>,
) -> Result<usize> {
    let cloud = read_lidar(source)?;
    let n = cloud.len();
    let opts = PointWriteOptions {
        point_format: point_format_id,
        minor_version,
        ..Default::default()
    };
    write_lidar(&cloud, destination, &opts)?;
    Ok(n)
}

/// Classifies ground points in a [`PointCloud`] using either Cloth Simulation Filter (CSF)
/// or Progressive Morphological Filter (PMF).
pub fn classify_ground(cloud: &PointCloud, method: GroundFilterMethod) -> Result<PointCloud> {
    match method {
        GroundFilterMethod::Preclassified => Ok(cloud.clone()),
        GroundFilterMethod::Csf => Ok(terrio_csf(cloud, &CsfOptions::default())?),
        GroundFilterMethod::Pmf => Ok(terrio_pmf(cloud, &PmfOptions::default())?),
    }
}

/// Filters a [`PointCloud`] in memory to keep only points matching `classes`.
pub fn filter_by_classification(cloud: &PointCloud, classes: &[u8]) -> PointCloud {
    if classes.is_empty() {
        return cloud.clone();
    }
    let Some(dim) = cloud
        .dim("classification")
        .or_else(|| cloud.dim("Classification"))
    else {
        return cloud.clone();
    };
    let keep: Vec<bool> = match &dim.values {
        Values::U8(v) => v.iter().map(|c| classes.contains(c)).collect(),
        Values::U16(v) => v.iter().map(|c| classes.contains(&(*c as u8))).collect(),
        Values::I32(v) => v.iter().map(|c| classes.contains(&(*c as u8))).collect(),
        _ => vec![true; cloud.len()],
    };
    cloud.select(&keep)
}

/// Generates a Digital Surface Model (DSM) GeoTIFF from a LiDAR point cloud.
///
/// Supports optional ASPRS classification filtering, pulse return-kind filtering,
/// and elevation bounds before gridding via pure-Rust `terrio::pointcloud::grid`.
pub fn lidar_to_dsm(
    in_lidar: impl AsRef<Path>,
    out_dsm: impl AsRef<Path>,
    options: &LidarToDsmOptions,
) -> Result<(usize, usize)> {
    if !options.resolution.is_finite() || options.resolution <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "resolution must be finite and > 0, got {}",
            options.resolution
        )));
    }
    let out_path = out_dsm.as_ref();
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let filter = PointFilter {
        bounds: None,
        z_range: match (options.min_z, options.max_z) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            (Some(lo), None) => Some((lo, f64::INFINITY)),
            (None, Some(hi)) => Some((f64::NEG_INFINITY, hi)),
            (None, None) => None,
        },
        classes: options.classifications.clone().unwrap_or_default(),
        returns: options.return_kinds.clone().unwrap_or_default(),
    };

    let cloud = if filter.is_empty() {
        read_lidar(in_lidar)?
    } else {
        read_lidar_filtered(in_lidar, filter)?
    };

    if cloud.is_empty() {
        return Err(Error::Algorithm(
            "no points remain after classification / return / elevation filtering".into(),
        ));
    }

    let grid_opts = GridOptions {
        resolution: options.resolution,
        method: options.method,
        stats: vec![options.stat],
        dimension: "z".to_string(),
        radius: options.radius,
        window_size: options.window_size,
        nodata: options.nodata,
        ..Default::default()
    };
    let dsm = terrio::pointcloud::grid::grid(&cloud, &grid_opts)?;
    let dims = (dsm.raster.cols, dsm.raster.rows);
    terrio::write_raster_dataset(&dsm, out_str, &RasterWriteOptions::default())?;
    Ok(dims)
}

/// Generates a bare-earth Digital Terrain Model (DTM) GeoTIFF from a LiDAR point cloud.
///
/// Uses existing ASPRS ground returns (`Preclassified`) or runs pure-Rust ground
/// classification (`Csf` or `Pmf`), selects ground points, and interpolates a continuous
/// bare-earth surface.
pub fn lidar_to_dtm(
    in_lidar: impl AsRef<Path>,
    out_dtm: impl AsRef<Path>,
    options: &LidarToDtmOptions,
) -> Result<(usize, usize)> {
    if !options.resolution.is_finite() || options.resolution <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "resolution must be finite and > 0, got {}",
            options.resolution
        )));
    }
    let out_path = out_dtm.as_ref();
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let ground = match options.ground_method {
        GroundFilterMethod::Preclassified => {
            let filter = PointFilter {
                classes: vec![options.ground_class],
                ..Default::default()
            };
            read_lidar_filtered(in_lidar, filter)?
        }
        method => {
            let raw = read_lidar(in_lidar)?;
            let classified = classify_ground(&raw, method)?;
            filter_by_classification(&classified, &[options.ground_class])
        }
    };

    if ground.is_empty() {
        return Err(Error::Algorithm(format!(
            "no ground points found (class={})",
            options.ground_class
        )));
    }

    let grid_opts = GridOptions {
        resolution: options.resolution,
        method: options.method,
        stats: vec![options.stat],
        dimension: "z".to_string(),
        radius: options.radius,
        window_size: options.window_size,
        nodata: options.nodata,
        ..Default::default()
    };
    let dtm = terrio::pointcloud::grid::grid(&ground, &grid_opts)?;
    let dims = (dtm.raster.cols, dtm.raster.rows);
    terrio::write_raster_dataset(&dtm, out_str, &RasterWriteOptions::default())?;
    Ok(dims)
}

/// Generates a Canopy Height Model (`CHM = max(0, DSM - DTM)`) GeoTIFF directly from a
/// LiDAR point cloud in pure Rust.
pub fn lidar_to_chm(
    in_lidar: impl AsRef<Path>,
    out_chm: impl AsRef<Path>,
    options: &LidarToDtmOptions,
) -> Result<(usize, usize)> {
    if !options.resolution.is_finite() || options.resolution <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "resolution must be finite and > 0, got {}",
            options.resolution
        )));
    }
    let out_path = out_chm.as_ref();
    ensure_parent_dir(out_path)?;

    let raw = read_lidar(in_lidar)?;
    if raw.is_empty() {
        return Err(Error::Algorithm("input point cloud is empty".into()));
    }
    let classified = match options.ground_method {
        GroundFilterMethod::Preclassified => raw.clone(),
        method => classify_ground(&raw, method)?,
    };
    let ground = filter_by_classification(&classified, &[options.ground_class]);
    if ground.is_empty() {
        return Err(Error::Algorithm(format!(
            "no ground points found (class={})",
            options.ground_class
        )));
    }

    let dsm_opts = GridOptions {
        resolution: options.resolution,
        method: GridMethod::Statistics,
        stats: vec![GridStat::Max],
        dimension: "z".to_string(),
        radius: options.radius,
        window_size: options.window_size,
        nodata: options.nodata,
        ..Default::default()
    };
    let dtm_opts = GridOptions {
        resolution: options.resolution,
        method: options.method,
        stats: vec![options.stat],
        dimension: "z".to_string(),
        radius: options.radius,
        window_size: options.window_size,
        nodata: options.nodata,
        ..Default::default()
    };

    let dsm_ds = terrio::pointcloud::grid::grid(&raw, &dsm_opts)?;
    let dtm_ds = terrio::pointcloud::grid::grid(&ground, &dtm_opts)?;

    let dsm_arr = richdem::dataset_to_dynamic(dsm_ds, "dsm")?.to_f64();
    let dtm_arr = richdem::dataset_to_dynamic(dtm_ds, "dtm")?.to_f64();

    let w = dsm_arr.width();
    let h = dsm_arr.height();
    let nd = options.nodata;
    let dsm_nd = dsm_arr.no_data();
    let dtm_nd = dtm_arr.no_data();
    let gt_dsm = dsm_arr.geotransform;
    let gt_dtm = dtm_arr.geotransform;

    let mut chm_data = vec![nd; w * h];
    for r in 0..h {
        let wy = gt_dsm[3] + (r as f64 + 0.5) * gt_dsm[5];
        let dr = ((wy - gt_dtm[3]) / gt_dtm[5]).floor() as isize;
        if dr < 0 || dr >= dtm_arr.height() as isize {
            continue;
        }
        let dru = dr as usize;
        for c in 0..w {
            let z_s = dsm_arr.get(c, r);
            if !z_s.is_finite() || z_s.rd_eq(dsm_nd) {
                continue;
            }
            let wx = gt_dsm[0] + (c as f64 + 0.5) * gt_dsm[1];
            let dc = ((wx - gt_dtm[0]) / gt_dtm[1]).floor() as isize;
            if dc < 0 || dc >= dtm_arr.width() as isize {
                continue;
            }
            let z_t = dtm_arr.get(dc as usize, dru);
            if !z_t.is_finite() || z_t.rd_eq(dtm_nd) {
                continue;
            }
            chm_data[r * w + c] = (z_s - z_t).max(0.0);
        }
    }

    let mut chm = richdem::Array2D::from_vec(w, h, chm_data)?;
    chm.copy_metadata_from(&dsm_arr);
    chm.set_no_data(nd);
    richdem::save_raster(&chm, out_path, Some("lidar_to_chm"), false)?;
    Ok((w, h))
}
