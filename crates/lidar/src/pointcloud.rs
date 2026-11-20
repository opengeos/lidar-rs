//! Pure-Rust LiDAR point cloud I/O, format conversion, and DSM gridding via `terrio`.
//!
//! Replaces `laspy` and `WhiteboxTools` (`read_lidar`, `write_lidar`,
//! `convert_lidar`, `lidar_to_dsm`) from `lidar/common.py` with zero C dependencies.

use std::path::Path;

pub use terrio::pointcloud::{
    Dimension, LasHeader, PointCloud, PointFilter, PointFormat, PointReadOptions,
    PointWriteOptions, ValueType, Values,
    grid::{GridMethod, GridOptions, GridStat},
};

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
            stat: GridStat::Max,
            method: GridMethod::Statistics,
            window_size: 0,
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

/// Generates a Digital Surface Model (DSM) GeoTIFF from a LiDAR point cloud.
///
/// Mirrors `lidar.lidar_to_dsm` using pure-Rust `terrio::pointcloud::grid`.
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

    let mut cloud = read_lidar(in_lidar)?;
    if (options.min_z.is_some() || options.max_z.is_some())
        && let Some(zs) = cloud.coords("z")
    {
        let keep: Vec<bool> = zs
            .iter()
            .map(|&z| {
                options.min_z.is_none_or(|lo| z >= lo) && options.max_z.is_none_or(|hi| z <= hi)
            })
            .collect();
        cloud = cloud.select(&keep);
    }

    if cloud.is_empty() {
        return Err(Error::Algorithm(
            "no points remain after elevation filtering".into(),
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
