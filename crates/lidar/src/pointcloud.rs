//! Pure-Rust LiDAR point cloud I/O, format conversion, ground classification,
//! and DSM / DTM / CHM gridding via `terrio`.
//!
//! Replaces `laspy` and `WhiteboxTools` (`read_lidar`, `write_lidar`,
//! `convert_lidar`, `lidar_to_dsm`) from `lidar/common.py` with zero C dependencies,
//! and adds ASPRS classification / return filtering, Cloth Simulation Filter (CSF)
//! and Progressive Morphological Filter (PMF) ground classification, bare-earth DTM
//! generation, and Canopy Height Model (CHM) creation.

use std::path::Path;
use std::str::FromStr;

pub use terrio::pointcloud::{
    Dimension, LasHeader, PointCloud, PointFilter, PointFormat, PointReadOptions,
    PointWriteOptions, ReturnKind, ValueType, Values,
    filters::{
        CsfOptions, PmfOptions, csf as terrio_csf, csf_classes, pmf as terrio_pmf, pmf_classes,
    },
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

impl FromStr for GroundFilterMethod {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "preclassified" => Ok(Self::Preclassified),
            "csf" => Ok(Self::Csf),
            "pmf" => Ok(Self::Pmf),
            other => Err(Error::InvalidArgument(format!(
                "unknown ground filter method '{other}'; expected preclassified, csf, or pmf"
            ))),
        }
    }
}

/// Parses a single LiDAR pulse return kind name (`"first"`, `"last"`, `"intermediate"`,
/// or `"only"` / `"single"`, case-insensitive).
pub fn parse_return_kind(s: &str) -> Result<ReturnKind> {
    match s.trim().to_ascii_lowercase().as_str() {
        "first" => Ok(ReturnKind::First),
        "last" => Ok(ReturnKind::Last),
        "intermediate" => Ok(ReturnKind::Intermediate),
        "only" | "single" => Ok(ReturnKind::Only),
        other => Err(Error::InvalidArgument(format!(
            "unknown return kind '{other}'; expected first, last, intermediate, or only"
        ))),
    }
}

/// Parses a comma-separated list of LiDAR return kinds using [`parse_return_kind`].
pub fn parse_return_kinds(s: &str) -> Result<Vec<ReturnKind>> {
    let parsed: Vec<ReturnKind> = s
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(parse_return_kind)
        .collect::<Result<Vec<_>>>()?;
    if parsed.is_empty() {
        return Err(Error::InvalidArgument(
            "return kind list cannot be empty".into(),
        ));
    }
    Ok(parsed)
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

/// Converts a LiDAR point cloud between formats, LAS point formats, or LAS minor versions
/// using `terrio::pointcloud::convert` for bounded-memory chunked streaming.
///
/// Mirrors `lidar.convert_lidar`.
pub fn convert_lidar(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    point_format_id: Option<u8>,
    minor_version: Option<u8>,
) -> Result<usize> {
    let src_path = source.as_ref();
    let dst_path = destination.as_ref();
    crate::raster_ops::ensure_no_path_collision(src_path, &[dst_path])?;
    ensure_parent_dir(dst_path)?;
    let src_str = path_to_str(src_path)?;
    let dst_str = path_to_str(dst_path)?;

    // Read only the source header (O(1) memory) before converting so we neither re-decode
    // destination formats like PLY nor mask destination write/open failures. Since
    // `convert_lidar` applies no filtering or thinning, the converted count equals the source count.
    let count = terrio::pointcloud::open_points(src_str)?
        .header()
        .point_count;
    let count = usize::try_from(count).map_err(|_| {
        Error::Algorithm(format!(
            "converted point count {count} exceeds platform usize"
        ))
    })?;

    let opts = terrio::pointcloud::ConvertOptions {
        write: PointWriteOptions {
            point_format: point_format_id,
            minor_version,
            ..Default::default()
        },
        ..Default::default()
    };
    terrio::pointcloud::convert(src_str, dst_str, &opts)?;
    Ok(count)
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
///
/// Returns an error if `classes` is non-empty and the point cloud lacks a supported
/// integer `classification` dimension.
pub fn filter_by_classification(cloud: &PointCloud, classes: &[u8]) -> Result<PointCloud> {
    if classes.is_empty() {
        return Ok(cloud.clone());
    }
    let Some(dim) = cloud
        .dim("classification")
        .or_else(|| cloud.dim("Classification"))
    else {
        return Err(Error::InvalidArgument(
            "point cloud has no classification dimension for classification filtering".into(),
        ));
    };
    let keep: Vec<bool> = match &dim.values {
        Values::U8(v) => v.iter().map(|c| classes.contains(c)).collect(),
        Values::U16(v) => v
            .iter()
            .map(|&c| u8::try_from(c).is_ok_and(|b| classes.contains(&b)))
            .collect(),
        Values::I32(v) => v
            .iter()
            .map(|&c| u8::try_from(c).is_ok_and(|b| classes.contains(&b)))
            .collect(),
        Values::U32(v) => v
            .iter()
            .map(|&c| u8::try_from(c).is_ok_and(|b| classes.contains(&b)))
            .collect(),
        Values::I16(v) => v
            .iter()
            .map(|&c| u8::try_from(c).is_ok_and(|b| classes.contains(&b)))
            .collect(),
        Values::I8(v) => v
            .iter()
            .map(|&c| u8::try_from(c).is_ok_and(|b| classes.contains(&b)))
            .collect(),
        other => {
            return Err(Error::InvalidArgument(format!(
                "unsupported classification dimension type: {other:?}"
            )));
        }
    };
    Ok(cloud.select(&keep))
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
    if let Some(r) = options.radius
        && (!r.is_finite() || r <= 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "radius must be finite and > 0, got {r}"
        )));
    }
    if let Some(lo) = options.min_z
        && !lo.is_finite()
    {
        return Err(Error::InvalidArgument(format!(
            "min_z must be finite, got {lo}"
        )));
    }
    if let Some(hi) = options.max_z
        && !hi.is_finite()
    {
        return Err(Error::InvalidArgument(format!(
            "max_z must be finite, got {hi}"
        )));
    }
    if let (Some(lo), Some(hi)) = (options.min_z, options.max_z)
        && lo > hi
    {
        return Err(Error::InvalidArgument(format!(
            "min_z ({lo}) must be <= max_z ({hi})"
        )));
    }
    let out_path = out_dsm.as_ref();
    crate::raster_ops::ensure_no_path_collision(in_lidar.as_ref(), &[out_path])?;
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

    let mut grid_opts = GridOptions {
        resolution: options.resolution,
        method: options.method,
        stats: vec![options.stat],
        dimension: "z".to_string(),
        radius: options.radius,
        window_size: options.window_size,
        nodata: options.nodata,
        ..Default::default()
    };

    // Stream statistics gridding in bounded chunks via `terrio::pointcloud::grid::grid_reader`
    // so large LAS/LAZ/COPC/PLY/E57 files do not load all points into RAM at once.
    let dsm = if options.method == GridMethod::Statistics {
        let in_str = path_to_str(in_lidar.as_ref())?;
        let mut reader = terrio::pointcloud::open_points(in_str)?;
        if reader.header().point_count == 0 {
            return Err(Error::Algorithm(
                "no points remain after classification / return / elevation filtering".into(),
            ));
        }
        let read_opts = PointReadOptions {
            filter,
            ..Default::default()
        };
        // When filtering is active (or format bounds require scanning), compute filtered bounds
        // in one streaming pass up front so an empty filtered cloud returns our typed error
        // without string-matching `terrio`'s error message and `grid_reader` runs a single pass.
        let needs_bounds_scan =
            !read_opts.filter.is_empty() || !matches!(reader.format_name(), "LAS" | "LAZ" | "COPC");
        if needs_bounds_scan {
            let mut bounds: Option<[f64; 4]> = None;
            reader.start_chunks(&read_opts, 0)?;
            while let Some(chunk) = reader.next_chunk()? {
                if let Some(c) = chunk.bounds() {
                    bounds = Some(match bounds {
                        None => [c[0], c[1], c[3], c[4]],
                        Some(a) => [
                            a[0].min(c[0]),
                            a[1].min(c[1]),
                            a[2].max(c[3]),
                            a[3].max(c[4]),
                        ],
                    });
                }
            }
            let Some(b) = bounds else {
                return Err(Error::Algorithm(
                    "no points remain after classification / return / elevation filtering".into(),
                ));
            };
            grid_opts.bounds = Some(b);
        }
        terrio::pointcloud::grid::grid_reader(&mut reader, &read_opts, &grid_opts, None)?
    } else {
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
        terrio::pointcloud::grid::grid(&cloud, &grid_opts)?
    };
    let dims = (dsm.raster.cols, dsm.raster.rows);
    terrio::write_raster_dataset(&dsm, out_str, &RasterWriteOptions::default())?;
    Ok(dims)
}

fn select_ground_points(
    raw: &PointCloud,
    method: GroundFilterMethod,
    ground_class: u8,
) -> Result<PointCloud> {
    match method {
        GroundFilterMethod::Preclassified => filter_by_classification(raw, &[ground_class]),
        GroundFilterMethod::Csf | GroundFilterMethod::Pmf => {
            if ground_class != 2 {
                return Err(Error::InvalidArgument(format!(
                    "{method:?} always assigns ASPRS ground class 2; set ground_class=2 or use GroundFilterMethod::Preclassified (got ground_class={ground_class})"
                )));
            }
            let classes = match method {
                GroundFilterMethod::Csf => csf_classes(raw, &CsfOptions::default())?,
                GroundFilterMethod::Pmf => pmf_classes(raw, &PmfOptions::default())?,
                GroundFilterMethod::Preclassified => unreachable!(),
            };
            let keep: Vec<bool> = classes.iter().map(|&c| c == Some(2)).collect();
            Ok(raw.select(&keep))
        }
    }
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
    if let Some(r) = options.radius
        && (!r.is_finite() || r <= 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "radius must be finite and > 0, got {r}"
        )));
    }
    let out_path = out_dtm.as_ref();
    crate::raster_ops::ensure_no_path_collision(in_lidar.as_ref(), &[out_path])?;
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
            select_ground_points(&raw, method, options.ground_class)?
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
    if let Some(r) = options.radius
        && (!r.is_finite() || r <= 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "radius must be finite and > 0, got {r}"
        )));
    }
    let out_path = out_chm.as_ref();
    crate::raster_ops::ensure_no_path_collision(in_lidar.as_ref(), &[out_path])?;
    ensure_parent_dir(out_path)?;

    let raw = read_lidar(in_lidar)?;
    if raw.is_empty() {
        return Err(Error::Algorithm("input point cloud is empty".into()));
    }
    let ground = select_ground_points(&raw, options.ground_method, options.ground_class)?;
    if ground.is_empty() {
        return Err(Error::Algorithm(format!(
            "no ground points found (class={})",
            options.ground_class
        )));
    }

    // Use the raw cloud's XY extent for both DSM and DTM so both grids share one exact lattice.
    let shared_bounds = raw.bounds().map(|b| [b[0], b[1], b[3], b[4]]);

    let dsm_opts = GridOptions {
        resolution: options.resolution,
        method: GridMethod::Statistics,
        stats: vec![GridStat::Max],
        dimension: "z".to_string(),
        radius: options.radius,
        window_size: options.window_size,
        bounds: shared_bounds,
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
        bounds: shared_bounds,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_ground_points_rejects_stale_class2_canopy_returns() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut cls = Vec::new();
        let mut rn = Vec::new();
        let mut nr = Vec::new();
        for r in 0..16 {
            for c in 0..16 {
                xs.push(500_000.0 + c as f64);
                ys.push(4_000_000.0 + r as f64);
                zs.push(100.0);
                cls.push(1u8); // initially unclassified ground
                rn.push(2u8);
                nr.push(2u8);
            }
        }
        // Add elevated canopy points that carry a stale pre-existing class 2 label on first return
        // (which CSF ignores in its default Last/Only return set) and on only-return points.
        xs.push(500_008.0);
        ys.push(4_000_008.0);
        zs.push(125.0);
        cls.push(2u8);
        rn.push(1u8);
        nr.push(2u8);

        xs.push(500_009.0);
        ys.push(4_000_009.0);
        zs.push(125.0);
        cls.push(2u8);
        rn.push(1u8);
        nr.push(1u8);

        let cloud = PointCloud::new(
            vec![
                Dimension::new("x", Values::F64(xs)),
                Dimension::new("y", Values::F64(ys)),
                Dimension::new("z", Values::F64(zs)),
                Dimension::new("classification", Values::U8(cls)),
                Dimension::new("return_number", Values::U8(rn)),
                Dimension::new("number_of_returns", Values::U8(nr)),
            ],
            terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
        );

        let ground = select_ground_points(&cloud, GroundFilterMethod::Csf, 2).unwrap();
        assert!(!ground.is_empty());
        let Values::F64(gz) = &ground.dim("z").unwrap().values else {
            panic!("expected f64 z dimension");
        };
        assert!(
            gz.iter().all(|&z| z < 105.0),
            "stale class-2 canopy points must not leak into CSF ground subset"
        );
    }

    #[test]
    fn streaming_convert_and_lidar_to_dsm_match_in_memory_grid() {
        let tmp = tempfile::tempdir().unwrap();
        let las_path = tmp.path().join("in.las");
        let laz_path = tmp.path().join("out.laz");
        let dsm_unfiltered = tmp.path().join("dsm_all.tif");
        let dsm_filtered = tmp.path().join("dsm_filtered.tif");

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut cls = Vec::new();
        let mut rn = Vec::new();
        let mut nr = Vec::new();
        for r in 0..12 {
            for c in 0..12 {
                xs.push(500_000.0 + c as f64);
                ys.push(4_000_000.0 + r as f64);
                zs.push(100.0 + (r + c) as f64 * 0.25);
                cls.push(if (r + c) % 2 == 0 { 2u8 } else { 5u8 });
                rn.push(1u8);
                nr.push(1u8);
            }
        }
        let cloud = PointCloud::new(
            vec![
                Dimension::new("x", Values::F64(xs)),
                Dimension::new("y", Values::F64(ys)),
                Dimension::new("z", Values::F64(zs)),
                Dimension::new("classification", Values::U8(cls)),
                Dimension::new("return_number", Values::U8(rn)),
                Dimension::new("number_of_returns", Values::U8(nr)),
            ],
            terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
        );
        write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();

        let count = convert_lidar(&las_path, &laz_path, Some(6), Some(4)).unwrap();
        assert_eq!(count, 144);
        assert_eq!(read_lidar(&laz_path).unwrap().len(), 144);

        // Compare streaming unfiltered DSM against in-memory terrio grid.
        let opts_all = LidarToDsmOptions {
            resolution: 1.0,
            radius: Some(1.5),
            ..Default::default()
        };
        let (cols, rows) = lidar_to_dsm(&laz_path, &dsm_unfiltered, &opts_all).unwrap();
        let streamed_arr = richdem::load_raster::<f64>(&dsm_unfiltered, None).unwrap();
        let ref_ds = terrio::pointcloud::grid::grid(
            &read_lidar(&laz_path).unwrap(),
            &GridOptions {
                resolution: opts_all.resolution,
                method: opts_all.method,
                stats: vec![opts_all.stat],
                dimension: "z".to_string(),
                radius: opts_all.radius,
                window_size: opts_all.window_size,
                nodata: opts_all.nodata,
                ..Default::default()
            },
        )
        .unwrap();
        let ref_arr = richdem::dataset_to_dynamic(ref_ds, "ref").unwrap().to_f64();
        assert_eq!((cols, rows), (ref_arr.width(), ref_arr.height()));
        for (a, b) in streamed_arr.data().iter().zip(ref_arr.data().iter()) {
            assert!((a - b).abs() < 1e-12, "DSM mismatch: {a} vs {b}");
        }

        // Filtered streaming DSM (class 2, z range): verify exact extent and cell values match
        // in-memory gridding on the same filtered point cloud.
        let opts_filt = LidarToDsmOptions {
            resolution: 1.0,
            min_z: Some(100.0),
            max_z: Some(104.0),
            classifications: Some(vec![2]),
            return_kinds: Some(vec![ReturnKind::Only]),
            ..Default::default()
        };
        let (fc, fr) = lidar_to_dsm(&laz_path, &dsm_filtered, &opts_filt).unwrap();
        let filt_arr = richdem::load_raster::<f64>(&dsm_filtered, None).unwrap();
        let filt_cloud = read_lidar_filtered(
            &laz_path,
            PointFilter {
                bounds: None,
                z_range: Some((100.0, 104.0)),
                classes: vec![2],
                returns: vec![ReturnKind::Only],
            },
        )
        .unwrap();
        let filt_ref_ds = terrio::pointcloud::grid::grid(
            &filt_cloud,
            &GridOptions {
                resolution: opts_filt.resolution,
                method: opts_filt.method,
                stats: vec![opts_filt.stat],
                dimension: "z".to_string(),
                radius: opts_filt.radius,
                window_size: opts_filt.window_size,
                nodata: opts_filt.nodata,
                ..Default::default()
            },
        )
        .unwrap();
        let filt_ref_arr = richdem::dataset_to_dynamic(filt_ref_ds, "filt_ref")
            .unwrap()
            .to_f64();
        assert_eq!((fc, fr), (filt_ref_arr.width(), filt_ref_arr.height()));
        assert_eq!(filt_arr.geotransform, filt_ref_arr.geotransform);
        for (a, b) in filt_arr.data().iter().zip(filt_ref_arr.data().iter()) {
            assert!((a - b).abs() < 1e-12, "filtered DSM mismatch: {a} vs {b}");
        }

        // Empty filter result still returns an error.
        let opts_empty = LidarToDsmOptions {
            classifications: Some(vec![99]),
            ..Default::default()
        };
        assert!(lidar_to_dsm(&laz_path, tmp.path().join("empty.tif"), &opts_empty).is_err());
    }
}
