//! Surface depression (sink) extraction and DEM refinement.
//!
//! Pure-Rust port of `lidar/filling.py`, using [`richdem::fill_depressions`] for
//! Priority-Flood depression filling and [`terrio`] for GDAL-free GeoTIFF,
//! Shapefile, and GeoPackage I/O.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use richdem::{Array2D, DemNum, Topology, fill_depressions, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::raster_ops::{ensure_no_path_collision, join_csv_to_vector, polygonize};
use crate::region::{RegionProps, region_group};

/// Geometric and topographic attributes of an extracted sink (depression region).
///
/// Matches the `Depression` class and `regions_info.csv` schema in `lidar/filling.py`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SinkDepression {
    /// 1-based region ID.
    pub region_id: u32,
    /// Number of pixels in the depression.
    pub count: usize,
    /// Surface area in map units squared.
    pub area: f64,
    /// Depression relief volume below `max_elev` (`avg_depth * area`, matching `lidar/filling.py`).
    pub volume: f64,
    /// Mean depth in vertical elevation units.
    pub avg_depth: f64,
    /// Maximum depth in vertical elevation units.
    pub max_depth: f64,
    /// Lowest elevation in the depression.
    pub min_elev: f64,
    /// Spill / boundary (maximum) elevation in the depression.
    pub max_elev: f64,
    /// Perimeter in map units.
    pub perimeter: f64,
    /// Major axis length in map units.
    pub major_axis: f64,
    /// Minor axis length in map units.
    pub minor_axis: f64,
    /// Elongatedness (`major_axis / minor_axis`).
    pub elongatedness: f64,
    /// Equivalent ellipse eccentricity.
    pub eccentricity: f64,
    /// Orientation in degrees.
    pub orientation: f64,
    /// Ratio of region area to bounding-box area.
    pub area_bbox_ratio: f64,
}

impl From<&RegionProps> for SinkDepression {
    fn from(p: &RegionProps) -> Self {
        Self {
            region_id: p.label,
            count: p.count,
            area: p.area,
            volume: p.volume,
            avg_depth: p.mean_depth,
            max_depth: p.max_depth,
            min_elev: p.min_elev,
            max_elev: p.max_elev,
            perimeter: p.perimeter,
            major_axis: p.major_axis,
            minor_axis: p.minor_axis,
            elongatedness: p.elongatedness,
            eccentricity: p.eccentricity,
            orientation: p.orientation,
            area_bbox_ratio: p.area_bbox_ratio,
        }
    }
}

/// Options controlling [`extract_sinks`].
#[derive(Debug, Clone)]
pub struct ExtractSinksOptions {
    /// Minimum number of pixels to keep a depression (regions with `count > min_size` are kept).
    pub min_size: usize,
    /// Optional pre-filled DEM path; if `None`, fills the DEM with `richdem::fill_depressions`.
    pub filled_dem: Option<PathBuf>,
    /// Connectivity topology for depression filling (`Topology::D8` by default).
    pub topology: Topology,
    /// Whether to keep all intermediate rasters, CSV, and shapefile (`true` by default).
    /// When `false`, intermediate files are deleted and only the joined `.gpkg` is retained.
    pub keep_files: bool,
    /// Whether to DEFLATE-compress output GeoTIFFs.
    pub compress: bool,
}

impl Default for ExtractSinksOptions {
    fn default() -> Self {
        Self {
            min_size: 1000,
            filled_dem: None,
            topology: Topology::D8,
            keep_files: true,
            compress: false,
        }
    }
}

/// Output paths and metadata produced by [`extract_sinks`].
///
/// Intermediate paths are `Some(...)` when retained on disk and `None` when
/// removed by `keep_files = false`. `gpkg_path` is always retained.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtractSinksResult {
    /// Path to `sink.tif` (elevation inside sinks, 0 outside), or `None` if removed.
    pub sink_path: Option<PathBuf>,
    /// Path to refined `dem.tif`, or `None` if removed.
    pub refined_dem_path: Option<PathBuf>,
    /// Path to `dem_filled.tif` (or caller-supplied filled DEM), or `None` if removed.
    pub filled_dem_path: Option<PathBuf>,
    /// Path to `dem_diff.tif`, or `None` if removed.
    pub dem_diff_path: Option<PathBuf>,
    /// Path to `depth.tif`, or `None` if removed.
    pub depth_path: Option<PathBuf>,
    /// Path to `region.tif`, or `None` if removed.
    pub region_path: Option<PathBuf>,
    /// Path to `regions_info.csv`, or `None` if removed.
    pub csv_path: Option<PathBuf>,
    /// Path to `regions.shp`, or `None` if removed.
    pub shapefile_path: Option<PathBuf>,
    /// Path to `<stem>.gpkg` with region geometries joined to CSV attributes.
    pub gpkg_path: PathBuf,
    /// Extracted sink attributes.
    pub depressions: Vec<SinkDepression>,
}

/// In-memory outputs of [`extract_sinks_in_memory`].
#[derive(Debug, Clone)]
pub struct InMemorySinks {
    /// Completely filled DEM.
    pub dem_filled: Array2D<f64>,
    /// Raw difference (`dem_filled - dem`).
    pub dem_diff: Array2D<f64>,
    /// Depth raster of depressions exceeding `min_size` (0 outside).
    pub depth: Array2D<f64>,
    /// Sink elevation raster (original elevation inside retained depressions, 0 outside).
    pub sink: Array2D<f64>,
    /// Refined DEM with small depressions filled and large depressions kept intact.
    pub dem_refined: Array2D<f64>,
    /// Connected-component region IDs (`i32`, 0 outside).
    pub region: Array2D<i32>,
    /// Computed attributes for every retained depression region.
    pub depressions: Vec<SinkDepression>,
}

/// Writes sink depression properties to a CSV file matching `lidar/filling.py`.
pub fn write_sink_csv(depressions: &[SinkDepression], csv_path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(csv_path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "region_id,count,area,volume,avg_depth,max_depth,min_elev,max_elev,perimeter,major_axis,minor_axis,elongatedness,eccentricity,orientation,area_bbox_ratio"
    )?;
    for d in depressions {
        writeln!(
            w,
            "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            d.region_id,
            d.count,
            d.area,
            d.volume,
            d.avg_depth,
            d.max_depth,
            d.min_elev,
            d.max_elev,
            d.perimeter,
            d.major_axis,
            d.minor_axis,
            d.elongatedness,
            d.eccentricity,
            d.orientation,
            d.area_bbox_ratio,
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Extracts sinks from an in-memory DEM without writing files to disk.
pub fn extract_sinks_in_memory(
    dem: &Array2D<f64>,
    min_size: usize,
    topology: Topology,
) -> Result<InMemorySinks> {
    extract_sinks_with_filled_in_memory(dem, None, min_size, topology)
}

/// Extracts sinks in memory, optionally reusing a pre-filled DEM.
pub fn extract_sinks_with_filled_in_memory(
    dem: &Array2D<f64>,
    pre_filled: Option<&Array2D<f64>>,
    min_size: usize,
    topology: Topology,
) -> Result<InMemorySinks> {
    let w = dem.width();
    let h = dem.height();
    let n = w
        .checked_mul(h)
        .ok_or_else(|| Error::DimensionMismatch(format!("raster dimensions overflow ({w}x{h})")))?;

    let (mut dem_filled, filled_nd_opt) = match pre_filled {
        Some(f) => {
            if f.width() != w || f.height() != h {
                return Err(Error::DimensionMismatch(format!(
                    "filled DEM shape ({}x{}) does not match input DEM ({w}x{h})",
                    f.width(),
                    f.height()
                )));
            }
            for (k, (a, b)) in dem
                .geotransform
                .iter()
                .zip(f.geotransform.iter())
                .enumerate()
            {
                let tol = 1e-9 * a.abs().max(b.abs()).max(1.0);
                if !a.is_finite() || !b.is_finite() || (a - b).abs() > tol {
                    return Err(Error::DimensionMismatch(format!(
                        "filled DEM geotransform[{k}] ({b}) does not match input DEM ({a})"
                    )));
                }
            }
            if !dem.projection.is_empty()
                && !f.projection.is_empty()
                && dem.projection != f.projection
            {
                return Err(Error::InvalidArgument(
                    "filled DEM CRS/projection does not match input DEM".into(),
                ));
            }
            (f.clone(), Some(f.no_data()))
        }
        None => {
            let mut f = dem.clone();
            fill_depressions(&mut f, topology);
            (f, None)
        }
    };

    let no_data = dem.no_data();
    let gt = dem.geotransform;

    let is_missing = |v: f64| !v.is_finite() || v.rd_eq(no_data);

    // Preserve missing values in dem_filled and compute raw dem_diff.
    let mut diff_vec = vec![0.0f64; n];
    for (i, diff_cell) in diff_vec.iter_mut().enumerate() {
        let orig = dem.get_i(i);
        if is_missing(orig) {
            dem_filled.set_i(i, orig);
            *diff_cell = 0.0;
        } else {
            let filled = dem_filled.get_i(i);
            if !filled.is_finite() || filled_nd_opt.is_some_and(|fnd| filled.rd_eq(fnd)) {
                return Err(Error::InvalidArgument(format!(
                    "pre-filled DEM contains NoData/non-finite value at cell index {i} where input DEM is valid"
                )));
            }
            let d = filled - orig;
            *diff_cell = if d.is_finite() && d > 0.0 { d } else { 0.0 };
        }
    }

    let mut dem_diff = Array2D::from_vec(w, h, diff_vec)?;
    dem_diff.copy_metadata_from(dem);
    dem_diff.set_no_data(0.0);

    // 4-connected region grouping filtering out small artifact depressions (size <= min_size).
    let (labels, nb_labels) = region_group(dem_diff.data(), w, h, min_size, |v| v > 0.0);

    // Filtered depth raster.
    let mut depth_vec = vec![0.0f64; n];
    for i in 0..n {
        if labels[i] > 0 {
            depth_vec[i] = dem_diff.get_i(i);
        }
    }
    let mut depth = Array2D::from_vec(w, h, depth_vec)?;
    depth.copy_metadata_from(dem);
    depth.set_no_data(0.0);

    // Depression properties on original DEM elevations (CRS-aware metric basis).
    let (vx_m, vy_m, _, _, cell_area_m2) =
        crate::region::safe_crs_metric_basis(&dem.projection, &gt, w, h);
    let props = crate::region::region_props_affine_basis(
        &labels,
        nb_labels,
        dem.data(),
        w,
        h,
        vx_m,
        vy_m,
        cell_area_m2,
    );
    let depressions: Vec<SinkDepression> = props.iter().map(SinkDepression::from).collect();

    // Region ID raster (i32, 0 outside).
    let region_vec: Vec<i32> = labels.iter().map(|&l| l as i32).collect();
    let mut region = Array2D::from_vec(w, h, region_vec)?;
    region.copy_metadata_from(dem);
    region.set_no_data(0);

    // Sink DEM (original elevation inside retained depressions, collision-free NoData outside).
    // Preserve 0.0 as NoData whenever no retained sink cell equals 0.0 for backward compatibility.
    // When any retained sink cell has z == 0.0 (for example coastal sea-level depressions), select
    // a NoData sentinel strictly outside the retained sink elevation range.
    let mut has_zero_sink = false;
    let mut min_sink_z = f64::INFINITY;
    let mut max_sink_z = f64::NEG_INFINITY;
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl > 0 {
            let z = dem.get_i(i);
            if z == 0.0 {
                has_zero_sink = true;
            }
            if z < min_sink_z {
                min_sink_z = z;
            }
            if z > max_sink_z {
                max_sink_z = z;
            }
        }
    }
    let sink_nd = if !has_zero_sink {
        0.0
    } else if no_data.is_finite() && (no_data < min_sink_z || no_data > max_sink_z) {
        no_data
    } else if -9999.0 < min_sink_z {
        -9999.0
    } else {
        let shifted = min_sink_z - 9999.0;
        if shifted.is_finite() && shifted < min_sink_z {
            shifted
        } else if -f64::MAX < min_sink_z {
            -f64::MAX
        } else if f64::MAX > max_sink_z {
            f64::MAX
        } else {
            f64::NAN
        }
    };
    let mut sink_vec = vec![sink_nd; n];
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl > 0 {
            sink_vec[i] = dem.get_i(i);
        }
    }
    let mut sink = Array2D::from_vec(w, h, sink_vec)?;
    sink.copy_metadata_from(dem);
    sink.set_no_data(sink_nd);

    // Refined DEM: small depressions remain filled; large depressions keep original elevation.
    let mut refined_vec = dem_filled.data().to_vec();
    for i in 0..n {
        let orig = dem.get_i(i);
        if is_missing(orig) || labels[i] > 0 {
            refined_vec[i] = orig;
        }
    }
    let mut dem_refined = Array2D::from_vec(w, h, refined_vec)?;
    dem_refined.copy_metadata_from(dem);
    dem_refined.set_no_data(no_data);

    Ok(InMemorySinks {
        dem_filled,
        dem_diff,
        depth,
        sink,
        dem_refined,
        region,
        depressions,
    })
}

/// Extracts surface depressions (sinks) from a DEM file and writes all output
/// rasters, CSV statistics, shapefile, and attribute-joined GeoPackage to `out_dir`.
///
/// Mirrors `lidar.ExtractSinks` from `opengeos/lidar`.
pub fn extract_sinks(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &ExtractSinksOptions,
) -> Result<ExtractSinksResult> {
    let in_dem = in_dem.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let out_dem = out_dir.join("dem.tif");
    let out_dem_diff = out_dir.join("dem_diff.tif");
    let out_sink = out_dir.join("sink.tif");
    let out_region = out_dir.join("region.tif");
    let out_depth = out_dir.join("depth.tif");
    let out_csv_file = out_dir.join("regions_info.csv");
    let out_vec_file = out_dir.join("regions.shp");

    let stem = in_dem
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("regions");
    let out_gpkg = out_dir.join(format!("{stem}.gpkg"));

    let out_dem_filled = options
        .filled_dem
        .clone()
        .unwrap_or_else(|| out_dir.join("dem_filled.tif"));

    let mut output_paths: Vec<&Path> = vec![
        &out_dem,
        &out_dem_diff,
        &out_sink,
        &out_region,
        &out_depth,
        &out_csv_file,
        &out_vec_file,
        &out_gpkg,
    ];
    if options.filled_dem.is_none() {
        output_paths.push(&out_dem_filled);
    }
    ensure_no_path_collision(in_dem, &output_paths)?;
    if let Some(ref fpath) = options.filled_dem {
        ensure_no_path_collision(fpath, &output_paths)?;
    }

    let dem = load_raster::<f64>(in_dem, None)?;
    let pre_filled = match &options.filled_dem {
        Some(path) => Some(load_raster::<f64>(path, None)?),
        None => None,
    };

    let mem = extract_sinks_with_filled_in_memory(
        &dem,
        pre_filled.as_ref(),
        options.min_size,
        options.topology,
    )?;

    if options.filled_dem.is_none() {
        save_raster(
            &mem.dem_filled,
            &out_dem_filled,
            Some("extract_sinks: dem_filled"),
            options.compress,
        )?;
    }
    save_raster(
        &mem.dem_diff,
        &out_dem_diff,
        Some("extract_sinks: dem_diff"),
        options.compress,
    )?;
    save_raster(
        &mem.depth,
        &out_depth,
        Some("extract_sinks: depth"),
        options.compress,
    )?;
    save_raster(
        &mem.sink,
        &out_sink,
        Some("extract_sinks: sink"),
        options.compress,
    )?;
    save_raster(
        &mem.dem_refined,
        &out_dem,
        Some("extract_sinks: refined_dem"),
        options.compress,
    )?;
    save_raster(
        &mem.region,
        &out_region,
        Some("extract_sinks: region"),
        options.compress,
    )?;

    write_sink_csv(&mem.depressions, &out_csv_file)?;

    // Polygonize region raster to ESRI Shapefile and join CSV attributes into GeoPackage.
    polygonize(&out_region, &out_vec_file, "id")?;
    join_csv_to_vector(&out_vec_file, &out_csv_file, &out_gpkg, "id", "region_id")?;

    let result = if options.keep_files {
        ExtractSinksResult {
            sink_path: Some(out_sink),
            refined_dem_path: Some(out_dem),
            filled_dem_path: Some(out_dem_filled),
            dem_diff_path: Some(out_dem_diff),
            depth_path: Some(out_depth),
            region_path: Some(out_region),
            csv_path: Some(out_csv_file),
            shapefile_path: Some(out_vec_file),
            gpkg_path: out_gpkg,
            depressions: mem.depressions,
        }
    } else {
        for f in [
            &out_dem,
            &out_dem_diff,
            &out_depth,
            &out_sink,
            &out_region,
            &out_csv_file,
        ] {
            let _ = std::fs::remove_file(f);
        }
        if options.filled_dem.is_none() {
            let _ = std::fs::remove_file(&out_dem_filled);
        }
        for ext in ["shp", "dbf", "shx", "prj", "cpg"] {
            let sidecar = out_vec_file.with_extension(ext);
            let _ = std::fs::remove_file(sidecar);
        }
        ExtractSinksResult {
            sink_path: None,
            refined_dem_path: None,
            filled_dem_path: options.filled_dem.clone(),
            dem_diff_path: None,
            depth_path: None,
            region_path: None,
            csv_path: None,
            shapefile_path: None,
            gpkg_path: out_gpkg,
            depressions: mem.depressions,
        }
    };

    Ok(result)
}
