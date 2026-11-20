//! Catchment delineation, overland flow-path tracing, and catchment hierarchy.
//!
//! Pure-Rust ports of the ArcGIS Hydrology Analyst toolbox scripts from `opengeos/lidar`:
//! - `2_Delineate_Catchment.py` → [`delineate_catchments`] / [`delineate_catchments_in_memory`]
//! - `3_Flow_Path.py` → [`delineate_flow_paths`]
//! - `5_Catchment_Hierarchy.py` → [`delineate_catchment_hierarchy`]
//!
//! All flow-direction routing uses [`richdem::barnes_flat_resolution_d8`] and
//! [`richdem::fill_depressions_epsilon`], and all raster/vector I/O uses [`terrio`]
//! with zero ArcPy or GDAL dependencies.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use richdem::{
    Array2D, D8Flowdir, D8X, D8Y, DemNum, DynamicArray2D, FLOWDIR_NO_DATA, Topology,
    array2d_to_dataset, barnes_flat_resolution_d8, dataset_to_dynamic, fill_depressions_epsilon,
    load_raster, save_raster,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::SpatialRef;
use terrio::ops::polygonize::{PolygonizeOptions, polygonize as terrio_polygonize};
use terrio::ops::rasterize::{Grid, RasterizeOptions, rasterize as terrio_rasterize};
use terrio::vector::Coord;
use terrio::{
    DataType, Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer,
    VectorOpenOptions, VectorWriteOptions,
};

use crate::error::{Error, Result};
use crate::raster_ops::ensure_no_path_collision;
use crate::region::region_group;

/// Attributes of a single delineated sink catchment (from `2_Delineate_Catchment.py`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CatchmentRecord {
    /// 1-based sink / catchment ID.
    pub id: u32,
    /// Number of pixels in the sink footprint.
    pub sink_cells: usize,
    /// Area of the sink footprint in map units squared.
    pub sink_area: f64,
    /// Number of pixels in the contributing catchment (including the sink itself).
    pub catchment_cells: usize,
    /// Area of the contributing catchment in map units squared (`cat_area`).
    pub catchment_area: f64,
    /// Ratio of sink area to catchment area (`dep2catR = sink_area / cat_area`).
    pub dep_to_cat_ratio: f64,
}

/// In-memory output of [`delineate_catchments_in_memory`].
#[derive(Debug, Clone)]
pub struct InMemoryCatchments {
    /// D8 flow directions (`0..=8`, `255` = NoData).
    pub flow_directions: Array2D<D8Flowdir>,
    /// Watershed label grid (`0` = background / drains off grid, `> 0` = sink ID).
    pub watershed: Array2D<u32>,
    /// Per-catchment summary metrics.
    pub records: Vec<CatchmentRecord>,
}

/// Options controlling [`delineate_catchments`].
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DelineateCatchmentsOptions {
    /// Compress output GeoTIFFs with DEFLATE.
    pub compress: bool,
}

/// Output paths and records produced by [`delineate_catchments`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DelineateCatchmentsResult {
    /// Path to `watershed.tif`.
    pub watershed_path: PathBuf,
    /// Path to the polygonized catchment vector (`catchment.gpkg` or `.shp`).
    pub vector_path: PathBuf,
    /// Path to `catchments_info.csv`.
    pub csv_path: PathBuf,
    /// Per-catchment summary metrics.
    pub records: Vec<CatchmentRecord>,
}

/// Attributes of a depression at one hierarchy level paired with its contributing
/// catchment and zonal DEM statistics (from `5_Catchment_Hierarchy.py`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CatchmentHierarchyRecord {
    /// Hierarchy level (1 = innermost leaf level, higher = enclosing level).
    pub level: u32,
    /// 1-based depression ID within this level.
    pub level_fid: u32,
    /// Number of cells inside the depression footprint.
    pub count: usize,
    /// Depression footprint area in map units squared.
    pub area: f64,
    /// Contributing catchment area in map units squared (`cat_area`).
    pub cat_area: f64,
    /// Depression-to-catchment area ratio (`dep2catR = area / cat_area`).
    pub dep_to_cat_ratio: f64,
    /// Minimum DEM elevation within the depression.
    pub min_elev: f64,
    /// Maximum DEM elevation within the depression.
    pub max_elev: f64,
    /// Elevation range (`max_elev - min_elev`).
    pub range: f64,
    /// Mean DEM elevation within the depression.
    pub mean_elev: f64,
    /// Standard deviation of DEM elevation within the depression.
    pub std_elev: f64,
    /// Sum of DEM elevations within the depression.
    pub sum_elev: f64,
    /// Relief volume below `max_elev`: `(count * max_elev - sum_elev) * cell_area`.
    pub volume: f64,
    /// Mean depth (`volume / area`).
    pub mean_depth: f64,
}

/// Options controlling [`delineate_catchment_hierarchy`].
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CatchmentHierarchyOptions {
    /// Compress output GeoTIFFs with DEFLATE.
    pub compress: bool,
}

/// Output paths and records produced by [`delineate_catchment_hierarchy`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CatchmentHierarchyResult {
    /// Path to the output catchment hierarchy raster (e.g. `catchment_hir.tif`), where
    /// each depression cell stores its contributing catchment area (`u32`, NoData = `0`).
    pub catchment_raster_path: PathBuf,
    /// Per-level catchment polygon shapefiles (`shapefiles/Catchment_level_*.shp`).
    pub catchment_shapefiles: Vec<PathBuf>,
    /// Per-level depression shapefiles enriched with catchment & zonal stats
    /// (`shapefiles/level/Single_level_*.shp`).
    pub level_shapefiles: Vec<PathBuf>,
    /// Merged depression hierarchy shapefile (`shapefiles/Merge_levels.shp`).
    pub merged_shapefile_path: PathBuf,
    /// Summary CSV table (`catchment_hierarchy.csv`).
    pub csv_path: PathBuf,
    /// All depression-level catchment and zonal statistics records.
    pub records: Vec<CatchmentHierarchyRecord>,
}

/// Attributes of an overland flow path connecting an upstream sink to its downstream sink
/// or the DEM outlet boundary (from `3_Flow_Path.py`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FlowPathRecord {
    /// 1-based flow-path ID.
    pub id: u32,
    /// Upstream sink ID where this overland spill path begins.
    pub start_sink: u32,
    /// Downstream sink ID where this path terminates (`0` if it exits the DEM boundary).
    pub end_sink: u32,
    /// DEM elevation at the path start (spill point).
    pub start_elev: f64,
    /// DEM elevation at the path end.
    pub end_elev: f64,
    /// Elevation drop (`start_elev - end_elev`).
    pub diff_elev: f64,
    /// Overland path length in map units.
    pub length: f64,
    /// Upstream sink storage/relief volume in cubic map units.
    pub volume: f64,
    /// Upstream sink contributing catchment area in map units squared.
    pub cat_area: f64,
    /// Equivalent rainfall depth to fill the upstream sink (`volume / cat_area`).
    pub simu_depth: f64,
    /// Rainfall intensity in m/h (`rain_intensity_cm_h / 100.0`).
    pub rain_intensity_m_h: f64,
    /// Time in hours for the upstream sink to fill and spill (`simu_depth / rain_intensity_m_h`).
    pub time_inund: f64,
    /// 1-based inundation rank ordered by `time_inund` ascending.
    pub rank: u32,
}

/// Options controlling [`delineate_flow_paths`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DelineateFlowPathsOptions {
    /// Rainfall intensity in **cm/h** (default `5.0`, converted to m/h internally).
    pub rain_intensity_cm_h: f64,
    /// Optional path to the partially-filled / refined DEM (`dem.tif` from `extract_sinks`)
    /// used to compute sink relief volumes and pre-fill catchment areas. When omitted,
    /// `in_dem_filled` is used directly (and volumes default to `0.0`).
    pub refined_dem: Option<PathBuf>,
}

impl Default for DelineateFlowPathsOptions {
    fn default() -> Self {
        Self {
            rain_intensity_cm_h: 5.0,
            refined_dem: None,
        }
    }
}

/// Output paths and records produced by [`delineate_flow_paths`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DelineateFlowPathsResult {
    /// Path to the output polyline vector file (`flowpath.shp`, `.gpkg`, or `.geojson`).
    pub flowpath_vector_path: PathBuf,
    /// Path to the summary CSV table (`flowpaths_info.csv` beside `flowpath_vector_path`).
    pub csv_path: PathBuf,
    /// Per-flow-path attributes and sink connectivity.
    pub records: Vec<FlowPathRecord>,
}

fn affine_cell_area(gt: &[f64; 6]) -> f64 {
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    if det > 0.0 && det.is_finite() {
        det
    } else {
        1.0
    }
}

fn affine_cell_size(gt: &[f64; 6]) -> f64 {
    let res_x = gt[1].hypot(gt[4]);
    let res_y = gt[2].hypot(gt[5]);
    if res_x > 0.0 && res_y > 0.0 && res_x.is_finite() && res_y.is_finite() {
        0.5 * (res_x + res_y)
    } else {
        1.0
    }
}

fn cell_center_xy(gt: &[f64; 6], col: usize, row: usize) -> (f64, f64) {
    let c = col as f64 + 0.5;
    let r = row as f64 + 0.5;
    (gt[0] + gt[1] * c + gt[2] * r, gt[3] + gt[4] * c + gt[5] * r)
}

fn is_missing(dem: &Array2D<f64>, v: f64) -> bool {
    !v.is_finite() || v.rd_eq(dem.no_data())
}

fn validate_same_grid(a: &Array2D<f64>, b: &Array2D<f64>, label: &str) -> Result<()> {
    if a.width() != b.width() || a.height() != b.height() {
        return Err(Error::DimensionMismatch(format!(
            "{label} dimensions ({}x{}) do not match DEM ({}x{})",
            b.width(),
            b.height(),
            a.width(),
            a.height()
        )));
    }
    for i in 0..6 {
        let diff = (a.geotransform[i] - b.geotransform[i]).abs();
        let tol = 1e-6 * a.geotransform[i].abs().max(1.0);
        if diff > tol {
            return Err(Error::InvalidArgument(format!(
                "{label} geotransform differs from DEM at index {i}: {} vs {}",
                b.geotransform[i], a.geotransform[i]
            )));
        }
    }
    Ok(())
}

/// Computes D8 flow directions with Barnes (2014) flat resolution without altering
/// real depressions in the DEM.
pub fn d8_flow_directions_resolved(dem: &Array2D<f64>) -> Result<Array2D<D8Flowdir>> {
    let mut copy = dem.clone();
    Ok(barnes_flat_resolution_d8(&mut copy, false)?)
}

/// Traces upslope D8 watersheds draining into each nonzero pour-point / sink label.
///
/// Mirrors `arcpy.sa.Watershed(flow_direction, pour_points)` in O(N) time with
/// iterative path compression (no recursion limit).
///
/// Cells where `pour_points[idx] > 0` receive their own label. Every other cell
/// follows `flowdirs` downstream until reaching a labeled pour point (inheriting its
/// label) or exiting the grid / hitting a pit (`0`), NoData (`255`), or a cycle (`0`).
pub fn d8_watersheds(flowdirs: &Array2D<D8Flowdir>, pour_points: &[u32]) -> Result<Array2D<u32>> {
    let w = flowdirs.width();
    let h = flowdirs.height();
    let n = w * h;
    if pour_points.len() != n {
        return Err(Error::DimensionMismatch(format!(
            "pour_points length ({}) does not match flowdirs size ({w}x{h} = {n})",
            pour_points.len()
        )));
    }

    let fdata = flowdirs.data();
    let mut ws = vec![0u32; n];
    // 0 = unvisited, 1 = on current path, 2 = resolved
    let mut state = vec![0u8; n];

    for i in 0..n {
        if pour_points[i] > 0 {
            ws[i] = pour_points[i];
            state[i] = 2;
        } else if fdata[i] == FLOWDIR_NO_DATA {
            state[i] = 2;
        }
    }

    let mut path = Vec::with_capacity(256);
    for start in 0..n {
        if state[start] == 2 {
            continue;
        }
        path.clear();
        let mut cur = start;
        let resolved_label = loop {
            if state[cur] == 2 {
                break ws[cur];
            }
            if state[cur] == 1 {
                // Cycle detected; break out safely with background label 0.
                break 0;
            }
            state[cur] = 1;
            path.push(cur);

            let dir = fdata[cur];
            if dir == 0 || dir == FLOWDIR_NO_DATA || dir > 8 {
                break 0;
            }
            let x = (cur % w) as i32 + D8X[dir as usize];
            let y = (cur / w) as i32 + D8Y[dir as usize];
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                break 0;
            }
            cur = (y as usize) * w + (x as usize);
        };

        for &idx in &path {
            ws[idx] = resolved_label;
            state[idx] = 2;
        }
    }

    let mut out = Array2D::from_vec(w, h, ws)?;
    out.copy_metadata_from(flowdirs);
    out.set_no_data(0);
    Ok(out)
}

fn is_raster_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "tif" | "tiff" | "img" | "asc" | "vrt" | "bil" | "bip" | "bsq" | "sdat" | "dem"
    )
}

/// Loads a sink raster (dense 1..=N integer region IDs or elevation mask) or rasterizes
/// a sink polygon vector onto `dem`'s exact grid, returning 1-based sink labels (`0` = background).
fn load_or_rasterize_sink_labels(in_sink: &Path, dem: &Array2D<f64>) -> Result<Vec<u32>> {
    if is_raster_extension(in_sink) {
        let sink_arr = load_raster::<f64>(in_sink, None)?;
        validate_same_grid(dem, &sink_arr, "sink raster")?;
        let nd = sink_arr.no_data();
        let mut all_pos_int = true;
        let mut unique_ids: HashSet<u32> = HashSet::new();
        let mut max_id = 0u32;
        for &v in sink_arr.data() {
            if !v.is_finite() || v.rd_eq(nd) || v == 0.0 {
                continue;
            }
            if v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f64 {
                all_pos_int = false;
                break;
            }
            let id = v as u32;
            unique_ids.insert(id);
            if id > max_id {
                max_id = id;
            }
        }
        // Only treat the raster as pre-labeled region IDs when its positive values form a
        // dense 1..=N integer sequence (matching `region.tif`). Integer-valued elevation
        // rasters (e.g., elevations 100..150) fall through to 4-connected `region_group`.
        let is_dense_label_map =
            all_pos_int && !unique_ids.is_empty() && max_id as usize == unique_ids.len();
        if is_dense_label_map {
            let labels = sink_arr
                .data()
                .iter()
                .map(|&v| {
                    if !v.is_finite() || v.rd_eq(nd) || v <= 0.0 {
                        0
                    } else {
                        v as u32
                    }
                })
                .collect();
            Ok(labels)
        } else {
            // Elevation sink raster (e.g. sink.tif) or non-dense 0-background mask: group
            // valid nonzero foreground cells into 4-connected regions.
            let (labels, _) = region_group(sink_arr.data(), dem.width(), dem.height(), 1, |v| {
                v.is_finite() && !v.rd_eq(nd) && v != 0.0
            });
            Ok(labels)
        }
    } else {
        let (labels, _) = rasterize_vector_to_labels(in_sink, dem)?;
        Ok(labels)
    }
}

/// Rasterizes a polygon vector file onto `dem`'s grid with sequential 1-based feature IDs,
/// returning `(labels, layer)` so attributes can be joined back onto the vector features.
fn rasterize_vector_to_labels(in_vec: &Path, dem: &Array2D<f64>) -> Result<(Vec<u32>, Layer)> {
    let vec_str = in_vec
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", in_vec.display())))?;
    let ds = terrio::open_vector(vec_str, &VectorOpenOptions::default())?;
    let layer = ds.layers.into_iter().next().ok_or_else(|| {
        Error::InvalidArgument(format!("vector file has no layers: {}", in_vec.display()))
    })?;

    if layer.features.is_empty() {
        return Ok((vec![0u32; dem.width() * dem.height()], layer));
    }

    let burn_field = "__lidar_fid";
    let mut burn_layer = Layer::new("burn".to_string());
    burn_layer.geom_type = layer.geom_type;
    burn_layer.crs = layer.crs.clone();
    burn_layer
        .schema
        .add_field(FieldDef::new(burn_field, FieldType::Integer));

    for (idx, feat) in layer.features.iter().enumerate() {
        burn_layer.features.push(Feature {
            fid: feat.fid,
            geometry: feat.geometry.clone(),
            attributes: vec![FieldValue::Integer((idx + 1) as i64)],
        });
    }

    let dem_ds = array2d_to_dataset(dem)?;
    let opts = RasterizeOptions {
        attribute: Some(burn_field.to_string()),
        grid: Some(Grid::like(&dem_ds)),
        data_type: Some(DataType::U32),
        nodata: Some(0.0),
        ..Default::default()
    };
    let burned = terrio_rasterize(&[burn_layer], &opts)?;
    let dyn_arr = dataset_to_dynamic(burned, "rasterized")?;
    let labels = match dyn_arr {
        DynamicArray2D::U32(a) => a.into_vec(),
        other => other
            .to_f64()
            .into_vec()
            .into_iter()
            .map(|v| {
                if v > 0.0 && v.is_finite() {
                    v as u32
                } else {
                    0
                }
            })
            .collect(),
    };
    Ok((labels, layer))
}

fn polygonize_u32_with_records(
    arr: &Array2D<u32>,
    out_vector: &Path,
    layer_name: &str,
    area_by_id: &HashMap<u32, (f64, f64)>,
) -> Result<usize> {
    if let Some(parent) = out_vector.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let ds = array2d_to_dataset(arr)?;
    let opts = PolygonizeOptions {
        band: 0,
        field_name: "id".to_string(),
        layer_name: layer_name.to_string(),
        connectedness8: false,
        ignore_nodata: false,
        mask: None,
    };
    let mut layer = terrio_polygonize(&ds, &opts)?;
    layer.features.retain(|f| match f.attributes.first() {
        Some(FieldValue::Integer(i)) => *i > 0,
        Some(FieldValue::Float(v)) => v.is_finite() && *v > 0.0,
        _ => false,
    });
    layer
        .schema
        .add_field(FieldDef::new("cat_area", FieldType::Float));
    layer
        .schema
        .add_field(FieldDef::new("dep2catR", FieldType::Float));

    for feat in &mut layer.features {
        let id = match feat.attributes.first() {
            Some(FieldValue::Integer(i)) if *i > 0 => *i as u32,
            Some(FieldValue::Float(f)) if *f > 0.0 && f.is_finite() => *f as u32,
            _ => 0,
        };
        let (cat_area, ratio) = area_by_id.get(&id).copied().unwrap_or((0.0, 0.0));
        feat.attributes.push(FieldValue::Float(cat_area));
        feat.attributes.push(FieldValue::Float(ratio));
    }

    let count = layer.features.len();
    let out_str = out_vector.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!("non-UTF-8 path: {}", out_vector.display()))
    })?;
    terrio::write_vector(&[layer], out_str, &VectorWriteOptions::default())?;
    Ok(count)
}

/// Delineates contributing catchments for labeled sinks in memory.
///
/// Pure-Rust core of `2_Delineate_Catchment.py`.
pub fn delineate_catchments_in_memory(
    dem: &Array2D<f64>,
    sink_labels: &[u32],
) -> Result<InMemoryCatchments> {
    let flow_directions = d8_flow_directions_resolved(dem)?;
    let watershed = d8_watersheds(&flow_directions, sink_labels)?;
    let cell_area = affine_cell_area(&dem.geotransform);

    let mut sink_counts: BTreeMap<u32, usize> = BTreeMap::new();
    for &lbl in sink_labels {
        if lbl > 0 {
            *sink_counts.entry(lbl).or_insert(0) += 1;
        }
    }
    let mut cat_counts: BTreeMap<u32, usize> = BTreeMap::new();
    for &lbl in watershed.data() {
        if lbl > 0 {
            *cat_counts.entry(lbl).or_insert(0) += 1;
        }
    }

    let mut records = Vec::with_capacity(sink_counts.len());
    for (id, s_cells) in sink_counts {
        let c_cells = cat_counts.get(&id).copied().unwrap_or(s_cells);
        let sink_area = s_cells as f64 * cell_area;
        let catchment_area = c_cells as f64 * cell_area;
        let dep_to_cat_ratio = if catchment_area > 0.0 {
            sink_area / catchment_area
        } else {
            0.0
        };
        records.push(CatchmentRecord {
            id,
            sink_cells: s_cells,
            sink_area,
            catchment_cells: c_cells,
            catchment_area,
            dep_to_cat_ratio,
        });
    }

    Ok(InMemoryCatchments {
        flow_directions,
        watershed,
        records,
    })
}

/// Delineates contributing catchments for sinks on disk (GeoTIFF + vector + CSV).
///
/// Pure-Rust port of `2_Delineate_Catchment.py`. `in_sink` may be either a sink
/// region raster (e.g., `region.tif` or `sink.tif`) or a sink polygon vector
/// (`.shp`, `.gpkg`, `.geojson`).
pub fn delineate_catchments(
    in_dem: impl AsRef<Path>,
    in_sink: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &DelineateCatchmentsOptions,
) -> Result<DelineateCatchmentsResult> {
    let in_dem = in_dem.as_ref();
    let in_sink = in_sink.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let watershed_path = out_dir.join("watershed.tif");
    let vector_path = out_dir.join("catchment.gpkg");
    let csv_path = out_dir.join("catchments_info.csv");
    ensure_no_path_collision(in_dem, &[&watershed_path, &vector_path, &csv_path])?;
    ensure_no_path_collision(in_sink, &[&watershed_path, &vector_path, &csv_path])?;

    let dem = load_raster::<f64>(in_dem, None)?;
    let sink_labels = load_or_rasterize_sink_labels(in_sink, &dem)?;
    let mem = delineate_catchments_in_memory(&dem, &sink_labels)?;

    save_raster(
        &mem.watershed,
        &watershed_path,
        Some("delineate_catchments: watershed"),
        options.compress,
    )?;

    let area_by_id: HashMap<u32, (f64, f64)> = mem
        .records
        .iter()
        .map(|r| (r.id, (r.catchment_area, r.dep_to_cat_ratio)))
        .collect();
    polygonize_u32_with_records(&mem.watershed, &vector_path, "catchment", &area_by_id)?;
    write_catchments_csv(&mem.records, &csv_path)?;

    Ok(DelineateCatchmentsResult {
        watershed_path,
        vector_path,
        csv_path,
        records: mem.records,
    })
}

/// Writes `catchments_info.csv` summarizing delineated sink catchments.
pub fn write_catchments_csv(records: &[CatchmentRecord], path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "id,sink_cells,sink_area,cat_cells,cat_area,dep2catR")?;
    for r in records {
        writeln!(
            w,
            "{},{},{:.4},{},{:.4},{:.6}",
            r.id,
            r.sink_cells,
            r.sink_area,
            r.catchment_cells,
            r.catchment_area,
            r.dep_to_cat_ratio
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Parses the integer level number from filenames like `Single_level_1.shp` or `Single_level_02.tif`.
fn parse_single_level_number(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    let suffix = stem.strip_prefix("Single_level_")?;
    suffix.parse::<u32>().ok()
}

/// Delineates the nested catchment hierarchy across all depression levels produced by
/// [`crate::delineate_depressions`].
///
/// Pure-Rust port of `5_Catchment_Hierarchy.py`:
/// 1. Computes D8 flow directions (with Barnes flat resolution) once from `in_dem`.
/// 2. For each `Single_level_*.shp` (or `Single_level_*.tif`) in `in_level_dir`:
///    - Delineates each depression's contributing watershed (`shapefiles/Catchment_level_*.shp`),
///    - Computes `cat_area`, `dep2catR`, zonal DEM stats (`count`, `area`, `min`, `max`,
///      `range`, `mean`, `std`, `sum`), `volume`, and `mean_depth`,
///    - Writes the enriched depression shapefile (`shapefiles/level/Single_level_*.shp`).
/// 3. Merges all enriched level shapefiles into `shapefiles/Merge_levels.shp` and writes
///    `catchment_hierarchy.csv`.
/// 4. Writes `out_catchment_img` (e.g. `catchment_hir.tif`, `u32` rounded `cat_area`,
///    NoData `0`), burning from highest level down to level 1 so every depression level's
///    rim carries its own contributing catchment area for [`crate::simulate_inundation`].
pub fn delineate_catchment_hierarchy(
    in_dem: impl AsRef<Path>,
    in_level_dir: impl AsRef<Path>,
    out_catchment_img: impl AsRef<Path>,
    options: &CatchmentHierarchyOptions,
) -> Result<CatchmentHierarchyResult> {
    let in_dem = in_dem.as_ref();
    let in_level_dir = in_level_dir.as_ref();
    let out_catchment_img = out_catchment_img.as_ref();

    let out_parent = out_catchment_img
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let out_shp_dir = out_parent.join("shapefiles");
    let out_level_dir = out_shp_dir.join("level");
    std::fs::create_dir_all(&out_level_dir)?;

    let merged_shapefile_path = out_shp_dir.join("Merge_levels.shp");
    let csv_path = out_parent.join("catchment_hierarchy.csv");
    ensure_no_path_collision(
        in_dem,
        &[out_catchment_img, &merged_shapefile_path, &csv_path],
    )?;

    // Discover Single_level_*.shp (preferred, matching 5_Catchment_Hierarchy.py) or .tif.
    let mut shp_levels: Vec<(u32, PathBuf)> = Vec::new();
    let mut tif_levels: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(in_level_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(level) = parse_single_level_number(&path) else {
            continue;
        };
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("shp") {
                shp_levels.push((level, path));
            } else if ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff") {
                tif_levels.push((level, path));
            }
        }
    }
    let mut level_files = if !shp_levels.is_empty() {
        shp_levels
    } else {
        tif_levels
    };
    if level_files.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "no Single_level_*.shp or Single_level_*.tif files found in {}",
            in_level_dir.display()
        )));
    }
    level_files.sort_by_key(|(lvl, _)| *lvl);

    let dem = load_raster::<f64>(in_dem, None)?;
    let w = dem.width();
    let h = dem.height();
    let cell_area = affine_cell_area(&dem.geotransform);
    let flowdirs = d8_flow_directions_resolved(&dem)?;

    let mut catchment_shapefiles = Vec::with_capacity(level_files.len());
    let mut level_shapefiles = Vec::with_capacity(level_files.len());
    let mut all_records: Vec<CatchmentHierarchyRecord> = Vec::new();

    // Store per-level `(level, labels, cat_area_by_id)` so we can paint catchment_hir.tif
    // from highest level down to level 1 (innermost levels overwrite inner footprints while
    // outer rims retain their enclosing level's catchment area).
    let mut level_rasters: Vec<(u32, Vec<u32>, HashMap<u32, f64>)> =
        Vec::with_capacity(level_files.len());

    let mut merged_layer =
        Layer::new("Merge_levels".to_string()).with_geom_type(GeometryType::Polygon);
    if !dem.projection.is_empty()
        && let Ok(srs) = SpatialRef::parse(&dem.projection)
    {
        srs.apply_to_layer(&mut merged_layer);
    }
    for (name, ftype) in [
        ("level", FieldType::Integer),
        ("dep_id", FieldType::Integer),
        ("count", FieldType::Integer),
        ("area", FieldType::Float),
        ("cat_area", FieldType::Float),
        ("dep2catR", FieldType::Float),
        ("min", FieldType::Float),
        ("max", FieldType::Float),
        ("range", FieldType::Float),
        ("mean", FieldType::Float),
        ("std", FieldType::Float),
        ("sum", FieldType::Float),
        ("volume", FieldType::Float),
        ("mean_depth", FieldType::Float),
    ] {
        merged_layer.schema.add_field(FieldDef::new(name, ftype));
    }

    for (level, path) in &level_files {
        let stem_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Single_level");
        let cat_name = format!("{}.shp", stem_name.replacen("Single", "Catchment", 1));
        let lvl_name = format!("{stem_name}.shp");
        let out_cat_shp = out_shp_dir.join(&cat_name);
        let out_lvl_shp = out_level_dir.join(&lvl_name);

        let is_raster_level = is_raster_extension(path);
        let (labels, src_layer) = if is_raster_level {
            let r = load_raster::<f64>(path, None)?;
            validate_same_grid(&dem, &r, "level raster")?;
            let nd = r.no_data();
            let (lbls, _) = region_group(r.data(), w, h, 1, |v| {
                v.is_finite() && !v.rd_eq(nd) && v > 0.0
            });
            let mut lbl_arr = Array2D::from_vec(w, h, lbls.clone())?;
            lbl_arr.copy_metadata_from(&dem);
            lbl_arr.set_no_data(0);
            let ds = array2d_to_dataset(&lbl_arr)?;
            let lyr = terrio_polygonize(
                &ds,
                &PolygonizeOptions {
                    band: 0,
                    field_name: "id".to_string(),
                    layer_name: stem_name.to_string(),
                    connectedness8: false,
                    ignore_nodata: false,
                    mask: None,
                },
            )?;
            (lbls, lyr)
        } else {
            rasterize_vector_to_labels(path, &dem)?
        };

        let watershed = d8_watersheds(&flowdirs, &labels)?;

        // Compute per-feature catchment pixel counts and zonal DEM statistics.
        let mut cat_counts: HashMap<u32, usize> = HashMap::new();
        for &cid in watershed.data() {
            if cid > 0 {
                *cat_counts.entry(cid).or_insert(0) += 1;
            }
        }

        // Accumulators per depression label: (count, min, max, sum, sum_sq).
        let mut zonal: BTreeMap<u32, (usize, f64, f64, f64, f64)> = BTreeMap::new();
        let dem_data = dem.data();
        for (idx, &lbl) in labels.iter().enumerate() {
            if lbl == 0 {
                continue;
            }
            let z = dem_data[idx];
            if is_missing(&dem, z) {
                continue;
            }
            let entry = zonal
                .entry(lbl)
                .or_insert((0, f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0));
            entry.0 += 1;
            if z < entry.1 {
                entry.1 = z;
            }
            if z > entry.2 {
                entry.2 = z;
            }
            entry.3 += z;
            entry.4 += z * z;
        }

        let mut cat_area_by_id: HashMap<u32, f64> = HashMap::new();
        let mut area_ratio_by_id: HashMap<u32, (f64, f64)> = HashMap::new();
        let mut rec_by_id: HashMap<u32, CatchmentHierarchyRecord> = HashMap::new();

        for (fid, (count, zmin, zmax, zsum, zsum_sq)) in zonal {
            if count == 0 {
                continue;
            }
            let c_cells = cat_counts.get(&fid).copied().unwrap_or(count);
            let area = count as f64 * cell_area;
            let cat_area = c_cells as f64 * cell_area;
            let dep_to_cat_ratio = if cat_area > 0.0 { area / cat_area } else { 0.0 };
            let mean_elev = zsum / count as f64;
            let variance = ((zsum_sq / count as f64) - mean_elev * mean_elev).max(0.0);
            let std_elev = variance.sqrt();
            let range = zmax - zmin;
            let volume = ((count as f64) * zmax - zsum).max(0.0) * cell_area;
            let mean_depth = if area > 0.0 { volume / area } else { 0.0 };

            cat_area_by_id.insert(fid, cat_area);
            area_ratio_by_id.insert(fid, (cat_area, dep_to_cat_ratio));
            let rec = CatchmentHierarchyRecord {
                level: *level,
                level_fid: fid,
                count,
                area,
                cat_area,
                dep_to_cat_ratio,
                min_elev: zmin,
                max_elev: zmax,
                range,
                mean_elev,
                std_elev,
                sum_elev: zsum,
                volume,
                mean_depth,
            };
            all_records.push(rec.clone());
            rec_by_id.insert(fid, rec);
        }

        // Write Catchment_level_*.shp
        polygonize_u32_with_records(&watershed, &out_cat_shp, "catchment", &area_ratio_by_id)?;
        catchment_shapefiles.push(out_cat_shp);

        // Write enriched Single_level_*.shp and append to Merge_levels.shp
        let mut out_lvl_layer =
            Layer::new(stem_name.to_string()).with_geom_type(GeometryType::Polygon);
        out_lvl_layer.crs = src_layer.crs.clone();
        if out_lvl_layer.crs.is_none()
            && !dem.projection.is_empty()
            && let Ok(srs) = SpatialRef::parse(&dem.projection)
        {
            srs.apply_to_layer(&mut out_lvl_layer);
        }
        for f in merged_layer.schema.fields() {
            out_lvl_layer.schema.add_field(f.clone());
        }

        for (idx, feat) in src_layer.features.iter().enumerate() {
            let fid = if is_raster_level {
                match feat.attributes.first() {
                    Some(FieldValue::Integer(i)) if *i > 0 => *i as u32,
                    Some(FieldValue::Float(f)) if *f > 0.0 && f.is_finite() => *f as u32,
                    _ => continue,
                }
            } else {
                (idx + 1) as u32
            };
            let Some(rec) = rec_by_id.get(&fid) else {
                continue;
            };
            let attrs = vec![
                FieldValue::Integer(rec.level as i64),
                FieldValue::Integer(rec.level_fid as i64),
                FieldValue::Integer(rec.count as i64),
                FieldValue::Float(rec.area),
                FieldValue::Float(rec.cat_area),
                FieldValue::Float(rec.dep_to_cat_ratio),
                FieldValue::Float(rec.min_elev),
                FieldValue::Float(rec.max_elev),
                FieldValue::Float(rec.range),
                FieldValue::Float(rec.mean_elev),
                FieldValue::Float(rec.std_elev),
                FieldValue::Float(rec.sum_elev),
                FieldValue::Float(rec.volume),
                FieldValue::Float(rec.mean_depth),
            ];
            let new_feat = Feature {
                fid: feat.fid,
                geometry: feat.geometry.clone(),
                attributes: attrs,
            };
            out_lvl_layer.features.push(new_feat.clone());
            merged_layer.features.push(Feature {
                fid: merged_layer.features.len() as u64,
                geometry: new_feat.geometry,
                attributes: new_feat.attributes,
            });
        }

        let lvl_str = out_lvl_shp.to_str().ok_or_else(|| {
            Error::InvalidArgument(format!("non-UTF-8 path: {}", out_lvl_shp.display()))
        })?;
        terrio::write_vector(&[out_lvl_layer], lvl_str, &VectorWriteOptions::default())?;
        level_shapefiles.push(out_lvl_shp);

        level_rasters.push((*level, labels, cat_area_by_id));
    }

    // Write Merge_levels.shp
    let merge_str = merged_shapefile_path.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "non-UTF-8 path: {}",
            merged_shapefile_path.display()
        ))
    })?;
    terrio::write_vector(&[merged_layer], merge_str, &VectorWriteOptions::default())?;

    // Burn catchment_hir.tif from highest level down to level 1 so each depression level's
    // rim carries that level's contributing catchment area (u32, NoData = 0).
    let mut hir_data = vec![0u32; w * h];
    for (_, labels, cat_area_by_id) in level_rasters.iter().rev() {
        for (idx, &lbl) in labels.iter().enumerate() {
            if lbl > 0
                && let Some(&area) = cat_area_by_id.get(&lbl)
            {
                let v = area.round().clamp(0.0, u32::MAX as f64) as u32;
                if v > 0 {
                    hir_data[idx] = v;
                }
            }
        }
    }
    let mut hir_arr = Array2D::from_vec(w, h, hir_data)?;
    hir_arr.copy_metadata_from(&dem);
    hir_arr.set_no_data(0);
    save_raster(
        &hir_arr,
        out_catchment_img,
        Some("delineate_catchment_hierarchy"),
        options.compress,
    )?;

    write_catchment_hierarchy_csv(&all_records, &csv_path)?;

    Ok(CatchmentHierarchyResult {
        catchment_raster_path: out_catchment_img.to_path_buf(),
        catchment_shapefiles,
        level_shapefiles,
        merged_shapefile_path,
        csv_path,
        records: all_records,
    })
}

/// Delineates the nested catchment hierarchy completely in memory from a refined DEM
/// and the `depression_level` raster produced by [`crate::delineate_depressions_in_memory`].
///
/// Returns `(catchment_hir_f64, records)` with zero filesystem I/O (suitable for
/// WebAssembly / browser execution and high-throughput pipelines).
pub fn delineate_catchment_hierarchy_in_memory(
    dem: &Array2D<f64>,
    depression_level: &Array2D<i32>,
    min_size: usize,
) -> Result<(Array2D<f64>, Vec<CatchmentHierarchyRecord>)> {
    let w = dem.width();
    let h = dem.height();
    if depression_level.width() != w || depression_level.height() != h {
        return Err(Error::DimensionMismatch(format!(
            "depression_level ({}x{}) does not match DEM ({w}x{h})",
            depression_level.width(),
            depression_level.height()
        )));
    }
    let ldata = depression_level.data();
    let max_level = ldata.iter().copied().max().unwrap_or(0);
    if max_level <= 0 {
        let mut empty = Array2D::from_vec(w, h, vec![0.0; w * h])?;
        empty.copy_metadata_from(dem);
        empty.set_no_data(0.0);
        return Ok((empty, Vec::new()));
    }

    let cell_area = affine_cell_area(&dem.geotransform);
    let flowdirs = d8_flow_directions_resolved(dem)?;
    let dem_data = dem.data();
    let ldata_f64: Vec<f64> = ldata.iter().map(|&v| v as f64).collect();

    let mut all_records = Vec::new();
    let mut level_rasters: Vec<(u32, Vec<u32>, HashMap<u32, f64>)> = Vec::new();

    for level_i32 in 1..=max_level {
        let level = level_i32 as u32;
        let level_f64 = level_i32 as f64;
        // Mirror extract_levels: group cells with 0 < level_img <= level and retain
        // components whose peak level reaches `level`.
        let (raw_labels, nb_labels) =
            region_group(&ldata_f64, w, h, min_size, |v| v > 0.0 && v <= level_f64);
        if nb_labels == 0 {
            continue;
        }
        let mut max_in_comp = vec![0i32; nb_labels as usize + 1];
        for (&lbl, &lv) in raw_labels.iter().zip(ldata.iter()) {
            if lbl > 0 && lv > max_in_comp[lbl as usize] {
                max_in_comp[lbl as usize] = lv;
            }
        }
        let mut remap = vec![0u32; nb_labels as usize + 1];
        let mut next_id = 0u32;
        for id in 1..=nb_labels as usize {
            if max_in_comp[id] >= level_i32 {
                next_id += 1;
                remap[id] = next_id;
            }
        }
        if next_id == 0 {
            continue;
        }
        let labels: Vec<u32> = raw_labels.iter().map(|&l| remap[l as usize]).collect();
        let watershed = d8_watersheds(&flowdirs, &labels)?;

        let mut cat_counts: HashMap<u32, usize> = HashMap::new();
        for &cid in watershed.data() {
            if cid > 0 {
                *cat_counts.entry(cid).or_insert(0) += 1;
            }
        }
        let mut zonal: BTreeMap<u32, (usize, f64, f64, f64, f64)> = BTreeMap::new();
        for (idx, &lbl) in labels.iter().enumerate() {
            if lbl == 0 {
                continue;
            }
            let z = dem_data[idx];
            if is_missing(dem, z) {
                continue;
            }
            let e = zonal
                .entry(lbl)
                .or_insert((0, f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0));
            e.0 += 1;
            if z < e.1 {
                e.1 = z;
            }
            if z > e.2 {
                e.2 = z;
            }
            e.3 += z;
            e.4 += z * z;
        }

        let mut cat_area_by_id: HashMap<u32, f64> = HashMap::new();
        for (fid, (count, zmin, zmax, zsum, zsum_sq)) in zonal {
            if count == 0 {
                continue;
            }
            let c_cells = cat_counts.get(&fid).copied().unwrap_or(count);
            let area = count as f64 * cell_area;
            let cat_area = c_cells as f64 * cell_area;
            let dep_to_cat_ratio = if cat_area > 0.0 { area / cat_area } else { 0.0 };
            let mean_elev = zsum / count as f64;
            let variance = ((zsum_sq / count as f64) - mean_elev * mean_elev).max(0.0);
            let std_elev = variance.sqrt();
            let range = zmax - zmin;
            let volume = ((count as f64) * zmax - zsum).max(0.0) * cell_area;
            let mean_depth = if area > 0.0 { volume / area } else { 0.0 };
            cat_area_by_id.insert(fid, cat_area);
            all_records.push(CatchmentHierarchyRecord {
                level,
                level_fid: fid,
                count,
                area,
                cat_area,
                dep_to_cat_ratio,
                min_elev: zmin,
                max_elev: zmax,
                range,
                mean_elev,
                std_elev,
                sum_elev: zsum,
                volume,
                mean_depth,
            });
        }
        level_rasters.push((level, labels, cat_area_by_id));
    }

    let mut hir_data = vec![0.0f64; w * h];
    for (_, labels, cat_area_by_id) in level_rasters.iter().rev() {
        for (idx, &lbl) in labels.iter().enumerate() {
            if lbl > 0
                && let Some(&area) = cat_area_by_id.get(&lbl)
                && area > 0.0
            {
                hir_data[idx] = area;
            }
        }
    }
    let mut hir_arr = Array2D::from_vec(w, h, hir_data)?;
    hir_arr.copy_metadata_from(dem);
    hir_arr.set_no_data(0.0);
    Ok((hir_arr, all_records))
}

/// Writes `catchment_hierarchy.csv`.
pub fn write_catchment_hierarchy_csv(
    records: &[CatchmentHierarchyRecord],
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "level,dep_id,count,area,cat_area,dep2catR,min,max,range,mean,std,sum,volume,mean_depth"
    )?;
    for r in records {
        writeln!(
            w,
            "{},{},{},{:.4},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            r.level,
            r.level_fid,
            r.count,
            r.area,
            r.cat_area,
            r.dep_to_cat_ratio,
            r.min_elev,
            r.max_elev,
            r.range,
            r.mean_elev,
            r.std_elev,
            r.sum_elev,
            r.volume,
            r.mean_depth
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Delineates overland spill flow paths connecting sinks to their downstream sinks or the
/// DEM outlet boundary, and ranks sinks by time to inundation.
///
/// Pure-Rust port of `3_Flow_Path.py`.
///
/// - `in_dem_filled`: fully filled DEM (e.g., `dem_filled.tif` from [`crate::extract_sinks`]).
/// - `in_sink`: sink region raster (`region.tif` / `sink.tif`) or sink polygon vector.
/// - `out_flowpath`: output polyline vector file (`.shp`, `.gpkg`, or `.geojson`).
pub fn delineate_flow_paths(
    in_dem_filled: impl AsRef<Path>,
    in_sink: impl AsRef<Path>,
    out_flowpath: impl AsRef<Path>,
    options: &DelineateFlowPathsOptions,
) -> Result<DelineateFlowPathsResult> {
    if !options.rain_intensity_cm_h.is_finite() || options.rain_intensity_cm_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rain_intensity_cm_h must be finite and > 0, got {}",
            options.rain_intensity_cm_h
        )));
    }
    let in_dem_filled = in_dem_filled.as_ref();
    let in_sink = in_sink.as_ref();
    let out_flowpath = out_flowpath.as_ref();
    if let Some(parent) = out_flowpath.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let csv_path = out_flowpath.with_file_name("flowpaths_info.csv");
    ensure_no_path_collision(in_dem_filled, &[out_flowpath, &csv_path])?;
    ensure_no_path_collision(in_sink, &[out_flowpath, &csv_path])?;

    let mut dem_filled = load_raster::<f64>(in_dem_filled, None)?;
    let w = dem_filled.width();
    let h = dem_filled.height();
    let cell_area = affine_cell_area(&dem_filled.geotransform);
    let cell_size = affine_cell_size(&dem_filled.geotransform);
    let min_path_len = 2.0 * cell_size;
    let rain_m_h = options.rain_intensity_cm_h / 100.0;

    let sink_labels = load_or_rasterize_sink_labels(in_sink, &dem_filled)?;

    // Optional refined DEM gives true pre-fill catchment area and sink relief volume.
    let (volumes, cat_areas) = if let Some(ref ref_path) = options.refined_dem {
        let refined = load_raster::<f64>(ref_path, None)?;
        validate_same_grid(&dem_filled, &refined, "refined DEM")?;
        let cats = delineate_catchments_in_memory(&refined, &sink_labels)?;
        let cat_map: HashMap<u32, f64> = cats
            .records
            .into_iter()
            .map(|r| (r.id, r.catchment_area))
            .collect();
        let mut vol_map: HashMap<u32, f64> = HashMap::new();
        let fdata = dem_filled.data();
        let rdata = refined.data();
        for (idx, &sid) in sink_labels.iter().enumerate() {
            if sid == 0 {
                continue;
            }
            let zf = fdata[idx];
            let zr = rdata[idx];
            if !is_missing(&dem_filled, zf) && !is_missing(&refined, zr) && zf > zr {
                *vol_map.entry(sid).or_insert(0.0) += (zf - zr) * cell_area;
            }
        }
        (vol_map, cat_map)
    } else {
        (HashMap::new(), HashMap::new())
    };

    // Ensure any remaining flats or pits in dem_filled drain cleanly to the DEM boundary.
    fill_depressions_epsilon(&mut dem_filled, Topology::D8)?;
    let flowdirs = barnes_flat_resolution_d8(&mut dem_filled, false)?;
    let fdata = flowdirs.data();
    let zdata = dem_filled.data();

    // Find a representative seed cell inside each sink (nearest interior cell to centroid).
    let mut sum_rc: BTreeMap<u32, (f64, f64, usize)> = BTreeMap::new();
    let mut sink_cells: BTreeMap<u32, Vec<(usize, usize)>> = BTreeMap::new();
    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let sid = sink_labels[off + c];
            if sid > 0 {
                let e = sum_rc.entry(sid).or_insert((0.0, 0.0, 0));
                e.0 += r as f64;
                e.1 += c as f64;
                e.2 += 1;
                sink_cells.entry(sid).or_default().push((r, c));
            }
        }
    }

    let mut raw_paths: Vec<(FlowPathRecord, Vec<Coord>)> = Vec::new();
    let max_steps = w * h;
    // Reuse a single generation-stamped buffer across all sinks so tracing is O(total_path_len)
    // rather than O(num_sinks * w * h).
    let mut visited_stamp = vec![0u32; w * h];
    let mut trace_gen = 0u32;

    for (sid, cells) in sink_cells {
        trace_gen = trace_gen.wrapping_add(1);
        if trace_gen == 0 {
            visited_stamp.fill(0);
            trace_gen = 1;
        }
        let (sr, sc, cnt) = sum_rc[&sid];
        let cr = sr / cnt as f64;
        let cc = sc / cnt as f64;
        let &(start_r, start_c) = cells
            .iter()
            .min_by(|&&(r1, c1), &&(r2, c2)| {
                let d1 = (r1 as f64 - cr).powi(2) + (c1 as f64 - cc).powi(2);
                let d2 = (r2 as f64 - cr).powi(2) + (c2 as f64 - cc).powi(2);
                d1.total_cmp(&d2)
            })
            .expect("non-empty sink cells");

        let mut cur_r = start_r;
        let mut cur_c = start_c;
        let mut last_origin_rc = (start_r, start_c);
        let mut steps = 0usize;
        let mut left_origin = false;
        let mut coords: Vec<Coord> = Vec::new();
        let mut elevs: Vec<f64> = Vec::new();
        let mut end_sink = 0u32;

        while steps < max_steps {
            let idx = cur_r * w + cur_c;
            if visited_stamp[idx] == trace_gen {
                break;
            }
            visited_stamp[idx] = trace_gen;
            steps += 1;

            let lbl = sink_labels[idx];
            if !left_origin {
                if lbl == sid {
                    last_origin_rc = (cur_r, cur_c);
                } else {
                    left_origin = true;
                    // Include the origin sink's spill-rim cell so direct sink-to-sink spills
                    // (e.g., diagonally adjacent sinks) retain a valid 2-point segment.
                    let (or_r, or_c) = last_origin_rc;
                    let (ox, oy) = cell_center_xy(&dem_filled.geotransform, or_c, or_r);
                    coords.push(Coord::xy(ox, oy));
                    elevs.push(zdata[or_r * w + or_c]);

                    let (x, y) = cell_center_xy(&dem_filled.geotransform, cur_c, cur_r);
                    coords.push(Coord::xy(x, y));
                    elevs.push(zdata[idx]);
                    if lbl > 0 {
                        end_sink = lbl;
                        break;
                    }
                }
            } else {
                let (x, y) = cell_center_xy(&dem_filled.geotransform, cur_c, cur_r);
                coords.push(Coord::xy(x, y));
                elevs.push(zdata[idx]);
                if lbl > 0 && lbl != sid {
                    end_sink = lbl;
                    break;
                }
            }

            let dir = fdata[idx];
            if dir == 0 || dir == FLOWDIR_NO_DATA || dir > 8 {
                break;
            }
            let nx = cur_c as i32 + D8X[dir as usize];
            let ny = cur_r as i32 + D8Y[dir as usize];
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                break;
            }
            cur_c = nx as usize;
            cur_r = ny as usize;
        }

        if coords.len() < 2 {
            continue;
        }

        let mut length = 0.0;
        for pair in coords.windows(2) {
            length += (pair[1].x - pair[0].x).hypot(pair[1].y - pair[0].y);
        }
        // Keep direct sink-to-sink connections regardless of length; filter tiny boundary stubs.
        if end_sink == 0 && length <= min_path_len {
            continue;
        }

        let start_elev = elevs.first().copied().unwrap_or(0.0);
        let end_elev = elevs.last().copied().unwrap_or(0.0);
        let volume = volumes.get(&sid).copied().unwrap_or(0.0);
        // Report 0.0 when refined_dem was not supplied rather than misrepresenting sink footprint
        // area as contributing catchment area.
        let cat_area = cat_areas.get(&sid).copied().unwrap_or(0.0);
        let simu_depth = if cat_area > 0.0 {
            volume / cat_area
        } else {
            0.0
        };
        let time_inund = if rain_m_h > 0.0 {
            simu_depth / rain_m_h
        } else {
            0.0
        };

        raw_paths.push((
            FlowPathRecord {
                id: 0,
                start_sink: sid,
                end_sink,
                start_elev,
                end_elev,
                diff_elev: start_elev - end_elev,
                length,
                volume,
                cat_area,
                simu_depth,
                rain_intensity_m_h: rain_m_h,
                time_inund,
                rank: 0,
            },
            coords,
        ));
    }

    // Assign inundation rank (1..=N) ordered by time_inund ascending only when refined_dem
    // was supplied; otherwise leave rank = 0 (unranked) so callers aren't misled by arbitrary ranks.
    if options.refined_dem.is_some() {
        let mut order: Vec<usize> = (0..raw_paths.len()).collect();
        order.sort_by(|&a, &b| {
            raw_paths[a]
                .0
                .time_inund
                .total_cmp(&raw_paths[b].0.time_inund)
        });
        for (rank_idx, &idx) in order.iter().enumerate() {
            raw_paths[idx].0.rank = (rank_idx + 1) as u32;
        }
    }

    let mut layer = Layer::new("flowpath".to_string()).with_geom_type(GeometryType::LineString);
    if !dem_filled.projection.is_empty()
        && let Ok(srs) = SpatialRef::parse(&dem_filled.projection)
    {
        srs.apply_to_layer(&mut layer);
    }
    for (name, ftype) in [
        ("id", FieldType::Integer),
        ("start_sink", FieldType::Integer),
        ("end_sink", FieldType::Integer),
        ("StartElev", FieldType::Float),
        ("EndElev", FieldType::Float),
        ("DiffElev", FieldType::Float),
        ("Length", FieldType::Float),
        ("volume", FieldType::Float),
        ("cat_area", FieldType::Float),
        ("simu_depth", FieldType::Float),
        ("rain_inten", FieldType::Float),
        ("time_inund", FieldType::Float),
        ("rank", FieldType::Integer),
    ] {
        layer.schema.add_field(FieldDef::new(name, ftype));
    }

    let mut records = Vec::with_capacity(raw_paths.len());
    for (idx, (mut rec, coords)) in raw_paths.into_iter().enumerate() {
        rec.id = (idx + 1) as u32;
        layer.features.push(Feature {
            fid: idx as u64,
            geometry: Some(Geometry::LineString(coords)),
            attributes: vec![
                FieldValue::Integer(rec.id as i64),
                FieldValue::Integer(rec.start_sink as i64),
                FieldValue::Integer(rec.end_sink as i64),
                FieldValue::Float(rec.start_elev),
                FieldValue::Float(rec.end_elev),
                FieldValue::Float(rec.diff_elev),
                FieldValue::Float(rec.length),
                FieldValue::Float(rec.volume),
                FieldValue::Float(rec.cat_area),
                FieldValue::Float(rec.simu_depth),
                FieldValue::Float(rec.rain_intensity_m_h),
                FieldValue::Float(rec.time_inund),
                FieldValue::Integer(rec.rank as i64),
            ],
        });
        records.push(rec);
    }

    let out_str = out_flowpath.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!("non-UTF-8 path: {}", out_flowpath.display()))
    })?;
    terrio::write_vector(&[layer], out_str, &VectorWriteOptions::default())?;
    write_flowpaths_csv(&records, &csv_path)?;

    Ok(DelineateFlowPathsResult {
        flowpath_vector_path: out_flowpath.to_path_buf(),
        csv_path,
        records,
    })
}

/// Writes `flowpaths_info.csv`.
pub fn write_flowpaths_csv(records: &[FlowPathRecord], path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "id,start_sink,end_sink,start_elev,end_elev,diff_elev,length,volume,cat_area,simu_depth,rain_inten,time_inund,rank"
    )?;
    for r in records {
        writeln!(
            w,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.4},{}",
            r.id,
            r.start_sink,
            r.end_sink,
            r.start_elev,
            r.end_elev,
            r.diff_elev,
            r.length,
            r.volume,
            r.cat_area,
            r.simu_depth,
            r.rain_intensity_m_h,
            r.time_inund,
            r.rank
        )?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d8_watersheds_drains_bowl_into_sink() {
        // 7x7 funnel sloping inward toward a central 3x3 sink labeled 1.
        let mut data = vec![0.0; 49];
        let mut pour = vec![0u32; 49];
        for r in 0..7 {
            for c in 0..7 {
                let d = ((r as i32 - 3).abs().max((c as i32 - 3).abs())) as f64;
                data[r * 7 + c] = 10.0 + d * 2.0;
                if (2..=4).contains(&r) && (2..=4).contains(&c) {
                    pour[r * 7 + c] = 1;
                }
            }
        }
        let mut dem = Array2D::from_vec(7, 7, data).unwrap();
        dem.geotransform = [0.0, 2.0, 0.0, 14.0, 0.0, -2.0];
        let res = delineate_catchments_in_memory(&dem, &pour).unwrap();
        assert_eq!(res.records.len(), 1);
        let rec = &res.records[0];
        assert_eq!(rec.sink_cells, 9);
        // Inner 5x5 cells all drain inward into the 3x3 sink (outermost edge cells drain off grid).
        assert_eq!(rec.catchment_cells, 25);
        assert!((rec.sink_area - 36.0).abs() < 1e-9);
        assert!((rec.catchment_area - 100.0).abs() < 1e-9);
        assert!((rec.dep_to_cat_ratio - 0.36).abs() < 1e-9);
    }

    #[test]
    fn integer_elevation_sink_raster_groups_as_single_region() {
        let tmp = tempfile::tempdir().unwrap();
        let dem_path = tmp.path().join("dem.tif");
        let sink_path = tmp.path().join("sink.tif");
        let dem = Array2D::from_vec(5, 5, vec![110.0; 25]).unwrap();
        // Integer-valued elevations (100, 101, 102) inside one connected 3x3 sink (0 = NoData).
        let mut sink_data = vec![0.0; 25];
        for r in 1..4 {
            for c in 1..4 {
                sink_data[r * 5 + c] = 100.0 + (r - 1) as f64;
            }
        }
        let mut sink = Array2D::from_vec(5, 5, sink_data).unwrap();
        sink.set_no_data(0.0);
        save_raster(&dem, &dem_path, None, false).unwrap();
        save_raster(&sink, &sink_path, None, false).unwrap();

        let labels = load_or_rasterize_sink_labels(&sink_path, &dem).unwrap();
        let max_label = labels.iter().copied().max().unwrap_or(0);
        assert_eq!(
            max_label, 1,
            "connected integer-valued sink must produce 1 region"
        );
    }

    #[test]
    fn in_memory_catchment_hierarchy_populates_levels_and_areas() {
        // 15x15 bowl with two nested depression levels (level 1 inner 3x3, level 2 outer 7x7).
        let w = 15usize;
        let h = 15usize;
        let mut dem_data = vec![0.0; w * h];
        let mut level_data = vec![0i32; w * h];
        for r in 0..h {
            for c in 0..w {
                let d = ((r as i32 - 7).abs().max((c as i32 - 7).abs())) as f64;
                dem_data[r * w + c] = 100.0 + d;
                if (6..=8).contains(&r) && (6..=8).contains(&c) {
                    level_data[r * w + c] = 1;
                } else if (4..=10).contains(&r) && (4..=10).contains(&c) {
                    level_data[r * w + c] = 2;
                }
            }
        }
        let mut dem = Array2D::from_vec(w, h, dem_data).unwrap();
        dem.geotransform = [0.0, 2.0, 0.0, 30.0, 0.0, -2.0];
        let mut dep_level = Array2D::from_vec(w, h, level_data).unwrap();
        dep_level.copy_metadata_from(&dem);
        dep_level.set_no_data(0);

        let (hir, records) = delineate_catchment_hierarchy_in_memory(&dem, &dep_level, 1).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].level, 1);
        assert_eq!(records[1].level, 2);
        assert!(records[1].area > records[0].area);
        assert!(records[0].cat_area >= records[0].area);
        assert!(hir.get(7, 7) > 0.0);
        assert!(hir.get(4, 4) > 0.0);
        assert_eq!(hir.get(0, 0), 0.0);
    }
}
