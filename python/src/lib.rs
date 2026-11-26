//! PyO3 bindings exposing the `lidar-rs` API to Python with zero GDAL dependencies.

use std::path::{Path, PathBuf};

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use terrio::crs::SpatialRef;

use lidar::pointcloud::{
    CsfOptions, Dimension, GridMethod, GridStat, GroundClassifyOptions, GroundFilterMethod,
    LidarReadOptions, LidarToDsmOptions, LidarToDtmOptions, PmfOptions, PointCloud,
    PointWriteOptions, ReturnKind, SmrfOptions, Values,
};
use lidar::{
    AnimateInundationOptions, Array2D, ArtifactClassifierOptions, Bbox, BreachMode, BreachOptions,
    BurnCulvertsOptions, CatchmentHierarchyOptions, CatchmentHierarchyRecord, ConnectivityGraph,
    DelineateCatchmentsOptions, DelineateFlowPathsOptions, DelineateOptions, DemProduct,
    DownloadOptions, EmbankmentArtifactCandidate, ExtractSinksOptions, FederalBundleOptions,
    FsmBasinSummary, FsmMassBalance, GraphEdge, GraphNode, HillshadeOptions, InMemoryDepressions,
    InundationForcing, NestedDepression, NoaaAtlas14Table, Region, RoutingMode, RunPipelineOptions,
    SimulateInundationOptions, SinkDepression, SpillFlowPath, SynthesizeCulvertsOptions,
    TiledDelineationOptions, TnmProductItem, Topology, WbdLevel,
};

fn to_py_err(e: lidar::Error) -> PyErr {
    match e {
        lidar::Error::InvalidArgument(_) | lidar::Error::DimensionMismatch(_) => {
            PyValueError::new_err(e.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn parse_topology(topo: &str) -> PyResult<Topology> {
    match topo.to_ascii_lowercase().as_str() {
        "d8" => Ok(Topology::D8),
        "d4" => Ok(Topology::D4),
        other => Err(PyValueError::new_err(format!("Unknown topology: {other}"))),
    }
}

fn from_py_array2(arr: PyReadonlyArray2<'_, f64>, no_data: Option<f64>) -> PyResult<Array2D<f64>> {
    let view = arr.as_array();
    let (h, w) = (view.nrows(), view.ncols());
    let data: Vec<f64> = view
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| view.iter().copied().collect());
    let mut a = Array2D::from_vec(w, h, data).map_err(|e| to_py_err(e.into()))?;
    a.set_no_data(no_data.unwrap_or(f64::NAN));
    Ok(a)
}

fn to_py_array2<'py>(py: Python<'py>, arr: Array2D<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (w, h) = (arr.width(), arr.height());
    let nd = Array2::from_shape_vec((h, w), arr.into_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(nd.into_pyarray(py))
}

fn to_py_array2_i32<'py>(
    py: Python<'py>,
    arr: Array2D<i32>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let (w, h) = (arr.width(), arr.height());
    let nd = Array2::from_shape_vec((h, w), arr.into_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(nd.into_pyarray(py))
}

fn to_py_array1_f64<'py>(py: Python<'py>, data: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(data).into_pyarray(py)
}

fn to_py_array1_u8<'py>(py: Python<'py>, data: Vec<u8>) -> Bound<'py, PyArray1<u8>> {
    Array1::from_vec(data).into_pyarray(py)
}

fn to_py_array1_u16<'py>(py: Python<'py>, data: Vec<u16>) -> Bound<'py, PyArray1<u16>> {
    Array1::from_vec(data).into_pyarray(py)
}

fn to_py_array3_u8<'py>(
    py: Python<'py>,
    h: usize,
    w: usize,
    c: usize,
    data: Vec<u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let nd = Array3::from_shape_vec((h, w, c), data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(nd.into_pyarray(py))
}

fn validate_geotransform(gt: [f64; 6]) -> PyResult<[f64; 6]> {
    if !gt.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(
            "geotransform elements must all be finite",
        ));
    }
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    if !det.is_finite() || det <= 0.0 {
        return Err(PyValueError::new_err(
            "geotransform must have a non-zero determinant (pixel_w * pixel_h - rot_x * rot_y != 0)",
        ));
    }
    Ok(gt)
}

fn geotransform_to_py(gt: [f64; 6]) -> (f64, f64, f64, f64, f64, f64) {
    (gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
}

fn crs_to_py(projection: &str) -> Option<String> {
    let trimmed = projection.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn resolve_spatial_meta(
    height: usize,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<([f64; 6], Option<String>)> {
    let gt = if let Some(gt) = geotransform {
        validate_geotransform(gt)?
    } else {
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "cell_size must be finite and > 0, got {cell_size}"
            )));
        }
        [
            0.0,
            cell_size,
            0.0,
            (height as f64) * cell_size,
            0.0,
            -cell_size,
        ]
    };
    let proj = crs
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    Ok((gt, proj))
}

fn from_py_array2_spatial(
    arr: PyReadonlyArray2<'_, f64>,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Array2D<f64>> {
    let mut a = from_py_array2(arr, no_data)?;
    let (gt, proj) = resolve_spatial_meta(a.height(), cell_size, geotransform, crs)?;
    a.geotransform = gt;
    if let Some(p) = proj {
        a.projection = p;
    }
    Ok(a)
}

fn from_py_array2_i32_spatial(
    arr: PyReadonlyArray2<'_, i32>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Array2D<i32>> {
    let view = arr.as_array();
    let (h, w) = (view.nrows(), view.ncols());
    let data: Vec<i32> = view
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| view.iter().copied().collect());
    let mut a = Array2D::from_vec(w, h, data).map_err(|e| to_py_err(e.into()))?;
    a.set_no_data(0);
    let (gt, proj) = resolve_spatial_meta(a.height(), cell_size, geotransform, crs)?;
    a.geotransform = gt;
    if let Some(p) = proj {
        a.projection = p;
    }
    Ok(a)
}

fn xy_to_row_col(gt: &[f64; 6], xy: [f64; 2]) -> Option<(usize, usize)> {
    let det = gt[1] * gt[5] - gt[2] * gt[4];
    if !det.is_finite() || det.abs() <= 1e-18 {
        return None;
    }
    let dx = xy[0] - gt[0];
    let dy = xy[1] - gt[3];
    let col_f = (dx * gt[5] - dy * gt[2]) / det - 0.5;
    let row_f = (gt[1] * dy - gt[4] * dx) / det - 0.5;
    if !col_f.is_finite() || !row_f.is_finite() {
        return None;
    }
    let col = col_f.round() as i64;
    let row = row_f.round() as i64;
    if row < 0 || col < 0 {
        return None;
    }
    Some((row as usize, col as usize))
}

fn parse_breach_mode(mode: &str) -> PyResult<BreachMode> {
    mode.parse::<BreachMode>().map_err(to_py_err)
}

fn parse_routing_mode(mode: &str) -> PyResult<RoutingMode> {
    mode.parse::<RoutingMode>().map_err(to_py_err)
}

fn fsm_mass_balance_to_dict<'py>(
    py: Python<'py>,
    mb: &FsmMassBalance,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("duration_h", mb.duration_h)?;
    dict.set_item("total_antecedent_m3", mb.total_antecedent_m3)?;
    dict.set_item("total_runoff_m3", mb.total_runoff_m3)?;
    dict.set_item("total_stored_m3", mb.total_stored_m3)?;
    dict.set_item("total_infiltration_m3", mb.total_infiltration_m3)?;
    dict.set_item("total_boundary_outflow_m3", mb.total_boundary_outflow_m3)?;
    dict.set_item("mass_balance_error_m3", mb.mass_balance_error_m3)?;
    Ok(dict)
}

fn fsm_basin_to_dict<'py>(py: Python<'py>, b: &FsmBasinSummary) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("dep_id", b.dep_id)?;
    dict.set_item("parent_id", b.parent_id)?;
    dict.set_item("level", b.level)?;
    dict.set_item("capacity_m3", b.capacity_m3)?;
    dict.set_item("local_cat_area_m2", b.local_cat_area_m2)?;
    dict.set_item("local_eff_cat_area_m2", b.local_eff_cat_area_m2)?;
    dict.set_item("final_routed_cat_area_m2", b.final_routed_cat_area_m2)?;
    dict.set_item("initial_volume_m3", b.initial_volume_m3)?;
    dict.set_item("final_volume_m3", b.final_volume_m3)?;
    dict.set_item("fill_time_h", b.fill_time_h)?;
    dict.set_item("spill_to_dep_id", b.spill_to_dep_id)?;
    dict.set_item("merge_time_h", b.merge_time_h)?;
    dict.set_item("spilled_volume_m3", b.spilled_volume_m3)?;
    dict.set_item("peak_spill_rate_m3_h", b.peak_spill_rate_m3_h)?;
    dict.set_item("infiltrated_volume_m3", b.infiltrated_volume_m3)?;
    Ok(dict)
}

fn parse_ground_method(method: &str) -> PyResult<GroundFilterMethod> {
    method.parse::<GroundFilterMethod>().map_err(to_py_err)
}

fn parse_return_kinds_py(kinds: Option<Vec<String>>) -> PyResult<Option<Vec<ReturnKind>>> {
    let Some(list) = kinds else {
        return Ok(None);
    };
    list.iter()
        .map(|s| lidar::pointcloud::parse_return_kind(s).map_err(to_py_err))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn sink_dep_to_dict<'py>(py: Python<'py>, d: &SinkDepression) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("region_id", d.region_id)?;
    dict.set_item("count", d.count)?;
    dict.set_item("area", d.area)?;
    dict.set_item("volume", d.volume)?;
    dict.set_item("avg_depth", d.avg_depth)?;
    dict.set_item("max_depth", d.max_depth)?;
    dict.set_item("min_elev", d.min_elev)?;
    dict.set_item("max_elev", d.max_elev)?;
    dict.set_item("perimeter", d.perimeter)?;
    dict.set_item("major_axis", d.major_axis)?;
    dict.set_item("minor_axis", d.minor_axis)?;
    dict.set_item("elongatedness", d.elongatedness)?;
    dict.set_item("eccentricity", d.eccentricity)?;
    dict.set_item("orientation", d.orientation)?;
    dict.set_item("area_bbox_ratio", d.area_bbox_ratio)?;
    Ok(dict)
}

fn nested_dep_to_dict<'py>(py: Python<'py>, d: &NestedDepression) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", d.id)?;
    dict.set_item("level", d.level)?;
    dict.set_item("count", d.count)?;
    dict.set_item("area", d.area)?;
    dict.set_item("volume", d.volume)?;
    dict.set_item("avg_depth", d.avg_depth)?;
    dict.set_item("max_depth", d.max_depth)?;
    dict.set_item("min_elev", d.min_elev)?;
    dict.set_item("max_elev", d.max_elev)?;
    dict.set_item("children_ids", d.children_ids.clone())?;
    dict.set_item("region_id", d.region_id)?;
    dict.set_item("perimeter", d.perimeter)?;
    dict.set_item("major_axis", d.major_axis)?;
    dict.set_item("minor_axis", d.minor_axis)?;
    dict.set_item("elongatedness", d.elongatedness)?;
    dict.set_item("eccentricity", d.eccentricity)?;
    dict.set_item("orientation", d.orientation)?;
    dict.set_item("area_bbox_ratio", d.area_bbox_ratio)?;
    dict.set_item("parent_id", d.parent_id)?;
    dict.set_item("circularity", d.circularity)?;
    dict.set_item("depth_diam_ratio", d.depth_diam_ratio)?;
    dict.set_item("hypsometric_integral", d.hypsometric_integral)?;
    dict.set_item("rim_slope_deg", d.rim_slope_deg)?;
    dict.set_item("artifact_flag", d.artifact_flag)?;
    Ok(dict)
}

fn catchment_hierarchy_record_to_dict<'py>(
    py: Python<'py>,
    r: &CatchmentHierarchyRecord,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("level", r.level)?;
    d.set_item("level_fid", r.level_fid)?;
    d.set_item("count", r.count)?;
    d.set_item("area", r.area)?;
    d.set_item("cat_area", r.cat_area)?;
    d.set_item("dep_to_cat_ratio", r.dep_to_cat_ratio)?;
    d.set_item("min_elev", r.min_elev)?;
    d.set_item("max_elev", r.max_elev)?;
    d.set_item("range", r.range)?;
    d.set_item("mean_elev", r.mean_elev)?;
    d.set_item("std_elev", r.std_elev)?;
    d.set_item("sum_elev", r.sum_elev)?;
    d.set_item("volume", r.volume)?;
    d.set_item("mean_depth", r.mean_depth)?;
    Ok(d)
}

fn graph_node_to_dict<'py>(py: Python<'py>, n: &GraphNode) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", n.id)?;
    d.set_item("parent_id", n.parent_id)?;
    d.set_item("level", n.level)?;
    d.set_item("region_id", n.region_id)?;
    d.set_item("area", n.area)?;
    d.set_item("volume", n.volume)?;
    d.set_item("avg_depth", n.avg_depth)?;
    d.set_item("max_depth", n.max_depth)?;
    d.set_item("min_elev", n.min_elev)?;
    d.set_item("max_elev", n.max_elev)?;
    d.set_item("spill_elev", n.max_elev)?;
    d.set_item("circularity", n.circularity)?;
    d.set_item("hypsometric_integral", n.hypsometric_integral)?;
    d.set_item("cat_area", n.cat_area)?;
    d.set_item("fill_time_h", n.fill_time_h)?;
    d.set_item("fill_runoff_mm", n.fill_runoff_mm)?;
    Ok(d)
}

fn graph_edge_to_dict<'py>(
    py: Python<'py>,
    e: &GraphEdge,
    gt: Option<&[f64; 6]>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("from_id", e.from_id)?;
    d.set_item("to_id", e.to_id)?;
    d.set_item("edge_type", &e.edge_type)?;
    d.set_item("spill_time_h", e.spill_time_h)?;
    d.set_item("spill_runoff_mm", e.spill_runoff_mm)?;
    d.set_item("merge_time_h", e.merge_time_h)?;
    d.set_item("spill_elev", e.spill_elev)?;
    d.set_item("length_m", e.length_m)?;
    d.set_item("start_xy", e.start_xy.map(|xy| (xy[0], xy[1])))?;
    d.set_item("end_xy", e.end_xy.map(|xy| (xy[0], xy[1])))?;
    let rc = match (e.start_xy, gt) {
        (Some(xy), Some(g)) => xy_to_row_col(g, xy),
        _ => None,
    };
    d.set_item("pour_row", rc.map(|(r, _)| r))?;
    d.set_item("pour_col", rc.map(|(_, c)| c))?;
    d.set_item("dist_to_stream", e.dist_to_stream)?;
    Ok(d)
}

fn connectivity_graph_to_dict<'py>(
    py: Python<'py>,
    g: &ConnectivityGraph,
    gt: Option<&[f64; 6]>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("rain_intensity_cm_h", g.rain_intensity_cm_h)?;
    let nodes = PyList::empty(py);
    for n in &g.nodes {
        nodes.append(graph_node_to_dict(py, n)?)?;
    }
    d.set_item("nodes", nodes)?;
    let edges = PyList::empty(py);
    for e in &g.edges {
        edges.append(graph_edge_to_dict(py, e, gt)?)?;
    }
    d.set_item("edges", edges)?;
    Ok(d)
}

fn dict_get_or<'py, T>(d: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    for<'a> T: pyo3::FromPyObject<'a, 'py>,
{
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<T>().map_err(|e| {
            let py_err: PyErr = e.into();
            PyValueError::new_err(format!("invalid field '{key}': {py_err}"))
        }),
        _ => Ok(default),
    }
}

fn dict_to_nested_dep(d: &Bound<'_, PyDict>) -> PyResult<NestedDepression> {
    let id: u32 = d
        .get_item("id")?
        .ok_or_else(|| PyValueError::new_err("depression dict missing required key 'id'"))?
        .extract()?;
    Ok(NestedDepression {
        id,
        level: dict_get_or(d, "level", 1u32)?,
        count: dict_get_or(d, "count", 0usize)?,
        area: dict_get_or(d, "area", 0.0f64)?,
        volume: dict_get_or(d, "volume", 0.0f64)?,
        avg_depth: dict_get_or(d, "avg_depth", 0.0f64)?,
        max_depth: dict_get_or(d, "max_depth", 0.0f64)?,
        min_elev: dict_get_or(d, "min_elev", 0.0f64)?,
        max_elev: dict_get_or(d, "max_elev", 0.0f64)?,
        children_ids: dict_get_or(d, "children_ids", Vec::<u32>::new())?,
        region_id: dict_get_or(d, "region_id", 1u32)?,
        perimeter: dict_get_or(d, "perimeter", 0.0f64)?,
        major_axis: dict_get_or(d, "major_axis", 0.0f64)?,
        minor_axis: dict_get_or(d, "minor_axis", 0.0f64)?,
        elongatedness: dict_get_or(d, "elongatedness", 0.0f64)?,
        eccentricity: dict_get_or(d, "eccentricity", 0.0f64)?,
        orientation: dict_get_or(d, "orientation", 0.0f64)?,
        area_bbox_ratio: dict_get_or(d, "area_bbox_ratio", 0.0f64)?,
        parent_id: dict_get_or(d, "parent_id", 0u32)?,
        circularity: dict_get_or(d, "circularity", 0.0f64)?,
        depth_diam_ratio: dict_get_or(d, "depth_diam_ratio", 0.0f64)?,
        hypsometric_integral: dict_get_or(d, "hypsometric_integral", 0.0f64)?,
        rim_slope_deg: dict_get_or(d, "rim_slope_deg", 0.0f64)?,
        artifact_flag: dict_get_or(d, "artifact_flag", false)?,
    })
}

fn dict_to_catchment_record(d: &Bound<'_, PyDict>) -> PyResult<CatchmentHierarchyRecord> {
    Ok(CatchmentHierarchyRecord {
        level: dict_get_or(d, "level", 1u32)?,
        level_fid: dict_get_or(d, "level_fid", 1u32)?,
        count: dict_get_or(d, "count", 0usize)?,
        area: dict_get_or(d, "area", 0.0f64)?,
        cat_area: dict_get_or(d, "cat_area", 0.0f64)?,
        dep_to_cat_ratio: dict_get_or(d, "dep_to_cat_ratio", 0.0f64)?,
        min_elev: dict_get_or(d, "min_elev", 0.0f64)?,
        max_elev: dict_get_or(d, "max_elev", 0.0f64)?,
        range: dict_get_or(d, "range", 0.0f64)?,
        mean_elev: dict_get_or(d, "mean_elev", 0.0f64)?,
        std_elev: dict_get_or(d, "std_elev", 0.0f64)?,
        sum_elev: dict_get_or(d, "sum_elev", 0.0f64)?,
        volume: dict_get_or(d, "volume", 0.0f64)?,
        mean_depth: dict_get_or(d, "mean_depth", 0.0f64)?,
    })
}

fn dict_to_spill_path(d: &Bound<'_, PyDict>) -> PyResult<SpillFlowPath> {
    Ok(SpillFlowPath {
        from_dep_id: dict_get_or(d, "from_dep_id", 0u32)?,
        from_level: dict_get_or(d, "from_level", 1u32)?,
        to_dep_id: dict_get_or(d, "to_dep_id", 0u32)?,
        spill_time_h: dict_get_or(d, "spill_time_h", 0.0f64)?,
        merge_time_h: dict_get_or(d, "merge_time_h", None)?,
        spill_elev: dict_get_or(d, "spill_elev", 0.0f64)?,
        cells: dict_get_or(d, "cells", Vec::<(usize, usize)>::new())?,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_ground_options(
    ground_class: u8,
    cloth_resolution: Option<f64>,
    rigidness: Option<usize>,
    class_threshold: Option<f64>,
    slope_smooth: Option<bool>,
    iterations: Option<usize>,
    time_step: Option<f64>,
    max_window_size: Option<f64>,
    initial_distance: Option<f64>,
    max_distance: Option<f64>,
    cell_size: Option<f64>,
    slope: Option<f64>,
    window_radius: Option<f64>,
    elevation_threshold: Option<f64>,
    elevation_scalar: Option<f64>,
) -> PyResult<(CsfOptions, PmfOptions, SmrfOptions)> {
    let mut csf = CsfOptions {
        ground_class,
        ..Default::default()
    };
    if let Some(v) = cloth_resolution {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "cloth_resolution must be finite and > 0, got {v}"
            )));
        }
        csf.resolution = v;
    }
    if let Some(v) = rigidness {
        if !(1..=3).contains(&v) {
            return Err(PyValueError::new_err(format!(
                "rigidness must be 1, 2, or 3, got {v}"
            )));
        }
        csf.rigidness = v;
    }
    if let Some(v) = class_threshold {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "class_threshold must be finite and > 0, got {v}"
            )));
        }
        csf.threshold = v;
    }
    if let Some(v) = slope_smooth {
        csf.smooth = v;
    }
    if let Some(v) = iterations {
        if v == 0 {
            return Err(PyValueError::new_err("iterations must be > 0"));
        }
        csf.iterations = v;
    }
    if let Some(v) = time_step {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "time_step must be finite and > 0, got {v}"
            )));
        }
        csf.step = v;
    }

    let mut pmf = PmfOptions {
        ground_class,
        ..Default::default()
    };
    if let Some(v) = max_window_size {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "max_window_size must be finite and > 0, got {v}"
            )));
        }
        pmf.max_window = v;
    }
    if let Some(v) = initial_distance {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(format!(
                "initial_distance must be finite and >= 0, got {v}"
            )));
        }
        pmf.initial_distance = v;
    }
    if let Some(v) = max_distance {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(format!(
                "max_distance must be finite and >= 0, got {v}"
            )));
        }
        pmf.max_distance = v;
    }

    let mut smrf = SmrfOptions {
        ground_class,
        ..Default::default()
    };
    if let Some(v) = window_radius {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "window_radius must be finite and > 0, got {v}"
            )));
        }
        smrf.window = Some(v);
    }
    if let Some(v) = elevation_threshold {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(format!(
                "elevation_threshold must be finite and >= 0, got {v}"
            )));
        }
        smrf.threshold = v;
    }
    if let Some(v) = elevation_scalar {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(format!(
                "elevation_scalar must be finite and >= 0, got {v}"
            )));
        }
        smrf.scalar = v;
    }

    if let Some(v) = cell_size {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "cell_size must be finite and > 0, got {v}"
            )));
        }
        pmf.cell = v;
        smrf.cell = v;
    }
    if let Some(v) = slope {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(format!(
                "slope must be finite and >= 0, got {v}"
            )));
        }
        pmf.slope = v;
        smrf.slope = v;
    }

    Ok((csf, pmf, smrf))
}

fn extract_dim_f64(cloud: &PointCloud, lower: &str, upper: &str) -> PyResult<Vec<f64>> {
    if let Some(slice) = cloud.coords(lower) {
        return Ok(slice.to_vec());
    }
    cloud
        .dim(lower)
        .or_else(|| cloud.dim(upper))
        .map(|d| d.to_f64())
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "point cloud is missing required '{lower}' dimension"
            ))
        })
}

fn extract_dim_u8(cloud: &PointCloud, lower: &str, upper: &str) -> Option<Vec<u8>> {
    let dim = cloud.dim(lower).or_else(|| cloud.dim(upper))?;
    match &dim.values {
        Values::U8(v) => Some(v.clone()),
        _ => Some(
            dim.to_f64()
                .into_iter()
                .map(|x| x.round().clamp(0.0, u8::MAX as f64) as u8)
                .collect(),
        ),
    }
}

fn extract_dim_u16(cloud: &PointCloud, lower: &str, upper: &str) -> Option<Vec<u16>> {
    let dim = cloud.dim(lower).or_else(|| cloud.dim(upper))?;
    match &dim.values {
        Values::U16(v) => Some(v.clone()),
        _ => Some(
            dim.to_f64()
                .into_iter()
                .map(|x| x.round().clamp(0.0, u16::MAX as f64) as u16)
                .collect(),
        ),
    }
}

fn point_cloud_crs_str(cloud: &PointCloud) -> Option<String> {
    if cloud.crs.is_unknown() {
        None
    } else {
        cloud
            .crs
            .authority_code()
            .or_else(|| cloud.crs.to_wkt())
            .filter(|s| !s.trim().is_empty())
    }
}

fn point_cloud_to_dict<'py>(py: Python<'py>, cloud: &PointCloud) -> PyResult<Bound<'py, PyDict>> {
    let xs = extract_dim_f64(cloud, "x", "X")?;
    let ys = extract_dim_f64(cloud, "y", "Y")?;
    let zs = extract_dim_f64(cloud, "z", "Z")?;
    let count = xs.len();

    let dict = PyDict::new(py);
    dict.set_item("x", to_py_array1_f64(py, xs))?;
    dict.set_item("y", to_py_array1_f64(py, ys))?;
    dict.set_item("z", to_py_array1_f64(py, zs))?;

    match extract_dim_u8(cloud, "classification", "Classification") {
        Some(v) => dict.set_item("classification", to_py_array1_u8(py, v))?,
        None => dict.set_item("classification", py.None())?,
    }
    match extract_dim_u16(cloud, "intensity", "Intensity") {
        Some(v) => dict.set_item("intensity", to_py_array1_u16(py, v))?,
        None => dict.set_item("intensity", py.None())?,
    }
    match extract_dim_u8(cloud, "return_number", "ReturnNumber") {
        Some(v) => dict.set_item("return_number", to_py_array1_u8(py, v))?,
        None => dict.set_item("return_number", py.None())?,
    }
    match extract_dim_u8(cloud, "number_of_returns", "NumberOfReturns") {
        Some(v) => dict.set_item("number_of_returns", to_py_array1_u8(py, v))?,
        None => dict.set_item("number_of_returns", py.None())?,
    }
    dict.set_item("crs", point_cloud_crs_str(cloud))?;
    dict.set_item("count", count)?;
    Ok(dict)
}

#[allow(clippy::too_many_arguments)]
fn arrays_to_point_cloud(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    classification: Option<PyReadonlyArray1<'_, u8>>,
    intensity: Option<PyReadonlyArray1<'_, u16>>,
    return_number: Option<PyReadonlyArray1<'_, u8>>,
    number_of_returns: Option<PyReadonlyArray1<'_, u8>>,
    crs: Option<&str>,
) -> PyResult<PointCloud> {
    let xs: Vec<f64> = x.as_array().iter().copied().collect();
    let ys: Vec<f64> = y.as_array().iter().copied().collect();
    let zs: Vec<f64> = z.as_array().iter().copied().collect();
    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err(PyValueError::new_err(format!(
            "x, y, and z must have identical lengths; got {}, {}, {}",
            xs.len(),
            ys.len(),
            zs.len()
        )));
    }
    let mut dims = vec![
        Dimension::new("x", Values::F64(xs)),
        Dimension::new("y", Values::F64(ys)),
        Dimension::new("z", Values::F64(zs)),
    ];
    if let Some(arr) = classification {
        let v: Vec<u8> = arr.as_array().iter().copied().collect();
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "classification length ({}) must match point count ({n})",
                v.len()
            )));
        }
        dims.push(Dimension::new("classification", Values::U8(v)));
    }
    if let Some(arr) = intensity {
        let v: Vec<u16> = arr.as_array().iter().copied().collect();
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "intensity length ({}) must match point count ({n})",
                v.len()
            )));
        }
        dims.push(Dimension::new("intensity", Values::U16(v)));
    }
    if let Some(arr) = return_number {
        let v: Vec<u8> = arr.as_array().iter().copied().collect();
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "return_number length ({}) must match point count ({n})",
                v.len()
            )));
        }
        dims.push(Dimension::new("return_number", Values::U8(v)));
    }
    if let Some(arr) = number_of_returns {
        let v: Vec<u8> = arr.as_array().iter().copied().collect();
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "number_of_returns length ({}) must match point count ({n})",
                v.len()
            )));
        }
        dims.push(Dimension::new("number_of_returns", Values::U8(v)));
    }
    let spatial_ref = match crs.map(str::trim).filter(|s| !s.is_empty()) {
        Some(c) => SpatialRef::parse(c)
            .map_err(|e| PyValueError::new_err(format!("invalid CRS '{c}': {e}")))?,
        None => SpatialRef::default(),
    };
    Ok(PointCloud::new(dims, spatial_ref))
}

fn ground_classify_result_to_dict<'py>(
    py: Python<'py>,
    method: &str,
    ground_class: u8,
    output_path: Option<String>,
    classified: &PointCloud,
) -> PyResult<Bound<'py, PyDict>> {
    let classes = extract_dim_u8(classified, "classification", "Classification")
        .unwrap_or_else(|| vec![0u8; classified.len()]);
    let total_points = classes.len();
    let ground_points = classes.iter().filter(|&&c| c == ground_class).count();
    let non_ground_points = total_points.saturating_sub(ground_points);

    let dict = PyDict::new(py);
    dict.set_item("method", method)?;
    dict.set_item("total_points", total_points)?;
    dict.set_item("ground_points", ground_points)?;
    dict.set_item("non_ground_points", non_ground_points)?;
    dict.set_item("ground_class", ground_class)?;
    dict.set_item("output_path", output_path)?;
    dict.set_item("classification", to_py_array1_u8(py, classes))?;
    Ok(dict)
}

fn opt_path_to_py(p: Option<&PathBuf>) -> Option<String> {
    p.map(|path| path.to_string_lossy().into_owned())
}

fn run_file_filter<F>(
    in_dem: &str,
    out_file: Option<&str>,
    no_data: Option<f64>,
    history: &str,
    f: F,
) -> lidar::Result<Array2D<f64>>
where
    F: FnOnce(&Array2D<f64>) -> lidar::Result<Array2D<f64>>,
{
    let mut dem = lidar::load_raster::<f64>(in_dem, None)?;
    if let Some(nd) = no_data {
        dem.set_no_data(nd);
    }
    let out = f(&dem)?;
    if let Some(out_path) = out_file.map(Path::new) {
        if let Some(parent) = out_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        lidar::save_raster(&out, out_path, Some(history), false)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (arr, kernel_size=3, no_data=None))]
fn mean_filter_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    kernel_size: usize,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2(arr, no_data)?;
    let out = py
        .detach(move || lidar::mean_filter(&dem, kernel_size))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (arr, kernel_size=3, no_data=None))]
fn median_filter_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    kernel_size: usize,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2(arr, no_data)?;
    let out = py
        .detach(move || lidar::median_filter(&dem, kernel_size))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (arr, sigma=1.0, no_data=None))]
fn gaussian_filter_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    sigma: f64,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2(arr, no_data)?;
    let out = py
        .detach(move || lidar::gaussian_filter(&dem, sigma))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (arr, delta=100.0, no_data=None))]
fn flip_dem_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    delta: f64,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2(arr, no_data)?;
    let out = py
        .detach(move || lidar::flip_dem(&dem, delta))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, kernel_size=3, out_file=None, no_data=None))]
fn mean_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    kernel_size: usize,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || -> lidar::Result<Option<Array2D<f64>>> {
            let has_out = out_file.is_some();
            let arr = if no_data.is_none() {
                lidar::mean_filter_file(&in_dem, kernel_size, out_file.as_deref().map(Path::new))?
            } else {
                let hist = format!("mean_filter(kernel_size={kernel_size})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::mean_filter(dem, kernel_size)
                })?
            };
            Ok(if has_out { None } else { Some(arr) })
        })
        .map_err(to_py_err)?;
    out.map(|a| to_py_array2(py, a)).transpose()
}

#[pyfunction]
#[pyo3(signature = (in_dem, kernel_size=3, out_file=None, no_data=None))]
fn median_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    kernel_size: usize,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || -> lidar::Result<Option<Array2D<f64>>> {
            let has_out = out_file.is_some();
            let arr = if no_data.is_none() {
                lidar::median_filter_file(&in_dem, kernel_size, out_file.as_deref().map(Path::new))?
            } else {
                let hist = format!("median_filter(kernel_size={kernel_size})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::median_filter(dem, kernel_size)
                })?
            };
            Ok(if has_out { None } else { Some(arr) })
        })
        .map_err(to_py_err)?;
    out.map(|a| to_py_array2(py, a)).transpose()
}

#[pyfunction]
#[pyo3(signature = (in_dem, sigma=1.0, out_file=None, no_data=None))]
fn gaussian_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    sigma: f64,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || -> lidar::Result<Option<Array2D<f64>>> {
            let has_out = out_file.is_some();
            let arr = if no_data.is_none() {
                lidar::gaussian_filter_file(&in_dem, sigma, out_file.as_deref().map(Path::new))?
            } else {
                let hist = format!("gaussian_filter(sigma={sigma})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::gaussian_filter(dem, sigma)
                })?
            };
            Ok(if has_out { None } else { Some(arr) })
        })
        .map_err(to_py_err)?;
    out.map(|a| to_py_array2(py, a)).transpose()
}

#[pyfunction]
#[pyo3(signature = (in_dem, delta=100.0, out_file=None, no_data=None))]
fn flip_dem_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    delta: f64,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || -> lidar::Result<Option<Array2D<f64>>> {
            let has_out = out_file.is_some();
            let arr = if no_data.is_none() {
                lidar::flip_dem_file(&in_dem, delta, out_file.as_deref().map(Path::new))?
            } else {
                let hist = format!("flip_dem(delta={delta})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::flip_dem(dem, delta)
                })?
            };
            Ok(if has_out { None } else { Some(arr) })
        })
        .map_err(to_py_err)?;
    out.map(|a| to_py_array2(py, a)).transpose()
}

#[pyfunction]
#[pyo3(signature = (in_dem, min_size, out_dir, filled_dem=None, topology="D8", keep_files=true, compress=false))]
#[allow(clippy::too_many_arguments)]
fn extract_sinks<'py>(
    py: Python<'py>,
    in_dem: &str,
    min_size: usize,
    out_dir: &str,
    filled_dem: Option<&str>,
    topology: &str,
    keep_files: bool,
    compress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = ExtractSinksOptions {
        min_size,
        filled_dem: filled_dem.map(PathBuf::from),
        topology: parse_topology(topology)?,
        keep_files,
        compress,
    };
    let in_dem = in_dem.to_string();
    let out_dir = out_dir.to_string();
    let res = py
        .detach(move || lidar::extract_sinks(&in_dem, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("sink_path", opt_path_to_py(res.sink_path.as_ref()))?;
    dict.set_item(
        "refined_dem_path",
        opt_path_to_py(res.refined_dem_path.as_ref()),
    )?;
    dict.set_item(
        "filled_dem_path",
        opt_path_to_py(res.filled_dem_path.as_ref()),
    )?;
    dict.set_item("dem_diff_path", opt_path_to_py(res.dem_diff_path.as_ref()))?;
    dict.set_item("depth_path", opt_path_to_py(res.depth_path.as_ref()))?;
    dict.set_item("region_path", opt_path_to_py(res.region_path.as_ref()))?;
    dict.set_item("csv_path", opt_path_to_py(res.csv_path.as_ref()))?;
    dict.set_item(
        "shapefile_path",
        opt_path_to_py(res.shapefile_path.as_ref()),
    )?;
    dict.set_item("gpkg_path", res.gpkg_path.to_string_lossy().as_ref())?;

    let deps = PyList::empty(py);
    for d in &res.depressions {
        deps.append(sink_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_sink, min_size, min_depth, interval, out_dir, bool_level_shp=false, bool_comb_tif=false, compress=false, export_gpkg=false))]
#[allow(clippy::too_many_arguments)]
fn delineate_depressions<'py>(
    py: Python<'py>,
    in_sink: &str,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    out_dir: &str,
    bool_level_shp: bool,
    bool_comb_tif: bool,
    compress: bool,
    export_gpkg: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = DelineateOptions {
        min_size,
        min_depth,
        interval,
        bool_level_shp,
        bool_comb_tif,
        export_gpkg,
        compress,
    };
    let in_sink = in_sink.to_string();
    let out_dir = out_dir.to_string();
    let res = py
        .detach(move || lidar::delineate_depressions(&in_sink, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "depression_id_path",
        res.depression_id_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "depression_level_path",
        res.depression_level_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "shapefile_path",
        res.shapefile_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("gpkg_path", opt_path_to_py(res.gpkg_path.as_ref()))?;
    dict.set_item("csv_path", res.csv_path.to_string_lossy().as_ref())?;
    let level_shps: Vec<String> = res
        .level_shapefiles
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    dict.set_item("level_shapefiles", level_shps)?;
    dict.set_item("max_level", res.max_level)?;
    dict.set_item("num_regions", res.num_regions)?;

    let deps = PyList::empty(py);
    for d in &res.depressions {
        deps.append(nested_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem, min_size, min_height, interval, out_dir, bool_shp=false, compress=false, export_gpkg=false, bool_comb_tif=false))]
#[allow(clippy::too_many_arguments)]
fn delineate_mounts<'py>(
    py: Python<'py>,
    in_dem: &str,
    min_size: usize,
    min_height: f64,
    interval: f64,
    out_dir: &str,
    bool_shp: bool,
    compress: bool,
    export_gpkg: bool,
    bool_comb_tif: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = DelineateOptions {
        min_size,
        min_depth: min_height,
        interval,
        bool_level_shp: bool_shp,
        bool_comb_tif,
        export_gpkg,
        compress,
    };
    let in_dem = in_dem.to_string();
    let out_dir = out_dir.to_string();
    let res = py
        .detach(move || lidar::delineate_mounts(&in_dem, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "flipped_dem_path",
        res.flipped_dem_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "depression_id_path",
        res.mounts.depression_id_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "depression_level_path",
        res.mounts.depression_level_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "shapefile_path",
        res.mounts.shapefile_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("gpkg_path", opt_path_to_py(res.mounts.gpkg_path.as_ref()))?;
    dict.set_item("csv_path", res.mounts.csv_path.to_string_lossy().as_ref())?;
    let level_shps: Vec<String> = res
        .mounts
        .level_shapefiles
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    dict.set_item("level_shapefiles", level_shps)?;
    dict.set_item("max_level", res.mounts.max_level)?;
    dict.set_item("num_regions", res.mounts.num_regions)?;

    let deps = PyList::empty(py);
    for d in &res.mounts.depressions {
        deps.append(nested_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_raster, out_vector, field_name="id"))]
fn polygonize(
    py: Python<'_>,
    in_raster: &str,
    out_vector: &str,
    field_name: &str,
) -> PyResult<usize> {
    let in_raster = in_raster.to_string();
    let out_vector = out_vector.to_string();
    let field_name = field_name.to_string();
    py.detach(move || lidar::polygonize(&in_raster, &out_vector, &field_name))
        .map_err(to_py_err)
}

#[pyfunction]
fn mosaic(py: Python<'_>, inputs: Vec<String>, output: &str) -> PyResult<()> {
    let output = output.to_string();
    py.detach(move || lidar::mosaic(&inputs, &output))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, output, dst_crs="EPSG:4326", resampling="nearest", resolution=None))]
fn reproject_image(
    py: Python<'_>,
    input: &str,
    output: &str,
    dst_crs: &str,
    resampling: &str,
    resolution: Option<(f64, f64)>,
) -> PyResult<()> {
    let r = lidar::parse_resampling(resampling).map_err(to_py_err)?;
    let input = input.to_string();
    let output = output.to_string();
    let dst_crs = dst_crs.to_string();
    py.detach(move || lidar::reproject_image(&input, &output, &dst_crs, r, resolution))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, output, resolution, resampling="bilinear"))]
fn resample(
    py: Python<'_>,
    input: &str,
    output: &str,
    resolution: f64,
    resampling: &str,
) -> PyResult<()> {
    let r = lidar::parse_resampling(resampling).map_err(to_py_err)?;
    let input = input.to_string();
    let output = output.to_string();
    py.detach(move || lidar::resample(&input, &output, resolution, r))
        .map_err(to_py_err)
}

#[pyfunction]
fn clip_image(py: Python<'_>, input: &str, mask: &str, output: &str) -> PyResult<()> {
    let input = input.to_string();
    let mask = mask.to_string();
    let output = output.to_string();
    py.detach(move || lidar::clip_image(&input, &mask, &output))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, output, compression="DEFLATE"))]
fn image_to_cog(py: Python<'_>, input: &str, output: &str, compression: &str) -> PyResult<()> {
    let input = input.to_string();
    let output = output.to_string();
    let compression = compression.to_string();
    py.detach(move || lidar::image_to_cog(&input, &output, Some(&compression)))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, crs, output=None))]
fn add_crs(py: Python<'_>, input: &str, crs: &str, output: Option<&str>) -> PyResult<()> {
    let input = input.to_string();
    let crs = crs.to_string();
    let output = output.map(str::to_string);
    py.detach(move || lidar::add_crs(&input, &crs, output.as_deref().map(Path::new)))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_vector, in_csv, out_vector, vec_col="id", csv_col="region_id"))]
fn join_tables(
    py: Python<'_>,
    in_vector: &str,
    in_csv: &str,
    out_vector: &str,
    vec_col: &str,
    csv_col: &str,
) -> PyResult<usize> {
    let in_vector = in_vector.to_string();
    let in_csv = in_csv.to_string();
    let out_vector = out_vector.to_string();
    let vec_col = vec_col.to_string();
    let csv_col = csv_col.to_string();
    py.detach(move || {
        lidar::join_csv_to_vector(&in_vector, &in_csv, &out_vector, &vec_col, &csv_col)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_csv, out_vector, latitude="latitude", longitude="longitude"))]
fn csv_to_vector(
    py: Python<'_>,
    in_csv: &str,
    out_vector: &str,
    latitude: &str,
    longitude: &str,
) -> PyResult<usize> {
    let in_csv = in_csv.to_string();
    let out_vector = out_vector.to_string();
    let latitude = latitude.to_string();
    let longitude = longitude.to_string();
    py.detach(move || lidar::csv_points_to_vector(&in_csv, &out_vector, &latitude, &longitude))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (source, destination, point_format_id=None, minor_version=None))]
fn convert_lidar(
    py: Python<'_>,
    source: &str,
    destination: &str,
    point_format_id: Option<u8>,
    minor_version: Option<u8>,
) -> PyResult<usize> {
    let source = source.to_string();
    let destination = destination.to_string();
    py.detach(move || lidar::convert_lidar(&source, &destination, point_format_id, minor_version))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_lidar, out_dsm, resolution=1.0, radius=None, min_z=None, max_z=None, stat="max", method="statistics", window_size=0, classifications=None, return_kinds=None))]
#[allow(clippy::too_many_arguments)]
fn lidar_to_dsm(
    py: Python<'_>,
    in_lidar: &str,
    out_dsm: &str,
    resolution: f64,
    radius: Option<f64>,
    min_z: Option<f64>,
    max_z: Option<f64>,
    stat: &str,
    method: &str,
    window_size: usize,
    classifications: Option<Vec<u8>>,
    return_kinds: Option<Vec<String>>,
) -> PyResult<(usize, usize)> {
    let opts = LidarToDsmOptions {
        resolution,
        radius,
        min_z,
        max_z,
        classifications,
        return_kinds: parse_return_kinds_py(return_kinds)?,
        stat: GridStat::parse(stat).map_err(|e| to_py_err(e.into()))?,
        method: GridMethod::parse(method).map_err(|e| to_py_err(e.into()))?,
        window_size,
        ..Default::default()
    };
    let in_lidar = in_lidar.to_string();
    let out_dsm = out_dsm.to_string();
    py.detach(move || lidar::lidar_to_dsm(&in_lidar, &out_dsm, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_lidar, out_dtm, resolution=1.0, ground_method="csf", ground_class=2, radius=None, method="linear", stat="idw", window_size=5, cloth_resolution=None, rigidness=None, class_threshold=None, slope_smooth=None, iterations=None, time_step=None, max_window_size=None, initial_distance=None, max_distance=None, cell_size=None, slope=None, window_radius=None, elevation_threshold=None, elevation_scalar=None))]
#[allow(clippy::too_many_arguments)]
fn lidar_to_dtm(
    py: Python<'_>,
    in_lidar: &str,
    out_dtm: &str,
    resolution: f64,
    ground_method: &str,
    ground_class: u8,
    radius: Option<f64>,
    method: &str,
    stat: &str,
    window_size: usize,
    cloth_resolution: Option<f64>,
    rigidness: Option<usize>,
    class_threshold: Option<f64>,
    slope_smooth: Option<bool>,
    iterations: Option<usize>,
    time_step: Option<f64>,
    max_window_size: Option<f64>,
    initial_distance: Option<f64>,
    max_distance: Option<f64>,
    cell_size: Option<f64>,
    slope: Option<f64>,
    window_radius: Option<f64>,
    elevation_threshold: Option<f64>,
    elevation_scalar: Option<f64>,
) -> PyResult<(usize, usize)> {
    let (csf, pmf, smrf) = build_ground_options(
        ground_class,
        cloth_resolution,
        rigidness,
        class_threshold,
        slope_smooth,
        iterations,
        time_step,
        max_window_size,
        initial_distance,
        max_distance,
        cell_size,
        slope,
        window_radius,
        elevation_threshold,
        elevation_scalar,
    )?;
    let opts = LidarToDtmOptions {
        resolution,
        ground_method: parse_ground_method(ground_method)?,
        ground_class,
        radius,
        method: GridMethod::parse(method).map_err(|e| to_py_err(e.into()))?,
        stat: GridStat::parse(stat).map_err(|e| to_py_err(e.into()))?,
        window_size,
        csf,
        pmf,
        smrf,
        ..Default::default()
    };
    let in_lidar = in_lidar.to_string();
    let out_dtm = out_dtm.to_string();
    py.detach(move || lidar::lidar_to_dtm(&in_lidar, &out_dtm, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_lidar, out_chm, resolution=1.0, ground_method="csf", ground_class=2, radius=None, window_size=5, cloth_resolution=None, rigidness=None, class_threshold=None, slope_smooth=None, iterations=None, time_step=None, max_window_size=None, initial_distance=None, max_distance=None, cell_size=None, slope=None, window_radius=None, elevation_threshold=None, elevation_scalar=None))]
#[allow(clippy::too_many_arguments)]
fn lidar_to_chm(
    py: Python<'_>,
    in_lidar: &str,
    out_chm: &str,
    resolution: f64,
    ground_method: &str,
    ground_class: u8,
    radius: Option<f64>,
    window_size: usize,
    cloth_resolution: Option<f64>,
    rigidness: Option<usize>,
    class_threshold: Option<f64>,
    slope_smooth: Option<bool>,
    iterations: Option<usize>,
    time_step: Option<f64>,
    max_window_size: Option<f64>,
    initial_distance: Option<f64>,
    max_distance: Option<f64>,
    cell_size: Option<f64>,
    slope: Option<f64>,
    window_radius: Option<f64>,
    elevation_threshold: Option<f64>,
    elevation_scalar: Option<f64>,
) -> PyResult<(usize, usize)> {
    let (csf, pmf, smrf) = build_ground_options(
        ground_class,
        cloth_resolution,
        rigidness,
        class_threshold,
        slope_smooth,
        iterations,
        time_step,
        max_window_size,
        initial_distance,
        max_distance,
        cell_size,
        slope,
        window_radius,
        elevation_threshold,
        elevation_scalar,
    )?;
    let opts = LidarToDtmOptions {
        resolution,
        ground_method: parse_ground_method(ground_method)?,
        ground_class,
        radius,
        window_size,
        csf,
        pmf,
        smrf,
        ..Default::default()
    };
    let in_lidar = in_lidar.to_string();
    let out_chm = out_chm.to_string();
    py.detach(move || lidar::lidar_to_chm(&in_lidar, &out_chm, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (path, bounds=None, polygon_path=None, max_points=None, min_z=None, max_z=None, classifications=None, return_kinds=None, max_level=None, resolution=None))]
#[allow(clippy::too_many_arguments)]
fn read_lidar<'py>(
    py: Python<'py>,
    path: &str,
    bounds: Option<[f64; 4]>,
    polygon_path: Option<&str>,
    max_points: Option<usize>,
    min_z: Option<f64>,
    max_z: Option<f64>,
    classifications: Option<Vec<u8>>,
    return_kinds: Option<Vec<String>>,
    max_level: Option<u32>,
    resolution: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = LidarReadOptions {
        bounds,
        max_level,
        resolution,
        max_points,
        classifications: classifications.unwrap_or_default(),
        return_kinds: parse_return_kinds_py(return_kinds)?.unwrap_or_default(),
        min_z,
        max_z,
        polygon_path: polygon_path.map(PathBuf::from),
    };
    let path_s = path.to_string();
    let cloud = py
        .detach(move || lidar::read_lidar_windowed(&path_s, &opts))
        .map_err(to_py_err)?;
    point_cloud_to_dict(py, &cloud)
}

#[pyfunction]
#[pyo3(signature = (path, x, y, z, classification=None, intensity=None, return_number=None, number_of_returns=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn write_lidar(
    py: Python<'_>,
    path: &str,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    classification: Option<PyReadonlyArray1<'_, u8>>,
    intensity: Option<PyReadonlyArray1<'_, u16>>,
    return_number: Option<PyReadonlyArray1<'_, u8>>,
    number_of_returns: Option<PyReadonlyArray1<'_, u8>>,
    crs: Option<&str>,
) -> PyResult<usize> {
    let cloud = arrays_to_point_cloud(
        x,
        y,
        z,
        classification,
        intensity,
        return_number,
        number_of_returns,
        crs,
    )?;
    let n = cloud.len();
    let path_s = path.to_string();
    py.detach(move || lidar::write_lidar(&cloud, &path_s, &PointWriteOptions::default()))
        .map_err(to_py_err)?;
    Ok(n)
}

#[pyfunction]
#[pyo3(signature = (in_lidar, out_lidar=None, method="csf", ground_class=2, cloth_resolution=None, rigidness=None, class_threshold=None, slope_smooth=None, iterations=None, time_step=None, max_window_size=None, initial_distance=None, max_distance=None, cell_size=None, slope=None, window_radius=None, elevation_threshold=None, elevation_scalar=None))]
#[allow(clippy::too_many_arguments)]
fn classify_ground_file<'py>(
    py: Python<'py>,
    in_lidar: &str,
    out_lidar: Option<&str>,
    method: &str,
    ground_class: u8,
    cloth_resolution: Option<f64>,
    rigidness: Option<usize>,
    class_threshold: Option<f64>,
    slope_smooth: Option<bool>,
    iterations: Option<usize>,
    time_step: Option<f64>,
    max_window_size: Option<f64>,
    initial_distance: Option<f64>,
    max_distance: Option<f64>,
    cell_size: Option<f64>,
    slope: Option<f64>,
    window_radius: Option<f64>,
    elevation_threshold: Option<f64>,
    elevation_scalar: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let parsed_method = parse_ground_method(method)?;
    if parsed_method == GroundFilterMethod::Preclassified {
        return Err(PyValueError::new_err(
            "classify_ground method must be 'csf', 'pmf', or 'smrf'",
        ));
    }
    let method_norm = match parsed_method {
        GroundFilterMethod::Csf => "csf",
        GroundFilterMethod::Pmf => "pmf",
        GroundFilterMethod::Smrf => "smrf",
        GroundFilterMethod::Preclassified => unreachable!(),
    };
    let (csf, pmf, smrf) = build_ground_options(
        ground_class,
        cloth_resolution,
        rigidness,
        class_threshold,
        slope_smooth,
        iterations,
        time_step,
        max_window_size,
        initial_distance,
        max_distance,
        cell_size,
        slope,
        window_radius,
        elevation_threshold,
        elevation_scalar,
    )?;
    let opts = GroundClassifyOptions {
        method: parsed_method,
        csf,
        pmf,
        smrf,
    };
    let in_path = in_lidar.to_string();
    let out_path = out_lidar.map(str::to_string);
    let out_ret = out_path.clone();
    let classified = py
        .detach(move || -> lidar::Result<PointCloud> {
            let cloud = lidar::read_lidar(&in_path)?;
            let out_cloud = lidar::classify_ground_with_options(&cloud, &opts)?;
            if let Some(ref dst) = out_path {
                lidar::write_lidar(&out_cloud, dst, &PointWriteOptions::default())?;
            }
            Ok(out_cloud)
        })
        .map_err(to_py_err)?;
    ground_classify_result_to_dict(py, method_norm, ground_class, out_ret, &classified)
}

#[pyfunction]
#[pyo3(signature = (x, y, z, classification=None, intensity=None, return_number=None, number_of_returns=None, crs=None, out_lidar=None, method="csf", ground_class=2, cloth_resolution=None, rigidness=None, class_threshold=None, slope_smooth=None, iterations=None, time_step=None, max_window_size=None, initial_distance=None, max_distance=None, cell_size=None, slope=None, window_radius=None, elevation_threshold=None, elevation_scalar=None))]
#[allow(clippy::too_many_arguments)]
fn classify_ground_array<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    classification: Option<PyReadonlyArray1<'_, u8>>,
    intensity: Option<PyReadonlyArray1<'_, u16>>,
    return_number: Option<PyReadonlyArray1<'_, u8>>,
    number_of_returns: Option<PyReadonlyArray1<'_, u8>>,
    crs: Option<&str>,
    out_lidar: Option<&str>,
    method: &str,
    ground_class: u8,
    cloth_resolution: Option<f64>,
    rigidness: Option<usize>,
    class_threshold: Option<f64>,
    slope_smooth: Option<bool>,
    iterations: Option<usize>,
    time_step: Option<f64>,
    max_window_size: Option<f64>,
    initial_distance: Option<f64>,
    max_distance: Option<f64>,
    cell_size: Option<f64>,
    slope: Option<f64>,
    window_radius: Option<f64>,
    elevation_threshold: Option<f64>,
    elevation_scalar: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let parsed_method = parse_ground_method(method)?;
    if parsed_method == GroundFilterMethod::Preclassified {
        return Err(PyValueError::new_err(
            "classify_ground method must be 'csf', 'pmf', or 'smrf'",
        ));
    }
    let method_norm = match parsed_method {
        GroundFilterMethod::Csf => "csf",
        GroundFilterMethod::Pmf => "pmf",
        GroundFilterMethod::Smrf => "smrf",
        GroundFilterMethod::Preclassified => unreachable!(),
    };
    let (csf, pmf, smrf) = build_ground_options(
        ground_class,
        cloth_resolution,
        rigidness,
        class_threshold,
        slope_smooth,
        iterations,
        time_step,
        max_window_size,
        initial_distance,
        max_distance,
        cell_size,
        slope,
        window_radius,
        elevation_threshold,
        elevation_scalar,
    )?;
    let opts = GroundClassifyOptions {
        method: parsed_method,
        csf,
        pmf,
        smrf,
    };
    let cloud = arrays_to_point_cloud(
        x,
        y,
        z,
        classification,
        intensity,
        return_number,
        number_of_returns,
        crs,
    )?;
    let out_path = out_lidar.map(str::to_string);
    let out_ret = out_path.clone();
    let classified = py
        .detach(move || -> lidar::Result<PointCloud> {
            let out_cloud = lidar::classify_ground_with_options(&cloud, &opts)?;
            if let Some(ref dst) = out_path {
                lidar::write_lidar(&out_cloud, dst, &PointWriteOptions::default())?;
            }
            Ok(out_cloud)
        })
        .map_err(to_py_err)?;
    ground_classify_result_to_dict(py, method_norm, ground_class, out_ret, &classified)
}

#[pyfunction]
#[pyo3(signature = (in_dem, in_sink, out_dir, compress=false))]
fn delineate_catchments<'py>(
    py: Python<'py>,
    in_dem: &str,
    in_sink: &str,
    out_dir: &str,
    compress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let in_dem = in_dem.to_string();
    let in_sink = in_sink.to_string();
    let out_dir = out_dir.to_string();
    let opts = DelineateCatchmentsOptions { compress };
    let res = py
        .detach(move || lidar::delineate_catchments(&in_dem, &in_sink, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "watershed_path",
        res.watershed_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("vector_path", res.vector_path.to_string_lossy().as_ref())?;
    dict.set_item("csv_path", res.csv_path.to_string_lossy().as_ref())?;
    let recs = PyList::empty(py);
    for r in &res.records {
        let d = PyDict::new(py);
        d.set_item("id", r.id)?;
        d.set_item("sink_cells", r.sink_cells)?;
        d.set_item("sink_area", r.sink_area)?;
        d.set_item("catchment_cells", r.catchment_cells)?;
        d.set_item("catchment_area", r.catchment_area)?;
        d.set_item("dep_to_cat_ratio", r.dep_to_cat_ratio)?;
        recs.append(d)?;
    }
    dict.set_item("records", recs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem_filled, in_sink, out_flowpath, rain_intensity_cm_h=5.0, refined_dem=None))]
fn delineate_flow_paths<'py>(
    py: Python<'py>,
    in_dem_filled: &str,
    in_sink: &str,
    out_flowpath: &str,
    rain_intensity_cm_h: f64,
    refined_dem: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let in_dem_filled = in_dem_filled.to_string();
    let in_sink = in_sink.to_string();
    let out_flowpath = out_flowpath.to_string();
    let opts = DelineateFlowPathsOptions {
        rain_intensity_cm_h,
        refined_dem: refined_dem.map(PathBuf::from),
    };
    let res = py
        .detach(move || lidar::delineate_flow_paths(&in_dem_filled, &in_sink, &out_flowpath, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "flowpath_vector_path",
        res.flowpath_vector_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("csv_path", res.csv_path.to_string_lossy().as_ref())?;
    let recs = PyList::empty(py);
    for r in &res.records {
        let d = PyDict::new(py);
        d.set_item("id", r.id)?;
        d.set_item("start_sink", r.start_sink)?;
        d.set_item("end_sink", r.end_sink)?;
        d.set_item("start_elev", r.start_elev)?;
        d.set_item("end_elev", r.end_elev)?;
        d.set_item("diff_elev", r.diff_elev)?;
        d.set_item("length", r.length)?;
        d.set_item("volume", r.volume)?;
        d.set_item("cat_area", r.cat_area)?;
        d.set_item("simu_depth", r.simu_depth)?;
        d.set_item("rain_intensity_m_h", r.rain_intensity_m_h)?;
        d.set_item("time_inund", r.time_inund)?;
        d.set_item("rank", r.rank)?;
        recs.append(d)?;
    }
    dict.set_item("records", recs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem, in_level_dir, out_catchment_img, compress=false))]
fn delineate_catchment_hierarchy<'py>(
    py: Python<'py>,
    in_dem: &str,
    in_level_dir: &str,
    out_catchment_img: &str,
    compress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let in_dem = in_dem.to_string();
    let in_level_dir = in_level_dir.to_string();
    let out_catchment_img = out_catchment_img.to_string();
    let opts = CatchmentHierarchyOptions { compress };
    let res = py
        .detach(move || {
            lidar::delineate_catchment_hierarchy(&in_dem, &in_level_dir, &out_catchment_img, &opts)
        })
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "catchment_raster_path",
        res.catchment_raster_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "merged_shapefile_path",
        res.merged_shapefile_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("csv_path", res.csv_path.to_string_lossy().as_ref())?;
    let cat_shps: Vec<String> = res
        .catchment_shapefiles
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let lvl_shps: Vec<String> = res
        .level_shapefiles
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    dict.set_item("catchment_shapefiles", cat_shps)?;
    dict.set_item("level_shapefiles", lvl_shps)?;
    let recs = PyList::empty(py);
    for r in &res.records {
        recs.append(catchment_hierarchy_record_to_dict(py, r)?)?;
    }
    dict.set_item("records", recs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_sink, in_catchment, out_dir, min_size=1000, min_depth=0.2, interval=0.2, rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0, compress=false))]
#[allow(clippy::too_many_arguments)]
fn simulate_inundation<'py>(
    py: Python<'py>,
    in_sink: &str,
    in_catchment: &str,
    out_dir: &str,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    rain_intensity_cm_h: f64,
    rain_duration_h: f64,
    time_step_h: f64,
    compress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let in_sink = in_sink.to_string();
    let in_catchment = in_catchment.to_string();
    let out_dir = out_dir.to_string();
    let opts = SimulateInundationOptions {
        min_size,
        min_depth,
        interval,
        rain_intensity_cm_h,
        rain_duration_h,
        time_step_h,
        compress,
    };
    let res = py
        .detach(move || lidar::simulate_inundation(&in_sink, &in_catchment, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    let frames: Vec<String> = res
        .frame_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    dict.set_item("frame_paths", frames)?;
    dict.set_item(
        "fill_time_path",
        res.fill_time_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "summary_csv_path",
        res.summary_csv_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("num_regions", res.num_regions)?;
    let steps = PyList::empty(py);
    for s in &res.steps {
        let d = PyDict::new(py);
        d.set_item("step", s.step)?;
        d.set_item("time_hours", s.time_hours)?;
        d.set_item("inundated_cells", s.inundated_cells)?;
        d.set_item("inundated_area", s.inundated_area)?;
        d.set_item("inundated_fraction", s.inundated_fraction)?;
        d.set_item("raster_path", s.raster_path.to_string_lossy().as_ref())?;
        steps.append(d)?;
    }
    dict.set_item("steps", steps)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem, in_flood_dir, out_gif, frame_delay_ms=200, ping_pong=true, loops=0, water_alpha=0.55, max_dimension=None, time_step_h=Some(1.0), rain_intensity_cm_h=None, azimuth_deg=135.0, elevation_deg=45.0, show_scale_bar=false, show_legend=false))]
#[allow(clippy::too_many_arguments)]
fn animate_inundation<'py>(
    py: Python<'py>,
    in_dem: &str,
    in_flood_dir: &str,
    out_gif: &str,
    frame_delay_ms: u16,
    ping_pong: bool,
    loops: u16,
    water_alpha: f32,
    max_dimension: Option<usize>,
    time_step_h: Option<f64>,
    rain_intensity_cm_h: Option<f64>,
    azimuth_deg: f64,
    elevation_deg: f64,
    show_scale_bar: bool,
    show_legend: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let in_dem = in_dem.to_string();
    let in_flood_dir = in_flood_dir.to_string();
    let out_gif = out_gif.to_string();
    let opts = AnimateInundationOptions {
        frame_delay_ms,
        ping_pong,
        loops,
        water_alpha,
        max_dimension,
        time_step_h,
        rain_intensity_cm_h,
        show_scale_bar,
        show_legend,
        hillshade: HillshadeOptions {
            azimuth_deg,
            elevation_deg,
            ..Default::default()
        },
        ..Default::default()
    };
    let res = py
        .detach(move || lidar::animate_inundation(&in_dem, &in_flood_dir, &out_gif, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("gif_path", res.gif_path.to_string_lossy().as_ref())?;
    dict.set_item("frame_count", res.frame_count)?;
    dict.set_item("width", res.width)?;
    dict.set_item("height", res.height)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem, in_fill_time, out_dir, num_steps=10, time_step_h=1.0, rain_intensity_cm_h=None, water_alpha=0.55, max_dimension=None, show_scale_bar=true, show_legend=true, azimuth_deg=135.0, elevation_deg=45.0))]
#[allow(clippy::too_many_arguments)]
fn export_inundation_frames_png(
    py: Python<'_>,
    in_dem: &str,
    in_fill_time: &str,
    out_dir: &str,
    num_steps: usize,
    time_step_h: f64,
    rain_intensity_cm_h: Option<f64>,
    water_alpha: f32,
    max_dimension: Option<usize>,
    show_scale_bar: bool,
    show_legend: bool,
    azimuth_deg: f64,
    elevation_deg: f64,
) -> PyResult<Vec<String>> {
    let in_dem = in_dem.to_string();
    let in_fill_time = in_fill_time.to_string();
    let out_dir = out_dir.to_string();
    let opts = AnimateInundationOptions {
        water_alpha,
        max_dimension,
        time_step_h: Some(time_step_h),
        rain_intensity_cm_h,
        show_scale_bar,
        show_legend,
        hillshade: HillshadeOptions {
            azimuth_deg,
            elevation_deg,
            ..Default::default()
        },
        ..Default::default()
    };
    let paths = py
        .detach(move || {
            let dem = lidar::load_raster::<f64>(&in_dem, None)?;
            let ft = lidar::load_raster::<f64>(&in_fill_time, None)?;
            lidar::export_inundation_frames_png(
                &dem,
                &ft,
                num_steps,
                time_step_h,
                &[],
                &out_dir,
                &opts,
            )
        })
        .map_err(to_py_err)?;
    Ok(paths
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}

#[pyfunction]
#[pyo3(signature = (arr, mode="selective", max_depth=None, max_length=None, fill_remaining=false, no_data=None, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn breach_depressions_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    mode: &str,
    max_depth: Option<f64>,
    max_length: Option<u32>,
    fill_remaining: bool,
    no_data: Option<f64>,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2_spatial(arr, no_data, 1.0, geotransform, crs)?;
    let mut opts = BreachOptions::for_mode(parse_breach_mode(mode)?, max_depth, max_length);
    opts.fill_remaining = fill_remaining;
    let out = py
        .detach(move || lidar::breach_depressions(&dem, &opts))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, out_dem, mode="selective", max_depth=None, max_length=None, fill_remaining=false, compress=false, no_data=None))]
#[allow(clippy::too_many_arguments)]
fn breach_depressions(
    py: Python<'_>,
    in_dem: &str,
    out_dem: &str,
    mode: &str,
    max_depth: Option<f64>,
    max_length: Option<u32>,
    fill_remaining: bool,
    compress: bool,
    no_data: Option<f64>,
) -> PyResult<String> {
    let in_dem = in_dem.to_string();
    let out_dem_s = out_dem.to_string();
    let mut opts = BreachOptions::for_mode(parse_breach_mode(mode)?, max_depth, max_length);
    opts.fill_remaining = fill_remaining;
    opts.compress = compress;
    let out_ret = out_dem_s.clone();
    py.detach(move || {
        lidar::breach_depressions_file_with_no_data(&in_dem, &out_dem_s, &opts, no_data)
    })
    .map_err(to_py_err)?;
    Ok(out_ret)
}

#[pyfunction]
#[pyo3(signature = (in_dem, culverts_vector, out_dem, half_width_cells=0, depth_offset=0.05, point_radius_cells=5, compress=false))]
#[allow(clippy::too_many_arguments)]
fn burn_culverts(
    py: Python<'_>,
    in_dem: &str,
    culverts_vector: &str,
    out_dem: &str,
    half_width_cells: usize,
    depth_offset: f64,
    point_radius_cells: usize,
    compress: bool,
) -> PyResult<usize> {
    let in_dem = in_dem.to_string();
    let culverts_vector = culverts_vector.to_string();
    let out_dem = out_dem.to_string();
    let opts = BurnCulvertsOptions {
        half_width_cells,
        depth_offset,
        point_radius_cells,
        compress,
    };
    let (_out, carved) = py
        .detach(move || lidar::burn_culverts_file(&in_dem, &culverts_vector, &out_dem, &opts))
        .map_err(to_py_err)?;
    Ok(carved)
}

#[pyfunction]
#[pyo3(signature = (cn_arr, precip_mm, no_data=None))]
fn scs_curve_number_runoff<'py>(
    py: Python<'py>,
    cn_arr: PyReadonlyArray2<'_, f64>,
    precip_mm: f64,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cn = from_py_array2(cn_arr, no_data)?;
    let out = py
        .detach(move || lidar::scs_curve_number_runoff(&cn, precip_mm))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (total_precip_mm, duration_h=24.0, time_step_h=1.0))]
fn scs_type2_hyetograph(
    total_precip_mm: f64,
    duration_h: f64,
    time_step_h: f64,
) -> PyResult<Vec<(f64, f64)>> {
    lidar::scs_type2_hyetograph(total_precip_mm, duration_h, time_step_h).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (arr, min_size=1000, topology="D8", no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn extract_sinks_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    min_size: usize,
    topology: &str,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let dem = from_py_array2_spatial(arr, no_data, cell_size, geotransform, crs)?;
    let out_gt = dem.geotransform;
    let out_crs = crs_to_py(&dem.projection);
    let topo = parse_topology(topology)?;
    let mem = py
        .detach(move || lidar::extract_sinks_in_memory(&dem, min_size, topo))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("sink", to_py_array2(py, mem.sink)?)?;
    dict.set_item("dem_refined", to_py_array2(py, mem.dem_refined)?)?;
    dict.set_item("dem_filled", to_py_array2(py, mem.dem_filled)?)?;
    dict.set_item("dem_diff", to_py_array2(py, mem.dem_diff)?)?;
    dict.set_item("depth", to_py_array2(py, mem.depth)?)?;
    dict.set_item("region", to_py_array2_i32(py, mem.region)?)?;
    let deps = PyList::empty(py);
    for d in &mem.depressions {
        deps.append(sink_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    dict.set_item("geotransform", geotransform_to_py(out_gt))?;
    dict.set_item("crs", out_crs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (sink_arr, min_size=1000, min_depth=0.3, interval=0.3, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn delineate_depressions_array<'py>(
    py: Python<'py>,
    sink_arr: PyReadonlyArray2<'_, f64>,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let sink = from_py_array2_spatial(sink_arr, no_data, cell_size, geotransform, crs)?;
    let out_gt = sink.geotransform;
    let out_crs = crs_to_py(&sink.projection);
    let mem = py
        .detach(move || {
            lidar::delineate_depressions_in_memory(&sink, min_size, min_depth, interval)
        })
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("depression_id", to_py_array2_i32(py, mem.depression_id)?)?;
    dict.set_item(
        "depression_level",
        to_py_array2_i32(py, mem.depression_level)?,
    )?;
    dict.set_item("max_level", mem.max_level)?;
    dict.set_item("num_regions", mem.num_regions)?;
    let deps = PyList::empty(py);
    for d in &mem.depressions {
        deps.append(nested_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    dict.set_item("geotransform", geotransform_to_py(out_gt))?;
    dict.set_item("crs", out_crs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (dsm_arr, min_size=1000, min_height=0.3, interval=0.3, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn delineate_mounts_array<'py>(
    py: Python<'py>,
    dsm_arr: PyReadonlyArray2<'_, f64>,
    min_size: usize,
    min_height: f64,
    interval: f64,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let dsm = from_py_array2_spatial(dsm_arr, no_data, cell_size, geotransform, crs)?;
    let out_gt = dsm.geotransform;
    let out_crs = crs_to_py(&dsm.projection);
    let mem = py
        .detach(move || lidar::delineate_mounts_in_memory(&dsm, min_size, min_height, interval))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("depression_id", to_py_array2_i32(py, mem.depression_id)?)?;
    dict.set_item(
        "depression_level",
        to_py_array2_i32(py, mem.depression_level)?,
    )?;
    dict.set_item("max_level", mem.max_level)?;
    dict.set_item("num_regions", mem.num_regions)?;
    let deps = PyList::empty(py);
    for d in &mem.depressions {
        deps.append(nested_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    dict.set_item("geotransform", geotransform_to_py(out_gt))?;
    dict.set_item("crs", out_crs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (dem_arr, azimuth=315.0, altitude=45.0, z_factor=1.0, cell_size=1.0, geotransform=None, no_data=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn compute_hillshade_array<'py>(
    py: Python<'py>,
    dem_arr: PyReadonlyArray2<'_, f64>,
    azimuth: f64,
    altitude: f64,
    z_factor: f64,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    no_data: Option<f64>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let dem = from_py_array2_spatial(dem_arr, no_data, cell_size, geotransform, crs)?;
    let (h, w) = (dem.height(), dem.width());
    let opts = HillshadeOptions {
        azimuth_deg: azimuth,
        elevation_deg: altitude,
        z_factor,
        ..Default::default()
    };
    let rgb = py.detach(move || lidar::compute_hillshade_rgb(&dem, &opts));
    to_py_array3_u8(py, h, w, 3, rgb)
}

#[pyfunction]
#[pyo3(signature = (in_dem, azimuth=315.0, altitude=45.0, z_factor=1.0, no_data=None))]
fn compute_hillshade_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    azimuth: f64,
    altitude: f64,
    z_factor: f64,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let in_dem = in_dem.to_string();
    let opts = HillshadeOptions {
        azimuth_deg: azimuth,
        elevation_deg: altitude,
        z_factor,
        ..Default::default()
    };
    let (h, w, rgb) = py
        .detach(move || -> lidar::Result<(usize, usize, Vec<u8>)> {
            let mut dem = lidar::load_raster::<f64>(&in_dem, None)?;
            if let Some(nd) = no_data {
                dem.set_no_data(nd);
            }
            let (h, w) = (dem.height(), dem.width());
            let rgb = lidar::compute_hillshade_rgb(&dem, &opts);
            Ok((h, w, rgb))
        })
        .map_err(to_py_err)?;
    to_py_array3_u8(py, h, w, 3, rgb)
}

fn from_py_array2_u8_spatial(
    arr: PyReadonlyArray2<'_, u8>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Array2D<u8>> {
    let view = arr.as_array();
    let (h, w) = (view.nrows(), view.ncols());
    let data: Vec<u8> = view
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| view.iter().copied().collect());
    let mut a = Array2D::from_vec(w, h, data).map_err(|e| to_py_err(e.into()))?;
    a.set_no_data(0);
    let (gt, proj) = resolve_spatial_meta(a.height(), cell_size, geotransform, crs)?;
    a.geotransform = gt;
    if let Some(p) = proj {
        a.projection = p;
    }
    Ok(a)
}

fn canonical_or_lexical_path(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
        let base = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(cp) = std::fs::canonicalize(base) {
            return cp.join(name);
        }
    }
    p.to_path_buf()
}

fn ensure_no_output_collision(input: &str, output: &str, label: &str) -> PyResult<()> {
    let in_p = canonical_or_lexical_path(Path::new(input));
    let out_p = canonical_or_lexical_path(Path::new(output));
    if in_p == out_p {
        return Err(PyValueError::new_err(format!(
            "out_file ({output}) must not overwrite input {label} ({input})"
        )));
    }
    Ok(())
}

fn save_raster_to_path(
    grid: &Array2D<f64>,
    out_path: &Path,
    history: &str,
    compress: bool,
) -> lidar::Result<()> {
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    lidar::save_raster(grid, out_path, Some(history), compress)?;
    Ok(())
}

fn embankment_artifact_to_dict<'py>(
    py: Python<'py>,
    c: &EmbankmentArtifactCandidate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("dep_id", c.dep_id)?;
    d.set_item("level", c.level)?;
    d.set_item("area_m2", c.area_m2)?;
    d.set_item("max_depth", c.max_depth)?;
    d.set_item("elongatedness", c.elongatedness)?;
    d.set_item("circularity", c.circularity)?;
    d.set_item("compactness", c.compactness)?;
    d.set_item("barrier_width_m", c.barrier_width_m)?;
    d.set_item("barrier_height_ratio", c.barrier_height_ratio)?;
    d.set_item("rim_linearity", c.rim_linearity)?;
    d.set_item("artifact_score", c.artifact_score)?;
    d.set_item("is_likely_artifact", c.is_likely_artifact)?;
    d.set_item("suggested_breach", c.suggested_breach)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (dem_arr, flow_acc=None, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
fn compute_twi_array<'py>(
    py: Python<'py>,
    dem_arr: PyReadonlyArray2<'_, f64>,
    flow_acc: Option<PyReadonlyArray2<'_, f64>>,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2_spatial(dem_arr, no_data, cell_size, geotransform, crs)?;
    let gt = dem.geotransform;
    let proj = dem.projection.clone();
    let fa_grid = match flow_acc {
        Some(fa) => Some(from_py_array2_spatial(
            fa,
            None,
            cell_size,
            Some(gt),
            Some(&proj),
        )?),
        None => None,
    };
    let out = py
        .detach(move || lidar::compute_twi(&dem, fa_grid.as_ref()))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, out_file=None, flow_acc=None, no_data=None, compress=false))]
fn compute_twi_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    out_file: Option<&str>,
    flow_acc: Option<&str>,
    no_data: Option<f64>,
    compress: bool,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    if let Some(out) = out_file {
        ensure_no_output_collision(in_dem, out, "in_dem")?;
        if let Some(fa) = flow_acc {
            ensure_no_output_collision(fa, out, "flow_acc")?;
        }
    }
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let flow_acc = flow_acc.map(str::to_string);
    let out = py
        .detach(move || -> lidar::Result<Option<Array2D<f64>>> {
            let mut dem = lidar::load_raster::<f64>(&in_dem, None)?;
            if let Some(nd) = no_data {
                dem.set_no_data(nd);
            }
            let fa = match flow_acc.as_deref() {
                Some(p) => Some(lidar::load_raster::<f64>(p, None)?),
                None => None,
            };
            let res = lidar::compute_twi(&dem, fa.as_ref())?;
            if let Some(out_path) = out_file.as_deref().map(Path::new) {
                save_raster_to_path(&res, out_path, "compute_twi", compress)?;
                Ok(None)
            } else {
                Ok(Some(res))
            }
        })
        .map_err(to_py_err)?;
    out.map(|a| to_py_array2(py, a)).transpose()
}

#[pyfunction]
#[pyo3(signature = (dem_arr, stream_mask=None, acc_threshold_cells=None, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn compute_hand_array<'py>(
    py: Python<'py>,
    dem_arr: PyReadonlyArray2<'_, f64>,
    stream_mask: Option<PyReadonlyArray2<'_, u8>>,
    acc_threshold_cells: Option<f64>,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2_spatial(dem_arr, no_data, cell_size, geotransform, crs)?;
    let gt = dem.geotransform;
    let proj = dem.projection.clone();
    let out = match (stream_mask, acc_threshold_cells) {
        (Some(mask_arr), None) => {
            let mask = from_py_array2_u8_spatial(mask_arr, cell_size, Some(gt), Some(&proj))?;
            py.detach(move || lidar::compute_hand(&dem, &mask))
                .map_err(to_py_err)?
        }
        (None, Some(thresh)) => py
            .detach(move || lidar::compute_hand_from_threshold(&dem, thresh))
            .map_err(to_py_err)?,
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "specify either stream_mask or acc_threshold_cells, not both",
            ));
        }
        (None, None) => {
            return Err(PyValueError::new_err(
                "specify either stream_mask or acc_threshold_cells",
            ));
        }
    };
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, out_file=None, stream_mask=None, acc_threshold_cells=None, no_data=None, compress=false))]
#[allow(clippy::too_many_arguments)]
fn compute_hand_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    out_file: Option<&str>,
    stream_mask: Option<&str>,
    acc_threshold_cells: Option<f64>,
    no_data: Option<f64>,
    compress: bool,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    if stream_mask.is_some() && acc_threshold_cells.is_some() {
        return Err(PyValueError::new_err(
            "specify either stream_mask or acc_threshold_cells, not both",
        ));
    }
    if stream_mask.is_none() && acc_threshold_cells.is_none() {
        return Err(PyValueError::new_err(
            "specify either stream_mask or acc_threshold_cells",
        ));
    }
    if let Some(out) = out_file {
        ensure_no_output_collision(in_dem, out, "in_dem")?;
        if let Some(mask) = stream_mask {
            ensure_no_output_collision(mask, out, "stream_mask")?;
        }
    }
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let stream_mask = stream_mask.map(str::to_string);
    let out = py
        .detach(move || -> lidar::Result<Option<Array2D<f64>>> {
            let mut dem = lidar::load_raster::<f64>(&in_dem, None)?;
            if let Some(nd) = no_data {
                dem.set_no_data(nd);
            }
            let res = if let Some(mask_path) = stream_mask.as_deref() {
                let mask_f64 = lidar::load_raster::<f64>(mask_path, None)?;
                let m_nd = mask_f64.no_data();
                let m_data: Vec<u8> = mask_f64
                    .data()
                    .iter()
                    .map(|&v| u8::from(v.is_finite() && (m_nd.is_nan() || v != m_nd) && v > 0.0))
                    .collect();
                let mut mask_u8 = Array2D::from_vec(mask_f64.width(), mask_f64.height(), m_data)?;
                mask_u8.copy_metadata_from(&mask_f64);
                mask_u8.set_no_data(0);
                lidar::compute_hand(&dem, &mask_u8)?
            } else {
                lidar::compute_hand_from_threshold(&dem, acc_threshold_cells.unwrap())?
            };
            if let Some(out_path) = out_file.as_deref().map(Path::new) {
                save_raster_to_path(&res, out_path, "compute_hand", compress)?;
                Ok(None)
            } else {
                Ok(Some(res))
            }
        })
        .map_err(to_py_err)?;
    out.map(|a| to_py_array2(py, a)).transpose()
}

#[pyfunction]
#[pyo3(signature = (dem_arr, depression_id=None, depressions=None, max_barrier_width_m=60.0, min_exit_drop=0.05, score_threshold=0.55, leaves_only=true, min_size=1000, min_depth=0.3, interval=0.3, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn classify_embankment_artifacts_array<'py>(
    py: Python<'py>,
    dem_arr: PyReadonlyArray2<'_, f64>,
    depression_id: Option<PyReadonlyArray2<'_, i32>>,
    depressions: Option<Bound<'py, PyList>>,
    max_barrier_width_m: f64,
    min_exit_drop: f64,
    score_threshold: f64,
    leaves_only: bool,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyList>> {
    let dem = from_py_array2_spatial(dem_arr, no_data, cell_size, geotransform, crs)?;
    let gt = dem.geotransform;
    let proj = dem.projection.clone();
    let opts = ArtifactClassifierOptions {
        max_barrier_width_m,
        min_exit_drop,
        score_threshold,
        leaves_only,
    };

    let precomputed = match (depression_id, depressions) {
        (Some(dep_id_arr), Some(dep_list)) => {
            let id_grid = from_py_array2_i32_spatial(dep_id_arr, cell_size, Some(gt), Some(&proj))?;
            let mut lvl_grid = Array2D::new(id_grid.width(), id_grid.height(), 0i32);
            lvl_grid.copy_metadata_from(&id_grid);
            let mut dep_records = Vec::with_capacity(dep_list.len());
            let mut max_level = 0u32;
            let mut max_region = 0u32;
            for item in dep_list.iter() {
                let d = item.cast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each element of depressions must be a dict")
                })?;
                let rec = dict_to_nested_dep(d)?;
                max_level = max_level.max(rec.level);
                max_region = max_region.max(rec.region_id);
                dep_records.push(rec);
            }
            Some(InMemoryDepressions {
                depression_id: id_grid,
                depression_level: lvl_grid,
                depressions: dep_records,
                max_level,
                num_regions: max_region,
            })
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "provide both depression_id and depressions, or omit both to delineate automatically",
            ));
        }
    };

    let candidates = py
        .detach(move || -> lidar::Result<Vec<EmbankmentArtifactCandidate>> {
            let deps = match precomputed {
                Some(d) => d,
                None => {
                    let sinks = lidar::extract_sinks_in_memory(&dem, min_size, Topology::D8)?;
                    lidar::delineate_depressions_in_memory(
                        &sinks.sink,
                        min_size,
                        min_depth,
                        interval,
                    )?
                }
            };
            lidar::classify_embankment_artifacts(&dem, &deps, &opts)
        })
        .map_err(to_py_err)?;

    let out = PyList::empty(py);
    for c in &candidates {
        out.append(embankment_artifact_to_dict(py, c)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, max_barrier_width_m=60.0, min_exit_drop=0.05, score_threshold=0.55, leaves_only=true, min_size=1000, min_depth=0.3, interval=0.3, no_data=None))]
#[allow(clippy::too_many_arguments)]
fn classify_embankment_artifacts_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    max_barrier_width_m: f64,
    min_exit_drop: f64,
    score_threshold: f64,
    leaves_only: bool,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyList>> {
    let in_dem = in_dem.to_string();
    let opts = ArtifactClassifierOptions {
        max_barrier_width_m,
        min_exit_drop,
        score_threshold,
        leaves_only,
    };
    let candidates = py
        .detach(move || -> lidar::Result<Vec<EmbankmentArtifactCandidate>> {
            let mut dem = lidar::load_raster::<f64>(&in_dem, None)?;
            if let Some(nd) = no_data {
                dem.set_no_data(nd);
            }
            let sinks = lidar::extract_sinks_in_memory(&dem, min_size, Topology::D8)?;
            let deps =
                lidar::delineate_depressions_in_memory(&sinks.sink, min_size, min_depth, interval)?;
            lidar::classify_embankment_artifacts(&dem, &deps, &opts)
        })
        .map_err(to_py_err)?;

    let out = PyList::empty(py);
    for c in &candidates {
        out.append(embankment_artifact_to_dict(py, c)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (sink_arr, depression_id=None, depression_level=None, depressions=None, interval=0.3, min_size=1000, min_depth=0.3, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn compute_rating_curves<'py>(
    py: Python<'py>,
    sink_arr: PyReadonlyArray2<'_, f64>,
    depression_id: Option<PyReadonlyArray2<'_, i32>>,
    depression_level: Option<PyReadonlyArray2<'_, i32>>,
    depressions: Option<Bound<'py, PyList>>,
    interval: f64,
    min_size: usize,
    min_depth: f64,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyList>> {
    let sink = from_py_array2_spatial(sink_arr, no_data, cell_size, geotransform, crs)?;
    let gt = sink.geotransform;
    let proj = sink.projection.clone();

    let precomputed = match (depression_id, depressions) {
        (Some(dep_id_arr), Some(dep_list)) => {
            let id_grid = from_py_array2_i32_spatial(dep_id_arr, cell_size, Some(gt), Some(&proj))?;
            let lvl_grid = match depression_level {
                Some(lvl_arr) => {
                    from_py_array2_i32_spatial(lvl_arr, cell_size, Some(gt), Some(&proj))?
                }
                None => {
                    let mut z = Array2D::new(id_grid.width(), id_grid.height(), 0i32);
                    z.copy_metadata_from(&id_grid);
                    z
                }
            };
            let mut dep_records = Vec::with_capacity(dep_list.len());
            let mut max_level = 0u32;
            let mut max_region = 0u32;
            for item in dep_list.iter() {
                let d = item.cast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each element of depressions must be a dict")
                })?;
                let rec = dict_to_nested_dep(d)?;
                max_level = max_level.max(rec.level);
                max_region = max_region.max(rec.region_id);
                dep_records.push(rec);
            }
            Some(InMemoryDepressions {
                depression_id: id_grid,
                depression_level: lvl_grid,
                depressions: dep_records,
                max_level,
                num_regions: max_region,
            })
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "provide both depression_id and depressions, or omit both to delineate automatically",
            ));
        }
    };

    let points = py
        .detach(move || -> lidar::Result<Vec<lidar::RatingCurvePoint>> {
            let deps = match precomputed {
                Some(d) => d,
                None => {
                    lidar::delineate_depressions_in_memory(&sink, min_size, min_depth, interval)?
                }
            };
            lidar::compute_rating_curves(&sink, &deps, interval)
        })
        .map_err(to_py_err)?;

    struct RatingCurveGroup {
        level: u32,
        region_id: u32,
        stages: Vec<f64>,
        depths: Vec<f64>,
        areas: Vec<f64>,
        volumes: Vec<f64>,
    }
    let mut order: Vec<u32> = Vec::new();
    let mut grouped: std::collections::HashMap<u32, RatingCurveGroup> =
        std::collections::HashMap::new();
    for pt in points {
        let entry = grouped.entry(pt.dep_id).or_insert_with(|| {
            order.push(pt.dep_id);
            RatingCurveGroup {
                level: pt.level,
                region_id: pt.region_id,
                stages: Vec::new(),
                depths: Vec::new(),
                areas: Vec::new(),
                volumes: Vec::new(),
            }
        });
        entry.stages.push(pt.stage);
        entry.depths.push(pt.depth);
        entry.areas.push(pt.area);
        entry.volumes.push(pt.volume);
    }

    let out = PyList::empty(py);
    for dep_id in order {
        if let Some(group) = grouped.remove(&dep_id) {
            let d = PyDict::new(py);
            d.set_item("id", dep_id)?;
            d.set_item("dep_id", dep_id)?;
            d.set_item("level", group.level)?;
            d.set_item("region_id", group.region_id)?;
            d.set_item("stages", group.stages)?;
            d.set_item("depths", group.depths)?;
            d.set_item("areas", group.areas)?;
            d.set_item("volumes", group.volumes)?;
            out.append(d)?;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (dem_arr, min_size=1000, min_depth=0.3, interval=0.3, rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0, kernel_size=1, hyetograph=None, stream_vector=None, no_data=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn build_connectivity_graph_from_dem<'py>(
    py: Python<'py>,
    dem_arr: PyReadonlyArray2<'_, f64>,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    rain_intensity_cm_h: f64,
    rain_duration_h: f64,
    time_step_h: f64,
    kernel_size: usize,
    hyetograph: Option<Vec<(f64, f64)>>,
    stream_vector: Option<&str>,
    no_data: Option<f64>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let raw_dem = from_py_array2_spatial(dem_arr, no_data, cell_size, geotransform, crs)?;
    let gt = raw_dem.geotransform;
    let stream_path = stream_vector.map(PathBuf::from);
    let sim_opts = SimulateInundationOptions {
        min_size,
        min_depth,
        interval,
        rain_intensity_cm_h,
        rain_duration_h,
        time_step_h,
        compress: false,
    };
    let graph = py
        .detach(move || -> lidar::Result<ConnectivityGraph> {
            let opts = lidar::InMemoryFullPipelineOptions {
                kernel_size,
                breach: None,
                simulation: sim_opts,
                routing: RoutingMode::Static,
                forcing: InundationForcing {
                    runoff_coeff: None,
                    infiltration_cm_h: None,
                    antecedent_depth: None,
                    hyetograph: hyetograph.as_deref(),
                },
                scalar_runoff_coeff: 1.0,
                stream_vector: stream_path.as_deref(),
            };
            let pipe = lidar::run_full_pipeline_in_memory(&raw_dem, &opts)?;
            Ok(pipe.connectivity_graph)
        })
        .map_err(to_py_err)?;
    connectivity_graph_to_dict(py, &graph, Some(&gt))
}

#[pyfunction]
#[pyo3(signature = (depression_id, depression_level, depressions, fill_time=None, catchment_records=None, spill_paths=None, rain_intensity_cm_h=5.0, hyetograph=None, stream_vector=None, cell_size=1.0, geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn build_connectivity_graph_from_parts<'py>(
    py: Python<'py>,
    depression_id: PyReadonlyArray2<'_, i32>,
    depression_level: Option<PyReadonlyArray2<'_, i32>>,
    depressions: Bound<'py, PyList>,
    fill_time: Option<PyReadonlyArray2<'_, f64>>,
    catchment_records: Option<Bound<'py, PyList>>,
    spill_paths: Option<Bound<'py, PyList>>,
    rain_intensity_cm_h: f64,
    hyetograph: Option<Vec<(f64, f64)>>,
    stream_vector: Option<&str>,
    cell_size: f64,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let id_grid = from_py_array2_i32_spatial(depression_id, cell_size, geotransform, crs)?;
    let gt = id_grid.geotransform;
    let proj = id_grid.projection.clone();
    let lvl_grid = match depression_level {
        Some(lvl_arr) => from_py_array2_i32_spatial(lvl_arr, cell_size, Some(gt), Some(&proj))?,
        None => {
            let mut z = Array2D::new(id_grid.width(), id_grid.height(), 0i32);
            z.copy_metadata_from(&id_grid);
            z
        }
    };
    let ft_grid = match fill_time {
        Some(ft_arr) => from_py_array2_spatial(ft_arr, None, cell_size, Some(gt), Some(&proj))?,
        None => {
            let mut ft = Array2D::new(id_grid.width(), id_grid.height(), f64::NAN);
            ft.set_no_data(f64::NAN);
            ft.geotransform = gt;
            ft.projection = proj.clone();
            ft
        }
    };

    let mut dep_records = Vec::with_capacity(depressions.len());
    let mut max_level = 0u32;
    let mut max_region = 0u32;
    for item in depressions.iter() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each element of depressions must be a dict"))?;
        let rec = dict_to_nested_dep(d)?;
        max_level = max_level.max(rec.level);
        max_region = max_region.max(rec.region_id);
        dep_records.push(rec);
    }

    let mut cat_records = Vec::new();
    if let Some(list) = catchment_records {
        for item in list.iter() {
            let d = item.cast::<PyDict>().map_err(|_| {
                PyValueError::new_err("each element of catchment_records must be a dict")
            })?;
            cat_records.push(dict_to_catchment_record(d)?);
        }
    }

    let mut spill_records = Vec::new();
    if let Some(list) = spill_paths {
        for item in list.iter() {
            let d = item
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("each element of spill_paths must be a dict"))?;
            spill_records.push(dict_to_spill_path(d)?);
        }
    }

    let in_mem_deps = InMemoryDepressions {
        depression_id: id_grid,
        depression_level: lvl_grid,
        depressions: dep_records,
        max_level,
        num_regions: max_region,
    };
    let stream_path = stream_vector.map(PathBuf::from);
    let graph = py
        .detach(move || {
            lidar::build_connectivity_graph(
                &in_mem_deps,
                &cat_records,
                &ft_grid,
                &spill_records,
                rain_intensity_cm_h,
                hyetograph.as_deref(),
                stream_path.as_deref(),
            )
        })
        .map_err(to_py_err)?;
    connectivity_graph_to_dict(py, &graph, Some(&gt))
}

#[pyfunction]
#[pyo3(signature = (dem_arr, kernel_size=3, min_size=1000, min_depth=0.3, interval=0.3, rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0, runoff_coeff_arr=None, infiltration_arr=None, antecedent_depth_arr=None, hyetograph=None, breach_max_depth=None, no_data=None, cell_size=1.0, routing="static", geotransform=None, crs=None))]
#[allow(clippy::too_many_arguments)]
fn simulate_inundation_array<'py>(
    py: Python<'py>,
    dem_arr: PyReadonlyArray2<'_, f64>,
    kernel_size: usize,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    rain_intensity_cm_h: f64,
    rain_duration_h: f64,
    time_step_h: f64,
    runoff_coeff_arr: Option<PyReadonlyArray2<'_, f64>>,
    infiltration_arr: Option<PyReadonlyArray2<'_, f64>>,
    antecedent_depth_arr: Option<PyReadonlyArray2<'_, f64>>,
    hyetograph: Option<Vec<(f64, f64)>>,
    breach_max_depth: Option<f64>,
    no_data: Option<f64>,
    cell_size: f64,
    routing: &str,
    geotransform: Option<[f64; 6]>,
    crs: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let routing_mode = parse_routing_mode(routing)?;
    let raw_dem = from_py_array2_spatial(dem_arr, no_data, cell_size, geotransform, crs)?;
    let out_gt = raw_dem.geotransform;
    let out_crs = crs_to_py(&raw_dem.projection);
    let rc = runoff_coeff_arr
        .map(|a| from_py_array2_spatial(a, None, cell_size, Some(out_gt), crs))
        .transpose()?;
    let inf = infiltration_arr
        .map(|a| from_py_array2_spatial(a, None, cell_size, Some(out_gt), crs))
        .transpose()?;
    let ant = antecedent_depth_arr
        .map(|a| from_py_array2_spatial(a, None, cell_size, Some(out_gt), crs))
        .transpose()?;

    let sim_opts = SimulateInundationOptions {
        min_size,
        min_depth,
        interval,
        rain_intensity_cm_h,
        rain_duration_h,
        time_step_h,
        compress: false,
    };

    let pipe = py
        .detach(move || {
            let breach = breach_max_depth.map(|md| BreachOptions {
                mode: BreachMode::Selective,
                max_depth: Some(md),
                ..Default::default()
            });
            let opts = lidar::InMemoryFullPipelineOptions {
                kernel_size,
                breach,
                simulation: sim_opts,
                routing: routing_mode,
                forcing: InundationForcing {
                    runoff_coeff: rc.as_ref(),
                    infiltration_cm_h: inf.as_ref(),
                    antecedent_depth: ant.as_ref(),
                    hyetograph: hyetograph.as_deref(),
                },
                scalar_runoff_coeff: 1.0,
                stream_vector: None,
            };
            lidar::run_full_pipeline_in_memory(&raw_dem, &opts)
        })
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "depression_id",
        to_py_array2_i32(py, pipe.depressions.depression_id)?,
    )?;
    dict.set_item(
        "depression_level",
        to_py_array2_i32(py, pipe.depressions.depression_level)?,
    )?;
    dict.set_item(
        "catchment_hierarchy",
        to_py_array2(py, pipe.catchment_hierarchy)?,
    )?;
    dict.set_item("fill_time", to_py_array2(py, pipe.fill_time)?)?;

    let py_deps = PyList::empty(py);
    for d in &pipe.depressions.depressions {
        py_deps.append(nested_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", py_deps)?;

    let py_steps = PyList::empty(py);
    for s in &pipe.steps {
        let d = PyDict::new(py);
        d.set_item("step", s.step)?;
        d.set_item("time_hours", s.time_hours)?;
        d.set_item("inundated_cells", s.inundated_cells)?;
        d.set_item("inundated_area", s.inundated_area)?;
        d.set_item("inundated_fraction", s.inundated_fraction)?;
        py_steps.append(d)?;
    }
    dict.set_item("steps", py_steps)?;

    let py_spills = PyList::empty(py);
    for sp in &pipe.spill_paths {
        let d = PyDict::new(py);
        d.set_item("from_dep_id", sp.from_dep_id)?;
        d.set_item("from_level", sp.from_level)?;
        d.set_item("to_dep_id", sp.to_dep_id)?;
        d.set_item("spill_time_h", sp.spill_time_h)?;
        d.set_item("merge_time_h", sp.merge_time_h)?;
        d.set_item("spill_elev", sp.spill_elev)?;
        d.set_item("cells", sp.cells.clone())?;
        py_spills.append(d)?;
    }
    dict.set_item("spill_paths", py_spills)?;

    let py_rc = PyList::empty(py);
    for pt in &pipe.rating_curves {
        let d = PyDict::new(py);
        d.set_item("dep_id", pt.dep_id)?;
        d.set_item("level", pt.level)?;
        d.set_item("region_id", pt.region_id)?;
        d.set_item("stage", pt.stage)?;
        d.set_item("depth", pt.depth)?;
        d.set_item("area", pt.area)?;
        d.set_item("volume", pt.volume)?;
        py_rc.append(d)?;
    }
    dict.set_item("rating_curves", py_rc)?;
    dict.set_item(
        "connectivity_graph",
        connectivity_graph_to_dict(py, &pipe.connectivity_graph, Some(&out_gt))?,
    )?;
    dict.set_item(
        "connectivity_graphml",
        lidar::connectivity_graph_to_graphml_string(&pipe.connectivity_graph),
    )?;
    dict.set_item("inundation_curve_csv", &pipe.inundation_curve_csv)?;
    let py_cat_recs = PyList::empty(py);
    for r in &pipe.catchment_records {
        py_cat_recs.append(catchment_hierarchy_record_to_dict(py, r)?)?;
    }
    dict.set_item("catchment_records", py_cat_recs)?;
    dict.set_item("graph_node_count", pipe.connectivity_graph.nodes.len())?;
    dict.set_item("graph_edge_count", pipe.connectivity_graph.edges.len())?;
    if let Some(ref mb) = pipe.fsm_mass_balance {
        dict.set_item("fsm_mass_balance", fsm_mass_balance_to_dict(py, mb)?)?;
    } else {
        dict.set_item("fsm_mass_balance", py.None())?;
    }
    if let Some(ref basins) = pipe.fsm_basins {
        let py_basins = PyList::empty(py);
        for b in basins {
            py_basins.append(fsm_basin_to_dict(py, b)?)?;
        }
        dict.set_item("fsm_basins", py_basins)?;
    } else {
        dict.set_item("fsm_basins", py.None())?;
    }
    dict.set_item("geotransform", geotransform_to_py(out_gt))?;
    dict.set_item("crs", out_crs)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem, out_dir, kernel_size=3, min_size=1000, min_depth=0.3, interval=0.3, rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0, breach_max_depth=None, breach_max_length=40, culverts_vector=None, runoff_coeff_path=None, infiltration_path=None, antecedent_depth_path=None, hyetograph=None, stream_vector=None, use_f32=false, save_rasters=true, export_vectors=true, export_graph=true, compress=false, routing="static"))]
#[allow(clippy::too_many_arguments)]
fn run_pipeline<'py>(
    py: Python<'py>,
    in_dem: &str,
    out_dir: &str,
    kernel_size: usize,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    rain_intensity_cm_h: f64,
    rain_duration_h: f64,
    time_step_h: f64,
    breach_max_depth: Option<f64>,
    breach_max_length: u32,
    culverts_vector: Option<&str>,
    runoff_coeff_path: Option<&str>,
    infiltration_path: Option<&str>,
    antecedent_depth_path: Option<&str>,
    hyetograph: Option<Vec<(f64, f64)>>,
    stream_vector: Option<&str>,
    use_f32: bool,
    save_rasters: bool,
    export_vectors: bool,
    export_graph: bool,
    compress: bool,
    routing: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let routing_mode = parse_routing_mode(routing)?;
    let in_dem = in_dem.to_string();
    let out_dir = out_dir.to_string();
    let breach = breach_max_depth.map(|md| BreachOptions {
        mode: BreachMode::Selective,
        max_depth: Some(md),
        max_length: Some(breach_max_length),
        ..Default::default()
    });
    let opts = RunPipelineOptions {
        kernel_size,
        breach,
        culverts_vector: culverts_vector.map(PathBuf::from),
        simulation: SimulateInundationOptions {
            min_size,
            min_depth,
            interval,
            rain_intensity_cm_h,
            rain_duration_h,
            time_step_h,
            compress,
        },
        routing: routing_mode,
        runoff_coeff_path: runoff_coeff_path.map(PathBuf::from),
        infiltration_path: infiltration_path.map(PathBuf::from),
        antecedent_depth_path: antecedent_depth_path.map(PathBuf::from),
        hyetograph,
        stream_vector: stream_vector.map(PathBuf::from),
        use_f32,
        save_rasters,
        export_vectors,
        export_graph,
        ..Default::default()
    };
    let res = py
        .detach(move || lidar::run_pipeline(&in_dem, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("out_dir", res.out_dir.to_string_lossy().as_ref())?;
    dict.set_item("width", res.width)?;
    dict.set_item("height", res.height)?;
    dict.set_item("num_sink_regions", res.num_sink_regions)?;
    dict.set_item("num_depressions", res.num_depressions)?;
    dict.set_item("max_level", res.max_level)?;
    dict.set_item("total_storage_m3", res.total_storage_m3)?;
    dict.set_item("num_spill_paths", res.num_spill_paths)?;
    dict.set_item(
        "depressions_csv",
        res.depressions_csv.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "catchments_csv",
        res.catchments_csv.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "inundation_csv",
        res.inundation_csv.to_string_lossy().as_ref(),
    )?;
    dict.set_item("summary_json", res.summary_json.to_string_lossy().as_ref())?;
    dict.set_item(
        "depressions_gpkg",
        opt_path_to_py(res.depressions_gpkg.as_ref()),
    )?;
    dict.set_item(
        "spill_paths_gpkg",
        opt_path_to_py(res.spill_paths_gpkg.as_ref()),
    )?;
    dict.set_item(
        "rating_curves_csv",
        opt_path_to_py(res.rating_curves_csv.as_ref()),
    )?;
    dict.set_item(
        "connectivity_graph_json",
        opt_path_to_py(res.connectivity_graph_json.as_ref()),
    )?;
    dict.set_item(
        "connectivity_graphml",
        opt_path_to_py(res.connectivity_graphml.as_ref()),
    )?;
    dict.set_item(
        "routing",
        match res.routing {
            RoutingMode::Static => "static",
            RoutingMode::FillSpillMerge => "fsm",
        },
    )?;
    dict.set_item(
        "fsm_basins_csv",
        opt_path_to_py(res.fsm_basins_csv.as_ref()),
    )?;
    if let Some(ref mb) = res.fsm_mass_balance {
        dict.set_item("fsm_mass_balance", fsm_mass_balance_to_dict(py, mb)?)?;
    } else {
        dict.set_item("fsm_mass_balance", py.None())?;
    }
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (in_dem, out_dir, tile_size=2048, overlap=128, kernel_size=3, min_size=1000, min_depth=0.3, interval=0.3, use_f32=true))]
#[allow(clippy::too_many_arguments)]
fn delineate_depressions_tiled<'py>(
    py: Python<'py>,
    in_dem: &str,
    out_dir: &str,
    tile_size: usize,
    overlap: usize,
    kernel_size: usize,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    use_f32: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let in_dem = in_dem.to_string();
    let out_dir = out_dir.to_string();
    let opts = TiledDelineationOptions {
        tile_size,
        overlap,
        kernel_size,
        min_size,
        min_depth,
        interval,
        use_f32,
        ..Default::default()
    };
    let res = py
        .detach(move || lidar::delineate_depressions_tiled(&in_dem, &out_dir, &opts))
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("num_tiles", res.num_tiles)?;
    dict.set_item("max_level", res.max_level)?;
    dict.set_item("csv_path", res.csv_path.to_string_lossy().as_ref())?;
    let deps = PyList::empty(py);
    for d in &res.depressions {
        deps.append(nested_dep_to_dict(py, d)?)?;
    }
    dict.set_item("depressions", deps)?;
    Ok(dict)
}

fn parse_py_region(
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
) -> PyResult<Region> {
    match (bbox, huc, region) {
        (Some((x0, y0, x1, y1)), None, None) => {
            Ok(Region::Bbox(Bbox::new(x0, y0, x1, y1).map_err(to_py_err)?))
        }
        (None, Some(code), None) => {
            Ok(Region::Huc(Region::validate_huc(&code).map_err(to_py_err)?))
        }
        (None, None, Some(path)) => Ok(Region::VectorFile(PathBuf::from(path))),
        _ => Err(PyValueError::new_err(
            "specify exactly one of bbox, huc, or region",
        )),
    }
}

fn tnm_item_to_dict<'py>(py: Python<'py>, it: &TnmProductItem) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("grid_key", &it.grid_key)?;
    d.set_item("title", &it.title)?;
    d.set_item("publication_date", &it.publication_date)?;
    d.set_item("download_url", &it.download_url)?;
    d.set_item("size_bytes", it.size_bytes)?;
    d.set_item("format", &it.format)?;
    d.set_item(
        "bbox",
        (
            it.bbox.min_lon,
            it.bbox.min_lat,
            it.bbox.max_lon,
            it.bbox.max_lat,
        ),
    )?;
    Ok(d)
}

fn noaa_table_to_dict<'py>(
    py: Python<'py>,
    tbl: &NoaaAtlas14Table,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("lat", tbl.lat)?;
    d.set_item("lon", tbl.lon)?;
    d.set_item("units", &tbl.units)?;
    let depths = PyDict::new(py);
    for (dur, row) in &tbl.depths_mm {
        let row_d = PyDict::new(py);
        for (&ari, &val) in row {
            row_d.set_item(ari, val)?;
        }
        depths.set_item(dur, row_d)?;
    }
    d.set_item("depths_mm", depths)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (streams_vector, roads_vector, out_culverts, half_length_m=25.0, snap_distance_m=15.0, target_crs=None))]
fn synthesize_culverts(
    py: Python<'_>,
    streams_vector: &str,
    roads_vector: &str,
    out_culverts: &str,
    half_length_m: f64,
    snap_distance_m: f64,
    target_crs: Option<String>,
) -> PyResult<usize> {
    let streams = streams_vector.to_string();
    let roads = roads_vector.to_string();
    let out = out_culverts.to_string();
    let opts = SynthesizeCulvertsOptions {
        half_length_m,
        snap_distance_m,
        target_crs,
        ..Default::default()
    };
    py.detach(move || lidar::synthesize_culverts(&streams, &roads, &out, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (bbox=None, huc=None, region=None, dem_product="1m", deduplicate=true))]
fn query_3dep_products<'py>(
    py: Python<'py>,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    dem_product: &str,
    deduplicate: bool,
) -> PyResult<Bound<'py, PyList>> {
    let reg = parse_py_region(bbox, huc, region)?;
    let opts = DownloadOptions {
        dem_product: dem_product.parse::<DemProduct>().map_err(to_py_err)?,
        deduplicate_tiles: deduplicate,
        ..Default::default()
    };
    let items = py
        .detach(move || lidar::query_3dep_products(&reg, &opts))
        .map_err(to_py_err)?;
    let list = PyList::empty(py);
    for it in &items {
        list.append(tnm_item_to_dict(py, it)?)?;
    }
    Ok(list)
}

#[pyfunction]
#[pyo3(signature = (out_dir, bbox=None, huc=None, region=None, dem_product="1m", mosaic_dem=true, clip_to_region=true, tile_cache_dir=None, overwrite=false))]
#[allow(clippy::too_many_arguments)]
fn download_3dep_dem<'py>(
    py: Python<'py>,
    out_dir: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    dem_product: &str,
    mosaic_dem: bool,
    clip_to_region: bool,
    tile_cache_dir: Option<String>,
    overwrite: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let reg = parse_py_region(bbox, huc, region)?;
    let out = out_dir.to_string();
    let opts = DownloadOptions {
        dem_product: dem_product.parse::<DemProduct>().map_err(to_py_err)?,
        mosaic_dem,
        clip_to_region,
        tile_cache_dir: tile_cache_dir.map(PathBuf::from),
        overwrite,
        ..Default::default()
    };
    let res = py
        .detach(move || lidar::download_3dep_dem(&reg, &out, &opts))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item(
        "bbox",
        (
            res.bbox.min_lon,
            res.bbox.min_lat,
            res.bbox.max_lon,
            res.bbox.max_lat,
        ),
    )?;
    let prods = PyList::empty(py);
    for it in &res.products {
        prods.append(tnm_item_to_dict(py, it)?)?;
    }
    d.set_item("products", prods)?;
    let tiles: Vec<String> = res
        .tile_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    d.set_item("tile_paths", tiles)?;
    d.set_item(
        "mosaic_path",
        res.mosaic_path.map(|p| p.to_string_lossy().into_owned()),
    )?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (out_path, bbox=None, huc=None, region=None, level=12, overwrite=false))]
fn download_wbd(
    py: Python<'_>,
    out_path: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    level: usize,
    overwrite: bool,
) -> PyResult<usize> {
    let reg = parse_py_region(bbox, huc, region)?;
    let out = out_path.to_string();
    let opts = DownloadOptions {
        wbd_level: WbdLevel::from_digits(level).map_err(to_py_err)?,
        overwrite,
        ..Default::default()
    };
    py.detach(move || lidar::download_wbd(&reg, &out, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (out_path, bbox=None, huc=None, region=None, clip_to_region=true, overwrite=false))]
fn download_nhd(
    py: Python<'_>,
    out_path: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    clip_to_region: bool,
    overwrite: bool,
) -> PyResult<usize> {
    let reg = parse_py_region(bbox, huc, region)?;
    let out = out_path.to_string();
    let opts = DownloadOptions {
        clip_to_region,
        overwrite,
        ..Default::default()
    };
    py.detach(move || lidar::download_nhd(&reg, &out, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (out_path, bbox=None, huc=None, region=None, clip_to_region=true, overwrite=false))]
fn download_nwi(
    py: Python<'_>,
    out_path: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    clip_to_region: bool,
    overwrite: bool,
) -> PyResult<usize> {
    let reg = parse_py_region(bbox, huc, region)?;
    let out = out_path.to_string();
    let opts = DownloadOptions {
        clip_to_region,
        overwrite,
        ..Default::default()
    };
    py.detach(move || lidar::download_nwi(&reg, &out, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (out_path, bbox=None, huc=None, region=None, clip_to_region=true, overwrite=false))]
fn download_roads(
    py: Python<'_>,
    out_path: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    clip_to_region: bool,
    overwrite: bool,
) -> PyResult<usize> {
    let reg = parse_py_region(bbox, huc, region)?;
    let out = out_path.to_string();
    let opts = DownloadOptions {
        clip_to_region,
        overwrite,
        ..Default::default()
    };
    py.detach(move || lidar::download_roads(&reg, &out, &opts))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (lat, lon, out_json=None, overwrite=false))]
fn fetch_noaa_atlas14<'py>(
    py: Python<'py>,
    lat: f64,
    lon: f64,
    out_json: Option<String>,
    overwrite: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out_p = out_json.map(PathBuf::from);
    let opts = DownloadOptions {
        overwrite,
        ..Default::default()
    };
    let tbl = py
        .detach(move || lidar::fetch_noaa_atlas14(lat, lon, out_p.as_deref(), &opts))
        .map_err(to_py_err)?;
    noaa_table_to_dict(py, &tbl)
}

#[pyfunction]
#[pyo3(signature = (out_dir, bbox=None, huc=None, region=None, datasets="all", dem_product="1m", wbd_level=12, mosaic_dem=true, clip_to_region=true, tile_cache_dir=None, overwrite=false))]
#[allow(clippy::too_many_arguments)]
fn download_federal_data<'py>(
    py: Python<'py>,
    out_dir: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    huc: Option<String>,
    region: Option<String>,
    datasets: &str,
    dem_product: &str,
    wbd_level: usize,
    mosaic_dem: bool,
    clip_to_region: bool,
    tile_cache_dir: Option<String>,
    overwrite: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let reg = parse_py_region(bbox, huc, region)?;
    let ds_list = lidar::parse_dataset_list(datasets).map_err(to_py_err)?;
    let out = out_dir.to_string();
    let opts = FederalBundleOptions {
        datasets: ds_list,
        download: DownloadOptions {
            dem_product: dem_product.parse::<DemProduct>().map_err(to_py_err)?,
            wbd_level: WbdLevel::from_digits(wbd_level).map_err(to_py_err)?,
            mosaic_dem,
            clip_to_region,
            tile_cache_dir: tile_cache_dir.map(PathBuf::from),
            overwrite,
            ..Default::default()
        },
    };
    let res = py
        .detach(move || lidar::download_federal_data(&reg, &out, &opts))
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "region_bbox",
        (
            res.region_bbox.min_lon,
            res.region_bbox.min_lat,
            res.region_bbox.max_lon,
            res.region_bbox.max_lat,
        ),
    )?;
    d.set_item(
        "wbd_path",
        res.wbd_path.map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item("wbd_count", res.wbd_count)?;
    d.set_item(
        "dem_tiles",
        res.dem_tiles
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "dem_mosaic_path",
        res.dem_mosaic_path
            .map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item(
        "nhd_path",
        res.nhd_path.map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item("nhd_count", res.nhd_count)?;
    d.set_item(
        "nwi_path",
        res.nwi_path.map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item("nwi_count", res.nwi_count)?;
    d.set_item(
        "roads_path",
        res.roads_path.map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item("roads_count", res.roads_count)?;
    d.set_item(
        "culverts_path",
        res.culverts_path.map(|p| p.to_string_lossy().into_owned()),
    )?;
    d.set_item("culverts_count", res.culverts_count)?;
    d.set_item(
        "noaa_atlas14_path",
        res.noaa_atlas14_path
            .map(|p| p.to_string_lossy().into_owned()),
    )?;
    if let Some(ref tbl) = res.noaa_atlas14 {
        d.set_item("noaa_atlas14", noaa_table_to_dict(py, tbl)?)?;
    } else {
        d.set_item("noaa_atlas14", py.None())?;
    }
    d.set_item(
        "manifest_path",
        res.manifest_path.to_string_lossy().as_ref(),
    )?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (args, lidar_cmd=None))]
fn cli_main(
    py: Python<'_>,
    args: Vec<std::ffi::OsString>,
    lidar_cmd: Option<Vec<std::ffi::OsString>>,
) -> i32 {
    py.detach(move || lidar_cli::run_as(args, lidar_cmd))
}

#[pymodule]
fn _lidar_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", lidar::VERSION)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(wrap_pyfunction!(mean_filter_array, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_array, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter_array, m)?)?;
    m.add_function(wrap_pyfunction!(flip_dem_array, m)?)?;
    m.add_function(wrap_pyfunction!(mean_filter_file, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_file, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter_file, m)?)?;
    m.add_function(wrap_pyfunction!(flip_dem_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sinks, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sinks_array, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_depressions, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_depressions_array, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_depressions_tiled, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_mounts, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_mounts_array, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hillshade_array, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hillshade_file, m)?)?;
    m.add_function(wrap_pyfunction!(compute_twi_array, m)?)?;
    m.add_function(wrap_pyfunction!(compute_twi_file, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hand_array, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hand_file, m)?)?;
    m.add_function(wrap_pyfunction!(classify_embankment_artifacts_array, m)?)?;
    m.add_function(wrap_pyfunction!(classify_embankment_artifacts_file, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rating_curves, m)?)?;
    m.add_function(wrap_pyfunction!(build_connectivity_graph_from_dem, m)?)?;
    m.add_function(wrap_pyfunction!(build_connectivity_graph_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_catchments, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_flow_paths, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_catchment_hierarchy, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_inundation, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_inundation_array, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(breach_depressions, m)?)?;
    m.add_function(wrap_pyfunction!(breach_depressions_array, m)?)?;
    m.add_function(wrap_pyfunction!(burn_culverts, m)?)?;
    m.add_function(wrap_pyfunction!(synthesize_culverts, m)?)?;
    m.add_function(wrap_pyfunction!(query_3dep_products, m)?)?;
    m.add_function(wrap_pyfunction!(download_3dep_dem, m)?)?;
    m.add_function(wrap_pyfunction!(download_wbd, m)?)?;
    m.add_function(wrap_pyfunction!(download_nhd, m)?)?;
    m.add_function(wrap_pyfunction!(download_nwi, m)?)?;
    m.add_function(wrap_pyfunction!(download_roads, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_noaa_atlas14, m)?)?;
    m.add_function(wrap_pyfunction!(download_federal_data, m)?)?;
    m.add_function(wrap_pyfunction!(scs_curve_number_runoff, m)?)?;
    m.add_function(wrap_pyfunction!(scs_type2_hyetograph, m)?)?;
    m.add_function(wrap_pyfunction!(animate_inundation, m)?)?;
    m.add_function(wrap_pyfunction!(export_inundation_frames_png, m)?)?;
    m.add_function(wrap_pyfunction!(polygonize, m)?)?;
    m.add_function(wrap_pyfunction!(mosaic, m)?)?;
    m.add_function(wrap_pyfunction!(reproject_image, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(clip_image, m)?)?;
    m.add_function(wrap_pyfunction!(image_to_cog, m)?)?;
    m.add_function(wrap_pyfunction!(add_crs, m)?)?;
    m.add_function(wrap_pyfunction!(join_tables, m)?)?;
    m.add_function(wrap_pyfunction!(csv_to_vector, m)?)?;
    m.add_function(wrap_pyfunction!(convert_lidar, m)?)?;
    m.add_function(wrap_pyfunction!(read_lidar, m)?)?;
    m.add_function(wrap_pyfunction!(write_lidar, m)?)?;
    m.add_function(wrap_pyfunction!(classify_ground_file, m)?)?;
    m.add_function(wrap_pyfunction!(classify_ground_array, m)?)?;
    m.add_function(wrap_pyfunction!(lidar_to_dsm, m)?)?;
    m.add_function(wrap_pyfunction!(lidar_to_dtm, m)?)?;
    m.add_function(wrap_pyfunction!(lidar_to_chm, m)?)?;
    Ok(())
}
