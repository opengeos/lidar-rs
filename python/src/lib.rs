//! PyO3 bindings exposing the `lidar-rs` API to Python with zero GDAL dependencies.

use std::path::{Path, PathBuf};

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use lidar::pointcloud::{GridMethod, GridStat, LidarToDsmOptions};
use lidar::{
    AnimateInundationOptions, Array2D, CatchmentHierarchyOptions, DelineateCatchmentsOptions,
    DelineateFlowPathsOptions, DelineateOptions, ExtractSinksOptions, HillshadeOptions,
    NestedDepression, SimulateInundationOptions, SinkDepression, Topology,
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
    let data: Vec<f64> = view.iter().copied().collect();
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || {
            if no_data.is_none() {
                lidar::mean_filter_file(&in_dem, kernel_size, out_file.as_deref().map(Path::new))
            } else {
                let hist = format!("mean_filter(kernel_size={kernel_size})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::mean_filter(dem, kernel_size)
                })
            }
        })
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, kernel_size=3, out_file=None, no_data=None))]
fn median_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    kernel_size: usize,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || {
            if no_data.is_none() {
                lidar::median_filter_file(&in_dem, kernel_size, out_file.as_deref().map(Path::new))
            } else {
                let hist = format!("median_filter(kernel_size={kernel_size})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::median_filter(dem, kernel_size)
                })
            }
        })
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, sigma=1.0, out_file=None, no_data=None))]
fn gaussian_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    sigma: f64,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || {
            if no_data.is_none() {
                lidar::gaussian_filter_file(&in_dem, sigma, out_file.as_deref().map(Path::new))
            } else {
                let hist = format!("gaussian_filter(sigma={sigma})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::gaussian_filter(dem, sigma)
                })
            }
        })
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, delta=100.0, out_file=None, no_data=None))]
fn flip_dem_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    delta: f64,
    out_file: Option<&str>,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let in_dem = in_dem.to_string();
    let out_file = out_file.map(str::to_string);
    let out = py
        .detach(move || {
            if no_data.is_none() {
                lidar::flip_dem_file(&in_dem, delta, out_file.as_deref().map(Path::new))
            } else {
                let hist = format!("flip_dem(delta={delta})");
                run_file_filter(&in_dem, out_file.as_deref(), no_data, &hist, |dem| {
                    lidar::flip_dem(dem, delta)
                })
            }
        })
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, min_size, out_dir, filled_dem=None, topology="D8", keep_files=true, compress=false))]
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
#[pyo3(signature = (in_sink, min_size, min_depth, interval, out_dir, bool_level_shp=false, bool_comb_tif=false, compress=false))]
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
) -> PyResult<Bound<'py, PyDict>> {
    let opts = DelineateOptions {
        min_size,
        min_depth,
        interval,
        bool_level_shp,
        bool_comb_tif,
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
#[pyo3(signature = (in_dem, min_size, min_height, interval, out_dir, bool_shp=false, compress=false))]
fn delineate_mounts<'py>(
    py: Python<'py>,
    in_dem: &str,
    min_size: usize,
    min_height: f64,
    interval: f64,
    out_dir: &str,
    bool_shp: bool,
    compress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = DelineateOptions {
        min_size,
        min_depth: min_height,
        interval,
        bool_level_shp: bool_shp,
        bool_comb_tif: false,
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
    dict.set_item("csv_path", res.mounts.csv_path.to_string_lossy().as_ref())?;
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
#[pyo3(signature = (in_lidar, out_dsm, resolution=1.0, radius=None, min_z=None, max_z=None, stat="max", method="statistics", window_size=0))]
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
) -> PyResult<(usize, usize)> {
    let opts = LidarToDsmOptions {
        resolution,
        radius,
        min_z,
        max_z,
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
        recs.append(d)?;
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
#[pyo3(signature = (in_dem, in_flood_dir, out_gif, frame_delay_ms=200, ping_pong=true, loops=0, water_alpha=0.55, max_dimension=None, time_step_h=Some(1.0), rain_intensity_cm_h=None, azimuth_deg=135.0, elevation_deg=45.0))]
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

#[pymodule]
fn _lidar_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", lidar::VERSION)?;
    m.add_function(wrap_pyfunction!(mean_filter_array, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_array, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter_array, m)?)?;
    m.add_function(wrap_pyfunction!(flip_dem_array, m)?)?;
    m.add_function(wrap_pyfunction!(mean_filter_file, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_file, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter_file, m)?)?;
    m.add_function(wrap_pyfunction!(flip_dem_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sinks, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_depressions, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_mounts, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_catchments, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_flow_paths, m)?)?;
    m.add_function(wrap_pyfunction!(delineate_catchment_hierarchy, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_inundation, m)?)?;
    m.add_function(wrap_pyfunction!(animate_inundation, m)?)?;
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
    m.add_function(wrap_pyfunction!(lidar_to_dsm, m)?)?;
    Ok(())
}
