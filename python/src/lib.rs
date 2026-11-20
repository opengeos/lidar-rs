//! PyO3 bindings exposing the `lidar-rs` API to Python with zero GDAL dependencies.

use std::path::{Path, PathBuf};

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use lidar::pointcloud::{GridMethod, GridStat, LidarToDsmOptions};
use lidar::{
    Array2D, DelineateOptions, ExtractSinksOptions, NestedDepression, SinkDepression, Topology,
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

fn from_py_array2(
    arr: PyReadonlyArray2<'_, f64>,
    no_data: Option<f64>,
) -> PyResult<Array2D<f64>> {
    let view = arr.as_array();
    let (h, w) = (view.nrows(), view.ncols());
    let data: Vec<f64> = view.iter().copied().collect();
    let mut a = Array2D::from_vec(w, h, data).map_err(|e| to_py_err(e.into()))?;
    a.set_no_data(no_data.unwrap_or(f64::NAN));
    Ok(a)
}

fn to_py_array2<'py>(
    py: Python<'py>,
    arr: Array2D<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
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

fn nested_dep_to_dict<'py>(
    py: Python<'py>,
    d: &NestedDepression,
) -> PyResult<Bound<'py, PyDict>> {
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

#[pyfunction]
#[pyo3(signature = (arr, kernel_size=3, no_data=None))]
fn mean_filter_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'_, f64>,
    kernel_size: usize,
    no_data: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dem = from_py_array2(arr, no_data)?;
    let out = lidar::mean_filter(&dem, kernel_size).map_err(to_py_err)?;
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
    let out = lidar::median_filter(&dem, kernel_size).map_err(to_py_err)?;
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
    let out = lidar::gaussian_filter(&dem, sigma).map_err(to_py_err)?;
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
    let out = lidar::flip_dem(&dem, delta).map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, kernel_size=3, out_file=None))]
fn mean_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    kernel_size: usize,
    out_file: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = lidar::mean_filter_file(in_dem, kernel_size, out_file.map(Path::new))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, kernel_size=3, out_file=None))]
fn median_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    kernel_size: usize,
    out_file: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = lidar::median_filter_file(in_dem, kernel_size, out_file.map(Path::new))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, sigma=1.0, out_file=None))]
fn gaussian_filter_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    sigma: f64,
    out_file: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = lidar::gaussian_filter_file(in_dem, sigma, out_file.map(Path::new))
        .map_err(to_py_err)?;
    to_py_array2(py, out)
}

#[pyfunction]
#[pyo3(signature = (in_dem, delta=100.0, out_file=None))]
fn flip_dem_file<'py>(
    py: Python<'py>,
    in_dem: &str,
    delta: f64,
    out_file: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = lidar::flip_dem_file(in_dem, delta, out_file.map(Path::new)).map_err(to_py_err)?;
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
    let res = lidar::extract_sinks(in_dem, out_dir, &opts).map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("sink_path", res.sink_path.to_string_lossy().as_ref())?;
    dict.set_item("refined_dem_path", res.refined_dem_path.to_string_lossy().as_ref())?;
    dict.set_item("filled_dem_path", res.filled_dem_path.to_string_lossy().as_ref())?;
    dict.set_item("dem_diff_path", res.dem_diff_path.to_string_lossy().as_ref())?;
    dict.set_item("depth_path", res.depth_path.to_string_lossy().as_ref())?;
    dict.set_item("region_path", res.region_path.to_string_lossy().as_ref())?;
    dict.set_item("csv_path", res.csv_path.to_string_lossy().as_ref())?;
    dict.set_item("shapefile_path", res.shapefile_path.to_string_lossy().as_ref())?;
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
    let res = lidar::delineate_depressions(in_sink, out_dir, &opts).map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "depression_id_path",
        res.depression_id_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item(
        "depression_level_path",
        res.depression_level_path.to_string_lossy().as_ref(),
    )?;
    dict.set_item("shapefile_path", res.shapefile_path.to_string_lossy().as_ref())?;
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
    let res = lidar::delineate_mounts(in_dem, out_dir, &opts).map_err(to_py_err)?;

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
fn polygonize(in_raster: &str, out_vector: &str, field_name: &str) -> PyResult<usize> {
    lidar::polygonize(in_raster, out_vector, field_name).map_err(to_py_err)
}

#[pyfunction]
fn mosaic(inputs: Vec<String>, output: &str) -> PyResult<()> {
    lidar::mosaic(&inputs, output).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, output, dst_crs="EPSG:4326", resampling="nearest", resolution=None))]
fn reproject_image(
    input: &str,
    output: &str,
    dst_crs: &str,
    resampling: &str,
    resolution: Option<(f64, f64)>,
) -> PyResult<()> {
    let r = lidar::parse_resampling(resampling).map_err(to_py_err)?;
    lidar::reproject_image(input, output, dst_crs, r, resolution).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, output, resolution, resampling="bilinear"))]
fn resample(input: &str, output: &str, resolution: f64, resampling: &str) -> PyResult<()> {
    let r = lidar::parse_resampling(resampling).map_err(to_py_err)?;
    lidar::resample(input, output, resolution, r).map_err(to_py_err)
}

#[pyfunction]
fn clip_image(input: &str, mask: &str, output: &str) -> PyResult<()> {
    lidar::clip_image(input, mask, output).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, output, compression="DEFLATE"))]
fn image_to_cog(input: &str, output: &str, compression: &str) -> PyResult<()> {
    lidar::image_to_cog(input, output, Some(compression)).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (input, crs, output=None))]
fn add_crs(input: &str, crs: &str, output: Option<&str>) -> PyResult<()> {
    lidar::add_crs(input, crs, output.map(Path::new)).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_vector, in_csv, out_vector, vec_col="id", csv_col="region_id"))]
fn join_tables(
    in_vector: &str,
    in_csv: &str,
    out_vector: &str,
    vec_col: &str,
    csv_col: &str,
) -> PyResult<usize> {
    lidar::join_csv_to_vector(in_vector, in_csv, out_vector, vec_col, csv_col)
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_csv, out_vector, latitude="latitude", longitude="longitude"))]
fn csv_to_vector(
    in_csv: &str,
    out_vector: &str,
    latitude: &str,
    longitude: &str,
) -> PyResult<usize> {
    lidar::csv_points_to_vector(in_csv, out_vector, latitude, longitude).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (source, destination, point_format_id=None, minor_version=None))]
fn convert_lidar(
    source: &str,
    destination: &str,
    point_format_id: Option<u8>,
    minor_version: Option<u8>,
) -> PyResult<usize> {
    lidar::convert_lidar(source, destination, point_format_id, minor_version).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (in_lidar, out_dsm, resolution=1.0, radius=None, min_z=None, max_z=None, stat="max", method="statistics", window_size=0))]
#[allow(clippy::too_many_arguments)]
fn lidar_to_dsm(
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
    lidar::lidar_to_dsm(in_lidar, out_dsm, &opts).map_err(to_py_err)
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
