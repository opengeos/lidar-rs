//! WebAssembly (`wasm-bindgen`) bindings for `lidar-rs`.
//!
//! Thin marshalling layer operating completely in memory on GeoTIFF / GeoJSON / GIF
//! byte buffers (`Uint8Array`) with zero filesystem or GDAL dependencies, for browsers
//! and Node.js. Core hydrology and delineation algorithms live in `crates/lidar`.

use richdem::{Array2D, DemNum, Topology, array2d_to_dataset, dataset_to_dynamic};
use serde::Serialize;
use terrio::formats::RasterDriver;
use terrio::memory::{open_raster_bytes, write_raster_bytes, write_vector_bytes};
use terrio::ops::polygonize::{PolygonizeOptions, polygonize as terrio_polygonize};
use terrio::{RasterWriteOptions, VectorWriteOptions};
use wasm_bindgen::prelude::*;

use lidar::{
    ANIMATION_BANNER_HEIGHT, AnimateInundationOptions, HillshadeOptions, InundationPipelineOptions,
    SimulateInundationOptions, animate_inundation_to_bytes, compute_hillshade_rgb,
    delineate_catchments_in_memory, delineate_depressions_in_memory, delineate_mounts_in_memory,
    extract_sink_labels_from_raster, extract_sinks_in_memory, flip_dem as core_flip_dem,
    gaussian_filter as core_gaussian_filter, mean_filter as core_mean_filter,
    median_filter as core_median_filter, simulate_inundation_pipeline_in_memory,
};

/// Upper bound on total pixels (`frames * width * height`) encoded in a single synchronous
/// browser GIF export to prevent tab freezes on extreme step counts.
const MAX_WASM_GIF_FRAME_PIXELS: u64 = 40_000_000;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn to_js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn read_array2d_f64(bytes: &[u8], filename: &str) -> Result<Array2D<f64>, JsValue> {
    let ds = open_raster_bytes(bytes.to_vec(), filename).map_err(to_js_err)?;
    let dyn_arr = dataset_to_dynamic(ds, filename).map_err(to_js_err)?;
    Ok(dyn_arr.to_f64())
}

fn write_array2d_tif<T: DemNum>(arr: &Array2D<T>, name: &str) -> Result<Vec<u8>, JsValue> {
    let ds = array2d_to_dataset(arr).map_err(to_js_err)?;
    write_raster_bytes(&ds, name, &RasterWriteOptions::default()).map_err(to_js_err)
}

fn polygonize_array_to_geojson<T: DemNum>(
    arr: &Array2D<T>,
    field_name: &str,
) -> Result<String, JsValue> {
    let ds = array2d_to_dataset(arr).map_err(to_js_err)?;
    let opts = PolygonizeOptions {
        band: 0,
        field_name: field_name.to_string(),
        layer_name: "features".to_string(),
        connectedness8: false,
        ignore_nodata: false,
        mask: None,
    };
    let layer = terrio_polygonize(&ds, &opts).map_err(to_js_err)?;
    let bytes = write_vector_bytes(&[layer], "features.geojson", &VectorWriteOptions::default())
        .map_err(to_js_err)?;
    String::from_utf8(bytes).map_err(to_js_err)
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let (chunks, _) = rgb.as_chunks::<3>();
    let mut out = Vec::with_capacity(chunks.len() * 4);
    for &[r, g, b] in chunks {
        out.push(r);
        out.push(g);
        out.push(b);
        out.push(255);
    }
    out
}

/// Simple viridis-like ramp (`t` in `[0, 1]`).
fn viridis_rgb(t: f64) -> [u8; 3] {
    const STOPS: [(f64, [f64; 3]); 5] = [
        (0.00, [68.0, 1.0, 84.0]),
        (0.25, [59.0, 82.0, 139.0]),
        (0.50, [33.0, 145.0, 140.0]),
        (0.75, [94.0, 201.0, 98.0]),
        (1.00, [253.0, 231.0, 37.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return [
                (c0[0] + f * (c1[0] - c0[0])).round() as u8,
                (c0[1] + f * (c1[1] - c0[1])).round() as u8,
                (c0[2] + f * (c1[2] - c0[2])).round() as u8,
            ];
        }
    }
    [253, 231, 37]
}

/// Blues ramp (`t` in `[0, 1]`).
fn blues_rgb(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (222.0 * (1.0 - t) + 8.0 * t).round() as u8;
    let g = (235.0 * (1.0 - t) + 48.0 * t).round() as u8;
    let b = (247.0 * (1.0 - t) + 107.0 * t).round() as u8;
    [r, g, b]
}

/// Colorizes a floating-point raster to RGBA (`0` alpha for invalid/NoData cells).
///
/// When `include_zero` is `true`, values `>= 0.0` (excluding `arr.no_data()`) are rendered
/// (appropriate for fill-time rasters where `0.0 h` is wet and `-1.0` is NoData). When
/// `false`, only strictly positive values (`> 0.0`) are rendered (appropriate for depth).
fn colorize_f64_rgba(arr: &Array2D<f64>, blues: bool, invert: bool, include_zero: bool) -> Vec<u8> {
    let nd = arr.no_data();
    let is_valid = |v: f64| -> bool {
        v.is_finite() && !v.rd_eq(nd) && if include_zero { v >= 0.0 } else { v > 0.0 }
    };
    let data = arr.data();
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in data {
        if is_valid(v) {
            if v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
        }
    }
    let span = (vmax - vmin).max(1e-12);
    let mut out = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        if !is_valid(v) {
            continue;
        }
        let mut t = ((v - vmin) / span).clamp(0.0, 1.0);
        if invert {
            t = 1.0 - t;
        }
        let c = if blues { blues_rgb(t) } else { viridis_rgb(t) };
        let p = i * 4;
        out[p] = c[0];
        out[p + 1] = c[1];
        out[p + 2] = c[2];
        out[p + 3] = 255;
    }
    out
}

fn colorize_i32_rgba(arr: &Array2D<i32>) -> Vec<u8> {
    let data = arr.data();
    let vmax = data.iter().copied().max().unwrap_or(0).max(1) as f64;
    let mut out = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        if v <= 0 {
            continue;
        }
        let t = (v as f64 / vmax).clamp(0.0, 1.0);
        let c = viridis_rgb(t);
        let p = i * 4;
        out[p] = c[0];
        out[p + 1] = c[1];
        out[p + 2] = c[2];
        out[p + 3] = 255;
    }
    out
}

#[derive(Serialize)]
struct RasterInfoJson {
    width: usize,
    height: usize,
    geotransform: [f64; 6],
    projection: String,
    no_data: f64,
    min: f64,
    max: f64,
}

/// Returns the `lidar-rs` crate version string.
#[wasm_bindgen]
pub fn version() -> String {
    lidar::VERSION.to_string()
}

/// Returns basic raster metadata and min/max elevation as a JSON string.
#[wasm_bindgen(js_name = rasterInfo)]
pub fn raster_info(bytes: &[u8], filename: &str) -> Result<String, JsValue> {
    let arr = read_array2d_f64(bytes, filename)?;
    let nd = arr.no_data();
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in arr.data() {
        if v.is_finite() && !v.rd_eq(nd) {
            if v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
        }
    }
    let info = RasterInfoJson {
        width: arr.width(),
        height: arr.height(),
        geotransform: arr.geotransform,
        projection: arr.projection,
        no_data: if nd.is_finite() { nd } else { f64::NAN },
        min: if vmin.is_finite() { vmin } else { 0.0 },
        max: if vmax.is_finite() { vmax } else { 0.0 },
    };
    serde_json::to_string(&info).map_err(to_js_err)
}

/// Applies a moving-window mean filter to a GeoTIFF byte buffer and returns GeoTIFF bytes.
#[wasm_bindgen(js_name = meanFilter)]
pub fn mean_filter(bytes: &[u8], filename: &str, kernel_size: usize) -> Result<Vec<u8>, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let out = core_mean_filter(&dem, kernel_size).map_err(to_js_err)?;
    write_array2d_tif(&out, "mean.tif")
}

/// Applies a moving-window median filter to a GeoTIFF byte buffer and returns GeoTIFF bytes.
#[wasm_bindgen(js_name = medianFilter)]
pub fn median_filter(bytes: &[u8], filename: &str, kernel_size: usize) -> Result<Vec<u8>, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let out = core_median_filter(&dem, kernel_size).map_err(to_js_err)?;
    write_array2d_tif(&out, "median.tif")
}

/// Applies a 2D Gaussian filter to a GeoTIFF byte buffer and returns GeoTIFF bytes.
#[wasm_bindgen(js_name = gaussianFilter)]
pub fn gaussian_filter(bytes: &[u8], filename: &str, sigma: f64) -> Result<Vec<u8>, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let out = core_gaussian_filter(&dem, sigma).map_err(to_js_err)?;
    write_array2d_tif(&out, "gaussian.tif")
}

/// Flips a DEM upside-down (`-dem + max + delta`) and returns GeoTIFF bytes.
#[wasm_bindgen(js_name = flipDem)]
pub fn flip_dem(bytes: &[u8], filename: &str, delta: f64) -> Result<Vec<u8>, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let out = core_flip_dem(&dem, delta).map_err(to_js_err)?;
    write_array2d_tif(&out, "dem_flip.tif")
}

/// Computes an RGBA shaded-relief preview buffer (`width * height * 4` bytes) suitable
/// for direct display in an HTML5 `<canvas>` via `ImageData`.
#[wasm_bindgen(js_name = hillshadeRgba)]
pub fn hillshade_rgba(
    bytes: &[u8],
    filename: &str,
    azimuth_deg: f64,
    elevation_deg: f64,
    z_factor: f64,
) -> Result<Vec<u8>, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let rgb = compute_hillshade_rgb(
        &dem,
        &HillshadeOptions {
            azimuth_deg,
            elevation_deg,
            z_factor,
            ..Default::default()
        },
    );
    Ok(rgb_to_rgba(&rgb))
}

/// Converts a raster byte buffer into a Cloud Optimized GeoTIFF (COG) byte buffer.
#[wasm_bindgen(js_name = imageToCog)]
pub fn image_to_cog(bytes: &[u8], filename: &str, compression: &str) -> Result<Vec<u8>, JsValue> {
    let ds = open_raster_bytes(bytes.to_vec(), filename).map_err(to_js_err)?;
    let opts = RasterWriteOptions {
        driver: Some(RasterDriver::Cog),
        compression: Some(compression.to_string()),
        ..Default::default()
    };
    write_raster_bytes(&ds, "output_cog.tif", &opts).map_err(to_js_err)
}

/// Polygonizes a raster byte buffer into a GeoJSON string.
#[wasm_bindgen(js_name = polygonizeToGeojson)]
pub fn polygonize_to_geojson(
    bytes: &[u8],
    filename: &str,
    field_name: &str,
) -> Result<String, JsValue> {
    let ds = open_raster_bytes(bytes.to_vec(), filename).map_err(to_js_err)?;
    let opts = PolygonizeOptions {
        band: 0,
        field_name: field_name.to_string(),
        layer_name: "features".to_string(),
        connectedness8: false,
        ignore_nodata: false,
        mask: None,
    };
    let layer = terrio_polygonize(&ds, &opts).map_err(to_js_err)?;
    let out = write_vector_bytes(&[layer], "features.geojson", &VectorWriteOptions::default())
        .map_err(to_js_err)?;
    String::from_utf8(out).map_err(to_js_err)
}

/// Output container returned by [`extract_sinks`].
#[wasm_bindgen]
pub struct SinksOutput {
    width: usize,
    height: usize,
    depressions_json: String,
    sink_tif: Vec<u8>,
    refined_dem_tif: Vec<u8>,
    depth_tif: Vec<u8>,
    region_tif: Vec<u8>,
    geojson: String,
    depth_rgba: Vec<u8>,
}

#[wasm_bindgen]
impl SinksOutput {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.height
    }
    #[wasm_bindgen(js_name = depressionsJson)]
    pub fn depressions_json(&self) -> String {
        self.depressions_json.clone()
    }
    #[wasm_bindgen(js_name = sinkTif)]
    pub fn sink_tif(&self) -> Vec<u8> {
        self.sink_tif.clone()
    }
    #[wasm_bindgen(js_name = refinedDemTif)]
    pub fn refined_dem_tif(&self) -> Vec<u8> {
        self.refined_dem_tif.clone()
    }
    #[wasm_bindgen(js_name = depthTif)]
    pub fn depth_tif(&self) -> Vec<u8> {
        self.depth_tif.clone()
    }
    #[wasm_bindgen(js_name = regionTif)]
    pub fn region_tif(&self) -> Vec<u8> {
        self.region_tif.clone()
    }
    pub fn geojson(&self) -> String {
        self.geojson.clone()
    }
    #[wasm_bindgen(js_name = depthRgba)]
    pub fn depth_rgba(&self) -> Vec<u8> {
        self.depth_rgba.clone()
    }
}

/// Extracts surface depressions (sinks) from a DEM GeoTIFF byte buffer in memory.
#[wasm_bindgen(js_name = extractSinks)]
pub fn extract_sinks(
    bytes: &[u8],
    filename: &str,
    min_size: usize,
) -> Result<SinksOutput, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let mem = extract_sinks_in_memory(&dem, min_size, Topology::D8).map_err(to_js_err)?;
    let depressions_json = serde_json::to_string(&mem.depressions).map_err(to_js_err)?;
    let sink_tif = write_array2d_tif(&mem.sink, "sink.tif")?;
    let refined_dem_tif = write_array2d_tif(&mem.dem_refined, "dem_refined.tif")?;
    let depth_tif = write_array2d_tif(&mem.depth, "depth.tif")?;
    let region_tif = write_array2d_tif(&mem.region, "region.tif")?;
    let geojson = polygonize_array_to_geojson(&mem.region, "region_id")?;
    let depth_rgba = colorize_f64_rgba(&mem.depth, true, false, false);
    Ok(SinksOutput {
        width: dem.width(),
        height: dem.height(),
        depressions_json,
        sink_tif,
        refined_dem_tif,
        depth_tif,
        region_tif,
        geojson,
        depth_rgba,
    })
}

/// Output container returned by [`delineate_depressions`] and [`delineate_mounts`].
#[wasm_bindgen]
pub struct DepressionsOutput {
    width: usize,
    height: usize,
    max_level: u32,
    num_regions: usize,
    depressions_json: String,
    id_tif: Vec<u8>,
    level_tif: Vec<u8>,
    geojson: String,
    level_rgba: Vec<u8>,
}

#[wasm_bindgen]
impl DepressionsOutput {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.height
    }
    #[wasm_bindgen(getter, js_name = maxLevel)]
    pub fn max_level(&self) -> u32 {
        self.max_level
    }
    #[wasm_bindgen(getter, js_name = numRegions)]
    pub fn num_regions(&self) -> usize {
        self.num_regions
    }
    #[wasm_bindgen(js_name = depressionsJson)]
    pub fn depressions_json(&self) -> String {
        self.depressions_json.clone()
    }
    #[wasm_bindgen(js_name = idTif)]
    pub fn id_tif(&self) -> Vec<u8> {
        self.id_tif.clone()
    }
    #[wasm_bindgen(js_name = levelTif)]
    pub fn level_tif(&self) -> Vec<u8> {
        self.level_tif.clone()
    }
    pub fn geojson(&self) -> String {
        self.geojson.clone()
    }
    #[wasm_bindgen(js_name = levelRgba)]
    pub fn level_rgba(&self) -> Vec<u8> {
        self.level_rgba.clone()
    }
}

fn marshal_depressions(
    mem: &lidar::InMemoryDepressions,
    id_name: &str,
    level_name: &str,
) -> Result<DepressionsOutput, JsValue> {
    let depressions_json = serde_json::to_string(&mem.depressions).map_err(to_js_err)?;
    let id_tif = write_array2d_tif(&mem.depression_id, id_name)?;
    let level_tif = write_array2d_tif(&mem.depression_level, level_name)?;
    let geojson = polygonize_array_to_geojson(&mem.depression_id, "id")?;
    let level_rgba = colorize_i32_rgba(&mem.depression_level);
    Ok(DepressionsOutput {
        width: mem.depression_id.width(),
        height: mem.depression_id.height(),
        max_level: mem.max_level,
        num_regions: mem.num_regions as usize,
        depressions_json,
        id_tif,
        level_tif,
        geojson,
        level_rgba,
    })
}

/// Delineates the nested depression hierarchy (Wu et al. 2019) from a sink GeoTIFF buffer.
#[wasm_bindgen(js_name = delineateDepressions)]
pub fn delineate_depressions(
    sink_bytes: &[u8],
    filename: &str,
    min_size: usize,
    min_depth: f64,
    interval: f64,
) -> Result<DepressionsOutput, JsValue> {
    let sink = read_array2d_f64(sink_bytes, filename)?;
    let mem =
        delineate_depressions_in_memory(&sink, min_size, min_depth, interval).map_err(to_js_err)?;
    marshal_depressions(&mem, "depression_id.tif", "depression_level.tif")
}

/// Delineates nested elevated features (mounts / buildings / tree canopies) from a DSM buffer.
#[wasm_bindgen(js_name = delineateMounts)]
pub fn delineate_mounts(
    dsm_bytes: &[u8],
    filename: &str,
    min_size: usize,
    min_height: f64,
    interval: f64,
) -> Result<DepressionsOutput, JsValue> {
    let dsm = read_array2d_f64(dsm_bytes, filename)?;
    let mem =
        delineate_mounts_in_memory(&dsm, min_size, min_height, interval).map_err(to_js_err)?;
    marshal_depressions(&mem, "mount_id.tif", "mount_level.tif")
}

/// Output container returned by [`delineate_catchments`].
#[wasm_bindgen]
pub struct CatchmentsOutput {
    records_json: String,
    watershed_tif: Vec<u8>,
    geojson: String,
}

#[wasm_bindgen]
impl CatchmentsOutput {
    #[wasm_bindgen(js_name = recordsJson)]
    pub fn records_json(&self) -> String {
        self.records_json.clone()
    }
    #[wasm_bindgen(js_name = watershedTif)]
    pub fn watershed_tif(&self) -> Vec<u8> {
        self.watershed_tif.clone()
    }
    pub fn geojson(&self) -> String {
        self.geojson.clone()
    }
}

/// Delineates contributing catchments from a refined DEM and a sink/region raster buffer.
///
/// Validates grid dimensions and geotransform alignment, preserving dense `1..=N` region IDs
/// and falling back to 4-connected grouping for raw elevation sink rasters.
#[wasm_bindgen(js_name = delineateCatchments)]
pub fn delineate_catchments(
    dem_bytes: &[u8],
    sink_or_region_bytes: &[u8],
    dem_filename: &str,
    sink_filename: &str,
) -> Result<CatchmentsOutput, JsValue> {
    let dem = read_array2d_f64(dem_bytes, dem_filename)?;
    let sink_arr = read_array2d_f64(sink_or_region_bytes, sink_filename)?;
    let labels = extract_sink_labels_from_raster(&dem, &sink_arr).map_err(to_js_err)?;
    let mem = delineate_catchments_in_memory(&dem, &labels).map_err(to_js_err)?;
    let records_json = serde_json::to_string(&mem.records).map_err(to_js_err)?;
    let watershed_tif = write_array2d_tif(&mem.watershed, "watershed.tif")?;
    let geojson = polygonize_array_to_geojson(&mem.watershed, "id")?;
    Ok(CatchmentsOutput {
        records_json,
        watershed_tif,
        geojson,
    })
}

/// Output container returned by [`simulate_inundation`].
///
/// Returns live playback data (`fillTimeValues`, `hillshadeRgba`, `stepsJson`, spatial previews,
/// and downloadable rasters/GeoJSON) immediately without blocking on GIF quantization. Call
/// [`InundationOutput::encode_gif`] on demand to generate an animated GIF byte buffer.
#[wasm_bindgen]
pub struct InundationOutput {
    width: usize,
    height: usize,
    projection: String,
    max_level: u32,
    num_regions: usize,
    num_steps: usize,
    time_step_h: f64,
    rain_intensity_cm_h: f64,
    dem_refined: Array2D<f64>,
    fill_time: Array2D<f64>,
    depressions_json: String,
    catchment_hierarchy_json: String,
    steps_json: String,
    fill_time_tif: Vec<u8>,
    catchment_hir_tif: Vec<u8>,
    sink_tif: Vec<u8>,
    level_tif: Vec<u8>,
    depressions_geojson: String,
    hillshade_rgba: Vec<u8>,
    depth_rgba: Vec<u8>,
    level_rgba: Vec<u8>,
    fill_time_rgba: Vec<u8>,
    fill_time_values: Vec<f64>,
}

#[wasm_bindgen]
impl InundationOutput {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.height
    }
    #[wasm_bindgen(getter)]
    pub fn projection(&self) -> String {
        self.projection.clone()
    }
    #[wasm_bindgen(getter, js_name = maxLevel)]
    pub fn max_level(&self) -> u32 {
        self.max_level
    }
    #[wasm_bindgen(getter, js_name = numRegions)]
    pub fn num_regions(&self) -> usize {
        self.num_regions
    }
    #[wasm_bindgen(getter, js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }
    #[wasm_bindgen(js_name = depressionsJson)]
    pub fn depressions_json(&self) -> String {
        self.depressions_json.clone()
    }
    #[wasm_bindgen(js_name = catchmentHierarchyJson)]
    pub fn catchment_hierarchy_json(&self) -> String {
        self.catchment_hierarchy_json.clone()
    }
    #[wasm_bindgen(js_name = stepsJson)]
    pub fn steps_json(&self) -> String {
        self.steps_json.clone()
    }
    #[wasm_bindgen(js_name = fillTimeTif)]
    pub fn fill_time_tif(&self) -> Vec<u8> {
        self.fill_time_tif.clone()
    }
    #[wasm_bindgen(js_name = catchmentHirTif)]
    pub fn catchment_hir_tif(&self) -> Vec<u8> {
        self.catchment_hir_tif.clone()
    }
    #[wasm_bindgen(js_name = sinkTif)]
    pub fn sink_tif(&self) -> Vec<u8> {
        self.sink_tif.clone()
    }
    #[wasm_bindgen(js_name = levelTif)]
    pub fn level_tif(&self) -> Vec<u8> {
        self.level_tif.clone()
    }
    #[wasm_bindgen(js_name = depressionsGeojson)]
    pub fn depressions_geojson(&self) -> String {
        self.depressions_geojson.clone()
    }
    #[wasm_bindgen(js_name = hillshadeRgba)]
    pub fn hillshade_rgba(&self) -> Vec<u8> {
        self.hillshade_rgba.clone()
    }
    #[wasm_bindgen(js_name = depthRgba)]
    pub fn depth_rgba(&self) -> Vec<u8> {
        self.depth_rgba.clone()
    }
    #[wasm_bindgen(js_name = levelRgba)]
    pub fn level_rgba(&self) -> Vec<u8> {
        self.level_rgba.clone()
    }
    #[wasm_bindgen(js_name = fillTimeRgba)]
    pub fn fill_time_rgba(&self) -> Vec<u8> {
        self.fill_time_rgba.clone()
    }
    /// Full-precision (`Float64Array`) per-cell fill times in hours (`-1.0` for dry/NoData).
    #[wasm_bindgen(js_name = fillTimeValues)]
    pub fn fill_time_values(&self) -> Vec<f64> {
        self.fill_time_values.clone()
    }
    /// Encodes a shaded-relief animated inundation GIF on demand, guarding against
    /// excessive total frame-pixel counts in synchronous browser execution.
    #[wasm_bindgen(js_name = encodeGif)]
    pub fn encode_gif(
        &self,
        frame_delay_ms: u16,
        ping_pong: bool,
        max_dim: usize,
    ) -> Result<Vec<u8>, JsValue> {
        let total_frames = if ping_pong && self.num_steps > 1 {
            self.num_steps.saturating_mul(2).saturating_sub(1)
        } else {
            self.num_steps
        };
        let (dw, dh) = if max_dim > 0 && (self.width > max_dim || self.height > max_dim) {
            if self.width >= self.height {
                let nh = ((self.height as f64) * (max_dim as f64) / (self.width as f64))
                    .round()
                    .max(1.0) as usize;
                (max_dim, nh)
            } else {
                let nw = ((self.width as f64) * (max_dim as f64) / (self.height as f64))
                    .round()
                    .max(1.0) as usize;
                (nw, max_dim)
            }
        } else {
            (self.width, self.height)
        };
        let total_pixels = (total_frames as u64)
            .saturating_mul(dw as u64)
            .saturating_mul((dh + ANIMATION_BANNER_HEIGHT) as u64);
        if total_pixels > MAX_WASM_GIF_FRAME_PIXELS {
            return Err(to_js_err(format!(
                "GIF export requires {total_frames} frames at {dw}x{} ({total_pixels} frame-pixels), \
                 exceeding the browser budget of {MAX_WASM_GIF_FRAME_PIXELS}; increase time_step, \
                 reduce duration, or set a smaller max_dim",
                dh + ANIMATION_BANNER_HEIGHT
            )));
        }

        let anim_opts = AnimateInundationOptions {
            frame_delay_ms,
            ping_pong,
            loops: 0,
            max_dimension: if max_dim > 0 { Some(max_dim) } else { None },
            time_step_h: Some(self.time_step_h),
            rain_intensity_cm_h: Some(self.rain_intensity_cm_h),
            ..Default::default()
        };
        let (gif_bytes, _, _, _) = animate_inundation_to_bytes(
            &self.dem_refined,
            &self.fill_time,
            self.num_steps,
            self.time_step_h,
            &anim_opts,
        )
        .map_err(to_js_err)?;
        Ok(gif_bytes)
    }
}

/// Runs the full in-memory hydrology & inundation pipeline:
/// optional median filter → sink extraction → nested depressions → catchment hierarchy →
/// single-pass inundation simulation.
///
/// GIF encoding is deferred to [`InundationOutput::encode_gif`] so live playback starts
/// immediately without blocking on frame quantization.
#[wasm_bindgen(js_name = simulateInundation)]
#[allow(clippy::too_many_arguments)]
pub fn simulate_inundation(
    dem_bytes: &[u8],
    filename: &str,
    kernel_size: usize,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    rain_intensity_cm_h: f64,
    rain_duration_h: f64,
    time_step_h: f64,
) -> Result<InundationOutput, JsValue> {
    let raw_dem = read_array2d_f64(dem_bytes, filename)?;
    let opts = InundationPipelineOptions {
        kernel_size,
        simulation: SimulateInundationOptions {
            min_size,
            min_depth,
            interval,
            rain_intensity_cm_h,
            rain_duration_h,
            time_step_h,
            compress: false,
        },
    };
    let pipe = simulate_inundation_pipeline_in_memory(&raw_dem, &opts).map_err(to_js_err)?;

    let hillshade_rgb =
        compute_hillshade_rgb(&pipe.sinks.dem_refined, &HillshadeOptions::default());
    let hillshade_rgba = rgb_to_rgba(&hillshade_rgb);
    let depth_rgba = colorize_f64_rgba(&pipe.sinks.depth, true, false, false);
    let level_rgba = colorize_i32_rgba(&pipe.depressions.depression_level);
    let fill_time_rgba = colorize_f64_rgba(&pipe.fill_time, false, true, true);

    let ft_nd = pipe.fill_time.no_data();
    let fill_time_values: Vec<f64> = pipe
        .fill_time
        .data()
        .iter()
        .map(|&v| {
            if v.is_finite() && !v.rd_eq(ft_nd) && v >= 0.0 {
                v
            } else {
                -1.0
            }
        })
        .collect();

    let num_steps = pipe.steps.len();
    let depressions_json =
        serde_json::to_string(&pipe.depressions.depressions).map_err(to_js_err)?;
    let catchment_hierarchy_json =
        serde_json::to_string(&pipe.catchment_records).map_err(to_js_err)?;
    let steps_json = serde_json::to_string(&pipe.steps).map_err(to_js_err)?;
    let fill_time_tif = write_array2d_tif(&pipe.fill_time, "fill_time.tif")?;
    let catchment_hir_tif = write_array2d_tif(&pipe.catchment_hierarchy, "catchment_hir.tif")?;
    let sink_tif = write_array2d_tif(&pipe.sinks.sink, "sink.tif")?;
    let level_tif = write_array2d_tif(&pipe.depressions.depression_level, "depression_level.tif")?;
    let depressions_geojson = polygonize_array_to_geojson(&pipe.depressions.depression_id, "id")?;

    Ok(InundationOutput {
        width: raw_dem.width(),
        height: raw_dem.height(),
        projection: raw_dem.projection,
        max_level: pipe.depressions.max_level,
        num_regions: pipe.num_regions,
        num_steps,
        time_step_h,
        rain_intensity_cm_h,
        dem_refined: pipe.sinks.dem_refined,
        fill_time: pipe.fill_time,
        depressions_json,
        catchment_hierarchy_json,
        steps_json,
        fill_time_tif,
        catchment_hir_tif,
        sink_tif,
        level_tif,
        depressions_geojson,
        hillshade_rgba,
        depth_rgba,
        level_rgba,
        fill_time_rgba,
        fill_time_values,
    })
}
