//! WebAssembly (`wasm-bindgen`) bindings for `lidar-rs`.
//!
//! Thin marshalling layer operating completely in memory on GeoTIFF / GeoJSON / GIF
//! byte buffers (`Uint8Array`) with zero filesystem or GDAL dependencies, for browsers
//! and Node.js. Core hydrology and delineation algorithms live in `crates/lidar`.

use std::collections::HashMap;

use richdem::{Array2D, DemNum, Topology, array2d_to_dataset, dataset_to_dynamic};
use serde::Serialize;
use terrio::formats::RasterDriver;
use terrio::memory::{open_raster_bytes, write_raster_bytes, write_vector_bytes};
use terrio::ops::polygonize::{PolygonizeOptions, polygonize as terrio_polygonize};
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, GeometryType, Layer, RasterWriteOptions,
    VectorWriteOptions,
};
use wasm_bindgen::prelude::*;

use lidar::{
    ANIMATION_BANNER_HEIGHT, AnimateInundationOptions, BreachMode, BreachOptions, HillshadeOptions,
    InMemoryFullPipelineOptions, InundationForcing, NestedDepression, RoutingMode,
    SimulateInundationOptions, SinkDepression, SpillFlowPath,
    animate_inundation_to_bytes_with_spill_paths, breach_depressions as core_breach_depressions,
    compute_hillshade_rgb, connectivity_graph_to_graphml_string, delineate_catchments_in_memory,
    delineate_depressions_in_memory, delineate_mounts_in_memory, extract_sink_labels_from_raster,
    extract_sinks_in_memory, flip_dem as core_flip_dem, fsm_basins_to_csv_string,
    gaussian_filter as core_gaussian_filter, mean_filter as core_mean_filter,
    median_filter as core_median_filter, rating_curves_to_csv_string,
    render_inundation_frame_png_bytes, run_full_pipeline_in_memory, scs_type2_hyetograph,
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
    let opts = RasterWriteOptions {
        compression: Some("DEFLATE".to_string()),
        ..Default::default()
    };
    write_raster_bytes(&ds, name, &opts).map_err(to_js_err)
}

/// Reprojects a vector layer to RFC 7946 WGS 84 (`EPSG:4326`) when its source CRS is known
/// and reprojection succeeds, omitting the deprecated `crs` member once reprojected. Falls
/// back cleanly to the original layer when the input raster has no CRS metadata or uses an
/// unrecognized local CRS.
fn reproject_layer_to_wgs84_if_known(mut layer: Layer, fallback_projection: &str) -> Layer {
    if layer.crs.is_none() {
        let trimmed = fallback_projection.trim();
        if !trimmed.is_empty()
            && let Ok(srs) = terrio::crs::SpatialRef::parse(trimmed)
        {
            srs.apply_to_layer(&mut layer);
        }
    }
    let src_srs = terrio::crs::SpatialRef::from_layer(&layer);
    if !src_srs.is_unknown() {
        if src_srs.epsg == Some(4326) {
            // RFC 7946 omits the deprecated `crs` member for WGS 84 coordinates.
            layer.crs = None;
        } else if let Ok(wgs84) = terrio::crs::SpatialRef::from_epsg(4326)
            && let Ok(mut reprojected) = terrio::vector::reproject_layer(&layer, &wgs84)
        {
            reprojected.crs = None;
            return reprojected;
        }
    }
    layer
}

/// Computes the WGS 84 (`EPSG:4326`) bounding box `[minLon, minLat, maxLon, maxLat]`
/// of a georeferenced raster by transforming its corners and edge midpoints.
/// Returns `None` when the raster has no CRS metadata, an unknown CRS, or fails reprojection.
fn compute_wgs84_bounds(
    width: usize,
    height: usize,
    geotransform: &[f64; 6],
    projection: &str,
) -> Option<Vec<f64>> {
    if width == 0 || height == 0 {
        return None;
    }
    let trimmed = projection.trim();
    if trimmed.is_empty() {
        return None;
    }
    let src_srs = terrio::crs::SpatialRef::parse(trimmed).ok()?;
    if src_srs.is_unknown() {
        return None;
    }
    let transformer = if src_srs.epsg == Some(4326) {
        None
    } else {
        let wgs84 = terrio::crs::SpatialRef::from_epsg(4326).ok()?;
        Some(terrio::crs::Transformer::new(&src_srs, &wgs84).ok()?)
    };
    let wf = width as f64;
    let hf = height as f64;
    let sample_pts = [
        (0.0, 0.0),
        (wf, 0.0),
        (0.0, hf),
        (wf, hf),
        (wf * 0.5, 0.0),
        (wf * 0.5, hf),
        (0.0, hf * 0.5),
        (wf, hf * 0.5),
    ];
    let mut min_lon = f64::INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    for (c, r) in sample_pts {
        let x = geotransform[0] + c * geotransform[1] + r * geotransform[2];
        let y = geotransform[3] + c * geotransform[4] + r * geotransform[5];
        let (lon, lat) = if let Some(ref tr) = transformer {
            tr.transform(x, y).ok()?
        } else {
            (x, y)
        };
        if !lon.is_finite()
            || !lat.is_finite()
            || !(-180.0..=180.0).contains(&lon)
            || !(-90.0..=90.0).contains(&lat)
        {
            return None;
        }
        min_lon = min_lon.min(lon);
        min_lat = min_lat.min(lat);
        max_lon = max_lon.max(lon);
        max_lat = max_lat.max(lat);
    }
    if min_lon <= max_lon && min_lat <= max_lat {
        Some(vec![min_lon, min_lat, max_lon, max_lat])
    } else {
        None
    }
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
    let layer = reproject_layer_to_wgs84_if_known(layer, &arr.projection);
    let bytes = write_vector_bytes(&[layer], "features.geojson", &VectorWriteOptions::default())
        .map_err(to_js_err)?;
    String::from_utf8(bytes).map_err(to_js_err)
}

fn feature_positive_id(feat: &Feature) -> Option<u32> {
    match feat.attributes.first() {
        Some(FieldValue::Integer(i)) if *i > 0 => u32::try_from(*i).ok(),
        Some(FieldValue::Float(f)) if *f > 0.0 && f.is_finite() => Some(*f as u32),
        _ => None,
    }
}

fn polygonize_with_attributes_to_geojson<A, FId, FAttrs>(
    id_raster: &Array2D<i32>,
    layer_name: &str,
    id_field_name: &str,
    fields: &[(&str, FieldType)],
    items: &[A],
    item_id: FId,
    build_attrs: FAttrs,
) -> Result<String, JsValue>
where
    FId: Fn(&A) -> u32,
    FAttrs: Fn(&A) -> Vec<FieldValue>,
{
    let ds = array2d_to_dataset(id_raster).map_err(to_js_err)?;
    let poly_layer = terrio_polygonize(
        &ds,
        &PolygonizeOptions {
            band: 0,
            field_name: id_field_name.to_string(),
            layer_name: layer_name.to_string(),
            connectedness8: false,
            ignore_nodata: false,
            mask: None,
        },
    )
    .map_err(to_js_err)?;

    let item_map: HashMap<u32, &A> = items.iter().map(|item| (item_id(item), item)).collect();
    let mut layer = Layer::new(layer_name.to_string()).with_geom_type(GeometryType::Polygon);
    layer.crs = poly_layer.crs.clone();
    for &(name, ftype) in fields {
        layer.schema.add_field(FieldDef::new(name, ftype));
    }

    for feat in poly_layer.features {
        let Some(id) = feature_positive_id(&feat) else {
            continue;
        };
        let Some(item) = item_map.get(&id) else {
            continue;
        };
        layer.features.push(Feature {
            fid: (layer.features.len() + 1) as u64,
            geometry: feat.geometry,
            attributes: build_attrs(item),
        });
    }

    let layer = reproject_layer_to_wgs84_if_known(layer, &id_raster.projection);
    let filename = format!("{layer_name}.geojson");
    let bytes = write_vector_bytes(&[layer], &filename, &VectorWriteOptions::default())
        .map_err(to_js_err)?;
    String::from_utf8(bytes).map_err(to_js_err)
}

fn polygonize_sinks_to_geojson(
    region: &Array2D<i32>,
    depressions: &[SinkDepression],
) -> Result<String, JsValue> {
    const FIELDS: &[(&str, FieldType)] = &[
        ("region_id", FieldType::Integer),
        ("count", FieldType::Integer),
        ("area", FieldType::Float),
        ("volume", FieldType::Float),
        ("avg_depth", FieldType::Float),
        ("max_depth", FieldType::Float),
        ("min_elev", FieldType::Float),
        ("max_elev", FieldType::Float),
        ("perimeter", FieldType::Float),
        ("major_axis", FieldType::Float),
        ("minor_axis", FieldType::Float),
        ("elongatedness", FieldType::Float),
        ("eccentricity", FieldType::Float),
        ("orientation", FieldType::Float),
        ("area_bbox_ratio", FieldType::Float),
    ];
    polygonize_with_attributes_to_geojson(
        region,
        "sinks",
        "region_id",
        FIELDS,
        depressions,
        |d| d.region_id,
        |d| {
            vec![
                FieldValue::Integer(d.region_id as i64),
                FieldValue::Integer(d.count as i64),
                FieldValue::Float(d.area),
                FieldValue::Float(d.volume),
                FieldValue::Float(d.avg_depth),
                FieldValue::Float(d.max_depth),
                FieldValue::Float(d.min_elev),
                FieldValue::Float(d.max_elev),
                FieldValue::Float(d.perimeter),
                FieldValue::Float(d.major_axis),
                FieldValue::Float(d.minor_axis),
                FieldValue::Float(d.elongatedness),
                FieldValue::Float(d.eccentricity),
                FieldValue::Float(d.orientation),
                FieldValue::Float(d.area_bbox_ratio),
            ]
        },
    )
}

fn polygonize_depressions_to_geojson(
    depression_id: &Array2D<i32>,
    depressions: &[NestedDepression],
) -> Result<String, JsValue> {
    const FIELDS: &[(&str, FieldType)] = &[
        ("id", FieldType::Integer),
        ("parent_id", FieldType::Integer),
        ("level", FieldType::Integer),
        ("region_id", FieldType::Integer),
        ("count", FieldType::Integer),
        ("area", FieldType::Float),
        ("volume", FieldType::Float),
        ("avg_depth", FieldType::Float),
        ("max_depth", FieldType::Float),
        ("min_elev", FieldType::Float),
        ("max_elev", FieldType::Float),
        ("perimeter", FieldType::Float),
        ("major_axis", FieldType::Float),
        ("minor_axis", FieldType::Float),
        ("elongatedness", FieldType::Float),
        ("eccentricity", FieldType::Float),
        ("orientation", FieldType::Float),
        ("area_bbox_ratio", FieldType::Float),
        ("circularity", FieldType::Float),
        ("depth_diam_ratio", FieldType::Float),
        ("hypsometric_integral", FieldType::Float),
        ("rim_slope_deg", FieldType::Float),
        ("artifact_flag", FieldType::Integer),
    ];
    polygonize_with_attributes_to_geojson(
        depression_id,
        "depressions",
        "id",
        FIELDS,
        depressions,
        |d| d.id,
        |d| {
            vec![
                FieldValue::Integer(d.id as i64),
                FieldValue::Integer(d.parent_id as i64),
                FieldValue::Integer(d.level as i64),
                FieldValue::Integer(d.region_id as i64),
                FieldValue::Integer(d.count as i64),
                FieldValue::Float(d.area),
                FieldValue::Float(d.volume),
                FieldValue::Float(d.avg_depth),
                FieldValue::Float(d.max_depth),
                FieldValue::Float(d.min_elev),
                FieldValue::Float(d.max_elev),
                FieldValue::Float(d.perimeter),
                FieldValue::Float(d.major_axis),
                FieldValue::Float(d.minor_axis),
                FieldValue::Float(d.elongatedness),
                FieldValue::Float(d.eccentricity),
                FieldValue::Float(d.orientation),
                FieldValue::Float(d.area_bbox_ratio),
                FieldValue::Float(d.circularity),
                FieldValue::Float(d.depth_diam_ratio),
                FieldValue::Float(d.hypsometric_integral),
                FieldValue::Float(d.rim_slope_deg),
                FieldValue::Integer(i64::from(d.artifact_flag)),
            ]
        },
    )
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

/// Breaches depressions in a DEM GeoTIFF buffer using Lindsay (2016) least-cost breaching
/// (`mode`: `"complete"`, `"selective"`, or `"constrained"`).
#[wasm_bindgen(js_name = breachDem)]
pub fn breach_dem(
    bytes: &[u8],
    filename: &str,
    mode: &str,
    max_depth: Option<f64>,
    max_length: Option<u32>,
    fill_remaining: bool,
) -> Result<Vec<u8>, JsValue> {
    let dem = read_array2d_f64(bytes, filename)?;
    let breach_mode: BreachMode = mode.parse().map_err(to_js_err)?;
    let mut opts = BreachOptions::for_mode(breach_mode, max_depth, max_length);
    opts.fill_remaining = fill_remaining;
    let out = core_breach_depressions(&dem, &opts).map_err(to_js_err)?;
    write_array2d_tif(&out, "dem_breached.tif")
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
    let layer = reproject_layer_to_wgs84_if_known(layer, "");
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
    let geojson = polygonize_sinks_to_geojson(&mem.region, &mem.depressions)?;
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
    projection: String,
    max_level: u32,
    num_regions: usize,
    depressions_json: String,
    depression_id: Array2D<i32>,
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
    #[wasm_bindgen(getter)]
    pub fn projection(&self) -> String {
        self.projection.clone()
    }
    /// Returns the raster bounding box `[minLon, minLat, maxLon, maxLat]` in WGS 84 (`EPSG:4326`),
    /// or `undefined` when the raster lacks a recognized CRS.
    #[wasm_bindgen(js_name = boundsWgs84)]
    pub fn bounds_wgs84(&self) -> Option<Vec<f64>> {
        compute_wgs84_bounds(
            self.width,
            self.height,
            &self.depression_id.geotransform,
            &self.projection,
        )
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
    /// Alias of [`Self::depressions_json`] for mount delineation outputs.
    #[wasm_bindgen(js_name = mountsJson)]
    pub fn mounts_json(&self) -> String {
        self.depressions_json.clone()
    }
    /// Per-cell depression/mount ID grid (`Int32Array`, `0` outside features) for interactive inspection.
    #[wasm_bindgen(js_name = depressionIdValues)]
    pub fn depression_id_values(&self) -> Vec<i32> {
        self.depression_id.data().to_vec()
    }
    /// Alias of [`Self::depression_id_values`] for mount delineation outputs.
    #[wasm_bindgen(js_name = mountIdValues)]
    pub fn mount_id_values(&self) -> Vec<i32> {
        self.depression_id.data().to_vec()
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
    mem: lidar::InMemoryDepressions,
    id_name: &str,
    level_name: &str,
) -> Result<DepressionsOutput, JsValue> {
    let depressions_json = serde_json::to_string(&mem.depressions).map_err(to_js_err)?;
    let id_tif = write_array2d_tif(&mem.depression_id, id_name)?;
    let level_tif = write_array2d_tif(&mem.depression_level, level_name)?;
    let geojson = polygonize_depressions_to_geojson(&mem.depression_id, &mem.depressions)?;
    let level_rgba = colorize_i32_rgba(&mem.depression_level);
    let width = mem.depression_id.width();
    let height = mem.depression_id.height();
    let projection = mem.depression_id.projection.clone();
    Ok(DepressionsOutput {
        width,
        height,
        projection,
        max_level: mem.max_level,
        num_regions: mem.num_regions as usize,
        depressions_json,
        depression_id: mem.depression_id,
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
    marshal_depressions(mem, "depression_id.tif", "depression_level.tif")
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
    marshal_depressions(mem, "mount_id.tif", "mount_level.tif")
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

fn spill_paths_to_geojson(
    paths: &[SpillFlowPath],
    geotransform: &[f64; 6],
    projection: &str,
) -> Result<String, JsValue> {
    let trimmed = projection.trim();
    if trimmed.is_empty() {
        return Err(JsValue::from_str(
            "Cannot export spill flow paths as RFC 7946 GeoJSON: input DEM has no CRS metadata",
        ));
    }
    let src_srs = terrio::crs::SpatialRef::parse(trimmed).map_err(|e| {
        JsValue::from_str(&format!(
            "Cannot export spill flow paths as RFC 7946 GeoJSON: unsupported DEM CRS ({e})"
        ))
    })?;
    if src_srs.is_unknown() {
        return Err(JsValue::from_str(
            "Cannot export spill flow paths as RFC 7946 GeoJSON: input DEM CRS is unknown",
        ));
    }
    let transformer = if src_srs.epsg == Some(4326) {
        None
    } else {
        let wgs84 = terrio::crs::SpatialRef::parse("EPSG:4326").map_err(to_js_err)?;
        Some(
            terrio::crs::Transformer::new(&src_srs, &wgs84).map_err(|e| {
                JsValue::from_str(&format!(
                    "Cannot reproject spill flow paths to WGS 84 (EPSG:4326): {e}"
                ))
            })?,
        )
    };

    let mut features: Vec<serde_json::Value> = Vec::with_capacity(paths.len());
    for (idx, p) in paths.iter().enumerate() {
        let mut coords: Vec<[f64; 2]> = Vec::with_capacity(p.cells.len());
        for &(col, row) in &p.cells {
            let cf = col as f64 + 0.5;
            let rf = row as f64 + 0.5;
            let x = geotransform[0] + cf * geotransform[1] + rf * geotransform[2];
            let y = geotransform[3] + cf * geotransform[4] + rf * geotransform[5];
            let (lon, lat) = if let Some(ref tr) = transformer {
                tr.transform(x, y).map_err(to_js_err)?
            } else {
                (x, y)
            };
            coords.push([lon, lat]);
        }
        features.push(serde_json::json!({
            "type": "Feature",
            "id": idx + 1,
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "from_dep_id": p.from_dep_id,
                "from_level": p.from_level,
                "to_dep_id": p.to_dep_id,
                "spill_time_h": p.spill_time_h,
                "merge_time_h": p.merge_time_h,
                "spill_elev": p.spill_elev,
                "cell_count": p.cells.len(),
            },
        }));
    }
    serde_json::to_string(&serde_json::json!({
        "type": "FeatureCollection",
        "name": "spill_flow_paths",
        "features": features,
    }))
    .map_err(to_js_err)
}

/// Output container returned by [`simulate_inundation`].
///
/// Returns live playback data (`fillTimeValues`, `hillshadeRgba`, `stepsJson`, `spillPathsJson`,
/// spatial previews, and downloadable rasters/GeoJSON) immediately without blocking on GIF
/// quantization. Call [`InundationOutput::encode_gif`] on demand to generate an animated GIF.
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
    runoff_coeff: f64,
    scaled_hyetograph: Option<Vec<(f64, f64)>>,
    dem_refined: Array2D<f64>,
    fill_time: Array2D<f64>,
    catchment_hierarchy: Array2D<f64>,
    sink: Array2D<f64>,
    depression_level: Array2D<i32>,
    depression_id: Array2D<i32>,
    depressions: Vec<NestedDepression>,
    spill_paths: Vec<SpillFlowPath>,
    depressions_json: String,
    catchment_hierarchy_json: String,
    steps_json: String,
    spill_paths_json: String,
    rating_curves_json: String,
    rating_curves_csv: String,
    connectivity_graph_json: String,
    connectivity_graphml: String,
    inundation_curve_csv: String,
    hillshade_rgba: Vec<u8>,
    depth_rgba: Vec<u8>,
    level_rgba: Vec<u8>,
    fill_time_rgba: Vec<u8>,
    fsm_mass_balance_json: String,
    fsm_basins_json: String,
    fsm_basins_csv: String,
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
    /// Returns the DEM bounding box `[minLon, minLat, maxLon, maxLat]` in WGS 84 (`EPSG:4326`),
    /// or `undefined` when the DEM lacks a recognized CRS.
    #[wasm_bindgen(js_name = boundsWgs84)]
    pub fn bounds_wgs84(&self) -> Option<Vec<f64>> {
        compute_wgs84_bounds(
            self.width,
            self.height,
            &self.dem_refined.geotransform,
            &self.projection,
        )
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
    #[wasm_bindgen(js_name = spillPathsJson)]
    pub fn spill_paths_json(&self) -> String {
        self.spill_paths_json.clone()
    }
    /// Exports delineated spill flow paths as an RFC 7946 GeoJSON FeatureCollection in WGS 84
    /// longitude/latitude coordinates (reprojected from the DEM CRS when necessary).
    #[wasm_bindgen(js_name = spillPathsGeojson)]
    pub fn spill_paths_geojson(&self) -> Result<String, JsValue> {
        spill_paths_to_geojson(
            &self.spill_paths,
            &self.dem_refined.geotransform,
            &self.projection,
        )
    }
    #[wasm_bindgen(js_name = fillTimeTif)]
    pub fn fill_time_tif(&self) -> Result<Vec<u8>, JsValue> {
        write_array2d_tif(&self.fill_time, "fill_time.tif")
    }
    #[wasm_bindgen(js_name = catchmentHirTif)]
    pub fn catchment_hir_tif(&self) -> Result<Vec<u8>, JsValue> {
        write_array2d_tif(&self.catchment_hierarchy, "catchment_hir.tif")
    }
    #[wasm_bindgen(js_name = sinkTif)]
    pub fn sink_tif(&self) -> Result<Vec<u8>, JsValue> {
        write_array2d_tif(&self.sink, "sink.tif")
    }
    #[wasm_bindgen(js_name = levelTif)]
    pub fn level_tif(&self) -> Result<Vec<u8>, JsValue> {
        write_array2d_tif(&self.depression_level, "depression_level.tif")
    }
    #[wasm_bindgen(js_name = depressionsGeojson)]
    pub fn depressions_geojson(&self) -> Result<String, JsValue> {
        polygonize_depressions_to_geojson(&self.depression_id, &self.depressions)
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
        let ft_nd = self.fill_time.no_data();
        self.fill_time
            .data()
            .iter()
            .map(|&v| {
                if v.is_finite() && !v.rd_eq(ft_nd) && v >= 0.0 {
                    v
                } else {
                    -1.0
                }
            })
            .collect()
    }
    /// Per-cell depression ID grid (`Int32Array`, `0` outside depressions) for interactive click-to-inspect.
    #[wasm_bindgen(js_name = depressionIdValues)]
    pub fn depression_id_values(&self) -> Vec<i32> {
        self.depression_id.data().to_vec()
    }
    /// Effective runoff coefficient used in the simulation (`(0, 1]`).
    #[wasm_bindgen(getter, js_name = runoffCoeff)]
    pub fn runoff_coeff(&self) -> f64 {
        self.runoff_coeff
    }
    /// Stage-storage-area rating curves across all depressions as JSON.
    #[wasm_bindgen(js_name = ratingCurvesJson)]
    pub fn rating_curves_json(&self) -> String {
        self.rating_curves_json.clone()
    }
    /// Stage-storage-area rating curves formatted as CSV.
    #[wasm_bindgen(js_name = ratingCurvesCsv)]
    pub fn rating_curves_csv(&self) -> String {
        self.rating_curves_csv.clone()
    }
    /// Directed nested connectivity graph as JSON.
    #[wasm_bindgen(js_name = connectivityGraphJson)]
    pub fn connectivity_graph_json(&self) -> String {
        self.connectivity_graph_json.clone()
    }
    /// Directed nested connectivity graph as GraphML XML (NetworkX / Gephi compatible).
    #[wasm_bindgen(js_name = connectivityGraphml)]
    pub fn connectivity_graphml(&self) -> String {
        self.connectivity_graphml.clone()
    }
    /// Time-series inundation curve (with cumulative runoff in mm) formatted as CSV.
    #[wasm_bindgen(js_name = inundationCurveCsv)]
    pub fn inundation_curve_csv(&self) -> String {
        self.inundation_curve_csv.clone()
    }
    /// Dynamic Fill-Spill-Merge watershed mass balance as JSON (`"null"` in static mode).
    #[wasm_bindgen(js_name = fsmMassBalanceJson)]
    pub fn fsm_mass_balance_json(&self) -> String {
        self.fsm_mass_balance_json.clone()
    }
    /// Dynamic Fill-Spill-Merge per-basin summaries as JSON (`"[]"` in static mode).
    #[wasm_bindgen(js_name = fsmBasinsJson)]
    pub fn fsm_basins_json(&self) -> String {
        self.fsm_basins_json.clone()
    }
    /// Dynamic Fill-Spill-Merge per-basin summaries formatted as CSV (empty string in static mode).
    #[wasm_bindgen(js_name = fsmBasinsCsv)]
    pub fn fsm_basins_csv(&self) -> String {
        self.fsm_basins_csv.clone()
    }
    /// Renders a single inundation time step (`0`-based, or `0` = dry, `1..=numSteps`) as PNG bytes
    /// with optional scale bar, legend, and active spill flow paths.
    #[wasm_bindgen(js_name = renderFramePng)]
    pub fn render_frame_png(
        &self,
        step_index: usize,
        show_scale_bar: bool,
        show_legend: bool,
        show_spill_paths: Option<bool>,
    ) -> Result<Vec<u8>, JsValue> {
        let clamped = step_index.min(self.num_steps);
        let t_hours = (clamped as f64) * self.time_step_h;
        let progress = if self.num_steps > 0 {
            clamped as f64 / self.num_steps as f64
        } else {
            0.0
        };
        let eff_intensity_cm_h = if t_hours > 0.0 {
            let cum_mm = lidar::cumulative_runoff_mm(
                t_hours,
                self.rain_intensity_cm_h * self.runoff_coeff,
                self.scaled_hyetograph.as_deref(),
            );
            cum_mm / (t_hours * 10.0)
        } else {
            self.rain_intensity_cm_h * self.runoff_coeff
        };
        let opts = AnimateInundationOptions {
            time_step_h: Some(self.time_step_h),
            rain_intensity_cm_h: Some(eff_intensity_cm_h),
            show_scale_bar,
            show_legend,
            ..Default::default()
        };
        let paths: &[SpillFlowPath] = if show_spill_paths.unwrap_or(true) {
            &self.spill_paths
        } else {
            &[]
        };
        render_inundation_frame_png_bytes(
            &self.dem_refined,
            &self.fill_time,
            t_hours,
            progress,
            paths,
            &opts,
        )
        .map_err(to_js_err)
    }
    /// Encodes a shaded-relief animated inundation GIF on demand, guarding against
    /// excessive total frame-pixel counts in synchronous browser execution.
    #[wasm_bindgen(js_name = encodeGif)]
    pub fn encode_gif(
        &self,
        frame_delay_ms: u16,
        ping_pong: bool,
        max_dim: usize,
        show_spill_paths: Option<bool>,
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

        let total_duration_h = (self.num_steps as f64) * self.time_step_h;
        let eff_intensity_cm_h = if self.scaled_hyetograph.is_some() && total_duration_h > 0.0 {
            let cum_mm = lidar::cumulative_runoff_mm(
                total_duration_h,
                self.rain_intensity_cm_h * self.runoff_coeff,
                self.scaled_hyetograph.as_deref(),
            );
            cum_mm / (total_duration_h * 10.0)
        } else {
            self.rain_intensity_cm_h * self.runoff_coeff
        };
        let anim_opts = AnimateInundationOptions {
            frame_delay_ms,
            ping_pong,
            loops: 0,
            max_dimension: if max_dim > 0 { Some(max_dim) } else { None },
            time_step_h: Some(self.time_step_h),
            rain_intensity_cm_h: Some(eff_intensity_cm_h),
            ..Default::default()
        };
        let paths: &[SpillFlowPath] = if show_spill_paths.unwrap_or(true) {
            &self.spill_paths
        } else {
            &[]
        };
        let (gif_bytes, _, _, _) = animate_inundation_to_bytes_with_spill_paths(
            &self.dem_refined,
            &self.fill_time,
            self.num_steps,
            self.time_step_h,
            paths,
            &anim_opts,
        )
        .map_err(to_js_err)?;
        Ok(gif_bytes)
    }
}

/// Runs the full in-memory hydrology & inundation pipeline:
/// optional median filter -> sink extraction -> nested depressions -> catchment hierarchy ->
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
    simulate_inundation_advanced(
        dem_bytes,
        filename,
        kernel_size,
        min_size,
        min_depth,
        interval,
        rain_intensity_cm_h,
        rain_duration_h,
        time_step_h,
        None,
        1.0,
        None,
        None,
    )
}

/// Runs the full in-memory hydrology & inundation pipeline with optional selective breaching
/// (`breach_max_depth`), effective runoff coefficient (`runoff_coeff` in `(0, 1]`), optional
/// SCS Type II design storm depth (`scs_type2_mm`), and optional routing mode
/// (`"static"` default, or `"fsm"` for dynamic Fill-Spill-Merge cascade mass-balance routing).
#[wasm_bindgen(js_name = simulateInundationAdvanced)]
#[allow(clippy::too_many_arguments)]
pub fn simulate_inundation_advanced(
    dem_bytes: &[u8],
    filename: &str,
    kernel_size: usize,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    rain_intensity_cm_h: f64,
    rain_duration_h: f64,
    time_step_h: f64,
    breach_max_depth: Option<f64>,
    runoff_coeff: f64,
    scs_type2_mm: Option<f64>,
    routing: Option<String>,
) -> Result<InundationOutput, JsValue> {
    if !runoff_coeff.is_finite() || runoff_coeff <= 0.0 || runoff_coeff > 1.0 {
        return Err(to_js_err(format!(
            "runoff_coeff must be in (0, 1], got {runoff_coeff}"
        )));
    }
    let routing_mode: RoutingMode = match routing.as_deref() {
        Some(r) if !r.trim().is_empty() => r.parse().map_err(to_js_err)?,
        _ => RoutingMode::Static,
    };
    let raw_dem = read_array2d_f64(dem_bytes, filename)?;
    let breach = match breach_max_depth {
        Some(md) => {
            if !md.is_finite() || md < 0.0 {
                return Err(to_js_err(format!(
                    "breach_max_depth must be finite and >= 0, got {md}"
                )));
            }
            if md > 0.0 {
                Some(BreachOptions {
                    mode: BreachMode::Selective,
                    max_depth: Some(md),
                    max_length: Some(40),
                    ..Default::default()
                })
            } else {
                None
            }
        }
        None => None,
    };
    let rc_arr = if (runoff_coeff - 1.0).abs() > 1e-12 {
        let mut a = Array2D::from_vec(
            raw_dem.width(),
            raw_dem.height(),
            vec![runoff_coeff; raw_dem.data().len()],
        )
        .map_err(to_js_err)?;
        a.copy_metadata_from(&raw_dem);
        a.set_no_data(f64::NAN);
        Some(a)
    } else {
        None
    };
    let hyeto = match scs_type2_mm {
        Some(mm) => {
            if !mm.is_finite() || mm < 0.0 {
                return Err(to_js_err(format!(
                    "scs_type2_mm must be finite and >= 0, got {mm}"
                )));
            }
            if mm > 0.0 {
                Some(scs_type2_hyetograph(mm, rain_duration_h, time_step_h).map_err(to_js_err)?)
            } else {
                None
            }
        }
        None => None,
    };
    let scaled_hyetograph: Option<Vec<(f64, f64)>> = hyeto.as_ref().map(|h| {
        h.iter()
            .map(|&(dt, cm_h)| (dt, cm_h * runoff_coeff))
            .collect()
    });
    let opts = InMemoryFullPipelineOptions {
        kernel_size,
        breach,
        simulation: SimulateInundationOptions {
            min_size,
            min_depth,
            interval,
            rain_intensity_cm_h,
            rain_duration_h,
            time_step_h,
            compress: false,
        },
        routing: routing_mode,
        forcing: InundationForcing {
            runoff_coeff: rc_arr.as_ref(),
            infiltration_cm_h: None,
            antecedent_depth: None,
            hyetograph: hyeto.as_deref(),
        },
        scalar_runoff_coeff: runoff_coeff,
        stream_vector: None,
    };
    let pipe = run_full_pipeline_in_memory(&raw_dem, &opts).map_err(to_js_err)?;

    let rating_curves_json = serde_json::to_string(&pipe.rating_curves).map_err(to_js_err)?;
    let rating_curves_csv = rating_curves_to_csv_string(&pipe.rating_curves);
    let connectivity_graph_json =
        serde_json::to_string(&pipe.connectivity_graph).map_err(to_js_err)?;
    let connectivity_graphml = connectivity_graph_to_graphml_string(&pipe.connectivity_graph);
    let inundation_curve_csv = pipe.inundation_curve_csv;
    let fsm_mass_balance_json = serde_json::to_string(&pipe.fsm_mass_balance).map_err(to_js_err)?;
    let fsm_basins_json = match pipe.fsm_basins.as_ref() {
        Some(b) => serde_json::to_string(b).map_err(to_js_err)?,
        None => "[]".to_string(),
    };
    let fsm_basins_csv = pipe
        .fsm_basins
        .as_deref()
        .map(fsm_basins_to_csv_string)
        .unwrap_or_default();

    let hillshade_rgb =
        compute_hillshade_rgb(&pipe.sinks.dem_refined, &HillshadeOptions::default());
    let hillshade_rgba = rgb_to_rgba(&hillshade_rgb);
    let depth_rgba = colorize_f64_rgba(&pipe.sinks.depth, true, false, false);
    let level_rgba = colorize_i32_rgba(&pipe.depressions.depression_level);
    let fill_time_rgba = colorize_f64_rgba(&pipe.fill_time, false, true, true);

    let depressions_json =
        serde_json::to_string(&pipe.depressions.depressions).map_err(to_js_err)?;
    let catchment_hierarchy_json =
        serde_json::to_string(&pipe.catchment_records).map_err(to_js_err)?;
    let steps_json = serde_json::to_string(&pipe.steps).map_err(to_js_err)?;
    let spill_paths_json = serde_json::to_string(&pipe.spill_paths).map_err(to_js_err)?;

    Ok(InundationOutput {
        width: raw_dem.width(),
        height: raw_dem.height(),
        projection: raw_dem.projection,
        max_level: pipe.depressions.max_level,
        num_regions: pipe.num_regions,
        num_steps: pipe.steps.len(),
        time_step_h,
        rain_intensity_cm_h,
        runoff_coeff,
        scaled_hyetograph,
        dem_refined: pipe.sinks.dem_refined,
        fill_time: pipe.fill_time,
        catchment_hierarchy: pipe.catchment_hierarchy,
        sink: pipe.sinks.sink,
        depression_level: pipe.depressions.depression_level,
        depression_id: pipe.depressions.depression_id,
        depressions: pipe.depressions.depressions,
        spill_paths: pipe.spill_paths,
        depressions_json,
        catchment_hierarchy_json,
        steps_json,
        spill_paths_json,
        rating_curves_json,
        rating_curves_csv,
        connectivity_graph_json,
        connectivity_graphml,
        inundation_curve_csv,
        hillshade_rgba,
        depth_rgba,
        level_rgba,
        fill_time_rgba,
        fsm_mass_balance_json,
        fsm_basins_json,
        fsm_basins_csv,
    })
}
