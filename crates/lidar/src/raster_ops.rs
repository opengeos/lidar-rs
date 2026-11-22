//! Geospatial raster and vector utilities powered by [`terrio`].
//!
//! Pure-Rust ports of the GDAL / Rasterio / Fiona / GeoPandas utilities in
//! `lidar/common.py` and `lidar/filling.py`:
//! - [`polygonize`] / [`polygonize_array`]
//! - [`mosaic`]
//! - [`reproject_image`]
//! - [`resample`]
//! - [`clip_image`]
//! - [`image_to_cog`]
//! - [`add_crs`]
//! - [`join_csv_to_vector`]
//! - [`csv_points_to_vector`]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use richdem::{Array2D, DemNum, array2d_to_dataset};
use terrio::crs::SpatialRef;
use terrio::formats::RasterDriver;
use terrio::ops::Resampling;
use terrio::ops::mosaic::{MosaicOptions, mosaic_to_path};
use terrio::ops::polygonize::{PolygonizeOptions, polygonize as terrio_polygonize};
use terrio::ops::warp::{Cutline, WarpOptions, warp};
use terrio::vector::Coord;
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer, RasterDataset,
    RasterWriteOptions, VectorOpenOptions, VectorWriteOptions,
};

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

/// Parses a resampling method name (`nearest`, `bilinear`, `cubic`, `cubicspline`,
/// `lanczos`, `average`, `mode`, `min`, `max`, `med`, `q1`, `q3`, `sum`, `rms`).
pub fn parse_resampling(name: &str) -> Result<Resampling> {
    Resampling::parse(name).map_err(|e| Error::InvalidArgument(e.to_string()))
}

/// Polygonizes a raster dataset on disk into a vector file (Shapefile, GeoPackage,
/// GeoJSON, etc., chosen from the output file extension).
///
/// Mirrors `lidar`'s `polygonize(img, shp_path)` using pure-Rust
/// `terrio::ops::polygonize` (4-connected, skipping NoData pixels).
pub fn polygonize(
    in_raster: impl AsRef<Path>,
    out_vector: impl AsRef<Path>,
    field_name: &str,
) -> Result<usize> {
    let in_str = path_to_str(in_raster.as_ref())?;
    let ds = terrio::open_raster(in_str)?;
    polygonize_dataset(&ds, out_vector.as_ref(), field_name)
}

/// Polygonizes an in-memory [`richdem::Array2D`] directly to a vector file.
pub fn polygonize_array<T: DemNum>(
    arr: &Array2D<T>,
    out_vector: impl AsRef<Path>,
    field_name: &str,
) -> Result<usize> {
    let ds = array2d_to_dataset(arr)?;
    polygonize_dataset(&ds, out_vector.as_ref(), field_name)
}

fn polygonize_dataset(ds: &RasterDataset, out_vector: &Path, field_name: &str) -> Result<usize> {
    ensure_parent_dir(out_vector)?;
    let opts = PolygonizeOptions {
        band: 0,
        field_name: field_name.to_string(),
        layer_name: "Shape".to_string(),
        connectedness8: false,
        ignore_nodata: false,
        mask: None,
    };
    let layer = terrio_polygonize(ds, &opts)?;
    let count = layer.features.len();
    let out_str = path_to_str(out_vector)?;
    terrio::write_vector(&[layer], out_str, &VectorWriteOptions::default())?;
    Ok(count)
}

fn resolve_path_for_compare(path: &Path) -> Option<PathBuf> {
    if let Ok(canon) = std::fs::canonicalize(path) {
        return Some(canon);
    }
    let parent = path.parent()?;
    let file_name = path.file_name()?;
    let parent_dir = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let canon_parent = std::fs::canonicalize(parent_dir).ok()?;
    Some(canon_parent.join(file_name))
}

/// Returns an error if `input` resolves to the same path as any entry in `outputs`.
pub(crate) fn ensure_no_path_collision(input: &Path, outputs: &[&Path]) -> Result<()> {
    let in_canon = resolve_path_for_compare(input);
    let mut seen_outputs: Vec<(PathBuf, &Path)> = Vec::with_capacity(outputs.len());
    for &out in outputs {
        if let Some(out_canon) = resolve_path_for_compare(out) {
            if in_canon.as_ref() == Some(&out_canon) {
                return Err(Error::InvalidArgument(format!(
                    "output path '{}' would overwrite input file '{}'",
                    out.display(),
                    input.display()
                )));
            }
            if let Some((_, prev)) = seen_outputs.iter().find(|(p, _)| *p == out_canon) {
                return Err(Error::InvalidArgument(format!(
                    "output paths '{}' and '{}' resolve to the same destination",
                    prev.display(),
                    out.display()
                )));
            }
            seen_outputs.push((out_canon, out));
        }
    }
    Ok(())
}

/// Mosaics multiple raster files into a single output GeoTIFF.
///
/// Mirrors `lidar.mosaic`.
pub fn mosaic<P: AsRef<Path>>(inputs: &[P], output: impl AsRef<Path>) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::InvalidArgument(
            "mosaic requires at least one input raster".into(),
        ));
    }
    let output = output.as_ref();
    ensure_parent_dir(output)?;

    for inp in inputs {
        ensure_no_path_collision(inp.as_ref(), &[output])?;
    }

    let paths: Vec<String> = inputs
        .iter()
        .map(|p| path_to_str(p.as_ref()).map(str::to_string))
        .collect::<Result<Vec<_>>>()?;
    let out_str = path_to_str(output)?;
    mosaic_to_path(
        &paths,
        &MosaicOptions::default(),
        out_str,
        &RasterWriteOptions::default(),
    )?;
    Ok(())
}

/// Reprojects a raster to `dst_crs` (e.g., `"EPSG:4326"`, `"EPSG:5070"`, WKT, or PROJ).
///
/// Mirrors `lidar.reproject_image`.
pub fn reproject_image(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    dst_crs: &str,
    resampling: Resampling,
    resolution: Option<(f64, f64)>,
) -> Result<()> {
    let in_path = input.as_ref();
    let out_path = output.as_ref();
    ensure_no_path_collision(in_path, &[out_path])?;
    let in_str = path_to_str(in_path)?;
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let src = terrio::open_raster(in_str)?;
    let srs = SpatialRef::parse(dst_crs)?;
    let opts = WarpOptions {
        dst_srs: Some(srs),
        resolution,
        resampling,
        ..Default::default()
    };
    let warped = warp(&[&src], &opts)?;
    terrio::write_raster_dataset(&warped, out_str, &RasterWriteOptions::default())?;
    Ok(())
}

/// Resamples a raster to a new target pixel resolution `(res, res)`.
///
/// Mirrors `lidar.resample`.
pub fn resample(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    resolution: f64,
    resampling: Resampling,
) -> Result<()> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "resolution must be finite and > 0, got {resolution}"
        )));
    }
    let in_path = input.as_ref();
    let out_path = output.as_ref();
    ensure_no_path_collision(in_path, &[out_path])?;
    let in_str = path_to_str(in_path)?;
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let src = terrio::open_raster(in_str)?;
    let opts = WarpOptions {
        resolution: Some((resolution, resolution)),
        resampling,
        ..Default::default()
    };
    let warped = warp(&[&src], &opts)?;
    terrio::write_raster_dataset(&warped, out_str, &RasterWriteOptions::default())?;
    Ok(())
}

/// Clips a raster by a polygon vector mask (GeoJSON, Shapefile, GeoPackage, etc.),
/// cropping to the cutline extent.
///
/// Mirrors `lidar.clip_image`.
pub fn clip_image(
    input: impl AsRef<Path>,
    mask: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<()> {
    let in_path = input.as_ref();
    let out_path = output.as_ref();
    ensure_no_path_collision(in_path, &[out_path])?;
    let in_str = path_to_str(in_path)?;
    let mask_str = path_to_str(mask.as_ref())?;
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let src = terrio::open_raster(in_str)?;
    let cutline = Cutline::open(mask_str)?;
    let opts = WarpOptions {
        cutline: Some(cutline),
        crop_to_cutline: true,
        ..Default::default()
    };
    let clipped = warp(&[&src], &opts)?;
    terrio::write_raster_dataset(&clipped, out_str, &RasterWriteOptions::default())?;
    Ok(())
}

/// Converts a raster to a Cloud Optimized GeoTIFF (COG).
///
/// Mirrors `lidar.image_to_cog`.
pub fn image_to_cog(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    compression: Option<&str>,
) -> Result<()> {
    let in_path = input.as_ref();
    let out_path = output.as_ref();
    ensure_no_path_collision(in_path, &[out_path])?;
    let in_str = path_to_str(in_path)?;
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let src = terrio::open_raster(in_str)?;
    let opts = RasterWriteOptions {
        driver: Some(RasterDriver::Cog),
        compression: Some(compression.unwrap_or("DEFLATE").to_string()),
        ..Default::default()
    };
    terrio::write_raster_dataset(&src, out_str, &opts)?;
    Ok(())
}

/// Truncates a string to at most `max_bytes` UTF-8 bytes without splitting
/// a multibyte character (required for ESRI Shapefile DBF 10-byte field names).
fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> String {
    s.char_indices()
        .take_while(|(i, c)| i + c.len_utf8() <= max_bytes)
        .map(|(_, c)| c)
        .collect()
}

/// Assigns a CRS (EPSG code such as `"4326"` / `"EPSG:4326"`, WKT, or PROJ string)
/// to a raster dataset without reprojecting pixels.
///
/// When `output` is `None` (in-place update), writes atomically via a temporary
/// sibling file and renames it over `input` so a write failure cannot corrupt the
/// original raster.
///
/// Mirrors `lidar.add_crs`.
pub fn add_crs(input: impl AsRef<Path>, crs: &str, output: Option<&Path>) -> Result<()> {
    let in_path = input.as_ref();
    let in_str = path_to_str(in_path)?;
    let out_path = output.unwrap_or(in_path);
    ensure_parent_dir(out_path)?;

    let crs_norm = if crs.trim().chars().all(|c| c.is_ascii_digit()) {
        format!("EPSG:{}", crs.trim())
    } else {
        crs.trim().to_string()
    };
    let sref = SpatialRef::parse(&crs_norm)?;
    let mut ds = terrio::open_raster(in_str)?;
    ds.raster.crs = sref.to_crs_info();

    let in_place = output.is_none() || out_path == in_path;
    if in_place {
        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("tif");
        let stem = out_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("raster");
        let tmp_path =
            out_path.with_file_name(format!(".{stem}.lidar_tmp_{}.{ext}", std::process::id()));
        let tmp_str = path_to_str(&tmp_path)?;
        if let Err(e) = terrio::write_raster_dataset(&ds, tmp_str, &RasterWriteOptions::default()) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        if let Err(e) = std::fs::rename(&tmp_path, out_path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
    } else {
        let out_str = path_to_str(out_path)?;
        terrio::write_raster_dataset(&ds, out_str, &RasterWriteOptions::default())?;
    }
    Ok(())
}

/// Parses RFC-4180 CSV content into complete records, properly preserving
/// embedded newlines and escaped quotes inside quoted fields.
fn parse_csv_records(content: &str) -> Vec<Vec<String>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut chars = content.chars().peekable();
    let mut in_quotes = false;
    let mut had_field_content = false;

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' => {
                    in_quotes = true;
                    had_field_content = true;
                }
                ',' => {
                    row.push(field.trim().to_string());
                    field.clear();
                    had_field_content = true;
                }
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    if had_field_content || !field.trim().is_empty() {
                        row.push(field.trim().to_string());
                        records.push(std::mem::take(&mut row));
                    }
                    field.clear();
                    had_field_content = false;
                }
                '\n' => {
                    if had_field_content || !field.trim().is_empty() {
                        row.push(field.trim().to_string());
                        records.push(std::mem::take(&mut row));
                    }
                    field.clear();
                    had_field_content = false;
                }
                _ => {
                    field.push(c);
                }
            }
        }
    }

    if had_field_content || !field.trim().is_empty() {
        row.push(field.trim().to_string());
        records.push(row);
    }
    records
}

fn read_csv_table(csv_path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let raw = std::fs::read_to_string(csv_path)?;
    let content = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
    let mut records = parse_csv_records(content);
    if records.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let headers = records.remove(0);
    Ok((headers, records))
}

fn infer_field_type(rows: &[Vec<String>], col: usize) -> FieldType {
    let mut saw_any = false;
    let mut all_int = true;
    let mut all_float = true;
    for row in rows {
        if let Some(val) = row.get(col) {
            let v = val.trim();
            if v.is_empty() {
                continue;
            }
            saw_any = true;
            if v.parse::<i64>().is_err() {
                all_int = false;
            }
            if v.parse::<f64>().is_err() {
                all_float = false;
            }
        }
    }
    if !saw_any {
        FieldType::Text
    } else if all_int {
        FieldType::Integer
    } else if all_float {
        FieldType::Float
    } else {
        FieldType::Text
    }
}

fn parse_field_value(raw: Option<&String>, ftype: FieldType) -> FieldValue {
    let Some(s) = raw.map(|v| v.trim()).filter(|v| !v.is_empty()) else {
        return FieldValue::Null;
    };
    match ftype {
        FieldType::Integer => s
            .parse::<i64>()
            .map(FieldValue::Integer)
            .unwrap_or_else(|_| FieldValue::Text(s.to_string())),
        FieldType::Float => s
            .parse::<f64>()
            .map(FieldValue::Float)
            .unwrap_or_else(|_| FieldValue::Text(s.to_string())),
        _ => FieldValue::Text(s.to_string()),
    }
}

fn normalize_text_join_key(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Ok(i) = trimmed.parse::<i64>() {
        return i.to_string();
    }
    if let Ok(f) = trimmed.parse::<f64>()
        && f.is_finite()
        && f.fract() == 0.0
        && f >= (i64::MIN as f64)
        && f < (i64::MAX as f64)
        && f.abs() <= ((1u64 << 53) as f64)
    {
        return (f as i64).to_string();
    }
    trimmed.to_string()
}

fn field_value_join_key(val: &FieldValue) -> Option<String> {
    match val {
        FieldValue::Integer(i) => Some(i.to_string()),
        FieldValue::Float(f) => {
            if f.is_finite() && f.fract() == 0.0 && f.abs() <= ((1u64 << 53) as f64) {
                Some((*f as i64).to_string())
            } else {
                Some(f.to_string())
            }
        }
        FieldValue::Text(s) => Some(normalize_text_join_key(s)),
        _ => None,
    }
}

/// Allocates a field name unique within `used_names` (case-insensitive),
/// truncating safely to 10 UTF-8 bytes when writing ESRI Shapefiles.
fn allocate_unique_field_name(
    raw_name: &str,
    is_shp: bool,
    used_names: &mut HashSet<String>,
) -> String {
    let base_raw = raw_name.replace('-', "_");
    let base = if is_shp && base_raw.len() > 10 {
        truncate_utf8_bytes(&base_raw, 10)
    } else {
        base_raw
    };
    if used_names.insert(base.to_ascii_lowercase()) {
        return base;
    }
    let mut counter = 1usize;
    loop {
        let suffix = format!("_{counter}");
        let mut candidate = if is_shp && base.len() + suffix.len() > 10 {
            truncate_utf8_bytes(&base, 10usize.saturating_sub(suffix.len()))
        } else {
            base.clone()
        };
        candidate.push_str(&suffix);
        if used_names.insert(candidate.to_ascii_lowercase()) {
            return candidate;
        }
        counter += 1;
    }
}

/// Joins a CSV table onto a vector file by matching `vec_join_col` to `csv_join_col`,
/// writing the joined vector layer to `out_vector` (GeoPackage, Shapefile, GeoJSON, etc.).
///
/// Mirrors `lidar.join_tables` and `lidar.join_csv_to_gdf`.
pub fn join_csv_to_vector(
    in_vector: impl AsRef<Path>,
    in_csv: impl AsRef<Path>,
    out_vector: impl AsRef<Path>,
    vec_join_col: &str,
    csv_join_col: &str,
) -> Result<usize> {
    let in_vec_str = path_to_str(in_vector.as_ref())?;
    let out_path = out_vector.as_ref();
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let vds = terrio::open_vector(in_vec_str, &VectorOpenOptions::default())?;
    let src_layer = vds.layers.first().ok_or_else(|| {
        Error::InvalidArgument(format!("vector file has no layers: {in_vec_str}"))
    })?;

    let (headers, rows) = read_csv_table(in_csv.as_ref())?;
    let is_shp = out_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("shp"));

    let mut out_layer = Layer::new(src_layer.name.clone());
    if let Some(gt) = src_layer.geom_type {
        out_layer = out_layer.with_geom_type(gt);
    }
    SpatialRef::from_layer(src_layer).apply_to_layer(&mut out_layer);

    if headers.is_empty() || rows.is_empty() {
        out_layer.schema = src_layer.schema.clone();
        out_layer.features = src_layer.features.clone();
        let n = out_layer.features.len();
        terrio::write_vector(&[out_layer], out_str, &VectorWriteOptions::default())?;
        return Ok(n);
    }

    let vec_key_idx = src_layer
        .schema
        .fields()
        .iter()
        .position(|f| f.name.eq_ignore_ascii_case(vec_join_col))
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "join column '{vec_join_col}' not found in vector layer"
            ))
        })?;
    let csv_key_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case(csv_join_col))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("join column '{csv_join_col}' not found in CSV"))
        })?;

    // Drop the vector join column when `csv_join_col` uses a different name
    // (matching `ExtractSinks` dropping `id` in favor of `region_id`). When both
    // join columns share the same name, keep the vector key and skip the duplicate CSV key.
    let mut keep_vec_indices = Vec::new();
    for (idx, field) in src_layer.schema.fields().iter().enumerate() {
        if idx == vec_key_idx && !vec_join_col.eq_ignore_ascii_case(csv_join_col) {
            continue;
        }
        keep_vec_indices.push(idx);
        out_layer.schema.add_field(field.clone());
    }

    // Add CSV fields with guaranteed unique names (truncated to 10 UTF-8 bytes for Shapefile DBF).
    let mut csv_col_types = Vec::with_capacity(headers.len());
    let mut used_names: HashSet<String> = out_layer
        .schema
        .fields()
        .iter()
        .map(|f| f.name.to_ascii_lowercase())
        .collect();
    let mut csv_keep_cols = Vec::new();

    for (ci, raw_name) in headers.iter().enumerate() {
        if ci == csv_key_idx && vec_join_col.eq_ignore_ascii_case(csv_join_col) {
            continue;
        }
        let ftype = infer_field_type(&rows, ci);
        let name = allocate_unique_field_name(raw_name, is_shp, &mut used_names);
        out_layer.schema.add_field(FieldDef::new(name, ftype));
        csv_col_types.push(ftype);
        csv_keep_cols.push(ci);
    }

    // Build lookup map from normalized CSV join key to row.
    let mut lookup: HashMap<String, &Vec<String>> = HashMap::with_capacity(rows.len());
    for row in &rows {
        if let Some(k) = row.get(csv_key_idx) {
            lookup.insert(normalize_text_join_key(k), row);
        }
    }

    for feat in &src_layer.features {
        let mut attrs = Vec::with_capacity(out_layer.schema.fields().len());
        for &vi in &keep_vec_indices {
            attrs.push(feat.attributes.get(vi).cloned().unwrap_or(FieldValue::Null));
        }
        let matched = feat
            .attributes
            .get(vec_key_idx)
            .and_then(field_value_join_key)
            .and_then(|k| lookup.get(&k).copied());

        for (&ci, &ftype) in csv_keep_cols.iter().zip(&csv_col_types) {
            let val = matched
                .map(|row| parse_field_value(row.get(ci), ftype))
                .unwrap_or(FieldValue::Null);
            attrs.push(val);
        }

        out_layer.features.push(Feature {
            fid: feat.fid,
            geometry: feat.geometry.clone(),
            attributes: attrs,
        });
    }

    let count = out_layer.features.len();
    terrio::write_vector(&[out_layer], out_str, &VectorWriteOptions::default())?;
    Ok(count)
}

/// Converts a CSV file with latitude and longitude columns into a point vector
/// dataset in WGS 84 (`EPSG:4326`).
///
/// Mirrors `lidar.csv_points_to_shp` / `lidar.csv_to_shp`.
pub fn csv_points_to_vector(
    in_csv: impl AsRef<Path>,
    out_vector: impl AsRef<Path>,
    latitude_col: &str,
    longitude_col: &str,
) -> Result<usize> {
    let out_path = out_vector.as_ref();
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let (headers, rows) = read_csv_table(in_csv.as_ref())?;
    if headers.is_empty() {
        return Err(Error::Parse("CSV file has no header row".into()));
    }
    let lat_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case(latitude_col))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("latitude column '{latitude_col}' not found in CSV"))
        })?;
    let lon_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case(longitude_col))
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "longitude column '{longitude_col}' not found in CSV"
            ))
        })?;

    let is_shp = out_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("shp"));

    let mut layer = Layer::new("points".to_string()).with_geom_type(GeometryType::Point);
    SpatialRef::parse("EPSG:4326")?.apply_to_layer(&mut layer);

    let mut used_names = HashSet::new();
    let mut col_types = Vec::with_capacity(headers.len());
    for (ci, h) in headers.iter().enumerate() {
        let ftype = infer_field_type(&rows, ci);
        let name = allocate_unique_field_name(h, is_shp, &mut used_names);
        layer.schema.add_field(FieldDef::new(name, ftype));
        col_types.push(ftype);
    }

    for (fid, row) in rows.iter().enumerate() {
        let lat: f64 = row
            .get(lat_idx)
            .ok_or_else(|| Error::Parse(format!("missing latitude at row {}", fid + 1)))?
            .parse()
            .map_err(|e| Error::Parse(format!("invalid latitude at row {}: {e}", fid + 1)))?;
        if !lat.is_finite() || !(-90.0..=90.0).contains(&lat) {
            return Err(Error::Parse(format!(
                "latitude out of WGS 84 range [-90, 90] at row {}: {lat}",
                fid + 1
            )));
        }
        let lon: f64 = row
            .get(lon_idx)
            .ok_or_else(|| Error::Parse(format!("missing longitude at row {}", fid + 1)))?
            .parse()
            .map_err(|e| Error::Parse(format!("invalid longitude at row {}: {e}", fid + 1)))?;
        if !lon.is_finite() || !(-180.0..=180.0).contains(&lon) {
            return Err(Error::Parse(format!(
                "longitude out of WGS 84 range [-180, 180] at row {}: {lon}",
                fid + 1
            )));
        }

        let attrs: Vec<FieldValue> = (0..headers.len())
            .map(|ci| parse_field_value(row.get(ci), col_types[ci]))
            .collect();

        layer.features.push(Feature {
            fid: fid as u64,
            geometry: Some(Geometry::Point(Coord::xy(lon, lat))),
            attributes: attrs,
        });
    }

    let count = layer.features.len();
    terrio::write_vector(&[layer], out_str, &VectorWriteOptions::default())?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_utf8_bytes_never_splits_multibyte_chars() {
        let a = truncate_utf8_bytes("elevación_m", 10);
        assert!(a.len() <= 10);
        assert_eq!(a, "elevación"); // 10 UTF-8 bytes ('ó' is 2 bytes)

        let b = truncate_utf8_bytes("aaaaaaaaa°C", 10);
        assert!(b.len() <= 10);
        assert_eq!(b, "aaaaaaaaa"); // 9 bytes; '°' would cross byte 10
    }
}
