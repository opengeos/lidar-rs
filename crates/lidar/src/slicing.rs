//! Wu et al. (2019) level-set algorithm for delineating nested surface depressions.
//!
//! Pure-Rust port of `lidar/slicing.py`, using pure-Rust connected-component
//! labeling, moment-based region geometry, parallel per-region level-set
//! slicing, and `terrio` for GeoTIFF and Shapefile output.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, array2d_to_dataset, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::SpatialRef;
use terrio::ops::polygonize::{PolygonizeOptions, polygonize as terrio_polygonize};
use terrio::vector::Ring;
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer, VectorWriteOptions,
};

use crate::error::{Error, Result};
use crate::raster_ops::{ensure_no_path_collision, polygonize};
use crate::region::{RegionProps, region_group, region_props_affine};

/// Maximum vertical level-set slices permitted per region to guard against
/// pathologically tiny slicing intervals hanging execution.
const MAX_SLICES_PER_REGION: usize = 100_000;

/// A single depression in the nested depression hierarchy.
///
/// Matches the `Depression` class and 18-column `depressions_info.csv` schema
/// in `lidar/slicing.py`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NestedDepression {
    /// Unique depression ID (1-based).
    pub id: u32,
    /// Hierarchy level (1 = innermost leaf depression; higher = enclosing compound depression).
    pub level: u32,
    /// Number of pixels in this depression.
    pub count: usize,
    /// Surface area in map units squared.
    pub area: f64,
    /// Depression relief volume below `max_elev` (`avg_depth * area`, matching `lidar/slicing.py`).
    pub volume: f64,
    /// Mean depth relative to spill elevation.
    pub avg_depth: f64,
    /// Maximum depth (`max_elev - min_elev`).
    pub max_depth: f64,
    /// Minimum elevation inside the depression.
    pub min_elev: f64,
    /// Spill / boundary (maximum) elevation.
    pub max_elev: f64,
    /// Direct child depression IDs (inner neighbors in the hierarchy graph).
    pub children_ids: Vec<u32>,
    /// Parent top-level sink region ID (1-based).
    pub region_id: u32,
    /// Perimeter in map units.
    pub perimeter: f64,
    /// Major axis length in map units.
    pub major_axis: f64,
    /// Minor axis length in map units.
    pub minor_axis: f64,
    /// Elongatedness (`major_axis / minor_axis`).
    pub elongatedness: f64,
    /// Eccentricity of the equivalent ellipse.
    pub eccentricity: f64,
    /// Orientation in degrees.
    pub orientation: f64,
    /// Ratio of pixel count to bounding-box pixel area.
    pub area_bbox_ratio: f64,
    /// Direct enclosing parent depression ID (`0` for top-level root depressions).
    #[cfg_attr(feature = "serde", serde(default))]
    pub parent_id: u32,
    /// Isoperimetric circularity (`4 * pi * area / perimeter^2`, clamped to `[0, 1]`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub circularity: f64,
    /// Depth-to-equivalent-diameter ratio (`max_depth / (2 * sqrt(area / pi))`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_diam_ratio: f64,
    /// Hypsometric integral (`avg_depth / max_depth`, clamped to `[0, 1]`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub hypsometric_integral: f64,
    /// Mean rim-to-floor side-slope angle in degrees (`atan(max_depth / eq_radius)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rim_slope_deg: f64,
    /// Heuristic flag for likely linear ditch / road-embankment artifacts.
    #[cfg_attr(feature = "serde", serde(default))]
    pub artifact_flag: bool,
}

/// Options controlling [`delineate_depressions`] and [`crate::mounts::delineate_mounts`].
#[derive(Debug, Clone)]
pub struct DelineateOptions {
    /// Minimum number of pixels for a depression (`count > min_size`).
    pub min_size: usize,
    /// Minimum depression depth (in elevation units).
    pub min_depth: f64,
    /// Vertical slicing interval (positive value; sliced top-down).
    pub interval: f64,
    /// Whether to export individual shapefiles per hierarchy level (`shp-level/Single_level_*.shp`).
    pub bool_level_shp: bool,
    /// Whether to also save combined level GeoTIFFs when `bool_level_shp` is true.
    pub bool_comb_tif: bool,
    /// Whether to DEFLATE-compress output GeoTIFFs.
    pub compress: bool,
    /// Whether to also export a unified multi-level GeoPackage (`depressions.gpkg`) with full attribute names.
    pub export_gpkg: bool,
}

impl Default for DelineateOptions {
    fn default() -> Self {
        Self {
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            bool_level_shp: false,
            bool_comb_tif: false,
            compress: false,
            export_gpkg: false,
        }
    }
}

/// Output paths and hierarchy summary produced by [`delineate_depressions`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DelineateResult {
    /// Path to `depression_id.tif` (`i32` unique depression ID per pixel).
    pub depression_id_path: PathBuf,
    /// Path to `depression_level.tif` (`i32` hierarchy level per pixel).
    pub depression_level_path: PathBuf,
    /// Path to `depressions.shp` (polygonized depression IDs).
    pub shapefile_path: PathBuf,
    /// Path to `depressions_info.csv`.
    pub csv_path: PathBuf,
    /// Per-level shapefiles generated when `bool_level_shp = true`.
    pub level_shapefiles: Vec<PathBuf>,
    /// Optional unified multi-level GeoPackage path (`depressions.gpkg`) when `export_gpkg = true`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gpkg_path: Option<PathBuf>,
    /// All delineated nested depressions across all sink regions.
    pub depressions: Vec<NestedDepression>,
    /// Maximum hierarchy level observed (`0` if no depressions).
    pub max_level: u32,
    /// Number of top-level sink regions processed.
    pub num_regions: u32,
}

/// In-memory outputs of [`delineate_depressions_in_memory`].
#[derive(Debug, Clone)]
pub struct InMemoryDepressions {
    /// Unique depression ID per pixel (`i32`, 0 outside).
    pub depression_id: Array2D<i32>,
    /// Hierarchy level per pixel (`i32`, 0 outside).
    pub depression_level: Array2D<i32>,
    /// All delineated nested depressions.
    pub depressions: Vec<NestedDepression>,
    /// Maximum hierarchy level observed (`0` if no depressions).
    pub max_level: u32,
    /// Number of top-level sink regions processed.
    pub num_regions: u32,
}

fn extended_morphometrics(
    area: f64,
    perimeter: f64,
    avg_depth: f64,
    max_depth: f64,
    elongatedness: f64,
    area_bbox_ratio: f64,
) -> (f64, f64, f64, f64, bool) {
    let circularity = if perimeter > 0.0 && area > 0.0 {
        (4.0 * PI * area / (perimeter * perimeter)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eq_radius = if area > 0.0 { (area / PI).sqrt() } else { 0.0 };
    let depth_diam_ratio = if eq_radius > 0.0 {
        max_depth / (2.0 * eq_radius)
    } else {
        0.0
    };
    let hypsometric_integral = if max_depth > 0.0 {
        (avg_depth / max_depth).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let rim_slope_deg = if eq_radius > 0.0 && max_depth > 0.0 {
        (max_depth / eq_radius).atan().to_degrees()
    } else {
        0.0
    };
    let artifact_flag = elongatedness > 6.0 && area_bbox_ratio < 0.25;
    (
        circularity,
        depth_diam_ratio,
        hypsometric_integral,
        rim_slope_deg,
        artifact_flag,
    )
}

fn depression_from_props(id: u32, region_id: u32, p: &RegionProps) -> NestedDepression {
    let (circularity, depth_diam_ratio, hypsometric_integral, rim_slope_deg, artifact_flag) =
        extended_morphometrics(
            p.area,
            p.perimeter,
            p.mean_depth,
            p.max_depth,
            p.elongatedness,
            p.area_bbox_ratio,
        );
    NestedDepression {
        id,
        level: 1,
        count: p.count,
        area: p.area,
        volume: p.volume,
        avg_depth: p.mean_depth,
        max_depth: p.max_depth,
        min_elev: p.min_elev,
        max_elev: p.max_elev,
        children_ids: Vec::new(),
        region_id,
        perimeter: p.perimeter,
        major_axis: p.major_axis,
        minor_axis: p.minor_axis,
        elongatedness: p.elongatedness,
        eccentricity: p.eccentricity,
        orientation: p.orientation,
        area_bbox_ratio: p.area_bbox_ratio,
        parent_id: 0,
        circularity,
        depth_diam_ratio,
        hypsometric_integral,
        rim_slope_deg,
        artifact_flag,
    }
}

/// Runs the Wu et al. (2019) top-down level-set algorithm on a single sink region's
/// cropped elevation sub-image.
///
/// Local depression IDs start at `1` inside this region and are shifted by `obj_uid`
/// when merged into the global image.
#[allow(clippy::too_many_arguments)]
fn level_set_region(
    sub_img: &[f64],
    bw: usize,
    bh: usize,
    region_id: u32,
    min_size: usize,
    min_depth: f64,
    interval: f64,
    gt: &[f64; 6],
    res_x: f64,
    res_y: f64,
    cell_area: f64,
) -> Result<(Vec<u32>, Vec<NestedDepression>)> {
    let mut img = sub_img.to_vec();
    let mut level_img = vec![0u32; bw * bh];

    let mut min_elev = f64::INFINITY;
    let mut max_elev = f64::NEG_INFINITY;
    let mut valid_count = 0usize;
    for &v in &img {
        if v.is_finite() {
            if v < min_elev {
                min_elev = v;
            }
            if v > max_elev {
                max_elev = v;
            }
            valid_count += 1;
        }
    }
    if valid_count == 0 {
        return Ok((level_img, Vec::new()));
    }
    let mut sum_above_min = 0.0f64;
    for &v in &img {
        if v.is_finite() {
            sum_above_min += v - min_elev;
        }
    }

    let step = interval.abs();
    let span = max_elev - min_elev;
    let est_slices = (span / step).ceil();
    if !est_slices.is_finite() || est_slices > MAX_SLICES_PER_REGION as f64 {
        return Err(Error::InvalidArgument(format!(
            "interval={step} produces too many slices ({est_slices}) for elevation range {span:.4} in region {region_id} (max {MAX_SLICES_PER_REGION})"
        )));
    }

    // Degenerate 1-row or 1-column depression (matches slicing.py lines 289-334).
    if bh == 1 || bw == 1 {
        let max_depth = max_elev - min_elev;
        if max_depth < min_depth {
            return Ok((level_img, Vec::new()));
        }
        let cells = valid_count;
        let area = cells as f64 * cell_area;
        let mean_depth = ((max_elev - min_elev) * valid_count as f64 - sum_above_min).max(0.0)
            / valid_count as f64;
        let volume = mean_depth * area;
        let (perimeter, major_axis, minor_axis, elongatedness, orientation) = if bh == 1 {
            let maj = valid_count as f64 * res_x;
            let min = (cell_area / res_x).min(res_y);
            let perim = 2.0 * (maj + min);
            (perim, maj, min, maj / min, 0.0)
        } else {
            let maj = valid_count as f64 * res_y;
            let min = (cell_area / res_y).min(res_x);
            let perim = 2.0 * (maj + min);
            (perim, maj, min, maj / min, 90.0)
        };
        let area_bbox_ratio = valid_count as f64 / (bh * bw) as f64;
        let (circularity, depth_diam_ratio, hypsometric_integral, rim_slope_deg, artifact_flag) =
            extended_morphometrics(
                area,
                perimeter,
                mean_depth,
                max_depth,
                elongatedness,
                area_bbox_ratio,
            );
        let dep = NestedDepression {
            id: 1,
            level: 1,
            count: cells,
            area,
            volume,
            avg_depth: mean_depth,
            max_depth,
            min_elev,
            max_elev,
            children_ids: Vec::new(),
            region_id,
            perimeter,
            major_axis,
            minor_axis,
            elongatedness,
            eccentricity: 1.0,
            orientation,
            area_bbox_ratio,
            parent_id: 0,
            circularity,
            depth_diam_ratio,
            hypsometric_integral,
            rim_slope_deg,
            artifact_flag,
        };
        for i in 0..img.len() {
            if img[i].is_finite() {
                level_img[i] = 1;
            }
        }
        return Ok((level_img, vec![dep]));
    }

    let mut unique_id = 0u32;
    let mut parent_counts: BTreeMap<u32, usize> = BTreeMap::new();
    let mut nbr_indices: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut dep_list: Vec<NestedDepression> = Vec::new();

    // Top-down slicing: always evaluate the full footprint at slice_idx == 0 (elev = max_elev)
    // so flat depressions (max_elev == min_elev) are retained when min_depth == 0.0.
    let mut slice_idx = 0usize;
    loop {
        let elev = max_elev - (slice_idx as f64) * step;
        if slice_idx > 0
            && (if min_depth > 0.0 {
                elev <= min_elev
            } else {
                elev < min_elev
            })
        {
            break;
        }
        slice_idx += 1;

        for v in &mut img {
            if v.is_finite() && *v > elev {
                *v = f64::NAN;
            }
        }

        let (label_objects, nb_labels) = region_group(&img, bw, bh, min_size, |v| v.is_finite());
        if nb_labels == 0 {
            break;
        }

        let objects = region_props_affine(&label_objects, nb_labels, &img, bw, bh, gt);
        for (i, obj) in objects.iter().enumerate() {
            if obj.max_depth < min_depth {
                continue;
            }
            let (fr, fc) = obj.first_pixel;
            let parent_id = level_img[fr * bw + fc];

            if parent_counts.is_empty() || parent_id == 0 {
                unique_id += 1;
                dep_list.push(depression_from_props(unique_id, region_id, obj));
                parent_counts.insert(unique_id, 0);
                nbr_indices.insert(unique_id, Vec::new());

                // Paint root depression ID onto level_img.
                let (r0, c0, r1, c1) = obj.bbox;
                for r in r0..r1 {
                    let off = r * bw;
                    for c in c0..c1 {
                        if label_objects[off + c] == obj.label {
                            level_img[off + c] = unique_id;
                        }
                    }
                }
            } else {
                if let Some(cnt) = parent_counts.get_mut(&parent_id) {
                    *cnt += 1;
                }
                if let Some(nbrs) = nbr_indices.get_mut(&parent_id) {
                    nbrs.push(i);
                }
            }
        }

        let keys: Vec<u32> = parent_counts.keys().copied().collect();
        for key in keys {
            let count = parent_counts.get(&key).copied().unwrap_or(0);
            if count > 1 {
                let children = nbr_indices.get(&key).cloned().unwrap_or_default();
                for child_idx in children {
                    let obj = &objects[child_idx];
                    unique_id += 1;
                    dep_list.push(depression_from_props(unique_id, region_id, obj));
                    dep_list[(key - 1) as usize].children_ids.push(unique_id);
                    parent_counts.insert(unique_id, 0);
                    nbr_indices.insert(unique_id, Vec::new());

                    let (r0, c0, r1, c1) = obj.bbox;
                    for r in r0..r1 {
                        let off = r * bw;
                        for c in c0..c1 {
                            if label_objects[off + c] == obj.label {
                                level_img[off + c] = unique_id;
                            }
                        }
                    }
                }
                parent_counts.remove(&key);
                nbr_indices.remove(&key);
            } else {
                if let Some(cnt) = parent_counts.get_mut(&key) {
                    *cnt = 0;
                }
                if let Some(nbrs) = nbr_indices.get_mut(&key) {
                    nbrs.clear();
                }
            }
        }
    }

    update_levels(&mut dep_list);
    Ok((level_img, dep_list))
}

/// Bottom-up hierarchy level assignment: leaf depressions have `level = 1`,
/// parent depressions have `level = 1 + max(child.level)`, and each child's
/// `parent_id` is populated from its enclosing parent.
fn update_levels(dep_list: &mut [NestedDepression]) {
    for i in (0..dep_list.len()).rev() {
        let cur_id = dep_list[i].id;
        if dep_list[i].children_ids.is_empty() {
            dep_list[i].level = 1;
        } else {
            let mut max_child_level = 0u32;
            let children = dep_list[i].children_ids.clone();
            for cid in children {
                let cidx = (cid - 1) as usize;
                let child_lvl = dep_list[cidx].level;
                if child_lvl > max_child_level {
                    max_child_level = child_lvl;
                }
                dep_list[cidx].parent_id = cur_id;
            }
            dep_list[i].level = max_child_level + 1;
        }
    }
}

/// Writes nested depression hierarchy attributes (24 columns: original 18 columns plus
/// `parent_id`, `circularity`, `depth_diam_ratio`, `hypsometric_integral`, `rim_slope_deg`,
/// and `artifact_flag`) to a CSV file.
pub fn write_depressions_csv(
    depressions: &[NestedDepression],
    csv_path: impl AsRef<Path>,
) -> Result<()> {
    let file = File::create(csv_path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "id,level,count,area,volume,avg_depth,max_depth,min_elev,max_elev,children_id,region_id,perimeter,major_axis,minor_axis,elongatedness,eccentricity,orientation,area_bbox_ratio,parent_id,circularity,depth_diam_ratio,hypsometric_integral,rim_slope_deg,artifact_flag"
    )?;
    for d in depressions {
        let children_str = if d.children_ids.is_empty() {
            "[]".to_string()
        } else {
            let joined = d
                .children_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(": ");
            format!("[{joined}]")
        };
        writeln!(
            w,
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{:.4},{:.4},{:.4},{:.2},{}",
            d.id,
            d.level,
            d.count,
            d.area,
            d.volume,
            d.avg_depth,
            d.max_depth,
            d.min_elev,
            d.max_elev,
            children_str,
            d.region_id,
            d.perimeter,
            d.major_axis,
            d.minor_axis,
            d.elongatedness,
            d.eccentricity,
            d.orientation,
            d.area_bbox_ratio,
            d.parent_id,
            d.circularity,
            d.depth_diam_ratio,
            d.hypsometric_integral,
            d.rim_slope_deg,
            d.artifact_flag,
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Exports all nested depressions into a unified multi-level vector file (`.gpkg` or `.geojson`)
/// with full untrimmed attribute names.
pub fn write_depressions_vector(
    depression_id: &Array2D<i32>,
    depressions: &[NestedDepression],
    out_path: impl AsRef<Path>,
) -> Result<()> {
    let out_path = out_path.as_ref();
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let out_str = out_path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", out_path.display())))?;

    let ds = array2d_to_dataset(depression_id)?;
    let poly_layer = terrio_polygonize(
        &ds,
        &PolygonizeOptions {
            band: 0,
            field_name: "id".to_string(),
            layer_name: "depressions".to_string(),
            connectedness8: false,
            ignore_nodata: false,
            mask: None,
        },
    )?;

    let dep_map: HashMap<u32, &NestedDepression> = depressions.iter().map(|d| (d.id, d)).collect();
    // Dissolve polygon parts sharing the same depression ID into one feature so multi-part
    // parent rims never duplicate `area`/`volume` attributes in vector exports.
    let mut polys_by_id: BTreeMap<u32, Vec<(Ring, Vec<Ring>)>> = BTreeMap::new();
    for feat in poly_layer.features {
        let did = match feat.attributes.first() {
            Some(FieldValue::Integer(i)) if *i > 0 => *i as u32,
            Some(FieldValue::Float(f)) if *f > 0.0 && f.is_finite() => *f as u32,
            _ => continue,
        };
        if !dep_map.contains_key(&did) {
            continue;
        }
        match feat.geometry {
            Some(Geometry::Polygon {
                exterior,
                interiors,
            }) => {
                polys_by_id
                    .entry(did)
                    .or_default()
                    .push((exterior, interiors));
            }
            Some(Geometry::MultiPolygon(polys)) => {
                polys_by_id.entry(did).or_default().extend(polys);
            }
            _ => {}
        }
    }

    let has_multipart = polys_by_id.values().any(|parts| parts.len() > 1);
    let mut layer = Layer::new("depressions".to_string()).with_geom_type(if has_multipart {
        GeometryType::MultiPolygon
    } else {
        GeometryType::Polygon
    });
    layer.crs = poly_layer.crs.clone();
    if layer.crs.is_none()
        && !depression_id.projection.is_empty()
        && let Ok(srs) = SpatialRef::parse(&depression_id.projection)
    {
        srs.apply_to_layer(&mut layer);
    }

    for (name, ftype) in [
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
    ] {
        layer.schema.add_field(FieldDef::new(name, ftype));
    }

    for (did, mut parts) in polys_by_id {
        let Some(d) = dep_map.get(&did) else {
            continue;
        };
        if parts.is_empty() {
            continue;
        }
        let geometry = if has_multipart {
            Some(Geometry::MultiPolygon(parts))
        } else {
            let (exterior, interiors) = parts.remove(0);
            Some(Geometry::Polygon {
                exterior,
                interiors,
            })
        };
        layer.features.push(Feature {
            fid: (layer.features.len() + 1) as u64,
            geometry,
            attributes: vec![
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
            ],
        });
    }

    terrio::write_vector(&[layer], out_str, &VectorWriteOptions::default())?;
    Ok(())
}

/// Runs the level-set nested depression delineation algorithm in memory on a sink raster.
///
/// The `sink` raster should have valid elevation values inside sink regions and
/// its declared NoData sentinel (e.g., `0` from [`crate::filling::extract_sinks`])
/// outside. Depressions whose `max_depth < min_depth` are filtered out.
pub fn delineate_depressions_in_memory(
    sink: &Array2D<f64>,
    min_size: usize,
    min_depth: f64,
    interval: f64,
) -> Result<InMemoryDepressions> {
    if !min_depth.is_finite() || min_depth < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "min_depth must be finite and >= 0, got {min_depth}"
        )));
    }
    if !interval.is_finite() || interval == 0.0 {
        return Err(Error::InvalidArgument(format!(
            "interval must be finite and non-zero, got {interval}"
        )));
    }
    let w = sink.width();
    let h = sink.height();
    let n = w
        .checked_mul(h)
        .ok_or_else(|| Error::DimensionMismatch(format!("raster dimensions overflow ({w}x{h})")))?;

    let no_data_raw = sink.no_data();
    let gt = sink.geotransform;
    let res_x = gt[1].hypot(gt[4]);
    let res_y = gt[2].hypot(gt[5]);
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let (res_x, res_y, cell_area) = if res_x > 0.0
        && res_x.is_finite()
        && res_y > 0.0
        && res_y.is_finite()
        && det > 0.0
        && det.is_finite()
    {
        (res_x, res_y, det)
    } else {
        (1.0, 1.0, 1.0)
    };

    // Treat only non-finite cells and the declared NoData sentinel as background,
    // preserving valid negative (below-sea-level) sink elevations.
    let is_valid_sink = |v: f64| -> bool { v.is_finite() && !v.rd_eq(no_data_raw) };

    let (label_objects, nb_labels) = region_group(sink.data(), w, h, min_size, is_valid_sink);

    // Extract bounding boxes for each top-level region.
    let k = nb_labels as usize;
    let mut bboxes = vec![(usize::MAX, usize::MAX, 0usize, 0usize); k];
    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let lbl = label_objects[off + c];
            if lbl == 0 {
                continue;
            }
            let b = &mut bboxes[(lbl - 1) as usize];
            if r < b.0 {
                b.0 = r;
            }
            if c < b.1 {
                b.1 = c;
            }
            if r + 1 > b.2 {
                b.2 = r + 1;
            }
            if c + 1 > b.3 {
                b.3 = c + 1;
            }
        }
    }

    type RegionBbox = (usize, usize, usize, usize);
    type RegionResult = Result<(usize, RegionBbox, Vec<u32>, Vec<NestedDepression>)>;

    // Build cropped sub-images per region and run level_set independently (parallel when enabled).
    let run_one = |idx: usize| -> RegionResult {
        let region_id = (idx + 1) as u32;
        let bbox = bboxes[idx];
        let (r0, c0, r1, c1) = bbox;
        let bh = r1 - r0;
        let bw = c1 - c0;
        let mut sub_img = vec![f64::NAN; bw * bh];
        for r in r0..r1 {
            let src_off = r * w;
            let dst_off = (r - r0) * bw;
            for c in c0..c1 {
                if label_objects[src_off + c] == region_id {
                    sub_img[dst_off + (c - c0)] = sink.get_i(src_off + c);
                }
            }
        }
        let (local_obj, local_deps) = level_set_region(
            &sub_img, bw, bh, region_id, min_size, min_depth, interval, &gt, res_x, res_y,
            cell_area,
        )?;
        Ok((idx, bbox, local_obj, local_deps))
    };

    #[cfg(feature = "parallel")]
    let mut region_results: Vec<_> = (0..k)
        .into_par_iter()
        .map(run_one)
        .collect::<Result<Vec<_>>>()?;
    #[cfg(not(feature = "parallel"))]
    let mut region_results: Vec<_> = (0..k).map(run_one).collect::<Result<Vec<_>>>()?;

    // Ensure deterministic region ordering before assigning global sequential IDs.
    region_results.sort_by_key(|r| r.0);

    let mut obj_vec = vec![0i32; n];
    let mut level_vec = vec![0i32; n];
    let mut global_deps: Vec<NestedDepression> = Vec::new();
    let mut max_level = 0u32;
    let mut uid_offset = 0u32;

    for (_idx, bbox, local_obj, mut local_deps) in region_results {
        let (r0, c0, r1, c1) = bbox;
        let bw = c1 - c0;

        for dep in &mut local_deps {
            dep.id += uid_offset;
            if dep.parent_id > 0 {
                dep.parent_id += uid_offset;
            }
            for cid in &mut dep.children_ids {
                *cid += uid_offset;
            }
            if dep.level > max_level {
                max_level = dep.level;
            }
        }

        for r in r0..r1 {
            let dst_off = r * w;
            let src_off = (r - r0) * bw;
            for c in c0..c1 {
                let local_id = local_obj[src_off + (c - c0)];
                if local_id > 0 {
                    let global_id = local_id + uid_offset;
                    let lvl = local_deps[(local_id - 1) as usize].level;
                    obj_vec[dst_off + c] = global_id as i32;
                    level_vec[dst_off + c] = lvl as i32;
                }
            }
        }

        uid_offset += local_deps.len() as u32;
        global_deps.extend(local_deps);
    }

    let mut depression_id = Array2D::from_vec(w, h, obj_vec)?;
    depression_id.copy_metadata_from(sink);
    depression_id.set_no_data(0);

    let mut depression_level = Array2D::from_vec(w, h, level_vec)?;
    depression_level.copy_metadata_from(sink);
    depression_level.set_no_data(0);

    Ok(InMemoryDepressions {
        depression_id,
        depression_level,
        depressions: global_deps,
        max_level,
        num_regions: nb_labels,
    })
}

/// Extracts individual hierarchy-level shapefiles (`Single_level_01.shp`, ...)
/// matching `extract_levels` in `lidar/slicing.py`.
fn extract_levels(
    level_img: &Array2D<i32>,
    obj_img: &Array2D<i32>,
    min_size: usize,
    out_shp_dir: &Path,
    bool_comb_tif: bool,
    compress: bool,
) -> Result<Vec<PathBuf>> {
    let w = level_img.width();
    let h = level_img.height();
    let max_level = level_img.data().iter().copied().max().unwrap_or(0);
    if max_level <= 0 {
        return Ok(Vec::new());
    }

    std::fs::create_dir_all(out_shp_dir)?;
    let digits = ((max_level as f64).log10().floor() as usize) + 1;
    let mut img: Vec<i32> = level_img.data().to_vec();
    let mut shapefiles = Vec::with_capacity(max_level as usize);

    for lvl in 1..=max_level {
        for v in &mut img {
            if *v > 0 && *v <= lvl {
                *v = lvl;
            }
        }

        let tmp_f64: Vec<f64> = img
            .iter()
            .map(|&v| if v > 0 && v <= lvl { v as f64 } else { 0.0 })
            .collect();

        if bool_comb_tif {
            let comb_i32: Vec<i32> = img
                .iter()
                .map(|&v| if v > 0 && v <= lvl { v } else { 0 })
                .collect();
            let mut comb_arr = Array2D::from_vec(w, h, comb_i32)?;
            comb_arr.copy_metadata_from(level_img);
            comb_arr.set_no_data(0);
            let comb_name = format!("Combined_level_{lvl:0digits$}.tif");
            save_raster(
                &comb_arr,
                out_shp_dir.join(comb_name),
                Some("extract_levels: combined"),
                compress,
            )?;
        }

        let (lbl_objects, nb_labels) = region_group(&tmp_f64, w, h, min_size, |v| v > 0.0);

        // For each grouped component, find max level_img and min obj_img (matching slicing.py:669-676).
        let k = nb_labels as usize;
        let mut max_lvl_per_reg = vec![0i32; k];
        let mut min_uid_per_reg = vec![i32::MAX; k];
        for (i, &lbl) in lbl_objects.iter().enumerate() {
            if lbl == 0 {
                continue;
            }
            let idx = (lbl - 1) as usize;
            let lv = level_img.get_i(i);
            let oid = obj_img.get_i(i);
            if lv > max_lvl_per_reg[idx] {
                max_lvl_per_reg[idx] = lv;
            }
            if oid < min_uid_per_reg[idx] {
                min_uid_per_reg[idx] = oid;
            }
        }

        let mut sin_vec = vec![0i32; w * h];
        for (i, &lbl) in lbl_objects.iter().enumerate() {
            if lbl == 0 {
                continue;
            }
            let idx = (lbl - 1) as usize;
            if max_lvl_per_reg[idx] >= lvl && min_uid_per_reg[idx] != i32::MAX {
                sin_vec[i] = min_uid_per_reg[idx];
            }
        }

        let mut sin_arr = Array2D::from_vec(w, h, sin_vec)?;
        sin_arr.copy_metadata_from(level_img);
        sin_arr.set_no_data(0);

        let shp_name = format!("Single_level_{lvl:0digits$}.shp");
        let shp_path = out_shp_dir.join(shp_name);
        crate::raster_ops::polygonize_array(&sin_arr, &shp_path, "id")?;
        shapefiles.push(shp_path);
    }

    Ok(shapefiles)
}

/// Delineates the nested hierarchy of surface depressions from a sink GeoTIFF
/// using the Wu et al. (2019) level-set method.
///
/// Mirrors `lidar.DelineateDepressions` from `opengeos/lidar`.
pub fn delineate_depressions(
    in_sink: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &DelineateOptions,
) -> Result<DelineateResult> {
    let in_sink = in_sink.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let out_obj_file = out_dir.join("depression_id.tif");
    let out_level_file = out_dir.join("depression_level.tif");
    let out_vec_file = out_dir.join("depressions.shp");
    let out_csv_file = out_dir.join("depressions_info.csv");
    let out_gpkg_file = out_dir.join("depressions.gpkg");
    let out_shp_dir = out_dir.join("shp-level");

    let mut out_paths: Vec<&Path> = vec![
        out_obj_file.as_path(),
        out_level_file.as_path(),
        out_vec_file.as_path(),
        out_csv_file.as_path(),
    ];
    if options.export_gpkg {
        out_paths.push(out_gpkg_file.as_path());
    }
    ensure_no_path_collision(in_sink, &out_paths)?;

    let sink = load_raster::<f64>(in_sink, None)?;
    let mem = delineate_depressions_in_memory(
        &sink,
        options.min_size,
        options.min_depth,
        options.interval,
    )?;

    save_raster(
        &mem.depression_level,
        &out_level_file,
        Some("delineate_depressions: level"),
        options.compress,
    )?;
    save_raster(
        &mem.depression_id,
        &out_obj_file,
        Some("delineate_depressions: id"),
        options.compress,
    )?;

    polygonize(&out_obj_file, &out_vec_file, "id")?;
    write_depressions_csv(&mem.depressions, &out_csv_file)?;

    let gpkg_path = if options.export_gpkg {
        let p = out_dir.join("depressions.gpkg");
        write_depressions_vector(&mem.depression_id, &mem.depressions, &p)?;
        Some(p)
    } else {
        None
    };

    let level_shapefiles = if options.bool_level_shp {
        // Remove only previously exported Single_level_* / Combined_level_* artifacts,
        // preserving any unrelated caller files inside shp-level/.
        if out_shp_dir.exists() {
            for entry in std::fs::read_dir(&out_shp_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file()
                    && let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && (name.starts_with("Single_level_") || name.starts_with("Combined_level_"))
                {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        extract_levels(
            &mem.depression_level,
            &mem.depression_id,
            options.min_size,
            &out_shp_dir,
            options.bool_comb_tif,
            options.compress,
        )?
    } else {
        Vec::new()
    };

    Ok(DelineateResult {
        depression_id_path: out_obj_file,
        depression_level_path: out_level_file,
        shapefile_path: out_vec_file,
        csv_path: out_csv_file,
        level_shapefiles,
        gpkg_path,
        depressions: mem.depressions,
        max_level: mem.max_level,
        num_regions: mem.num_regions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_nested_double_bowl_splits_into_two_levels() {
        // 30x16 sink with two inner bowls separated by a saddle ridge.
        let (w, h) = (30, 16);
        let mut data = vec![0.0f64; w * h];
        for r in 1..h - 1 {
            for c in 1..w - 1 {
                let d1 = ((r as f64 - 8.0).powi(2) + (c as f64 - 8.0).powi(2)).sqrt();
                let d2 = ((r as f64 - 8.0).powi(2) + (c as f64 - 21.0).powi(2)).sqrt();
                let z = (d1.min(d2) * 0.5 + 10.0).min(15.0);
                data[r * w + c] = z;
            }
        }
        let mut sink = Array2D::from_vec(w, h, data).unwrap();
        sink.set_no_data(0.0);

        let res = delineate_depressions_in_memory(&sink, 10, 0.3, 0.3).unwrap();
        assert_eq!(res.num_regions, 1);
        assert!(
            res.max_level >= 2,
            "expected nested hierarchy, got max_level={}",
            res.max_level
        );
        assert!(
            res.depressions.len() >= 3,
            "expected parent + 2 children, got {}",
            res.depressions.len()
        );
        assert_eq!(res.depressions[0].level, res.max_level);
        assert_eq!(res.depressions[0].children_ids.len(), 2);
    }

    #[test]
    fn top_slice_multiple_roots_are_preserved() {
        // Two disconnected 4x4 sub-components passed directly to level_set_region.
        let (bw, bh) = (10, 4);
        let mut sub = vec![f64::NAN; bw * bh];
        for r in 0..4 {
            for c in 0..4 {
                sub[r * bw + c] = 10.0 + (r + c) as f64 * 0.1;
                sub[r * bw + (c + 6)] = 10.0 + (r + c) as f64 * 0.1;
            }
        }
        let gt = [0.0, 1.0, 0.0, bh as f64, 0.0, -1.0];
        let (obj, deps) =
            level_set_region(&sub, bw, bh, 1, 2, 0.0, 0.2, &gt, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(deps.len(), 2);
        assert!(obj[0] > 0 && obj[6] > 0 && obj[0] != obj[6]);
    }

    #[test]
    fn enforces_min_depth_and_handles_flat_and_negative_sinks() {
        // Flat 2x2 sink at negative elevation (-10.0) with nodata=0.0.
        let mut flat = Array2D::from_vec(2, 2, vec![-10.0; 4]).unwrap();
        flat.set_no_data(0.0);
        let res_zero_depth = delineate_depressions_in_memory(&flat, 0, 0.0, 0.5).unwrap();
        assert_eq!(res_zero_depth.depressions.len(), 1);
        assert_eq!(res_zero_depth.depressions[0].min_elev, -10.0);

        // Same flat sink filtered out when min_depth > 0.
        let res_filtered = delineate_depressions_in_memory(&flat, 0, 0.5, 0.5).unwrap();
        assert!(res_filtered.depressions.is_empty());

        // Tiny interval rejected cleanly instead of hanging.
        let mut tiny = Array2D::from_vec(2, 2, vec![10.0, 10.0, 11.0, 11.0]).unwrap();
        tiny.set_no_data(0.0);
        assert!(delineate_depressions_in_memory(&tiny, 0, 0.0, 1e-30).is_err());
    }
}
