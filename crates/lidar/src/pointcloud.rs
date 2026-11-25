//! Pure-Rust LiDAR point cloud I/O, format conversion, ground classification,
//! outlier removal, height-above-ground normalization, COPC octree windowing,
//! and DSM / DTM / CHM gridding via `terrio`.
//!
//! Replaces `laspy` and `WhiteboxTools` (`read_lidar`, `write_lidar`,
//! `convert_lidar`, `lidar_to_dsm`) from `lidar/common.py` with zero C dependencies,
//! and adds ASPRS classification / return filtering, Cloth Simulation Filter (CSF),
//! Progressive Morphological Filter (PMF), and Simple Morphological Filter (SMRF)
//! ground classification, Extended Local Minimum (ELM) and statistical outlier
//! filtering, Height Above Ground (HAG) normalization, COPC octree LOD/spatial
//! windowing, bare-earth DTM generation, and Canopy Height Model (CHM) creation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub use terrio::pointcloud::{
    Dimension, LasHeader, PointCloud, PointFilter, PointFormat, PointReadOptions,
    PointWriteOptions, ReturnKind, ValueType, Values,
    filters::{
        CsfOptions, HagOptions, OutlierMethod, OutlierOptions, PmfOptions, SmrfOptions,
        csf as terrio_csf, csf_classes, elm as terrio_elm, hag_delaunay, hag_nn,
        outlier as terrio_outlier, outlier_mask, pmf as terrio_pmf, pmf_classes,
        smrf as terrio_smrf, smrf_classes,
    },
    grid::{GridMethod, GridOptions, GridStat},
};

use richdem::{Array2D, DemNum, array2d_to_dataset};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::formats::RasterDriver;
use terrio::{Geometry, RasterWriteOptions, VectorOpenOptions};

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
    /// Simple Morphological Filter (Pingel et al., 2013) via `terrio`.
    Smrf,
}

impl FromStr for GroundFilterMethod {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "preclassified" => Ok(Self::Preclassified),
            "csf" => Ok(Self::Csf),
            "pmf" => Ok(Self::Pmf),
            "smrf" => Ok(Self::Smrf),
            other => Err(Error::InvalidArgument(format!(
                "unknown ground filter method '{other}'; expected preclassified, csf, pmf, or smrf"
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

/// Configuration for [`classify_ground_with_options`], pairing a [`GroundFilterMethod`]
/// with tunable algorithm parameters for CSF, PMF, and SMRF.
#[derive(Debug, Clone, Default)]
pub struct GroundClassifyOptions {
    /// Ground classification algorithm (`GroundFilterMethod::Csf` by default).
    pub method: GroundFilterMethod,
    /// Cloth Simulation Filter parameters (Zhang et al., 2016).
    pub csf: CsfOptions,
    /// Progressive Morphological Filter parameters (Zhang et al., 2003).
    pub pmf: PmfOptions,
    /// Simple Morphological Filter parameters (Pingel et al., 2013).
    pub smrf: SmrfOptions,
}

/// Options for [`lidar_to_dsm`] and [`points_to_dsm`].
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
    /// Write a DEFLATE-compressed GeoTIFF when `true` and `cog` is `false` (default `false`).
    pub compress: bool,
    /// Write a Cloud Optimized GeoTIFF (COG) with overviews when `true` (default `false`).
    pub cog: bool,
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
            compress: false,
            cog: false,
        }
    }
}

/// Options for [`lidar_to_dtm`], [`lidar_to_chm`], [`points_to_dtm`], and [`points_to_chm`].
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
    /// Strip ASPRS low/high noise outliers (classes 7 and 18) via ELM before ground selection (default `false`).
    pub remove_outliers: bool,
    /// ELM grid cell size in CRS horizontal units when `remove_outliers` is `true` (default `10.0`).
    pub outlier_cell: f64,
    /// ELM elevation threshold in CRS vertical units when `remove_outliers` is `true` (default `1.0`).
    pub outlier_threshold: f64,
    /// Tunable Cloth Simulation Filter parameters used when `ground_method == GroundFilterMethod::Csf`.
    pub csf: CsfOptions,
    /// Tunable Progressive Morphological Filter parameters used when `ground_method == GroundFilterMethod::Pmf`.
    pub pmf: PmfOptions,
    /// Tunable Simple Morphological Filter parameters used when `ground_method == GroundFilterMethod::Smrf`.
    pub smrf: SmrfOptions,
    /// Optional Radius Outlier Removal (ROR) search radius; when `Some(r)`, points with fewer than
    /// `ror_min_neighbors` neighbors within Euclidean radius `r` are removed before ground selection.
    pub ror_radius: Option<f64>,
    /// Minimum neighbor count for ROR when `ror_radius` is `Some` (default `2`).
    pub ror_min_neighbors: usize,
    /// Optional Statistical Outlier Removal (SOR) neighbor count `mean_k`; when `Some(k)`, statistical
    /// outliers beyond `sor_sigma` standard deviations are removed before ground selection.
    pub sor_k: Option<usize>,
    /// Standard-deviation multiplier for SOR when `sor_k` is `Some` (default `2.0`).
    pub sor_sigma: f64,
    /// Write a DEFLATE-compressed GeoTIFF when `true` and `cog` is `false` (default `false`).
    pub compress: bool,
    /// Write a Cloud Optimized GeoTIFF (COG) with overviews when `true` (default `false`).
    pub cog: bool,
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
            remove_outliers: false,
            outlier_cell: 10.0,
            outlier_threshold: 1.0,
            csf: CsfOptions::default(),
            pmf: PmfOptions::default(),
            smrf: SmrfOptions::default(),
            ror_radius: None,
            ror_min_neighbors: 2,
            sor_k: None,
            sor_sigma: 2.0,
            compress: false,
            cog: false,
        }
    }
}

/// Options for Khosravipour et al. (2014) pit-free Canopy Height Model generation
/// ([`points_to_chm_pit_free`] and [`lidar_to_chm_pit_free`]).
#[derive(Debug, Clone)]
pub struct PitFreeChmOptions {
    /// Output raster pixel resolution (default `1.0`).
    pub resolution: f64,
    /// Height-above-ground thresholds at which partial canopy TIN surfaces are constructed
    /// (default `vec![0.0, 2.0, 5.0, 10.0, 15.0]`).
    pub height_thresholds: Vec<f64>,
    /// Maximum 2D triangle edge length for partial canopy TINs (`h_k > 0.0`). Triangles with any
    /// edge longer than this limit are skipped so interior crown pits are bridged without bridging
    /// true open gaps between trees. When `None`, defaults to `3.0 * resolution`.
    pub max_triangle_edge: Option<f64>,
    /// Bare-earth DTM, ground-classification, and outlier-filtering options.
    pub dtm: LidarToDtmOptions,
    /// Write a DEFLATE-compressed GeoTIFF when `true` and `cog` is `false` (default `false`).
    pub compress: bool,
    /// Write a Cloud Optimized GeoTIFF (COG) with overviews when `true` (default `false`).
    pub cog: bool,
}

impl Default for PitFreeChmOptions {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            height_thresholds: vec![0.0, 2.0, 5.0, 10.0, 15.0],
            max_triangle_edge: None,
            dtm: LidarToDtmOptions::default(),
            compress: false,
            cog: false,
        }
    }
}

/// Windowing, COPC octree level-of-detail (LOD), polygon clipping, and point filtering options
/// for [`read_lidar_windowed`] and [`read_lidar_in_polygon`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LidarReadOptions {
    /// Optional 2D spatial bounding box `[min_x, min_y, max_x, max_y]` in the file's CRS.
    pub bounds: Option<[f64; 4]>,
    /// Optional maximum COPC octree level (`0` = root; ignored for non-COPC formats).
    pub max_level: Option<u32>,
    /// Optional target point spacing in CRS units for COPC LOD selection (ignored for non-COPC formats).
    pub resolution: Option<f64>,
    /// Optional approximate maximum point count (COPC chooses octree level; other formats thin by stride).
    ///
    /// When [`Self::polygon_path`] is set, the reader fetches the polygon bounding box without
    /// reader-level point capping (still applying COPC octree chunk pruning against the bbox) and
    /// applies `max_points` after exact polygon clipping so small polygons inside large tiles are
    /// not starved by bbox-level thinning.
    pub max_points: Option<usize>,
    /// Optional ASPRS classification codes to keep (empty keeps all).
    pub classifications: Vec<u8>,
    /// Optional pulse return kinds to keep (empty keeps all).
    pub return_kinds: Vec<ReturnKind>,
    /// Optional minimum Z threshold for point inclusion.
    pub min_z: Option<f64>,
    /// Optional maximum Z threshold for point inclusion.
    pub max_z: Option<f64>,
    /// Optional path to a polygon vector file (GeoJSON, Shapefile, GeoPackage, etc.) used for
    /// bbox-accelerated COPC/LAS streaming followed by exact ray-casting point-in-polygon filtering.
    pub polygon_path: Option<PathBuf>,
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

/// Parsed 2D polygon with exterior ring, optional interior hole rings, and precomputed bbox.
struct Polygon2D {
    bbox: [f64; 4],
    exterior: Vec<(f64, f64)>,
    interiors: Vec<Vec<(f64, f64)>>,
}

fn ring_to_xy(ring: &terrio::vector::Ring) -> Vec<(f64, f64)> {
    ring.coords().iter().map(|c| (c.x, c.y)).collect()
}

fn collect_polygons_from_geometry(geom: &Geometry, out: &mut Vec<Polygon2D>) {
    match geom {
        Geometry::Polygon {
            exterior,
            interiors,
        } => {
            let ext = ring_to_xy(exterior);
            if ext.len() < 3 {
                return;
            }
            let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
            let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &(x, y) in &ext {
                if x.is_finite() && y.is_finite() {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
            if !min_x.is_finite() || !min_y.is_finite() {
                return;
            }
            let holes: Vec<Vec<(f64, f64)>> = interiors
                .iter()
                .map(ring_to_xy)
                .filter(|r| r.len() >= 3)
                .collect();
            out.push(Polygon2D {
                bbox: [min_x, min_y, max_x, max_y],
                exterior: ext,
                interiors: holes,
            });
        }
        Geometry::MultiPolygon(parts) => {
            for (exterior, interiors) in parts {
                let ext = ring_to_xy(exterior);
                if ext.len() < 3 {
                    continue;
                }
                let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
                let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &(x, y) in &ext {
                    if x.is_finite() && y.is_finite() {
                        min_x = min_x.min(x);
                        min_y = min_y.min(y);
                        max_x = max_x.max(x);
                        max_y = max_y.max(y);
                    }
                }
                if !min_x.is_finite() || !min_y.is_finite() {
                    continue;
                }
                let holes: Vec<Vec<(f64, f64)>> = interiors
                    .iter()
                    .map(ring_to_xy)
                    .filter(|r| r.len() >= 3)
                    .collect();
                out.push(Polygon2D {
                    bbox: [min_x, min_y, max_x, max_y],
                    exterior: ext,
                    interiors: holes,
                });
            }
        }
        Geometry::GeometryCollection(geoms) => {
            for g in geoms {
                collect_polygons_from_geometry(g, out);
            }
        }
        _ => {}
    }
}

fn load_polygons_and_bbox(
    polygon_vector_path: &Path,
    target_crs: &terrio::crs::SpatialRef,
) -> Result<(Vec<Polygon2D>, [f64; 4])> {
    let poly_str = path_to_str(polygon_vector_path)?;
    let vds = terrio::open_vector(poly_str, &VectorOpenOptions::default())?;
    let mut polygons = Vec::new();
    for layer in &vds.layers {
        let src_srs = terrio::crs::SpatialRef::from_layer(layer);
        let reprojected;
        let effective_layer =
            if !src_srs.is_unknown() && !target_crs.is_unknown() && !src_srs.same_as(target_crs) {
                reprojected = terrio::vector::reproject_layer(layer, target_crs).map_err(|e| {
                    Error::InvalidArgument(format!(
                        "failed to reproject polygon layer '{}' to point cloud CRS: {e}",
                        layer.name
                    ))
                })?;
                &reprojected
            } else {
                layer
            };
        for feat in &effective_layer.features {
            if let Some(ref geom) = feat.geometry {
                collect_polygons_from_geometry(geom, &mut polygons);
            }
        }
    }
    if polygons.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "no polygon geometries found in {}",
            polygon_vector_path.display()
        )));
    }
    let mut union = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in &polygons {
        union[0] = union[0].min(p.bbox[0]);
        union[1] = union[1].min(p.bbox[1]);
        union[2] = union[2].max(p.bbox[2]);
        union[3] = union[3].max(p.bbox[3]);
    }
    Ok((polygons, union))
}

fn point_on_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
    const EPS: f64 = 1e-12;
    let min_x = ax.min(bx) - EPS;
    let max_x = ax.max(bx) + EPS;
    let min_y = ay.min(by) - EPS;
    let max_y = ay.max(by) + EPS;
    if px < min_x || px > max_x || py < min_y || py > max_y {
        return false;
    }
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 <= EPS * EPS {
        return (px - ax).hypot(py - ay) <= EPS;
    }
    let cross = (px - ax) * dy - (py - ay) * dx;
    (cross * cross) <= EPS * EPS * len2
}

fn point_on_ring_boundary(px: f64, py: f64, ring: &[(f64, f64)]) -> bool {
    if ring.len() < 2 {
        return false;
    }
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (ax, ay) = ring[j];
        let (bx, by) = ring[i];
        if point_on_segment(px, py, ax, ay, bx, by) {
            return true;
        }
        j = i;
    }
    false
}

/// Ray-casting point-in-ring test (even-odd rule).
fn point_in_ring_raycast(px: f64, py: f64, ring: &[(f64, f64)]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Returns `true` if `(px, py)` lies inside `poly` (including exterior boundary, excluding interior holes).
fn point_in_polygon(px: f64, py: f64, poly: &Polygon2D) -> bool {
    if px < poly.bbox[0] || px > poly.bbox[2] || py < poly.bbox[1] || py > poly.bbox[3] {
        return false;
    }
    let in_exterior = point_on_ring_boundary(px, py, &poly.exterior)
        || point_in_ring_raycast(px, py, &poly.exterior);
    if !in_exterior {
        return false;
    }
    for hole in &poly.interiors {
        if point_in_ring_raycast(px, py, hole) && !point_on_ring_boundary(px, py, hole) {
            return false;
        }
    }
    true
}

/// Lightweight 2D uniform grid spatial index over polygon bounding boxes for fast
/// point-in-polygon queries when a clip layer contains many polygons (`polygons.len() > 8`).
struct PolygonGridIndex {
    bbox: [f64; 4],
    nx: usize,
    ny: usize,
    inv_dx: f64,
    inv_dy: f64,
    cells: Vec<Vec<usize>>,
}

impl PolygonGridIndex {
    fn new(polygons: &[Polygon2D], bbox: [f64; 4]) -> Self {
        let dim = (polygons.len() as f64).sqrt().ceil().clamp(2.0, 256.0) as usize;
        let span_x = bbox[2] - bbox[0];
        let span_y = bbox[3] - bbox[1];
        let (nx, inv_dx) = if span_x > 0.0 && span_x.is_finite() {
            (dim, dim as f64 / span_x)
        } else {
            (1, 0.0)
        };
        let (ny, inv_dy) = if span_y > 0.0 && span_y.is_finite() {
            (dim, dim as f64 / span_y)
        } else {
            (1, 0.0)
        };
        let mut cells = vec![Vec::new(); nx * ny];
        for (pi, poly) in polygons.iter().enumerate() {
            let cx0 = (((poly.bbox[0] - bbox[0]) * inv_dx).floor() as isize)
                .clamp(0, nx as isize - 1) as usize;
            let cx1 = (((poly.bbox[2] - bbox[0]) * inv_dx).floor() as isize)
                .clamp(0, nx as isize - 1) as usize;
            let cy0 = (((poly.bbox[1] - bbox[1]) * inv_dy).floor() as isize)
                .clamp(0, ny as isize - 1) as usize;
            let cy1 = (((poly.bbox[3] - bbox[1]) * inv_dy).floor() as isize)
                .clamp(0, ny as isize - 1) as usize;
            for cy in cy0..=cy1 {
                let row = cy * nx;
                for cx in cx0..=cx1 {
                    cells[row + cx].push(pi);
                }
            }
        }
        Self {
            bbox,
            nx,
            ny,
            inv_dx,
            inv_dy,
            cells,
        }
    }

    #[inline]
    fn contains(&self, px: f64, py: f64, polygons: &[Polygon2D]) -> bool {
        if px < self.bbox[0] || px > self.bbox[2] || py < self.bbox[1] || py > self.bbox[3] {
            return false;
        }
        let cx = (((px - self.bbox[0]) * self.inv_dx).floor() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let cy = (((py - self.bbox[1]) * self.inv_dy).floor() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        self.cells[cy * self.nx + cx]
            .iter()
            .any(|&pi| point_in_polygon(px, py, &polygons[pi]))
    }
}

fn filter_cloud_by_polygons(
    cloud: &PointCloud,
    polygons: &[Polygon2D],
    union_bbox: [f64; 4],
) -> Result<PointCloud> {
    if cloud.is_empty() {
        return Ok(cloud.clone());
    }
    let (xs, ys) = extract_xy(cloud)?;
    let keep: Vec<bool> = if polygons.len() > 8 {
        let index = PolygonGridIndex::new(polygons, union_bbox);
        (0..cloud.len())
            .map(|i| {
                let x = xs[i];
                let y = ys[i];
                x.is_finite() && y.is_finite() && index.contains(x, y, polygons)
            })
            .collect()
    } else {
        (0..cloud.len())
            .map(|i| {
                let x = xs[i];
                let y = ys[i];
                x.is_finite()
                    && y.is_finite()
                    && polygons.iter().any(|poly| point_in_polygon(x, y, poly))
            })
            .collect()
    };
    Ok(cloud.select(&keep))
}

/// Uniformly thins `cloud` to at most `max_points` points.
fn cap_point_cloud(cloud: &PointCloud, max_points: usize) -> PointCloud {
    let n = cloud.len();
    if n <= max_points {
        return cloud.clone();
    }
    let indices: Vec<usize> = (0..max_points)
        .map(|k| ((k as u128 * n as u128) / max_points as u128) as usize)
        .collect();
    cloud.take(&indices)
}

/// Reads a LiDAR point cloud with COPC octree LOD (`max_level`, `resolution`), spatial
/// bounding-box windowing (`bounds`), optional polygon clipping (`polygon_path`), approximate
/// point cap (`max_points`), and attribute filters (`classifications`, `return_kinds`,
/// `min_z`, `max_z`).
///
/// When `opts.polygon_path` is set, polygon layers whose CRS differs from the point cloud CRS
/// are automatically reprojected into the point cloud's CRS. The polygon's 2D bounding box is
/// intersected with `opts.bounds` and forwarded into `PointFilter::bounds` so COPC octree
/// readers only fetch chunks overlapping the polygon extent before exact ray-casting
/// point-in-polygon filtering. In this mode, all points inside the polygon bounding box are read
/// first (without reader-level `max_points` thinning) and `opts.max_points` caps the post-clip
/// output point count so small polygons inside large tiles are not starved.
pub fn read_lidar_windowed(path: impl AsRef<Path>, opts: &LidarReadOptions) -> Result<PointCloud> {
    if let Some([x0, y0, x1, y1]) = opts.bounds {
        if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
            return Err(Error::InvalidArgument(
                "bounds coordinates must be finite".into(),
            ));
        }
        if x0 > x1 || y0 > y1 {
            return Err(Error::InvalidArgument(format!(
                "invalid bounds [{x0}, {y0}, {x1}, {y1}]: expected min <= max"
            )));
        }
    }
    if let Some(res) = opts.resolution
        && (!res.is_finite() || res <= 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "resolution must be finite and > 0, got {res}"
        )));
    }
    if let Some(lo) = opts.min_z
        && !lo.is_finite()
    {
        return Err(Error::InvalidArgument(format!(
            "min_z must be finite, got {lo}"
        )));
    }
    if let Some(hi) = opts.max_z
        && !hi.is_finite()
    {
        return Err(Error::InvalidArgument(format!(
            "max_z must be finite, got {hi}"
        )));
    }
    if let (Some(lo), Some(hi)) = (opts.min_z, opts.max_z)
        && lo > hi
    {
        return Err(Error::InvalidArgument(format!(
            "min_z ({lo}) must be <= max_z ({hi})"
        )));
    }
    if opts.max_points == Some(0) {
        return Err(Error::InvalidArgument("max_points must be > 0".into()));
    }

    let p = path_to_str(path.as_ref())?;
    let mut reader = terrio::pointcloud::open_points(p)?;
    let cloud_crs = reader.crs().clone();

    let loaded_polygons = match &opts.polygon_path {
        Some(poly_path) => Some(load_polygons_and_bbox(poly_path, &cloud_crs)?),
        None => None,
    };

    let effective_bounds = match (opts.bounds, &loaded_polygons) {
        (None, None) => None,
        (Some(b), None) => Some(b),
        (None, Some((_, pb))) => Some(*pb),
        (Some(b), Some((_, pb))) => {
            let inter = [
                b[0].max(pb[0]),
                b[1].max(pb[1]),
                b[2].min(pb[2]),
                b[3].min(pb[3]),
            ];
            if inter[0] > inter[2] || inter[1] > inter[3] {
                return Ok(PointCloud::new(Vec::new(), cloud_crs));
            }
            Some(inter)
        }
    };

    let filter = PointFilter {
        bounds: effective_bounds,
        z_range: match (opts.min_z, opts.max_z) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            (Some(lo), None) => Some((lo, f64::INFINITY)),
            (None, Some(hi)) => Some((f64::NEG_INFINITY, hi)),
            (None, None) => None,
        },
        classes: opts.classifications.clone(),
        returns: opts.return_kinds.clone(),
    };
    // Defer `max_points` thinning until after exact polygon clipping so small polygons are not
    // starved when the reader thins across the larger bounding box.
    let reader_max_points = if loaded_polygons.is_some() {
        None
    } else {
        opts.max_points.map(|n| n as u64)
    };
    let read_opts = PointReadOptions {
        filter,
        max_level: opts.max_level,
        resolution: opts.resolution,
        max_points: reader_max_points,
    };
    let cloud = reader.read(&read_opts)?;
    if let Some((polygons, union_bbox)) = loaded_polygons {
        let clipped = filter_cloud_by_polygons(&cloud, &polygons, union_bbox)?;
        if let Some(cap) = opts.max_points {
            Ok(cap_point_cloud(&clipped, cap))
        } else {
            Ok(clipped)
        }
    } else {
        Ok(cloud)
    }
}

/// Reads points from a LiDAR archive (LAS, LAZ, COPC, PLY, E57, EPT) that fall inside any
/// polygon geometry in `polygon_vector_path` (GeoJSON, Shapefile, GeoPackage, etc.).
///
/// Computes the exact 2D bounding box of the polygon(s) and passes it into `PointFilter::bounds`
/// (intersected with `opts.bounds` when present) so COPC octree streaming only fetches chunks
/// intersecting the polygon's extent, then filters retained points with exact ray-casting
/// point-in-polygon respecting exterior rings and interior holes across `Polygon`,
/// `MultiPolygon`, and `GeometryCollection`.
pub fn read_lidar_in_polygon(
    path: impl AsRef<Path>,
    polygon_vector_path: impl AsRef<Path>,
    opts: &LidarReadOptions,
) -> Result<PointCloud> {
    let mut merged = opts.clone();
    merged.polygon_path = Some(polygon_vector_path.as_ref().to_path_buf());
    read_lidar_windowed(path, &merged)
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

    // `terrio::pointcloud::convert` returns `Result<()>` without a point count. Since
    // `convert_lidar` applies no filtering or thinning, obtain the exact count from the
    // source header up front (`header().point_count` is exact for all terrio readers:
    // LAS, LAZ, COPC, PLY, E57, and EPT) without loading points into RAM or reopening the output.
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

/// Classifies ground points in a [`PointCloud`] using tunable Cloth Simulation Filter (CSF),
/// Progressive Morphological Filter (PMF), or Simple Morphological Filter (SMRF) parameters.
pub fn classify_ground_with_options(
    cloud: &PointCloud,
    options: &GroundClassifyOptions,
) -> Result<PointCloud> {
    match options.method {
        GroundFilterMethod::Preclassified => Ok(cloud.clone()),
        GroundFilterMethod::Csf => Ok(terrio_csf(cloud, &options.csf)?),
        GroundFilterMethod::Pmf => Ok(terrio_pmf(cloud, &options.pmf)?),
        GroundFilterMethod::Smrf => Ok(terrio_smrf(cloud, &options.smrf)?),
    }
}

/// Classifies ground points in a [`PointCloud`] using Cloth Simulation Filter (CSF),
/// Progressive Morphological Filter (PMF), or Simple Morphological Filter (SMRF) with defaults.
pub fn classify_ground(cloud: &PointCloud, method: GroundFilterMethod) -> Result<PointCloud> {
    classify_ground_with_options(
        cloud,
        &GroundClassifyOptions {
            method,
            ..Default::default()
        },
    )
}

fn extract_xy(cloud: &PointCloud) -> Result<(Vec<f64>, Vec<f64>)> {
    let xs = match cloud.coords("x") {
        Some(slice) => slice.to_vec(),
        None => cloud
            .dim("x")
            .or_else(|| cloud.dim("X"))
            .ok_or_else(|| Error::InvalidArgument("point cloud has no x dimension".into()))?
            .to_f64(),
    };
    let ys = match cloud.coords("y") {
        Some(slice) => slice.to_vec(),
        None => cloud
            .dim("y")
            .or_else(|| cloud.dim("Y"))
            .ok_or_else(|| Error::InvalidArgument("point cloud has no y dimension".into()))?
            .to_f64(),
    };
    Ok((xs, ys))
}

fn extract_xyz(cloud: &PointCloud) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (xs, ys) = extract_xy(cloud)?;
    let zs = match cloud.coords("z") {
        Some(slice) => slice.to_vec(),
        None => cloud
            .dim("z")
            .or_else(|| cloud.dim("Z"))
            .ok_or_else(|| Error::InvalidArgument("point cloud has no z dimension".into()))?
            .to_f64(),
    };
    Ok((xs, ys, zs))
}

/// Removes isolated noise points using Radius Outlier Removal (ROR).
///
/// Every point with fewer than `min_neighbors` other points within 3D Euclidean distance
/// `radius` is stripped from the returned [`PointCloud`]. Uses a 3D spatial hash grid for
/// expected `O(N)` neighborhood counting with early exit once `min_neighbors` is reached.
pub fn filter_outliers_ror(
    cloud: &PointCloud,
    radius: f64,
    min_neighbors: usize,
) -> Result<PointCloud> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "radius must be finite and > 0 for ROR filtering, got {radius}"
        )));
    }
    if min_neighbors == 0 {
        return Err(Error::InvalidArgument(
            "min_neighbors must be >= 1 for ROR filtering".into(),
        ));
    }
    let n = cloud.len();
    if n == 0 {
        return Ok(cloud.clone());
    }
    let (xs, ys, zs) = extract_xyz(cloud)?;

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for i in 0..n {
        let (x, y, z) = (xs[i], ys[i], zs[i]);
        if x.is_finite() && y.is_finite() && z.is_finite() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            max_z = max_z.max(z);
        }
    }
    if !min_x.is_finite() {
        return Ok(cloud.select(&vec![false; n]));
    }

    let r2 = radius * radius;
    let inv_r = 1.0 / radius;
    let max_span_cells = ((max_x - min_x).max(max_y - min_y).max(max_z - min_z)) * inv_r;

    // If the shifted spatial grid fits safely in i64 cell indices, run O(N) 3D spatial hashing.
    if max_span_cells.is_finite() && max_span_cells < (i64::MAX / 4) as f64 {
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::with_capacity(n);
        let mut cell_of: Vec<Option<(i64, i64, i64)>> = Vec::with_capacity(n);
        for i in 0..n {
            let (x, y, z) = (xs[i], ys[i], zs[i]);
            if x.is_finite() && y.is_finite() && z.is_finite() {
                let key = (
                    ((x - min_x) * inv_r).floor() as i64,
                    ((y - min_y) * inv_r).floor() as i64,
                    ((z - min_z) * inv_r).floor() as i64,
                );
                grid.entry(key).or_default().push(i);
                cell_of.push(Some(key));
            } else {
                cell_of.push(None);
            }
        }

        let mut keep = vec![false; n];
        for i in 0..n {
            let Some((cx, cy, cz)) = cell_of[i] else {
                continue;
            };
            let (xi, yi, zi) = (xs[i], ys[i], zs[i]);
            let mut count = 0usize;
            'neighbor_search: for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if j == i {
                                    continue;
                                }
                                let ddx = xs[j] - xi;
                                let ddy = ys[j] - yi;
                                let ddz = zs[j] - zi;
                                if ddx * ddx + ddy * ddy + ddz * ddz <= r2 {
                                    count += 1;
                                    if count >= min_neighbors {
                                        keep[i] = true;
                                        break 'neighbor_search;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return Ok(cloud.select(&keep));
    }

    // Fallback for extreme coordinate spans that exceed i64 grid indices: delegate to terrio's KD-tree.
    let opts = OutlierOptions {
        method: OutlierMethod::Radius,
        radius,
        min_k: min_neighbors,
        class: 7,
        ..Default::default()
    };
    let mask = outlier_mask(cloud, &opts)?;
    let keep: Vec<bool> = mask
        .into_iter()
        .enumerate()
        .map(|(i, is_out)| !is_out && xs[i].is_finite() && ys[i].is_finite() && zs[i].is_finite())
        .collect();
    Ok(cloud.select(&keep))
}

/// Normalizes a capitalized `"Classification"` dimension to `"classification"` when
/// lowercase `"classification"` is absent so `terrio` filters preserve existing classes.
fn normalize_classification_dim(cloud: &PointCloud) -> std::borrow::Cow<'_, PointCloud> {
    if cloud.dim("classification").is_none() && cloud.dim("Classification").is_some() {
        let mut cloned = cloud.clone();
        if let Some(dim) = cloned.dim_mut("Classification") {
            dim.name = "classification".to_string();
        }
        std::borrow::Cow::Owned(cloned)
    } else {
        std::borrow::Cow::Borrowed(cloud)
    }
}

/// Removes points classified as ASPRS low noise (`7`) or high noise (`18`).
fn strip_noise_classes(cloud: &PointCloud) -> PointCloud {
    let Some(dim) = cloud
        .dim("classification")
        .or_else(|| cloud.dim("Classification"))
    else {
        return cloud.clone();
    };
    let n = cloud.len();
    let keep: Vec<bool> = (0..n)
        .map(|i| {
            let c = dim.values.get_f64(i);
            !(c == 7.0 || c == 18.0)
        })
        .collect();
    cloud.select(&keep)
}

/// Identifies low-noise outliers using the Extended Local Minimum (ELM) filter
/// (`terrio::pointcloud::filters::elm`, matching PDAL's `filters.elm`).
///
/// Points flagged as low noise are assigned ASPRS class `7`. When `remove_noise` is `true`,
/// all ASPRS low-noise (`7`) and high-noise (`18`) points are stripped from the returned cloud.
pub fn filter_outliers_elm(
    cloud: &PointCloud,
    cell: f64,
    threshold: f64,
    remove_noise: bool,
) -> Result<PointCloud> {
    if !cell.is_finite() || cell <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "cell must be finite and > 0, got {cell}"
        )));
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "threshold must be finite and >= 0, got {threshold}"
        )));
    }
    let normalized = normalize_classification_dim(cloud);
    let classified = terrio_elm(&normalized, cell, threshold, 7)?;
    if remove_noise {
        Ok(strip_noise_classes(&classified))
    } else {
        Ok(classified)
    }
}

/// Identifies statistical outliers using k-nearest-neighbor mean-distance thresholding
/// (`terrio::pointcloud::filters::outlier`, matching PDAL's `filters.outlier` in statistical mode).
///
/// Outliers are assigned ASPRS class `7`. When `remove_noise` is `true`, all ASPRS low-noise
/// (`7`) and high-noise (`18`) points are stripped from the returned cloud.
pub fn filter_outliers_statistical(
    cloud: &PointCloud,
    mean_k: usize,
    multiplier: f64,
    remove_noise: bool,
) -> Result<PointCloud> {
    if mean_k == 0 {
        return Err(Error::InvalidArgument(
            "mean_k must be >= 1 for statistical outlier filtering".into(),
        ));
    }
    if !multiplier.is_finite() || multiplier < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "multiplier must be finite and >= 0, got {multiplier}"
        )));
    }
    let opts = OutlierOptions {
        method: OutlierMethod::Statistical,
        mean_k,
        multiplier,
        class: 7,
        ..Default::default()
    };
    let normalized = normalize_classification_dim(cloud);
    let classified = terrio_outlier(&normalized, &opts)?;
    if remove_noise {
        Ok(strip_noise_classes(&classified))
    } else {
        Ok(classified)
    }
}

/// Computes Height Above Ground (HAG) for every point in `cloud`, storing the result in the
/// `height_above_ground` dimension using either Delaunay triangulation (`use_delaunay = true`,
/// PDAL `filters.hag_delaunay`) or nearest-neighbor interpolation (`use_delaunay = false`,
/// PDAL `filters.hag_nn`).
///
/// Requires a `classification` dimension with ASPRS ground points (class `2`).
pub fn normalize_hag(cloud: &PointCloud, use_delaunay: bool) -> Result<PointCloud> {
    if use_delaunay {
        Ok(hag_delaunay(cloud, &HagOptions::delaunay())?)
    } else {
        Ok(hag_nn(cloud, &HagOptions::nn())?)
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

fn raster_write_options(compress: bool, cog: bool) -> RasterWriteOptions {
    if cog {
        RasterWriteOptions {
            driver: Some(RasterDriver::Cog),
            compression: Some("DEFLATE".to_string()),
            ..Default::default()
        }
    } else if compress {
        RasterWriteOptions {
            compression: Some("DEFLATE".to_string()),
            ..Default::default()
        }
    } else {
        RasterWriteOptions::default()
    }
}

fn write_array2d_raster(
    arr: &Array2D<f64>,
    out_path: &Path,
    compress: bool,
    cog: bool,
) -> Result<()> {
    let out_str = path_to_str(out_path)?;
    let ds = array2d_to_dataset(arr)?;
    terrio::write_raster_dataset(&ds, out_str, &raster_write_options(compress, cog))?;
    Ok(())
}

fn validate_dsm_options(options: &LidarToDsmOptions) -> Result<()> {
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
    Ok(())
}

fn validate_dtm_options(options: &LidarToDtmOptions) -> Result<()> {
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
    if let Some(r) = options.ror_radius {
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "ror_radius must be finite and > 0, got {r}"
            )));
        }
        if options.ror_min_neighbors == 0 {
            return Err(Error::InvalidArgument(
                "ror_min_neighbors must be >= 1 when ror_radius is set".into(),
            ));
        }
    }
    if let Some(k) = options.sor_k {
        if k == 0 {
            return Err(Error::InvalidArgument("sor_k must be >= 1 when set".into()));
        }
        if !options.sor_sigma.is_finite() || options.sor_sigma < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "sor_sigma must be finite and >= 0, got {}",
                options.sor_sigma
            )));
        }
    }
    Ok(())
}

fn dsm_point_filter(options: &LidarToDsmOptions) -> PointFilter {
    PointFilter {
        bounds: None,
        z_range: match (options.min_z, options.max_z) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            (Some(lo), None) => Some((lo, f64::INFINITY)),
            (None, Some(hi)) => Some((f64::NEG_INFINITY, hi)),
            (None, None) => None,
        },
        classes: options.classifications.clone().unwrap_or_default(),
        returns: options.return_kinds.clone().unwrap_or_default(),
    }
}

/// Grids an in-memory [`PointCloud`] into a Digital Surface Model (DSM) [`Array2D<f64>`].
///
/// Applies optional ASPRS classification, return-kind, and elevation filtering configured in
/// `options` before gridding via pure-Rust `terrio::pointcloud::grid`.
pub fn points_to_dsm(cloud: &PointCloud, options: &LidarToDsmOptions) -> Result<Array2D<f64>> {
    validate_dsm_options(options)?;
    let filter = dsm_point_filter(options);
    let filtered = if filter.is_empty() {
        cloud.clone()
    } else {
        let normalized = normalize_classification_dim(cloud);
        normalized.filter(&filter)
    };
    if filtered.is_empty() {
        return Err(Error::Algorithm(
            "no points remain after classification / return / elevation filtering".into(),
        ));
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
    let dsm_ds = terrio::pointcloud::grid::grid(&filtered, &grid_opts)?;
    Ok(richdem::dataset_to_dynamic(dsm_ds, "dsm")?.to_f64())
}

/// Generates a Digital Surface Model (DSM) GeoTIFF (or COG) from a LiDAR point cloud.
///
/// Supports optional ASPRS classification filtering, pulse return-kind filtering,
/// elevation bounds, DEFLATE compression (`options.compress`), and Cloud Optimized GeoTIFF
/// output (`options.cog`) via pure-Rust `terrio::pointcloud::grid`.
pub fn lidar_to_dsm(
    in_lidar: impl AsRef<Path>,
    out_dsm: impl AsRef<Path>,
    options: &LidarToDsmOptions,
) -> Result<(usize, usize)> {
    validate_dsm_options(options)?;
    let out_path = out_dsm.as_ref();
    crate::raster_ops::ensure_no_path_collision(in_lidar.as_ref(), &[out_path])?;
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let filter = dsm_point_filter(options);

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
    terrio::write_raster_dataset(
        &dsm,
        out_str,
        &raster_write_options(options.compress, options.cog),
    )?;
    Ok(dims)
}

fn has_dtm_outlier_filters(options: &LidarToDtmOptions) -> bool {
    options.remove_outliers || options.ror_radius.is_some() || options.sor_k.is_some()
}

fn apply_dtm_outlier_filters(raw: &PointCloud, options: &LidarToDtmOptions) -> Result<PointCloud> {
    let mut cloud = raw.clone();
    if options.remove_outliers {
        cloud = filter_outliers_elm(
            &cloud,
            options.outlier_cell,
            options.outlier_threshold,
            true,
        )?;
    }
    if let Some(r) = options.ror_radius {
        cloud = filter_outliers_ror(&cloud, r, options.ror_min_neighbors)?;
    }
    if let Some(k) = options.sor_k {
        cloud = filter_outliers_statistical(&cloud, k, options.sor_sigma, true)?;
    }
    Ok(cloud)
}

fn select_ground_points(raw: &PointCloud, options: &LidarToDtmOptions) -> Result<PointCloud> {
    match options.ground_method {
        GroundFilterMethod::Preclassified => filter_by_classification(raw, &[options.ground_class]),
        GroundFilterMethod::Csf | GroundFilterMethod::Pmf | GroundFilterMethod::Smrf => {
            let expected_class = match options.ground_method {
                GroundFilterMethod::Csf => options.csf.ground_class,
                GroundFilterMethod::Pmf => options.pmf.ground_class,
                GroundFilterMethod::Smrf => options.smrf.ground_class,
                GroundFilterMethod::Preclassified => unreachable!(),
            };
            if options.ground_class != expected_class {
                return Err(Error::InvalidArgument(format!(
                    "{:?} assigns ground class {expected_class}; set ground_class={expected_class} or use GroundFilterMethod::Preclassified (got ground_class={})",
                    options.ground_method, options.ground_class
                )));
            }
            let classes = match options.ground_method {
                GroundFilterMethod::Csf => csf_classes(raw, &options.csf)?,
                GroundFilterMethod::Pmf => pmf_classes(raw, &options.pmf)?,
                GroundFilterMethod::Smrf => smrf_classes(raw, &options.smrf)?,
                GroundFilterMethod::Preclassified => unreachable!(),
            };
            let keep: Vec<bool> = classes.iter().map(|&c| c == Some(expected_class)).collect();
            Ok(raw.select(&keep))
        }
    }
}

/// Grids an in-memory [`PointCloud`] into a bare-earth Digital Terrain Model (DTM)
/// [`Array2D<f64>`], applying configured outlier filters and ground classification.
pub fn points_to_dtm(cloud: &PointCloud, options: &LidarToDtmOptions) -> Result<Array2D<f64>> {
    validate_dtm_options(options)?;
    let cleaned = if has_dtm_outlier_filters(options) {
        apply_dtm_outlier_filters(cloud, options)?
    } else {
        cloud.clone()
    };
    let ground = select_ground_points(&cleaned, options)?;
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
    let dtm_ds = terrio::pointcloud::grid::grid(&ground, &grid_opts)?;
    Ok(richdem::dataset_to_dynamic(dtm_ds, "dtm")?.to_f64())
}

/// Generates a bare-earth Digital Terrain Model (DTM) GeoTIFF (or COG) from a LiDAR point cloud.
///
/// Uses existing ASPRS ground returns (`Preclassified`) or runs pure-Rust ground
/// classification (`Csf`, `Pmf`, or `Smrf`) with tunable parameters, selects ground points,
/// and interpolates a continuous bare-earth surface.
pub fn lidar_to_dtm(
    in_lidar: impl AsRef<Path>,
    out_dtm: impl AsRef<Path>,
    options: &LidarToDtmOptions,
) -> Result<(usize, usize)> {
    validate_dtm_options(options)?;
    let out_path = out_dtm.as_ref();
    crate::raster_ops::ensure_no_path_collision(in_lidar.as_ref(), &[out_path])?;
    ensure_parent_dir(out_path)?;
    let out_str = path_to_str(out_path)?;

    let ground = if has_dtm_outlier_filters(options) {
        let raw = read_lidar(in_lidar)?;
        let cleaned = apply_dtm_outlier_filters(&raw, options)?;
        select_ground_points(&cleaned, options)?
    } else {
        match options.ground_method {
            GroundFilterMethod::Preclassified => {
                let filter = PointFilter {
                    classes: vec![options.ground_class],
                    ..Default::default()
                };
                read_lidar_filtered(in_lidar, filter)?
            }
            _ => {
                let raw = read_lidar(in_lidar)?;
                select_ground_points(&raw, options)?
            }
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
    terrio::write_raster_dataset(
        &dtm,
        out_str,
        &raster_write_options(options.compress, options.cog),
    )?;
    Ok(dims)
}

/// Computes an in-memory Canopy Height Model (`CHM = max(0, DSM - DTM)`) [`Array2D<f64>`]
/// directly from a [`PointCloud`].
pub fn points_to_chm(cloud: &PointCloud, options: &LidarToDtmOptions) -> Result<Array2D<f64>> {
    validate_dtm_options(options)?;
    let raw = if has_dtm_outlier_filters(options) {
        apply_dtm_outlier_filters(cloud, options)?
    } else {
        cloud.clone()
    };
    if raw.is_empty() {
        return Err(Error::Algorithm("input point cloud is empty".into()));
    }
    let ground = select_ground_points(&raw, options)?;
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

    let mut chm = Array2D::from_vec(w, h, chm_data)?;
    chm.copy_metadata_from(&dsm_arr);
    chm.set_no_data(nd);
    Ok(chm)
}

/// Generates a Canopy Height Model (`CHM = max(0, DSM - DTM)`) GeoTIFF (or COG) directly from a
/// LiDAR point cloud in pure Rust.
pub fn lidar_to_chm(
    in_lidar: impl AsRef<Path>,
    out_chm: impl AsRef<Path>,
    options: &LidarToDtmOptions,
) -> Result<(usize, usize)> {
    validate_dtm_options(options)?;
    let out_path = out_chm.as_ref();
    crate::raster_ops::ensure_no_path_collision(in_lidar.as_ref(), &[out_path])?;
    ensure_parent_dir(out_path)?;

    let raw = read_lidar(in_lidar)?;
    let chm = points_to_chm(&raw, options)?;
    let dims = (chm.width(), chm.height());
    write_array2d_raster(&chm, out_path, options.compress, options.cog)?;
    Ok(dims)
}

// ---------------------------------------------------------------------------
// Pit-free CHM (Khosravipour et al. 2014) and Morton-ordered 2D Delaunay TIN
// ---------------------------------------------------------------------------

#[inline]
fn spread_bits_16(v: u32) -> u32 {
    let mut x = v & 0xFFFF;
    x = (x | (x << 8)) & 0x00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333;
    x = (x | (x << 1)) & 0x5555_5555;
    x
}

#[inline]
fn morton_code_2d(nx: f64, ny: f64) -> u32 {
    let qx = ((nx + 0.5).clamp(0.0, 1.0) * 65535.0) as u32;
    let qy = ((ny + 0.5).clamp(0.0, 1.0) * 65535.0) as u32;
    spread_bits_16(qx) | (spread_bits_16(qy) << 1)
}

#[inline]
fn orient2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Deterministic splitmix64 perturbation in `[-1.0, 1.0]` for symbolic tie-breaking of
/// collinear and co-circular point configurations during Bowyer-Watson triangulation.
#[inline]
fn deterministic_jitter(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    unit * 2.0 - 1.0
}

/// Returns `true` if point `p` lies strictly inside the circumcircle of CCW triangle
/// `(a, b, c)` in normalized coordinates.
#[inline]
fn in_circumcircle_strict(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> bool {
    let adx = a.0 - p.0;
    let ady = a.1 - p.1;
    let bdx = b.0 - p.0;
    let bdy = b.1 - p.1;
    let cdx = c.0 - p.0;
    let cdy = c.1 - p.1;

    let ad2 = adx * adx + ady * ady;
    let bd2 = bdx * bdx + bdy * bdy;
    let cd2 = cdx * cdx + cdy * cdy;

    let det = adx * (bdy * cd2 - bd2 * cdy) - ady * (bdx * cd2 - bd2 * cdx)
        + ad2 * (bdx * cdy - bdy * cdx);
    det > 0.0
}

#[derive(Clone, Copy)]
struct DelaunayTri {
    v: [usize; 3],
    /// Neighbor triangle opposite vertex `v[k]`, i.e., sharing edge `(v[(k+1)%3], v[(k+2)%3])`.
    adj: [Option<usize>; 3],
    alive: bool,
}

/// Deduplicates coincident 2D points, retaining the maximum height `h` at each `(x, y)` location.
fn deduplicate_max_height(pts: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    let mut best: HashMap<(u64, u64), (f64, f64, f64)> = HashMap::with_capacity(pts.len());
    for &(x, y, h) in pts {
        if !x.is_finite() || !y.is_finite() || !h.is_finite() {
            continue;
        }
        // Quantize to 1e-6 CRS units to merge multi-return pulses along the same vertical beam.
        let qx = (x * 1e6).round() as i64 as u64;
        let qy = (y * 1e6).round() as i64 as u64;
        best.entry((qx, qy))
            .and_modify(|e| {
                if h > e.2 {
                    *e = (x, y, h);
                }
            })
            .or_insert((x, y, h));
    }
    let mut out: Vec<(f64, f64, f64)> = best.into_values().collect();
    // Deterministic ordering independent of HashMap iteration order.
    out.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.total_cmp(&b.2))
    });
    out
}

/// Computes a 2D Delaunay triangulation of `pts` using Morton-ordered Bowyer-Watson with
/// coordinate normalization and triangle-adjacency walking.
///
/// Returns triangle vertex index triples `[i, j, k]` referencing `pts`.
fn delaunay_triangulate_2d(pts: &[(f64, f64, f64)]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }

    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y, _) in pts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let span = (max_x - min_x).max(max_y - min_y);
    if !span.is_finite() || span <= 1e-12 {
        return Vec::new();
    }
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    let inv_span = 1.0 / span;

    // Normalized coordinates in [-0.5, 0.5] with deterministic sub-nanometer symbolic jitter
    // (~1e-11) used exclusively for triangulation predicates so regular scan-line / integer
    // grid points are never dropped as collinear or co-circular. Original unperturbed coordinates
    // in `pts` are still used for triangle edge-length checks and barycentric z-interpolation.
    const JITTER_SCALE: f64 = 1e-11;
    let mut coords: Vec<(f64, f64)> = Vec::with_capacity(n + 3);
    for (i, &(x, y, _)) in pts.iter().enumerate() {
        let jx = deterministic_jitter((i as u64).wrapping_mul(2).wrapping_add(1)) * JITTER_SCALE;
        let jy = deterministic_jitter((i as u64).wrapping_mul(2).wrapping_add(2)) * JITTER_SCALE;
        coords.push(((x - cx) * inv_span + jx, (y - cy) * inv_span + jy));
    }
    coords.push((-10.0, -10.0));
    coords.push((10.0, -10.0));
    coords.push((0.0, 20.0));

    // Sort insertion order by 2D Morton Z-order for O(1) amortized visibility walk.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| {
        let (nx, ny) = coords[i];
        morton_code_2d(nx, ny)
    });

    let mut tris: Vec<DelaunayTri> = Vec::with_capacity(2 * n + 4);
    let mut free_slots: Vec<usize> = Vec::new();
    tris.push(DelaunayTri {
        v: [n, n + 1, n + 2],
        adj: [None, None, None],
        alive: true,
    });
    let mut last_tri: usize = 0;

    let mut bad_set: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    // Each boundary edge stores `(a, b, outer_tri, outer_adj_slot)`.
    let mut boundary: Vec<(usize, usize, Option<usize>, Option<usize>)> = Vec::new();
    let mut new_indices: Vec<usize> = Vec::new();

    for &p in &order {
        let base_nx = (pts[p].0 - cx) * inv_span;
        let base_ny = (pts[p].1 - cy) * inv_span;

        // Retry up to 4 times with alternative deterministic jitter offsets if a rare numerical
        // degeneracy causes the enclosing triangle search or cavity star-shape check to fail.
        // Duplicate or coincident XY points (zero-area degeneracy) that cannot form positive-area
        // triangles after all retries are skipped.
        const MAX_JITTER_RETRIES: u64 = 4;
        for attempt in 0..=MAX_JITTER_RETRIES {
            if attempt > 0 {
                let seed_base = (p as u64)
                    .wrapping_mul(17)
                    .wrapping_add(attempt.wrapping_mul(0x9E37_79B9));
                let scale = JITTER_SCALE * (1.0 + attempt as f64);
                let jx = deterministic_jitter(seed_base.wrapping_add(1)) * scale;
                let jy = deterministic_jitter(seed_base.wrapping_add(2)) * scale;
                coords[p] = (base_nx + jx, base_ny + jy);
            }
            let (px, py) = coords[p];

            // 1. Locate a triangle containing `p` via visibility walk from `last_tri`.
            let mut cur = last_tri;
            if !tris[cur].alive {
                cur = tris.iter().rposition(|t| t.alive).unwrap_or(0);
            }
            let mut found_walk = false;
            let max_steps = tris.len().min(4096) + 16;
            for _ in 0..max_steps {
                let tv = tris[cur].v;
                let mut stepped = false;
                for k in 0..3 {
                    let a = tv[(k + 1) % 3];
                    let b = tv[(k + 2) % 3];
                    let (ax, ay) = coords[a];
                    let (bx, by) = coords[b];
                    if orient2d(ax, ay, bx, by, px, py) < 0.0
                        && let Some(next) = tris[cur].adj[k]
                        && tris[next].alive
                    {
                        cur = next;
                        stepped = true;
                        break;
                    }
                }
                if !stepped {
                    found_walk = true;
                    break;
                }
            }

            // Fallback linear scan if walk hit a numerical cycle.
            let start_tri = if found_walk {
                cur
            } else {
                let mut candidate = None;
                for (ti, t) in tris.iter().enumerate() {
                    if !t.alive {
                        continue;
                    }
                    let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
                    let (ax, ay) = coords[a];
                    let (bx, by) = coords[b];
                    let (cx0, cy0) = coords[c];
                    if orient2d(ax, ay, bx, by, px, py) >= 0.0
                        && orient2d(bx, by, cx0, cy0, px, py) >= 0.0
                        && orient2d(cx0, cy0, ax, ay, px, py) >= 0.0
                    {
                        candidate = Some(ti);
                        break;
                    }
                }
                match candidate {
                    Some(ti) => ti,
                    None => continue,
                }
            };

            // 2. BFS from `start_tri` to collect all triangles whose circumcircle strictly contains `p`.
            bad_set.clear();
            queue.clear();
            bad_set.insert(start_tri);
            queue.push_back(start_tri);
            while let Some(ti) = queue.pop_front() {
                for nb in tris[ti].adj.into_iter().flatten() {
                    if !tris[nb].alive || bad_set.contains(&nb) {
                        continue;
                    }
                    let tv = tris[nb].v;
                    if in_circumcircle_strict(coords[tv[0]], coords[tv[1]], coords[tv[2]], (px, py))
                    {
                        bad_set.insert(nb);
                        queue.push_back(nb);
                    }
                }
            }

            // 3. Ensure cavity is star-shaped with respect to `p` (handles points lying on an edge).
            let mut expanded = true;
            while expanded {
                expanded = false;
                let mut to_add = Vec::new();
                for &ti in &bad_set {
                    let tv = tris[ti].v;
                    for k in 0..3 {
                        let nb = tris[ti].adj[k];
                        if nb.is_some_and(|n_idx| bad_set.contains(&n_idx)) {
                            continue;
                        }
                        let a = tv[(k + 1) % 3];
                        let b = tv[(k + 2) % 3];
                        let (ax, ay) = coords[a];
                        let (bx, by) = coords[b];
                        if orient2d(ax, ay, bx, by, px, py) <= 0.0
                            && let Some(outer) = nb
                            && tris[outer].alive
                        {
                            to_add.push(outer);
                        }
                    }
                }
                for outer in to_add {
                    if bad_set.insert(outer) {
                        expanded = true;
                    }
                }
            }

            // 4. Extract boundary edges `(a, b, outer_neighbor, outer_adj_slot)`.
            boundary.clear();
            let mut all_positive = true;
            for &ti in &bad_set {
                let tv = tris[ti].v;
                for k in 0..3 {
                    let nb = tris[ti].adj[k];
                    if nb.is_some_and(|n_idx| bad_set.contains(&n_idx)) {
                        continue;
                    }
                    let a = tv[(k + 1) % 3];
                    let b = tv[(k + 2) % 3];
                    let (ax, ay) = coords[a];
                    let (bx, by) = coords[b];
                    if orient2d(ax, ay, bx, by, px, py) <= 0.0 {
                        all_positive = false;
                    }
                    let outer_slot = nb.and_then(|o_idx| {
                        tris[o_idx].adj.iter().position(|&slot| slot == Some(ti))
                    });
                    boundary.push((a, b, nb, outer_slot));
                }
            }
            if !all_positive || boundary.len() < 3 {
                continue;
            }

            // 5. Remove bad triangles into the free list and stitch new triangles around `p`.
            for &ti in &bad_set {
                tris[ti].alive = false;
                free_slots.push(ti);
            }

            new_indices.clear();
            for &(a, b, outer, outer_slot) in &boundary {
                let tri = DelaunayTri {
                    v: [a, b, p],
                    // adj[0] opposite a -> edge (b, p); adj[1] opposite b -> edge (p, a); adj[2] opposite p -> edge (a, b).
                    adj: [None, None, outer],
                    alive: true,
                };
                let new_idx = if let Some(slot) = free_slots.pop() {
                    tris[slot] = tri;
                    slot
                } else {
                    let idx = tris.len();
                    tris.push(tri);
                    idx
                };
                if let (Some(o_idx), Some(o_slot)) = (outer, outer_slot) {
                    tris[o_idx].adj[o_slot] = Some(new_idx);
                }
                new_indices.push(new_idx);
            }

            // Pair radial edges between newly created triangles:
            // Triangle `t_i = (a_i, b_i, p)` shares edge `(b_i, p)` with `t_j = (a_j, b_j, p)` where `a_j == b_i`.
            for (i, &ti) in new_indices.iter().enumerate() {
                let bi = tris[ti].v[1];
                for (j, &tj) in new_indices.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    if tris[tj].v[0] == bi {
                        tris[ti].adj[0] = Some(tj);
                        tris[tj].adj[1] = Some(ti);
                        break;
                    }
                }
            }
            last_tri = new_indices[0];
            break;
        }
    }

    tris.into_iter()
        .filter(|t| t.alive && t.v[0] < n && t.v[1] < n && t.v[2] < n)
        .map(|t| t.v)
        .collect()
}

/// Bilinearly samples a DTM `Array2D<f64>` at world coordinate `(wx, wy)`, falling back to the
/// nearest valid cell in a 3x3 neighborhood when on the raster boundary.
fn sample_dtm_bilinear(dtm: &Array2D<f64>, wx: f64, wy: f64) -> Option<f64> {
    let w = dtm.width();
    let h = dtm.height();
    if w == 0 || h == 0 {
        return None;
    }
    let gt = dtm.geotransform;
    let nd = dtm.no_data();
    let is_valid = |v: f64| v.is_finite() && !v.rd_eq(nd);

    // Continuous pixel index centered at (c + 0.5, r + 0.5).
    let fc = (wx - gt[0]) / gt[1] - 0.5;
    let fr = (wy - gt[3]) / gt[5] - 0.5;

    let c0 = fc.floor() as isize;
    let r0 = fr.floor() as isize;
    let dc = fc - c0 as f64;
    let dr = fr - r0 as f64;

    if c0 >= 0 && r0 >= 0 && (c0 + 1) < w as isize && (r0 + 1) < h as isize {
        let (cu, ru) = (c0 as usize, r0 as usize);
        let z00 = dtm.get(cu, ru);
        let z10 = dtm.get(cu + 1, ru);
        let z01 = dtm.get(cu, ru + 1);
        let z11 = dtm.get(cu + 1, ru + 1);
        if is_valid(z00) && is_valid(z10) && is_valid(z01) && is_valid(z11) {
            let z0 = z00 + (z10 - z00) * dc;
            let z1 = z01 + (z11 - z01) * dc;
            return Some(z0 + (z1 - z0) * dr);
        }
    }

    // Clamp to nearest valid cell within 3x3 around the rounded cell index.
    let cr = (fc.round() as isize).clamp(0, w as isize - 1);
    let rr = (fr.round() as isize).clamp(0, h as isize - 1);
    let center = dtm.get(cr as usize, rr as usize);
    if is_valid(center) {
        return Some(center);
    }
    let mut best: Option<(f64, f64)> = None;
    for dr_i in -1..=1 {
        let nr = rr + dr_i;
        if nr < 0 || nr >= h as isize {
            continue;
        }
        for dc_i in -1..=1 {
            let nc = cr + dc_i;
            if nc < 0 || nc >= w as isize {
                continue;
            }
            let val = dtm.get(nc as usize, nr as usize);
            if is_valid(val) {
                let d2 = (nc as f64 - fc).powi(2) + (nr as f64 - fr).powi(2);
                if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, val));
                }
            }
        }
    }
    best.map(|(_, v)| v)
}

struct TinRasterGrid {
    width: usize,
    height: usize,
    geotransform: [f64; 6],
    nodata: f64,
}

/// Rasterizes a set of 2D Delaunay triangles into `chm` by barycentric interpolation at cell
/// centers, keeping the pixel-wise maximum and optionally skipping triangles with any 2D edge
/// longer than `max_edge`.
fn rasterize_tin_max(
    pts: &[(f64, f64, f64)],
    tris: &[[usize; 3]],
    grid: &TinRasterGrid,
    max_edge: Option<f64>,
    chm: &mut [f64],
) {
    let w = grid.width;
    let h = grid.height;
    let gt = grid.geotransform;
    let nodata = grid.nodata;
    let max_edge2 = max_edge.map(|e| e * e);
    let cell_x = gt[1];
    let cell_y = gt[5]; // negative for north-up rasters
    const EPS: f64 = 1e-10;

    for &[a, b, c] in tris {
        let (x1, y1, z1) = pts[a];
        let (x2, y2, z2) = pts[b];
        let (x3, y3, z3) = pts[c];

        if let Some(me2) = max_edge2 {
            let d12 = (x2 - x1).powi(2) + (y2 - y1).powi(2);
            let d23 = (x3 - x2).powi(2) + (y3 - y2).powi(2);
            let d31 = (x1 - x3).powi(2) + (y1 - y3).powi(2);
            if d12 > me2 || d23 > me2 || d31 > me2 {
                continue;
            }
        }

        let denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
        if denom.abs() < 1e-10 {
            continue;
        }
        let m1x = (y2 - y3) / denom;
        let m1y = (x3 - x2) / denom;
        let m2x = (y3 - y1) / denom;
        let m2y = (x1 - x3) / denom;

        let min_x = x1.min(x2).min(x3);
        let max_x = x1.max(x2).max(x3);
        let min_y = y1.min(y2).min(y3);
        let max_y = y1.max(y2).max(y3);

        let c_lo_f = ((min_x - gt[0]) / cell_x - 0.5).ceil();
        let c_hi_f = ((max_x - gt[0]) / cell_x - 0.5).floor();
        if c_hi_f < 0.0 || c_lo_f >= w as f64 {
            continue;
        }
        let c_lo = c_lo_f.max(0.0) as usize;
        let c_hi = (c_hi_f.min((w - 1) as f64)) as usize;

        // For north-up rasters (`cell_y < 0`), larger y corresponds to smaller row index `r`.
        let r_a = (max_y - gt[3]) / cell_y - 0.5;
        let r_b = (min_y - gt[3]) / cell_y - 0.5;
        let r_lo_f = r_a.min(r_b).ceil();
        let r_hi_f = r_a.max(r_b).floor();
        if r_hi_f < 0.0 || r_lo_f >= h as f64 {
            continue;
        }
        let r_lo = r_lo_f.max(0.0) as usize;
        let r_hi = (r_hi_f.min((h - 1) as f64)) as usize;

        for r in r_lo..=r_hi {
            let py = gt[3] + (r as f64 + 0.5) * cell_y;
            let row_off = r * w;
            for col in c_lo..=c_hi {
                let px = gt[0] + (col as f64 + 0.5) * cell_x;
                let l1 = m1x * (px - x3) + m1y * (py - y3);
                let l2 = m2x * (px - x3) + m2y * (py - y3);
                let l3 = 1.0 - l1 - l2;
                if (-EPS..=1.0 + EPS).contains(&l1)
                    && (-EPS..=1.0 + EPS).contains(&l2)
                    && (-EPS..=1.0 + EPS).contains(&l3)
                {
                    let val = (l1 * z1 + l2 * z2 + l3 * z3).max(0.0);
                    let idx = row_off + col;
                    let cur = chm[idx];
                    if !cur.is_finite() || cur.rd_eq(nodata) || val > cur {
                        chm[idx] = val;
                    }
                }
            }
        }
    }
}

/// Computes a pit-free Canopy Height Model [`Array2D<f64>`] in memory using the multi-level
/// partial TIN algorithm of Khosravipour et al. (2014).
///
/// Steps:
/// 1. Apply configured outlier filters (`options.dtm`), classify ground points, and build a
///    bare-earth DTM on the point cloud's exact extent at `options.resolution`.
/// 2. Compute Height Above Ground (`h = max(0, z - dtm(x, y))`) for every point. Canopy returns
///    falling over DTM NoData cells (where no valid bilinear or 3x3 neighbor exists) are excluded
///    from the CHM TIN rather than fabricating ground elevations.
/// 3. Construct the base CHM TIN from all normalized returns (`h >= 0.0`, no edge-length cutoff).
/// 4. For each positive height threshold `h_k > 0.0` in `options.height_thresholds`, build a
///    partial Delaunay TIN from returns with `h >= h_k`, skipping triangles with any 2D edge
///    longer than `max_triangle_edge` (default `3.0 * options.resolution`), and composite the
///    pixel-wise maximum across all surfaces.
pub fn points_to_chm_pit_free(
    cloud: &PointCloud,
    options: &PitFreeChmOptions,
) -> Result<Array2D<f64>> {
    if !options.resolution.is_finite() || options.resolution <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "resolution must be finite and > 0, got {}",
            options.resolution
        )));
    }
    if let Some(edge) = options.max_triangle_edge
        && (!edge.is_finite() || edge <= 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "max_triangle_edge must be finite and > 0, got {edge}"
        )));
    }
    for &t in &options.height_thresholds {
        if !t.is_finite() || t < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "height_thresholds entries must be finite and >= 0, got {t}"
            )));
        }
    }

    let mut dtm_opts = options.dtm.clone();
    dtm_opts.resolution = options.resolution;
    validate_dtm_options(&dtm_opts)?;

    let cleaned = if has_dtm_outlier_filters(&dtm_opts) {
        apply_dtm_outlier_filters(cloud, &dtm_opts)?
    } else {
        cloud.clone()
    };
    if cleaned.is_empty() {
        return Err(Error::Algorithm("input point cloud is empty".into()));
    }

    let ground = select_ground_points(&cleaned, &dtm_opts)?;
    if ground.is_empty() {
        return Err(Error::Algorithm(format!(
            "no ground points found (class={})",
            dtm_opts.ground_class
        )));
    }

    let shared_bounds = cleaned.bounds().map(|b| [b[0], b[1], b[3], b[4]]);
    let dtm_grid_opts = GridOptions {
        resolution: options.resolution,
        method: dtm_opts.method,
        stats: vec![dtm_opts.stat],
        dimension: "z".to_string(),
        radius: dtm_opts.radius,
        window_size: dtm_opts.window_size,
        bounds: shared_bounds,
        nodata: dtm_opts.nodata,
        ..Default::default()
    };
    let dtm_ds = terrio::pointcloud::grid::grid(&ground, &dtm_grid_opts)?;
    let dtm_arr = richdem::dataset_to_dynamic(dtm_ds, "dtm")?.to_f64();

    let w = dtm_arr.width();
    let h = dtm_arr.height();
    let gt = dtm_arr.geotransform;
    let nd = dtm_opts.nodata;

    // Normalize point elevations to Height Above Ground (HAG), clamping negative values to 0.0.
    let (xs, ys, zs) = extract_xyz(&cleaned)?;
    let mut raw_hag_pts: Vec<(f64, f64, f64)> = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let (x, y, z) = (xs[i], ys[i], zs[i]);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        if let Some(zg) = sample_dtm_bilinear(&dtm_arr, x, y) {
            raw_hag_pts.push((x, y, (z - zg).max(0.0)));
        }
    }
    if raw_hag_pts.is_empty() {
        return Err(Error::Algorithm(
            "no valid height-above-ground points could be normalized against DTM".into(),
        ));
    }

    let base_pts = deduplicate_max_height(&raw_hag_pts);

    // 1. Base CHM surface (all returns, h >= 0.0) via full Delaunay TIN (no edge cutoff).
    let tin_grid = TinRasterGrid {
        width: w,
        height: h,
        geotransform: gt,
        nodata: nd,
    };
    let mut chm_data = vec![nd; w * h];
    let base_tris = delaunay_triangulate_2d(&base_pts);
    rasterize_tin_max(&base_pts, &base_tris, &tin_grid, None, &mut chm_data);

    // Stamp every point into its containing cell so isolated boundary/collinear cells outside
    // the convex hull also receive their exact canopy height.
    for &(x, y, hag) in &base_pts {
        let c = ((x - gt[0]) / gt[1]).floor() as isize;
        let r = ((y - gt[3]) / gt[5]).floor() as isize;
        if c >= 0 && r >= 0 && c < w as isize && r < h as isize {
            let idx = r as usize * w + c as usize;
            let cur = chm_data[idx];
            if !cur.is_finite() || cur.rd_eq(nd) || hag > cur {
                chm_data[idx] = hag;
            }
        }
    }

    // 2. Partial canopy surfaces for each positive height threshold h_k > 0.0, enforcing
    //    `max_triangle_edge` so interior crown pits are bridged while gaps between trees stay open.
    let max_edge = options
        .max_triangle_edge
        .unwrap_or(options.resolution * 3.0);
    let mut thresholds: Vec<f64> = options
        .height_thresholds
        .iter()
        .copied()
        .filter(|&t| t > 0.0)
        .collect();
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup();

    for hk in thresholds {
        let partial_pts: Vec<(f64, f64, f64)> = base_pts
            .iter()
            .copied()
            .filter(|&(_, _, hag)| hag >= hk)
            .collect();
        if partial_pts.len() < 3 {
            continue;
        }
        let partial_tris = delaunay_triangulate_2d(&partial_pts);
        rasterize_tin_max(
            &partial_pts,
            &partial_tris,
            &tin_grid,
            Some(max_edge),
            &mut chm_data,
        );
    }

    let mut chm = Array2D::from_vec(w, h, chm_data)?;
    chm.copy_metadata_from(&dtm_arr);
    chm.set_no_data(nd);
    Ok(chm)
}

/// Generates a pit-free Canopy Height Model (Khosravipour et al. 2014) from a LiDAR file,
/// writes it to `out_tif` (supporting DEFLATE compression and Cloud Optimized GeoTIFF output),
/// and returns the resulting [`Array2D<f64>`].
///
/// Canopy returns falling over DTM NoData cells (where no valid bilinear or 3x3 neighbor exists)
/// are excluded from the CHM TIN rather than fabricating ground elevations.
pub fn lidar_to_chm_pit_free(
    source: impl AsRef<Path>,
    out_tif: impl AsRef<Path>,
    options: &PitFreeChmOptions,
) -> Result<Array2D<f64>> {
    let out_path = out_tif.as_ref();
    crate::raster_ops::ensure_no_path_collision(source.as_ref(), &[out_path])?;
    ensure_parent_dir(out_path)?;

    let cloud = read_lidar(source)?;
    let chm = points_to_chm_pit_free(&cloud, options)?;
    write_array2d_raster(&chm, out_path, options.compress, options.cog)?;
    Ok(chm)
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

        let ground = select_ground_points(&cloud, &LidarToDtmOptions::default()).unwrap();
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

    #[test]
    fn delaunay_regular_integer_grid_drops_zero_points() {
        let mut pts = Vec::new();
        for r in 0..20 {
            for c in 0..20 {
                pts.push((500_000.0 + c as f64, 4_000_000.0 + r as f64, 10.0));
            }
        }
        let tris = delaunay_triangulate_2d(&pts);
        assert!(
            !tris.is_empty(),
            "regular integer grid must produce triangles"
        );
        let mut seen = vec![false; pts.len()];
        for [a, b, c] in tris {
            seen[a] = true;
            seen[b] = true;
            seen[c] = true;
        }
        let dropped = seen.iter().filter(|&&v| !v).count();
        assert_eq!(
            dropped, 0,
            "regular integer grid triangulation must drop zero points"
        );
    }

    #[test]
    fn pit_free_chm_skips_canopy_points_over_dtm_nodata_hole() {
        let nd = -9999.0;
        let mut dtm = Array2D::new(10, 10, 100.0);
        dtm.geotransform = [500_000.0, 1.0, 0.0, 4_000_010.0, 0.0, -1.0];
        dtm.set_no_data(nd);
        // Carve a 5x5 NoData hole in the middle so the center cell (5, 5) has no valid 3x3 neighbor.
        for r in 3..8 {
            for c in 3..8 {
                dtm.set(c, r, nd);
            }
        }
        // Center of the NoData hole returns None (excluded cleanly rather than fabricating ground).
        assert!(sample_dtm_bilinear(&dtm, 500_005.5, 4_000_004.5).is_none());
        // Valid area outside the hole returns 100.0.
        assert_eq!(
            sample_dtm_bilinear(&dtm, 500_001.5, 4_000_008.5),
            Some(100.0)
        );

        // End-to-end check: ground points only on x in [0..5], canopy point far away at x = 25
        // over a DTM NoData region (GridMethod::Statistics, window_size=0, radius=1.0).
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut cls = Vec::new();
        for r in 0..6 {
            for c in 0..6 {
                xs.push(500_000.0 + c as f64 + 0.5);
                ys.push(4_000_000.0 + r as f64 + 0.5);
                zs.push(100.0);
                cls.push(2u8);
            }
        }
        // Valid canopy point over valid ground.
        xs.push(500_003.5);
        ys.push(4_000_003.5);
        zs.push(115.0);
        cls.push(5u8);
        // Isolated canopy return far outside ground coverage (over DTM NoData hole).
        xs.push(500_025.5);
        ys.push(4_000_003.5);
        zs.push(150.0);
        cls.push(5u8);

        let cloud = PointCloud::new(
            vec![
                Dimension::new("x", Values::F64(xs)),
                Dimension::new("y", Values::F64(ys)),
                Dimension::new("z", Values::F64(zs)),
                Dimension::new("classification", Values::U8(cls)),
            ],
            terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
        );
        let chm = points_to_chm_pit_free(
            &cloud,
            &PitFreeChmOptions {
                resolution: 1.0,
                dtm: LidarToDtmOptions {
                    ground_method: GroundFilterMethod::Preclassified,
                    method: GridMethod::Statistics,
                    stat: GridStat::Mean,
                    radius: Some(1.0),
                    window_size: 0,
                    nodata: nd,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let gt = chm.geotransform;
        let hole_c = ((500_025.5 - gt[0]) / gt[1]).floor() as usize;
        let hole_r = ((4_000_003.5 - gt[3]) / gt[5]).floor() as usize;
        assert!(
            chm.get(hole_c, hole_r).rd_eq(nd),
            "canopy point over DTM NoData hole must remain NoData (got {})",
            chm.get(hole_c, hole_r)
        );
    }
}
