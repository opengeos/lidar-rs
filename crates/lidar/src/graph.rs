//! Stage-storage-area rating curves and directed fill-and-spill connectivity graph export.
//!
//! Produces hydrologic model-ready lookup tables (`rating_curves.csv`), directed
//! depression spill/merge graphs (`connectivity_graph.json`, `.graphml`), and
//! georeferenced spill-path vector layers (`.gpkg`, `.geojson`, `.shp`).

use std::collections::HashMap;
use std::path::Path;

use richdem::{Array2D, DemNum};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::SpatialRef;
use terrio::vector::Coord;
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer, VectorOpenOptions,
    VectorWriteOptions,
};

use crate::catchment::CatchmentHierarchyRecord;
use crate::error::{Error, Result};
use crate::inundation::SpillFlowPath;
use crate::slicing::{InMemoryDepressions, NestedDepression};

/// One elevation sample on a depression's stage-storage-area rating curve.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RatingCurvePoint {
    /// Depression ID (1-based).
    pub dep_id: u32,
    /// Hierarchy level (1 = leaf).
    pub level: u32,
    /// Top-level sink region ID.
    pub region_id: u32,
    /// Water surface stage elevation in DEM vertical units.
    pub stage: f64,
    /// Depth above the depression floor (`stage - min_elev`).
    pub depth: f64,
    /// Inundated surface area at this stage in map units squared.
    pub area: f64,
    /// Cumulative stored volume below this stage in map units cubed.
    pub volume: f64,
}

/// Computes stage-storage-area rating curves for every delineated depression directly
/// from the sink elevation raster and nested depression hierarchy.
///
/// Each depression's elevation span `[min_elev, max_elev]` is sampled at `interval`
/// steps (always including the floor `min_elev` and spill rim `max_elev`). When a
/// depression's span requires more than 500 steps at `interval`, sampling is capped at
/// 500 evenly spaced intervals (`span / 500`) across the full `[min_elev, max_elev]` range.
pub fn compute_rating_curves(
    sink: &Array2D<f64>,
    deps: &InMemoryDepressions,
    interval: f64,
) -> Result<Vec<RatingCurvePoint>> {
    if !interval.is_finite() || interval <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rating curve interval must be finite and > 0, got {interval}"
        )));
    }
    let w = sink.width();
    let h = sink.height();
    if deps.depression_id.width() != w || deps.depression_id.height() != h {
        return Err(Error::DimensionMismatch(format!(
            "depression_id ({}x{}) does not match sink ({w}x{h})",
            deps.depression_id.width(),
            deps.depression_id.height()
        )));
    }
    let gt = sink.geotransform;
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let cell_area = if det.is_finite() && det > 0.0 {
        det
    } else {
        1.0
    };
    let nd = sink.no_data();

    // Build parent lookup and gather owned elevation values per depression ID.
    let max_id = deps.depressions.iter().map(|d| d.id).max().unwrap_or(0) as usize;
    if max_id == 0 {
        return Ok(Vec::new());
    }
    let mut owned_elevs: Vec<Vec<f64>> = vec![Vec::new(); max_id + 1];
    let sdata = sink.data();
    let id_data = deps.depression_id.data();
    for (&did, &z) in id_data.iter().zip(sdata.iter()) {
        if did > 0 && (did as usize) <= max_id && z.is_finite() && !z.rd_eq(nd) {
            owned_elevs[did as usize].push(z);
        }
    }

    // Build children adjacency so compound (level >= 2) depressions include descendant cells.
    let dep_by_id: HashMap<u32, &NestedDepression> =
        deps.depressions.iter().map(|d| (d.id, d)).collect();

    fn collect_descendant_elevs(
        id: u32,
        dep_by_id: &HashMap<u32, &NestedDepression>,
        owned_elevs: &[Vec<f64>],
        out: &mut Vec<f64>,
    ) {
        if let Some(v) = owned_elevs.get(id as usize) {
            out.extend_from_slice(v);
        }
        if let Some(d) = dep_by_id.get(&id) {
            for &cid in &d.children_ids {
                collect_descendant_elevs(cid, dep_by_id, owned_elevs, out);
            }
        }
    }

    let mut curves = Vec::new();
    let mut buf = Vec::new();
    for d in &deps.depressions {
        buf.clear();
        collect_descendant_elevs(d.id, &dep_by_id, &owned_elevs, &mut buf);
        if buf.is_empty() {
            continue;
        }
        buf.sort_by(|a, b| a.total_cmp(b));
        let z_min = buf[0];
        let z_max = d.max_elev.max(*buf.last().unwrap_or(&z_min));
        let span = (z_max - z_min).max(0.0);

        // Prefix sums for O(log M) exact area & volume at each stage.
        let mut prefix_sum = Vec::with_capacity(buf.len() + 1);
        prefix_sum.push(0.0f64);
        let mut acc = 0.0f64;
        for &z in &buf {
            acc += z;
            prefix_sum.push(acc);
        }

        let n_steps = ((span / interval).ceil() as usize).clamp(1, 500);
        let eff_interval = if n_steps == 500 && span > 500.0 * interval {
            span / n_steps as f64
        } else {
            interval
        };
        for s in 0..=n_steps {
            let stage = if s == n_steps {
                z_max
            } else {
                (z_min + s as f64 * eff_interval).min(z_max)
            };
            // Number of cells with z <= stage
            let k = buf.partition_point(|&z| z <= stage);
            let area = k as f64 * cell_area;
            let vol = ((k as f64) * stage - prefix_sum[k]).max(0.0) * cell_area;
            curves.push(RatingCurvePoint {
                dep_id: d.id,
                level: d.level,
                region_id: d.region_id,
                stage,
                depth: (stage - z_min).max(0.0),
                area,
                volume: vol,
            });
        }
    }

    Ok(curves)
}

/// Writes stage-storage-area rating curves to CSV.
pub fn write_rating_curves_csv(points: &[RatingCurvePoint], path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, rating_curves_to_csv_string(points))?;
    Ok(())
}

/// A depression node in the directed fill-and-spill connectivity graph.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphNode {
    pub id: u32,
    pub parent_id: u32,
    pub level: u32,
    pub region_id: u32,
    pub area: f64,
    pub volume: f64,
    pub avg_depth: f64,
    pub max_depth: f64,
    pub min_elev: f64,
    pub max_elev: f64,
    pub circularity: f64,
    pub hypsometric_integral: f64,
    pub cat_area: f64,
    pub fill_time_h: Option<f64>,
    pub fill_runoff_mm: Option<f64>,
}

/// A directed hydrological linkage (spillover or hierarchical merge) between depressions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphEdge {
    pub from_id: u32,
    /// Destination depression ID (`0` = watershed boundary exit).
    pub to_id: u32,
    /// `"spill"` (overland pour-point cascade) or `"merge"` (child-to-parent nesting).
    pub edge_type: String,
    pub spill_time_h: Option<f64>,
    pub spill_runoff_mm: Option<f64>,
    pub merge_time_h: Option<f64>,
    pub spill_elev: Option<f64>,
    pub length_m: f64,
    pub start_xy: Option<[f64; 2]>,
    pub end_xy: Option<[f64; 2]>,
    /// Distance from spill endpoint to nearest stream feature when a stream layer is supplied.
    pub dist_to_stream: Option<f64>,
}

/// Complete directed fill-and-spill and hierarchical merge connectivity graph.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConnectivityGraph {
    pub rain_intensity_cm_h: f64,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[inline]
fn cell_center_xy(gt: &[f64; 6], col: usize, row: usize) -> [f64; 2] {
    let cf = col as f64 + 0.5;
    let rf = row as f64 + 0.5;
    [
        gt[0] + cf * gt[1] + rf * gt[2],
        gt[3] + cf * gt[4] + rf * gt[5],
    ]
}

fn point_to_segment_dist(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 1e-18 {
        return (px - ax).hypot(py - ay);
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len_sq).clamp(0.0, 1.0);
    let proj_x = ax + t * dx;
    let proj_y = ay + t * dy;
    (px - proj_x).hypot(py - proj_y)
}

struct StreamSegmentIndex {
    segs: Vec<([f64; 2], [f64; 2])>,
    min_x: f64,
    min_y: f64,
    cell_w: f64,
    cell_h: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<usize>>,
    overflow: Vec<usize>,
}

impl StreamSegmentIndex {
    fn build(segs: Vec<([f64; 2], [f64; 2])>) -> Self {
        if segs.len() <= 32 {
            return Self {
                segs,
                min_x: 0.0,
                min_y: 0.0,
                cell_w: 1.0,
                cell_h: 1.0,
                nx: 0,
                ny: 0,
                buckets: Vec::new(),
                overflow: Vec::new(),
            };
        }
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &(a, b) in &segs {
            min_x = min_x.min(a[0]).min(b[0]);
            min_y = min_y.min(a[1]).min(b[1]);
            max_x = max_x.max(a[0]).max(b[0]);
            max_y = max_y.max(a[1]).max(b[1]);
        }
        let dx = (max_x - min_x).max(1e-9);
        let dy = (max_y - min_y).max(1e-9);
        let side = (segs.len() as f64).sqrt().clamp(8.0, 128.0);
        let aspect = (dx / dy).clamp(0.125, 8.0);
        let nx = (side * aspect.sqrt()).round().clamp(4.0, 256.0) as usize;
        let ny = (side / aspect.sqrt()).round().clamp(4.0, 256.0) as usize;
        let cell_w = dx / nx as f64;
        let cell_h = dy / ny as f64;
        let mut buckets = vec![Vec::new(); nx * ny];
        let mut overflow = Vec::new();
        for (idx, &(a, b)) in segs.iter().enumerate() {
            let x0 = a[0].min(b[0]);
            let x1 = a[0].max(b[0]);
            let y0 = a[1].min(b[1]);
            let y1 = a[1].max(b[1]);
            let gx0 = (((x0 - min_x) / cell_w).floor() as isize).clamp(0, nx as isize - 1) as usize;
            let gx1 = (((x1 - min_x) / cell_w).floor() as isize).clamp(0, nx as isize - 1) as usize;
            let gy0 = (((y0 - min_y) / cell_h).floor() as isize).clamp(0, ny as isize - 1) as usize;
            let gy1 = (((y1 - min_y) / cell_h).floor() as isize).clamp(0, ny as isize - 1) as usize;
            if (gx1 - gx0 + 1) * (gy1 - gy0 + 1) > 64 {
                overflow.push(idx);
            } else {
                for gy in gy0..=gy1 {
                    let row_off = gy * nx;
                    for gx in gx0..=gx1 {
                        buckets[row_off + gx].push(idx);
                    }
                }
            }
        }
        Self {
            segs,
            min_x,
            min_y,
            cell_w,
            cell_h,
            nx,
            ny,
            buckets,
            overflow,
        }
    }

    fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    fn nearest_dist(&self, px: f64, py: f64, seen: &mut [u32], stamp: u32) -> f64 {
        if self.segs.is_empty() {
            return f64::INFINITY;
        }
        if self.nx == 0 {
            let mut best = f64::INFINITY;
            for &(a, b) in &self.segs {
                let d = point_to_segment_dist(px, py, a[0], a[1], b[0], b[1]);
                if d < best {
                    best = d;
                }
            }
            return best;
        }
        let mut best = f64::INFINITY;
        for &idx in &self.overflow {
            let (a, b) = self.segs[idx];
            let d = point_to_segment_dist(px, py, a[0], a[1], b[0], b[1]);
            if d < best {
                best = d;
            }
        }
        let qgx = (((px - self.min_x) / self.cell_w).floor() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let qgy = (((py - self.min_y) / self.cell_h).floor() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        let max_r = qgx.max(self.nx - 1 - qgx).max(qgy).max(self.ny - 1 - qgy);

        let mut visit_cell = |gx: usize, gy: usize, best: &mut f64| {
            for &idx in &self.buckets[gy * self.nx + gx] {
                if seen[idx] == stamp {
                    continue;
                }
                seen[idx] = stamp;
                let (a, b) = self.segs[idx];
                let d = point_to_segment_dist(px, py, a[0], a[1], b[0], b[1]);
                if d < *best {
                    *best = d;
                }
            }
        };

        for r in 0..=max_r {
            if r > 0 {
                let mut ring_lb = f64::INFINITY;
                if qgx >= r {
                    let x_hi = self.min_x + (qgx - r + 1) as f64 * self.cell_w;
                    ring_lb = ring_lb.min((px - x_hi).max(0.0));
                }
                if qgx + r < self.nx {
                    let x_lo = self.min_x + (qgx + r) as f64 * self.cell_w;
                    ring_lb = ring_lb.min((x_lo - px).max(0.0));
                }
                if qgy >= r {
                    let y_hi = self.min_y + (qgy - r + 1) as f64 * self.cell_h;
                    ring_lb = ring_lb.min((py - y_hi).max(0.0));
                }
                if qgy + r < self.ny {
                    let y_lo = self.min_y + (qgy + r) as f64 * self.cell_h;
                    ring_lb = ring_lb.min((y_lo - py).max(0.0));
                }
                if ring_lb >= best {
                    break;
                }
            }

            let gx_lo = qgx.saturating_sub(r);
            let gx_hi = (qgx + r).min(self.nx - 1);
            let gy_lo = qgy.saturating_sub(r);
            let gy_hi = (qgy + r).min(self.ny - 1);

            if r == 0 {
                visit_cell(qgx, qgy, &mut best);
            } else {
                if qgy >= r {
                    for gx in gx_lo..=gx_hi {
                        visit_cell(gx, gy_lo, &mut best);
                    }
                }
                if qgy + r < self.ny {
                    for gx in gx_lo..=gx_hi {
                        visit_cell(gx, gy_hi, &mut best);
                    }
                }
                let inner_y_lo = if qgy >= r { gy_lo + 1 } else { gy_lo };
                let inner_y_hi = if qgy + r < self.ny {
                    gy_hi.saturating_sub(1)
                } else {
                    gy_hi
                };
                if inner_y_lo <= inner_y_hi {
                    if qgx >= r {
                        for gy in inner_y_lo..=inner_y_hi {
                            visit_cell(gx_lo, gy, &mut best);
                        }
                    }
                    if qgx + r < self.nx {
                        for gy in inner_y_lo..=inner_y_hi {
                            visit_cell(gx_hi, gy, &mut best);
                        }
                    }
                }
            }
        }
        best
    }
}

fn load_stream_segments(path: &Path, dem_projection: &str) -> Result<StreamSegmentIndex> {
    let p_str = path.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!("non-UTF-8 stream vector path: {}", path.display()))
    })?;
    let ds = terrio::open_vector(p_str, &VectorOpenOptions::default())?;
    let dem_srs = if dem_projection.is_empty() {
        None
    } else {
        SpatialRef::parse(dem_projection)
            .ok()
            .filter(|s| !s.is_unknown())
    };
    let mut segs = Vec::new();
    fn push_geom(g: &Geometry, segs: &mut Vec<([f64; 2], [f64; 2])>) {
        match g {
            Geometry::LineString(coords) => {
                for w in coords.windows(2) {
                    segs.push(([w[0].x, w[0].y], [w[1].x, w[1].y]));
                }
            }
            Geometry::MultiLineString(lines) => {
                for line in lines {
                    for w in line.windows(2) {
                        segs.push(([w[0].x, w[0].y], [w[1].x, w[1].y]));
                    }
                }
            }
            Geometry::GeometryCollection(gs) => {
                for sub in gs {
                    push_geom(sub, segs);
                }
            }
            _ => {}
        }
    }
    for layer in &ds.layers {
        if let Some(ref d_srs) = dem_srs {
            let l_srs = SpatialRef::from_layer(layer);
            if !l_srs.is_unknown() && !d_srs.same_as(&l_srs) {
                return Err(Error::InvalidArgument(format!(
                    "stream vector layer '{}' CRS ({}) does not match DEM CRS ({})",
                    layer.name,
                    l_srs.label(),
                    d_srs.label()
                )));
            }
        }
        for feat in &layer.features {
            if let Some(ref g) = feat.geometry {
                push_geom(g, &mut segs);
            }
        }
    }
    Ok(StreamSegmentIndex::build(segs))
}

/// Builds a directed connectivity graph combining nested depression nodes, hierarchical
/// child-to-parent merge edges, and simulated overland spill-path edges.
///
/// When `hyetograph` is provided (`[(dt_h, intensity_cm_h)]`), `fill_runoff_mm` and
/// `spill_runoff_mm` integrate cumulative depth along the hyetograph instead of assuming
/// constant `rain_intensity_cm_h`.
///
/// When `stream_vector` is provided (e.g., USGS NHD flowlines in the same CRS as the DEM),
/// each spill edge is annotated with the exact Euclidean distance from its endpoint to the
/// nearest stream segment.
#[allow(clippy::too_many_arguments)]
pub fn build_connectivity_graph(
    deps: &InMemoryDepressions,
    catchments: &[CatchmentHierarchyRecord],
    fill_time: &Array2D<f64>,
    spill_paths: &[SpillFlowPath],
    rain_intensity_cm_h: f64,
    hyetograph: Option<&[(f64, f64)]>,
    stream_vector: Option<&Path>,
) -> Result<ConnectivityGraph> {
    let gt = deps.depression_id.geotransform;
    if fill_time.width() != deps.depression_id.width()
        || fill_time.height() != deps.depression_id.height()
    {
        return Err(Error::DimensionMismatch(format!(
            "fill_time ({}x{}) does not match depression_id ({}x{})",
            fill_time.width(),
            fill_time.height(),
            deps.depression_id.width(),
            deps.depression_id.height()
        )));
    }

    // Approximate per-depression fill time (max finite fill time across its owned cells).
    let max_id = deps.depressions.iter().map(|d| d.id).max().unwrap_or(0) as usize;
    let mut dep_max_ft = vec![f64::NEG_INFINITY; max_id + 1];
    let nd = fill_time.no_data();
    for (&did, &ft) in deps
        .depression_id
        .data()
        .iter()
        .zip(fill_time.data().iter())
    {
        if did > 0
            && (did as usize) <= max_id
            && ft.is_finite()
            && !ft.rd_eq(nd)
            && ft >= 0.0
            && ft > dep_max_ft[did as usize]
        {
            dep_max_ft[did as usize] = ft;
        }
    }

    // Match catchment area by `(level, cell_count)` heuristic. When two catchment records share
    // the same `(level, count)` key, mark the key ambiguous (`None`) and fall back to the
    // depression's own area rather than silently assigning another basin's catchment area.
    let mut cat_by_lvl_area: HashMap<(u32, usize), Option<f64>> = HashMap::new();
    for r in catchments {
        cat_by_lvl_area
            .entry((r.level, r.count))
            .and_modify(|slot| *slot = None)
            .or_insert(Some(r.cat_area));
    }

    let nodes: Vec<GraphNode> = deps
        .depressions
        .iter()
        .map(|d| {
            let ft = dep_max_ft
                .get(d.id as usize)
                .copied()
                .filter(|v| v.is_finite() && *v >= 0.0);
            let cat_area = cat_by_lvl_area
                .get(&(d.level, d.count))
                .and_then(|v| *v)
                .unwrap_or(d.area);
            GraphNode {
                id: d.id,
                parent_id: d.parent_id,
                level: d.level,
                region_id: d.region_id,
                area: d.area,
                volume: d.volume,
                avg_depth: d.avg_depth,
                max_depth: d.max_depth,
                min_elev: d.min_elev,
                max_elev: d.max_elev,
                circularity: d.circularity,
                hypsometric_integral: d.hypsometric_integral,
                cat_area,
                fill_time_h: ft,
                fill_runoff_mm: ft.map(|t| {
                    crate::inundation::cumulative_runoff_mm(t, rain_intensity_cm_h, hyetograph)
                }),
            }
        })
        .collect();

    let stream_index = match stream_vector {
        Some(p) => Some(load_stream_segments(p, &deps.depression_id.projection)?),
        None => None,
    };
    let mut stream_seen: Vec<u32> = stream_index
        .as_ref()
        .map(|idx| vec![0u32; idx.segs.len()])
        .unwrap_or_default();
    let mut query_stamp = 0u32;

    let mut edges = Vec::with_capacity(spill_paths.len() + deps.depressions.len());
    // 1. Overland spill edges
    for sp in spill_paths {
        let start_xy = sp.cells.first().map(|&(c, r)| cell_center_xy(&gt, c, r));
        let end_xy = sp.cells.last().map(|&(c, r)| cell_center_xy(&gt, c, r));
        let mut length_m = 0.0f64;
        for pair in sp.cells.windows(2) {
            let a = cell_center_xy(&gt, pair[0].0, pair[0].1);
            let b = cell_center_xy(&gt, pair[1].0, pair[1].1);
            length_m += (b[0] - a[0]).hypot(b[1] - a[1]);
        }
        let dist_to_stream = match (end_xy, &stream_index) {
            (Some([ex, ey]), Some(idx)) if !idx.is_empty() => {
                query_stamp = query_stamp.wrapping_add(1);
                if query_stamp == 0 {
                    stream_seen.fill(0);
                    query_stamp = 1;
                }
                Some(idx.nearest_dist(ex, ey, &mut stream_seen, query_stamp))
            }
            _ => None,
        };
        edges.push(GraphEdge {
            from_id: sp.from_dep_id,
            to_id: sp.to_dep_id,
            edge_type: "spill".to_string(),
            spill_time_h: Some(sp.spill_time_h),
            spill_runoff_mm: Some(crate::inundation::cumulative_runoff_mm(
                sp.spill_time_h,
                rain_intensity_cm_h,
                hyetograph,
            )),
            merge_time_h: sp.merge_time_h,
            spill_elev: Some(sp.spill_elev),
            length_m,
            start_xy,
            end_xy,
            dist_to_stream,
        });
    }

    // 2. Hierarchical child -> parent merge edges
    for d in &deps.depressions {
        if d.parent_id > 0 {
            let parent_ft = dep_max_ft
                .get(d.parent_id as usize)
                .copied()
                .filter(|v| v.is_finite() && *v >= 0.0);
            edges.push(GraphEdge {
                from_id: d.id,
                to_id: d.parent_id,
                edge_type: "merge".to_string(),
                spill_time_h: None,
                spill_runoff_mm: None,
                merge_time_h: parent_ft,
                spill_elev: Some(d.max_elev),
                length_m: 0.0,
                start_xy: None,
                end_xy: None,
                dist_to_stream: None,
            });
        }
    }

    Ok(ConnectivityGraph {
        rain_intensity_cm_h,
        nodes,
        edges,
    })
}

/// Writes a [`ConnectivityGraph`] to pretty-printed JSON.
#[cfg(feature = "serde")]
pub fn write_connectivity_graph_json(
    graph: &ConnectivityGraph,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(graph)
        .map_err(|e| Error::Algorithm(format!("failed to serialize connectivity graph: {e}")))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Formats stage-storage-area rating curves as a CSV string.
pub fn rating_curves_to_csv_string(curves: &[RatingCurvePoint]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("dep_id,level,region_id,stage,depth,area,volume\n");
    for pt in curves {
        let _ = writeln!(
            out,
            "{},{},{},{:.6},{:.6},{:.4},{:.4}",
            pt.dep_id, pt.level, pt.region_id, pt.stage, pt.depth, pt.area, pt.volume
        );
    }
    out
}

/// Formats a [`ConnectivityGraph`] as a GraphML XML string (compatible with NetworkX, Gephi, igraph).
pub fn connectivity_graph_to_graphml_string(graph: &ConnectivityGraph) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    let _ = writeln!(
        out,
        r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#
    );
    for (id, target, name, ty) in [
        ("n_level", "node", "level", "int"),
        ("n_region", "node", "region_id", "int"),
        ("n_parent", "node", "parent_id", "int"),
        ("n_area", "node", "area", "double"),
        ("n_volume", "node", "volume", "double"),
        ("n_max_depth", "node", "max_depth", "double"),
        ("n_cat_area", "node", "cat_area", "double"),
        ("n_fill_mm", "node", "fill_runoff_mm", "double"),
        ("e_type", "edge", "edge_type", "string"),
        ("e_spill_mm", "edge", "spill_runoff_mm", "double"),
        ("e_len_m", "edge", "length_m", "double"),
        ("e_stream_m", "edge", "dist_to_stream", "double"),
    ] {
        let _ = writeln!(
            out,
            r#"  <key id="{id}" for="{target}" attr.name="{name}" attr.type="{ty}"/>"#
        );
    }
    let _ = writeln!(out, r#"  <graph id="connectivity" edgedefault="directed">"#);
    if graph.edges.iter().any(|e| e.to_id == 0) {
        let _ = writeln!(out, r#"    <node id="n0">"#);
        let _ = writeln!(out, r#"      <data key="n_level">0</data>"#);
        let _ = writeln!(out, r#"    </node>"#);
    }
    for n in &graph.nodes {
        let _ = writeln!(out, r#"    <node id="n{}">"#, n.id);
        let _ = writeln!(out, r#"      <data key="n_level">{}</data>"#, n.level);
        let _ = writeln!(out, r#"      <data key="n_region">{}</data>"#, n.region_id);
        let _ = writeln!(out, r#"      <data key="n_parent">{}</data>"#, n.parent_id);
        let _ = writeln!(out, r#"      <data key="n_area">{:.4}</data>"#, n.area);
        let _ = writeln!(out, r#"      <data key="n_volume">{:.4}</data>"#, n.volume);
        let _ = writeln!(
            out,
            r#"      <data key="n_max_depth">{:.4}</data>"#,
            n.max_depth
        );
        let _ = writeln!(
            out,
            r#"      <data key="n_cat_area">{:.4}</data>"#,
            n.cat_area
        );
        if let Some(fm) = n.fill_runoff_mm {
            let _ = writeln!(out, r#"      <data key="n_fill_mm">{fm:.4}</data>"#);
        }
        let _ = writeln!(out, r#"    </node>"#);
    }
    fn xml_escape(s: &str) -> String {
        let mut escaped = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '&' => escaped.push_str("&amp;"),
                '<' => escaped.push_str("&lt;"),
                '>' => escaped.push_str("&gt;"),
                '"' => escaped.push_str("&quot;"),
                '\'' => escaped.push_str("&apos;"),
                other => escaped.push(other),
            }
        }
        escaped
    }
    for (ei, e) in graph.edges.iter().enumerate() {
        let _ = writeln!(
            out,
            r#"    <edge id="e{ei}" source="n{}" target="n{}">"#,
            e.from_id, e.to_id
        );
        let _ = writeln!(
            out,
            r#"      <data key="e_type">{}</data>"#,
            xml_escape(&e.edge_type)
        );
        if let Some(sm) = e.spill_runoff_mm {
            let _ = writeln!(out, r#"      <data key="e_spill_mm">{sm:.4}</data>"#);
        }
        let _ = writeln!(out, r#"      <data key="e_len_m">{:.4}</data>"#, e.length_m);
        if let Some(ds) = e.dist_to_stream {
            let _ = writeln!(out, r#"      <data key="e_stream_m">{ds:.4}</data>"#);
        }
        let _ = writeln!(out, r#"    </edge>"#);
    }
    let _ = writeln!(out, r#"  </graph>"#);
    let _ = writeln!(out, r#"</graphml>"#);
    out
}

/// Writes a [`ConnectivityGraph`] to GraphML XML (directly readable by NetworkX, Gephi, and igraph).
pub fn write_connectivity_graphml(graph: &ConnectivityGraph, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, connectivity_graph_to_graphml_string(graph))?;
    Ok(())
}

/// Writes simulated overland spill flow paths as a georeferenced polyline vector file
/// (`.gpkg`, `.geojson`, or `.shp` determined by file extension).
pub fn write_spill_paths_vector(
    spill_paths: &[SpillFlowPath],
    geotransform: &[f64; 6],
    projection: &str,
    rain_intensity_cm_h: f64,
    hyetograph: Option<&[(f64, f64)]>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let out_str = path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", path.display())))?;

    let mut layer = Layer::new("spill_paths".to_string()).with_geom_type(GeometryType::LineString);
    if !projection.is_empty()
        && let Ok(srs) = SpatialRef::parse(projection)
    {
        srs.apply_to_layer(&mut layer);
    }
    for (name, ftype) in [
        ("from_dep_id", FieldType::Integer),
        ("from_level", FieldType::Integer),
        ("to_dep_id", FieldType::Integer),
        ("spill_time_h", FieldType::Float),
        ("spill_runoff_mm", FieldType::Float),
        ("merge_time_h", FieldType::Float),
        ("spill_elev", FieldType::Float),
        ("length_m", FieldType::Float),
    ] {
        layer.schema.add_field(FieldDef::new(name, ftype));
    }

    for (idx, sp) in spill_paths.iter().enumerate() {
        if sp.cells.len() < 2 {
            continue;
        }
        let coords: Vec<Coord> = sp
            .cells
            .iter()
            .map(|&(c, r)| {
                let [x, y] = cell_center_xy(geotransform, c, r);
                Coord::xy(x, y)
            })
            .collect();
        let mut length_m = 0.0f64;
        for w in coords.windows(2) {
            length_m += (w[1].x - w[0].x).hypot(w[1].y - w[0].y);
        }
        layer.features.push(Feature {
            fid: (idx + 1) as u64,
            geometry: Some(Geometry::LineString(coords)),
            attributes: vec![
                FieldValue::Integer(sp.from_dep_id as i64),
                FieldValue::Integer(sp.from_level as i64),
                FieldValue::Integer(sp.to_dep_id as i64),
                FieldValue::Float(sp.spill_time_h),
                FieldValue::Float(crate::inundation::cumulative_runoff_mm(
                    sp.spill_time_h,
                    rain_intensity_cm_h,
                    hyetograph,
                )),
                sp.merge_time_h
                    .map(FieldValue::Float)
                    .unwrap_or(FieldValue::Null),
                FieldValue::Float(sp.spill_elev),
                FieldValue::Float(length_m),
            ],
        });
    }

    terrio::write_vector(&[layer], out_str, &VectorWriteOptions::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dep(id: u32, level: u32, count: usize, area: f64) -> NestedDepression {
        NestedDepression {
            id,
            level,
            count,
            area,
            volume: area,
            avg_depth: 1.0,
            max_depth: 2.0,
            min_elev: 100.0,
            max_elev: 102.0,
            children_ids: Vec::new(),
            region_id: id,
            perimeter: 10.0,
            major_axis: 3.0,
            minor_axis: 2.0,
            elongatedness: 1.5,
            eccentricity: 0.7,
            orientation: 0.0,
            area_bbox_ratio: 0.8,
            parent_id: 0,
            circularity: 0.8,
            depth_diam_ratio: 0.5,
            hypsometric_integral: 0.5,
            rim_slope_deg: 20.0,
            artifact_flag: false,
        }
    }

    fn make_cat_rec(
        level: u32,
        level_fid: u32,
        count: usize,
        cat_area: f64,
    ) -> CatchmentHierarchyRecord {
        CatchmentHierarchyRecord {
            level,
            level_fid,
            count,
            area: count as f64,
            cat_area,
            dep_to_cat_ratio: count as f64 / cat_area,
            min_elev: 100.0,
            max_elev: 102.0,
            range: 2.0,
            mean_elev: 101.0,
            std_elev: 0.5,
            sum_elev: count as f64 * 101.0,
            volume: count as f64,
            mean_depth: 1.0,
        }
    }

    #[test]
    fn stream_segment_index_matches_brute_force() {
        let mut segs = Vec::new();
        for i in 0..80 {
            let x = (i % 10) as f64 * 10.0 + (i as f64 * 0.37).sin();
            let y = (i / 10) as f64 * 12.0 + (i as f64 * 0.53).cos();
            segs.push(([x, y], [x + 4.5, y + 3.0]));
        }
        // Add a long segment spanning the full grid to exercise the overflow list.
        segs.push(([0.0, 0.0], [95.0, 90.0]));

        let idx = StreamSegmentIndex::build(segs.clone());
        let mut seen = vec![0u32; segs.len()];
        let queries = [
            [15.2, 24.8],
            [50.0, 50.0],
            [-25.0, 40.0],
            [140.0, 120.0],
            [0.0, 95.0],
            [82.3, 11.7],
        ];
        for (qi, &[qx, qy]) in queries.iter().enumerate() {
            let indexed = idx.nearest_dist(qx, qy, &mut seen, (qi + 1) as u32);
            let brute = segs
                .iter()
                .map(|&(a, b)| point_to_segment_dist(qx, qy, a[0], a[1], b[0], b[1]))
                .fold(f64::INFINITY, f64::min);
            assert!(
                (indexed - brute).abs() < 1e-12,
                "mismatch for query ({qx}, {qy}): indexed={indexed}, brute={brute}"
            );
        }
    }

    #[test]
    fn connectivity_graph_falls_back_on_catchment_key_collision() {
        let mut dep_id = Array2D::from_vec(4, 4, vec![0i32; 16]).unwrap();
        dep_id.geotransform = [0.0, 1.0, 0.0, 4.0, 0.0, -1.0];
        let dep_lvl = Array2D::from_vec(4, 4, vec![0i32; 16]).unwrap();
        let fill_time = Array2D::from_vec(4, 4, vec![-1.0f64; 16]).unwrap();

        let deps = InMemoryDepressions {
            depression_id: dep_id,
            depression_level: dep_lvl,
            max_level: 1,
            num_regions: 3,
            depressions: vec![
                make_dep(1, 1, 25, 25.0),
                make_dep(2, 1, 25, 25.0),
                make_dep(3, 1, 40, 40.0),
            ],
        };
        let catchments = vec![
            make_cat_rec(1, 1, 25, 100.0),
            make_cat_rec(1, 2, 25, 250.0),
            make_cat_rec(1, 3, 40, 300.0),
        ];

        let graph =
            build_connectivity_graph(&deps, &catchments, &fill_time, &[], 5.0, None, None).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        // Colliding `(level=1, count=25)` keys fall back to `d.area` rather than overwriting.
        assert!((graph.nodes[0].cat_area - 25.0).abs() < 1e-12);
        assert!((graph.nodes[1].cat_area - 25.0).abs() < 1e-12);
        // Unique `(level=1, count=40)` key retains its catchment area.
        assert!((graph.nodes[2].cat_area - 300.0).abs() < 1e-12);
    }

    #[test]
    fn load_stream_segments_rejects_crs_mismatch() {
        let tmp = std::env::temp_dir().join(format!(
            "lidar_rs_stream_crs_test_{}.gpkg",
            std::process::id()
        ));
        let mut layer = Layer::new("streams".to_string()).with_geom_type(GeometryType::LineString);
        SpatialRef::parse("EPSG:4326")
            .unwrap()
            .apply_to_layer(&mut layer);
        layer.features.push(Feature {
            fid: 1,
            geometry: Some(Geometry::LineString(vec![
                Coord::xy(-86.0, 37.0),
                Coord::xy(-85.9, 37.1),
            ])),
            attributes: Vec::new(),
        });
        terrio::write_vector(
            &[layer],
            tmp.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let err = load_stream_segments(&tmp, "EPSG:26916");
        let _ = std::fs::remove_file(&tmp);
        assert!(err.is_err(), "mismatched stream CRS must be rejected");
    }
}
