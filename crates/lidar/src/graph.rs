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
/// steps (always including the floor `min_elev` and spill rim `max_elev`).
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

fn load_stream_segments(path: &Path) -> Result<Vec<([f64; 2], [f64; 2])>> {
    let p_str = path.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!("non-UTF-8 stream vector path: {}", path.display()))
    })?;
    let ds = terrio::open_vector(p_str, &VectorOpenOptions::default())?;
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
        for feat in &layer.features {
            if let Some(ref g) = feat.geometry {
                push_geom(g, &mut segs);
            }
        }
    }
    Ok(segs)
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

    // Match catchment area by `(level, cell_count)` heuristic (falling back to depression area
    // when no matching catchment record exists; ties at identical `(level, count)` share `cat_area`).
    let mut cat_by_lvl_area: HashMap<(u32, usize), f64> = HashMap::new();
    for r in catchments {
        cat_by_lvl_area.insert((r.level, r.count), r.cat_area);
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
                .copied()
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

    let stream_segs = match stream_vector {
        Some(p) => Some(load_stream_segments(p)?),
        None => None,
    };

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
        let dist_to_stream = match (end_xy, &stream_segs) {
            (Some([ex, ey]), Some(segs)) if !segs.is_empty() => {
                let mut best = f64::INFINITY;
                for &(a, b) in segs {
                    let d = point_to_segment_dist(ex, ey, a[0], a[1], b[0], b[1]);
                    if d < best {
                        best = d;
                    }
                }
                Some(best)
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
