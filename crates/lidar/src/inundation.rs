//! Level-set surface inundation dynamics simulation.
//!
//! Pure-Rust port of `6_Simulate Inundation.py` from `opengeos/lidar`'s ArcGIS
//! Hydrology Analyst toolbox.
//!
//! Given a sink elevation raster (`sink.tif`) and a catchment hierarchy raster
//! (`catchment_hir.tif`, where each depression pixel stores its contributing
//! catchment area in map units squared), [`simulate_inundation`] computes the
//! time-to-inundation for every depression cell in a **single parallel level-set pass**
//! (avoiding redundant re-slicing across timesteps) and emits time-series inundation
//! GeoTIFFs (`flood_01.tif`, `flood_02.tif`, …), `fill_time.tif`, and
//! `inundation_summary.csv`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, D8X, D8Y, DemNum, NO_FLOW, Topology, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::catchment::{
    CatchmentHierarchyRecord, d8_flow_directions_resolved, delineate_catchment_hierarchy_in_memory,
    write_catchment_hierarchy_csv,
};
use crate::conditioning::{BreachOptions, BurnCulvertsOptions, breach_depressions, burn_culverts};
use crate::error::{Error, Result};
use crate::filling::{InMemorySinks, extract_sinks_in_memory};
use crate::filtering::median_filter;
#[cfg(feature = "serde")]
use crate::graph::write_connectivity_graph_json;
use crate::graph::{
    build_connectivity_graph, compute_rating_curves, write_connectivity_graphml,
    write_rating_curves_csv, write_spill_paths_vector,
};
use crate::raster_ops::ensure_no_path_collision;
use crate::region::{region_group, region_props_xy};
use crate::slicing::{
    InMemoryDepressions, delineate_depressions_in_memory, write_depressions_csv,
    write_depressions_vector,
};

/// Maximum vertical slices permitted per region to guard against tiny intervals.
const MAX_SLICES_PER_REGION: usize = 100_000;
/// Maximum number of output simulation frames permitted in one run.
const MAX_SIMULATION_STEPS: usize = 10_000;

/// Options controlling [`simulate_inundation`] and [`simulate_inundation_in_memory`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulateInundationOptions {
    /// Minimum number of pixels for a valid depression slice (default `1000`).
    pub min_size: usize,
    /// Minimum depth for a valid depression slice (default `0.2`).
    pub min_depth: f64,
    /// Vertical level-set slicing interval (positive; sliced top-down, default `0.2`).
    pub interval: f64,
    /// Rainfall intensity in **cm/h** (default `5.0`, converted to m/h internally).
    pub rain_intensity_cm_h: f64,
    /// Total rainfall duration in **hours** (default `50.0`).
    pub rain_duration_h: f64,
    /// Output simulation time step in **hours** (default `1.0`).
    pub time_step_h: f64,
    /// Compress output GeoTIFFs with DEFLATE.
    pub compress: bool,
}

impl Default for SimulateInundationOptions {
    fn default() -> Self {
        Self {
            min_size: 1000,
            min_depth: 0.2,
            interval: 0.2,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 50.0,
            time_step_h: 1.0,
            compress: false,
        }
    }
}

/// Summary metrics for one time step of [`simulate_inundation`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InundationStep {
    /// 1-based iteration index.
    pub step: usize,
    /// Elapsed rainfall duration in hours.
    pub time_hours: f64,
    /// Number of inundated pixels at this time step.
    pub inundated_cells: usize,
    /// Inundated surface area in map units squared.
    pub inundated_area: f64,
    /// Fraction of total valid sink area inundated (`0.0..=1.0`).
    pub inundated_fraction: f64,
    /// Path to `flood_<step>.tif`.
    pub raster_path: PathBuf,
}

/// Output paths and per-step summary produced by [`simulate_inundation`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulateInundationResult {
    /// Time-ordered paths to `flood_01.tif`, `flood_02.tif`, …
    pub frame_paths: Vec<PathBuf>,
    /// Path to `fill_time.tif` (`f64`, hours until each sink pixel inundates; `-1.0` = dry/NoData).
    pub fill_time_path: PathBuf,
    /// Path to `inundation_summary.csv`.
    pub summary_csv_path: PathBuf,
    /// Number of top-level sink regions simulated.
    pub num_regions: usize,
    /// Per-timestep inundation summary records.
    pub steps: Vec<InundationStep>,
}

fn validate_options(opts: &SimulateInundationOptions) -> Result<usize> {
    if !opts.min_depth.is_finite() || opts.min_depth < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "min_depth must be finite and >= 0, got {}",
            opts.min_depth
        )));
    }
    if !opts.interval.is_finite() || opts.interval == 0.0 {
        return Err(Error::InvalidArgument(format!(
            "interval must be finite and non-zero, got {}",
            opts.interval
        )));
    }
    if !opts.rain_intensity_cm_h.is_finite() || opts.rain_intensity_cm_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rain_intensity_cm_h must be finite and > 0, got {}",
            opts.rain_intensity_cm_h
        )));
    }
    if !opts.rain_duration_h.is_finite() || opts.rain_duration_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rain_duration_h must be finite and > 0, got {}",
            opts.rain_duration_h
        )));
    }
    if !opts.time_step_h.is_finite() || opts.time_step_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "time_step_h must be finite and > 0, got {}",
            opts.time_step_h
        )));
    }
    let n_steps_f = ((opts.rain_duration_h + 1e-9) / opts.time_step_h).floor();
    if !n_steps_f.is_finite() || n_steps_f < 1.0 || n_steps_f > MAX_SIMULATION_STEPS as f64 {
        return Err(Error::InvalidArgument(format!(
            "simulation produces {n_steps_f} steps; must be between 1 and {MAX_SIMULATION_STEPS}"
        )));
    }
    Ok(n_steps_f as usize)
}

fn affine_metrics(gt: &[f64; 6]) -> (f64, f64, f64) {
    let res_x = gt[1].hypot(gt[4]);
    let res_y = gt[2].hypot(gt[5]);
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let rx = if res_x > 0.0 && res_x.is_finite() {
        res_x
    } else {
        1.0
    };
    let ry = if res_y > 0.0 && res_y.is_finite() {
        res_y
    } else {
        1.0
    };
    let area = if det > 0.0 && det.is_finite() {
        det
    } else {
        rx * ry
    };
    (rx, ry, area)
}

struct RegionFillTask {
    bbox: (usize, usize, usize, usize),
    roi_elev: Vec<f64>,
    roi_catchment: Vec<f64>,
}

/// Computes the minimum inundation time (in hours) for every cell in a single sink region
/// in one top-down level-set pass.
fn region_fill_times(
    task: &RegionFillTask,
    options: &SimulateInundationOptions,
    metrics: (f64, f64, f64),
) -> Result<Vec<f64>> {
    let min_size = options.min_size;
    let min_depth = options.min_depth;
    let step = options.interval.abs();
    let rain_m_h = options.rain_intensity_cm_h / 100.0;
    let (res_x, res_y, cell_area) = metrics;
    let (r0, c0, r1, c1) = task.bbox;
    let bh = r1 - r0;
    let bw = c1 - c0;
    let mut img = task.roi_elev.clone();
    let mut fill_time = vec![f64::INFINITY; bw * bh];

    let mut min_elev = f64::INFINITY;
    let mut max_elev = f64::NEG_INFINITY;
    for &v in &img {
        if v.is_finite() {
            if v < min_elev {
                min_elev = v;
            }
            if v > max_elev {
                max_elev = v;
            }
        }
    }
    if !min_elev.is_finite() || !max_elev.is_finite() {
        return Ok(fill_time);
    }

    let elev_span = (max_elev - min_elev).max(0.0);
    let est_slices = (elev_span / step).ceil();
    if !est_slices.is_finite() || est_slices > MAX_SLICES_PER_REGION as f64 {
        return Err(Error::InvalidArgument(format!(
            "interval={step} requires {est_slices} slices (exceeds MAX_SLICES_PER_REGION={MAX_SLICES_PER_REGION})"
        )));
    }

    let mut slice_idx = 0usize;
    loop {
        let elev = max_elev - (slice_idx as f64) * step;
        if slice_idx > 0 && elev <= min_elev {
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

        let objects = region_props_xy(
            &label_objects,
            nb_labels,
            &img,
            bw,
            bh,
            res_x,
            res_y,
            cell_area,
        );
        for obj in &objects {
            if obj.max_depth < min_depth {
                continue;
            }
            let (fr, fc) = obj.first_pixel;
            let raw_cat = task.roi_catchment[fr * bw + fc];
            // If catchment raster is 0/missing at this cell, fall back to direct rainfall
            // over the depression's own footprint so valid depressions still simulate cleanly.
            let cat_area = if raw_cat.is_finite() && raw_cat > 0.0 {
                raw_cat
            } else {
                obj.area
            };
            if cat_area <= 0.0 {
                continue;
            }
            let t_fill = obj.volume / (cat_area * rain_m_h);
            let (or0, oc0, or1, oc1) = obj.bbox;
            for r in or0..or1 {
                let off = r * bw;
                for c in oc0..oc1 {
                    if label_objects[off + c] == obj.label && t_fill < fill_time[off + c] {
                        fill_time[off + c] = t_fill;
                    }
                }
            }
        }
    }

    Ok(fill_time)
}

/// Computes the per-pixel minimum time to inundation (hours) in memory.
///
/// Pixels that never inundate (or lie outside valid sink regions) are assigned `-1.0`
/// (the output raster's NoData value).
pub fn simulate_inundation_in_memory(
    sink: &Array2D<f64>,
    catchment_area: &Array2D<f64>,
    options: &SimulateInundationOptions,
) -> Result<(Array2D<f64>, usize)> {
    let _ = validate_options(options)?;
    let w = sink.width();
    let h = sink.height();
    if catchment_area.width() != w || catchment_area.height() != h {
        return Err(Error::DimensionMismatch(format!(
            "catchment raster ({}x{}) does not match sink raster ({w}x{h})",
            catchment_area.width(),
            catchment_area.height()
        )));
    }
    for i in 0..6 {
        let diff = (sink.geotransform[i] - catchment_area.geotransform[i]).abs();
        let tol = 1e-6 * sink.geotransform[i].abs().max(1.0);
        if diff > tol {
            return Err(Error::InvalidArgument(format!(
                "catchment geotransform differs from sink raster at index {i}: {} vs {}",
                catchment_area.geotransform[i], sink.geotransform[i]
            )));
        }
    }

    let metrics = affine_metrics(&sink.geotransform);
    let nd = sink.no_data();
    let c_nd = catchment_area.no_data();

    let (labels, nb_regions_u32) = region_group(sink.data(), w, h, options.min_size, |v| {
        v.is_finite() && !v.rd_eq(nd)
    });
    let nb_regions = nb_regions_u32 as usize;

    let mut fill_time_data = vec![-1.0f64; w * h];
    if nb_regions == 0 {
        let mut out = Array2D::from_vec(w, h, fill_time_data)?;
        out.copy_metadata_from(sink);
        out.set_no_data(-1.0);
        return Ok((out, 0));
    }

    // Compute per-region bounding boxes.
    let mut bboxes = vec![(usize::MAX, usize::MAX, 0usize, 0usize); nb_regions];
    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let lbl = labels[off + c];
            if lbl > 0 {
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
    }

    let sdata = sink.data();
    let cdata = catchment_area.data();
    let tasks: Vec<RegionFillTask> = (0..nb_regions)
        .map(|ri| {
            let lbl = (ri + 1) as u32;
            let bbox = bboxes[ri];
            let (r0, c0, r1, c1) = bbox;
            let bh = r1 - r0;
            let bw = c1 - c0;
            let mut roi_elev = vec![f64::NAN; bw * bh];
            let mut roi_catchment = vec![0.0f64; bw * bh];
            for r in r0..r1 {
                let g_off = r * w;
                let l_off = (r - r0) * bw;
                for c in c0..c1 {
                    if labels[g_off + c] == lbl {
                        roi_elev[l_off + (c - c0)] = sdata[g_off + c];
                        let cv = cdata[g_off + c];
                        roi_catchment[l_off + (c - c0)] =
                            if cv.is_finite() && !cv.rd_eq(c_nd) && cv > 0.0 {
                                cv
                            } else {
                                0.0
                            };
                    }
                }
            }
            RegionFillTask {
                bbox,
                roi_elev,
                roi_catchment,
            }
        })
        .collect();

    #[cfg(feature = "parallel")]
    let region_results: Result<Vec<Vec<f64>>> = tasks
        .par_iter()
        .map(|t| region_fill_times(t, options, metrics))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let region_results: Result<Vec<Vec<f64>>> = tasks
        .iter()
        .map(|t| region_fill_times(t, options, metrics))
        .collect();

    let region_results = region_results?;
    for (task, times) in tasks.iter().zip(region_results) {
        let (r0, c0, r1, c1) = task.bbox;
        let bw = c1 - c0;
        for r in r0..r1 {
            let g_off = r * w;
            let l_off = (r - r0) * bw;
            for c in c0..c1 {
                let t = times[l_off + (c - c0)];
                if t.is_finite() {
                    fill_time_data[g_off + c] = t;
                }
            }
        }
    }

    let mut out = Array2D::from_vec(w, h, fill_time_data)?;
    out.copy_metadata_from(sink);
    out.set_no_data(-1.0);
    Ok((out, nb_regions))
}

/// Per-step inundation metrics without a filesystem path (used by in-memory pipelines).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InundationStepSummary {
    /// 1-based iteration index.
    pub step: usize,
    /// Elapsed rainfall duration in hours.
    pub time_hours: f64,
    /// Number of inundated pixels at this time step.
    pub inundated_cells: usize,
    /// Inundated surface area in map units squared.
    pub inundated_area: f64,
    /// Fraction of total valid sink area inundated (`0.0..=1.0`).
    pub inundated_fraction: f64,
}

/// Options for the end-to-end in-memory hydrology & inundation pipeline
/// ([`simulate_inundation_pipeline_in_memory`]).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InundationPipelineOptions {
    /// Median smoothing filter kernel size (`<= 1` skips smoothing; default `3`).
    pub kernel_size: usize,
    /// Simulation and level-set parameters.
    pub simulation: SimulateInundationOptions,
}

impl Default for InundationPipelineOptions {
    fn default() -> Self {
        Self {
            kernel_size: 3,
            simulation: SimulateInundationOptions::default(),
        }
    }
}

/// Overland spill flow path connecting a fully filled upstream depression to a
/// downstream receiving depression (or the DEM outlet boundary).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpillFlowPath {
    /// Upstream depression ID that fills and begins spilling.
    pub from_dep_id: u32,
    /// Hierarchy level of the spilling depression.
    pub from_level: u32,
    /// Downstream receiving depression ID, or `0` when spilling to the DEM boundary.
    ///
    /// When non-zero, identifies a distinct downstream depression (never the spilling basin, its
    /// descendants, or an enclosing ancestor ring). Interior `NO_FLOW` micro-pits are stepped
    /// across rather than used as fallback targets, and a basin with no reachable overland outlet
    /// emits no spill path.
    pub to_dep_id: u32,
    /// Time in hours when the upstream depression becomes fully filled and begins spilling.
    pub spill_time_h: f64,
    /// Time in hours when the overland saddle submerges and the two water bodies merge
    /// (`None` if the channel does not submerge within the basin).
    pub merge_time_h: Option<f64>,
    /// Saddle / pour-point elevation where water overflows out of the upstream depression.
    pub spill_elev: f64,
    /// Ordered raster cell coordinates `(col, row)` along the overland spill path from
    /// the upstream rim across the saddle to the downstream shoreline.
    pub cells: Vec<(usize, usize)>,
}

/// Full in-memory outputs of [`simulate_inundation_pipeline_in_memory`].
#[derive(Debug, Clone)]
pub struct InMemoryInundationPipeline {
    /// Intermediate sink extraction outputs (`dem_refined`, `sink`, `depth`, `region`, ...).
    pub sinks: InMemorySinks,
    /// Nested depression hierarchy outputs (`depression_id`, `depression_level`, `depressions`, ...).
    pub depressions: InMemoryDepressions,
    /// Nested catchment hierarchy raster (contributing area in map units squared).
    pub catchment_hierarchy: Array2D<f64>,
    /// Per-level catchment hierarchy statistics.
    pub catchment_records: Vec<CatchmentHierarchyRecord>,
    /// Per-cell minimum time to inundation in hours (`-1.0` = dry / NoData).
    pub fill_time: Array2D<f64>,
    /// Number of top-level sink regions simulated.
    pub num_regions: usize,
    /// Per-timestep cumulative inundation metrics computed in a single O(N + steps) pass.
    pub steps: Vec<InundationStepSummary>,
    /// Overland fill-and-spill flow paths connecting upstream depressions to downstream
    /// receivers once each upstream depression fills to its spill point.
    pub spill_paths: Vec<SpillFlowPath>,
}

#[derive(Debug, Clone, Copy)]
struct HeapCell {
    elev: f64,
    idx: usize,
}

impl PartialEq for HeapCell {
    fn eq(&self, other: &Self) -> bool {
        self.elev.to_bits() == other.elev.to_bits() && self.idx == other.idx
    }
}

impl Eq for HeapCell {}

impl PartialOrd for HeapCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCell {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by elevation, tie-broken deterministically by cell index.
        other
            .elev
            .total_cmp(&self.elev)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

/// Traces D8 flow downhill from `start_idx` to a receiving pit (`NO_FLOW`) or DEM boundary,
/// returning `Some((trace, target_dep_id))` when the path drains away from `in_source` into a
/// distinct downstream depression (`target_dep_id != 0`, excluded from `desc_ids` and
/// `ancestor_ids`, truncated to the last cell inside `target_dep_id`, and not terminating in an
/// enclosing ancestor-ring pit) or reaches the DEM boundary (`target_dep_id == 0`).
///
/// Tracing all the way to the receiving pit (rather than stopping at a dry outer rim) produces
/// the complete physical overland flow channel so rendering can dynamically trim submerged ends
/// at any simulation timestamp `t`.
///
/// Returns `None` when the path drains back into `in_source` (memoized in `drains_to_source`)
/// or terminates at an interior `NO_FLOW` micro-pit or cycle without a valid downstream target
/// (memoized in `trapped_in_pit`), so the outward Priority-Flood search steps across
/// sub-threshold micro-pits until it locates a valid overland outlet.
#[allow(clippy::too_many_arguments)]
fn trace_downstream_away_from_source(
    start_idx: usize,
    w: usize,
    h: usize,
    flowdirs: &[u8],
    dep_ids: &[i32],
    in_source: &[bool],
    desc_ids: &HashSet<u32>,
    ancestor_ids: &HashSet<u32>,
    drains_to_source: &mut [u32],
    trapped_in_pit: &mut [u32],
    reject_gen: u32,
    seen_stamp: &mut [u32],
    stamp: u32,
) -> Option<(Vec<usize>, u32)> {
    let is_ancestor =
        |idx: usize| dep_ids[idx] > 0 && ancestor_ids.contains(&(dep_ids[idx] as u32));
    let mut trace = Vec::new();
    let mut target_dep = 0u32;
    let mut last_target_len = 0usize;
    let mut cur = start_idx;
    let max_steps = w * h;

    for _ in 0..max_steps {
        if in_source[cur] || drains_to_source[cur] == reject_gen {
            for &idx in &trace {
                drains_to_source[idx] = reject_gen;
            }
            return None;
        }
        // Only short-circuit on `trapped_in_pit` before entering a downstream target: a trace
        // that already passed through `target_dep` still needs to check whether the terminal pit
        // is an ancestor-ring trap (rejected) or an unlabelled pit (truncated back to `target_dep`).
        if target_dep == 0 && trapped_in_pit[cur] == reject_gen {
            for &idx in &trace {
                trapped_in_pit[idx] = reject_gen;
            }
            return None;
        }
        if seen_stamp[cur] == stamp {
            break;
        }
        seen_stamp[cur] = stamp;
        trace.push(cur);

        let did = dep_ids[cur];
        if did > 0 {
            let uid = did as u32;
            if !desc_ids.contains(&uid) && !ancestor_ids.contains(&uid) {
                target_dep = uid;
                last_target_len = trace.len();
            }
        }

        let r = cur / w;
        let c = cur % w;
        let on_boundary = r == 0 || r + 1 == h || c == 0 || c + 1 == w;
        let dir = flowdirs[cur] as usize;
        if dir == NO_FLOW as usize || dir > 8 {
            if target_dep != 0 && !is_ancestor(cur) {
                trace.truncate(last_target_len);
                return Some((trace, target_dep));
            }
            if on_boundary && !is_ancestor(cur) {
                return Some((trace, 0));
            }
            for &idx in &trace {
                trapped_in_pit[idx] = reject_gen;
            }
            return None;
        }
        let nr = r as isize + D8Y[dir] as isize;
        let nc = c as isize + D8X[dir] as isize;
        if nr < 0 || nc < 0 || nr >= h as isize || nc >= w as isize {
            if target_dep != 0 {
                trace.truncate(last_target_len);
            }
            return Some((trace, target_dep));
        }
        cur = (nr as usize) * w + (nc as usize);
    }

    let end_in_ancestor = trace.last().is_some_and(|&last| is_ancestor(last));
    if target_dep != 0 && !end_in_ancestor {
        trace.truncate(last_target_len);
        Some((trace, target_dep))
    } else {
        for &idx in &trace {
            trapped_in_pit[idx] = reject_gen;
        }
        None
    }
}

/// Delineates overland fill-and-spill flow paths connecting each nested depression to its
/// downstream receiving depression (or DEM boundary) once the upstream depression fills.
///
/// For each depression `A`, expands outward from `A`'s footprint in elevation order
/// (Priority-Flood) on `dem_refined` to locate the lowest saddle cell whose resolved D8
/// flow path drains away from `A`, then traces downhill into the receiving downstream
/// depression `B`. Opposing sibling paths across a shared saddle are resolved so only the
/// upstream (earlier-filling / higher) depression spills into the downstream basin.
pub fn delineate_spill_flow_paths_in_memory(
    dem_refined: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    fill_time: &Array2D<f64>,
) -> Result<Vec<SpillFlowPath>> {
    let w = dem_refined.width();
    let h = dem_refined.height();
    let n = w * h;
    if n == 0 || depressions.depressions.is_empty() {
        return Ok(Vec::new());
    }
    if depressions.depression_id.width() != w || depressions.depression_id.height() != h {
        return Err(Error::DimensionMismatch(format!(
            "depression_id ({}x{}) does not match DEM ({w}x{h})",
            depressions.depression_id.width(),
            depressions.depression_id.height()
        )));
    }
    if fill_time.width() != w || fill_time.height() != h {
        return Err(Error::DimensionMismatch(format!(
            "fill_time ({}x{}) does not match DEM ({w}x{h})",
            fill_time.width(),
            fill_time.height()
        )));
    }

    let flowdirs = d8_flow_directions_resolved(dem_refined)?;
    let fdata = flowdirs.data();
    let zdata = dem_refined.data();
    let z_nd = dem_refined.no_data();
    let dep_ids = depressions.depression_id.data();
    let ft_data = fill_time.data();
    let ft_nd = fill_time.no_data();

    // Build parent/children lookup tables for the nested depression tree.
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut parent_map: HashMap<u32, u32> = HashMap::new();
    let mut dep_by_id: HashMap<u32, &crate::slicing::NestedDepression> = HashMap::new();
    for d in &depressions.depressions {
        dep_by_id.insert(d.id, d);
        children_map.insert(d.id, d.children_ids.clone());
        for &cid in &d.children_ids {
            parent_map.insert(cid, d.id);
        }
    }

    // Group pixel indices by innermost depression ID once (O(N)).
    let mut pixels_by_id: HashMap<u32, Vec<usize>> = HashMap::new();
    for (idx, &did) in dep_ids.iter().enumerate() {
        if did > 0 {
            pixels_by_id.entry(did as u32).or_default().push(idx);
        }
    }

    // Reusable work buffers across depressions.
    let mut in_source = vec![false; n];
    let mut visited_pf = vec![0u32; n];
    let mut drains_to_source = vec![0u32; n];
    let mut trapped_in_pit = vec![0u32; n];
    let mut pf_gen = 0u32;
    let mut parent_cell = vec![usize::MAX; n];
    let mut trace_seen = vec![0u32; n];
    let mut trace_gen = 0u32;
    let mut heap: BinaryHeap<HeapCell> = BinaryHeap::new();

    let mut raw_paths: Vec<SpillFlowPath> = Vec::new();

    // Process from lowest level upward so child spills are discovered deterministically.
    let mut ordered_deps: Vec<&crate::slicing::NestedDepression> =
        depressions.depressions.iter().collect();
    ordered_deps.sort_by_key(|d| (d.level, d.id));

    for dep in ordered_deps {
        // Only delineate inter-depression spill paths for nested depressions (those enclosed
        // by a parent) OR when multiple top-level sink regions exist. A single top-level root
        // covering the entire basin spills off the DEM boundary only after everything is full.
        if !parent_map.contains_key(&dep.id) && depressions.num_regions <= 1 {
            continue;
        }

        // Collect all descendant IDs (including dep.id) and strictly enclosing ancestor IDs.
        let mut desc_ids: HashSet<u32> = HashSet::new();
        let mut stack = vec![dep.id];
        while let Some(cur_id) = stack.pop() {
            if desc_ids.insert(cur_id)
                && let Some(kids) = children_map.get(&cur_id)
            {
                stack.extend(kids.iter().copied());
            }
        }
        let mut ancestor_ids: HashSet<u32> = HashSet::new();
        let mut anc = parent_map.get(&dep.id).copied();
        while let Some(pid) = anc {
            ancestor_ids.insert(pid);
            anc = parent_map.get(&pid).copied();
        }

        // Gather source cell mask M_A.
        let mut source_cells: Vec<usize> = Vec::new();
        for &id in &desc_ids {
            if let Some(pts) = pixels_by_id.get(&id) {
                source_cells.extend(pts.iter().copied());
            }
        }
        if source_cells.is_empty() {
            continue;
        }

        // Compute baseline full-fill time of `dep` as the 95th-percentile valid `fill_time`
        // over `dep.id`'s own cells (falling back to all cells in `source_cells`). Once the
        // spill outlet cell `cell_indices[0]` is found below, `spill_time_h` is raised to at
        // least that outlet cell's valid fill time so spilling never starts before water
        // reaches the saddle rim.
        let mut rim_times: Vec<f64> = pixels_by_id
            .get(&dep.id)
            .into_iter()
            .flatten()
            .filter_map(|&idx| {
                let ft = ft_data[idx];
                (ft.is_finite() && !ft.rd_eq(ft_nd) && ft >= 0.0).then_some(ft)
            })
            .collect();
        if rim_times.is_empty() {
            rim_times = source_cells
                .iter()
                .filter_map(|&idx| {
                    let ft = ft_data[idx];
                    (ft.is_finite() && !ft.rd_eq(ft_nd) && ft >= 0.0).then_some(ft)
                })
                .collect();
        }
        if rim_times.is_empty() {
            continue;
        }
        rim_times.sort_by(f64::total_cmp);
        let p95_idx = ((rim_times.len() - 1) * 95) / 100;
        let baseline_spill_time_h = rim_times[p95_idx];

        pf_gen = pf_gen.wrapping_add(1);
        if pf_gen == 0 {
            visited_pf.fill(0);
            drains_to_source.fill(0);
            trapped_in_pit.fill(0);
            pf_gen = 1;
        }
        for &idx in &source_cells {
            in_source[idx] = true;
            visited_pf[idx] = pf_gen;
            parent_cell[idx] = usize::MAX;
        }

        // Seed min-heap with perimeter cells of M_A.
        heap.clear();
        for &idx in &source_cells {
            let r = idx / w;
            let c = idx % w;
            let mut on_perimeter = false;
            for d in 1..=8 {
                let nr = r as isize + D8Y[d] as isize;
                let nc = c as isize + D8X[d] as isize;
                if nr >= 0 && nc >= 0 && nr < h as isize && nc < w as isize {
                    let nidx = (nr as usize) * w + (nc as usize);
                    if !in_source[nidx] {
                        on_perimeter = true;
                        break;
                    }
                }
            }
            if on_perimeter {
                let z = zdata[idx];
                if z.is_finite() && !z.rd_eq(z_nd) {
                    heap.push(HeapCell { elev: z, idx });
                }
            }
        }

        let mut spill_elev = f64::NEG_INFINITY;
        let mut found_path: Option<(Vec<usize>, u32, f64)> = None;

        while let Some(HeapCell { elev, idx: cur }) = heap.pop() {
            if elev > spill_elev {
                spill_elev = elev;
            }
            if !in_source[cur] {
                trace_gen = trace_gen.wrapping_add(1);
                if trace_gen == 0 {
                    trace_seen.fill(0);
                    trace_gen = 1;
                }
                if let Some((down_trace, target_dep)) = trace_downstream_away_from_source(
                    cur,
                    w,
                    h,
                    fdata,
                    dep_ids,
                    &in_source,
                    &desc_ids,
                    &ancestor_ids,
                    &mut drains_to_source,
                    &mut trapped_in_pit,
                    pf_gen,
                    &mut trace_seen,
                    trace_gen,
                ) {
                    // Backtrack from `cur` to the source rim cell in M_A.
                    let mut up_path = Vec::new();
                    let mut p = cur;
                    while p != usize::MAX {
                        up_path.push(p);
                        if in_source[p] {
                            break;
                        }
                        p = parent_cell[p];
                    }
                    up_path.reverse();
                    // Combine rim -> saddle (`up_path`) + saddle -> downstream (`down_trace[1..]`).
                    if down_trace.len() > 1 {
                        up_path.extend_from_slice(&down_trace[1..]);
                    }
                    found_path = Some((up_path, target_dep, spill_elev));
                    break;
                }
            }

            let r = cur / w;
            let c = cur % w;
            for d in 1..=8 {
                let nr = r as isize + D8Y[d] as isize;
                let nc = c as isize + D8X[d] as isize;
                if nr < 0 || nc < 0 || nr >= h as isize || nc >= w as isize {
                    continue;
                }
                let nidx = (nr as usize) * w + (nc as usize);
                if visited_pf[nidx] == pf_gen {
                    continue;
                }
                let nz = zdata[nidx];
                if !nz.is_finite() || nz.rd_eq(z_nd) {
                    continue;
                }
                visited_pf[nidx] = pf_gen;
                parent_cell[nidx] = cur;
                heap.push(HeapCell {
                    elev: nz,
                    idx: nidx,
                });
            }
        }

        // Clear source flags for the next depression.
        for &idx in &source_cells {
            in_source[idx] = false;
        }

        let Some((cell_indices, to_dep_id, s_elev)) = found_path else {
            continue;
        };
        if cell_indices.len() < 2 {
            continue;
        }

        // Ensure spill_time_h is at least the selected outlet rim cell's fill time when valid,
        // falling back to the 95th-percentile basin fill time when outlet timing is unavailable.
        let outlet_ft = ft_data[cell_indices[0]];
        let spill_time_h = if outlet_ft.is_finite() && !outlet_ft.rd_eq(ft_nd) && outlet_ft >= 0.0 {
            baseline_spill_time_h.max(outlet_ft)
        } else {
            baseline_spill_time_h
        };

        // Compute merge_time_h from the interior overland channel cells (excluding source rim
        // and final target shoreline pixel).
        let interior = if cell_indices.len() > 2 {
            &cell_indices[1..cell_indices.len() - 1]
        } else {
            &cell_indices[..]
        };
        let mut max_saddle_ft = 0.0f64;
        let mut all_submerge = true;
        for &idx in interior {
            let ft = ft_data[idx];
            if ft.is_finite() && !ft.rd_eq(ft_nd) && ft >= 0.0 {
                if ft > max_saddle_ft {
                    max_saddle_ft = ft;
                }
            } else {
                all_submerge = false;
            }
        }
        let merge_time_h = if all_submerge {
            if max_saddle_ft <= spill_time_h + 1e-6 {
                // Saddle is already submerged when upstream fills; no visible overland channel.
                continue;
            }
            Some(max_saddle_ft)
        } else {
            None
        };

        let cells: Vec<(usize, usize)> = cell_indices
            .into_iter()
            .map(|idx| (idx % w, idx / w))
            .collect();

        raw_paths.push(SpillFlowPath {
            from_dep_id: dep.id,
            from_level: dep.level,
            to_dep_id,
            spill_time_h,
            merge_time_h,
            spill_elev: s_elev,
            cells,
        });
    }

    // Deduplicate opposing sibling spills (A -> B and B -> A): keep only the earlier-filling
    // (or higher-floor) upstream basin spilling into the downstream receiver.
    let mut keep = vec![true; raw_paths.len()];
    for i in 0..raw_paths.len() {
        if !keep[i] || raw_paths[i].to_dep_id == 0 {
            continue;
        }
        for j in (i + 1)..raw_paths.len() {
            if !keep[j] {
                continue;
            }
            if raw_paths[i].from_dep_id == raw_paths[j].to_dep_id
                && raw_paths[i].to_dep_id == raw_paths[j].from_dep_id
            {
                let i_earlier = raw_paths[i].spill_time_h < raw_paths[j].spill_time_h - 1e-6;
                let j_earlier = raw_paths[j].spill_time_h < raw_paths[i].spill_time_h - 1e-6;
                if j_earlier {
                    keep[i] = false;
                    break;
                } else if i_earlier {
                    keep[j] = false;
                } else {
                    let i_min = dep_by_id
                        .get(&raw_paths[i].from_dep_id)
                        .map_or(0.0, |d| d.min_elev);
                    let j_min = dep_by_id
                        .get(&raw_paths[j].from_dep_id)
                        .map_or(0.0, |d| d.min_elev);
                    if j_min > i_min {
                        keep[i] = false;
                        break;
                    } else {
                        keep[j] = false;
                    }
                }
            }
        }
    }

    let mut result: Vec<SpillFlowPath> = raw_paths
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect();
    result.sort_by(|a, b| {
        a.spill_time_h
            .total_cmp(&b.spill_time_h)
            .then_with(|| a.from_dep_id.cmp(&b.from_dep_id))
    });
    Ok(result)
}

/// Computes cumulative per-step inundation metrics in O(W*H + num_steps) time without
/// rescanning the raster for each time step.
pub fn compute_inundation_step_summaries(
    sink: &Array2D<f64>,
    fill_time: &Array2D<f64>,
    num_steps: usize,
    time_step_h: f64,
) -> Vec<InundationStepSummary> {
    let (_, _, cell_area) = affine_metrics(&sink.geotransform);
    let sink_nd = sink.no_data();
    let total_sink_cells = sink
        .data()
        .iter()
        .filter(|&&v| v.is_finite() && !v.rd_eq(sink_nd))
        .count();

    let ft_nd = fill_time.no_data();
    let mut newly_wet = vec![0usize; num_steps];
    for &ft in fill_time.data() {
        if !ft.is_finite() || ft.rd_eq(ft_nd) || ft < 0.0 {
            continue;
        }
        let mut step_1based = if ft <= 0.0 {
            1usize
        } else {
            let est = (ft / time_step_h).ceil();
            if !est.is_finite() || est > num_steps as f64 + 1.0 {
                continue;
            }
            (est as usize).max(1)
        };
        // Guard against floating-point boundary rounding so the bucket matches `ft <= k * dt` exactly.
        while step_1based > 1 && ((step_1based - 1) as f64) * time_step_h >= ft {
            step_1based -= 1;
        }
        while step_1based <= num_steps && (step_1based as f64) * time_step_h < ft {
            step_1based += 1;
        }
        if step_1based <= num_steps {
            newly_wet[step_1based - 1] += 1;
        }
    }

    let mut steps = Vec::with_capacity(num_steps);
    let mut cumulative = 0usize;
    for (idx, count) in newly_wet.into_iter().enumerate() {
        cumulative += count;
        let k = idx + 1;
        let t_hours = k as f64 * time_step_h;
        steps.push(InundationStepSummary {
            step: k,
            time_hours: t_hours,
            inundated_cells: cumulative,
            inundated_area: cumulative as f64 * cell_area,
            inundated_fraction: if total_sink_cells > 0 {
                cumulative as f64 / total_sink_cells as f64
            } else {
                0.0
            },
        });
    }
    steps
}

/// Runs the end-to-end in-memory hydrology & inundation pipeline:
/// optional median filter → sink extraction → nested depression hierarchy →
/// nested catchment hierarchy → inundation simulation → O(N + steps) step summaries.
pub fn simulate_inundation_pipeline_in_memory(
    raw_dem: &Array2D<f64>,
    options: &InundationPipelineOptions,
) -> Result<InMemoryInundationPipeline> {
    let num_steps = validate_options(&options.simulation)?;
    let sim_opts = &options.simulation;

    let smoothed;
    let dem = if options.kernel_size > 1 {
        smoothed = median_filter(raw_dem, options.kernel_size)?;
        &smoothed
    } else {
        raw_dem
    };

    let sinks = extract_sinks_in_memory(dem, sim_opts.min_size, Topology::D8)?;
    let depressions = delineate_depressions_in_memory(
        &sinks.sink,
        sim_opts.min_size,
        sim_opts.min_depth,
        sim_opts.interval,
    )?;
    let (catchment_hierarchy, catchment_records) = delineate_catchment_hierarchy_in_memory(
        &sinks.dem_refined,
        &depressions.depression_level,
        sim_opts.min_size,
    )?;
    let (fill_time, num_regions) =
        simulate_inundation_in_memory(&sinks.sink, &catchment_hierarchy, sim_opts)?;
    let steps =
        compute_inundation_step_summaries(&sinks.sink, &fill_time, num_steps, sim_opts.time_step_h);
    let spill_paths =
        delineate_spill_flow_paths_in_memory(&sinks.dem_refined, &depressions, &fill_time)?;

    Ok(InMemoryInundationPipeline {
        sinks,
        depressions,
        catchment_hierarchy,
        catchment_records,
        fill_time,
        num_regions,
        steps,
        spill_paths,
    })
}

/// Optional spatially variable hydrologic forcing and storm hyetograph inputs for
/// [`simulate_inundation_with_forcing_in_memory`].
#[derive(Debug, Clone, Default)]
pub struct InundationForcing<'a> {
    /// Optional per-cell effective runoff coefficient raster (`0.0..=1.0`).
    /// When `None`, spatially uniform `C = 1.0` is assumed.
    pub runoff_coeff: Option<&'a Array2D<f64>>,
    /// Optional per-cell vertical infiltration / karst swallow-hole loss rate raster (`cm/h`).
    pub infiltration_cm_h: Option<&'a Array2D<f64>>,
    /// Optional per-cell antecedent water depth raster (vertical DEM units `>= 0.0`) pre-filling
    /// depressions before storm onset.
    pub antecedent_depth: Option<&'a Array2D<f64>>,
    /// Optional piecewise-constant storm hyetograph as `(step_duration_h, intensity_cm_h)` pairs.
    /// When `Some`, overrides constant `options.rain_intensity_cm_h`.
    pub hyetograph: Option<&'a [(f64, f64)]>,
}

/// Converts an NRCS Curve Number (`CN` in `(0, 100]`) raster and storm precipitation
/// depth (`precip_mm > 0`) into a spatially variable effective runoff-coefficient raster
/// (`Q / P` in `[0, 1]`) using the standard NRCS TR-55 rainfall-runoff equation
/// (`S = 25400 / CN - 254`, `Ia = 0.2 * S`).
///
/// Non-finite and NoData CN cells propagate as NoData (`-9999.0`) in the output raster.
pub fn scs_curve_number_runoff(cn: &Array2D<f64>, precip_mm: f64) -> Result<Array2D<f64>> {
    if !precip_mm.is_finite() || precip_mm <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "precip_mm must be finite and > 0, got {precip_mm}"
        )));
    }
    const OUT_NODATA: f64 = -9999.0;
    let nd = cn.no_data();
    let data: Vec<f64> = cn
        .data()
        .iter()
        .map(|&v| {
            if !v.is_finite() || v.rd_eq(nd) {
                return OUT_NODATA;
            }
            if v <= 0.0 {
                return 0.0;
            }
            let c = v.min(100.0);
            let s = 25_400.0 / c - 254.0;
            let ia = 0.2 * s;
            if !s.is_finite() || !ia.is_finite() || precip_mm <= ia {
                0.0
            } else {
                let q = (precip_mm - ia).powi(2) / (precip_mm - ia + s);
                (q / precip_mm).clamp(0.0, 1.0)
            }
        })
        .collect();
    let mut out = Array2D::from_vec(cn.width(), cn.height(), data)?;
    out.copy_metadata_from(cn);
    out.set_no_data(OUT_NODATA);
    Ok(out)
}

/// Generates a piecewise-constant SCS Type II design storm hyetograph as
/// `(step_duration_h, intensity_cm_h)` pairs whose cumulative depth equals `total_precip_mm`.
pub fn scs_type2_hyetograph(
    total_precip_mm: f64,
    duration_h: f64,
    time_step_h: f64,
) -> Result<Vec<(f64, f64)>> {
    if !total_precip_mm.is_finite() || total_precip_mm <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "total_precip_mm must be finite and > 0, got {total_precip_mm}"
        )));
    }
    if !duration_h.is_finite()
        || duration_h <= 0.0
        || !time_step_h.is_finite()
        || time_step_h <= 0.0
    {
        return Err(Error::InvalidArgument(
            "duration_h and time_step_h must be finite and > 0".into(),
        ));
    }
    // Standard 24-h NRCS SCS Type II dimensionless cumulative rainfall curve `(t/T, P/P_tot)`.
    const TYPE2: &[(f64, f64)] = &[
        (0.000, 0.000),
        (0.083, 0.022),
        (0.167, 0.048),
        (0.250, 0.080),
        (0.333, 0.120),
        (0.417, 0.181),
        (0.458, 0.235),
        (0.479, 0.283),
        (0.490, 0.357),
        (0.500, 0.663),
        (0.521, 0.735),
        (0.542, 0.772),
        (0.583, 0.820),
        (0.667, 0.880),
        (0.750, 0.922),
        (0.833, 0.952),
        (0.917, 0.977),
        (1.000, 1.000),
    ];
    let interp = |u: f64| -> f64 {
        let u = u.clamp(0.0, 1.0);
        for w in TYPE2.windows(2) {
            if u <= w[1].0 {
                let f = if w[1].0 > w[0].0 {
                    (u - w[0].0) / (w[1].0 - w[0].0)
                } else {
                    0.0
                };
                return w[0].1 + f * (w[1].1 - w[0].1);
            }
        }
        1.0
    };
    let raw_steps = (duration_h / time_step_h).ceil();
    if !raw_steps.is_finite() || raw_steps < 1.0 || raw_steps > MAX_SIMULATION_STEPS as f64 {
        return Err(Error::InvalidArgument(format!(
            "hyetograph requires {raw_steps} steps (duration_h={duration_h}, time_step_h={time_step_h}), exceeding MAX_SIMULATION_STEPS ({MAX_SIMULATION_STEPS})"
        )));
    }
    let n_steps = raw_steps as usize;
    let mut out = Vec::with_capacity(n_steps);
    let mut prev_t = 0.0f64;
    let mut prev_frac = 0.0f64;
    for k in 1..=n_steps {
        let t = (k as f64 * time_step_h).min(duration_h);
        let dt = (t - prev_t).max(1e-9);
        let frac = interp(t / duration_h);
        let d_mm = ((frac - prev_frac) * total_precip_mm).max(0.0);
        let intensity_cm_h = (d_mm / 10.0) / dt;
        out.push((dt, intensity_cm_h));
        prev_t = t;
        prev_frac = frac;
    }
    Ok(out)
}

/// Integrates cumulative storm rainfall / effective runoff depth (mm) up to `t_hours`.
///
/// When `hyetograph` is `Some`, integrates piecewise-constant `(dt_h, intensity_cm_h)` steps
/// up to `t_hours`. Otherwise uses constant `t_hours * rain_intensity_cm_h * 10.0`.
pub fn cumulative_runoff_mm(
    t_hours: f64,
    rain_intensity_cm_h: f64,
    hyetograph: Option<&[(f64, f64)]>,
) -> f64 {
    if !t_hours.is_finite() || t_hours <= 0.0 {
        return 0.0;
    }
    let Some(steps) = hyetograph else {
        return t_hours * rain_intensity_cm_h * 10.0;
    };
    let mut rem_t = t_hours;
    let mut total_mm = 0.0f64;
    for &(dt, intensity_cm_h) in steps {
        if rem_t <= 0.0 {
            break;
        }
        if dt <= 0.0 {
            continue;
        }
        let used = rem_t.min(dt);
        total_mm += used * intensity_cm_h * 10.0;
        rem_t -= used;
    }
    total_mm
}

/// Formats time-series inundation step summaries as CSV with integrated cumulative runoff (mm).
pub fn format_inundation_curve_csv(
    steps: &[InundationStepSummary],
    rain_intensity_cm_h: f64,
    hyetograph: Option<&[(f64, f64)]>,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "step,time_hours,runoff_mm,inundated_cells,inundated_area,inundated_fraction\n",
    );
    for s in steps {
        let runoff_mm = cumulative_runoff_mm(s.time_hours, rain_intensity_cm_h, hyetograph);
        let _ = writeln!(
            out,
            "{},{:.4},{:.4},{},{:.4},{:.6}",
            s.step,
            s.time_hours,
            runoff_mm,
            s.inundated_cells,
            s.inundated_area,
            s.inundated_fraction
        );
    }
    out
}

/// Maps a required cumulative effective runoff depth (meters) onto elapsed time (hours)
/// along a piecewise-constant hyetograph `[(dt_h, intensity_cm_h)]`, deducting optional
/// continuous vertical loss rate `loss_m_h`.
fn invert_hyetograph_time(
    req_volume_m3: f64,
    eff_cat_area_m2: f64,
    dep_area_m2: f64,
    loss_m_h: f64,
    hyetograph: &[(f64, f64)],
) -> f64 {
    if req_volume_m3 <= 0.0 {
        return 0.0;
    }
    let mut rem_vol = req_volume_m3;
    let mut elapsed_h = 0.0f64;
    for &(dt, rain_cm_h) in hyetograph {
        if dt <= 0.0 {
            continue;
        }
        let inflow_m3_h = eff_cat_area_m2 * (rain_cm_h / 100.0) - dep_area_m2 * loss_m_h;
        if inflow_m3_h > 0.0 {
            let step_vol = inflow_m3_h * dt;
            if step_vol >= rem_vol {
                return elapsed_h + rem_vol / inflow_m3_h;
            }
            rem_vol -= step_vol;
        } else if inflow_m3_h < 0.0 {
            // Vertical infiltration exceeds rainfall: previously accumulated water drains
            // (capped at empty basin `req_volume_m3`).
            rem_vol = (rem_vol - inflow_m3_h * dt).min(req_volume_m3);
        }
        elapsed_h += dt;
    }
    f64::INFINITY
}

struct ForcedRegionFillTask {
    bbox: (usize, usize, usize, usize),
    roi_elev: Vec<f64>,
    roi_catchment: Vec<f64>,
    roi_runoff_coeff: Option<Vec<f64>>,
    roi_infil_m_h: Option<Vec<f64>>,
    roi_antecedent: Option<Vec<f64>>,
}

fn region_fill_times_forced(
    task: &ForcedRegionFillTask,
    options: &SimulateInundationOptions,
    hyetograph: Option<&[(f64, f64)]>,
    metrics: (f64, f64, f64),
) -> Result<Vec<f64>> {
    let min_size = options.min_size;
    let min_depth = options.min_depth;
    let step = options.interval.abs();
    let rain_m_h = options.rain_intensity_cm_h / 100.0;
    let (res_x, res_y, cell_area) = metrics;
    let (r0, c0, r1, c1) = task.bbox;
    let bh = r1 - r0;
    let bw = c1 - c0;
    let mut img = task.roi_elev.clone();
    let mut fill_time = vec![f64::INFINITY; bw * bh];

    // Cells with antecedent depth > 0 are already inundated at t = 0.0 h.
    if let Some(ref ant) = task.roi_antecedent {
        for (ft, (&z, &h0)) in fill_time.iter_mut().zip(img.iter().zip(ant.iter())) {
            if z.is_finite() && h0 > 0.0 {
                *ft = 0.0;
            }
        }
    }

    let mut min_elev = f64::INFINITY;
    let mut max_elev = f64::NEG_INFINITY;
    for &v in &img {
        if v.is_finite() {
            if v < min_elev {
                min_elev = v;
            }
            if v > max_elev {
                max_elev = v;
            }
        }
    }
    if !min_elev.is_finite() || !max_elev.is_finite() {
        return Ok(fill_time);
    }

    let elev_span = (max_elev - min_elev).max(0.0);
    let est_slices = (elev_span / step).ceil();
    if !est_slices.is_finite() || est_slices > MAX_SLICES_PER_REGION as f64 {
        return Err(Error::InvalidArgument(format!(
            "interval={step} requires {est_slices} slices (exceeds MAX_SLICES_PER_REGION={MAX_SLICES_PER_REGION})"
        )));
    }

    let mut slice_idx = 0usize;
    loop {
        let elev = max_elev - (slice_idx as f64) * step;
        if slice_idx > 0 && elev <= min_elev {
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

        let objects = region_props_xy(
            &label_objects,
            nb_labels,
            &img,
            bw,
            bh,
            res_x,
            res_y,
            cell_area,
        );
        for obj in &objects {
            if obj.max_depth < min_depth {
                continue;
            }
            let (fr, fc) = obj.first_pixel;
            let raw_cat = task.roi_catchment[fr * bw + fc];
            let cat_area = if raw_cat.is_finite() && raw_cat > 0.0 {
                raw_cat
            } else {
                obj.area
            };
            if cat_area <= 0.0 {
                continue;
            }

            let (or0, oc0, or1, oc1) = obj.bbox;
            // Compute remaining unfilled volume (deducting antecedent water) and mean C / infiltration.
            let mut rem_vol = obj.volume;
            let mut c_sum = 0.0f64;
            let mut c_cnt = 0usize;
            let mut inf_sum = 0.0f64;
            let mut inf_cnt = 0usize;
            if task.roi_antecedent.is_some()
                || task.roi_runoff_coeff.is_some()
                || task.roi_infil_m_h.is_some()
            {
                let mut ant_vol = 0.0f64;
                for r in or0..or1 {
                    let off = r * bw;
                    for c in oc0..oc1 {
                        if label_objects[off + c] != obj.label {
                            continue;
                        }
                        if let Some(ref ant) = task.roi_antecedent {
                            let h0 = ant[off + c].max(0.0);
                            if h0 > 0.0 {
                                let cell_depth = (obj.max_elev - img[off + c]).max(0.0);
                                ant_vol += h0.min(cell_depth) * cell_area;
                            }
                        }
                        if let Some(ref rc) = task.roi_runoff_coeff {
                            c_sum += rc[off + c].clamp(0.0, 1.0);
                            c_cnt += 1;
                        }
                        if let Some(ref inf) = task.roi_infil_m_h {
                            inf_sum += inf[off + c].max(0.0);
                            inf_cnt += 1;
                        }
                    }
                }
                rem_vol = (obj.volume - ant_vol).max(0.0);
            }
            let mean_c = if c_cnt > 0 { c_sum / c_cnt as f64 } else { 1.0 };
            let mean_inf_m_h = if inf_cnt > 0 {
                inf_sum / inf_cnt as f64
            } else {
                0.0
            };
            let eff_cat = cat_area * mean_c;

            let t_fill = if rem_vol <= 0.0 {
                0.0
            } else if let Some(hyeto) = hyetograph {
                invert_hyetograph_time(rem_vol, eff_cat, obj.area, mean_inf_m_h, hyeto)
            } else {
                let net_q = eff_cat * rain_m_h - obj.area * mean_inf_m_h;
                if net_q <= 0.0 {
                    f64::INFINITY
                } else {
                    rem_vol / net_q
                }
            };
            if !t_fill.is_finite() {
                continue;
            }
            for r in or0..or1 {
                let off = r * bw;
                for c in oc0..oc1 {
                    if label_objects[off + c] == obj.label && t_fill < fill_time[off + c] {
                        fill_time[off + c] = t_fill;
                    }
                }
            }
        }
    }

    Ok(fill_time)
}

/// Computes per-pixel inundation times in memory with optional spatially variable runoff
/// coefficients, vertical infiltration losses, antecedent ponded water depths, and a
/// time-varying storm hyetograph.
pub fn simulate_inundation_with_forcing_in_memory(
    sink: &Array2D<f64>,
    catchment_area: &Array2D<f64>,
    options: &SimulateInundationOptions,
    forcing: &InundationForcing<'_>,
) -> Result<(Array2D<f64>, usize)> {
    if forcing.runoff_coeff.is_none()
        && forcing.infiltration_cm_h.is_none()
        && forcing.antecedent_depth.is_none()
        && forcing.hyetograph.is_none()
    {
        return simulate_inundation_in_memory(sink, catchment_area, options);
    }
    let _ = validate_options(options)?;
    if let Some(hyeto) = forcing.hyetograph {
        if hyeto.is_empty() {
            return Err(Error::InvalidArgument(
                "hyetograph must not be empty when provided".into(),
            ));
        }
        for (idx, &(dt, intensity)) in hyeto.iter().enumerate() {
            if !dt.is_finite() || dt <= 0.0 || !intensity.is_finite() || intensity < 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "invalid hyetograph step {idx}: dt={dt}, intensity_cm_h={intensity}"
                )));
            }
        }
    }
    let w = sink.width();
    let h = sink.height();
    let sink_gt = sink.geotransform;
    let cell_x = sink_gt[1].hypot(sink_gt[2]).max(1e-9);
    let cell_y = sink_gt[4].hypot(sink_gt[5]).max(1e-9);
    let sink_srs = if sink.projection.is_empty() {
        None
    } else {
        terrio::crs::SpatialRef::parse(&sink.projection)
            .ok()
            .filter(|s| !s.is_unknown())
    };
    let check_grid = |arr: &Array2D<f64>, name: &str| -> Result<()> {
        if arr.width() != w || arr.height() != h {
            return Err(Error::DimensionMismatch(format!(
                "{name} ({}x{}) does not match sink ({w}x{h})",
                arr.width(),
                arr.height()
            )));
        }
        for (k, (&a, &b)) in arr.geotransform.iter().zip(sink_gt.iter()).enumerate() {
            let tol = match k {
                0 => 1e-3 * cell_x,
                3 => 1e-3 * cell_y,
                _ => 1e-6 * a.abs().max(b.abs()).max(1e-12),
            };
            if (a - b).abs() > tol {
                return Err(Error::DimensionMismatch(format!(
                    "{name} geotransform[{k}] ({a}) does not match sink ({b})"
                )));
            }
        }
        if let Some(ref s_srs) = sink_srs
            && !arr.projection.is_empty()
            && let Ok(a_srs) = terrio::crs::SpatialRef::parse(&arr.projection)
            && !a_srs.is_unknown()
            && !s_srs.same_as(&a_srs)
        {
            return Err(Error::InvalidArgument(format!(
                "{name} CRS ({}) does not match sink CRS ({})",
                a_srs.label(),
                s_srs.label()
            )));
        }
        Ok(())
    };
    check_grid(catchment_area, "catchment raster")?;
    if let Some(rc) = forcing.runoff_coeff {
        check_grid(rc, "runoff_coeff")?;
    }
    if let Some(inf) = forcing.infiltration_cm_h {
        check_grid(inf, "infiltration_cm_h")?;
    }
    if let Some(ant) = forcing.antecedent_depth {
        check_grid(ant, "antecedent_depth")?;
    }

    let metrics = affine_metrics(&sink.geotransform);
    let nd = sink.no_data();
    let c_nd = catchment_area.no_data();

    let (labels, nb_regions_u32) = region_group(sink.data(), w, h, options.min_size, |v| {
        v.is_finite() && !v.rd_eq(nd)
    });
    let nb_regions = nb_regions_u32 as usize;

    let mut fill_time_data = vec![-1.0f64; w * h];
    if nb_regions == 0 {
        let mut out = Array2D::from_vec(w, h, fill_time_data)?;
        out.copy_metadata_from(sink);
        out.set_no_data(-1.0);
        return Ok((out, 0));
    }

    let mut bboxes = vec![(usize::MAX, usize::MAX, 0usize, 0usize); nb_regions];
    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let lbl = labels[off + c];
            if lbl > 0 {
                let b = &mut bboxes[(lbl - 1) as usize];
                b.0 = b.0.min(r);
                b.1 = b.1.min(c);
                b.2 = b.2.max(r + 1);
                b.3 = b.3.max(c + 1);
            }
        }
    }

    let sdata = sink.data();
    let cdata = catchment_area.data();
    let tasks: Vec<ForcedRegionFillTask> = (0..nb_regions)
        .map(|ri| {
            let lbl = (ri + 1) as u32;
            let bbox = bboxes[ri];
            let (r0, c0, r1, c1) = bbox;
            let bh = r1 - r0;
            let bw = c1 - c0;
            let mut roi_elev = vec![f64::NAN; bw * bh];
            let mut roi_catchment = vec![0.0f64; bw * bh];
            let mut roi_rc = forcing.runoff_coeff.map(|_| vec![1.0f64; bw * bh]);
            let mut roi_inf = forcing.infiltration_cm_h.map(|_| vec![0.0f64; bw * bh]);
            let mut roi_ant = forcing.antecedent_depth.map(|_| vec![0.0f64; bw * bh]);
            for r in r0..r1 {
                let g_off = r * w;
                let l_off = (r - r0) * bw;
                for c in c0..c1 {
                    if labels[g_off + c] == lbl {
                        let gi = g_off + c;
                        let li = l_off + (c - c0);
                        roi_elev[li] = sdata[gi];
                        let cv = cdata[gi];
                        roi_catchment[li] = if cv.is_finite() && !cv.rd_eq(c_nd) && cv > 0.0 {
                            cv
                        } else {
                            0.0
                        };
                        if let (Some(dst), Some(src)) = (&mut roi_rc, forcing.runoff_coeff) {
                            let v = src.get_i(gi);
                            if v.is_finite() && !v.rd_eq(src.no_data()) {
                                dst[li] = v.clamp(0.0, 1.0);
                            }
                        }
                        if let (Some(dst), Some(src)) = (&mut roi_inf, forcing.infiltration_cm_h) {
                            let v = src.get_i(gi);
                            if v.is_finite() && !v.rd_eq(src.no_data()) && v > 0.0 {
                                dst[li] = v / 100.0; // cm/h -> m/h
                            }
                        }
                        if let (Some(dst), Some(src)) = (&mut roi_ant, forcing.antecedent_depth) {
                            let v = src.get_i(gi);
                            if v.is_finite() && !v.rd_eq(src.no_data()) && v > 0.0 {
                                dst[li] = v;
                            }
                        }
                    }
                }
            }
            ForcedRegionFillTask {
                bbox,
                roi_elev,
                roi_catchment,
                roi_runoff_coeff: roi_rc,
                roi_infil_m_h: roi_inf,
                roi_antecedent: roi_ant,
            }
        })
        .collect();

    #[cfg(feature = "parallel")]
    let region_results: Result<Vec<Vec<f64>>> = tasks
        .par_iter()
        .map(|t| region_fill_times_forced(t, options, forcing.hyetograph, metrics))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let region_results: Result<Vec<Vec<f64>>> = tasks
        .iter()
        .map(|t| region_fill_times_forced(t, options, forcing.hyetograph, metrics))
        .collect();

    for (task, times) in tasks.iter().zip(region_results?) {
        let (r0, c0, r1, c1) = task.bbox;
        let bw = c1 - c0;
        for r in r0..r1 {
            let g_off = r * w;
            let l_off = (r - r0) * bw;
            for c in c0..c1 {
                let t = times[l_off + (c - c0)];
                if t.is_finite() {
                    fill_time_data[g_off + c] = t;
                }
            }
        }
    }

    let mut out = Array2D::from_vec(w, h, fill_time_data)?;
    out.copy_metadata_from(sink);
    out.set_no_data(-1.0);
    Ok((out, nb_regions))
}

/// Options for [`run_full_pipeline_in_memory`] (shared core orchestration for Rust, Python, and WASM).
pub struct InMemoryFullPipelineOptions<'a> {
    /// Median filter kernel size (`<= 1` skips smoothing).
    pub kernel_size: usize,
    /// Optional selective/constrained depression breaching applied prior to smoothing and sink extraction.
    pub breach: Option<BreachOptions>,
    /// Level-set and inundation simulation parameters.
    pub simulation: SimulateInundationOptions,
    /// Hydrologic forcing (runoff coefficient raster, infiltration raster, antecedent depth, hyetograph).
    pub forcing: InundationForcing<'a>,
    /// Effective scalar runoff multiplier used for reporting cumulative runoff (mm) when no spatially
    /// variable raster is used (default `1.0`).
    pub scalar_runoff_coeff: f64,
    /// Optional stream polyline vector for computing spill-to-stream distances on the connectivity graph.
    pub stream_vector: Option<&'a Path>,
}

/// Complete in-memory outputs from [`run_full_pipeline_in_memory`].
pub struct InMemoryFullPipeline {
    pub sinks: InMemorySinks,
    pub depressions: InMemoryDepressions,
    pub catchment_hierarchy: Array2D<f64>,
    pub catchment_records: Vec<CatchmentHierarchyRecord>,
    pub fill_time: Array2D<f64>,
    pub num_regions: usize,
    pub steps: Vec<InundationStepSummary>,
    pub spill_paths: Vec<SpillFlowPath>,
    pub rating_curves: Vec<crate::graph::RatingCurvePoint>,
    pub connectivity_graph: crate::graph::ConnectivityGraph,
    pub inundation_curve_csv: String,
}

/// Runs the complete in-memory hydrology pipeline (optional breaching -> optional median filter ->
/// sink extraction -> nested depression hierarchy -> catchment hierarchy -> forced inundation ->
/// spill flow paths -> rating curves -> connectivity graph) without touching the filesystem.
pub fn run_full_pipeline_in_memory(
    raw_dem: &Array2D<f64>,
    options: &InMemoryFullPipelineOptions<'_>,
) -> Result<InMemoryFullPipeline> {
    let num_steps = validate_options(&options.simulation)?;
    let sim_opts = &options.simulation;

    let mut dem = raw_dem.clone();
    if options.kernel_size > 1 {
        dem = median_filter(&dem, options.kernel_size)?;
    }
    if let Some(ref breach_opts) = options.breach {
        dem = breach_depressions(&dem, breach_opts)?;
    }

    let sinks = extract_sinks_in_memory(&dem, sim_opts.min_size, Topology::D8)?;
    let depressions = delineate_depressions_in_memory(
        &sinks.sink,
        sim_opts.min_size,
        sim_opts.min_depth,
        sim_opts.interval,
    )?;
    let (catchment_hierarchy, catchment_records) = delineate_catchment_hierarchy_in_memory(
        &sinks.dem_refined,
        &depressions.depression_level,
        sim_opts.min_size,
    )?;

    let (fill_time, num_regions) = simulate_inundation_with_forcing_in_memory(
        &sinks.sink,
        &catchment_hierarchy,
        sim_opts,
        &options.forcing,
    )?;
    let steps =
        compute_inundation_step_summaries(&sinks.sink, &fill_time, num_steps, sim_opts.time_step_h);
    let spill_paths =
        delineate_spill_flow_paths_in_memory(&sinks.dem_refined, &depressions, &fill_time)?;
    let rating_curves = compute_rating_curves(&sinks.sink, &depressions, sim_opts.interval.abs())?;

    // Scale hyetograph / constant rainfall intensity by scalar_runoff_coeff for reported runoff depth.
    let eff_rain = sim_opts.rain_intensity_cm_h * options.scalar_runoff_coeff;
    let scaled_hyeto: Option<Vec<(f64, f64)>> = options.forcing.hyetograph.map(|h| {
        h.iter()
            .map(|&(dt, cm_h)| (dt, cm_h * options.scalar_runoff_coeff))
            .collect()
    });
    let connectivity_graph = build_connectivity_graph(
        &depressions,
        &catchment_records,
        &fill_time,
        &spill_paths,
        eff_rain,
        scaled_hyeto.as_deref(),
        options.stream_vector,
    )?;
    let inundation_curve_csv =
        format_inundation_curve_csv(&steps, eff_rain, scaled_hyeto.as_deref());

    Ok(InMemoryFullPipeline {
        sinks,
        depressions,
        catchment_hierarchy,
        catchment_records,
        fill_time,
        num_regions,
        steps,
        spill_paths,
        rating_curves,
        connectivity_graph,
        inundation_curve_csv,
    })
}

/// Options for the all-in-one end-to-end [`run_pipeline`] workflow.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RunPipelineOptions {
    /// Median filter kernel size (`<= 1` skips smoothing; default `3`).
    pub kernel_size: usize,
    /// Optional selective depression breaching applied prior to sink extraction.
    pub breach: Option<BreachOptions>,
    /// Optional culvert vector path burned into the DEM prior to sink extraction.
    pub culverts_vector: Option<PathBuf>,
    /// Culvert burning parameters used when `culverts_vector` is `Some`.
    pub burn_culverts: BurnCulvertsOptions,
    /// Level-set and inundation simulation parameters.
    pub simulation: SimulateInundationOptions,
    /// Optional spatially variable runoff-coefficient raster path (`0.0..=1.0`).
    pub runoff_coeff_path: Option<PathBuf>,
    /// Optional spatially variable infiltration loss raster path (`cm/h`).
    pub infiltration_path: Option<PathBuf>,
    /// Optional antecedent water-depth raster path (vertical DEM units).
    pub antecedent_depth_path: Option<PathBuf>,
    /// Optional piecewise-constant storm hyetograph `[(dt_h, intensity_cm_h)]`.
    pub hyetograph: Option<Vec<(f64, f64)>>,
    /// Optional stream/ditch polyline vector path for computing spill-to-stream distances.
    pub stream_vector: Option<PathBuf>,
    /// Load source DEM as `f32` before widening to reduce peak I/O memory (default `false`).
    pub use_f32: bool,
    /// Save raster GeoTIFF outputs (`depth.tif`, `depression_id.tif`, `depression_level.tif`, `catchment_hir.tif`, `fill_time.tif`).
    pub save_rasters: bool,
    /// Export unified `depressions.gpkg` and `spill_paths.gpkg` vector layers (default `true`).
    pub export_vectors: bool,
    /// Export `rating_curves.csv`, `connectivity_graph.graphml`, and (when compiled with
    /// the `serde` feature) `connectivity_graph.json` (default `true`). Note: `summary.json`
    /// also serializes the full [`RunPipelineResult`] under `serde`, and a minimal numeric
    /// JSON object when built without `serde`.
    pub export_graph: bool,
}

impl Default for RunPipelineOptions {
    fn default() -> Self {
        Self {
            kernel_size: 3,
            breach: None,
            culverts_vector: None,
            burn_culverts: BurnCulvertsOptions::default(),
            simulation: SimulateInundationOptions::default(),
            runoff_coeff_path: None,
            infiltration_path: None,
            antecedent_depth_path: None,
            hyetograph: None,
            stream_vector: None,
            use_f32: false,
            save_rasters: true,
            export_vectors: true,
            export_graph: true,
        }
    }
}

/// Output summary and generated file paths from [`run_pipeline`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RunPipelineResult {
    pub out_dir: PathBuf,
    pub width: usize,
    pub height: usize,
    pub num_sink_regions: usize,
    pub num_depressions: usize,
    pub max_level: u32,
    pub total_storage_m3: f64,
    pub num_spill_paths: usize,
    pub depressions_csv: PathBuf,
    pub catchments_csv: PathBuf,
    pub inundation_csv: PathBuf,
    pub summary_json: PathBuf,
    pub depressions_gpkg: Option<PathBuf>,
    pub spill_paths_gpkg: Option<PathBuf>,
    pub rating_curves_csv: Option<PathBuf>,
    pub connectivity_graph_json: Option<PathBuf>,
    pub connectivity_graphml: Option<PathBuf>,
}

/// Runs the complete `lidar-rs` workflow end to end in memory from a single DEM file
/// with zero intermediate GeoTIFF re-reads, writing structured CSV, JSON, GraphML,
/// GeoPackage, and optional compressed GeoTIFF outputs into `out_dir`.
pub fn run_pipeline(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &RunPipelineOptions,
) -> Result<RunPipelineResult> {
    let in_dem = in_dem.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let num_steps = validate_options(&options.simulation)?;
    let sim_opts = &options.simulation;

    // Guard against output paths resolving to the input DEM before writing anything.
    let depressions_csv = out_dir.join("depressions.csv");
    let catchments_csv = out_dir.join("catchments.csv");
    let inundation_csv = out_dir.join("inundation_curve.csv");
    let summary_json = out_dir.join("summary.json");
    let mut candidate_outputs = vec![
        depressions_csv.clone(),
        catchments_csv.clone(),
        inundation_csv.clone(),
        summary_json.clone(),
    ];
    if options.save_rasters {
        for name in [
            "depth.tif",
            "depression_id.tif",
            "depression_level.tif",
            "catchment_hir.tif",
            "fill_time.tif",
        ] {
            candidate_outputs.push(out_dir.join(name));
        }
    }
    if options.export_vectors {
        candidate_outputs.push(out_dir.join("depressions.gpkg"));
        candidate_outputs.push(out_dir.join("spill_paths.gpkg"));
    }
    if options.export_graph {
        candidate_outputs.push(out_dir.join("rating_curves.csv"));
        candidate_outputs.push(out_dir.join("connectivity_graph.graphml"));
        candidate_outputs.push(out_dir.join("connectivity_graph.json"));
    }
    let candidate_refs: Vec<&Path> = candidate_outputs.iter().map(PathBuf::as_path).collect();
    crate::raster_ops::ensure_no_path_collision(in_dem, &candidate_refs)?;
    for opt_in in [
        options.culverts_vector.as_deref(),
        options.runoff_coeff_path.as_deref(),
        options.infiltration_path.as_deref(),
        options.antecedent_depth_path.as_deref(),
        options.stream_vector.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        crate::raster_ops::ensure_no_path_collision(opt_in, &candidate_refs)?;
    }

    let mut dem = if options.use_f32 {
        let d32 = load_raster::<f32>(in_dem, None)?;
        let nd32 = d32.no_data();
        let out_nd = if nd32.is_finite() {
            nd32 as f64
        } else {
            f64::NAN
        };
        let data: Vec<f64> = d32
            .data()
            .iter()
            .map(|&v| {
                if v.rd_eq(nd32) || !v.is_finite() {
                    out_nd
                } else {
                    v as f64
                }
            })
            .collect();
        let mut out = Array2D::from_vec(d32.width(), d32.height(), data)?;
        out.copy_metadata_from(&d32);
        out.set_no_data(out_nd);
        out
    } else {
        load_raster::<f64>(in_dem, None)?
    };

    // Smooth raw DEM noise FIRST, then apply culvert burning and selective breaching so a
    // median filter never smooths away a 1-cell culvert cut or breach channel.
    if options.kernel_size > 1 {
        dem = median_filter(&dem, options.kernel_size)?;
    }
    if let Some(ref culv) = options.culverts_vector {
        let (burned, _) = burn_culverts(&dem, culv, &options.burn_culverts)?;
        dem = burned;
    }
    if let Some(ref breach_opts) = options.breach {
        dem = breach_depressions(&dem, breach_opts)?;
    }

    let sinks = extract_sinks_in_memory(&dem, sim_opts.min_size, Topology::D8)?;
    let depressions = delineate_depressions_in_memory(
        &sinks.sink,
        sim_opts.min_size,
        sim_opts.min_depth,
        sim_opts.interval,
    )?;
    let (catchment_hierarchy, catchment_records) = delineate_catchment_hierarchy_in_memory(
        &sinks.dem_refined,
        &depressions.depression_level,
        sim_opts.min_size,
    )?;

    let rc_arr = match &options.runoff_coeff_path {
        Some(p) => Some(load_raster::<f64>(p, None)?),
        None => None,
    };
    let inf_arr = match &options.infiltration_path {
        Some(p) => Some(load_raster::<f64>(p, None)?),
        None => None,
    };
    let ant_arr = match &options.antecedent_depth_path {
        Some(p) => Some(load_raster::<f64>(p, None)?),
        None => None,
    };
    let forcing = InundationForcing {
        runoff_coeff: rc_arr.as_ref(),
        infiltration_cm_h: inf_arr.as_ref(),
        antecedent_depth: ant_arr.as_ref(),
        hyetograph: options.hyetograph.as_deref(),
    };
    let (fill_time, _num_sim_regions) = simulate_inundation_with_forcing_in_memory(
        &sinks.sink,
        &catchment_hierarchy,
        sim_opts,
        &forcing,
    )?;
    let steps =
        compute_inundation_step_summaries(&sinks.sink, &fill_time, num_steps, sim_opts.time_step_h);
    let spill_paths =
        delineate_spill_flow_paths_in_memory(&sinks.dem_refined, &depressions, &fill_time)?;

    let (_, _, cell_area) = affine_metrics(&dem.geotransform);
    let d_nd = sinks.depth.no_data();
    let total_storage_m3: f64 = sinks
        .depth
        .data()
        .iter()
        .filter(|&&v| v.is_finite() && !v.rd_eq(d_nd) && v > 0.0)
        .map(|&v| v * cell_area)
        .sum();

    write_depressions_csv(&depressions.depressions, &depressions_csv)?;
    write_catchment_hierarchy_csv(&catchment_records, &catchments_csv)?;
    std::fs::write(
        &inundation_csv,
        format_inundation_curve_csv(
            &steps,
            sim_opts.rain_intensity_cm_h,
            options.hyetograph.as_deref(),
        ),
    )?;

    if options.save_rasters {
        let comp = sim_opts.compress;
        save_raster(&sinks.depth, out_dir.join("depth.tif"), None, comp)?;
        save_raster(
            &depressions.depression_id,
            out_dir.join("depression_id.tif"),
            None,
            comp,
        )?;
        save_raster(
            &depressions.depression_level,
            out_dir.join("depression_level.tif"),
            None,
            comp,
        )?;
        save_raster(
            &catchment_hierarchy,
            out_dir.join("catchment_hir.tif"),
            None,
            comp,
        )?;
        save_raster(&fill_time, out_dir.join("fill_time.tif"), None, comp)?;
    }

    let (depressions_gpkg, spill_paths_gpkg) = if options.export_vectors {
        let dg = out_dir.join("depressions.gpkg");
        let sg = out_dir.join("spill_paths.gpkg");
        write_depressions_vector(&depressions.depression_id, &depressions.depressions, &dg)?;
        write_spill_paths_vector(
            &spill_paths,
            &dem.geotransform,
            &dem.projection,
            sim_opts.rain_intensity_cm_h,
            options.hyetograph.as_deref(),
            &sg,
        )?;
        (Some(dg), Some(sg))
    } else {
        (None, None)
    };

    let (rating_curves_csv, connectivity_graph_json, connectivity_graphml) = if options.export_graph
    {
        let rc_path = out_dir.join("rating_curves.csv");
        let curves = compute_rating_curves(&sinks.sink, &depressions, sim_opts.interval.abs())?;
        write_rating_curves_csv(&curves, &rc_path)?;

        let graph = build_connectivity_graph(
            &depressions,
            &catchment_records,
            &fill_time,
            &spill_paths,
            sim_opts.rain_intensity_cm_h,
            options.hyetograph.as_deref(),
            options.stream_vector.as_deref(),
        )?;
        let gml_path = out_dir.join("connectivity_graph.graphml");
        write_connectivity_graphml(&graph, &gml_path)?;

        #[cfg(feature = "serde")]
        let gj_path = {
            let p = out_dir.join("connectivity_graph.json");
            write_connectivity_graph_json(&graph, &p)?;
            Some(p)
        };
        #[cfg(not(feature = "serde"))]
        let gj_path = None;

        (Some(rc_path), gj_path, Some(gml_path))
    } else {
        (None, None, None)
    };

    let result = RunPipelineResult {
        out_dir: out_dir.to_path_buf(),
        width: dem.width(),
        height: dem.height(),
        num_sink_regions: sinks.depressions.len(),
        num_depressions: depressions.depressions.len(),
        max_level: depressions.max_level,
        total_storage_m3,
        num_spill_paths: spill_paths.len(),
        depressions_csv,
        catchments_csv,
        inundation_csv,
        summary_json: summary_json.clone(),
        depressions_gpkg,
        spill_paths_gpkg,
        rating_curves_csv,
        connectivity_graph_json,
        connectivity_graphml,
    };

    #[cfg(feature = "serde")]
    {
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| Error::Algorithm(format!("failed to serialize summary: {e}")))?;
        std::fs::write(&summary_json, json)?;
    }
    #[cfg(not(feature = "serde"))]
    {
        std::fs::write(
            &summary_json,
            format!(
                "{{\"width\":{},\"height\":{},\"num_depressions\":{},\"max_level\":{},\"total_storage_m3\":{:.4},\"num_spill_paths\":{}}}\n",
                result.width,
                result.height,
                result.num_depressions,
                result.max_level,
                result.total_storage_m3,
                result.num_spill_paths
            ),
        )?;
    }

    Ok(result)
}

/// Simulates surface inundation dynamics over time and writes `flood_*.tif`,
/// `fill_time.tif`, and `inundation_summary.csv` into `out_dir`.
///
/// Pure-Rust port of `6_Simulate Inundation.py`.
pub fn simulate_inundation(
    in_sink: impl AsRef<Path>,
    in_catchment: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &SimulateInundationOptions,
) -> Result<SimulateInundationResult> {
    let n_steps = validate_options(options)?;
    let in_sink = in_sink.as_ref();
    let in_catchment = in_catchment.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let fill_time_path = out_dir.join("fill_time.tif");
    let summary_csv_path = out_dir.join("inundation_summary.csv");

    let digits = (n_steps as f64).log10().floor() as usize + 1;
    let frame_paths: Vec<PathBuf> = (1..=n_steps)
        .map(|k| out_dir.join(format!("flood_{k:0digits$}.tif")))
        .collect();

    let mut protected: Vec<&Path> = Vec::with_capacity(frame_paths.len() + 2);
    protected.push(&fill_time_path);
    protected.push(&summary_csv_path);
    for p in &frame_paths {
        protected.push(p.as_path());
    }
    ensure_no_path_collision(in_sink, &protected)?;
    ensure_no_path_collision(in_catchment, &protected)?;

    let sink = load_raster::<f64>(in_sink, None)?;
    let catchment = load_raster::<f64>(in_catchment, None)?;
    let (fill_time, num_regions) = simulate_inundation_in_memory(&sink, &catchment, options)?;

    // Remove stale flood_*.tif(f) frames from any prior run in out_dir so rerun outputs
    // with a different step count cannot interleave with old frames.
    if let Ok(entries) = std::fs::read_dir(out_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file()
                && let Some(name) = p.file_name().and_then(|n| n.to_str())
            {
                let lower = name.to_ascii_lowercase();
                if lower.starts_with("flood_")
                    && (lower.ends_with(".tif") || lower.ends_with(".tiff"))
                {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }

    save_raster(
        &fill_time,
        &fill_time_path,
        Some("simulate_inundation: fill_time_hours"),
        options.compress,
    )?;

    let w = sink.width();
    let h = sink.height();
    let (_, _, cell_area) = affine_metrics(&sink.geotransform);
    let nd = sink.no_data();
    let total_sink_cells = sink
        .data()
        .iter()
        .filter(|&&v| v.is_finite() && !v.rd_eq(nd))
        .count();

    let ft_data = fill_time.data();
    let mut steps = Vec::with_capacity(n_steps);
    let mut flood_buf = vec![0u32; w * h];

    for (i, frame_path) in frame_paths.iter().enumerate() {
        let step_num = i + 1;
        let t_hours = (step_num as f64) * options.time_step_h;
        let mut inundated_cells = 0usize;
        for (dst, &ft) in flood_buf.iter_mut().zip(ft_data.iter()) {
            if ft >= 0.0 && ft.is_finite() && ft <= t_hours {
                *dst = 1;
                inundated_cells += 1;
            } else {
                *dst = 0;
            }
        }
        let mut frame_arr = Array2D::from_vec(w, h, flood_buf.clone())?;
        frame_arr.copy_metadata_from(&sink);
        frame_arr.set_no_data(0);
        save_raster(
            &frame_arr,
            frame_path,
            Some(&format!("simulate_inundation(t={t_hours:.2}h)")),
            options.compress,
        )?;

        let inundated_area = inundated_cells as f64 * cell_area;
        let inundated_fraction = if total_sink_cells > 0 {
            inundated_cells as f64 / total_sink_cells as f64
        } else {
            0.0
        };
        steps.push(InundationStep {
            step: step_num,
            time_hours: t_hours,
            inundated_cells,
            inundated_area,
            inundated_fraction,
            raster_path: frame_path.clone(),
        });
    }

    write_inundation_summary_csv(&steps, &summary_csv_path)?;

    Ok(SimulateInundationResult {
        frame_paths,
        fill_time_path,
        summary_csv_path,
        num_regions,
        steps,
    })
}

/// Writes `inundation_summary.csv`.
pub fn write_inundation_summary_csv(
    steps: &[InundationStep],
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "step,time_hours,inundated_cells,inundated_area,inundated_fraction"
    )?;
    for s in steps {
        writeln!(
            w,
            "{},{:.4},{},{:.4},{:.6}",
            s.step, s.time_hours, s.inundated_cells, s.inundated_area, s.inundated_fraction
        )?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_bowl_inundates_monotonically_from_bottom_up() {
        // 10x10 sink with a deep inner 4x4 core (elev 10.0) inside a shallower rim (elev 12.0).
        let mut sink_data = vec![12.0; 100];
        for r in 3..7 {
            for c in 3..7 {
                sink_data[r * 10 + c] = 10.0;
            }
        }
        let mut sink = Array2D::from_vec(10, 10, sink_data).unwrap();
        sink.set_no_data(0.0);
        let mut cat = Array2D::from_vec(10, 10, vec![200.0; 100]).unwrap();
        cat.set_no_data(0.0);

        let opts = SimulateInundationOptions {
            min_size: 4,
            min_depth: 0.0,
            interval: 0.5,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 20.0,
            time_step_h: 2.0,
            compress: false,
        };
        let (ft, n_reg) = simulate_inundation_in_memory(&sink, &cat, &opts).unwrap();
        assert_eq!(n_reg, 1);
        let core_t = ft.get(4, 4);
        let rim_t = ft.get(0, 0);
        assert!(core_t >= 0.0 && rim_t >= 0.0);
        assert!(
            core_t <= rim_t,
            "inner core ({core_t}h) should inundate no later than outer rim ({rim_t}h)"
        );
    }

    #[test]
    fn spill_path_crosses_sub_threshold_micropit_into_downstream_depression() {
        // 11x25 terrain inside a compound parent sink (id=1, level=2) containing two
        // sibling leaf basins:
        //   - upstream basin A (id=2, level=1) at cols 2..6, rows 3..8 (floor 10.0)
        //   - downstream basin B (id=3, level=1) at cols 18..22, rows 3..8 (floor 8.0)
        // Separated by a saddle containing a small sub-threshold micro-pit at (col=11, row=5, z=11.4)
        // between a 11.6 rim on A's side and a 11.55 bump on B's side. The spill path from A
        // must cross the micro-pit and terminate inside downstream basin B (id=3), never stopping
        // at the parent ring (id=1).
        let w = 25usize;
        let h = 11usize;
        let mut z = vec![20.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];
        let mut ft = vec![-1.0f64; w * h];

        for r in 1..10 {
            for c in 1..24 {
                z[r * w + c] = 14.0;
                dep_id[r * w + c] = 1;
                dep_lvl[r * w + c] = 2;
                ft[r * w + c] = 10.0;
            }
        }
        // Saddle corridor along row 5 from col 6..18 with a micro-pit at col 11.
        let corridor_z = [
            (6, 11.55),
            (7, 11.60),
            (8, 11.58),
            (9, 11.52),
            (10, 11.46),
            (11, 11.40), // local micro-pit inside parent ring (id=1)
            (12, 11.48),
            (13, 11.55), // low micro-ridge separating pit from B
            (14, 11.30),
            (15, 10.80),
            (16, 10.20),
            (17, 9.60),
        ];
        for (c, elev) in corridor_z {
            z[5 * w + c] = elev;
        }
        // Basin A (id=2, fills early at 2.0h)
        for r in 3..8 {
            for c in 2..6 {
                let d = ((r as isize - 5).abs() + (c as isize - 4).abs()) as f64;
                z[r * w + c] = 10.0 + 0.3 * d;
                dep_id[r * w + c] = 2;
                dep_lvl[r * w + c] = 1;
                ft[r * w + c] = 2.0;
            }
        }
        // Basin B (id=3, deeper, fills later at 5.0h)
        for r in 3..8 {
            for c in 18..22 {
                let d = ((r as isize - 5).abs() + (c as isize - 20).abs()) as f64;
                z[r * w + c] = 8.0 + 0.3 * d;
                dep_id[r * w + c] = 3;
                dep_lvl[r * w + c] = 1;
                ft[r * w + c] = 5.0;
            }
        }

        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
        id_arr.set_no_data(0);
        let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
        lvl_arr.set_no_data(0);
        let mut ft_arr = Array2D::from_vec(w, h, ft).unwrap();
        ft_arr.set_no_data(-1.0);

        let dummy_dep = |id: u32, level: u32, min_elev: f64, children_ids: Vec<u32>| {
            crate::slicing::NestedDepression {
                id,
                level,
                count: 20,
                area: 20.0,
                volume: 20.0,
                avg_depth: 1.0,
                max_depth: 2.0,
                min_elev,
                max_elev: min_elev + 2.0,
                children_ids,
                region_id: 1,
                perimeter: 16.0,
                major_axis: 5.0,
                minor_axis: 5.0,
                elongatedness: 1.0,
                eccentricity: 0.0,
                orientation: 0.0,
                area_bbox_ratio: 1.0,
                parent_id: 0,
                circularity: 1.0,
                depth_diam_ratio: 0.2,
                hypsometric_integral: 0.5,
                rim_slope_deg: 20.0,
                artifact_flag: false,
            }
        };
        let deps = InMemoryDepressions {
            depression_id: id_arr,
            depression_level: lvl_arr,
            depressions: vec![
                dummy_dep(1, 2, 8.0, vec![2, 3]),
                dummy_dep(2, 1, 10.0, vec![]),
                dummy_dep(3, 1, 8.0, vec![]),
            ],
            num_regions: 1,
            max_level: 2,
        };

        let paths = delineate_spill_flow_paths_in_memory(&dem, &deps, &ft_arr).unwrap();
        assert_eq!(paths.len(), 1, "expected 1 deduplicated spill path A -> B");
        let p = &paths[0];
        assert_eq!(p.from_dep_id, 2);
        assert_eq!(
            p.to_dep_id, 3,
            "spill path must cross the micro-pit and connect to downstream basin 3, not parent 1"
        );
        let &(end_c, end_r) = p.cells.last().unwrap();
        assert_eq!(
            deps.depression_id.get(end_c, end_r),
            3,
            "path endpoint ({end_c}, {end_r}) must lie inside downstream basin 3"
        );
    }

    #[test]
    fn trace_rejects_ancestor_micropit_even_after_visiting_downstream_basin() {
        // 1x4 strip: cell 0 (source) -> cell 1 (downstream basin B=3) -> cell 2 (parent ring=1)
        // -> cell 3 (NO_FLOW pit in parent ring=1).
        // D8 code 5 moves east (+1 col, 0 row).
        let w = 4usize;
        let h = 1usize;
        let flowdirs = [NO_FLOW as u8, 5u8, 5u8, NO_FLOW as u8];
        let dep_ids = [2i32, 3, 1, 1];
        let in_source = [true, false, false, false];
        let desc_ids: HashSet<u32> = [2].into_iter().collect();
        let ancestor_ids: HashSet<u32> = [1].into_iter().collect();
        let mut drains_to_source = vec![0u32; w * h];
        let mut trapped_in_pit = vec![0u32; w * h];
        let mut seen_stamp = vec![0u32; w * h];

        let res = trace_downstream_away_from_source(
            1,
            w,
            h,
            &flowdirs,
            &dep_ids,
            &in_source,
            &desc_ids,
            &ancestor_ids,
            &mut drains_to_source,
            &mut trapped_in_pit,
            1,
            &mut seen_stamp,
            1,
        );
        assert!(
            res.is_none(),
            "trace terminating in an ancestor-ring NO_FLOW pit must be rejected even after visiting basin 3"
        );
        assert_eq!(trapped_in_pit[3], 1);
    }

    #[test]
    fn spill_path_omitted_when_no_downstream_outlet_is_reachable() {
        // Isolated child basin (id=2) inside parent ring (id=1) whose only neighbor outside
        // the child is a micro-pit inside parent ring (id=1), surrounded completely by NoData
        // so neither a distinct downstream depression nor the DEM edge is reachable.
        let w = 9usize;
        let h = 7usize;
        let nd = -9999.0f64;
        let mut z = vec![nd; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];
        let mut ft = vec![-1.0f64; w * h];

        for r in 1..6 {
            for c in 1..8 {
                z[r * w + c] = 14.0;
                dep_id[r * w + c] = 1;
                dep_lvl[r * w + c] = 2;
                ft[r * w + c] = 10.0;
            }
        }
        // Child basin (id=2) at cols 2..4, rows 2..5
        for r in 2..5 {
            for c in 2..4 {
                z[r * w + c] = 10.0;
                dep_id[r * w + c] = 2;
                dep_lvl[r * w + c] = 1;
                ft[r * w + c] = 2.0;
            }
        }
        // Micro-pit inside parent ring (id=1) at (col=6, row=3)
        z[3 * w + 4] = 12.0;
        z[3 * w + 5] = 11.5;
        z[3 * w + 6] = 11.0;

        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(nd);
        let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
        id_arr.set_no_data(0);
        let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
        lvl_arr.set_no_data(0);
        let mut ft_arr = Array2D::from_vec(w, h, ft).unwrap();
        ft_arr.set_no_data(-1.0);

        let dummy_dep = |id: u32, level: u32, min_elev: f64, children_ids: Vec<u32>| {
            crate::slicing::NestedDepression {
                id,
                level,
                count: 6,
                area: 6.0,
                volume: 6.0,
                avg_depth: 1.0,
                max_depth: 2.0,
                min_elev,
                max_elev: min_elev + 2.0,
                children_ids,
                region_id: 1,
                perimeter: 10.0,
                major_axis: 3.0,
                minor_axis: 2.0,
                elongatedness: 1.5,
                eccentricity: 0.0,
                orientation: 0.0,
                area_bbox_ratio: 1.0,
                parent_id: 0,
                circularity: 1.0,
                depth_diam_ratio: 0.2,
                hypsometric_integral: 0.5,
                rim_slope_deg: 20.0,
                artifact_flag: false,
            }
        };
        let deps = InMemoryDepressions {
            depression_id: id_arr,
            depression_level: lvl_arr,
            depressions: vec![
                dummy_dep(1, 2, 10.0, vec![2]),
                dummy_dep(2, 1, 10.0, vec![]),
            ],
            num_regions: 1,
            max_level: 2,
        };

        let paths = delineate_spill_flow_paths_in_memory(&dem, &deps, &ft_arr).unwrap();
        assert!(
            paths.is_empty(),
            "isolated basin with only ancestor-ring micro-pits and no reachable outlet should emit no spill path, got {paths:?}"
        );
    }

    #[test]
    fn trace_handles_boundary_no_flow_and_truncates_target_before_grid_exit() {
        // 3x5 grid:
        // Row 1 has interior cells (cols 1..4) and boundary cell (col 4, row 1).
        // Case 1: unlabelled interior NO_FLOW pit at (col=2, row=1) -> rejected (None).
        // Case 2: flow from (col=3, row=1) east to boundary NO_FLOW cell (col=4, row=1) -> Some((.., 0)).
        // Case 3: flow from (col=1, row=1, dep=3) east through unlabelled (col=2..4) and exiting
        //         grid -> Some((trace, 3)) truncated so endpoint lies inside dep 3.
        let w = 5usize;
        let h = 3usize;
        let mut flowdirs = vec![NO_FLOW as u8; w * h];
        let mut dep_ids = vec![0i32; w * h];
        let in_source = vec![false; w * h];
        let desc_ids: HashSet<u32> = [2].into_iter().collect();
        let ancestor_ids: HashSet<u32> = HashSet::new();
        let mut drains_to_source = vec![0u32; w * h];
        let mut trapped_in_pit = vec![0u32; w * h];
        let mut seen_stamp = vec![0u32; w * h];

        // Interior pit at (2, 1), boundary NO_FLOW at (4, 1), east flow at (3, 1) -> (4, 1)
        flowdirs[w + 3] = 5u8; // east -> (4, 1) which is NO_FLOW on boundary

        let interior_pit = trace_downstream_away_from_source(
            w + 2,
            w,
            h,
            &flowdirs,
            &dep_ids,
            &in_source,
            &desc_ids,
            &ancestor_ids,
            &mut drains_to_source,
            &mut trapped_in_pit,
            1,
            &mut seen_stamp,
            1,
        );
        assert!(
            interior_pit.is_none(),
            "unlabelled interior NO_FLOW micro-pit must be rejected"
        );

        let boundary_exit = trace_downstream_away_from_source(
            w + 3,
            w,
            h,
            &flowdirs,
            &dep_ids,
            &in_source,
            &desc_ids,
            &ancestor_ids,
            &mut drains_to_source,
            &mut trapped_in_pit,
            2,
            &mut seen_stamp,
            2,
        );
        let (b_trace, b_target) =
            boundary_exit.expect("boundary NO_FLOW cell must be accepted as a DEM-edge spill");
        assert_eq!(b_target, 0);
        assert_eq!(*b_trace.last().unwrap(), w + 4);

        // Now connect (1, 1, dep=3) -> (2, 1) -> (3, 1) -> (4, 1) -> off-grid east
        dep_ids[w + 1] = 3;
        flowdirs[w + 1] = 5u8;
        flowdirs[w + 2] = 5u8;
        flowdirs[w + 4] = 5u8;
        let trunc = trace_downstream_away_from_source(
            w + 1,
            w,
            h,
            &flowdirs,
            &dep_ids,
            &in_source,
            &desc_ids,
            &ancestor_ids,
            &mut drains_to_source,
            &mut trapped_in_pit,
            3,
            &mut seen_stamp,
            3,
        );
        let (t_trace, t_target) = trunc.expect("trace visiting target 3 should succeed");
        assert_eq!(t_target, 3);
        assert_eq!(
            dep_ids[*t_trace.last().unwrap()],
            3,
            "trace must be truncated to end inside target depression 3"
        );
    }

    #[test]
    fn scs_cn_preserves_small_positive_cn_values() {
        let mut cn = Array2D::from_vec(4, 1, vec![5e-7, 0.0, 100.0, -9999.0]).unwrap();
        cn.set_no_data(-9999.0);
        // For CN = 5e-7, S ~= 5.08e10 and Ia ~= 1.016e10 > 6e9 -> Q must be 0.0.
        let coeff = scs_curve_number_runoff(&cn, 6e9).unwrap();
        assert_eq!(coeff.data()[0], 0.0);
        assert_eq!(coeff.data()[1], 0.0);
        assert!((coeff.data()[2] - 1.0).abs() < 1e-12);
        assert_eq!(coeff.data()[3], coeff.no_data());
    }

    #[test]
    fn forced_inundation_rejects_one_pixel_shift_and_crs_mismatch() {
        let mut sink = Array2D::from_vec(4, 4, vec![100.0f64; 16]).unwrap();
        sink.set_no_data(-9999.0);
        sink.geotransform = [500_000.0, 1.0, 0.0, 5_000_000.0, 0.0, -1.0];
        sink.projection = "EPSG:26916".to_string();

        let mut cat = Array2D::from_vec(4, 4, vec![16.0f64; 16]).unwrap();
        cat.set_no_data(0.0);
        cat.geotransform = sink.geotransform;
        cat.projection = sink.projection.clone();

        let opts = SimulateInundationOptions::default();

        // 1-pixel origin shift at northing 5,000,000 must fail.
        let mut shifted_rc = Array2D::from_vec(4, 4, vec![0.5f64; 16]).unwrap();
        shifted_rc.geotransform = [500_000.0, 1.0, 0.0, 5_000_001.0, 0.0, -1.0];
        shifted_rc.projection = sink.projection.clone();
        let err_shift = simulate_inundation_with_forcing_in_memory(
            &sink,
            &cat,
            &opts,
            &InundationForcing {
                runoff_coeff: Some(&shifted_rc),
                ..Default::default()
            },
        );
        assert!(err_shift.is_err(), "1-pixel origin shift must be rejected");

        // CRS mismatch must fail.
        let mut wrong_crs_rc = Array2D::from_vec(4, 4, vec![0.5f64; 16]).unwrap();
        wrong_crs_rc.geotransform = sink.geotransform;
        wrong_crs_rc.projection = "EPSG:4326".to_string();
        let err_crs = simulate_inundation_with_forcing_in_memory(
            &sink,
            &cat,
            &opts,
            &InundationForcing {
                runoff_coeff: Some(&wrong_crs_rc),
                ..Default::default()
            },
        );
        assert!(err_crs.is_err(), "CRS mismatch must be rejected");
    }
}
