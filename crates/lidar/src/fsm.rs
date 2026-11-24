//! Dynamic fill-and-spill cascade mass-balance routing (Fill-Spill-Merge).
//!
//! Implements an event-driven, mass-conserving Fill-Spill-Merge (FSM) hydrologic
//! routing engine over nested surface depression hierarchies (inspired by Barnes et al.,
//! 2020/2021, and Wu et al., 2019).
//!
//! Unlike the static level-set simulation in [`crate::inundation::simulate_inundation_in_memory`],
//! where each depression fills independently using its static pre-spill catchment area:
//!
//! 1. **Single-pass catchment-weighted attribution (`O(N)`):** Every grid cell traces its
//!    resolved D8 flow path to its initial receiving depression (or watershed boundary),
//!    accumulating both geometric catchment area and catchment-weighted effective runoff
//!    area (`cell_area * runoff_coeff`) without double-counting between nested levels.
//! 2. **Dynamic spill cascades:** Once an upstream depression `A` reaches its spill capacity
//!    at `t_spill(A)`, its excess discharge `(Q_in - loss).max(0)` routes over the lowest
//!    saddle into downstream basin `B` (or the watershed boundary `0`), accelerating `B`'s
//!    inundation and updating effective contributing area dynamically.
//! 3. **Hierarchical merging:** When all child depressions of an enclosing compound parent
//!    `P` reach spill capacity, they merge into `P` at `t_merge`, transferring their stored
//!    water and contributing catchments to `P`, which then continues filling smoothly up to
//!    its own spill rim.
//! 4. **Continuous hypsometric cell timing & exact mass conservation:** Per-cell `fill_time`
//!    is interpolated continuously from each depression's exact sorted-elevation storage
//!    curve in `O(D log D)`, and every run reports the residual of exact mass conservation:
//!    `antecedent + runoff == stored + infiltration + boundary_outflow`.
//!
//! ## Scaling notes
//!
//! Outlet discovery checks out grid-sized scratch buffers (~25 bytes per cell) from a small pool
//! bounded by active worker concurrency and releases them immediately once outlets are found.
//! The event-driven routing loop performs `O(m)` passes per fill/merge event (`O(m^2)` worst case
//! for `m` depressions) and retains sorted hypsometric cell tables (`16` bytes per entry,
//! `O(N * max_level)` memory across `N` valid sink pixels).

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, D8X, D8Y, DemNum, NO_FLOW};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::catchment::d8_flow_directions_resolved;
use crate::error::{Error, Result};
use crate::inundation::{
    HeapCell, InundationForcing, SimulateInundationOptions, SpillFlowPath, affine_metrics,
    trace_downstream_away_from_source, validate_forcing_grids, validate_grid_alignment,
    validate_options_with_hyetograph,
};
use crate::slicing::{InMemoryDepressions, NestedDepression};

/// Time tolerance in hours for deciding whether an event falls within the storm duration.
const STORM_TIME_EPS_H: f64 = 1e-9;

/// Hydrologic inundation routing strategy for pipeline simulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RoutingMode {
    /// Classic Wu et al. (2019) static level-set inundation using pre-spill catchment areas.
    #[default]
    Static,
    /// Dynamic event-driven Fill-Spill-Merge cascade mass-balance routing.
    FillSpillMerge,
}

impl FromStr for RoutingMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "static" => Ok(Self::Static),
            "fsm" | "fill-spill-merge" | "fill_spill_merge" => Ok(Self::FillSpillMerge),
            other => Err(Error::InvalidArgument(format!(
                "unknown routing mode '{other}'; expected 'static' or 'fsm' ('fill-spill-merge')"
            ))),
        }
    }
}

/// Per-depression dynamic mass-balance and routing summary produced by
/// [`simulate_fill_spill_merge_in_memory`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FsmBasinSummary {
    /// 1-based depression ID.
    pub dep_id: u32,
    /// Direct enclosing parent depression ID (`0` for top-level roots).
    pub parent_id: u32,
    /// Hierarchy level (`1` = leaf).
    pub level: u32,
    /// Storage capacity below the depression spill rim in cubic map units.
    pub capacity_m3: f64,
    /// Local direct D8 contributing area prior to upstream spill cascades (map units squared).
    pub local_cat_area_m2: f64,
    /// Catchment-weighted effective local contributing area (`sum(cell_area * C_i)`).
    pub local_eff_cat_area_m2: f64,
    /// Total contributing area reaching this basin at the end of the storm duration,
    /// including merged children and active upstream spill cascades (map units squared).
    /// For basins merged into a parent before storm end, this falls back to their pre-merge
    /// `local_cat_area_m2`.
    pub final_routed_cat_area_m2: f64,
    /// Initial antecedent ponded water volume at `t = 0` (cubic map units).
    pub initial_volume_m3: f64,
    /// Stored water volume at the end of the simulated storm duration (cubic map units).
    /// For basins merged into a parent before storm end, this records their capacity at merge.
    pub final_volume_m3: f64,
    /// Time in hours when this depression first reached full spill capacity within the simulated
    /// storm duration (`None` if it never filled during the storm). Every basin that is full at
    /// storm end has a value; with variable rainfall or infiltration a basin can also fill and
    /// then drain below its rim, so a value does not guarantee `final_volume_m3 == capacity_m3`.
    /// With constant rainfall, `fill_time.tif` keeps extrapolated sustained-rainfall timing for
    /// basins that never filled.
    pub fill_time_h: Option<f64>,
    /// Downstream depression ID receiving spillover (`Some(0)` for DEM boundary, `None` if it did
    /// not spill within the simulated storm duration).
    pub spill_to_dep_id: Option<u32>,
    /// Time in hours when this depression merged into its parent (`None` if it had not merged by
    /// the end of the simulated storm duration).
    pub merge_time_h: Option<f64>,
    /// Merge time including the constant-rainfall extension past the storm, the timing convention
    /// shared by `fill_time.tif`, the spill flow paths, and the connectivity graph's edges. Equals
    /// `merge_time_h` for merges within the storm. Internal: used only to time graph `merge` edges.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) sustained_merge_time_h: Option<f64>,
    /// Cumulative excess volume spilled downstream over the simulated storm duration (cubic map units).
    pub spilled_volume_m3: f64,
    /// Peak spill discharge rate during the simulated storm duration (cubic map units per hour).
    pub peak_spill_rate_m3_h: f64,
    /// Cumulative vertical infiltration loss over the simulated storm duration (cubic map units).
    pub infiltrated_volume_m3: f64,
}

/// Watershed-scale mass-balance verification summary at the end of the simulated storm duration.
///
/// Conservation invariant:
/// `total_antecedent_m3 + total_runoff_m3 == total_stored_m3 + total_infiltration_m3 + total_boundary_outflow_m3`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FsmMassBalance {
    /// Evaluated storm duration in hours.
    pub duration_h: f64,
    /// Total initial antecedent ponded volume across all depressions at `t = 0`.
    pub total_antecedent_m3: f64,
    /// Total effective runoff generated across the valid DEM over `[0, duration_h]`.
    pub total_runoff_m3: f64,
    /// Total water volume stored in depressions at `t = duration_h`.
    pub total_stored_m3: f64,
    /// Total vertical infiltration loss across depressions over `[0, duration_h]`.
    pub total_infiltration_m3: f64,
    /// Total water exported across the DEM boundary (direct runoff + boundary spills) over `[0, duration_h]`.
    pub total_boundary_outflow_m3: f64,
    /// Residual `(antecedent + runoff) - (stored + infiltration + boundary_outflow)`.
    pub mass_balance_error_m3: f64,
}

/// Complete outputs of [`simulate_fill_spill_merge_in_memory`].
#[derive(Debug, Clone)]
pub struct FsmSimulationResult {
    /// Per-cell continuous time to inundation in hours (`-1.0` = dry / NoData).
    ///
    /// With constant rainfall, rainfall is extended past the storm duration, so cells in basins
    /// that are still filling at storm end get sustained-rainfall timing that can exceed
    /// `duration_h`. With a hyetograph, routing stops at the end of the hyetograph, and cells not
    /// yet inundated by then stay NoData (`-1.0`). Values are first-inundation times: a cell keeps
    /// its time even if its basin later drains.
    pub fill_time: Array2D<f64>,
    /// Dynamically realized overland spill flow paths with exact cascade `spill_time_h`
    /// and `merge_time_h`.
    pub spill_paths: Vec<SpillFlowPath>,
    /// Per-depression dynamic mass-balance and routing summary records.
    pub basins: Vec<FsmBasinSummary>,
    /// Watershed-scale mass-balance accounting at the end of the storm duration.
    pub mass_balance: FsmMassBalance,
    /// Number of top-level sink regions.
    pub num_regions: usize,
    /// Number of depressions that filled and initiated overland spilling. Like `spill_paths` and
    /// `fill_time`, this includes events in the constant-rainfall extension past the storm.
    pub num_spill_events: usize,
    /// Number of compound parent depressions formed by merging filled children. Includes events in
    /// the constant-rainfall extension past the storm, like `num_spill_events`.
    pub num_merge_events: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasinState {
    /// Compound parent awaiting all direct children to fill before activating.
    Waiting,
    /// Currently filling (or draining), `stored_vol < capacity`.
    Active,
    /// Full (`stored_vol == capacity`), routing excess discharge downstream.
    FullSpilling,
    /// Merged into its enclosing parent depression.
    Merged,
}

#[derive(Debug, Clone)]
struct TopographicOutlet {
    to_dep_id: u32,
    spill_elev: f64,
    cells: Vec<(usize, usize)>,
}

/// Sorted cell entry for exact hypsometric timing.
#[derive(Debug, Clone, Copy)]
struct HypsoCell {
    grid_idx: usize,
    /// Cumulative storage volume required for water surface to reach this cell's elevation.
    threshold_vol: f64,
}

#[derive(Debug)]
struct BasinNode {
    id: u32,
    parent_id: u32,
    level: u32,
    children_ids: Vec<u32>,
    min_elev: f64,
    capacity_m3: f64,
    /// Exact sorted cells across this basin + all descendants (for continuous stage-volume timing).
    hypso_cells: Vec<HypsoCell>,
    /// Cursor into `hypso_cells` of the next cell not yet inundated.
    hypso_cursor: usize,
    /// Highest volume ever reached by this basin (for monotonic `fill_time` updates).
    max_vol_reached: f64,
    /// Local direct catchment area (for leaves: D8 attributed; for merged parents: sum of children).
    local_cat_area_m2: f64,
    /// Local effective catchment area weighted by `runoff_coeff`.
    local_eff_cat_area_m2: f64,
    /// Total vertical infiltration loss rate over this basin's footprint (`m^3/h`).
    loss_rate_m3_h: f64,
    /// Dynamic state.
    state: BasinState,
    /// Current stored volume (`m^3`).
    stored_vol_m3: f64,
    /// Initial antecedent volume at `t = 0` (`m^3`).
    initial_vol_m3: f64,
    /// Time when basin first reached full capacity.
    first_fill_time_h: Option<f64>,
    /// Time when basin merged into its parent.
    merge_time_h: Option<f64>,
    /// Topographic spill outlet discovered on `dem_refined`.
    outlet: Option<TopographicOutlet>,
    /// Optional dynamic cycle-breaking target override (used only for same-level sibling ties).
    spill_override: Option<u32>,
    /// Realized downstream receiver when spilling.
    realized_spill_to: Option<u32>,
    /// First time this basin actively spilled excess discharge (or reached full capacity with a target).
    first_spill_time_h: Option<f64>,
    /// Accumulated spilled volume during the storm duration window.
    spilled_vol_storm_m3: f64,
    /// Peak spill discharge during the storm duration window.
    peak_spill_rate_m3_h: f64,
    /// Accumulated vertical infiltration during the storm duration window.
    infiltrated_vol_storm_m3: f64,
    /// Snapshot of volume at `t = storm_duration_h`.
    storm_end_vol_m3: f64,
    /// Whether this basin had already merged into its parent at `t = storm_duration_h`.
    storm_end_merged: bool,
    /// Snapshot of routed contributing area at `t = storm_duration_h`.
    storm_end_routed_cat_m2: f64,
}

/// Reusable grid-sized scratch buffers (~25 bytes per cell) for outward Priority-Flood outlet
/// discovery.
///
/// Checked out from a pool bounded by concurrent worker count (or allocated once sequentially)
/// and reused across depressions via generation stamps so outlet discovery never re-allocates
/// `O(N)` buffers per depression or per Rayon work-stealing split.
struct OutletSearchScratch {
    in_source: Vec<bool>,
    visited_pf: Vec<u32>,
    drains_to_source: Vec<u32>,
    trapped_in_pit: Vec<u32>,
    parent_cell: Vec<usize>,
    trace_seen: Vec<u32>,
    pf_gen: u32,
    trace_gen: u32,
    heap: BinaryHeap<HeapCell>,
}

impl OutletSearchScratch {
    fn new(n: usize) -> Self {
        Self {
            in_source: vec![false; n],
            visited_pf: vec![0u32; n],
            drains_to_source: vec![0u32; n],
            trapped_in_pit: vec![0u32; n],
            parent_cell: vec![usize::MAX; n],
            trace_seen: vec![0u32; n],
            pf_gen: 0,
            trace_gen: 0,
            heap: BinaryHeap::new(),
        }
    }
}

/// Discovers the lowest topographic overland spill outlet for a single depression `dep`
/// using outward Priority-Flood on `dem_refined` and D8 downhill tracing.
#[allow(clippy::too_many_arguments)]
fn find_topographic_outlet(
    dep: &NestedDepression,
    w: usize,
    h: usize,
    zdata: &[f64],
    z_nd: f64,
    fdata: &[u8],
    dep_ids: &[i32],
    children_map: &HashMap<u32, Vec<u32>>,
    parent_map: &HashMap<u32, u32>,
    pixels_by_id: &HashMap<u32, Vec<usize>>,
    scratch: &mut OutletSearchScratch,
) -> Option<TopographicOutlet> {
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

    let mut source_cells: Vec<usize> = Vec::new();
    for &id in &desc_ids {
        if let Some(pts) = pixels_by_id.get(&id) {
            source_cells.extend(pts.iter().copied());
        }
    }
    if source_cells.is_empty() {
        return None;
    }

    scratch.pf_gen = scratch.pf_gen.wrapping_add(1);
    if scratch.pf_gen == 0 {
        scratch.visited_pf.fill(0);
        scratch.drains_to_source.fill(0);
        scratch.trapped_in_pit.fill(0);
        scratch.pf_gen = 1;
    }
    let pf_gen = scratch.pf_gen;
    scratch.heap.clear();

    for &idx in &source_cells {
        scratch.in_source[idx] = true;
        scratch.visited_pf[idx] = pf_gen;
        scratch.parent_cell[idx] = usize::MAX;
    }

    // Seed min-heap with perimeter cells of source footprint.
    for &idx in &source_cells {
        let r = idx / w;
        let c = idx % w;
        let mut on_perimeter = r == 0 || r + 1 == h || c == 0 || c + 1 == w;
        if !on_perimeter {
            for d in 1..=8 {
                let nr = r as isize + D8Y[d] as isize;
                let nc = c as isize + D8X[d] as isize;
                if nr >= 0 && nc >= 0 && nr < h as isize && nc < w as isize {
                    let nidx = (nr as usize) * w + (nc as usize);
                    if !scratch.in_source[nidx] {
                        on_perimeter = true;
                        break;
                    }
                }
            }
        }
        if on_perimeter {
            let z = zdata[idx];
            if z.is_finite() && !z.rd_eq(z_nd) {
                scratch.heap.push(HeapCell { elev: z, idx });
            }
        }
    }

    let mut spill_elev = f64::NEG_INFINITY;
    let mut found: Option<TopographicOutlet> = None;
    while let Some(HeapCell { elev, idx: cur }) = scratch.heap.pop() {
        if elev > spill_elev {
            spill_elev = elev;
        }
        let r = cur / w;
        let c = cur % w;
        if !scratch.in_source[cur] {
            scratch.trace_gen = scratch.trace_gen.wrapping_add(1);
            if scratch.trace_gen == 0 {
                scratch.trace_seen.fill(0);
                scratch.trace_gen = 1;
            }
            if let Some((down_trace, target_dep)) = trace_downstream_away_from_source(
                cur,
                w,
                h,
                fdata,
                dep_ids,
                &scratch.in_source,
                &desc_ids,
                &ancestor_ids,
                &mut scratch.drains_to_source,
                &mut scratch.trapped_in_pit,
                pf_gen,
                &mut scratch.trace_seen,
                scratch.trace_gen,
            ) {
                let mut up_path = Vec::new();
                let mut p = cur;
                while p != usize::MAX {
                    up_path.push(p);
                    if scratch.in_source[p] {
                        break;
                    }
                    p = scratch.parent_cell[p];
                }
                up_path.reverse();
                if down_trace.len() > 1 {
                    up_path.extend_from_slice(&down_trace[1..]);
                }
                let cells: Vec<(usize, usize)> =
                    up_path.into_iter().map(|idx| (idx % w, idx / w)).collect();
                found = Some(TopographicOutlet {
                    to_dep_id: target_dep,
                    spill_elev,
                    cells,
                });
                break;
            }
        } else {
            let on_edge = r == 0 || r + 1 == h || c == 0 || c + 1 == w;
            let touches_nodata = !on_edge
                && ancestor_ids.is_empty()
                && (1..=8).any(|d| {
                    let nr = r as isize + D8Y[d] as isize;
                    let nc = c as isize + D8X[d] as isize;
                    if nr >= 0 && nc >= 0 && nr < h as isize && nc < w as isize {
                        let nz = zdata[(nr as usize) * w + (nc as usize)];
                        !nz.is_finite() || nz.rd_eq(z_nd)
                    } else {
                        false
                    }
                });
            if on_edge || touches_nodata {
                // Source basin directly touches the DEM boundary or a clipped NoData border.
                found = Some(TopographicOutlet {
                    to_dep_id: 0,
                    spill_elev,
                    cells: vec![(c, r)],
                });
                break;
            }
        }

        for d in 1..=8 {
            let nr = r as isize + D8Y[d] as isize;
            let nc = c as isize + D8X[d] as isize;
            if nr < 0 || nc < 0 || nr >= h as isize || nc >= w as isize {
                continue;
            }
            let nidx = (nr as usize) * w + (nc as usize);
            if scratch.visited_pf[nidx] == pf_gen {
                continue;
            }
            let nz = zdata[nidx];
            if !nz.is_finite() || nz.rd_eq(z_nd) {
                continue;
            }
            scratch.visited_pf[nidx] = pf_gen;
            scratch.parent_cell[nidx] = cur;
            scratch.heap.push(HeapCell {
                elev: nz,
                idx: nidx,
            });
        }
    }

    for &idx in &source_cells {
        scratch.in_source[idx] = false;
    }

    found
}

/// Resolves the lowest-elevation leaf descendant of `dep_id` in the nesting tree.
fn lowest_leaf_descendant(
    dep_id: u32,
    dep_by_id: &HashMap<u32, &NestedDepression>,
    memo: &mut HashMap<u32, u32>,
) -> u32 {
    if let Some(&leaf) = memo.get(&dep_id) {
        return leaf;
    }
    let Some(d) = dep_by_id.get(&dep_id) else {
        return dep_id;
    };
    if d.children_ids.is_empty() {
        memo.insert(dep_id, dep_id);
        return dep_id;
    }
    let mut best_leaf = dep_id;
    let mut best_elev = f64::INFINITY;
    for &cid in &d.children_ids {
        let leaf = lowest_leaf_descendant(cid, dep_by_id, memo);
        let z = dep_by_id.get(&leaf).map_or(f64::INFINITY, |ld| ld.min_elev);
        if z < best_elev || ((z - best_elev).abs() <= 1e-12 && leaf < best_leaf) {
            best_elev = z;
            best_leaf = leaf;
        }
    }
    memo.insert(dep_id, best_leaf);
    best_leaf
}

/// Resolves the live receiving basin (`Active` or `FullSpilling`) for a spill targeting `id`.
///
/// - `Merged` targets walk upward to their active ancestor.
/// - `Waiting` compound parents route downward to their lowest-elevation `Active` descendant
///   (so spillover entering a compound rim fills from the bottom up without dropping mass).
/// - Returns `0` (DEM boundary) if no eligible live basin exists or if it would self-loop.
fn resolve_spill_target(
    basins: &[BasinNode],
    id_to_idx: &[usize],
    src_id: u32,
    mut id: u32,
) -> u32 {
    while id != 0 {
        let idx = id_to_idx.get(id as usize).copied().unwrap_or(usize::MAX);
        if idx == usize::MAX {
            return 0;
        }
        match basins[idx].state {
            BasinState::Merged => {
                id = basins[idx].parent_id;
            }
            BasinState::Active | BasinState::FullSpilling => {
                return if id == src_id { 0 } else { id };
            }
            BasinState::Waiting => {
                return lowest_active_descendant(id, |x| x == src_id, basins, id_to_idx)
                    .unwrap_or(0);
            }
        }
    }
    0
}

/// Finds the lowest common ancestor of a set of depression IDs in the nesting tree, if one exists.
fn find_lca(nodes: &[u32], basins: &[BasinNode], id_to_idx: &[usize]) -> Option<u32> {
    let first = *nodes.first()?;
    let mut common: Vec<u32> = Vec::new();
    let mut cur = first;
    while cur != 0 {
        let idx = id_to_idx.get(cur as usize).copied().unwrap_or(usize::MAX);
        if idx == usize::MAX {
            break;
        }
        common.push(cur);
        cur = basins[idx].parent_id;
    }
    for &nid in &nodes[1..] {
        let mut anc = HashSet::new();
        let mut c = nid;
        while c != 0 {
            let idx = id_to_idx.get(c as usize).copied().unwrap_or(usize::MAX);
            if idx == usize::MAX {
                break;
            }
            anc.insert(c);
            c = basins[idx].parent_id;
        }
        common.retain(|a| anc.contains(a));
        if common.is_empty() {
            return None;
        }
    }
    common.first().copied()
}

/// Returns the lowest-elevation `Active` descendant of `root_id` (excluding IDs matched by
/// `is_excluded`), if any.
fn lowest_active_descendant(
    root_id: u32,
    is_excluded: impl Fn(u32) -> bool,
    basins: &[BasinNode],
    id_to_idx: &[usize],
) -> Option<u32> {
    let mut stack = vec![root_id];
    let mut best: Option<(f64, u32)> = None;
    while let Some(cur) = stack.pop() {
        let idx = id_to_idx.get(cur as usize).copied().unwrap_or(usize::MAX);
        if idx == usize::MAX {
            continue;
        }
        let b = &basins[idx];
        if b.state == BasinState::Active && !is_excluded(b.id) {
            let cand = (b.min_elev, b.id);
            if best.is_none_or(|prev| {
                cand.0 < prev.0 || ((cand.0 - prev.0).abs() <= 1e-12 && cand.1 < prev.1)
            }) {
                best = Some(cand);
            }
        }
        for &cid in &b.children_ids {
            stack.push(cid);
        }
    }
    best.map(|(_, id)| id)
}

/// Evaluates topological spill inflows (`q_in`), routed contributing areas (`routed_cat`),
/// downstream target indices (`target_idx_buf`), and total boundary spill discharge.
///
/// When `allow_drain_revert` is true and a `FullSpilling` basin loses more to vertical
/// infiltration than it receives, it reverts to `Active` and the routing DAG is recomputed.
#[allow(clippy::too_many_arguments)]
fn evaluate_topological_routing(
    basins: &mut [BasinNode],
    id_to_idx: &[usize],
    rain_m_h: f64,
    allow_drain_revert: bool,
    q_in: &mut [f64],
    routed_cat: &mut [f64],
    indegree: &mut [usize],
    target_idx_buf: &mut [usize],
) -> f64 {
    let m = basins.len();
    loop {
        q_in.fill(0.0);
        routed_cat.fill(0.0);
        indegree.fill(0);
        target_idx_buf.fill(usize::MAX);
        let mut q_boundary_spill = 0.0;

        for i in 0..m {
            match basins[i].state {
                BasinState::Active | BasinState::FullSpilling => {
                    q_in[i] = basins[i].local_eff_cat_area_m2 * rain_m_h;
                    routed_cat[i] = basins[i].local_cat_area_m2;
                }
                BasinState::Waiting | BasinState::Merged => {}
            }
        }

        for i in 0..m {
            if basins[i].state != BasinState::FullSpilling {
                continue;
            }
            let raw_dst = basins[i]
                .spill_override
                .or_else(|| basins[i].outlet.as_ref().map(|o| o.to_dep_id))
                .unwrap_or(0);
            let rep = resolve_spill_target(basins, id_to_idx, basins[i].id, raw_dst);
            if rep != 0 {
                let dst_idx = id_to_idx.get(rep as usize).copied().unwrap_or(usize::MAX);
                if dst_idx != usize::MAX && dst_idx != i {
                    target_idx_buf[i] = dst_idx;
                    indegree[dst_idx] += 1;
                }
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..m {
            if basins[i].state == BasinState::FullSpilling && indegree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut drained_any = false;
        while let Some(u) = queue.pop_front() {
            let net_spill = q_in[u] - basins[u].loss_rate_m3_h;
            if allow_drain_revert && net_spill < -1e-15 && basins[u].stored_vol_m3 > 1e-9 {
                // Infiltration exceeds inflow on this full basin; un-spill and drain.
                basins[u].state = BasinState::Active;
                basins[u].spill_override = None;
                drained_any = true;
                break;
            }
            let q_out = net_spill.max(0.0);
            let dst = target_idx_buf[u];
            if dst != usize::MAX {
                q_in[dst] += q_out;
                routed_cat[dst] += routed_cat[u];
                indegree[dst] = indegree[dst].saturating_sub(1);
                if basins[dst].state == BasinState::FullSpilling && indegree[dst] == 0 {
                    queue.push_back(dst);
                }
            } else {
                q_boundary_spill += q_out;
            }
        }

        if !drained_any {
            return q_boundary_spill;
        }
    }
}

/// Advances the hypsometric cursor on `basin`, recording exact continuous `fill_time` for all
/// cells whose threshold volume is now `<= basin.stored_vol_m3`.
fn mark_newly_inundated_cells(
    basin: &mut BasinNode,
    t_start: f64,
    vol_start: f64,
    q_net: f64,
    fill_time_data: &mut [f64],
) {
    let vol_end = basin.stored_vol_m3;
    if vol_end <= 0.0 || (basin.max_vol_reached >= 0.0 && vol_end <= basin.max_vol_reached) {
        return;
    }
    while basin.hypso_cursor < basin.hypso_cells.len() {
        let cell = basin.hypso_cells[basin.hypso_cursor];
        if cell.threshold_vol > vol_end + 1e-12 {
            break;
        }
        let t_cell = if cell.threshold_vol <= vol_start || q_net <= 1e-15 {
            t_start
        } else {
            t_start + ((cell.threshold_vol - vol_start) / q_net).max(0.0)
        };
        let slot = &mut fill_time_data[cell.grid_idx];
        if *slot < 0.0 || t_cell < *slot {
            *slot = t_cell;
        }
        basin.hypso_cursor += 1;
    }
    if vol_end > basin.max_vol_reached {
        basin.max_vol_reached = vol_end;
    }
}

/// Immediately marks all cells below `basin.stored_vol_m3` as inundated at timestamp `t_now`
/// (used at `t = 0` for antecedent ponding and upon parent merges).
fn mark_cells_at_instant(basin: &mut BasinNode, t_now: f64, fill_time_data: &mut [f64]) {
    if basin.stored_vol_m3 <= 0.0 {
        return;
    }
    while basin.hypso_cursor < basin.hypso_cells.len() {
        let cell = basin.hypso_cells[basin.hypso_cursor];
        if cell.threshold_vol > basin.stored_vol_m3 + 1e-12 {
            break;
        }
        let slot = &mut fill_time_data[cell.grid_idx];
        if *slot < 0.0 || t_now < *slot {
            *slot = t_now;
        }
        basin.hypso_cursor += 1;
    }
    if basin.stored_vol_m3 > basin.max_vol_reached {
        basin.max_vol_reached = basin.stored_vol_m3;
    }
}

/// Runs dynamic event-driven Fill-Spill-Merge cascade mass-balance routing in memory.
///
/// Combines D8 catchment-weighted runoff attribution, exact hypsometric stage-storage
/// curves, dynamic overland spill cascades between depressions, hierarchical child-to-parent
/// basin merging, and watershed-scale mass conservation verification.
pub fn simulate_fill_spill_merge_in_memory(
    dem_refined: &Array2D<f64>,
    sink: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    options: &SimulateInundationOptions,
    forcing: &InundationForcing<'_>,
) -> Result<FsmSimulationResult> {
    let _ = validate_options_with_hyetograph(options, forcing.hyetograph.is_some())?;
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

    let w = dem_refined.width();
    let h = dem_refined.height();
    let n = w * h;
    validate_grid_alignment(dem_refined, sink, "sink")?;
    validate_grid_alignment(dem_refined, &depressions.depression_id, "depression_id")?;
    validate_forcing_grids(sink, forcing)?;

    let (_, _, cell_area) = affine_metrics(&dem_refined.geotransform);
    let mut fill_time_data = vec![-1.0f64; n];

    let storm_duration_h = if let Some(hyeto) = forcing.hyetograph {
        let total_hyeto_h: f64 = hyeto.iter().map(|&(dt, _)| dt).sum();
        total_hyeto_h.min(options.rain_duration_h)
    } else {
        options.rain_duration_h
    };

    if depressions.depressions.is_empty() || n == 0 {
        // Compute direct boundary runoff for mass balance when no depressions exist.
        let zdata = dem_refined.data();
        let z_nd = dem_refined.no_data();
        let mut eff_valid_area = 0.0f64;
        for (idx, &z) in zdata.iter().enumerate() {
            if !z.is_finite() || z.rd_eq(z_nd) {
                continue;
            }
            let c = forcing
                .runoff_coeff
                .map(|rc| {
                    let v = rc.get_i(idx);
                    if v.is_finite() && !v.rd_eq(rc.no_data()) {
                        v.clamp(0.0, 1.0)
                    } else {
                        1.0
                    }
                })
                .unwrap_or(1.0);
            eff_valid_area += cell_area * c;
        }
        let total_rain_m = crate::inundation::cumulative_runoff_mm(
            storm_duration_h,
            options.rain_intensity_cm_h,
            forcing.hyetograph,
        ) / 1000.0;
        let total_runoff = eff_valid_area * total_rain_m;
        let mut ft = Array2D::from_vec(w, h, fill_time_data)?;
        ft.copy_metadata_from(sink);
        ft.set_no_data(-1.0);
        return Ok(FsmSimulationResult {
            fill_time: ft,
            spill_paths: Vec::new(),
            basins: Vec::new(),
            mass_balance: FsmMassBalance {
                duration_h: storm_duration_h,
                total_antecedent_m3: 0.0,
                total_runoff_m3: total_runoff,
                total_stored_m3: 0.0,
                total_infiltration_m3: 0.0,
                total_boundary_outflow_m3: total_runoff,
                mass_balance_error_m3: 0.0,
            },
            num_regions: depressions.num_regions as usize,
            num_spill_events: 0,
            num_merge_events: 0,
        });
    }

    // 1. Build hierarchy lookup maps and group pixels by innermost depression ID.
    let max_id = depressions
        .depressions
        .iter()
        .map(|d| d.id)
        .max()
        .unwrap_or(0) as usize;
    let mut id_to_idx = vec![usize::MAX; max_id + 1];
    let mut dep_by_id: HashMap<u32, &NestedDepression> =
        HashMap::with_capacity(depressions.depressions.len());
    let mut children_map: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(depressions.depressions.len());
    let mut parent_map: HashMap<u32, u32> = HashMap::new();
    for (idx, d) in depressions.depressions.iter().enumerate() {
        id_to_idx[d.id as usize] = idx;
        dep_by_id.insert(d.id, d);
        children_map.insert(d.id, d.children_ids.clone());
        if d.parent_id > 0 {
            parent_map.insert(d.id, d.parent_id);
        }
        for &cid in &d.children_ids {
            parent_map.entry(cid).or_insert(d.id);
        }
    }

    let dep_ids = depressions.depression_id.data();
    let mut pixels_by_id: HashMap<u32, Vec<usize>> = HashMap::new();
    for (idx, &did) in dep_ids.iter().enumerate() {
        if did > 0 && id_to_idx.get(did as usize).copied().unwrap_or(usize::MAX) != usize::MAX {
            pixels_by_id.entry(did as u32).or_default().push(idx);
        }
    }

    // 2. Compute D8 flow directions on dem_refined once (`O(N)`).
    let flowdirs = d8_flow_directions_resolved(dem_refined)?;
    let fdata = flowdirs.data();
    let zdata = dem_refined.data();
    let z_nd = dem_refined.no_data();
    let sdata = sink.data();
    let s_nd = sink.no_data();

    // 3. Single-pass D8 catchment-weighted runoff attribution (`O(N)`).
    // Every valid cell flows downhill along D8 to its terminal pit (or DEM exit).
    // If it terminates inside a depression `d`, we credit its catchment area and
    // `cell_area * runoff_coeff` to `lowest_leaf_descendant(d)` so water always fills
    // from the bottom up with exact 100% watershed mass conservation (zero double counting).
    let mut leaf_memo: HashMap<u32, u32> = HashMap::new();
    let mut local_cat_area = vec![0.0f64; max_id + 1];
    let mut local_eff_cat_area = vec![0.0f64; max_id + 1];
    let mut boundary_eff_cat_area = 0.0f64;

    const UNVISITED: u32 = u32::MAX;
    const VISITING: u32 = u32::MAX - 1;
    let mut terminal_dep = vec![UNVISITED; n];
    let mut path_buf: Vec<usize> = Vec::with_capacity(128);

    for start in 0..n {
        let z = zdata[start];
        if !z.is_finite() || z.rd_eq(z_nd) {
            terminal_dep[start] = 0;
            continue;
        }
        let target = if terminal_dep[start] != UNVISITED {
            terminal_dep[start]
        } else {
            path_buf.clear();
            let mut cur = start;
            let resolved = loop {
                if terminal_dep[cur] != UNVISITED {
                    break if terminal_dep[cur] == VISITING {
                        0
                    } else {
                        terminal_dep[cur]
                    };
                }
                terminal_dep[cur] = VISITING;
                path_buf.push(cur);

                let did = dep_ids[cur];
                // Stop immediately upon reaching a leaf depression cell so its cells stay inside it.
                if did > 0 {
                    let uid = did as u32;
                    if let Some(d) = dep_by_id.get(&uid)
                        && d.children_ids.is_empty()
                    {
                        break uid;
                    }
                }

                let dir = fdata[cur] as usize;
                if dir == NO_FLOW as usize || dir > 8 {
                    if did > 0
                        && id_to_idx.get(did as usize).copied().unwrap_or(usize::MAX) != usize::MAX
                    {
                        break lowest_leaf_descendant(did as u32, &dep_by_id, &mut leaf_memo);
                    }
                    break 0;
                }
                let nr = (cur / w) as isize + D8Y[dir] as isize;
                let nc = (cur % w) as isize + D8X[dir] as isize;
                if nr < 0 || nc < 0 || nr >= h as isize || nc >= w as isize {
                    if did > 0
                        && id_to_idx.get(did as usize).copied().unwrap_or(usize::MAX) != usize::MAX
                    {
                        break lowest_leaf_descendant(did as u32, &dep_by_id, &mut leaf_memo);
                    }
                    break 0;
                }
                let next = (nr as usize) * w + (nc as usize);
                let nz = zdata[next];
                if !nz.is_finite() || nz.rd_eq(z_nd) {
                    if did > 0
                        && id_to_idx.get(did as usize).copied().unwrap_or(usize::MAX) != usize::MAX
                    {
                        break lowest_leaf_descendant(did as u32, &dep_by_id, &mut leaf_memo);
                    }
                    break 0;
                }
                cur = next;
            };
            for &p in &path_buf {
                terminal_dep[p] = resolved;
            }
            resolved
        };

        let c_val = forcing
            .runoff_coeff
            .map(|rc| {
                let v = rc.get_i(start);
                if v.is_finite() && !v.rd_eq(rc.no_data()) {
                    v.clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .unwrap_or(1.0);

        if target > 0 {
            local_cat_area[target as usize] += cell_area;
            local_eff_cat_area[target as usize] += cell_area * c_val;
        } else {
            boundary_eff_cat_area += cell_area * c_val;
        }
    }

    // 4. Discover topographic spill outlets for every depression (parallel when enabled),
    // checking out reusable scratch buffers from a pool strictly bounded by active worker count.
    #[cfg(feature = "parallel")]
    let outlets: Vec<Option<TopographicOutlet>> = {
        let pool = std::sync::Mutex::new(Vec::<OutletSearchScratch>::new());
        depressions
            .depressions
            .par_iter()
            .map(|d| {
                let mut scratch = pool
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .pop()
                    .unwrap_or_else(|| OutletSearchScratch::new(n));
                let out = find_topographic_outlet(
                    d,
                    w,
                    h,
                    zdata,
                    z_nd,
                    fdata,
                    dep_ids,
                    &children_map,
                    &parent_map,
                    &pixels_by_id,
                    &mut scratch,
                );
                pool.lock().unwrap_or_else(|e| e.into_inner()).push(scratch);
                out
            })
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let outlets: Vec<Option<TopographicOutlet>> = {
        let mut scratch = OutletSearchScratch::new(n);
        depressions
            .depressions
            .iter()
            .map(|d| {
                find_topographic_outlet(
                    d,
                    w,
                    h,
                    zdata,
                    z_nd,
                    fdata,
                    dep_ids,
                    &children_map,
                    &parent_map,
                    &pixels_by_id,
                    &mut scratch,
                )
            })
            .collect()
    };

    // 5. Build exact hypsometric storage curves, infiltration rates, and antecedent volumes.
    // Collect sorted footprint cells and initial rim elevations first, incorporating each
    // basin's discovered saddle elevation (`outlet.spill_elev`), then propagate rim elevations
    // bottom-up so every parent's `z_rim >= max(z_rim(children))` and `V_cap(P) >= sum(V_cap(c))`.
    let mut all_cells_by_dep: Vec<Vec<(f64, usize)>> =
        Vec::with_capacity(depressions.depressions.len());
    let mut z_rims: Vec<f64> = Vec::with_capacity(depressions.depressions.len());

    for (dep_idx, d) in depressions.depressions.iter().enumerate() {
        let mut all_cells: Vec<(f64, usize)> = Vec::new();
        let mut stack = vec![d.id];
        let mut visited_ids = HashSet::new();
        while let Some(cur_id) = stack.pop() {
            if !visited_ids.insert(cur_id) {
                continue;
            }
            if let Some(pts) = pixels_by_id.get(&cur_id) {
                for &gi in pts {
                    let sv = sdata[gi];
                    let z = if sv.is_finite() && !sv.rd_eq(s_nd) {
                        sv
                    } else {
                        zdata[gi]
                    };
                    if z.is_finite() && !z.rd_eq(z_nd) {
                        all_cells.push((z, gi));
                    }
                }
            }
            if let Some(kids) = children_map.get(&cur_id) {
                stack.extend(kids.iter().copied());
            }
        }
        all_cells.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut z_rim = d
            .max_elev
            .max(all_cells.last().map_or(d.max_elev, |&(z, _)| z));
        if let Some(outlet) = &outlets[dep_idx]
            && outlet.spill_elev.is_finite()
        {
            z_rim = z_rim.max(outlet.spill_elev);
        }
        all_cells_by_dep.push(all_cells);
        z_rims.push(z_rim);
    }

    let mut level_order: Vec<usize> = (0..depressions.depressions.len()).collect();
    level_order.sort_by_key(|&idx| {
        (
            depressions.depressions[idx].level,
            depressions.depressions[idx].id,
        )
    });
    for &dep_idx in &level_order {
        let d = &depressions.depressions[dep_idx];
        let pid = parent_map.get(&d.id).copied().unwrap_or(d.parent_id);
        if pid > 0 {
            let pidx = id_to_idx.get(pid as usize).copied().unwrap_or(usize::MAX);
            if pidx != usize::MAX && z_rims[dep_idx] > z_rims[pidx] {
                z_rims[pidx] = z_rims[dep_idx];
            }
        }
    }

    let mut basins: Vec<BasinNode> = Vec::with_capacity(depressions.depressions.len());
    let mut total_antecedent_m3 = 0.0f64;

    for (dep_idx, (d, all_cells)) in depressions
        .depressions
        .iter()
        .zip(all_cells_by_dep)
        .enumerate()
    {
        let z_rim = z_rims[dep_idx];

        let mut hypso_cells: Vec<HypsoCell> = Vec::with_capacity(all_cells.len());
        let z_base = all_cells.first().map_or(0.0, |&(z, _)| z);
        let mut prefix_sum = 0.0f64;
        let mut capacity_m3 = 0.0f64;
        for (i, &(z, gi)) in all_cells.iter().enumerate() {
            let dz = z - z_base;
            let thresh = if i == 0 {
                0.0
            } else if (z - all_cells[i - 1].0).abs() <= 1e-12 {
                hypso_cells[i - 1].threshold_vol
            } else {
                ((i as f64) * dz - prefix_sum).max(0.0) * cell_area
            };
            hypso_cells.push(HypsoCell {
                grid_idx: gi,
                threshold_vol: thresh,
            });
            prefix_sum += dz;
            capacity_m3 += (z_rim - z).max(0.0) * cell_area;
        }

        // Compute total infiltration loss rate over this depression's entire footprint.
        let loss_rate_m3_h = if let Some(inf) = forcing.infiltration_cm_h {
            let inf_nd = inf.no_data();
            let mut sum_m_h = 0.0f64;
            for &(_, gi) in &all_cells {
                let v = inf.get_i(gi);
                if v.is_finite() && !v.rd_eq(inf_nd) && v > 0.0 {
                    sum_m_h += (v / 100.0) * cell_area;
                }
            }
            sum_m_h
        } else {
            0.0
        };

        // Compute initial antecedent ponded water over OWNED cells (`dep_id == d.id`) only,
        // so children and parents never double-count antecedent water; upon merge, children's
        // volumes combine with the parent's owned rim antecedent water.
        let mut owned_ant_vol = 0.0f64;
        if let Some(ant) = forcing.antecedent_depth {
            let ant_nd = ant.no_data();
            if let Some(pts) = pixels_by_id.get(&d.id) {
                for &gi in pts {
                    let h0 = ant.get_i(gi);
                    if h0.is_finite() && !h0.rd_eq(ant_nd) && h0 > 0.0 {
                        let sv = sdata[gi];
                        let z = if sv.is_finite() && !sv.rd_eq(s_nd) {
                            sv
                        } else {
                            zdata[gi]
                        };
                        if z.is_finite() && !z.rd_eq(z_nd) {
                            let max_cell_depth = (z_rim - z).max(0.0);
                            owned_ant_vol += h0.min(max_cell_depth) * cell_area;
                            fill_time_data[gi] = 0.0;
                        }
                    }
                }
            }
        }
        owned_ant_vol = owned_ant_vol.min(capacity_m3);
        total_antecedent_m3 += owned_ant_vol;

        let is_leaf = d.children_ids.is_empty();
        let pid = parent_map.get(&d.id).copied().unwrap_or(d.parent_id);

        let mut node = BasinNode {
            id: d.id,
            parent_id: pid,
            level: d.level,
            children_ids: d.children_ids.clone(),
            min_elev: d.min_elev,
            capacity_m3,
            hypso_cells,
            hypso_cursor: 0,
            max_vol_reached: -1.0,
            local_cat_area_m2: local_cat_area[d.id as usize],
            local_eff_cat_area_m2: local_eff_cat_area[d.id as usize],
            loss_rate_m3_h,
            state: if is_leaf {
                BasinState::Active
            } else {
                BasinState::Waiting
            },
            stored_vol_m3: owned_ant_vol,
            initial_vol_m3: owned_ant_vol,
            first_fill_time_h: None,
            merge_time_h: None,
            outlet: outlets[dep_idx].clone(),
            spill_override: None,
            realized_spill_to: None,
            first_spill_time_h: None,
            spilled_vol_storm_m3: 0.0,
            peak_spill_rate_m3_h: 0.0,
            infiltrated_vol_storm_m3: 0.0,
            storm_end_vol_m3: owned_ant_vol,
            storm_end_merged: false,
            storm_end_routed_cat_m2: local_cat_area[d.id as usize],
        };

        if is_leaf {
            if node.stored_vol_m3 > 0.0 {
                mark_cells_at_instant(&mut node, 0.0, &mut fill_time_data);
            }
            if node.stored_vol_m3 >= node.capacity_m3 - 1e-9 {
                node.stored_vol_m3 = node.capacity_m3;
                node.state = BasinState::FullSpilling;
                node.first_fill_time_h = Some(0.0);
                mark_cells_at_instant(&mut node, 0.0, &mut fill_time_data);
            }
        }

        basins.push(node);
    }

    // 6. Build piecewise-constant rainfall segments.
    // Each segment is `(dt_h, rain_m_h, within_storm_window, snapshot_after_segment)`.
    // For constant rainfall, Segment 1 covers `[0, storm_duration_h]` (used for mass balance
    // and storm volume summaries), and Segment 2 covers `[storm_duration_h, infinity]` so
    // deep basins still receive exact finite `fill_time` values in `fill_time.tif`.
    let mut segments: Vec<(f64, f64, bool, bool)> = Vec::new();
    if let Some(hyeto) = forcing.hyetograph {
        let mut elapsed = 0.0f64;
        for &(dt, cm_h) in hyeto {
            if dt <= 0.0 {
                continue;
            }
            let rain_m_h = cm_h / 100.0;
            if elapsed < storm_duration_h - 1e-12 && elapsed + dt > storm_duration_h + 1e-12 {
                let first_dt = storm_duration_h - elapsed;
                let second_dt = dt - first_dt;
                segments.push((first_dt, rain_m_h, true, false));
                segments.push((second_dt, rain_m_h, false, false));
            } else {
                let in_storm = elapsed < storm_duration_h - 1e-12;
                segments.push((dt, rain_m_h, in_storm, false));
            }
            elapsed += dt;
        }
    } else {
        let rain_m_h = options.rain_intensity_cm_h / 100.0;
        segments.push((storm_duration_h, rain_m_h, true, false));
        segments.push((f64::INFINITY, rain_m_h, false, false));
    }
    if let Some(last_storm) = segments.iter_mut().rev().find(|s| s.2) {
        last_storm.3 = true;
    }

    // Helper: resolve all ready parent merges and break any full spill cycles so the
    // `FullSpilling` routing graph is guaranteed to be a DAG.
    let mut cycle_color = vec![0u8; basins.len()];
    let resolve_merges_and_cycles = |basins: &mut [BasinNode],
                                     id_to_idx: &[usize],
                                     t_now: f64,
                                     num_merge_events: &mut usize,
                                     fill_time_data: &mut [f64],
                                     color: &mut [u8]| {
        loop {
            let mut changed = false;

            // (a) Check if any `Waiting` parent now has all direct children FullSpilling/Merged.
            for idx in 0..basins.len() {
                if basins[idx].state != BasinState::Waiting || basins[idx].children_ids.is_empty() {
                    continue;
                }
                let all_children_full = basins[idx].children_ids.iter().all(|&cid| {
                    let cidx = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                    cidx != usize::MAX
                        && matches!(
                            basins[cidx].state,
                            BasinState::FullSpilling | BasinState::Merged
                        )
                });
                if !all_children_full {
                    continue;
                }

                let child_ids = basins[idx].children_ids.clone();
                let mut merged_vol = basins[idx].stored_vol_m3;
                let mut merged_cat = basins[idx].local_cat_area_m2;
                let mut merged_eff_cat = basins[idx].local_eff_cat_area_m2;
                for cid in child_ids {
                    let cidx = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                    if cidx != usize::MAX && basins[cidx].state == BasinState::FullSpilling {
                        merged_vol += basins[cidx].stored_vol_m3;
                        merged_cat += basins[cidx].local_cat_area_m2;
                        merged_eff_cat += basins[cidx].local_eff_cat_area_m2;
                        basins[cidx].state = BasinState::Merged;
                        basins[cidx].merge_time_h = Some(t_now);
                    }
                }
                let p = &mut basins[idx];
                p.stored_vol_m3 = merged_vol.min(p.capacity_m3);
                p.local_cat_area_m2 = merged_cat;
                p.local_eff_cat_area_m2 = merged_eff_cat;
                p.state = BasinState::Active;
                *num_merge_events += 1;

                mark_cells_at_instant(p, t_now, fill_time_data);
                if p.stored_vol_m3 >= p.capacity_m3 - 1e-9 {
                    p.stored_vol_m3 = p.capacity_m3;
                    p.state = BasinState::FullSpilling;
                    if p.first_fill_time_h.is_none() {
                        p.first_fill_time_h = Some(t_now);
                    }
                    mark_cells_at_instant(p, t_now, fill_time_data);
                }
                changed = true;
            }

            if changed {
                continue;
            }

            // (b) Detect and break any cycle among `FullSpilling` nodes.
            let m = basins.len();
            color.fill(0); // 0 = unvisited, 1 = visiting, 2 = done
            let mut cycle_nodes: Option<Vec<u32>> = None;

            for start_idx in 0..m {
                if basins[start_idx].state != BasinState::FullSpilling || color[start_idx] != 0 {
                    continue;
                }
                let mut chain: Vec<usize> = Vec::new();
                let mut cur_idx = start_idx;
                loop {
                    if color[cur_idx] == 2 {
                        break;
                    }
                    if color[cur_idx] == 1 {
                        if let Some(pos) = chain.iter().position(|&x| x == cur_idx) {
                            cycle_nodes =
                                Some(chain[pos..].iter().map(|&i| basins[i].id).collect());
                        }
                        break;
                    }
                    color[cur_idx] = 1;
                    chain.push(cur_idx);

                    let raw_dst = basins[cur_idx]
                        .spill_override
                        .or_else(|| basins[cur_idx].outlet.as_ref().map(|o| o.to_dep_id))
                        .unwrap_or(0);
                    let rep = resolve_spill_target(basins, id_to_idx, basins[cur_idx].id, raw_dst);
                    if rep == 0 {
                        break;
                    }
                    let next_idx = id_to_idx.get(rep as usize).copied().unwrap_or(usize::MAX);
                    if next_idx == usize::MAX || basins[next_idx].state != BasinState::FullSpilling
                    {
                        break;
                    }
                    cur_idx = next_idx;
                }
                for &ci in &chain {
                    color[ci] = 2;
                }
                if cycle_nodes.is_some() {
                    break;
                }
            }

            if let Some(cyc) = cycle_nodes {
                let cyc_set: HashSet<u32> = cyc.iter().copied().collect();
                // Pick the lowest-rim member of the cycle as the break point.
                let break_id = *cyc
                    .iter()
                    .min_by(|&&a, &&b| {
                        let za = id_to_idx
                            .get(a as usize)
                            .copied()
                            .filter(|&i| i != usize::MAX)
                            .and_then(|i| basins[i].outlet.as_ref())
                            .map_or(f64::INFINITY, |o| o.spill_elev);
                        let zb = id_to_idx
                            .get(b as usize)
                            .copied()
                            .filter(|&i| i != usize::MAX)
                            .and_then(|i| basins[i].outlet.as_ref())
                            .map_or(f64::INFINITY, |o| o.spill_elev);
                        za.total_cmp(&zb).then_with(|| a.cmp(&b))
                    })
                    .unwrap();

                let new_target = find_lca(&cyc, basins, id_to_idx)
                    .and_then(|lca| {
                        lowest_active_descendant(lca, |x| cyc_set.contains(&x), basins, id_to_idx)
                    })
                    .unwrap_or(0);
                let bidx = id_to_idx
                    .get(break_id as usize)
                    .copied()
                    .unwrap_or(usize::MAX);
                if bidx != usize::MAX {
                    basins[bidx].spill_override = Some(new_target);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
    };

    // 7. Event-driven integration loop.
    let m = basins.len();
    let mut t_cur = 0.0f64;
    let mut total_runoff_storm_m3 = 0.0f64;
    let mut total_infiltration_storm_m3 = 0.0f64;
    let mut total_boundary_outflow_storm_m3 = 0.0f64;
    let mut num_merge_events = 0usize;
    let mut snapshot_taken = false;
    let mut last_rain_m_h = 0.0f64;

    let mut q_in = vec![0.0f64; m];
    let mut routed_cat = vec![0.0f64; m];
    let mut indegree = vec![0usize; m];
    let mut target_idx_buf = vec![usize::MAX; m];

    for (seg_idx, (seg_dt, rain_m_h, in_storm_window, snapshot_after_segment)) in
        segments.into_iter().enumerate()
    {
        if seg_idx > 0 && rain_m_h < last_rain_m_h - 1e-15 {
            for b in &mut basins {
                b.spill_override = None;
            }
        }
        last_rain_m_h = rain_m_h;
        let t_seg_end = if seg_dt.is_finite() {
            t_cur + seg_dt
        } else {
            f64::INFINITY
        };
        let mut iter_guard = 0usize;
        let max_iters = (m + 1) * 16;
        let mut segment_done_early = false;

        while t_cur < t_seg_end - 1e-12 && iter_guard < max_iters {
            iter_guard += 1;

            resolve_merges_and_cycles(
                &mut basins,
                &id_to_idx,
                t_cur,
                &mut num_merge_events,
                &mut fill_time_data,
                &mut cycle_color,
            );

            // Evaluate topological spill inflows; if a FullSpilling basin loses more to
            // infiltration than it receives during a dry/low-rain step, revert it to Active
            // so it drains cleanly rather than spilling negative water.
            let q_boundary_spill = evaluate_topological_routing(
                &mut basins,
                &id_to_idx,
                rain_m_h,
                true,
                &mut q_in,
                &mut routed_cat,
                &mut indegree,
                &mut target_idx_buf,
            );

            // Record realized spill targets and start times for newly spilling basins.
            for i in 0..m {
                if basins[i].state == BasinState::FullSpilling {
                    let q_out = (q_in[i] - basins[i].loss_rate_m3_h).max(0.0);
                    if q_out > 1e-15 {
                        if basins[i].first_spill_time_h.is_none() {
                            basins[i].first_spill_time_h = Some(t_cur);
                        }
                        if basins[i].realized_spill_to.is_none() {
                            let dst = target_idx_buf[i];
                            let dst_id = if dst != usize::MAX { basins[dst].id } else { 0 };
                            basins[i].realized_spill_to = Some(dst_id);
                        }
                        if in_storm_window && q_out > basins[i].peak_spill_rate_m3_h {
                            basins[i].peak_spill_rate_m3_h = q_out;
                        }
                    }
                }
            }

            // Determine next event time step `dt`.
            let mut dt_next = t_seg_end - t_cur;
            let mut any_filling = false;
            for i in 0..m {
                if basins[i].state != BasinState::Active {
                    continue;
                }
                let q_net = q_in[i] - basins[i].loss_rate_m3_h;
                if q_net > 1e-15 {
                    any_filling = true;
                    let rem = (basins[i].capacity_m3 - basins[i].stored_vol_m3).max(0.0);
                    let dt_fill = rem / q_net;
                    if dt_fill < dt_next {
                        dt_next = dt_fill;
                    }
                }
            }

            if !dt_next.is_finite() {
                // Unbounded post-storm tail segment with no remaining filling basins.
                segment_done_early = true;
                break;
            }
            if !any_filling && !t_seg_end.is_finite() {
                segment_done_early = true;
                break;
            }

            // Advance all basins by `dt_next`.
            let t_start = t_cur;
            let t_end = t_cur + dt_next;

            if in_storm_window {
                let mut total_local_eff_cat = boundary_eff_cat_area;
                for b in &basins {
                    if matches!(b.state, BasinState::Active | BasinState::FullSpilling) {
                        total_local_eff_cat += b.local_eff_cat_area_m2;
                    }
                }
                total_runoff_storm_m3 += total_local_eff_cat * rain_m_h * dt_next;
                total_boundary_outflow_storm_m3 +=
                    (boundary_eff_cat_area * rain_m_h + q_boundary_spill) * dt_next;
            }

            for i in 0..m {
                match basins[i].state {
                    BasinState::Active => {
                        let q_net = q_in[i] - basins[i].loss_rate_m3_h;
                        let vol_old = basins[i].stored_vol_m3;
                        let vol_new = (vol_old + q_net * dt_next).clamp(0.0, basins[i].capacity_m3);
                        basins[i].stored_vol_m3 = vol_new;

                        if in_storm_window && basins[i].loss_rate_m3_h > 0.0 {
                            // Exact actual infiltration: `Q_in * dt - (vol_new - vol_old)`.
                            let actual_infil = (q_in[i] * dt_next - (vol_new - vol_old)).max(0.0);
                            basins[i].infiltrated_vol_storm_m3 += actual_infil;
                            total_infiltration_storm_m3 += actual_infil;
                        }

                        if q_net > 1e-15 {
                            mark_newly_inundated_cells(
                                &mut basins[i],
                                t_start,
                                vol_old,
                                q_net,
                                &mut fill_time_data,
                            );
                        }

                        if q_net > 1e-15 && basins[i].stored_vol_m3 >= basins[i].capacity_m3 - 1e-9
                        {
                            basins[i].stored_vol_m3 = basins[i].capacity_m3;
                            basins[i].state = BasinState::FullSpilling;
                            if basins[i].first_fill_time_h.is_none() {
                                basins[i].first_fill_time_h = Some(t_end);
                            }
                            mark_cells_at_instant(&mut basins[i], t_end, &mut fill_time_data);
                        }
                    }
                    BasinState::FullSpilling => {
                        if in_storm_window {
                            let actual_infil = basins[i].loss_rate_m3_h.min(q_in[i]) * dt_next;
                            let q_out = (q_in[i] - basins[i].loss_rate_m3_h).max(0.0);
                            basins[i].infiltrated_vol_storm_m3 += actual_infil;
                            basins[i].spilled_vol_storm_m3 += q_out * dt_next;
                            total_infiltration_storm_m3 += actual_infil;
                        }
                    }
                    BasinState::Waiting | BasinState::Merged => {}
                }
            }

            t_cur = t_end;
        }

        if !segment_done_early && t_cur < t_seg_end - 1e-12 && iter_guard >= max_iters {
            return Err(Error::Algorithm(format!(
                "FSM routing exceeded iteration guard ({max_iters}) at t={t_cur:.6} h before segment end {t_seg_end:.6} h"
            )));
        }

        // Snapshot storm-end volumes and routed contributing areas right when the storm window ends.
        if snapshot_after_segment && !snapshot_taken {
            resolve_merges_and_cycles(
                &mut basins,
                &id_to_idx,
                t_cur,
                &mut num_merge_events,
                &mut fill_time_data,
                &mut cycle_color,
            );
            let _ = evaluate_topological_routing(
                &mut basins,
                &id_to_idx,
                rain_m_h,
                false,
                &mut q_in,
                &mut routed_cat,
                &mut indegree,
                &mut target_idx_buf,
            );
            for i in 0..m {
                basins[i].storm_end_vol_m3 = basins[i].stored_vol_m3;
                basins[i].storm_end_merged = basins[i].state == BasinState::Merged;
                basins[i].storm_end_routed_cat_m2 = if routed_cat[i] > 0.0 {
                    routed_cat[i]
                } else {
                    basins[i].local_cat_area_m2
                };
            }
            snapshot_taken = true;
        }
    }

    if !snapshot_taken {
        resolve_merges_and_cycles(
            &mut basins,
            &id_to_idx,
            t_cur,
            &mut num_merge_events,
            &mut fill_time_data,
            &mut cycle_color,
        );
        let _ = evaluate_topological_routing(
            &mut basins,
            &id_to_idx,
            last_rain_m_h,
            false,
            &mut q_in,
            &mut routed_cat,
            &mut indegree,
            &mut target_idx_buf,
        );
        for i in 0..m {
            basins[i].storm_end_vol_m3 = basins[i].stored_vol_m3;
            basins[i].storm_end_merged = basins[i].state == BasinState::Merged;
            basins[i].storm_end_routed_cat_m2 = if routed_cat[i] > 0.0 {
                routed_cat[i]
            } else {
                basins[i].local_cat_area_m2
            };
        }
    }

    // 8. Compute exact watershed-scale mass balance at `t = storm_duration_h`.
    // Sum stored volumes over all basins that had NOT merged at the storm-end snapshot
    // (including any `Waiting` parent holding owned rim antecedent water).
    let mut total_stored_storm_m3 = 0.0f64;
    for b in &basins {
        if !b.storm_end_merged {
            total_stored_storm_m3 += b.storm_end_vol_m3;
        }
    }
    let mass_balance_error_m3 = (total_antecedent_m3 + total_runoff_storm_m3)
        - (total_stored_storm_m3 + total_infiltration_storm_m3 + total_boundary_outflow_storm_m3);

    let mass_balance = FsmMassBalance {
        duration_h: storm_duration_h,
        total_antecedent_m3,
        total_runoff_m3: total_runoff_storm_m3,
        total_stored_m3: total_stored_storm_m3,
        total_infiltration_m3: total_infiltration_storm_m3,
        total_boundary_outflow_m3: total_boundary_outflow_storm_m3,
        mass_balance_error_m3,
    };

    // 9. Build dynamically realized `SpillFlowPath` list and per-basin summaries.
    let is_ancestor = |anc_id: u32, mut cur_id: u32| -> bool {
        if anc_id == 0 || cur_id == 0 {
            return false;
        }
        while let Some(&pid) = parent_map.get(&cur_id) {
            if pid == anc_id {
                return true;
            }
            cur_id = pid;
        }
        false
    };

    let mut spill_paths: Vec<SpillFlowPath> = Vec::new();
    let mut num_spill_events = 0usize;
    let mut summaries: Vec<FsmBasinSummary> = Vec::with_capacity(m);
    let storm_end = storm_duration_h + STORM_TIME_EPS_H;

    for b in &basins {
        if b.first_spill_time_h.is_some() {
            num_spill_events += 1;
        }
        if let (Some(spill_t), Some(outlet)) = (b.first_spill_time_h, b.outlet.as_ref()) {
            let realized_to = b.realized_spill_to.unwrap_or(outlet.to_dep_id);
            let geometry_matches = realized_to == outlet.to_dep_id
                || is_ancestor(realized_to, outlet.to_dep_id)
                || is_ancestor(outlet.to_dep_id, realized_to);
            let valid_duration = b.merge_time_h.is_none_or(|mt| mt > spill_t + 1e-6);
            let show_boundary = b.parent_id > 0 || depressions.num_regions > 1;
            // Require >= 2 vertices so exported GeoJSON/GPKG LineString geometries remain valid
            // per RFC 7946; single-cell border touches are still reported via `spill_to_dep_id`.
            if geometry_matches
                && valid_duration
                && outlet.cells.len() >= 2
                && (realized_to != 0 || show_boundary)
            {
                spill_paths.push(SpillFlowPath {
                    from_dep_id: b.id,
                    from_level: b.level,
                    to_dep_id: realized_to,
                    spill_time_h: spill_t,
                    merge_time_h: b.merge_time_h,
                    spill_elev: outlet.spill_elev,
                    cells: outlet.cells.clone(),
                });
            }
        }

        summaries.push(FsmBasinSummary {
            dep_id: b.id,
            parent_id: b.parent_id,
            level: b.level,
            capacity_m3: b.capacity_m3,
            local_cat_area_m2: b.local_cat_area_m2,
            local_eff_cat_area_m2: b.local_eff_cat_area_m2,
            final_routed_cat_area_m2: b.storm_end_routed_cat_m2,
            initial_volume_m3: b.initial_vol_m3,
            final_volume_m3: b.storm_end_vol_m3,
            // Report only fill/spill/merge events realized within the storm; the unbounded
            // post-storm tail segment exists only to give `fill_time.tif` sustained-rainfall
            // timing for deep basins. `realized_spill_to` is always set together with
            // `first_spill_time_h`, so gating on the latter bounds the spill target too.
            // `sustained_merge_time_h` keeps the unbounded merge time for the connectivity graph.
            fill_time_h: b.first_fill_time_h.filter(|&t| t <= storm_end),
            spill_to_dep_id: b
                .first_spill_time_h
                .filter(|&t| t <= storm_end)
                .and(b.realized_spill_to),
            merge_time_h: b.merge_time_h.filter(|&t| t <= storm_end),
            sustained_merge_time_h: b.merge_time_h,
            spilled_volume_m3: b.spilled_vol_storm_m3,
            peak_spill_rate_m3_h: b.peak_spill_rate_m3_h,
            infiltrated_volume_m3: b.infiltrated_vol_storm_m3,
        });
    }

    spill_paths.sort_by(|a, b| {
        a.spill_time_h
            .total_cmp(&b.spill_time_h)
            .then_with(|| a.from_dep_id.cmp(&b.from_dep_id))
    });

    let mut fill_time = Array2D::from_vec(w, h, fill_time_data)?;
    fill_time.copy_metadata_from(sink);
    fill_time.set_no_data(-1.0);

    Ok(FsmSimulationResult {
        fill_time,
        spill_paths,
        basins: summaries,
        mass_balance,
        num_regions: depressions.num_regions as usize,
        num_spill_events,
        num_merge_events,
    })
}

/// Formats per-basin Fill-Spill-Merge routing summaries as a CSV string.
pub fn fsm_basins_to_csv_string(basins: &[FsmBasinSummary]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "dep_id,parent_id,level,capacity_m3,local_cat_area_m2,local_eff_cat_area_m2,final_routed_cat_area_m2,initial_volume_m3,final_volume_m3,fill_time_h,spill_to_dep_id,merge_time_h,spilled_volume_m3,peak_spill_rate_m3_h,infiltrated_volume_m3\n",
    );
    for b in basins {
        let ft = b.fill_time_h.map(|v| format!("{v:.6}")).unwrap_or_default();
        let st = b.spill_to_dep_id.map(|v| v.to_string()).unwrap_or_default();
        let mt = b
            .merge_time_h
            .map(|v| format!("{v:.6}"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4}",
            b.dep_id,
            b.parent_id,
            b.level,
            b.capacity_m3,
            b.local_cat_area_m2,
            b.local_eff_cat_area_m2,
            b.final_routed_cat_area_m2,
            b.initial_volume_m3,
            b.final_volume_m3,
            ft,
            st,
            mt,
            b.spilled_volume_m3,
            b.peak_spill_rate_m3_h,
            b.infiltrated_volume_m3,
        );
    }
    out
}

/// Writes per-basin Fill-Spill-Merge routing summaries to a CSV file.
pub fn write_fsm_basins_csv(basins: &[FsmBasinSummary], path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(fsm_basins_to_csv_string(basins).as_bytes())?;
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filling::extract_sinks_in_memory;
    use crate::slicing::delineate_depressions_in_memory;
    use richdem::Topology;

    fn build_two_sibling_compound_dem() -> (Array2D<f64>, Array2D<f64>, InMemoryDepressions) {
        // 30x16 DEM with a compound depression containing two asymmetric sibling bowls:
        //   - Left bowl A (shallow, smaller capacity -> fills first)
        //   - Right bowl B (deeper, larger capacity -> receives spillover from A)
        // Surrounded by a flat 18.0 m rim sloping inward so all interior cells drain into A or B.
        let (w, h) = (30usize, 16usize);
        let mut z = vec![20.0f64; w * h];
        for r in 1..h - 1 {
            for c in 1..w - 1 {
                let d1 = ((r as f64 - 8.0).powi(2) + (c as f64 - 7.0).powi(2)).sqrt();
                let d2 = ((r as f64 - 8.0).powi(2) + (c as f64 - 21.0).powi(2)).sqrt();
                // Left bowl floor = 13.0, Right bowl floor = 9.0, saddle around ~15.5, outer rim = 18.0
                let z1 = 13.0 + 0.45 * d1;
                let z2 = 9.0 + 0.85 * d2;
                z[r * w + c] = z1.min(z2).min(18.0);
            }
        }
        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

        let sinks = extract_sinks_in_memory(&dem, 10, Topology::D8).unwrap();
        let deps = delineate_depressions_in_memory(&sinks.sink, 10, 0.3, 0.3).unwrap();
        (sinks.dem_refined, sinks.sink, deps)
    }

    #[test]
    fn fsm_two_sibling_cascade_accelerates_downstream_and_conserves_mass() {
        let (dem_refined, sink, deps) = build_two_sibling_compound_dem();
        assert!(deps.max_level >= 2, "expected compound hierarchy");
        assert!(deps.depressions.len() >= 3, "expected parent + 2 children");

        let opts = SimulateInundationOptions {
            min_size: 10,
            min_depth: 0.3,
            interval: 0.3,
            rain_intensity_cm_h: 5.0, // 0.05 m/h
            // Long enough for every basin to fill within the storm (`fill_time_h` is storm-bounded).
            rain_duration_h: 500.0,
            time_step_h: 1.0,
            compress: false,
        };

        let res = simulate_fill_spill_merge_in_memory(
            &dem_refined,
            &sink,
            &deps,
            &opts,
            &InundationForcing::default(),
        )
        .unwrap();

        // Verify exact mass conservation to machine precision.
        let mb = &res.mass_balance;
        assert!(mb.total_runoff_m3 > 0.0);
        assert!(
            mb.mass_balance_error_m3.abs() < 1e-6 * mb.total_runoff_m3.max(1.0),
            "mass balance violated: {mb:?}"
        );

        // Verify spill + merge events occurred.
        assert!(res.num_spill_events >= 1, "expected at least 1 spill event");
        assert!(res.num_merge_events >= 1, "expected at least 1 merge event");

        // Find the two leaf children and verify the slower downstream child filled strictly
        // faster than its isolated static fill time `capacity / (local_cat * rain_m_h)`.
        let rain_m_h = opts.rain_intensity_cm_h / 100.0;
        let mut leaves: Vec<&FsmBasinSummary> =
            res.basins.iter().filter(|b| b.level == 1).collect();
        leaves.sort_by(|a, b| {
            a.fill_time_h
                .unwrap_or(f64::INFINITY)
                .total_cmp(&b.fill_time_h.unwrap_or(f64::INFINITY))
        });
        assert!(leaves.len() >= 2);
        let upstream = leaves[0];
        let downstream = leaves[1];
        let t_up = upstream.fill_time_h.expect("upstream must fill");
        let t_down = downstream.fill_time_h.expect("downstream must fill");
        let static_t_down = downstream.capacity_m3 / (downstream.local_cat_area_m2 * rain_m_h);

        assert!(
            t_up < t_down,
            "upstream ({t_up}) should fill before downstream ({t_down})"
        );
        assert!(
            t_down < static_t_down - 1e-6,
            "downstream fill time with FSM cascade ({t_down:.4} h) must be strictly faster than static isolated fill time ({static_t_down:.4} h)"
        );
        assert!(
            upstream.spilled_volume_m3 > 0.0,
            "upstream basin must spill positive excess volume"
        );
        assert_eq!(upstream.spill_to_dep_id, Some(downstream.dep_id));

        // Verify the parent merged right when the downstream sibling filled.
        let merge_t = upstream.merge_time_h.expect("siblings should merge");
        assert!((merge_t - t_down).abs() < 1e-6);

        // Verify every cell in a leaf has finite fill_time <= its basin's fill_time_h.
        let id_data = deps.depression_id.data();
        let ft_data = res.fill_time.data();
        for (&did, &ft) in id_data.iter().zip(ft_data.iter()) {
            if did == upstream.dep_id as i32 {
                assert!(
                    ft >= 0.0 && ft <= t_up + 1e-6,
                    "upstream cell fill time {ft} exceeded basin fill time {t_up}"
                );
            }
        }
    }

    #[test]
    fn fsm_exact_analytical_hand_calculation_two_basins() {
        // Synthetic exact test: 10x6 DEM with an outer 1-cell border (z=20, dep_id=0) draining
        // inward into an 8x4 compound depression P (id=1, level=2) with two flat-bottomed bowls:
        //   - Basin A (id=2, level=1): 2x2 floor at z=10, saddle at z=12.5 -> V_cap(A) = 4 * 2.5 = 10 m^3
        //   - Basin B (id=3, level=1): 2x2 floor at z=6, saddle at z=12.5 -> V_cap(B) = 4 * 6.5 = 26 m^3
        //   - West half slopes strictly into A, east half slopes strictly into B.
        // With rain = 100 cm/h = 1.0 m/h:
        //   - A fills FIRST at t_a = V_A / Q_A
        //   - A then spills Q_A into B, accelerating B to fill at
        //     t_b = t_a + (V_B - Q_B * t_a) / (Q_A + Q_B)
        //   - A & B merge into P at t_b, and P fills at t_p = t_b + (V_P - V_A - V_B) / (Q_A + Q_B).
        let (w, h) = (10usize, 6usize);
        let mut z = vec![20.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];

        for r in 1..h - 1 {
            for c in 1..w - 1 {
                dep_id[r * w + c] = 1;
                dep_lvl[r * w + c] = 2;
                if (2..=3).contains(&r) && (2..=3).contains(&c) {
                    z[r * w + c] = 10.0;
                    dep_id[r * w + c] = 2;
                    dep_lvl[r * w + c] = 1;
                } else if (2..=3).contains(&r) && (6..=7).contains(&c) {
                    z[r * w + c] = 6.0;
                    dep_id[r * w + c] = 3;
                    dep_lvl[r * w + c] = 1;
                } else if c < 5 {
                    let dist = ((c as f64 - 2.5).powi(2) + (r as f64 - 2.5).powi(2)).sqrt();
                    z[r * w + c] = 12.6 + 0.2 * dist;
                } else {
                    let dist = ((c as f64 - 6.5).powi(2) + (r as f64 - 2.5).powi(2)).sqrt();
                    z[r * w + c] = 12.6 + 0.2 * dist;
                }
            }
        }
        // Lower saddle between A and B at (4, 2) and (5, 2) so A spills directly into B.
        z[2 * w + 4] = 12.5;
        z[2 * w + 5] = 12.4;

        let mut dem = Array2D::from_vec(w, h, z.clone()).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

        let mut sink_data = vec![0.0f64; w * h];
        for i in 0..w * h {
            if dep_id[i] > 0 {
                sink_data[i] = z[i];
            }
        }
        let mut sink = Array2D::from_vec(w, h, sink_data).unwrap();
        sink.set_no_data(0.0);
        sink.geotransform = dem.geotransform;

        let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
        id_arr.set_no_data(0);
        id_arr.geotransform = dem.geotransform;
        let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
        lvl_arr.set_no_data(0);
        lvl_arr.geotransform = dem.geotransform;

        let make_dep = |id: u32,
                        level: u32,
                        parent_id: u32,
                        children_ids: Vec<u32>,
                        min_elev: f64,
                        max_elev: f64,
                        count: usize| NestedDepression {
            id,
            level,
            count,
            area: count as f64,
            volume: (max_elev - min_elev) * count as f64,
            avg_depth: max_elev - min_elev,
            max_depth: max_elev - min_elev,
            min_elev,
            max_elev,
            children_ids,
            region_id: 1,
            perimeter: 8.0,
            major_axis: 2.0,
            minor_axis: 2.0,
            elongatedness: 1.0,
            eccentricity: 0.0,
            orientation: 0.0,
            area_bbox_ratio: 1.0,
            parent_id,
            circularity: 1.0,
            depth_diam_ratio: 0.5,
            hypsometric_integral: 1.0,
            rim_slope_deg: 45.0,
            artifact_flag: false,
        };

        let deps = InMemoryDepressions {
            depression_id: id_arr,
            depression_level: lvl_arr,
            depressions: vec![
                make_dep(1, 2, 0, vec![2, 3], 6.0, 14.0, 32),
                make_dep(2, 1, 1, vec![], 10.0, 12.0, 4),
                make_dep(3, 1, 1, vec![], 6.0, 12.0, 4),
            ],
            max_level: 2,
            num_regions: 1,
        };

        let opts = SimulateInundationOptions {
            min_size: 2,
            min_depth: 0.1,
            interval: 0.5,
            rain_intensity_cm_h: 100.0, // 1.0 m/h
            // Long enough for the parent to fill within the storm (`fill_time_h` is storm-bounded).
            rain_duration_h: 50.0,
            time_step_h: 0.5,
            compress: false,
        };

        let res = simulate_fill_spill_merge_in_memory(
            &dem,
            &sink,
            &deps,
            &opts,
            &InundationForcing::default(),
        )
        .unwrap();

        let p = res.basins.iter().find(|b| b.dep_id == 1).unwrap();
        let a = res.basins.iter().find(|b| b.dep_id == 2).unwrap();
        let b = res.basins.iter().find(|b| b.dep_id == 3).unwrap();

        assert_eq!(a.capacity_m3, 10.0);
        assert_eq!(b.capacity_m3, 26.0);
        assert!(p.capacity_m3 >= a.capacity_m3 + b.capacity_m3);
        assert_eq!(a.spill_to_dep_id, Some(3));

        let t_a = a.fill_time_h.unwrap();
        let t_b = b.fill_time_h.unwrap();
        let t_p = p.fill_time_h.unwrap();

        let q_a = a.local_eff_cat_area_m2;
        let q_b = b.local_eff_cat_area_m2;
        let expected_t_a = a.capacity_m3 / q_a;
        let expected_t_b = t_a + (b.capacity_m3 - q_b * t_a) / (q_a + q_b);
        let expected_t_p =
            expected_t_b + (p.capacity_m3 - a.capacity_m3 - b.capacity_m3) / (q_a + q_b);

        assert!(t_a < t_b, "A ({t_a}) must fill before B ({t_b})");
        assert!(
            (t_a - expected_t_a).abs() < 1e-9,
            "t_a={t_a} vs {expected_t_a}"
        );
        assert!(
            (t_b - expected_t_b).abs() < 1e-9,
            "t_b={t_b} vs {expected_t_b}"
        );
        assert!(
            (t_p - expected_t_p).abs() < 1e-9,
            "t_p={t_p} vs {expected_t_p}"
        );
        assert!(res.mass_balance.mass_balance_error_m3.abs() < 1e-9);
        assert!(res.mass_balance.total_boundary_outflow_m3 > 0.0);
    }

    #[test]
    fn fsm_serial_three_basin_cascade_handles_out_of_order_filling() {
        // Three separate top-level basins terraced downhill from west to east:
        //   A (high, large) -> B (mid, tiny -> fills FIRST) -> C (low, large) -> boundary (0).
        // Because B fills before A, B starts spilling into C first, and when A later fills,
        // A's spillover cascades cleanly through full B straight into C.
        let (w, h) = (19usize, 7usize);
        let mut z = vec![0.0f64; w * h];
        for r in 0..h {
            for c in 0..w {
                let base = 30.0 - (c as f64) + (r as f64 - 3.0).abs() * 0.4;
                z[r * w + c] = base;
            }
        }
        // Basin A around col 3 (medium depth ~3.5 m -> fills SECOND)
        for r in 2..=4 {
            for c in 2..=4 {
                let d = ((r as isize - 3).abs() + (c as isize - 3).abs()) as f64;
                z[r * w + c] = 20.0 + 1.5 * d;
            }
        }
        // Basin B around col 9 (shallow ~0.8 m -> fills FIRST)
        for r in 2..=4 {
            for c in 8..=10 {
                let d = ((r as isize - 3).abs() + (c as isize - 9).abs()) as f64;
                z[r * w + c] = 17.6 + 0.4 * d;
            }
        }
        // Basin C around col 15 (deep ~6.0 m -> fills LAST, accelerated by both B and A)
        for r in 2..=4 {
            for c in 14..=16 {
                let d = ((r as isize - 3).abs() + (c as isize - 15).abs()) as f64;
                z[r * w + c] = 5.0 + 2.5 * d;
            }
        }

        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

        let sinks = extract_sinks_in_memory(&dem, 4, Topology::D8).unwrap();
        let deps = delineate_depressions_in_memory(&sinks.sink, 4, 0.2, 0.2).unwrap();
        assert_eq!(deps.num_regions, 3, "expected 3 separate terraced basins");
        assert_eq!(
            deps.depressions.len(),
            3,
            "expected 3 delineated depressions"
        );

        let opts = SimulateInundationOptions {
            min_size: 4,
            min_depth: 0.2,
            interval: 0.2,
            rain_intensity_cm_h: 50.0, // 0.5 m/h
            rain_duration_h: 10.0,
            time_step_h: 0.5,
            compress: false,
        };
        let res = simulate_fill_spill_merge_in_memory(
            &sinks.dem_refined,
            &sinks.sink,
            &deps,
            &opts,
            &InundationForcing::default(),
        )
        .unwrap();

        let mb = &res.mass_balance;
        assert!(
            mb.mass_balance_error_m3.abs() < 1e-6 * mb.total_runoff_m3.max(1.0),
            "3-basin serial cascade mass balance violated: {mb:?}"
        );
        assert_eq!(
            res.num_spill_events, 3,
            "all 3 terraced basins should spill"
        );
        let rain_m_h = opts.rain_intensity_cm_h / 100.0;
        let deepest = res
            .basins
            .iter()
            .max_by(|a, b| a.capacity_m3.total_cmp(&b.capacity_m3))
            .unwrap();
        let t_c = deepest.fill_time_h.expect("deepest basin C should fill");
        let static_t_c = deepest.capacity_m3 / (deepest.local_cat_area_m2 * rain_m_h);
        assert!(
            t_c < static_t_c - 1e-6,
            "cascaded basin C ({t_c:.4} h) must fill strictly faster than static ({static_t_c:.4} h)"
        );
    }

    #[test]
    fn fsm_catchment_weighted_runoff_and_hyetograph_infiltration_conserve_mass() {
        let (dem_refined, sink, deps) = build_two_sibling_compound_dem();
        let w = dem_refined.width();
        let h = dem_refined.height();

        // Spatially variable runoff_coeff: upland west = 0.9, upland east = 0.25, depression floors = 0.5
        let mut rc_data = vec![0.25f64; w * h];
        for r in 0..h {
            for c in 0..w / 2 {
                rc_data[r * w + c] = 0.9;
            }
        }
        let mut rc = Array2D::from_vec(w, h, rc_data).unwrap();
        rc.copy_metadata_from(&sink);

        // Infiltration = 1.0 cm/h everywhere; antecedent ponding = 0.5 m on the deep floors.
        let mut inf = Array2D::from_vec(w, h, vec![1.0f64; w * h]).unwrap();
        inf.copy_metadata_from(&sink);

        let mut ant_data = vec![0.0f64; w * h];
        for (i, &z) in sink.data().iter().enumerate() {
            if z > 0.0 && z < 11.0 {
                ant_data[i] = 0.5;
            }
        }
        let mut ant = Array2D::from_vec(w, h, ant_data).unwrap();
        ant.copy_metadata_from(&sink);

        // Multi-pulse hyetograph including a completely dry mid-storm spell (to test drainage + refill).
        let hyeto = [
            (2.0, 8.0),
            (3.0, 0.0), // dry spell: infiltration drains stored water
            (4.0, 12.0),
            (2.0, 0.5),
            (4.0, 15.0),
        ];
        let opts = SimulateInundationOptions {
            min_size: 10,
            min_depth: 0.3,
            interval: 0.3,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 15.0,
            time_step_h: 1.0,
            compress: false,
        };
        let forcing = InundationForcing {
            runoff_coeff: Some(&rc),
            infiltration_cm_h: Some(&inf),
            antecedent_depth: Some(&ant),
            hyetograph: Some(&hyeto),
        };

        let res = simulate_fill_spill_merge_in_memory(&dem_refined, &sink, &deps, &opts, &forcing)
            .unwrap();
        let mb = &res.mass_balance;
        assert!(mb.total_antecedent_m3 > 0.0);
        assert!(mb.total_infiltration_m3 > 0.0);
        assert!(
            mb.mass_balance_error_m3.abs()
                < 1e-6 * (mb.total_runoff_m3 + mb.total_antecedent_m3).max(1.0),
            "forced FSM mass balance error too large: {mb:?}"
        );
    }

    #[test]
    fn fsm_upland_runoff_coeff_outside_depression_controls_fill_time() {
        // Single 3x3 depression centered in a 9x9 funnel draining inward.
        // Depression floor has C = 0.1 everywhere, while surrounding upland has C = 1.0 vs C = 0.2.
        // Under catchment-weighted FSM, changing upland C must directly accelerate/delay filling!
        let (w, h) = (9usize, 9usize);
        let mut z = vec![0.0f64; w * h];
        for r in 0..h {
            for c in 0..w {
                let d = ((r as f64 - 4.0).powi(2) + (c as f64 - 4.0).powi(2)).sqrt();
                z[r * w + c] = 10.0 + d;
            }
        }
        // Flatten the inner 3x3 bowl to z = 10.0; the surrounding 72 upland cells slope
        // inward from z = 10.0 + d (12.0 to ~15.7) into the explicitly delineated 3x3 depression.
        for r in 3..=5 {
            for c in 3..=5 {
                z[r * w + c] = 10.0;
            }
        }
        let mut dem = Array2D::from_vec(w, h, z.clone()).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

        let mut sink_data = vec![0.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];
        for r in 3..=5 {
            for c in 3..=5 {
                sink_data[r * w + c] = 10.0;
                dep_id[r * w + c] = 1;
                dep_lvl[r * w + c] = 1;
            }
        }
        let mut sink = Array2D::from_vec(w, h, sink_data).unwrap();
        sink.set_no_data(0.0);
        sink.geotransform = dem.geotransform;
        let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
        id_arr.set_no_data(0);
        id_arr.geotransform = dem.geotransform;
        let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
        lvl_arr.set_no_data(0);
        lvl_arr.geotransform = dem.geotransform;

        let dep = NestedDepression {
            id: 1,
            level: 1,
            count: 9,
            area: 9.0,
            volume: 18.0,
            avg_depth: 2.0,
            max_depth: 2.0,
            min_elev: 10.0,
            max_elev: 12.0,
            children_ids: vec![],
            region_id: 1,
            perimeter: 12.0,
            major_axis: 3.0,
            minor_axis: 3.0,
            elongatedness: 1.0,
            eccentricity: 0.0,
            orientation: 0.0,
            area_bbox_ratio: 1.0,
            parent_id: 0,
            circularity: 1.0,
            depth_diam_ratio: 0.5,
            hypsometric_integral: 1.0,
            rim_slope_deg: 45.0,
            artifact_flag: false,
        };
        let deps = InMemoryDepressions {
            depression_id: id_arr,
            depression_level: lvl_arr,
            depressions: vec![dep],
            max_level: 1,
            num_regions: 1,
        };

        let make_rc = |upland_c: f64| {
            let mut d = vec![upland_c; w * h];
            for r in 3..=5 {
                for c in 3..=5 {
                    d[r * w + c] = 0.1; // identical depression floor C in both runs
                }
            }
            let mut a = Array2D::from_vec(w, h, d).unwrap();
            a.geotransform = dem.geotransform;
            a
        };

        let opts = SimulateInundationOptions {
            min_size: 4,
            min_depth: 0.1,
            interval: 0.5,
            rain_intensity_cm_h: 100.0,
            rain_duration_h: 5.0,
            time_step_h: 0.5,
            compress: false,
        };

        let rc_high = make_rc(1.0);
        let rc_low = make_rc(0.2);
        let res_high = simulate_fill_spill_merge_in_memory(
            &dem,
            &sink,
            &deps,
            &opts,
            &InundationForcing {
                runoff_coeff: Some(&rc_high),
                ..Default::default()
            },
        )
        .unwrap();
        let res_low = simulate_fill_spill_merge_in_memory(
            &dem,
            &sink,
            &deps,
            &opts,
            &InundationForcing {
                runoff_coeff: Some(&rc_low),
                ..Default::default()
            },
        )
        .unwrap();

        let t_high = res_high.basins[0].fill_time_h.unwrap();
        let t_low = res_low.basins[0].fill_time_h.unwrap();
        assert!(
            t_high < t_low * 0.5,
            "higher upland runoff_coeff must dramatically speed up filling (t_high={t_high}, t_low={t_low})"
        );
    }

    #[test]
    fn fsm_rejects_shifted_sink_or_depression_id_grid() {
        let (dem_refined, mut sink, mut deps) = build_two_sibling_compound_dem();
        let opts = SimulateInundationOptions {
            min_size: 10,
            min_depth: 0.3,
            interval: 0.3,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 10.0,
            time_step_h: 1.0,
            compress: false,
        };

        // Shift sink geotransform origin -> must be rejected.
        sink.geotransform[0] += 5.0;
        let err = simulate_fill_spill_merge_in_memory(
            &dem_refined,
            &sink,
            &deps,
            &opts,
            &InundationForcing::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::DimensionMismatch(_)),
            "expected DimensionMismatch for shifted sink, got {err:?}"
        );

        // Restore sink, shift depression_id geotransform origin -> must be rejected.
        sink.geotransform = dem_refined.geotransform;
        deps.depression_id.geotransform[3] += 5.0;
        let err = simulate_fill_spill_merge_in_memory(
            &dem_refined,
            &sink,
            &deps,
            &opts,
            &InundationForcing::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::DimensionMismatch(_)),
            "expected DimensionMismatch for shifted depression_id, got {err:?}"
        );
    }

    #[test]
    fn fsm_spill_into_waiting_compound_parent_routes_to_active_descendant_and_conserves_mass() {
        // Upstream leaf U (id=1) sits on a high west terrace and spills over a saddle directly
        // onto a cell labeled with Waiting compound parent P (id=2, children C1=3, C2=4).
        // Spillover entering P's rim must route down into P's lowest Active child C1 (id=3)
        // rather than being dropped while P is Waiting, conserving mass to machine precision.
        let (w, h) = (13usize, 7usize);
        let mut z = vec![30.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];

        // West upland terrace around U (cols 1..=3, rows 2..=4)
        for r in 1..h - 1 {
            for c in 1..=4 {
                z[r * w + c] = 22.0 + ((r as f64 - 3.0).abs() + (c as f64 - 2.0).abs()) * 0.5;
            }
        }
        // Upstream shallow leaf U (id=1): 1x1 bowl at (2, 3), floor 19.0
        z[3 * w + 2] = 19.0;
        dep_id[3 * w + 2] = 1;
        dep_lvl[3 * w + 2] = 1;
        // Saddle at (3, 3) = 20.0 -> (4, 3) = 19.5 -> enters P rim at (5, 3) = 17.0
        z[3 * w + 3] = 20.0;
        z[3 * w + 4] = 19.5;

        // Compound parent P (id=2, level=2) covers cols 5..=11, rows 1..=5
        for r in 1..h - 1 {
            for c in 5..=11 {
                dep_id[r * w + c] = 2;
                dep_lvl[r * w + c] = 2;
                z[r * w + c] = 16.0 + ((r as f64 - 3.0).abs() + (c as f64 - 8.0).abs()) * 0.4;
            }
        }
        // Child C1 (id=3, deeper floor 5.0) at (7, 3) and Child C2 (id=4, floor 8.0) at (10, 3)
        z[3 * w + 7] = 5.0;
        dep_id[3 * w + 7] = 3;
        dep_lvl[3 * w + 7] = 1;
        z[3 * w + 10] = 8.0;
        dep_id[3 * w + 10] = 4;
        dep_lvl[3 * w + 10] = 1;
        // Ensure (5, 3) flows downhill into (6, 3) -> C1 (7, 3)
        z[3 * w + 5] = 15.0;
        z[3 * w + 6] = 12.0;

        let mut dem = Array2D::from_vec(w, h, z.clone()).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

        let mut sink_data = vec![0.0f64; w * h];
        for i in 0..w * h {
            if dep_id[i] > 0 {
                sink_data[i] = z[i];
            }
        }
        let mut sink = Array2D::from_vec(w, h, sink_data).unwrap();
        sink.set_no_data(0.0);
        sink.geotransform = dem.geotransform;

        let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
        id_arr.set_no_data(0);
        id_arr.geotransform = dem.geotransform;
        let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
        lvl_arr.set_no_data(0);
        lvl_arr.geotransform = dem.geotransform;

        let make_dep = |id: u32,
                        level: u32,
                        parent_id: u32,
                        children_ids: Vec<u32>,
                        region_id: u32,
                        min_elev: f64,
                        max_elev: f64,
                        count: usize| NestedDepression {
            id,
            level,
            count,
            area: count as f64,
            volume: (max_elev - min_elev) * count as f64,
            avg_depth: max_elev - min_elev,
            max_depth: max_elev - min_elev,
            min_elev,
            max_elev,
            children_ids,
            region_id,
            perimeter: 4.0,
            major_axis: 1.0,
            minor_axis: 1.0,
            elongatedness: 1.0,
            eccentricity: 0.0,
            orientation: 0.0,
            area_bbox_ratio: 1.0,
            parent_id,
            circularity: 1.0,
            depth_diam_ratio: 0.5,
            hypsometric_integral: 1.0,
            rim_slope_deg: 45.0,
            artifact_flag: false,
        };

        let deps = InMemoryDepressions {
            depression_id: id_arr,
            depression_level: lvl_arr,
            depressions: vec![
                make_dep(1, 1, 0, vec![], 1, 19.0, 20.0, 1),
                make_dep(2, 2, 0, vec![3, 4], 2, 5.0, 18.0, 35),
                make_dep(3, 1, 2, vec![], 2, 5.0, 15.0, 1),
                make_dep(4, 1, 2, vec![], 2, 8.0, 15.0, 1),
            ],
            max_level: 2,
            num_regions: 2,
        };

        let opts = SimulateInundationOptions {
            min_size: 1,
            min_depth: 0.1,
            interval: 0.5,
            rain_intensity_cm_h: 100.0, // 1.0 m/h
            rain_duration_h: 2.0,       // U fills & spills well before P merges
            time_step_h: 0.5,
            compress: false,
        };

        let res = simulate_fill_spill_merge_in_memory(
            &dem,
            &sink,
            &deps,
            &opts,
            &InundationForcing::default(),
        )
        .unwrap();

        let u = res.basins.iter().find(|b| b.dep_id == 1).unwrap();
        let c1 = res.basins.iter().find(|b| b.dep_id == 3).unwrap();
        assert!(u.spilled_volume_m3 > 0.0, "U must spill during storm");
        assert_eq!(
            u.spill_to_dep_id,
            Some(3),
            "spill targeting Waiting parent P must route to its lowest Active child C1"
        );
        let t_c1 = c1.fill_time_h.expect("C1 should fill");
        let static_t_c1 = c1.capacity_m3 / c1.local_cat_area_m2;
        assert!(
            t_c1 < static_t_c1 - 1e-6,
            "C1 ({t_c1:.4} h) must be accelerated by U's spillover (static {static_t_c1:.4} h)"
        );
        assert!(
            res.mass_balance.mass_balance_error_m3.abs() < 1e-9,
            "mass balance must be exact when spilling into Waiting compound parent: {:?}",
            res.mass_balance
        );
    }

    #[test]
    fn fsm_storm_end_routed_cat_reflects_events_at_storm_end() {
        // Use the two-basin analytical DEM and stop the storm right at the exact fill time of A
        // (`t_a`), so A becomes FullSpilling on the final step of the storm window.
        // At `t = storm_duration_h = t_a`, B's `final_routed_cat_area_m2` must already reflect
        // A's routed contributing area rather than the stale pre-event value.
        let (w, h) = (10usize, 6usize);
        let mut z = vec![20.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];

        for r in 1..h - 1 {
            for c in 1..w - 1 {
                dep_id[r * w + c] = 1;
                dep_lvl[r * w + c] = 2;
                if (2..=3).contains(&r) && (2..=3).contains(&c) {
                    z[r * w + c] = 10.0;
                    dep_id[r * w + c] = 2;
                    dep_lvl[r * w + c] = 1;
                } else if (2..=3).contains(&r) && (6..=7).contains(&c) {
                    z[r * w + c] = 6.0;
                    dep_id[r * w + c] = 3;
                    dep_lvl[r * w + c] = 1;
                } else if c < 5 {
                    let dist = ((c as f64 - 2.5).powi(2) + (r as f64 - 2.5).powi(2)).sqrt();
                    z[r * w + c] = 12.6 + 0.2 * dist;
                } else {
                    let dist = ((c as f64 - 6.5).powi(2) + (r as f64 - 2.5).powi(2)).sqrt();
                    z[r * w + c] = 12.6 + 0.2 * dist;
                }
            }
        }
        z[2 * w + 4] = 12.5;
        z[2 * w + 5] = 12.4;

        let mut dem = Array2D::from_vec(w, h, z.clone()).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

        let mut sink_data = vec![0.0f64; w * h];
        for i in 0..w * h {
            if dep_id[i] > 0 {
                sink_data[i] = z[i];
            }
        }
        let mut sink = Array2D::from_vec(w, h, sink_data).unwrap();
        sink.set_no_data(0.0);
        sink.geotransform = dem.geotransform;

        let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
        id_arr.set_no_data(0);
        id_arr.geotransform = dem.geotransform;
        let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
        lvl_arr.set_no_data(0);
        lvl_arr.geotransform = dem.geotransform;

        let make_dep = |id: u32,
                        level: u32,
                        parent_id: u32,
                        children_ids: Vec<u32>,
                        min_elev: f64,
                        max_elev: f64,
                        count: usize| NestedDepression {
            id,
            level,
            count,
            area: count as f64,
            volume: (max_elev - min_elev) * count as f64,
            avg_depth: max_elev - min_elev,
            max_depth: max_elev - min_elev,
            min_elev,
            max_elev,
            children_ids,
            region_id: 1,
            perimeter: 8.0,
            major_axis: 2.0,
            minor_axis: 2.0,
            elongatedness: 1.0,
            eccentricity: 0.0,
            orientation: 0.0,
            area_bbox_ratio: 1.0,
            parent_id,
            circularity: 1.0,
            depth_diam_ratio: 0.5,
            hypsometric_integral: 1.0,
            rim_slope_deg: 45.0,
            artifact_flag: false,
        };

        let deps = InMemoryDepressions {
            depression_id: id_arr,
            depression_level: lvl_arr,
            depressions: vec![
                make_dep(1, 2, 0, vec![2, 3], 6.0, 14.0, 32),
                make_dep(2, 1, 1, vec![], 10.0, 12.0, 4),
                make_dep(3, 1, 1, vec![], 6.0, 12.0, 4),
            ],
            max_level: 2,
            num_regions: 1,
        };

        // First find exact t_a under rain = 100 cm/h.
        let probe_opts = SimulateInundationOptions {
            min_size: 2,
            min_depth: 0.1,
            interval: 0.5,
            rain_intensity_cm_h: 100.0,
            rain_duration_h: 5.0,
            time_step_h: 0.5,
            compress: false,
        };
        let probe = simulate_fill_spill_merge_in_memory(
            &dem,
            &sink,
            &deps,
            &probe_opts,
            &InundationForcing::default(),
        )
        .unwrap();
        let t_a = probe
            .basins
            .iter()
            .find(|b| b.dep_id == 2)
            .unwrap()
            .fill_time_h
            .unwrap();

        // Now run with rain_duration_h == t_a exactly.
        let exact_opts = SimulateInundationOptions {
            rain_duration_h: t_a,
            time_step_h: t_a,
            ..probe_opts
        };
        let res = simulate_fill_spill_merge_in_memory(
            &dem,
            &sink,
            &deps,
            &exact_opts,
            &InundationForcing::default(),
        )
        .unwrap();
        let a = res.basins.iter().find(|b| b.dep_id == 2).unwrap();
        let b = res.basins.iter().find(|b| b.dep_id == 3).unwrap();
        assert!(
            (b.final_routed_cat_area_m2 - (a.local_cat_area_m2 + b.local_cat_area_m2)).abs() < 1e-9,
            "expected B's storm-end routed_cat ({}) to include A's area ({} + {})",
            b.final_routed_cat_area_m2,
            a.local_cat_area_m2,
            b.local_cat_area_m2
        );
        assert!(res.mass_balance.mass_balance_error_m3.abs() < 1e-9);
    }
}
