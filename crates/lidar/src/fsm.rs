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
//! Sorted hypsometric cell tables store only each depression's own cells (`O(N)` total memory
//! across `N` valid sink pixels, down from `O(N * max_level)`), and the event-driven routing loop
//! tracks only live/waiting basins with reusable buffers to avoid redundant scans and allocations.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, D8Flowdir, D8X, D8Y, DemNum, NO_FLOW, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::catchment::d8_flow_directions_resolved;
use crate::error::{Error, Result};
use crate::inundation::{
    GreenAmptParams, HeapCell, InundationForcing, SimulateInundationOptions, SpillFlowPath,
    affine_metrics, compute_exterior_nodata_mask, green_ampt_infiltration_rate_cm_h,
    solve_green_ampt_ponded, trace_downstream_away_from_source, validate_forcing_grids,
    validate_grid_alignment, validate_options_with_hyetograph,
};
use crate::region::crs_metric_basis;
use crate::slicing::{InMemoryDepressions, NestedDepression};

/// Time tolerance in hours for deciding whether an event falls within the storm duration.
const STORM_TIME_EPS_H: f64 = 1e-9;

/// Upper bound on total Green-Ampt subdivided storm segments to prevent runaway allocations.
const MAX_GREEN_AMPT_SUBSTEPS: usize = 100_000;

/// Upper bound on output bins produced by [`attenuate_fsm_hydrograph_with_options`].
const MAX_ATTENUATION_STEPS: usize = 200_000;

/// Piecewise-linear basin volume segment `(t_start_h, t_end_h, vol_start_m3, vol_end_m3)`
/// used to compute drain-aware inundation durations.
pub(crate) type BasinVolumeSegment = (f64, f64, f64, f64);

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

/// Watershed-scale outlet discharge and dynamic threshold connectivity sample
/// recorded along the Fill-Spill-Merge event timeline.
///
/// Discharge `q_boundary_m3_s` is piecewise-constant on each interval
/// `[pt[k].time_h, pt[k + 1].time_h)`, so the exact watershed boundary outflow volume
/// equals `sum_k q_boundary_m3_s[k] * 3600.0 * (time_h[k + 1] - time_h[k])`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FsmHydrographPoint {
    /// Elapsed simulation time in hours.
    pub time_h: f64,
    /// Total watershed boundary outflow rate in cubic meters per second (`m^3/s`):
    /// `(boundary_eff_cat_area * rain_m_h + q_boundary_spill) / 3600.0`.
    pub q_boundary_m3_s: f64,
    /// Total water volume currently stored across all unmerged depressions (`m^3`).
    pub total_stored_m3: f64,
    /// Fraction of total valid watershed area currently hydrologically connected to
    /// the watershed boundary (`0.0..=1.0`), following Shook et al. (2013) and
    /// Leibowitz et al. (2016) dynamic fill-and-spill threshold connectivity.
    pub connected_area_fraction: f64,
    /// Number of depressions currently in the `FullSpilling` state.
    pub spilling_basin_count: usize,
}

/// Parameters for linear-reservoir / Nash-cascade spill-path travel-time attenuation
/// of an FSM watershed outlet hydrograph.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpillAttenuationOptions {
    /// Mean overland/channel travel time in hours (`>= 0.0`).
    pub mean_travel_time_h: f64,
    /// Uniform output sampling interval in hours (`> 0.0`, default `0.25`).
    pub dt_h: f64,
    /// Number of equal linear reservoirs in series (`>= 1`, default `2` so peaks are
    /// both physically lagged and attenuated while conserving total outflow volume).
    pub num_reservoirs: usize,
    /// Optional trailing recession duration in hours after the storm ends. When `None`,
    /// automatically extends until residual channel storage drops below `1e-12` of total
    /// outflow volume.
    pub tail_duration_h: Option<f64>,
}

impl Default for SpillAttenuationOptions {
    fn default() -> Self {
        Self {
            mean_travel_time_h: 1.0,
            dt_h: 0.25,
            num_reservoirs: 2,
            tail_duration_h: None,
        }
    }
}

/// Flood-hazard raster products derived from a Fill-Spill-Merge simulation.
#[derive(Debug, Clone)]
pub struct FsmHazardRasters {
    /// Peak water depth above ground elevation (`(wse - dem_z).max(0.0)`), `0.0` on dry
    /// valid DEM cells and NoData (`-9999.0`) on NoData DEM cells.
    pub max_depth: Array2D<f64>,
    /// Peak water surface elevation (`wse >= dem_z`) on cells inundated within the storm
    /// window, NoData on dry and NoData DEM cells.
    pub wse: Array2D<f64>,
    /// Total inundated duration in hours (`0.0..=storm_duration_h`) on valid DEM cells,
    /// `0.0` on dry valid cells and NoData (`-1.0`) on NoData DEM cells.
    pub duration_h: Array2D<f64>,
}

/// Extended Fill-Spill-Merge hydrologic forcing options adding physically based
/// Green-Ampt soil infiltration on top of [`InundationForcing`].
#[derive(Debug, Clone, Default)]
pub struct FsmForcing<'a> {
    /// Base spatially variable forcing (runoff coefficients, static infiltration raster,
    /// antecedent depth raster, and piecewise-constant hyetograph).
    pub base: InundationForcing<'a>,
    /// Optional Green-Ampt wetting-front soil infiltration parameters applied dynamically
    /// to depression beds during storm segments.
    pub green_ampt: Option<GreenAmptParams>,
    /// Maximum sub-step duration in hours used to resolve Green-Ampt wetting-front decay
    /// inside long or constant-rainfall segments (`None` defaults to `0.25` h).
    pub green_ampt_substep_h: Option<f64>,
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
    /// Watershed outlet discharge and dynamic hydrologic connectivity time series over
    /// `[0, duration_h]`, recorded at `t = 0.0`, every event / segment boundary, and
    /// storm end with deduplicated strictly increasing timestamps.
    pub hydrograph: Vec<FsmHydrographPoint>,
    /// Number of top-level sink regions.
    pub num_regions: usize,
    /// Number of depressions that filled and initiated overland spilling. Like `spill_paths` and
    /// `fill_time`, this includes events in the constant-rainfall extension past the storm.
    pub num_spill_events: usize,
    /// Number of compound parent depressions formed by merging filled children. Includes events in
    /// the constant-rainfall extension past the storm, like `num_spill_events`.
    pub num_merge_events: usize,
    /// Peak stored volume reached by each basin during `[0, duration_h]` (aligned with `basins`),
    /// used by [`compute_fsm_hazard_rasters`] for exact peak stage inversion even if basins drain.
    pub(crate) basin_peak_vol_storm_m3: Vec<f64>,
    /// Per-basin piecewise-linear volume history `(t0, t1, v0, v1)` within `[0, duration_h]`,
    /// used by [`compute_fsm_hazard_rasters`] for sub-window clipping and drain-aware durations.
    pub(crate) basin_vol_history: Vec<Vec<BasinVolumeSegment>>,
    /// Whether any basin experienced a draining interval (`q_net < 0` with stored water) during
    /// the storm window.
    pub(crate) basin_had_draining: bool,
    /// Time in hours when each basin first became active (`Some(0.0)` for leaves; `Some(t_merge)`
    /// for compound parents that merged during the storm; `None` for parents still `Waiting`).
    pub(crate) basin_activation_time_h: Vec<Option<f64>>,
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
    /// Exact sorted cells owned by this basin (`dep_id == id`), storing each cell's cumulative
    /// subtree storage threshold volume (total memory across all basins is `O(N)`).
    hypso_cells: Vec<HypsoCell>,
    /// Cursor into `hypso_cells` of the next cell not yet inundated.
    hypso_cursor: usize,
    /// Highest volume ever reached by this basin (for monotonic `fill_time` updates).
    max_vol_reached: f64,
    /// Local direct catchment area (for leaves: D8 attributed; for merged parents: sum of children).
    local_cat_area_m2: f64,
    /// Local effective catchment area weighted by `runoff_coeff`.
    local_eff_cat_area_m2: f64,
    /// Total valid sink footprint area of this basin's subtree (`m^2`).
    footprint_area_m2: f64,
    /// Static vertical infiltration loss rate over this basin's footprint (`m^3/h`).
    static_loss_rate_m3_h: f64,
    /// Current dynamic Green-Ampt vertical infiltration loss rate (`m^3/h`).
    ga_loss_rate_m3_h: f64,
    /// Cumulative Green-Ampt infiltrated depth over this basin's bed (`cm`).
    ga_cum_infil_cm: f64,
    /// Total active vertical infiltration loss rate over this basin's footprint (`m^3/h`).
    loss_rate_m3_h: f64,
    /// Dynamic state.
    state: BasinState,
    /// Current stored volume (`m^3`).
    stored_vol_m3: f64,
    /// Initial antecedent volume at `t = 0` (`m^3`).
    initial_vol_m3: f64,
    /// Peak stored volume reached during `[0, storm_duration_h]` (`m^3`).
    storm_peak_vol_m3: f64,
    /// Piecewise-linear `(t0, t1, v0, v1)` volume trajectory within `[0, storm_duration_h]`.
    storm_vol_history: Vec<BasinVolumeSegment>,
    /// Time when this basin first entered an active state (`Some(0.0)` for leaves,
    /// `Some(t_merge)` when a `Waiting` parent merges, `None` while still `Waiting`).
    activation_time_h: Option<f64>,
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
    /// True when one or more merged zero-capacity descendant basins still hold unmarked cells
    /// waiting for this parent to gain positive stored volume.
    has_unmarked_merged_children: bool,
}

/// Reusable grid-sized scratch buffers (~25 bytes per cell) for outward Priority-Flood outlet
/// discovery.
///
/// Checked out from a pool bounded by concurrent worker count (or allocated once sequentially)
/// and reused across depressions via generation stamps so outlet discovery never re-allocates
/// `O(N)` buffers per depression or per Rayon work-stealing split.
pub(crate) struct OutletSearchScratch {
    pub(crate) in_source: Vec<bool>,
    pub(crate) visited_pf: Vec<u32>,
    pub(crate) drains_to_source: Vec<u32>,
    pub(crate) trapped_in_pit: Vec<u32>,
    pub(crate) parent_cell: Vec<usize>,
    pub(crate) trace_seen: Vec<u32>,
    pub(crate) pf_gen: u32,
    pub(crate) trace_gen: u32,
    pub(crate) heap: BinaryHeap<HeapCell>,
}

impl OutletSearchScratch {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            in_source: vec![false; n],
            visited_pf: vec![0u32; n],
            drains_to_source: vec![0u32; n],
            trapped_in_pit: vec![0u32; n],
            // Zero-initialized (uses OS zero pages lazily); source cells are explicitly set
            // to `usize::MAX` and every pushed non-source cell is written before it is read.
            parent_cell: vec![0usize; n],
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
    exterior_nodata: &[bool],
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
                exterior_nodata,
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
            let touches_exterior_nodata = !on_edge
                && ancestor_ids.is_empty()
                && (1..=8).any(|d| {
                    let nr = r as isize + D8Y[d] as isize;
                    let nc = c as isize + D8X[d] as isize;
                    nr >= 0
                        && nc >= 0
                        && nr < h as isize
                        && nc < w as isize
                        && exterior_nodata[(nr as usize) * w + (nc as usize)]
                });
            if on_edge || touches_exterior_nodata {
                // Top-level source basin directly touches the raster edge or an exterior-connected
                // NoData border (never an interior NoData void).
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
    live_indices: &[usize],
    rain_m_h: f64,
    allow_drain_revert: bool,
    q_in: &mut [f64],
    routed_cat: &mut [f64],
    indegree: &mut [usize],
    target_idx_buf: &mut [usize],
    queue: &mut VecDeque<usize>,
) -> f64 {
    loop {
        let mut q_boundary_spill = 0.0;
        let mut spilling_count = 0usize;

        // Zero all entries so non-live basins (`Waiting`/`Merged`) never retain stale values
        // across routing passes or into storm-end snapshots.
        q_in.fill(0.0);
        routed_cat.fill(0.0);

        for &i in live_indices {
            q_in[i] = basins[i].local_eff_cat_area_m2 * rain_m_h;
            routed_cat[i] = basins[i].local_cat_area_m2;
            indegree[i] = 0;
            target_idx_buf[i] = usize::MAX;
            if basins[i].state == BasinState::FullSpilling {
                spilling_count += 1;
            }
        }

        if spilling_count == 0 {
            return 0.0;
        }

        for &i in live_indices {
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

        queue.clear();
        for &i in live_indices {
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
    let fsm_forcing = FsmForcing {
        base: forcing.clone(),
        green_ampt: None,
        green_ampt_substep_h: None,
    };
    simulate_fill_spill_merge_impl(dem_refined, sink, depressions, options, &fsm_forcing, None)
}

/// Runs dynamic event-driven Fill-Spill-Merge routing with extended [`FsmForcing`],
/// including optional physically based Green-Ampt wetting-front soil infiltration.
pub fn simulate_fill_spill_merge_with_green_ampt_in_memory(
    dem_refined: &Array2D<f64>,
    sink: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    options: &SimulateInundationOptions,
    forcing: &FsmForcing<'_>,
) -> Result<FsmSimulationResult> {
    simulate_fill_spill_merge_impl(dem_refined, sink, depressions, options, forcing, None)
}

/// Internal variant of [`simulate_fill_spill_merge_in_memory`] reusing precomputed
/// resolved D8 flow directions (`d8_flow_directions_resolved`) across pipeline stages.
pub(crate) fn simulate_fill_spill_merge_with_flowdirs(
    dem_refined: &Array2D<f64>,
    sink: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    options: &SimulateInundationOptions,
    forcing: &InundationForcing<'_>,
    flowdirs: &Array2D<D8Flowdir>,
) -> Result<FsmSimulationResult> {
    let fsm_forcing = FsmForcing {
        base: forcing.clone(),
        green_ampt: None,
        green_ampt_substep_h: None,
    };
    simulate_fill_spill_merge_with_forcing_and_flowdirs(
        dem_refined,
        sink,
        depressions,
        options,
        &fsm_forcing,
        flowdirs,
    )
}

/// Internal variant accepting full [`FsmForcing`] and precomputed resolved D8 flow directions.
pub(crate) fn simulate_fill_spill_merge_with_forcing_and_flowdirs(
    dem_refined: &Array2D<f64>,
    sink: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    options: &SimulateInundationOptions,
    forcing: &FsmForcing<'_>,
    flowdirs: &Array2D<D8Flowdir>,
) -> Result<FsmSimulationResult> {
    if flowdirs.width() != dem_refined.width() || flowdirs.height() != dem_refined.height() {
        return Err(Error::DimensionMismatch(format!(
            "flowdirs ({}x{}) does not match DEM ({}x{})",
            flowdirs.width(),
            flowdirs.height(),
            dem_refined.width(),
            dem_refined.height()
        )));
    }
    simulate_fill_spill_merge_impl(
        dem_refined,
        sink,
        depressions,
        options,
        forcing,
        Some(flowdirs),
    )
}

fn simulate_fill_spill_merge_impl(
    dem_refined: &Array2D<f64>,
    sink: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    options: &SimulateInundationOptions,
    fsm_forcing: &FsmForcing<'_>,
    precomputed_flowdirs: Option<&Array2D<D8Flowdir>>,
) -> Result<FsmSimulationResult> {
    let forcing = &fsm_forcing.base;
    let green_ampt = fsm_forcing.green_ampt;
    if let Some(ga) = green_ampt {
        ga.validate()?;
    }
    let ga_substep_h = fsm_forcing.green_ampt_substep_h.unwrap_or(0.25);
    if !ga_substep_h.is_finite() || ga_substep_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "green_ampt_substep_h must be finite and > 0, got {ga_substep_h}"
        )));
    }
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
        // Compute direct boundary runoff and hydrograph when no depressions exist.
        let zdata = dem_refined.data();
        let z_nd = dem_refined.no_data();
        let mut eff_valid_area = 0.0f64;
        let mut total_valid_area = 0.0f64;
        for (idx, &z) in zdata.iter().enumerate() {
            if !z.is_finite() || z.rd_eq(z_nd) {
                continue;
            }
            total_valid_area += cell_area;
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
        let conn_frac = if total_valid_area > 0.0 { 1.0 } else { 0.0 };
        let mut hydrograph = Vec::new();
        let mut total_runoff = 0.0f64;
        if let Some(hyeto) = forcing.hyetograph {
            let mut t = 0.0f64;
            let mut last_q = 0.0f64;
            for &(dt, cm_h) in hyeto {
                if t >= storm_duration_h - 1e-12 {
                    break;
                }
                let used = dt.min(storm_duration_h - t);
                let rain_m_h = cm_h / 100.0;
                let q_m3_s = (eff_valid_area * rain_m_h) / 3600.0;
                hydrograph.push(FsmHydrographPoint {
                    time_h: t,
                    q_boundary_m3_s: q_m3_s,
                    total_stored_m3: 0.0,
                    connected_area_fraction: conn_frac,
                    spilling_basin_count: 0,
                });
                total_runoff += eff_valid_area * rain_m_h * used;
                last_q = q_m3_s;
                t += used;
            }
            hydrograph.push(FsmHydrographPoint {
                time_h: storm_duration_h,
                q_boundary_m3_s: last_q,
                total_stored_m3: 0.0,
                connected_area_fraction: conn_frac,
                spilling_basin_count: 0,
            });
        } else {
            let rain_m_h = options.rain_intensity_cm_h / 100.0;
            let q_m3_s = (eff_valid_area * rain_m_h) / 3600.0;
            total_runoff = eff_valid_area * rain_m_h * storm_duration_h;
            hydrograph.push(FsmHydrographPoint {
                time_h: 0.0,
                q_boundary_m3_s: q_m3_s,
                total_stored_m3: 0.0,
                connected_area_fraction: conn_frac,
                spilling_basin_count: 0,
            });
            hydrograph.push(FsmHydrographPoint {
                time_h: storm_duration_h,
                q_boundary_m3_s: q_m3_s,
                total_stored_m3: 0.0,
                connected_area_fraction: conn_frac,
                spilling_basin_count: 0,
            });
        }
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
            hydrograph,
            num_regions: depressions.num_regions as usize,
            num_spill_events: 0,
            num_merge_events: 0,
            basin_peak_vol_storm_m3: Vec::new(),
            basin_vol_history: Vec::new(),
            basin_had_draining: false,
            basin_activation_time_h: Vec::new(),
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

    // 2. Reuse precomputed D8 flow directions on dem_refined, or compute them once (`O(N)`).
    let owned_flowdirs;
    let flowdirs = match precomputed_flowdirs {
        Some(f) => f,
        None => {
            owned_flowdirs = d8_flow_directions_resolved(dem_refined)?;
            &owned_flowdirs
        }
    };
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
    let mut boundary_cat_area = 0.0f64;
    let mut boundary_eff_cat_area = 0.0f64;
    let mut total_valid_area = 0.0f64;

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

        total_valid_area += cell_area;
        if target > 0 {
            local_cat_area[target as usize] += cell_area;
            local_eff_cat_area[target as usize] += cell_area * c_val;
        } else {
            boundary_cat_area += cell_area;
            boundary_eff_cat_area += cell_area * c_val;
        }
    }

    // 4. Discover topographic spill outlets for every depression (parallel when enabled),
    // checking out reusable scratch buffers from a pool strictly bounded by active worker count.
    let exterior_nodata = compute_exterior_nodata_mask(fdata, w, h);
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
                    &exterior_nodata,
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
                    &exterior_nodata,
                    dep_ids,
                    &children_map,
                    &parent_map,
                    &pixels_by_id,
                    &mut scratch,
                )
            })
            .collect()
    };

    // 5. Build O(N) exact hypsometric storage curves, infiltration rates, and antecedent volumes.
    // Every sink pixel belongs to exactly one depression (`pixels_by_id`), so each depression
    // collects and sorts ONLY its own cells (`O(N)` total memory, sorted in parallel with Rayon).
    // Subtree capacities, rim elevations, and infiltration loss rates propagate bottom-up in O(m),
    // and each compound parent's owned rim cells receive their exact subtree `threshold_vol`.
    let m = depressions.depressions.len();
    type OwnedCellData = (Vec<(f64, usize)>, Vec<f64>, f64);
    let collect_owned = |d: &NestedDepression| -> OwnedCellData {
        let mut cells: Vec<(f64, usize)> = Vec::new();
        let mut owned_loss = 0.0f64;
        if let Some(pts) = pixels_by_id.get(&d.id) {
            cells.reserve(pts.len());
            for &gi in pts {
                let sv = sdata[gi];
                let z = if sv.is_finite() && !sv.rd_eq(s_nd) {
                    sv
                } else {
                    zdata[gi]
                };
                if z.is_finite() && !z.rd_eq(z_nd) {
                    cells.push((z, gi));
                    if let Some(inf) = forcing.infiltration_cm_h {
                        let v = inf.get_i(gi);
                        if v.is_finite() && !v.rd_eq(inf.no_data()) && v > 0.0 {
                            owned_loss += (v / 100.0) * cell_area;
                        }
                    }
                }
            }
        }
        cells.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut prefix = Vec::with_capacity(cells.len() + 1);
        prefix.push(0.0f64);
        if let Some(&(z0, _)) = cells.first() {
            let mut acc = 0.0f64;
            for &(z, _) in &cells {
                acc += z - z0;
                prefix.push(acc);
            }
        }
        (cells, prefix, owned_loss)
    };

    #[cfg(feature = "parallel")]
    let owned_data: Vec<OwnedCellData> = depressions
        .depressions
        .par_iter()
        .map(collect_owned)
        .collect();
    #[cfg(not(feature = "parallel"))]
    let owned_data: Vec<OwnedCellData> =
        depressions.depressions.iter().map(collect_owned).collect();

    let mut owned_cells: Vec<Vec<(f64, usize)>> = Vec::with_capacity(m);
    let mut owned_prefix: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut owned_min_z = vec![f64::INFINITY; m];
    let mut owned_max_z = vec![f64::NEG_INFINITY; m];
    let mut subtree_count = vec![0usize; m];
    let mut subtree_min_z = vec![f64::INFINITY; m];
    let mut subtree_max_z = vec![f64::NEG_INFINITY; m];
    let mut subtree_z_sum_above_min = vec![0.0f64; m];
    let mut subtree_loss_rate = vec![0.0f64; m];

    for (idx, (cells, prefix, loss)) in owned_data.into_iter().enumerate() {
        let cnt = cells.len();
        if cnt > 0 {
            let zmin = cells[0].0;
            let zmax = cells[cnt - 1].0;
            owned_min_z[idx] = zmin;
            owned_max_z[idx] = zmax;
            subtree_count[idx] = cnt;
            subtree_min_z[idx] = zmin;
            subtree_max_z[idx] = zmax;
            subtree_z_sum_above_min[idx] = prefix[cnt];
        }
        subtree_loss_rate[idx] = loss;
        owned_cells.push(cells);
        owned_prefix.push(prefix);
    }

    let mut level_order: Vec<usize> = (0..m).collect();
    level_order.sort_by_key(|&idx| {
        (
            depressions.depressions[idx].level,
            depressions.depressions[idx].id,
        )
    });

    // Propagate subtree cell counts, min/max elevations, shifted elevation sums, loss rates,
    // and rim elevations bottom-up so every parent satisfies `z_rim(P) >= max(z_rim(c))` and
    // `V_cap(P) >= sum(V_cap(c))`.
    let mut z_rims = vec![f64::NEG_INFINITY; m];
    for &dep_idx in &level_order {
        let d = &depressions.depressions[dep_idx];
        let mut z_rim = d.max_elev;
        if subtree_max_z[dep_idx].is_finite() {
            z_rim = z_rim.max(subtree_max_z[dep_idx]);
        }
        if let Some(outlet) = &outlets[dep_idx]
            && outlet.spill_elev.is_finite()
        {
            z_rim = z_rim.max(outlet.spill_elev);
        }

        for &cid in &d.children_ids {
            let cidx = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
            if cidx == usize::MAX {
                continue;
            }
            if z_rims[cidx] > z_rim {
                z_rim = z_rims[cidx];
            }
            subtree_loss_rate[dep_idx] += subtree_loss_rate[cidx];
            let n_b = subtree_count[cidx];
            if n_b == 0 {
                continue;
            }
            let n_a = subtree_count[dep_idx];
            if n_a == 0 {
                subtree_count[dep_idx] = n_b;
                subtree_min_z[dep_idx] = subtree_min_z[cidx];
                subtree_max_z[dep_idx] = subtree_max_z[cidx];
                subtree_z_sum_above_min[dep_idx] = subtree_z_sum_above_min[cidx];
            } else {
                let new_min = subtree_min_z[dep_idx].min(subtree_min_z[cidx]);
                subtree_z_sum_above_min[dep_idx] = (subtree_z_sum_above_min[dep_idx]
                    + (n_a as f64) * (subtree_min_z[dep_idx] - new_min))
                    + (subtree_z_sum_above_min[cidx]
                        + (n_b as f64) * (subtree_min_z[cidx] - new_min));
                subtree_min_z[dep_idx] = new_min;
                if subtree_max_z[cidx] > subtree_max_z[dep_idx] {
                    subtree_max_z[dep_idx] = subtree_max_z[cidx];
                }
                subtree_count[dep_idx] = n_a + n_b;
            }
            if subtree_max_z[dep_idx].is_finite() && subtree_max_z[dep_idx] > z_rim {
                z_rim = subtree_max_z[dep_idx];
            }
        }
        z_rims[dep_idx] = z_rim;
    }

    // Build exact per-depression `hypso_cells` containing ONLY owned cells (`dep_id == d.id`)
    // in parallel. Total entries across all depressions equals the number of valid sink cells.
    let build_hypso = |dep_idx: usize| -> Vec<HypsoCell> {
        let d = &depressions.depressions[dep_idx];
        let owned = &owned_cells[dep_idx];
        if owned.is_empty() {
            return Vec::new();
        }
        let mut hypso: Vec<HypsoCell> = Vec::with_capacity(owned.len());
        if d.children_ids.is_empty() {
            let z_base = owned[0].0;
            let pref = &owned_prefix[dep_idx];
            for (i, &(z, gi)) in owned.iter().enumerate() {
                let thresh = if i == 0 {
                    0.0
                } else if (z - owned[i - 1].0).abs() <= 1e-12 {
                    hypso[i - 1].threshold_vol
                } else {
                    ((i as f64) * (z - z_base) - pref[i]).max(0.0) * cell_area
                };
                hypso.push(HypsoCell {
                    grid_idx: gi,
                    threshold_vol: thresh,
                });
            }
            return hypso;
        }

        // Compound parent: fold all descendant subtrees whose max elevation <= z_min_owned
        // in O(1), and track two-pointer cursors only for descendants with overlapping elevations.
        let z_min_owned = owned[0].0;
        let z_base = subtree_min_z[dep_idx];
        let mut base_cnt = 0usize;
        let mut base_sum_above_zbase = 0.0f64;
        let mut overlap_nodes: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = d
            .children_ids
            .iter()
            .filter_map(|&cid| {
                let ci = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                (ci != usize::MAX).then_some(ci)
            })
            .collect();

        while let Some(u) = stack.pop() {
            if subtree_count[u] == 0 {
                continue;
            }
            if subtree_max_z[u] <= z_min_owned {
                base_cnt += subtree_count[u];
                base_sum_above_zbase += subtree_z_sum_above_min[u]
                    + (subtree_count[u] as f64) * (subtree_min_z[u] - z_base);
            } else {
                if !owned_cells[u].is_empty() {
                    if owned_max_z[u] <= z_min_owned {
                        let clen = owned_cells[u].len();
                        base_cnt += clen;
                        base_sum_above_zbase +=
                            owned_prefix[u][clen] + (clen as f64) * (owned_min_z[u] - z_base);
                    } else {
                        overlap_nodes.push(u);
                    }
                }
                for &cid in &depressions.depressions[u].children_ids {
                    let ci = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                    if ci != usize::MAX {
                        stack.push(ci);
                    }
                }
            }
        }

        let mut ptrs = vec![0usize; overlap_nodes.len()];
        let pref_self = &owned_prefix[dep_idx];
        let self_min_shift = owned_min_z[dep_idx] - z_base;

        for (i, &(z, gi)) in owned.iter().enumerate() {
            let thresh = if i > 0 && (z - owned[i - 1].0).abs() <= 1e-12 {
                hypso[i - 1].threshold_vol
            } else {
                let mut overlap_cnt = 0usize;
                let mut overlap_sum_dz = 0.0f64;
                for (k, &u) in overlap_nodes.iter().enumerate() {
                    let u_cells = &owned_cells[u];
                    let mut p = ptrs[k];
                    while p < u_cells.len() && u_cells[p].0 < z {
                        p += 1;
                    }
                    ptrs[k] = p;
                    if p > 0 {
                        overlap_cnt += p;
                        overlap_sum_dz +=
                            owned_prefix[u][p] + (p as f64) * (owned_min_z[u] - z_base);
                    }
                }
                let total_cnt = base_cnt + i + overlap_cnt;
                if total_cnt == 0 {
                    0.0
                } else {
                    let total_sum_dz = base_sum_above_zbase
                        + (pref_self[i] + (i as f64) * self_min_shift)
                        + overlap_sum_dz;
                    ((total_cnt as f64) * (z - z_base) - total_sum_dz).max(0.0) * cell_area
                }
            };
            hypso.push(HypsoCell {
                grid_idx: gi,
                threshold_vol: thresh,
            });
        }
        hypso
    };

    #[cfg(feature = "parallel")]
    let hypso_by_dep: Vec<Vec<HypsoCell>> = (0..m).into_par_iter().map(build_hypso).collect();
    #[cfg(not(feature = "parallel"))]
    let hypso_by_dep: Vec<Vec<HypsoCell>> = (0..m).map(build_hypso).collect();

    drop(owned_cells);
    drop(owned_prefix);

    let mut basins: Vec<BasinNode> = Vec::with_capacity(m);
    let mut total_antecedent_m3 = 0.0f64;

    for (dep_idx, (d, hypso_cells)) in depressions.depressions.iter().zip(hypso_by_dep).enumerate()
    {
        let z_rim = z_rims[dep_idx];
        let capacity_m3 = if subtree_count[dep_idx] > 0 {
            ((subtree_count[dep_idx] as f64) * (z_rim - subtree_min_z[dep_idx])
                - subtree_z_sum_above_min[dep_idx])
                .max(0.0)
                * cell_area
        } else {
            0.0
        };
        let loss_rate_m3_h = subtree_loss_rate[dep_idx];

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
            footprint_area_m2: subtree_count[dep_idx] as f64 * cell_area,
            static_loss_rate_m3_h: loss_rate_m3_h,
            ga_loss_rate_m3_h: 0.0,
            ga_cum_infil_cm: 0.0,
            loss_rate_m3_h,
            state: if is_leaf {
                BasinState::Active
            } else {
                BasinState::Waiting
            },
            stored_vol_m3: owned_ant_vol,
            initial_vol_m3: owned_ant_vol,
            storm_peak_vol_m3: owned_ant_vol,
            storm_vol_history: Vec::new(),
            activation_time_h: is_leaf.then_some(0.0),
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
            has_unmarked_merged_children: false,
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
    // When Green-Ampt infiltration is enabled, in-storm segments are subdivided into
    // `<= ga_substep_h` slices so wetting-front infiltration decay is resolved smoothly.
    let mut raw_segments: Vec<(f64, f64, bool)> = Vec::new();
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
                raw_segments.push((first_dt, rain_m_h, true));
                raw_segments.push((second_dt, rain_m_h, false));
            } else {
                let in_storm = elapsed < storm_duration_h - 1e-12;
                raw_segments.push((dt, rain_m_h, in_storm));
            }
            elapsed += dt;
        }
    } else {
        let rain_m_h = options.rain_intensity_cm_h / 100.0;
        raw_segments.push((storm_duration_h, rain_m_h, true));
        raw_segments.push((f64::INFINITY, rain_m_h, false));
    }
    let mut total_segments = 0usize;
    for &(dt, _, in_storm) in &raw_segments {
        if in_storm && green_ampt.is_some() && dt.is_finite() && dt > ga_substep_h + 1e-12 {
            let n_sub_f = (dt / ga_substep_h).ceil();
            if !n_sub_f.is_finite() || n_sub_f > MAX_GREEN_AMPT_SUBSTEPS as f64 {
                return Err(Error::InvalidArgument(format!(
                    "green_ampt_substep_h={ga_substep_h} requires {n_sub_f} substeps (exceeds limit {MAX_GREEN_AMPT_SUBSTEPS})"
                )));
            }
            total_segments = total_segments.saturating_add(n_sub_f as usize);
        } else {
            total_segments = total_segments.saturating_add(1);
        }
        if total_segments > MAX_GREEN_AMPT_SUBSTEPS {
            return Err(Error::InvalidArgument(format!(
                "Green-Ampt storm subdivision requires > {MAX_GREEN_AMPT_SUBSTEPS} segments; increase green_ampt_substep_h"
            )));
        }
    }
    let mut segments: Vec<(f64, f64, bool, bool)> = Vec::with_capacity(total_segments);
    for (dt, rain_m_h, in_storm) in raw_segments {
        if in_storm && green_ampt.is_some() && dt.is_finite() && dt > ga_substep_h + 1e-12 {
            let n_sub = (dt / ga_substep_h).ceil() as usize;
            let sub_dt = dt / n_sub as f64;
            for _ in 0..n_sub {
                segments.push((sub_dt, rain_m_h, true, false));
            }
        } else {
            segments.push((dt, rain_m_h, in_storm, false));
        }
    }
    if let Some(last_storm) = segments.iter_mut().rev().find(|s| s.2) {
        last_storm.3 = true;
    }

    // Track `live_indices` (Active | FullSpilling) and `waiting_indices` (Waiting) in ascending
    // index order so every event-loop pass skips Merged/Waiting basins with deterministic order.
    let mut live_indices: Vec<usize> = Vec::with_capacity(m);
    let mut waiting_indices: Vec<usize> = Vec::new();
    for (idx, b) in basins.iter().enumerate() {
        match b.state {
            BasinState::Active | BasinState::FullSpilling => live_indices.push(idx),
            BasinState::Waiting => waiting_indices.push(idx),
            BasinState::Merged => {}
        }
    }

    // Helper: resolve all ready parent merges and break any full spill cycles so the
    // `FullSpilling` routing graph is guaranteed to be a DAG.
    let mut cycle_color = vec![0u8; m];
    let mut chain_buf: Vec<usize> = Vec::with_capacity(32);
    let mut q_in = vec![0.0f64; m];
    let mut routed_cat = vec![0.0f64; m];
    let mut indegree = vec![0usize; m];
    let mut target_idx_buf = vec![usize::MAX; m];
    let mut topo_queue: VecDeque<usize> = VecDeque::with_capacity(m.min(256));

    fn flush_unmarked_descendants(
        root_idx: usize,
        basins: &mut [BasinNode],
        id_to_idx: &[usize],
        t_mark: f64,
        fill_time_data: &mut [f64],
    ) {
        let mut stack: Vec<usize> = basins[root_idx]
            .children_ids
            .iter()
            .filter_map(|&cid| {
                let ci = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                (ci != usize::MAX).then_some(ci)
            })
            .collect();
        basins[root_idx].has_unmarked_merged_children = false;
        while let Some(cidx) = stack.pop() {
            if basins[cidx].hypso_cursor < basins[cidx].hypso_cells.len() {
                for cell in &basins[cidx].hypso_cells[basins[cidx].hypso_cursor..] {
                    let slot = &mut fill_time_data[cell.grid_idx];
                    if *slot < 0.0 || t_mark < *slot {
                        *slot = t_mark;
                    }
                }
                basins[cidx].hypso_cursor = basins[cidx].hypso_cells.len();
            }
            if basins[cidx].has_unmarked_merged_children {
                basins[cidx].has_unmarked_merged_children = false;
                for &gcid in &basins[cidx].children_ids {
                    let gidx = id_to_idx.get(gcid as usize).copied().unwrap_or(usize::MAX);
                    if gidx != usize::MAX {
                        stack.push(gidx);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_merges_and_cycles(
        basins: &mut [BasinNode],
        id_to_idx: &[usize],
        live_indices: &mut Vec<usize>,
        waiting_indices: &mut Vec<usize>,
        t_now: f64,
        in_storm_window: bool,
        green_ampt: Option<GreenAmptParams>,
        rem_seg_dt_h: f64,
        num_merge_events: &mut usize,
        fill_time_data: &mut [f64],
        color: &mut [u8],
        chain: &mut Vec<usize>,
        q_in: &mut [f64],
        routed_cat: &mut [f64],
        indegree: &mut [usize],
        target_idx_buf: &mut [usize],
    ) {
        loop {
            let mut changed = false;
            let mut merged_any = false;

            // (a) Check only `Waiting` parents to see if all direct children are FullSpilling/Merged.
            for &idx in waiting_indices.iter() {
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
                let mut child_ga_weighted = 0.0f64;
                for &cid in &child_ids {
                    let cidx = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                    if cidx != usize::MAX && basins[cidx].state == BasinState::FullSpilling {
                        merged_vol += basins[cidx].stored_vol_m3;
                        merged_cat += basins[cidx].local_cat_area_m2;
                        merged_eff_cat += basins[cidx].local_eff_cat_area_m2;
                        child_ga_weighted +=
                            basins[cidx].ga_cum_infil_cm * basins[cidx].footprint_area_m2;
                        basins[cidx].state = BasinState::Merged;
                        basins[cidx].merge_time_h = Some(t_now);
                        q_in[cidx] = 0.0;
                        routed_cat[cidx] = 0.0;
                        indegree[cidx] = 0;
                        target_idx_buf[cidx] = usize::MAX;
                    }
                }
                // Track or immediately flush any un-marked cells of degenerate 0-capacity flat
                // children so they receive exact inundation timestamps once the parent holds water.
                let any_unmarked_child = child_ids.iter().any(|&cid| {
                    let cidx = id_to_idx.get(cid as usize).copied().unwrap_or(usize::MAX);
                    cidx != usize::MAX
                        && (basins[cidx].hypso_cursor < basins[cidx].hypso_cells.len()
                            || basins[cidx].has_unmarked_merged_children)
                });
                if any_unmarked_child {
                    if merged_vol > 0.0 {
                        flush_unmarked_descendants(idx, basins, id_to_idx, t_now, fill_time_data);
                    } else {
                        basins[idx].has_unmarked_merged_children = true;
                    }
                }
                let p = &mut basins[idx];
                p.stored_vol_m3 = merged_vol.min(p.capacity_m3);
                if in_storm_window && p.stored_vol_m3 > p.storm_peak_vol_m3 {
                    p.storm_peak_vol_m3 = p.stored_vol_m3;
                }
                p.local_cat_area_m2 = merged_cat;
                p.local_eff_cat_area_m2 = merged_eff_cat;
                if let Some(ga) = green_ampt {
                    if p.footprint_area_m2 > 0.0 {
                        p.ga_cum_infil_cm = child_ga_weighted / p.footprint_area_m2;
                    }
                    let ga_rate_m_h =
                        if in_storm_window && rem_seg_dt_h.is_finite() && rem_seg_dt_h > 1e-12 {
                            let f_new =
                                solve_green_ampt_ponded(ga, p.ga_cum_infil_cm, rem_seg_dt_h);
                            ((f_new - p.ga_cum_infil_cm) / rem_seg_dt_h) / 100.0
                        } else if in_storm_window && p.ga_cum_infil_cm > 1e-12 {
                            green_ampt_infiltration_rate_cm_h(ga, p.ga_cum_infil_cm) / 100.0
                        } else {
                            ga.ks_cm_h / 100.0
                        };
                    p.ga_loss_rate_m3_h = ga_rate_m_h * p.footprint_area_m2;
                    p.loss_rate_m3_h = p.static_loss_rate_m3_h + p.ga_loss_rate_m3_h;
                }
                p.state = BasinState::Active;
                if p.activation_time_h.is_none() {
                    p.activation_time_h = Some(t_now);
                }
                *num_merge_events += 1;

                mark_cells_at_instant(p, t_now, fill_time_data);
                if p.stored_vol_m3 >= p.capacity_m3 - 1e-9 {
                    p.stored_vol_m3 = p.capacity_m3;
                    if in_storm_window && p.stored_vol_m3 > p.storm_peak_vol_m3 {
                        p.storm_peak_vol_m3 = p.stored_vol_m3;
                    }
                    p.state = BasinState::FullSpilling;
                    if p.first_fill_time_h.is_none() {
                        p.first_fill_time_h = Some(t_now);
                    }
                    mark_cells_at_instant(p, t_now, fill_time_data);
                }
                if basins[idx].state == BasinState::FullSpilling
                    && basins[idx].has_unmarked_merged_children
                {
                    flush_unmarked_descendants(idx, basins, id_to_idx, t_now, fill_time_data);
                }
                changed = true;
                merged_any = true;
            }

            if merged_any {
                waiting_indices.retain(|&i| basins[i].state == BasinState::Waiting);
                live_indices.clear();
                for (i, b) in basins.iter().enumerate() {
                    if matches!(b.state, BasinState::Active | BasinState::FullSpilling) {
                        live_indices.push(i);
                    }
                }
            }

            if changed {
                continue;
            }

            // (b) Detect and break any cycle among `FullSpilling` nodes (scanning only live nodes).
            let mut has_spilling = false;
            for &i in live_indices.iter() {
                color[i] = 0; // 0 = unvisited, 1 = visiting, 2 = done
                if basins[i].state == BasinState::FullSpilling {
                    has_spilling = true;
                }
            }
            if !has_spilling {
                break;
            }

            let mut cycle_nodes: Option<Vec<u32>> = None;
            for &start_idx in live_indices.iter() {
                if basins[start_idx].state != BasinState::FullSpilling || color[start_idx] != 0 {
                    continue;
                }
                chain.clear();
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
                for &ci in chain.iter() {
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
    }

    // Helper: sample watershed outlet hydrograph & dynamic threshold connectivity at `t_sample`.
    // `reach_state`: 0 = unknown, 1 = does not reach outlet, 2 = reaches outlet 0.
    let mut reach_state = vec![0u8; m];
    let mut reach_path: Vec<usize> = Vec::with_capacity(32);
    #[allow(clippy::too_many_arguments)]
    fn record_hydrograph_point(
        hydrograph: &mut Vec<FsmHydrographPoint>,
        t_sample: f64,
        rain_m_h: f64,
        q_boundary_spill: f64,
        basins: &[BasinNode],
        live_indices: &[usize],
        target_idx_buf: &[usize],
        boundary_cat_area: f64,
        boundary_eff_cat_area: f64,
        total_valid_area: f64,
        reach_state: &mut [u8],
        reach_path: &mut Vec<usize>,
    ) {
        let q_boundary_m3_s = (boundary_eff_cat_area * rain_m_h + q_boundary_spill) / 3600.0;
        let mut total_stored_m3 = 0.0f64;
        for b in basins {
            if b.state != BasinState::Merged {
                total_stored_m3 += b.stored_vol_m3;
            }
        }

        let mut spilling_basin_count = 0usize;
        for &i in live_indices {
            reach_state[i] = 0;
            if basins[i].state == BasinState::FullSpilling {
                spilling_basin_count += 1;
            }
        }
        let mut connected_area = boundary_cat_area;
        for &i in live_indices {
            if basins[i].state != BasinState::FullSpilling {
                continue;
            }
            if reach_state[i] == 0 {
                reach_path.clear();
                let mut cur = i;
                let outcome = loop {
                    if cur == usize::MAX {
                        break 2u8;
                    }
                    if basins[cur].state != BasinState::FullSpilling {
                        break 1u8;
                    }
                    if reach_state[cur] != 0 {
                        break reach_state[cur];
                    }
                    if reach_path.contains(&cur) {
                        break 1u8;
                    }
                    reach_path.push(cur);
                    cur = target_idx_buf[cur];
                };
                for &node in reach_path.iter() {
                    reach_state[node] = outcome;
                }
            }
            if reach_state[i] == 2 {
                connected_area += basins[i].local_cat_area_m2;
            }
        }
        let connected_area_fraction = if total_valid_area > 0.0 {
            (connected_area / total_valid_area).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let pt = FsmHydrographPoint {
            time_h: t_sample,
            q_boundary_m3_s,
            total_stored_m3,
            connected_area_fraction,
            spilling_basin_count,
        };
        if let Some(last) = hydrograph.last_mut()
            && (last.time_h - t_sample).abs() <= 1e-12
        {
            *last = pt;
        } else {
            hydrograph.push(pt);
        }
    }

    // 7. Event-driven integration loop.
    let mut t_cur = 0.0f64;
    let mut total_runoff_storm_m3 = 0.0f64;
    let mut total_infiltration_storm_m3 = 0.0f64;
    let mut total_boundary_outflow_storm_m3 = 0.0f64;
    let mut num_merge_events = 0usize;
    let mut snapshot_taken = false;
    let mut last_rain_m_h = 0.0f64;
    let mut last_storm_seg_dt = ga_substep_h;
    let mut topology_dirty = true;
    let mut hydrograph: Vec<FsmHydrographPoint> = Vec::with_capacity(segments.len() + 8);
    let mut any_basin_drained_in_storm = false;

    for (seg_idx, (seg_dt, rain_m_h, in_storm_window, snapshot_after_segment)) in
        segments.into_iter().enumerate()
    {
        if in_storm_window && seg_dt.is_finite() && seg_dt > 0.0 {
            last_storm_seg_dt = seg_dt;
        }
        if seg_idx > 0 && rain_m_h < last_rain_m_h - 1e-15 {
            for &i in &live_indices {
                basins[i].spill_override = None;
            }
            topology_dirty = true;
        }
        if let Some(ga) = green_ampt {
            let mut any_loss_increased = false;
            for &i in &live_indices {
                let b = &mut basins[i];
                let prev_loss = b.loss_rate_m3_h;
                let ga_rate_m_h = if in_storm_window && seg_dt.is_finite() {
                    let f_new = solve_green_ampt_ponded(ga, b.ga_cum_infil_cm, seg_dt);
                    ((f_new - b.ga_cum_infil_cm) / seg_dt) / 100.0
                } else {
                    ga.ks_cm_h / 100.0
                };
                b.ga_loss_rate_m3_h = ga_rate_m_h * b.footprint_area_m2;
                b.loss_rate_m3_h = b.static_loss_rate_m3_h + b.ga_loss_rate_m3_h;
                if b.loss_rate_m3_h > prev_loss + 1e-15 {
                    any_loss_increased = true;
                }
            }
            if any_loss_increased {
                for &i in &live_indices {
                    basins[i].spill_override = None;
                }
                topology_dirty = true;
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

            if topology_dirty {
                resolve_merges_and_cycles(
                    &mut basins,
                    &id_to_idx,
                    &mut live_indices,
                    &mut waiting_indices,
                    t_cur,
                    in_storm_window,
                    green_ampt,
                    t_seg_end - t_cur,
                    &mut num_merge_events,
                    &mut fill_time_data,
                    &mut cycle_color,
                    &mut chain_buf,
                    &mut q_in,
                    &mut routed_cat,
                    &mut indegree,
                    &mut target_idx_buf,
                );
                topology_dirty = false;
            }

            // Evaluate topological spill inflows; if a FullSpilling basin loses more to
            // infiltration than it receives during a dry/low-rain step, revert it to Active
            // so it drains cleanly rather than spilling negative water.
            let q_boundary_spill = evaluate_topological_routing(
                &mut basins,
                &id_to_idx,
                &live_indices,
                rain_m_h,
                true,
                &mut q_in,
                &mut routed_cat,
                &mut indegree,
                &mut target_idx_buf,
                &mut topo_queue,
            );

            if in_storm_window {
                record_hydrograph_point(
                    &mut hydrograph,
                    t_cur,
                    rain_m_h,
                    q_boundary_spill,
                    &basins,
                    &live_indices,
                    &target_idx_buf,
                    boundary_cat_area,
                    boundary_eff_cat_area,
                    total_valid_area,
                    &mut reach_state,
                    &mut reach_path,
                );
            }

            // Record realized spill targets, determine next event time step `dt_next`, and
            // sum live effective catchment area in a single pass over `live_indices`.
            let mut dt_next = t_seg_end - t_cur;
            let mut any_filling = false;
            let mut total_local_eff_cat = boundary_eff_cat_area;

            for &i in &live_indices {
                total_local_eff_cat += basins[i].local_eff_cat_area_m2;
                match basins[i].state {
                    BasinState::FullSpilling => {
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
                    BasinState::Active => {
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
                    BasinState::Waiting | BasinState::Merged => {}
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

            // Advance all live basins by `dt_next`.
            let t_start = t_cur;
            let t_end = t_cur + dt_next;

            if in_storm_window {
                total_runoff_storm_m3 += total_local_eff_cat * rain_m_h * dt_next;
                total_boundary_outflow_storm_m3 +=
                    (boundary_eff_cat_area * rain_m_h + q_boundary_spill) * dt_next;
            }

            for &i in &live_indices {
                match basins[i].state {
                    BasinState::Active => {
                        let q_net = q_in[i] - basins[i].loss_rate_m3_h;
                        let vol_old = basins[i].stored_vol_m3;
                        let vol_new = (vol_old + q_net * dt_next).clamp(0.0, basins[i].capacity_m3);
                        basins[i].stored_vol_m3 = vol_new;

                        if in_storm_window {
                            if vol_new > basins[i].storm_peak_vol_m3 {
                                basins[i].storm_peak_vol_m3 = vol_new;
                            }
                            if q_net < -1e-15 && vol_old > 1e-12 {
                                any_basin_drained_in_storm = true;
                            }
                            if dt_next > 0.0 {
                                if q_net < -1e-15 && vol_old > 0.0 && vol_new <= 0.0 {
                                    let t_empty = (t_start + vol_old / (-q_net)).min(t_end);
                                    if t_empty > t_start {
                                        basins[i]
                                            .storm_vol_history
                                            .push((t_start, t_empty, vol_old, 0.0));
                                    }
                                    if t_end > t_empty {
                                        basins[i]
                                            .storm_vol_history
                                            .push((t_empty, t_end, 0.0, 0.0));
                                    }
                                } else {
                                    basins[i]
                                        .storm_vol_history
                                        .push((t_start, t_end, vol_old, vol_new));
                                }
                            }
                        }

                        if in_storm_window && basins[i].loss_rate_m3_h > 0.0 {
                            // Exact actual infiltration: `Q_in * dt - (vol_new - vol_old)`.
                            let actual_infil = (q_in[i] * dt_next - (vol_new - vol_old)).max(0.0);
                            basins[i].infiltrated_vol_storm_m3 += actual_infil;
                            total_infiltration_storm_m3 += actual_infil;
                            if basins[i].ga_loss_rate_m3_h > 0.0
                                && basins[i].footprint_area_m2 > 0.0
                            {
                                let ga_share =
                                    basins[i].ga_loss_rate_m3_h / basins[i].loss_rate_m3_h;
                                basins[i].ga_cum_infil_cm +=
                                    (actual_infil * ga_share / basins[i].footprint_area_m2) * 100.0;
                            }
                        }

                        if q_net > 1e-15 {
                            if basins[i].has_unmarked_merged_children && vol_new > 0.0 {
                                flush_unmarked_descendants(
                                    i,
                                    &mut basins,
                                    &id_to_idx,
                                    t_start,
                                    &mut fill_time_data,
                                );
                            }
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
                            if in_storm_window
                                && basins[i].stored_vol_m3 > basins[i].storm_peak_vol_m3
                            {
                                basins[i].storm_peak_vol_m3 = basins[i].stored_vol_m3;
                            }
                            basins[i].state = BasinState::FullSpilling;
                            topology_dirty = true;
                            if basins[i].first_fill_time_h.is_none() {
                                basins[i].first_fill_time_h = Some(t_end);
                            }
                            mark_cells_at_instant(&mut basins[i], t_end, &mut fill_time_data);
                        }
                    }
                    BasinState::FullSpilling => {
                        if in_storm_window {
                            if basins[i].capacity_m3 > basins[i].storm_peak_vol_m3 {
                                basins[i].storm_peak_vol_m3 = basins[i].capacity_m3;
                            }
                            if dt_next > 0.0 {
                                let cap = basins[i].capacity_m3;
                                basins[i].storm_vol_history.push((t_start, t_end, cap, cap));
                            }
                            let actual_infil = basins[i].loss_rate_m3_h.min(q_in[i]) * dt_next;
                            let q_out = (q_in[i] - basins[i].loss_rate_m3_h).max(0.0);
                            basins[i].infiltrated_vol_storm_m3 += actual_infil;
                            basins[i].spilled_vol_storm_m3 += q_out * dt_next;
                            total_infiltration_storm_m3 += actual_infil;
                            if basins[i].ga_loss_rate_m3_h > 0.0
                                && basins[i].loss_rate_m3_h > 0.0
                                && basins[i].footprint_area_m2 > 0.0
                            {
                                let ga_share =
                                    basins[i].ga_loss_rate_m3_h / basins[i].loss_rate_m3_h;
                                basins[i].ga_cum_infil_cm +=
                                    (actual_infil * ga_share / basins[i].footprint_area_m2) * 100.0;
                            }
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
            if topology_dirty {
                resolve_merges_and_cycles(
                    &mut basins,
                    &id_to_idx,
                    &mut live_indices,
                    &mut waiting_indices,
                    t_cur,
                    true,
                    green_ampt,
                    last_storm_seg_dt,
                    &mut num_merge_events,
                    &mut fill_time_data,
                    &mut cycle_color,
                    &mut chain_buf,
                    &mut q_in,
                    &mut routed_cat,
                    &mut indegree,
                    &mut target_idx_buf,
                );
                topology_dirty = false;
            }
            let q_boundary_spill_end = evaluate_topological_routing(
                &mut basins,
                &id_to_idx,
                &live_indices,
                rain_m_h,
                false,
                &mut q_in,
                &mut routed_cat,
                &mut indegree,
                &mut target_idx_buf,
                &mut topo_queue,
            );
            record_hydrograph_point(
                &mut hydrograph,
                storm_duration_h,
                rain_m_h,
                q_boundary_spill_end,
                &basins,
                &live_indices,
                &target_idx_buf,
                boundary_cat_area,
                boundary_eff_cat_area,
                total_valid_area,
                &mut reach_state,
                &mut reach_path,
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
        if topology_dirty {
            resolve_merges_and_cycles(
                &mut basins,
                &id_to_idx,
                &mut live_indices,
                &mut waiting_indices,
                t_cur,
                true,
                green_ampt,
                last_storm_seg_dt,
                &mut num_merge_events,
                &mut fill_time_data,
                &mut cycle_color,
                &mut chain_buf,
                &mut q_in,
                &mut routed_cat,
                &mut indegree,
                &mut target_idx_buf,
            );
        }
        let q_boundary_spill_end = evaluate_topological_routing(
            &mut basins,
            &id_to_idx,
            &live_indices,
            last_rain_m_h,
            false,
            &mut q_in,
            &mut routed_cat,
            &mut indegree,
            &mut target_idx_buf,
            &mut topo_queue,
        );
        record_hydrograph_point(
            &mut hydrograph,
            storm_duration_h,
            last_rain_m_h,
            q_boundary_spill_end,
            &basins,
            &live_indices,
            &target_idx_buf,
            boundary_cat_area,
            boundary_eff_cat_area,
            total_valid_area,
            &mut reach_state,
            &mut reach_path,
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

    let basin_peak_vol_storm_m3: Vec<f64> = basins.iter().map(|b| b.storm_peak_vol_m3).collect();
    let basin_activation_time_h: Vec<Option<f64>> = basins
        .iter()
        .map(|b| b.activation_time_h.filter(|&t| t <= storm_end))
        .collect();
    let basin_vol_history: Vec<Vec<BasinVolumeSegment>> =
        basins.into_iter().map(|b| b.storm_vol_history).collect();

    Ok(FsmSimulationResult {
        fill_time,
        spill_paths,
        basins: summaries,
        mass_balance,
        hydrograph,
        num_regions: depressions.num_regions as usize,
        num_spill_events,
        num_merge_events,
        basin_peak_vol_storm_m3,
        basin_vol_history,
        basin_had_draining: any_basin_drained_in_storm,
        basin_activation_time_h,
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

/// Integrates total boundary outflow volume (in cubic meters) from a piecewise-constant
/// FSM hydrograph: `sum_{k} pt[k].q_boundary_m3_s * 3600.0 * (pt[k + 1].time_h - pt[k].time_h)`.
pub fn integrate_fsm_hydrograph_m3(hydrograph: &[FsmHydrographPoint]) -> f64 {
    hydrograph
        .windows(2)
        .map(|w| {
            let dt_h = (w[1].time_h - w[0].time_h).max(0.0);
            w[0].q_boundary_m3_s * 3600.0 * dt_h
        })
        .sum()
}

/// Formats an FSM watershed outlet hydrograph and dynamic connectivity curve as a CSV string.
///
/// Uses full floating-point round-trip formatting for `time_h`, `q_boundary_m3_s`, and
/// `total_stored_m3` so small positive discharges (e.g., `2.78e-7 m^3/s`) are never
/// truncated to zero.
pub fn fsm_hydrograph_to_csv_string(hydrograph: &[FsmHydrographPoint]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "time_h,q_boundary_m3_s,total_stored_m3,connected_area_fraction,spilling_basin_count\n",
    );
    for pt in hydrograph {
        let _ = writeln!(
            out,
            "{},{},{},{:.6},{}",
            pt.time_h,
            pt.q_boundary_m3_s,
            pt.total_stored_m3,
            pt.connected_area_fraction,
            pt.spilling_basin_count,
        );
    }
    out
}

/// Writes an FSM watershed outlet hydrograph and dynamic connectivity curve to a CSV file.
pub fn write_fsm_hydrograph_csv(
    hydrograph: &[FsmHydrographPoint],
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(fsm_hydrograph_to_csv_string(hydrograph).as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Estimates mean overland spill-path travel time in hours from a set of delineated
/// [`SpillFlowPath`] polylines, the DEM's CRS (`projection`), affine `geotransform`,
/// raster dimensions (`width`, `height`), and a characteristic overland/channel flow
/// velocity (`velocity_m_s > 0.0`).
///
/// Uses the same CRS-aware metric basis as [`crate::region::crs_metric_scale`] so metric
/// projected CRSs, foot-based projected CRSs (e.g. State Plane US survey feet), rotated
/// affine grids, and geographic CRSs (`EPSG:4326`, `EPSG:4269`) all measure path lengths
/// in SI meters. Returns `0.0` when `spill_paths` is empty.
pub fn estimate_mean_travel_time_h(
    spill_paths: &[SpillFlowPath],
    projection: &str,
    geotransform: &[f64; 6],
    width: usize,
    height: usize,
    velocity_m_s: f64,
) -> Result<f64> {
    if !velocity_m_s.is_finite() || velocity_m_s <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "velocity_m_s must be finite and > 0, got {velocity_m_s}"
        )));
    }
    if spill_paths.is_empty() {
        return Ok(0.0);
    }
    let (vx_m, vy_m, _) = crs_metric_basis(projection, geotransform, width, height);
    let mut total_len_m = 0.0f64;
    let mut count = 0usize;
    for sp in spill_paths {
        if sp.cells.len() < 2 {
            continue;
        }
        let mut len_m = 0.0f64;
        for w in sp.cells.windows(2) {
            let dc = w[1].0 as f64 - w[0].0 as f64;
            let dr = w[1].1 as f64 - w[0].1 as f64;
            let dx_m = dc * vx_m.0 + dr * vy_m.0;
            let dy_m = dc * vx_m.1 + dr * vy_m.1;
            len_m += dx_m.hypot(dy_m);
        }
        total_len_m += len_m;
        count += 1;
    }
    if count == 0 {
        return Ok(0.0);
    }
    let mean_len_m = total_len_m / count as f64;
    Ok(mean_len_m / (velocity_m_s * 3600.0))
}

/// Applies volume-conserving Nash linear-reservoir travel-time attenuation to an FSM
/// watershed outlet hydrograph using `mean_travel_time_h` and uniform step `dt_h`
/// (defaults to a 2-reservoir cascade for physical peak lag and attenuation).
pub fn attenuate_fsm_hydrograph(
    hydrograph: &[FsmHydrographPoint],
    mean_travel_time_h: f64,
    dt_h: f64,
) -> Result<Vec<FsmHydrographPoint>> {
    attenuate_fsm_hydrograph_with_options(
        hydrograph,
        &SpillAttenuationOptions {
            mean_travel_time_h,
            dt_h,
            ..Default::default()
        },
    )
}

/// Applies volume-conserving Nash linear-reservoir travel-time attenuation to an FSM
/// watershed outlet hydrograph with configurable reservoir count and recession tail.
///
/// For `num_reservoirs == 1`, each uniform bin `[t_k, t_k + dt_h]` integrates the linear
/// reservoir ODE analytically against exact bin inflow volume. For `num_reservoirs > 1`,
/// reservoirs are routed sequentially with piecewise-constant bin-averaged coupling
/// between stages, guaranteeing strict non-negativity, unconditional stability, and exact
/// conservation of total outflow volume (`integrate_fsm_hydrograph_m3`) across all `dt_h > 0`
/// while converging to the continuous Nash cascade as `dt_h -> 0`.
///
/// Returns [`Error::InvalidArgument`] if the required number of bins exceeds
/// `200_000` (e.g., for very small `dt_h` relative to storm duration and recession tail).
pub fn attenuate_fsm_hydrograph_with_options(
    hydrograph: &[FsmHydrographPoint],
    options: &SpillAttenuationOptions,
) -> Result<Vec<FsmHydrographPoint>> {
    if !options.mean_travel_time_h.is_finite() || options.mean_travel_time_h < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "mean_travel_time_h must be finite and >= 0, got {}",
            options.mean_travel_time_h
        )));
    }
    if !options.dt_h.is_finite() || options.dt_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "dt_h must be finite and > 0, got {}",
            options.dt_h
        )));
    }
    if options.num_reservoirs == 0 {
        return Err(Error::InvalidArgument("num_reservoirs must be >= 1".into()));
    }
    if let Some(tail) = options.tail_duration_h
        && (!tail.is_finite() || tail < 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "tail_duration_h must be finite and >= 0, got {tail}"
        )));
    }
    if hydrograph.len() < 2 || options.mean_travel_time_h == 0.0 {
        return Ok(hydrograph.to_vec());
    }

    let t0 = hydrograph[0].time_h;
    let t_storm_end = hydrograph.last().unwrap().time_h;
    if t_storm_end <= t0 {
        return Ok(hydrograph.to_vec());
    }
    let dt = options.dt_h;
    let n_res = options.num_reservoirs;
    let k_res = options.mean_travel_time_h / n_res as f64;
    let total_in_vol = integrate_fsm_hydrograph_m3(hydrograph);

    let auto_tail_h = (30.0 * options.mean_travel_time_h).max(dt);
    let max_t_end = t_storm_end + options.tail_duration_h.unwrap_or(auto_tail_h);
    let raw_steps = ((max_t_end - t0) / dt).ceil();
    if !raw_steps.is_finite() || raw_steps > MAX_ATTENUATION_STEPS as f64 {
        return Err(Error::InvalidArgument(format!(
            "attenuation requires {raw_steps} steps (exceeds limit {MAX_ATTENUATION_STEPS}); increase dt_h or reduce tail_duration_h"
        )));
    }
    let max_steps = (raw_steps as usize).max(1);
    let vol_tol = 1e-12 * total_in_vol.max(1.0);

    // Advancing cursors keep bin integration and state sampling O(N + M) overall.
    let mut in_cursor = 0usize;
    let mut state_cursor = 0usize;
    let last_in_idx = hydrograph.len() - 1;

    let mut storage = vec![0.0f64; n_res];
    let mut out: Vec<FsmHydrographPoint> = Vec::with_capacity(max_steps.min(4096) + 2);
    let mut t_cur = t0;

    for step_idx in 0..max_steps {
        let t_next = if options.tail_duration_h.is_some() && step_idx + 1 == max_steps {
            max_t_end
        } else {
            (t_cur + dt).min(max_t_end)
        };
        let step_dt = t_next - t_cur;
        if step_dt <= 1e-15 {
            break;
        }

        // Sample start-of-bin watershed state and channel storage at `t_cur`.
        while state_cursor < last_in_idx && hydrograph[state_cursor + 1].time_h <= t_cur + 1e-12 {
            state_cursor += 1;
        }
        let p_state = &hydrograph[state_cursor];
        let channel_storage_start: f64 = storage.iter().sum();

        // Integrate input hydrograph over [t_cur, t_next] using advancing cursor.
        while in_cursor < last_in_idx && hydrograph[in_cursor + 1].time_h <= t_cur {
            in_cursor += 1;
        }
        let mut vol_through = 0.0f64;
        if t_cur < t_storm_end && t_next > t0 {
            let mut k = in_cursor;
            while k < last_in_idx {
                let s0 = hydrograph[k].time_h;
                if s0 >= t_next {
                    break;
                }
                let s1 = hydrograph[k + 1].time_h;
                let ov = s1.min(t_next) - s0.max(t_cur);
                if ov > 0.0 {
                    vol_through += hydrograph[k].q_boundary_m3_s * 3600.0 * ov;
                }
                k += 1;
            }
        }

        let alpha = (-step_dt / k_res).exp();
        for s in &mut storage {
            let inflow_rate = vol_through / step_dt; // m^3/h
            let s_old = *s;
            let s_new = s_old * alpha + inflow_rate * k_res * (1.0 - alpha);
            let v_out = (vol_through + s_old - s_new).max(0.0);
            *s = s_new.max(0.0);
            vol_through = v_out;
        }
        let q_step_m3_s = vol_through / (3600.0 * step_dt);

        out.push(FsmHydrographPoint {
            time_h: t_cur,
            q_boundary_m3_s: q_step_m3_s,
            total_stored_m3: p_state.total_stored_m3 + channel_storage_start,
            connected_area_fraction: p_state.connected_area_fraction,
            spilling_basin_count: p_state.spilling_basin_count,
        });
        t_cur = t_next;

        let channel_storage_end: f64 = storage.iter().sum();
        if options.tail_duration_h.is_none()
            && t_cur >= t_storm_end - 1e-12
            && channel_storage_end <= vol_tol
        {
            break;
        }
    }

    let channel_storage_final: f64 = storage.iter().sum();
    if options.tail_duration_h.is_none() && channel_storage_final > vol_tol {
        return Err(Error::InvalidArgument(format!(
            "attenuation recession did not converge within {MAX_ATTENUATION_STEPS} steps (residual channel storage {channel_storage_final:.4} m^3); increase dt_h"
        )));
    }
    while state_cursor < last_in_idx && hydrograph[state_cursor + 1].time_h <= t_cur + 1e-12 {
        state_cursor += 1;
    }
    let p_end = &hydrograph[state_cursor];
    let last_q = out.last().map_or(0.0, |p| p.q_boundary_m3_s);
    out.push(FsmHydrographPoint {
        time_h: t_cur,
        q_boundary_m3_s: if t_cur > t_storm_end + 1e-12 {
            0.0
        } else {
            last_q
        },
        total_stored_m3: p_end.total_stored_m3 + channel_storage_final,
        connected_area_fraction: p_end.connected_area_fraction,
        spilling_basin_count: p_end.spilling_basin_count,
    });
    Ok(out)
}

/// Evaluates peak stored volume along a piecewise-linear volume history clipped strictly
/// to `[0.0, window_end_h]`.
fn peak_vol_from_history(
    history: &[BasinVolumeSegment],
    window_end_h: f64,
    initial_vol: f64,
) -> f64 {
    let mut peak = initial_vol.max(0.0);
    for &(t0, t1, v0, v1) in history {
        if t0 >= window_end_h {
            break;
        }
        let dt = t1 - t0;
        if dt <= 0.0 {
            continue;
        }
        if v0 > peak {
            peak = v0;
        }
        let v_at_end = if t1 <= window_end_h {
            v1
        } else {
            let frac = ((window_end_h - t0) / dt).clamp(0.0, 1.0);
            v0 + frac * (v1 - v0)
        };
        if v_at_end > peak {
            peak = v_at_end;
        }
    }
    peak
}

/// Computes exact wet duration (in hours) spent at or above storage threshold `thresh_vol`
/// along a piecewise-linear volume history `[(t0, t1, v0, v1)]`, clipped to `[0.0, window_end_h]`.
fn wet_duration_from_history(
    history: &[BasinVolumeSegment],
    thresh_vol: f64,
    window_end_h: f64,
) -> f64 {
    let mut dur = 0.0f64;
    for &(t0, t1, v0, v1) in history {
        if t0 >= window_end_h {
            break;
        }
        let full_dt = t1 - t0;
        if full_dt <= 0.0 {
            continue;
        }
        let (eff_t1, eff_v1) = if t1 <= window_end_h {
            (t1, v1)
        } else {
            let frac = ((window_end_h - t0) / full_dt).clamp(0.0, 1.0);
            (window_end_h, v0 + frac * (v1 - v0))
        };
        let dt = (eff_t1 - t0).max(0.0);
        if dt <= 0.0 {
            continue;
        }
        let wet0 = if thresh_vol <= 0.0 {
            v0 > 1e-12
        } else {
            v0 >= thresh_vol - 1e-12
        };
        let wet1 = if thresh_vol <= 0.0 {
            eff_v1 > 1e-12
        } else {
            eff_v1 >= thresh_vol - 1e-12
        };
        if wet0 && wet1 {
            dur += dt;
        } else if !wet0 && wet1 {
            let dv = eff_v1 - v0;
            if dv > 1e-15 {
                let frac = ((eff_v1 - thresh_vol) / dv).clamp(0.0, 1.0);
                dur += dt * frac;
            }
        } else if wet0 && !wet1 {
            let dv = v0 - eff_v1;
            if dv > 1e-15 {
                let frac = ((v0 - thresh_vol) / dv).clamp(0.0, 1.0);
                dur += dt * frac;
            }
        }
    }
    dur
}

/// Computes flood-hazard raster products (`max_depth`, `wse`, and `duration_h`) from a
/// Fill-Spill-Merge simulation over `[0, storm_duration_h]`.
///
/// `storm_duration_h` must satisfy `0.0 <= storm_duration_h <= fsm.mass_balance.duration_h`.
/// For sub-windows (`storm_duration_h < fsm.mass_balance.duration_h`), both peak stage and
/// wet duration are clipped strictly to `[0.0, storm_duration_h]`.
///
/// For every cell inundated within the window (`0 <= fill_time <= storm_duration_h`):
/// - `wse` is inverted from the peak stored volume reached by the cell's enclosing
///   basin (and any compound parent it merged into within `[0, storm_duration_h]`), capped
///   at the basin's spill rim elevation and guaranteed `>= dem_z`.
/// - `max_depth = (wse - dem_z).max(0.0)`.
/// - `duration_h` is the inundated duration in `[0.0, storm_duration_h]` (accounting for
///   draining intervals and preserving antecedent ponding while a compound parent waits
///   for its children to merge).
///
/// Non-inundated valid DEM cells receive `0.0` in `max_depth` and `duration_h`, and NoData
/// in `wse`. NoData DEM cells remain NoData in all three rasters.
pub fn compute_fsm_hazard_rasters(
    dem: &Array2D<f64>,
    fill_time: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    fsm: &FsmSimulationResult,
    storm_duration_h: f64,
) -> Result<FsmHazardRasters> {
    if !storm_duration_h.is_finite() || storm_duration_h < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "storm_duration_h must be finite and >= 0, got {storm_duration_h}"
        )));
    }
    if storm_duration_h > fsm.mass_balance.duration_h + STORM_TIME_EPS_H {
        return Err(Error::InvalidArgument(format!(
            "storm_duration_h ({storm_duration_h}) exceeds simulated FSM storm duration ({})",
            fsm.mass_balance.duration_h
        )));
    }
    validate_grid_alignment(dem, fill_time, "fill_time")?;
    validate_grid_alignment(dem, &depressions.depression_id, "depression_id")?;

    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let ft_data = fill_time.data();
    let ft_nd = fill_time.no_data();
    let dep_ids = depressions.depression_id.data();
    let (_, _, cell_area) = affine_metrics(&dem.geotransform);

    let depth_nd = -9999.0f64;
    let wse_nd = if z_nd.is_finite() { z_nd } else { -9999.0f64 };
    let dur_nd = -1.0f64;

    let mut max_depth_data = vec![depth_nd; n];
    let mut wse_data = vec![wse_nd; n];
    let mut dur_data = vec![dur_nd; n];

    // Initialize valid DEM cells to dry (depth = 0, duration = 0, wse = NoData).
    for idx in 0..n {
        let z = zdata[idx];
        if z.is_finite() && !z.rd_eq(z_nd) {
            max_depth_data[idx] = 0.0;
            dur_data[idx] = 0.0;
        }
    }

    let m = depressions.depressions.len();
    if m == 0 || n == 0 {
        let mut max_depth = Array2D::from_vec(w, h, max_depth_data)?;
        max_depth.copy_metadata_from(dem);
        max_depth.set_no_data(depth_nd);
        let mut wse = Array2D::from_vec(w, h, wse_data)?;
        wse.copy_metadata_from(dem);
        wse.set_no_data(wse_nd);
        let mut duration_h = Array2D::from_vec(w, h, dur_data)?;
        duration_h.copy_metadata_from(dem);
        duration_h.set_no_data(dur_nd);
        return Ok(FsmHazardRasters {
            max_depth,
            wse,
            duration_h,
        });
    }

    let is_subwindow = storm_duration_h < fsm.mass_balance.duration_h - STORM_TIME_EPS_H;
    let has_history = fsm.basin_vol_history.len() == fsm.basins.len();

    // Build hierarchy maps and per-depression sorted subtree elevations.
    let mut id_to_idx: HashMap<u32, usize> = HashMap::with_capacity(m);
    let mut parent_map: HashMap<u32, u32> = HashMap::with_capacity(m);
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::with_capacity(m);
    for (idx, d) in depressions.depressions.iter().enumerate() {
        id_to_idx.insert(d.id, idx);
        children_map.insert(d.id, d.children_ids.clone());
        if d.parent_id > 0 {
            parent_map.insert(d.id, d.parent_id);
        }
        for &cid in &d.children_ids {
            parent_map.entry(cid).or_insert(d.id);
        }
    }
    let mut fsm_idx_by_id: HashMap<u32, usize> = HashMap::with_capacity(fsm.basins.len());
    for (idx, b) in fsm.basins.iter().enumerate() {
        fsm_idx_by_id.insert(b.dep_id, idx);
    }

    let mut pixels_by_id: HashMap<u32, Vec<usize>> = HashMap::new();
    for (idx, &did) in dep_ids.iter().enumerate() {
        if did > 0 {
            pixels_by_id.entry(did as u32).or_default().push(idx);
        }
    }

    // For each basin, compute exact sorted subtree elevations, cumulative storage table,
    // and storm peak WSE clipped to `[0, storm_duration_h]`.
    let mut subtree_elevs: Vec<Vec<f64>> = vec![Vec::new(); m];
    let mut subtree_prefix_v: Vec<Vec<f64>> = vec![Vec::new(); m];
    let mut basin_peak_wse: Vec<Option<f64>> = vec![None; m];

    for (idx, d) in depressions.depressions.iter().enumerate() {
        let mut desc_ids: HashSet<u32> = HashSet::new();
        let mut stack = vec![d.id];
        while let Some(cid) = stack.pop() {
            if desc_ids.insert(cid)
                && let Some(kids) = children_map.get(&cid)
            {
                stack.extend(kids.iter().copied());
            }
        }
        let mut elevs: Vec<f64> = Vec::new();
        for &cid in &desc_ids {
            if let Some(pts) = pixels_by_id.get(&cid) {
                for &gi in pts {
                    let z = zdata[gi];
                    if z.is_finite() && !z.rd_eq(z_nd) {
                        elevs.push(z);
                    }
                }
            }
        }
        elevs.sort_by(f64::total_cmp);
        let k = elevs.len();
        let mut pref_v = Vec::with_capacity(k);
        let mut sum_z_above_min = 0.0f64;
        if k > 0 {
            let z0 = elevs[0];
            let mut v_cum = 0.0f64;
            pref_v.push(0.0);
            for i in 1..k {
                v_cum += (i as f64) * cell_area * (elevs[i] - elevs[i - 1]);
                pref_v.push(v_cum);
                sum_z_above_min += elevs[i] - z0;
            }

            let fidx_opt = fsm_idx_by_id.get(&d.id).copied();
            let (peak_vol, cap_m3) = if let Some(fidx) = fidx_opt {
                let summary = &fsm.basins[fidx];
                let pv = if is_subwindow && has_history {
                    peak_vol_from_history(
                        &fsm.basin_vol_history[fidx],
                        storm_duration_h,
                        summary.initial_volume_m3,
                    )
                } else if fsm.basin_peak_vol_storm_m3.len() == fsm.basins.len() {
                    fsm.basin_peak_vol_storm_m3[fidx]
                } else {
                    summary
                        .final_volume_m3
                        .max(if summary.fill_time_h.is_some() {
                            summary.capacity_m3
                        } else {
                            0.0
                        })
                };
                (pv, summary.capacity_m3)
            } else {
                (0.0, d.volume)
            };

            let z_rim_from_cap =
                elevs[0] + (sum_z_above_min + cap_m3 / cell_area.max(1e-12)) / (k as f64);
            let z_rim = z_rim_from_cap.max(elevs[k - 1]).max(d.max_elev);
            let eff_vol = peak_vol.min(cap_m3);

            let filled_in_window = fidx_opt.is_some_and(|fi| {
                fsm.basins[fi]
                    .fill_time_h
                    .is_some_and(|ft| ft <= storm_duration_h + STORM_TIME_EPS_H)
            });
            if eff_vol > 1e-12 || (cap_m3 <= 1e-12 && filled_in_window) {
                let wse = if eff_vol >= cap_m3 - 1e-9 {
                    z_rim
                } else {
                    let pos = pref_v.partition_point(|&v| v <= eff_vol).saturating_sub(1);
                    let extra = (eff_vol - pref_v[pos]).max(0.0);
                    (elevs[pos] + extra / (((pos + 1) as f64) * cell_area)).min(z_rim)
                };
                basin_peak_wse[idx] = Some(wse);
            }
        }
        subtree_elevs[idx] = elevs;
        subtree_prefix_v[idx] = pref_v;
    }

    // Helper: exact cumulative storage volume threshold for elevation `z` inside basin `bidx`.
    let cell_threshold_vol = |bidx: usize, z: f64| -> f64 {
        let elevs = &subtree_elevs[bidx];
        let pref_v = &subtree_prefix_v[bidx];
        if elevs.is_empty() {
            return 0.0;
        }
        let pos = elevs.partition_point(|&ez| ez < z);
        if pos == 0 {
            0.0
        } else {
            let prev = pos - 1;
            pref_v[prev] + (pos as f64) * cell_area * (z - elevs[prev]).max(0.0)
        }
    };

    let storm_end = storm_duration_h + STORM_TIME_EPS_H;
    let use_history_duration = fsm.basin_had_draining && has_history;

    for idx in 0..n {
        let z = zdata[idx];
        if !z.is_finite() || z.rd_eq(z_nd) {
            continue;
        }
        let ft = ft_data[idx];
        if !ft.is_finite() || ft.rd_eq(ft_nd) || ft < 0.0 || ft > storm_end {
            continue;
        }
        let did = dep_ids[idx];
        if did <= 0 {
            continue;
        }
        let start_id = did as u32;
        let Some(&start_bidx) = id_to_idx.get(&start_id) else {
            continue;
        };

        // Determine highest WSE across `start_id` and any enclosing parent that it merged into
        // within `[0, storm_duration_h]`.
        let mut best_wse = basin_peak_wse[start_bidx].unwrap_or(z);
        let mut cur_id = start_id;
        let mut merged_chain: Vec<(u32, usize)> = vec![(start_id, start_bidx)];
        while let Some(&pid) = parent_map.get(&cur_id) {
            let cur_merged_in_storm = fsm_idx_by_id.get(&cur_id).is_some_and(|&fi| {
                fsm.basins[fi]
                    .merge_time_h
                    .is_some_and(|mt| mt <= storm_end)
            });
            if !cur_merged_in_storm {
                break;
            }
            if let Some(&p_bidx) = id_to_idx.get(&pid) {
                if let Some(pw) = basin_peak_wse[p_bidx]
                    && pw > best_wse
                {
                    best_wse = pw;
                }
                merged_chain.push((pid, p_bidx));
                cur_id = pid;
            } else {
                break;
            }
        }
        let final_wse = best_wse.max(z);
        wse_data[idx] = final_wse;
        max_depth_data[idx] = (final_wse - z).max(0.0);

        let dur = if use_history_duration {
            let mut total_dur = 0.0f64;
            // Preserve antecedent ponding duration [0, t_activate] on parent-owned cells
            // while their compound parent waits for its children to fill and merge.
            if ft <= 1e-12
                && !depressions.depressions[start_bidx].children_ids.is_empty()
                && let Some(&start_fidx) = fsm_idx_by_id.get(&start_id)
            {
                let act_t = fsm
                    .basin_activation_time_h
                    .get(start_fidx)
                    .copied()
                    .flatten()
                    .unwrap_or(storm_duration_h);
                total_dur += act_t.clamp(0.0, storm_duration_h);
            }
            for &(bid, bidx) in &merged_chain {
                if let Some(&fidx) = fsm_idx_by_id.get(&bid)
                    && let Some(hist) = fsm.basin_vol_history.get(fidx)
                {
                    let thresh = cell_threshold_vol(bidx, z);
                    total_dur += wet_duration_from_history(hist, thresh, storm_duration_h);
                }
            }
            total_dur.clamp(0.0, storm_duration_h)
        } else {
            (storm_duration_h - ft).clamp(0.0, storm_duration_h)
        };
        dur_data[idx] = dur;
    }

    let mut max_depth = Array2D::from_vec(w, h, max_depth_data)?;
    max_depth.copy_metadata_from(dem);
    max_depth.set_no_data(depth_nd);

    let mut wse = Array2D::from_vec(w, h, wse_data)?;
    wse.copy_metadata_from(dem);
    wse.set_no_data(wse_nd);

    let mut duration_h = Array2D::from_vec(w, h, dur_data)?;
    duration_h.copy_metadata_from(dem);
    duration_h.set_no_data(dur_nd);

    Ok(FsmHazardRasters {
        max_depth,
        wse,
        duration_h,
    })
}

/// Writes [`FsmHazardRasters`] to `max_depth.tif`, `wse.tif`, and `duration_h.tif` inside `out_dir`.
pub fn write_fsm_hazard_rasters(
    hazards: &FsmHazardRasters,
    out_dir: impl AsRef<Path>,
    compress: bool,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;
    let max_depth_path = out_dir.join("max_depth.tif");
    let wse_path = out_dir.join("wse.tif");
    let duration_path = out_dir.join("duration_h.tif");
    save_raster(
        &hazards.max_depth,
        &max_depth_path,
        Some("fsm_hazard: max_depth"),
        compress,
    )?;
    save_raster(&hazards.wse, &wse_path, Some("fsm_hazard: wse"), compress)?;
    save_raster(
        &hazards.duration_h,
        &duration_path,
        Some("fsm_hazard: duration_h"),
        compress,
    )?;
    Ok((max_depth_path, wse_path, duration_path))
}

/// Parameters controlling post-storm vertical drainage and recession modeling
/// ([`simulate_post_storm_recession`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecessionOptions {
    /// Evapotranspiration loss rate in `cm/h` (must be finite and `>= 0.0`, default `0.05`).
    pub et_cm_h: f64,
    /// Vertical seepage / percolation loss rate in `cm/h` (must be finite and `>= 0.0`, default `0.10`).
    pub seepage_cm_h: f64,
    /// Maximum post-storm recession duration to simulate in hours (must be finite and `> 0.0`, default `168.0`).
    pub max_recession_h: f64,
}

impl Default for RecessionOptions {
    fn default() -> Self {
        Self {
            et_cm_h: 0.05,
            seepage_cm_h: 0.10,
            max_recession_h: 168.0,
        }
    }
}

impl RecessionOptions {
    /// Validates post-storm recession parameters.
    pub fn validate(&self) -> Result<()> {
        if !self.et_cm_h.is_finite() || self.et_cm_h < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "et_cm_h must be finite and >= 0, got {}",
                self.et_cm_h
            )));
        }
        if !self.seepage_cm_h.is_finite() || self.seepage_cm_h < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "seepage_cm_h must be finite and >= 0, got {}",
                self.seepage_cm_h
            )));
        }
        if !self.max_recession_h.is_finite() || self.max_recession_h <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "max_recession_h must be finite and > 0, got {}",
                self.max_recession_h
            )));
        }
        Ok(())
    }
}

/// One sample point along the post-storm recession curve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecessionCurvePoint {
    /// Elapsed hours after storm end (`>= 0.0`).
    pub time_h: f64,
    /// Total remaining ponded water volume across all depressions (`m^3`).
    pub stored_vol_m3: f64,
    /// Total currently inundated surface area across all wet depressions (`m^2`).
    pub inundated_area_m2: f64,
    /// Number of active basins still holding ponded water (`stored_vol > 0`).
    pub wet_basin_count: usize,
}

/// Output of [`simulate_post_storm_recession`].
#[derive(Debug, Clone)]
pub struct RecessionResult {
    /// Per-cell drain time in hours after storm end:
    /// `-1.0` (NoData) if never wet at storm end, finite `>= 0.0` if drained within
    /// `max_recession_h`, or `f64::INFINITY` if still inundated at `max_recession_h`.
    pub drain_time_h: Array2D<f64>,
    /// Total remaining stored water volume across all depressions at `t = max_recession_h` (`m^3`).
    pub remaining_volume_m3: f64,
    /// Total water volume drained via ET and seepage over `[0, max_recession_h]` (`m^3`).
    pub total_drained_m3: f64,
    /// Number of initially wet basins that drained completely within `max_recession_h`.
    pub drained_basin_count: usize,
    /// Time-series recession curve starting at `t = 0.0` (storm end).
    pub curve: Vec<RecessionCurvePoint>,
}

/// Evaluates exact `(stored_volume_m3, wet_cell_count)` inside a sorted hypsometric basin
/// at water surface elevation `wse`.
fn hypso_state_at_wse(elevs: &[f64], pref_v: &[f64], cell_area: f64, wse: f64) -> (f64, usize) {
    if elevs.is_empty() || wse <= elevs[0] + 1e-12 {
        return (0.0, 0);
    }
    let wet_cnt = elevs.partition_point(|&z| z < wse);
    if wet_cnt == 0 {
        return (0.0, 0);
    }
    let idx = wet_cnt - 1;
    let vol = pref_v[idx] + (wet_cnt as f64) * cell_area * (wse - elevs[idx]).max(0.0);
    (vol.max(0.0), wet_cnt)
}

/// Simulates post-storm vertical drainage and pond recession starting from the storm-end
/// stored volumes produced by [`simulate_fill_spill_merge_in_memory`].
///
/// Over any wet surface area `A_wet(W)`, vertical losses at rate `r = (et_cm_h + seepage_cm_h) / 100`
/// satisfy `dV/dt = -r * A_wet(W)` while hypsometric storage satisfies `dV/dW = A_wet(W)`,
/// yielding exact piecewise-analytical stage recession `dW/dt = -r` across every cell's
/// hypsometric step. A cell at elevation `z < W_0` dries at exact time `(W_0 - z) / r` after
/// storm end, conserving volume `total_drained_m3 + remaining_volume_m3 == initial_stored_m3`
/// to machine precision.
pub fn simulate_post_storm_recession(
    dem: &Array2D<f64>,
    depressions: &InMemoryDepressions,
    fsm: &FsmSimulationResult,
    options: &RecessionOptions,
) -> Result<RecessionResult> {
    options.validate()?;
    validate_grid_alignment(dem, &depressions.depression_id, "depression_id")?;
    validate_grid_alignment(dem, &fsm.fill_time, "fill_time")?;

    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let dep_ids = depressions.depression_id.data();
    let (_, _, cell_area) = affine_metrics(&dem.geotransform);

    let mut drain_data = vec![-1.0f64; n];
    if n == 0 || depressions.depressions.is_empty() || fsm.basins.is_empty() {
        let mut drain_time_h = Array2D::from_vec(w, h, drain_data)?;
        drain_time_h.copy_metadata_from(dem);
        drain_time_h.set_no_data(-1.0);
        return Ok(RecessionResult {
            drain_time_h,
            remaining_volume_m3: 0.0,
            total_drained_m3: 0.0,
            drained_basin_count: 0,
            curve: vec![RecessionCurvePoint {
                time_h: 0.0,
                stored_vol_m3: 0.0,
                inundated_area_m2: 0.0,
                wet_basin_count: 0,
            }],
        });
    }

    let mut children_map: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(depressions.depressions.len());
    for d in &depressions.depressions {
        children_map.insert(d.id, d.children_ids.clone());
    }
    let mut merged_in_storm: HashSet<u32> = HashSet::new();
    for b in &fsm.basins {
        if b.merge_time_h.is_some() {
            merged_in_storm.insert(b.dep_id);
        }
    }
    let mut pixels_by_id: HashMap<u32, Vec<usize>> = HashMap::new();
    for (idx, &did) in dep_ids.iter().enumerate() {
        if did > 0 {
            pixels_by_id.entry(did as u32).or_default().push(idx);
        }
    }

    let rate_m_h = (options.et_cm_h + options.seepage_cm_h) / 100.0;
    let max_t = options.max_recession_h;

    struct WetBasinRecession {
        elevs: Vec<f64>,
        pref_v: Vec<f64>,
        v0: f64,
        w0: f64,
        t_empty: f64,
    }

    let mut wet_basins: Vec<WetBasinRecession> = Vec::new();
    let mut total_drained_m3 = 0.0f64;
    let mut remaining_volume_m3 = 0.0f64;
    let mut drained_basin_count = 0usize;

    for b in &fsm.basins {
        if b.merge_time_h.is_some() {
            continue;
        }
        let v0 = b.final_volume_m3.min(b.capacity_m3);
        if !v0.is_finite() || v0 <= 1e-12 {
            continue;
        }

        // Collect cells owned by this basin plus any descendants that merged into it during
        // the storm (unmerged children retain their own separate pools).
        let mut owned_ids: Vec<u32> = vec![b.dep_id];
        let mut stack: Vec<u32> = children_map
            .get(&b.dep_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|cid| merged_in_storm.contains(cid))
            .collect();
        let mut seen: HashSet<u32> = HashSet::new();
        seen.insert(b.dep_id);
        while let Some(cid) = stack.pop() {
            if seen.insert(cid) {
                owned_ids.push(cid);
                if let Some(kids) = children_map.get(&cid) {
                    for &gc in kids {
                        if merged_in_storm.contains(&gc) {
                            stack.push(gc);
                        }
                    }
                }
            }
        }

        let mut cells: Vec<(f64, usize)> = Vec::new();
        for id in owned_ids {
            if let Some(pts) = pixels_by_id.get(&id) {
                for &gi in pts {
                    let z = zdata[gi];
                    if z.is_finite() && !z.rd_eq(z_nd) {
                        cells.push((z, gi));
                    }
                }
            }
        }
        if cells.is_empty() {
            continue;
        }
        cells.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let k = cells.len();
        let elevs: Vec<f64> = cells.iter().map(|&(z, _)| z).collect();
        let mut pref_v = Vec::with_capacity(k);
        pref_v.push(0.0);
        let mut v_cum = 0.0f64;
        for i in 1..k {
            v_cum += (i as f64) * cell_area * (elevs[i] - elevs[i - 1]);
            pref_v.push(v_cum);
        }

        let pos = pref_v.partition_point(|&v| v <= v0).saturating_sub(1);
        let extra = (v0 - pref_v[pos]).max(0.0);
        let w0 = elevs[pos] + extra / (((pos + 1) as f64) * cell_area);

        // Assign per-cell drain times for all cells wet at storm end (`z < w0`).
        for &(z, gi) in &cells {
            if z < w0 - 1e-12 {
                if rate_m_h <= 0.0 {
                    drain_data[gi] = f64::INFINITY;
                } else {
                    let t_dry = ((w0 - z) / rate_m_h).max(0.0);
                    drain_data[gi] = if t_dry <= max_t + 1e-12 {
                        t_dry.min(max_t)
                    } else {
                        f64::INFINITY
                    };
                }
            }
        }

        let t_empty = if rate_m_h > 0.0 {
            ((w0 - elevs[0]) / rate_m_h).max(0.0)
        } else {
            f64::INFINITY
        };

        let v_rem = if rate_m_h <= 0.0 {
            v0
        } else {
            let w_end = w0 - rate_m_h * max_t;
            let (vr, _) = hypso_state_at_wse(&elevs, &pref_v, cell_area, w_end);
            vr.min(v0)
        };
        if v_rem <= 1e-12 {
            drained_basin_count += 1;
            total_drained_m3 += v0;
        } else {
            remaining_volume_m3 += v_rem;
            total_drained_m3 += (v0 - v_rem).max(0.0);
        }

        wet_basins.push(WetBasinRecession {
            elevs,
            pref_v,
            v0,
            w0,
            t_empty,
        });
    }

    // Construct strictly increasing sample timestamps combining a uniform grid across
    // `[0, t_horizon]` with exact basin drain-completion timestamps.
    let t_horizon = if wet_basins.is_empty() {
        0.0
    } else {
        let max_empty = wet_basins.iter().map(|b| b.t_empty).fold(0.0f64, f64::max);
        max_empty.min(max_t)
    };

    let mut sample_times: Vec<f64> = Vec::with_capacity(wet_basins.len() + 32);
    sample_times.push(0.0);
    if t_horizon > 0.0 {
        const NUM_UNIFORM_STEPS: usize = 24;
        for step in 1..=NUM_UNIFORM_STEPS {
            sample_times.push(t_horizon * (step as f64 / NUM_UNIFORM_STEPS as f64));
        }
        for b in &wet_basins {
            if b.t_empty.is_finite() && b.t_empty > 0.0 && b.t_empty <= max_t {
                sample_times.push(b.t_empty);
            }
        }
    }
    sample_times.sort_by(f64::total_cmp);
    let mut dedup_times: Vec<f64> = Vec::with_capacity(sample_times.len());
    for t in sample_times {
        if dedup_times
            .last()
            .is_none_or(|&prev| (t - prev).abs() > 1e-9)
        {
            dedup_times.push(t);
        }
    }

    let mut curve = Vec::with_capacity(dedup_times.len());
    for t in dedup_times {
        let mut stored_vol_m3 = 0.0f64;
        let mut wet_cells = 0usize;
        let mut wet_basin_count = 0usize;
        for b in &wet_basins {
            let (vol, cnt) = if rate_m_h <= 0.0 {
                let cnt = b.elevs.partition_point(|&z| z < b.w0 - 1e-12);
                (b.v0, cnt)
            } else {
                let w_cur = b.w0 - rate_m_h * t;
                let (v, c) = hypso_state_at_wse(&b.elevs, &b.pref_v, cell_area, w_cur);
                (v.min(b.v0), c)
            };
            if vol > 1e-12 {
                stored_vol_m3 += vol;
                wet_cells += cnt;
                wet_basin_count += 1;
            }
        }
        curve.push(RecessionCurvePoint {
            time_h: t,
            stored_vol_m3,
            inundated_area_m2: wet_cells as f64 * cell_area,
            wet_basin_count,
        });
    }

    let mut drain_time_h = Array2D::from_vec(w, h, drain_data)?;
    drain_time_h.copy_metadata_from(dem);
    drain_time_h.set_no_data(-1.0);

    Ok(RecessionResult {
        drain_time_h,
        remaining_volume_m3,
        total_drained_m3,
        drained_basin_count,
        curve,
    })
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

    #[test]
    fn fsm_owned_hypso_handles_overlapping_parent_rim_pocket_and_conserves_mass() {
        // Compound depression (id=1) with two child bowls (id=2, id=3) plus a sub-threshold
        // pocket owned directly by parent (id=1) whose floor (10.5) is lower than the child
        // rims (12.0), verifying exact subtree threshold_vol computation in O(N) memory.
        let (w, h) = (10usize, 8usize);
        let mut z = vec![20.0f64; w * h];
        let mut sink_data = vec![0.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];

        for r in 1..7 {
            for c in 1..9 {
                let idx = r * w + c;
                dep_id[idx] = 1;
                dep_lvl[idx] = 2;
                z[idx] = 13.0;
                sink_data[idx] = 13.0;
            }
        }
        // Child 2 (left bowl): floor 9.0, rim 12.0
        for r in 2..5 {
            for c in 2..4 {
                let idx = r * w + c;
                dep_id[idx] = 2;
                dep_lvl[idx] = 1;
                let elev = if r == 3 && c == 2 { 9.0 } else { 11.5 };
                z[idx] = elev;
                sink_data[idx] = elev;
            }
        }
        // Child 3 (right bowl): floor 8.0, rim 12.0
        for r in 2..5 {
            for c in 6..8 {
                let idx = r * w + c;
                dep_id[idx] = 3;
                dep_lvl[idx] = 1;
                let elev = if r == 3 && c == 6 { 8.0 } else { 11.5 };
                z[idx] = elev;
                sink_data[idx] = elev;
            }
        }
        // Sub-threshold pocket owned by parent (id=1) with elevation 10.5 < child rim 12.0.
        let pocket_idx = 5 * w + 4;
        z[pocket_idx] = 10.5;
        sink_data[pocket_idx] = 10.5;

        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

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
                make_dep(1, 2, 0, vec![2, 3], 8.0, 13.0, 48),
                make_dep(2, 1, 1, vec![], 9.0, 12.0, 6),
                make_dep(3, 1, 1, vec![], 8.0, 12.0, 6),
            ],
            max_level: 2,
            num_regions: 1,
        };

        let opts = SimulateInundationOptions {
            min_size: 2,
            min_depth: 0.1,
            interval: 0.5,
            rain_intensity_cm_h: 50.0,
            rain_duration_h: 10.0,
            time_step_h: 1.0,
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

        assert!(
            res.mass_balance.mass_balance_error_m3.abs() < 1e-9,
            "mass balance error too large: {}",
            res.mass_balance.mass_balance_error_m3
        );
        let parent_sum = res.basins.iter().find(|b| b.dep_id == 1).unwrap();
        let child2 = res.basins.iter().find(|b| b.dep_id == 2).unwrap();
        let child3 = res.basins.iter().find(|b| b.dep_id == 3).unwrap();
        assert!(parent_sum.capacity_m3 > child2.capacity_m3 + child3.capacity_m3);
        let pocket_ft = res.fill_time.get_i(pocket_idx);
        let rim_ft = res.fill_time.get_i(w + 1);
        assert!(
            pocket_ft >= 0.0 && rim_ft >= pocket_ft,
            "pocket fill_time ({pocket_ft}) should be finite and <= rim fill_time ({rim_ft})"
        );
    }

    #[test]
    fn fsm_zero_capacity_flat_child_marks_cells_when_parent_gains_water() {
        // Compound parent (id=1, rim=12.0) enclosing two completely flat zero-capacity children
        // (id=2 and id=3, both flat at z=10.0, max_elev=10.0) with no antecedent water.
        // At t=0 both children are FullSpilling with stored_vol=0 and merge into parent with
        // merged_vol=0; once rain begins filling parent, every flat child cell must receive a
        // finite fill_time >= 0.0 rather than remaining -1.0.
        let (w, h) = (8usize, 6usize);
        let mut z = vec![20.0f64; w * h];
        let mut sink_data = vec![0.0f64; w * h];
        let mut dep_id = vec![0i32; w * h];
        let mut dep_lvl = vec![0i32; w * h];

        for r in 1..5 {
            for c in 1..7 {
                let idx = r * w + c;
                dep_id[idx] = 1;
                dep_lvl[idx] = 2;
                z[idx] = 12.0;
                sink_data[idx] = 12.0;
            }
        }
        let child2_cells = [2 * w + 2, 2 * w + 3];
        let child3_cells = [3 * w + 4, 3 * w + 5];
        for &idx in &child2_cells {
            dep_id[idx] = 2;
            dep_lvl[idx] = 1;
            z[idx] = 10.0;
            sink_data[idx] = 10.0;
        }
        for &idx in &child3_cells {
            dep_id[idx] = 3;
            dep_lvl[idx] = 1;
            z[idx] = 10.0;
            sink_data[idx] = 10.0;
        }

        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

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
            perimeter: 6.0,
            major_axis: 2.0,
            minor_axis: 1.0,
            elongatedness: 2.0,
            eccentricity: 0.5,
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
                make_dep(1, 2, 0, vec![2, 3], 10.0, 12.0, 24),
                make_dep(2, 1, 1, vec![], 10.0, 10.0, 2),
                make_dep(3, 1, 1, vec![], 10.0, 10.0, 2),
            ],
            max_level: 2,
            num_regions: 1,
        };

        let opts = SimulateInundationOptions {
            min_size: 1,
            min_depth: 0.0,
            interval: 0.5,
            rain_intensity_cm_h: 50.0,
            rain_duration_h: 5.0,
            time_step_h: 1.0,
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

        assert!(res.mass_balance.mass_balance_error_m3.abs() < 1e-9);
        for &idx in child2_cells.iter().chain(child3_cells.iter()) {
            let ft = res.fill_time.get_i(idx);
            assert!(
                ft >= 0.0 && ft.is_finite(),
                "zero-capacity flat child cell {idx} must be inundated once parent gains water, got {ft}"
            );
        }
    }

    #[test]
    fn fsm_topological_routing_resets_non_live_routed_cat_across_passes() {
        // Verify that `evaluate_topological_routing` resets `q_in` and `routed_cat` for ALL
        // basins (including non-live `Waiting` / `Merged` entries) across repeated routing passes
        // so stale values never accumulate into storm-end snapshots.
        let (dem_refined, sink, deps) = build_two_sibling_compound_dem();
        let probe_opts = SimulateInundationOptions {
            min_size: 10,
            min_depth: 0.3,
            interval: 0.3,
            rain_intensity_cm_h: 50.0,
            rain_duration_h: 50.0,
            time_step_h: 1.0,
            compress: false,
        };
        let probe = simulate_fill_spill_merge_in_memory(
            &dem_refined,
            &sink,
            &deps,
            &probe_opts,
            &InundationForcing::default(),
        )
        .unwrap();

        // Pick a storm duration between the first leaf fill time and the parent merge time,
        // using a multi-step hyetograph to force multiple topological routing passes while the
        // compound parent remains `Waiting` and one child is actively spilling into its sibling.
        let mut leaf_fills: Vec<f64> = probe
            .basins
            .iter()
            .filter(|b| b.level == 1)
            .filter_map(|b| b.fill_time_h)
            .collect();
        leaf_fills.sort_by(f64::total_cmp);
        assert!(leaf_fills.len() >= 2);
        let t_mid = 0.5 * (leaf_fills[0] + leaf_fills[1]);
        let step_dt = t_mid / 4.0;
        let hyeto = vec![(step_dt, 50.0); 4];

        let opts = SimulateInundationOptions {
            rain_duration_h: t_mid,
            time_step_h: step_dt,
            ..probe_opts
        };
        let forcing = InundationForcing {
            hyetograph: Some(&hyeto),
            ..Default::default()
        };
        let res = simulate_fill_spill_merge_in_memory(&dem_refined, &sink, &deps, &opts, &forcing)
            .unwrap();

        assert!(res.mass_balance.mass_balance_error_m3.abs() < 1e-9);
        let parent = res.basins.iter().find(|b| b.level > 1).unwrap();
        assert!(
            parent.fill_time_h.is_none(),
            "parent should still be Waiting at t_mid"
        );
        assert!(
            (parent.final_routed_cat_area_m2 - parent.local_cat_area_m2).abs() < 1e-9,
            "Waiting parent final_routed_cat_area_m2 ({}) must equal local_cat_area_m2 ({}) with no stale accumulation",
            parent.final_routed_cat_area_m2,
            parent.local_cat_area_m2
        );
    }
}
