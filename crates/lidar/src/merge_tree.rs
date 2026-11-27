//! Exact bottom-up merge-tree (contour-tree) nested depression hierarchy.
//!
//! Replaces discrete vertical level-set slicing with an exact bottom-up Union-Find
//! sweep over sink cells sorted once in ascending elevation order. Internal saddles
//! and pour points are recovered to machine precision independent of any slicing
//! interval, same-elevation multi-basin junctions are grouped into a single multi-child
//! merge event, sub-threshold micro-pits are absorbed via persistence pruning
//! (`min_size` and `min_depth`), and exact continuous stage-storage-area rating curves
//! retain every distinct elevation breakpoint without downsampling.
//!
//! ## Complexity
//!
//! - **Merge-tree sweep:** `O(M log M)` time and `O(M)` memory per sink region of `M`
//!   occupied pixels (using sparse indexing whenever a region's bounding box is sparse).
//! - **Per-node morphometrics and exact rating curves:** `O(sum_v (k_v log k_v + S_v))`
//!   time where `k_v = |subtree(v)|` and `S_v <= k_v + 2` is the number of retained
//!   rating-curve stages for node `v` (sorting each node's subtree elevations and stages,
//!   followed by a single linear sweep to compute interval-by-interval storage), and `O(M)`
//!   peak working memory per region without bounding-box-sized dense grids for sparse regions.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::f64::consts::{FRAC_PI_4, PI};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, Topology};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::graph::RatingCurvePoint;
use crate::inundation::validate_grid_alignment;
use crate::region::{
    benkrid_crookes_perimeter_from_border_set, extended_morphometrics, safe_crs_metric_basis,
};
use crate::slicing::{InMemoryDepressions, NestedDepression};

/// Options controlling [`delineate_depressions_merge_tree_in_memory`] and
/// [`delineate_depressions_merge_tree_with_dem_in_memory`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MergeTreeOptions {
    /// Minimum pixel count for a depression branch to survive persistence pruning (default `1000`).
    pub min_size: usize,
    /// Minimum vertical relief (`saddle_elev - min_elev`) for a branch to survive persistence
    /// pruning (default `0.3`).
    pub min_depth: f64,
    /// Grid connectivity topology for sink grouping and bottom-up merges (`Topology::D8` by default).
    pub topology: Topology,
    /// Set to `true` when the `source_dem` passed to
    /// [`delineate_depressions_merge_tree_with_dem_in_memory`] is already a pit-refined DEM
    /// (such as `InMemorySinks::dem_refined`), skipping an extra full-DEM clone and Priority-Flood
    /// pass (default `false`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub dem_is_refined: bool,
    /// Optional cap on emitted rating-curve points per merge-tree node (default `None`, retaining
    /// every distinct elevation breakpoint). Set to `Some(0)` to disable rating-curve emission
    /// completely while still computing exact per-node volumes via bottom-up elevation moments,
    /// or `Some(k)` (`k >= 2`, values of `1` are treated as `2`) to emit at most `k` evenly
    /// spaced breakpoints per node (always preserving floor and spill endpoints with exact
    /// integrated storage volume).
    #[cfg_attr(feature = "serde", serde(default))]
    pub max_rating_curve_points: Option<usize>,
}

impl Default for MergeTreeOptions {
    fn default() -> Self {
        Self {
            min_size: 1000,
            min_depth: 0.3,
            topology: Topology::D8,
            dem_is_refined: false,
            max_rating_curve_points: None,
        }
    }
}

/// A single node in the exact bottom-up depression merge tree.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MergeTreeNode {
    /// Unique 1-based node / depression ID.
    pub id: u32,
    /// Direct enclosing parent node ID (`0` for top-level sink roots).
    pub parent_id: u32,
    /// Direct child node IDs that merge into this node at their shared saddle.
    pub children_ids: Vec<u32>,
    /// Hierarchy level (`1` = leaf basin; `1 + max(child.level)` for compound basins).
    pub level: u32,
    /// 1-based top-level sink region ID.
    pub region_id: u32,
    /// Lowest elevation in this depression (including descendants).
    pub min_elev: f64,
    /// Exact saddle / spill elevation where this depression merges or overflows.
    /// For root nodes delineated from a masked sink raster alone (without the source DEM),
    /// this is the lower-bound rim estimate equal to the highest retained sink cell; pass the
    /// source DEM via [`delineate_depressions_merge_tree_with_dem_in_memory`] to obtain the
    /// exact exterior outlet elevation.
    pub saddle_elev: f64,
    /// Row index of the saddle / pour point cell in the full raster.
    pub pour_point_row: usize,
    /// Column index of the saddle / pour point cell in the full raster.
    pub pour_point_col: usize,
    /// Map X coordinate of the pour point cell center.
    pub pour_point_x: f64,
    /// Map Y coordinate of the pour point cell center.
    pub pour_point_y: f64,
    /// Total pixel count enclosed by this depression (including descendants).
    pub count: usize,
    /// Total surface area in map units squared (`count * cell_area`).
    pub area: f64,
    /// Exact storage volume below `saddle_elev` in cubic map units, computed relative to
    /// this node's own floor elevation so it matches its rating-curve endpoint identically.
    pub volume: f64,
    /// Maximum vertical depth (`saddle_elev - min_elev`).
    pub max_depth: f64,
    /// Mean depth below `saddle_elev` (`volume / area`).
    pub avg_depth: f64,
    /// `true` if this node is a leaf depression (`children_ids.is_empty()`).
    pub is_leaf: bool,
}

/// Complete output of [`delineate_depressions_merge_tree_in_memory`] and
/// [`delineate_depressions_merge_tree_with_dem_in_memory`].
#[derive(Debug, Clone)]
pub struct MergeTreeResult {
    /// Exact merge-tree nodes ordered by 1-based `id` (`nodes[i].id == (i + 1) as u32`).
    pub nodes: Vec<MergeTreeNode>,
    /// Standard [`InMemoryDepressions`] representation directly compatible with
    /// [`crate::fsm`], [`crate::graph`], and [`crate::inundation`].
    pub depressions: InMemoryDepressions,
    /// Exact continuous stage-storage-area rating curves retaining every distinct
    /// cell elevation breakpoint (plus floor and spill stage) without downsampling.
    pub rating_curves: Vec<RatingCurvePoint>,
}

#[derive(Clone, Copy)]
struct SinkCell {
    z: f64,
    r: usize,
    c: usize,
}

#[derive(Clone, Copy)]
struct RootOutlet {
    z: f64,
    r: usize,
    c: usize,
}

struct LocalNode {
    parent_idx: Option<usize>,
    children_idxs: Vec<usize>,
    level: u32,
    min_elev: f64,
    saddle_elev: f64,
    pour_point_row: usize,
    pour_point_col: usize,
    count: usize,
    area: f64,
    max_depth: f64,
    is_leaf: bool,
}

/// Compact or sparse lookup from grid coordinate `(r, c)` to sorted cell index `0..m`.
/// Keeps temporary memory strictly `O(m)` even for sparse diagonal sink regions.
enum CellIndex {
    Dense {
        r0: usize,
        c0: usize,
        bw: usize,
        bh: usize,
        data: Vec<usize>,
    },
    Sparse(HashMap<(usize, usize), usize>),
}

impl CellIndex {
    fn new(r0: usize, c0: usize, bw: usize, bh: usize, m: usize) -> Self {
        let bbox_area = bw.saturating_mul(bh);
        if bbox_area <= m.saturating_mul(4).max(256) {
            Self::Dense {
                r0,
                c0,
                bw,
                bh,
                data: vec![usize::MAX; bbox_area],
            }
        } else {
            Self::Sparse(HashMap::with_capacity(m))
        }
    }

    #[inline]
    fn insert(&mut self, r: usize, c: usize, idx: usize) {
        match self {
            Self::Dense {
                r0, c0, bw, data, ..
            } => {
                data[(r - *r0) * *bw + (c - *c0)] = idx;
            }
            Self::Sparse(map) => {
                map.insert((r, c), idx);
            }
        }
    }

    #[inline]
    fn get(&self, r: isize, c: isize) -> Option<usize> {
        if r < 0 || c < 0 {
            return None;
        }
        let ru = r as usize;
        let cu = c as usize;
        match self {
            Self::Dense {
                r0,
                c0,
                bw,
                bh,
                data,
            } => {
                if ru < *r0 || cu < *c0 || ru >= *r0 + *bh || cu >= *c0 + *bw {
                    return None;
                }
                let v = data[(ru - *r0) * *bw + (cu - *c0)];
                (v != usize::MAX).then_some(v)
            }
            Self::Sparse(map) => map.get(&(ru, cu)).copied(),
        }
    }
}

#[inline]
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[inline]
fn uf_find_u32(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[inline]
fn cell_center_xy(gt: &[f64; 6], row: usize, col: usize) -> (f64, f64) {
    let cf = col as f64 + 0.5;
    let rf = row as f64 + 0.5;
    (
        gt[0] + cf * gt[1] + rf * gt[2],
        gt[3] + cf * gt[4] + rf * gt[5],
    )
}

/// Groups valid sink pixels into connected components under `topology` (D4 or D8),
/// retaining components with `count >= min_count`.
fn label_sink_regions<F: Fn(f64) -> bool>(
    data: &[f64],
    width: usize,
    height: usize,
    min_count: usize,
    topology: Topology,
    is_fg: F,
) -> (Vec<u32>, u32) {
    let n = width * height;
    let mut labels = vec![0u32; n];
    if n == 0 {
        return (labels, 0);
    }
    let mut parent: Vec<u32> = vec![0];
    let use_d8 = matches!(topology, Topology::D8);

    for r in 0..height {
        let row_off = r * width;
        for c in 0..width {
            let i = row_off + c;
            if !is_fg(data[i]) {
                continue;
            }
            let mut nbrs = [0u32; 4];
            let mut k = 0usize;
            if c > 0 && labels[i - 1] > 0 {
                nbrs[k] = uf_find_u32(&mut parent, labels[i - 1]);
                k += 1;
            }
            if r > 0 && labels[i - width] > 0 {
                nbrs[k] = uf_find_u32(&mut parent, labels[i - width]);
                k += 1;
            }
            if use_d8 && r > 0 {
                if c > 0 && labels[i - width - 1] > 0 {
                    nbrs[k] = uf_find_u32(&mut parent, labels[i - width - 1]);
                    k += 1;
                }
                if c + 1 < width && labels[i - width + 1] > 0 {
                    nbrs[k] = uf_find_u32(&mut parent, labels[i - width + 1]);
                    k += 1;
                }
            }
            if k == 0 {
                let next = parent.len() as u32;
                parent.push(next);
                labels[i] = next;
            } else {
                let mut root = nbrs[0];
                for &nr in &nbrs[1..k] {
                    let ra = uf_find_u32(&mut parent, root);
                    let rb = uf_find_u32(&mut parent, nr);
                    if ra != rb {
                        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                        parent[hi as usize] = lo;
                        root = lo;
                    }
                }
                labels[i] = uf_find_u32(&mut parent, root);
            }
        }
    }

    let mut counts = vec![0usize; parent.len()];
    for v in &mut labels {
        if *v > 0 {
            let r = uf_find_u32(&mut parent, *v);
            *v = r;
            counts[r as usize] += 1;
        }
    }

    let mut remap = vec![0u32; parent.len()];
    let mut next_id = 0u32;
    for v in &mut labels {
        let cur = *v;
        if cur == 0 {
            continue;
        }
        if counts[cur as usize] < min_count {
            *v = 0;
            continue;
        }
        let mapped = remap[cur as usize];
        if mapped == 0 {
            next_id += 1;
            remap[cur as usize] = next_id;
            *v = next_id;
        } else {
            *v = mapped;
        }
    }

    (labels, next_id)
}

struct RegionMergeOutput {
    region_idx: usize,
    /// Sparse occupied pixel assignments `(global_linear_idx, local_node_id)` so peak
    /// memory scales strictly with occupied cells rather than `sum(bbox_area)`.
    pixel_assignments: Vec<(usize, u32)>,
    nodes: Vec<MergeTreeNode>,
    depressions: Vec<NestedDepression>,
    rating_curves: Vec<RatingCurvePoint>,
}

const D4_OFFSETS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const D8_OFFSETS: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

#[allow(clippy::too_many_arguments)]
fn build_region_merge_tree(
    region_idx: usize,
    region_id: u32,
    full_w: usize,
    bbox: (usize, usize, usize, usize),
    mut cells: Vec<SinkCell>,
    root_outlet: Option<RootOutlet>,
    gt: &[f64; 6],
    vx_m: (f64, f64),
    vy_m: (f64, f64),
    cell_area: f64,
    options: &MergeTreeOptions,
) -> RegionMergeOutput {
    let (r0, c0, r1, c1) = bbox;
    let bh = r1 - r0;
    let bw = c1 - c0;
    let m = cells.len();
    let min_size_eff = options.min_size.max(1);

    if m < min_size_eff {
        return RegionMergeOutput {
            region_idx,
            pixel_assignments: Vec::new(),
            nodes: Vec::new(),
            depressions: Vec::new(),
            rating_curves: Vec::new(),
        };
    }

    // Sort all sink cells once in ascending (z, row, col) order: O(M log M).
    cells.sort_unstable_by(|a, b| a.z.total_cmp(&b.z).then(a.r.cmp(&b.r)).then(a.c.cmp(&b.c)));

    let reg_min_elev = cells[0].z;
    let reg_max_elev = cells[m - 1].z;
    let root_spill_elev = root_outlet.map_or(reg_max_elev, |o| o.z.max(reg_max_elev));
    if (root_spill_elev - reg_min_elev) < options.min_depth {
        return RegionMergeOutput {
            region_idx,
            pixel_assignments: Vec::new(),
            nodes: Vec::new(),
            depressions: Vec::new(),
            rating_curves: Vec::new(),
        };
    }

    let offsets: &[(isize, isize)] = match options.topology {
        Topology::D4 => &D4_OFFSETS,
        Topology::D8 => &D8_OFFSETS,
    };

    let mut cell_index = CellIndex::new(r0, c0, bw, bh, m);
    let mut uf_parent: Vec<usize> = (0..m).collect();
    let mut uf_count = vec![0usize; m];
    let mut uf_min_elev = vec![f64::INFINITY; m];
    let mut uf_min_idx = vec![usize::MAX; m];
    let mut uf_active_node: Vec<Option<usize>> = vec![None; m];

    // Intrusive O(1)-splice singly linked list of pixels owned by each active UF root
    // that have not yet been stamped onto a finalized child node.
    let mut next_pix = vec![usize::MAX; m];
    let mut list_head = vec![usize::MAX; m];
    let mut list_tail = vec![usize::MAX; m];

    // Owned local node index (into `local_nodes`) for each sorted cell `0..m`.
    let mut owned_node = vec![usize::MAX; m];
    let mut local_nodes: Vec<LocalNode> = Vec::new();

    let splice_list =
        |dst: usize, src: usize, head: &mut [usize], tail: &mut [usize], next: &mut [usize]| {
            let sh = head[src];
            if sh == usize::MAX {
                return;
            }
            let st = tail[src];
            if head[dst] == usize::MAX {
                head[dst] = sh;
                tail[dst] = st;
            } else {
                next[tail[dst]] = sh;
                tail[dst] = st;
            }
            head[src] = usize::MAX;
            tail[src] = usize::MAX;
        };

    let finalize_component = |root: usize,
                              z_saddle: f64,
                              pp_r: usize,
                              pp_c: usize,
                              uf_count: &[usize],
                              uf_min_elev: &[f64],
                              uf_active_node: &mut [Option<usize>],
                              list_head: &mut [usize],
                              list_tail: &mut [usize],
                              next_pix: &[usize],
                              owned_node: &mut [usize],
                              local_nodes: &mut Vec<LocalNode>|
     -> usize {
        let cnt = uf_count[root];
        let min_z = uf_min_elev[root];
        let area = cnt as f64 * cell_area;
        let max_depth = (z_saddle - min_z).max(0.0);

        let nidx = if let Some(existing) = uf_active_node[root].take() {
            let n = &mut local_nodes[existing];
            n.min_elev = min_z;
            n.saddle_elev = z_saddle;
            n.pour_point_row = pp_r;
            n.pour_point_col = pp_c;
            n.count = cnt;
            n.area = area;
            n.max_depth = max_depth;
            existing
        } else {
            let idx = local_nodes.len();
            local_nodes.push(LocalNode {
                parent_idx: None,
                children_idxs: Vec::new(),
                level: 1,
                min_elev: min_z,
                saddle_elev: z_saddle,
                pour_point_row: pp_r,
                pour_point_col: pp_c,
                count: cnt,
                area,
                max_depth,
                is_leaf: true,
            });
            idx
        };

        let mut cur = list_head[root];
        while cur != usize::MAX {
            owned_node[cur] = nidx;
            cur = next_pix[cur];
        }
        list_head[root] = usize::MAX;
        list_tail[root] = usize::MAX;
        nidx
    };

    // Process cells in exact equal-elevation plateaus so simultaneous multi-basin
    // junctions at elevation `z_batch` emit one multi-child compound node rather than
    // artificial chained binary compounds at identical elevation.
    let mut batch_start = 0usize;
    let mut plateau_parent: Vec<usize> = Vec::new();
    let mut pre_root_to_slot: HashMap<usize, usize> = HashMap::new();
    let mut pre_root_first_contact: HashMap<usize, usize> = HashMap::new();
    let mut slot_to_pre_root: Vec<usize> = Vec::new();
    let mut touched_edges: Vec<(usize, usize)> = Vec::new();

    while batch_start < m {
        let z_batch = cells[batch_start].z;
        let mut batch_end = batch_start + 1;
        while batch_end < m && cells[batch_end].z == z_batch {
            batch_end += 1;
        }
        let batch_len = batch_end - batch_start;

        // 1. Activate every cell in this elevation plateau.
        for i in batch_start..batch_end {
            let cell = cells[i];
            cell_index.insert(cell.r, cell.c, i);
            uf_count[i] = 1;
            uf_min_elev[i] = z_batch;
            uf_min_idx[i] = i;
            list_head[i] = i;
            list_tail[i] = i;
        }

        // 2. Discover all connections between plateau cells (`0..batch_len`) and
        //    pre-existing strictly-lower UF roots (`< batch_start`), recording each
        //    pre-root's deterministic first plateau contact cell in `O(1)` amortized time.
        pre_root_to_slot.clear();
        pre_root_first_contact.clear();
        slot_to_pre_root.clear();
        touched_edges.clear();

        for local_i in 0..batch_len {
            let i = batch_start + local_i;
            let cell = cells[i];
            for &(dr, dc) in offsets {
                let Some(n_sorted) = cell_index.get(cell.r as isize + dr, cell.c as isize + dc)
                else {
                    continue;
                };
                if n_sorted < batch_start {
                    let pre_root = uf_find(&mut uf_parent, n_sorted);
                    pre_root_first_contact.entry(pre_root).or_insert(i);
                    let slot = *pre_root_to_slot.entry(pre_root).or_insert_with(|| {
                        let s = batch_len + slot_to_pre_root.len();
                        slot_to_pre_root.push(pre_root);
                        s
                    });
                    touched_edges.push((local_i, slot));
                } else if n_sorted < i {
                    touched_edges.push((local_i, n_sorted - batch_start));
                }
            }
        }

        // 3. Connected components of the plateau graph (vertices: batch cells + touched pre-roots).
        let num_verts = batch_len + slot_to_pre_root.len();
        plateau_parent.clear();
        plateau_parent.extend(0..num_verts);
        for &(u, v) in &touched_edges {
            let ru = uf_find(&mut plateau_parent, u);
            let rv = uf_find(&mut plateau_parent, v);
            if ru != rv {
                let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
                plateau_parent[hi] = lo;
            }
        }

        // Group vertices by plateau component root in deterministic ascending order.
        let mut comp_groups: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
        for local_i in 0..batch_len {
            let root = uf_find(&mut plateau_parent, local_i);
            comp_groups
                .entry(root)
                .or_default()
                .0
                .push(batch_start + local_i);
        }
        for (k, &pre_root) in slot_to_pre_root.iter().enumerate() {
            let root = uf_find(&mut plateau_parent, batch_len + k);
            comp_groups.entry(root).or_default().1.push(pre_root);
        }

        let mut group_keys: Vec<usize> = comp_groups.keys().copied().collect();
        group_keys.sort_unstable();

        for gk in group_keys {
            let (batch_cells, pre_roots) = comp_groups.remove(&gk).unwrap();
            if pre_roots.is_empty() {
                // Pure local-minimum flat plateau at `z_batch`: union all plateau cells.
                let anchor = batch_cells[0];
                let mut total_cnt = 1usize;
                for &ci in &batch_cells[1..] {
                    uf_parent[ci] = anchor;
                    total_cnt += 1;
                    splice_list(anchor, ci, &mut list_head, &mut list_tail, &mut next_pix);
                }
                uf_count[anchor] = total_cnt;
                continue;
            }

            let mut sig_roots: Vec<usize> = Vec::new();
            let mut noise_roots: Vec<usize> = Vec::new();
            for r in pre_roots {
                let is_sig = uf_active_node[r].is_some()
                    || (uf_count[r] >= min_size_eff
                        && (z_batch - uf_min_elev[r]) >= options.min_depth);
                if is_sig {
                    sig_roots.push(r);
                } else {
                    noise_roots.push(r);
                }
            }

            if sig_roots.len() >= 2 {
                // Multi-branch saddle event at exact elevation `z_batch`: emit ONE compound
                // parent with ALL qualifying branches merged at `z_batch` as direct children.
                sig_roots.sort_unstable_by(|&a, &b| {
                    uf_min_elev[a]
                        .total_cmp(&uf_min_elev[b])
                        .then(uf_min_idx[a].cmp(&uf_min_idx[b]))
                });

                // Each finalized child receives its own deterministic plateau contact cell
                // recorded in `pre_root_first_contact` (two children bordering the same saddle
                // cell naturally share that coordinate, while distinct contact saddles keep
                // their respective coordinates without any linear membership scan).
                let fallback_pp = cells[batch_cells[0]];
                let mut parent_pp = fallback_pp;
                let mut parent_pp_idx = usize::MAX;
                let mut child_nidxs = Vec::with_capacity(sig_roots.len());
                for &sr in &sig_roots {
                    let contact_idx = pre_root_first_contact.get(&sr).copied();
                    let child_pp = contact_idx.map_or(fallback_pp, |ci| cells[ci]);
                    if let Some(ci) = contact_idx
                        && ci < parent_pp_idx
                    {
                        parent_pp_idx = ci;
                        parent_pp = child_pp;
                    }
                    let cn = finalize_component(
                        sr,
                        z_batch,
                        child_pp.r,
                        child_pp.c,
                        &uf_count,
                        &uf_min_elev,
                        &mut uf_active_node,
                        &mut list_head,
                        &mut list_tail,
                        &next_pix,
                        &mut owned_node,
                        &mut local_nodes,
                    );
                    child_nidxs.push(cn);
                }

                let merged = sig_roots[0];
                let mut total_cnt = uf_count[merged];
                let mut min_z = uf_min_elev[merged];
                let mut min_idx = uf_min_idx[merged];

                for &sr in &sig_roots[1..] {
                    uf_parent[sr] = merged;
                    total_cnt += uf_count[sr];
                    if uf_min_elev[sr] < min_z
                        || (uf_min_elev[sr] == min_z && uf_min_idx[sr] < min_idx)
                    {
                        min_z = uf_min_elev[sr];
                        min_idx = uf_min_idx[sr];
                    }
                }
                for &nr in &noise_roots {
                    uf_parent[nr] = merged;
                    total_cnt += uf_count[nr];
                    if uf_min_elev[nr] < min_z
                        || (uf_min_elev[nr] == min_z && uf_min_idx[nr] < min_idx)
                    {
                        min_z = uf_min_elev[nr];
                        min_idx = uf_min_idx[nr];
                    }
                    splice_list(merged, nr, &mut list_head, &mut list_tail, &mut next_pix);
                }
                for &ci in &batch_cells {
                    uf_parent[ci] = merged;
                    total_cnt += 1;
                    splice_list(merged, ci, &mut list_head, &mut list_tail, &mut next_pix);
                }

                uf_count[merged] = total_cnt;
                uf_min_elev[merged] = min_z;
                uf_min_idx[merged] = min_idx;

                let parent_nidx = local_nodes.len();
                for &cn in &child_nidxs {
                    local_nodes[cn].parent_idx = Some(parent_nidx);
                }
                local_nodes.push(LocalNode {
                    parent_idx: None,
                    children_idxs: child_nidxs,
                    level: 1,
                    min_elev: min_z,
                    saddle_elev: z_batch,
                    pour_point_row: parent_pp.r,
                    pour_point_col: parent_pp.c,
                    count: total_cnt,
                    area: total_cnt as f64 * cell_area,
                    max_depth: 0.0,
                    is_leaf: false,
                });
                uf_active_node[merged] = Some(parent_nidx);
            } else {
                // At most one significant branch: absorb noise branches and plateau cells into winner.
                let winner = if sig_roots.len() == 1 {
                    sig_roots[0]
                } else {
                    *noise_roots
                        .iter()
                        .min_by(|&&a, &&b| {
                            uf_min_elev[a]
                                .total_cmp(&uf_min_elev[b])
                                .then(Reverse(uf_count[a]).cmp(&Reverse(uf_count[b])))
                                .then(uf_min_idx[a].cmp(&uf_min_idx[b]))
                        })
                        .unwrap()
                };

                let mut total_cnt = uf_count[winner];
                let mut min_z = uf_min_elev[winner];
                let mut min_idx = uf_min_idx[winner];

                for &nr in &noise_roots {
                    if nr == winner {
                        continue;
                    }
                    uf_parent[nr] = winner;
                    total_cnt += uf_count[nr];
                    if uf_min_elev[nr] < min_z
                        || (uf_min_elev[nr] == min_z && uf_min_idx[nr] < min_idx)
                    {
                        min_z = uf_min_elev[nr];
                        min_idx = uf_min_idx[nr];
                    }
                    splice_list(winner, nr, &mut list_head, &mut list_tail, &mut next_pix);
                }
                for &ci in &batch_cells {
                    uf_parent[ci] = winner;
                    total_cnt += 1;
                    splice_list(winner, ci, &mut list_head, &mut list_tail, &mut next_pix);
                }

                uf_count[winner] = total_cnt;
                uf_min_elev[winner] = min_z;
                uf_min_idx[winner] = min_idx;
            }
        }

        batch_start = batch_end;
    }

    // Finalize the single region root component using exact exterior outlet when provided,
    // or the highest retained sink cell as lower-bound rim estimate otherwise.
    let root_uf = uf_find(&mut uf_parent, m - 1);
    let top_cell = cells[m - 1];
    let (root_z, root_r, root_c) = match root_outlet {
        Some(out) if out.z >= top_cell.z => (out.z, out.r, out.c),
        _ => (top_cell.z, top_cell.r, top_cell.c),
    };
    let root_nidx = finalize_component(
        root_uf,
        root_z,
        root_r,
        root_c,
        &uf_count,
        &uf_min_elev,
        &mut uf_active_node,
        &mut list_head,
        &mut list_tail,
        &next_pix,
        &mut owned_node,
        &mut local_nodes,
    );

    // Bottom-up level assignment (every child was pushed before its parent).
    for idx in 0..local_nodes.len() {
        if local_nodes[idx].children_idxs.is_empty() {
            local_nodes[idx].level = 1;
            local_nodes[idx].is_leaf = true;
        } else {
            let max_child_lvl = local_nodes[idx]
                .children_idxs
                .iter()
                .map(|&c| local_nodes[c].level)
                .max()
                .unwrap_or(0);
            local_nodes[idx].level = max_child_lvl + 1;
            local_nodes[idx].is_leaf = false;
        }
    }

    // Reorder local nodes in deterministic pre-order starting from `root_nidx`
    // so local ID 1 is the root (matching `slicing.rs` conventions).
    let mut preorder: Vec<usize> = Vec::with_capacity(local_nodes.len());
    let mut stack = vec![root_nidx];
    while let Some(u) = stack.pop() {
        preorder.push(u);
        for &c in local_nodes[u].children_idxs.iter().rev() {
            stack.push(c);
        }
    }

    let num_nodes = preorder.len();
    let mut old_to_new_id = vec![0u32; local_nodes.len()];
    for (pos, &old_idx) in preorder.iter().enumerate() {
        old_to_new_id[old_idx] = (pos + 1) as u32;
    }

    // Build sparse `(global_linear_idx, local_node_id)` assignments and per-node owned cell lists.
    let mut pixel_assignments: Vec<(usize, u32)> = Vec::with_capacity(m);
    let mut node_owned_cells: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for (i, cell) in cells.iter().enumerate() {
        let old_n = owned_node[i];
        if old_n != usize::MAX {
            let nid = old_to_new_id[old_n];
            pixel_assignments.push((cell.r * full_w + cell.c, nid));
            node_owned_cells[(nid - 1) as usize].push(i);
        }
    }

    // Build pre-order MergeTreeNode vector (volume & avg_depth finalized below using each
    // node's own local floor `node.min_elev` to eliminate cross-basin cancellation and match
    // the exact rating-curve endpoint identically).
    let mut nodes = Vec::with_capacity(num_nodes);
    for (pos, &old_idx) in preorder.iter().enumerate() {
        let ln = &local_nodes[old_idx];
        let id = (pos + 1) as u32;
        let parent_id = ln.parent_idx.map_or(0, |p| old_to_new_id[p]);
        let children_ids: Vec<u32> = ln.children_idxs.iter().map(|&c| old_to_new_id[c]).collect();
        let (pour_point_x, pour_point_y) = cell_center_xy(gt, ln.pour_point_row, ln.pour_point_col);
        nodes.push(MergeTreeNode {
            id,
            parent_id,
            children_ids,
            level: ln.level,
            region_id,
            min_elev: ln.min_elev,
            saddle_elev: ln.saddle_elev,
            pour_point_row: ln.pour_point_row,
            pour_point_col: ln.pour_point_col,
            pour_point_x,
            pour_point_y,
            count: ln.count,
            area: ln.area,
            volume: 0.0,
            max_depth: ln.max_depth,
            avg_depth: 0.0,
            is_leaf: ln.is_leaf,
        });
    }

    // Bottom-up Chan's central-moment merge and Euler-tour border-set merge avoid
    // `O(sum |subtree|)` coordinate rescans while preserving exact second central moments,
    // bounding boxes, and Benkrid-Crookes perimeters.
    #[derive(Clone, Copy)]
    struct NodeMoments {
        n: f64,
        mean_r: f64,
        mean_c: f64,
        m20: f64,
        m02: f64,
        m11: f64,
        mean_z: f64,
        r0: usize,
        c0: usize,
        r1: usize,
        c1: usize,
    }
    impl NodeMoments {
        fn empty() -> Self {
            Self {
                n: 0.0,
                mean_r: 0.0,
                mean_c: 0.0,
                m20: 0.0,
                m02: 0.0,
                m11: 0.0,
                mean_z: 0.0,
                r0: usize::MAX,
                c0: usize::MAX,
                r1: 0,
                c1: 0,
            }
        }
        fn merge_into(&mut self, other: &Self) {
            if other.n <= 0.0 {
                return;
            }
            if self.n <= 0.0 {
                *self = *other;
                return;
            }
            let n_ab = self.n + other.n;
            let dr = other.mean_r - self.mean_r;
            let dc = other.mean_c - self.mean_c;
            let dz = other.mean_z - self.mean_z;
            let w_ab = (self.n * other.n) / n_ab;
            self.mean_r += dr * (other.n / n_ab);
            self.mean_c += dc * (other.n / n_ab);
            self.mean_z += dz * (other.n / n_ab);
            self.m20 += other.m20 + dr * dr * w_ab;
            self.m02 += other.m02 + dc * dc * w_ab;
            self.m11 += other.m11 + dr * dc * w_ab;
            self.n = n_ab;
            self.r0 = self.r0.min(other.r0);
            self.c0 = self.c0.min(other.c0);
            self.r1 = self.r1.max(other.r1);
            self.c1 = self.c1.max(other.c1);
        }
    }

    let (vx, vy) = (vx_m, vy_m);
    let res_x = vx.0.hypot(vx.1);
    let res_y = vy.0.hypot(vy.1);
    let max_side = res_x.max(res_y);
    let min_res = if max_side > 0.0 && cell_area > 0.0 {
        (cell_area / max_side).min(res_x.min(res_y))
    } else {
        res_x.min(res_y)
    };

    // Because `nodes` are already in DFS preorder (1-based IDs `1..=num_nodes`), subtree
    // membership for node `u` is a contiguous half-open ID range `[u_id, subtree_end[u])`.
    let mut subtree_end = vec![0u32; num_nodes];
    for u in (0..num_nodes).rev() {
        let mut end = nodes[u].id + 1;
        for &cid in &nodes[u].children_ids {
            end = end.max(subtree_end[(cid - 1) as usize]);
        }
        subtree_end[u] = end;
    }

    // Map each region cell's `(row, col)` to its owned preorder node ID for O(1) 4-neighbor
    // interior/border checks during bottom-up border-set merging.
    let bbox_cells = bw.saturating_mul(bh);
    let use_dense_owner = bbox_cells <= m.saturating_mul(4).max(256);
    let mut dense_owner: Vec<u32> = if use_dense_owner {
        vec![0u32; bbox_cells]
    } else {
        Vec::new()
    };
    let mut sparse_owner: HashMap<(usize, usize), u32> = if use_dense_owner {
        HashMap::new()
    } else {
        HashMap::with_capacity(m)
    };
    for (u, owned) in node_owned_cells.iter().enumerate() {
        let nid = (u + 1) as u32;
        for &ci in owned {
            let cell = cells[ci];
            if use_dense_owner {
                dense_owner[(cell.r - r0) * bw + (cell.c - c0)] = nid;
            } else {
                sparse_owner.insert((cell.r, cell.c), nid);
            }
        }
    }
    let owner_at = |r: isize, c: isize| -> u32 {
        if r < r0 as isize || c < c0 as isize || r >= r1 as isize || c >= c1 as isize {
            return 0;
        }
        let ru = r as usize;
        let cu = c as usize;
        if use_dense_owner {
            dense_owner[(ru - r0) * bw + (cu - c0)]
        } else {
            sparse_owner.get(&(ru, cu)).copied().unwrap_or(0)
        }
    };

    // Initialize owned-cell moments (shifted relative to region bbox origin `(r0, c0)` to
    // prevent catastrophic cancellation) and merge bottom-up using Chan's formula.
    let mut moments: Vec<NodeMoments> = Vec::with_capacity(num_nodes);
    let mut perimeters: Vec<f64> = vec![0.0; num_nodes];
    let mut borders: Vec<HashSet<(isize, isize)>> =
        (0..num_nodes).map(|_| HashSet::new()).collect();
    let r0_f = r0 as f64;
    let c0_f = c0 as f64;
    for owned in &node_owned_cells {
        let mut nm = NodeMoments::empty();
        if !owned.is_empty() {
            let mut sum_r = 0.0f64;
            let mut sum_c = 0.0f64;
            let mut sum_z = 0.0f64;
            let mut lr0 = usize::MAX;
            let mut lc0 = usize::MAX;
            let mut lr1 = 0usize;
            let mut lc1 = 0usize;
            for &ci in owned {
                let cell = cells[ci];
                let dr = cell.r as f64 - r0_f;
                let dc = cell.c as f64 - c0_f;
                let cf = dc * vx.0 + dr * vy.0;
                let rf = -(dc * vx.1 + dr * vy.1);
                sum_r += rf;
                sum_c += cf;
                sum_z += cell.z;
                lr0 = lr0.min(cell.r);
                lc0 = lc0.min(cell.c);
                lr1 = lr1.max(cell.r + 1);
                lc1 = lc1.max(cell.c + 1);
            }
            let nf = owned.len() as f64;
            let mean_r = sum_r / nf;
            let mean_c = sum_c / nf;
            let mean_z = sum_z / nf;
            let mut m20 = 0.0f64;
            let mut m02 = 0.0f64;
            let mut m11 = 0.0f64;
            for &ci in owned {
                let cell = cells[ci];
                let dr = cell.r as f64 - r0_f;
                let dc = cell.c as f64 - c0_f;
                let cf = dc * vx.0 + dr * vy.0 - mean_c;
                let rf = -(dc * vx.1 + dr * vy.1) - mean_r;
                m20 += rf * rf;
                m02 += cf * cf;
                m11 += rf * cf;
            }
            nm = NodeMoments {
                n: nf,
                mean_r,
                mean_c,
                m20,
                m02,
                m11,
                mean_z,
                r0: lr0,
                c0: lc0,
                r1: lr1,
                c1: lc1,
            };
        }
        moments.push(nm);
    }

    // Single bottom-up pass (reverse preorder): merge moments in O(1) per edge and merge
    // border sets small-to-large while pruning cells whose 4-neighbors all lie inside the
    // node's preorder interval `[u_id, u_end)`.
    for u in (0..num_nodes).rev() {
        let u_id = nodes[u].id;
        let u_end = subtree_end[u];
        let is_border = |ri: isize, ci: isize| -> bool {
            let n1 = owner_at(ri - 1, ci);
            let n2 = owner_at(ri + 1, ci);
            let n3 = owner_at(ri, ci - 1);
            let n4 = owner_at(ri, ci + 1);
            !(n1 >= u_id
                && n1 < u_end
                && n2 >= u_id
                && n2 < u_end
                && n3 >= u_id
                && n3 < u_end
                && n4 >= u_id
                && n4 < u_end)
        };

        // Pick largest child border set as base (small-to-large merge).
        let mut cur_border: HashSet<(isize, isize)> = HashSet::new();
        for &cid in &nodes[u].children_ids {
            let cidx = (cid - 1) as usize;
            let child_m = moments[cidx];
            moments[u].merge_into(&child_m);
            let child_b = std::mem::take(&mut borders[cidx]);
            if child_b.len() > cur_border.len() {
                let prev = std::mem::replace(&mut cur_border, child_b);
                cur_border.extend(prev);
            } else {
                cur_border.extend(child_b);
            }
        }
        if !nodes[u].children_ids.is_empty() {
            cur_border.retain(|&(ri, ci)| is_border(ri, ci));
        }
        for &ci in &node_owned_cells[u] {
            let cell = cells[ci];
            let ri = cell.r as isize;
            let cj = cell.c as isize;
            if is_border(ri, cj) {
                cur_border.insert((ri, cj));
            }
        }
        perimeters[u] = benkrid_crookes_perimeter_from_border_set(&cur_border, vx, vy);
        borders[u] = cur_border;
    }

    // Compute exact component-local storage volumes, optional (and optionally capped) exact
    // continuous rating curves, and NestedDepression morphometrics directly from the bottom-up
    // Chan moments + perimeters.
    let mut depressions = Vec::with_capacity(num_nodes);
    let mut rating_curves = Vec::new();
    let emit_curves = options.max_rating_curve_points != Some(0);
    let curve_cap = options.max_rating_curve_points.map(|c| c.max(2));
    let mut subtree_elevs: Vec<f64> = Vec::new();
    let mut stages: Vec<f64> = Vec::new();
    let mut desc_stack: Vec<u32> = Vec::new();

    for node_idx in 0..num_nodes {
        let node_id = nodes[node_idx].id;
        let node_min = nodes[node_idx].min_elev;
        let node_saddle = nodes[node_idx].saddle_elev;
        let node_level = nodes[node_idx].level;
        let node_area = nodes[node_idx].area;
        let node_max_depth = nodes[node_idx].max_depth;
        let nm = moments[node_idx];

        let node_volume = if !emit_curves {
            // Exact storage volume below `node_saddle` from bottom-up Chan elevation mean
            // (all subtree cells have z <= node_saddle; clamp to >= 0.0 for floating-point safety).
            ((node_saddle - nm.mean_z).max(0.0) * nm.n * cell_area).max(0.0)
        } else {
            subtree_elevs.clear();
            stages.clear();
            stages.push(node_min);
            stages.push(node_saddle);

            desc_stack.clear();
            desc_stack.push(node_id);
            while let Some(cur_id) = desc_stack.pop() {
                let cur = &nodes[(cur_id - 1) as usize];
                if cur_id != node_id {
                    stages.push(cur.min_elev.clamp(node_min, node_saddle));
                    stages.push(cur.saddle_elev.clamp(node_min, node_saddle));
                }
                for &ci in &node_owned_cells[(cur_id - 1) as usize] {
                    subtree_elevs.push(cells[ci].z);
                }
                for &cid in &cur.children_ids {
                    desc_stack.push(cid);
                }
            }

            // Sort subtree elevations and stages, deduplicating numerically equal stages
            // (`*a == *b`, collapsing -0.0 and +0.0 while retaining distinct tight saddles).
            subtree_elevs.sort_unstable_by(f64::total_cmp);
            stages.extend_from_slice(&subtree_elevs);
            stages.sort_unstable_by(f64::total_cmp);
            stages.dedup_by(|a, b| *a == *b);

            let n_stages = stages.len();
            let emit_stride_cap = curve_cap.filter(|&cap| n_stages > cap);
            let mut next_emit_slot = 0usize;
            let mut next_emit_idx = 0usize;

            // Accumulate storage interval-by-interval across EVERY consecutive sorted stage
            // so even capped rating curves preserve exact integrated cumulative volume.
            let mut cum_vol = 0.0f64;
            let mut vol_at_saddle = 0.0f64;
            let mut k = 0usize;
            let mut prev_stage = stages[0];
            let mut prev_k = 0usize;
            for (s_idx, &stage) in stages.iter().enumerate() {
                if s_idx > 0 {
                    let dz = (stage - prev_stage).max(0.0);
                    cum_vol += (prev_k as f64) * dz * cell_area;
                }
                while k < subtree_elevs.len() && subtree_elevs[k] <= stage {
                    k += 1;
                }
                if stage <= node_saddle {
                    vol_at_saddle = cum_vol;
                }
                let should_emit = match emit_stride_cap {
                    None => true,
                    Some(cap) => {
                        if s_idx == next_emit_idx || s_idx + 1 == n_stages {
                            next_emit_slot += 1;
                            if next_emit_slot < cap {
                                next_emit_idx = (next_emit_slot * (n_stages - 1)) / (cap - 1);
                            } else {
                                next_emit_idx = usize::MAX;
                            }
                            true
                        } else {
                            false
                        }
                    }
                };
                if should_emit {
                    let area = k as f64 * cell_area;
                    let depth = (stage - node_min).max(0.0);
                    rating_curves.push(RatingCurvePoint {
                        dep_id: node_id,
                        level: node_level,
                        region_id,
                        stage,
                        depth,
                        area,
                        volume: cum_vol,
                    });
                }
                prev_stage = stage;
                prev_k = k;
            }
            vol_at_saddle
        };

        let node_avg_depth = if node_area > 0.0 {
            node_volume / node_area
        } else {
            0.0
        };
        nodes[node_idx].volume = node_volume;
        nodes[node_idx].avg_depth = node_avg_depth;

        let (
            perimeter,
            major_axis,
            minor_axis,
            elongatedness,
            eccentricity,
            orientation,
            area_bbox_ratio,
        ) = if nm.n > 0.0 {
            let mu20 = (nm.m20 / nm.n).max(0.0);
            let mu02 = (nm.m02 / nm.n).max(0.0);
            let mu11 = nm.m11 / nm.n;
            let common = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();
            let l1 = (0.5 * (mu20 + mu02 + common)).max(0.0);
            let l2 = (0.5 * (mu20 + mu02 - common)).max(0.0);
            let maj = 4.0 * l1.sqrt();
            let min_ax = (4.0 * l2.sqrt()).max(min_res);
            let elong = if min_ax > 0.0 { maj / min_ax } else { 0.0 };
            let ecc = if l1 > 0.0 {
                (1.0 - l2 / l1).max(0.0).sqrt()
            } else {
                0.0
            };
            let orient_rad = if (mu20 - mu02).abs() == 0.0 {
                if mu11 < 0.0 { -FRAC_PI_4 } else { FRAC_PI_4 }
            } else {
                0.5 * (2.0 * mu11).atan2(mu20 - mu02)
            };
            let orient = orient_rad / PI * 180.0;
            let bw_n = nm.c1.saturating_sub(nm.c0) as f64;
            let bh_n = nm.r1.saturating_sub(nm.r0) as f64;
            let bbox_area = bw_n * bh_n;
            let abr = if bbox_area > 0.0 {
                nm.n / bbox_area
            } else {
                0.0
            };
            (perimeters[node_idx], maj, min_ax, elong, ecc, orient, abr)
        } else {
            (0.0, 0.0, min_res, 0.0, 0.0, 0.0, 0.0)
        };

        let (circularity, depth_diam_ratio, hypsometric_integral, rim_slope_deg, artifact_flag) =
            extended_morphometrics(
                node_area,
                perimeter,
                node_avg_depth,
                node_max_depth,
                elongatedness,
                area_bbox_ratio,
            );
        depressions.push(NestedDepression {
            id: node_id,
            level: node_level,
            count: nodes[node_idx].count,
            area: node_area,
            volume: node_volume,
            avg_depth: node_avg_depth,
            max_depth: node_max_depth,
            min_elev: node_min,
            max_elev: node_saddle,
            children_ids: nodes[node_idx].children_ids.clone(),
            region_id,
            perimeter,
            major_axis,
            minor_axis,
            elongatedness,
            eccentricity,
            orientation,
            area_bbox_ratio,
            parent_id: nodes[node_idx].parent_id,
            circularity,
            depth_diam_ratio,
            hypsometric_integral,
            rim_slope_deg,
            artifact_flag,
        });
    }

    RegionMergeOutput {
        region_idx,
        pixel_assignments,
        nodes,
        depressions,
        rating_curves,
    }
}

fn delineate_merge_tree_impl(
    sink: &Array2D<f64>,
    source_dem: Option<&Array2D<f64>>,
    options: &MergeTreeOptions,
) -> Result<MergeTreeResult> {
    if !options.min_depth.is_finite() || options.min_depth < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "min_depth must be finite and >= 0, got {}",
            options.min_depth
        )));
    }
    if let Some(dem) = source_dem {
        validate_grid_alignment(sink, dem, "dem")?;
    }

    let w = sink.width();
    let h = sink.height();
    let n = w
        .checked_mul(h)
        .ok_or_else(|| Error::DimensionMismatch(format!("raster dimensions overflow ({w}x{h})")))?;

    let gt = sink.geotransform;
    let (vx_m, vy_m, _, _, cell_area) = safe_crs_metric_basis(&sink.projection, &gt, w, h);
    let no_data_raw = sink.no_data();
    let is_valid_sink = |v: f64| -> bool { v.is_finite() && !v.rd_eq(no_data_raw) };

    let min_count = options.min_size.max(1);
    let (label_objects, nb_labels) = label_sink_regions(
        sink.data(),
        w,
        h,
        min_count,
        options.topology,
        is_valid_sink,
    );

    let k = nb_labels as usize;
    let mut bboxes = vec![(usize::MAX, usize::MAX, 0usize, 0usize); k];
    let mut region_cells: Vec<Vec<SinkCell>> = vec![Vec::new(); k];
    let mut reg_max_elev = vec![f64::NEG_INFINITY; k];
    let mut region_outlets: Vec<Option<RootOutlet>> = vec![None; k];

    let sdata = sink.data();
    let offsets: &[(isize, isize)] = match options.topology {
        Topology::D4 => &D4_OFFSETS,
        Topology::D8 => &D8_OFFSETS,
    };

    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let lbl = label_objects[off + c];
            if lbl == 0 {
                continue;
            }
            let idx = (lbl - 1) as usize;
            let b = &mut bboxes[idx];
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
            let z = sdata[off + c];
            if z > reg_max_elev[idx] {
                reg_max_elev[idx] = z;
            }
            region_cells[idx].push(SinkCell { z, r, c });
        }
    }

    // When the source DEM is provided, identify the exact lowest exterior rim/outlet neighbor
    // at or above the region's maximum interior elevation so root spill elevation and storage
    // volume reflect the true topographic pour point without being hijacked by pruned exterior
    // pits. When `options.dem_is_refined` is true (e.g., `InMemorySinks::dem_refined`), all
    // pruned exterior pits are already filled so we avoid cloning and re-filling `source_dem`.
    if let Some(dem) = source_dem
        && k > 0
    {
        let filled_opt = if options.dem_is_refined {
            None
        } else {
            let mut filled = dem.clone();
            richdem::fill_depressions(&mut filled, options.topology);
            Some(filled)
        };
        let ddata = dem.data();
        let d_nd = dem.no_data();
        let fdata = filled_opt.as_ref().map(Array2D::data);
        let mut fallback_outlets: Vec<Option<RootOutlet>> = vec![None; k];

        for (idx, cells) in region_cells.iter().enumerate() {
            let lbl = (idx + 1) as u32;
            let max_z = reg_max_elev[idx];
            for cell in cells {
                for &(dr, dc) in offsets {
                    let nr = cell.r as isize + dr;
                    let nc = cell.c as isize + dc;
                    if nr < 0 || nc < 0 || nr >= h as isize || nc >= w as isize {
                        continue;
                    }
                    let nru = nr as usize;
                    let ncu = nc as usize;
                    let nidx = nru * w + ncu;
                    if label_objects[nidx] == lbl {
                        continue;
                    }
                    let dz = ddata[nidx];
                    if !dz.is_finite() || dz.rd_eq(d_nd) || dz < max_z {
                        continue;
                    }
                    let is_unfilled_rim = match fdata {
                        Some(fd) => fd[nidx].is_finite() && fd[nidx] <= dz + 1e-12,
                        None => true,
                    };
                    let slot = if is_unfilled_rim {
                        &mut region_outlets[idx]
                    } else {
                        &mut fallback_outlets[idx]
                    };
                    let replace = match *slot {
                        None => true,
                        Some(cur) => dz < cur.z || (dz == cur.z && (nru, ncu) < (cur.r, cur.c)),
                    };
                    if replace {
                        *slot = Some(RootOutlet {
                            z: dz,
                            r: nru,
                            c: ncu,
                        });
                    }
                }
            }
            if region_outlets[idx].is_none() {
                region_outlets[idx] = fallback_outlets[idx];
            }
        }
    }

    type RegionBbox = (usize, usize, usize, usize);
    type RegionWorkItem = (usize, RegionBbox, Vec<SinkCell>, Option<RootOutlet>);

    let work: Vec<RegionWorkItem> = bboxes
        .into_iter()
        .zip(region_cells)
        .zip(region_outlets)
        .enumerate()
        .map(|(idx, ((bbox, cells), outlet))| (idx, bbox, cells, outlet))
        .collect();

    #[cfg(feature = "parallel")]
    let mut region_results: Vec<RegionMergeOutput> = work
        .into_par_iter()
        .map(|(idx, bbox, cells, outlet)| {
            build_region_merge_tree(
                idx,
                (idx + 1) as u32,
                w,
                bbox,
                cells,
                outlet,
                &gt,
                vx_m,
                vy_m,
                cell_area,
                options,
            )
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let mut region_results: Vec<RegionMergeOutput> = work
        .into_iter()
        .map(|(idx, bbox, cells, outlet)| {
            build_region_merge_tree(
                idx,
                (idx + 1) as u32,
                w,
                bbox,
                cells,
                outlet,
                &gt,
                vx_m,
                vy_m,
                cell_area,
                options,
            )
        })
        .collect();

    region_results.sort_by_key(|r| r.region_idx);

    let mut obj_vec = vec![0i32; n];
    let mut level_vec = vec![0i32; n];
    let mut global_nodes: Vec<MergeTreeNode> = Vec::new();
    let mut global_deps: Vec<NestedDepression> = Vec::new();
    let mut global_curves: Vec<RatingCurvePoint> = Vec::new();
    let mut max_level = 0u32;
    let mut uid_offset = 0u32;

    for mut res in region_results {
        if res.nodes.is_empty() {
            continue;
        }

        for node in &mut res.nodes {
            node.id += uid_offset;
            if node.parent_id > 0 {
                node.parent_id += uid_offset;
            }
            for cid in &mut node.children_ids {
                *cid += uid_offset;
            }
            if node.level > max_level {
                max_level = node.level;
            }
        }
        for dep in &mut res.depressions {
            dep.id += uid_offset;
            if dep.parent_id > 0 {
                dep.parent_id += uid_offset;
            }
            for cid in &mut dep.children_ids {
                *cid += uid_offset;
            }
        }
        for pt in &mut res.rating_curves {
            pt.dep_id += uid_offset;
        }

        // Stitch only occupied cells from the sparse assignment vector.
        for (gidx, local_id) in res.pixel_assignments {
            let global_id = local_id + uid_offset;
            let lvl = res.nodes[(local_id - 1) as usize].level;
            obj_vec[gidx] = global_id as i32;
            level_vec[gidx] = lvl as i32;
        }

        uid_offset += res.nodes.len() as u32;
        global_nodes.extend(res.nodes);
        global_deps.extend(res.depressions);
        global_curves.extend(res.rating_curves);
    }

    let mut depression_id = Array2D::from_vec(w, h, obj_vec)?;
    depression_id.copy_metadata_from(sink);
    depression_id.set_no_data(0);

    let mut depression_level = Array2D::from_vec(w, h, level_vec)?;
    depression_level.copy_metadata_from(sink);
    depression_level.set_no_data(0);

    Ok(MergeTreeResult {
        nodes: global_nodes,
        depressions: InMemoryDepressions {
            depression_id,
            depression_level,
            depressions: global_deps,
            max_level,
            num_regions: nb_labels,
        },
        rating_curves: global_curves,
    })
}

/// Delineates the exact nested depression hierarchy using a bottom-up Union-Find
/// merge tree over sink cells sorted once in ascending elevation order.
///
/// Internal saddles and pour points are recovered to machine precision independent of
/// any discrete slicing interval. Because a masked `sink` raster excludes exterior rim
/// cells whose fill depth is zero, root nodes use the highest retained sink cell as a
/// lower-bound rim estimate; pass the original DEM via
/// [`delineate_depressions_merge_tree_with_dem_in_memory`] when exact exterior root
/// outlet elevations and volumes are required.
///
/// The returned [`MergeTreeResult::depressions`] is directly compatible with
/// [`crate::fsm::simulate_fill_spill_merge_in_memory`],
/// [`crate::graph::build_connectivity_graph`], and [`crate::inundation`].
pub fn delineate_depressions_merge_tree_in_memory(
    sink: &Array2D<f64>,
    options: &MergeTreeOptions,
) -> Result<MergeTreeResult> {
    delineate_merge_tree_impl(sink, None, options)
}

/// Delineates the exact nested depression hierarchy using both the masked `sink` raster
/// and the source `dem` (e.g., `dem_refined` or original DEM) so root depressions are
/// finalized against their true lowest exterior outlet cell on the surrounding rim.
///
/// # Performance and Memory
///
/// When at least one sink region is present, this function clones `dem` and runs a
/// full-grid Priority-Flood depression fill (`O(W * H)` extra memory and `O(N log N)`
/// time) to distinguish true unfilled rim cells from pruned exterior pits regardless of
/// whether the caller supplied the original DEM or `dem_refined`.
pub fn delineate_depressions_merge_tree_with_dem_in_memory(
    sink: &Array2D<f64>,
    dem: &Array2D<f64>,
    options: &MergeTreeOptions,
) -> Result<MergeTreeResult> {
    delineate_merge_tree_impl(sink, Some(dem), options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filling::extract_sinks_in_memory;

    #[test]
    fn exact_saddle_recovered_below_coarse_interval() {
        // Two 5x5 bowls connected by a single exact saddle cell at (3, 6) with z = 10.37.
        let (w, h) = (13, 7);
        let mut data = vec![0.0f64; w * h];
        for r in 1..6 {
            for c in 1..12 {
                data[r * w + c] = 12.0;
            }
        }
        for r in 2..5 {
            for c in 2..5 {
                let d = ((r as i32 - 3).abs() + (c as i32 - 3).abs()) as f64;
                data[r * w + c] = 8.0 + d * 0.5;
            }
        }
        for r in 2..5 {
            for c in 8..11 {
                let d = ((r as i32 - 3).abs() + (c as i32 - 9).abs()) as f64;
                data[r * w + c] = 8.5 + d * 0.5;
            }
        }
        data[3 * w + 5] = 10.20;
        data[3 * w + 6] = 10.37;
        data[3 * w + 7] = 10.20;

        let mut sink = Array2D::from_vec(w, h, data).unwrap();
        sink.set_no_data(0.0);
        sink.geotransform = [100.0, 2.0, 0.0, 200.0, 0.0, -2.0];

        let opts = MergeTreeOptions {
            min_size: 5,
            min_depth: 0.5,
            topology: Topology::D4,
            ..Default::default()
        };
        let res = delineate_depressions_merge_tree_in_memory(&sink, &opts).unwrap();
        assert_eq!(res.depressions.max_level, 2);
        assert_eq!(res.nodes.len(), 3);

        let root = &res.nodes[0];
        assert_eq!(root.parent_id, 0);
        assert_eq!(root.level, 2);
        assert!(!root.is_leaf);
        assert_eq!(root.children_ids.len(), 2);

        for &cid in &root.children_ids {
            let child = &res.nodes[(cid - 1) as usize];
            assert!(child.is_leaf);
            assert_eq!(child.parent_id, root.id);
            assert!((child.saddle_elev - 10.37).abs() < 1e-12);
            assert_eq!((child.pour_point_row, child.pour_point_col), (3, 6));
            assert!((child.pour_point_x - (100.0 + 6.5 * 2.0)).abs() < 1e-12);
            assert!((child.pour_point_y - (200.0 - 3.5 * 2.0)).abs() < 1e-12);
        }

        let root_curve: Vec<&RatingCurvePoint> = res
            .rating_curves
            .iter()
            .filter(|p| p.dep_id == root.id)
            .collect();
        assert!(root_curve.iter().any(|p| (p.stage - 10.37).abs() < 1e-12));
        let last = root_curve.last().unwrap();
        assert!((last.volume - root.volume).abs() < 1e-10);
        assert!((last.area - root.area).abs() < 1e-10);
    }

    #[test]
    fn component_local_volume_avoids_cancellation_from_deep_sibling() {
        // D4 row [-1e16, 11, 10, 12] with unit cells, min_size=1, min_depth=1.
        // Shallow branch at z=10 spills at saddle 11 -> exact volume = 1.0 (not 2.0!).
        let mut sink = Array2D::from_vec(4, 1, vec![-1e16, 11.0, 10.0, 12.0]).unwrap();
        sink.set_no_data(-9999.0);
        sink.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];

        let res = delineate_depressions_merge_tree_in_memory(
            &sink,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();

        let shallow = res
            .nodes
            .iter()
            .find(|n| (n.min_elev - 10.0).abs() < 1e-12)
            .expect("expected shallow branch at min_elev=10");
        assert!((shallow.saddle_elev - 11.0).abs() < 1e-12);
        assert!(
            (shallow.volume - 1.0).abs() < 1e-12,
            "expected exact local volume 1.0, got {}",
            shallow.volume
        );
        let rc_last = res
            .rating_curves
            .iter()
            .rfind(|p| p.dep_id == shallow.id)
            .unwrap();
        assert!((rc_last.volume - shallow.volume).abs() < 1e-12);
    }

    #[test]
    fn same_elevation_multi_basin_merge_emits_single_compound_parent() {
        // D4 row [0, 5, 1, 5, 2, 6] with min_size=1, min_depth=1:
        // all three basins (0, 1, 2) join at elevation 5 -> ONE level-2 compound with 3 children!
        let mut sink = Array2D::from_vec(6, 1, vec![0.0, 5.0, 1.0, 5.0, 2.0, 6.0]).unwrap();
        sink.set_no_data(-9999.0);
        sink.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];

        let res = delineate_depressions_merge_tree_in_memory(
            &sink,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(res.depressions.max_level, 2);
        assert_eq!(res.nodes.len(), 4);
        let root = &res.nodes[0];
        assert_eq!(root.level, 2);
        assert_eq!(root.children_ids.len(), 3);
        for &cid in &root.children_ids {
            let child = &res.nodes[(cid - 1) as usize];
            assert_eq!(child.level, 1);
            assert!((child.saddle_elev - 5.0).abs() < 1e-12);
            if (child.min_elev - 2.0).abs() < 1e-12 {
                // Basin at col 4 contacts the plateau at col 3 (not col 1).
                assert_eq!((child.pour_point_row, child.pour_point_col), (0, 3));
            } else {
                // Basins at cols 0 and 2 border the saddle at col 1.
                assert_eq!((child.pour_point_row, child.pour_point_col), (0, 1));
            }
        }
    }

    #[test]
    fn interval_storage_stays_finite_for_extreme_range_and_collapses_signed_zero_stages() {
        // 1. Extreme [0.0, 1e308] basin: exact storage at stage 1e308 is 1e308 (no overflow).
        let mut ext = Array2D::from_vec(2, 1, vec![0.0, 1e308]).unwrap();
        ext.set_no_data(-9999.0);
        ext.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];
        let res_ext = delineate_depressions_merge_tree_in_memory(
            &ext,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res_ext.nodes.len(), 1);
        assert!(res_ext.nodes[0].volume.is_finite());
        assert_eq!(res_ext.nodes[0].volume, 1e308);
        assert_eq!(res_ext.rating_curves.last().unwrap().volume, 1e308);

        // 2. [-0.0, +0.0, 1.0]: -0.0 and +0.0 collapse into one rating-curve point.
        let mut sz = Array2D::from_vec(3, 1, vec![-0.0, 0.0, 1.0]).unwrap();
        sz.set_no_data(-9999.0);
        sz.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];
        let res_sz = delineate_depressions_merge_tree_in_memory(
            &sz,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 0.5,
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res_sz.nodes.len(), 1);
        assert_eq!(
            res_sz.rating_curves.len(),
            2,
            "expected -0.0 and +0.0 to collapse into 1 stage, got {:?}",
            res_sz.rating_curves
        );
        assert_eq!(res_sz.rating_curves[0].stage, 0.0);
        assert_eq!(res_sz.rating_curves[1].stage, 1.0);
        assert!((res_sz.nodes[0].volume - 2.0).abs() < 1e-12);
    }

    #[test]
    fn root_finalization_with_dem_recovers_exact_outlet_and_storage() {
        // 5x5 DEM: boundary 20 except one outlet at (0, 2) = 10; 9 interior cells = 4 except center = 1.
        let mut data = vec![20.0f64; 25];
        data[2] = 10.0; // exact outlet at row 0, col 2
        for r in 1..=3 {
            for c in 1..=3 {
                data[r * 5 + c] = 4.0;
            }
        }
        data[2 * 5 + 2] = 1.0;

        let mut dem = Array2D::from_vec(5, 5, data).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, 5.0, 0.0, -1.0];

        let sinks = extract_sinks_in_memory(&dem, 1, Topology::D8).unwrap();
        let res = delineate_depressions_merge_tree_with_dem_in_memory(
            &sinks.sink,
            &dem,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D8,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(res.nodes.len(), 1);
        let root = &res.nodes[0];
        assert!((root.saddle_elev - 10.0).abs() < 1e-12);
        assert_eq!((root.pour_point_row, root.pour_point_col), (0, 2));
        // Exact volume: 8*(10-4) + 1*(10-1) = 48 + 9 = 57.
        assert!(
            (root.volume - 57.0).abs() < 1e-12,
            "expected exact root volume 57.0, got {}",
            root.volume
        );
    }

    #[test]
    fn rating_curve_retains_every_distinct_breakpoint_and_tight_saddle() {
        // 1. Single basin with 65 unit cells at elevations 1..=65: must keep stage 2 (volume 1.0).
        let vals: Vec<f64> = (1..=65).map(|v| v as f64).collect();
        let mut sink65 = Array2D::from_vec(65, 1, vals).unwrap();
        sink65.set_no_data(-9999.0);
        sink65.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];
        let res65 = delineate_depressions_merge_tree_in_memory(
            &sink65,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res65.rating_curves.len(), 65);
        let p2 = res65
            .rating_curves
            .iter()
            .find(|p| p.stage == 2.0)
            .expect("exact rating curve must retain stage 2 without downsampling");
        assert!((p2.volume - 1.0).abs() < 1e-12);

        // 2. D4 row [1, 1 + 2^-41, 1, 2]: exact dedup preserves representable saddle 1 + 2^-41
        //    on the root curve and preserves distinct floor & spill endpoints on both leaves!
        let saddle = 1.0 + 2.0f64.powi(-41);
        let mut tight = Array2D::from_vec(4, 1, vec![1.0, saddle, 1.0, 2.0]).unwrap();
        tight.set_no_data(-9999.0);
        tight.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];
        let res_t = delineate_depressions_merge_tree_in_memory(
            &tight,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 2.0f64.powi(-42),
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res_t.nodes.len(), 3);
        let root_id = res_t.nodes[0].id;
        assert!(
            res_t
                .rating_curves
                .iter()
                .any(|p| p.dep_id == root_id && p.stage == saddle),
            "root curve must preserve exact representable saddle 1 + 2^-41"
        );
        for &cid in &res_t.nodes[0].children_ids {
            let pts: Vec<&RatingCurvePoint> = res_t
                .rating_curves
                .iter()
                .filter(|p| p.dep_id == cid)
                .collect();
            assert_eq!(pts.len(), 2, "leaf must retain both floor and spill stage");
            assert_eq!(pts[0].stage, 1.0);
            assert_eq!(pts[1].stage, saddle);
        }
    }

    #[test]
    fn rating_curve_cap_and_dem_is_refined_preserve_exact_volumes_and_morphometrics() {
        let vals: Vec<f64> = (1..=65).map(|v| v as f64).collect();
        let mut sink65 = Array2D::from_vec(65, 1, vals).unwrap();
        sink65.set_no_data(-9999.0);
        sink65.geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0];

        let full = delineate_depressions_merge_tree_in_memory(
            &sink65,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                ..Default::default()
            },
        )
        .unwrap();
        let capped = delineate_depressions_merge_tree_in_memory(
            &sink65,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                max_rating_curve_points: Some(10),
                ..Default::default()
            },
        )
        .unwrap();
        let disabled = delineate_depressions_merge_tree_in_memory(
            &sink65,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D4,
                max_rating_curve_points: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(capped.rating_curves.len(), 10);
        assert_eq!(capped.rating_curves.first().unwrap().stage, 1.0);
        assert_eq!(capped.rating_curves.last().unwrap().stage, 65.0);
        assert!((capped.rating_curves.last().unwrap().volume - full.nodes[0].volume).abs() < 1e-9);
        assert!(disabled.rating_curves.is_empty());
        assert!((disabled.nodes[0].volume - full.nodes[0].volume).abs() < 1e-9);
        assert!(
            (disabled.depressions.depressions[0].major_axis
                - full.depressions.depressions[0].major_axis)
                .abs()
                < 1e-9
        );
        assert!(
            (disabled.depressions.depressions[0].perimeter
                - full.depressions.depressions[0].perimeter)
                .abs()
                < 1e-9
        );

        // Verify `dem_is_refined: true` produces identical root outlet & volume on a refined DEM.
        let mut dem = Array2D::from_vec(5, 5, vec![12.0; 25]).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, 5.0, 0.0, -1.0];
        dem.set(2, 0, 10.0);
        for r in 1..4 {
            for c in 1..4 {
                dem.set(c, r, 4.0);
            }
        }
        dem.set(2, 2, 1.0);
        let sinks = crate::extract_sinks_in_memory(&dem, 1, Topology::D8).unwrap();
        let r_unrefined = delineate_depressions_merge_tree_with_dem_in_memory(
            &sinks.sink,
            &sinks.dem_refined,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D8,
                dem_is_refined: false,
                ..Default::default()
            },
        )
        .unwrap();
        let r_refined = delineate_depressions_merge_tree_with_dem_in_memory(
            &sinks.sink,
            &sinks.dem_refined,
            &MergeTreeOptions {
                min_size: 1,
                min_depth: 1.0,
                topology: Topology::D8,
                dem_is_refined: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r_refined.nodes.len(), r_unrefined.nodes.len());
        assert!((r_refined.nodes[0].saddle_elev - r_unrefined.nodes[0].saddle_elev).abs() < 1e-12);
        assert!((r_refined.nodes[0].volume - r_unrefined.nodes[0].volume).abs() < 1e-12);
    }
}
