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
//! - **Per-node morphometrics and exact rating curves:** `O(sum_v |subtree(v)|)` time
//!   and `O(M)` peak memory per region (`O(M * depth)` in the worst case for deep nested
//!   chains), without allocating bounding-box-sized dense grids for sparse regions.

use std::cmp::Reverse;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, Topology};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::graph::RatingCurvePoint;
use crate::inundation::validate_grid_alignment;
use crate::region::{affine_basis, extended_morphometrics, region_props_from_sparse_cells};
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
}

impl Default for MergeTreeOptions {
    fn default() -> Self {
        Self {
            min_size: 1000,
            min_depth: 0.3,
            topology: Topology::D8,
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
        //    pre-existing strictly-lower UF roots (`< batch_start`).
        pre_root_to_slot.clear();
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

                // Deterministic pour point: earliest plateau cell (in row-major order)
                // that directly touches at least one significant branch, else `batch_cells[0]`.
                let mut pp_cell = cells[batch_cells[0]];
                'find_pp: for &ci in &batch_cells {
                    let c = cells[ci];
                    for &(dr, dc) in offsets {
                        if let Some(n_sorted) = cell_index.get(c.r as isize + dr, c.c as isize + dc)
                            && n_sorted < batch_start
                        {
                            let pr = uf_find(&mut uf_parent, n_sorted);
                            if sig_roots.contains(&pr) {
                                pp_cell = c;
                                break 'find_pp;
                            }
                        }
                    }
                }

                let mut child_nidxs = Vec::with_capacity(sig_roots.len());
                for &sr in &sig_roots {
                    let cn = finalize_component(
                        sr,
                        z_batch,
                        pp_cell.r,
                        pp_cell.c,
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
                    pour_point_row: pp_cell.r,
                    pour_point_col: pp_cell.c,
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
    let (root_z, root_r, root_c) = if let Some(out) = root_outlet {
        (out.z.max(top_cell.z), out.r, out.c)
    } else {
        (top_cell.z, top_cell.r, top_cell.c)
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

    // Compute exact component-local storage volumes, exact continuous rating curves (retaining
    // every distinct elevation breakpoint), and sparse-memory NestedDepression morphometrics.
    let mut depressions = Vec::with_capacity(num_nodes);
    let mut rating_curves = Vec::new();
    let mut subtree_coords: Vec<(usize, usize)> = Vec::new();
    let mut subtree_elevs: Vec<f64> = Vec::new();
    let mut stages: Vec<f64> = Vec::new();

    for node_idx in 0..num_nodes {
        let node_id = nodes[node_idx].id;
        let node_min = nodes[node_idx].min_elev;
        let node_saddle = nodes[node_idx].saddle_elev;
        let node_level = nodes[node_idx].level;
        let node_area = nodes[node_idx].area;
        let node_max_depth = nodes[node_idx].max_depth;

        subtree_coords.clear();
        subtree_elevs.clear();
        stages.clear();
        stages.push(node_min);
        stages.push(node_saddle);

        let mut desc_stack = vec![node_id];
        while let Some(cur_id) = desc_stack.pop() {
            let cur = &nodes[(cur_id - 1) as usize];
            if cur_id != node_id {
                stages.push(cur.min_elev.clamp(node_min, node_saddle));
                stages.push(cur.saddle_elev.clamp(node_min, node_saddle));
            }
            for &ci in &node_owned_cells[(cur_id - 1) as usize] {
                let c = cells[ci];
                subtree_coords.push((c.r, c.c));
                subtree_elevs.push(c.z);
            }
            for &cid in &cur.children_ids {
                desc_stack.push(cid);
            }
        }

        // Exact sorted elevations and prefix sums relative to this component's OWN floor `node_min`.
        subtree_elevs.sort_unstable_by(f64::total_cmp);
        let mut prefix_sum = Vec::with_capacity(subtree_elevs.len() + 1);
        prefix_sum.push(0.0f64);
        let mut acc = 0.0f64;
        for &z in &subtree_elevs {
            acc += z - node_min;
            prefix_sum.push(acc);
        }

        // Exact storage volume below `node_saddle` using this node's own local floor reference.
        let k_end = subtree_elevs.partition_point(|&z| z <= node_saddle);
        let node_volume = ((k_end as f64) * (node_saddle - node_min).max(0.0) - prefix_sum[k_end])
            .max(0.0)
            * cell_area;
        let node_avg_depth = if node_area > 0.0 {
            node_volume / node_area
        } else {
            0.0
        };
        nodes[node_idx].volume = node_volume;
        nodes[node_idx].avg_depth = node_avg_depth;

        // Retain EVERY distinct cell elevation breakpoint (no downsampling) and deduplicate
        // only exact equal stages so closely spaced representable saddles are never merged.
        stages.extend_from_slice(&subtree_elevs);
        stages.sort_unstable_by(f64::total_cmp);
        stages.dedup_by(|a, b| a.total_cmp(b).is_eq());

        for &stage in &stages {
            let k = subtree_elevs.partition_point(|&z| z <= stage);
            let area = k as f64 * cell_area;
            let depth = (stage - node_min).max(0.0);
            let vol = ((k as f64) * depth - prefix_sum[k]).max(0.0) * cell_area;
            rating_curves.push(RatingCurvePoint {
                dep_id: node_id,
                level: node_level,
                region_id,
                stage,
                depth,
                area,
                volume: vol,
            });
        }

        let props = region_props_from_sparse_cells(
            node_id,
            &subtree_coords,
            gt,
            node_area,
            node_volume,
            node_avg_depth,
            node_max_depth,
            node_min,
            node_saddle,
        );
        let (circularity, depth_diam_ratio, hypsometric_integral, rim_slope_deg, artifact_flag) =
            extended_morphometrics(
                node_area,
                props.perimeter,
                node_avg_depth,
                node_max_depth,
                props.elongatedness,
                props.area_bbox_ratio,
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
            perimeter: props.perimeter,
            major_axis: props.major_axis,
            minor_axis: props.minor_axis,
            elongatedness: props.elongatedness,
            eccentricity: props.eccentricity,
            orientation: props.orientation,
            area_bbox_ratio: props.area_bbox_ratio,
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
    let (_, _, cell_area) = affine_basis(&gt);
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
    let mut region_outlets: Vec<Option<RootOutlet>> = vec![None; k];

    let sdata = sink.data();
    let offsets: &[(isize, isize)] = match options.topology {
        Topology::D4 => &D4_OFFSETS,
        Topology::D8 => &D8_OFFSETS,
    };
    let dem_info = source_dem.map(|d| (d.data(), d.no_data()));

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
            region_cells[idx].push(SinkCell {
                z: sdata[off + c],
                r,
                c,
            });

            // When the source DEM is provided, identify the exact lowest exterior rim/outlet
            // neighbor bordering this sink region so root spill elevation and storage volume
            // reflect the true topographic pour point rather than the highest interior cell.
            if let Some((ddata, d_nd)) = dem_info {
                for &(dr, dc) in offsets {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
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
                    if !dz.is_finite() || dz.rd_eq(d_nd) {
                        continue;
                    }
                    let replace = match region_outlets[idx] {
                        None => true,
                        Some(cur) => dz < cur.z || (dz == cur.z && (nru, ncu) < (cur.r, cur.c)),
                    };
                    if replace {
                        region_outlets[idx] = Some(RootOutlet {
                            z: dz,
                            r: nru,
                            c: ncu,
                        });
                    }
                }
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
        }
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
}
