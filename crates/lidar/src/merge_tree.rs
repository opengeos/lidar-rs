//! Exact bottom-up merge-tree (contour-tree) nested depression hierarchy.
//!
//! Replaces discrete vertical level-set slicing with an exact `O(M log M)`
//! bottom-up Union-Find sweep over sink cells sorted once in ascending
//! elevation order. Every saddle / pour point is recovered to machine precision
//! independent of any slicing interval, sub-threshold micro-pits are absorbed
//! via persistence pruning (`min_size` and `min_depth`), and exact continuous
//! stage-storage-area rating curves are emitted at every topological critical
//! elevation (floors, internal saddles, and spill rims).

use std::cmp::Reverse;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, Topology};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::graph::RatingCurvePoint;
use crate::region::{RegionBasicStats, affine_basis, region_props_for_single_label};
use crate::slicing::{InMemoryDepressions, NestedDepression};

/// Options controlling [`delineate_depressions_merge_tree_in_memory`].
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
    pub saddle_elev: f64,
    /// Row index of the exact saddle / pour point cell in the full raster.
    pub pour_point_row: usize,
    /// Column index of the exact saddle / pour point cell in the full raster.
    pub pour_point_col: usize,
    /// Map X coordinate of the pour point cell center.
    pub pour_point_x: f64,
    /// Map Y coordinate of the pour point cell center.
    pub pour_point_y: f64,
    /// Total pixel count enclosed by this depression (including descendants).
    pub count: usize,
    /// Total surface area in map units squared (`count * cell_area`).
    pub area: f64,
    /// Exact storage volume below `saddle_elev` in cubic map units.
    pub volume: f64,
    /// Maximum vertical depth (`saddle_elev - min_elev`).
    pub max_depth: f64,
    /// Mean depth below `saddle_elev` (`volume / area`).
    pub avg_depth: f64,
    /// `true` if this node is a leaf depression (`children_ids.is_empty()`).
    pub is_leaf: bool,
}

/// Complete output of [`delineate_depressions_merge_tree_in_memory`].
#[derive(Debug, Clone)]
pub struct MergeTreeResult {
    /// Exact merge-tree nodes ordered by 1-based `id` (`nodes[i].id == (i + 1) as u32`).
    pub nodes: Vec<MergeTreeNode>,
    /// Standard [`InMemoryDepressions`] representation directly compatible with
    /// [`crate::fsm`], [`crate::graph`], and [`crate::inundation`].
    pub depressions: InMemoryDepressions,
    /// Exact continuous stage-storage-area rating curves evaluated at every
    /// depression's floor, internal descendant saddles, and spill rim.
    pub rating_curves: Vec<RatingCurvePoint>,
}

#[derive(Clone, Copy)]
struct SinkCell {
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
    volume: f64,
    max_depth: f64,
    avg_depth: f64,
    is_leaf: bool,
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

fn extended_morphometrics(
    area: f64,
    perimeter: f64,
    avg_depth: f64,
    max_depth: f64,
    elongatedness: f64,
    area_bbox_ratio: f64,
) -> (f64, f64, f64, f64, bool) {
    let pi = std::f64::consts::PI;
    let circularity = if perimeter > 0.0 && area > 0.0 {
        (4.0 * pi * area / (perimeter * perimeter)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eq_radius = if area > 0.0 { (area / pi).sqrt() } else { 0.0 };
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

struct RegionMergeOutput {
    region_idx: usize,
    bbox: (usize, usize, usize, usize),
    /// Per-cell local node ID (1-based) over `bbox`, 0 outside.
    local_dep_id: Vec<u32>,
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
    bbox: (usize, usize, usize, usize),
    mut cells: Vec<SinkCell>,
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
            bbox,
            local_dep_id: vec![0u32; bw * bh],
            nodes: Vec::new(),
            depressions: Vec::new(),
            rating_curves: Vec::new(),
        };
    }

    // Sort all sink cells once in ascending (z, row, col) order: O(M log M).
    cells.sort_unstable_by(|a, b| a.z.total_cmp(&b.z).then(a.r.cmp(&b.r)).then(a.c.cmp(&b.c)));

    let reg_min_elev = cells[0].z;
    let reg_max_elev = cells[m - 1].z;
    if (reg_max_elev - reg_min_elev) < options.min_depth {
        return RegionMergeOutput {
            region_idx,
            bbox,
            local_dep_id: vec![0u32; bw * bh],
            nodes: Vec::new(),
            depressions: Vec::new(),
            rating_curves: Vec::new(),
        };
    }

    let offsets: &[(isize, isize)] = match options.topology {
        Topology::D4 => &D4_OFFSETS,
        Topology::D8 => &D8_OFFSETS,
    };

    let mut cell_to_sorted = vec![usize::MAX; bw * bh];
    let mut uf_parent: Vec<usize> = (0..m).collect();
    let mut uf_count = vec![0usize; m];
    let mut uf_min_elev = vec![f64::INFINITY; m];
    let mut uf_min_idx = vec![usize::MAX; m];
    let mut uf_sum_z = vec![0.0f64; m];
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
                              uf_sum_z: &[f64],
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
        let volume =
            ((cnt as f64) * (z_saddle - reg_min_elev) - uf_sum_z[root]).max(0.0) * cell_area;
        let avg_depth = if area > 0.0 { volume / area } else { 0.0 };

        let nidx = if let Some(existing) = uf_active_node[root].take() {
            let n = &mut local_nodes[existing];
            n.min_elev = min_z;
            n.saddle_elev = z_saddle;
            n.pour_point_row = pp_r;
            n.pour_point_col = pp_c;
            n.count = cnt;
            n.area = area;
            n.volume = volume;
            n.max_depth = max_depth;
            n.avg_depth = avg_depth;
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
                volume,
                max_depth,
                avg_depth,
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

    // Bottom-up sweep.
    for i in 0..m {
        let cell = cells[i];
        let lr = cell.r - r0;
        let lc = cell.c - c0;
        cell_to_sorted[lr * bw + lc] = i;

        uf_count[i] = 1;
        uf_min_elev[i] = cell.z;
        uf_min_idx[i] = i;
        uf_sum_z[i] = cell.z - reg_min_elev;
        list_head[i] = i;
        list_tail[i] = i;

        let mut nbr_roots = [0usize; 8];
        let mut nbr_len = 0usize;
        for &(dr, dc) in offsets {
            let nr = lr as isize + dr;
            let nc = lc as isize + dc;
            if nr < 0 || nc < 0 || nr >= bh as isize || nc >= bw as isize {
                continue;
            }
            let n_sorted = cell_to_sorted[nr as usize * bw + nc as usize];
            if n_sorted == usize::MAX {
                continue;
            }
            let root = uf_find(&mut uf_parent, n_sorted);
            if !nbr_roots[..nbr_len].contains(&root) {
                nbr_roots[nbr_len] = root;
                nbr_len += 1;
            }
        }

        if nbr_len == 0 {
            continue;
        }

        let mut sig_roots = [0usize; 8];
        let mut sig_len = 0usize;
        let mut noise_roots = [0usize; 8];
        let mut noise_len = 0usize;
        for &r in &nbr_roots[..nbr_len] {
            let is_sig = uf_active_node[r].is_some()
                || (uf_count[r] >= min_size_eff && (cell.z - uf_min_elev[r]) >= options.min_depth);
            if is_sig {
                sig_roots[sig_len] = r;
                sig_len += 1;
            } else {
                noise_roots[noise_len] = r;
                noise_len += 1;
            }
        }

        if sig_len >= 2 {
            // Exact saddle! Sort meeting significant branches deterministically.
            sig_roots[..sig_len].sort_unstable_by(|&a, &b| {
                uf_min_elev[a]
                    .total_cmp(&uf_min_elev[b])
                    .then(uf_min_idx[a].cmp(&uf_min_idx[b]))
            });

            let mut child_nidxs = Vec::with_capacity(sig_len);
            for &sr in &sig_roots[..sig_len] {
                let cn = finalize_component(
                    sr,
                    cell.z,
                    cell.r,
                    cell.c,
                    &uf_count,
                    &uf_min_elev,
                    &uf_sum_z,
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
            let mut total_cnt = uf_count[merged] + 1;
            let mut total_sum_z = uf_sum_z[merged] + (cell.z - reg_min_elev);
            let mut min_z = uf_min_elev[merged];
            let mut min_idx = uf_min_idx[merged];

            // Attach other significant roots to `merged`.
            for &sr in &sig_roots[1..sig_len] {
                uf_parent[sr] = merged;
                total_cnt += uf_count[sr];
                total_sum_z += uf_sum_z[sr];
                if uf_min_elev[sr] < min_z || (uf_min_elev[sr] == min_z && uf_min_idx[sr] < min_idx)
                {
                    min_z = uf_min_elev[sr];
                    min_idx = uf_min_idx[sr];
                }
            }
            // Absorb any sub-threshold neighbor components into the newly formed parent.
            for &nr in &noise_roots[..noise_len] {
                uf_parent[nr] = merged;
                total_cnt += uf_count[nr];
                total_sum_z += uf_sum_z[nr];
                if uf_min_elev[nr] < min_z || (uf_min_elev[nr] == min_z && uf_min_idx[nr] < min_idx)
                {
                    min_z = uf_min_elev[nr];
                    min_idx = uf_min_idx[nr];
                }
                splice_list(merged, nr, &mut list_head, &mut list_tail, &mut next_pix);
            }
            // Attach saddle cell `i` to `merged` (it belongs to the new parent node).
            uf_parent[i] = merged;
            splice_list(merged, i, &mut list_head, &mut list_tail, &mut next_pix);

            uf_count[merged] = total_cnt;
            uf_sum_z[merged] = total_sum_z;
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
                saddle_elev: cell.z,
                pour_point_row: cell.r,
                pour_point_col: cell.c,
                count: total_cnt,
                area: total_cnt as f64 * cell_area,
                volume: 0.0,
                max_depth: 0.0,
                avg_depth: 0.0,
                is_leaf: false,
            });
            uf_active_node[merged] = Some(parent_nidx);
        } else {
            // No hierarchy split: absorb sub-threshold components and cell `i` into `winner`.
            let winner = if sig_len == 1 {
                sig_roots[0]
            } else {
                *nbr_roots[..nbr_len]
                    .iter()
                    .min_by(|&&a, &&b| {
                        uf_min_elev[a]
                            .total_cmp(&uf_min_elev[b])
                            .then(Reverse(uf_count[a]).cmp(&Reverse(uf_count[b])))
                            .then(uf_min_idx[a].cmp(&uf_min_idx[b]))
                    })
                    .unwrap()
            };

            let mut total_cnt = uf_count[winner] + 1;
            let mut total_sum_z = uf_sum_z[winner] + (cell.z - reg_min_elev);
            let mut min_z = uf_min_elev[winner];
            let mut min_idx = uf_min_idx[winner];

            for &nr in &nbr_roots[..nbr_len] {
                if nr == winner {
                    continue;
                }
                uf_parent[nr] = winner;
                total_cnt += uf_count[nr];
                total_sum_z += uf_sum_z[nr];
                if uf_min_elev[nr] < min_z || (uf_min_elev[nr] == min_z && uf_min_idx[nr] < min_idx)
                {
                    min_z = uf_min_elev[nr];
                    min_idx = uf_min_idx[nr];
                }
                splice_list(winner, nr, &mut list_head, &mut list_tail, &mut next_pix);
            }
            uf_parent[i] = winner;
            splice_list(winner, i, &mut list_head, &mut list_tail, &mut next_pix);

            uf_count[winner] = total_cnt;
            uf_sum_z[winner] = total_sum_z;
            uf_min_elev[winner] = min_z;
            uf_min_idx[winner] = min_idx;
        }
    }

    // Finalize the single region root component at `reg_max_elev`.
    let root_uf = uf_find(&mut uf_parent, m - 1);
    let top_cell = cells[m - 1];
    let root_nidx = finalize_component(
        root_uf,
        top_cell.z,
        top_cell.r,
        top_cell.c,
        &uf_count,
        &uf_min_elev,
        &uf_sum_z,
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

    // Build local_dep_id grid and per-node owned pixel index lists (in sorted cell order).
    let mut local_dep_id = vec![0u32; bw * bh];
    let mut node_owned_cells: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for (i, cell) in cells.iter().enumerate() {
        let old_n = owned_node[i];
        if old_n != usize::MAX {
            let nid = old_to_new_id[old_n];
            local_dep_id[(cell.r - r0) * bw + (cell.c - c0)] = nid;
            node_owned_cells[(nid - 1) as usize].push(i);
        }
    }

    // Build pre-order MergeTreeNode vector.
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
            volume: ln.volume,
            max_depth: ln.max_depth,
            avg_depth: ln.avg_depth,
            is_leaf: ln.is_leaf,
        });
    }

    // Compute full NestedDepression morphometrics and exact continuous rating curves
    // for every node in this region.
    let mut depressions = Vec::with_capacity(num_nodes);
    let mut rating_curves = Vec::new();
    let mut subtree_mask_labels = vec![0u32; bw * bh];
    let mut stamp = 0u32;
    let mut subtree_elevs: Vec<f64> = Vec::new();
    let mut crit_stages: Vec<f64> = Vec::new();

    for node in &nodes {
        stamp += 1;
        // Collect all descendant node IDs (including self).
        let mut desc_stack = vec![node.id];
        let mut min_lr = usize::MAX;
        let mut min_lc = usize::MAX;
        let mut max_lr = 0usize;
        let mut max_lc = 0usize;
        let mut first_pixel = (usize::MAX, usize::MAX);
        subtree_elevs.clear();
        crit_stages.clear();
        crit_stages.push(node.min_elev);
        crit_stages.push(node.saddle_elev);

        while let Some(cur_id) = desc_stack.pop() {
            let cur_node = &nodes[(cur_id - 1) as usize];
            if cur_id != node.id {
                crit_stages.push(cur_node.min_elev.clamp(node.min_elev, node.saddle_elev));
                crit_stages.push(cur_node.saddle_elev.clamp(node.min_elev, node.saddle_elev));
            }
            for &ci in &node_owned_cells[(cur_id - 1) as usize] {
                let c = cells[ci];
                let lr = c.r - r0;
                let lc = c.c - c0;
                subtree_mask_labels[lr * bw + lc] = stamp;
                if lr < min_lr {
                    min_lr = lr;
                }
                if lc < min_lc {
                    min_lc = lc;
                }
                if lr + 1 > max_lr {
                    max_lr = lr + 1;
                }
                if lc + 1 > max_lc {
                    max_lc = lc + 1;
                }
                if (lr, lc) < first_pixel {
                    first_pixel = (lr, lc);
                }
                subtree_elevs.push(c.z);
            }
            for &cid in &cur_node.children_ids {
                desc_stack.push(cid);
            }
        }

        // Compute affine-aware perimeter & ellipse moments using the same estimator as slicing.rs.
        let basic = RegionBasicStats {
            label: stamp,
            count: node.count,
            bbox: (min_lr, min_lc, max_lr, max_lc),
            first_pixel,
            area: node.area,
            volume: node.volume,
            mean_depth: node.avg_depth,
            max_depth: node.max_depth,
            min_elev: node.min_elev,
            max_elev: node.saddle_elev,
        };
        let props = region_props_for_single_label(&subtree_mask_labels, &basic, bw, gt);
        let (circularity, depth_diam_ratio, hypsometric_integral, rim_slope_deg, artifact_flag) =
            extended_morphometrics(
                node.area,
                props.perimeter,
                node.avg_depth,
                node.max_depth,
                props.elongatedness,
                props.area_bbox_ratio,
            );
        depressions.push(NestedDepression {
            id: node.id,
            level: node.level,
            count: node.count,
            area: node.area,
            volume: node.volume,
            avg_depth: node.avg_depth,
            max_depth: node.max_depth,
            min_elev: node.min_elev,
            max_elev: node.saddle_elev,
            children_ids: node.children_ids.clone(),
            region_id,
            perimeter: props.perimeter,
            major_axis: props.major_axis,
            minor_axis: props.minor_axis,
            elongatedness: props.elongatedness,
            eccentricity: props.eccentricity,
            orientation: props.orientation,
            area_bbox_ratio: props.area_bbox_ratio,
            parent_id: node.parent_id,
            circularity,
            depth_diam_ratio,
            hypsometric_integral,
            rim_slope_deg,
            artifact_flag,
        });

        // Build exact continuous breakpoint rating curve for this node.
        subtree_elevs.sort_unstable_by(f64::total_cmp);
        let z_min = node.min_elev;
        let mut prefix_sum = Vec::with_capacity(subtree_elevs.len() + 1);
        prefix_sum.push(0.0f64);
        let mut acc = 0.0f64;
        for &z in &subtree_elevs {
            acc += z - z_min;
            prefix_sum.push(acc);
        }

        // Include exact critical points (floor, every descendant floor & saddle, and own saddle).
        // Also include intermediate unique elevations when compact (<= 64) or quantile samples
        // between critical stages so continuous stage-storage curves resolve curvature smoothly.
        let mut stages = crit_stages.clone();
        if subtree_elevs.len() <= 64 {
            stages.extend_from_slice(&subtree_elevs);
        } else {
            let n_quant = 16usize;
            for q in 1..n_quant {
                let idx = (q * (subtree_elevs.len() - 1)) / n_quant;
                stages.push(subtree_elevs[idx]);
            }
        }
        stages.sort_unstable_by(f64::total_cmp);
        stages.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
        // Guarantee exact endpoints are preserved without rounding drift.
        if let Some(first) = stages.first_mut() {
            *first = node.min_elev;
        }
        if let Some(last) = stages.last_mut() {
            *last = node.saddle_elev;
        }

        for stage in stages {
            let k = subtree_elevs.partition_point(|&z| z <= stage);
            let area = k as f64 * cell_area;
            let depth = (stage - z_min).max(0.0);
            let vol = ((k as f64) * depth - prefix_sum[k]).max(0.0) * cell_area;
            rating_curves.push(RatingCurvePoint {
                dep_id: node.id,
                level: node.level,
                region_id,
                stage,
                depth,
                area,
                volume: vol,
            });
        }
    }

    RegionMergeOutput {
        region_idx,
        bbox,
        local_dep_id,
        nodes,
        depressions,
        rating_curves,
    }
}

/// Delineates the exact nested depression hierarchy using a bottom-up `O(M log M)`
/// Union-Find merge tree over sink cells sorted once in ascending elevation order.
///
/// Unlike top-down discrete level-set slicing, every saddle elevation and pour-point
/// coordinate `(pour_point_row, pour_point_col)` is recovered to machine precision
/// independent of any slicing interval. The returned [`MergeTreeResult::depressions`]
/// is directly compatible with [`crate::fsm::simulate_fill_spill_merge_in_memory`],
/// [`crate::graph::build_connectivity_graph`], and [`crate::inundation`].
pub fn delineate_depressions_merge_tree_in_memory(
    sink: &Array2D<f64>,
    options: &MergeTreeOptions,
) -> Result<MergeTreeResult> {
    if !options.min_depth.is_finite() || options.min_depth < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "min_depth must be finite and >= 0, got {}",
            options.min_depth
        )));
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

    let sdata = sink.data();
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
        }
    }

    type RegionBbox = (usize, usize, usize, usize);
    type RegionWorkItem = (usize, RegionBbox, Vec<SinkCell>);

    let work: Vec<RegionWorkItem> = bboxes
        .into_iter()
        .zip(region_cells)
        .enumerate()
        .map(|(idx, (bbox, cells))| (idx, bbox, cells))
        .collect();

    #[cfg(feature = "parallel")]
    let mut region_results: Vec<RegionMergeOutput> = work
        .into_par_iter()
        .map(|(idx, bbox, cells)| {
            build_region_merge_tree(idx, (idx + 1) as u32, bbox, cells, &gt, cell_area, options)
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let mut region_results: Vec<RegionMergeOutput> = work
        .into_iter()
        .map(|(idx, bbox, cells)| {
            build_region_merge_tree(idx, (idx + 1) as u32, bbox, cells, &gt, cell_area, options)
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
        let (r0, c0, r1, c1) = res.bbox;
        let bw = c1 - c0;

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

        for r in r0..r1 {
            let dst_off = r * w;
            let src_off = (r - r0) * bw;
            for c in c0..c1 {
                let local_id = res.local_dep_id[src_off + (c - c0)];
                if local_id > 0 {
                    let global_id = local_id + uid_offset;
                    let lvl = res.nodes[(local_id - 1) as usize].level;
                    obj_vec[dst_off + c] = global_id as i32;
                    level_vec[dst_off + c] = lvl as i32;
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_saddle_recovered_below_coarse_interval() {
        // Two 5x5 bowls connected by a single exact saddle cell at (3, 6) with z = 10.37.
        let (w, h) = (13, 7);
        let mut data = vec![0.0f64; w * h];
        // Outer compound rim: rows 1..6, cols 1..12 at z = 12.0.
        for r in 1..6 {
            for c in 1..12 {
                data[r * w + c] = 12.0;
            }
        }
        // Left bowl: rows 2..5, cols 2..5 (9 cells, z from 8.0 up to 9.5).
        for r in 2..5 {
            for c in 2..5 {
                let d = ((r as i32 - 3).abs() + (c as i32 - 3).abs()) as f64;
                data[r * w + c] = 8.0 + d * 0.5;
            }
        }
        // Right bowl: rows 2..5, cols 8..11 (9 cells, z from 8.5 up to 10.0).
        for r in 2..5 {
            for c in 8..11 {
                let d = ((r as i32 - 3).abs() + (c as i32 - 9).abs()) as f64;
                data[r * w + c] = 8.5 + d * 0.5;
            }
        }
        // Connecting channel at row 3: cols 5 and 7 at 10.20, exact saddle at col 6 = 10.37.
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
            assert!(
                (child.saddle_elev - 10.37).abs() < 1e-12,
                "expected exact saddle 10.37, got {}",
                child.saddle_elev
            );
            assert_eq!((child.pour_point_row, child.pour_point_col), (3, 6));
            assert!((child.pour_point_x - (100.0 + 6.5 * 2.0)).abs() < 1e-12);
            assert!((child.pour_point_y - (200.0 - 3.5 * 2.0)).abs() < 1e-12);
        }

        // Verify rating curve for root contains exact breakpoint at saddle 10.37.
        let root_curve: Vec<&RatingCurvePoint> = res
            .rating_curves
            .iter()
            .filter(|p| p.dep_id == root.id)
            .collect();
        assert!(
            root_curve.iter().any(|p| (p.stage - 10.37).abs() < 1e-12),
            "root rating curve must include exact saddle stage 10.37"
        );
        let last = root_curve.last().unwrap();
        assert!((last.volume - root.volume).abs() < 1e-10);
        assert!((last.area - root.area).abs() < 1e-10);
    }
}
