//! True out-of-core tiled regional depression delineation with windowed raster I/O,
//! overlapping halos, optional per-tile breaching, and seamless cross-tile spatial stitching.
//!
//! Splits large DEM rasters into overlapping square tiles, reads each tile's padded window
//! on demand via `terrio::open_raster_lazy` + `read_window` (keeping peak DEM memory
//! `O(tile_size^2)` instead of loading the entire raster into RAM; falling back cleanly to
//! eager `load_raster` only for non-lazy formats such as legacy ASCII grids), delineates
//! nested depressions independently per tile (in parallel when the `parallel` feature is
//! enabled), deduplicates features across tile seams via Union-Find clustering, and
//! optionally stitches full-DEM `depression_id.tif`, `depression_level.tif`, `depth.tif`,
//! and `depressions.gpkg` spatial outputs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, Topology, dataset_to_dynamic, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::conditioning::{BreachOptions, breach_depressions};
use crate::error::{Error, Result};
use crate::filling::extract_sinks_in_memory;
use crate::filtering::median_filter;
use crate::slicing::{
    NestedDepression, delineate_depressions_in_memory, write_depressions_csv,
    write_depressions_vector_with_enclosing,
};

/// Options controlling [`delineate_depressions_tiled`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TiledDelineationOptions {
    /// Core (non-overlapping) tile width and height in pixels (default `2048`).
    pub tile_size: usize,
    /// Overlap halo added around each side of a tile in pixels (default `128`).
    /// Note: seam deduplication memory and merge time scale with the number of sink pixels
    /// falling inside tile overlap bands, so keep `overlap` proportional to expected basin width.
    pub overlap: usize,
    /// Optional median filter kernel size (`<= 1` skips smoothing; default `3`).
    pub kernel_size: usize,
    /// Minimum depression area in pixels (default `1000`).
    pub min_size: usize,
    /// Minimum depression depth in vertical elevation units (default `0.3`).
    pub min_depth: f64,
    /// Vertical level-set slicing interval (default `0.3`).
    pub interval: f64,
    /// Connectivity topology for depression filling (`Topology::D8` by default).
    pub topology: Topology,
    /// Load source DEM as `f32` when falling back to eager loading on non-lazy formats
    /// (default `true`).
    pub use_f32: bool,
}

impl Default for TiledDelineationOptions {
    fn default() -> Self {
        Self {
            tile_size: 2048,
            overlap: 128,
            kernel_size: 3,
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            topology: Topology::D8,
            use_f32: true,
        }
    }
}

/// Spatial output and per-tile conditioning options for
/// [`delineate_depressions_tiled_with_outputs`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TiledOutputOptions {
    /// Stitch and write full-DEM `depression_id.tif`, `depression_level.tif`, and `depth.tif`
    /// into `out_dir` (default `false`).
    pub write_rasters: bool,
    /// Stitch and write full-DEM `depressions.gpkg` (including `"depressions"` and
    /// `"depressions_full"` layers) into `out_dir` (default `false`).
    pub write_gpkg: bool,
    /// Optional per-tile least-cost depression breaching applied before sink extraction
    /// (default `None`).
    pub breach: Option<BreachOptions>,
    /// Apply DEFLATE compression to stitched GeoTIFF outputs (default `true`).
    pub compress: bool,
}

impl Default for TiledOutputOptions {
    fn default() -> Self {
        Self {
            write_rasters: false,
            write_gpkg: false,
            breach: None,
            compress: true,
        }
    }
}

/// Summary and merged outputs produced by [`delineate_depressions_tiled`] and
/// [`delineate_depressions_tiled_with_outputs`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TiledDelineationResult {
    /// Total number of tiles processed.
    pub num_tiles: usize,
    /// Maximum hierarchy level observed across all tiles.
    pub max_level: u32,
    /// Total deduplicated nested depressions across the full DEM.
    pub depressions: Vec<NestedDepression>,
    /// Path to merged `depressions_info.csv`.
    pub csv_path: PathBuf,
    /// Optional path to stitched `depression_id.tif` when `write_rasters = true`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depression_id_tif: Option<PathBuf>,
    /// Optional path to stitched `depression_level.tif` when `write_rasters = true`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depression_level_tif: Option<PathBuf>,
    /// Optional path to stitched `depth.tif` when `write_rasters = true`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_tif: Option<PathBuf>,
    /// Optional path to stitched `depressions.gpkg` when `write_gpkg = true`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gpkg_path: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct TileWindow {
    core_r0: usize,
    core_c0: usize,
    core_r1: usize,
    core_c1: usize,
    pad_r0: usize,
    pad_c0: usize,
    pad_r1: usize,
    pad_c1: usize,
}

#[derive(Clone)]
struct RasterMeta {
    width: usize,
    height: usize,
    geotransform: [f64; 6],
    projection: String,
}

#[derive(Clone, Copy)]
struct StitchedPixel {
    gr: usize,
    gc: usize,
    local_did: u32,
    level: u32,
    depth: f64,
    in_core: bool,
}

fn shifted_geotransform(gt: &[f64; 6], col_off: usize, row_off: usize) -> [f64; 6] {
    let c0f = col_off as f64;
    let r0f = row_off as f64;
    [
        gt[0] + c0f * gt[1] + r0f * gt[2],
        gt[1],
        gt[2],
        gt[3] + c0f * gt[4] + r0f * gt[5],
        gt[4],
        gt[5],
    ]
}

/// Extracts a tile sub-grid as `Array2D<f64>` from either an `f32` or `f64` in-memory raster,
/// preserving the source NoData sentinel so valid elevations are never misclassified as NoData.
fn extract_tile_f64<T: DemNum>(src: &Array2D<T>, win: TileWindow) -> Result<Array2D<f64>> {
    let w = src.width();
    let tw = win.pad_c1 - win.pad_c0;
    let th = win.pad_r1 - win.pad_r0;
    let nd = src.no_data();
    let nd_f64 = nd.to_f64();
    let out_nd = if nd_f64.is_finite() { nd_f64 } else { f64::NAN };
    let sdata = src.data();
    let mut data = Vec::with_capacity(tw * th);
    for r in win.pad_r0..win.pad_r1 {
        let off = r * w;
        for c in win.pad_c0..win.pad_c1 {
            let v = sdata[off + c];
            if v.rd_eq(nd) {
                data.push(out_nd);
            } else {
                let f = v.to_f64();
                data.push(if f.is_finite() { f } else { out_nd });
            }
        }
    }
    let mut tile = Array2D::from_vec(tw, th, data)?;
    tile.set_no_data(out_nd);
    tile.geotransform = shifted_geotransform(&src.geotransform, win.pad_c0, win.pad_r0);
    tile.projection = src.projection.clone();
    Ok(tile)
}

/// Reads a single padded tile window directly from disk via `terrio::open_raster_lazy`
/// without loading the full raster into memory.
fn read_tile_window_lazy(
    path_str: &str,
    meta: &RasterMeta,
    win: TileWindow,
) -> Result<Array2D<f64>> {
    let mut lazy = terrio::open_raster_lazy(path_str)?;
    let ds = lazy.read_window(
        terrio::Window {
            x: win.pad_c0,
            y: win.pad_r0,
            width: win.pad_c1 - win.pad_c0,
            height: win.pad_r1 - win.pad_r0,
        },
        &terrio::WindowOptions {
            bands: vec![1],
            overview: None,
        },
    )?;
    let mut tile = dataset_to_dynamic(ds, path_str)?.to_f64();
    tile.geotransform = shifted_geotransform(&meta.geotransform, win.pad_c0, win.pad_r0);
    if tile.projection.is_empty() && !meta.projection.is_empty() {
        tile.projection = meta.projection.clone();
    }
    Ok(tile)
}

struct TileRegionOutput {
    /// Global `(row, col)` of the lowest-elevation cell in this sink region.
    min_cell: (usize, usize),
    /// Total pixel count of this sink region inside the tile (larger = less truncated).
    total_cells: usize,
    /// Whether any pixel of this tile copy lies inside the tile's non-overlapping core.
    intersects_core: bool,
    /// Whether any pixel of this tile copy touches an internal padded tile boundary
    /// (meaning the basin may be clipped by this tile's halo).
    touches_internal_pad_edge: bool,
    /// Global linear cell indices (`gr * full_w + gc`) for regions that overlap a tile seam
    /// halo band, used for exact pixel-overlap deduplication across neighboring tiles.
    seam_cells: Vec<usize>,
    /// Complete nested depression hierarchy for this sink region.
    depressions: Vec<NestedDepression>,
    /// Sparse per-pixel spatial payload populated only when `write_rasters || write_gpkg`.
    pixels: Vec<StitchedPixel>,
}

fn process_one_tile(
    tile: Array2D<f64>,
    win: TileWindow,
    full_w: usize,
    full_h: usize,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
) -> Result<Vec<TileRegionOutput>> {
    let smoothed = if options.kernel_size > 1 {
        median_filter(&tile, options.kernel_size)?
    } else {
        tile
    };
    let conditioned = if let Some(ref breach_opts) = outputs.breach {
        breach_depressions(&smoothed, breach_opts)?
    } else {
        smoothed
    };
    let sinks = extract_sinks_in_memory(&conditioned, options.min_size, options.topology)?;
    let deps = delineate_depressions_in_memory(
        &sinks.sink,
        options.min_size,
        options.min_depth,
        options.interval,
    )?;
    if deps.depressions.is_empty() {
        return Ok(Vec::new());
    }

    let need_spatial = outputs.write_rasters || outputs.write_gpkg;
    let tw = deps.depression_id.width();
    let th = deps.depression_id.height();
    let id_data = deps.depression_id.data();
    let lvl_data = deps.depression_level.data();
    let sink_data = sinks.sink.data();
    let depth_data = sinks.depth.data();
    let sink_nd = sinks.sink.no_data();
    let dep_region: HashMap<u32, u32> = deps
        .depressions
        .iter()
        .map(|d| (d.id, d.region_id))
        .collect();

    // Seam band inside this tile: any cell within `overlap` pixels of an internal core edge
    // can also be observed by a neighboring tile's halo.
    let ov = options.overlap;
    let inner_r0 = if win.core_r0 > 0 {
        win.core_r0.saturating_add(ov)
    } else {
        0
    };
    let inner_r1 = if win.core_r1 < full_h {
        win.core_r1.saturating_sub(ov)
    } else {
        full_h
    };
    let inner_c0 = if win.core_c0 > 0 {
        win.core_c0.saturating_add(ov)
    } else {
        0
    };
    let inner_c1 = if win.core_c1 < full_w {
        win.core_c1.saturating_sub(ov)
    } else {
        full_w
    };

    struct RegAcc {
        count: usize,
        min_elev: f64,
        min_cell: (usize, usize),
        intersects_core: bool,
        touches_internal_pad_edge: bool,
        seam_cells: Vec<usize>,
        pixels: Vec<StitchedPixel>,
    }
    let mut reg_acc: HashMap<u32, RegAcc> = HashMap::new();
    for r in 0..th {
        let off = r * tw;
        let gr = win.pad_r0 + r;
        let on_pad_r = (r == 0 && win.pad_r0 > 0) || (r + 1 == th && win.pad_r1 < full_h);
        for c in 0..tw {
            let did = id_data[off + c];
            if did > 0
                && let Some(&rid) = dep_region.get(&(did as u32))
            {
                let gc = win.pad_c0 + c;
                let z = sink_data[off + c];
                let e = reg_acc.entry(rid).or_insert(RegAcc {
                    count: 0,
                    min_elev: f64::INFINITY,
                    min_cell: (gr, gc),
                    intersects_core: false,
                    touches_internal_pad_edge: false,
                    seam_cells: Vec::new(),
                    pixels: Vec::new(),
                });
                e.count += 1;
                let in_core =
                    gr >= win.core_r0 && gr < win.core_r1 && gc >= win.core_c0 && gc < win.core_c1;
                if in_core {
                    e.intersects_core = true;
                }
                let on_pad_c = (c == 0 && win.pad_c0 > 0) || (c + 1 == tw && win.pad_c1 < full_w);
                if on_pad_r || on_pad_c {
                    e.touches_internal_pad_edge = true;
                }
                // Only cells in this tile's seam band can appear inside a neighboring tile's
                // padded window; interior-core cells are never shared across tiles.
                if gr < inner_r0 || gr >= inner_r1 || gc < inner_c0 || gc >= inner_c1 {
                    e.seam_cells.push(gr * full_w + gc);
                }
                if need_spatial {
                    let dval = depth_data[off + c];
                    let depth = if dval.is_finite() && dval > 0.0 {
                        dval
                    } else {
                        0.0
                    };
                    e.pixels.push(StitchedPixel {
                        gr,
                        gc,
                        local_did: did as u32,
                        level: lvl_data[off + c].max(0) as u32,
                        depth,
                        in_core,
                    });
                }
                // Row-major scan visits `(gr, gc)` in strictly ascending order, so strict `<`
                // deterministically retains the earliest pixel on elevation ties.
                if z.is_finite() && !z.rd_eq(sink_nd) && z < e.min_elev {
                    e.min_elev = z;
                    e.min_cell = (gr, gc);
                }
            }
        }
    }

    let mut by_reg: HashMap<u32, Vec<NestedDepression>> = HashMap::new();
    for d in deps.depressions {
        if reg_acc.contains_key(&d.region_id) {
            by_reg.entry(d.region_id).or_default().push(d);
        }
    }

    let mut out = Vec::with_capacity(by_reg.len());
    for (rid, reg_deps) in by_reg {
        let Some(acc) = reg_acc.remove(&rid) else {
            continue;
        };
        if acc.count == 0 {
            continue;
        }
        out.push(TileRegionOutput {
            min_cell: acc.min_cell,
            total_cells: acc.count,
            intersects_core: acc.intersects_core,
            touches_internal_pad_edge: acc.touches_internal_pad_edge,
            seam_cells: acc.seam_cells,
            depressions: reg_deps,
            pixels: acc.pixels,
        });
    }
    Ok(out)
}

fn build_tile_windows(w: usize, h: usize, ts: usize, ov: usize) -> Vec<TileWindow> {
    let mut windows = Vec::new();
    let mut r0 = 0usize;
    while r0 < h {
        let r1 = (r0 + ts).min(h);
        let mut c0 = 0usize;
        while c0 < w {
            let c1 = (c0 + ts).min(w);
            windows.push(TileWindow {
                core_r0: r0,
                core_c0: c0,
                core_r1: r1,
                core_c1: c1,
                pad_r0: r0.saturating_sub(ov),
                pad_c0: c0.saturating_sub(ov),
                pad_r1: r1.saturating_add(ov).min(h),
                pad_c1: c1.saturating_add(ov).min(w),
            });
            c0 = c1;
        }
        r0 = r1;
    }
    windows
}

struct KeptRegion {
    winner: TileRegionOutput,
    /// Optional fallback core pixels from non-winning cluster copies when the winning
    /// tile copy was itself clipped by an internal halo edge (giant cross-tile basin).
    extra_core_pixels: Vec<StitchedPixel>,
}

fn merge_and_export_tiled(
    per_tile: Vec<Vec<TileRegionOutput>>,
    num_tiles: usize,
    meta: &RasterMeta,
    out_dir: &Path,
    outputs: &TiledOutputOptions,
) -> Result<TiledDelineationResult> {
    let need_spatial = outputs.write_rasters || outputs.write_gpkg;
    let mut kept_regions: Vec<KeptRegion> = Vec::new();
    let mut seam_regions: Vec<TileRegionOutput> = Vec::new();

    for tile_regions in per_tile {
        for reg in tile_regions {
            if reg.seam_cells.is_empty() {
                if reg.intersects_core {
                    kept_regions.push(KeptRegion {
                        winner: reg,
                        extra_core_pixels: Vec::new(),
                    });
                }
            } else {
                seam_regions.push(reg);
            }
        }
    }

    if !seam_regions.is_empty() {
        let m = seam_regions.len();
        let mut parent: Vec<usize> = (0..m).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        let total_seam_cells: usize = seam_regions.iter().map(|r| r.seam_cells.len()).sum();
        let mut cell_owner: HashMap<usize, usize> = HashMap::with_capacity(total_seam_cells);
        let mut min_cell_owner: HashMap<(usize, usize), usize> = HashMap::with_capacity(m);
        for (i, reg) in seam_regions.iter().enumerate() {
            if let Some(&prev) = min_cell_owner.get(&reg.min_cell) {
                let ra = find(&mut parent, prev);
                let rb = find(&mut parent, i);
                if ra != rb {
                    parent[rb] = ra;
                }
            } else {
                min_cell_owner.insert(reg.min_cell, i);
            }
            for &gidx in &reg.seam_cells {
                if let Some(&prev) = cell_owner.get(&gidx) {
                    let ra = find(&mut parent, prev);
                    let rb = find(&mut parent, i);
                    if ra != rb {
                        parent[rb] = ra;
                    }
                } else {
                    cell_owner.insert(gidx, i);
                }
            }
        }

        // For each cluster root, track `(any_intersects_core, best_member_idx, member_indices)`.
        struct ClusterInfo {
            any_core: bool,
            best_idx: usize,
            members: Vec<usize>,
        }
        let mut clusters: HashMap<usize, ClusterInfo> = HashMap::new();
        for (i, reg) in seam_regions.iter().enumerate() {
            let root = find(&mut parent, i);
            clusters
                .entry(root)
                .and_modify(|ci| {
                    ci.any_core |= reg.intersects_core;
                    ci.members.push(i);
                    let best = &seam_regions[ci.best_idx];
                    let cand_rank = (!reg.touches_internal_pad_edge, reg.total_cells);
                    let best_rank = (!best.touches_internal_pad_edge, best.total_cells);
                    if cand_rank > best_rank
                        || (cand_rank == best_rank && reg.min_cell < best.min_cell)
                    {
                        ci.best_idx = i;
                    }
                })
                .or_insert(ClusterInfo {
                    any_core: reg.intersects_core,
                    best_idx: i,
                    members: vec![i],
                });
        }

        let mut seam_opts: Vec<Option<TileRegionOutput>> =
            seam_regions.into_iter().map(Some).collect();
        let mut cluster_list: Vec<ClusterInfo> =
            clusters.into_values().filter(|c| c.any_core).collect();
        cluster_list.sort_by_key(|c| c.best_idx);

        for ci in cluster_list {
            let Some(winner) = seam_opts[ci.best_idx].take() else {
                continue;
            };
            let mut extra_core_pixels = Vec::new();
            if need_spatial && winner.touches_internal_pad_edge {
                for idx in ci.members {
                    if idx == ci.best_idx {
                        continue;
                    }
                    if let Some(other) = seam_opts[idx].take() {
                        extra_core_pixels.extend(other.pixels.into_iter().filter(|p| p.in_core));
                    }
                }
            }
            kept_regions.push(KeptRegion {
                winner,
                extra_core_pixels,
            });
        }
    }

    kept_regions.sort_by_key(|r| (r.winner.min_cell, r.winner.total_cells));

    // Reassign globally unique sequential IDs, parent_ids, children_ids, and region_ids,
    // and stitch spatial rasters if requested.
    let n_pix = meta.width.checked_mul(meta.height).ok_or_else(|| {
        Error::DimensionMismatch(format!(
            "raster dimensions overflow ({}x{})",
            meta.width, meta.height
        ))
    })?;
    let mut stitched_id = if need_spatial {
        vec![0i32; n_pix]
    } else {
        Vec::new()
    };
    let mut stitched_lvl = if outputs.write_rasters {
        vec![0i32; n_pix]
    } else {
        Vec::new()
    };
    let mut stitched_depth = if outputs.write_rasters {
        vec![0.0f64; n_pix]
    } else {
        Vec::new()
    };

    let mut merged = Vec::new();
    let mut next_dep_id = 1u32;
    let mut next_reg_id = 1u32;
    let mut max_level = 0u32;

    for kr in kept_regions {
        let tile_deps = kr.winner.depressions;
        if tile_deps.is_empty() {
            continue;
        }
        let mut id_map: HashMap<u32, u32> = HashMap::with_capacity(tile_deps.len());
        let mut reg_map: HashMap<u32, u32> = HashMap::new();
        let mut root_global_id = 0u32;
        let mut root_level = 0u32;

        for d in &tile_deps {
            let gid = next_dep_id;
            id_map.insert(d.id, gid);
            next_dep_id += 1;
            if d.parent_id == 0 && d.level >= root_level {
                root_global_id = gid;
                root_level = d.level;
            }
            reg_map.entry(d.region_id).or_insert_with(|| {
                let r = next_reg_id;
                next_reg_id += 1;
                r
            });
        }
        if root_global_id == 0 {
            root_global_id = id_map[&tile_deps[0].id];
            root_level = tile_deps[0].level;
        }

        if need_spatial {
            // Paint non-winning tail pixels first (for giant cross-tile basins), then
            // paint the winning copy's exact unclipped pixels on top.
            for p in &kr.extra_core_pixels {
                let idx = p.gr * meta.width + p.gc;
                if stitched_id[idx] == 0 {
                    stitched_id[idx] = root_global_id as i32;
                    if outputs.write_rasters {
                        stitched_lvl[idx] = root_level as i32;
                        stitched_depth[idx] = p.depth;
                    }
                }
            }
            for p in &kr.winner.pixels {
                if let Some(&gid) = id_map.get(&p.local_did) {
                    let idx = p.gr * meta.width + p.gc;
                    stitched_id[idx] = gid as i32;
                    if outputs.write_rasters {
                        stitched_lvl[idx] = p.level as i32;
                        stitched_depth[idx] = p.depth;
                    }
                }
            }
        }

        for mut d in tile_deps {
            d.id = id_map[&d.id];
            d.parent_id = if d.parent_id > 0 {
                id_map.get(&d.parent_id).copied().unwrap_or(0)
            } else {
                0
            };
            d.children_ids = d
                .children_ids
                .into_iter()
                .filter_map(|c| id_map.get(&c).copied())
                .collect();
            d.region_id = reg_map[&d.region_id];
            if d.level > max_level {
                max_level = d.level;
            }
            merged.push(d);
        }
    }

    std::fs::create_dir_all(out_dir)?;
    let csv_path = out_dir.join("depressions_info.csv");
    write_depressions_csv(&merged, &csv_path)?;

    let mut depression_id_tif = None;
    let mut depression_level_tif = None;
    let mut depth_tif = None;
    let mut gpkg_path = None;

    if need_spatial {
        let mut id_arr = Array2D::from_vec(meta.width, meta.height, stitched_id)?;
        id_arr.geotransform = meta.geotransform;
        id_arr.projection = meta.projection.clone();
        id_arr.set_no_data(0);

        if outputs.write_rasters {
            let id_path = out_dir.join("depression_id.tif");
            save_raster(
                &id_arr,
                &id_path,
                Some("delineate_depressions_tiled: id"),
                outputs.compress,
            )?;
            depression_id_tif = Some(id_path);

            let mut lvl_arr = Array2D::from_vec(meta.width, meta.height, stitched_lvl)?;
            lvl_arr.geotransform = meta.geotransform;
            lvl_arr.projection = meta.projection.clone();
            lvl_arr.set_no_data(0);
            let lvl_path = out_dir.join("depression_level.tif");
            save_raster(
                &lvl_arr,
                &lvl_path,
                Some("delineate_depressions_tiled: level"),
                outputs.compress,
            )?;
            depression_level_tif = Some(lvl_path);

            let mut depth_arr = Array2D::from_vec(meta.width, meta.height, stitched_depth)?;
            depth_arr.geotransform = meta.geotransform;
            depth_arr.projection = meta.projection.clone();
            depth_arr.set_no_data(0.0);
            let d_path = out_dir.join("depth.tif");
            save_raster(
                &depth_arr,
                &d_path,
                Some("delineate_depressions_tiled: depth"),
                outputs.compress,
            )?;
            depth_tif = Some(d_path);
        }

        if outputs.write_gpkg {
            let gp = out_dir.join("depressions.gpkg");
            write_depressions_vector_with_enclosing(&id_arr, &merged, &gp, true)?;
            gpkg_path = Some(gp);
        }
    }

    Ok(TiledDelineationResult {
        num_tiles,
        max_level,
        depressions: merged,
        csv_path,
        depression_id_tif,
        depression_level_tif,
        depth_tif,
        gpkg_path,
    })
}

fn run_tiled_in_memory<T: DemNum>(
    dem: &Array2D<T>,
    out_dir: &Path,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
) -> Result<TiledDelineationResult> {
    let meta = RasterMeta {
        width: dem.width(),
        height: dem.height(),
        geotransform: dem.geotransform,
        projection: dem.projection.clone(),
    };
    let windows = build_tile_windows(meta.width, meta.height, options.tile_size, options.overlap);
    let num_tiles = windows.len();
    let run_one = |win: TileWindow| -> Result<Vec<TileRegionOutput>> {
        let tile = extract_tile_f64(dem, win)?;
        process_one_tile(tile, win, meta.width, meta.height, options, outputs)
    };

    #[cfg(feature = "parallel")]
    let per_tile: Vec<Vec<TileRegionOutput>> = windows
        .into_par_iter()
        .map(run_one)
        .collect::<Result<Vec<_>>>()?;

    #[cfg(not(feature = "parallel"))]
    let per_tile: Vec<Vec<TileRegionOutput>> = windows
        .into_iter()
        .map(run_one)
        .collect::<Result<Vec<_>>>()?;

    merge_and_export_tiled(per_tile, num_tiles, &meta, out_dir, outputs)
}

fn validate_tiled_options(options: &TiledDelineationOptions) -> Result<()> {
    if options.tile_size < 16 {
        return Err(Error::InvalidArgument(format!(
            "tile_size must be >= 16, got {}",
            options.tile_size
        )));
    }
    if options.kernel_size > 1025 {
        return Err(Error::InvalidArgument(format!(
            "kernel_size must be <= 1025, got {}",
            options.kernel_size
        )));
    }
    if options.kernel_size > 1 && options.overlap < options.kernel_size / 2 {
        return Err(Error::InvalidArgument(format!(
            "overlap ({}) must be >= kernel_size / 2 ({}) when kernel_size > 1",
            options.overlap,
            options.kernel_size / 2
        )));
    }
    if !options.min_depth.is_finite() || options.min_depth < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "min_depth must be finite and >= 0, got {}",
            options.min_depth
        )));
    }
    if !options.interval.is_finite() || options.interval == 0.0 {
        return Err(Error::InvalidArgument(format!(
            "interval must be finite and non-zero, got {}",
            options.interval
        )));
    }
    Ok(())
}

/// Runs tiled nested depression delineation on a DEM file with true windowed out-of-core
/// reading (`O(tile_size^2)` peak DEM memory) and overlapping halos.
pub fn delineate_depressions_tiled(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &TiledDelineationOptions,
) -> Result<TiledDelineationResult> {
    delineate_depressions_tiled_with_outputs(
        in_dem,
        out_dir,
        options,
        &TiledOutputOptions::default(),
    )
}

/// Runs out-of-core tiled nested depression delineation with optional per-tile breaching
/// and full-DEM stitched spatial outputs (`depression_id.tif`, `depression_level.tif`,
/// `depth.tif`, and `depressions.gpkg`).
pub fn delineate_depressions_tiled_with_outputs(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
) -> Result<TiledDelineationResult> {
    validate_tiled_options(options)?;
    let in_dem = in_dem.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join("depressions_info.csv");
    let id_path = out_dir.join("depression_id.tif");
    let lvl_path = out_dir.join("depression_level.tif");
    let depth_path = out_dir.join("depth.tif");
    let gpkg_path = out_dir.join("depressions.gpkg");
    let mut out_paths: Vec<&Path> = vec![csv_path.as_path()];
    if outputs.write_rasters {
        out_paths.push(id_path.as_path());
        out_paths.push(lvl_path.as_path());
        out_paths.push(depth_path.as_path());
    }
    if outputs.write_gpkg {
        out_paths.push(gpkg_path.as_path());
    }
    crate::raster_ops::ensure_no_path_collision(in_dem, &out_paths)?;

    // Out-of-core windowed path via `terrio::open_raster_lazy` for GeoTIFF / COG / lazy drivers.
    if let Some(p_str) = in_dem.to_str()
        && let Ok(lazy) = terrio::open_raster_lazy(p_str)
    {
        let width = lazy.width();
        let height = lazy.height();
        if width > 0 && height > 0 {
            let crs = lazy.crs();
            let projection = crs.to_wkt().or_else(|| crs.to_proj4()).unwrap_or_default();
            let meta = RasterMeta {
                width,
                height,
                geotransform: lazy.geo_transform(),
                projection,
            };
            drop(lazy);

            let windows =
                build_tile_windows(meta.width, meta.height, options.tile_size, options.overlap);
            let num_tiles = windows.len();
            let run_one = |win: TileWindow| -> Result<Vec<TileRegionOutput>> {
                let tile = read_tile_window_lazy(p_str, &meta, win)?;
                process_one_tile(tile, win, meta.width, meta.height, options, outputs)
            };

            #[cfg(feature = "parallel")]
            let per_tile: Vec<Vec<TileRegionOutput>> = windows
                .into_par_iter()
                .map(run_one)
                .collect::<Result<Vec<_>>>()?;

            #[cfg(not(feature = "parallel"))]
            let per_tile: Vec<Vec<TileRegionOutput>> = windows
                .into_iter()
                .map(run_one)
                .collect::<Result<Vec<_>>>()?;

            return merge_and_export_tiled(per_tile, num_tiles, &meta, out_dir, outputs);
        }
    }

    // Fallback for non-lazy formats (e.g., legacy ESRI ASCII grids or non-UTF-8 paths).
    if options.use_f32 {
        let dem = load_raster::<f32>(in_dem, None)?;
        run_tiled_in_memory(&dem, out_dir, options, outputs)
    } else {
        let dem = load_raster::<f64>(in_dem, None)?;
        run_tiled_in_memory(&dem, out_dir, options, outputs)
    }
}

/// Loads a DEM raster using `f32` storage internally (halving RAM footprint on large
/// float32 3DEP rasters) and converts on demand.
pub fn load_dem_f32(path: impl AsRef<Path>) -> Result<Array2D<f32>> {
    Ok(load_raster::<f32>(path, None)?)
}
