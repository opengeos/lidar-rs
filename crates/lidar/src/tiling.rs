//! Tiled regional depression delineation with overlapping halos and optional `f32` loading.
//!
//! Splits large DEM rasters into overlapping square tiles, delineates nested
//! depressions independently per tile (in parallel when the `parallel` feature is
//! enabled), deduplicates features along tile overlap seams using non-overlapping
//! tile cores and minimum-elevation keys, and merges globally numbered depression
//! attributes. Peak per-tile working memory is bounded by the tile size, while the
//! source raster is loaded into memory (optionally as `f32` to halve RAM usage).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, Topology, load_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::filling::extract_sinks_in_memory;
use crate::filtering::median_filter;
use crate::slicing::{NestedDepression, delineate_depressions_in_memory, write_depressions_csv};

/// Options controlling [`delineate_depressions_tiled`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TiledDelineationOptions {
    /// Core (non-overlapping) tile width and height in pixels (default `2048`).
    pub tile_size: usize,
    /// Overlap halo added around each side of a tile in pixels (default `128`).
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
    /// Load source DEM as `f32` to halve peak memory usage on large rasters (default `true`).
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

/// Summary and merged outputs produced by [`delineate_depressions_tiled`].
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

/// Extracts a tile sub-grid as `Array2D<f64>` from either an `f32` or `f64` source raster,
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
    let gt = src.geotransform;
    let c0f = win.pad_c0 as f64;
    let r0f = win.pad_r0 as f64;
    tile.geotransform = [
        gt[0] + c0f * gt[1] + r0f * gt[2],
        gt[1],
        gt[2],
        gt[3] + c0f * gt[4] + r0f * gt[5],
        gt[4],
        gt[5],
    ];
    tile.projection = src.projection.clone();
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
}

fn process_one_tile(
    tile: Array2D<f64>,
    win: TileWindow,
    full_w: usize,
    full_h: usize,
    options: &TiledDelineationOptions,
) -> Result<Vec<TileRegionOutput>> {
    let smoothed = if options.kernel_size > 1 {
        median_filter(&tile, options.kernel_size)?
    } else {
        tile
    };
    let sinks = extract_sinks_in_memory(&smoothed, options.min_size, options.topology)?;
    let deps = delineate_depressions_in_memory(
        &sinks.sink,
        options.min_size,
        options.min_depth,
        options.interval,
    )?;
    if deps.depressions.is_empty() {
        return Ok(Vec::new());
    }

    let tw = deps.depression_id.width();
    let th = deps.depression_id.height();
    let id_data = deps.depression_id.data();
    let sink_data = sinks.sink.data();
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
                });
                e.count += 1;
                if gr >= win.core_r0 && gr < win.core_r1 && gc >= win.core_c0 && gc < win.core_c1 {
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
        });
    }
    Ok(out)
}

fn run_tiled_generic<T: DemNum>(
    dem: &Array2D<T>,
    out_dir: &Path,
    options: &TiledDelineationOptions,
) -> Result<TiledDelineationResult> {
    let w = dem.width();
    let h = dem.height();
    let ts = options.tile_size;
    let ov = options.overlap;

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

    let num_tiles = windows.len();
    let run_one = |win: TileWindow| -> Result<Vec<TileRegionOutput>> {
        let tile = extract_tile_f64(dem, win)?;
        process_one_tile(tile, win, w, h, options)
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

    // Global deduplication across tile seams:
    // - Regions strictly inside a tile interior (`seam_cells.is_empty()`) cannot overlap any
    //   neighbor tile and are kept directly.
    // - Regions overlapping a tile seam halo band are clustered via Union-Find whenever two tile
    //   copies share any global pixel or share the same `min_cell`. Within each cluster we require
    //   at least one copy to intersect its tile core (rejecting spurious halo-only edge artifacts)
    //   and retain the best copy (preferring unclipped copies, then largest pixel count).
    let mut dedup_regions: Vec<TileRegionOutput> = Vec::new();
    let mut seam_regions: Vec<TileRegionOutput> = Vec::new();
    for tile_regions in per_tile {
        for reg in tile_regions {
            if reg.seam_cells.is_empty() {
                if reg.intersects_core {
                    dedup_regions.push(reg);
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
        let mut cell_owner: HashMap<usize, usize> = HashMap::new();
        let mut min_cell_owner: HashMap<(usize, usize), usize> = HashMap::new();
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

        // For each cluster root, track `(any_intersects_core, best_member_idx)`.
        let mut clusters: HashMap<usize, (bool, usize)> = HashMap::new();
        for (i, reg) in seam_regions.iter().enumerate() {
            let root = find(&mut parent, i);
            clusters
                .entry(root)
                .and_modify(|(any_core, best_idx)| {
                    *any_core |= reg.intersects_core;
                    let best = &seam_regions[*best_idx];
                    let cand_rank = (!reg.touches_internal_pad_edge, reg.total_cells);
                    let best_rank = (!best.touches_internal_pad_edge, best.total_cells);
                    if cand_rank > best_rank
                        || (cand_rank == best_rank && reg.min_cell < best.min_cell)
                    {
                        *best_idx = i;
                    }
                })
                .or_insert((reg.intersects_core, i));
        }

        let mut winners: Vec<usize> = clusters
            .into_values()
            .filter_map(|(any_core, idx)| any_core.then_some(idx))
            .collect();
        winners.sort_unstable();
        let mut keep = vec![false; m];
        for idx in winners {
            keep[idx] = true;
        }
        for (reg, is_kept) in seam_regions.into_iter().zip(keep) {
            if is_kept {
                dedup_regions.push(reg);
            }
        }
    }

    dedup_regions.sort_by_key(|r| (r.min_cell, r.total_cells));

    // Reassign globally unique sequential IDs, parent_ids, children_ids, and region_ids.
    let mut merged = Vec::new();
    let mut next_dep_id = 1u32;
    let mut next_reg_id = 1u32;
    let mut max_level = 0u32;

    for reg in dedup_regions {
        let tile_deps = reg.depressions;
        if tile_deps.is_empty() {
            continue;
        }
        let mut id_map: HashMap<u32, u32> = HashMap::with_capacity(tile_deps.len());
        let mut reg_map: HashMap<u32, u32> = HashMap::new();
        for d in &tile_deps {
            id_map.insert(d.id, next_dep_id);
            next_dep_id += 1;
            reg_map.entry(d.region_id).or_insert_with(|| {
                let r = next_reg_id;
                next_reg_id += 1;
                r
            });
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

    Ok(TiledDelineationResult {
        num_tiles,
        max_level,
        depressions: merged,
        csv_path,
    })
}

/// Runs tiled nested depression delineation on a DEM file with overlapping halos and
/// optional `f32` loading for reduced peak memory consumption.
pub fn delineate_depressions_tiled(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &TiledDelineationOptions,
) -> Result<TiledDelineationResult> {
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
    let in_dem = in_dem.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;
    let csv_path = out_dir.join("depressions_info.csv");
    crate::raster_ops::ensure_no_path_collision(in_dem, &[&csv_path])?;
    if options.use_f32 {
        let dem = load_raster::<f32>(in_dem, None)?;
        run_tiled_generic(&dem, out_dir, options)
    } else {
        let dem = load_raster::<f64>(in_dem, None)?;
        run_tiled_generic(&dem, out_dir, options)
    }
}

/// Loads a DEM raster using `f32` storage internally (halving RAM footprint on large
/// float32 3DEP rasters) and converts on demand.
pub fn load_dem_f32(path: impl AsRef<Path>) -> Result<Array2D<f32>> {
    Ok(load_raster::<f32>(path, None)?)
}
