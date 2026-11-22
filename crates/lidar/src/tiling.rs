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
    /// Global `(row, col)` of the lowest-elevation cell in this sink region (used for global deduplication across seams).
    min_cell: (usize, usize),
    /// Total pixel count of this sink region inside the tile (larger = less truncated).
    total_cells: usize,
    /// Whether this tile copy's centroid falls inside the tile's non-overlapping core.
    core_owned: bool,
    /// Complete nested depression hierarchy for this sink region.
    depressions: Vec<NestedDepression>,
}

fn process_one_tile(
    tile: Array2D<f64>,
    win: TileWindow,
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

    // Compute pixel centroid and global minimum-elevation cell for each top-level sink region.
    // Keeping complete regions whose centroid falls inside this tile's non-overlapping core
    // preserves parent-child hierarchy integrity for depressions that fit within the overlap halo.
    // Global deduplication by `min_cell` removes duplicates when both neighbouring tiles see the
    // same lowest cell; basins wider than `overlap` can still be clipped at a tile edge, so
    // callers should set `overlap` larger than the expected maximum basin width.
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
    struct RegAcc {
        sum_r: f64,
        sum_c: f64,
        count: usize,
        min_elev: f64,
        min_cell: (usize, usize),
    }
    let mut reg_acc: HashMap<u32, RegAcc> = HashMap::new();
    for r in 0..th {
        let off = r * tw;
        let gr = win.pad_r0 + r;
        for c in 0..tw {
            let did = id_data[off + c];
            if did > 0
                && let Some(&rid) = dep_region.get(&(did as u32))
            {
                let gc = win.pad_c0 + c;
                let z = sink_data[off + c];
                let e = reg_acc.entry(rid).or_insert(RegAcc {
                    sum_r: 0.0,
                    sum_c: 0.0,
                    count: 0,
                    min_elev: f64::INFINITY,
                    min_cell: (gr, gc),
                });
                e.sum_r += gr as f64;
                e.sum_c += gc as f64;
                e.count += 1;
                if z.is_finite()
                    && !z.rd_eq(sink_nd)
                    && (z < e.min_elev
                        || ((z - e.min_elev).abs() <= 1e-12 && (gr, gc) < e.min_cell))
                {
                    e.min_elev = z;
                    e.min_cell = (gr, gc);
                }
            }
        }
    }

    let mut reg_meta: HashMap<u32, ((usize, usize), usize, bool)> = HashMap::new();
    for (rid, acc) in reg_acc {
        if acc.count == 0 {
            continue;
        }
        let gr = acc.sum_r / acc.count as f64;
        let gc = acc.sum_c / acc.count as f64;
        let core_owned = gr >= win.core_r0 as f64
            && gr < win.core_r1 as f64
            && gc >= win.core_c0 as f64
            && gc < win.core_c1 as f64;
        reg_meta.insert(rid, (acc.min_cell, acc.count, core_owned));
    }

    let mut by_reg: HashMap<u32, Vec<NestedDepression>> = HashMap::new();
    for d in deps.depressions {
        if reg_meta.contains_key(&d.region_id) {
            by_reg.entry(d.region_id).or_default().push(d);
        }
    }

    let mut out = Vec::with_capacity(by_reg.len());
    for (rid, reg_deps) in by_reg {
        let (min_cell, total_cells, core_owned) = reg_meta[&rid];
        out.push(TileRegionOutput {
            min_cell,
            total_cells,
            core_owned,
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
        process_one_tile(tile, win, options)
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

    // Global deduplication across tile seams keyed by each sink region's lowest pixel within
    // the padded tile: require at least one core-owned observation to emit a group, while
    // selecting the largest (least-clipped) copy observed across all overlapping tiles.
    let mut best_by_min_cell: HashMap<(usize, usize), (bool, TileRegionOutput)> = HashMap::new();
    for tile_regions in per_tile {
        for reg in tile_regions {
            match best_by_min_cell.entry(reg.min_cell) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((reg.core_owned, reg));
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let (any_core, best) = e.get_mut();
                    *any_core |= reg.core_owned;
                    if reg.total_cells > best.total_cells {
                        *best = reg;
                    }
                }
            }
        }
    }
    let mut dedup_regions: Vec<TileRegionOutput> = best_by_min_cell
        .into_values()
        .filter_map(|(any_core, reg)| any_core.then_some(reg))
        .collect();
    dedup_regions.sort_by_key(|r| r.min_cell);

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
