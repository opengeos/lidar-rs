//! Tiled/out-of-core regional depression delineation and `f32` memory-efficient helpers.
//!
//! Splits large DEM rasters into overlapping square tiles, delineates nested
//! depressions independently per tile (in parallel when the `parallel` feature is
//! enabled), deduplicates features along tile overlap seams using non-overlapping
//! tile cores, and merges globally numbered depression attributes.

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

/// Extracts a tile sub-grid as `Array2D<f64>` from either an `f32` or `f64` source raster.
fn extract_tile_f64<T: DemNum>(src: &Array2D<T>, win: TileWindow) -> Result<Array2D<f64>> {
    let w = src.width();
    let tw = win.pad_c1 - win.pad_c0;
    let th = win.pad_r1 - win.pad_r0;
    let nd = src.no_data();
    let sdata = src.data();
    let mut data = Vec::with_capacity(tw * th);
    for r in win.pad_r0..win.pad_r1 {
        let off = r * w;
        for c in win.pad_c0..win.pad_c1 {
            let v = sdata[off + c];
            if v.rd_eq(nd) {
                data.push(-9999.0f64);
            } else {
                let f = v.to_f64();
                data.push(if f.is_finite() { f } else { -9999.0 });
            }
        }
    }
    let mut tile = Array2D::from_vec(tw, th, data)?;
    tile.set_no_data(-9999.0);
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

fn process_one_tile(
    tile: Array2D<f64>,
    win: TileWindow,
    options: &TiledDelineationOptions,
) -> Result<Vec<NestedDepression>> {
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

    // Compute pixel centroid (in global DEM coordinates) for each top-level sink region.
    // Keeping complete regions whose centroid falls inside this tile's non-overlapping core
    // preserves parent-child hierarchy integrity without cutting multi-level complexes in half.
    let tw = deps.depression_id.width();
    let th = deps.depression_id.height();
    let id_data = deps.depression_id.data();
    let dep_region: HashMap<u32, u32> = deps
        .depressions
        .iter()
        .map(|d| (d.id, d.region_id))
        .collect();
    let mut reg_acc: HashMap<u32, (f64, f64, usize)> = HashMap::new();
    for r in 0..th {
        let off = r * tw;
        for c in 0..tw {
            let did = id_data[off + c];
            if did > 0
                && let Some(&rid) = dep_region.get(&(did as u32))
            {
                let e = reg_acc.entry(rid).or_insert((0.0, 0.0, 0));
                e.0 += (win.pad_r0 + r) as f64;
                e.1 += (win.pad_c0 + c) as f64;
                e.2 += 1;
            }
        }
    }

    let mut kept_regions = std::collections::HashSet::new();
    for (rid, (sr, sc, cnt)) in reg_acc {
        if cnt == 0 {
            continue;
        }
        let gr = sr / cnt as f64;
        let gc = sc / cnt as f64;
        if gr >= win.core_r0 as f64
            && gr < win.core_r1 as f64
            && gc >= win.core_c0 as f64
            && gc < win.core_c1 as f64
        {
            kept_regions.insert(rid);
        }
    }

    Ok(deps
        .depressions
        .into_iter()
        .filter(|d| kept_regions.contains(&d.region_id))
        .collect())
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
                pad_r1: (r1 + ov).min(h),
                pad_c1: (c1 + ov).min(w),
            });
            c0 = c1;
        }
        r0 = r1;
    }

    let num_tiles = windows.len();
    let run_one = |win: TileWindow| -> Result<Vec<NestedDepression>> {
        let tile = extract_tile_f64(dem, win)?;
        process_one_tile(tile, win, options)
    };

    #[cfg(feature = "parallel")]
    let per_tile: Vec<Vec<NestedDepression>> = windows
        .into_par_iter()
        .map(run_one)
        .collect::<Result<Vec<_>>>()?;

    #[cfg(not(feature = "parallel"))]
    let per_tile: Vec<Vec<NestedDepression>> = windows
        .into_iter()
        .map(run_one)
        .collect::<Result<Vec<_>>>()?;

    // Reassign globally unique sequential IDs, parent_ids, children_ids, and region_ids.
    let mut merged = Vec::new();
    let mut next_dep_id = 1u32;
    let mut next_reg_id = 1u32;
    let mut max_level = 0u32;

    for tile_deps in per_tile {
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
    let out_dir = out_dir.as_ref();
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
