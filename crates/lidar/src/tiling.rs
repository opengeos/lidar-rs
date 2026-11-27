//! Tiled regional depression delineation with windowed out-of-core raster I/O,
//! overlapping halos, optional per-tile breaching, and optional full-DEM stitched spatial outputs.
//!
//! Splits large DEM rasters into overlapping square tiles, reads each tile's padded window
//! on demand via `terrio::open_raster_lazy` + `read_window` (falling back to `load_raster`
//! only for legacy non-lazy formats such as ESRI ASCII `.asc` grids), delineates nested
//! depressions independently per tile (in parallel when the `parallel` feature is enabled),
//! deduplicates features across tile seams via Union-Find clustering, and optionally
//! stitches full-DEM `depression_id.tif`, `depression_level.tif`, `depth.tif`, and
//! `depressions.gpkg` outputs.
//!
//! ## Memory Bounds
//!
//! - **Metadata-only mode** (`write_rasters = false`, `write_gpkg = false`, default in
//!   [`delineate_depressions_tiled`]): peak DEM memory is `O(tile_size^2)` per worker tile
//!   plus merged depression attributes and seam-band cell indices.
//! - **Stitched spatial output mode** ([`delineate_depressions_tiled_with_outputs`] with
//!   `write_rasters = true` or `write_gpkg = true`): interior sink pixels are spooled by
//!   row band to temporary scratch files and streamed out through `terrio::GeoTiffWriter`
//!   in fixed-height row strips (`STREAM_BAND_ROWS = 256`) so `write_rasters = true` never
//!   allocates full-DEM stitched buffers in RAM. When `write_gpkg = true`, only the `i32`
//!   depression-ID raster (`4` bytes/pixel) is assembled for polygonization.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(test)]
use richdem::save_raster;
use richdem::{Array2D, DemNum, Topology, dataset_to_dynamic, load_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::SpatialRef;
use terrio::raster::RasterData;
use terrio::{DataType, GeoTiffWriter, Raster, RasterDataset, RasterWriteOptions};

use crate::conditioning::{BreachOptions, breach_depressions, validate_breach_options};
use crate::error::{Error, Result};
use crate::filling::extract_sinks_in_memory;
use crate::filtering::median_filter;
use crate::progress::ExecutionControl;
use crate::slicing::{
    NestedDepression, delineate_depressions_in_memory, write_depressions_csv,
    write_depressions_vector_with_enclosing,
};

#[cfg(test)]
static LAZY_WINDOW_READS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static EAGER_FALLBACK_READS: AtomicUsize = AtomicUsize::new(0);

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
    /// Load source DEM as `f32` when running on the eager/in-memory tiled path (used for
    /// formats that do not support windowed lazy reads; default `true`). Windowed GeoTIFF/COG
    /// inputs read one padded tile at a time directly into `f64`.
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
///
/// Note on memory: enabling `write_rasters` allocates full-extent stitched buffers
/// (`16` bytes/pixel: `i32` ID, `i32` level, `f64` depth) and enabling `write_gpkg` alone
/// allocates a full-extent `i32` ID buffer (`4` bytes/pixel) to polygonize the mosaic.
/// When both are `false`, peak DEM memory remains `O(tile_size^2)` per worker tile.
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
    ///
    /// Note: breaching runs independently on each padded tile window, so least-cost breach
    /// paths extending beyond `overlap` cells can vary with tile layout. Increase `overlap`
    /// or run global [`crate::breach_depressions`] beforehand when tile-invariant breaching
    /// is required.
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

/// Compact per-pixel payload retained ONLY for seam-crossing regions when spatial
/// outputs are enabled (interior regions are written directly into [`StitchedBuffers`]).
#[derive(Clone, Copy)]
struct SeamPixel {
    gidx: usize,
    local_did: u32,
    level: u32,
    depth: f64,
}

/// Row-band height used when spooling and streaming stitched output GeoTIFFs so peak RAM
/// remains `O(STREAM_BAND_ROWS * width)` instead of `O(height * width)`.
const STREAM_BAND_ROWS: usize = 256;

/// Packed byte size of one spooled interior pixel record:
/// `(local_row: u16, _pad: u16, col: u32, temp_id: u32, lvl: i32, depth: f64)` = 24 bytes.
const SPOOL_REC_BYTES: usize = 24;

static SPOOL_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

/// Disk-backed per-row-band spool for strictly interior depression pixels, avoiding
/// full-DEM `StitchedBuffers` RAM allocations during tiled delineation.
struct RowBandSpool {
    dir: PathBuf,
    num_bands: usize,
    width: usize,
    height: usize,
    next_temp_id: u32,
    band_files: Vec<Option<BufWriter<File>>>,
}

impl RowBandSpool {
    fn new(out_dir: &Path, width: usize, height: usize) -> Result<Self> {
        fs::create_dir_all(out_dir)?;
        let seq = SPOOL_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = out_dir.join(format!(".lidar_tiled_spool_{}_{}", std::process::id(), seq));
        fs::create_dir_all(&dir)?;
        let num_bands = height.div_ceil(STREAM_BAND_ROWS);
        let mut band_files = Vec::with_capacity(num_bands);
        for _ in 0..num_bands {
            band_files.push(None);
        }
        Ok(Self {
            dir,
            num_bands,
            width,
            height,
            next_temp_id: 1,
            band_files,
        })
    }

    fn band_path(&self, band: usize) -> PathBuf {
        self.dir.join(format!("band_{band}.bin"))
    }

    fn append_band_bytes(&mut self, band: usize, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if self.band_files[band].is_none() {
            let f = File::create(self.band_path(band))?;
            self.band_files[band] = Some(BufWriter::new(f));
        }
        if let Some(w) = self.band_files[band].as_mut() {
            w.write_all(bytes)?;
        }
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for w in self.band_files.iter_mut().flatten() {
            w.flush()?;
        }
        self.band_files.clear();
        Ok(())
    }
}

impl Drop for RowBandSpool {
    fn drop(&mut self) {
        self.band_files.clear();
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[inline]
fn encode_spool_record(
    buf: &mut Vec<u8>,
    local_row: u16,
    col: u32,
    temp_id: u32,
    lvl: i32,
    depth: f64,
) {
    buf.extend_from_slice(&local_row.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&col.to_le_bytes());
    buf.extend_from_slice(&temp_id.to_le_bytes());
    buf.extend_from_slice(&lvl.to_le_bytes());
    buf.extend_from_slice(&depth.to_le_bytes());
}

fn build_streaming_header(
    meta: &RasterMeta,
    data_type: DataType,
    nodata: f64,
) -> Result<RasterDataset> {
    let mut raster = Raster {
        cols: meta.width,
        rows: meta.height,
        bands: 1,
        x_min: meta.geotransform[0],
        y_min: meta.geotransform[3] + meta.height as f64 * meta.geotransform[5],
        cell_size_x: meta.geotransform[1].abs().max(1e-12),
        cell_size_y: meta.geotransform[5].abs().max(1e-12),
        nodata,
        data_type,
        crs: Default::default(),
        metadata: Default::default(),
        data: RasterData::new_filled(data_type, 0, 0.0),
    };
    if !meta.projection.is_empty() {
        let sref = SpatialRef::parse(&meta.projection).map_err(|e| {
            Error::InvalidArgument(format!("invalid projection '{}': {e}", meta.projection))
        })?;
        raster.crs = sref.to_crs_info();
    }
    let mut ds = RasterDataset::new(raster, true);
    ds.set_geo_transform(meta.geotransform);
    Ok(ds)
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

/// Reads a single padded tile window directly from disk via a reusable `terrio::LazyRaster`
/// handle without loading the full raster into memory or reopening the TIFF IFD per tile.
fn read_tile_window_with_lazy(
    lazy: &mut terrio::LazyRaster,
    path_str: &str,
    meta: &RasterMeta,
    win: TileWindow,
) -> Result<Array2D<f64>> {
    #[cfg(test)]
    LAZY_WINDOW_READS.fetch_add(1, Ordering::Relaxed);

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
    /// Sparse per-pixel spatial payload retained ONLY for seam-crossing regions when
    /// `write_rasters || write_gpkg`. Interior regions are written directly into
    /// [`StitchedBuffers`] with negative temporary IDs recorded in `direct_temp_ids`.
    seam_pixels: Vec<SeamPixel>,
    /// Mapping from `local_did -> temp_id` (positive magnitude of negative tag written into
    /// `StitchedBuffers::id`) for strictly interior regions streamed directly during the tile pass.
    direct_temp_ids: Vec<(u32, u32)>,
}

fn process_one_tile(
    tile: Array2D<f64>,
    win: TileWindow,
    full_w: usize,
    full_h: usize,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
    spool: Option<&Mutex<RowBandSpool>>,
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

    let need_spatial = spool.is_some();
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

    // Prepare interior write-back entries and seam-region pixels completely OUTSIDE the
    // stitched-buffer mutex so parallel tile workers never serialize during pixel scanning.
    let mut interior_temp_map: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut seam_pixel_map: HashMap<u32, Vec<SeamPixel>> = HashMap::new();

    if need_spatial {
        let mut interior_did_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut interior_rids: Vec<u32> = Vec::new();
        for (&rid, acc) in &reg_acc {
            if acc.seam_cells.is_empty()
                && acc.intersects_core
                && let Some(rdeps) = by_reg.get(&rid)
            {
                interior_rids.push(rid);
                for d in rdeps {
                    interior_did_set.insert(d.id);
                }
            }
        }

        // Scan tile pixels without holding any lock.
        let mut interior_writes: Vec<(usize, u32, i32, f64)> = Vec::new();
        for r in 0..th {
            let off = r * tw;
            let gr = win.pad_r0 + r;
            for c in 0..tw {
                let did = id_data[off + c];
                if did <= 0 {
                    continue;
                }
                let did_u = did as u32;
                let Some(&rid) = dep_region.get(&did_u) else {
                    continue;
                };
                let gc = win.pad_c0 + c;
                let gidx = gr * full_w + gc;
                let dval = depth_data[off + c];
                let depth = if dval.is_finite() && dval > 0.0 {
                    dval
                } else {
                    0.0
                };
                let lvl = lvl_data[off + c].max(0);
                if interior_did_set.contains(&did_u) {
                    interior_writes.push((gidx, did_u, lvl, depth));
                } else if reg_acc.get(&rid).is_some_and(|a| !a.seam_cells.is_empty()) {
                    seam_pixel_map.entry(rid).or_default().push(SeamPixel {
                        gidx,
                        local_did: did_u,
                        level: lvl as u32,
                        depth,
                    });
                }
            }
        }

        // Acquire lock ONLY to allocate sequential temporary IDs and append encoded per-band
        // interior pixel records to the disk-backed row-band spool.
        if !interior_rids.is_empty()
            && let Some(mtx) = spool
        {
            let mut local_did_to_temp: HashMap<u32, u32> =
                HashMap::with_capacity(interior_did_set.len());
            {
                let mut sp = mtx
                    .lock()
                    .map_err(|_| Error::Algorithm("row-band spool mutex poisoned".into()))?;
                for rid in interior_rids {
                    if let Some(rdeps) = by_reg.get(&rid) {
                        let mut pairs = Vec::with_capacity(rdeps.len());
                        for d in rdeps {
                            let tid = sp.next_temp_id;
                            sp.next_temp_id = sp.next_temp_id.checked_add(1).ok_or_else(|| {
                                Error::Algorithm("exceeded maximum temporary depression IDs".into())
                            })?;
                            local_did_to_temp.insert(d.id, tid);
                            pairs.push((d.id, tid));
                        }
                        interior_temp_map.insert(rid, pairs);
                    }
                }
            }
            // Encode per-band binary records completely outside the lock.
            let mut per_band: HashMap<usize, Vec<u8>> = HashMap::new();
            for (gidx, did_u, lvl, depth) in interior_writes {
                if let Some(&tid) = local_did_to_temp.get(&did_u) {
                    let gr = gidx / full_w;
                    let gc = gidx % full_w;
                    let band = gr / STREAM_BAND_ROWS;
                    let local_row = (gr % STREAM_BAND_ROWS) as u16;
                    let buf = per_band.entry(band).or_default();
                    encode_spool_record(buf, local_row, gc as u32, tid, lvl, depth);
                }
            }
            if !per_band.is_empty() {
                let mut sp = mtx
                    .lock()
                    .map_err(|_| Error::Algorithm("row-band spool mutex poisoned".into()))?;
                for (band, bytes) in per_band {
                    sp.append_band_bytes(band, &bytes)?;
                }
            }
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
            seam_pixels: seam_pixel_map.remove(&rid).unwrap_or_default(),
            direct_temp_ids: interior_temp_map.remove(&rid).unwrap_or_default(),
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

fn merge_and_export_tiled(
    per_tile: Vec<Vec<TileRegionOutput>>,
    num_tiles: usize,
    meta: &RasterMeta,
    out_dir: &Path,
    outputs: &TiledOutputOptions,
    spool: Option<RowBandSpool>,
) -> Result<TiledDelineationResult> {
    let mut kept_regions: Vec<TileRegionOutput> = Vec::new();
    let mut seam_regions: Vec<TileRegionOutput> = Vec::new();

    for tile_regions in per_tile {
        for reg in tile_regions {
            if reg.seam_cells.is_empty() {
                if reg.intersects_core {
                    kept_regions.push(reg);
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

        // For each cluster root, retain ONLY the single best tile copy (preferring unclipped
        // copies, then largest pixel count) so exported polygon footprints strictly agree with
        // the winner's `count` and `area` attributes.
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
                kept_regions.push(reg);
            }
        }
    }

    kept_regions.sort_by_key(|r| (r.min_cell, r.total_cells));

    // Reassign globally unique sequential IDs, parent_ids, children_ids, and region_ids.
    let mut spool_opt = spool;
    let mut temp_to_global: Vec<i32> = spool_opt
        .as_ref()
        .map_or_else(Vec::new, |sp| vec![0i32; sp.next_temp_id as usize]);
    let num_bands = meta.height.div_ceil(STREAM_BAND_ROWS);
    // Bucket winning seam pixels by row band `(local_row, col, gid, lvl, depth)` so they
    // stream directly alongside spooled interior pixels without full-DEM RAM buffers.
    type BandSeamRec = (u16, u32, i32, i32, f64);
    let mut seam_by_band: Vec<Vec<BandSeamRec>> = if spool_opt.is_some() {
        vec![Vec::new(); num_bands]
    } else {
        Vec::new()
    };

    let mut merged = Vec::new();
    let mut next_dep_id = 1u32;
    let mut next_reg_id = 1u32;
    let mut max_level = 0u32;

    for reg in kept_regions {
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

        if spool_opt.is_some() {
            // Record remap entries for interior regions already spooled to disk.
            for &(local_did, temp_id) in &reg.direct_temp_ids {
                if let Some(&gid) = id_map.get(&local_did) {
                    temp_to_global[temp_id as usize] = gid as i32;
                }
            }
            // Bucket winning seam-crossing region pixels by output row band.
            for p in &reg.seam_pixels {
                if let Some(&gid) = id_map.get(&p.local_did) {
                    let gr = p.gidx / meta.width;
                    let gc = p.gidx % meta.width;
                    let band = gr / STREAM_BAND_ROWS;
                    let local_row = (gr % STREAM_BAND_ROWS) as u16;
                    seam_by_band[band].push((
                        local_row,
                        gc as u32,
                        gid as i32,
                        p.level as i32,
                        p.depth,
                    ));
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

    if let Some(mut spool) = spool_opt.take() {
        spool.flush_all()?;
        let width = spool.width;
        let height = spool.height;
        let write_opts = RasterWriteOptions {
            compression: outputs.compress.then(|| "DEFLATE".to_string()),
            ..Default::default()
        };

        let mut id_writer = if outputs.write_rasters {
            let p = out_dir.join("depression_id.tif");
            let p_str = p.to_str().ok_or_else(|| {
                Error::InvalidArgument(format!("non-UTF-8 path: {}", p.display()))
            })?;
            let header = build_streaming_header(meta, DataType::I32, 0.0)?;
            let w = GeoTiffWriter::create(p_str, &header, &write_opts)?;
            depression_id_tif = Some(p);
            Some(w)
        } else {
            None
        };
        let mut lvl_writer = if outputs.write_rasters {
            let p = out_dir.join("depression_level.tif");
            let p_str = p.to_str().ok_or_else(|| {
                Error::InvalidArgument(format!("non-UTF-8 path: {}", p.display()))
            })?;
            let header = build_streaming_header(meta, DataType::I32, 0.0)?;
            let w = GeoTiffWriter::create(p_str, &header, &write_opts)?;
            depression_level_tif = Some(p);
            Some(w)
        } else {
            None
        };
        let mut depth_writer = if outputs.write_rasters {
            let p = out_dir.join("depth.tif");
            let p_str = p.to_str().ok_or_else(|| {
                Error::InvalidArgument(format!("non-UTF-8 path: {}", p.display()))
            })?;
            let header = build_streaming_header(meta, DataType::F64, 0.0)?;
            let w = GeoTiffWriter::create(p_str, &header, &write_opts)?;
            depth_tif = Some(p);
            Some(w)
        } else {
            None
        };

        // Only allocate full-extent `id` vector when GeoPackage polygonization is requested;
        // `lvl` and `depth` are always streamed band-by-band without full-DEM RAM buffers.
        let mut full_id_for_gpkg: Option<Vec<i32>> = if outputs.write_gpkg {
            let n_pix = width.checked_mul(height).ok_or_else(|| {
                Error::DimensionMismatch(format!("raster dimensions overflow ({width}x{height})"))
            })?;
            Some(vec![0i32; n_pix])
        } else {
            None
        };

        let mut id_band: Vec<i32> = Vec::new();
        let mut lvl_band: Vec<i32> = Vec::new();
        let mut depth_band: Vec<f64> = Vec::new();
        let mut read_buf = [0u8; SPOOL_REC_BYTES * 1024];

        for (band, seam_recs) in seam_by_band.iter().enumerate().take(spool.num_bands) {
            let r0 = band * STREAM_BAND_ROWS;
            let r1 = (r0 + STREAM_BAND_ROWS).min(height);
            let band_h = r1 - r0;
            let band_pix = band_h * width;

            id_band.clear();
            id_band.resize(band_pix, 0);
            if outputs.write_rasters {
                lvl_band.clear();
                lvl_band.resize(band_pix, 0);
                depth_band.clear();
                depth_band.resize(band_pix, 0.0);
            }

            // 1. Replay spooled interior pixels for this row band.
            let bpath = spool.band_path(band);
            if bpath.exists() {
                let mut reader = BufReader::new(File::open(&bpath)?);
                let mut leftover = 0usize;
                loop {
                    let nread = reader.read(&mut read_buf[leftover..])?;
                    let total = leftover + nread;
                    if total == 0 {
                        break;
                    }
                    let complete = (total / SPOOL_REC_BYTES) * SPOOL_REC_BYTES;
                    for rec in read_buf[..complete].as_chunks::<SPOOL_REC_BYTES>().0 {
                        let local_row = u16::from_le_bytes([rec[0], rec[1]]) as usize;
                        let col = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
                        let tid = u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]) as usize;
                        let lvl = i32::from_le_bytes([rec[12], rec[13], rec[14], rec[15]]);
                        let depth = f64::from_le_bytes([
                            rec[16], rec[17], rec[18], rec[19], rec[20], rec[21], rec[22], rec[23],
                        ]);
                        let gid = temp_to_global.get(tid).copied().unwrap_or(0);
                        if gid > 0 && local_row < band_h && col < width {
                            let idx = local_row * width + col;
                            id_band[idx] = gid;
                            if outputs.write_rasters {
                                lvl_band[idx] = lvl;
                                depth_band[idx] = depth;
                            }
                        }
                    }
                    leftover = total - complete;
                    if leftover > 0 {
                        read_buf.copy_within(complete..total, 0);
                    }
                    if nread == 0 {
                        break;
                    }
                }
            }

            // 2. Paint winning seam-crossing pixels for this row band.
            for &(local_row, col, gid, lvl, depth) in seam_recs {
                let lr = local_row as usize;
                let c = col as usize;
                if lr < band_h && c < width {
                    let idx = lr * width + c;
                    id_band[idx] = gid;
                    if outputs.write_rasters {
                        lvl_band[idx] = lvl;
                        depth_band[idx] = depth;
                    }
                }
            }

            // 3. Copy band into `full_id_for_gpkg` if GPKG polygonization was requested.
            if let Some(ref mut full_id) = full_id_for_gpkg {
                let dst_start = r0 * width;
                full_id[dst_start..dst_start + band_pix].copy_from_slice(&id_band);
            }

            // 4. Stream row band directly to GeoTiffWriter(s).
            if let Some(ref mut w) = id_writer {
                let slab = std::mem::take(&mut id_band);
                let rd = RasterData::I32(slab);
                w.write_rows(&rd, band_h, None)?;
                if let RasterData::I32(v) = rd {
                    id_band = v;
                }
            }
            if let Some(ref mut w) = lvl_writer {
                let slab = std::mem::take(&mut lvl_band);
                let rd = RasterData::I32(slab);
                w.write_rows(&rd, band_h, None)?;
                if let RasterData::I32(v) = rd {
                    lvl_band = v;
                }
            }
            if let Some(ref mut w) = depth_writer {
                let slab = std::mem::take(&mut depth_band);
                let rd = RasterData::F64(slab);
                w.write_rows(&rd, band_h, None)?;
                if let RasterData::F64(v) = rd {
                    depth_band = v;
                }
            }
        }

        if let Some(w) = id_writer {
            w.finish()?;
        }
        if let Some(w) = lvl_writer {
            w.finish()?;
        }
        if let Some(w) = depth_writer {
            w.finish()?;
        }

        if let Some(full_id) = full_id_for_gpkg {
            let mut id_arr = Array2D::from_vec(width, height, full_id)?;
            id_arr.geotransform = meta.geotransform;
            id_arr.projection = meta.projection.clone();
            id_arr.set_no_data(0);
            let gp = out_dir.join("depressions.gpkg");
            write_depressions_vector_with_enclosing(&id_arr, &merged, &gp, true)?;
            gpkg_path = Some(gp);
        }
        // `spool` drops here and cleans up its temporary band directory.
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

fn report_tile_completion(
    progress_mtx: &Mutex<usize>,
    num_tiles: usize,
    control: &ExecutionControl,
) {
    // Hold the progress mutex across increment + callback invocation so parallel tile
    // workers deliver strictly monotonic completion fractions (1/N, 2/N, ..., N/N).
    // Callbacks are expected to be fast and non-reentrant with respect to tiling.
    let mut completed = progress_mtx
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *completed += 1;
    let done = *completed;
    let frac = 0.05 + 0.85 * (done as f64 / num_tiles.max(1) as f64);
    control.report(
        "tile",
        frac,
        Some(format!("Completed tile {done}/{num_tiles}")),
    );
}

fn run_tiled_in_memory<T: DemNum>(
    dem: &Array2D<T>,
    out_dir: &Path,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
    control: &ExecutionControl,
) -> Result<TiledDelineationResult> {
    #[cfg(test)]
    EAGER_FALLBACK_READS.fetch_add(1, Ordering::Relaxed);

    control.check_cancelled()?;

    let meta = RasterMeta {
        width: dem.width(),
        height: dem.height(),
        geotransform: dem.geotransform,
        projection: dem.projection.clone(),
    };
    meta.width.checked_mul(meta.height).ok_or_else(|| {
        Error::DimensionMismatch(format!(
            "raster dimensions overflow ({}x{})",
            meta.width, meta.height
        ))
    })?;
    let need_spatial = outputs.write_rasters || outputs.write_gpkg;
    let spool_mtx = if need_spatial {
        Some(Mutex::new(RowBandSpool::new(
            out_dir,
            meta.width,
            meta.height,
        )?))
    } else {
        None
    };

    let windows = build_tile_windows(meta.width, meta.height, options.tile_size, options.overlap);
    let num_tiles = windows.len();
    control.report(
        "prepare",
        0.05,
        Some(format!("Scheduled {num_tiles} tile(s)")),
    );
    let progress_mtx = Mutex::new(0usize);
    let run_one = |win: TileWindow| -> Result<Vec<TileRegionOutput>> {
        control.check_cancelled()?;
        let tile = extract_tile_f64(dem, win)?;
        let res = process_one_tile(
            tile,
            win,
            meta.width,
            meta.height,
            options,
            outputs,
            spool_mtx.as_ref(),
        )?;
        control.check_cancelled()?;
        report_tile_completion(&progress_mtx, num_tiles, control);
        Ok(res)
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

    control.check_cancelled()?;
    control.report(
        "merge",
        0.95,
        Some(format!("Merging {num_tiles} tile(s) and writing outputs")),
    );
    control.check_cancelled()?;

    let spool = match spool_mtx {
        Some(m) => Some(
            m.into_inner()
                .map_err(|_| Error::Algorithm("row-band spool mutex poisoned".into()))?,
        ),
        None => None,
    };
    let res = merge_and_export_tiled(per_tile, num_tiles, &meta, out_dir, outputs, spool)?;
    control.report("complete", 1.0, Some("Tiled delineation complete".into()));
    Ok(res)
}

fn validate_tiled_options(
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
) -> Result<()> {
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
    if let Some(ref breach_opts) = outputs.breach {
        validate_breach_options(breach_opts)?;
    }
    Ok(())
}

/// Returns `true` for GeoTIFF / COG extensions where `terrio::open_raster_lazy` failures
/// should be propagated explicitly instead of silently falling back to full-raster eager loading.
fn is_geotiff_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .is_some_and(|ext| matches!(ext.as_str(), "tif" | "tiff" | "gtiff" | "cog"))
}

/// Runs tiled nested depression delineation on a DEM file with windowed out-of-core
/// reading (`O(tile_size^2)` peak DEM memory per worker tile) and overlapping halos.
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

/// Runs windowed tiled nested depression delineation with optional per-tile breaching
/// and optional full-DEM stitched spatial outputs (`depression_id.tif`,
/// `depression_level.tif`, `depth.tif`, and `depressions.gpkg`).
///
/// Peak DEM memory is `O(tile_size^2)` per worker tile when `write_rasters` and
/// `write_gpkg` are both `false`. Enabling full-extent stitched outputs allocates
/// full-raster stitched buffers (`16` bytes/pixel for rasters, `4` bytes/pixel for
/// GPKG-only) to write the single mosaic file.
pub fn delineate_depressions_tiled_with_outputs(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
) -> Result<TiledDelineationResult> {
    delineate_depressions_tiled_with_control(
        in_dem,
        out_dir,
        options,
        outputs,
        &ExecutionControl::default(),
    )
}

/// Runs [`delineate_depressions_tiled_with_outputs`] with progress reporting and
/// cooperative cancellation via [`ExecutionControl`].
///
/// Reports monotonic tile completion progress across parallel workers and checks
/// `control.check_cancelled()` before each tile and before final merging/export.
pub fn delineate_depressions_tiled_with_control(
    in_dem: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &TiledDelineationOptions,
    outputs: &TiledOutputOptions,
    control: &ExecutionControl,
) -> Result<TiledDelineationResult> {
    validate_tiled_options(options, outputs)?;
    control.check_cancelled()?;
    let in_dem = in_dem.as_ref();
    let out_dir = out_dir.as_ref();

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
    control.check_cancelled()?;
    std::fs::create_dir_all(out_dir)?;
    // Recheck collisions after `out_dir` exists so non-canonical paths like
    // `<new_dir>/../dem.tif` resolve via canonicalization before any writes.
    crate::raster_ops::ensure_no_path_collision(in_dem, &out_paths)?;

    // Out-of-core windowed path via `terrio::open_raster_lazy`:
    // - If `open_raster_lazy` succeeds AND `lazy.reads_by_window()` is true, read each padded
    //   tile on demand via `read_window` (`O(tile_size^2)` DEM memory per worker).
    // - If `open_raster_lazy` succeeds AND `!lazy.reads_by_window()` (e.g., whole-file drivers
    //   such as `.xyz` or in-memory formats), fall through to one shared in-memory load rather
    //   than re-decoding the entire file once per tile.
    // - If `open_raster_lazy` fails on a GeoTIFF/COG extension, propagate the error explicitly
    //   instead of silently loading a large GeoTIFF into RAM; fall back cleanly for other formats.
    if let Some(p_str) = in_dem.to_str() {
        match terrio::open_raster_lazy(p_str) {
            #[allow(unused_mut)]
            Ok(mut lazy) if lazy.reads_by_window() => {
                let width = lazy.width();
                let height = lazy.height();
                let crs = lazy.crs();
                let projection = crs.to_wkt().or_else(|| crs.to_proj4()).unwrap_or_default();
                let meta = RasterMeta {
                    width,
                    height,
                    geotransform: lazy.geo_transform(),
                    projection,
                };
                #[cfg(feature = "parallel")]
                drop(lazy);

                meta.width.checked_mul(meta.height).ok_or_else(|| {
                    Error::DimensionMismatch(format!(
                        "raster dimensions overflow ({}x{})",
                        meta.width, meta.height
                    ))
                })?;
                let need_spatial = outputs.write_rasters || outputs.write_gpkg;
                let spool_mtx = if need_spatial {
                    Some(Mutex::new(RowBandSpool::new(
                        out_dir,
                        meta.width,
                        meta.height,
                    )?))
                } else {
                    None
                };

                let windows =
                    build_tile_windows(meta.width, meta.height, options.tile_size, options.overlap);
                let num_tiles = windows.len();
                control.report(
                    "prepare",
                    0.05,
                    Some(format!("Scheduled {num_tiles} tile(s)")),
                );
                let progress_mtx = Mutex::new(0usize);

                #[cfg(feature = "parallel")]
                let per_tile: Vec<Vec<TileRegionOutput>> = windows
                    .into_par_iter()
                    .map_init(
                        || terrio::open_raster_lazy(p_str).map_err(|e| e.to_string()),
                        |lazy_res, win| -> Result<Vec<TileRegionOutput>> {
                            control.check_cancelled()?;
                            let lazy = match lazy_res {
                                Ok(l) => l,
                                Err(msg) => {
                                    return Err(Error::Algorithm(format!(
                                        "failed to open lazy raster '{p_str}': {msg}"
                                    )));
                                }
                            };
                            let tile = read_tile_window_with_lazy(lazy, p_str, &meta, win)?;
                            let res = process_one_tile(
                                tile,
                                win,
                                meta.width,
                                meta.height,
                                options,
                                outputs,
                                spool_mtx.as_ref(),
                            )?;
                            control.check_cancelled()?;
                            report_tile_completion(&progress_mtx, num_tiles, control);
                            Ok(res)
                        },
                    )
                    .collect::<Result<Vec<_>>>()?;

                #[cfg(not(feature = "parallel"))]
                let per_tile: Vec<Vec<TileRegionOutput>> = windows
                    .into_iter()
                    .map(|win| -> Result<Vec<TileRegionOutput>> {
                        control.check_cancelled()?;
                        let tile = read_tile_window_with_lazy(&mut lazy, p_str, &meta, win)?;
                        let res = process_one_tile(
                            tile,
                            win,
                            meta.width,
                            meta.height,
                            options,
                            outputs,
                            spool_mtx.as_ref(),
                        )?;
                        control.check_cancelled()?;
                        report_tile_completion(&progress_mtx, num_tiles, control);
                        Ok(res)
                    })
                    .collect::<Result<Vec<_>>>()?;

                control.check_cancelled()?;
                control.report(
                    "merge",
                    0.95,
                    Some(format!("Merging {num_tiles} tile(s) and writing outputs")),
                );
                control.check_cancelled()?;

                let spool =
                    match spool_mtx {
                        Some(m) => Some(m.into_inner().map_err(|_| {
                            Error::Algorithm("row-band spool mutex poisoned".into())
                        })?),
                        None => None,
                    };
                let res =
                    merge_and_export_tiled(per_tile, num_tiles, &meta, out_dir, outputs, spool)?;
                control.report("complete", 1.0, Some("Tiled delineation complete".into()));
                return Ok(res);
            }
            Ok(_) => {
                // Driver does not read by window; fall through to one shared in-memory load.
            }
            Err(err) if is_geotiff_extension(in_dem) => {
                return Err(err.into());
            }
            Err(_) => {
                // Non-GeoTIFF format not supported by lazy open; fall back to `load_raster`.
            }
        }
    }

    // Fallback for legacy non-lazy formats (e.g., ESRI ASCII grids or non-UTF-8 paths).
    control.check_cancelled()?;
    if options.use_f32 {
        let dem = load_raster::<f32>(in_dem, None)?;
        run_tiled_in_memory(&dem, out_dir, options, outputs, control)
    } else {
        let dem = load_raster::<f64>(in_dem, None)?;
        run_tiled_in_memory(&dem, out_dir, options, outputs, control)
    }
}

/// Loads a DEM raster using `f32` storage internally (halving RAM footprint on large
/// float32 3DEP rasters) and converts on demand.
pub fn load_dem_f32(path: impl AsRef<Path>) -> Result<Array2D<f32>> {
    Ok(load_raster::<f32>(path, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize tests that inspect the global lazy/eager read counters.
    static COUNTER_LOCK: StdMutex<()> = StdMutex::new(());

    #[test]
    fn tiled_geotiff_uses_lazy_window_reads_without_eager_fallback() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (36usize, 36usize);
        let mut z = vec![20.0f64; w * h];
        for r in 14..22 {
            for c in 14..22 {
                z[r * w + c] = 15.0 + ((r as i32 - 18).abs() + (c as i32 - 18).abs()) as f64 * 0.3;
            }
        }
        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [100.0, 2.0, 0.0, 500.0, 0.0, -2.0];

        let tmp = tempfile::tempdir().unwrap();
        let tif = tmp.path().join("lazy_test.tif");
        save_raster(&dem, &tif, None, false).unwrap();

        LAZY_WINDOW_READS.store(0, Ordering::SeqCst);
        EAGER_FALLBACK_READS.store(0, Ordering::SeqCst);

        let opts = TiledDelineationOptions {
            tile_size: 18,
            overlap: 6,
            kernel_size: 1,
            min_size: 5,
            min_depth: 0.2,
            interval: 0.5,
            topology: Topology::D8,
            use_f32: false,
        };
        let res = delineate_depressions_tiled_with_outputs(
            &tif,
            tmp.path().join("out"),
            &opts,
            &TiledOutputOptions::default(),
        )
        .unwrap();

        assert_eq!(res.num_tiles, 4);
        assert_eq!(
            LAZY_WINDOW_READS.load(Ordering::SeqCst),
            4,
            "expected exactly 4 lazy window reads for 2x2 tiles"
        );
        assert_eq!(
            EAGER_FALLBACK_READS.load(Ordering::SeqCst),
            0,
            "eager fallback must NOT run for GeoTIFF inputs"
        );
    }

    #[test]
    fn read_tile_window_lazy_matches_full_raster_slice_with_nodata() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (36usize, 36usize);
        let mut z = vec![30.0f64; w * h];
        // Place a real -9999 NoData hole straddling the tile seam (rows 16..20, cols 16..20)
        // plus a depression at rows 4..10, cols 4..10 and a seam-crossing basin at rows 22..28, cols 14..22.
        for r in 16..20 {
            for c in 16..20 {
                z[r * w + c] = -9999.0;
            }
        }
        for r in 4..10 {
            for c in 4..10 {
                z[r * w + c] = 22.0 + ((r as i32 - 7).abs() + (c as i32 - 7).abs()) as f64 * 0.5;
            }
        }
        for r in 22..28 {
            for c in 14..22 {
                z[r * w + c] = 20.0 + ((r as i32 - 25).abs() + (c as i32 - 18).abs()) as f64 * 0.4;
            }
        }
        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [200.0, 2.5, 0.0, 600.0, 0.0, -2.5];

        let tmp = tempfile::tempdir().unwrap();
        let tif = tmp.path().join("nodata_seam.tif");
        save_raster(&dem, &tif, None, false).unwrap();

        let meta = RasterMeta {
            width: w,
            height: h,
            geotransform: dem.geotransform,
            projection: dem.projection.clone(),
        };
        let win = TileWindow {
            core_r0: 18,
            core_c0: 18,
            core_r1: 36,
            core_c1: 36,
            pad_r0: 12,
            pad_c0: 12,
            pad_r1: 36,
            pad_c1: 36,
        };
        let mut lazy = terrio::open_raster_lazy(tif.to_str().unwrap()).unwrap();
        let lazy_tile =
            read_tile_window_with_lazy(&mut lazy, tif.to_str().unwrap(), &meta, win).unwrap();
        let mem_tile = extract_tile_f64(&dem, win).unwrap();
        assert_eq!(
            (lazy_tile.width(), lazy_tile.height()),
            (mem_tile.width(), mem_tile.height())
        );
        assert_eq!(lazy_tile.geotransform, mem_tile.geotransform);
        assert_eq!(lazy_tile.no_data(), -9999.0);
        assert_eq!(lazy_tile.data(), mem_tile.data());

        // Compare full tiled lazy run vs in-memory tiled run on the NoData-seam DEM!
        let opts = TiledDelineationOptions {
            tile_size: 18,
            overlap: 8,
            kernel_size: 1,
            min_size: 8,
            min_depth: 0.3,
            interval: 0.5,
            topology: Topology::D8,
            use_f32: false,
        };
        let out_opts = TiledOutputOptions {
            write_rasters: true,
            write_gpkg: false,
            breach: None,
            compress: false,
        };
        let lazy_res = delineate_depressions_tiled_with_outputs(
            &tif,
            tmp.path().join("lazy_out"),
            &opts,
            &out_opts,
        )
        .unwrap();
        let mem_res = run_tiled_in_memory(
            &dem,
            &tmp.path().join("mem_out"),
            &opts,
            &out_opts,
            &ExecutionControl::default(),
        )
        .unwrap();

        assert_eq!(lazy_res.depressions.len(), mem_res.depressions.len());
        assert_eq!(lazy_res.depressions.len(), 2);
        let lazy_id =
            load_raster::<i32>(lazy_res.depression_id_tif.as_ref().unwrap(), None).unwrap();
        let mem_id = load_raster::<i32>(mem_res.depression_id_tif.as_ref().unwrap(), None).unwrap();
        assert_eq!(lazy_id.data(), mem_id.data());
        // NoData hole pixels must never be classified as depressions.
        for r in 16..20 {
            for c in 16..20 {
                assert_eq!(lazy_id.get(c, r), 0);
            }
        }
    }

    #[test]
    fn tiled_rejects_invalid_breach_options_before_reading_tiles() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let nonexistent = tmp.path().join("never_opened.tif");

        LAZY_WINDOW_READS.store(0, Ordering::SeqCst);
        let bad_neg = TiledOutputOptions {
            breach: Some(BreachOptions {
                max_depth: Some(-1.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = delineate_depressions_tiled_with_outputs(
            &nonexistent,
            tmp.path().join("out1"),
            &TiledDelineationOptions::default(),
            &bad_neg,
        );
        assert!(matches!(err, Err(Error::InvalidArgument(_))));
        assert_eq!(LAZY_WINDOW_READS.load(Ordering::SeqCst), 0);

        let bad_nan = TiledOutputOptions {
            breach: Some(BreachOptions {
                max_depth: Some(f64::NAN),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err_nan = delineate_depressions_tiled_with_outputs(
            &nonexistent,
            tmp.path().join("out2"),
            &TiledDelineationOptions::default(),
            &bad_nan,
        );
        assert!(matches!(err_nan, Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn tiled_with_control_reports_monotonic_tiles_and_honors_cancellation() {
        use std::sync::Arc;

        use crate::progress::{CancellationToken, ProgressEvent};

        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (64usize, 64usize);
        let mut z = vec![20.0f64; w * h];
        for r in 10..20 {
            for c in 10..20 {
                z[r * w + c] = 15.0 + ((r as i32 - 15).abs() + (c as i32 - 15).abs()) as f64 * 0.3;
            }
        }
        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [100.0, 2.0, 0.0, 500.0, 0.0, -2.0];

        let tmp = tempfile::tempdir().unwrap();
        let tif = tmp.path().join("tiled_progress.tif");
        save_raster(&dem, &tif, None, false).unwrap();

        let opts = TiledDelineationOptions {
            tile_size: 16,
            overlap: 4,
            kernel_size: 1,
            min_size: 5,
            min_depth: 0.2,
            interval: 0.5,
            topology: Topology::D8,
            use_f32: false,
        };

        // 1. Monotonic tile progress across 16 tiles (4x4).
        let events: Arc<StdMutex<Vec<ProgressEvent>>> = Arc::new(StdMutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let ctrl = ExecutionControl::new().with_progress_fn(move |ev| {
            captured.lock().unwrap().push(ev.clone());
        });
        let res = delineate_depressions_tiled_with_control(
            &tif,
            tmp.path().join("ok"),
            &opts,
            &TiledOutputOptions::default(),
            &ctrl,
        )
        .unwrap();
        assert_eq!(res.num_tiles, 16);

        let recorded = events.lock().unwrap();
        let tile_events: Vec<&ProgressEvent> =
            recorded.iter().filter(|e| e.stage == "tile").collect();
        assert_eq!(tile_events.len(), 16);
        assert_eq!(recorded.first().unwrap().stage, "prepare");
        assert_eq!(recorded.last().unwrap().stage, "complete");
        assert_eq!(recorded.last().unwrap().fraction, 1.0);

        let mut prev = -1.0f64;
        for ev in recorded.iter() {
            assert!((0.0..=1.0).contains(&ev.fraction));
            assert!(
                ev.fraction >= prev,
                "non-monotonic fraction: {} < {prev}",
                ev.fraction
            );
            prev = ev.fraction;
        }

        // 2. Pre-cancelled token aborts cleanly before reading any tile.
        LAZY_WINDOW_READS.store(0, Ordering::SeqCst);
        let pre_tok = CancellationToken::new();
        pre_tok.cancel();
        let pre_out = tmp.path().join("pre_cancel");
        let pre_err = delineate_depressions_tiled_with_control(
            &tif,
            &pre_out,
            &opts,
            &TiledOutputOptions::default(),
            &ExecutionControl::new().with_cancel(pre_tok),
        );
        assert!(matches!(pre_err, Err(Error::Cancelled)));
        assert_eq!(LAZY_WINDOW_READS.load(Ordering::SeqCst), 0);
        assert!(!pre_out.exists());

        // 3. Mid-run cancellation after first tile aborts cleanly.
        let mid_tok = CancellationToken::new();
        let trigger = mid_tok.clone();
        let mid_ctrl = ExecutionControl::new()
            .with_cancel(mid_tok)
            .with_progress_fn(move |ev| {
                if ev.stage == "tile" {
                    trigger.cancel();
                }
            });
        let mid_out = tmp.path().join("mid_cancel");
        let mid_err = delineate_depressions_tiled_with_control(
            &tif,
            &mid_out,
            &opts,
            &TiledOutputOptions::default(),
            &mid_ctrl,
        );
        assert!(matches!(mid_err, Err(Error::Cancelled)));
        assert!(!mid_out.join("depressions_info.csv").exists());
    }
}
