//! Utilities for downloading authoritative open U.S. federal geospatial
//! datasets within a user-specified region of interest (zero API keys, zero GDAL
//! dependencies; gated behind the opt-in `download` feature using `ureq` + `rustls`
//! for HTTPS).
//!
//! Supported federal web services:
//! - **USGS WBD** (Watershed Boundary Dataset, HUC-2 through HUC-12):
//!   [`download_wbd`]
//! - **USGS 3DEP** (1-m, 1/3 arc-sec, and 1 arc-sec DEMs, plus Lidar Point Cloud):
//!   [`query_3dep_products`], [`download_3dep_dem`]
//! - **USGS NHD** (National Hydrography Dataset flowlines):
//!   [`download_nhd`]
//! - **USFWS NWI** (National Wetlands Inventory polygons):
//!   [`download_nwi`]
//! - **USGS National Transportation** (road and railway centerlines):
//!   [`download_roads`]
//! - **NOAA Atlas 14** (partial-duration-series precipitation depth frequency tables):
//!   [`fetch_noaa_atlas14`]
//! - **End-to-end bundle** (downloads all selected federal datasets, optionally mosaics/clips
//!   the 3DEP DEM and synthesizes road-stream culverts): [`download_federal_data`]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use terrio::crs::{SpatialRef, Transformer};
use terrio::ops::Resampling;
use terrio::ops::topology::make_valid as terrio_make_valid;
use terrio::ops::vector_clip::{ClipRegion, ClipTemplateOptions, clip as terrio_vector_clip};
use terrio::{VectorOpenOptions, VectorWriteOptions};

use crate::conditioning::{SynthesizeCulvertsOptions, synthesize_culverts};
use crate::error::{Error, Result};
use crate::raster_ops::{clip_image, mosaic_with_resampling, resolve_path_for_compare};

/// USGS Watershed Boundary Dataset (WBD) MapServer base URL.
pub const USGS_WBD_BASE_URL: &str =
    "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer";
/// USGS National Hydrography Dataset (NHD) flowlines (Layer 6) query endpoint.
pub const USGS_NHD_FLOWLINE_URL: &str =
    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query";
/// USFWS National Wetlands Inventory (NWI) MapServer (Layer 0) query endpoint.
pub const USFWS_NWI_URL: &str = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query";
/// USGS The National Map (TNM) Access products API endpoint.
pub const USGS_TNM_PRODUCTS_URL: &str = "https://tnmaccess.nationalmap.gov/api/v1/products";
/// USGS National Transportation MapServer base URL.
pub const USGS_TRANSPORTATION_BASE_URL: &str =
    "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer";
/// Default USGS National Transportation layer IDs (controlled-access, secondary, local,
/// ramp roadways: 29..=33; railways: 38).
///
/// Layer 38 (railways) is included by default because raised rail embankments also act as
/// artificial flow barriers in bare-earth LiDAR DEMs; override
/// [`DownloadOptions::transportation_layers`] to exclude railways when only road crossings are
/// desired.
pub const DEFAULT_TRANSPORTATION_LAYERS: &[u32] = &[29, 30, 31, 32, 33, 38];
/// NOAA Atlas 14 precipitation frequency query CGI endpoint.
pub const NOAA_ATLAS14_URL: &str = "https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/cgi_readH5.py";

const NOAA_DURATIONS: &[&str] = &[
    "5-min", "10-min", "15-min", "30-min", "60-min", "2-hr", "3-hr", "6-hr", "12-hr", "24-hr",
    "2-day", "3-day", "4-day", "7-day", "10-day", "20-day", "30-day", "45-day", "60-day",
];
const NOAA_ARIS: &[u32] = &[1, 2, 5, 10, 25, 50, 100, 200, 500, 1000];

/// Axis-aligned geographic bounding box in WGS 84 (`EPSG:4326`) decimal degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bbox {
    /// Western longitude bound in degrees (`-180.0..=180.0`).
    pub min_lon: f64,
    /// Southern latitude bound in degrees (`-90.0..=90.0`).
    pub min_lat: f64,
    /// Eastern longitude bound in degrees (`-180.0..=180.0`).
    pub max_lon: f64,
    /// Northern latitude bound in degrees (`-90.0..=90.0`).
    pub max_lat: f64,
}

impl Bbox {
    /// Creates and validates a WGS 84 bounding box.
    pub fn new(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Result<Self> {
        let bbox = Self {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        };
        bbox.validate()?;
        Ok(bbox)
    }

    /// Validates coordinate finiteness, geographic ranges, and ordering (`min < max`).
    pub fn validate(&self) -> Result<()> {
        if !self.min_lon.is_finite()
            || !self.min_lat.is_finite()
            || !self.max_lon.is_finite()
            || !self.max_lat.is_finite()
        {
            return Err(Error::InvalidArgument(
                "bounding box coordinates must be finite".into(),
            ));
        }
        if self.min_lon < -180.0 || self.max_lon > 180.0 || self.min_lon >= self.max_lon {
            return Err(Error::InvalidArgument(format!(
                "invalid longitude bounds [{}, {}]; expected -180 <= min_lon < max_lon <= 180",
                self.min_lon, self.max_lon
            )));
        }
        if self.min_lat < -90.0 || self.max_lat > 90.0 || self.min_lat >= self.max_lat {
            return Err(Error::InvalidArgument(format!(
                "invalid latitude bounds [{}, {}]; expected -90 <= min_lat < max_lat <= 90",
                self.min_lat, self.max_lat
            )));
        }
        Ok(())
    }

    /// Returns `(centroid_lon, centroid_lat)`.
    pub fn centroid(&self) -> (f64, f64) {
        (
            0.5 * (self.min_lon + self.max_lon),
            0.5 * (self.min_lat + self.max_lat),
        )
    }

    /// Returns `true` if `self` intersects `other` (including touching edges).
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_lon <= other.max_lon
            && self.max_lon >= other.min_lon
            && self.min_lat <= other.max_lat
            && self.max_lat >= other.min_lat
    }

    /// Formats as `"min_lon,min_lat,max_lon,max_lat"` for ArcGIS and USGS TNM APIs.
    pub fn to_csv_string(&self) -> String {
        format!(
            "{},{},{},{}",
            self.min_lon, self.min_lat, self.max_lon, self.max_lat
        )
    }
}

impl FromStr for Bbox {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(Error::InvalidArgument(format!(
                "expected 4 comma-separated WGS 84 bounds 'min_lon,min_lat,max_lon,max_lat', got '{s}'"
            )));
        }
        let nums: Vec<f64> = parts
            .iter()
            .map(|p| {
                p.parse::<f64>()
                    .map_err(|_| Error::InvalidArgument(format!("invalid bbox coordinate '{p}'")))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(nums[0], nums[1], nums[2], nums[3])
    }
}

/// Region of interest specification for federal data downloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Region {
    /// Explicit WGS 84 bounding box (`min_lon, min_lat, max_lon, max_lat`).
    Bbox(Bbox),
    /// USGS Watershed Boundary Dataset Hydrologic Unit Code (2, 4, 6, 8, 10, or 12 digits,
    /// e.g. `"10160002"` for a HUC-8 subbasin or `"101600020502"` for a single HUC-12).
    Huc(String),
    /// Local polygon vector file (`.geojson`, `.gpkg`, `.shp`) in any CRS supported by `terrio`.
    VectorFile(PathBuf),
}

impl Region {
    /// Validates a HUC code string (2, 4, 6, 8, 10, or 12 ASCII digits).
    pub fn validate_huc(huc: &str) -> Result<String> {
        let trimmed = huc.trim();
        let valid_len = matches!(trimmed.len(), 2 | 4 | 6 | 8 | 10 | 12);
        if !valid_len || !trimmed.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidArgument(format!(
                "invalid HUC code '{huc}'; expected 2, 4, 6, 8, 10, or 12 digits (e.g. '10160002')"
            )));
        }
        Ok(trimmed.to_string())
    }
}

/// USGS Watershed Boundary Dataset (WBD) hierarchy level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WbdLevel {
    /// HUC-2 Region (MapServer Layer 1).
    Huc2,
    /// HUC-4 Subregion (MapServer Layer 2).
    Huc4,
    /// HUC-6 Basin (MapServer Layer 3).
    Huc6,
    /// HUC-8 Subbasin (MapServer Layer 4).
    Huc8,
    /// HUC-10 Watershed (MapServer Layer 5).
    Huc10,
    /// HUC-12 Subwatershed (MapServer Layer 6).
    #[default]
    Huc12,
}

impl WbdLevel {
    /// USGS WBD MapServer layer index.
    pub fn layer_id(self) -> u32 {
        match self {
            Self::Huc2 => 1,
            Self::Huc4 => 2,
            Self::Huc6 => 3,
            Self::Huc8 => 4,
            Self::Huc10 => 5,
            Self::Huc12 => 6,
        }
    }

    /// Primary HUC attribute field name on USGS WBD MapServer.
    pub fn field_name(self) -> &'static str {
        match self {
            Self::Huc2 => "huc2",
            Self::Huc4 => "huc4",
            Self::Huc6 => "huc6",
            Self::Huc8 => "huc8",
            Self::Huc10 => "huc10",
            Self::Huc12 => "huc12",
        }
    }

    /// Digit length associated with this WBD level.
    pub fn digits(self) -> usize {
        match self {
            Self::Huc2 => 2,
            Self::Huc4 => 4,
            Self::Huc6 => 6,
            Self::Huc8 => 8,
            Self::Huc10 => 10,
            Self::Huc12 => 12,
        }
    }

    /// Parses numeric digits (`2`, `4`, `6`, `8`, `10`, `12`) into a [`WbdLevel`].
    pub fn from_digits(digits: usize) -> Result<Self> {
        match digits {
            2 => Ok(Self::Huc2),
            4 => Ok(Self::Huc4),
            6 => Ok(Self::Huc6),
            8 => Ok(Self::Huc8),
            10 => Ok(Self::Huc10),
            12 => Ok(Self::Huc12),
            other => Err(Error::InvalidArgument(format!(
                "invalid WBD HUC level {other}; expected 2, 4, 6, 8, 10, or 12"
            ))),
        }
    }

    /// Full ArcGIS REST query endpoint URL for this WBD level.
    pub fn query_url(self) -> String {
        format!("{USGS_WBD_BASE_URL}/{}/query", self.layer_id())
    }
}

impl FromStr for WbdLevel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let norm = s.trim().to_ascii_lowercase();
        let digits = norm
            .strip_prefix("huc")
            .unwrap_or(&norm)
            .trim_start_matches('-')
            .trim_start_matches('_');
        let n = digits.parse::<usize>().map_err(|_| {
            Error::InvalidArgument(format!(
                "invalid WBD level '{s}'; expected 2, 4, 6, 8, 10, or 12"
            ))
        })?;
        Self::from_digits(n)
    }
}

/// USGS 3D Elevation Program (3DEP) product selector for The National Map API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DemProduct {
    /// 1-meter bare-earth GeoTIFF DEM (`Digital Elevation Model (DEM) 1 meter`).
    #[default]
    Meter1,
    /// 1/3 arc-second (~10-m) GeoTIFF DEM (`National Elevation Dataset (NED) 1/3 arc-second`).
    ThirdArcSec,
    /// 1 arc-second (~30-m) GeoTIFF DEM (`National Elevation Dataset (NED) 1 arc-second`).
    OneArcSec,
    /// Raw airborne LiDAR Point Cloud (`Lidar Point Cloud (LPC)` in LAZ/LAS).
    LidarPointCloud,
}

impl DemProduct {
    /// Dataset name accepted by the USGS TNM Access `datasets` query parameter.
    pub fn tnm_dataset_name(self) -> &'static str {
        match self {
            Self::Meter1 => "Digital Elevation Model (DEM) 1 meter",
            Self::ThirdArcSec => "National Elevation Dataset (NED) 1/3 arc-second",
            Self::OneArcSec => "National Elevation Dataset (NED) 1 arc-second",
            Self::LidarPointCloud => "Lidar Point Cloud (LPC)",
        }
    }

    /// Optional `prodFormats` filter for USGS TNM Access API.
    pub fn prod_formats(self) -> Option<&'static str> {
        match self {
            Self::Meter1 | Self::ThirdArcSec | Self::OneArcSec => Some("GeoTIFF"),
            Self::LidarPointCloud => Some("LAZ,LAS"),
        }
    }

    /// Returns `true` for raster DEM products (eligible for mosaicking and clipping).
    pub fn is_raster(self) -> bool {
        !matches!(self, Self::LidarPointCloud)
    }
}

impl FromStr for DemProduct {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1m" | "meter1" | "1-m" | "1_m" => Ok(Self::Meter1),
            "10m" | "1/3" | "third_arc_sec" | "third-arc-sec" | "1_3_arc_sec" | "ned13" => {
                Ok(Self::ThirdArcSec)
            }
            "30m" | "1arcsec" | "one_arc_sec" | "one-arc-sec" | "ned1" => Ok(Self::OneArcSec),
            "lpc" | "lidar" | "pointcloud" | "laz" => Ok(Self::LidarPointCloud),
            other => Err(Error::InvalidArgument(format!(
                "unknown 3DEP product '{other}'; expected 1m, 10m, 30m, or lpc"
            ))),
        }
    }
}

/// Individual federal datasets available in [`download_federal_data`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FederalDataset {
    /// USGS Watershed Boundary Dataset polygons.
    Wbd,
    /// USGS 3DEP DEM (or LPC) tiles plus optional mosaic/clip.
    Dem,
    /// USGS NHD flowlines.
    Nhd,
    /// USFWS NWI wetland polygons.
    Nwi,
    /// USGS National Transportation roads and railways.
    Roads,
    /// Deterministic road- and rail-stream culverts synthesized from NHD x Transportation.
    ///
    /// Includes railway embankments (Layer 38) by default; omit Layer 38 from
    /// [`DownloadOptions::transportation_layers`] to synthesize road-only crossings.
    Culverts,
    /// NOAA Atlas 14 PDS precipitation depth frequency table at region centroid.
    NoaaAtlas14,
}

impl FederalDataset {
    /// Returns all seven datasets in dependency order.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Wbd,
            Self::Dem,
            Self::Nhd,
            Self::Nwi,
            Self::Roads,
            Self::Culverts,
            Self::NoaaAtlas14,
        ]
    }
}

impl FromStr for FederalDataset {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "wbd" | "huc" => Ok(Self::Wbd),
            "dem" | "3dep" | "elevation" => Ok(Self::Dem),
            "nhd" | "streams" | "flowlines" => Ok(Self::Nhd),
            "nwi" | "wetlands" => Ok(Self::Nwi),
            "roads" | "transportation" | "trans" => Ok(Self::Roads),
            "culverts" => Ok(Self::Culverts),
            "noaa" | "atlas14" | "noaa_atlas14" | "precip" => Ok(Self::NoaaAtlas14),
            other => Err(Error::InvalidArgument(format!(
                "unknown federal dataset '{other}'; expected wbd, dem, nhd, nwi, roads, culverts, or noaa"
            ))),
        }
    }
}

/// Parses a comma-separated list of dataset names (or `"all"`).
pub fn parse_dataset_list(spec: &str) -> Result<Vec<FederalDataset>> {
    let trimmed = spec.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok(FederalDataset::all());
    }
    let mut out = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let ds = p.parse::<FederalDataset>()?;
        if !out.contains(&ds) {
            out.push(ds);
        }
    }
    if out.is_empty() {
        return Err(Error::InvalidArgument(
            "dataset list must contain at least one dataset or 'all'".into(),
        ));
    }
    Ok(out)
}

/// Configuration for federal web-service queries, retries, caching, and post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadOptions {
    /// Per-request HTTP timeout in seconds (default `120`).
    pub timeout_secs: u64,
    /// Maximum retry attempts for transient network or 5xx errors (default `4`).
    pub max_retries: usize,
    /// ArcGIS ObjectID chunk size for paginated vector downloads (default `250`).
    pub oid_chunk_size: usize,
    /// Re-download files even if they already exist locally with matching size (default `false`).
    pub overwrite: bool,
    /// Optional shared cache directory for 3DEP tiles so tiles can be reused across regions.
    pub tile_cache_dir: Option<PathBuf>,
    /// USGS 3DEP product to query and download (default [`DemProduct::Meter1`]).
    pub dem_product: DemProduct,
    /// USGS WBD HUC hierarchy level to download (default [`WbdLevel::Huc12`]).
    pub wbd_level: WbdLevel,
    /// Deduplicate overlapping 3DEP grid tiles (`x\d+y\d+`) to newest `publicationDate` (default `true`).
    pub deduplicate_tiles: bool,
    /// Mosaic downloaded raster DEM tiles into a single `dem.tif` (default `true`).
    pub mosaic_dem: bool,
    /// Clip downloaded DEM mosaic and vector layers to the region boundary polygon when the
    /// region is specified via `Region::Huc` or `Region::VectorFile` (default `true`).
    pub clip_to_region: bool,
    /// Maximum TNM product records to scan across paginated pages (default `1000`).
    pub max_products: usize,
    /// Culvert synthesis half-length in meters on each side of crossing (default `25.0`).
    pub culvert_half_length_m: f64,
    /// Culvert synthesis deduplication snap distance in meters (default `15.0`).
    pub culvert_snap_distance_m: f64,
    /// USGS National Transportation MapServer layer IDs to query (default `[29, 30, 31, 32, 33, 38]`).
    pub transportation_layers: Vec<u32>,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            timeout_secs: 120,
            max_retries: 4,
            oid_chunk_size: 250,
            overwrite: false,
            tile_cache_dir: None,
            dem_product: DemProduct::Meter1,
            wbd_level: WbdLevel::Huc12,
            deduplicate_tiles: true,
            mosaic_dem: true,
            clip_to_region: true,
            max_products: 1000,
            culvert_half_length_m: 25.0,
            culvert_snap_distance_m: 15.0,
            transportation_layers: DEFAULT_TRANSPORTATION_LAYERS.to_vec(),
        }
    }
}

/// Options for [`download_federal_data`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederalBundleOptions {
    /// Federal datasets to download (defaults to all seven datasets).
    pub datasets: Vec<FederalDataset>,
    /// Shared download, caching, and processing options.
    pub download: DownloadOptions,
}

impl Default for FederalBundleOptions {
    fn default() -> Self {
        Self {
            datasets: FederalDataset::all(),
            download: DownloadOptions::default(),
        }
    }
}

/// Metadata for a single USGS The National Map (3DEP) product tile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TnmProductItem {
    /// Deduplication grid key (e.g. `"x47y523"` when present in title, else title).
    pub grid_key: String,
    /// Product title reported by USGS TNM.
    pub title: String,
    /// Publication date string (ISO `YYYY-MM-DD`, empty if unavailable).
    pub publication_date: String,
    /// Direct download URL.
    pub download_url: String,
    /// Expected file size in bytes if reported by USGS TNM.
    pub size_bytes: Option<u64>,
    /// File format string (e.g. `"GeoTIFF"`, `"LAZ"`).
    pub format: String,
    /// Product WGS 84 bounding box.
    pub bbox: Bbox,
}

/// Output summary from [`download_3dep_dem`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemDownloadResult {
    /// Resolved WGS 84 bounding box queried.
    pub bbox: Bbox,
    /// Product metadata items selected for download.
    pub products: Vec<TnmProductItem>,
    /// Paths to downloaded (or cached) tile files on disk.
    pub tile_paths: Vec<PathBuf>,
    /// Path to the mosaicked (and optionally clipped) GeoTIFF when `mosaic_dem` is enabled.
    pub mosaic_path: Option<PathBuf>,
}

/// Structured NOAA Atlas 14 Partial Duration Series (PDS) precipitation depth table (mm).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoaaAtlas14Table {
    /// Query latitude in decimal degrees (WGS 84).
    pub lat: f64,
    /// Query longitude in decimal degrees (WGS 84).
    pub lon: f64,
    /// Depth units (`"mm"`).
    pub units: String,
    /// Mapping `duration_label -> (ari_yr -> depth_mm)`.
    pub depths_mm: BTreeMap<String, BTreeMap<u32, f64>>,
}

impl NoaaAtlas14Table {
    /// Looks up the design storm precipitation depth in millimeters for a given duration
    /// (e.g. `"24-hr"`, `"60-min"`) and average recurrence interval in years (e.g. `2`, `10`, `25`, `100`).
    pub fn depth_mm(&self, duration: &str, ari_yr: u32) -> Option<f64> {
        self.depths_mm
            .get(duration)
            .and_then(|row| row.get(&ari_yr))
            .copied()
    }
}

/// Resolved region geometry and geographic extent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRegion {
    /// WGS 84 bounding box of the region.
    pub bbox: Bbox,
    /// Optional local vector file representing the exact polygon boundary (present for
    /// `Region::VectorFile` or fetched `Region::Huc`).
    pub boundary_path: Option<PathBuf>,
    /// Number of boundary polygons fetched (for `Region::Huc`).
    pub boundary_feature_count: usize,
}

/// Complete manifest returned by [`download_federal_data`] and written to `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederalBundleResult {
    /// Resolved WGS 84 bounding box of the region.
    pub region_bbox: Bbox,
    /// Canonical region identity (`"bbox"`, `"huc:<code>"`, or `"file:<path>"`) for cache checks.
    #[serde(default)]
    pub region_key: String,
    /// USGS 3DEP DEM product configured for this bundle.
    #[serde(default)]
    pub dem_product: DemProduct,
    /// USGS WBD hierarchy level configured for this bundle.
    #[serde(default)]
    pub wbd_level: WbdLevel,
    /// Whether clipping to the region boundary was enabled for this bundle.
    #[serde(default)]
    pub clip_to_region: bool,
    /// Whether DEM mosaicking was enabled for this bundle.
    #[serde(default)]
    pub mosaic_dem: bool,
    /// Effective USGS National Transportation layer IDs queried for this bundle.
    #[serde(default)]
    pub transportation_layers: Vec<u32>,
    /// Path to downloaded WBD vector file, if requested.
    pub wbd_path: Option<PathBuf>,
    /// Number of WBD polygon features downloaded.
    pub wbd_count: usize,
    /// Downloaded 3DEP tile paths.
    pub dem_tiles: Vec<PathBuf>,
    /// Path to mosaicked (and optionally clipped) 3DEP DEM GeoTIFF, if produced.
    pub dem_mosaic_path: Option<PathBuf>,
    /// Path to downloaded NHD flowlines vector file, if requested.
    pub nhd_path: Option<PathBuf>,
    /// Number of NHD flowlines downloaded.
    pub nhd_count: usize,
    /// Path to downloaded NWI wetland polygons vector file, if requested.
    pub nwi_path: Option<PathBuf>,
    /// Number of NWI polygons downloaded.
    pub nwi_count: usize,
    /// Path to downloaded USGS Transportation roads/railways vector file, if requested.
    pub roads_path: Option<PathBuf>,
    /// Number of road/railway segments downloaded.
    pub roads_count: usize,
    /// Path to synthesized culverts vector file, if requested.
    pub culverts_path: Option<PathBuf>,
    /// Number of synthesized culvert breach segments.
    pub culverts_count: usize,
    /// Path to saved NOAA Atlas 14 JSON file, if requested.
    pub noaa_atlas14_path: Option<PathBuf>,
    /// Parsed NOAA Atlas 14 precipitation frequency table, if requested.
    pub noaa_atlas14: Option<NoaaAtlas14Table>,
    /// Path to the written `manifest.json`.
    pub manifest_path: PathBuf,
}

fn make_agent(timeout_secs: u64) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(timeout_secs.max(5))))
        .user_agent(format!("lidar-rs/{}", crate::VERSION))
        .build();
    ureq::Agent::new_with_config(config)
}

/// Builds an HTTP agent configured for large streaming file downloads using per-phase timeouts
/// (connect, response headers, per-read, per-write) instead of a fixed total transfer timeout.
fn make_download_agent(timeout_secs: u64) -> ureq::Agent {
    let io_timeout = Duration::from_secs(timeout_secs.max(30));
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(io_timeout))
        .timeout_per_read(Some(io_timeout))
        .timeout_per_write(Some(Duration::from_secs(30)))
        .user_agent(format!("lidar-rs/{}", crate::VERSION))
        .build();
    ureq::Agent::new_with_config(config)
}

pub(crate) fn is_retryable_ureq_error(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::BodyExceedsLimit(_)
        | ureq::Error::BadUri(_)
        | ureq::Error::Http(_)
        | ureq::Error::InvalidProxyUrl
        | ureq::Error::TooManyRedirects
        | ureq::Error::RedirectFailed
        | ureq::Error::RequireHttpsOnly(_)
        | ureq::Error::LargeResponseHeader(_, _)
        | ureq::Error::Tls(_)
        | ureq::Error::Pem(_)
        | ureq::Error::Rustls(_) => false,
        ureq::Error::StatusCode(code) => *code == 408 || *code == 429 || *code >= 500,
        _ => true,
    }
}

fn retry_backoff_duration(base_ms: u64, attempt: usize) -> Duration {
    let linear_ms = base_ms.saturating_mul(attempt as u64 + 1);
    let max_jitter = (linear_ms / 4).max(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() as u64);
    let mixed = nanos
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((attempt as u64).wrapping_mul(1_442_695_040_888_963_407));
    Duration::from_millis(linear_ms.saturating_add(mixed % max_jitter))
}

fn write_file_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempFileGuard(Option<PathBuf>);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            if let Some(p) = self.0.take() {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    ensure_parent_dir(path)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    // Stage via OpenOptions::create_new instead of NamedTempFile so the written JSON/metadata
    // file honors the process umask (e.g. 0644) rather than forcing 0600 permissions.
    let pid = std::process::id();
    let mut opened = None;
    for _ in 0..1024 {
        let seq = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".lidar_tmp_{pid}_{seq}.part"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                opened = Some((candidate, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let (tmp_path, mut file) = opened.ok_or_else(|| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "failed to allocate unique temporary file for atomic write",
        ))
    })?;
    let mut guard = TempFileGuard(Some(tmp_path.clone()));
    file.write_all(contents)?;
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp_path, path)?;
    guard.0 = None;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn url_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

fn build_query_url(base: &str, params: &[(&str, String)]) -> String {
    if params.is_empty() {
        return base.to_string();
    }
    let qs: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode_component(k), url_encode_component(v)))
        .collect();
    if base.contains('?') {
        format!("{base}&{}", qs.join("&"))
    } else {
        format!("{base}?{}", qs.join("&"))
    }
}

/// Maximum response size accepted for JSON/GeoJSON/text federal API responses (256 MiB).
const MAX_TEXT_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;

fn http_get_text(
    agent: &ureq::Agent,
    url: &str,
    params: &[(&str, String)],
    max_retries: usize,
) -> Result<String> {
    let full_url = build_query_url(url, params);
    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        let res = agent.get(&full_url).call().and_then(|mut resp| {
            resp.body_mut()
                .with_config()
                .limit(MAX_TEXT_RESPONSE_BYTES)
                .read_to_string()
        });
        match res {
            Ok(text) => return Ok(text),
            Err(e) => {
                let retryable = is_retryable_ureq_error(&e);
                last_err = format!("{e}");
                if !retryable {
                    return Err(Error::Http(format!("GET {url} failed: {last_err}")));
                }
                if attempt < max_retries {
                    std::thread::sleep(retry_backoff_duration(1200, attempt));
                }
            }
        }
    }
    Err(Error::Http(format!(
        "GET {url} failed after {} attempts: {last_err}",
        max_retries + 1
    )))
}

fn http_post_form_text(
    agent: &ureq::Agent,
    url: &str,
    form: &[(&str, String)],
    max_retries: usize,
) -> Result<String> {
    let body = form
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode_component(k), url_encode_component(v)))
        .collect::<Vec<_>>()
        .join("&");
    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        let res = agent
            .post(url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send(body.as_bytes())
            .and_then(|mut resp| {
                resp.body_mut()
                    .with_config()
                    .limit(MAX_TEXT_RESPONSE_BYTES)
                    .read_to_string()
            });
        match res {
            Ok(text) => return Ok(text),
            Err(e) => {
                let retryable = is_retryable_ureq_error(&e);
                last_err = format!("{e}");
                if !retryable {
                    return Err(Error::Http(format!("POST {url} failed: {last_err}")));
                }
                if attempt < max_retries {
                    std::thread::sleep(retry_backoff_duration(1500, attempt));
                }
            }
        }
    }
    Err(Error::Http(format!(
        "POST {url} failed after {} attempts: {last_err}",
        max_retries + 1
    )))
}

fn http_download_file(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    expected_bytes: Option<u64>,
    overwrite: bool,
    max_retries: usize,
) -> Result<PathBuf> {
    let expected_bytes = expected_bytes.filter(|&n| n > 0);
    if !overwrite
        && dest.exists()
        && let Ok(meta) = std::fs::metadata(dest)
        && meta.is_file()
        && meta.len() > 0
        && expected_bytes.is_none_or(|exp| meta.len() == exp)
    {
        return Ok(dest.to_path_buf());
    }
    ensure_parent_dir(dest)?;
    let parent = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        let mut tmp_file = tempfile::Builder::new()
            .prefix(".lidar_dl_")
            .suffix(".part")
            .tempfile_in(parent)?;

        enum DownloadAttemptErr {
            Fatal(Error),
            Retryable(Error),
        }

        let res = (|| -> std::result::Result<u64, DownloadAttemptErr> {
            let mut resp = agent.get(url).call().map_err(|e| {
                let retryable = is_retryable_ureq_error(&e);
                let err = Error::Http(format!("GET {url}: {e}"));
                if retryable {
                    DownloadAttemptErr::Retryable(err)
                } else {
                    DownloadAttemptErr::Fatal(err)
                }
            })?;
            let has_content_encoding = resp
                .headers()
                .get("content-encoding")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| {
                    let trimmed = v.trim();
                    !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("identity")
                });
            let header_content_length = if expected_bytes.is_none() && !has_content_encoding {
                resp.headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .filter(|&n| n > 0)
            } else {
                None
            };
            let mut reader = resp.body_mut().as_reader();
            let mut buf = vec![0u8; 256 * 1024];
            let mut written: u64 = 0;
            loop {
                let n = reader
                    .read(&mut buf)
                    .map_err(|e| DownloadAttemptErr::Retryable(Error::Io(e)))?;
                if n == 0 {
                    break;
                }
                tmp_file
                    .as_file_mut()
                    .write_all(&buf[..n])
                    .map_err(|e| DownloadAttemptErr::Fatal(Error::Io(e)))?;
                written += n as u64;
            }
            tmp_file
                .as_file_mut()
                .flush()
                .map_err(|e| DownloadAttemptErr::Fatal(Error::Io(e)))?;
            if written == 0 {
                return Err(DownloadAttemptErr::Retryable(Error::Http(format!(
                    "GET {url} returned empty body"
                ))));
            }
            let verify_len = expected_bytes.or(header_content_length);
            if let Some(exp) = verify_len
                && written != exp
            {
                return Err(DownloadAttemptErr::Retryable(Error::Http(format!(
                    "GET {url} short read: expected {exp} bytes, got {written}"
                ))));
            }
            Ok(written)
        })();
        match res {
            Ok(_) => {
                tmp_file.persist(dest).map_err(|e| Error::Io(e.error))?;
                return Ok(dest.to_path_buf());
            }
            Err(DownloadAttemptErr::Fatal(e)) => {
                drop(tmp_file);
                return Err(e);
            }
            Err(DownloadAttemptErr::Retryable(e)) => {
                drop(tmp_file);
                last_err = e.to_string();
                if attempt < max_retries {
                    std::thread::sleep(retry_backoff_duration(2000, attempt));
                }
            }
        }
    }
    Err(Error::Http(format!(
        "download {url} failed after {} attempts: {last_err}",
        max_retries + 1
    )))
}

/// Computes the WGS 84 bounding box of a local vector file via `terrio`.
pub fn vector_file_bbox_wgs84(path: impl AsRef<Path>) -> Result<Bbox> {
    let p = path.as_ref();
    let p_str = p
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", p.display())))?;
    let ds = terrio::open_vector(p_str, &VectorOpenOptions::default())?;
    let wgs84 = SpatialRef::from_epsg(4326)?;

    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );

    for layer in &ds.layers {
        let src_srs = SpatialRef::from_layer(layer);
        let mut lmin_x = f64::INFINITY;
        let mut lmin_y = f64::INFINITY;
        let mut lmax_x = f64::NEG_INFINITY;
        let mut lmax_y = f64::NEG_INFINITY;
        for feat in &layer.features {
            if let Some(ref g) = feat.geometry
                && let Some((gx0, gy0, gx1, gy1)) = geometry_bounds(g)
            {
                lmin_x = lmin_x.min(gx0);
                lmin_y = lmin_y.min(gy0);
                lmax_x = lmax_x.max(gx1);
                lmax_y = lmax_y.max(gy1);
            }
        }
        if !lmin_x.is_finite() {
            continue;
        }
        let (bx0, by0, bx1, by1) = if !src_srs.is_unknown() && !src_srs.same_as(&wgs84) {
            let tx = Transformer::new(&src_srs, &wgs84)?;
            tx.transform_bounds_densified((lmin_x, lmin_y, lmax_x, lmax_y), 21)?
        } else {
            (lmin_x, lmin_y, lmax_x, lmax_y)
        };
        min_x = min_x.min(bx0);
        min_y = min_y.min(by0);
        max_x = max_x.max(bx1);
        max_y = max_y.max(by1);
    }

    if !min_x.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "vector file '{}' contains no valid geometries",
            p.display()
        )));
    }
    Bbox::new(min_x, min_y, max_x, max_y)
}

fn geometry_bounds(geom: &terrio::Geometry) -> Option<(f64, f64, f64, f64)> {
    let mut b = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    fn push_coord(b: &mut (f64, f64, f64, f64), c: &terrio::vector::Coord) {
        if c.x.is_finite() && c.y.is_finite() {
            b.0 = b.0.min(c.x);
            b.1 = b.1.min(c.y);
            b.2 = b.2.max(c.x);
            b.3 = b.3.max(c.y);
        }
    }
    fn walk(b: &mut (f64, f64, f64, f64), g: &terrio::Geometry) {
        match g {
            terrio::Geometry::Point(c) => push_coord(b, c),
            terrio::Geometry::MultiPoint(pts) | terrio::Geometry::LineString(pts) => {
                for c in pts {
                    push_coord(b, c);
                }
            }
            terrio::Geometry::MultiLineString(lines) => {
                for line in lines {
                    for c in line {
                        push_coord(b, c);
                    }
                }
            }
            terrio::Geometry::Polygon {
                exterior,
                interiors,
            } => {
                for c in exterior.coords() {
                    push_coord(b, c);
                }
                for ring in interiors {
                    for c in ring.coords() {
                        push_coord(b, c);
                    }
                }
            }
            terrio::Geometry::MultiPolygon(polys) => {
                for (ext, ints) in polys {
                    for c in ext.coords() {
                        push_coord(b, c);
                    }
                    for ring in ints {
                        for c in ring.coords() {
                            push_coord(b, c);
                        }
                    }
                }
            }
            terrio::Geometry::GeometryCollection(geoms) => {
                for child in geoms {
                    walk(b, child);
                }
            }
        }
    }
    walk(&mut b, geom);
    if b.0.is_finite() { Some(b) } else { None }
}

fn geojson_bbox(fc: &Value) -> Result<Bbox> {
    let mut b = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    fn walk_coords(v: &Value, b: &mut (f64, f64, f64, f64)) {
        if let Some(arr) = v.as_array() {
            if arr.len() >= 2 && arr[0].is_number() && arr[1].is_number() {
                if let (Some(x), Some(y)) = (arr[0].as_f64(), arr[1].as_f64()) {
                    b.0 = b.0.min(x);
                    b.1 = b.1.min(y);
                    b.2 = b.2.max(x);
                    b.3 = b.3.max(y);
                }
            } else {
                for child in arr {
                    walk_coords(child, b);
                }
            }
        }
    }
    if let Some(feats) = fc.get("features").and_then(Value::as_array) {
        for feat in feats {
            if let Some(geom) = feat.get("geometry") {
                if let Some(coords) = geom.get("coordinates") {
                    walk_coords(coords, &mut b);
                }
                if let Some(geoms) = geom.get("geometries").and_then(Value::as_array) {
                    for g in geoms {
                        if let Some(coords) = g.get("coordinates") {
                            walk_coords(coords, &mut b);
                        }
                    }
                }
            }
        }
    }
    if !b.0.is_finite() {
        return Err(Error::Parse(
            "GeoJSON FeatureCollection contains no valid coordinates".into(),
        ));
    }
    Bbox::new(b.0, b.1, b.2, b.3)
}

fn is_geometry_validity_error(err: &terrio::Error) -> bool {
    match err {
        terrio::Error::InvalidData(msg) => {
            let lower = msg.to_ascii_lowercase();
            lower.contains("polygon")
                || lower.contains("ring")
                || lower.contains("topology")
                || lower.contains("clip")
                || lower.contains("vert")
                || lower.contains("self-intersection")
                || lower.contains("invalid")
        }
        terrio::Error::InvalidArgument(msg) => {
            let lower = msg.to_ascii_lowercase();
            lower.contains("make-valid")
                || lower.contains("self-intersection")
                || lower.contains("is invalid")
                || lower.contains("topology")
        }
        _ => false,
    }
}

fn clip_with_lazy_repair(layer: &terrio::Layer, clip_region: &ClipRegion) -> Result<terrio::Layer> {
    let mut out = terrio::Layer::new(layer.name.clone());
    out.geom_type = layer.geom_type;
    out.crs = layer.crs.clone();
    out.schema = layer.schema.clone();

    let mut single = terrio::Layer::new(layer.name.clone());
    single.geom_type = layer.geom_type;
    single.crs = layer.crs.clone();
    single.schema = layer.schema.clone();

    let mut skipped: usize = 0;
    for feat in &layer.features {
        single.features.clear();
        single.features.push(feat.clone());
        let clipped_feat_layer = match terrio_vector_clip(&single, clip_region) {
            Ok(c) => c,
            Err(ref e) if is_geometry_validity_error(e) => match terrio_make_valid(&single) {
                Ok(repaired) => match terrio_vector_clip(&repaired, clip_region) {
                    Ok(c) => c,
                    Err(ref e2) if is_geometry_validity_error(e2) => {
                        skipped += 1;
                        continue;
                    }
                    Err(e2) => return Err(e2.into()),
                },
                Err(ref e2) if is_geometry_validity_error(e2) => {
                    skipped += 1;
                    continue;
                }
                Err(e2) => return Err(e2.into()),
            },
            Err(e) => return Err(e.into()),
        };
        for mut f in clipped_feat_layer.features {
            f.fid = out.features.len() as u64 + 1;
            out.features.push(f);
        }
    }
    if skipped > 0 {
        eprintln!(
            "warning: skipped {skipped} unrepairable geometry feature(s) while clipping layer '{}'",
            layer.name
        );
    }
    Ok(out)
}

/// Writes a GeoJSON `FeatureCollection` `Value` to `out_path` (supports `.geojson`/`.json`
/// directly, or converts via `terrio` for `.gpkg`, `.shp`, etc., with optional polygon clipping).
fn write_geojson_feature_collection(
    fc: &Value,
    out_path: &Path,
    layer_name: &str,
    clip_mask: Option<&Path>,
) -> Result<usize> {
    ensure_parent_dir(out_path)?;
    let ext = out_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let raw_count = fc
        .get("features")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);

    if clip_mask.is_none() && (ext == "geojson" || ext == "json") {
        let serialized = serde_json::to_string(fc)
            .map_err(|e| Error::Parse(format!("failed serializing GeoJSON: {e}")))?;
        let parent = out_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut staged = tempfile::Builder::new()
            .prefix(".lidar_vec_")
            .suffix(".geojson")
            .tempfile_in(parent)?;
        staged.as_file_mut().write_all(serialized.as_bytes())?;
        staged.as_file_mut().flush()?;
        staged.persist(out_path).map_err(|e| Error::Io(e.error))?;
        return Ok(raw_count);
    }

    // Use a temporary GeoJSON file to convert/clip cleanly via terrio
    let tmp = tempfile::Builder::new()
        .prefix("lidar_fed_")
        .suffix(".geojson")
        .tempfile()?;
    let serialized = serde_json::to_string(fc)
        .map_err(|e| Error::Parse(format!("failed serializing GeoJSON: {e}")))?;
    std::fs::write(tmp.path(), serialized)?;

    let tmp_str = tmp
        .path()
        .to_str()
        .ok_or_else(|| Error::InvalidArgument("non-UTF-8 temporary path".into()))?;
    let out_str = out_path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", out_path.display())))?;

    let ds = terrio::open_vector(tmp_str, &VectorOpenOptions::default())?;
    let mut layers = ds.layers;
    if let Some(first) = layers.first_mut() {
        first.name = layer_name.to_string();
    }

    if let Some(mask_path) = clip_mask {
        let mask_str = mask_path.to_str().ok_or_else(|| {
            Error::InvalidArgument(format!("non-UTF-8 mask path: {}", mask_path.display()))
        })?;
        let clip_region = ClipRegion::from_path(mask_str, &ClipTemplateOptions::default())?;
        let mut clipped = Vec::with_capacity(layers.len());
        for layer in &layers {
            // Try clipping directly first (fast path when geometries are already valid);
            // only fall back to per-feature geometry repair on invalid federal polygons
            // (e.g. occasional self-intersecting rings in USFWS NWI) so huge valid
            // polygons never pay an unnecessary MakeValid cost.
            let mut cl = match terrio_vector_clip(layer, &clip_region) {
                Ok(c) => c,
                Err(ref e) if is_geometry_validity_error(e) => {
                    clip_with_lazy_repair(layer, &clip_region)?
                }
                Err(e) => return Err(e.into()),
            };
            cl.name = layer.name.clone();
            clipped.push(cl);
        }
        layers = clipped;
    }

    let final_count: usize = layers.iter().map(|l| l.features.len()).sum();
    let layers_to_write = if layers.is_empty() {
        let mut empty = terrio::Layer::new(layer_name.to_string());
        SpatialRef::from_epsg(4326)?.apply_to_layer(&mut empty);
        vec![empty]
    } else {
        layers
    };

    // Atomically stage single-file vector formats (.gpkg/.geojson/.json) in the destination
    // directory so interrupted writes never leave truncated files that cache checks reuse.
    if matches!(ext.as_str(), "gpkg" | "geojson" | "json") {
        let parent = out_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let staged = tempfile::Builder::new()
            .prefix(".lidar_vec_")
            .suffix(&format!(".{ext}"))
            .tempfile_in(parent)?;
        let staged_path = staged.into_temp_path();
        let staged_str = staged_path
            .to_str()
            .ok_or_else(|| Error::InvalidArgument("non-UTF-8 temporary path".into()))?;
        terrio::write_vector(&layers_to_write, staged_str, &VectorWriteOptions::default())?;
        staged_path
            .persist(out_path)
            .map_err(|e| Error::Io(e.error))?;
    } else {
        terrio::write_vector(&layers_to_write, out_str, &VectorWriteOptions::default())?;
    }
    Ok(final_count)
}

/// Maximum recursive bisection depth for short ArcGIS OID pages when the server does not set
/// `exceededTransferLimit: true`. Recovers silent server caps below the chunk size while bounding
/// request amplification when a few OIDs are deleted or have null geometry.
const MAX_NON_EXCEEDED_BISECT_DEPTH: usize = 4;

/// Decides whether an ArcGIS `objectIds` chunk response was truncated by a server transfer
/// limit and should be bisected into smaller chunks.
///
/// - Never bisects single-OID chunks (`chunk_len <= 1`).
/// - Always bisects when the server explicitly sets `exceededTransferLimit: true`.
/// - When `!exceeded` and `features_len == 0`, returns `false` so missing or null-geometry
///   OIDs do not trigger recursive empty-chunk amplification.
/// - When `!exceeded` and `0 < features_len < chunk_len`, bisects only while
///   `non_exceeded_depth_left > 0`.
pub(crate) fn should_bisect_arcgis_chunk(
    exceeded: bool,
    features_len: usize,
    chunk_len: usize,
    non_exceeded_depth_left: usize,
) -> bool {
    if chunk_len <= 1 {
        return false;
    }
    if exceeded {
        return true;
    }
    if features_len == 0 {
        return false;
    }
    features_len < chunk_len && non_exceeded_depth_left > 0
}

fn fetch_arcgis_oid_chunk(
    agent: &ureq::Agent,
    query_url: &str,
    chunk: &[i64],
    out_fields: &str,
    max_retries: usize,
) -> Result<Vec<Value>> {
    fetch_arcgis_oid_chunk_inner(
        agent,
        query_url,
        chunk,
        out_fields,
        max_retries,
        MAX_NON_EXCEEDED_BISECT_DEPTH,
    )
}

fn fetch_arcgis_oid_chunk_inner(
    agent: &ureq::Agent,
    query_url: &str,
    chunk: &[i64],
    out_fields: &str,
    max_retries: usize,
    non_exceeded_depth_left: usize,
) -> Result<Vec<Value>> {
    let oid_str = chunk
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let form = vec![
        ("objectIds", oid_str),
        ("outFields", out_fields.to_string()),
        ("outSR", "4326".to_string()),
        ("f", "geojson".to_string()),
    ];
    let text = http_post_form_text(agent, query_url, &form, max_retries)?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| Error::Parse(format!("invalid GeoJSON chunk from {query_url}: {e}")))?;
    if let Some(err) = v.get("error") {
        return Err(Error::Http(format!(
            "ArcGIS chunk error from {query_url}: {err}"
        )));
    }
    let features = v
        .get("features")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let exceeded = v
        .get("exceededTransferLimit")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || v.get("properties")
            .and_then(|p| p.get("exceededTransferLimit"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

    if !exceeded && features.is_empty() {
        return Ok(features);
    }
    if should_bisect_arcgis_chunk(
        exceeded,
        features.len(),
        chunk.len(),
        non_exceeded_depth_left,
    ) {
        let next_depth = if exceeded {
            non_exceeded_depth_left
        } else {
            non_exceeded_depth_left.saturating_sub(1)
        };
        let mid = chunk.len() / 2;
        let mut left = fetch_arcgis_oid_chunk_inner(
            agent,
            query_url,
            &chunk[..mid],
            out_fields,
            max_retries,
            next_depth,
        )?;
        let right = fetch_arcgis_oid_chunk_inner(
            agent,
            query_url,
            &chunk[mid..],
            out_fields,
            max_retries,
            next_depth,
        )?;
        left.extend(right);
        return Ok(left);
    }
    if exceeded && chunk.len() == 1 {
        return Err(Error::Http(format!(
            "ArcGIS server reported exceededTransferLimit on single OID {} from {query_url}",
            chunk[0]
        )));
    }
    Ok(features)
}

/// Fetches all features matching `base_params` from an ArcGIS REST MapServer/FeatureServer
/// query endpoint using `returnIdsOnly=true` + chunked POST requests so large regions never
/// silently truncate at the server's maxRecordCount.
fn fetch_arcgis_paginated(
    agent: &ureq::Agent,
    query_url: &str,
    base_params: &[(&str, String)],
    out_fields: &str,
    opts: &DownloadOptions,
) -> Result<Value> {
    let mut id_params: Vec<(&str, String)> = base_params.to_vec();
    id_params.push(("returnIdsOnly", "true".to_string()));
    id_params.push(("f", "json".to_string()));

    let id_text = http_get_text(agent, query_url, &id_params, opts.max_retries)?;
    let id_json: Value = serde_json::from_str(&id_text)
        .map_err(|e| Error::Parse(format!("invalid ArcGIS OID JSON from {query_url}: {e}")))?;

    if let Some(err) = id_json.get("error") {
        return Err(Error::Http(format!(
            "ArcGIS server error from {query_url}: {err}"
        )));
    }

    let oids: Vec<i64> = id_json
        .get("objectIds")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default();

    if oids.is_empty() {
        return Ok(serde_json::json!({
            "type": "FeatureCollection",
            "features": []
        }));
    }

    let chunk_size = opts.oid_chunk_size.max(1);
    let chunks: Vec<Vec<i64>> = oids.chunks(chunk_size).map(|c| c.to_vec()).collect();

    #[cfg(feature = "parallel")]
    let chunk_results: Result<Vec<Vec<Value>>> = {
        // Limit concurrency to 6 so federal ArcGIS servers stay responsive
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .map_err(|e| Error::Algorithm(format!("failed creating download thread pool: {e}")))?;
        pool.install(|| {
            chunks
                .par_iter()
                .map(|c| fetch_arcgis_oid_chunk(agent, query_url, c, out_fields, opts.max_retries))
                .collect()
        })
    };
    #[cfg(not(feature = "parallel"))]
    let chunk_results: Result<Vec<Vec<Value>>> = chunks
        .iter()
        .map(|c| fetch_arcgis_oid_chunk(agent, query_url, c, out_fields, opts.max_retries))
        .collect();

    let mut all_features = Vec::with_capacity(oids.len());
    for batch in chunk_results? {
        all_features.extend(batch);
    }

    Ok(serde_json::json!({
        "type": "FeatureCollection",
        "features": all_features
    }))
}

/// Resolves a [`Region`] (`Bbox`, `Huc`, or `VectorFile`) into a validated WGS 84 [`Bbox`]
/// and (when applicable) a boundary vector file path.
///
/// For `Region::Huc`, if `boundary_out_dir` is `Some(dir)`, the fetched WBD boundary is saved
/// to `dir/region_huc_<huc>.geojson`; otherwise it is fetched in memory to determine the exact
/// extent.
pub fn resolve_region(
    region: &Region,
    boundary_out_dir: Option<&Path>,
    opts: &DownloadOptions,
) -> Result<ResolvedRegion> {
    match region {
        Region::Bbox(bbox) => {
            bbox.validate()?;
            Ok(ResolvedRegion {
                bbox: *bbox,
                boundary_path: None,
                boundary_feature_count: 0,
            })
        }
        Region::VectorFile(path) => {
            if !path.exists() {
                return Err(Error::InvalidArgument(format!(
                    "region vector file does not exist: {}",
                    path.display()
                )));
            }
            let bbox = vector_file_bbox_wgs84(path)?;
            Ok(ResolvedRegion {
                bbox,
                boundary_path: Some(path.clone()),
                boundary_feature_count: 1,
            })
        }
        Region::Huc(raw_huc) => {
            let huc = Region::validate_huc(raw_huc)?;
            let level = WbdLevel::from_digits(huc.len())?;
            let agent = make_agent(opts.timeout_secs);
            let where_clause = format!("{} = '{huc}'", level.field_name());
            let out_fields = format!("{},name,areasqkm,states", level.field_name());
            let fc = fetch_arcgis_paginated(
                &agent,
                &level.query_url(),
                &[("where", where_clause)],
                &out_fields,
                opts,
            )?;
            let count = fc
                .get("features")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if count == 0 {
                return Err(Error::InvalidArgument(format!(
                    "USGS WBD returned no watershed boundary for HUC '{huc}'"
                )));
            }
            let bbox = geojson_bbox(&fc)?;
            let boundary_path = if let Some(dir) = boundary_out_dir {
                std::fs::create_dir_all(dir)?;
                let p = dir.join(format!("region_huc_{huc}.geojson"));
                write_geojson_feature_collection(&fc, &p, "region", None)?;
                Some(p)
            } else {
                None
            };
            Ok(ResolvedRegion {
                bbox,
                boundary_path,
                boundary_feature_count: count,
            })
        }
    }
}

/// Downloads USGS Watershed Boundary Dataset (WBD) polygons for `region` at
/// `opts.wbd_level` (default `Huc12`) to `out_path` (`.geojson`, `.gpkg`, or `.shp`).
///
/// When `region` is `Region::Huc(code)` and `code` is shorter than `opts.wbd_level`
/// (e.g., HUC-8 code `"10160002"` with `WbdLevel::Huc12`), fetches all complete child
/// subwatershed polygons matching `huc12 LIKE '10160002%'`. When `region` is
/// `Region::VectorFile(path)` and `opts.clip_to_region` is `true`, the returned WBD polygons are
/// clipped to `path`.
///
/// Returns the number of WBD features written.
pub fn download_wbd(
    region: &Region,
    out_path: impl AsRef<Path>,
    opts: &DownloadOptions,
) -> Result<usize> {
    let out_path = out_path.as_ref();
    if !opts.overwrite && out_path.exists() {
        let ds = terrio::open_vector(
            out_path
                .to_str()
                .ok_or_else(|| Error::InvalidArgument("non-UTF-8 output path".into()))?,
            &VectorOpenOptions::default(),
        )?;
        return Ok(ds.layers.iter().map(|l| l.features.len()).sum());
    }

    let agent = make_agent(opts.timeout_secs);
    let level = opts.wbd_level;
    let out_fields = match level {
        WbdLevel::Huc12 => "huc12,name,areasqkm,states,hutype".to_string(),
        other => format!("{},name,areasqkm,states", other.field_name()),
    };

    let (base_params, clip_mask): (Vec<(&str, String)>, Option<&Path>) = match region {
        Region::Huc(raw_huc) => {
            let huc = Region::validate_huc(raw_huc)?;
            if huc.len() <= level.digits() {
                let clause = if huc.len() == level.digits() {
                    format!("{} = '{huc}'", level.field_name())
                } else {
                    format!("{} LIKE '{huc}%'", level.field_name())
                };
                (vec![("where", clause)], None)
            } else {
                // Requested coarser level than the provided HUC; query by prefix match on parent
                let prefix = &huc[..level.digits()];
                (
                    vec![("where", format!("{} = '{prefix}'", level.field_name()))],
                    None,
                )
            }
        }
        Region::Bbox(bbox) => {
            bbox.validate()?;
            (envelope_query_params(bbox), None)
        }
        Region::VectorFile(path) => {
            let bbox = vector_file_bbox_wgs84(path)?;
            let mask = opts.clip_to_region.then_some(path.as_path());
            (envelope_query_params(&bbox), mask)
        }
    };

    let fc = fetch_arcgis_paginated(&agent, &level.query_url(), &base_params, &out_fields, opts)?;
    write_geojson_feature_collection(&fc, out_path, level.field_name(), clip_mask)
}

fn envelope_query_params(bbox: &Bbox) -> Vec<(&'static str, String)> {
    vec![
        ("geometry", bbox.to_csv_string()),
        ("geometryType", "esriGeometryEnvelope".to_string()),
        ("inSR", "4326".to_string()),
        ("spatialRel", "esriSpatialRelIntersects".to_string()),
    ]
}

/// Extracts a 3DEP grid tile key from a USGS product title if present:
/// - 1-m UTM 10-km tiles: `<zone>_x\d+y\d+` (e.g. `"14_x47y523"`) when a 1..=60 UTM zone token
///   precedes `x\d+y\d+`, or `"x47y523"` if no zone token is present
/// - 1/3 and 1 arc-second 1x1-degree tiles: `[ns]\d+[ew]\d+` (e.g. `"n48w100"`)
///
/// Falls back to the full title if neither pattern is found.
pub(crate) fn extract_3dep_grid_key(title: &str) -> String {
    let bytes = title.as_bytes();
    let match_pair = |a0: u8, a1: u8, b0: u8, b1: u8| -> Option<(usize, String)> {
        for i in 0..bytes.len() {
            if bytes[i] == a0 || bytes[i] == a1 {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 1 && j < bytes.len() && (bytes[j] == b0 || bytes[j] == b1) {
                    let y_start = j + 1;
                    let mut k = y_start;
                    while k < bytes.len() && bytes[k].is_ascii_digit() {
                        k += 1;
                    }
                    if k > y_start {
                        let prev_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                        let next_ok = k == bytes.len() || !bytes[k].is_ascii_alphanumeric();
                        if prev_ok && next_ok {
                            return Some((i, title[i..k].to_ascii_lowercase()));
                        }
                    }
                }
            }
        }
        None
    };

    if let Some((start_idx, key)) = match_pair(b'x', b'X', b'y', b'Y') {
        // Prefix the preceding UTM zone number (1..=60) so tiles straddling a UTM zone boundary
        // with identical local x/y indices never collide during deduplication.
        let prefix = title[..start_idx].trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
        let zone_tok = prefix
            .rsplit(|c: char| !c.is_ascii_alphanumeric())
            .next()
            .unwrap_or_default();
        if (1..=2).contains(&zone_tok.len())
            && zone_tok.bytes().all(|b| b.is_ascii_digit())
            && let Ok(zone) = zone_tok.parse::<u8>()
            && (1..=60).contains(&zone)
        {
            return format!("{zone}_{key}");
        }
        return key;
    }
    if let Some((_, k)) = match_pair(b'n', b'N', b'w', b'W')
        .or_else(|| match_pair(b'n', b'N', b'e', b'E'))
        .or_else(|| match_pair(b's', b'S', b'w', b'W'))
        .or_else(|| match_pair(b's', b'S', b'e', b'E'))
    {
        return k;
    }
    title.to_string()
}

/// Sanitizes a remote tile download URL into a safe basename confined to `tiles_dir`.
pub(crate) fn sanitize_tile_filename(download_url: &str, grid_key: &str) -> String {
    let without_query = download_url
        .split(['?', '#'])
        .next()
        .unwrap_or(download_url);
    let raw = without_query.rsplit('/').next().unwrap_or_default();
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .collect();
    if !cleaned.is_empty() && cleaned != "." && cleaned != ".." && !cleaned.starts_with('.') {
        return cleaned;
    }
    let fallback_key: String = grid_key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let fallback_key = fallback_key.trim_matches('_');
    if fallback_key.is_empty() {
        "tile.tif".to_string()
    } else {
        format!("{fallback_key}.tif")
    }
}

/// Parses USGS TNM products JSON and optionally deduplicates overlapping grid tiles to the
/// newest `publicationDate`.
pub(crate) fn parse_tnm_products_json(
    json: &Value,
    filter_bbox: Option<&Bbox>,
    deduplicate: bool,
) -> Vec<TnmProductItem> {
    let Some(items) = json.get("items").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut parsed: Vec<TnmProductItem> = Vec::new();
    for it in items {
        let Some(url) = it
            .get("downloadURL")
            .and_then(Value::as_str)
            .filter(|s| s.starts_with("https://"))
        else {
            continue;
        };
        let Some(bb) = it.get("boundingBox") else {
            continue;
        };
        let (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) = (
            bb.get("minX").and_then(Value::as_f64),
            bb.get("minY").and_then(Value::as_f64),
            bb.get("maxX").and_then(Value::as_f64),
            bb.get("maxY").and_then(Value::as_f64),
        ) else {
            continue;
        };
        let Ok(bbox) = Bbox::new(min_x, min_y, max_x, max_y) else {
            continue;
        };
        if let Some(fb) = filter_bbox
            && !bbox.intersects(fb)
        {
            continue;
        }
        let title = it
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let grid_key = extract_3dep_grid_key(&title);
        let publication_date = it
            .get("publicationDate")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let size_bytes = it
            .get("sizeInBytes")
            .and_then(Value::as_u64)
            .filter(|&n| n > 0);
        let format = it
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        parsed.push(TnmProductItem {
            grid_key,
            title,
            publication_date,
            download_url: url.to_string(),
            size_bytes,
            format,
            bbox,
        });
    }

    if !deduplicate {
        parsed.sort_by(|a, b| a.grid_key.cmp(&b.grid_key));
        return parsed;
    }

    let mut by_grid: HashMap<String, TnmProductItem> = HashMap::new();
    for item in parsed {
        match by_grid.get(&item.grid_key) {
            Some(prev) if prev.publication_date >= item.publication_date => {}
            _ => {
                by_grid.insert(item.grid_key.clone(), item);
            }
        }
    }
    let mut deduped: Vec<TnmProductItem> = by_grid.into_values().collect();
    deduped.sort_by(|a, b| a.grid_key.cmp(&b.grid_key));
    deduped
}

/// Queries the USGS The National Map (TNM) API for 3DEP elevation or point-cloud tiles
/// intersecting `region` without downloading tile payloads.
pub fn query_3dep_products(region: &Region, opts: &DownloadOptions) -> Result<Vec<TnmProductItem>> {
    let resolved = resolve_region(region, None, opts)?;
    query_3dep_products_for_bbox(&resolved.bbox, opts)
}

fn query_3dep_products_for_bbox(
    bbox: &Bbox,
    opts: &DownloadOptions,
) -> Result<Vec<TnmProductItem>> {
    let agent = make_agent(opts.timeout_secs);
    let max_products = opts.max_products.max(1);
    let mut offset = 0usize;
    let mut raw_items: Vec<Value> = Vec::new();

    loop {
        let remaining = max_products.saturating_sub(raw_items.len());
        if remaining == 0 {
            break;
        }
        let req_size = 100usize.min(remaining);
        let mut params = vec![
            ("datasets", opts.dem_product.tnm_dataset_name().to_string()),
            ("bbox", bbox.to_csv_string()),
            ("max", req_size.to_string()),
            ("offset", offset.to_string()),
        ];
        if let Some(fmt) = opts.dem_product.prod_formats() {
            params.push(("prodFormats", fmt.to_string()));
        }
        let text = http_get_text(&agent, USGS_TNM_PRODUCTS_URL, &params, opts.max_retries)?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Parse(format!("invalid USGS TNM JSON response: {e}")))?;
        let batch = json
            .get("items")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let batch_len = batch.len();
        raw_items.extend(batch);
        let total_opt = json
            .get("total")
            .and_then(Value::as_u64)
            .map(|t| t as usize);
        offset += batch_len;
        if let Some(total) = total_opt {
            if offset < total && raw_items.len() >= max_products {
                return Err(Error::InvalidArgument(format!(
                    "USGS TNM matched {total} '{}' products for bbox {}, exceeding max_products={max_products}; increase max_products or narrow the region",
                    opts.dem_product.tnm_dataset_name(),
                    bbox.to_csv_string()
                )));
            }
            if batch_len == 0 || offset >= total {
                break;
            }
        } else {
            if batch_len == 0 || batch_len < req_size {
                break;
            }
            if raw_items.len() >= max_products {
                return Err(Error::InvalidArgument(format!(
                    "USGS TNM returned at least {max_products} '{}' products for bbox {} without a total count; increase max_products or narrow the region",
                    opts.dem_product.tnm_dataset_name(),
                    bbox.to_csv_string()
                )));
            }
        }
    }

    let combined = serde_json::json!({ "items": raw_items });
    Ok(parse_tnm_products_json(
        &combined,
        Some(bbox),
        opts.deduplicate_tiles,
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DemMosaicCacheMeta {
    bbox: Bbox,
    dem_product: DemProduct,
    clip_to_region: bool,
    boundary_path: Option<PathBuf>,
    tile_paths: Vec<PathBuf>,
}

fn dem_mosaic_cache_matches(meta_path: &Path, expected: &DemMosaicCacheMeta) -> bool {
    let Ok(txt) = std::fs::read_to_string(meta_path) else {
        return false;
    };
    let Ok(prev) = serde_json::from_str::<DemMosaicCacheMeta>(&txt) else {
        return false;
    };
    let tol = 1e-6;
    let bbox_same = (prev.bbox.min_lon - expected.bbox.min_lon).abs() <= tol
        && (prev.bbox.min_lat - expected.bbox.min_lat).abs() <= tol
        && (prev.bbox.max_lon - expected.bbox.max_lon).abs() <= tol
        && (prev.bbox.max_lat - expected.bbox.max_lat).abs() <= tol;
    bbox_same
        && prev.dem_product == expected.dem_product
        && prev.clip_to_region == expected.clip_to_region
        && prev.boundary_path == expected.boundary_path
        && prev.tile_paths == expected.tile_paths
}

/// Queries, downloads, and optionally mosaics and clips USGS 3DEP DEM (or LPC) tiles for `region`.
pub fn download_3dep_dem(
    region: &Region,
    out_dir: impl AsRef<Path>,
    opts: &DownloadOptions,
) -> Result<DemDownloadResult> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;
    let resolved = resolve_region(region, Some(out_dir), opts)?;
    download_3dep_dem_resolved(&resolved, out_dir, opts, opts.overwrite)
}

fn fnv1a_hash64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn insert_filename_tag(filename: &str, tag: &str) -> String {
    if let Some(dot) = filename.rfind('.')
        && dot > 0
    {
        format!("{}_{tag}{}", &filename[..dot], &filename[dot..])
    } else {
        format!("{filename}_{tag}")
    }
}

/// Assigns collision-free local tile filenames for `products` so parallel tile downloads never
/// race on the same destination path. Non-colliding tiles keep their exact sanitized basename.
pub(crate) fn assign_unique_tile_filenames(products: &[TnmProductItem]) -> Vec<String> {
    let base_names: Vec<String> = products
        .iter()
        .map(|it| sanitize_tile_filename(&it.download_url, &it.grid_key))
        .collect();
    let mut counts: HashMap<String, usize> = HashMap::with_capacity(base_names.len());
    for name in &base_names {
        *counts.entry(name.to_ascii_lowercase()).or_insert(0) += 1;
    }

    let mut used: HashSet<String> = HashSet::with_capacity(base_names.len());
    let mut out = Vec::with_capacity(base_names.len());
    for (idx, (item, base)) in products.iter().zip(base_names).enumerate() {
        let lower = base.to_ascii_lowercase();
        let candidate = if counts.get(&lower).copied().unwrap_or(0) <= 1 && used.insert(lower) {
            base
        } else {
            let short_hash = format!(
                "{:08x}",
                (fnv1a_hash64(item.download_url.as_bytes()) as u32)
            );
            let tagged = insert_filename_tag(&base, &short_hash);
            if used.insert(tagged.to_ascii_lowercase()) {
                tagged
            } else {
                let mut disambig = insert_filename_tag(&base, &format!("{short_hash}_{idx}"));
                let mut extra = 1usize;
                while !used.insert(disambig.to_ascii_lowercase()) {
                    disambig = insert_filename_tag(&base, &format!("{short_hash}_{idx}_{extra}"));
                    extra += 1;
                }
                disambig
            }
        };
        out.push(candidate);
    }
    out
}

fn download_3dep_dem_resolved(
    resolved: &ResolvedRegion,
    out_dir: &Path,
    opts: &DownloadOptions,
    overwrite_mosaic: bool,
) -> Result<DemDownloadResult> {
    let products = query_3dep_products_for_bbox(&resolved.bbox, opts)?;
    if products.is_empty() {
        return Err(Error::Http(format!(
            "no USGS 3DEP '{}' tiles found for bbox {}",
            opts.dem_product.tnm_dataset_name(),
            resolved.bbox.to_csv_string()
        )));
    }

    let tiles_dir = opts
        .tile_cache_dir
        .clone()
        .unwrap_or_else(|| out_dir.join("tiles"));
    std::fs::create_dir_all(&tiles_dir)?;

    let filenames = assign_unique_tile_filenames(&products);
    let pairs: Vec<(&TnmProductItem, &String)> = products.iter().zip(filenames.iter()).collect();

    let agent = make_download_agent(opts.timeout_secs);
    let download_one = |(item, fname): &(&TnmProductItem, &String)| -> Result<PathBuf> {
        let dest = tiles_dir.join(fname);
        http_download_file(
            &agent,
            &item.download_url,
            &dest,
            item.size_bytes,
            opts.overwrite,
            opts.max_retries,
        )
    };

    #[cfg(feature = "parallel")]
    let tile_paths: Vec<PathBuf> = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .map_err(|e| Error::Algorithm(format!("failed creating download thread pool: {e}")))?;
        pool.install(|| {
            pairs
                .par_iter()
                .map(download_one)
                .collect::<Result<Vec<_>>>()
        })?
    };
    #[cfg(not(feature = "parallel"))]
    let tile_paths: Vec<PathBuf> = pairs.iter().map(download_one).collect::<Result<Vec<_>>>()?;

    let mosaic_path = if opts.mosaic_dem && opts.dem_product.is_raster() && !tile_paths.is_empty() {
        let final_dem = out_dir.join("dem.tif");
        let meta_path = out_dir.join(".dem.meta.json");
        let expected_meta = DemMosaicCacheMeta {
            bbox: resolved.bbox,
            dem_product: opts.dem_product,
            clip_to_region: opts.clip_to_region,
            boundary_path: resolved.boundary_path.clone(),
            tile_paths: tile_paths.clone(),
        };
        let cache_valid = !overwrite_mosaic
            && final_dem.is_file()
            && std::fs::metadata(&final_dem).is_ok_and(|m| m.len() > 0)
            && dem_mosaic_cache_matches(&meta_path, &expected_meta);

        if !cache_valid {
            // Stage the final mosaic/clip into a unique temp file in out_dir and rename atomically
            // so an interrupted run never leaves a truncated dem.tif behind.
            let staged_tmp = tempfile::Builder::new()
                .prefix(".dem_stage_")
                .suffix(".tif")
                .tempfile_in(out_dir)?;
            let staged_path = staged_tmp.into_temp_path();
            let staged_pb = staged_path.to_path_buf();

            let build_res = (|| -> Result<()> {
                let needs_clip = opts.clip_to_region && resolved.boundary_path.is_some();
                if needs_clip {
                    if tile_paths.len() == 1 {
                        clip_image(
                            &tile_paths[0],
                            resolved.boundary_path.as_ref().expect("boundary path"),
                            &staged_pb,
                        )?;
                    } else {
                        let raw_tmp = tempfile::Builder::new()
                            .prefix(".dem_mosaic_")
                            .suffix(".tif")
                            .tempfile_in(out_dir)?;
                        let raw_path = raw_tmp.into_temp_path();
                        let raw_pb = raw_path.to_path_buf();
                        mosaic_with_resampling(&tile_paths, &raw_pb, Resampling::Bilinear)?;
                        clip_image(
                            &raw_pb,
                            resolved.boundary_path.as_ref().expect("boundary path"),
                            &staged_pb,
                        )?;
                    }
                } else if tile_paths.len() == 1 {
                    std::fs::copy(&tile_paths[0], &staged_pb)?;
                } else {
                    mosaic_with_resampling(&tile_paths, &staged_pb, Resampling::Bilinear)?;
                }
                Ok(())
            })();
            build_res?;
            staged_path
                .persist(&final_dem)
                .map_err(|e| Error::Io(e.error))?;
            if let Ok(meta_json) = serde_json::to_string(&expected_meta) {
                let _ = write_file_atomic(&meta_path, meta_json.as_bytes());
            }
        }
        Some(final_dem)
    } else {
        None
    };

    Ok(DemDownloadResult {
        bbox: resolved.bbox,
        products,
        tile_paths,
        mosaic_path,
    })
}

/// Downloads USGS National Hydrography Dataset (NHD) flowlines intersecting `region`.
pub fn download_nhd(
    region: &Region,
    out_path: impl AsRef<Path>,
    opts: &DownloadOptions,
) -> Result<usize> {
    let tmp_dir = tempfile::tempdir()?;
    let resolved = resolve_region(region, Some(tmp_dir.path()), opts)?;
    download_nhd_resolved(&resolved, out_path.as_ref(), opts)
}

fn download_nhd_resolved(
    resolved: &ResolvedRegion,
    out_path: &Path,
    opts: &DownloadOptions,
) -> Result<usize> {
    if !opts.overwrite && out_path.exists() {
        let ds = terrio::open_vector(
            out_path
                .to_str()
                .ok_or_else(|| Error::InvalidArgument("non-UTF-8 output path".into()))?,
            &VectorOpenOptions::default(),
        )?;
        return Ok(ds.layers.iter().map(|l| l.features.len()).sum());
    }
    let agent = make_agent(opts.timeout_secs);
    let fc = fetch_arcgis_paginated(
        &agent,
        USGS_NHD_FLOWLINE_URL,
        &envelope_query_params(&resolved.bbox),
        "gnis_name,ftype,fcode,lengthkm",
        opts,
    )?;
    let mask = if opts.clip_to_region {
        resolved.boundary_path.as_deref()
    } else {
        None
    };
    write_geojson_feature_collection(&fc, out_path, "nhd_flowlines", mask)
}

/// Downloads USFWS National Wetlands Inventory (NWI) wetland polygons intersecting `region`.
pub fn download_nwi(
    region: &Region,
    out_path: impl AsRef<Path>,
    opts: &DownloadOptions,
) -> Result<usize> {
    let tmp_dir = tempfile::tempdir()?;
    let resolved = resolve_region(region, Some(tmp_dir.path()), opts)?;
    download_nwi_resolved(&resolved, out_path.as_ref(), opts)
}

fn download_nwi_resolved(
    resolved: &ResolvedRegion,
    out_path: &Path,
    opts: &DownloadOptions,
) -> Result<usize> {
    if !opts.overwrite && out_path.exists() {
        let ds = terrio::open_vector(
            out_path
                .to_str()
                .ok_or_else(|| Error::InvalidArgument("non-UTF-8 output path".into()))?,
            &VectorOpenOptions::default(),
        )?;
        return Ok(ds.layers.iter().map(|l| l.features.len()).sum());
    }
    let agent = make_agent(opts.timeout_secs);
    let fc = fetch_arcgis_paginated(
        &agent,
        USFWS_NWI_URL,
        &envelope_query_params(&resolved.bbox),
        "ATTRIBUTE,WETLAND_TYPE,ACRES",
        opts,
    )?;
    let mask = if opts.clip_to_region {
        resolved.boundary_path.as_deref()
    } else {
        None
    };
    write_geojson_feature_collection(&fc, out_path, "nwi_wetlands", mask)
}

/// Downloads USGS National Transportation road and railway centerlines intersecting `region`.
pub fn download_roads(
    region: &Region,
    out_path: impl AsRef<Path>,
    opts: &DownloadOptions,
) -> Result<usize> {
    let tmp_dir = tempfile::tempdir()?;
    let resolved = resolve_region(region, Some(tmp_dir.path()), opts)?;
    download_roads_resolved(&resolved, out_path.as_ref(), opts)
}

fn download_roads_resolved(
    resolved: &ResolvedRegion,
    out_path: &Path,
    opts: &DownloadOptions,
) -> Result<usize> {
    if !opts.overwrite && out_path.exists() {
        let ds = terrio::open_vector(
            out_path
                .to_str()
                .ok_or_else(|| Error::InvalidArgument("non-UTF-8 output path".into()))?,
            &VectorOpenOptions::default(),
        )?;
        return Ok(ds.layers.iter().map(|l| l.features.len()).sum());
    }
    let agent = make_agent(opts.timeout_secs);
    let base_params = envelope_query_params(&resolved.bbox);
    let layers = if opts.transportation_layers.is_empty() {
        DEFAULT_TRANSPORTATION_LAYERS
    } else {
        &opts.transportation_layers
    };

    let mut all_features: Vec<Value> = Vec::new();
    for &lyr in layers {
        let url = format!("{USGS_TRANSPORTATION_BASE_URL}/{lyr}/query");
        // Use "*" because USGS Transportation layers do not share identical attribute schemas
        // (for example, Railroads layer 29 publishes `name` and `mtfcc_code` rather than
        // `FULL_STREET_NAME`); properties are normalized to a compact schema below.
        let fc = fetch_arcgis_paginated(&agent, &url, &base_params, "*", opts)?;
        if let Some(feats) = fc.get("features").and_then(Value::as_array) {
            for feat in feats {
                let mut f = feat.clone();
                // Normalize to compact consistent schema across road and railway layers
                let old_props = f
                    .get("properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let mut props = Map::new();
                props.insert("trans_layer".to_string(), Value::from(lyr));
                if let Some(name) = old_props.get("FULL_STREET_NAME") {
                    props.insert("FULL_STREET_NAME".to_string(), name.clone());
                } else {
                    props.insert("FULL_STREET_NAME".to_string(), Value::Null);
                }
                if let Some(mtfcc) = old_props.get("MTFCC_CODE") {
                    props.insert("MTFCC_CODE".to_string(), mtfcc.clone());
                } else {
                    props.insert("MTFCC_CODE".to_string(), Value::Null);
                }
                if let Some(obj) = f.as_object_mut() {
                    obj.insert("properties".to_string(), Value::Object(props));
                }
                all_features.push(f);
            }
        }
    }

    let merged = serde_json::json!({
        "type": "FeatureCollection",
        "features": all_features
    });
    let mask = if opts.clip_to_region {
        resolved.boundary_path.as_deref()
    } else {
        None
    };
    write_geojson_feature_collection(&merged, out_path, "usgs_roads", mask)
}

fn find_js_assignment_rhs<'a>(body: &'a str, ident: &str) -> Option<&'a str> {
    let bytes = body.as_bytes();
    let mut offset = 0usize;
    while let Some(rel) = body[offset..].find(ident) {
        let idx = offset + rel;
        let end_ident = idx + ident.len();
        let prev_ok =
            idx == 0 || !(bytes[idx - 1].is_ascii_alphanumeric() || bytes[idx - 1] == b'_');
        if prev_ok {
            let mut pos = end_ident;
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'=' && bytes.get(pos + 1) != Some(&b'=') {
                return Some(&body[pos + 1..]);
            }
        }
        offset = end_ident;
    }
    None
}

/// Parses the JavaScript-formatted `quantiles = [[...]];` response from NOAA Atlas 14 CGI.
pub(crate) fn parse_noaa_atlas14_response(
    lat: f64,
    lon: f64,
    body: &str,
) -> Result<NoaaAtlas14Table> {
    let after_eq = find_js_assignment_rhs(body, "quantiles").ok_or_else(|| {
        Error::Parse("NOAA Atlas 14 response missing 'quantiles' array (point may be outside Atlas 14 coverage)".into())
    })?;
    let start = after_eq
        .find("[[")
        .ok_or_else(|| Error::Parse("NOAA Atlas 14 quantiles opening '[[' not found".into()))?;
    let rest = &after_eq[start..];
    let end = rest
        .find("]];")
        .map(|i| i + 2)
        .or_else(|| rest.find("]]").map(|i| i + 2))
        .ok_or_else(|| Error::Parse("NOAA Atlas 14 quantiles closing ']]' not found".into()))?;
    let raw_matrix = rest[..end].replace('\'', "\"");
    let matrix: Vec<Vec<Value>> = serde_json::from_str(&raw_matrix).map_err(|e| {
        Error::Parse(format!(
            "failed parsing NOAA Atlas 14 quantiles matrix: {e}"
        ))
    })?;

    let mut depths_mm: BTreeMap<String, BTreeMap<u32, f64>> = BTreeMap::new();
    for (&dur, row) in NOAA_DURATIONS.iter().zip(matrix.iter()) {
        let mut by_ari = BTreeMap::new();
        for (&ari, cell) in NOAA_ARIS.iter().zip(row.iter()) {
            let val = match cell {
                Value::Number(n) => n.as_f64(),
                Value::String(s) => s.trim().parse::<f64>().ok(),
                _ => None,
            };
            if let Some(v) = val
                && v.is_finite()
            {
                by_ari.insert(ari, v);
            }
        }
        if !by_ari.is_empty() {
            depths_mm.insert(dur.to_string(), by_ari);
        }
    }

    if depths_mm.is_empty() {
        return Err(Error::Parse(
            "NOAA Atlas 14 quantiles matrix contained no numeric values".into(),
        ));
    }

    Ok(NoaaAtlas14Table {
        lat: (lat * 10_000.0).round() / 10_000.0,
        lon: (lon * 10_000.0).round() / 10_000.0,
        units: "mm".to_string(),
        depths_mm,
    })
}

/// Fetches NOAA Atlas 14 partial-duration-series precipitation depth frequency values (mm)
/// for a WGS 84 `(lat, lon)` point, optionally saving the parsed JSON table to `out_json`.
pub fn fetch_noaa_atlas14(
    lat: f64,
    lon: f64,
    out_json: Option<&Path>,
    opts: &DownloadOptions,
) -> Result<NoaaAtlas14Table> {
    if !lat.is_finite() || !(-90.0..=90.0).contains(&lat) {
        return Err(Error::InvalidArgument(format!("invalid latitude: {lat}")));
    }
    if !lon.is_finite() || !(-180.0..=180.0).contains(&lon) {
        return Err(Error::InvalidArgument(format!("invalid longitude: {lon}")));
    }
    let req_lat = (lat * 10_000.0).round() / 10_000.0;
    let req_lon = (lon * 10_000.0).round() / 10_000.0;
    if let Some(p) = out_json
        && !opts.overwrite
        && p.exists()
        && let Ok(txt) = std::fs::read_to_string(p)
        && let Ok(parsed) = serde_json::from_str::<NoaaAtlas14Table>(&txt)
        && (parsed.lat - req_lat).abs() <= 1e-5
        && (parsed.lon - req_lon).abs() <= 1e-5
    {
        return Ok(parsed);
    }

    let agent = make_agent(opts.timeout_secs.min(60));
    let params = vec![
        ("lat", format!("{lat:.4}")),
        ("lon", format!("{lon:.4}")),
        ("type", "pf".to_string()),
        ("data", "depth".to_string()),
        ("units", "metric".to_string()),
        ("series", "pds".to_string()),
    ];
    let body = http_get_text(&agent, NOAA_ATLAS14_URL, &params, opts.max_retries)?;
    let table = parse_noaa_atlas14_response(lat, lon, &body)?;

    if let Some(p) = out_json {
        let pretty = serde_json::to_string_pretty(&table)
            .map_err(|e| Error::Parse(format!("failed serializing NOAA Atlas 14 JSON: {e}")))?;
        write_file_atomic(p, pretty.as_bytes())?;
    }
    Ok(table)
}

fn region_cache_key(region: &Region) -> String {
    match region {
        Region::Bbox(_) => "bbox".to_string(),
        Region::Huc(code) => format!("huc:{}", code.trim()),
        Region::VectorFile(path) => {
            let norm = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
            format!("file:{}", norm.display())
        }
    }
}

fn effective_transportation_layers(opts: &DownloadOptions) -> Vec<u32> {
    if opts.transportation_layers.is_empty() {
        DEFAULT_TRANSPORTATION_LAYERS.to_vec()
    } else {
        opts.transportation_layers.clone()
    }
}

fn manifest_matches_request(
    manifest_path: &Path,
    region_key: &str,
    bbox: &Bbox,
    opts: &DownloadOptions,
) -> bool {
    let Ok(txt) = std::fs::read_to_string(manifest_path) else {
        return false;
    };
    let Ok(prev) = serde_json::from_str::<FederalBundleResult>(&txt) else {
        return false;
    };
    let tol = 1e-6;
    let bbox_same = (prev.region_bbox.min_lon - bbox.min_lon).abs() <= tol
        && (prev.region_bbox.min_lat - bbox.min_lat).abs() <= tol
        && (prev.region_bbox.max_lon - bbox.max_lon).abs() <= tol
        && (prev.region_bbox.max_lat - bbox.max_lat).abs() <= tol;
    bbox_same
        && prev.region_key == region_key
        && prev.dem_product == opts.dem_product
        && prev.wbd_level == opts.wbd_level
        && prev.clip_to_region == opts.clip_to_region
        && prev.mosaic_dem == opts.mosaic_dem
        && prev.transportation_layers == effective_transportation_layers(opts)
}

/// Removes stale per-region bundle artifacts in `out_dir` when invalidating an existing
/// mismatched `manifest.json`, preserving `keep_boundary` (resolved canonically so relative vs
/// absolute path spellings never delete the active boundary) and leaving tile caches untouched.
///
/// Only called when `manifest_path` already exists so pointing `-o` at a directory without a
/// prior `manifest.json` never deletes unrelated user files. When `manifest.json` parses cleanly,
/// only direct-child bundle paths in `out_dir` recorded in that prior manifest (plus DEM metadata
/// sidecars and stale `region_huc_*.geojson` files in `out_dir`) are removed. If `manifest.json`
/// fails to parse, only `manifest.json` itself is removed so arbitrary unverified files in
/// `out_dir` are never deleted.
pub(crate) fn clean_stale_bundle_outputs(
    out_dir: &Path,
    manifest_path: &Path,
    keep_boundary: Option<&Path>,
) {
    let keep_canon = keep_boundary.and_then(resolve_path_for_compare);
    let is_kept = |candidate: &Path| -> bool {
        if let Some(kb) = keep_boundary
            && candidate == kb
        {
            return true;
        }
        match (&keep_canon, resolve_path_for_compare(candidate)) {
            (Some(k), Some(c)) => *k == c,
            _ => false,
        }
    };
    let remove_if_not_kept = |candidate: &Path| {
        if !is_kept(candidate) {
            let _ = std::fs::remove_file(candidate);
        }
    };

    let parsed_prev = std::fs::read_to_string(manifest_path)
        .ok()
        .and_then(|txt| serde_json::from_str::<FederalBundleResult>(&txt).ok());

    let out_dir_canon = std::fs::canonicalize(out_dir).ok();
    let is_direct_child_of_out_dir = |candidate: &Path| -> bool {
        let Some(ref od) = out_dir_canon else {
            return false;
        };
        resolve_path_for_compare(candidate)
            .and_then(|c| c.parent().map(Path::to_path_buf))
            .is_some_and(|parent| parent == *od)
    };

    if let Some(prev) = parsed_prev {
        for p in [
            prev.wbd_path.as_deref(),
            prev.dem_mosaic_path.as_deref(),
            prev.nhd_path.as_deref(),
            prev.nwi_path.as_deref(),
            prev.roads_path.as_deref(),
            prev.culverts_path.as_deref(),
            prev.noaa_atlas14_path.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if is_direct_child_of_out_dir(p) {
                remove_if_not_kept(p);
            }
        }
        if prev.dem_mosaic_path.is_some() {
            remove_if_not_kept(&out_dir.join(".dem.meta.json"));
            remove_if_not_kept(&out_dir.join("dem.tif.meta.json"));
        }
        if let Ok(entries) = std::fs::read_dir(out_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name();
                if let Some(s) = fname.to_str()
                    && s.starts_with("region_huc_")
                    && s.ends_with(".geojson")
                {
                    remove_if_not_kept(&entry.path());
                }
            }
        }
    }

    remove_if_not_kept(manifest_path);
}

/// Downloads a complete bundle of open U.S. federal datasets for `region` into `out_dir`.
///
/// Downloads every dataset listed in `opts.datasets` (defaulting to WBD, 3DEP DEM, NHD,
/// NWI, USGS Transportation roads and railways, synthesized road- and rail-stream culverts, and
/// NOAA Atlas 14 precipitation frequencies), caches files on disk, and writes `manifest.json`.
///
/// `out_dir` is scoped to a single region and option set (if `out_dir/manifest.json` records a
/// different region or options, bundle outputs in `out_dir` are automatically refreshed), while
/// [`DownloadOptions::tile_cache_dir`] can be shared safely across regions.
pub fn download_federal_data(
    region: &Region,
    out_dir: impl AsRef<Path>,
    opts: &FederalBundleOptions,
) -> Result<FederalBundleResult> {
    let has = |d: FederalDataset| opts.datasets.contains(&d);
    let need_culverts = has(FederalDataset::Culverts);
    if need_culverts {
        SynthesizeCulvertsOptions {
            half_length_m: opts.download.culvert_half_length_m,
            snap_distance_m: opts.download.culvert_snap_distance_m,
            ..Default::default()
        }
        .validate()?;
    }

    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let resolved = resolve_region(region, Some(out_dir), &opts.download)?;
    let reg_key = region_cache_key(region);
    let trans_layers = effective_transportation_layers(&opts.download);
    let manifest_path = out_dir.join("manifest.json");
    let cache_matches =
        manifest_matches_request(&manifest_path, &reg_key, &resolved.bbox, &opts.download);
    let mut bundle_dl_opts = opts.download.clone();
    if !cache_matches {
        // Invalidate prior manifest-managed per-region bundle files before writing new outputs
        // so a partial subset run for Region B can never inherit stale Region A files, without
        // deleting unrelated user files in directories that had no prior manifest.json.
        if manifest_path.exists() {
            clean_stale_bundle_outputs(out_dir, &manifest_path, resolved.boundary_path.as_deref());
        }
        bundle_dl_opts.overwrite = true;
    }

    let (wbd_path, wbd_count) = if has(FederalDataset::Wbd) {
        let p = out_dir.join("wbd.gpkg");
        let n = download_wbd(region, &p, &bundle_dl_opts)?;
        (Some(p), n)
    } else {
        (None, 0)
    };

    let (dem_tiles, dem_mosaic_path) = if has(FederalDataset::Dem) {
        // Keep original opts.download.overwrite for tile cache reuse, while refreshing dem.tif
        // whenever bundle_dl_opts.overwrite is true.
        let res = download_3dep_dem_resolved(
            &resolved,
            out_dir,
            &opts.download,
            bundle_dl_opts.overwrite,
        )?;
        (res.tile_paths, res.mosaic_path)
    } else {
        (Vec::new(), None)
    };

    let (nhd_path, nhd_count) = if has(FederalDataset::Nhd) || need_culverts {
        let p = out_dir.join("nhd_flowlines.gpkg");
        let n = download_nhd_resolved(&resolved, &p, &bundle_dl_opts)?;
        (Some(p), n)
    } else {
        (None, 0)
    };

    let (nwi_path, nwi_count) = if has(FederalDataset::Nwi) {
        let p = out_dir.join("nwi_wetlands.gpkg");
        let n = download_nwi_resolved(&resolved, &p, &bundle_dl_opts)?;
        (Some(p), n)
    } else {
        (None, 0)
    };

    let (roads_path, roads_count) = if has(FederalDataset::Roads) || need_culverts {
        let p = out_dir.join("usgs_roads.gpkg");
        let n = download_roads_resolved(&resolved, &p, &bundle_dl_opts)?;
        (Some(p), n)
    } else {
        (None, 0)
    };

    // Match DEM CRS for culverts only when the DEM raster has a projected CRS (e.g., 1-m UTM);
    // geographic DEMs (e.g., 10-m / 30-m EPSG:4269) fall back to local WGS 84 UTM meters so
    // culvert lengths and snap distances are never interpreted in degrees.
    let dem_crs = dem_mosaic_path
        .as_deref()
        .or_else(|| dem_tiles.first().map(PathBuf::as_path))
        .and_then(|p| p.to_str())
        .and_then(|s| terrio::open_raster(s).ok())
        .and_then(|ds| {
            ds.raster
                .crs
                .wkt
                .or(ds.raster.crs.epsg.map(|e| format!("EPSG:{e}")))
        })
        .filter(|crs_str| {
            SpatialRef::parse(crs_str)
                .map(|srs| !srs.is_unknown() && !srs.is_geographic())
                .unwrap_or(false)
        });

    let (culverts_path, culverts_count) = if need_culverts {
        if let (Some(nhd_p), Some(roads_p)) = (nhd_path.as_deref(), roads_path.as_deref()) {
            let p = out_dir.join("culverts.gpkg");
            let c_opts = SynthesizeCulvertsOptions {
                half_length_m: opts.download.culvert_half_length_m,
                snap_distance_m: opts.download.culvert_snap_distance_m,
                target_crs: dem_crs,
                ..Default::default()
            };
            let n = synthesize_culverts(nhd_p, roads_p, &p, &c_opts)?;
            (Some(p), n)
        } else {
            (None, 0)
        }
    } else {
        (None, 0)
    };

    let (noaa_atlas14_path, noaa_atlas14) = if has(FederalDataset::NoaaAtlas14) {
        let (lon, lat) = resolved.bbox.centroid();
        let p = out_dir.join("noaa_atlas14.json");
        let table = fetch_noaa_atlas14(lat, lon, Some(&p), &bundle_dl_opts)?;
        (Some(p), Some(table))
    } else {
        (None, None)
    };

    let result = FederalBundleResult {
        region_bbox: resolved.bbox,
        region_key: reg_key,
        dem_product: opts.download.dem_product,
        wbd_level: opts.download.wbd_level,
        clip_to_region: opts.download.clip_to_region,
        mosaic_dem: opts.download.mosaic_dem,
        transportation_layers: trans_layers,
        wbd_path,
        wbd_count,
        dem_tiles,
        dem_mosaic_path,
        nhd_path,
        nhd_count,
        nwi_path,
        nwi_count,
        roads_path,
        roads_count,
        culverts_path,
        culverts_count,
        noaa_atlas14_path,
        noaa_atlas14,
        manifest_path: manifest_path.clone(),
    };
    let manifest_json = serde_json::to_string_pretty(&result)
        .map_err(|e| Error::Parse(format!("failed serializing manifest JSON: {e}")))?;
    write_file_atomic(&manifest_path, manifest_json.as_bytes())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_parse_validate_and_intersect() {
        let b: Bbox = "-99.5, 47.0, -99.0, 47.5".parse().unwrap();
        assert!((b.centroid().0 - (-99.25)).abs() < 1e-9);
        assert!((b.centroid().1 - 47.25).abs() < 1e-9);
        assert_eq!(b.to_csv_string(), "-99.5,47,-99,47.5");

        let b2 = Bbox::new(-99.1, 47.2, -98.8, 47.6).unwrap();
        assert!(b.intersects(&b2));
        let b3 = Bbox::new(-98.5, 47.2, -98.0, 47.6).unwrap();
        assert!(!b.intersects(&b3));

        assert!(Bbox::new(-99.0, 47.0, -99.5, 47.5).is_err());
        assert!(Bbox::new(-200.0, 47.0, -99.0, 47.5).is_err());
        assert!("-99.5,47.0,-99.0".parse::<Bbox>().is_err());
    }

    #[test]
    fn region_huc_and_enums_parse() {
        assert_eq!(Region::validate_huc("10160002").unwrap(), "10160002");
        assert_eq!(
            Region::validate_huc("101600020502").unwrap(),
            "101600020502"
        );
        assert!(Region::validate_huc("1016000").is_err());
        assert!(Region::validate_huc("1016000a").is_err());

        assert_eq!("12".parse::<WbdLevel>().unwrap(), WbdLevel::Huc12);
        assert_eq!("huc8".parse::<WbdLevel>().unwrap(), WbdLevel::Huc8);
        assert_eq!(WbdLevel::Huc12.layer_id(), 6);
        assert_eq!(WbdLevel::Huc8.field_name(), "huc8");

        assert_eq!("1m".parse::<DemProduct>().unwrap(), DemProduct::Meter1);
        assert_eq!(
            "10m".parse::<DemProduct>().unwrap(),
            DemProduct::ThirdArcSec
        );
        assert_eq!("30m".parse::<DemProduct>().unwrap(), DemProduct::OneArcSec);
        assert_eq!(
            "lpc".parse::<DemProduct>().unwrap(),
            DemProduct::LidarPointCloud
        );

        let ds = parse_dataset_list("wbd, dem, nhd, culverts").unwrap();
        assert_eq!(
            ds,
            vec![
                FederalDataset::Wbd,
                FederalDataset::Dem,
                FederalDataset::Nhd,
                FederalDataset::Culverts
            ]
        );
        assert_eq!(parse_dataset_list("all").unwrap().len(), 7);
    }

    #[test]
    fn extract_grid_key_and_deduplicate_tnm_tiles() {
        assert_eq!(
            extract_3dep_grid_key("USGS 1 Meter 14 x47y523 ND_3DEPProcessing_D22"),
            "14_x47y523"
        );
        assert_eq!(
            extract_3dep_grid_key("USGS_1M_14_x48y524_ND_2022.tif"),
            "14_x48y524"
        );
        assert_eq!(
            extract_3dep_grid_key("USGS 1 Meter 15 x47y523 MN_2023"),
            "15_x47y523"
        );
        assert_eq!(
            extract_3dep_grid_key("USGS 1/3 Arc Second n48w100 20241024"),
            "n48w100"
        );
        assert_eq!(
            extract_3dep_grid_key("USGS 1 arc-second n48w100 1 x 1 degree"),
            "n48w100"
        );
        assert_eq!(extract_3dep_grid_key("custom_title"), "custom_title");

        let sample = serde_json::json!({
            "items": [
                {
                    "title": "USGS 1 Meter 14 x47y523 ND_Old_2018",
                    "publicationDate": "2018-05-01",
                    "downloadURL": "https://example.org/old.tif",
                    "sizeInBytes": 1000,
                    "format": "GeoTIFF",
                    "boundingBox": {"minX": -99.3, "minY": 47.0, "maxX": -99.1, "maxY": 47.2}
                },
                {
                    "title": "USGS 1 Meter 14 x47y523 ND_3DEPProcessing_D22",
                    "publicationDate": "2023-08-15",
                    "downloadURL": "https://example.org/new.tif",
                    "sizeInBytes": 2000,
                    "format": "GeoTIFF",
                    "boundingBox": {"minX": -99.3, "minY": 47.0, "maxX": -99.1, "maxY": 47.2}
                },
                {
                    "title": "USGS 1 Meter 15 x47y523 Adjacent_Zone_2022",
                    "publicationDate": "2022-06-01",
                    "downloadURL": "https://example.org/z15.tif",
                    "sizeInBytes": 2500,
                    "format": "GeoTIFF",
                    "boundingBox": {"minX": -99.2, "minY": 47.0, "maxX": -99.0, "maxY": 47.2}
                },
                {
                    "title": "USGS 1 Meter 14 x99y999 Far_Away",
                    "publicationDate": "2024-01-01",
                    "downloadURL": "https://example.org/far.tif",
                    "sizeInBytes": 3000,
                    "format": "GeoTIFF",
                    "boundingBox": {"minX": -80.0, "minY": 35.0, "maxX": -79.8, "maxY": 35.2}
                }
            ]
        });

        let region_bb = Bbox::new(-99.4, 47.05, -99.05, 47.25).unwrap();
        let items = parse_tnm_products_json(&sample, Some(&region_bb), true);
        assert_eq!(items.len(), 2, "distinct UTM zones must not collide");
        assert_eq!(items[0].grid_key, "14_x47y523");
        assert_eq!(items[0].publication_date, "2023-08-15");
        assert_eq!(items[0].download_url, "https://example.org/new.tif");
        assert_eq!(items[1].grid_key, "15_x47y523");
    }

    #[test]
    fn sanitize_tile_filename_and_noaa_edge_cases() {
        assert_eq!(
            sanitize_tile_filename(
                "https://prd-tnm.s3.amazonaws.com/StagedProducts/USGS_1M_14_x47y523.tif?sig=abc#top",
                "14_x47y523"
            ),
            "USGS_1M_14_x47y523.tif"
        );
        assert_eq!(
            sanitize_tile_filename("https://example.org/..", "14_x47y523"),
            "14_x47y523.tif"
        );
        assert_eq!(
            sanitize_tile_filename("https://example.org/.hidden", "14_x47y523"),
            "14_x47y523.tif"
        );
        assert_eq!(
            sanitize_tile_filename("https://example.org/", "14/x47?y523"),
            "14_x47_y523.tif"
        );

        // Reject all-null NOAA Atlas 14 response
        assert!(parse_noaa_atlas14_response(47.0, -99.0, "quantiles = [[null, null]];").is_err());

        // Cached NOAA Atlas 14 JSON matching coordinates is reused without network access
        let tmp = tempfile::tempdir().unwrap();
        let cache_path = tmp.path().join("noaa.json");
        let mut depths = BTreeMap::new();
        let mut row = BTreeMap::new();
        row.insert(100, 123.4);
        depths.insert("24-hr".to_string(), row);
        let tbl = NoaaAtlas14Table {
            lat: 47.1235,
            lon: -99.2346,
            units: "mm".to_string(),
            depths_mm: depths,
        };
        std::fs::write(&cache_path, serde_json::to_string(&tbl).unwrap()).unwrap();
        let loaded = fetch_noaa_atlas14(
            47.12349,
            -99.23458,
            Some(&cache_path),
            &DownloadOptions::default(),
        )
        .unwrap();
        assert_eq!(loaded, tbl);

        // Non-HTTPS downloadURL items are skipped
        let insecure = serde_json::json!({
            "items": [{
                "title": "USGS 1 Meter 14 x47y523 HTTP_Only",
                "publicationDate": "2024-01-01",
                "downloadURL": "http://example.org/insecure.tif",
                "sizeInBytes": 1000,
                "format": "GeoTIFF",
                "boundingBox": {"minX": -99.3, "minY": 47.0, "maxX": -99.1, "maxY": 47.2}
            }]
        });
        assert!(parse_tnm_products_json(&insecure, None, true).is_empty());

        // Manifest cache matcher detects region kind, bbox, and option changes
        let bb = Bbox::new(-99.2, 47.0, -99.0, 47.2).unwrap();
        let opts = DownloadOptions::default();
        let manifest_path = tmp.path().join("manifest.json");
        let manifest = FederalBundleResult {
            region_bbox: bb,
            region_key: "bbox".to_string(),
            dem_product: opts.dem_product,
            wbd_level: opts.wbd_level,
            clip_to_region: opts.clip_to_region,
            mosaic_dem: opts.mosaic_dem,
            transportation_layers: effective_transportation_layers(&opts),
            wbd_path: None,
            wbd_count: 0,
            dem_tiles: vec![],
            dem_mosaic_path: None,
            nhd_path: None,
            nhd_count: 0,
            nwi_path: None,
            nwi_count: 0,
            roads_path: None,
            roads_count: 0,
            culverts_path: None,
            culverts_count: 0,
            noaa_atlas14_path: None,
            noaa_atlas14: None,
            manifest_path: manifest_path.clone(),
        };
        std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();
        assert!(manifest_matches_request(&manifest_path, "bbox", &bb, &opts));
        // Same bbox but different region kind (VectorFile / Huc) must NOT match
        assert!(!manifest_matches_request(
            &manifest_path,
            "file:/tmp/study.geojson",
            &bb,
            &opts
        ));
        let other_bb = Bbox::new(-98.2, 46.0, -98.0, 46.2).unwrap();
        assert!(!manifest_matches_request(
            &manifest_path,
            "bbox",
            &other_bb,
            &opts
        ));
        let mut other_opts = opts.clone();
        other_opts.dem_product = DemProduct::OneArcSec;
        assert!(!manifest_matches_request(
            &manifest_path,
            "bbox",
            &bb,
            &other_opts
        ));
        let mut road_only_opts = opts.clone();
        road_only_opts.transportation_layers = vec![29, 30, 31, 32, 33];
        assert!(!manifest_matches_request(
            &manifest_path,
            "bbox",
            &bb,
            &road_only_opts
        ));
    }

    #[test]
    fn parse_noaa_atlas14_quantiles_response() {
        let mut rows = Vec::new();
        for i in 0..19 {
            let base = (i + 1) as f64 * 10.0;
            let vals: Vec<String> = (0..10)
                .map(|j| format!("'{:.2}'", base + j as f64))
                .collect();
            rows.push(format!("[{}]", vals.join(",")));
        }
        let body = format!(
            "result = 'ok';\nquantiles = [{}];\nupper = [];\n",
            rows.join(",")
        );
        let tbl = parse_noaa_atlas14_response(47.12345, -99.23456, &body).unwrap();
        assert_eq!(tbl.units, "mm");
        assert!((tbl.lat - 47.1235).abs() < 1e-6);
        assert!((tbl.lon - (-99.2346)).abs() < 1e-6);
        // 24-hr is index 9 (base = 100.0); 100-yr is ARI index 6 (+6.0 = 106.0)
        assert!((tbl.depth_mm("24-hr", 100).unwrap() - 106.0).abs() < 1e-6);
        // 2-yr is ARI index 1 (+1.0 = 101.0)
        assert!((tbl.depth_mm("24-hr", 2).unwrap() - 101.0).abs() < 1e-6);
    }

    #[test]
    fn write_geojson_and_convert_to_gpkg_and_read_bbox() {
        let tmp = tempfile::tempdir().unwrap();
        let fc = serde_json::json!({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"huc12": "101600020502", "name": "Pearl Lake"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-99.2, 47.0], [-99.0, 47.0], [-99.0, 47.2], [-99.2, 47.2], [-99.2, 47.0]
                    ]]
                }
            }]
        });
        let gpkg_path = tmp.path().join("wbd.gpkg");
        let n = write_geojson_feature_collection(&fc, &gpkg_path, "huc12", None).unwrap();
        assert_eq!(n, 1);

        let bb = vector_file_bbox_wgs84(&gpkg_path).unwrap();
        assert!((bb.min_lon - (-99.2)).abs() < 1e-6);
        assert!((bb.max_lon - (-99.0)).abs() < 1e-6);
        assert!((bb.min_lat - 47.0).abs() < 1e-6);
        assert!((bb.max_lat - 47.2).abs() < 1e-6);

        // Verify self-intersecting (bowtie) polygon is repaired cleanly before clipping
        let bowtie_fc = serde_json::json!({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"ATTRIBUTE": "PEM1C"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-99.18, 47.02], [-99.02, 47.18], [-99.02, 47.02],
                        [-99.18, 47.18], [-99.18, 47.02]
                    ]]
                }
            }]
        });
        let clipped_path = tmp.path().join("nwi_clipped.gpkg");
        let n_clip = write_geojson_feature_collection(
            &bowtie_fc,
            &clipped_path,
            "nwi_wetlands",
            Some(&gpkg_path),
        )
        .unwrap();
        assert_eq!(n_clip, 1);
    }

    #[test]
    fn stale_cross_region_cache_invalidation_cleans_bundle_files_but_keeps_tiles() {
        let tmp = tempfile::tempdir().unwrap();
        let out_dir = tmp.path();
        let tiles_dir = out_dir.join("tiles");
        std::fs::create_dir_all(&tiles_dir).unwrap();
        let cached_tile = tiles_dir.join("USGS_1M_14_x47y523.tif");
        std::fs::write(&cached_tile, b"tile-bytes").unwrap();

        let region_a = Bbox::new(-99.2, 47.0, -99.0, 47.2).unwrap();
        let opts_a = DownloadOptions::default();
        let manifest_path = out_dir.join("manifest.json");
        let manifest_a = FederalBundleResult {
            region_bbox: region_a,
            region_key: "bbox".to_string(),
            dem_product: opts_a.dem_product,
            wbd_level: opts_a.wbd_level,
            clip_to_region: opts_a.clip_to_region,
            mosaic_dem: opts_a.mosaic_dem,
            transportation_layers: effective_transportation_layers(&opts_a),
            wbd_path: Some(out_dir.join("wbd.gpkg")),
            wbd_count: 1,
            dem_tiles: vec![cached_tile.clone()],
            dem_mosaic_path: Some(out_dir.join("dem.tif")),
            nhd_path: Some(out_dir.join("nhd_flowlines.gpkg")),
            nhd_count: 1,
            nwi_path: Some(out_dir.join("nwi_wetlands.gpkg")),
            nwi_count: 1,
            roads_path: Some(out_dir.join("usgs_roads.gpkg")),
            roads_count: 1,
            culverts_path: Some(out_dir.join("culverts.gpkg")),
            culverts_count: 1,
            noaa_atlas14_path: Some(out_dir.join("noaa_atlas14.json")),
            noaa_atlas14: None,
            manifest_path: manifest_path.clone(),
        };
        std::fs::write(&manifest_path, serde_json::to_string(&manifest_a).unwrap()).unwrap();

        let stale_files = [
            "wbd.gpkg",
            "dem.tif",
            "nhd_flowlines.gpkg",
            "nwi_wetlands.gpkg",
            "usgs_roads.gpkg",
            "culverts.gpkg",
            "noaa_atlas14.json",
            ".dem.meta.json",
            "dem.tif.meta.json",
            "region_huc_10160002.geojson",
        ];
        for name in stale_files {
            std::fs::write(out_dir.join(name), b"stale-region-a").unwrap();
        }

        // Run download_federal_data for a DIFFERENT region (Region B) with an empty dataset subset
        // (simulating a partial request like [Dem] that leaves wbd.gpkg unrequested).
        let region_b = Bbox::new(-98.2, 46.0, -98.0, 46.2).unwrap();
        let res = download_federal_data(
            &Region::Bbox(region_b),
            out_dir,
            &FederalBundleOptions {
                datasets: Vec::new(),
                download: DownloadOptions::default(),
            },
        )
        .unwrap();
        assert_eq!(res.region_bbox, region_b);

        for name in stale_files {
            assert!(
                !out_dir.join(name).exists(),
                "stale Region A file '{name}' should have been removed on cache invalidation"
            );
        }
        assert!(
            cached_tile.exists(),
            "tile cache files must never be removed on bundle invalidation"
        );
        assert!(manifest_path.exists());
        assert!(manifest_matches_request(
            &manifest_path,
            "bbox",
            &region_b,
            &DownloadOptions::default()
        ));
    }

    #[test]
    fn no_manifest_preserves_user_files_and_keep_boundary_is_preserved_across_path_spellings() {
        // (a) A directory with NO prior manifest.json must never have pre-existing user files
        // deleted when download_federal_data initializes it.
        let user_dir = tempfile::tempdir().unwrap();
        let user_dem = user_dir.path().join("dem.tif");
        let user_wbd = user_dir.path().join("wbd.gpkg");
        std::fs::write(&user_dem, b"user-owned-dem").unwrap();
        std::fs::write(&user_wbd, b"user-owned-wbd").unwrap();

        let bb = Bbox::new(-98.2, 46.0, -98.0, 46.2).unwrap();
        download_federal_data(
            &Region::Bbox(bb),
            user_dir.path(),
            &FederalBundleOptions {
                datasets: Vec::new(),
                download: DownloadOptions::default(),
            },
        )
        .unwrap();
        assert_eq!(std::fs::read(&user_dem).unwrap(), b"user-owned-dem");
        assert_eq!(std::fs::read(&user_wbd).unwrap(), b"user-owned-wbd");

        // (b) A mismatched valid manifest deletes ONLY paths recorded in that manifest, keeping
        // unrecorded files in out_dir untouched.
        let partial_dir = tempfile::tempdir().unwrap();
        let recorded_nhd = partial_dir.path().join("nhd_flowlines.gpkg");
        let unrecorded_user_dem = partial_dir.path().join("dem.tif");
        std::fs::write(&recorded_nhd, b"old-nhd").unwrap();
        std::fs::write(&unrecorded_user_dem, b"user-dem").unwrap();
        let region_a = Bbox::new(-99.2, 47.0, -99.0, 47.2).unwrap();
        let manifest_path = partial_dir.path().join("manifest.json");
        let opts_a = DownloadOptions::default();
        let partial_manifest = FederalBundleResult {
            region_bbox: region_a,
            region_key: "bbox".to_string(),
            dem_product: opts_a.dem_product,
            wbd_level: opts_a.wbd_level,
            clip_to_region: opts_a.clip_to_region,
            mosaic_dem: opts_a.mosaic_dem,
            transportation_layers: effective_transportation_layers(&opts_a),
            wbd_path: None,
            wbd_count: 0,
            dem_tiles: Vec::new(),
            dem_mosaic_path: None,
            nhd_path: Some(recorded_nhd.clone()),
            nhd_count: 1,
            nwi_path: None,
            nwi_count: 0,
            roads_path: None,
            roads_count: 0,
            culverts_path: None,
            culverts_count: 0,
            noaa_atlas14_path: None,
            noaa_atlas14: None,
            manifest_path: manifest_path.clone(),
        };
        std::fs::write(
            &manifest_path,
            serde_json::to_string(&partial_manifest).unwrap(),
        )
        .unwrap();
        clean_stale_bundle_outputs(partial_dir.path(), &manifest_path, None);
        assert!(
            !recorded_nhd.exists(),
            "manifest-recorded NHD should be removed"
        );
        assert!(!manifest_path.exists());
        assert!(
            unrecorded_user_dem.exists(),
            "unrecorded user dem.tif must not be deleted"
        );

        // (b2) A crafted or copied manifest pointing outside out_dir must NEVER delete external
        // files outside out_dir.
        let external_dir = tempfile::tempdir().unwrap();
        let external_file = external_dir.path().join("external_important.tif");
        std::fs::write(&external_file, b"must-not-be-deleted").unwrap();
        let crafted_dir = tempfile::tempdir().unwrap();
        let crafted_manifest_path = crafted_dir.path().join("manifest.json");
        let crafted_manifest = FederalBundleResult {
            region_bbox: region_a,
            region_key: "bbox".to_string(),
            dem_product: opts_a.dem_product,
            wbd_level: opts_a.wbd_level,
            clip_to_region: opts_a.clip_to_region,
            mosaic_dem: opts_a.mosaic_dem,
            transportation_layers: effective_transportation_layers(&opts_a),
            wbd_path: Some(external_file.clone()),
            wbd_count: 1,
            dem_tiles: Vec::new(),
            dem_mosaic_path: Some(external_file.clone()),
            nhd_path: None,
            nhd_count: 0,
            nwi_path: None,
            nwi_count: 0,
            roads_path: None,
            roads_count: 0,
            culverts_path: None,
            culverts_count: 0,
            noaa_atlas14_path: None,
            noaa_atlas14: None,
            manifest_path: crafted_manifest_path.clone(),
        };
        std::fs::write(
            &crafted_manifest_path,
            serde_json::to_string(&crafted_manifest).unwrap(),
        )
        .unwrap();
        clean_stale_bundle_outputs(crafted_dir.path(), &crafted_manifest_path, None);
        assert!(
            external_file.exists(),
            "manifest pointing outside out_dir must never delete external files"
        );
        assert!(!crafted_manifest_path.exists());

        // (c) keep_boundary pointing to out_dir/wbd.gpkg or aliased via ./ must never be deleted
        // when cleaning a valid mismatched manifest.
        let keep_dir = tempfile::tempdir().unwrap();
        let keep_wbd = keep_dir.path().join("wbd.gpkg");
        let keep_huc = keep_dir.path().join("region_huc_10160002.geojson");
        let stale_nhd = keep_dir.path().join("nhd_flowlines.gpkg");
        let valid_manifest_path = keep_dir.path().join("manifest.json");
        std::fs::write(&keep_wbd, b"active-boundary-wbd").unwrap();
        std::fs::write(&keep_huc, b"active-huc-boundary").unwrap();
        std::fs::write(&stale_nhd, b"stale-nhd").unwrap();
        let valid_manifest = FederalBundleResult {
            region_bbox: region_a,
            region_key: "bbox".to_string(),
            dem_product: opts_a.dem_product,
            wbd_level: opts_a.wbd_level,
            clip_to_region: opts_a.clip_to_region,
            mosaic_dem: opts_a.mosaic_dem,
            transportation_layers: effective_transportation_layers(&opts_a),
            wbd_path: Some(keep_wbd.clone()),
            wbd_count: 1,
            dem_tiles: Vec::new(),
            dem_mosaic_path: None,
            nhd_path: Some(stale_nhd.clone()),
            nhd_count: 1,
            nwi_path: None,
            nwi_count: 0,
            roads_path: None,
            roads_count: 0,
            culverts_path: None,
            culverts_count: 0,
            noaa_atlas14_path: None,
            noaa_atlas14: None,
            manifest_path: valid_manifest_path.clone(),
        };
        std::fs::write(
            &valid_manifest_path,
            serde_json::to_string(&valid_manifest).unwrap(),
        )
        .unwrap();

        let aliased_wbd = keep_dir.path().join(".").join("wbd.gpkg");
        clean_stale_bundle_outputs(keep_dir.path(), &valid_manifest_path, Some(&aliased_wbd));
        assert!(
            keep_wbd.exists(),
            "active wbd.gpkg boundary must be preserved"
        );
        assert!(
            !stale_nhd.exists(),
            "stale nhd recorded in valid manifest should be removed"
        );
        assert!(
            !keep_huc.exists(),
            "unkept region_huc file should be removed when valid manifest is invalidated"
        );

        // Re-create valid manifest and verify aliased region_huc_ keep_boundary is preserved.
        std::fs::write(&keep_huc, b"active-huc-boundary").unwrap();
        std::fs::write(
            &valid_manifest_path,
            serde_json::to_string(&valid_manifest).unwrap(),
        )
        .unwrap();
        let aliased_huc = keep_dir
            .path()
            .join(".")
            .join("region_huc_10160002.geojson");
        clean_stale_bundle_outputs(keep_dir.path(), &valid_manifest_path, Some(&aliased_huc));
        assert!(
            keep_huc.exists(),
            "active region_huc boundary must be preserved across aliased paths"
        );

        // (d) When manifest.json is corrupt/unparseable, ONLY manifest.json itself is removed so
        // arbitrary unverified files in out_dir are never deleted.
        let corrupt_dir = tempfile::tempdir().unwrap();
        let unverified_dem = corrupt_dir.path().join("dem.tif");
        let unverified_wbd = corrupt_dir.path().join("wbd.gpkg");
        let corrupt_manifest = corrupt_dir.path().join("manifest.json");
        std::fs::write(&unverified_dem, b"keep-dem").unwrap();
        std::fs::write(&unverified_wbd, b"keep-wbd").unwrap();
        std::fs::write(&corrupt_manifest, b"{not valid json").unwrap();
        clean_stale_bundle_outputs(corrupt_dir.path(), &corrupt_manifest, None);
        assert!(!corrupt_manifest.exists());
        assert!(
            unverified_dem.exists(),
            "corrupt manifest must not delete unverified dem.tif"
        );
        assert!(
            unverified_wbd.exists(),
            "corrupt manifest must not delete unverified wbd.gpkg"
        );
    }

    /// Reads an HTTP/1.1 request (headers up to `\r\n\r\n` plus the exact `Content-Length` body)
    /// from `stream` before replying, avoiding Winsock TCP RST (`os error 10053`) on Windows when
    /// closing a connection while unread POST body bytes remain in flight.
    fn drain_http_request_body(stream: &mut std::net::TcpStream) -> String {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut req_bytes = Vec::new();
        let mut tmp_buf = [0u8; 1024];
        let mut header_end = None;
        let mut content_len = 0usize;
        loop {
            let n = stream.read(&mut tmp_buf).unwrap_or(0);
            if n == 0 {
                break;
            }
            req_bytes.extend_from_slice(&tmp_buf[..n]);
            if header_end.is_none()
                && let Some(pos) = req_bytes.windows(4).position(|w| w == b"\r\n\r\n")
            {
                header_end = Some(pos + 4);
                let hdr = String::from_utf8_lossy(&req_bytes[..pos]);
                for line in hdr.lines() {
                    if let Some((k, v)) = line.split_once(':')
                        && k.trim().eq_ignore_ascii_case("content-length")
                    {
                        content_len = v.trim().parse::<usize>().unwrap_or(0);
                    }
                }
            }
            if let Some(he) = header_end
                && req_bytes.len() >= he + content_len
            {
                break;
            }
        }
        header_end
            .map(|he| String::from_utf8_lossy(&req_bytes[he..]).into_owned())
            .unwrap_or_default()
    }

    #[test]
    fn tnm_size_in_bytes_zero_parses_as_none_and_content_length_verified() {
        let json = serde_json::json!({
            "items": [
                {
                    "title": "USGS 1 Meter 14 x47y523 ZeroSize",
                    "publicationDate": "2024-01-01",
                    "downloadURL": "https://example.org/zero.tif",
                    "sizeInBytes": 0,
                    "format": "GeoTIFF",
                    "boundingBox": {"minX": -99.3, "minY": 47.0, "maxX": -99.1, "maxY": 47.2}
                },
                {
                    "title": "USGS 1 Meter 14 x48y523 KnownSize",
                    "publicationDate": "2024-01-01",
                    "downloadURL": "https://example.org/known.tif",
                    "sizeInBytes": 4096,
                    "format": "GeoTIFF",
                    "boundingBox": {"minX": -99.3, "minY": 47.0, "maxX": -99.1, "maxY": 47.2}
                }
            ]
        });
        let items = parse_tnm_products_json(&json, None, false);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].size_bytes, None);
        assert_eq!(items[1].size_bytes, Some(4096));

        // Verify http_download_file enforces Content-Length when expected_bytes is None and retries
        // on a truncated transfer.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let _ = drain_http_request_body(&mut stream);
                if attempt == 0 {
                    // Claim 16 bytes but send only 5 bytes before closing.
                    let _ = stream.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\nhello",
                    );
                } else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\n0123456789abcdef",
                    );
                }
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Write);
            }
        });

        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("tile.tif");
        let agent = make_download_agent(10);
        let url = format!("http://{addr}/tile.tif");
        let saved = http_download_file(&agent, &url, &dest, None, true, 1).unwrap();
        server.join().unwrap();
        assert_eq!(std::fs::read(&saved).unwrap(), b"0123456789abcdef");
    }

    #[test]
    fn arcgis_oid_bisection_avoids_empty_amplification_and_recovers_sub_1000_caps() {
        // Never bisect empty chunks when exceeded is false.
        assert!(!should_bisect_arcgis_chunk(false, 0, 50, 4));
        // Full chunk never bisects.
        assert!(!should_bisect_arcgis_chunk(false, 50, 50, 4));
        // Short page below 1000 bisects while non-exceeded budget remains, then stops at 0.
        assert!(should_bisect_arcgis_chunk(false, 250, 500, 4));
        assert!(!should_bisect_arcgis_chunk(false, 250, 500, 0));
        // Explicit exceededTransferLimit always bisects for chunk_len > 1 even at budget 0.
        assert!(should_bisect_arcgis_chunk(true, 10, 50, 0));
        assert!(!should_bisect_arcgis_chunk(true, 0, 1, 4));

        // Live local HTTP check: (1) empty chunk of 20 OIDs sends exactly 1 request; (2) silent
        // cap at 2 features per request (without exceededTransferLimit) bisects and recovers all 4.
        let empty_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let empty_addr = empty_listener.local_addr().unwrap();
        let empty_hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let empty_hits_srv = empty_hits.clone();
        let empty_srv = std::thread::spawn(move || {
            let (mut stream, _) = empty_listener.accept().unwrap();
            empty_hits_srv.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = drain_http_request_body(&mut stream);
            let body = r#"{"type":"FeatureCollection","features":[]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);
        });
        let agent = make_agent(10);
        let empty_oids: Vec<i64> = (1..=20).collect();
        let empty_res = fetch_arcgis_oid_chunk(
            &agent,
            &format!("http://{empty_addr}/query"),
            &empty_oids,
            "*",
            0,
        )
        .unwrap();
        empty_srv.join().unwrap();
        assert!(empty_res.is_empty());
        assert_eq!(empty_hits.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Silent cap at 2 features per response for 4 requested OIDs -> initial request gets 2,
        // bisects into 2 + 2, and recovers all 4 features.
        let cap_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let cap_addr = cap_listener.local_addr().unwrap();
        let cap_srv = std::thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = cap_listener.accept().unwrap();
                let body_part = drain_http_request_body(&mut stream);
                let oids: Vec<i64> = body_part
                    .split('&')
                    .find_map(|kv| kv.strip_prefix("objectIds="))
                    .unwrap_or("")
                    .split("%2C")
                    .filter_map(|s| s.parse::<i64>().ok())
                    .collect();
                let capped: Vec<Value> = oids
                    .into_iter()
                    .take(2)
                    .map(|id| serde_json::json!({"type": "Feature", "id": id, "properties": {"id": id}, "geometry": null}))
                    .collect();
                let body = serde_json::json!({"type": "FeatureCollection", "features": capped})
                    .to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Write);
            }
        });
        let cap_res = fetch_arcgis_oid_chunk(
            &agent,
            &format!("http://{cap_addr}/query"),
            &[1, 2, 3, 4],
            "*",
            0,
        )
        .unwrap();
        cap_srv.join().unwrap();
        assert_eq!(cap_res.len(), 4);
    }

    #[test]
    fn bilinear_mosaic_masks_nodata_seam_without_blending() {
        let tmp = tempfile::tempdir().unwrap();
        let tile_a = tmp.path().join("tile_a.tif");
        let tile_b = tmp.path().join("tile_b.tif");
        let mosaic_out = tmp.path().join("mosaic.tif");

        let nd = -9999.0_f64;
        // Left tile: valid 100.0 on columns 0..3, NoData (-9999.0) on right seam column 3.
        let mut data_a = vec![100.0_f64; 4 * 4];
        for r in 0..4 {
            data_a[r * 4 + 3] = nd;
        }
        let mut arr_a = richdem::Array2D::from_vec(4, 4, data_a).unwrap();
        arr_a.set_no_data(nd);
        arr_a.geotransform = [500_000.0, 1.0, 0.0, 4_500_004.0, 0.0, -1.0];
        arr_a.projection = "EPSG:32615".to_string();
        richdem::save_raster(&arr_a, &tile_a, None, false).unwrap();

        // Right tile shifted by 0.5 px (forces bilinear resampling) with NoData on its left edge.
        let mut data_b = vec![120.0_f64; 4 * 4];
        for r in 0..4 {
            data_b[r * 4] = nd;
        }
        let mut arr_b = richdem::Array2D::from_vec(4, 4, data_b).unwrap();
        arr_b.set_no_data(nd);
        arr_b.geotransform = [500_002.5, 1.0, 0.0, 4_500_004.0, 0.0, -1.0];
        arr_b.projection = "EPSG:32615".to_string();
        richdem::save_raster(&arr_b, &tile_b, None, false).unwrap();

        mosaic_with_resampling(&[&tile_a, &tile_b], &mosaic_out, Resampling::Bilinear).unwrap();
        let mosaicked = richdem::load_raster::<f64>(&mosaic_out, None).unwrap();
        let out_nd = mosaicked.no_data();
        let mut valid_count = 0usize;
        for &v in mosaicked.data() {
            if !v.is_finite() || (v - out_nd).abs() < 1e-6 {
                continue;
            }
            valid_count += 1;
            assert!(
                (99.0..=121.0).contains(&v),
                "bilinear mosaic blended NoData (-9999.0) into valid pixel: {v}"
            );
        }
        assert!(valid_count > 0);
    }

    #[test]
    fn retry_classification_tile_filename_disambiguation_and_noaa_comment_parsing() {
        assert!(is_retryable_ureq_error(&ureq::Error::HostNotFound));
        assert!(!is_retryable_ureq_error(&ureq::Error::BadUri("bad".into())));
        assert!(!is_retryable_ureq_error(&ureq::Error::Tls("cert failed")));
        assert!(!is_retryable_ureq_error(&ureq::Error::StatusCode(400)));
        assert!(!is_retryable_ureq_error(&ureq::Error::StatusCode(404)));
        assert!(is_retryable_ureq_error(&ureq::Error::StatusCode(408)));
        assert!(is_retryable_ureq_error(&ureq::Error::StatusCode(429)));
        assert!(is_retryable_ureq_error(&ureq::Error::StatusCode(503)));

        let bb = Bbox::new(-99.3, 47.0, -99.1, 47.2).unwrap();
        let products = vec![
            TnmProductItem {
                grid_key: "14_x47y523".into(),
                title: "A".into(),
                publication_date: "2023-01-01".into(),
                download_url: "https://example.org/a/tile.tif".into(),
                size_bytes: None,
                format: "GeoTIFF".into(),
                bbox: bb,
            },
            TnmProductItem {
                grid_key: "14_x48y523".into(),
                title: "B".into(),
                publication_date: "2023-01-01".into(),
                download_url: "https://example.org/b/tile.tif".into(),
                size_bytes: None,
                format: "GeoTIFF".into(),
                bbox: bb,
            },
            TnmProductItem {
                grid_key: "14_x49y523".into(),
                title: "C".into(),
                publication_date: "2023-01-01".into(),
                download_url: "https://example.org/c/unique.tif".into(),
                size_bytes: None,
                format: "GeoTIFF".into(),
                bbox: bb,
            },
        ];
        let names = assign_unique_tile_filenames(&products);
        assert_eq!(names.len(), 3);
        assert_ne!(names[0], names[1]);
        assert!(names[0].ends_with(".tif") && names[1].ends_with(".tif"));
        assert_eq!(names[2], "unique.tif");

        // Comment mentioning "quantiles" and prefixed "upper_quantiles" must not hijack parsing.
        let body = "// Note: quantiles are reported in metric mm; see [[0]]\n\
                    upper_quantiles = [['999.0']];\n\
                    quantiles = [['12.5', '18.0']];\n";
        let tbl = parse_noaa_atlas14_response(47.0, -99.0, body).unwrap();
        assert!((tbl.depth_mm("5-min", 1).unwrap() - 12.5).abs() < 1e-6);
        assert!((tbl.depth_mm("5-min", 2).unwrap() - 18.0).abs() < 1e-6);
    }
}
