//! Command-line interface and MCP server for `lidar-rs`: DEM filtering, sink extraction,
//! nested surface depression and mount delineation, and geospatial utilities
//! in pure Rust (powered by `terrio` and `richdem`).

mod mcp;
mod usage;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use lidar::pointcloud::{
    GridMethod, GridStat, GroundFilterMethod, LidarToDsmOptions, LidarToDtmOptions,
    PointWriteOptions, ReturnKind, classify_ground, convert_lidar, lidar_to_chm, lidar_to_dsm,
    lidar_to_dtm, read_lidar, write_lidar,
};
use lidar::{
    AnimateInundationOptions, BreachMode, BreachOptions, BurnCulvertsOptions,
    CatchmentHierarchyOptions, DelineateCatchmentsOptions, DelineateFlowPathsOptions,
    DelineateOptions, ExtractSinksOptions, ForcingInput, HillshadeOptions, RoutingMode,
    RunPipelineOptions, SimulateInundationAdvancedOptions, SimulateInundationOptions,
    SynthesizeCulvertsOptions, TiledDelineationOptions, Topology, add_crs, animate_inundation,
    breach_depressions_file, burn_culverts_file, clip_image, csv_points_to_vector,
    delineate_catchment_hierarchy, delineate_catchments, delineate_depressions,
    delineate_depressions_tiled, delineate_flow_paths, delineate_mounts,
    export_inundation_frames_png, extract_sinks, flip_dem_file, gaussian_filter_file, image_to_cog,
    join_csv_to_vector, load_dem_for_preview, load_raster, mean_filter_file, median_filter_file,
    mosaic, parse_resampling, polygonize, render_dem_preview_png_bytes,
    render_inundation_frame_png_bytes, reproject_image, resample, run_pipeline, save_raster,
    scs_curve_number_runoff, scs_type2_hyetograph, simulate_inundation,
    simulate_inundation_advanced, synthesize_culverts,
};
#[cfg(feature = "download")]
use lidar::{
    Bbox, DemProduct, DownloadOptions, FederalBundleOptions, Region, WbdLevel,
    download_federal_data, fetch_noaa_atlas14, parse_dataset_list, query_3dep_products,
};
use richdem::DemNum;
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "lidar",
    version,
    about = "Nested surface depression and mount delineation from DEMs in pure Rust",
    long_about = "Pure-Rust port of opengeos/lidar (Wu et al. 2019; Wu 2021) powered by terrio and richdem, with zero GDAL dependencies."
)]
pub(crate) struct Cli {
    /// Print the versioned JSON command catalog without running a command
    #[arg(id = "json_usage", long = "json-usage", global = true, hide = true)]
    _json_usage: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, ValueEnum)]
enum CliTopology {
    D8,
    D4,
}

impl From<CliTopology> for Topology {
    fn from(t: CliTopology) -> Self {
        match t {
            CliTopology::D8 => Topology::D8,
            CliTopology::D4 => Topology::D4,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum CliBreachMode {
    Complete,
    Selective,
    Constrained,
}

impl From<CliBreachMode> for BreachMode {
    fn from(m: CliBreachMode) -> Self {
        match m {
            CliBreachMode::Complete => BreachMode::Complete,
            CliBreachMode::Selective => BreachMode::Selective,
            CliBreachMode::Constrained => BreachMode::Constrained,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum CliRoutingMode {
    Static,
    Fsm,
}

impl From<CliRoutingMode> for RoutingMode {
    fn from(m: CliRoutingMode) -> Self {
        match m {
            CliRoutingMode::Static => RoutingMode::Static,
            CliRoutingMode::Fsm => RoutingMode::FillSpillMerge,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum CliGroundMethod {
    Preclassified,
    Csf,
    Pmf,
    Smrf,
}

impl From<CliGroundMethod> for GroundFilterMethod {
    fn from(m: CliGroundMethod) -> Self {
        match m {
            CliGroundMethod::Preclassified => GroundFilterMethod::Preclassified,
            CliGroundMethod::Csf => GroundFilterMethod::Csf,
            CliGroundMethod::Pmf => GroundFilterMethod::Pmf,
            CliGroundMethod::Smrf => GroundFilterMethod::Smrf,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum CliClassifyGroundMethod {
    Csf,
    Pmf,
    Smrf,
}

impl CliClassifyGroundMethod {
    fn as_str(self) -> &'static str {
        match self {
            CliClassifyGroundMethod::Csf => "csf",
            CliClassifyGroundMethod::Pmf => "pmf",
            CliClassifyGroundMethod::Smrf => "smrf",
        }
    }
}

impl From<CliClassifyGroundMethod> for GroundFilterMethod {
    fn from(m: CliClassifyGroundMethod) -> Self {
        match m {
            CliClassifyGroundMethod::Csf => GroundFilterMethod::Csf,
            CliClassifyGroundMethod::Pmf => GroundFilterMethod::Pmf,
            CliClassifyGroundMethod::Smrf => GroundFilterMethod::Smrf,
        }
    }
}

fn print_json(value: &impl serde::Serialize) -> lidar::Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).map_err(|e| lidar::Error::Algorithm(e.to_string()))?
    );
    Ok(())
}

#[cfg(feature = "download")]
fn parse_u32_list(s: &str) -> lidar::Result<Vec<u32>> {
    let parsed: Vec<u32> = s
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|_| lidar::Error::InvalidArgument(format!("invalid layer ID: '{part}'")))
        })
        .collect::<lidar::Result<Vec<_>>>()?;
    if parsed.is_empty() {
        return Err(lidar::Error::InvalidArgument(
            "transportation layer list cannot be empty".into(),
        ));
    }
    Ok(parsed)
}

fn parse_u8_list(s: &str) -> lidar::Result<Vec<u8>> {
    let parsed: Vec<u8> = s
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim().parse::<u8>().map_err(|_| {
                lidar::Error::InvalidArgument(format!(
                    "invalid ASPRS classification code: '{part}'"
                ))
            })
        })
        .collect::<lidar::Result<Vec<_>>>()?;
    if parsed.is_empty() {
        return Err(lidar::Error::InvalidArgument(
            "ASPRS classification code list cannot be empty".into(),
        ));
    }
    Ok(parsed)
}

fn parse_return_kinds(s: &str) -> lidar::Result<Vec<ReturnKind>> {
    lidar::pointcloud::parse_return_kinds(s)
}

#[derive(Subcommand)]
enum Commands {
    /// Apply a moving-window mean filter to a DEM
    MeanFilter {
        /// Input DEM raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Square kernel size in pixels
        #[arg(short = 'k', long, default_value_t = 3)]
        kernel_size: usize,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Apply a moving-window median filter to a DEM
    MedianFilter {
        /// Input DEM raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Square kernel size in pixels
        #[arg(short = 'k', long, default_value_t = 3)]
        kernel_size: usize,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Apply a 2D Gaussian smoothing filter to a DEM
    GaussianFilter {
        /// Input DEM raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Standard deviation (sigma) in pixels
        #[arg(short = 's', long, default_value_t = 1.0)]
        sigma: f64,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Extract surface depressions (sinks) and generate a refined DEM
    ExtractSinks {
        /// Input DEM raster
        input: PathBuf,
        /// Output directory
        out_dir: PathBuf,
        /// Minimum number of pixels for a depression
        #[arg(short = 'm', long, default_value_t = 1000)]
        min_size: usize,
        /// Optional pre-filled DEM raster
        #[arg(long)]
        filled_dem: Option<PathBuf>,
        /// Connectivity topology for depression filling (d8 or d4)
        #[arg(short = 't', long, value_enum, default_value_t = CliTopology::D8)]
        topology: CliTopology,
        /// Remove intermediate files and keep only the joined GeoPackage
        #[arg(long)]
        drop_intermediates: bool,
        /// Compress output GeoTIFFs with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Delineate nested surface depression hierarchy using the level-set method (Wu et al. 2019)
    DelineateDepressions {
        /// Input sink raster (e.g., sink.tif produced by extract-sinks)
        in_sink: PathBuf,
        /// Output directory
        out_dir: PathBuf,
        /// Minimum number of pixels for a depression
        #[arg(short = 'm', long, default_value_t = 1000)]
        min_size: usize,
        /// Minimum depth of a depression
        #[arg(short = 'd', long, default_value_t = 0.3)]
        min_depth: f64,
        /// Vertical slicing interval (positive; sliced top-down)
        #[arg(short = 'i', long, default_value_t = 0.3)]
        interval: f64,
        /// Export individual shapefiles per hierarchy level under shp-level/
        #[arg(long)]
        level_shp: bool,
        /// Also export combined level GeoTIFFs when --level-shp is set
        #[arg(long, requires = "level_shp")]
        comb_tif: bool,
        /// Also export a unified multi-level GeoPackage (depressions.gpkg) with full attribute names
        #[arg(long)]
        gpkg: bool,
        /// Compress output GeoTIFFs with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Delineate nested hierarchy of elevated features (mounts / peaks / canopies)
    DelineateMounts {
        /// Input DEM or DSM raster
        input: PathBuf,
        /// Output directory
        out_dir: PathBuf,
        /// Minimum number of pixels for a mount
        #[arg(short = 'm', long, default_value_t = 1000)]
        min_size: usize,
        /// Minimum height of a mount
        #[arg(short = 'H', long, default_value_t = 0.3)]
        min_height: f64,
        /// Vertical slicing interval
        #[arg(short = 'i', long, default_value_t = 0.3)]
        interval: f64,
        /// Export individual shapefiles per hierarchy level
        #[arg(long)]
        level_shp: bool,
        /// Also export combined level GeoTIFFs when --level-shp is set
        #[arg(long, requires = "level_shp")]
        comb_tif: bool,
        /// Also export a unified multi-level GeoPackage with full attribute names
        #[arg(long)]
        gpkg: bool,
        /// Compress output GeoTIFFs with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Invert (flip) a DEM upside-down so elevated features become depressions
    FlipDem {
        /// Input DEM raster
        input: PathBuf,
        /// Output inverted DEM raster
        output: PathBuf,
        /// Base elevation offset added to the flipped DEM
        #[arg(long, default_value_t = 100.0, allow_hyphen_values = true)]
        delta: f64,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Polygonize a raster band into a vector file (Shapefile, GeoPackage, GeoJSON, etc.)
    Polygonize {
        /// Input raster
        input: PathBuf,
        /// Output vector file
        output: PathBuf,
        /// Output attribute field name
        #[arg(short = 'f', long, default_value = "id")]
        field_name: String,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Mosaic multiple rasters into a single GeoTIFF
    Mosaic {
        /// Input raster files
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output mosaic GeoTIFF
        #[arg(short = 'o', long)]
        output: PathBuf,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Reproject a raster to a target CRS
    Reproject {
        /// Input raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Destination CRS (e.g., EPSG:4326, EPSG:5070, WKT, or PROJ)
        #[arg(long)]
        crs: String,
        /// Resampling method (nearest, bilinear, cubic, etc.)
        #[arg(short = 'r', long, default_value = "nearest")]
        resampling: String,
        /// Optional target pixel resolution
        #[arg(long)]
        resolution: Option<f64>,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Resample a raster to a target pixel resolution
    Resample {
        /// Input raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Target pixel resolution
        #[arg(long)]
        resolution: f64,
        /// Resampling method (nearest, bilinear, cubic, etc.)
        #[arg(short = 'r', long, default_value = "bilinear")]
        resampling: String,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Clip a raster by a polygon vector mask
    Clip {
        /// Input raster
        input: PathBuf,
        /// Polygon mask vector file (GeoJSON, Shapefile, GeoPackage, etc.)
        mask: PathBuf,
        /// Output clipped raster
        output: PathBuf,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Convert a raster to a Cloud Optimized GeoTIFF (COG)
    ToCog {
        /// Input raster
        input: PathBuf,
        /// Output COG file
        output: PathBuf,
        /// Compression codec (DEFLATE, LZW, ZSTD, etc.)
        #[arg(short = 'c', long, default_value = "DEFLATE")]
        compression: String,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Assign a CRS to a raster without reprojecting pixels
    AddCrs {
        /// Input raster
        input: PathBuf,
        /// CRS string or numeric EPSG code (e.g., 4326 or EPSG:5070)
        crs: String,
        /// Output raster (overwrites input in place if omitted)
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Join a CSV attribute table onto a vector file
    JoinTables {
        /// Input vector file
        in_vector: PathBuf,
        /// Input CSV file
        in_csv: PathBuf,
        /// Output vector file (.gpkg, .shp, .geojson, etc.)
        out_vector: PathBuf,
        /// Join column in vector file
        #[arg(long, default_value = "id")]
        vec_col: String,
        /// Join column in CSV file
        #[arg(long, default_value = "region_id")]
        csv_col: String,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Convert a CSV file with latitude/longitude columns into a WGS 84 point vector file
    CsvToVector {
        /// Input CSV file
        in_csv: PathBuf,
        /// Output vector file (.shp, .gpkg, .geojson, etc.)
        out_vector: PathBuf,
        /// Latitude column name
        #[arg(long, default_value = "latitude")]
        latitude: String,
        /// Longitude column name
        #[arg(long, default_value = "longitude")]
        longitude: String,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Convert a LiDAR point cloud between formats (LAS, LAZ, COPC, PLY, E57)
    ConvertLidar {
        /// Input point cloud file
        input: PathBuf,
        /// Output point cloud file
        output: PathBuf,
        /// Optional LAS point data record format (0..=10)
        #[arg(long)]
        point_format: Option<u8>,
        /// Optional LAS minor version (e.g. 4 for LAS 1.4)
        #[arg(long)]
        minor_version: Option<u8>,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Classify ground points in a LiDAR point cloud using CSF, PMF, or SMRF
    ClassifyGround {
        /// Input point cloud file (LAS, LAZ, COPC, PLY, E57)
        input: PathBuf,
        /// Output classified point cloud file
        output: PathBuf,
        /// Ground classification algorithm (csf, pmf, smrf)
        #[arg(long, value_enum, default_value_t = CliClassifyGroundMethod::Csf)]
        method: CliClassifyGroundMethod,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Grid a LiDAR point cloud into a Digital Surface Model (DSM) GeoTIFF
    LidarToDsm {
        /// Input point cloud file (LAS, LAZ, COPC, PLY, E57)
        input: PathBuf,
        /// Output DSM GeoTIFF
        output: PathBuf,
        /// Output pixel resolution
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
        /// Optional search radius around each cell center
        #[arg(long)]
        radius: Option<f64>,
        /// Optional minimum Z filter
        #[arg(long, allow_hyphen_values = true)]
        min_z: Option<f64>,
        /// Optional maximum Z filter
        #[arg(long, allow_hyphen_values = true)]
        max_z: Option<f64>,
        /// Optional ASPRS classification codes to keep (comma-separated, e.g. "2" or "2,6")
        #[arg(long)]
        classification: Option<String>,
        /// Optional pulse return kinds to keep (comma-separated: first,last,intermediate,only)
        #[arg(long)]
        return_kind: Option<String>,
        /// Grid statistic (max, min, mean, idw, count, stdev)
        #[arg(long, default_value = "max")]
        stat: String,
        /// Gridding method (statistics, linear, nearest)
        #[arg(long, default_value = "statistics")]
        method: String,
        /// Gap-filling window size in cells (0 = none)
        #[arg(long, default_value_t = 0)]
        window_size: usize,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Generate a bare-earth Digital Terrain Model (DTM) from a LiDAR point cloud using CSF/PMF/SMRF/preclassified ground points
    LidarToDtm {
        /// Input point cloud file (LAS, LAZ, COPC, PLY, E57)
        input: PathBuf,
        /// Output DTM GeoTIFF
        output: PathBuf,
        /// Ground classification method (preclassified, csf, pmf, smrf)
        #[arg(long, value_enum, default_value_t = CliGroundMethod::Csf)]
        ground_method: CliGroundMethod,
        /// ASPRS ground class code (default 2)
        #[arg(long, default_value_t = 2)]
        ground_class: u8,
        /// Output pixel resolution
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
        /// Optional search radius around each cell center
        #[arg(long)]
        radius: Option<f64>,
        /// Gridding method (linear, statistics, nearest)
        #[arg(long, default_value = "linear")]
        method: String,
        /// Grid statistic when --method=statistics (idw, min, mean, max, count, stdev)
        #[arg(long, default_value = "idw")]
        stat: String,
        /// Gap-filling window size in cells
        #[arg(long, default_value_t = 5)]
        window_size: usize,
        /// Strip ASPRS low/high noise outliers via ELM before ground selection
        #[arg(long)]
        remove_outliers: bool,
        /// ELM cell size in CRS horizontal units when --remove-outliers is set
        #[arg(long, default_value_t = 10.0)]
        outlier_cell: f64,
        /// ELM elevation drop threshold in CRS vertical units when --remove-outliers is set
        #[arg(long, default_value_t = 1.0)]
        outlier_threshold: f64,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Generate a Canopy Height Model (CHM = DSM - DTM) from a LiDAR point cloud
    LidarToChm {
        /// Input point cloud file (LAS, LAZ, COPC, PLY, E57)
        input: PathBuf,
        /// Output CHM GeoTIFF
        output: PathBuf,
        /// Ground classification method (preclassified, csf, pmf, smrf)
        #[arg(long, value_enum, default_value_t = CliGroundMethod::Csf)]
        ground_method: CliGroundMethod,
        /// ASPRS ground class code (default 2)
        #[arg(long, default_value_t = 2)]
        ground_class: u8,
        /// Output pixel resolution
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
        /// Optional search radius around each cell center
        #[arg(long)]
        radius: Option<f64>,
        /// Gap-filling window size in cells
        #[arg(long, default_value_t = 5)]
        window_size: usize,
        /// Strip ASPRS low/high noise outliers via ELM before ground selection
        #[arg(long)]
        remove_outliers: bool,
        /// ELM cell size in CRS horizontal units when --remove-outliers is set
        #[arg(long, default_value_t = 10.0)]
        outlier_cell: f64,
        /// ELM elevation drop threshold in CRS vertical units when --remove-outliers is set
        #[arg(long, default_value_t = 1.0)]
        outlier_threshold: f64,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Breach depressions using Lindsay (2016) least-cost breaching (complete, selective, or constrained)
    BreachDepressions {
        /// Input DEM raster
        input: PathBuf,
        /// Output conditioned DEM raster
        output: PathBuf,
        /// Breaching mode (complete, selective, constrained)
        #[arg(long, value_enum, default_value_t = CliBreachMode::Selective)]
        mode: CliBreachMode,
        /// Maximum breach depth in elevation units (defaults to 1.5 for selective/constrained; pass 0 for unlimited)
        #[arg(long)]
        max_depth: Option<f64>,
        /// Maximum breach path length in cells (defaults to 50 for selective/constrained; pass 0 for unlimited)
        #[arg(long)]
        max_length: Option<u32>,
        /// Fill remaining unbreached depressions after selective/constrained breaching
        #[arg(long)]
        fill_remaining: bool,
        /// Compress output GeoTIFF with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Burn polyline or point culvert vectors across road embankments into a DEM
    BurnCulverts {
        /// Input DEM raster
        input: PathBuf,
        /// Input culvert polyline or point vector (.geojson, .gpkg, .shp)
        culverts: PathBuf,
        /// Output hydro-conditioned DEM raster
        output: PathBuf,
        /// Half-width of carved culvert channel in grid cells (0 = 1-cell wide)
        #[arg(long, default_value_t = 0)]
        half_width: usize,
        /// Elevation drop below endpoint elevation (vertical DEM units)
        #[arg(long, default_value_t = 0.05)]
        depth_offset: f64,
        /// Search radius in cells around culvert point features
        #[arg(long, default_value_t = 5)]
        point_radius: usize,
        /// Compress output GeoTIFF with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Run the complete in-memory hydrology pipeline from a raw DEM to vectors, rating curves, connectivity graph, and inundation rasters
    Pipeline {
        /// Input raw DEM raster
        input: PathBuf,
        /// Output directory
        out_dir: PathBuf,
        /// Median filter kernel size (1 = skip filtering)
        #[arg(short = 'k', long, default_value_t = 3)]
        kernel: usize,
        /// Minimum depression size in pixels
        #[arg(short = 'm', long, default_value_t = 1000)]
        min_size: usize,
        /// Minimum depression depth
        #[arg(short = 'd', long, default_value_t = 0.3)]
        min_depth: f64,
        /// Vertical level-set slicing interval
        #[arg(short = 'i', long, default_value_t = 0.3)]
        interval: f64,
        /// Rainfall intensity in cm/h
        #[arg(long, default_value_t = 5.0)]
        rain_intensity: f64,
        /// Total rainfall duration in hours (defaults to 24.0 when --scs-type2-mm is set, else 50.0)
        #[arg(long)]
        duration: Option<f64>,
        /// Simulation time step in hours
        #[arg(long, default_value_t = 1.0)]
        time_step: f64,
        /// Optional breaching mode (complete, selective, constrained; enables Lindsay 2016 pre-conditioning)
        #[arg(long, value_enum)]
        breach_mode: Option<CliBreachMode>,
        /// Optional selective breaching maximum depth (enables Lindsay 2016 pre-conditioning; pass 0 for unlimited)
        #[arg(long)]
        breach_max_depth: Option<f64>,
        /// Optional selective breaching maximum path length in cells (defaults to 50 when --breach-max-depth is set; pass 0 for unlimited)
        #[arg(long, requires = "breach_max_depth")]
        breach_max_length: Option<u32>,
        /// Fill remaining unbreached depressions after selective/constrained breaching
        #[arg(long)]
        breach_fill_remaining: bool,
        /// Optional culvert vector (.gpkg, .geojson, .shp) to burn before sink extraction
        #[arg(long)]
        culverts: Option<PathBuf>,
        /// Optional spatially variable runoff-coefficient raster (GeoTIFF, values 0..=1)
        #[arg(long, conflicts_with = "curve_number")]
        runoff_coeff: Option<PathBuf>,
        /// Optional NRCS Curve Number (0..=100) raster (GeoTIFF) converted to runoff coefficient via TR-55
        #[arg(long, conflicts_with = "runoff_coeff")]
        curve_number: Option<PathBuf>,
        /// Optional spatially variable infiltration loss raster (GeoTIFF, cm/h)
        #[arg(long)]
        infiltration: Option<PathBuf>,
        /// Optional antecedent water depth raster (GeoTIFF)
        #[arg(long)]
        antecedent_depth: Option<PathBuf>,
        /// Optional SCS Type II design storm total precipitation in mm (overrides constant rainfall)
        #[arg(long)]
        scs_type2_mm: Option<f64>,
        /// Hydrologic routing mode (static = independent level-set timing, fsm = dynamic Fill-Spill-Merge cascade mass-balance)
        #[arg(long, value_enum, default_value_t = CliRoutingMode::Static)]
        routing: CliRoutingMode,
        /// Optional stream network vector (.gpkg, .geojson, .shp) for distance-to-stream graph metrics
        #[arg(long)]
        streams: Option<PathBuf>,
        /// Load DEM as f32 (halves peak RAM on large rasters)
        #[arg(long)]
        use_f32: bool,
        /// Skip writing output GeoTIFF rasters (keep CSV, GeoPackage, and graph outputs)
        #[arg(long)]
        no_rasters: bool,
        /// Skip writing output GeoPackage vectors
        #[arg(long)]
        no_vectors: bool,
        /// Skip writing rating curves and connectivity graph
        #[arg(long)]
        no_graph: bool,
        /// Compress output GeoTIFFs with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Delineate nested depressions across large DEMs using parallel overlapping tiles
    TiledDepressions {
        /// Input DEM raster
        input: PathBuf,
        /// Output directory
        out_dir: PathBuf,
        /// Core tile width/height in pixels
        #[arg(long, default_value_t = 2048)]
        tile_size: usize,
        /// Overlap halo around each tile in pixels
        #[arg(long, default_value_t = 128)]
        overlap: usize,
        /// Median filter kernel size (1 = skip)
        #[arg(short = 'k', long, default_value_t = 3)]
        kernel: usize,
        /// Minimum depression size in pixels
        #[arg(short = 'm', long, default_value_t = 1000)]
        min_size: usize,
        /// Minimum depression depth
        #[arg(short = 'd', long, default_value_t = 0.3)]
        min_depth: f64,
        /// Vertical level-set slicing interval
        #[arg(short = 'i', long, default_value_t = 0.3)]
        interval: f64,
        /// Connectivity topology for per-tile depression filling (d8 or d4)
        #[arg(short = 't', long, value_enum, default_value_t = CliTopology::D8)]
        topology: CliTopology,
        /// Load DEM in f64 precision instead of the default f32 precision
        #[arg(long, conflicts_with = "use_f32")]
        use_f64: bool,
        /// Load DEM in f32 precision (default; kept for backward compatibility)
        #[arg(long, hide = true)]
        use_f32: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Export numbered PNG inundation frames with optional scale bar and legend
    ExportFrames {
        /// Background DEM raster
        dem: PathBuf,
        /// Fill-time raster (fill_time.tif)
        fill_time: PathBuf,
        /// Output directory for numbered PNG frames
        out_dir: PathBuf,
        /// Number of frames to export
        #[arg(long, default_value_t = 10)]
        num_steps: usize,
        /// Time step in hours
        #[arg(long, default_value_t = 1.0)]
        time_step: f64,
        /// Optional rainfall intensity in cm/h for banner labels
        #[arg(long)]
        rain_intensity: Option<f64>,
        /// Water overlay opacity (0.0..=1.0)
        #[arg(long, default_value_t = 0.55)]
        water_alpha: f32,
        /// Optional maximum width/height in pixels
        #[arg(long)]
        max_dim: Option<usize>,
        /// Draw a metric scale bar on each frame
        #[arg(long)]
        scale_bar: bool,
        /// Draw a legend box on each frame
        #[arg(long)]
        legend: bool,
        /// Hillshade azimuth in degrees
        #[arg(long, default_value_t = 135.0)]
        azimuth: f64,
        /// Hillshade elevation angle in degrees
        #[arg(long, default_value_t = 45.0)]
        elevation: f64,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Delineate contributing catchments for extracted sinks (ports 2_Delineate_Catchment.py)
    DelineateCatchments {
        /// Partially-filled / refined DEM raster (e.g. dem.tif from extract-sinks)
        dem: PathBuf,
        /// Sink region raster (region.tif / sink.tif) or sink polygon vector (.shp / .gpkg)
        sink: PathBuf,
        /// Output directory
        out_dir: PathBuf,
        /// Compress output GeoTIFFs with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Delineate overland spill flow paths between sinks (ports 3_Flow_Path.py)
    DelineateFlowPaths {
        /// Fully filled DEM raster (e.g. dem_filled.tif from extract-sinks)
        dem_filled: PathBuf,
        /// Sink region raster (region.tif / sink.tif) or sink polygon vector (.shp / .gpkg)
        sink: PathBuf,
        /// Output polyline vector file (.shp, .gpkg, .geojson)
        output: PathBuf,
        /// Rainfall intensity in cm/h
        #[arg(long, default_value_t = 5.0)]
        rain_intensity: f64,
        /// Optional refined/partially-filled DEM (dem.tif); required to compute sink volume, cat_area, time_inund, and rank (otherwise left 0 / unranked)
        #[arg(long)]
        refined_dem: Option<PathBuf>,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Delineate nested catchment hierarchy across depression levels (ports 5_Catchment_Hierarchy.py)
    CatchmentHierarchy {
        /// Partially-filled / refined DEM raster (e.g. dem.tif from extract-sinks)
        dem: PathBuf,
        /// Directory containing Single_level_*.shp (or .tif) from delineate-depressions --level-shp
        level_dir: PathBuf,
        /// Output catchment hierarchy GeoTIFF (e.g. catchment_hir.tif)
        output: PathBuf,
        /// Compress output GeoTIFF with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Simulate surface inundation dynamics over time (ports 6_Simulate Inundation.py)
    SimulateInundation {
        /// Input sink elevation raster (sink.tif)
        sink: PathBuf,
        /// Input catchment hierarchy raster (catchment_hir.tif)
        catchment: PathBuf,
        /// Output directory for flood_*.tif frames, fill_time.tif, and summary CSV
        out_dir: PathBuf,
        /// Minimum depression size in pixels
        #[arg(short = 'm', long, default_value_t = 1000)]
        min_size: usize,
        /// Minimum depression depth
        #[arg(short = 'd', long, default_value_t = 0.2)]
        min_depth: f64,
        /// Vertical level-set slicing interval (positive; sliced top-down)
        #[arg(short = 'i', long, default_value_t = 0.2)]
        interval: f64,
        /// Rainfall intensity in cm/h
        #[arg(long, default_value_t = 5.0)]
        rain_intensity: f64,
        /// Total rainfall duration in hours (defaults to 24.0 when --scs-type2-mm is set, else 50.0)
        #[arg(long)]
        duration: Option<f64>,
        /// Simulation time step in hours
        #[arg(long, default_value_t = 1.0)]
        time_step: f64,
        /// Hydrologic routing mode (static = independent level-set timing, fsm = dynamic Fill-Spill-Merge cascade mass-balance)
        #[arg(long, value_enum, default_value_t = CliRoutingMode::Static)]
        routing: CliRoutingMode,
        /// Optional refined DEM raster (e.g. dem.tif) for --routing fsm overland routing (defaults to sink raster if omitted)
        #[arg(long)]
        dem: Option<PathBuf>,
        /// Optional SCS Type II design storm total precipitation in mm (overrides constant rainfall)
        #[arg(long)]
        scs_type2_mm: Option<f64>,
        /// Optional runoff coefficient scalar (0..=1) or aligned GeoTIFF raster path
        #[arg(long)]
        runoff_coeff: Option<String>,
        /// Optional vertical infiltration loss rate scalar (cm/h, >= 0) or aligned GeoTIFF raster path
        #[arg(long)]
        infiltration: Option<String>,
        /// Optional antecedent ponded water depth scalar (>= 0) or aligned GeoTIFF raster path
        #[arg(long)]
        antecedent_depth: Option<String>,
        /// Compress output GeoTIFFs with DEFLATE
        #[arg(short = 'c', long)]
        compress: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Render shaded-relief inundation animation as an animated GIF (ports 7_Play_Animation.py)
    AnimateInundation {
        /// Background DEM raster (e.g. dem.tif)
        dem: PathBuf,
        /// Directory containing flood_*.tif rasters from simulate-inundation
        flood_dir: PathBuf,
        /// Output animated GIF path (e.g. inundation.gif)
        output: PathBuf,
        /// Delay between frames in milliseconds
        #[arg(long, default_value_t = 200)]
        delay_ms: u16,
        /// Play only rising water (disable rising-then-falling ping-pong playback)
        #[arg(long)]
        no_ping_pong: bool,
        /// Number of GIF loops (0 = infinite)
        #[arg(long, default_value_t = 0)]
        loops: u16,
        /// Water overlay opacity (0.0..=1.0)
        #[arg(long, default_value_t = 0.55)]
        water_alpha: f32,
        /// Optional maximum width/height in pixels for downscaling large DEMs
        #[arg(long)]
        max_dim: Option<usize>,
        /// Optional time step in hours for frame labels
        #[arg(long, default_value_t = 1.0)]
        time_step: f64,
        /// Optional rainfall intensity in cm/h for the banner title
        #[arg(long)]
        rain_intensity: Option<f64>,
        /// Hillshade azimuth in degrees
        #[arg(long, default_value_t = 135.0)]
        azimuth: f64,
        /// Hillshade elevation angle in degrees
        #[arg(long, default_value_t = 45.0)]
        elevation: f64,
        /// Draw a metric scale bar on each frame
        #[arg(long)]
        scale_bar: bool,
        /// Draw a legend box on each frame
        #[arg(long)]
        legend: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Download authoritative open U.S. federal data (USGS 3DEP DEM/LPC, WBD, NHD, Transportation, USFWS NWI, NOAA Atlas 14) for a region
    #[cfg(feature = "download")]
    Download {
        /// Output directory for downloaded federal datasets and manifest.json
        #[arg(short = 'o', long)]
        out_dir: PathBuf,
        /// WGS 84 bounding box as 'min_lon,min_lat,max_lon,max_lat'
        #[arg(long, allow_hyphen_values = true, conflicts_with_all = ["huc", "region"])]
        bbox: Option<String>,
        /// USGS WBD Hydrologic Unit Code (2, 4, 6, 8, 10, or 12 digits, e.g. 10160002)
        #[arg(long, conflicts_with_all = ["bbox", "region"])]
        huc: Option<String>,
        /// Local polygon vector file (.geojson, .gpkg, .shp) defining the region of interest
        #[arg(long, conflicts_with_all = ["bbox", "huc"])]
        region: Option<PathBuf>,
        /// Comma-separated datasets to download ('all' or subset of wbd,dem,nhd,nwi,roads,culverts,noaa)
        #[arg(long, default_value = "all")]
        datasets: String,
        /// USGS 3DEP product to download (1m, 10m, 30m, lpc)
        #[arg(long, default_value = "1m")]
        dem_product: String,
        /// USGS WBD hierarchy level (2, 4, 6, 8, 10, or 12)
        #[arg(long, default_value = "12")]
        wbd_level: String,
        /// Optional shared cache directory for 3DEP tiles
        #[arg(long)]
        tile_cache: Option<PathBuf>,
        /// Keep all overlapping 3DEP tile vintages instead of deduplicating to newest publicationDate
        #[arg(long)]
        no_deduplicate: bool,
        /// Skip mosaicking downloaded 3DEP GeoTIFF tiles into dem.tif
        #[arg(long)]
        no_mosaic: bool,
        /// Skip clipping downloaded DEM/vectors to the HUC or vector region boundary
        #[arg(long)]
        no_clip: bool,
        /// Culvert synthesis half-length along stream flowlines in meters
        #[arg(long, default_value_t = 25.0)]
        culvert_half_length: f64,
        /// Culvert synthesis snapping distance in meters
        #[arg(long, default_value_t = 15.0)]
        culvert_snap_distance: f64,
        /// Maximum TNM product records to scan across paginated pages (default 1000)
        #[arg(long)]
        max_products: Option<usize>,
        /// Comma-separated USGS National Transportation MapServer layer IDs (default "29,30,31,32,33,38")
        #[arg(long)]
        transportation_layers: Option<String>,
        /// Per-request HTTP timeout in seconds (default 120)
        #[arg(long)]
        timeout_secs: Option<u64>,
        /// Maximum retry attempts for transient network or 5xx errors (default 4)
        #[arg(long)]
        max_retries: Option<usize>,
        /// ArcGIS ObjectID chunk size for paginated vector downloads (default 250)
        #[arg(long)]
        oid_chunk_size: Option<usize>,
        /// Re-download files even if they already exist locally
        #[arg(long)]
        overwrite: bool,
        /// Query matching USGS 3DEP tiles without downloading data
        #[arg(long)]
        dry_run: bool,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Fetch NOAA Atlas 14 precipitation frequency depth table (mm) for a WGS 84 latitude/longitude point
    #[cfg(feature = "download")]
    NoaaAtlas14 {
        /// Query latitude in decimal degrees (WGS 84, -90..=90)
        #[arg(long, allow_hyphen_values = true)]
        lat: f64,
        /// Query longitude in decimal degrees (WGS 84, -180..=180)
        #[arg(long, allow_hyphen_values = true)]
        lon: f64,
        /// Optional output JSON file path to save the parsed table
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
        /// Per-request HTTP timeout in seconds (default 120)
        #[arg(long)]
        timeout_secs: Option<u64>,
        /// Maximum retry attempts for transient network or 5xx errors (default 4)
        #[arg(long)]
        max_retries: Option<usize>,
        /// Re-fetch even if --output already exists with matching coordinates
        #[arg(long)]
        overwrite: bool,
        /// Print full table as JSON
        #[arg(long)]
        json: bool,
    },
    /// Synthesize deterministic road-stream culvert breach polylines by intersecting stream and road vectors
    SynthesizeCulverts {
        /// Stream flowline vector file (e.g. NHD .gpkg, .geojson, .shp)
        streams: PathBuf,
        /// Road/railway centerline vector file (e.g. USGS Transportation .gpkg, .geojson, .shp)
        roads: PathBuf,
        /// Output culvert breach polyline vector file (.gpkg or .shp)
        output: PathBuf,
        /// Half-length extracted along each stream on both sides of a crossing (meters)
        #[arg(long, default_value_t = 25.0)]
        half_length: f64,
        /// Minimum distance between distinct crossings in meters
        #[arg(long, default_value_t = 15.0)]
        snap_distance: f64,
        /// Optional target projected CRS (e.g. EPSG:26914); auto-selects local UTM if omitted for WGS 84 inputs
        #[arg(long)]
        crs: Option<String>,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Render a shaded-relief PNG preview of a DEM, optional depression depth overlay, or inundation snapshot
    Preview {
        /// Input DEM or DSM raster
        dem: PathBuf,
        /// Output PNG file path (in MCP mode, omit -o to return the PNG directly as inline image content)
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
        /// Optional fill-time raster (fill_time.tif) to overlay inundation at --hour
        #[arg(long, conflicts_with = "depth")]
        fill_time: Option<PathBuf>,
        /// Inundation snapshot time in hours when --fill-time is set; shows cells first inundated by
        /// this time (defaults to max finite fill time, which can exceed the storm duration because
        /// constant-rain fill_time.tif assumes sustained rainfall; pass the storm duration to stop
        /// at storm end)
        #[arg(long, requires = "fill_time")]
        hour: Option<f64>,
        /// Optional depression depth raster (depth.tif or sink.tif) to overlay positive-depth basins
        #[arg(long, conflicts_with = "fill_time")]
        depth: Option<PathBuf>,
        /// Maximum output width/height in pixels (0 = native resolution)
        #[arg(long, default_value_t = 1024)]
        max_dim: usize,
        /// Overlay opacity (0.0..=1.0)
        #[arg(long, default_value_t = 0.55)]
        water_alpha: f32,
        /// Draw a metric scale bar on the preview
        #[arg(long)]
        scale_bar: bool,
        /// Draw a color legend box on the preview (when an overlay is shown)
        #[arg(long)]
        legend: bool,
        /// Hillshade azimuth in degrees
        #[arg(long, default_value_t = 135.0)]
        azimuth: f64,
        /// Hillshade elevation angle in degrees
        #[arg(long, default_value_t = 45.0)]
        elevation: f64,
        /// Print summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Print metadata summary for a raster, vector dataset, or LiDAR point cloud
    Info {
        /// Input raster, vector, or point cloud file
        input: PathBuf,
        /// Compute per-band raster statistics (min, max, mean, stddev)
        #[arg(long)]
        stats: bool,
        /// Print metadata as JSON
        #[arg(long)]
        json: bool,
    },
    /// Run a Model Context Protocol (MCP) server over stdin/stdout for AI agents
    Mcp {
        /// Cooperative guardrail restricting MCP tool path arguments to the given directory before invocation (repeatable; pair with OS sandboxing for kernel-level isolation)
        #[arg(long = "allow-dir", value_name = "DIR")]
        allow_dir: Vec<PathBuf>,
    },
}

fn dispatch(command: Commands, lidar_cmd: Option<Vec<OsString>>) -> lidar::Result<()> {
    match command {
        Commands::MeanFilter {
            input,
            output,
            kernel_size,
            json,
        } => {
            mean_filter_file(&input, kernel_size, Some(&output))?;
            if json {
                print_json(&json!({
                    "output": output,
                    "kernel_size": kernel_size,
                }))?;
            } else {
                println!("Saved mean-filtered raster to {}", output.display());
            }
        }
        Commands::MedianFilter {
            input,
            output,
            kernel_size,
            json,
        } => {
            median_filter_file(&input, kernel_size, Some(&output))?;
            if json {
                print_json(&json!({
                    "output": output,
                    "kernel_size": kernel_size,
                }))?;
            } else {
                println!("Saved median-filtered raster to {}", output.display());
            }
        }
        Commands::GaussianFilter {
            input,
            output,
            sigma,
            json,
        } => {
            gaussian_filter_file(&input, sigma, Some(&output))?;
            if json {
                print_json(&json!({
                    "output": output,
                    "sigma": sigma,
                }))?;
            } else {
                println!("Saved Gaussian-filtered raster to {}", output.display());
            }
        }
        Commands::ExtractSinks {
            input,
            out_dir,
            min_size,
            filled_dem,
            topology,
            drop_intermediates,
            compress,
            json,
        } => {
            let opts = ExtractSinksOptions {
                min_size,
                filled_dem,
                topology: topology.into(),
                keep_files: !drop_intermediates,
                compress,
            };
            let res = extract_sinks(&input, &out_dir, &opts)?;
            if json {
                print_json(&res)?;
            } else {
                println!("Extracted {} sink region(s)", res.depressions.len());
                if let Some(ref sink_path) = res.sink_path {
                    println!("  sink: {}", sink_path.display());
                }
                println!("  gpkg: {}", res.gpkg_path.display());
            }
        }
        Commands::DelineateDepressions {
            in_sink,
            out_dir,
            min_size,
            min_depth,
            interval,
            level_shp,
            comb_tif,
            gpkg,
            compress,
            json,
        } => {
            let opts = DelineateOptions {
                min_size,
                min_depth,
                interval,
                bool_level_shp: level_shp,
                bool_comb_tif: comb_tif,
                export_gpkg: gpkg,
                compress,
            };
            let res = delineate_depressions(&in_sink, &out_dir, &opts)?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Delineated {} nested depression(s) across {} region(s) (max level = {})",
                    res.depressions.len(),
                    res.num_regions,
                    res.max_level
                );
                println!("  depression_id:    {}", res.depression_id_path.display());
                println!(
                    "  depression_level: {}",
                    res.depression_level_path.display()
                );
                println!("  shapefile:        {}", res.shapefile_path.display());
                if let Some(ref gp) = res.gpkg_path {
                    println!("  gpkg:             {}", gp.display());
                }
                println!("  csv:              {}", res.csv_path.display());
            }
        }
        Commands::DelineateMounts {
            input,
            out_dir,
            min_size,
            min_height,
            interval,
            level_shp,
            comb_tif,
            gpkg,
            compress,
            json,
        } => {
            let opts = DelineateOptions {
                min_size,
                min_depth: min_height,
                interval,
                bool_level_shp: level_shp,
                bool_comb_tif: comb_tif,
                export_gpkg: gpkg,
                compress,
            };
            let res = delineate_mounts(&input, &out_dir, &opts)?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Delineated {} nested mount(s) across {} region(s) (max level = {})",
                    res.mounts.depressions.len(),
                    res.mounts.num_regions,
                    res.mounts.max_level
                );
                println!(
                    "  depression_id:    {}",
                    res.mounts.depression_id_path.display()
                );
                println!(
                    "  depression_level: {}",
                    res.mounts.depression_level_path.display()
                );
                println!(
                    "  shapefile:        {}",
                    res.mounts.shapefile_path.display()
                );
                if let Some(ref gp) = res.mounts.gpkg_path {
                    println!("  gpkg:             {}", gp.display());
                }
                println!("  csv:              {}", res.mounts.csv_path.display());
            }
        }
        Commands::FlipDem {
            input,
            output,
            delta,
            json,
        } => {
            flip_dem_file(&input, delta, Some(&output))?;
            if json {
                print_json(&json!({
                    "output": output,
                    "delta": delta,
                }))?;
            } else {
                println!("Saved flipped DEM to {}", output.display());
            }
        }
        Commands::Polygonize {
            input,
            output,
            field_name,
            json,
        } => {
            let count = polygonize(&input, &output, &field_name)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "field_name": field_name,
                    "feature_count": count,
                }))?;
            } else {
                println!("Polygonized {count} feature(s) to {}", output.display());
            }
        }
        Commands::Mosaic {
            inputs,
            output,
            json,
        } => {
            mosaic(&inputs, &output)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "input_count": inputs.len(),
                }))?;
            } else {
                println!(
                    "Mosaicked {} raster(s) to {}",
                    inputs.len(),
                    output.display()
                );
            }
        }
        Commands::Reproject {
            input,
            output,
            crs,
            resampling,
            resolution,
            json,
        } => {
            let r = parse_resampling(&resampling)?;
            let res_pair = resolution.map(|v| (v, v));
            reproject_image(&input, &output, &crs, r, res_pair)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "crs": crs,
                    "resampling": resampling,
                    "resolution": resolution,
                }))?;
            } else {
                println!("Reprojected to {} ({crs})", output.display());
            }
        }
        Commands::Resample {
            input,
            output,
            resolution,
            resampling,
            json,
        } => {
            let r = parse_resampling(&resampling)?;
            resample(&input, &output, resolution, r)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "resolution": resolution,
                    "resampling": resampling,
                }))?;
            } else {
                println!(
                    "Resampled to {} (resolution={resolution})",
                    output.display()
                );
            }
        }
        Commands::Clip {
            input,
            mask,
            output,
            json,
        } => {
            clip_image(&input, &mask, &output)?;
            if json {
                print_json(&json!({
                    "input": input,
                    "mask": mask,
                    "output": output,
                }))?;
            } else {
                println!("Clipped raster saved to {}", output.display());
            }
        }
        Commands::ToCog {
            input,
            output,
            compression,
            json,
        } => {
            image_to_cog(&input, &output, Some(&compression))?;
            if json {
                print_json(&json!({
                    "output": output,
                    "compression": compression,
                }))?;
            } else {
                println!("Saved COG to {}", output.display());
            }
        }
        Commands::AddCrs {
            input,
            crs,
            output,
            json,
        } => {
            add_crs(&input, &crs, output.as_deref())?;
            let target = output.as_deref().unwrap_or(&input);
            if json {
                print_json(&json!({
                    "output": target,
                    "crs": crs,
                }))?;
            } else {
                println!("Assigned CRS {crs} to {}", target.display());
            }
        }
        Commands::JoinTables {
            in_vector,
            in_csv,
            out_vector,
            vec_col,
            csv_col,
            json,
        } => {
            let count = join_csv_to_vector(&in_vector, &in_csv, &out_vector, &vec_col, &csv_col)?;
            if json {
                print_json(&json!({
                    "output": out_vector,
                    "feature_count": count,
                    "vec_col": vec_col,
                    "csv_col": csv_col,
                }))?;
            } else {
                println!("Joined {count} feature(s) to {}", out_vector.display());
            }
        }
        Commands::CsvToVector {
            in_csv,
            out_vector,
            latitude,
            longitude,
            json,
        } => {
            let count = csv_points_to_vector(&in_csv, &out_vector, &latitude, &longitude)?;
            if json {
                print_json(&json!({
                    "output": out_vector,
                    "point_count": count,
                }))?;
            } else {
                println!("Converted {count} point(s) to {}", out_vector.display());
            }
        }
        Commands::ConvertLidar {
            input,
            output,
            point_format,
            minor_version,
            json,
        } => {
            let n = convert_lidar(&input, &output, point_format, minor_version)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "point_count": n,
                    "point_format": point_format,
                    "minor_version": minor_version,
                }))?;
            } else {
                println!("Converted {n} point(s) to {}", output.display());
            }
        }
        Commands::ClassifyGround {
            input,
            output,
            method,
            json,
        } => {
            ensure_no_cli_path_collision(&input, &output)?;
            let cloud = read_lidar(&input)?;
            let classified = classify_ground(&cloud, method.into())?;
            write_lidar(&classified, &output, &PointWriteOptions::default())?;
            let point_count = classified.len();
            let ground_count = classified
                .classification_counts()
                .into_iter()
                .find(|&(c, _)| c == 2)
                .map_or(0, |(_, n)| n);
            if json {
                print_json(&json!({
                    "input": input,
                    "output": output,
                    "method": method.as_str(),
                    "point_count": point_count,
                    "ground_count": ground_count,
                }))?;
            } else {
                println!(
                    "Classified {ground_count}/{point_count} ground point(s) ({}) to {}",
                    method.as_str(),
                    output.display()
                );
            }
        }
        Commands::LidarToDsm {
            input,
            output,
            resolution,
            radius,
            min_z,
            max_z,
            classification,
            return_kind,
            stat,
            method,
            window_size,
            json,
        } => {
            let classifications = classification.as_deref().map(parse_u8_list).transpose()?;
            let return_kinds = return_kind.as_deref().map(parse_return_kinds).transpose()?;
            let opts = LidarToDsmOptions {
                resolution,
                radius,
                min_z,
                max_z,
                classifications,
                return_kinds,
                stat: GridStat::parse(&stat)?,
                method: GridMethod::parse(&method)?,
                window_size,
                ..Default::default()
            };
            let (cols, rows) = lidar_to_dsm(&input, &output, &opts)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "width": cols,
                    "height": rows,
                    "resolution": resolution,
                }))?;
            } else {
                println!("Generated {cols}x{rows} DSM at {}", output.display());
            }
        }
        Commands::LidarToDtm {
            input,
            output,
            ground_method,
            ground_class,
            resolution,
            radius,
            method,
            stat,
            window_size,
            remove_outliers,
            outlier_cell,
            outlier_threshold,
            json,
        } => {
            let opts = LidarToDtmOptions {
                resolution,
                ground_method: ground_method.into(),
                ground_class,
                radius,
                method: GridMethod::parse(&method)?,
                stat: GridStat::parse(&stat)?,
                window_size,
                remove_outliers,
                outlier_cell,
                outlier_threshold,
                ..Default::default()
            };
            let (cols, rows) = lidar_to_dtm(&input, &output, &opts)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "width": cols,
                    "height": rows,
                    "resolution": resolution,
                    "remove_outliers": remove_outliers,
                }))?;
            } else {
                println!("Generated {cols}x{rows} DTM at {}", output.display());
            }
        }
        Commands::LidarToChm {
            input,
            output,
            ground_method,
            ground_class,
            resolution,
            radius,
            window_size,
            remove_outliers,
            outlier_cell,
            outlier_threshold,
            json,
        } => {
            let opts = LidarToDtmOptions {
                resolution,
                ground_method: ground_method.into(),
                ground_class,
                radius,
                window_size,
                remove_outliers,
                outlier_cell,
                outlier_threshold,
                ..Default::default()
            };
            let (cols, rows) = lidar_to_chm(&input, &output, &opts)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "width": cols,
                    "height": rows,
                    "resolution": resolution,
                    "remove_outliers": remove_outliers,
                }))?;
            } else {
                println!("Generated {cols}x{rows} CHM at {}", output.display());
            }
        }
        Commands::BreachDepressions {
            input,
            output,
            mode,
            max_depth,
            max_length,
            fill_remaining,
            compress,
            json,
        } => {
            let breach_mode: BreachMode = mode.into();
            let mut opts = BreachOptions::for_mode(breach_mode, max_depth, max_length);
            if max_depth == Some(0.0) {
                opts.max_depth = None;
            }
            if max_length == Some(0) {
                opts.max_length = None;
            }
            opts.fill_remaining = fill_remaining;
            opts.compress = compress;
            breach_depressions_file(&input, &output, &opts)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "mode": format!("{breach_mode:?}").to_ascii_lowercase(),
                    "max_depth": opts.max_depth,
                    "max_length": opts.max_length,
                    "fill_remaining": fill_remaining,
                }))?;
            } else {
                println!("Saved breached DEM to {}", output.display());
            }
        }
        Commands::BurnCulverts {
            input,
            culverts,
            output,
            half_width,
            depth_offset,
            point_radius,
            compress,
            json,
        } => {
            let opts = BurnCulvertsOptions {
                half_width_cells: half_width,
                depth_offset,
                point_radius_cells: point_radius,
                compress,
            };
            let (_out, carved) = burn_culverts_file(&input, &culverts, &output, &opts)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "cells_lowered": carved,
                }))?;
            } else {
                println!(
                    "Burned culverts ({carved} cells lowered) to {}",
                    output.display()
                );
            }
        }
        Commands::Pipeline {
            input,
            out_dir,
            kernel,
            min_size,
            min_depth,
            interval,
            rain_intensity,
            duration,
            time_step,
            breach_mode,
            breach_max_depth,
            breach_max_length,
            breach_fill_remaining,
            culverts,
            runoff_coeff,
            curve_number,
            infiltration,
            antecedent_depth,
            scs_type2_mm,
            routing,
            streams,
            use_f32,
            no_rasters,
            no_vectors,
            no_graph,
            compress,
            json,
        } => {
            let duration = duration.unwrap_or(if scs_type2_mm.is_some() { 24.0 } else { 50.0 });
            let hyetograph = scs_type2_mm
                .map(|mm| scs_type2_hyetograph(mm, duration, time_step))
                .transpose()?;
            let breach = if breach_mode.is_some() || breach_max_depth.is_some() {
                let mode: BreachMode = breach_mode.map(Into::into).unwrap_or(BreachMode::Selective);
                let md_arg = breach_max_depth.filter(|&md| md != 0.0);
                let mut opts = BreachOptions::for_mode(mode, md_arg, breach_max_length);
                if breach_max_depth == Some(0.0) {
                    opts.max_depth = None;
                }
                if breach_max_length == Some(0) {
                    opts.max_length = None;
                }
                opts.fill_remaining = breach_fill_remaining;
                Some(opts)
            } else if breach_fill_remaining {
                return Err(lidar::Error::InvalidArgument(
                    "--breach-fill-remaining requires --breach-mode or --breach-max-depth".into(),
                ));
            } else {
                None
            };
            let _cn_tmp_file;
            let resolved_runoff_coeff = if let Some(cn_path) = curve_number {
                let precip_mm = scs_type2_mm.unwrap_or(rain_intensity * duration * 10.0);
                let cn_arr = load_raster::<f64>(&cn_path, None)?;
                let rc_arr = scs_curve_number_runoff(&cn_arr, precip_mm)?;
                let tmp = tempfile::Builder::new()
                    .prefix("lidar-cn-runoff-")
                    .suffix(".tif")
                    .tempfile()?;
                save_raster(&rc_arr, tmp.path(), Some("scs_curve_number_runoff"), false)?;
                let tmp_path = tmp.path().to_path_buf();
                _cn_tmp_file = Some(tmp);
                Some(tmp_path)
            } else {
                _cn_tmp_file = None;
                runoff_coeff
            };
            let opts = RunPipelineOptions {
                kernel_size: kernel,
                breach,
                culverts_vector: culverts,
                simulation: SimulateInundationOptions {
                    min_size,
                    min_depth,
                    interval,
                    rain_intensity_cm_h: rain_intensity,
                    rain_duration_h: duration,
                    time_step_h: time_step,
                    compress,
                },
                routing: routing.into(),
                runoff_coeff_path: resolved_runoff_coeff,
                infiltration_path: infiltration,
                antecedent_depth_path: antecedent_depth,
                hyetograph,
                stream_vector: streams,
                use_f32,
                save_rasters: !no_rasters,
                export_vectors: !no_vectors,
                export_graph: !no_graph,
                ..Default::default()
            };
            let res = run_pipeline(&input, &out_dir, &opts)?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Pipeline complete: {} nested depression(s) across {} sink region(s) (max level = {}), {} spill path(s)",
                    res.num_depressions, res.num_sink_regions, res.max_level, res.num_spill_paths
                );
                println!("  depressions CSV:  {}", res.depressions_csv.display());
                if let Some(ref p) = res.depressions_gpkg {
                    println!("  depressions GPKG: {}", p.display());
                }
                if let Some(ref p) = res.spill_paths_gpkg {
                    println!("  spill paths GPKG: {}", p.display());
                }
                if let Some(ref p) = res.rating_curves_csv {
                    println!("  rating curves:    {}", p.display());
                }
                if let Some(ref p) = res.connectivity_graph_json {
                    println!("  graph JSON:       {}", p.display());
                }
                if let Some(ref p) = res.connectivity_graphml {
                    println!("  graph GraphML:    {}", p.display());
                }
                println!("  inundation curve: {}", res.inundation_csv.display());
                if let Some(ref p) = res.fsm_basins_csv {
                    println!("  FSM basins CSV:   {}", p.display());
                }
                if let Some(ref mb) = res.fsm_mass_balance {
                    println!(
                        "  FSM mass balance ({:.2} h): runoff={:.2} m³, stored={:.2} m³, infil={:.2} m³, outflow={:.2} m³, err={:.2e} m³",
                        mb.duration_h,
                        mb.total_runoff_m3,
                        mb.total_stored_m3,
                        mb.total_infiltration_m3,
                        mb.total_boundary_outflow_m3,
                        mb.mass_balance_error_m3,
                    );
                }
                println!("  summary JSON:     {}", res.summary_json.display());
            }
        }
        Commands::TiledDepressions {
            input,
            out_dir,
            tile_size,
            overlap,
            kernel,
            min_size,
            min_depth,
            interval,
            topology,
            use_f64,
            use_f32: _,
            json,
        } => {
            let opts = TiledDelineationOptions {
                tile_size,
                overlap,
                kernel_size: kernel,
                min_size,
                min_depth,
                interval,
                topology: topology.into(),
                use_f32: !use_f64,
            };
            let res = delineate_depressions_tiled(&input, &out_dir, &opts)?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Tiled delineation complete across {} tile(s): {} nested depression(s), max level = {}",
                    res.num_tiles,
                    res.depressions.len(),
                    res.max_level
                );
                println!("  csv: {}", res.csv_path.display());
            }
        }
        Commands::ExportFrames {
            dem,
            fill_time,
            out_dir,
            num_steps,
            time_step,
            rain_intensity,
            water_alpha,
            max_dim,
            scale_bar,
            legend,
            azimuth,
            elevation,
            json,
        } => {
            let dem_arr = load_raster::<f64>(&dem, None)?;
            let ft_arr = load_raster::<f64>(&fill_time, None)?;
            let opts = AnimateInundationOptions {
                water_alpha,
                max_dimension: max_dim,
                time_step_h: Some(time_step),
                rain_intensity_cm_h: rain_intensity,
                show_scale_bar: scale_bar,
                show_legend: legend,
                hillshade: HillshadeOptions {
                    azimuth_deg: azimuth,
                    elevation_deg: elevation,
                    ..Default::default()
                },
                ..Default::default()
            };
            let paths = export_inundation_frames_png(
                &dem_arr,
                &ft_arr,
                num_steps,
                time_step,
                &[],
                &out_dir,
                &opts,
            )?;
            if json {
                print_json(&json!({
                    "out_dir": out_dir,
                    "frame_count": paths.len(),
                    "frames": paths,
                }))?;
            } else {
                println!(
                    "Exported {} PNG frame(s) to {}",
                    paths.len(),
                    out_dir.display()
                );
            }
        }
        Commands::DelineateCatchments {
            dem,
            sink,
            out_dir,
            compress,
            json,
        } => {
            let res = delineate_catchments(
                &dem,
                &sink,
                &out_dir,
                &DelineateCatchmentsOptions { compress },
            )?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Delineated {} catchment(s):\n  watershed: {}\n  vector:    {}\n  csv:       {}",
                    res.records.len(),
                    res.watershed_path.display(),
                    res.vector_path.display(),
                    res.csv_path.display(),
                );
            }
        }
        Commands::DelineateFlowPaths {
            dem_filled,
            sink,
            output,
            rain_intensity,
            refined_dem,
            json,
        } => {
            let res = delineate_flow_paths(
                &dem_filled,
                &sink,
                &output,
                &DelineateFlowPathsOptions {
                    rain_intensity_cm_h: rain_intensity,
                    refined_dem,
                },
            )?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Delineated {} flow path(s):\n  vector: {}\n  csv:    {}",
                    res.records.len(),
                    res.flowpath_vector_path.display(),
                    res.csv_path.display(),
                );
            }
        }
        Commands::CatchmentHierarchy {
            dem,
            level_dir,
            output,
            compress,
            json,
        } => {
            let res = delineate_catchment_hierarchy(
                &dem,
                &level_dir,
                &output,
                &CatchmentHierarchyOptions { compress },
            )?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Delineated catchment hierarchy ({} feature(s) across {} level(s)):\n  raster: {}\n  merged: {}\n  csv:    {}",
                    res.records.len(),
                    res.level_shapefiles.len(),
                    res.catchment_raster_path.display(),
                    res.merged_shapefile_path.display(),
                    res.csv_path.display(),
                );
            }
        }
        Commands::SimulateInundation {
            sink,
            catchment,
            out_dir,
            min_size,
            min_depth,
            interval,
            rain_intensity,
            duration,
            time_step,
            routing,
            dem,
            scs_type2_mm,
            runoff_coeff,
            infiltration,
            antecedent_depth,
            compress,
            json,
        } => {
            let duration = duration.unwrap_or(if scs_type2_mm.is_some() { 24.0 } else { 50.0 });
            let routing_mode: RoutingMode = routing.into();
            let opts = SimulateInundationOptions {
                min_size,
                min_depth,
                interval,
                rain_intensity_cm_h: rain_intensity,
                rain_duration_h: duration,
                time_step_h: time_step,
                compress,
            };
            let has_advanced = routing_mode == RoutingMode::FillSpillMerge
                || scs_type2_mm.is_some()
                || runoff_coeff.is_some()
                || infiltration.is_some()
                || antecedent_depth.is_some()
                || dem.is_some();

            if !has_advanced {
                let res = simulate_inundation(&sink, &catchment, &out_dir, &opts)?;
                if json {
                    print_json(&res)?;
                } else {
                    println!(
                        "Simulated {} timestep(s) across {} region(s):\n  fill_time: {}\n  summary:   {}",
                        res.steps.len(),
                        res.num_regions,
                        res.fill_time_path.display(),
                        res.summary_csv_path.display(),
                    );
                }
            } else {
                let adv_opts = SimulateInundationAdvancedOptions {
                    simulation: opts,
                    routing: routing_mode,
                    dem_path: dem,
                    scs_type2_mm,
                    runoff_coeff: runoff_coeff
                        .as_deref()
                        .map(ForcingInput::parse_spec)
                        .transpose()?,
                    infiltration_cm_h: infiltration
                        .as_deref()
                        .map(ForcingInput::parse_spec)
                        .transpose()?,
                    antecedent_depth: antecedent_depth
                        .as_deref()
                        .map(ForcingInput::parse_spec)
                        .transpose()?,
                };
                let adv = simulate_inundation_advanced(&sink, &catchment, &out_dir, &adv_opts)?;
                if json {
                    if adv.routing == RoutingMode::FillSpillMerge {
                        print_json(&adv)?;
                    } else {
                        print_json(&adv.simulation)?;
                    }
                } else {
                    println!(
                        "Simulated {} timestep(s) across {} region(s):\n  fill_time: {}\n  summary:   {}",
                        adv.simulation.steps.len(),
                        adv.simulation.num_regions,
                        adv.simulation.fill_time_path.display(),
                        adv.simulation.summary_csv_path.display(),
                    );
                    if let Some(ref p) = adv.fsm_basins_csv {
                        println!("  fsm_csv:   {}", p.display());
                    }
                    if let Some(ref mb) = adv.fsm_mass_balance {
                        println!(
                            "  FSM mass balance ({:.2} h): runoff={:.2} m³, stored={:.2} m³, infil={:.2} m³, outflow={:.2} m³, err={:.2e} m³",
                            mb.duration_h,
                            mb.total_runoff_m3,
                            mb.total_stored_m3,
                            mb.total_infiltration_m3,
                            mb.total_boundary_outflow_m3,
                            mb.mass_balance_error_m3,
                        );
                    }
                }
            }
        }
        Commands::AnimateInundation {
            dem,
            flood_dir,
            output,
            delay_ms,
            no_ping_pong,
            loops,
            water_alpha,
            max_dim,
            time_step,
            rain_intensity,
            azimuth,
            elevation,
            scale_bar,
            legend,
            json,
        } => {
            let opts = AnimateInundationOptions {
                frame_delay_ms: delay_ms,
                ping_pong: !no_ping_pong,
                loops,
                water_alpha,
                max_dimension: max_dim,
                time_step_h: Some(time_step),
                rain_intensity_cm_h: rain_intensity,
                show_scale_bar: scale_bar,
                show_legend: legend,
                hillshade: HillshadeOptions {
                    azimuth_deg: azimuth,
                    elevation_deg: elevation,
                    ..Default::default()
                },
                ..Default::default()
            };
            let res = animate_inundation(&dem, &flood_dir, &output, &opts)?;
            if json {
                print_json(&res)?;
            } else {
                println!(
                    "Rendered {} frame(s) ({}x{}) to {}",
                    res.frame_count,
                    res.width,
                    res.height,
                    res.gif_path.display(),
                );
            }
        }
        #[cfg(feature = "download")]
        Commands::Download {
            out_dir,
            bbox,
            huc,
            region,
            datasets,
            dem_product,
            wbd_level,
            tile_cache,
            no_deduplicate,
            no_mosaic,
            no_clip,
            culvert_half_length,
            culvert_snap_distance,
            max_products,
            transportation_layers,
            timeout_secs,
            max_retries,
            oid_chunk_size,
            overwrite,
            dry_run,
            json,
        } => {
            let reg = match (bbox, huc, region) {
                (Some(bb), None, None) => Region::Bbox(bb.parse::<Bbox>()?),
                (None, Some(code), None) => Region::Huc(Region::validate_huc(&code)?),
                (None, None, Some(path)) => Region::VectorFile(path),
                _ => {
                    return Err(lidar::Error::InvalidArgument(
                        "specify exactly one region selector: --bbox, --huc, or --region".into(),
                    ));
                }
            };
            let defaults = DownloadOptions::default();
            let parsed_layers = transportation_layers
                .as_deref()
                .map(parse_u32_list)
                .transpose()?;
            let dl_opts = DownloadOptions {
                timeout_secs: timeout_secs.unwrap_or(defaults.timeout_secs),
                max_retries: max_retries.unwrap_or(defaults.max_retries),
                oid_chunk_size: oid_chunk_size.unwrap_or(defaults.oid_chunk_size),
                overwrite,
                tile_cache_dir: tile_cache,
                dem_product: dem_product.parse::<DemProduct>()?,
                wbd_level: wbd_level.parse::<WbdLevel>()?,
                deduplicate_tiles: !no_deduplicate,
                mosaic_dem: !no_mosaic,
                clip_to_region: !no_clip,
                max_products: max_products.unwrap_or(defaults.max_products),
                culvert_half_length_m: culvert_half_length,
                culvert_snap_distance_m: culvert_snap_distance,
                transportation_layers: parsed_layers.unwrap_or(defaults.transportation_layers),
            };
            if dry_run {
                let items = query_3dep_products(&reg, &dl_opts)?;
                if json {
                    print_json(&items)?;
                } else {
                    let total_mb: f64 =
                        items.iter().filter_map(|i| i.size_bytes).sum::<u64>() as f64 / 1e6;
                    println!(
                        "Found {} USGS 3DEP '{}' product(s) (~{total_mb:.1} MB total):",
                        items.len(),
                        dl_opts.dem_product.tnm_dataset_name()
                    );
                    for it in &items {
                        let mb = it.size_bytes.map_or(0.0, |b| b as f64 / 1e6);
                        println!(
                            "  {} | {} | {:.1} MB | {}",
                            it.grid_key, it.publication_date, mb, it.title
                        );
                    }
                }
            } else {
                let ds_list = parse_dataset_list(&datasets)?;
                let bundle_opts = FederalBundleOptions {
                    datasets: ds_list,
                    download: dl_opts,
                };
                let res = download_federal_data(&reg, &out_dir, &bundle_opts)?;
                if json {
                    print_json(&res)?;
                } else {
                    println!(
                        "Downloaded federal data into {} (bbox: {}):",
                        out_dir.display(),
                        res.region_bbox.to_csv_string()
                    );
                    if let Some(ref p) = res.wbd_path {
                        println!("  WBD:      {} ({} feature(s))", p.display(), res.wbd_count);
                    }
                    if !res.dem_tiles.is_empty() {
                        println!("  3DEP:     {} tile(s)", res.dem_tiles.len());
                    }
                    if let Some(ref p) = res.dem_mosaic_path {
                        println!("  DEM:      {}", p.display());
                    }
                    if let Some(ref p) = res.nhd_path {
                        println!(
                            "  NHD:      {} ({} flowline(s))",
                            p.display(),
                            res.nhd_count
                        );
                    }
                    if let Some(ref p) = res.nwi_path {
                        println!("  NWI:      {} ({} polygon(s))", p.display(), res.nwi_count);
                    }
                    if let Some(ref p) = res.roads_path {
                        println!(
                            "  Roads:    {} ({} segment(s))",
                            p.display(),
                            res.roads_count
                        );
                    }
                    if let Some(ref p) = res.culverts_path {
                        println!(
                            "  Culverts: {} ({} crossing(s))",
                            p.display(),
                            res.culverts_count
                        );
                    }
                    if let Some(ref p) = res.noaa_atlas14_path {
                        let d100 = res
                            .noaa_atlas14
                            .as_ref()
                            .and_then(|t| t.depth_mm("24-hr", 100))
                            .map_or_else(|| "n/a".into(), |v| format!("{v:.1} mm"));
                        println!("  NOAA 14:  {} (24-hr 100-yr = {d100})", p.display());
                    }
                    println!("  Manifest: {}", res.manifest_path.display());
                }
            }
        }
        #[cfg(feature = "download")]
        Commands::NoaaAtlas14 {
            lat,
            lon,
            output,
            timeout_secs,
            max_retries,
            overwrite,
            json,
        } => {
            let defaults = DownloadOptions::default();
            let dl_opts = DownloadOptions {
                timeout_secs: timeout_secs.unwrap_or(defaults.timeout_secs),
                max_retries: max_retries.unwrap_or(defaults.max_retries),
                overwrite,
                ..defaults
            };
            let table = fetch_noaa_atlas14(lat, lon, output.as_deref(), &dl_opts)?;
            if json {
                print_json(&table)?;
            } else {
                println!(
                    "NOAA Atlas 14 PDS precipitation depths ({}) at lat={:.4}, lon={:.4}:",
                    table.units, table.lat, table.lon
                );
                for ari in [2u32, 10, 25, 100] {
                    if let Some(mm) = table.depth_mm("24-hr", ari) {
                        println!("  24-hr {ari:>3}-yr: {mm:.1} mm");
                    }
                }
                if let Some(ref p) = output {
                    println!("  saved: {}", p.display());
                }
            }
        }
        Commands::SynthesizeCulverts {
            streams,
            roads,
            output,
            half_length,
            snap_distance,
            crs,
            json,
        } => {
            let opts = SynthesizeCulvertsOptions {
                half_length_m: half_length,
                snap_distance_m: snap_distance,
                target_crs: crs,
                ..Default::default()
            };
            let n = synthesize_culverts(&streams, &roads, &output, &opts)?;
            if json {
                print_json(&json!({
                    "output": output,
                    "crossing_count": n,
                }))?;
            } else {
                println!(
                    "Synthesized {n} culvert crossing(s) to {}",
                    output.display()
                );
            }
        }
        Commands::Preview {
            dem,
            output,
            fill_time,
            hour,
            depth,
            max_dim,
            water_alpha,
            scale_bar,
            legend,
            azimuth,
            elevation,
            json,
        } => {
            let out_path = output.ok_or_else(|| {
                lidar::Error::InvalidArgument(
                    "output path (-o / --output) is required on the command line (in MCP mode, omit -o to receive inline image content)"
                        .into(),
                )
            })?;
            let resolve = |p: &Path| -> PathBuf {
                std::fs::canonicalize(p).unwrap_or_else(|_| {
                    if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
                        let parent_dir = if parent.as_os_str().is_empty() {
                            Path::new(".")
                        } else {
                            parent
                        };
                        std::fs::canonicalize(parent_dir)
                            .map(|cp| cp.join(name))
                            .unwrap_or_else(|_| p.to_path_buf())
                    } else {
                        p.to_path_buf()
                    }
                })
            };
            let out_resolved = resolve(&out_path);
            for input_path in [Some(dem.as_path()), fill_time.as_deref(), depth.as_deref()]
                .into_iter()
                .flatten()
            {
                if resolve(input_path) == out_resolved {
                    return Err(lidar::Error::InvalidArgument(format!(
                        "preview output '{}' would overwrite input raster '{}'",
                        out_path.display(),
                        input_path.display()
                    )));
                }
            }
            if let Some(h) = hour
                && (!h.is_finite() || h < 0.0)
            {
                return Err(lidar::Error::InvalidArgument(format!(
                    "--hour must be finite and >= 0, got {h}"
                )));
            }
            let allow_overview = max_dim > 0 && fill_time.is_none() && depth.is_none();
            let dem_arr = load_dem_for_preview(&dem, max_dim, allow_overview)?;
            let opts = AnimateInundationOptions {
                water_alpha,
                max_dimension: if max_dim == 0 { None } else { Some(max_dim) },
                show_scale_bar: scale_bar,
                show_legend: legend,
                hillshade: HillshadeOptions {
                    azimuth_deg: azimuth,
                    elevation_deg: elevation,
                    ..Default::default()
                },
                ..Default::default()
            };
            let png_bytes = if let Some(ft_path) = fill_time {
                let ft_arr = load_raster::<f64>(&ft_path, None)?;
                let ft_nd = ft_arr.no_data();
                let max_ft = ft_arr
                    .data()
                    .iter()
                    .copied()
                    .filter(|&v| v.is_finite() && v >= 0.0 && !v.rd_eq(ft_nd))
                    .fold(0.0f64, f64::max);
                let t_hours = hour.unwrap_or(if max_ft > 0.0 { max_ft } else { 1.0 });
                let progress = if max_ft > 0.0 {
                    (t_hours / max_ft).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                render_inundation_frame_png_bytes(&dem_arr, &ft_arr, t_hours, progress, &[], &opts)?
            } else if let Some(depth_path) = depth {
                let depth_arr = load_raster::<f64>(&depth_path, None)?;
                render_dem_preview_png_bytes(&dem_arr, Some(&depth_arr), &opts)?
            } else {
                render_dem_preview_png_bytes(&dem_arr, None, &opts)?
            };
            if let Some(parent) = out_path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out_path, &png_bytes)?;
            if json {
                print_json(&json!({
                    "output": out_path,
                    "bytes": png_bytes.len(),
                }))?;
            } else {
                println!(
                    "Rendered preview PNG ({} bytes) to {}",
                    png_bytes.len(),
                    out_path.display()
                );
            }
        }
        Commands::Info { input, stats, json } => {
            let p_str = input.to_str().ok_or_else(|| {
                lidar::Error::InvalidArgument(format!("non-UTF-8 path: {}", input.display()))
            })?;
            let opts = terrio::info::InfoOptions {
                statistics: stats,
                ..Default::default()
            };
            let report = terrio::info::info_path(p_str, &opts)?;
            if json {
                print_json(&report)?;
            } else {
                println!("{}", report.to_text());
            }
        }
        Commands::Mcp { allow_dir } => {
            if std::env::var_os("LIDAR_MCP_TOOL").is_some() {
                return Err(lidar::Error::InvalidArgument(
                    "refusing to start recursive `lidar mcp` inside an MCP tool subprocess".into(),
                ));
            }
            let cmd = match lidar_cmd {
                Some(c) => c,
                None => mcp::this_binary()?,
            };
            mcp::serve_with_allowed_dirs(cmd, allow_dir)?;
        }
    }

    Ok(())
}

fn ensure_no_cli_path_collision(input: &Path, output: &Path) -> lidar::Result<()> {
    let resolve = |p: &Path| -> PathBuf {
        std::fs::canonicalize(p).unwrap_or_else(|_| {
            if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
                let parent_dir = if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                };
                std::fs::canonicalize(parent_dir)
                    .map(|cp| cp.join(name))
                    .unwrap_or_else(|_| p.to_path_buf())
            } else {
                p.to_path_buf()
            }
        })
    };
    if resolve(input) == resolve(output) {
        return Err(lidar::Error::InvalidArgument(format!(
            "output '{}' would overwrite input '{}'",
            output.display(),
            input.display()
        )));
    }
    Ok(())
}

/// Runs the `lidar` CLI with `args` (program name first) and returns a process exit code
/// (`0` on success, `1` on runtime error, `2` on argument usage error).
pub fn run(args: Vec<OsString>) -> i32 {
    run_as(args, None)
}

/// Runs the `lidar` CLI with `args`, using `lidar_cmd` as the command line invoked by `lidar mcp`
/// for child tool calls (e.g., `[python, "-m", "lidar_rs"]` when run from the Python package).
pub fn run_as(args: Vec<OsString>, lidar_cmd: Option<Vec<OsString>>) -> i32 {
    usage::with_clap_stack(move || run_as_inner(args, lidar_cmd))
}

fn run_as_inner(args: Vec<OsString>, lidar_cmd: Option<Vec<OsString>>) -> i32 {
    if let Some(catalog_res) = usage::requested(&args) {
        match catalog_res {
            Ok(catalog) => match serde_json::to_string_pretty(&catalog) {
                Ok(json) => {
                    println!("{json}");
                    return 0;
                }
                Err(e) => {
                    eprintln!("error: failed serializing command catalog: {e}");
                    return 1;
                }
            },
            Err(msg) => {
                eprintln!("error: {msg}");
                return 2;
            }
        }
    }
    let cli = match Cli::try_parse_from(&args) {
        Ok(c) => c,
        Err(e) => {
            let code = e.exit_code();
            let _ = e.print();
            return code;
        }
    };
    match dispatch(cli.command, lidar_cmd) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_depressions_defaults_to_f32_and_accepts_f64_or_compat_f32() {
        usage::with_clap_stack(|| {
            let cli =
                Cli::try_parse_from(["lidar", "tiled-depressions", "dem.tif", "out"]).unwrap();
            match cli.command {
                Commands::TiledDepressions {
                    use_f64, use_f32, ..
                } => {
                    assert!(!use_f64);
                    assert!(!use_f32);
                }
                _ => panic!("expected TiledDepressions"),
            }

            let cli_compat =
                Cli::try_parse_from(["lidar", "tiled-depressions", "dem.tif", "out", "--use-f32"])
                    .unwrap();
            match cli_compat.command {
                Commands::TiledDepressions { use_f64, .. } => assert!(!use_f64),
                _ => panic!("expected TiledDepressions"),
            }

            let cli_f64 =
                Cli::try_parse_from(["lidar", "tiled-depressions", "dem.tif", "out", "--use-f64"])
                    .unwrap();
            match cli_f64.command {
                Commands::TiledDepressions { use_f64, .. } => assert!(use_f64),
                _ => panic!("expected TiledDepressions"),
            }

            assert!(
                Cli::try_parse_from([
                    "lidar",
                    "tiled-depressions",
                    "dem.tif",
                    "out",
                    "--use-f32",
                    "--use-f64",
                ])
                .is_err()
            );
        });
    }

    #[test]
    fn pipeline_breach_max_length_requires_breach_max_depth() {
        usage::with_clap_stack(|| {
            assert!(
                Cli::try_parse_from([
                    "lidar",
                    "pipeline",
                    "dem.tif",
                    "out",
                    "--breach-max-length",
                    "25",
                ])
                .is_err()
            );

            let cli = Cli::try_parse_from([
                "lidar",
                "pipeline",
                "dem.tif",
                "out",
                "--breach-max-depth",
                "1.5",
                "--breach-max-length",
                "0",
            ])
            .unwrap();
            match cli.command {
                Commands::Pipeline {
                    breach_max_depth,
                    breach_max_length,
                    ..
                } => {
                    assert_eq!(breach_max_depth, Some(1.5));
                    assert_eq!(breach_max_length, Some(0));
                }
                _ => panic!("expected Pipeline"),
            }
        });
    }

    #[test]
    fn parse_u8_list_and_return_kinds_reject_empty_input() {
        assert!(parse_u8_list("").is_err());
        assert!(parse_u8_list("  ,  ").is_err());
        assert_eq!(parse_u8_list("2, 6").unwrap(), vec![2, 6]);

        assert!(parse_return_kinds("").is_err());
        assert!(parse_return_kinds(" , ").is_err());
        assert_eq!(
            parse_return_kinds("first,only").unwrap(),
            vec![ReturnKind::First, ReturnKind::Only]
        );
    }

    #[test]
    fn preview_rejects_invalid_hour_and_avoids_creating_output_dir_on_missing_dem() {
        let tmp = tempfile::tempdir().unwrap();
        let missing_dem = tmp.path().join("does_not_exist.tif");
        let nested_out_dir = tmp.path().join("nested_out");
        let out_png = nested_out_dir.join("preview.png");

        let code = run(vec![
            "lidar".into(),
            "preview".into(),
            missing_dem.as_os_str().to_os_string(),
            "-o".into(),
            out_png.as_os_str().to_os_string(),
        ]);
        assert_eq!(code, 1);
        assert!(
            !nested_out_dir.exists(),
            "preview must not create output parent directories when input DEM fails to load"
        );

        let code_bad_hour = run(vec![
            "lidar".into(),
            "preview".into(),
            missing_dem.as_os_str().to_os_string(),
            "--fill-time".into(),
            missing_dem.as_os_str().to_os_string(),
            "--hour=-1".into(),
            "-o".into(),
            out_png.as_os_str().to_os_string(),
        ]);
        assert_eq!(code_bad_hour, 1);
        assert!(!nested_out_dir.exists());
    }

    #[test]
    fn preview_renders_cog_to_valid_png_with_requested_max_dim() {
        let tmp = tempfile::tempdir().unwrap();
        let src_tif = tmp.path().join("dem_1024.tif");
        let cog_tif = tmp.path().join("dem_1024_cog.tif");
        let out_png = tmp.path().join("preview_256.png");

        let (w, h) = (1024usize, 1024usize);
        let mut data = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                let dx = c as f64 - 512.0;
                let dy = r as f64 - 512.0;
                data.push(100.0 + 0.02 * (dx * dx + dy * dy).sqrt());
            }
        }
        let mut dem = lidar::Array2D::from_vec(w, h, data).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [500_000.0, 1.0, 0.0, 4_001_024.0, 0.0, -1.0];
        lidar::save_raster(&dem, &src_tif, None, false).unwrap();

        image_to_cog(&src_tif, &cog_tif, None).unwrap();

        let code = run(vec![
            "lidar".into(),
            "preview".into(),
            cog_tif.as_os_str().to_os_string(),
            "--max-dim".into(),
            "256".into(),
            "-o".into(),
            out_png.as_os_str().to_os_string(),
        ]);
        assert_eq!(code, 0);
        let bytes = std::fs::read(&out_png).unwrap();
        assert!(
            bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "preview output must be a valid PNG"
        );
        // PNG IHDR width/height are big-endian u32 at bytes 16..24.
        let png_w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let png_h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(png_w, 256);
        assert_eq!(png_h, 256);
    }

    #[test]
    fn every_exposed_subcommand_supports_json_flag() {
        let mut count = 0usize;
        for cmd in usage::commands() {
            count += 1;
            let has_json = cmd
                .get_arguments()
                .any(|arg| arg.get_long() == Some("json"));
            assert!(
                has_json,
                "subcommand `{}` is missing `--json` flag",
                cmd.get_name()
            );
        }
        assert!(
            count >= 28,
            "expected at least 28 exposed subcommands, got {count}"
        );
    }

    #[test]
    fn parses_new_subcommands_and_upgraded_flags() {
        usage::with_clap_stack(|| {
            // classify-ground with smrf and --json
            let cli = Cli::try_parse_from([
                "lidar",
                "classify-ground",
                "in.laz",
                "out.laz",
                "--method",
                "smrf",
                "--json",
            ])
            .unwrap();
            match cli.command {
                Commands::ClassifyGround { method, json, .. } => {
                    assert_eq!(method.as_str(), "smrf");
                    assert!(json);
                }
                _ => panic!("expected ClassifyGround"),
            }

            // simulate-inundation with FSM, SCS Type II, and forcing flags
            let cli = Cli::try_parse_from([
                "lidar",
                "simulate-inundation",
                "sink.tif",
                "cat.tif",
                "out",
                "--routing",
                "fsm",
                "--scs-type2-mm",
                "100",
                "--runoff-coeff",
                "0.75",
                "--infiltration",
                "0.2",
                "--antecedent-depth",
                "0.05",
                "--json",
            ])
            .unwrap();
            match cli.command {
                Commands::SimulateInundation {
                    routing,
                    scs_type2_mm,
                    runoff_coeff,
                    infiltration,
                    antecedent_depth,
                    duration,
                    json,
                    ..
                } => {
                    assert_eq!(RoutingMode::from(routing), RoutingMode::FillSpillMerge);
                    assert_eq!(scs_type2_mm, Some(100.0));
                    assert_eq!(runoff_coeff.as_deref(), Some("0.75"));
                    assert_eq!(infiltration.as_deref(), Some("0.2"));
                    assert_eq!(antecedent_depth.as_deref(), Some("0.05"));
                    assert_eq!(duration, None);
                    assert!(json);
                }
                _ => panic!("expected SimulateInundation"),
            }

            // pipeline with --curve-number, --breach-mode, --breach-fill-remaining
            let cli = Cli::try_parse_from([
                "lidar",
                "pipeline",
                "dem.tif",
                "out",
                "--curve-number",
                "cn.tif",
                "--breach-mode",
                "constrained",
                "--breach-fill-remaining",
            ])
            .unwrap();
            match cli.command {
                Commands::Pipeline {
                    curve_number,
                    breach_mode,
                    breach_fill_remaining,
                    ..
                } => {
                    assert_eq!(curve_number, Some(PathBuf::from("cn.tif")));
                    assert!(matches!(breach_mode, Some(CliBreachMode::Constrained)));
                    assert!(breach_fill_remaining);
                }
                _ => panic!("expected Pipeline"),
            }
            // --curve-number conflicts with --runoff-coeff
            assert!(
                Cli::try_parse_from([
                    "lidar",
                    "pipeline",
                    "dem.tif",
                    "out",
                    "--curve-number",
                    "cn.tif",
                    "--runoff-coeff",
                    "rc.tif",
                ])
                .is_err()
            );

            // delineate-mounts with --gpkg and --comb-tif
            let cli = Cli::try_parse_from([
                "lidar",
                "delineate-mounts",
                "dsm.tif",
                "out",
                "--level-shp",
                "--comb-tif",
                "--gpkg",
                "--json",
            ])
            .unwrap();
            match cli.command {
                Commands::DelineateMounts {
                    level_shp,
                    comb_tif,
                    gpkg,
                    json,
                    ..
                } => {
                    assert!(level_shp && comb_tif && gpkg && json);
                }
                _ => panic!("expected DelineateMounts"),
            }

            // tiled-depressions with -t d4
            let cli =
                Cli::try_parse_from(["lidar", "tiled-depressions", "dem.tif", "out", "-t", "d4"])
                    .unwrap();
            match cli.command {
                Commands::TiledDepressions { topology, .. } => {
                    assert_eq!(Topology::from(topology), Topology::D4);
                }
                _ => panic!("expected TiledDepressions"),
            }
        });
    }

    #[cfg(feature = "download")]
    #[test]
    fn parses_noaa_atlas14_and_download_tuning_flags() {
        usage::with_clap_stack(|| {
            let cli = Cli::try_parse_from([
                "lidar",
                "noaa-atlas14",
                "--lat",
                "47.098",
                "--lon",
                "-99.095",
                "-o",
                "noaa.json",
                "--json",
            ])
            .unwrap();
            match cli.command {
                Commands::NoaaAtlas14 {
                    lat,
                    lon,
                    output,
                    json,
                    ..
                } => {
                    assert!((lat - 47.098).abs() < 1e-9);
                    assert!((lon - (-99.095)).abs() < 1e-9);
                    assert_eq!(output, Some(PathBuf::from("noaa.json")));
                    assert!(json);
                }
                _ => panic!("expected NoaaAtlas14"),
            }

            let cli = Cli::try_parse_from([
                "lidar",
                "download",
                "-o",
                "out",
                "--bbox",
                "-99.1,47.0,-99.0,47.1",
                "--max-products",
                "50",
                "--transportation-layers",
                "29,30,38",
                "--timeout-secs",
                "45",
                "--max-retries",
                "2",
                "--oid-chunk-size",
                "100",
                "--dry-run",
                "--json",
            ])
            .unwrap();
            match cli.command {
                Commands::Download {
                    max_products,
                    transportation_layers,
                    timeout_secs,
                    max_retries,
                    oid_chunk_size,
                    ..
                } => {
                    assert_eq!(max_products, Some(50));
                    assert_eq!(transportation_layers.as_deref(), Some("29,30,38"));
                    assert_eq!(timeout_secs, Some(45));
                    assert_eq!(max_retries, Some(2));
                    assert_eq!(oid_chunk_size, Some(100));
                }
                _ => panic!("expected Download"),
            }
        });
    }

    #[test]
    fn classify_ground_and_simulate_inundation_advanced_and_pipeline_cn_execute_cleanly() {
        let tmp = tempfile::tempdir().unwrap();

        // 1. classify-ground (SMRF) end-to-end
        let in_las = tmp.path().join("pts.las");
        let out_las = tmp.path().join("pts_ground.las");
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut cls = Vec::new();
        for r in 0..12 {
            for c in 0..12 {
                xs.push(500_000.0 + c as f64);
                ys.push(4_000_000.0 + r as f64);
                zs.push(100.0);
                cls.push(1u8);
            }
        }
        xs.push(500_006.0);
        ys.push(4_000_006.0);
        zs.push(115.0);
        cls.push(1u8);
        let cloud = lidar::pointcloud::PointCloud::new(
            vec![
                lidar::pointcloud::Dimension::new("x", lidar::pointcloud::Values::F64(xs)),
                lidar::pointcloud::Dimension::new("y", lidar::pointcloud::Values::F64(ys)),
                lidar::pointcloud::Dimension::new("z", lidar::pointcloud::Values::F64(zs)),
                lidar::pointcloud::Dimension::new(
                    "classification",
                    lidar::pointcloud::Values::U8(cls),
                ),
            ],
            terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
        );
        write_lidar(&cloud, &in_las, &PointWriteOptions::default()).unwrap();
        assert_eq!(
            run(vec![
                "lidar".into(),
                "classify-ground".into(),
                in_las.as_os_str().to_os_string(),
                out_las.as_os_str().to_os_string(),
                "--method".into(),
                "smrf".into(),
                "--json".into(),
            ]),
            0
        );
        assert!(out_las.exists());
        // Path collision must fail.
        assert_eq!(
            run(vec![
                "lidar".into(),
                "classify-ground".into(),
                in_las.as_os_str().to_os_string(),
                in_las.as_os_str().to_os_string(),
            ]),
            1
        );

        // 2. pipeline with --curve-number and --breach-mode selective --breach-fill-remaining
        let dem_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dem.tif");
        let dem = load_raster::<f64>(&dem_path, None).unwrap();
        let cn_path = tmp.path().join("cn.tif");
        let mut cn =
            lidar::Array2D::from_vec(dem.width(), dem.height(), vec![80.0; dem.data().len()])
                .unwrap();
        cn.copy_metadata_from(&dem);
        cn.set_no_data(-9999.0);
        save_raster(&cn, &cn_path, None, false).unwrap();

        let pipe_dir = tmp.path().join("pipe_cn");
        assert_eq!(
            run(vec![
                "lidar".into(),
                "pipeline".into(),
                dem_path.as_os_str().to_os_string(),
                pipe_dir.as_os_str().to_os_string(),
                "--curve-number".into(),
                cn_path.as_os_str().to_os_string(),
                "--breach-mode".into(),
                "selective".into(),
                "--breach-max-depth".into(),
                "0.3".into(),
                "--breach-fill-remaining".into(),
                "--scs-type2-mm".into(),
                "100".into(),
                "--duration".into(),
                "12".into(),
                "--time-step".into(),
                "4".into(),
                "--no-vectors".into(),
                "--no-graph".into(),
                "--json".into(),
            ]),
            0
        );
        assert!(pipe_dir.join("summary.json").exists());

        // 3. simulate-inundation with --routing fsm, --runoff-coeff scalar, --scs-type2-mm
        let sink_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/sink.tif");
        let sim_dir = tmp.path().join("sim_fsm");
        assert_eq!(
            run(vec![
                "lidar".into(),
                "simulate-inundation".into(),
                sink_path.as_os_str().to_os_string(),
                sink_path.as_os_str().to_os_string(),
                sim_dir.as_os_str().to_os_string(),
                "--routing".into(),
                "fsm".into(),
                "--runoff-coeff".into(),
                "0.75".into(),
                "--infiltration".into(),
                "0.1".into(),
                "--antecedent-depth".into(),
                "0.0".into(),
                "--scs-type2-mm".into(),
                "100".into(),
                "--duration".into(),
                "12".into(),
                "--time-step".into(),
                "4".into(),
                "--json".into(),
            ]),
            0
        );
        assert!(sim_dir.join("fill_time.tif").exists());
        assert!(sim_dir.join("fsm_basins.csv").exists());
        assert!(sim_dir.join("inundation_summary.csv").exists());

        // 4. simulate-inundation with --dem and --routing static must return an error
        let invalid_dir = tmp.path().join("sim_static_dem");
        assert_eq!(
            run(vec![
                "lidar".into(),
                "simulate-inundation".into(),
                sink_path.as_os_str().to_os_string(),
                sink_path.as_os_str().to_os_string(),
                invalid_dir.as_os_str().to_os_string(),
                "--routing".into(),
                "static".into(),
                "--dem".into(),
                dem_path.as_os_str().to_os_string(),
            ]),
            1
        );
    }
}
