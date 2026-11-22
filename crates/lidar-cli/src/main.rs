//! Command-line interface for `lidar-rs`: DEM filtering, sink extraction,
//! nested surface depression and mount delineation, and geospatial utilities
//! in pure Rust (powered by `terrio` and `richdem`).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use lidar::pointcloud::{
    GridMethod, GridStat, GroundFilterMethod, LidarToDsmOptions, LidarToDtmOptions, ReturnKind,
    convert_lidar, lidar_to_chm, lidar_to_dsm, lidar_to_dtm,
};
use lidar::{
    AnimateInundationOptions, BreachMode, BreachOptions, BurnCulvertsOptions,
    CatchmentHierarchyOptions, DelineateCatchmentsOptions, DelineateFlowPathsOptions,
    DelineateOptions, ExtractSinksOptions, HillshadeOptions, RunPipelineOptions,
    SimulateInundationOptions, TiledDelineationOptions, Topology, add_crs, animate_inundation,
    breach_depressions_file, burn_culverts_file, clip_image, csv_points_to_vector,
    delineate_catchment_hierarchy, delineate_catchments, delineate_depressions,
    delineate_depressions_tiled, delineate_flow_paths, delineate_mounts,
    export_inundation_frames_png, extract_sinks, flip_dem_file, gaussian_filter_file, image_to_cog,
    join_csv_to_vector, load_raster, mean_filter_file, median_filter_file, mosaic,
    parse_resampling, polygonize, reproject_image, resample, run_pipeline, scs_type2_hyetograph,
    simulate_inundation,
};

#[derive(Parser)]
#[command(
    name = "lidar",
    version,
    about = "Nested surface depression and mount delineation from DEMs in pure Rust",
    long_about = "Pure-Rust port of opengeos/lidar (Wu et al. 2019; Wu 2021) powered by terrio and richdem, with zero GDAL or C/C++ dependencies."
)]
struct Cli {
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
enum CliGroundMethod {
    Preclassified,
    Csf,
    Pmf,
}

impl From<CliGroundMethod> for GroundFilterMethod {
    fn from(m: CliGroundMethod) -> Self {
        match m {
            CliGroundMethod::Preclassified => GroundFilterMethod::Preclassified,
            CliGroundMethod::Csf => GroundFilterMethod::Csf,
            CliGroundMethod::Pmf => GroundFilterMethod::Pmf,
        }
    }
}

fn parse_u8_list(s: &str) -> lidar::Result<Vec<u8>> {
    s.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim().parse::<u8>().map_err(|_| {
                lidar::Error::InvalidArgument(format!(
                    "invalid ASPRS classification code: '{part}'"
                ))
            })
        })
        .collect()
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
    },
    /// Mosaic multiple rasters into a single GeoTIFF
    Mosaic {
        /// Input raster files
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output mosaic GeoTIFF
        #[arg(short = 'o', long)]
        output: PathBuf,
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
    },
    /// Clip a raster by a polygon vector mask
    Clip {
        /// Input raster
        input: PathBuf,
        /// Polygon mask vector file (GeoJSON, Shapefile, GeoPackage, etc.)
        mask: PathBuf,
        /// Output clipped raster
        output: PathBuf,
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
    },
    /// Generate a bare-earth Digital Terrain Model (DTM) from a LiDAR point cloud using CSF/PMF/preclassified ground points
    LidarToDtm {
        /// Input point cloud file (LAS, LAZ, COPC, PLY, E57)
        input: PathBuf,
        /// Output DTM GeoTIFF
        output: PathBuf,
        /// Ground classification method (preclassified, csf, pmf)
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
    },
    /// Generate a Canopy Height Model (CHM = DSM - DTM) from a LiDAR point cloud
    LidarToChm {
        /// Input point cloud file (LAS, LAZ, COPC, PLY, E57)
        input: PathBuf,
        /// Output CHM GeoTIFF
        output: PathBuf,
        /// Ground classification method (preclassified, csf, pmf)
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
        /// Optional selective breaching maximum depth (enables Lindsay 2016 pre-conditioning)
        #[arg(long)]
        breach_max_depth: Option<f64>,
        /// Optional selective breaching maximum path length in cells
        #[arg(long, default_value_t = 40)]
        breach_max_length: u32,
        /// Optional culvert vector (.gpkg, .geojson, .shp) to burn before sink extraction
        #[arg(long)]
        culverts: Option<PathBuf>,
        /// Optional spatially variable runoff-coefficient raster (GeoTIFF, values 0..=1)
        #[arg(long)]
        runoff_coeff: Option<PathBuf>,
        /// Optional spatially variable infiltration loss raster (GeoTIFF, cm/h)
        #[arg(long)]
        infiltration: Option<PathBuf>,
        /// Optional antecedent water depth raster (GeoTIFF)
        #[arg(long)]
        antecedent_depth: Option<PathBuf>,
        /// Optional SCS Type II design storm total precipitation in mm (overrides constant rainfall)
        #[arg(long)]
        scs_type2_mm: Option<f64>,
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
        /// Load DEM in f32 precision to reduce peak memory
        #[arg(long)]
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
        /// Total rainfall duration in hours
        #[arg(long, default_value_t = 50.0)]
        duration: f64,
        /// Simulation time step in hours
        #[arg(long, default_value_t = 1.0)]
        time_step: f64,
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
    },
    /// Print metadata summary for a raster or LiDAR point cloud
    Info {
        /// Input raster or point cloud file
        input: PathBuf,
    },
}

fn run() -> lidar::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::MeanFilter {
            input,
            output,
            kernel_size,
        } => {
            mean_filter_file(&input, kernel_size, Some(&output))?;
            println!("Saved mean-filtered raster to {}", output.display());
        }
        Commands::MedianFilter {
            input,
            output,
            kernel_size,
        } => {
            median_filter_file(&input, kernel_size, Some(&output))?;
            println!("Saved median-filtered raster to {}", output.display());
        }
        Commands::GaussianFilter {
            input,
            output,
            sigma,
        } => {
            gaussian_filter_file(&input, sigma, Some(&output))?;
            println!("Saved Gaussian-filtered raster to {}", output.display());
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
            compress,
            json,
        } => {
            let opts = DelineateOptions {
                min_size,
                min_depth: min_height,
                interval,
                bool_level_shp: level_shp,
                bool_comb_tif: false,
                export_gpkg: false,
                compress,
            };
            let res = delineate_mounts(&input, &out_dir, &opts)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
            }
        }
        Commands::FlipDem {
            input,
            output,
            delta,
        } => {
            flip_dem_file(&input, delta, Some(&output))?;
            println!("Saved flipped DEM to {}", output.display());
        }
        Commands::Polygonize {
            input,
            output,
            field_name,
        } => {
            let count = polygonize(&input, &output, &field_name)?;
            println!("Polygonized {count} feature(s) to {}", output.display());
        }
        Commands::Mosaic { inputs, output } => {
            mosaic(&inputs, &output)?;
            println!(
                "Mosaicked {} raster(s) to {}",
                inputs.len(),
                output.display()
            );
        }
        Commands::Reproject {
            input,
            output,
            crs,
            resampling,
            resolution,
        } => {
            let r = parse_resampling(&resampling)?;
            let res_pair = resolution.map(|v| (v, v));
            reproject_image(&input, &output, &crs, r, res_pair)?;
            println!("Reprojected to {} ({crs})", output.display());
        }
        Commands::Resample {
            input,
            output,
            resolution,
            resampling,
        } => {
            let r = parse_resampling(&resampling)?;
            resample(&input, &output, resolution, r)?;
            println!(
                "Resampled to {} (resolution={resolution})",
                output.display()
            );
        }
        Commands::Clip {
            input,
            mask,
            output,
        } => {
            clip_image(&input, &mask, &output)?;
            println!("Clipped raster saved to {}", output.display());
        }
        Commands::ToCog {
            input,
            output,
            compression,
        } => {
            image_to_cog(&input, &output, Some(&compression))?;
            println!("Saved COG to {}", output.display());
        }
        Commands::AddCrs { input, crs, output } => {
            add_crs(&input, &crs, output.as_deref())?;
            let target = output.as_deref().unwrap_or(&input);
            println!("Assigned CRS {crs} to {}", target.display());
        }
        Commands::JoinTables {
            in_vector,
            in_csv,
            out_vector,
            vec_col,
            csv_col,
        } => {
            let count = join_csv_to_vector(&in_vector, &in_csv, &out_vector, &vec_col, &csv_col)?;
            println!("Joined {count} feature(s) to {}", out_vector.display());
        }
        Commands::CsvToVector {
            in_csv,
            out_vector,
            latitude,
            longitude,
        } => {
            let count = csv_points_to_vector(&in_csv, &out_vector, &latitude, &longitude)?;
            println!("Converted {count} point(s) to {}", out_vector.display());
        }
        Commands::ConvertLidar {
            input,
            output,
            point_format,
            minor_version,
        } => {
            let n = convert_lidar(&input, &output, point_format, minor_version)?;
            println!("Converted {n} point(s) to {}", output.display());
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
            println!("Generated {cols}x{rows} DSM at {}", output.display());
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
        } => {
            let opts = LidarToDtmOptions {
                resolution,
                ground_method: ground_method.into(),
                ground_class,
                radius,
                method: GridMethod::parse(&method)?,
                stat: GridStat::parse(&stat)?,
                window_size,
                ..Default::default()
            };
            let (cols, rows) = lidar_to_dtm(&input, &output, &opts)?;
            println!("Generated {cols}x{rows} DTM at {}", output.display());
        }
        Commands::LidarToChm {
            input,
            output,
            ground_method,
            ground_class,
            resolution,
            radius,
            window_size,
        } => {
            let opts = LidarToDtmOptions {
                resolution,
                ground_method: ground_method.into(),
                ground_class,
                radius,
                window_size,
                ..Default::default()
            };
            let (cols, rows) = lidar_to_chm(&input, &output, &opts)?;
            println!("Generated {cols}x{rows} CHM at {}", output.display());
        }
        Commands::BreachDepressions {
            input,
            output,
            mode,
            max_depth,
            max_length,
            fill_remaining,
            compress,
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
            println!("Saved breached DEM to {}", output.display());
        }
        Commands::BurnCulverts {
            input,
            culverts,
            output,
            half_width,
            depth_offset,
            point_radius,
            compress,
        } => {
            let opts = BurnCulvertsOptions {
                half_width_cells: half_width,
                depth_offset,
                point_radius_cells: point_radius,
                compress,
            };
            let (_out, carved) = burn_culverts_file(&input, &culverts, &output, &opts)?;
            println!(
                "Burned culverts ({carved} cells lowered) to {}",
                output.display()
            );
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
            breach_max_depth,
            breach_max_length,
            culverts,
            runoff_coeff,
            infiltration,
            antecedent_depth,
            scs_type2_mm,
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
            let breach = breach_max_depth.map(|md| BreachOptions {
                mode: BreachMode::Selective,
                max_depth: Some(md),
                max_length: Some(breach_max_length),
                ..Default::default()
            });
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
                runoff_coeff_path: runoff_coeff,
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
            use_f32,
            json,
        } => {
            let opts = TiledDelineationOptions {
                tile_size,
                overlap,
                kernel_size: kernel,
                min_size,
                min_depth,
                interval,
                use_f32,
                ..Default::default()
            };
            let res = delineate_depressions_tiled(&input, &out_dir, &opts)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
            println!(
                "Exported {} PNG frame(s) to {}",
                paths.len(),
                out_dir.display()
            );
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
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
            compress,
            json,
        } => {
            let opts = SimulateInundationOptions {
                min_size,
                min_depth,
                interval,
                rain_intensity_cm_h: rain_intensity,
                rain_duration_h: duration,
                time_step_h: time_step,
                compress,
            };
            let res = simulate_inundation(&sink, &catchment, &out_dir, &opts)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res)
                        .map_err(|e| lidar::Error::Algorithm(e.to_string()))?
                );
            } else {
                println!(
                    "Simulated {} timestep(s) across {} region(s):\n  fill_time: {}\n  summary:   {}",
                    res.steps.len(),
                    res.num_regions,
                    res.fill_time_path.display(),
                    res.summary_csv_path.display(),
                );
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
            println!(
                "Rendered {} frame(s) ({}x{}) to {}",
                res.frame_count,
                res.width,
                res.height,
                res.gif_path.display(),
            );
        }
        Commands::Info { input } => {
            let p_str = input.to_str().ok_or_else(|| {
                lidar::Error::InvalidArgument(format!("non-UTF-8 path: {}", input.display()))
            })?;
            let opts = terrio::info::InfoOptions {
                statistics: false,
                ..Default::default()
            };
            println!("{}", terrio::info::info_path(p_str, &opts)?.to_text());
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
