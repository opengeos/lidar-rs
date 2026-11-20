//! Command-line interface for `lidar-rs`: DEM filtering, sink extraction,
//! nested surface depression and mount delineation, and geospatial utilities
//! in pure Rust (powered by `terrio` and `richdem`).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use lidar::pointcloud::{GridMethod, GridStat, LidarToDsmOptions, convert_lidar, lidar_to_dsm};
use lidar::{
    DelineateOptions, ExtractSinksOptions, Topology, add_crs, clip_image, csv_points_to_vector,
    delineate_depressions, delineate_mounts, extract_sinks, flip_dem_file, gaussian_filter_file,
    image_to_cog, join_csv_to_vector, mean_filter_file, median_filter_file, mosaic,
    parse_resampling, polygonize, reproject_image, resample,
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
        #[arg(long)]
        comb_tif: bool,
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
            compress,
            json,
        } => {
            let opts = DelineateOptions {
                min_size,
                min_depth,
                interval,
                bool_level_shp: level_shp,
                bool_comb_tif: comb_tif,
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
            stat,
            method,
            window_size,
        } => {
            let opts = LidarToDsmOptions {
                resolution,
                radius,
                min_z,
                max_z,
                stat: GridStat::parse(&stat)?,
                method: GridMethod::parse(&method)?,
                window_size,
                ..Default::default()
            };
            let (cols, rows) = lidar_to_dsm(&input, &output, &opts)?;
            println!("Generated {cols}x{rows} DSM at {}", output.display());
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
