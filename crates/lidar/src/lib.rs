//! # lidar-rs
//!
//! Pure-Rust port of [`opengeos/lidar`](https://github.com/opengeos/lidar) for
//! delineating the nested hierarchy of surface depressions and elevated features
//! (mounts) in digital elevation models (DEMs), powered by
//! [`terrio`](https://github.com/opengeos/terrio) for GDAL-free geospatial I/O
//! and [`richdem`](https://github.com/opengeos/richdem-rs) for Priority-Flood
//! depression filling.
//!
//! ## Features
//!
//! - **DEM smoothing (`filtering`):** Mean, median, and Gaussian filters with
//!   NoData safety and symmetric boundary reflection.
//! - **Sink extraction (`filling`):** Priority-Flood depression filling via
//!   `richdem`, 4-connected region grouping, artifact depression removal,
//!   refined DEM generation, geometric/topographic depression attributes, and
//!   GeoTIFF / CSV / Shapefile / GeoPackage export via `terrio`.
//! - **Nested depression hierarchy (`slicing`):** Wu et al. (2019) level-set
//!   algorithm for delineating nested surface depressions, computing 18
//!   topological & geometric attributes, and exporting per-level shapefiles.
//! - **Nested mount hierarchy (`mounts`):** DEM inversion (`flip_dem`) and
//!   nested elevated feature delineation (`delineate_mounts`).
//! - **Geospatial raster & vector utilities (`raster_ops`):** Pure-Rust
//!   `polygonize`, `mosaic`, `reproject_image`, `resample`, `clip_image`,
//!   `image_to_cog`, `add_crs`, `join_csv_to_vector`, and `csv_points_to_vector`.
//! - **LiDAR point clouds (`pointcloud`):** Pure-Rust LAS/LAZ/COPC/PLY/E57
//!   reading, writing, format conversion, and DSM gridding via `terrio`.
//!
//! ## Citations
//!
//! - Wu, Q. (2021). lidar: A Python package for delineating nested surface
//!   depressions from digital elevation data. *Journal of Open Source Software*,
//!   6(59), 2965. <https://doi.org/10.21105/joss.02965>
//! - Wu, Q., Lane, C.R., Wang, L., Vanderhoof, M.K., Christensen, J.R., &
//!   Liu, H. (2019). Efficient Delineation of Nested Depression Hierarchy in
//!   Digital Elevation Models for Hydrological Analysis Using Level-Set Method.
//!   *Journal of the American Water Resources Association*, 55(2), 354–368.
//!   <https://doi.org/10.1111/1752-1688.12689>

pub mod error;
pub mod filling;
pub mod filtering;
pub mod mounts;
#[cfg(feature = "pointcloud")]
pub mod pointcloud;
pub mod raster_ops;
pub mod region;
pub mod slicing;

pub use error::{Error, Result};
pub use filling::{
    ExtractSinksOptions, ExtractSinksResult, InMemorySinks, SinkDepression, extract_sinks,
    extract_sinks_in_memory, extract_sinks_with_filled_in_memory, write_sink_csv,
};
pub use filtering::{
    gaussian_filter, gaussian_filter_file, mean_filter, mean_filter_file, median_filter,
    median_filter_file,
};
pub use mounts::{DelineateMountsResult, delineate_mounts, flip_dem, flip_dem_file};
#[cfg(feature = "pointcloud")]
pub use pointcloud::{LidarToDsmOptions, convert_lidar, lidar_to_dsm, read_lidar, write_lidar};
pub use raster_ops::{
    add_crs, clip_image, csv_points_to_vector, image_to_cog, join_csv_to_vector, mosaic,
    parse_resampling, polygonize, polygonize_array, reproject_image, resample,
};
pub use region::{RegionProps, region_group, region_props, region_props_xy};
pub use slicing::{
    DelineateOptions, DelineateResult, InMemoryDepressions, NestedDepression,
    delineate_depressions, delineate_depressions_in_memory, write_depressions_csv,
};

/// Re-exports from `richdem` (`Array2D`, `Topology`, `load_raster`, `save_raster`).
pub use richdem::{Array2D, Topology, load_raster, save_raster};
/// Re-export of `terrio::ops::Resampling`.
pub use terrio::ops::Resampling;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
