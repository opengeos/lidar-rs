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

pub mod animation;
pub mod catchment;
pub mod conditioning;
pub mod error;
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
pub mod federal;
pub mod filling;
pub mod filtering;
pub mod fsm;
pub mod graph;
pub mod inundation;
pub mod mounts;
#[cfg(feature = "pointcloud")]
pub mod pointcloud;
pub mod raster_ops;
pub mod region;
pub mod slicing;
pub mod tiling;

pub use animation::{
    ANIMATION_BANNER_HEIGHT, AnimateInundationOptions, AnimateInundationResult, HillshadeOptions,
    animate_inundation, animate_inundation_to_bytes, animate_inundation_to_bytes_with_spill_paths,
    compute_hillshade_rgb, encode_rgb_to_png_bytes, export_inundation_frames_png,
    load_dem_for_preview, render_dem_preview_png_bytes, render_inundation_frame_png_bytes,
};
pub use catchment::{
    CatchmentHierarchyOptions, CatchmentHierarchyRecord, CatchmentHierarchyResult, CatchmentRecord,
    DelineateCatchmentsOptions, DelineateCatchmentsResult, DelineateFlowPathsOptions,
    DelineateFlowPathsResult, FlowPathRecord, InMemoryCatchments, d8_flow_directions_resolved,
    d8_watersheds, delineate_catchment_hierarchy, delineate_catchment_hierarchy_in_memory,
    delineate_catchments, delineate_catchments_in_memory, delineate_flow_paths,
    extract_sink_labels_from_raster, write_catchment_hierarchy_csv, write_catchments_csv,
    write_flowpaths_csv,
};
pub use conditioning::{
    ArtifactClassifierOptions, BreachMode, BreachOptions, BurnCulvertsOptions,
    EmbankmentArtifactCandidate, SynthesizeCulvertsOptions, breach_depressions,
    breach_depressions_file, breach_depressions_file_with_no_data, burn_culverts,
    burn_culverts_file, classify_embankment_artifacts, synthesize_culverts,
};
pub use error::{Error, Result};
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
pub use federal::{
    Bbox, DemDownloadResult, DemProduct, DownloadOptions, FederalBundleOptions,
    FederalBundleResult, FederalDataset, NoaaAtlas14Table, Region, ResolvedRegion, TnmProductItem,
    WbdLevel, download_3dep_dem, download_federal_data, download_nhd, download_nwi, download_roads,
    download_wbd, fetch_noaa_atlas14, parse_dataset_list, query_3dep_products, resolve_region,
    vector_file_bbox_wgs84,
};
pub use filling::{
    ExtractSinksOptions, ExtractSinksResult, InMemorySinks, SinkDepression, extract_sinks,
    extract_sinks_in_memory, extract_sinks_with_filled_in_memory, write_sink_csv,
};
pub use filtering::{
    gaussian_filter, gaussian_filter_file, mean_filter, mean_filter_file, median_filter,
    median_filter_file,
};
pub use fsm::{
    FsmBasinSummary, FsmForcing, FsmHazardRasters, FsmHydrographPoint, FsmMassBalance,
    FsmSimulationResult, RoutingMode, SpillAttenuationOptions, attenuate_fsm_hydrograph,
    attenuate_fsm_hydrograph_with_options, compute_fsm_hazard_rasters, estimate_mean_travel_time_h,
    fsm_basins_to_csv_string, fsm_hydrograph_to_csv_string, integrate_fsm_hydrograph_m3,
    simulate_fill_spill_merge_in_memory, simulate_fill_spill_merge_with_green_ampt_in_memory,
    write_fsm_basins_csv, write_fsm_hazard_rasters, write_fsm_hydrograph_csv,
};
#[cfg(feature = "serde")]
pub use graph::write_connectivity_graph_json;
pub use graph::{
    ConnectivityGraph, GraphEdge, GraphNode, RatingCurvePoint, build_connectivity_graph,
    compute_rating_curves, connectivity_graph_to_graphml_string, rating_curves_to_csv_string,
    write_connectivity_graphml, write_rating_curves_csv, write_spill_paths_vector,
};
pub use inundation::{
    ForcingInput, GreenAmptParams, InMemoryFullPipeline, InMemoryFullPipelineOptions,
    InMemoryInundationPipeline, InundationForcing, InundationPipelineOptions, InundationStep,
    InundationStepSummary, RunPipelineOptions, RunPipelineResult,
    SimulateInundationAdvancedOptions, SimulateInundationAdvancedResult, SimulateInundationOptions,
    SimulateInundationResult, SpillFlowPath, compute_inundation_step_summaries,
    cumulative_runoff_mm, delineate_spill_flow_paths_in_memory, format_inundation_curve_csv,
    green_ampt_hyetograph_losses, green_ampt_infiltration_rate_cm_h, run_full_pipeline_in_memory,
    run_pipeline, scs_cn_incremental_hyetograph, scs_curve_number_runoff, scs_type2_hyetograph,
    simulate_inundation, simulate_inundation_advanced, simulate_inundation_in_memory,
    simulate_inundation_pipeline_in_memory, simulate_inundation_with_forcing_in_memory,
    write_inundation_summary_csv,
};
pub use mounts::{
    DelineateMountsResult, delineate_mounts, delineate_mounts_in_memory, flip_dem, flip_dem_file,
};
#[cfg(feature = "pointcloud")]
pub use pointcloud::{
    CsfOptions, GroundClassifyOptions, GroundFilterMethod, LidarReadOptions, LidarToDsmOptions,
    LidarToDtmOptions, PitFreeChmOptions, PmfOptions, ReturnKind, SmrfOptions, classify_ground,
    classify_ground_with_options, convert_lidar, filter_by_classification, filter_outliers_elm,
    filter_outliers_ror, filter_outliers_statistical, lidar_to_chm, lidar_to_chm_pit_free,
    lidar_to_dsm, lidar_to_dtm, normalize_hag, points_to_chm, points_to_chm_pit_free,
    points_to_dsm, points_to_dtm, read_lidar, read_lidar_filtered, read_lidar_in_polygon,
    read_lidar_windowed, write_lidar,
};
pub use raster_ops::{
    add_crs, clip_image, csv_points_to_vector, image_to_cog, join_csv_to_vector, mosaic,
    parse_resampling, polygonize, polygonize_array, reproject_image, resample,
};
pub use region::{
    CrsMetricScale, RegionProps, crs_metric_scale, region_group, region_group_slice, region_props,
    region_props_xy,
};
pub use slicing::{
    DelineateOptions, DelineateResult, InMemoryDepressions, NestedDepression,
    delineate_depressions, delineate_depressions_in_memory, write_depressions_csv,
    write_depressions_vector,
};
pub use tiling::{
    TiledDelineationOptions, TiledDelineationResult, delineate_depressions_tiled, load_dem_f32,
};

/// Re-exports from `richdem` (`Array2D`, `Topology`, `load_raster`, `save_raster`).
pub use richdem::{Array2D, Topology, load_raster, save_raster};
/// Re-export of `terrio::ops::Resampling`.
pub use terrio::ops::Resampling;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
