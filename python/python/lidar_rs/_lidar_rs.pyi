"""Type stubs for the compiled Rust extension module `lidar_rs._lidar_rs`."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import (
    AnimateInundationResult,
    CatchmentHierarchyRecord,
    CatchmentHierarchyResult,
    ConnectivityGraph,
    DelineateCatchmentsResult,
    DelineateFlowPathsResult,
    DelineateMountsResult,
    DelineationArrayResult,
    DelineationResult,
    DepressionRecord,
    Download3depResult,
    EmbankmentArtifactCandidate,
    ExtractSinksArrayResult,
    ExtractSinksResult,
    FederalDataResult,
    FsmHydrographPoint,
    Geotransform,
    GroundClassifyResult,
    InundationArrayResult,
    InundationResult,
    MergeTreeResult,
    NoaaAtlas14Table,
    PipelineResult,
    PointCloudDict,
    RatingCurve,
    SpillFlowPathRecord,
    TiledDelineationResult,
    TnmProductItem,
)

__version__: str

def cli_main(
    args: Sequence[str],
    lidar_cmd: Sequence[str] | None = None,
) -> int: ...

def mean_filter_array(
    arr: NDArray[np.float64],
    kernel_size: int = 3,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...

def median_filter_array(
    arr: NDArray[np.float64],
    kernel_size: int = 3,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...

def gaussian_filter_array(
    arr: NDArray[np.float64],
    sigma: float = 1.0,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...

def bilateral_filter_array(
    arr: NDArray[np.float64],
    spatial_sigma: float = 1.5,
    range_sigma: float = 0.5,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...

def bilateral_filter_file(
    in_dem: str,
    spatial_sigma: float = 1.5,
    range_sigma: float = 0.5,
    out_file: str | None = None,
    no_data: float | None = None,
) -> NDArray[np.float64] | None: ...

def feature_preserving_smoothing_array(
    arr: NDArray[np.float64],
    filter_size: int = 11,
    norm_diff: float = 15.0,
    num_iter: int = 3,
    max_z_change: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def feature_preserving_smoothing_file(
    in_dem: str,
    filter_size: int = 11,
    norm_diff: float = 15.0,
    num_iter: int = 3,
    max_z_change: float | None = None,
    out_file: str | None = None,
    no_data: float | None = None,
    compress: bool = False,
) -> NDArray[np.float64] | None: ...

def flip_dem_array(
    arr: NDArray[np.float64],
    delta: float = 100.0,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...

def mean_filter_file(
    in_dem: str,
    kernel_size: int = 3,
    out_file: str | None = None,
    no_data: float | None = None,
) -> NDArray[np.float64] | None: ...

def median_filter_file(
    in_dem: str,
    kernel_size: int = 3,
    out_file: str | None = None,
    no_data: float | None = None,
) -> NDArray[np.float64] | None: ...

def gaussian_filter_file(
    in_dem: str,
    sigma: float = 1.0,
    out_file: str | None = None,
    no_data: float | None = None,
) -> NDArray[np.float64] | None: ...

def flip_dem_file(
    in_dem: str,
    delta: float = 100.0,
    out_file: str | None = None,
    no_data: float | None = None,
) -> NDArray[np.float64] | None: ...

def extract_sinks(
    in_dem: str,
    min_size: int,
    out_dir: str,
    filled_dem: str | None = None,
    topology: str = "D8",
    keep_files: bool = True,
    compress: bool = False,
) -> ExtractSinksResult: ...

def extract_sinks_array(
    arr: NDArray[np.float64],
    min_size: int = 1000,
    topology: str = "D8",
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> ExtractSinksArrayResult: ...

def delineate_depressions(
    in_sink: str,
    min_size: int,
    min_depth: float,
    interval: float,
    out_dir: str,
    bool_level_shp: bool = False,
    bool_comb_tif: bool = False,
    compress: bool = False,
    export_gpkg: bool = False,
) -> DelineationResult: ...

def delineate_depressions_array(
    sink_arr: NDArray[np.float64],
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> DelineationArrayResult: ...

def delineate_depressions_tiled(
    in_dem: str,
    out_dir: str,
    tile_size: int = 2048,
    overlap: int = 128,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    use_f32: bool = True,
    delineation_method: str = "level_set",
    write_rasters: bool = False,
    write_gpkg: bool = False,
    compress: bool = False,
) -> TiledDelineationResult: ...

def delineate_mounts(
    in_dem: str,
    min_size: int,
    min_height: float,
    interval: float,
    out_dir: str,
    bool_shp: bool = False,
    compress: bool = False,
    export_gpkg: bool = False,
    bool_comb_tif: bool = False,
) -> DelineateMountsResult: ...

def delineate_mounts_array(
    dsm_arr: NDArray[np.float64],
    min_size: int = 1000,
    min_height: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> DelineationArrayResult: ...

def compute_hillshade_array(
    dem_arr: NDArray[np.float64],
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    no_data: float | None = None,
    crs: str | None = None,
) -> NDArray[np.uint8]: ...

def compute_hillshade_file(
    in_dem: str,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    no_data: float | None = None,
) -> NDArray[np.uint8]: ...

def compute_twi_array(
    dem_arr: NDArray[np.float64],
    flow_acc: NDArray[np.float64] | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def compute_twi_file(
    in_dem: str,
    out_file: str | None = None,
    flow_acc: str | None = None,
    no_data: float | None = None,
    compress: bool = False,
) -> NDArray[np.float64] | None: ...

def dinf_flow_direction_array(
    dem_arr: NDArray[np.float64],
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def dinf_flow_direction_file(
    in_dem: str,
    out_file: str | None = None,
    no_data: float | None = None,
    compress: bool = False,
) -> NDArray[np.float64] | None: ...

def dinf_flow_accumulation_array(
    dem_arr: NDArray[np.float64],
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def dinf_flow_accumulation_file(
    in_dem: str,
    out_file: str | None = None,
    no_data: float | None = None,
    compress: bool = False,
) -> NDArray[np.float64] | None: ...

def mfd_flow_accumulation_array(
    dem_arr: NDArray[np.float64],
    exponent: float = 1.1,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def mfd_flow_accumulation_file(
    in_dem: str,
    exponent: float = 1.1,
    out_file: str | None = None,
    no_data: float | None = None,
    compress: bool = False,
) -> NDArray[np.float64] | None: ...

def burn_streams(
    in_dem: str,
    streams_vector: str,
    out_dem: str,
    depth: float = 1.0,
    width_cells: int = 0,
    buffer_cells: int = 0,
    smooth_drop: float = 0.0,
    compress: bool = False,
) -> int: ...

def delineate_depressions_merge_tree(
    sink_arr: NDArray[np.float64],
    dem_arr: NDArray[np.float64] | None = None,
    min_size: int = 1000,
    min_depth: float = 0.3,
    topology: str = "D8",
    dem_is_refined: bool = False,
    max_rating_curve_points: int | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> MergeTreeResult: ...

def attenuate_fsm_hydrograph(
    hydrograph: Sequence[FsmHydrographPoint],
    mean_travel_time_h: float,
    dt_h: float = 0.25,
    num_reservoirs: int = 2,
    tail_duration_h: float | None = None,
) -> list[FsmHydrographPoint]: ...

def compute_hand_array(
    dem_arr: NDArray[np.float64],
    stream_mask: NDArray[np.uint8] | None = None,
    acc_threshold_cells: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def compute_hand_file(
    in_dem: str,
    out_file: str | None = None,
    stream_mask: str | None = None,
    acc_threshold_cells: float | None = None,
    no_data: float | None = None,
    compress: bool = False,
) -> NDArray[np.float64] | None: ...

def classify_embankment_artifacts_array(
    dem_arr: NDArray[np.float64],
    depression_id: NDArray[np.int32] | None = None,
    depressions: list[DepressionRecord] | None = None,
    max_barrier_width_m: float = 60.0,
    min_exit_drop: float = 0.05,
    score_threshold: float = 0.55,
    leaves_only: bool = True,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> list[EmbankmentArtifactCandidate]: ...

def classify_embankment_artifacts_file(
    in_dem: str,
    max_barrier_width_m: float = 60.0,
    min_exit_drop: float = 0.05,
    score_threshold: float = 0.55,
    leaves_only: bool = True,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
) -> list[EmbankmentArtifactCandidate]: ...

def compute_rating_curves(
    sink_arr: NDArray[np.float64],
    depression_id: NDArray[np.int32] | None = None,
    depression_level: NDArray[np.int32] | None = None,
    depressions: list[DepressionRecord] | None = None,
    interval: float = 0.3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    delineation_method: str = "level_set",
) -> list[RatingCurve]: ...

def build_connectivity_graph_from_dem(
    dem_arr: NDArray[np.float64],
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    kernel_size: int = 1,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    stream_vector: str | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> ConnectivityGraph: ...

def build_connectivity_graph_from_parts(
    depression_id: NDArray[np.int32],
    depression_level: NDArray[np.int32] | None,
    depressions: list[DepressionRecord],
    fill_time: NDArray[np.float64] | None = None,
    catchment_records: list[CatchmentHierarchyRecord] | None = None,
    spill_paths: list[SpillFlowPathRecord] | None = None,
    rain_intensity_cm_h: float = 5.0,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    stream_vector: str | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> ConnectivityGraph: ...

def delineate_catchments(
    in_dem: str,
    in_sink: str,
    out_dir: str,
    compress: bool = False,
) -> DelineateCatchmentsResult: ...

def delineate_flow_paths(
    in_dem_filled: str,
    in_sink: str,
    out_flowpath: str,
    rain_intensity_cm_h: float = 5.0,
    refined_dem: str | None = None,
) -> DelineateFlowPathsResult: ...

def delineate_catchment_hierarchy(
    in_dem: str,
    in_level_dir: str,
    out_catchment_img: str,
    compress: bool = False,
) -> CatchmentHierarchyResult: ...

def simulate_inundation(
    in_sink: str,
    in_catchment: str,
    out_dir: str,
    min_size: int = 1000,
    min_depth: float = 0.2,
    interval: float = 0.2,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    compress: bool = False,
) -> InundationResult: ...

def simulate_inundation_array(
    dem_arr: NDArray[np.float64],
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    runoff_coeff_arr: NDArray[np.float64] | None = None,
    infiltration_arr: NDArray[np.float64] | None = None,
    antecedent_depth_arr: NDArray[np.float64] | None = None,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    breach_max_depth: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    routing: str = "static",
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    runoff_coeff_no_data: float | None = None,
    infiltration_no_data: float | None = None,
    antecedent_depth_no_data: float | None = None,
    delineation_method: str = "level_set",
    green_ampt: tuple[float, float, float] | None = None,
    compute_hazard: bool = False,
    recession_et_cm_h: float | None = None,
    recession_seepage_cm_h: float | None = None,
    recession_max_h: float = 168.0,
) -> InundationArrayResult: ...

def run_pipeline(
    in_dem: str,
    out_dir: str,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    breach_max_depth: float | None = None,
    breach_max_length: int = 40,
    culverts_vector: str | None = None,
    runoff_coeff_path: str | None = None,
    infiltration_path: str | None = None,
    antecedent_depth_path: str | None = None,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    stream_vector: str | None = None,
    use_f32: bool = False,
    save_rasters: bool = True,
    export_vectors: bool = True,
    export_graph: bool = True,
    compress: bool = False,
    routing: str = "static",
    delineation_method: str = "level_set",
) -> PipelineResult: ...

def breach_depressions(
    in_dem: str,
    out_dem: str,
    mode: str = "selective",
    max_depth: float | None = None,
    max_length: int | None = None,
    fill_remaining: bool = False,
    compress: bool = False,
    no_data: float | None = None,
) -> str: ...

def breach_depressions_array(
    arr: NDArray[np.float64],
    mode: str = "selective",
    max_depth: float | None = None,
    max_length: int | None = None,
    fill_remaining: bool = False,
    no_data: float | None = None,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...

def burn_culverts(
    in_dem: str,
    culverts_vector: str,
    out_dem: str,
    half_width_cells: int = 0,
    depth_offset: float = 0.05,
    point_radius_cells: int = 5,
    compress: bool = False,
) -> int: ...

def synthesize_culverts(
    streams_vector: str,
    roads_vector: str,
    out_culverts: str,
    half_length_m: float = 25.0,
    snap_distance_m: float = 15.0,
    target_crs: str | None = None,
) -> int: ...

def query_3dep_products(
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    dem_product: str = "1m",
    deduplicate: bool = True,
) -> list[TnmProductItem]: ...

def download_3dep_dem(
    out_dir: str,
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    dem_product: str = "1m",
    mosaic_dem: bool = True,
    clip_to_region: bool = True,
    tile_cache_dir: str | None = None,
    overwrite: bool = False,
) -> Download3depResult: ...

def download_wbd(
    out_path: str,
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    level: int = 12,
    overwrite: bool = False,
) -> int: ...

def download_nhd(
    out_path: str,
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    clip_to_region: bool = True,
    overwrite: bool = False,
) -> int: ...

def download_nwi(
    out_path: str,
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    clip_to_region: bool = True,
    overwrite: bool = False,
) -> int: ...

def download_roads(
    out_path: str,
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    clip_to_region: bool = True,
    overwrite: bool = False,
) -> int: ...

def fetch_noaa_atlas14(
    lat: float,
    lon: float,
    out_json: str | None = None,
    overwrite: bool = False,
) -> NoaaAtlas14Table: ...

def download_federal_data(
    out_dir: str,
    bbox: tuple[float, float, float, float] | None = None,
    huc: str | None = None,
    region: str | None = None,
    datasets: str = "all",
    dem_product: str = "1m",
    wbd_level: int = 12,
    mosaic_dem: bool = True,
    clip_to_region: bool = True,
    tile_cache_dir: str | None = None,
    overwrite: bool = False,
) -> FederalDataResult: ...

def scs_curve_number_runoff(
    cn_arr: NDArray[np.float64],
    precip_mm: float,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...

def scs_type2_hyetograph(
    total_precip_mm: float,
    duration_h: float = 24.0,
    time_step_h: float = 1.0,
) -> list[tuple[float, float]]: ...

def animate_inundation(
    in_dem: str,
    in_flood_dir: str,
    out_gif: str,
    frame_delay_ms: int = 200,
    ping_pong: bool = True,
    loops: int = 0,
    water_alpha: float = 0.55,
    max_dimension: int | None = None,
    time_step_h: float | None = 1.0,
    rain_intensity_cm_h: float | None = None,
    azimuth_deg: float = 135.0,
    elevation_deg: float = 45.0,
    show_scale_bar: bool = False,
    show_legend: bool = False,
) -> AnimateInundationResult: ...

def export_inundation_frames_png(
    in_dem: str,
    in_fill_time: str,
    out_dir: str,
    num_steps: int = 10,
    time_step_h: float = 1.0,
    rain_intensity_cm_h: float | None = None,
    water_alpha: float = 0.55,
    max_dimension: int | None = None,
    show_scale_bar: bool = True,
    show_legend: bool = True,
    azimuth_deg: float = 135.0,
    elevation_deg: float = 45.0,
) -> list[str]: ...

def polygonize(
    in_raster: str,
    out_vector: str,
    field_name: str = "id",
) -> int: ...

def mosaic(
    inputs: Sequence[str],
    output: str,
    resampling: str = "nearest",
    compress: bool = False,
    cog: bool = False,
) -> None: ...

def reproject_image(
    input: str,
    output: str,
    dst_crs: str = "EPSG:4326",
    resampling: str = "nearest",
    resolution: tuple[float, float] | None = None,
    compress: bool = False,
    cog: bool = False,
) -> None: ...

def resample(
    input: str,
    output: str,
    resolution: float,
    resampling: str = "bilinear",
    compress: bool = False,
    cog: bool = False,
) -> None: ...

def clip_image(
    input: str,
    mask: str,
    output: str,
    compress: bool = False,
    cog: bool = False,
) -> None: ...

def image_to_cog(
    input: str,
    output: str,
    compression: str = "DEFLATE",
) -> None: ...

def add_crs(
    input: str,
    crs: str,
    output: str | None = None,
) -> None: ...

def join_tables(
    in_vector: str,
    in_csv: str,
    out_vector: str,
    vec_col: str = "id",
    csv_col: str = "region_id",
) -> int: ...

def csv_to_vector(
    in_csv: str,
    out_vector: str,
    latitude: str = "latitude",
    longitude: str = "longitude",
) -> int: ...

def convert_lidar(
    source: str,
    destination: str,
    point_format_id: int | None = None,
    minor_version: int | None = None,
) -> int: ...

def read_lidar(
    path: str,
    bounds: tuple[float, float, float, float] | None = None,
    polygon_path: str | None = None,
    max_points: int | None = None,
    min_z: float | None = None,
    max_z: float | None = None,
    classifications: Sequence[int] | None = None,
    return_kinds: Sequence[str] | None = None,
    max_level: int | None = None,
    resolution: float | None = None,
) -> PointCloudDict: ...

def write_lidar(
    path: str,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    classification: NDArray[np.uint8] | None = None,
    intensity: NDArray[np.uint16] | None = None,
    return_number: NDArray[np.uint8] | None = None,
    number_of_returns: NDArray[np.uint8] | None = None,
    crs: str | None = None,
) -> int: ...

def classify_ground_file(
    in_lidar: str,
    out_lidar: str | None = None,
    method: str = "csf",
    ground_class: int = 2,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
) -> GroundClassifyResult: ...

def classify_ground_array(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    classification: NDArray[np.uint8] | None = None,
    intensity: NDArray[np.uint16] | None = None,
    return_number: NDArray[np.uint8] | None = None,
    number_of_returns: NDArray[np.uint8] | None = None,
    crs: str | None = None,
    out_lidar: str | None = None,
    method: str = "csf",
    ground_class: int = 2,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
) -> GroundClassifyResult: ...

def lidar_to_dsm(
    in_lidar: str,
    out_dsm: str,
    resolution: float = 1.0,
    radius: float | None = None,
    min_z: float | None = None,
    max_z: float | None = None,
    stat: str = "max",
    method: str = "statistics",
    window_size: int = 0,
    classifications: Sequence[int] | None = None,
    return_kinds: Sequence[str] | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]: ...

def lidar_to_dtm(
    in_lidar: str,
    out_dtm: str,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    radius: float | None = None,
    method: str = "linear",
    stat: str = "idw",
    window_size: int = 5,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]: ...

def lidar_to_chm(
    in_lidar: str,
    out_chm: str,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    radius: float | None = None,
    window_size: int = 5,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]: ...

def lidar_to_chm_pit_free(
    in_lidar: str,
    out_chm: str,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    height_thresholds: Sequence[float] | None = None,
    max_triangle_edge: float | None = None,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]: ...

def lidar_to_dtm_tiled(
    in_lidar: str,
    out_dtm: str,
    tile_size: float = 500.0,
    overlap: float = 25.0,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    method: str = "linear",
    stat: str = "idw",
    window_size: int = 5,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
    read_resolution: float | None = None,
    max_level: int | None = None,
) -> tuple[int, int]: ...

def lidar_to_chm_tiled(
    in_lidar: str,
    out_chm: str,
    tile_size: float = 500.0,
    overlap: float = 25.0,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    window_size: int = 5,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
    read_resolution: float | None = None,
    max_level: int | None = None,
) -> tuple[int, int]: ...

def stream_remote_pointcloud_aoi(
    source_url: str,
    bounds_wgs84: tuple[float, float, float, float] | None = None,
    bounds_native: tuple[float, float, float, float] | None = None,
    classifications: Sequence[int] | None = None,
    return_kinds: Sequence[str] | None = None,
    max_level: int | None = None,
    resolution: float | None = None,
    max_points: int | None = None,
) -> PointCloudDict: ...
