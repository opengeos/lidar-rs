"""TypedDict definitions and type aliases for `lidar_rs`."""

from __future__ import annotations

import os
from typing import Any, NotRequired, Protocol, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _XarrayLike(Protocol):
    @property
    def values(self) -> Any: ...

    @property
    def dims(self) -> Any: ...


class _RasterioLike(Protocol):
    @property
    def transform(self) -> Any: ...

    def read(self, band: int = 1) -> Any: ...


PathLikeStr = str | os.PathLike[str]
Geotransform = tuple[float, float, float, float, float, float]
SpatialArrayLike = ArrayLike | _XarrayLike | _RasterioLike


class SinkDepressionRecord(TypedDict):
    """Geometric and topographic attributes of a single top-level sink region."""

    region_id: int
    count: int
    area: float
    volume: float
    avg_depth: float
    max_depth: float
    min_elev: float
    max_elev: float
    perimeter: float
    major_axis: float
    minor_axis: float
    elongatedness: float
    eccentricity: float
    orientation: float
    area_bbox_ratio: float


class DepressionRecord(TypedDict):
    """Attributes of a single depression or mount in the nested hierarchy."""

    id: int
    level: int
    count: int
    area: float
    volume: float
    avg_depth: float
    max_depth: float
    min_elev: float
    max_elev: float
    children_ids: list[int]
    region_id: int
    perimeter: float
    major_axis: float
    minor_axis: float
    elongatedness: float
    eccentricity: float
    orientation: float
    area_bbox_ratio: float
    parent_id: int
    circularity: float
    depth_diam_ratio: float
    hypsometric_integral: float
    rim_slope_deg: float
    artifact_flag: bool


class ExtractSinksResult(TypedDict):
    """Output paths and attributes produced by file-based `extract_sinks`."""

    sink_path: str | None
    refined_dem_path: str | None
    filled_dem_path: str | None
    dem_diff_path: str | None
    depth_path: str | None
    region_path: str | None
    csv_path: str | None
    shapefile_path: str | None
    gpkg_path: str
    depressions: list[SinkDepressionRecord]


class ExtractSinksArrayResult(TypedDict):
    """In-memory arrays and attributes produced by `extract_sinks_array`."""

    sink: NDArray[np.float64]
    dem_refined: NDArray[np.float64]
    dem_filled: NDArray[np.float64]
    dem_diff: NDArray[np.float64]
    depth: NDArray[np.float64]
    region: NDArray[np.int32]
    depressions: list[SinkDepressionRecord]
    geotransform: Geotransform
    crs: str | None


class DelineationResult(TypedDict):
    """Output paths and hierarchy summary produced by `delineate_depressions`."""

    depression_id_path: str
    depression_level_path: str
    shapefile_path: str
    gpkg_path: str | None
    csv_path: str
    level_shapefiles: list[str]
    max_level: int
    num_regions: int
    depressions: list[DepressionRecord]


class DelineateMountsResult(DelineationResult):
    """Output paths and hierarchy summary produced by `delineate_mounts`."""

    flipped_dem_path: str


class DelineationArrayResult(TypedDict):
    """In-memory arrays and hierarchy summary produced by `delineate_depressions_array` and `delineate_mounts_array`."""

    depression_id: NDArray[np.int32]
    depression_level: NDArray[np.int32]
    max_level: int
    num_regions: int
    depressions: list[DepressionRecord]
    geotransform: Geotransform
    crs: str | None


class TiledDelineationResult(TypedDict):
    """Summary produced by `delineate_depressions_tiled`."""

    num_tiles: int
    max_level: int
    csv_path: str
    depressions: list[DepressionRecord]


class CatchmentRecord(TypedDict):
    """Contributing catchment record for a single sink."""

    id: int
    sink_cells: int
    sink_area: float
    catchment_cells: int
    catchment_area: float
    dep_to_cat_ratio: float


class DelineateCatchmentsResult(TypedDict):
    """Output paths and records produced by `delineate_catchments`."""

    watershed_path: str
    vector_path: str
    csv_path: str
    records: list[CatchmentRecord]


class FlowPathRecord(TypedDict):
    """Overland spill flow-path record between sinks."""

    id: int
    start_sink: int
    end_sink: int
    start_elev: float
    end_elev: float
    diff_elev: float
    length: float
    volume: float
    cat_area: float
    simu_depth: float
    rain_intensity_m_h: float
    time_inund: float
    rank: int


class DelineateFlowPathsResult(TypedDict):
    """Output paths and records produced by `delineate_flow_paths`."""

    flowpath_vector_path: str
    csv_path: str
    records: list[FlowPathRecord]


class CatchmentHierarchyRecord(TypedDict):
    """Nested catchment statistics at one hierarchy level."""

    level: int
    level_fid: int
    count: int
    area: float
    cat_area: float
    dep_to_cat_ratio: float
    min_elev: float
    max_elev: float
    range: float
    mean_elev: float
    std_elev: float
    sum_elev: float
    volume: float
    mean_depth: float


class CatchmentHierarchyResult(TypedDict):
    """Output paths and records produced by `delineate_catchment_hierarchy`."""

    catchment_raster_path: str
    merged_shapefile_path: str
    csv_path: str
    catchment_shapefiles: list[str]
    level_shapefiles: list[str]
    records: list[CatchmentHierarchyRecord]


class InundationStepSummaryRecord(TypedDict):
    """In-memory inundation summary at one simulation time step."""

    step: int
    time_hours: float
    inundated_cells: int
    inundated_area: float
    inundated_fraction: float


class InundationStepRecord(InundationStepSummaryRecord):
    """File-based inundation summary at one simulation time step."""

    raster_path: str


class SpillFlowPathRecord(TypedDict):
    """Simulated overland spill path between depressions."""

    from_dep_id: int
    from_level: int
    to_dep_id: int
    spill_time_h: float
    merge_time_h: float | None
    spill_elev: float
    cells: list[tuple[int, int]]


class RatingCurvePoint(TypedDict):
    """Single stage-area-volume sample on a depression rating curve."""

    dep_id: int
    level: int
    region_id: int
    stage: float
    depth: float
    area: float
    volume: float


class RatingCurve(TypedDict):
    """Per-depression stage-area-volume rating curve."""

    id: int
    dep_id: int
    level: int
    region_id: int
    stages: list[float]
    depths: list[float]
    areas: list[float]
    volumes: list[float]


class GraphNode(TypedDict):
    """Node representing a depression in the directed connectivity graph."""

    id: int
    parent_id: int
    level: int
    region_id: int
    area: float
    volume: float
    avg_depth: float
    max_depth: float
    min_elev: float
    max_elev: float
    spill_elev: float
    circularity: float
    hypsometric_integral: float
    cat_area: float
    fill_time_h: float | None
    fill_runoff_mm: float | None


class GraphEdge(TypedDict):
    """Directed spill or hierarchical merge edge in the connectivity graph."""

    from_id: int
    to_id: int
    edge_type: str
    spill_time_h: float | None
    spill_runoff_mm: float | None
    merge_time_h: float | None
    spill_elev: float | None
    length_m: float
    start_xy: tuple[float, float] | None
    end_xy: tuple[float, float] | None
    pour_row: int | None
    pour_col: int | None
    dist_to_stream: float | None


class ConnectivityGraph(TypedDict):
    """Directed depression spill and hierarchical merge connectivity graph."""

    rain_intensity_cm_h: float
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class FsmMassBalance(TypedDict):
    """Domain-wide mass balance summary for Fill-Spill-Merge routing."""

    duration_h: float
    total_antecedent_m3: float
    total_runoff_m3: float
    total_stored_m3: float
    total_infiltration_m3: float
    total_boundary_outflow_m3: float
    mass_balance_error_m3: float


class FsmBasinSummary(TypedDict):
    """Per-depression mass balance and spill summary for Fill-Spill-Merge routing."""

    dep_id: int
    parent_id: int
    level: int
    capacity_m3: float
    local_cat_area_m2: float
    local_eff_cat_area_m2: float
    final_routed_cat_area_m2: float
    initial_volume_m3: float
    final_volume_m3: float
    fill_time_h: float | None
    spill_to_dep_id: int | None
    merge_time_h: float | None
    spilled_volume_m3: float
    peak_spill_rate_m3_h: float
    infiltrated_volume_m3: float


class FsmHydrographPoint(TypedDict):
    """Single point on a Fill-Spill-Merge watershed outlet hydrograph."""

    time_h: float
    q_boundary_m3_s: float
    total_stored_m3: float
    connected_area_fraction: float
    spilling_basin_count: int


class RecessionCurvePoint(TypedDict):
    """Single point on a post-storm pond recession curve."""

    time_h: float
    stored_vol_m3: float
    inundated_area_m2: float
    wet_basin_count: int


class MergeTreeNode(TypedDict):
    """Exact bottom-up merge-tree depression node."""

    id: int
    parent_id: int
    children_ids: list[int]
    level: int
    region_id: int
    min_elev: float
    saddle_elev: float
    pour_point_row: int
    pour_point_col: int
    pour_point_x: float
    pour_point_y: float
    count: int
    area: float
    volume: float
    max_depth: float
    avg_depth: float
    is_leaf: bool


class MergeTreeResult(TypedDict):
    """In-memory output of `delineate_depressions_merge_tree`."""

    depression_id: NDArray[np.int32]
    depression_level: NDArray[np.int32]
    max_level: int
    num_regions: int
    depressions: list[DepressionRecord]
    nodes: list[MergeTreeNode]
    rating_curves: list[RatingCurvePoint]
    geotransform: Geotransform
    crs: str | None


class InundationResult(TypedDict):
    """Output paths and time steps produced by file-based `simulate_inundation`."""

    frame_paths: list[str]
    fill_time_path: str
    summary_csv_path: str
    num_regions: int
    steps: list[InundationStepRecord]


class InundationArrayResult(TypedDict):
    """Full in-memory inundation pipeline result produced by `simulate_inundation_array`."""

    depression_id: NDArray[np.int32]
    depression_level: NDArray[np.int32]
    catchment_hierarchy: NDArray[np.float64]
    fill_time: NDArray[np.float64]
    depressions: list[DepressionRecord]
    steps: list[InundationStepSummaryRecord]
    spill_paths: list[SpillFlowPathRecord]
    rating_curves: list[RatingCurvePoint]
    connectivity_graph: ConnectivityGraph
    connectivity_graphml: str
    inundation_curve_csv: str
    catchment_records: list[CatchmentHierarchyRecord]
    graph_node_count: int
    graph_edge_count: int
    fsm_mass_balance: FsmMassBalance | None
    fsm_basins: list[FsmBasinSummary] | None
    fsm_hydrograph: list[FsmHydrographPoint] | None
    fsm_hazard_max_depth: NotRequired[NDArray[np.float64]]
    fsm_hazard_wse: NotRequired[NDArray[np.float64]]
    fsm_hazard_duration_h: NotRequired[NDArray[np.float64]]
    fsm_recession_drain_time_h: NotRequired[NDArray[np.float64]]
    fsm_recession_remaining_m3: NotRequired[float]
    fsm_recession_drained_m3: NotRequired[float]
    fsm_recession_curve: NotRequired[list[RecessionCurvePoint]]
    geotransform: Geotransform
    crs: str | None


class AnimateInundationResult(TypedDict):
    """Output metadata produced by `animate_inundation`."""

    gif_path: str
    frame_count: int
    width: int
    height: int


class PipelineResult(TypedDict):
    """Summary and output paths produced by `run_pipeline`."""

    out_dir: str
    width: int
    height: int
    num_sink_regions: int
    num_depressions: int
    max_level: int
    total_storage_m3: float
    num_spill_paths: int
    depressions_csv: str
    catchments_csv: str
    inundation_csv: str
    summary_json: str
    depressions_gpkg: str | None
    spill_paths_gpkg: str | None
    rating_curves_csv: str | None
    connectivity_graph_json: str | None
    connectivity_graphml: str | None
    routing: str
    fsm_basins_csv: str | None
    fsm_mass_balance: FsmMassBalance | None


class TnmProductItem(TypedDict):
    """Metadata record for a USGS 3DEP product item."""

    grid_key: str
    title: str
    publication_date: str
    download_url: str
    size_bytes: int | None
    format: str
    bbox: tuple[float, float, float, float]


class Download3depResult(TypedDict):
    """Output paths and metadata produced by `download_3dep_dem`."""

    bbox: tuple[float, float, float, float]
    products: list[TnmProductItem]
    tile_paths: list[str]
    mosaic_path: str | None


class NoaaAtlas14Table(TypedDict):
    """NOAA Atlas 14 precipitation frequency table."""

    lat: float
    lon: float
    units: str
    depths_mm: dict[str, dict[int, float]]


class FederalDataResult(TypedDict):
    """Output paths and counts produced by `download_federal_data`."""

    region_bbox: tuple[float, float, float, float]
    wbd_path: str | None
    wbd_count: int
    dem_tiles: list[str]
    dem_mosaic_path: str | None
    nhd_path: str | None
    nhd_count: int
    nwi_path: str | None
    nwi_count: int
    roads_path: str | None
    roads_count: int
    culverts_path: str | None
    culverts_count: int
    noaa_atlas14_path: str | None
    noaa_atlas14: NoaaAtlas14Table | None
    manifest_path: str


class PointCloudDict(TypedDict):
    """NumPy-backed LiDAR point cloud returned by `read_lidar`."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    classification: NDArray[np.uint8] | None
    intensity: NDArray[np.uint16] | None
    return_number: NDArray[np.uint8] | None
    number_of_returns: NDArray[np.uint8] | None
    crs: str | None
    count: int


class GroundClassifyResult(TypedDict):
    """Result returned by `classify_ground`."""

    method: str
    total_points: int
    ground_points: int
    non_ground_points: int
    ground_class: int
    output_path: str | None
    classification: NDArray[np.uint8]


class EmbankmentArtifactCandidate(TypedDict):
    """Morphometric and barrier-profile assessment of a potential embankment artifact depression."""

    dep_id: int
    level: int
    area_m2: float
    max_depth: float
    elongatedness: float
    circularity: float
    compactness: float
    barrier_width_m: float | None
    barrier_height_ratio: float
    rim_linearity: float
    artifact_score: float
    is_likely_artifact: bool
    suggested_breach: tuple[tuple[float, float], tuple[float, float]] | None


__all__ = [
    "AnimateInundationResult",
    "CatchmentHierarchyRecord",
    "CatchmentHierarchyResult",
    "CatchmentRecord",
    "ConnectivityGraph",
    "DelineateCatchmentsResult",
    "DelineateFlowPathsResult",
    "DelineateMountsResult",
    "DelineationArrayResult",
    "DelineationResult",
    "DepressionRecord",
    "Download3depResult",
    "EmbankmentArtifactCandidate",
    "ExtractSinksArrayResult",
    "ExtractSinksResult",
    "FederalDataResult",
    "FlowPathRecord",
    "FsmBasinSummary",
    "FsmHydrographPoint",
    "FsmMassBalance",
    "Geotransform",
    "GraphEdge",
    "GraphNode",
    "GroundClassifyResult",
    "InundationArrayResult",
    "InundationResult",
    "InundationStepRecord",
    "InundationStepSummaryRecord",
    "MergeTreeNode",
    "MergeTreeResult",
    "NoaaAtlas14Table",
    "PathLikeStr",
    "PipelineResult",
    "PointCloudDict",
    "RatingCurve",
    "RatingCurvePoint",
    "RecessionCurvePoint",
    "SinkDepressionRecord",
    "SpatialArrayLike",
    "SpillFlowPathRecord",
    "TiledDelineationResult",
    "TnmProductItem",
]
