/**
 * TypeScript definitions for `lidar-wasm` (WebAssembly bindings for `lidar-rs`).
 *
 * Covers every `#[wasm_bindgen]` export in `js/src/lib.rs` together with typed
 * interfaces for all parsed JSON and GeoJSON payloads.
 */

/** Supported least-cost depression breaching modes for {@link breachDem}. */
export type BreachMode = "complete" | "selective" | "constrained";

/** Supported hydrological routing modes for {@link simulateInundationAdvanced}. */
export type RoutingMode = "static" | "fsm";

/** Supported depression delineation algorithms for {@link simulateInundationAdvanced}. */
export type DelineationMethod = "level_set" | "merge_tree";

/** Parsed result of {@link rasterInfo}. */
export interface RasterInfo {
  width: number;
  height: number;
  geotransform: [number, number, number, number, number, number];
  projection: string;
  no_data: number | null;
  min: number;
  max: number;
}

/** Single surface depression record parsed from {@link SinksOutput.depressionsJson}. */
export interface SinkDepression {
  region_id: number;
  count: number;
  area: number;
  volume: number;
  avg_depth: number;
  max_depth: number;
  min_elev: number;
  max_elev: number;
  perimeter: number;
  major_axis: number;
  minor_axis: number;
  elongatedness: number;
  eccentricity: number;
  orientation: number;
  area_bbox_ratio: number;
}

/** Nested depression (or mount) record parsed from {@link DepressionsOutput.depressionsJson} or {@link InundationOutput.depressionsJson}. */
export interface NestedDepression {
  id: number;
  level: number;
  count: number;
  area: number;
  volume: number;
  avg_depth: number;
  max_depth: number;
  min_elev: number;
  max_elev: number;
  children_ids: number[];
  region_id: number;
  perimeter: number;
  major_axis: number;
  minor_axis: number;
  elongatedness: number;
  eccentricity: number;
  orientation: number;
  area_bbox_ratio: number;
  parent_id: number;
  circularity: number;
  depth_diam_ratio: number;
  hypsometric_integral: number;
  rim_slope_deg: number;
  artifact_flag: boolean;
}

/** Per-catchment summary parsed from {@link CatchmentsOutput.recordsJson}. */
export interface CatchmentRecord {
  id: number;
  sink_cells: number;
  sink_area: number;
  catchment_cells: number;
  catchment_area: number;
  dep_to_cat_ratio: number;
}

/** Per-level catchment hierarchy statistics parsed from {@link InundationOutput.catchmentHierarchyJson}. */
export interface CatchmentHierarchyRecord {
  level: number;
  level_fid: number;
  count: number;
  area: number;
  cat_area: number;
  dep_to_cat_ratio: number;
  min_elev: number;
  max_elev: number;
  range: number;
  mean_elev: number;
  std_elev: number;
  sum_elev: number;
  volume: number;
  mean_depth: number;
}

/** Single time-step inundation summary parsed from {@link InundationOutput.stepsJson}. */
export interface InundationStepSummary {
  step: number;
  time_hours: number;
  inundated_cells: number;
  inundated_area: number;
  inundated_fraction: number;
}

/** Overland spill flow path parsed from {@link InundationOutput.spillPathsJson}. */
export interface SpillFlowPath {
  from_dep_id: number;
  from_level: number;
  to_dep_id: number;
  spill_time_h: number;
  merge_time_h: number | null;
  spill_elev: number;
  cells: Array<[number, number]>;
}

/** Stage-storage-area sample parsed from {@link InundationOutput.ratingCurvesJson}. */
export interface RatingCurvePoint {
  dep_id: number;
  level: number;
  region_id: number;
  stage: number;
  depth: number;
  area: number;
  volume: number;
}

/** Node in the directed hydrological connectivity graph. */
export interface GraphNode {
  id: number;
  parent_id: number;
  level: number;
  region_id: number;
  area: number;
  volume: number;
  avg_depth: number;
  max_depth: number;
  min_elev: number;
  max_elev: number;
  circularity: number;
  hypsometric_integral: number;
  cat_area: number;
  fill_time_h: number | null;
  fill_runoff_mm: number | null;
}

/** Directed spillover or hierarchical merge edge in the connectivity graph. */
export interface GraphEdge {
  from_id: number;
  to_id: number;
  edge_type: "spill" | "merge" | string;
  spill_time_h: number | null;
  spill_runoff_mm: number | null;
  merge_time_h: number | null;
  spill_elev: number | null;
  length_m: number;
  start_xy: [number, number] | null;
  end_xy: [number, number] | null;
  dist_to_stream: number | null;
}

/** Directed nested connectivity graph parsed from {@link InundationOutput.connectivityGraphJson}. */
export interface ConnectivityGraph {
  rain_intensity_cm_h: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/** Dynamic Fill-Spill-Merge watershed mass balance parsed from {@link InundationOutput.fsmMassBalanceJson}. */
export interface FsmMassBalance {
  duration_h: number;
  total_antecedent_m3: number;
  total_runoff_m3: number;
  total_stored_m3: number;
  total_infiltration_m3: number;
  total_boundary_outflow_m3: number;
  mass_balance_error_m3: number;
}

/** Per-depression Fill-Spill-Merge summary parsed from {@link InundationOutput.fsmBasinsJson}. */
export interface FsmBasinSummary {
  dep_id: number;
  parent_id: number;
  level: number;
  capacity_m3: number;
  local_cat_area_m2: number;
  local_eff_cat_area_m2: number;
  final_routed_cat_area_m2: number;
  initial_volume_m3: number;
  final_volume_m3: number;
  fill_time_h: number | null;
  spill_to_dep_id: number | null;
  merge_time_h: number | null;
  spilled_volume_m3: number;
  peak_spill_rate_m3_h: number;
  infiltrated_volume_m3: number;
}

/** GeoJSON coordinate pair `[longitude, latitude]` (or projected `[x, y]` when unprojected). */
export type GeoJsonPosition = [number, number];

export interface GeoJsonPolygon {
  type: "Polygon";
  coordinates: GeoJsonPosition[][];
}

export interface GeoJsonMultiPolygon {
  type: "MultiPolygon";
  coordinates: GeoJsonPosition[][][];
}

export interface GeoJsonLineString {
  type: "LineString";
  coordinates: GeoJsonPosition[];
}

export type GeoJsonPolygonGeometry = GeoJsonPolygon | GeoJsonMultiPolygon;

export interface GeoJsonFeature<G, P> {
  type: "Feature";
  id?: number | string;
  geometry: G;
  properties: P;
}

export interface GeoJsonFeatureCollection<G, P> {
  type: "FeatureCollection";
  name?: string;
  features: Array<GeoJsonFeature<G, P>>;
}

/** Properties attached to each feature in {@link SinksOutput.geojson}. */
export interface SinkGeoJsonProperties {
  region_id: number;
  count: number;
  area: number;
  volume: number;
  avg_depth: number;
  max_depth: number;
  min_elev: number;
  max_elev: number;
  perimeter: number;
  major_axis: number;
  minor_axis: number;
  elongatedness: number;
  eccentricity: number;
  orientation: number;
  area_bbox_ratio: number;
}

/** Properties attached to each feature in {@link DepressionsOutput.geojson} and {@link InundationOutput.depressionsGeojson}. */
export interface DepressionGeoJsonProperties {
  id: number;
  parent_id: number;
  level: number;
  region_id: number;
  count: number;
  area: number;
  volume: number;
  avg_depth: number;
  max_depth: number;
  min_elev: number;
  max_elev: number;
  perimeter: number;
  major_axis: number;
  minor_axis: number;
  elongatedness: number;
  eccentricity: number;
  orientation: number;
  area_bbox_ratio: number;
  circularity: number;
  depth_diam_ratio: number;
  hypsometric_integral: number;
  rim_slope_deg: number;
  artifact_flag: number;
}

/** Properties attached to each feature in {@link InundationOutput.spillPathsGeojson}. */
export interface SpillPathGeoJsonProperties {
  from_dep_id: number;
  from_level: number;
  to_dep_id: number;
  spill_time_h: number;
  merge_time_h: number | null;
  spill_elev: number;
  cell_count: number;
}

export type SinksFeatureCollection = GeoJsonFeatureCollection<
  GeoJsonPolygonGeometry,
  SinkGeoJsonProperties
>;

export type DepressionsFeatureCollection = GeoJsonFeatureCollection<
  GeoJsonPolygonGeometry,
  DepressionGeoJsonProperties
>;

export type SpillPathsFeatureCollection = GeoJsonFeatureCollection<
  GeoJsonLineString,
  SpillPathGeoJsonProperties
>;

/** Output container returned by {@link extractSinks}. Call `.free()` when done. */
export class SinksOutput {
  free(): void;
  readonly width: number;
  readonly height: number;
  depressionsJson(): string;
  sinkTif(): Uint8Array;
  refinedDemTif(): Uint8Array;
  depthTif(): Uint8Array;
  regionTif(): Uint8Array;
  geojson(): string;
  depthRgba(): Uint8Array;
}

/** Single point on an FSM watershed outlet hydrograph parsed from {@link InundationOutput.fsmHydrographJson}. */
export interface FsmHydrographPoint {
  time_h: number;
  q_boundary_m3_s: number;
  total_stored_m3: number;
  connected_area_fraction: number;
  spilling_basin_count: number;
}

/** Merge-tree node metadata parsed from {@link MergeTreeOutput.nodesJson} (internal saddles are exact; sink-only root `saddle_elev` and rating-curve endpoints are lower-bound estimates). */
export interface MergeTreeNode {
  id: number;
  parent_id: number;
  children_ids: number[];
  level: number;
  region_id: number;
  min_elev: number;
  saddle_elev: number;
  pour_point_row: number;
  pour_point_col: number;
  pour_point_x: number;
  pour_point_y: number;
  count: number;
  area: number;
  volume: number;
  max_depth: number;
  avg_depth: number;
  is_leaf: boolean;
}

/** Output container returned by {@link delineateDepressionsMergeTree}. Call `.free()` when done. */
export class MergeTreeOutput {
  private constructor();
  free(): void;
  readonly width: number;
  readonly height: number;
  readonly projection: string;
  readonly maxLevel: number;
  readonly numRegions: number;
  boundsWgs84(): Float64Array | undefined;
  depressionsJson(): string;
  nodesJson(): string;
  ratingCurvesJson(): string;
  ratingCurvesCsv(): string;
  depressionIdValues(): Int32Array;
  idTif(): Uint8Array;
  levelTif(): Uint8Array;
  geojson(): string;
  levelRgba(): Uint8Array;
}

/** Output container returned by {@link delineateDepressions} and {@link delineateMounts}. Call `.free()` when done. */
export class DepressionsOutput {
  free(): void;
  readonly width: number;
  readonly height: number;
  readonly projection: string;
  readonly maxLevel: number;
  readonly numRegions: number;
  /** Raster bounding box `[minLon, minLat, maxLon, maxLat]` in WGS 84 (`EPSG:4326`), or `undefined` if unprojected. */
  boundsWgs84(): Float64Array | undefined;
  depressionsJson(): string;
  /** Alias of {@link depressionsJson} for mount delineation outputs. */
  mountsJson(): string;
  /** Per-cell depression/mount ID grid (`0` outside features). */
  depressionIdValues(): Int32Array;
  /** Alias of {@link depressionIdValues} for mount delineation outputs. */
  mountIdValues(): Int32Array;
  idTif(): Uint8Array;
  levelTif(): Uint8Array;
  geojson(): string;
  levelRgba(): Uint8Array;
}

/** Output container returned by {@link delineateCatchments}. Call `.free()` when done. */
export class CatchmentsOutput {
  free(): void;
  recordsJson(): string;
  watershedTif(): Uint8Array;
  geojson(): string;
}

/** Output container returned by {@link simulateInundation} and {@link simulateInundationAdvanced}. Call `.free()` when done. */
export class InundationOutput {
  free(): void;
  readonly width: number;
  readonly height: number;
  readonly projection: string;
  readonly maxLevel: number;
  readonly numRegions: number;
  readonly numSteps: number;
  readonly runoffCoeff: number;
  /** DEM bounding box `[minLon, minLat, maxLon, maxLat]` in WGS 84 (`EPSG:4326`), or `undefined` if unprojected. */
  boundsWgs84(): Float64Array | undefined;
  depressionsJson(): string;
  catchmentHierarchyJson(): string;
  stepsJson(): string;
  spillPathsJson(): string;
  spillPathsGeojson(): string;
  fillTimeTif(): Uint8Array;
  catchmentHirTif(): Uint8Array;
  sinkTif(): Uint8Array;
  levelTif(): Uint8Array;
  depressionsGeojson(): string;
  hillshadeRgba(): Uint8Array;
  depthRgba(): Uint8Array;
  levelRgba(): Uint8Array;
  fillTimeRgba(): Uint8Array;
  /** Per-cell fill times in hours (`-1.0` for dry / NoData). */
  fillTimeValues(): Float64Array;
  /** Per-cell depression ID grid (`0` outside depressions). */
  depressionIdValues(): Int32Array;
  ratingCurvesJson(): string;
  ratingCurvesCsv(): string;
  connectivityGraphJson(): string;
  connectivityGraphml(): string;
  inundationCurveCsv(): string;
  fsmMassBalanceJson(): string;
  fsmBasinsJson(): string;
  fsmBasinsCsv(): string;
  fsmHydrographJson(): string;
  fsmHydrographCsv(): string;
  renderFramePng(
    step_index: number,
    show_scale_bar: boolean,
    show_legend: boolean,
    show_spill_paths?: boolean | null,
  ): Uint8Array;
  encodeGif(
    frame_delay_ms: number,
    ping_pong: boolean,
    max_dim: number,
    show_spill_paths?: boolean | null,
  ): Uint8Array;
}

/** Initializes the `console_error_panic_hook` (automatically invoked on WASM start). */
export function init_panic_hook(): void;

/** Returns the `lidar-rs` crate version string. */
export function version(): string;

/** Returns basic raster metadata and min/max elevation as a JSON string ({@link RasterInfo}). */
export function rasterInfo(bytes: Uint8Array, filename: string): string;

/** Applies a moving-window mean filter to a GeoTIFF buffer and returns GeoTIFF bytes. */
export function meanFilter(bytes: Uint8Array, filename: string, kernel_size: number): Uint8Array;

/** Applies a moving-window median filter to a GeoTIFF buffer and returns GeoTIFF bytes. */
export function medianFilter(bytes: Uint8Array, filename: string, kernel_size: number): Uint8Array;

/** Applies a 2D Gaussian filter to a GeoTIFF buffer and returns GeoTIFF bytes. */
export function gaussianFilter(bytes: Uint8Array, filename: string, sigma: number): Uint8Array;

/** Applies an edge-preserving 2D bilateral filter to a DEM GeoTIFF buffer. */
export function bilateralFilter(
  bytes: Uint8Array,
  filename: string,
  spatial_sigma: number,
  range_sigma: number,
): Uint8Array;

/** Applies Lindsay et al. (2019) feature-preserving DEM smoothing to a GeoTIFF buffer. */
export function featurePreservingSmoothing(
  bytes: Uint8Array,
  filename: string,
  filter_size: number,
  norm_diff: number,
  num_iter: number,
  max_z_change?: number | null,
): Uint8Array;

/** Computes Tarboton (1997) D-infinity flow direction (radians in `[0, 2*pi)`). */
export function dinfFlowDirection(bytes: Uint8Array, filename: string): Uint8Array;

/** Computes Tarboton (1997) D-infinity flow accumulation (contributing area in cells). */
export function dinfFlowAccumulation(bytes: Uint8Array, filename: string): Uint8Array;

/** Computes Multiple Flow Direction (MFD) flow accumulation. */
export function mfdFlowAccumulation(
  bytes: Uint8Array,
  filename: string,
  exponent: number,
): Uint8Array;

/** Computes the Topographic Wetness Index (`TWI = ln(a / tan(beta))`) from a DEM GeoTIFF buffer. */
export function computeTwi(bytes: Uint8Array, filename: string): Uint8Array;

/** Computes Height Above Nearest Drainage (HAND) using a D-infinity flow-accumulation threshold. */
export function computeHand(
  bytes: Uint8Array,
  filename: string,
  acc_threshold_cells: number,
): Uint8Array;

/** Applies Nash linear-reservoir travel-time attenuation to a JSON-serialized FSM hydrograph. */
export function attenuateFsmHydrograph(
  hydrograph_json: string,
  mean_travel_time_h: number,
  dt_h: number,
  num_reservoirs: number,
): string;

/** Delineates exact bottom-up Union-Find merge-tree depressions and continuous rating curves. */
export function delineateDepressionsMergeTree(
  sink_bytes: Uint8Array,
  filename: string,
  min_size: number,
  min_depth: number,
  max_rating_curve_points?: number | null,
): MergeTreeOutput;

/** Flips a DEM upside-down (`-dem + max + delta`) and returns GeoTIFF bytes. */
export function flipDem(bytes: Uint8Array, filename: string, delta: number): Uint8Array;

/** Breaches depressions in a DEM GeoTIFF buffer using Lindsay (2016) least-cost breaching. */
export function breachDem(
  bytes: Uint8Array,
  filename: string,
  mode: BreachMode | string,
  max_depth: number | null | undefined,
  max_length: number | null | undefined,
  fill_remaining: boolean,
): Uint8Array;

/** Computes an RGBA shaded-relief preview buffer (`width * height * 4` bytes). */
export function hillshadeRgba(
  bytes: Uint8Array,
  filename: string,
  azimuth_deg: number,
  elevation_deg: number,
  z_factor: number,
): Uint8Array;

/** Converts a raster byte buffer into a Cloud Optimized GeoTIFF (COG) byte buffer. */
export function imageToCog(bytes: Uint8Array, filename: string, compression: string): Uint8Array;

/** Polygonizes a raster byte buffer into an RFC 7946 GeoJSON string. */
export function polygonizeToGeojson(
  bytes: Uint8Array,
  filename: string,
  field_name: string,
): string;

/** Extracts surface depressions (sinks) from a DEM GeoTIFF buffer in memory. */
export function extractSinks(
  bytes: Uint8Array,
  filename: string,
  min_size: number,
): SinksOutput;

/** Delineates the nested depression hierarchy (Wu et al. 2019) from a sink GeoTIFF buffer. */
export function delineateDepressions(
  sink_bytes: Uint8Array,
  filename: string,
  min_size: number,
  min_depth: number,
  interval: number,
): DepressionsOutput;

/** Delineates nested elevated features (mounts, buildings, or tree canopies) from a DSM buffer. */
export function delineateMounts(
  dsm_bytes: Uint8Array,
  filename: string,
  min_size: number,
  min_height: number,
  interval: number,
): DepressionsOutput;

/** Delineates contributing catchments from a refined DEM and a sink/region raster buffer. */
export function delineateCatchments(
  dem_bytes: Uint8Array,
  sink_or_region_bytes: Uint8Array,
  dem_filename: string,
  sink_filename: string,
): CatchmentsOutput;

/** Runs the full in-memory hydrology and inundation simulation pipeline. */
export function simulateInundation(
  dem_bytes: Uint8Array,
  filename: string,
  kernel_size: number,
  min_size: number,
  min_depth: number,
  interval: number,
  rain_intensity_cm_h: number,
  rain_duration_h: number,
  time_step_h: number,
): InundationOutput;

/** Runs the full in-memory hydrology and inundation pipeline with selective breaching, runoff coefficient, SCS Type II storm forcing, and static or FSM routing. */
export function simulateInundationAdvanced(
  dem_bytes: Uint8Array,
  filename: string,
  kernel_size: number,
  min_size: number,
  min_depth: number,
  interval: number,
  rain_intensity_cm_h: number,
  rain_duration_h: number,
  time_step_h: number,
  breach_max_depth: number | null | undefined,
  runoff_coeff: number,
  scs_type2_mm?: number | null,
  routing?: RoutingMode | string | null,
  delineation_method?: DelineationMethod | string | null,
): InundationOutput;
