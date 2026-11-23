# Rating Curves & Connectivity Graphs

Hydrologic models (SWAT+, HEC-HMS, GSSHA, MODFLOW, and custom Python/NetworkX routing scripts) require structured lookup tables and directed topological networks rather than raw rasters alone.

`lidar-rs` exports three model-ready connectivity artifacts directly from the nested depression hierarchy and fill-and-spill simulation:

1. **Stage-Storage-Area Rating Curves (`rating_curves.csv`):** Exact prefix-sum area and cumulative volume curves for every depression across `[min_elev, max_elev]`.
2. **Directed Connectivity Graph (`connectivity_graph.json` and `connectivity_graph.graphml`):** Full directed graph combining overland **spill** edges and hierarchical child-to-parent **merge** edges.
3. **Georeferenced Spill-Path Vector (`spill_paths.gpkg` / `.geojson` / `.shp`):** Polyline geometry for every overland pour-point cascade with spill timing, runoff threshold, elevation, and length.

![Stage-storage rating curves for the top five depressions and directed fill-and-spill plus hierarchical merge connectivity graph](https://files.opengeos.org/lidar/images/rating_curves_and_graph.png)

## Stage-Storage-Area Rating Curves

`compute_rating_curves(sink, deps, interval)` collects all owned and descendant cell elevations for each depression (so compound level >= 2 depressions include their nested sub-basins), sorts elevations once, and evaluates exact surface area and stored volume in `O(log M)` per stage via prefix sums.

- Curves always include both the basin floor (`min_elev`, `depth = 0`) and spill rim (`max_elev`).
- When a depression's vertical span requires more than 500 steps at `interval`, sampling caps at 500 evenly spaced intervals (`span / 500`) covering the full `[min_elev, max_elev]` range without gaps.

| `rating_curves.csv` Column | Units | Description |
| --- | --- | --- |
| `dep_id` | 1-based ID | Depression identifier matching `depressions.csv` |
| `level` | integer | Hierarchy level (`1` = leaf) |
| `region_id` | 1-based ID | Enclosing top-level sink region |
| `stage` | DEM vertical units | Water surface elevation |
| `depth` | DEM vertical units | Depth above depression floor (`stage - min_elev`) |
| `area` | map units squared | Inundated surface area at `stage` |
| `volume` | horizontal map units² × DEM vertical units | Cumulative stored volume below `stage` (cubic map units when horizontal and vertical units match) |

## Directed Connectivity Graph Schema

`build_connectivity_graph` produces a `ConnectivityGraph` exported to both pretty-printed JSON and standard GraphML XML (directly readable by NetworkX, Gephi, and igraph):

- **Full JSON schema (`connectivity_graph.json`):**
  - **Nodes (`GraphNode`):** `id`, `parent_id`, `level`, `region_id`, `area`, `volume`, `avg_depth`, `max_depth`, `min_elev`, `max_elev`, `circularity`, `hypsometric_integral`, `cat_area`, `fill_time_h`, and `fill_runoff_mm`.
  - **Edges (`GraphEdge`):**
    - `edge_type = "spill"`: Overland pour-point cascade from a filled basin to a downstream basin (or boundary exit `0`), with `spill_time_h`, `spill_runoff_mm`, `merge_time_h`, `spill_elev`, `length_m`, `start_xy`, `end_xy`, and optional `dist_to_stream`.
    - `edge_type = "merge"`: Hierarchical child-to-parent coalescence link (`from_id = child`, `to_id = parent`). In static routing `merge_time_h` is the parent's fill time; with `--routing fsm` it is the simulated child-to-parent merge time, including merges in the constant-rainfall extension past the storm, so it matches the `merge_time_h` on that child's `spill` edge.
- **GraphML attribute subset (`connectivity_graph.graphml`):**
  - **Node attributes:** `level`, `region_id`, `parent_id`, `area`, `volume`, `max_depth`, `cat_area`, and `fill_runoff_mm` (plus a virtual sink node `n0` when any spill edge exits to the watershed boundary `to_id == 0`).
  - **Edge attributes:** `edge_type`, `spill_runoff_mm`, `length_m`, and `dist_to_stream` (when finite). Use the JSON graph when you need `spill_time_h`, `merge_time_h`, `spill_elev`, `start_xy`, or `end_xy`.

## Stream Proximity Annotation (`--streams`)

When a stream flowline vector (such as USGS NHD flowlines, or the sample [sample_streams.gpkg](https://files.opengeos.org/lidar/data/sample_streams.gpkg)) is supplied:

1. `lidar-rs` validates that the stream layer CRS matches the DEM CRS via `SpatialRef::same_as`, rejecting mismatched projections cleanly.
2. All stream line segments are indexed in a pure-Rust uniform-grid spatial index (`StreamSegmentIndex`) with exact expanding-ring Euclidean lower bounds for fast nearest-segment lookup even on dense regional stream networks.
3. Each `"spill"` edge records `dist_to_stream` from its downstream spill endpoint (`end_xy`) to the nearest stream segment.

## Analyzing Outputs in Python (NetworkX & Pandas)

Requires optional analysis packages (`pip install pandas networkx`).

```python
import json
import pandas as pd
import networkx as nx
import lidar_rs as lidar

# Run the pipeline with stream proximity enabled
res = lidar.run_pipeline(
    "dem.tif",
    "out_pipe",
    min_size=1000,
    min_depth=0.3,
    interval=0.3,
    rain_intensity_cm_h=5.0,
    rain_duration_h=50.0,
    time_step_h=1.0,
    stream_vector="sample_streams.gpkg",
)

# Load GraphML directly into NetworkX
G = nx.read_graphml(res["connectivity_graphml"])
spill_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["edge_type"] == "spill"]
print(f"Loaded {G.number_of_nodes()} nodes and {len(spill_edges)} overland spill edges")

# Inspect JSON graph stream distances
graph_data = json.loads(open(res["connectivity_graph_json"]).read())
for e in graph_data["edges"]:
    if e["edge_type"] == "spill":
        dist = e.get("dist_to_stream")
        dist_str = f"{dist:.1f} m" if dist is not None else "n/a"
        print(
            f"Dep #{e['from_id']} -> #{e['to_id']} | "
            f"spill={e['spill_runoff_mm']:.1f} mm | "
            f"len={e['length_m']:.1f} m | "
            f"dist_to_stream={dist_str}"
        )

# Load stage-storage rating curves
rc = pd.read_csv(res["rating_curves_csv"])
print(rc.groupby("dep_id")[["depth", "area", "volume"]].max().head())
```

## Inspecting Interactively in the Browser

In the [WebAssembly Playground](../playground.md), clicking any depression on the inundation map or in the feature table highlights its contour, displays its parent/child hierarchy and morphometry, and plots its exact **Stage-Volume Rating Curve** immediately. You can also download `rating_curves.csv`, `connectivity_graph.json`, `connectivity_graph.graphml`, and `spill_paths.geojson` directly from the browser.
