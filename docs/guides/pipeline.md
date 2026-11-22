# All-in-One Pipeline & GeoPackage Export

Instead of chaining five intermediate GeoTIFF and Shapefile directories manually, `lidar pipeline` (and `run_pipeline` in Rust/Python) executes the entire depressional hydrology workflow **in memory** and writes clean, unified outputs once at the end.

## Running the Full Pipeline

=== "CLI"

    ```sh
    # Download sample DEM and stream flowlines
    curl -LO https://files.opengeos.org/lidar/data/dem.tif
    curl -LO https://files.opengeos.org/lidar/data/sample_streams.gpkg

    lidar pipeline dem.tif out_pipe \
      --kernel 3 \
      --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --rain-intensity 5 --duration 50 --time-step 1 \
      --streams sample_streams.gpkg \
      --compress
    ```

    Pass `--use-f32` to load large float32 DEMs at half RAM, `--breach-max-depth 1.5` to enable selective breaching before sink extraction, `--culverts sample_culverts.gpkg` to burn culverts, or `--scs-type2-mm 100` (omitting `--duration` or setting `--duration 24`) to drive an NRCS 24-hour Type II design storm.

=== "Python (File Pipeline)"

    ```python
    import lidar_rs as lidar

    res = lidar.run_pipeline(
        "dem.tif",
        "out_pipe",
        kernel_size=3,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=50.0,
        time_step_h=1.0,
        stream_vector="sample_streams.gpkg",
        compress=True,
    )
    print(res["num_depressions"], res["num_spill_paths"])
    print("GeoPackage:", res["depressions_gpkg"])
    print("GraphML:", res["connectivity_graphml"])
    ```

=== "Python (Zero-Temp-File NumPy)"

    ```python
    import lidar_rs as lidar

    dem_arr = lidar.MedianFilter("dem.tif", kernel_size=3)
    sim = lidar.simulate_inundation_array(
        dem_arr,
        kernel_size=1,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=50.0,
        time_step_h=1.0,
        cell_size=1.0,
    )
    fill_time = sim["fill_time"]         # 2D NumPy float64 array
    depressions = sim["depressions"]     # list of 24-field dicts
    rating_curves = sim["rating_curves"] # list of stage-area-volume dicts
    ```

=== "Rust"

    ```rust,no_run
    use std::path::PathBuf;
    use lidar::{RunPipelineOptions, SimulateInundationOptions, run_pipeline};

    fn main() -> lidar::Result<()> {
        let res = run_pipeline(
            "dem.tif",
            "out_pipe",
            &RunPipelineOptions {
                kernel_size: 3,
                simulation: SimulateInundationOptions {
                    min_size: 1000,
                    min_depth: 0.3,
                    interval: 0.3,
                    rain_intensity_cm_h: 5.0,
                    rain_duration_h: 50.0,
                    time_step_h: 1.0,
                    compress: true,
                },
                stream_vector: Some(PathBuf::from("sample_streams.gpkg")),
                ..Default::default()
            },
        )?;
        println!("Total storage: {:.1} m3", res.total_storage_m3);
        Ok(())
    }
    ```

## Pipeline Output Layout

A single `lidar pipeline` invocation produces an analysis-ready directory:

```text
out_pipe/
├── depressions.csv              # 24-column nested depression morphometry table
├── catchments.csv               # Per-level contributing catchment & zonal elevation stats
├── inundation_curve.csv         # Time-step inundation area, fraction, and cumulative runoff (mm)
├── summary.json                 # Machine-readable summary (counts, max level, storage, paths)
├── depressions.gpkg             # Unified GeoPackage polygon layer with full untruncated attributes
├── spill_paths.gpkg             # Georeferenced overland spill-path polylines with spill timing
├── rating_curves.csv            # Stage-storage-area lookup curves for every depression
├── connectivity_graph.json      # Directed fill-and-spill + hierarchical merge graph (JSON)
├── connectivity_graph.graphml   # Directed graph for NetworkX, Gephi, and igraph
├── depth.tif                    # Depression depth raster (DEFLATE-compressed when --compress)
├── depression_id.tif            # Unique depression ID raster
├── depression_level.tif         # Hierarchy level raster (1 = innermost leaf)
├── catchment_hir.tif            # Contributing catchment area raster
└── fill_time.tif                # Time to inundation in hours (-1.0 = dry/NoData)
```

Use `--no-rasters`, `--no-vectors`, or `--no-graph` on the CLI (or set `save_rasters=False`, `export_vectors=False`, `export_graph=False` in Rust/Python, which all default to `true`) to skip output groups you do not need.

## Unified GeoPackage Export (`--gpkg`)

Legacy ESRI Shapefiles truncate attribute column names to 10 characters (e.g., `hypsometric_integral` becomes `hypsometr`). Both `lidar pipeline` and `lidar delineate-depressions --gpkg` (or `export_gpkg=True` in Python) write a modern SQLite-backed **GeoPackage (`.gpkg`)** via pure-Rust `terrio`, preserving all 24 full attribute names and the source CRS in a single portable file.

```sh
lidar delineate-depressions out/sink.tif out \
  --min-size 1000 --min-depth 0.3 --interval 0.3 --gpkg
```

## Interactive Browser Exports

The exact same outputs (GeoJSON, GeoTIFFs, 24-column CSV, rating curves, GraphML/JSON connectivity graphs, and PNG/GIF snapshots) are also generated client-side in the [WebAssembly Playground](../playground.md):

![Spatial layers, 12 client-side download buttons, and expanded morphometry table in the lidar-rs WebAssembly Playground](https://files.opengeos.org/lidar/images/playground_downloads_table.png)

!!! note "Path Safety"
    `run_pipeline` and `delineate_depressions` validate before writing that no enabled output path resolves (including through symlinks) to an input file or to another enabled output path.
