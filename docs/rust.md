# Rust Library

Add `lidar` to your `Cargo.toml`:

```toml
[dependencies]
lidar = "0.5"
```

## Cargo Features

| Feature | Default | Description |
| --- | --- | --- |
| `parallel` | Yes | Multi-threaded parallelism via `rayon` (filters, level-set slicing, inundation, tiling) |
| `pointcloud` | Yes | Pure-Rust LiDAR I/O (LAS/LAZ/COPC/PLY/E57), CSF/PMF classification, and DSM/DTM/CHM gridding via `terrio/pointcloud` |
| `serde` | Yes | `Serialize` / `Deserialize` derives on all option/result structs plus `write_connectivity_graph_json` |
| `download` | No | Opt-in U.S. federal geospatial data downloaders (`federal`: USGS 3DEP, WBD, NHD, Transportation, USFWS NWI, NOAA Atlas 14; non-WASM, enables `ureq` with `rustls`) |

For minimal WASM or embedded builds without threads or point-cloud support:

```toml
[dependencies]
lidar = { version = "0.5", default-features = false, features = ["serde"] }
```

## Module Map

| Module | Key Public Types & Functions |
| --- | --- |
| `filtering` | `mean_filter`, `median_filter`, `gaussian_filter` (+ `*_file` wrappers) |
| `conditioning` | `BreachMode`, `BreachOptions` (`for_mode`), `breach_depressions`, `burn_culverts`, `BurnCulvertsOptions`, `synthesize_culverts`, `SynthesizeCulvertsOptions` |
| `federal` | `Region`, `Bbox`, `WbdLevel`, `DemProduct`, `FederalDataset`, `DownloadOptions`, `FederalBundleOptions`, `download_federal_data`, `query_3dep_products`, `download_3dep_dem`, `download_wbd`, `download_nhd`, `download_nwi`, `download_roads`, `fetch_noaa_atlas14` |
| `filling` | `extract_sinks`, `extract_sinks_in_memory`, `ExtractSinksOptions`, `InMemorySinks` |
| `slicing` | `delineate_depressions`, `delineate_depressions_in_memory`, `NestedDepression` (24 fields), `write_depressions_csv`, `write_depressions_vector` |
| `mounts` | `flip_dem`, `delineate_mounts`, `delineate_mounts_in_memory` |
| `catchment` | `delineate_catchments`, `delineate_flow_paths`, `delineate_catchment_hierarchy`, `delineate_catchment_hierarchy_in_memory` (compressed D8 + DSU) |
| `inundation` | `run_pipeline`, `run_full_pipeline_in_memory`, `simulate_inundation`, `simulate_inundation_with_forcing_in_memory`, `InundationForcing`, `scs_curve_number_runoff`, `scs_type2_hyetograph`, `cumulative_runoff_mm` |
| `fsm` | `RoutingMode`, `simulate_fill_spill_merge_in_memory`, `FsmSimulationResult`, `FsmBasinSummary`, `FsmMassBalance`, `fsm_basins_to_csv_string`, `write_fsm_basins_csv` |
| `graph` | `compute_rating_curves`, `write_rating_curves_csv`, `build_connectivity_graph`, `write_connectivity_graph_json`, `write_connectivity_graphml`, `write_spill_paths_vector` |
| `tiling` | `delineate_depressions_tiled`, `TiledDelineationOptions`, `load_dem_f32` |
| `animation` | `animate_inundation`, `animate_inundation_to_bytes_with_spill_paths`, `export_inundation_frames_png`, `render_inundation_frame_png_bytes`, `AnimateInundationOptions` |
| `pointcloud` | `read_lidar`, `write_lidar`, `convert_lidar`, `classify_ground`, `GroundFilterMethod`, `parse_return_kind`, `lidar_to_dsm`, `lidar_to_dtm`, `lidar_to_chm` |
| `raster_ops` | `polygonize`, `mosaic`, `reproject_image`, `resample`, `clip_image`, `image_to_cog`, `add_crs`, `join_csv_to_vector`, `csv_points_to_vector` |

## Example 1: All-in-One File Pipeline & Hydro-Conditioning

```rust,no_run
use std::path::PathBuf;
use lidar::{
    BreachMode, BreachOptions, RunPipelineOptions, SimulateInundationOptions,
    run_pipeline, scs_type2_hyetograph,
};

fn main() -> lidar::Result<()> {
    let hyeto = scs_type2_hyetograph(100.0, 24.0, 0.5)?;
    let res = run_pipeline(
        "dem.tif",
        "output_pipeline",
        &RunPipelineOptions {
            kernel_size: 3,
            culverts_vector: Some(PathBuf::from("sample_culverts.gpkg")),
            breach: Some(BreachOptions::for_mode(BreachMode::Selective, Some(1.5), Some(40))),
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 0.0, // overridden by hyetograph
                rain_duration_h: 24.0,
                time_step_h: 0.5,
                compress: true,
            },
            hyetograph: Some(hyeto),
            stream_vector: Some(PathBuf::from("sample_streams.gpkg")),
            ..Default::default()
        },
    )?;
    println!(
        "Delineated {} depressions ({} levels) and {} spill paths; total storage {:.1} m3",
        res.num_depressions, res.max_level, res.num_spill_paths, res.total_storage_m3
    );
    Ok(())
}
```

## Example 2: Pure In-Memory Orchestration (`Array2D<f64>`)

`run_full_pipeline_in_memory` runs the complete workflow on a borrowed `&Array2D<f64>` with zero filesystem I/O (borrowing the input DEM via `Cow::Borrowed` when smoothing and breaching are disabled):

```rust,no_run
use lidar::{
    InMemoryFullPipelineOptions, InundationForcing, SimulateInundationOptions,
    load_raster, run_full_pipeline_in_memory, scs_curve_number_runoff,
};

fn main() -> lidar::Result<()> {
    let dem = load_raster::<f64>("dem.tif", None)?;
    let cn = load_raster::<f64>("sample_cn.tif", None)?;
    let rc = scs_curve_number_runoff(&cn, 150.0)?;

    let pipe = run_full_pipeline_in_memory(
        &dem,
        &InMemoryFullPipelineOptions {
            kernel_size: 3,
            breach: None,
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 50.0,
                time_step_h: 1.0,
                compress: false,
            },
            routing: lidar::RoutingMode::FillSpillMerge,
            forcing: InundationForcing {
                runoff_coeff: Some(&rc),
                ..Default::default()
            },
            scalar_runoff_coeff: 1.0,
            stream_vector: None,
        },
    )?;

    println!(
        "Graph: {} nodes, {} edges; {} rating curve points",
        pipe.connectivity_graph.nodes.len(),
        pipe.connectivity_graph.edges.len(),
        pipe.rating_curves.len()
    );
    Ok(())
}
```

## Example 3: Tiled Regional Delineation & LiDAR Point Clouds

```rust,no_run
use lidar::{TiledDelineationOptions, delineate_depressions_tiled};
use lidar::pointcloud::{GroundFilterMethod, LidarToDtmOptions, lidar_to_chm, lidar_to_dtm};

fn main() -> lidar::Result<()> {
    // 1. Parallel overlapping tiled delineation with f32 DEM loading
    let tiled = delineate_depressions_tiled(
        "large_dem.tif",
        "output_tiled",
        &TiledDelineationOptions {
            tile_size: 2048,
            overlap: 256,
            use_f32: true,
            ..Default::default()
        },
    )?;
    println!("Processed {} tiles -> {} depressions", tiled.num_tiles, tiled.depressions.len());

    // 2. Pure-Rust CSF bare-earth DTM and Canopy Height Model from LAZ/COPC
    let dtm_opts = LidarToDtmOptions {
        resolution: 1.0,
        ground_method: GroundFilterMethod::Csf,
        ..Default::default()
    };
    lidar_to_dtm("sample_forest.laz", "output/dtm.tif", &dtm_opts)?;
    lidar_to_chm("sample_forest.laz", "output/chm.tif", &dtm_opts)?;
    Ok(())
}
```
