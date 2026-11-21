# Rust Library

Add `lidar` to your `Cargo.toml`:

```toml
[dependencies]
lidar = "0.4"
```

## Example Pipeline

```rust,no_run
use std::path::Path;
use lidar::{
    DelineateOptions, ExtractSinksOptions,
    delineate_depressions, extract_sinks, median_filter_file,
};

fn main() -> lidar::Result<()> {
    let out_dir = Path::new("output");
    let med_path = out_dir.join("median.tif");

    median_filter_file("dem.tif", 3, Some(&med_path))?;

    let sinks = extract_sinks(
        &med_path,
        out_dir,
        &ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )?;

    let deps = delineate_depressions(
        sinks.sink_path.as_ref().unwrap(),
        out_dir,
        &DelineateOptions {
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            bool_level_shp: true,
            ..Default::default()
        },
    )?;

    println!(
        "Delineated {} nested depressions across {} regions (max level {})",
        deps.depressions.len(),
        deps.num_regions,
        deps.max_level
    );
    Ok(())
}
```

## All-in-One Pipeline, Hydro-Conditioning, Rating Curves, and Tiling

```rust,no_run
use lidar::{
    BreachMode, BreachOptions, RunPipelineOptions, SimulateInundationOptions,
    TiledDelineationOptions, delineate_depressions_tiled, run_pipeline,
};

fn main() -> lidar::Result<()> {
    // End-to-end in-memory pipeline with selective breaching, GeoPackage, rating curves, and GraphML
    let res = run_pipeline(
        "dem.tif",
        "output_pipeline",
        &RunPipelineOptions {
            kernel_size: 3,
            breach: Some(BreachOptions {
                mode: BreachMode::Selective,
                max_depth: Some(1.0),
                max_length: Some(30),
                ..Default::default()
            }),
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 50.0,
                time_step_h: 1.0,
                compress: true,
            },
            ..Default::default()
        },
    )?;
    println!("Wrote {} depressions and {} spill paths", res.num_depressions, res.num_spill_paths);

    // Parallel overlapping tiled delineation for large regional DEMs
    let tiled = delineate_depressions_tiled("large_dem.tif", "output_tiled", &TiledDelineationOptions::default())?;
    println!("Processed {} tiles -> {} depressions", tiled.num_tiles, tiled.depressions.len());
    Ok(())
}
```
