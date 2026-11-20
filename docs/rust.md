# Rust Library

Add `lidar` to your `Cargo.toml`:

```toml
[dependencies]
lidar = "0.3"
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
