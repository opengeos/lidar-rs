//! Release-mode benchmark for `lidar-rs` core operations on the 400×400 sample datasets.
//!
//! Run with: `cargo run --release --example benchmark`

use std::path::PathBuf;
use std::time::Instant;

use lidar::{
    DelineateOptions, ExtractSinksOptions, Topology, delineate_depressions,
    delineate_depressions_in_memory, delineate_mounts, extract_sinks, extract_sinks_in_memory,
    gaussian_filter, load_raster, mean_filter, median_filter,
};

fn bench<F, R>(name: &str, runs: usize, mut f: F)
where
    F: FnMut() -> R,
{
    // Warmup
    let _ = f();
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let median = times[runs / 2];
    println!("{name:<42} {median:>8.2} ms (median of {runs})");
}

fn main() -> lidar::Result<()> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("test_data");
    let dem = load_raster::<f64>(data_dir.join("dem.tif"), None)?;
    let sink = load_raster::<f64>(data_dir.join("sink.tif"), None)?;
    let tmp = tempfile::tempdir()?;

    println!(
        "lidar-rs {} benchmark on {}x{} DEM ({} cells)\n",
        lidar::VERSION,
        dem.width(),
        dem.height(),
        dem.width() * dem.height()
    );

    bench("mean_filter (3x3, in-memory)", 5, || {
        mean_filter(&dem, 3).unwrap()
    });
    bench("median_filter (3x3, in-memory)", 5, || {
        median_filter(&dem, 3).unwrap()
    });
    bench("gaussian_filter (sigma=1.0, in-memory)", 5, || {
        gaussian_filter(&dem, 1.0).unwrap()
    });
    bench("extract_sinks (in-memory, min_size=1000)", 5, || {
        extract_sinks_in_memory(&dem, 1000, Topology::D8).unwrap()
    });
    bench("delineate_depressions (in-memory)", 5, || {
        delineate_depressions_in_memory(&sink, 1000, 0.3, 0.3).unwrap()
    });
    bench("extract_sinks (full file I/O + GPKG)", 3, || {
        extract_sinks(
            data_dir.join("dem.tif"),
            tmp.path().join("sinks"),
            &ExtractSinksOptions {
                min_size: 1000,
                ..Default::default()
            },
        )
        .unwrap()
    });
    bench("delineate_depressions (full file I/O)", 3, || {
        delineate_depressions(
            data_dir.join("sink.tif"),
            tmp.path().join("deps"),
            &DelineateOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                bool_level_shp: true,
                ..Default::default()
            },
        )
        .unwrap()
    });
    bench("delineate_mounts (dsm.tif, full I/O)", 3, || {
        delineate_mounts(
            data_dir.join("dsm.tif"),
            tmp.path().join("mounts"),
            &DelineateOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                bool_level_shp: false,
                ..Default::default()
            },
        )
        .unwrap()
    });

    Ok(())
}
