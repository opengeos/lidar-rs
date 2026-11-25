//! Release-mode benchmark for `lidar-rs` core operations on the 400x400 sample datasets
//! and optional synthetic DEMs.
//!
//! Run with:
//! `cargo run --release --example benchmark`
//! `cargo run --release --example benchmark -- --size 512 --iters 5`

use std::path::PathBuf;
use std::time::Instant;

use lidar::{
    Array2D, BreachMode, BreachOptions, DelineateOptions, ExtractSinksOptions,
    InMemoryFullPipelineOptions, InundationForcing, MergeTreeOptions, RoutingMode,
    RunPipelineOptions, SimulateInundationOptions, TiledDelineationOptions, Topology,
    breach_depressions, delineate_catchment_hierarchy_in_memory, delineate_depressions,
    delineate_depressions_in_memory, delineate_depressions_merge_tree_in_memory,
    delineate_depressions_tiled, delineate_mounts, extract_sinks, extract_sinks_in_memory,
    gaussian_filter, load_raster, mean_filter, median_filter, run_full_pipeline_in_memory,
    run_pipeline, save_raster, simulate_inundation_with_forcing_in_memory,
};

struct BenchRow {
    name: String,
    runs: usize,
    median_ms: f64,
    min_ms: f64,
    mcells_per_s: f64,
}

fn bench<F, R>(name: &str, runs: usize, cells: usize, mut f: F) -> BenchRow
where
    F: FnMut() -> R,
{
    let runs = runs.max(1);
    // Warmup
    let _ = f();
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let median_ms = times[runs / 2];
    let min_ms = times[0];
    let mcells_per_s = if median_ms > 0.0 {
        (cells as f64 / 1e6) / (median_ms / 1000.0)
    } else {
        0.0
    };
    BenchRow {
        name: name.to_string(),
        runs,
        median_ms,
        min_ms,
        mcells_per_s,
    }
}

fn print_table(title: &str, rows: &[BenchRow]) {
    println!("{title}");
    println!(
        "{:<48} {:>5} {:>11} {:>11} {:>12}",
        "Operation", "Runs", "Median (ms)", "Min (ms)", "Mcells/s"
    );
    println!("{}", "-".repeat(91));
    for row in rows {
        println!(
            "{:<48} {:>5} {:>11.2} {:>11.2} {:>12.2}",
            row.name, row.runs, row.median_ms, row.min_ms, row.mcells_per_s
        );
    }
    println!();
}

fn make_synthetic_dem(size: usize) -> lidar::Result<Array2D<f64>> {
    let mut data = Vec::with_capacity(size * size);
    let inv = 1.0 / size.max(1) as f64;
    for r in 0..size {
        let y = r as f64 * inv;
        for c in 0..size {
            let x = c as f64 * inv;
            // Gentle regional slope with multi-scale nested bowls and ridges.
            let slope = 100.0 + 4.0 * x + 2.5 * y;
            let bowl1 = -3.0 * ((x - 0.30).hypot(y - 0.35) * 6.0).cos().max(0.0);
            let bowl2 = -2.4 * ((x - 0.70).hypot(y - 0.65) * 7.0).cos().max(0.0);
            let ripple = 0.6 * (x * 24.0).sin() * (y * 24.0).cos();
            data.push(slope + bowl1 + bowl2 + ripple);
        }
    }
    let mut dem = Array2D::from_vec(size, size, data)?;
    dem.set_no_data(-9999.0);
    dem.geotransform = [500_000.0, 1.0, 0.0, 4_000_000.0 + size as f64, 0.0, -1.0];
    Ok(dem)
}

fn parse_args() -> lidar::Result<(Option<usize>, usize)> {
    let mut size: Option<usize> = None;
    let mut iters: usize = 3;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                let val = args.get(i + 1).ok_or_else(|| {
                    lidar::Error::InvalidArgument("--size requires an integer >= 32".into())
                })?;
                let n = val.parse::<usize>().map_err(|_| {
                    lidar::Error::InvalidArgument(format!(
                        "invalid --size value '{val}': expected an integer >= 32"
                    ))
                })?;
                if n < 32 {
                    return Err(lidar::Error::InvalidArgument(format!(
                        "--size must be >= 32, got {n}"
                    )));
                }
                size = Some(n);
                i += 2;
            }
            "--iters" => {
                let val = args.get(i + 1).ok_or_else(|| {
                    lidar::Error::InvalidArgument("--iters requires an integer >= 1".into())
                })?;
                let n = val.parse::<usize>().map_err(|_| {
                    lidar::Error::InvalidArgument(format!(
                        "invalid --iters value '{val}': expected an integer >= 1"
                    ))
                })?;
                if n < 1 {
                    return Err(lidar::Error::InvalidArgument(format!(
                        "--iters must be >= 1, got {n}"
                    )));
                }
                iters = n;
                i += 2;
            }
            other => {
                return Err(lidar::Error::InvalidArgument(format!(
                    "unknown benchmark argument '{other}'; supported flags: --size <N>, --iters <N>"
                )));
            }
        }
    }
    Ok((size, iters))
}

fn main() -> lidar::Result<()> {
    let (synth_size, iters) = parse_args()?;
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("test_data");
    let dem_path = data_dir.join("dem.tif");
    let sink_path = data_dir.join("sink.tif");
    let dsm_path = data_dir.join("dsm.tif");

    let dem = load_raster::<f64>(&dem_path, None)?;
    let sink = load_raster::<f64>(&sink_path, None)?;
    let cells = dem.width() * dem.height();
    let tmp = tempfile::tempdir()?;

    // Precompute intermediate rasters for isolated hydrology stage benchmarks.
    let sinks_mem = extract_sinks_in_memory(&dem, 1000, Topology::D8)?;
    let deps_mem = delineate_depressions_in_memory(&sinks_mem.sink, 1000, 0.3, 0.3)?;
    let (cat_hir, _) = delineate_catchment_hierarchy_in_memory(
        &sinks_mem.dem_refined,
        &deps_mem.depression_level,
        1000,
    )?;

    let mut runoff_coeff = Array2D::from_vec(dem.width(), dem.height(), vec![0.75; cells])?;
    runoff_coeff.copy_metadata_from(&sinks_mem.sink);
    let mut infiltration = Array2D::from_vec(dem.width(), dem.height(), vec![0.2; cells])?;
    infiltration.copy_metadata_from(&sinks_mem.sink);

    let sim_opts = SimulateInundationOptions {
        min_size: 1000,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 24.0,
        time_step_h: 2.0,
        compress: false,
    };

    let mut rows = Vec::new();

    rows.push(bench("mean_filter (3x3, in-memory)", iters, cells, || {
        mean_filter(&dem, 3).unwrap()
    }));
    rows.push(bench(
        "median_filter (3x3, in-memory)",
        iters,
        cells,
        || median_filter(&dem, 3).unwrap(),
    ));
    rows.push(bench(
        "gaussian_filter (sigma=1.0, in-memory)",
        iters,
        cells,
        || gaussian_filter(&dem, 1.0).unwrap(),
    ));
    rows.push(bench(
        "breach_depressions (selective, in-memory)",
        iters,
        cells,
        || {
            breach_depressions(
                &dem,
                &BreachOptions::for_mode(BreachMode::Selective, Some(1.0), Some(30)),
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "extract_sinks (in-memory, min_size=1000)",
        iters,
        cells,
        || extract_sinks_in_memory(&dem, 1000, Topology::D8).unwrap(),
    ));
    rows.push(bench(
        "delineate_depressions (level-set, in-memory)",
        iters,
        cells,
        || delineate_depressions_in_memory(&sink, 1000, 0.3, 0.3).unwrap(),
    ));
    rows.push(bench(
        "delineate_depressions_merge_tree (in-memory)",
        iters,
        cells,
        || {
            delineate_depressions_merge_tree_in_memory(
                &sink,
                &MergeTreeOptions {
                    min_size: 1000,
                    min_depth: 0.3,
                    topology: Topology::D8,
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "delineate_catchment_hierarchy (in-memory)",
        iters,
        cells,
        || {
            delineate_catchment_hierarchy_in_memory(
                &sinks_mem.dem_refined,
                &deps_mem.depression_level,
                1000,
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "simulate_inundation_with_forcing (in-memory)",
        iters,
        cells,
        || {
            simulate_inundation_with_forcing_in_memory(
                &sinks_mem.sink,
                &cat_hir,
                &sim_opts,
                &InundationForcing {
                    runoff_coeff: Some(&runoff_coeff),
                    infiltration_cm_h: Some(&infiltration),
                    ..Default::default()
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "run_full_pipeline_in_memory (static)",
        iters,
        cells,
        || {
            run_full_pipeline_in_memory(
                &dem,
                &InMemoryFullPipelineOptions {
                    kernel_size: 3,
                    simulation: sim_opts.clone(),
                    routing: RoutingMode::Static,
                    ..Default::default()
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "run_full_pipeline_in_memory (FSM)",
        iters,
        cells,
        || {
            run_full_pipeline_in_memory(
                &dem,
                &InMemoryFullPipelineOptions {
                    kernel_size: 3,
                    simulation: sim_opts.clone(),
                    routing: RoutingMode::FillSpillMerge,
                    ..Default::default()
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "run_pipeline (full file I/O, CSV+GPKG+Graph)",
        iters,
        cells,
        || {
            run_pipeline(
                &dem_path,
                tmp.path().join("pipe"),
                &RunPipelineOptions {
                    kernel_size: 3,
                    simulation: sim_opts.clone(),
                    save_rasters: true,
                    export_vectors: true,
                    export_graph: true,
                    ..Default::default()
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "delineate_depressions_tiled (200x200 tiles)",
        iters,
        cells,
        || {
            delineate_depressions_tiled(
                &dem_path,
                tmp.path().join("tiled"),
                &TiledDelineationOptions {
                    tile_size: 200,
                    overlap: 32,
                    kernel_size: 3,
                    min_size: 1000,
                    min_depth: 0.3,
                    interval: 0.3,
                    topology: Topology::D8,
                    use_f32: false,
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "extract_sinks (full file I/O + GPKG)",
        iters,
        cells,
        || {
            extract_sinks(
                &dem_path,
                tmp.path().join("sinks"),
                &ExtractSinksOptions {
                    min_size: 1000,
                    ..Default::default()
                },
            )
            .unwrap()
        },
    ));
    rows.push(bench(
        "delineate_depressions (full file I/O)",
        iters,
        cells,
        || {
            delineate_depressions(
                &sink_path,
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
        },
    ));
    rows.push(bench(
        "delineate_mounts (dsm.tif, full I/O)",
        iters,
        cells,
        || {
            delineate_mounts(
                &dsm_path,
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
        },
    ));

    print_table(
        &format!(
            "lidar-rs {} benchmark on {}x{} DEM ({cells} cells)",
            lidar::VERSION,
            dem.width(),
            dem.height()
        ),
        &rows,
    );

    if let Some(n) = synth_size {
        let synth = make_synthetic_dem(n)?;
        let synth_cells = n * n;
        let synth_tif = tmp.path().join("synthetic.tif");
        save_raster(&synth, &synth_tif, None, false)?;
        let synth_sinks = extract_sinks_in_memory(&synth, 200, Topology::D8)?;
        let synth_sim = SimulateInundationOptions {
            min_size: 200,
            min_depth: 0.2,
            interval: 0.3,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 12.0,
            time_step_h: 2.0,
            compress: false,
        };
        let mut srows = Vec::new();
        srows.push(bench(
            "breach_depressions (selective)",
            iters,
            synth_cells,
            || {
                breach_depressions(
                    &synth,
                    &BreachOptions::for_mode(BreachMode::Selective, Some(1.0), Some(30)),
                )
                .unwrap()
            },
        ));
        srows.push(bench(
            "delineate_depressions (level-set)",
            iters,
            synth_cells,
            || delineate_depressions_in_memory(&synth_sinks.sink, 200, 0.2, 0.3).unwrap(),
        ));
        srows.push(bench(
            "delineate_depressions_merge_tree",
            iters,
            synth_cells,
            || {
                delineate_depressions_merge_tree_in_memory(
                    &synth_sinks.sink,
                    &MergeTreeOptions {
                        min_size: 200,
                        min_depth: 0.2,
                        topology: Topology::D8,
                    },
                )
                .unwrap()
            },
        ));
        srows.push(bench(
            "run_full_pipeline_in_memory (static)",
            iters,
            synth_cells,
            || {
                run_full_pipeline_in_memory(
                    &synth,
                    &InMemoryFullPipelineOptions {
                        kernel_size: 3,
                        simulation: synth_sim.clone(),
                        routing: RoutingMode::Static,
                        ..Default::default()
                    },
                )
                .unwrap()
            },
        ));
        srows.push(bench(
            "delineate_depressions_tiled",
            iters,
            synth_cells,
            || {
                delineate_depressions_tiled(
                    &synth_tif,
                    tmp.path().join("synth_tiled"),
                    &TiledDelineationOptions {
                        tile_size: (n / 2).max(64),
                        overlap: 24,
                        kernel_size: 3,
                        min_size: 200,
                        min_depth: 0.2,
                        interval: 0.3,
                        topology: Topology::D8,
                        use_f32: false,
                    },
                )
                .unwrap()
            },
        ));
        print_table(
            &format!("Synthetic {n}x{n} DEM benchmark ({synth_cells} cells)"),
            &srows,
        );
    }

    Ok(())
}
