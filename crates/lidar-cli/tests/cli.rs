//! End-to-end subprocess tests for `lidar` CLI ergonomics (verbosity, completions, help text).

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
}

#[test]
fn verbose_pipeline_with_json_keeps_stdout_pure_json_and_emits_stage_timing_to_stderr() {
    let dem = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("pipe_v");

    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args([
            "-v",
            "pipeline",
            dem.to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--interval",
            "0.5",
            "--duration",
            "10",
            "--time-step",
            "5",
            "--no-rasters",
            "--no-vectors",
            "--no-graph",
            "--json",
        ])
        .output()
        .expect("spawn lidar -v pipeline");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout_str = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let parsed: Value = serde_json::from_str(&stdout_str)
        .expect("stdout must remain valid JSON even when --verbose is active");
    assert!(parsed["num_depressions"].as_u64().unwrap() > 0);

    let stderr_str = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr_str.contains("[load]") && stderr_str.contains("[extract_sinks]"),
        "expected stage progress on stderr, got:\n{stderr_str}"
    );
    assert!(
        stderr_str.contains("completed in"),
        "expected total timing on stderr, got:\n{stderr_str}"
    );
}

#[test]
fn quiet_suppresses_non_json_stdout() {
    let dem = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();
    let out_tif = tmp.path().join("mean.tif");

    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args([
            "-q",
            "mean-filter",
            dem.to_str().unwrap(),
            out_tif.to_str().unwrap(),
        ])
        .output()
        .expect("spawn lidar -q mean-filter");
    assert!(out.status.success());
    assert!(out_tif.exists());
    assert!(
        out.stdout.is_empty(),
        "expected empty stdout with --quiet, got: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn completions_generates_scripts_for_all_supported_shells() {
    for shell in ["bash", "zsh", "fish", "powershell", "elvish"] {
        let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
            .args(["completions", shell])
            .output()
            .expect("spawn lidar completions");
        assert!(
            out.status.success(),
            "completions {shell} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let script = String::from_utf8(out.stdout).expect("utf-8 completion script");
        assert!(
            script.contains("lidar"),
            "completion script for {shell} did not reference `lidar`"
        );
    }
}

#[test]
fn tiled_depressions_help_documents_f32_fallback_vs_windowed_f64() {
    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args(["tiled-depressions", "--help"])
        .output()
        .expect("spawn lidar tiled-depressions --help");
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).expect("utf-8 help");
    assert!(
        help.contains("eager/in-memory fallback") && help.contains("windowed GeoTIFF/COG"),
        "expected updated f32/f64 help text, got:\n{help}"
    );
}

#[test]
fn pipeline_config_is_rejected_inside_mcp_subprocess() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tmp.path().join("run.toml");
    std::fs::write(&cfg, "kernel = 3\n").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .env("LIDAR_MCP_TOOL", "1")
        .args(["pipeline", "--config", cfg.to_str().unwrap()])
        .output()
        .expect("spawn lidar pipeline --config under LIDAR_MCP_TOOL");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--config is disabled inside MCP"),
        "expected MCP --config rejection on stderr, got:\n{stderr}"
    );
}

#[test]
fn cn_to_runoff_computes_exact_values_for_scalar_and_raster_with_custom_ia_ratio() {
    // CN = 80 => S = 63.5 mm; P = 100 mm, Ia/S = 0.05 => Ia = 3.175 mm
    let s_mm: f64 = 25_400.0 / 80.0 - 254.0;
    let ia_mm = 0.05 * s_mm;
    let expected_q = (100.0_f64 - ia_mm).powi(2) / (100.0 - ia_mm + s_mm);
    let expected_rc = expected_q / 100.0;

    // 1. Scalar mode with --ia-ratio 0.05
    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args([
            "cn-to-runoff",
            "--cn",
            "80",
            "--precip-mm",
            "100",
            "--ia-ratio",
            "0.05",
            "--json",
        ])
        .output()
        .expect("spawn lidar cn-to-runoff scalar");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let parsed: Value = serde_json::from_slice(&out.stdout).expect("valid JSON");
    assert!((parsed["retention_s_mm"].as_f64().unwrap() - s_mm).abs() < 1e-9);
    assert!((parsed["initial_abstraction_mm"].as_f64().unwrap() - ia_mm).abs() < 1e-9);
    assert!((parsed["runoff_mm"].as_f64().unwrap() - expected_q).abs() < 1e-9);
    assert!((parsed["runoff_coeff"].as_f64().unwrap() - expected_rc).abs() < 1e-9);

    // 2. Scalar mode must reject --output cleanly
    let tmp = tempfile::tempdir().unwrap();
    let bogus_out = tmp.path().join("scalar_should_fail.tif");
    let out_err = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args([
            "cn-to-runoff",
            "--cn",
            "80",
            "--precip-mm",
            "100",
            "-o",
            bogus_out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn lidar cn-to-runoff scalar with -o");
    assert_eq!(out_err.status.code(), Some(1));
    assert!(!bogus_out.exists());

    // 3. Raster mode with --cn-raster and --ia-ratio 0.05
    let cn_tif = tmp.path().join("cn_grid.tif");
    let rc_tif = tmp.path().join("rc_grid.tif");
    let mut cn_arr = lidar::Array2D::from_vec(2, 2, vec![80.0, 80.0, 80.0, -9999.0]).unwrap();
    cn_arr.set_no_data(-9999.0);
    cn_arr.geotransform = [500_000.0, 1.0, 0.0, 4_000_002.0, 0.0, -1.0];
    lidar::save_raster(&cn_arr, &cn_tif, None, false).unwrap();

    let out_ras = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args([
            "cn-to-runoff",
            "--cn-raster",
            cn_tif.to_str().unwrap(),
            "--precip-mm",
            "100",
            "--ia-ratio",
            "0.05",
            "-o",
            rc_tif.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("spawn lidar cn-to-runoff --cn-raster");
    assert!(
        out_ras.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_ras.stderr)
    );
    let ras_json: Value = serde_json::from_slice(&out_ras.stdout).expect("valid JSON");
    assert_eq!(ras_json["valid_cells"].as_u64(), Some(3));
    assert!((ras_json["mean_runoff_coeff"].as_f64().unwrap() - expected_rc).abs() < 1e-9);
    let loaded_rc = lidar::load_raster::<f64>(&rc_tif, None).unwrap();
    assert!((loaded_rc.data()[0] - expected_rc).abs() < 1e-9);
    assert_eq!(loaded_rc.data()[3], loaded_rc.no_data());
}

fn run_lidar(args: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn lidar {args:?}: {e}"));
    assert!(
        out.status.success(),
        "lidar {args:?} failed (exit {:?}):\nstdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "lidar {args:?} did not emit valid JSON ({e}):\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

fn make_synthetic_dem(path: &std::path::Path, w: usize, h: usize) {
    let mut data = Vec::with_capacity(w * h);
    let cx1 = (w as f64) * 0.35;
    let cy1 = (h as f64) * 0.5;
    let cx2 = (w as f64) * 0.7;
    let cy2 = (h as f64) * 0.5;
    for r in 0..h {
        for c in 0..w {
            let x = c as f64;
            let y = r as f64;
            let slope = 110.0 - 0.04 * x + 0.02 * y;
            let d1 = ((x - cx1).powi(2) + (y - cy1).powi(2)).sqrt();
            let d2 = ((x - cx2).powi(2) + (y - cy2).powi(2)).sqrt();
            let bowl1 = if d1 < 7.0 {
                -2.5 * (1.0 - (d1 / 7.0).powi(2))
            } else {
                0.0
            };
            let bowl2 = if d2 < 6.0 {
                -1.8 * (1.0 - (d2 / 6.0).powi(2))
            } else {
                0.0
            };
            data.push(slope + bowl1 + bowl2);
        }
    }
    let mut arr = lidar::Array2D::from_vec(w, h, data).unwrap();
    arr.set_no_data(-9999.0);
    arr.geotransform = [
        500_000.0,
        2.0,
        0.0,
        4_500_000.0 + (h as f64) * 2.0,
        0.0,
        -2.0,
    ];
    arr.projection = "EPSG:32614".to_string();
    lidar::save_raster(&arr, path, None, false).unwrap();
}

#[test]
fn cli_v06_filters_derivatives_and_merge_tree_execute_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let dem = tmp.path().join("dem.tif");
    make_synthetic_dem(&dem, 40, 32);
    let dem_s = dem.to_str().unwrap();

    // median-filter, gaussian-filter, bilateral-filter, feature-preserving-smoothing
    let med = tmp.path().join("med.tif");
    let gau = tmp.path().join("gau.tif");
    let bil = tmp.path().join("bil.tif");
    let fps = tmp.path().join("fps.tif");
    run_lidar(&[
        "median-filter",
        dem_s,
        med.to_str().unwrap(),
        "-k",
        "3",
        "--json",
    ]);
    run_lidar(&[
        "gaussian-filter",
        dem_s,
        gau.to_str().unwrap(),
        "-s",
        "1.0",
        "--json",
    ]);
    run_lidar(&[
        "bilateral-filter",
        dem_s,
        bil.to_str().unwrap(),
        "--spatial-sigma",
        "1.5",
        "--range-sigma",
        "0.8",
        "--json",
    ]);
    run_lidar(&[
        "feature-preserving-smoothing",
        dem_s,
        fps.to_str().unwrap(),
        "--filter-size",
        "5",
        "--norm-diff",
        "15.0",
        "--num-iter",
        "2",
        "--compress",
        "--json",
    ]);
    assert!(med.exists() && gau.exists() && bil.exists() && fps.exists());

    // flip-dem & breach-depressions
    let flipped = tmp.path().join("flipped.tif");
    let breached = tmp.path().join("breached.tif");
    run_lidar(&[
        "flip-dem",
        dem_s,
        flipped.to_str().unwrap(),
        "--delta",
        "200",
        "--json",
    ]);
    run_lidar(&[
        "breach-depressions",
        dem_s,
        breached.to_str().unwrap(),
        "--mode",
        "selective",
        "--max-depth",
        "3.0",
        "--fill-remaining",
        "--compress",
        "--json",
    ]);
    assert!(flipped.exists() && breached.exists());

    // dinf, mfd, twi, hand
    let dinf_dir = tmp.path().join("dinf_dir.tif");
    let dinf_acc = tmp.path().join("dinf_acc.tif");
    let mfd_acc = tmp.path().join("mfd_acc.tif");
    let twi = tmp.path().join("twi.tif");
    let hand = tmp.path().join("hand.tif");
    run_lidar(&[
        "dinf",
        breached.to_str().unwrap(),
        "--direction",
        dinf_dir.to_str().unwrap(),
        "--accumulation",
        dinf_acc.to_str().unwrap(),
        "--compress",
        "--json",
    ]);
    run_lidar(&[
        "mfd",
        breached.to_str().unwrap(),
        mfd_acc.to_str().unwrap(),
        "-p",
        "1.1",
        "--compress",
        "--json",
    ]);
    run_lidar(&[
        "twi",
        breached.to_str().unwrap(),
        twi.to_str().unwrap(),
        "--flow-acc",
        dinf_acc.to_str().unwrap(),
        "--compress",
        "--json",
    ]);
    run_lidar(&[
        "hand",
        breached.to_str().unwrap(),
        hand.to_str().unwrap(),
        "--stream-threshold",
        "15.0",
        "--compress",
        "--json",
    ]);
    assert!(
        dinf_dir.exists() && dinf_acc.exists() && mfd_acc.exists() && twi.exists() && hand.exists()
    );

    // burn-streams + burn-culverts + synthesize-culverts
    let streams_gpkg = tmp.path().join("streams.gpkg");
    let roads_gpkg = tmp.path().join("roads.gpkg");
    let utm14 = terrio::vector::Crs::new().with_epsg(32614);
    let mut s_layer =
        terrio::Layer::new("streams").with_geom_type(terrio::GeometryType::LineString);
    s_layer.crs = Some(utm14.clone());
    s_layer.features.push(terrio::Feature {
        fid: 1,
        geometry: Some(terrio::Geometry::LineString(vec![
            terrio::vector::Coord::xy(500_010.0, 4_500_032.0),
            terrio::vector::Coord::xy(500_070.0, 4_500_032.0),
        ])),
        attributes: vec![],
    });
    terrio::write_vector(
        &[s_layer],
        streams_gpkg.to_str().unwrap(),
        &terrio::VectorWriteOptions::default(),
    )
    .unwrap();
    let mut r_layer = terrio::Layer::new("roads").with_geom_type(terrio::GeometryType::LineString);
    r_layer.crs = Some(utm14);
    r_layer.features.push(terrio::Feature {
        fid: 1,
        geometry: Some(terrio::Geometry::LineString(vec![
            terrio::vector::Coord::xy(500_040.0, 4_500_010.0),
            terrio::vector::Coord::xy(500_040.0, 4_500_054.0),
        ])),
        attributes: vec![],
    });
    terrio::write_vector(
        &[r_layer],
        roads_gpkg.to_str().unwrap(),
        &terrio::VectorWriteOptions::default(),
    )
    .unwrap();
    let burned_streams = tmp.path().join("burned_streams.tif");
    let synth_culverts = tmp.path().join("synth_culverts.gpkg");
    let burned_culverts = tmp.path().join("burned_culverts.tif");
    run_lidar(&[
        "burn-streams",
        dem_s,
        streams_gpkg.to_str().unwrap(),
        burned_streams.to_str().unwrap(),
        "--depth",
        "1.5",
        "--buffer-cells",
        "2",
        "--smooth-drop",
        "0.5",
        "--json",
    ]);
    run_lidar(&[
        "synthesize-culverts",
        streams_gpkg.to_str().unwrap(),
        roads_gpkg.to_str().unwrap(),
        synth_culverts.to_str().unwrap(),
        "--half-length",
        "10",
        "--snap-distance",
        "5",
        "--crs",
        "EPSG:32614",
        "--json",
    ]);
    run_lidar(&[
        "burn-culverts",
        dem_s,
        synth_culverts.to_str().unwrap(),
        burned_culverts.to_str().unwrap(),
        "--json",
    ]);
    assert!(burned_streams.exists() && synth_culverts.exists() && burned_culverts.exists());

    // extract-sinks -> merge-tree -> delineate-depressions -> delineate-catchments -> delineate-flow-paths -> catchment-hierarchy -> simulate-inundation -> export-frames -> animate-inundation
    let sinks_dir = tmp.path().join("sinks");
    run_lidar(&[
        "extract-sinks",
        dem_s,
        sinks_dir.to_str().unwrap(),
        "-m",
        "5",
        "--json",
    ]);
    let sink_tif = sinks_dir.join("sink.tif");
    let ref_dem = sinks_dir.join("dem.tif");
    let filled_dem = sinks_dir.join("dem_filled.tif");
    let region_tif = sinks_dir.join("region.tif");
    assert!(sink_tif.exists() && ref_dem.exists() && filled_dem.exists());

    let mt_dir = tmp.path().join("merge_tree");
    let mt_json = run_lidar(&[
        "merge-tree",
        sink_tif.to_str().unwrap(),
        mt_dir.to_str().unwrap(),
        "--dem",
        ref_dem.to_str().unwrap(),
        "--dem-is-refined",
        "-m",
        "5",
        "-d",
        "0.1",
        "--max-rating-curve-points",
        "16",
        "--compress",
        "--json",
    ]);
    assert!(mt_json["num_nodes"].as_u64().unwrap() >= 1);
    assert!(mt_dir.join("merge_tree_nodes.json").exists());
    assert!(mt_dir.join("rating_curves.csv").exists());

    let dep_dir = tmp.path().join("deps");
    run_lidar(&[
        "delineate-depressions",
        sink_tif.to_str().unwrap(),
        dep_dir.to_str().unwrap(),
        "-m",
        "5",
        "-d",
        "0.1",
        "-i",
        "0.2",
        "--level-shp",
        "--gpkg",
        "--json",
    ]);
    assert!(dep_dir.join("depressions.gpkg").exists());

    let mnt_dir = tmp.path().join("mounts");
    let mnt_json = run_lidar(&[
        "delineate-mounts",
        flipped.to_str().unwrap(),
        mnt_dir.to_str().unwrap(),
        "-m",
        "5",
        "-H",
        "0.1",
        "-i",
        "0.3",
        "--gpkg",
        "--json",
    ]);
    let mnt_gpkg = mnt_json["mounts"]["gpkg_path"].as_str().unwrap();
    assert!(std::path::Path::new(mnt_gpkg).exists());

    let cat_dir = tmp.path().join("catchments");
    let cat_json = run_lidar(&[
        "delineate-catchments",
        ref_dem.to_str().unwrap(),
        region_tif.to_str().unwrap(),
        cat_dir.to_str().unwrap(),
        "--json",
    ]);
    assert!(std::path::Path::new(cat_json["watershed_path"].as_str().unwrap()).exists());

    let fp_out = tmp.path().join("flow_paths.gpkg");
    run_lidar(&[
        "delineate-flow-paths",
        filled_dem.to_str().unwrap(),
        region_tif.to_str().unwrap(),
        fp_out.to_str().unwrap(),
        "--refined-dem",
        ref_dem.to_str().unwrap(),
        "--json",
    ]);
    assert!(fp_out.exists());

    let cat_hir = tmp.path().join("catchment_hir.tif");
    run_lidar(&[
        "catchment-hierarchy",
        ref_dem.to_str().unwrap(),
        dep_dir.join("shp-level").to_str().unwrap(),
        cat_hir.to_str().unwrap(),
        "--json",
    ]);
    assert!(cat_hir.exists());

    let sim_dir = tmp.path().join("sim");
    run_lidar(&[
        "simulate-inundation",
        sink_tif.to_str().unwrap(),
        cat_hir.to_str().unwrap(),
        sim_dir.to_str().unwrap(),
        "-m",
        "5",
        "-d",
        "0.1",
        "-i",
        "0.2",
        "--duration",
        "6",
        "--time-step",
        "2",
        "--json",
    ]);
    let fill_time = sim_dir.join("fill_time.tif");
    assert!(fill_time.exists());

    let frames_dir = tmp.path().join("frames");
    let frames_json = run_lidar(&[
        "export-frames",
        dem_s,
        fill_time.to_str().unwrap(),
        frames_dir.to_str().unwrap(),
        "--num-steps",
        "3",
        "--time-step",
        "2",
        "--scale-bar",
        "--legend",
        "--json",
    ]);
    assert_eq!(frames_json["frame_count"].as_u64(), Some(3));
    let first_frame = frames_json["frames"][0].as_str().unwrap();
    assert!(std::path::Path::new(first_frame).exists());

    let gif_path = tmp.path().join("anim.gif");
    run_lidar(&[
        "animate-inundation",
        dem_s,
        sim_dir.to_str().unwrap(),
        gif_path.to_str().unwrap(),
        "--no-ping-pong",
        "--json",
    ]);
    assert!(gif_path.exists());
}

#[test]
fn cli_raster_vector_pointcloud_tiled_and_fsm_pipeline_options_execute_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let dem = tmp.path().join("dem.tif");
    make_synthetic_dem(&dem, 36, 32);
    let dem_s = dem.to_str().unwrap();

    // info, polygonize, to-cog, add-crs, resample, reproject, clip, mosaic
    let info_json = run_lidar(&["info", dem_s, "--stats", "--json"]);
    assert!(
        info_json.is_object(),
        "info --json must return a JSON object"
    );

    let cog = tmp.path().join("dem_cog.tif");
    run_lidar(&["to-cog", dem_s, cog.to_str().unwrap(), "--json"]);
    assert!(cog.exists());

    let crs_tif = tmp.path().join("dem_crs.tif");
    run_lidar(&[
        "add-crs",
        dem_s,
        "EPSG:32614",
        "-o",
        crs_tif.to_str().unwrap(),
        "--json",
    ]);
    assert!(crs_tif.exists());

    let resampled = tmp.path().join("resampled_cog.tif");
    run_lidar(&[
        "resample",
        dem_s,
        resampled.to_str().unwrap(),
        "--resolution",
        "4.0",
        "--compress",
        "--cog",
        "--json",
    ]);
    assert!(resampled.exists());

    let reproj = tmp.path().join("reproj_cog.tif");
    run_lidar(&[
        "reproject",
        dem_s,
        reproj.to_str().unwrap(),
        "--crs",
        "EPSG:4326",
        "--compress",
        "--cog",
        "--json",
    ]);
    assert!(reproj.exists());

    let reproj_arr = lidar::load_raster::<f64>(&reproj, None).unwrap();
    let gt = reproj_arr.geotransform;
    let x0 = gt[0];
    let x1 = gt[0] + gt[1] * reproj_arr.width() as f64;
    let y0 = gt[3];
    let y1 = gt[3] + gt[5] * reproj_arr.height() as f64;
    let (min_x, max_x) = (x0.min(x1), x0.max(x1));
    let (min_y, max_y) = (y0.min(y1), y0.max(y1));
    let mx0 = min_x + (max_x - min_x) * 0.2;
    let mx1 = min_x + (max_x - min_x) * 0.8;
    let my0 = min_y + (max_y - min_y) * 0.2;
    let my1 = min_y + (max_y - min_y) * 0.8;
    let mask_gj = tmp.path().join("mask.geojson");
    std::fs::write(
        &mask_gj,
        format!(
            r#"{{"type":"FeatureCollection","features":[{{"type":"Feature","properties":{{"id":1}},"geometry":{{"type":"Polygon","coordinates":[[[{mx0},{my0}],[{mx1},{my0}],[{mx1},{my1}],[{mx0},{my1}],[{mx0},{my0}]]]}}}}]}}"#
        ),
    )
    .unwrap();
    let clipped = tmp.path().join("clipped_cog.tif");
    run_lidar(&[
        "clip",
        reproj.to_str().unwrap(),
        mask_gj.to_str().unwrap(),
        clipped.to_str().unwrap(),
        "--compress",
        "--cog",
        "--json",
    ]);
    assert!(clipped.exists());

    let mosaicked = tmp.path().join("mosaic_cog.tif");
    run_lidar(&[
        "mosaic",
        dem_s,
        crs_tif.to_str().unwrap(),
        "-o",
        mosaicked.to_str().unwrap(),
        "--resampling",
        "bilinear",
        "--compress",
        "--cog",
        "--json",
    ]);
    assert!(mosaicked.exists());

    // polygonize + join-tables + csv-to-vector
    let mut cls_ras = lidar::Array2D::from_vec(4, 4, vec![1.0; 16]).unwrap();
    cls_ras.set_no_data(0.0);
    cls_ras.geotransform = [500_000.0, 1.0, 0.0, 4_500_004.0, 0.0, -1.0];
    cls_ras.projection = "EPSG:32614".to_string();
    let cls_tif = tmp.path().join("cls.tif");
    lidar::save_raster(&cls_ras, &cls_tif, None, false).unwrap();
    let poly_gpkg = tmp.path().join("poly.gpkg");
    run_lidar(&[
        "polygonize",
        cls_tif.to_str().unwrap(),
        poly_gpkg.to_str().unwrap(),
        "-f",
        "id",
        "--json",
    ]);
    let attr_csv = tmp.path().join("attrs.csv");
    std::fs::write(&attr_csv, "region_id,label\n1,basin_a\n").unwrap();
    let joined_gpkg = tmp.path().join("joined.gpkg");
    run_lidar(&[
        "join-tables",
        poly_gpkg.to_str().unwrap(),
        attr_csv.to_str().unwrap(),
        joined_gpkg.to_str().unwrap(),
        "--vec-col",
        "id",
        "--csv-col",
        "region_id",
        "--json",
    ]);
    assert!(joined_gpkg.exists());

    let pts_csv = tmp.path().join("pts.csv");
    std::fs::write(&pts_csv, "latitude,longitude,name\n47.1,-99.1,site1\n").unwrap();
    let pts_gpkg = tmp.path().join("pts.gpkg");
    run_lidar(&[
        "csv-to-vector",
        pts_csv.to_str().unwrap(),
        pts_gpkg.to_str().unwrap(),
        "--json",
    ]);
    assert!(pts_gpkg.exists());

    // Point cloud subcommands: convert-lidar, lidar-to-dsm, lidar-to-dtm (incl --tiled & --cog & --sor-k), lidar-to-chm (incl --pit-free & --tiled)
    let in_las = tmp.path().join("cloud.las");
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    for r in 0..16 {
        for c in 0..16 {
            let x = 500_000.0 + c as f64;
            let y = 4_500_000.0 + r as f64;
            xs.push(x);
            ys.push(y);
            zs.push(100.0);
            cls.push(2u8);
            if (4..12).contains(&r) && (4..12).contains(&c) {
                xs.push(x + 0.2);
                ys.push(y + 0.2);
                zs.push(110.0);
                cls.push(5u8);
            }
        }
    }
    let cloud = lidar::pointcloud::PointCloud::new(
        vec![
            lidar::pointcloud::Dimension::new("x", lidar::pointcloud::Values::F64(xs)),
            lidar::pointcloud::Dimension::new("y", lidar::pointcloud::Values::F64(ys)),
            lidar::pointcloud::Dimension::new("z", lidar::pointcloud::Values::F64(zs)),
            lidar::pointcloud::Dimension::new("classification", lidar::pointcloud::Values::U8(cls)),
        ],
        terrio::crs::SpatialRef::parse("EPSG:32614").unwrap(),
    );
    lidar::pointcloud::write_lidar(
        &cloud,
        &in_las,
        &lidar::pointcloud::PointWriteOptions::default(),
    )
    .unwrap();

    let conv_laz = tmp.path().join("cloud.laz");
    run_lidar(&[
        "convert-lidar",
        in_las.to_str().unwrap(),
        conv_laz.to_str().unwrap(),
        "--json",
    ]);
    assert!(conv_laz.exists());

    let dsm_tif = tmp.path().join("dsm.tif");
    run_lidar(&[
        "lidar-to-dsm",
        in_las.to_str().unwrap(),
        dsm_tif.to_str().unwrap(),
        "--resolution",
        "2.0",
        "--compress",
        "--cog",
        "--json",
    ]);
    let dtm_tiled = tmp.path().join("dtm_tiled.tif");
    run_lidar(&[
        "lidar-to-dtm",
        in_las.to_str().unwrap(),
        dtm_tiled.to_str().unwrap(),
        "--ground-method",
        "preclassified",
        "--resolution",
        "2.0",
        "--tiled",
        "--tile-size",
        "8.0",
        "--overlap",
        "2.0",
        "--sor-k",
        "4",
        "--sor-sigma",
        "2.5",
        "--compress",
        "--json",
    ]);
    let chm_pf = tmp.path().join("chm_pitfree.tif");
    run_lidar(&[
        "lidar-to-chm",
        in_las.to_str().unwrap(),
        chm_pf.to_str().unwrap(),
        "--ground-method",
        "preclassified",
        "--resolution",
        "2.0",
        "--pit-free",
        "--height-thresholds",
        "0,2,5",
        "--max-triangle-edge",
        "8.0",
        "--compress",
        "--json",
    ]);
    let chm_tiled = tmp.path().join("chm_tiled.tif");
    run_lidar(&[
        "lidar-to-chm",
        in_las.to_str().unwrap(),
        chm_tiled.to_str().unwrap(),
        "--ground-method",
        "preclassified",
        "--resolution",
        "2.0",
        "--tiled",
        "--tile-size",
        "8.0",
        "--overlap",
        "2.0",
        "--compress",
        "--json",
    ]);
    assert!(dsm_tif.exists() && dtm_tiled.exists() && chm_pf.exists() && chm_tiled.exists());

    // tiled-depressions with --write-rasters, --write-gpkg, --delineation-method merge-tree
    let td_dir = tmp.path().join("tiled_deps");
    run_lidar(&[
        "tiled-depressions",
        dem_s,
        td_dir.to_str().unwrap(),
        "--tile-size",
        "20",
        "--overlap",
        "6",
        "-k",
        "1",
        "-m",
        "5",
        "-d",
        "0.1",
        "--delineation-method",
        "merge-tree",
        "--write-rasters",
        "--write-gpkg",
        "--compress",
        "--json",
    ]);
    assert!(td_dir.join("depression_id.tif").exists());
    assert!(td_dir.join("depressions.gpkg").exists());

    // pipeline --routing fsm with merge-tree, Green-Ampt, hazard-dir, recession, attenuation
    let pipe_dir = tmp.path().join("pipe_fsm");
    let hazard_dir = tmp.path().join("hazard_out");
    let pipe_json = run_lidar(&[
        "pipeline",
        dem_s,
        pipe_dir.to_str().unwrap(),
        "-k",
        "1",
        "-m",
        "5",
        "-d",
        "0.1",
        "-i",
        "0.2",
        "--delineation-method",
        "merge-tree",
        "--routing",
        "fsm",
        "--rain-intensity",
        "8.0",
        "--duration",
        "6.0",
        "--time-step",
        "2.0",
        "--green-ampt-ks",
        "0.5",
        "--green-ampt-psi",
        "10.0",
        "--green-ampt-dtheta",
        "0.3",
        "--hazard-dir",
        hazard_dir.to_str().unwrap(),
        "--recession-et-cm-h",
        "0.5",
        "--recession-seepage-cm-h",
        "0.5",
        "--recession-max-h",
        "24.0",
        "--attenuate-travel-time-h",
        "1.0",
        "--attenuate-dt-h",
        "0.5",
        "--attenuate-reservoirs",
        "2",
        "--no-vectors",
        "--json",
    ]);
    assert!(pipe_json["num_depressions"].as_u64().unwrap() >= 1);
    assert!(hazard_dir.join("max_depth.tif").exists());
    assert!(hazard_dir.join("wse.tif").exists());
    assert!(hazard_dir.join("duration_h.tif").exists());
    assert!(pipe_dir.join("fsm_recession_drain_time.tif").exists());
    assert!(pipe_dir.join("fsm_recession_curve.csv").exists());
    assert!(pipe_dir.join("fsm_hydrograph_attenuated.csv").exists());
}

#[test]
fn mcp_server_emits_notifications_progress_for_pipeline_with_progress_token() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::Stdio;

    let tmp = tempfile::tempdir().unwrap();
    let dem = tmp.path().join("dem.tif");
    make_synthetic_dem(&dem, 32, 28);
    let out_dir = tmp.path().join("mcp_pipe");

    let mut child = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args(["mcp", "--allow-dir", tmp.path().to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lidar mcp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    // 1. Initialize handshake
    writeln!(
        stdin,
        "{}",
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "cli-test", "version": "0.1" }
            }
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    let mut init_line = String::new();
    reader.read_line(&mut init_line).unwrap();
    let init_resp: Value = serde_json::from_str(&init_line).unwrap();
    assert_eq!(init_resp["id"], 1);

    // 2. Call `pipeline` with `_meta.progressToken`
    writeln!(
        stdin,
        "{}",
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "pipeline",
                "arguments": {
                    "args": [
                        dem.to_str().unwrap(),
                        out_dir.to_str().unwrap(),
                        "-k", "1",
                        "-m", "5",
                        "-d", "0.1",
                        "-i", "0.3",
                        "--duration", "4.0",
                        "--time-step", "2.0",
                        "--no-rasters",
                        "--no-vectors",
                        "--no-graph",
                        "--json"
                    ]
                },
                "_meta": { "progressToken": "tok-pipe-1" }
            }
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    let mut progress_frames = Vec::new();
    let mut final_resp: Option<Value> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).unwrap();
        if n == 0 {
            break;
        }
        let msg: Value = serde_json::from_str(&line).unwrap();
        if msg.get("method").and_then(Value::as_str) == Some("notifications/progress") {
            progress_frames.push(msg);
        } else if msg.get("id") == Some(&Value::from(2)) {
            final_resp = Some(msg);
            break;
        }
    }

    drop(stdin);
    let _ = child.wait();

    assert!(
        !progress_frames.is_empty(),
        "expected at least one notifications/progress frame during pipeline execution"
    );
    assert!(progress_frames.iter().all(|f| {
        f["params"]["progressToken"] == "tok-pipe-1"
            && f["params"]["total"] == 1.0
            && f["params"]["progress"]
                .as_f64()
                .is_some_and(|p| (0.0..=1.0).contains(&p))
    }));
    let resp = final_resp.expect("expected final tools/call response for id=2");
    assert_ne!(resp["result"]["isError"], Value::Bool(true));
}
