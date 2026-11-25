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
