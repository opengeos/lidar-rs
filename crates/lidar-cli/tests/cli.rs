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
