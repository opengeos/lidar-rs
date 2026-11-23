//! End-to-end tests driving `lidar mcp` over stdin/stdout, as an MCP client runs it.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
}

fn session(messages: &[Value]) -> HashMap<u64, Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn lidar mcp");
    let mut stdin = child.stdin.take().unwrap();
    for m in messages {
        writeln!(stdin, "{m}").unwrap();
    }
    let requests = messages.iter().filter(|m| m.get("id").is_some()).count();
    let replies: HashMap<u64, Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .take(requests)
        .map(|l| {
            let v: Value = serde_json::from_str(&l.unwrap()).unwrap();
            (v["id"].as_u64().unwrap(), v)
        })
        .collect();
    drop(stdin);
    assert!(child.wait().unwrap().success());
    replies
}

fn call(id: u32, name: &str, args: &[&str]) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {"name": name, "arguments": {"args": args}}
    })
}

#[test]
fn initializes_lists_tools_prompts_and_handles_errors() {
    let dem = test_data_dir().join("dem.tif");
    let dem_str = dem.to_str().unwrap();

    let replies = session(&[
        json!({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"}
            }
        }),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "prompts/list"}),
        call(3, "info", &[dem_str, "--json", "--stats"]),
        call(4, "info", &["/no/such/file.tif"]),
        call(5, "info", &["--no-such-flag"]),
    ]);

    assert_eq!(replies[&0]["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(replies[&0]["result"]["serverInfo"]["name"], "lidar-rs");

    let tools = replies[&1]["result"]["tools"].as_array().unwrap();
    assert!(tools.iter().any(|t| t["name"] == "pipeline"));
    assert!(tools.iter().any(|t| t["name"] == "preview"));
    assert!(!tools.iter().any(|t| t["name"] == "mcp"));

    let prompts = replies[&2]["result"]["prompts"].as_array().unwrap();
    assert!(prompts.iter().any(|p| p["name"] == "delineate-depressions"));

    let info_res = &replies[&3]["result"];
    assert_eq!(info_res["isError"], false, "{info_res}");
    let info_json: Value =
        serde_json::from_str(info_res["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(info_json.is_object(), "expected structured JSON info");

    assert_eq!(replies[&4]["result"]["isError"], true);
    assert!(
        replies[&4]["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("exit code 1")
    );
    assert_eq!(replies[&5]["result"]["isError"], true);
    assert!(
        replies[&5]["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("exit code 2")
    );
}

#[test]
fn preview_returns_inline_png_image_and_pipeline_runs_end_to_end() {
    let dem = test_data_dir().join("dem.tif");
    let dem_str = dem.to_str().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let pipe_dir = tmp.path().join("pipe_out");
    let pipe_str = pipe_dir.to_str().unwrap();

    // Run preview without -o (should attach inline base64 PNG) and pipeline with --json.
    let replies = session(&[
        call(1, "preview", &[dem_str, "--max-dim", "128", "--scale-bar"]),
        call(
            2,
            "pipeline",
            &[
                dem_str,
                pipe_str,
                "--min-size",
                "1000",
                "--min-depth",
                "0.3",
                "--interval",
                "0.3",
                "--rain-intensity",
                "5",
                "--duration",
                "10",
                "--time-step",
                "2",
                "--routing",
                "fsm",
                "--json",
            ],
        ),
    ]);

    let prev = &replies[&1]["result"];
    assert_eq!(prev["isError"], false, "{prev}");
    let content = prev["content"].as_array().unwrap();
    let img = content
        .iter()
        .find(|c| c["type"] == "image")
        .expect("preview must attach image content");
    assert_eq!(img["mimeType"], "image/png");
    // Base64 of PNG magic bytes `\x89PNG\r\n\x1a\n` starts with `iVBORw0KGgo`
    assert!(img["data"].as_str().unwrap().starts_with("iVBORw0KGgo"));

    let pipe_res = &replies[&2]["result"];
    assert_eq!(pipe_res["isError"], false, "{pipe_res}");
    let summary: Value =
        serde_json::from_str(pipe_res["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(summary["num_depressions"].as_u64().unwrap() > 0);
    assert!(summary["fsm_mass_balance"].is_object());

    // Now visually inspect the generated fill_time.tif overlay through MCP preview.
    let fill_time = pipe_dir.join("fill_time.tif");
    let fill_str = fill_time.to_str().unwrap();
    let replies2 = session(&[call(
        1,
        "preview",
        &[
            dem_str,
            "--fill-time",
            fill_str,
            "--hour",
            "6",
            "--max-dim",
            "128",
            "--scale-bar",
            "--legend",
        ],
    )]);
    let prev2 = &replies2[&1]["result"];
    assert_eq!(prev2["isError"], false, "{prev2}");
    let img2 = prev2["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["type"] == "image")
        .expect("inundation preview must attach image content");
    assert!(img2["data"].as_str().unwrap().starts_with("iVBORw0KGgo"));
}

#[test]
fn json_usage_catalog_describes_root_and_subcommand() {
    let out = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .arg("--json-usage")
        .output()
        .expect("run lidar --json-usage");
    assert!(out.status.success());
    let root: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(root["program"], "lidar");
    let subs = root["command"]["subcommands"].as_array().unwrap();
    assert!(subs.iter().any(|c| c["name"] == "pipeline"));
    assert!(!subs.iter().any(|c| c["name"] == "mcp"));

    let out_sub = Command::new(env!("CARGO_BIN_EXE_lidar"))
        .args(["pipeline", "--json-usage"])
        .output()
        .expect("run lidar pipeline --json-usage");
    assert!(out_sub.status.success());
    let sub: Value = serde_json::from_slice(&out_sub.stdout).unwrap();
    assert_eq!(sub["command"]["name"], "pipeline");
    assert!(
        sub["command"]["arguments"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["long"] == "--routing")
    );
}
