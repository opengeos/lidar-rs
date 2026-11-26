//! Subprocess regression tests for v0.6.0 CLI and MCP security & correctness audit findings.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn lidar_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lidar")
}

fn test_data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join(name)
}

#[test]
fn mcp_allow_dir_blocks_forcing_path_urls_and_zip_symlink_bypasses() {
    let tmp = tempfile::tempdir().unwrap();
    let allowed = tmp.path().join("allowed");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&allowed).unwrap();
    std::fs::create_dir_all(&outside).unwrap();

    let allowed_sink = allowed.join("sink.tif");
    let allowed_cat = allowed.join("cat.tif");
    let allowed_out = allowed.join("out");
    let outside_tif = outside.join("secret.tif");
    let outside_zip = outside.join("secret.zip");
    std::fs::write(&allowed_sink, b"dummy").unwrap();
    std::fs::write(&allowed_cat, b"dummy").unwrap();
    std::fs::write(&outside_tif, b"secret").unwrap();
    std::fs::write(&outside_zip, b"PK\x03\x04").unwrap();

    #[cfg(unix)]
    let zip_symlink = {
        let link = allowed.join("escape.zip");
        std::os::unix::fs::symlink(&outside_zip, &link).unwrap();
        link
    };

    let mut child = Command::new(lidar_bin())
        .args(["mcp", "--allow-dir", allowed.to_str().unwrap()])
        .current_dir(&allowed)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());

    let mut rpc = |req: Value| -> Value {
        writeln!(stdin, "{req}").unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    };

    // 1. SEC-1: simulate-inundation string-typed forcing path outside --allow-dir must be rejected.
    let r1 = rpc(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "simulate-inundation",
            "arguments": {
                "args": [
                    allowed_sink.to_str().unwrap(),
                    allowed_cat.to_str().unwrap(),
                    allowed_out.to_str().unwrap(),
                    "--runoff-coeff",
                    outside_tif.to_str().unwrap()
                ]
            }
        }
    }));
    assert_eq!(r1["error"]["code"], -32602, "expected INVALID_PARAMS: {r1}");

    // 2. SEC-2: remote URLs and VFS prefixes must be rejected under --allow-dir.
    for (id, target) in [
        (2, "http://169.254.169.254/latest/meta-data/"),
        (3, "WFS:http://127.0.0.1:8080/geoserver"),
        (4, "/vsicurl/https://example.org/dem.tif"),
        (40, "/vsimem/secret.tif"),
        (41, "DODS:http://127.0.0.1:8080/dem.tif"),
        (42, "PG:dbname=prod"),
        (43, "NETCDF:/outside/secret.nc:var"),
    ] {
        let r = rpc(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": "info",
                "arguments": { "args": [target] }
            }
        }));
        assert_eq!(
            r["error"]["code"], -32602,
            "URL/VFS '{target}' must be blocked under --allow-dir: {r}"
        );
    }

    // 3. SEC-2: symlinked `.zip!member` escaping --allow-dir must be rejected.
    #[cfg(unix)]
    {
        let vfs_arg = format!("{}!dem.tif", zip_symlink.display());
        let r5 = rpc(json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "info",
                "arguments": { "args": [vfs_arg] }
            }
        }));
        assert_eq!(
            r5["error"]["code"], -32602,
            "symlinked .zip! escape must be blocked under --allow-dir: {r5}"
        );
    }

    // 4. SEC-3: oversized batch (> 128) must be rejected cleanly.
    let huge_batch: Vec<Value> = (0..130)
        .map(|i| json!({"jsonrpc": "2.0", "id": i, "method": "ping"}))
        .collect();
    let r6 = rpc(Value::Array(huge_batch));
    assert_eq!(
        r6["error"]["code"], -32600,
        "oversized batch must fail: {r6}"
    );

    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success());
}

#[test]
fn mcp_preview_json_clustered_flags_and_invalid_utf8_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let dem = test_data("dem.tif");
    let explicit_out = tmp.path().join("clustered.png");

    let mut child = Command::new(lidar_bin())
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());

    let mut read_reply = || -> Value {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    };

    // 1. COR-5: invalid UTF-8 bytes emit PARSE_ERROR (-32700) without killing the server.
    stdin.write_all(b"\xff\xfe\n").unwrap();
    stdin.flush().unwrap();
    let err_utf8 = read_reply();
    assert_eq!(
        err_utf8["error"]["code"], -32700,
        "invalid UTF-8 must emit PARSE_ERROR: {err_utf8}"
    );

    // Server must still respond to ping after invalid UTF-8 line.
    writeln!(
        stdin,
        "{}",
        json!({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    )
    .unwrap();
    stdin.flush().unwrap();
    let pong = read_reply();
    assert_eq!(pong["id"], 1);
    assert!(pong["result"].is_object());

    // 2. COR-2: preview --json (without -o) must not leak deleted temp file path.
    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "preview",
                "arguments": { "args": [dem.to_str().unwrap(), "--max-dim", "64", "--json"] }
            }
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let prev_json = read_reply();
    assert_eq!(prev_json["result"]["isError"], false, "{prev_json}");
    let content = prev_json["result"]["content"].as_array().unwrap();
    assert!(content.iter().any(|c| c["type"] == "image"));
    let text_payload: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        text_payload["output"],
        Value::Null,
        "temp path must be scrubbed in JSON output: {text_payload}"
    );
    assert_eq!(text_payload["inline_image"], true);

    // 3. COR-4: clustered short flag `-qo <FILE>` must be recognized as explicit output.
    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "preview",
                "arguments": {
                    "args": ["-qo", explicit_out.to_str().unwrap(), "--max-dim", "64", dem.to_str().unwrap()]
                }
            }
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let prev_cluster = read_reply();
    assert_eq!(prev_cluster["result"]["isError"], false, "{prev_cluster}");
    assert!(
        explicit_out.exists(),
        "clustered -qo must write preview PNG"
    );

    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success());
}

#[test]
fn connectivity_graph_rejects_conflicting_format_with_json_on_stdout() {
    let dem = test_data("dem.tif");
    let out = Command::new(lidar_bin())
        .args([
            "connectivity-graph",
            dem.to_str().unwrap(),
            "--format",
            "graphml",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "connectivity-graph --format graphml --json without -o must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--json conflicts with --format"),
        "expected format conflict error, got: {stderr}"
    );
}
