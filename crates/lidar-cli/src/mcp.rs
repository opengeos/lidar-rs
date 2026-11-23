//! `lidar mcp`: a pure-Rust Model Context Protocol (MCP) server over standard input and output,
//! so AI agents (Claude Code, Claude Desktop, Cursor, VS Code, etc.) can inspect terrain/LiDAR
//! datasets, run depression delineation, hydro-conditioning, Fill-Spill-Merge inundation
//! simulations, federal data downloads, and visually inspect DEM/inundation previews.
//!
//! Modeled after `terrio mcp`:
//! - Every CLI subcommand (except `mcp` and `help`) is automatically exposed as an MCP tool whose
//!   arguments are its CLI `args` array and whose description is its full Clap `--help`.
//! - [`Server::handle`] processes one JSON-RPC 2.0 message (single or batch) and returns its reply
//!   or spawns background worker threads so the server stays responsive (`ping`, cancellations,
//!   other calls) while long hydrology pipelines run.
//! - Each tool runs in an isolated child process (`lidar <tool> <args>`) with bounded stdout/stderr
//!   capture, cancellation via `notifications/cancelled`, and clean child teardown on client EOF.
//! - `preview` automatically attaches its rendered PNG as MCP `image` content so multimodal agents
//!   can visually inspect DEMs, depressions, and flood inundation maps directly.
//! - Curated MCP `prompts` guide agents through end-to-end watershed hydrology and LiDAR workflows.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{Value, json};

use crate::usage::{command_names, commands, with_clap_stack};

/// Protocol versions supported by this server, newest first.
const VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Maximum combined bytes of command output (stderr + stdout + notices) returned to the agent.
const MAX_OUTPUT: usize = 100_000;

/// Maximum PNG size in bytes attached as MCP image content for `preview` (8 MiB).
const MAX_IMAGE: u64 = 8 << 20;

/// Maximum concurrent tool calls running across the entire session (standalone + batches).
const MAX_CONCURRENT_TOOLS: usize = 8;

/// Instructions sent to the AI agent during MCP `initialize`.
const INSTRUCTIONS: &str = "lidar-rs performs terrain depression delineation (Wu et al. 2019), hydro-conditioning, LiDAR point-cloud gridding (DSM/DTM/CHM), fill-and-spill inundation simulation (static and dynamic Fill-Spill-Merge), and U.S. federal geospatial data download in pure Rust with zero GDAL dependencies.\n\
- Each tool runs one `lidar` subcommand; pass `args` as an array of strings after the command name, e.g. `info` with [\"/abs/path/dem.tif\", \"--json\", \"--stats\"].\n\
- Each tool's description is the subcommand's full `--help` (arguments, flags, defaults). Always use absolute paths.\n\
- Start by inspecting inputs with `info` (pass `--json` and optionally `--stats`) to check CRS, pixel resolution, dimensions, NoData, and elevation range.\n\
- For end-to-end terrain hydrology (smoothing, optional breaching/culverts, sink extraction, nested depressions, catchment hierarchy, inundation simulation, rating curves, connectivity graph, and GeoPackage export), prefer the single-call `pipeline` tool with `--json` (use `--routing fsm` for dynamic mass-conserving Fill-Spill-Merge cascade routing and `--scs-type2-mm` for design storms).\n\
- When the `download` tool is available, run `download` with `--dry-run` first to inspect matching USGS 3DEP tiles before downloading full bundles (`--huc`, `--bbox`, or `--region` with `--json`).\n\
- Use `preview` to visually inspect a DEM, depression depth raster (`--depth`), or inundation snapshot (`--fill-time`, `--hour`); omit `-o` to receive the shaded-relief PNG directly as image content without leaving files behind.\n\
- Use `prompts/list` for guided step-by-step hydrology and LiDAR workflows.";

/// Commands that only read inputs and never write files.
const READ_ONLY: &[&str] = &["info"];

/// Commands that contact external network services.
const OPEN_WORLD: &[&str] = &["download"];

type RpcError = (i64, String);

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

/// A queued tool call: request id, tool name, and CLI arguments.
type ToolCall = (Value, String, Vec<OsString>);

/// Session-wide concurrency limiter shared across standalone calls and JSON-RPC batches.
struct ConcurrencyLimiter {
    active: Mutex<usize>,
    cv: Condvar,
    limit: usize,
}

struct ConcurrencyPermit {
    limiter: Arc<ConcurrencyLimiter>,
}

impl Drop for ConcurrencyPermit {
    fn drop(&mut self) {
        let mut active = self
            .limiter
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *active = active.saturating_sub(1);
        self.limiter.cv.notify_one();
    }
}

impl ConcurrencyLimiter {
    fn new(limit: usize) -> Self {
        Self {
            active: Mutex::new(0),
            cv: Condvar::new(),
            limit: limit.max(1),
        }
    }

    /// Acquires a permit, returning `None` if `cancel` is set before a slot opens.
    fn acquire(self: &Arc<Self>, cancel: &AtomicBool) -> Option<ConcurrencyPermit> {
        let mut active = self.active.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            if *active < self.limit {
                *active += 1;
                return Some(ConcurrencyPermit {
                    limiter: self.clone(),
                });
            }
            let (guard, _) = self
                .cv
                .wait_timeout(active, Duration::from_millis(50))
                .unwrap_or_else(PoisonError::into_inner);
            active = guard;
        }
    }
}

/// Stateful JSON-RPC 2.0 MCP server.
pub struct Server {
    /// Command line that invokes `lidar` (either the native binary or `python -m lidar_rs`).
    lidar: Arc<Vec<OsString>>,
    /// Channel carrying serialized JSON-RPC replies back to the single stdout writer thread.
    send: Sender<String>,
    /// Active tool calls keyed by serialized request id, paired with their cancellation flag.
    running: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    /// Shared concurrency cap across all single and batched tool calls in this session.
    limiter: Arc<ConcurrencyLimiter>,
    workers: Vec<JoinHandle<()>>,
}

impl Server {
    /// Creates a server whose tool calls run `lidar` and forward replies to `send`.
    pub fn new(lidar: Vec<OsString>, send: Sender<String>) -> Self {
        Self {
            lidar: Arc::new(lidar),
            send,
            running: Arc::default(),
            limiter: Arc::new(ConcurrencyLimiter::new(MAX_CONCURRENT_TOOLS)),
            workers: Vec::new(),
        }
    }

    /// Cancels every still-running tool process and joins worker threads (called on client EOF).
    pub fn shutdown(mut self) {
        for cancel in self
            .running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
        {
            cancel.store(true, Ordering::Relaxed);
        }
        self.limiter.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }

    /// Handles one line of JSON-RPC input. Returns an immediate serialized reply, or `None` for
    /// notifications, responses, or async tool calls (whose replies are delivered on `send`).
    pub fn handle(&mut self, message: &str) -> Option<String> {
        let msg: Value = match serde_json::from_str(message) {
            Ok(v) => v,
            Err(e) => {
                return Some(
                    error(&Value::Null, PARSE_ERROR, &format!("not JSON: {e}")).to_string(),
                );
            }
        };
        let reply = match &msg {
            Value::Array(batch) if batch.is_empty() => {
                error(&Value::Null, INVALID_REQUEST, "empty batch")
            }
            Value::Array(batch) => {
                let (mut calls, mut ids) = (Vec::new(), HashSet::new());
                let replies: Vec<Value> = batch
                    .iter()
                    .filter_map(|m| self.handle_value(m, Some((&mut calls, &mut ids))))
                    .collect();
                if !calls.is_empty() {
                    self.spawn_batch(replies, calls);
                    return None;
                }
                if replies.is_empty() {
                    return None;
                }
                Value::Array(replies)
            }
            _ => self.handle_value(&msg, None)?,
        };
        Some(reply.to_string())
    }

    fn handle_value(
        &mut self,
        msg: &Value,
        calls: Option<(&mut Vec<ToolCall>, &mut HashSet<String>)>,
    ) -> Option<Value> {
        let Some(obj) = msg.as_object() else {
            return Some(error(
                &Value::Null,
                INVALID_REQUEST,
                "expected a JSON-RPC object",
            ));
        };
        let raw_id = obj.get("id").cloned();
        let reply_id = raw_id
            .as_ref()
            .filter(|id| matches!(id, Value::String(_) | Value::Number(_)))
            .cloned()
            .unwrap_or(Value::Null);

        if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(error(&reply_id, INVALID_REQUEST, "jsonrpc must be \"2.0\""));
        }

        let Some(method_val) = obj.get("method") else {
            if obj.contains_key("result") || obj.contains_key("error") {
                return None;
            }
            return Some(error(&reply_id, INVALID_REQUEST, "expected a method"));
        };
        let Some(method) = method_val.as_str() else {
            return Some(error(&reply_id, INVALID_REQUEST, "method must be a string"));
        };
        if method == "notifications/cancelled" {
            let request = obj
                .get("params")
                .and_then(|p| p.get("requestId"))
                .map(Value::to_string);
            if let Some(cancel) = request.and_then(|r| {
                self.running
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get(&r)
                    .cloned()
            }) {
                cancel.store(true, Ordering::Relaxed);
                self.limiter.cv.notify_all();
            }
        }
        // Notifications (no id) produce no reply.
        let id = raw_id?;
        if !matches!(id, Value::String(_) | Value::Number(_)) {
            return Some(error(
                &Value::Null,
                INVALID_REQUEST,
                "id must be a string or a number",
            ));
        }
        let params = obj.get("params").cloned().unwrap_or(Value::Null);
        let ok_reply = |r: Value| json!({"jsonrpc": "2.0", "id": id, "result": r});
        if method == "tools/call" {
            return Some(match parse_tool_call(&params) {
                Err((code, message)) => error(&id, code, &message),
                Ok((name, args)) => {
                    if self
                        .running
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .contains_key(&id.to_string())
                    {
                        return Some(error(
                            &id,
                            INVALID_REQUEST,
                            "a request id already in use by a running call",
                        ));
                    }
                    match calls {
                        Some((calls, ids)) => {
                            if !ids.insert(id.to_string()) {
                                return Some(error(
                                    &id,
                                    INVALID_REQUEST,
                                    "a request id repeated within a batch",
                                ));
                            }
                            calls.push((id, name, args));
                        }
                        None => self.spawn(id, name, args),
                    }
                    return None;
                }
            });
        }
        Some(match self.call(method, &params) {
            Ok(r) => ok_reply(r),
            Err((code, message)) => error(&id, code, &message),
        })
    }

    fn spawn(&mut self, id: Value, name: String, args: Vec<OsString>) {
        let run = self.start((id, name, args));
        let send = self.send.clone();
        self.workers.retain(|w| !w.is_finished());
        self.workers.push(std::thread::spawn(move || {
            if let Some(reply) = run() {
                let _ = send.send(reply.to_string());
            }
        }));
    }

    fn spawn_batch(&mut self, mut replies: Vec<Value>, calls: Vec<ToolCall>) {
        let runs: Vec<_> = calls.into_iter().map(|call| self.start(call)).collect();
        let send = self.send.clone();
        self.workers.retain(|w| !w.is_finished());
        self.workers.push(std::thread::spawn(move || {
            let width = runs.len().min(MAX_CONCURRENT_TOOLS);
            let queue = Arc::new(Mutex::new(runs.into_iter()));
            let workers: Vec<_> = (0..width)
                .map(|_| {
                    let queue = queue.clone();
                    std::thread::spawn(move || {
                        let mut out = Vec::new();
                        loop {
                            let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
                            let Some(run) = next else { break };
                            out.extend(run());
                        }
                        out
                    })
                })
                .collect();
            replies.extend(
                workers
                    .into_iter()
                    .flat_map(|w| w.join().unwrap_or_default()),
            );
            if !replies.is_empty() {
                let _ = send.send(Value::Array(replies).to_string());
            }
        }));
    }

    fn start(
        &mut self,
        (id, name, args): ToolCall,
    ) -> impl FnOnce() -> Option<Value> + Send + 'static + use<> {
        let key = id.to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let previous = self
            .running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key.clone(), cancel.clone());
        debug_assert!(previous.is_none(), "a running call's id was reused");
        let (lidar, running, limiter) = (
            self.lidar.clone(),
            self.running.clone(),
            self.limiter.clone(),
        );
        move || {
            let reply = match limiter.acquire(&cancel) {
                None => None,
                Some(_permit) => {
                    if cancel.load(Ordering::Relaxed) {
                        None
                    } else {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            call_tool(&lidar, &name, args, &cancel)
                        }))
                        .map(|r| r.map(|r| json!({"jsonrpc": "2.0", "id": id, "result": r})))
                        .unwrap_or_else(|_| {
                            Some(error(
                                &id,
                                INTERNAL_ERROR,
                                "the tool call failed unexpectedly",
                            ))
                        })
                    }
                }
            };
            running
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&key);
            reply
        }
    }

    fn call(&self, method: &str, params: &Value) -> Result<Value, RpcError> {
        match method {
            "initialize" => Ok(initialize(params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_list()})),
            "prompts/list" => Ok(json!({"prompts": prompt_list()})),
            "prompts/get" => get_prompt(params),
            _ => Err((METHOD_NOT_FOUND, format!("unknown method '{method}'"))),
        }
    }
}

/// Validates and extracts `(tool_name, cli_args)` from `tools/call` parameters.
pub(crate) fn parse_tool_call(params: &Value) -> Result<(String, Vec<OsString>), RpcError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((INVALID_PARAMS, "name is required".to_string()))?;
    if !command_names().iter().any(|c| c == name) {
        return Err((INVALID_PARAMS, format!("unknown tool '{name}'")));
    }
    let args = match params.get("arguments").and_then(|a| a.get("args")) {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(OsString::from))
            .collect::<Option<Vec<_>>>()
            .ok_or((
                INVALID_PARAMS,
                "args must be an array of strings".to_string(),
            ))?,
        Some(_) => return Err((INVALID_PARAMS, "args must be an array of strings".into())),
    };
    Ok((name.to_string(), args))
}

/// Executes `lidar <name> <args>`, returning `None` if cancelled before completion.
pub(crate) fn call_tool(
    lidar: &[OsString],
    name: &str,
    args: Vec<OsString>,
    cancel: &AtomicBool,
) -> Option<Value> {
    if name == "preview" {
        return call_preview(lidar, args, cancel);
    }
    call_command(lidar, name, args, cancel)
}

/// Runs `lidar preview` and attaches the generated PNG as MCP `image` content so multimodal
/// agents can inspect the DEM/inundation directly. When `-o`/`--output` is omitted, writes into
/// a private temporary directory and removes it afterwards.
fn call_preview(lidar: &[OsString], mut args: Vec<OsString>, cancel: &AtomicBool) -> Option<Value> {
    let failed =
        |text: String| Some(json!({"content": [{"type": "text", "text": text}], "isError": true}));

    // Detect if the caller passed `-o` / `--output` (handling `--` terminator cleanly).
    let mut explicit_output: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy();
        if s == "--" {
            break;
        }
        if s == "-o" || s == "--output" {
            if let Some(val) = args.get(i + 1) {
                explicit_output = Some(PathBuf::from(val));
            }
            i += 2;
            continue;
        }
        if let Some(val) = s.strip_prefix("--output=") {
            explicit_output = Some(PathBuf::from(val));
        } else if let Some(val) = s.strip_prefix("-o")
            && !val.is_empty()
        {
            explicit_output = Some(PathBuf::from(val));
        }
        i += 1;
    }

    let temp_output = match explicit_output {
        Some(p) if p.as_os_str() == "-" => {
            return failed(
                "preview cannot send PNG bytes as tool text: pass -o FILE, or omit -o to receive image content"
                    .into(),
            );
        }
        Some(_) => None,
        None => {
            let dir = match tempfile::Builder::new().prefix("lidar-preview-").tempdir() {
                Ok(d) => d,
                Err(e) => return failed(format!("could not create temporary directory: {e}")),
            };
            let p = dir.path().join("preview.png");
            args.splice(0..0, [OsString::from("-o"), p.clone().into_os_string()]);
            Some((p, dir))
        }
    };

    let mut result = call_command(lidar, "preview", args, cancel);
    // Attach inline PNG image content only when the caller omitted -o (temporary server output);
    // explicit -o FILE saves to disk and returns the child's text confirmation only.
    if let (Some((path, _dir)), Some(res)) = (
        temp_output,
        result.as_mut().filter(|r| r["isError"] == false),
    ) {
        let png = std::fs::metadata(&path)
            .ok()
            .filter(|m| m.is_file() && m.len() > 0 && m.len() <= MAX_IMAGE)
            .and_then(|_| std::fs::read(&path).ok());
        let content = res["content"]
            .as_array_mut()
            .expect("tool results always have content array");
        match png {
            Some(bytes) => {
                content.push(json!({
                    "type": "image",
                    "data": encode_base64(&bytes),
                    "mimeType": "image/png"
                }));
            }
            None => {
                content.push(json!({
                    "type": "text",
                    "text": format!(
                        "The preview PNG exceeded {} MiB or could not be read; pass a smaller --max-dim.",
                        MAX_IMAGE >> 20
                    )
                }));
            }
        }
    }
    result
}

fn call_command(
    lidar: &[OsString],
    name: &str,
    args: Vec<OsString>,
    cancel: &AtomicBool,
) -> Option<Value> {
    let (text, failed) = match run_child(lidar, name, args, cancel) {
        Ok(None) => return None,
        Ok(Some((status, stdout, stderr))) => {
            let failed = !status.success();
            let footer = if failed {
                let code = status
                    .code()
                    .map_or_else(|| "a signal".to_string(), |c| format!("exit code {c}"));
                format!("\n(lidar {name} failed with {code})")
            } else {
                String::new()
            };
            let stream_budget = MAX_OUTPUT.saturating_sub(footer.len());
            // Preserve stderr first so diagnostics are never truncated by large stdout.
            let stderr_text = clip(
                String::from_utf8_lossy(&stderr.0).into_owned(),
                stderr.1,
                stream_budget,
            );
            let sep_len = usize::from(!stderr_text.is_empty());
            let stdout_budget = stream_budget
                .saturating_sub(stderr_text.len())
                .saturating_sub(sep_len);
            let mut text = if stdout_budget > 0 {
                clip(
                    String::from_utf8_lossy(&stdout.0).into_owned(),
                    stdout.1,
                    stdout_budget,
                )
            } else {
                String::new()
            };
            if !stderr_text.is_empty() {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&stderr_text);
            }
            if !footer.is_empty() {
                text.push_str(&footer);
            }
            if text.trim().is_empty() {
                text = format!("lidar {name} finished.");
            }
            (text, failed)
        }
        Err(e) => (format!("could not run lidar: {e}"), true),
    };
    Some(json!({"content": [{"type": "text", "text": text}], "isError": failed}))
}

type BoundedOutput = (Vec<u8>, usize);

/// Ensures a spawned child process is killed and reaped on any early return or panic.
struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn as_mut(&mut self) -> &mut Child {
        self.0.as_mut().expect("child still owned")
    }

    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run_child(
    lidar: &[OsString],
    name: &str,
    args: Vec<OsString>,
    cancel: &AtomicBool,
) -> std::io::Result<Option<(ExitStatus, BoundedOutput, BoundedOutput)>> {
    let (program, before) = lidar
        .split_first()
        .expect("lidar command line must contain a program");
    let child = Command::new(program)
        .args(before)
        .arg(name)
        .args(args)
        .env("LIDAR_MCP_TOOL", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut guard = ChildGuard(Some(child));
    let stdout = read_bounded(guard.as_mut().stdout.take().expect("piped stdout"))?;
    let stderr = read_bounded(guard.as_mut().stderr.take().expect("piped stderr"))?;
    let status = loop {
        if let Some(status) = guard.as_mut().try_wait()? {
            guard.disarm();
            break status;
        }
        if cancel.load(Ordering::Relaxed) {
            drop(guard);
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let joined = |h: JoinHandle<BoundedOutput>| h.join().unwrap_or_default();
    Ok(Some((status, joined(stdout), joined(stderr))))
}

fn read_bounded(mut r: impl Read + Send + 'static) -> std::io::Result<JoinHandle<BoundedOutput>> {
    std::thread::Builder::new().spawn(move || {
        let (mut kept, mut total, mut buf) = (Vec::new(), 0usize, [0u8; 8192]);
        loop {
            let n = match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            let room = MAX_OUTPUT.saturating_sub(kept.len());
            kept.extend_from_slice(&buf[..n.min(room)]);
            total += n;
        }
        (kept, total)
    })
}

/// Serves MCP over stdin and stdout (one JSON-RPC message per line) until EOF.
pub fn serve(lidar: Vec<OsString>) -> std::io::Result<()> {
    let (send, replies) = std::sync::mpsc::channel::<String>();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        let mut out = std::io::stdout().lock();
        for reply in replies {
            writeln!(out, "{reply}")?;
            out.flush()?;
        }
        Ok(())
    });
    let mut server = Server::new(lidar, send.clone());
    let read = (|| {
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Some(reply) = server.handle(&line) {
                let _ = send.send(reply);
            }
        }
        Ok(())
    })();
    server.shutdown();
    drop(send);
    writer.join().unwrap_or(Ok(())).and(read)
}

/// Returns the command line that re-invokes this `lidar` binary.
pub fn this_binary() -> std::io::Result<Vec<OsString>> {
    Ok(vec![std::env::current_exe()?.into_os_string()])
}

fn initialize(params: &Value) -> Value {
    let asked = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or("");
    let version = VERSIONS
        .iter()
        .find(|v| **v == asked)
        .copied()
        .unwrap_or(VERSIONS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": {
            "tools": {"listChanged": false},
            "prompts": {"listChanged": false}
        },
        "serverInfo": {
            "name": "lidar-rs",
            "title": "lidar-rs",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": INSTRUCTIONS,
    })
}

fn tool_list() -> Vec<Value> {
    static TOOLS: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
    TOOLS
        .get_or_init(|| {
            with_clap_stack(|| {
                commands()
                    .map(|mut c| {
                        let name = c.get_name().to_string();
                        let ro = READ_ONLY.contains(&name.as_str());
                        let open_world = OPEN_WORLD.contains(&name.as_str());
                        let description =
                            c.render_long_help().to_string().trim_end().to_string();
                        json!({
                            "name": name,
                            "description": description,
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "args": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": format!("The command line after `lidar {name}`, one argument per item"),
                                    }
                                },
                                "required": ["args"],
                            },
                            "annotations": {
                                "readOnlyHint": ro,
                                "destructiveHint": !ro,
                                "idempotentHint": ro,
                                "openWorldHint": open_world,
                            },
                        })
                    })
                    .collect()
            })
        })
        .clone()
}

fn prompt_list() -> Vec<Value> {
    let mut prompts = vec![json!({
        "name": "delineate-depressions",
        "description": "Inspect a DEM, choose resolution-appropriate thresholds, run the end-to-end nested depression and inundation pipeline, and preview the results.",
        "arguments": [
            {"name": "dem_path", "description": "Absolute path to the input DEM GeoTIFF", "required": true},
            {"name": "out_dir", "description": "Absolute path to the output directory", "required": true}
        ]
    })];
    #[cfg(feature = "download")]
    prompts.push(json!({
        "name": "watershed-fsm-study",
        "description": "Download authoritative U.S. federal datasets (USGS 3DEP DEM, WBD, NHD, NWI, Transportation, synthesized culverts, NOAA Atlas 14) for a watershed and run a hydro-conditioned Fill-Spill-Merge storm simulation.",
        "arguments": [
            {"name": "region", "description": "Region selector: a USGS HUC code (e.g. 101600020102), WGS 84 bbox 'min_lon,min_lat,max_lon,max_lat', or polygon file path", "required": true},
            {"name": "out_dir", "description": "Absolute path to the study output directory", "required": true},
            {"name": "dem_product", "description": "Optional USGS 3DEP DEM product (1m, 10m, or 30m; default 1m)", "required": false}
        ]
    }));
    prompts.push(json!({
        "name": "pointcloud-terrain-products",
        "description": "Inspect a LAS/LAZ/COPC/PLY/E57 LiDAR point cloud, generate bare-earth DTM, DSM, and Canopy Height Model (CHM), and preview the output rasters.",
        "arguments": [
            {"name": "pointcloud_path", "description": "Absolute path to the input LiDAR point cloud", "required": true},
            {"name": "out_dir", "description": "Absolute path to the output directory", "required": true},
            {"name": "resolution", "description": "Optional grid resolution in CRS units (default 1.0)", "required": false}
        ]
    }));
    prompts
}

#[cfg(feature = "download")]
fn download_region_flag(region: &str) -> &'static str {
    let trimmed = region.trim();
    if matches!(trimmed.len(), 2 | 4 | 6 | 8 | 10 | 12)
        && trimmed.chars().all(|c| c.is_ascii_digit())
    {
        "--huc"
    } else if trimmed.split(',').count() == 4
        && trimmed
            .split(',')
            .all(|part| part.trim().parse::<f64>().is_ok())
    {
        "--bbox"
    } else {
        "--region"
    }
}

fn get_prompt(params: &Value) -> Result<Value, RpcError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((INVALID_PARAMS, "prompt name is required".to_string()))?;
    let args = params.get("arguments").and_then(Value::as_object);
    let req_arg = |key: &str| -> Result<String, RpcError> {
        args.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(ToString::to_string)
            .ok_or((
                INVALID_PARAMS,
                format!("missing required prompt argument '{key}'"),
            ))
    };
    let opt_arg = |key: &str, default: &str| -> String {
        args.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(default)
            .to_string()
    };

    match name {
        "delineate-depressions" => {
            let dem = req_arg("dem_path")?;
            let out = req_arg("out_dir")?;
            Ok(json!({
                "description": "Inspect a DEM and run the lidar-rs nested depression and inundation pipeline.",
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": format!(
                            "Analyze surface depressions and inundation for DEM `{dem}` and write outputs to `{out}`:\n\
                             1. Call `info` with `[\"{dem}\", \"--json\", \"--stats\"]` to inspect CRS, pixel size, dimensions, NoData, and elevation range.\n\
                             2. Choose physically appropriate `--min-size`, `--min-depth`, and `--interval` values based on pixel resolution and terrain relief.\n\
                             3. Run `pipeline` with `[\"{dem}\", \"{out}\", \"--routing\", \"fsm\", \"--compress\", \"--json\"]` plus your chosen thresholds.\n\
                             4. Visually inspect the result with `preview` on `{dem}` overlaid with `--fill-time {out}/fill_time.tif --scale-bar --legend` (omit `-o` to view inline).\n\
                             5. Summarize depression count, maximum nesting level, total storage volume, spill path count, and FSM mass balance."
                        )
                    }
                }]
            }))
        }
        #[cfg(feature = "download")]
        "watershed-fsm-study" => {
            let region = req_arg("region")?;
            let out = req_arg("out_dir")?;
            let product = opt_arg("dem_product", "1m");
            let flag = download_region_flag(&region);
            Ok(json!({
                "description": "Download federal watershed data and run hydro-conditioned Fill-Spill-Merge modeling.",
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": format!(
                            "Run an end-to-end watershed Fill-Spill-Merge study for region `{region}` in `{out}` using USGS 3DEP `{product}`:\n\
                             1. Call `download` with `[\"{flag}\", \"{region}\", \"-o\", \"{out}/federal\", \"--dem-product\", \"{product}\", \"--dry-run\"]` first to check tile count and download size (use `--huc` for HUC codes, `--bbox` for WGS 84 bounds, or `--region` for polygon files).\n\
                             2. Run `download` with `[\"{flag}\", \"{region}\", \"-o\", \"{out}/federal\", \"--dem-product\", \"{product}\", \"--datasets\", \"all\", \"--json\"]` to obtain 3DEP DEM, WBD, NHD, NWI, roads/railways, synthesized culverts, and NOAA Atlas 14 design storms.\n\
                             3. Inspect `{out}/federal/dem.tif` with `info` (`--json`). If it is in a geographic CRS (e.g., 10m/30m EPSG:4269), reproject it with `reproject` to match `{out}/federal/culverts.gpkg`.\n\
                             4. Extract the 100-year 24-hour depth from `{out}/federal/noaa_atlas14.json` and run `pipeline` with `--routing fsm --culverts {out}/federal/culverts.gpkg --streams {out}/federal/nhd_flowlines.gpkg --scs-type2-mm <DEPTH> --compress --json` into `{out}/fsm_100yr`.\n\
                             5. Call `preview` with `--fill-time {out}/fsm_100yr/fill_time.tif --scale-bar --legend` and summarize watershed storage retention and connectivity."
                        )
                    }
                }]
            }))
        }
        "pointcloud-terrain-products" => {
            let pc = req_arg("pointcloud_path")?;
            let out = req_arg("out_dir")?;
            let res = opt_arg("resolution", "1.0");
            Ok(json!({
                "description": "Generate bare-earth DTM, DSM, and Canopy Height Model from a LiDAR point cloud.",
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": format!(
                            "Process LiDAR point cloud `{pc}` into `{out}` at resolution `{res}`:\n\
                             1. Call `info` with `[\"{pc}\", \"--json\"]` to check CRS, point count, bounds, and existing ASPRS classifications.\n\
                             2. Generate bare-earth DTM (`lidar-to-dtm`), DSM (`lidar-to-dsm`), and CHM (`lidar-to-chm`) in `{out}` at `--resolution {res}`.\n\
                             3. Verify each output raster with `info --json --stats` and visually inspect the DTM with `preview` (`--scale-bar`)."
                        )
                    }
                }]
            }))
        }
        _ => Err((INVALID_PARAMS, format!("unknown prompt '{name}'"))),
    }
}

/// Pure-Rust RFC 4648 base64 encoder (zero extra dependencies).
pub(crate) fn encode_base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn clip(mut text: String, total: usize, max: usize) -> String {
    if text.len() <= max && total <= text.len() {
        return text;
    }
    let full_total = total.max(text.len());
    let suffix_for = |kept: usize| format!("\n... (output cut at {kept} of {full_total} bytes)");
    let max_suffix_len = suffix_for(max).len();
    if max <= max_suffix_len {
        let mut end = max.min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        return text;
    }
    let mut end = (max - max_suffix_len).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    let suffix = suffix_for(end);
    text.push_str(&suffix);
    text
}

fn error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server() -> Server {
        Server::new(vec!["lidar".into()], std::sync::mpsc::channel().0)
    }

    fn request(s: &mut Server, method: &str, params: Value) -> Value {
        let msg =
            json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).to_string();
        serde_json::from_str(&s.handle(&msg).expect("a reply")).unwrap()
    }

    #[test]
    fn initializes_with_supported_version_and_capabilities() {
        let r = request(
            &mut server(),
            "initialize",
            json!({"protocolVersion": "2025-03-26"}),
        );
        assert_eq!(r["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(r["result"]["serverInfo"]["name"], "lidar-rs");
        assert!(r["result"]["capabilities"]["tools"].is_object());
        assert!(r["result"]["capabilities"]["prompts"].is_object());
        assert!(
            r["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("pipeline")
        );

        let r = request(
            &mut server(),
            "initialize",
            json!({"protocolVersion": "1999-01-01"}),
        );
        assert_eq!(r["result"]["protocolVersion"], VERSIONS[0]);
    }

    #[test]
    fn lists_cli_commands_with_accurate_annotations() {
        let r = request(&mut server(), "tools/list", json!({}));
        let tools = r["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for expected in [
            "info",
            "preview",
            "pipeline",
            "extract-sinks",
            "delineate-depressions",
            "breach-depressions",
            "burn-culverts",
            "synthesize-culverts",
            "lidar-to-dtm",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
        assert!(!names.contains(&"mcp"));
        assert!(!names.contains(&"help"));

        let info = tools.iter().find(|t| t["name"] == "info").unwrap();
        assert_eq!(info["annotations"]["readOnlyHint"], true);
        assert_eq!(info["annotations"]["destructiveHint"], false);
        assert_eq!(info["annotations"]["openWorldHint"], false);
        assert!(info["description"].as_str().unwrap().contains("--json"));

        let pipe = tools.iter().find(|t| t["name"] == "pipeline").unwrap();
        assert_eq!(pipe["annotations"]["readOnlyHint"], false);
        assert_eq!(pipe["annotations"]["openWorldHint"], false);

        #[cfg(feature = "download")]
        {
            let dl = tools.iter().find(|t| t["name"] == "download").unwrap();
            assert_eq!(dl["annotations"]["openWorldHint"], true);
        }
    }

    #[test]
    fn lists_and_renders_workflow_prompts() {
        let s = &mut server();
        let list = request(s, "prompts/list", json!({}));
        let prompts = list["result"]["prompts"].as_array().unwrap();
        #[cfg(feature = "download")]
        assert_eq!(prompts.len(), 3);
        #[cfg(not(feature = "download"))]
        {
            assert_eq!(prompts.len(), 2);
            assert!(!prompts.iter().any(|p| p["name"] == "watershed-fsm-study"));
        }

        let got = request(
            s,
            "prompts/get",
            json!({
                "name": "delineate-depressions",
                "arguments": {"dem_path": "/data/dem.tif", "out_dir": "/data/out"}
            }),
        );
        let text = got["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("/data/dem.tif") && text.contains("preview"));

        #[cfg(feature = "download")]
        {
            let huc = request(
                s,
                "prompts/get",
                json!({
                    "name": "watershed-fsm-study",
                    "arguments": {"region": "101600020102", "out_dir": "/data/study"}
                }),
            );
            assert!(
                huc["result"]["messages"][0]["content"]["text"]
                    .as_str()
                    .unwrap()
                    .contains("\"--huc\"")
            );
        }

        let missing = request(
            s,
            "prompts/get",
            json!({"name": "delineate-depressions", "arguments": {}}),
        );
        assert_eq!(missing["error"]["code"], INVALID_PARAMS);
    }

    #[test]
    fn rejects_invalid_requests_and_ignores_notifications() {
        let s = &mut server();
        assert_eq!(
            request(
                s,
                "tools/call",
                json!({"name": "mcp", "arguments": {"args": []}})
            )["error"]["code"],
            INVALID_PARAMS
        );
        assert_eq!(
            request(
                s,
                "tools/call",
                json!({"name": "info", "arguments": {"args": [1]}})
            )["error"]["code"],
            INVALID_PARAMS
        );
        assert_eq!(
            request(s, "nope", json!({}))["error"]["code"],
            METHOD_NOT_FOUND
        );
        assert_eq!(
            serde_json::from_str::<Value>(&s.handle("{").unwrap()).unwrap()["error"]["code"],
            PARSE_ERROR
        );
        // Invalid JSON-RPC envelopes (empty object, wrong version) return INVALID_REQUEST.
        assert_eq!(
            serde_json::from_str::<Value>(&s.handle("{}").unwrap()).unwrap()["error"]["code"],
            INVALID_REQUEST
        );
        assert_eq!(
            serde_json::from_str::<Value>(
                &s.handle(r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#)
                    .unwrap()
            )
            .unwrap()["error"]["code"],
            INVALID_REQUEST
        );
        assert!(
            s.handle(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .is_none()
        );
        assert!(
            s.handle(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#)
                .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn answers_while_running_and_honors_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("started");
        let script = format!("touch \"{}\"; exec sleep 30", marker.display());
        let (send, replies) = std::sync::mpsc::channel();
        let mut s = Server::new(vec!["sh".into(), "-c".into(), script.into()], send);
        let call = json!({
            "jsonrpc": "2.0",
            "id": "slow",
            "method": "tools/call",
            "params": {"name": "info", "arguments": {"args": []}}
        });
        let started = std::time::Instant::now();
        assert!(s.handle(&call.to_string()).is_none());
        // Wait until the child actually starts before testing cancellation.
        while !marker.exists() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "child never started"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(request(&mut s, "ping", json!({}))["result"].is_object());
        // Duplicate running id must be refused.
        let dup: Value =
            serde_json::from_str(&s.handle(&call.to_string()).expect("refusal")).unwrap();
        assert_eq!(dup["error"]["code"], INVALID_REQUEST);

        s.handle(
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "slow"}
            })
            .to_string(),
        );
        // Verify notifications/cancelled itself terminates and clears the call BEFORE shutdown().
        while !s.running.lock().unwrap().is_empty() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "notifications/cancelled did not stop running call"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        s.shutdown();
        assert!(replies.try_recv().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn batch_runs_calls_and_replies_together() {
        let (send, replies) = std::sync::mpsc::channel();
        let mut s = Server::new(vec!["sh".into(), "-c".into(), "echo done".into()], send);
        let batch = json!([
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "info", "arguments": {"args": []}}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ]);
        assert!(s.handle(&batch.to_string()).is_none());
        let reply: Value =
            serde_json::from_str(&replies.recv_timeout(Duration::from_secs(30)).unwrap()).unwrap();
        s.shutdown();
        let arr = reply.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(arr.iter().any(|r| r["id"] == 2 && r["result"].is_object()));
        let call = arr.iter().find(|r| r["id"] == 1).unwrap();
        assert!(
            call["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("done")
        );
    }

    #[test]
    fn panicking_call_returns_internal_error() {
        let (send, replies) = std::sync::mpsc::channel();
        let mut s = Server::new(Vec::new(), send);
        let call = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "info", "arguments": {"args": []}}
        });
        assert!(s.handle(&call.to_string()).is_none());
        let reply: Value =
            serde_json::from_str(&replies.recv_timeout(Duration::from_secs(10)).unwrap()).unwrap();
        assert_eq!(reply["error"]["code"], INTERNAL_ERROR);
        assert!(s.running.lock().unwrap().is_empty());
        s.shutdown();
    }

    #[test]
    fn base64_and_clip_match_rfc4648_and_utf8_boundaries() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");

        let long = "é".repeat(MAX_OUTPUT);
        let clipped = clip(long.clone(), long.len(), MAX_OUTPUT);
        assert!(clipped.len() <= MAX_OUTPUT);
        assert!(clipped.ends_with(&format!("of {} bytes)", long.len())));
        assert_eq!(clip("abc".into(), 3, 1), "a");
    }
}
