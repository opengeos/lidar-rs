# AI Agents (MCP)

`lidar mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server: an AI agent such as Claude Code, Claude Desktop, Cursor, or VS Code Copilot can inspect rasters and LiDAR point clouds, delineate nested surface depressions, run dynamic Fill-Spill-Merge inundation simulations, download open U.S. federal watershed datasets, and visually preview terrain and flooding on your machine.

Every [CLI subcommand](cli.md) except `mcp`, `completions`, and `help` is exposed automatically as an MCP tool whose `args` parameter is the command line after the command name and whose description is the command's full `--help`. The server speaks JSON-RPC 2.0 over standard input and output and runs each tool call in an isolated `lidar` child process.

## Setup

Install the `lidar` CLI (prebuilt binaries from [Installation](installation.md) include `lidar download` out of the box), or run directly through Python / `uvx` without installing the standalone binary.

### Claude Code

```sh
# Using the standalone lidar binary
claude mcp add lidar -- lidar mcp

# Or zero-install via uvx (includes federal data download)
claude mcp add lidar -- uvx --from lidar-rs lidar mcp
```

### Claude Desktop and Cursor

Add to `claude_desktop_config.json` or `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "lidar": {
      "command": "lidar",
      "args": ["mcp"]
    }
  }
}
```

Or with `uvx`:

```json
{
  "mcpServers": {
    "lidar": {
      "command": "uvx",
      "args": ["--from", "lidar-rs", "lidar", "mcp"]
    }
  }
}
```

### VS Code

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "lidar": {
      "command": "lidar",
      "args": ["mcp"]
    }
  }
}
```

## Agent-Centric Capabilities

- **Structured dataset inspection (`info`):** Pass `["/abs/path/dem.tif", "--json", "--stats"]` to inspect CRS, pixel resolution, dimensions, NoData, band statistics, vector layers, or point-cloud bounds before choosing modeling thresholds.
- **Single-call hydrology pipeline (`pipeline`):** Run smoothing, optional breaching and culvert burning, sink extraction, Wu et al. (2019) nested depression hierarchy, catchment hierarchy, static or dynamic Fill-Spill-Merge (`--routing fsm`) inundation simulation, rating curves, and connectivity graphs in one call with `--json`.
- **Visual inspection (`preview`):** Renders a shaded-relief hillshade PNG of any DEM/DSM, optionally overlaid with a depression depth raster (`--depth`) or inundation snapshot (`--fill-time`, `--hour`, `--scale-bar`, `--legend`). When `-o` is omitted, `lidar mcp` writes to a private temporary directory, attaches the PNG directly as MCP `image` content so multimodal agents can see the map, and removes the temporary file.
- **Federal watershed data download (`download`):** Available when built with `--features download` (enabled in prebuilt CLI releases and `lidar-rs` Python wheels). Agents can run `download --dry-run` first to check USGS 3DEP tile counts and sizes before downloading WBD, 3DEP DEM/LPC, NHD, NWI, USGS Transportation, synthesized culverts, and NOAA Atlas 14 design storms.
- **Built-in MCP prompts (`prompts/list`, `prompts/get`):** Includes three guided multi-step workflow prompts:
  - `delineate-depressions` (`dem_path`, `out_dir`)
  - `watershed-fsm-study` (`region`, `out_dir`, optional `dem_product`)
  - `pointcloud-terrain-products` (`pointcloud_path`, `out_dir`, optional `resolution`)
- **Machine-readable command catalog (`--json-usage`):** Run `lidar --json-usage` or `lidar pipeline --json-usage` to inspect the full JSON schema of subcommands, flags, types, and defaults.

## Permissions and Safety

By default, `lidar mcp` does not restrict filesystem paths beyond OS permissions. Pass `--allow-dir <DIR>` (repeatable) to cooperatively restrict tool path arguments to canonicalized paths inside the allowed directories (which also rejects remote URLs/VFS schemes and `pipeline --config`, which is always disabled inside MCP subprocesses). Pair `--allow-dir` with OS sandboxing (such as Landlock or containers) when kernel-enforced isolation is required.

- `info` is annotated `readOnlyHint: true`, `destructiveHint: false`, `idempotentHint: true`, `openWorldHint: false`.
- `download` and `noaa-atlas14` (when built with `--features download`) are annotated `openWorldHint: true` because they contact USGS, USFWS, and NOAA web services.
- All other local processing tools are annotated `openWorldHint: false` and `destructiveHint: true` because they write files and can overwrite existing outputs.
- Annotations are client hints, not enforcement.

## Operational Limits

- Always pass absolute file paths; relative paths resolve against the working directory where the MCP client spawned the server.
- Combined command output (`stderr` + `stdout`, including any truncation notices and failure status) is capped at 100,000 bytes (`stderr` diagnostics are preserved first so failure messages are not truncated).
- Inline `preview` PNG attachments are returned only when `-o`/`--output` is omitted (capped at 8 MiB; use `--max-dim`, default `1024`, to keep images compact). Passing `-o FILE` writes the PNG to disk and returns the command's text output only.
- Tool calls execute asynchronously in isolated child processes with a session-wide concurrency limit of 8 simultaneous tool runs (covering both standalone calls and JSON-RPC batches), so `ping` and concurrent requests stay responsive without exhausting memory.
- Client cancellations (`notifications/cancelled`) and EOF on standard input immediately terminate running child processes, so clients should keep stdin open until all expected replies arrive. JSON-RPC batches return all replies together once their tool calls finish.
