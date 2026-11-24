# lidar-rs

[![CI](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lidar.svg)](https://crates.io/crates/lidar)
[![PyPI](https://img.shields.io/pypi/v/lidar-rs)](https://pypi.org/project/lidar-rs/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb)
[![Docs](https://img.shields.io/badge/docs-lidar--rs.opengeos.org-teal.svg)](https://lidar-rs.opengeos.org)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24411444-blue.svg)](https://doi.org/10.5281/zenodo.24411444)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance nested surface depression delineation, hydro-conditioning, LiDAR point-cloud gridding, and fill-and-spill inundation simulation in **pure Rust**, ported from [`opengeos/lidar`](https://github.com/opengeos/lidar) and powered by [`terrio`](https://github.com/opengeos/terrio) and [`richdem`](https://github.com/opengeos/richdem-rs) with **zero GDAL or C/C++ dependencies**.

**Documentation:** [User Guides](https://lidar-rs.opengeos.org/guides/) | [Sample Datasets](https://lidar-rs.opengeos.org/datasets/) | [CLI](https://lidar-rs.opengeos.org/cli/) | [AI Agents (MCP)](https://lidar-rs.opengeos.org/mcp/) | [Rust](https://lidar-rs.opengeos.org/rust/) | [Python](https://lidar-rs.opengeos.org/python/) | [JavaScript / WASM](https://lidar-rs.opengeos.org/javascript/) | [Live Playground](https://lidar-rs.opengeos.org/playground/)

## Demo

[![Nested depression hierarchy and depression tree overview in lidar-rs](https://files.opengeos.org/depression_hierarchy.webp)](https://lidar-rs.opengeos.org/playground/)

[![Interactive inundation simulation and inundation curve in the lidar-rs WebAssembly Playground](https://files.opengeos.org/inundation_demo.gif)](https://lidar-rs.opengeos.org/playground/)

[![Progressive depression tree graph building bottom-up with rising water layers in the lidar-rs WebAssembly Playground](https://files.opengeos.org/depression_tree.gif)](https://lidar-rs.opengeos.org/playground/)

## Highlights

- **All-in-one in-memory pipeline (`run_pipeline` / `lidar pipeline`):** Run DEM smoothing, optional hydro-conditioning, Priority-Flood sink extraction, nested depression hierarchy, single-pass D8 catchment hierarchy, inundation simulation, rating curves, connectivity graph export, and unified multi-layer GeoPackage (`results.gpkg`) output in a single command.
- **Compressed D8 catchment engine:** Single-pass upland D8 forest compression plus per-level Union-Find (`O(N + L*D)`). In an illustrative 98M-cell, 43-level 1-m USGS 3DEP karst run it completed in ~65 seconds (see the [User Guides](https://lidar-rs.opengeos.org/guides/)).
- **Hydro-conditioning (`breach_depressions`, `burn_culverts`):** Pure-Rust Lindsay (2016) least-cost depression breaching (`complete`, `selective`, `constrained`) and vector-guided culvert burning across road embankments.
- **Inundation physics, storm forcing & Fill-Spill-Merge routing:** Dynamic event-driven Fill-Spill-Merge cascade mass-balance routing (`--routing fsm` / `RoutingMode::FillSpillMerge`), catchment-weighted runoff coefficients, NRCS TR-55 SCS Curve Number conversion (`scs_curve_number_runoff`), SCS Type II design storm hyetographs, vertical infiltration losses, and antecedent ponded water depths.
- **Rating curves & connectivity graphs:** Exact stage-storage-area rating curves (`rating_curves.csv`), directed fill-and-spill / merge networks (JSON + GraphML), georeferenced spill-path vectors (`spill_paths.gpkg`), and NHD stream-proximity metrics.
- **DEM smoothing, sink extraction & nested hierarchies:** Mean, median, and Gaussian filters; Priority-Flood sink extraction (`extract_sinks`); Wu et al. (2019) nested depression hierarchy (`delineate_depressions`, 18+ morphometric and topological attributes); and nested mount / canopy hierarchy (`delineate_mounts`, `flip_dem`).
- **LiDAR point clouds (`DSM`, `DTM`, `CHM`):** Native LAS/LAZ/COPC/PLY/E57 reading and conversion, ASPRS class and return filtering, pure-Rust CSF/PMF ground classification (`classify_ground`), bare-earth DTM (`lidar_to_dtm`), and Canopy Height Models (`lidar_to_chm`).
- **Tiled regional delineation & `f32` storage (`delineate_depressions_tiled` / `tiled-depressions`):** Parallel overlapping tile windows with core-owned seam deduplication and optional `f32` DEM grids that halve source raster memory.
- **Publication PNG & GIF rendering (`export_frames`, `animate_inundation`):** Shaded-relief inundation animations and individual PNG frames with overland spill arrows, metric scale bars, legends, and timestamp HUDs.
- **Federal geospatial data download (`download_federal_data` / `lidar download`):** Pure-Rust download utilities for authoritative open U.S. federal datasets within a WGS 84 bounding box, USGS HUC code (HUC-2 through HUC-12), or local polygon file: USGS 3DEP DEMs (1-m, 10-m, 30-m) and LiDAR Point Clouds (with newest-vintage tile deduplication, caching, mosaicking, and clipping), USGS WBD watershed boundaries, USGS NHD flowlines, USFWS NWI wetlands, USGS National Transportation roads and railways, deterministic road-stream culvert synthesis (`synthesize_culverts`), and NOAA Atlas 14 precipitation depth frequency tables.
- **AI Agents & MCP server (`lidar mcp`):** Pure-Rust Model Context Protocol (MCP) server over stdio for Claude Code, Claude Desktop, Cursor, and VS Code (`claude mcp add lidar -- lidar mcp` or zero-install via `uvx --from lidar-rs lidar mcp`), exposing CLI subcommands (excluding `mcp` and `help`; `download` is included when built with `--features download`), inline PNG terrain/inundation `preview` images, structured `info --json --stats`, and guided hydrology workflow prompts (plus the CLI `--json-usage` catalog flag).
- **Four interfaces, one pure-Rust core:** Rust crate (`lidar`), standalone CLI & MCP server (`lidar-cli` with JSON output), Python package (`lidar-rs`, PyO3 + zero-copy NumPy arrays + bundled `lidar` CLI, Python 3.11+), and WebAssembly (`lidar-js`) powering the interactive browser [Playground](https://lidar-rs.opengeos.org/playground/) with Depression Inspector and live Stage-Volume Rating Curves.

## Workspace Layout

| Path | Description |
|---|---|
| `crates/lidar` | Pure-Rust core library (published to crates.io as `lidar`) |
| `crates/lidar-cli` | Standalone `lidar` command-line tool (`lidar-cli`) |
| `python` | Python bindings (`lidar-rs`, PyO3 + NumPy) with drop-in `opengeos/lidar` API |
| `js` | WebAssembly (`wasm-bindgen`) bindings (`lidar-js`) for browsers and Node.js |
| `test_data` | Sample DEM/DSM/sink GeoTIFF datasets from `opengeos/lidar` |
| `examples` | Interactive Google Colab demo notebooks (`lidar_rs_demo.ipynb`, `inundation_demo.ipynb`, `federal_data_download.ipynb`) |
| `docs` | Zensical documentation site, interactive WASM Playground, and CLI installer scripts (`install.sh`, `install.ps1`) |

## Rust Usage

Add `lidar` to your `Cargo.toml`:

```toml
[dependencies]
lidar = "0.5"
```

```rust,no_run
use lidar::{RunPipelineOptions, SimulateInundationOptions, run_pipeline};

fn main() -> lidar::Result<()> {
    let res = run_pipeline(
        "dem.tif",
        "out_pipe",
        &RunPipelineOptions {
            kernel_size: 3,
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 50.0,
                time_step_h: 1.0,
                compress: true,
            },
            ..Default::default()
        },
    )?;

    println!(
        "Delineated {} nested depressions and {} spill paths",
        res.num_depressions, res.num_spill_paths,
    );
    Ok(())
}
```

## Command-Line Interface

Install the standalone `lidar` CLI:

```sh
# Linux and macOS
curl -LsSf https://lidar-rs.opengeos.org/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://lidar-rs.opengeos.org/install.ps1 | iex"

# Or from crates.io (add `--features download` for `lidar download`)
cargo install lidar-cli --features download
```

Example commands:

```sh
# End-to-end pipeline (GeoPackage, CSV, rating curves, connectivity graphs, rasters)
lidar pipeline dem.tif out_pipe --min-size 1000 --min-depth 0.3 --interval 0.3 \
  --rain-intensity 5 --duration 50 --time-step 1 --compress

# Hydro-conditioning (Lindsay 2016 breaching + culvert burning)
lidar breach-depressions dem.tif out/breached.tif --mode selective --max-depth 0.5
lidar burn-culverts dem.tif culverts.gpkg out/burned.tif --half-width 1 --depth-offset 0.1

# Publication PNG frames & shaded-relief animated GIF
lidar export-frames dem.tif out_pipe/fill_time.tif out_pipe/frames --num-steps 10 --time-step 5 --rain-intensity 5 --scale-bar --legend
lidar animate-inundation dem.tif out/simulation out/inundation.gif --rain-intensity 5 --scale-bar --legend

# LiDAR point clouds (CSF/PMF ground classification, DSM, bare-earth DTM, CHM)
lidar lidar-to-dsm cloud.laz dsm.tif --resolution 1.0
lidar lidar-to-dtm cloud.laz dtm.tif --resolution 1.0 --ground-method csf
lidar lidar-to-chm cloud.laz chm.tif --resolution 1.0 --ground-method csf

# Parallel tiled regional delineation
lidar tiled-depressions regional_dem.tif out_tiled --tile-size 2048 --overlap 128 --min-size 1000
```

## Python Usage

```sh
pip install lidar-rs
```

Or try it interactively in [Google Colab](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb).

```python
import lidar_rs as lidar

# End-to-end pipeline (GeoPackage, CSV, rating curves, GraphML/JSON, rasters)
pipe = lidar.run_pipeline(
    "dem.tif",
    "out_pipe",
    min_size=1000,
    min_depth=0.3,
    interval=0.3,
    rain_intensity_cm_h=5.0,
    rain_duration_h=50.0,
    time_step_h=1.0,
    compress=True,
)
print(pipe["num_depressions"], pipe["num_spill_paths"])

# Classic opengeos/lidar drop-in workflow
med = lidar.MedianFilter("dem.tif", kernel_size=3, out_file="out/median.tif")
sink = lidar.ExtractSinks(med, min_size=1000, out_dir="out")
dep_id, dep_level = lidar.DelineateDepressions(
    sink, min_size=1000, min_depth=0.3, interval=0.3, out_dir="out", bool_level_shp=True
)
```

## Performance

Measured in release mode (`cargo run --release --example benchmark`) on the 400 × 400 sample datasets (`test_data/dem.tif`, `sink.tif`, `dsm.tif`, 160,000 cells):

| Operation | Median Time |
|---|---:|
| `mean_filter` (3×3, in-memory) | 0.75 ms |
| `median_filter` (3×3, in-memory) | 1.27 ms |
| `gaussian_filter` (σ = 1.0, in-memory) | 1.51 ms |
| `extract_sinks` (in-memory, `min_size = 1000`) | 8.25 ms |
| `extract_sinks` (full file I/O: GeoTIFFs + CSV + Shapefile + GeoPackage) | 17.68 ms |
| `delineate_depressions` (in-memory, `interval = 0.3`) | 46.77 ms |
| `delineate_depressions` (full file I/O + per-level shapefiles) | 65.64 ms |
| `delineate_mounts` (`dsm.tif`, full pipeline + file I/O) | 78.39 ms |

## Citations

- **Wu, Q.** (2026). lidar-rs: High-performance nested depression delineation and fill-and-spill inundation simulation. *Zenodo*. <https://doi.org/10.5281/zenodo.24411444>
- **Wu, Q.** (2021). lidar: A Python package for delineating nested surface depressions from digital elevation data. *Journal of Open Source Software*, 6(59), 2965. <https://doi.org/10.21105/joss.02965>
- **Wu, Q.**, Lane, C.R., Wang, L., Vanderhoof, M.K., Christensen, J.R., & Liu, H. (2019). Efficient Delineation of Nested Depression Hierarchy in Digital Elevation Models for Hydrological Analysis Using Level-Set Method. *Journal of the American Water Resources Association*, 55(2), 354–368. <https://doi.org/10.1111/1752-1688.12689>

## License

`lidar-rs` source code is licensed under the [MIT License](LICENSE). See [NOTICE](NOTICE) for third-party crate license attributions (including `richdem` under GPL-3.0-only when linked in distributed binaries).
