# lidar-rs

[![CI](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lidar.svg)](https://crates.io/crates/lidar)
[![PyPI](https://img.shields.io/pypi/v/lidar-rs)](https://pypi.org/project/lidar-rs/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24411444-blue.svg)](https://doi.org/10.5281/zenodo.24411444)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/opengeos/lidar-rs/blob/main/LICENSE)

High-performance nested surface depression delineation, hydro-conditioning, LiDAR point-cloud gridding, and fill-and-spill inundation simulation in **pure Rust**, ported from [`opengeos/lidar`](https://github.com/opengeos/lidar) and powered by [`terrio`](https://github.com/opengeos/terrio) and [`richdem`](https://github.com/opengeos/richdem-rs) with **zero GDAL or C/C++ dependencies**.

Run it right now in your browser with zero installation via the live [WebAssembly Playground](playground.md) or try the [Google Colab notebook](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb).

## Demo

[![Interactive inundation simulation and inundation curve in the lidar-rs WebAssembly Playground](https://files.opengeos.org/inundation_demo.gif)](playground.md)

[![Nested depression hierarchy and depression tree overview in lidar-rs](https://files.opengeos.org/depression_hierarchy.webp)](playground.md)

[![Progressive depression tree graph building bottom-up with rising water layers in the lidar-rs WebAssembly Playground](https://files.opengeos.org/dem_depression_tree.gif)](playground.md)

## Key Capabilities

| Capability | Description | Guide |
| --- | --- | --- |
| **All-in-One In-Memory Pipeline** | Run DEM smoothing, optional hydro-conditioning, sink extraction, nested depression hierarchy, single-pass D8 catchment hierarchy, inundation simulation, rating curves, and connectivity graph export in a single command (`lidar pipeline` / `run_pipeline`). | [All-in-One Pipeline](guides/pipeline.md) |
| **Federal Geospatial Data Download** | Pure-Rust downloaders for USGS 3DEP DEM/LPC, USGS WBD, USGS NHD, USFWS NWI, USGS National Transportation, NOAA Atlas 14, and offline culvert synthesis (`lidar download` / `download_federal_data`). | [Colab Notebook](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/federal_data_download.ipynb) |
| **Compressed D8 Catchment Engine** | Single-pass upland D8 forest compression plus per-level Union-Find (`O(N + L*D)`); completed an illustrative 98M-cell, 43-level 1-m USGS 3DEP karst run in ~65 seconds. | [Overview & Architecture](guides/index.md) |
| **Hydro-Conditioning** | Pure-Rust Lindsay (2016) least-cost depression breaching (`complete`, `selective`, `constrained`) and vector-guided culvert burning across road embankments. | [Hydro-Conditioning](guides/hydro-conditioning.md) |
| **Inundation Physics & Storm Forcing** | Spatially variable runoff coefficients, NRCS TR-55 SCS Curve Number conversion, SCS Type II design storm hyetographs, vertical infiltration losses, and antecedent ponded water depths. | [Inundation Physics](guides/inundation-physics.md) |
| **Rating Curves & Connectivity Graphs** | Exact stage-storage-area rating curves (`rating_curves.csv`), directed fill-and-spill / merge networks (JSON + GraphML), georeferenced spill-path vectors, and NHD stream-proximity metrics. | [Connectivity Graphs](guides/connectivity-graphs.md) |
| **LiDAR Point Clouds (DSM, DTM, CHM)** | Native LAS/LAZ/COPC/PLY/E57 reading and conversion, ASPRS class & return filtering, pure-Rust CSF/PMF ground classification, bare-earth DTM, and Canopy Height Models. | [LiDAR Point Clouds](guides/point-clouds.md) |
| **Tiled Regional Delineation & `f32`** | Parallel overlapping tile windows with core-owned seam deduplication and optional `f32` storage (halving source DEM memory). | [Tiled Delineation](guides/tiling.md) |
| **Publication PNG & GIF Rendering** | Pure-Rust shaded-relief inundation animations, individual PNG frames, and inline MCP `preview` renders with overland spill arrows, metric scale bars, legends, and timestamp HUDs. | [Animation & Frames](guides/visualization.md) |
| **AI Agents (MCP Server)** | Pure-Rust Model Context Protocol server (`lidar mcp` or `uvx --from lidar-rs lidar mcp`) for Claude Code, Claude Desktop, Cursor, and VS Code with every CLI tool, inline `preview` PNG images, `info --json --stats`, and guided workflow prompts. | [AI Agents (MCP)](mcp.md) |

## Quick Example

=== "CLI"

    ```sh
    # Download the sample 1-m DEM (400x400)
    curl -LO https://files.opengeos.org/lidar/data/dem.tif

    # Run the complete hydrology, rating-curve, and connectivity pipeline
    lidar pipeline dem.tif out_pipe \
      --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --rain-intensity 5 --duration 50 --time-step 1 --compress

    # Export publication PNG frames with metric scale bar and legend
    lidar export-frames dem.tif out_pipe/fill_time.tif out_pipe/frames \
      --num-steps 10 --time-step 5 --rain-intensity 5 --scale-bar --legend
    ```

=== "Python"

    ```python
    import lidar_rs as lidar

    # End-to-end file pipeline (GeoPackage, CSV, rating curves, GraphML/JSON, rasters)
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
    print(f"Delineated {pipe['num_depressions']} depressions and {pipe['num_spill_paths']} spill paths")

    # Or stay 100% in memory with NumPy arrays (zero temporary files)
    dem_arr = lidar.MedianFilter("dem.tif", kernel_size=3)
    sim = lidar.simulate_inundation_array(
        dem_arr, kernel_size=1, min_size=1000, min_depth=0.3, interval=0.3,
        rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0,
    )
    ```

=== "Rust"

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
        println!("{} depressions, {} spill paths", res.num_depressions, res.num_spill_paths);
        Ok(())
    }
    ```

=== "JavaScript / WASM"

    ```js
    import init, { simulateInundationAdvanced } from "./wasm/lidar_js.js";

    await init();
    const demBytes = new Uint8Array(await (await fetch("https://files.opengeos.org/lidar/data/dem.tif")).arrayBuffer());
    const sim = simulateInundationAdvanced(demBytes, "dem.tif", 3, 1000, 0.3, 0.3, 5.0, 50.0, 1.0, 0.0, 1.0, null);
    try {
      const graph = JSON.parse(sim.connectivityGraphJson());
      const png = sim.renderFramePng(25, true, true, true);
      console.log(`Nodes: ${graph.nodes.length}, Edges: ${graph.edges.length}, PNG: ${png.byteLength} B`);
    } finally {
      sim.free();
    }
    ```

![Animated surface inundation with overland spill paths, scale bar, and legend rendered in pure Rust](https://files.opengeos.org/lidar/images/inundation_spill_demo.gif)

## Four Interfaces, One Pure-Rust Core

- **[Standalone CLI (`lidar-cli`)](cli.md):** Single static binary for Linux, macOS, and Windows with 33 subcommands and JSON output mode.
- **[Rust Library (`lidar`)](rust.md):** Ergonomic in-memory (`Array2D`) and file-based APIs with optional `parallel`, `pointcloud`, `serde`, and `download` Cargo features.
- **[Python Package (`lidar-rs`)](python.md):** Drop-in compatibility with `opengeos/lidar` plus NumPy zero-copy array workflows (`pip install lidar-rs`, Python 3.11+).
- **[WebAssembly (`lidar-js`)](javascript.md):** Full browser and Node.js bindings powering the client-side [Playground](playground.md).

## Citations

If you use `lidar-rs` in published research, please cite:

- **Wu, Q.** (2026). lidar-rs: High-performance nested depression delineation and fill-and-spill inundation simulation. *Zenodo*. <https://doi.org/10.5281/zenodo.24411444>
- **Wu, Q.** (2021). lidar: A Python package for delineating nested surface depressions from digital elevation data. *Journal of Open Source Software*, 6(59), 2965. <https://doi.org/10.21105/joss.02965>
- **Wu, Q.**, Lane, C.R., Wang, L., Vanderhoof, M.K., Christensen, J.R., & Liu, H. (2019). Efficient Delineation of Nested Depression Hierarchy in Digital Elevation Models for Hydrological Analysis Using Level-Set Method. *Journal of the American Water Resources Association*, 55(2), 354–368. <https://doi.org/10.1111/1752-1688.12689>
