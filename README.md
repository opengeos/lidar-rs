# lidar-rs

[![CI](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lidar.svg)](https://crates.io/crates/lidar)
[![PyPI](https://img.shields.io/pypi/v/lidar-rs)](https://pypi.org/project/lidar-rs/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb)
[![Docs](https://img.shields.io/badge/docs-lidar--rs.opengeos.org-teal.svg)](https://lidar-rs.opengeos.org)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24411444-blue.svg)](https://doi.org/10.5281/zenodo.24411444)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance nested depression delineation and fill-and-spill inundation simulation from Digital Elevation Models (DEMs) in **pure Rust**, ported from [`opengeos/lidar`](https://github.com/opengeos/lidar) and powered by [`terrio`](https://github.com/opengeos/terrio) and [`richdem`](https://github.com/opengeos/richdem-rs) with **zero GDAL or C/C++ dependencies**.

[![Interactive inundation simulation and inundation curve in the lidar-rs WebAssembly Playground](https://files.opengeos.org/inundation_demo.gif)](https://lidar-rs.opengeos.org/playground/)

## Highlights

- **DEM smoothing filters:** Mean (`mean_filter`), median (`median_filter`), and Gaussian (`gaussian_filter`) filters with symmetric boundary reflection, NoData preservation, and optional `rayon` row parallelism.
- **Sink extraction & DEM refinement (`extract_sinks`):** Priority-Flood depression filling via `richdem`, 4-connected region grouping, small artifact removal, refined DEM generation, geometric/topographic depression attributes (`regions_info.csv`), and polygonized Shapefile + attribute-joined GeoPackage export via `terrio`.
- **Nested depression hierarchy (`delineate_depressions`):** Wu et al. (2019) level-set algorithm delineating nested depressions across sink regions in parallel, computing all 18 geometric and topological attributes (`id`, `level`, `count`, `area`, `volume`, `avg_depth`, `max_depth`, `min_elev`, `max_elev`, `children_id`, `region_id`, `perimeter`, `major_axis`, `minor_axis`, `elongatedness`, `eccentricity`, `orientation`, `area_bbox_ratio`), and exporting per-level shapefiles (`shp-level/Single_level_*.shp`).
- **Nested mount hierarchy (`delineate_mounts` & `flip_dem`):** DEM/DSM inversion and elevated feature hierarchy delineation.
- **Catchment hierarchy, flow paths & inundation simulation (`catchment`, `inundation`, `animation`):** Pure-Rust ports of `opengeos/lidar`'s ArcGIS Hydrology Analyst toolbox (`delineate_catchments`, `delineate_flow_paths`, `delineate_catchment_hierarchy`, single-pass parallel `simulate_inundation`, and shaded-relief animated GIF export via `animate_inundation`).
- **Pure-Rust geospatial utilities:** `polygonize`, `mosaic`, `reproject_image`, `resample`, `clip_image`, `image_to_cog`, `add_crs`, `join_csv_to_vector`, `csv_points_to_vector`, and LiDAR point cloud I/O, conversion, and DSM gridding (`read_lidar`, `write_lidar`, `convert_lidar`, `lidar_to_dsm`) via `terrio`.
- **Rust crate (`lidar`), standalone CLI (`lidar-cli`), Python bindings (`lidar-rs` via PyO3 + maturin), and WebAssembly (`lidar-js` via `wasm-bindgen`) with a live browser [Playground](https://lidar-rs.opengeos.org/playground/).**

## Workspace Layout

| Path | Description |
|---|---|
| `crates/lidar` | Pure-Rust core library (published to crates.io as `lidar`) |
| `crates/lidar-cli` | Standalone `lidar` command-line tool (`lidar-cli`) |
| `python` | Python bindings (`lidar-rs`, PyO3 + NumPy) with drop-in `opengeos/lidar` API |
| `js` | WebAssembly (`wasm-bindgen`) bindings (`lidar-js`) for browsers and Node.js |
| `test_data` | Sample DEM/DSM/sink GeoTIFF datasets from `opengeos/lidar` |
| `examples` | Interactive Google Colab demo notebooks (`lidar_rs_demo.ipynb`, `inundation_demo.ipynb`) |
| `docs` | Zensical documentation site, interactive WASM Playground, and CLI installer scripts (`install.sh`, `install.ps1`) |

## Rust Usage

Add `lidar` to your `Cargo.toml`:

```toml
[dependencies]
lidar = "0.4"
```

```rust,no_run
use std::path::Path;
use lidar::{
    DelineateOptions, ExtractSinksOptions,
    delineate_depressions, extract_sinks, median_filter_file,
};

fn main() -> lidar::Result<()> {
    let out_dir = Path::new("output");
    let med_path = out_dir.join("median.tif");

    median_filter_file("dem.tif", 3, Some(&med_path))?;

    let sinks = extract_sinks(
        &med_path,
        out_dir,
        &ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )?;

    let deps = delineate_depressions(
        sinks.sink_path.as_ref().unwrap(),
        out_dir,
        &DelineateOptions {
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            bool_level_shp: true,
            ..Default::default()
        },
    )?;

    println!(
        "Delineated {} nested depressions across {} regions (max level = {})",
        deps.depressions.len(),
        deps.num_regions,
        deps.max_level,
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

# Or from crates.io
cargo install lidar-cli
```

Example commands:

```sh
lidar median-filter dem.tif out/median.tif -k 3
lidar extract-sinks out/median.tif out --min-size 1000
lidar delineate-depressions out/sink.tif out --min-size 1000 --min-depth 0.3 --interval 0.3 --level-shp
lidar catchment-hierarchy out/dem.tif out/shp-level out/catchment_hir.tif
lidar simulate-inundation out/sink.tif out/catchment_hir.tif out/simulation --rain-intensity 5 --duration 50 --time-step 1
lidar animate-inundation out/dem.tif out/simulation out/inundation.gif --rain-intensity 5
lidar delineate-mounts dsm.tif out_mounts --min-size 1000 --min-height 0.3 --interval 0.3
lidar lidar-to-dsm cloud.laz dsm.tif --resolution 1.0
```

## Python Usage

```sh
pip install lidar-rs
```

Or try it interactively in [Google Colab](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb).

```python
import lidar_rs as lidar

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
