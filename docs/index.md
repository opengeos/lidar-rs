# lidar-rs

[![CI](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/opengeos/lidar-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lidar.svg)](https://crates.io/crates/lidar)
[![PyPI](https://img.shields.io/pypi/v/lidar-rs)](https://pypi.org/project/lidar-rs/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24411444-blue.svg)](https://doi.org/10.5281/zenodo.24411444)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/opengeos/lidar-rs/blob/main/LICENSE)

High-performance nested depression delineation and fill-and-spill inundation simulation from digital elevation models (DEMs) in **pure Rust**, ported from [`opengeos/lidar`](https://github.com/opengeos/lidar) and powered by [`terrio`](https://github.com/opengeos/terrio) and [`richdem`](https://github.com/opengeos/richdem-rs) with **zero GDAL or C/C++ dependencies**.

Try it interactively in your browser with no local setup via the live [WebAssembly Playground](playground.md) or the [Google Colab notebook](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb).

[![Interactive inundation simulation and inundation curve in the lidar-rs WebAssembly Playground](https://files.opengeos.org/inundation_demo.gif)](playground.md)

## Key Features

- **DEM smoothing filters:** Mean, median, and Gaussian filters with symmetric boundary reflection and NoData safety.
- **Sink extraction & DEM refinement:** Priority-Flood depression filling via `richdem`, 4-connected region grouping, small artifact removal, refined DEM generation, and GeoTIFF / CSV / Shapefile / GeoPackage export via `terrio`.
- **Nested depression hierarchy (Wu et al. 2019):** Top-down level-set slicing, 18 geometric & topological attributes per depression, CSV export, and per-level shapefile extraction.
- **Nested mount hierarchy:** DEM inversion (`flip_dem`) and elevated feature hierarchy delineation (`delineate_mounts`).
- **Catchment hierarchy, overland flow paths & inundation animation:** Pure-Rust ports of the ArcGIS Hydrology Analyst workflow (`delineate_catchments`, `delineate_flow_paths`, `delineate_catchment_hierarchy`, `simulate_inundation`, and shaded-relief GIF export via `animate_inundation`).
- **Geospatial raster, vector & point-cloud utilities:** `polygonize`, `mosaic`, `reproject_image`, `resample`, `clip_image`, `image_to_cog`, `add_crs`, `join_csv_to_vector`, `csv_points_to_vector`, and pure-Rust LAS/LAZ/COPC/PLY/E57 I/O & DSM gridding (`lidar_to_dsm`).
- **Rust library (`lidar`), standalone CLI (`lidar-cli`), Python bindings (`lidar-rs`), and WebAssembly (`lidar-js`).**

## Citations

- **Wu, Q.** (2026). lidar-rs: High-performance nested depression delineation and fill-and-spill inundation simulation. *Zenodo*. <https://doi.org/10.5281/zenodo.24411444>
- **Wu, Q.** (2021). lidar: A Python package for delineating nested surface depressions from digital elevation data. *Journal of Open Source Software*, 6(59), 2965. <https://doi.org/10.21105/joss.02965>
- **Wu, Q.**, Lane, C.R., Wang, L., Vanderhoof, M.K., Christensen, J.R., & Liu, H. (2019). Efficient Delineation of Nested Depression Hierarchy in Digital Elevation Models for Hydrological Analysis Using Level-Set Method. *Journal of the American Water Resources Association*, 55(2), 354–368. <https://doi.org/10.1111/1752-1688.12689>
