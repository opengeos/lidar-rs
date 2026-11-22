# Overview & Architecture

`lidar-rs` implements a complete pure-Rust terrain and depressional hydrology stack with zero C/C++, GDAL, or proprietary GIS dependencies. All algorithms live inside the core `lidar` crate and are exposed uniformly through the CLI, Python, and WebAssembly front ends.

## End-to-End Processing Architecture

```text
Raw LiDAR (.las/.laz/.copc/.ply/.e57)
  └── Ground classification (CSF / PMF) ──► Bare-Earth DTM / DSM / CHM
                                              │
Raw DEM (.tif / COG) ◄────────────────────────┘
  ├── 1. Noise Smoothing (mean / median / Gaussian, NoData-safe reflection)
  ├── 2. Hydro-Conditioning (Lindsay 2016 breaching + vector culvert burning)
  ├── 3. Priority-Flood Sink Extraction (richdem) & 4-Connected Region Grouping
  ├── 4. Wu et al. (2019) Top-Down Level-Set Slicing (24 morphometric attributes)
  ├── 5. Single-Pass Compressed D8 Upland Forest + Per-Level DSU Catchments
  ├── 6. Forced Fill-and-Spill Simulation (SCS-CN, Type II hyetograph, losses, ponding)
  └── 7. Exports: GeoPackage, GeoTIFF, CSV, Rating Curves, JSON/GraphML, PNG/GIF
```

## Single-Pass Compressed D8 Catchment Engine

Earlier per-level watershed approaches ran a full raster D8 watershed pass at every hierarchy level `l = 1..L`, scaling as `O(L * N)` where `N` is total grid cells and `L` is the maximum nesting depth. On complex karst landscapes (30 to 50 hierarchy levels across 100M+ 1-m LiDAR cells), repeated full-grid scans dominated runtime.

`delineate_catchment_hierarchy_in_memory` replaces that per-level scan with a two-stage algorithm:

1. **Single-pass upland forest compression (`O(N)`):** Every non-depression upland cell (`level <= 0`) traces its resolved D8 flow path once, compressing chains so each upland cell directly credits its first entry pixel inside a depression (`upland_weight`).
2. **Incremental per-level Disjoint-Set Union (`O(L * D)`):** Only depression cells (`D << N`) are processed bottom-up from level `1` to `L`, merging adjacent sub-basins and accumulating area, catchment area, and elevation moments via DSU.

!!! tip "Regional 1-m 3DEP Karst Benchmark"
    On the 98M-cell Park City, Kentucky 1-m USGS 3DEP karst HUC-12 watershed (43 hierarchy levels, 3,462 nested depressions, 2,716 spill flow paths), the compressed D8 + DSU engine completes the entire catchment hierarchy and fill-and-spill pipeline in **64.6 seconds** on a single workstation, compared to tens of minutes with per-level raster scans.

## Nested Mount / Canopy Delineation

The same level-set hierarchy engine works on elevated features (hills, drumlins, dunes, tree crowns, and buildings) by inverting a Digital Surface Model (DSM) via `flip_dem` and slicing downward from peaks with `delineate_mounts`.

![Nested mount and canopy hierarchy delineated from a 400x400 DSM in the WebAssembly Playground](https://files.opengeos.org/lidar/images/playground_mounts.png)

## 24-Column Depression Morphometry Schema

Every delineated depression (and mount) computes 24 geometric, topological, and diagnostic attributes exported consistently to `depressions_info.csv`, `depressions.gpkg`, Python dicts, and WASM JSON:

| Column | Type | Definition |
| --- | --- | --- |
| `id` | `u32` | Unique 1-based depression identifier |
| `level` | `u32` | Hierarchy level (`1` = innermost leaf depression; higher = enclosing compound basin) |
| `count` | `usize` | Number of pixels inside the depression footprint |
| `area` | `f64` | Surface area in map units squared (`count * cell_area`) |
| `volume` | `f64` | Relief storage volume below `max_elev` (`avg_depth * area`) |
| `avg_depth` | `f64` | Mean depth below the spill rim (`max_elev - mean_elev`) |
| `max_depth` | `f64` | Maximum depth (`max_elev - min_elev`) |
| `min_elev` | `f64` | Minimum elevation (basin floor) |
| `max_elev` | `f64` | Maximum elevation (spill rim) |
| `children_id` | `string` | Colon-delimited direct child depression IDs (empty for leaves) |
| `region_id` | `u32` | Top-level enclosing sink region ID (1-based) |
| `perimeter` | `f64` | Perimeter in map units (Benkrid-Crookes estimator) |
| `major_axis` | `f64` | Equivalent-ellipse major axis length |
| `minor_axis` | `f64` | Equivalent-ellipse minor axis length |
| `elongatedness` | `f64` | Aspect ratio (`major_axis / minor_axis`) |
| `eccentricity` | `f64` | Equivalent-ellipse eccentricity (`[0, 1)`) |
| `orientation` | `f64` | Principal axis azimuth in degrees |
| `area_bbox_ratio` | `f64` | Extent ratio (pixel count divided by bounding-box area) |
| `parent_id` | `u32` | Direct enclosing parent depression ID (`0` for root basins) |
| `circularity` | `f64` | Isoperimetric circularity `4 * pi * area / perimeter^2`, clamped to `[0, 1]` |
| `depth_diam_ratio` | `f64` | Depth-to-equivalent-diameter ratio `max_depth / (2 * sqrt(area / pi))` |
| `hypsometric_integral` | `f64` | Basin shape factor `avg_depth / max_depth` (`~0.33` cone, `~0.5` paraboloid, `~1.0` flat-bottomed) |
| `rim_slope_deg` | `f64` | Mean rim-to-floor side slope `atan(max_depth / eq_radius)` in degrees |
| `artifact_flag` | `bool` | Heuristic flag for narrow linear ditches or road-embankment artifacts (`elongatedness > 6.0 && area_bbox_ratio < 0.25`) |

## Explore the Guides

- [All-in-One Pipeline & GeoPackage](pipeline.md): One-command DEM-to-hydrology workflow and zero-temp-file NumPy orchestration.
- [Hydro-Conditioning](hydro-conditioning.md): Selective Lindsay (2016) breaching and vector culvert burning.
- [Inundation Physics & Storm Forcing](inundation-physics.md): SCS-CN, SCS Type II design storms, infiltration, and antecedent ponding.
- [Rating Curves & Connectivity Graphs](connectivity-graphs.md): Stage-storage tables, JSON/GraphML networks, and stream proximity.
- [LiDAR Point Clouds (DSM, DTM, CHM)](point-clouds.md): Pure-Rust LAS/LAZ/COPC/PLY/E57 gridding and CSF/PMF bare-earth filtering.
- [Tiled Regional Delineation](tiling.md): Overlapping tile windows and `f32` memory reduction for large DEMs.
- [Animation & Publication Frames](visualization.md): Pure-Rust shaded-relief GIFs and PNG snapshots with scale bars and legends.
