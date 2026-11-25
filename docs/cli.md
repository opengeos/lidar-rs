# Command-Line Reference

The `lidar` CLI (`lidar-cli`) provides a standalone, zero-dependency binary wrapping every terrain, hydrology, point-cloud, and geospatial utility in `lidar-rs`. Run `lidar <subcommand> --help` for full flag details, and pass `--json` on any analysis or utility subcommand for structured machine-readable JSON output.

All sample files used below can be downloaded from [Sample Datasets](datasets.md) (`https://files.opengeos.org/lidar/data/`).

## Subcommand Overview

| Category | Subcommands |
| --- | --- |
| **All-in-One & Tiled Pipelines** | `pipeline`, `tiled-depressions` |
| **Federal Data Download** | `download`, `noaa-atlas14`, `synthesize-culverts` |
| **Hydro-Conditioning** | `breach-depressions`, `burn-culverts` |
| **DEM Smoothing & Sinks** | `mean-filter`, `median-filter`, `gaussian-filter`, `extract-sinks` |
| **Nested Hierarchies** | `delineate-depressions`, `delineate-mounts`, `flip-dem` |
| **Catchments & Inundation** | `delineate-catchments`, `delineate-flow-paths`, `catchment-hierarchy`, `simulate-inundation` |
| **Animation & Visualization** | `animate-inundation`, `export-frames`, `preview` |
| **AI Agents (MCP)** | `mcp` (plus `--json-usage` global catalog flag) |
| **LiDAR Point Clouds** | `convert-lidar`, `classify-ground`, `lidar-to-dsm`, `lidar-to-dtm`, `lidar-to-chm` |
| **Raster & Vector Utilities** | `polygonize`, `mosaic`, `reproject`, `resample`, `clip`, `to-cog`, `add-crs`, `join-tables`, `csv-to-vector`, `info` |

## All-in-One Pipeline (`lidar pipeline`)

Runs smoothing, optional hydro-conditioning, sink extraction, nested depression hierarchy, single-pass D8 catchment hierarchy, forced fill-and-spill simulation, stage-storage rating curves, and connectivity graph export in memory:

```sh
# Standard constant-rainfall pipeline
lidar pipeline dem.tif out_pipe \
  --kernel 3 \
  --min-size 1000 --min-depth 0.3 --interval 0.3 \
  --rain-intensity 5 --duration 50 --time-step 1 \
  --compress --json

# Full hydro-conditioned + SCS Type II 24-h design storm + NHD stream proximity
# (requires aligned forcing rasters; see the Inundation Physics guide for creating runoff_coeff.tif)
lidar pipeline dem.tif out_storm \
  --culverts sample_culverts.gpkg \
  --breach-max-depth 1.5 --breach-max-length 40 \
  --scs-type2-mm 100 --time-step 0.5 \
  --runoff-coeff runoff_coeff.tif \
  --infiltration infil_cm_h.tif \
  --antecedent-depth antecedent_m.tif \
  --streams sample_streams.gpkg \
  --use-f32 --compress
```

| Flag | Default | Description |
| --- | --- | --- |
| `--kernel <N>` | `3` | Median filter kernel size (`<= 1` skips smoothing) |
| `--breach-mode <MODE>` | off | Enables Lindsay (2016) pre-conditioning (`complete`, `selective`, or `constrained`) |
| `--breach-max-depth <M>` | off | Enables selective Lindsay (2016) breaching with max depth `M` (pass `0` for unlimited) |
| `--breach-max-length <N>` | `50` | Max breach path length in cells (requires `--breach-max-depth`; pass `0` for unlimited) |
| `--breach-fill-remaining` | `false` | Fill remaining unbreached depressions after selective/constrained breaching |
| `--culverts <PATH>` | off | Vector file (`.gpkg`, `.geojson`, `.shp`) burned before sink extraction |
| `--scs-type2-mm <MM>` | off | Enables an NRCS SCS Type II design storm (`--duration` defaults to `24.0` h when omitted) |
| `--routing <MODE>` | `static` | Inundation routing mode: `static` (independent level-set) or `fsm` (dynamic Fill-Spill-Merge cascade mass-balance) |
| `--runoff-coeff <TIF>` | off | Spatially variable runoff coefficient raster (`0..=1`) |
| `--curve-number <TIF>` | off | Spatially variable NRCS Curve Number (`0..=100`) raster converted to runoff coefficient via TR-55 (conflicts with `--runoff-coeff`) |
| `--infiltration <TIF>` | off | Spatially variable vertical infiltration loss raster (`cm/h`) |
| `--antecedent-depth <TIF>` | off | Antecedent ponded water depth raster (`m`) |
| `--streams <PATH>` | off | Stream flowline vector for computing `dist_to_stream` on spill edges |
| `--use-f32` | `false` | Load source DEM as `f32` to halve initial memory footprint |
| `--no-rasters`, `--no-vectors`, `--no-graph` | `false` | Skip writing raster, GeoPackage, or rating-curve/graph output groups |

## Federal Data Download (`lidar download`, `lidar noaa-atlas14`)

Download open U.S. federal datasets (USGS 3DEP DEM/LPC, USGS WBD, USGS NHD, USFWS NWI, USGS National Transportation roads and railways, synthesized culverts, and NOAA Atlas 14 precipitation frequencies) for any WGS 84 bounding box, USGS HUC code (HUC-2 through HUC-12), or local polygon file. Prebuilt release binaries include `lidar download` and `lidar noaa-atlas14`; when installing from crates.io, enable the opt-in feature with `cargo install lidar-cli --features download` (also enables `terrio/remote` HTTP range reads):

```sh
# Dry-run query for matching 1-m USGS 3DEP tiles in a HUC-12 watershed
lidar download --huc 101600020503 -o out_huc12 --dry-run

# Download all 7 federal layers, mosaic & clip 1-m DEM, and synthesize road-stream culverts
lidar download --huc 101600020503 -o out_huc12 --tile-cache tiles_cache --json

# Download selected layers with custom HTTP/API tuning
lidar download --bbox -99.14,47.04,-99.08,47.08 -o out_bbox \
  --datasets wbd,nhd,nwi,roads,culverts,noaa \
  --max-products 500 --transportation-layers 29,30,31,32,33 --timeout-secs 60 --max-retries 3 --oid-chunk-size 200
lidar download --region study_area.geojson -o out_poly --dem-product 10m

# Fetch a NOAA Atlas 14 precipitation frequency table directly for any point
lidar noaa-atlas14 --lat 47.098 --lon -99.095 -o noaa_atlas14.json --json

# Synthesize culvert breach polylines offline from any stream and road vector files
lidar synthesize-culverts nhd_flowlines.gpkg usgs_roads.gpkg culverts.gpkg --half-length 25 --snap-distance 15 --json
```

## Hydro-Conditioning

```sh
# Selective Lindsay (2016) breaching (defaults: max_depth=1.5, max_length=50; pass 0 for unlimited)
lidar breach-depressions dem.tif out/breached.tif --mode selective --max-depth 1.5 --max-length 40 --compress

# Unconstrained complete breaching
lidar breach-depressions dem.tif out/breached_complete.tif --mode complete --compress

# Vector-guided culvert burning (same CRS as DEM required)
lidar burn-culverts dem.tif sample_culverts.gpkg out/burned.tif --half-width 1 --depth-offset 0.05 --compress
```

## Tiled Regional Delineation

```sh
# Uses f32 source storage by default to halve RAM (pass --use-f64 for double precision, -t d4|d8 for topology)
lidar tiled-depressions large_dem.tif out_tiled \
  --tile-size 2048 --overlap 256 \
  --kernel 3 --min-size 1000 --min-depth 0.3 --interval 0.3 \
  --topology d8 --json
```

## Step-by-Step Depression, Mount & Inundation Workflow

```sh
# DEM smoothing filters
lidar median-filter dem.tif out/median.tif -k 3
lidar mean-filter dem.tif out/mean.tif -k 3
lidar gaussian-filter dem.tif out/gaussian.tif -s 1.0

# Sink extraction & DEM refinement
lidar extract-sinks out/median.tif out --min-size 1000

# Nested depression hierarchy (Wu et al. 2019), including unified GeoPackage
lidar delineate-depressions out/sink.tif out \
  --min-size 1000 --min-depth 0.3 --interval 0.3 --level-shp --gpkg

# Elevated features (nested mounts / canopies from DSM), with optional GeoPackage & combined level rasters
lidar delineate-mounts dsm.tif out_mounts \
  --min-size 1000 --min-height 0.3 --interval 0.3 --level-shp --comb-tif --gpkg

# Catchments, overland flow paths, catchment hierarchy, and inundation simulation
# (static routing uses catchment_hir.tif; --routing fsm requires --dem to compute D8 catchments directly)
lidar delineate-catchments out/dem.tif out/region.tif out/catchments
lidar delineate-flow-paths out/dem_filled.tif out/region.tif out/flowpath.shp --rain-intensity 5 --refined-dem out/dem.tif
lidar catchment-hierarchy out/dem.tif out/shp-level out/catchment_hir.tif
lidar simulate-inundation out/sink.tif out/catchment_hir.tif out/simulation \
  --min-size 1000 --min-depth 0.2 --interval 0.2 \
  --rain-intensity 5 --duration 50 --time-step 1
lidar simulate-inundation out/sink.tif out/catchment_hir.tif out/sim_fsm \
  --routing fsm --dem out/dem.tif --scs-type2-mm 100 --runoff-coeff 0.75 --infiltration 0.2 --antecedent-depth 0.05 --json

# Publication PNG frames & animated GIF with scale bar and legend
lidar export-frames dem.tif out_pipe/fill_time.tif out/frames \
  --num-steps 10 --time-step 5 --rain-intensity 5 --scale-bar --legend
lidar animate-inundation out/dem.tif out/simulation out/inundation.gif \
  --delay-ms 150 --rain-intensity 5 --scale-bar --legend
```

## LiDAR Point Clouds (LAS, LAZ, COPC, PLY, E57)

```sh
# Format conversion
lidar convert-lidar sample_forest.laz out/forest.las --point-format 6

# Standalone pure-Rust ground classification (CSF, PMF, or SMRF)
lidar classify-ground sample_forest.laz out/forest_ground.laz --method smrf --json

# Filtered DSM gridding (ASPRS classes & return kinds)
lidar lidar-to-dsm sample_forest.laz out/dsm.tif \
  --resolution 1.0 --radius 1.5 --classification 2,5 --return-kind first,only

# Bare-earth DTM using pure-Rust CSF, PMF, or SMRF ground classification (with optional ELM outlier removal)
lidar lidar-to-dtm sample_forest.laz out/dtm.tif --resolution 1.0 --ground-method smrf --remove-outliers

# Canopy Height Model (CHM = max(0, DSM - DTM)) on one shared grid
lidar lidar-to-chm sample_forest.laz out/chm.tif --resolution 1.0 --ground-method csf
```

## Geospatial Raster & Vector Utilities

```sh
lidar info dem.tif --json --stats
lidar preview dem.tif -o preview.png --scale-bar
lidar preview dem.tif -o flood.png --fill-time out_pipe/fill_time.tif --hour 10 --scale-bar --legend
lidar --json-usage   # dump machine-readable JSON command catalog
lidar polygonize out/region.tif out/regions.geojson --field-name id
lidar mosaic tile1.tif tile2.tif -o mosaic.tif
lidar reproject dem.tif dem_4326.tif --crs EPSG:4326 -r bilinear
lidar resample dem.tif dem_2m.tif --resolution 2.0 -r bilinear
lidar clip dem.tif mask.gpkg dem_clip.tif
lidar to-cog dem.tif dem_cog.tif --compression DEFLATE
lidar add-crs dem.tif EPSG:26915 -o dem_crs.tif
lidar join-tables out/regions.shp out/regions_info.csv out/joined.gpkg
lidar csv-to-vector points.csv points.shp --latitude latitude --longitude longitude
```

## AI Agents (MCP Server)

`lidar mcp` runs a foreground JSON-RPC 2.0 server over stdin/stdout (normally launched by an MCP client such as Claude Code, Claude Desktop, Cursor, or VS Code; see [AI Agents (MCP)](mcp.md)):

```sh
claude mcp add lidar -- lidar mcp
```
