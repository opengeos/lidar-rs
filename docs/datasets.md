# Sample Datasets

All sample datasets referenced throughout the documentation and tutorials are hosted on Cloudflare R2 (`https://files.opengeos.org/lidar/data/`) so the Git repository stays lightweight while every tutorial command runs out of the box.

## Downloadable Datasets

| Dataset | Format & CRS | Size | Description | Direct URL |
| --- | --- | --- | --- | --- |
| **`dem.tif`** | GeoTIFF (`EPSG:26915`, 400x400, 1 m) | 1.0 MB | Bare-earth DEM featuring a multi-level nested surface depression complex (5 levels, 13 nested depressions, 6 overland spill paths). | [dem.tif](https://files.opengeos.org/lidar/data/dem.tif) |
| **`dsm.tif`** | GeoTIFF (`EPSG:26915`, 400x400, 1 m) | 627 KB | Digital Surface Model for nested mount and canopy hierarchy delineation (`delineate_mounts`). | [dsm.tif](https://files.opengeos.org/lidar/data/dsm.tif) |
| **`sink.tif`** | GeoTIFF (`EPSG:26915`, 400x400, 1 m) | 626 KB | Pre-extracted sink elevation raster derived from `dem.tif` for direct `delineate-depressions` testing. | [sink.tif](https://files.opengeos.org/lidar/data/sink.tif) |
| **`sample_cn.tif`** | GeoTIFF (`EPSG:26915`, 400x400, 1 m) | 1.1 MB | Spatially variable NRCS Curve Number raster (CN 58 to 95) aligned to `dem.tif` for SCS-CN rainfall-runoff forcing tutorials. | [sample_cn.tif](https://files.opengeos.org/lidar/data/sample_cn.tif) |
| **`sample_culverts.gpkg`** | GeoPackage (`EPSG:26915`, LineString) | 96 KB | Projected culvert polylines intersecting `dem.tif` for testing `lidar burn-culverts` and `run_pipeline --culverts`. | [sample_culverts.gpkg](https://files.opengeos.org/lidar/data/sample_culverts.gpkg) |
| **`sample_streams.gpkg`** | GeoPackage (`EPSG:26915`, LineString) | 96 KB | Projected NHD-style stream flowline bordering `dem.tif` for computing spill-endpoint `dist_to_stream` connectivity metrics. | [sample_streams.gpkg](https://files.opengeos.org/lidar/data/sample_streams.gpkg) |
| **`sample_forest.laz`** | LAZ 1.4 (`EPSG:32615`, 7,821 points) | 22 KB | Synthetic forested terrain point cloud with undulating bare earth, a surface depression, 16 multi-return tree canopies, and a flat-roofed building for `lidar-to-dsm`, `lidar-to-dtm` (CSF/PMF), and `lidar-to-chm`. | [sample_forest.laz](https://files.opengeos.org/lidar/data/sample_forest.laz) |

## Quick Download Script

Download the complete tutorial bundle into a local `lidar_samples/` folder:

```sh
mkdir -p lidar_samples && cd lidar_samples
for f in dem.tif dsm.tif sink.tif sample_cn.tif sample_culverts.gpkg sample_streams.gpkg sample_forest.laz; do
  curl -LO "https://files.opengeos.org/lidar/data/$f"
done
```

## Documentation Figures & Demos

All screenshots, comparison figures, and animated GIFs embedded in this documentation are hosted under `https://files.opengeos.org/lidar/images/`:

- [inundation_spill_demo.gif](https://files.opengeos.org/lidar/images/inundation_spill_demo.gif): Pure-Rust shaded-relief inundation animation with overland spill paths, scale bar, and legend.
- [publication_frame_spill_paths.png](https://files.opengeos.org/lidar/images/publication_frame_spill_paths.png): Single publication-ready PNG snapshot exported by `lidar export-frames`.
- [playground_inspector.png](https://files.opengeos.org/lidar/images/playground_inspector.png): Interactive WebAssembly Playground showing Depression Inspector and Stage-Volume Rating Curve.
- [playground_downloads_table.png](https://files.opengeos.org/lidar/images/playground_downloads_table.png): WebAssembly Playground Spatial Layers, client-side downloads, and morphometry table.
- [playground_mounts.png](https://files.opengeos.org/lidar/images/playground_mounts.png): WebAssembly Playground DSM mount/canopy hierarchy delineation.
- [hydro_conditioning_comparison.png](https://files.opengeos.org/lidar/images/hydro_conditioning_comparison.png): Raw DEM vs. selective and complete Lindsay (2016) breaching.
- [inundation_forcing_comparison.png](https://files.opengeos.org/lidar/images/inundation_forcing_comparison.png): SCS Type II hyetograph, spatial SCS-CN runoff map, and inundation curves.
- [rating_curves_and_graph.png](https://files.opengeos.org/lidar/images/rating_curves_and_graph.png): Stage-storage rating curves and directed fill-and-spill connectivity graph.
- [pointcloud_dsm_dtm_chm.png](https://files.opengeos.org/lidar/images/pointcloud_dsm_dtm_chm.png): Pure-Rust LiDAR DSM, CSF bare-earth DTM, and Canopy Height Model.
- [tiled_delineation_diagram.png](https://files.opengeos.org/lidar/images/tiled_delineation_diagram.png): Overlapping tile windows with seam halos and `f32` RAM savings.
