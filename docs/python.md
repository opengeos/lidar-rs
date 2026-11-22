# Python API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb)

`lidar-rs` provides both drop-in compatibility with [`opengeos/lidar`](https://github.com/opengeos/lidar) and modern high-performance Rust-backed Python APIs (zero GDAL, SciPy, scikit-image, or GeoPandas required; supports Python 3.11+).

```sh
pip install lidar-rs
```

## Recommended Modern Workflows

### 1. All-in-One File Pipeline (`run_pipeline`)

Run smoothing, optional hydro-conditioning, nested depression delineation, single-pass D8 catchment hierarchy, forced inundation, stage-storage rating curves, and directed connectivity graph export in one call:

```python
import lidar_rs as lidar

pipe = lidar.run_pipeline(
    "dem.tif",
    "out_pipe",
    kernel_size=3,
    min_size=1000,
    min_depth=0.3,
    interval=0.3,
    rain_intensity_cm_h=5.0,
    rain_duration_h=50.0,
    time_step_h=1.0,
    breach_max_depth=None,        # set e.g. 1.5 to enable selective breaching
    culverts_vector=None,         # optional .gpkg/.geojson/.shp culvert layer
    stream_vector="sample_streams.gpkg",
    use_f32=False,
    compress=True,
)
print(f"Depressions: {pipe['num_depressions']}, Spill paths: {pipe['num_spill_paths']}")
print("GeoPackage:", pipe["depressions_gpkg"])
print("Rating curves:", pipe["rating_curves_csv"])
print("GraphML:", pipe["connectivity_graphml"])
```

### 2. Zero-Temp-File NumPy Array API

Stay 100% in memory with 2D NumPy `float64` arrays, ideal for Jupyter notebooks, Xarray/Rasterio pipelines, and batch parameter sweeps:

```python
import numpy as np
import tifffile
import lidar_rs as lidar

dem_arr = lidar.MedianFilter("dem.tif", kernel_size=3)  # returns 2D NumPy array when out_file=None

# In-memory sink extraction & nested depression hierarchy
sinks_mem = lidar.extract_sinks_array(dem_arr, min_size=1000, cell_size=1.0)
deps_mem = lidar.delineate_depressions_array(
    sinks_mem["sink"], min_size=1000, min_depth=0.3, interval=0.3, cell_size=1.0
)

# SCS Curve Number -> runoff coefficient + 24-hour SCS Type II design storm
cn = tifffile.imread("sample_cn.tif").astype(np.float64)
rc = lidar.scs_curve_number_runoff(cn, precip_mm=150.0)
rc = np.where(rc == -9999.0, np.nan, rc)  # map NoData sentinel to NaN so array sim falls back to 1.0
hyeto = lidar.scs_type2_hyetograph(total_precip_mm=150.0, duration_h=24.0, time_step_h=0.5)

# Complete in-memory hydrology & forced inundation simulation
sim = lidar.simulate_inundation_array(
    dem_arr,
    kernel_size=1,
    min_size=1000,
    min_depth=0.3,
    interval=0.3,
    rain_intensity_cm_h=0.0,
    rain_duration_h=24.0,
    time_step_h=0.5,
    runoff_coeff_arr=rc,
    hyetograph=hyeto,
    cell_size=1.0,
)
print(sim["fill_time"].shape, len(sim["depressions"]), len(sim["rating_curves"]), sim["graph_node_count"])
```

### 3. Hydro-Conditioning, Point Clouds, Tiling & PNG Frames

```python
import lidar_rs as lidar

# Lindsay (2016) breaching on GeoTIFF or NumPy array
breached_sel = lidar.breach_depressions(dem_arr, mode="selective", max_depth=1.5, max_length=40)
breached_all = lidar.breach_depressions(dem_arr, mode="complete")  # unconstrained

# Culvert burning (same CRS as DEM; returns carved cell count)
carved = lidar.burn_culverts("dem.tif", "sample_culverts.gpkg", "out/burned.tif", half_width_cells=1)

# Tiled regional delineation with f32 source loading
tiled = lidar.delineate_depressions_tiled(
    "dem.tif", "out_tiled",
    tile_size=200, overlap=50, min_size=1000, min_depth=0.3, interval=0.3, use_f32=True,
)

# Pure-Rust LiDAR point clouds: filtered DSM, CSF/PMF bare-earth DTM, and CHM
lidar.lidar_to_dsm("sample_forest.laz", "out/dsm.tif", resolution=1.0, classifications=[2, 5], return_kinds=["first", "only"])
lidar.lidar_to_dtm("sample_forest.laz", "out/dtm.tif", resolution=1.0, ground_method="csf")
lidar.lidar_to_chm("sample_forest.laz", "out/chm.tif", resolution=1.0, ground_method="csf")

# Publication-ready PNG frames with scale bar and legend
frames = lidar.export_inundation_frames_png(
    "dem.tif", "out_pipe/fill_time.tif", "out_pipe/frames",
    num_steps=10, time_step_h=5.0, rain_intensity_cm_h=5.0,
    show_scale_bar=True, show_legend=True,
)
```

## Drop-In `opengeos/lidar` Compatibility

Existing Python scripts written for `opengeos/lidar` work directly with `import lidar_rs as lidar`:

```python
import lidar_rs as lidar

med_path = lidar.MedianFilter("dem.tif", kernel_size=3, out_file="out/median.tif")
sink_path = lidar.ExtractSinks(med_path, min_size=1000, out_dir="out")
dep_id, dep_level = lidar.DelineateDepressions(
    sink_path, min_size=1000, min_depth=0.3, interval=0.3,
    out_dir="out", bool_level_shp=True,
)
# Use the dict-returning snake_case wrapper when exporting a unified GeoPackage:
# deps = lidar.delineate_depressions(sink_path, min_size=1000, min_depth=0.3, interval=0.3, out_dir="out", export_gpkg=True)
mount_id, mount_level = lidar.DelineateMounts(
    "dsm.tif", min_size=1000, min_height=0.3, interval=0.3, out_dir="out_mounts",
)

# Step-by-step inundation & inline Jupyter/Colab animation
sinks = lidar.extract_sinks(med_path, min_size=1000, out_dir="out")
lidar.delineate_catchments(sinks["refined_dem_path"], sinks["region_path"], "out/catchments")
lidar.delineate_flow_paths(
    sinks["filled_dem_path"], sinks["region_path"], "out/flowpath.shp",
    rain_intensity_cm_h=5.0, refined_dem=sinks["refined_dem_path"],
)
hir = lidar.delineate_catchment_hierarchy(sinks["refined_dem_path"], "out/shp-level", "out/catchment_hir.tif")
sim = lidar.simulate_inundation(
    sinks["sink_path"], hir["catchment_raster_path"], "out/simulation",
    min_size=1000, min_depth=0.2, interval=0.2,
    rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0,
)
lidar.PlayAnimation(
    sinks["refined_dem_path"], "out/simulation", "out/inundation.gif",
    rain_intensity_cm_h=5.0, show_scale_bar=True, show_legend=True, display=True,
)
```

## Function Quick Reference

| Group | Functions |
| --- | --- |
| **Pipelines & Tiling** | `run_pipeline`, `delineate_depressions_tiled` |
| **In-Memory NumPy** | `extract_sinks_array`, `delineate_depressions_array`, `simulate_inundation_array`, `scs_curve_number_runoff`, `scs_type2_hyetograph` |
| **Hydro-Conditioning** | `breach_depressions` (file or array), `burn_culverts` |
| **Smoothing & Hierarchies** | `MeanFilter`, `MedianFilter`, `GaussianFilter`, `ExtractSinks`, `extract_sinks`, `DelineateDepressions`, `delineate_depressions`, `DelineateMounts`, `delineate_mounts` |
| **Catchments & Inundation** | `delineate_catchments`, `delineate_flow_paths`, `delineate_catchment_hierarchy`, `simulate_inundation` |
| **Visualization** | `animate_inundation`, `PlayAnimation`, `export_inundation_frames_png` |
| **Point Clouds** | `convert_lidar`, `lidar_to_dsm`, `lidar_to_dtm`, `lidar_to_chm` |
| **Raster / Vector Utils** | `polygonize`, `mosaic`, `reproject_image`, `resample`, `clip_image`, `image_to_cog`, `add_crs`, `join_csv_to_vector`, `csv_points_to_vector` |
