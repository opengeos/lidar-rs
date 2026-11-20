# Python API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb)

`lidar-rs` provides drop-in Python functions matching [`opengeos/lidar`](https://github.com/opengeos/lidar) without requiring GDAL, SciPy, scikit-image, or GeoPandas. Try the [interactive Google Colab notebook](https://colab.research.google.com/github/opengeos/lidar-rs/blob/main/examples/lidar_rs_demo.ipynb) to run the full DEM/DSM pipeline in your browser.

```python
import lidar_rs as lidar

# Filter DEM (accepts file path or 2D NumPy array)
med_path = lidar.MedianFilter("dem.tif", kernel_size=3, out_file="out/median.tif")

# Extract sinks
sink_path = lidar.ExtractSinks(med_path, min_size=1000, out_dir="out")

# Delineate nested depression hierarchy
dep_id, dep_level = lidar.DelineateDepressions(
    sink_path,
    min_size=1000,
    min_depth=0.3,
    interval=0.3,
    out_dir="out",
    bool_level_shp=True,
)

# Delineate nested mounts from a DSM
mount_id, mount_level = lidar.DelineateMounts(
    "dsm.tif",
    min_size=1000,
    min_height=0.3,
    interval=0.3,
    out_dir="out_mounts",
)

# Full inundation dynamics workflow (pure Rust, zero ArcGIS/GDAL)
sinks = lidar.extract_sinks(med_path, min_size=1000, out_dir="out")
lidar.delineate_catchments(sinks["refined_dem_path"], sinks["region_path"], "out/catchments")
lidar.delineate_flow_paths(
    sinks["filled_dem_path"], sinks["region_path"], "out/flowpath.shp",
    rain_intensity_cm_h=5.0, refined_dem=sinks["refined_dem_path"],
)
hir = lidar.delineate_catchment_hierarchy(
    sinks["refined_dem_path"], "out/shp-level", "out/catchment_hir.tif"
)
sim = lidar.simulate_inundation(
    sinks["sink_path"], hir["catchment_raster_path"], "out/simulation",
    min_size=1000, min_depth=0.2, interval=0.2,
    rain_intensity_cm_h=5.0, rain_duration_h=50.0, time_step_h=1.0,
)
lidar.PlayAnimation(
    sinks["refined_dem_path"], "out/simulation", "out/inundation.gif",
    rain_intensity_cm_h=5.0, display=True,  # displays inline in Jupyter / Colab
)
```

For rich structured metadata (depression attribute lists, file paths, max hierarchy level), use the snake_case functions `lidar.extract_sinks(...)`, `lidar.delineate_depressions(...)`, `lidar.delineate_mounts(...)`, `lidar.delineate_catchments(...)`, `lidar.delineate_flow_paths(...)`, `lidar.delineate_catchment_hierarchy(...)`, `lidar.simulate_inundation(...)`, and `lidar.animate_inundation(...)`.
