# Python API

`lidar-rs` provides drop-in Python functions matching [`opengeos/lidar`](https://github.com/opengeos/lidar) without requiring GDAL, SciPy, scikit-image, or GeoPandas.

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
```

For rich structured metadata (depression attribute lists, file paths, max hierarchy level), use the snake_case functions `lidar.extract_sinks(...)`, `lidar.delineate_depressions(...)`, and `lidar.delineate_mounts(...)`.
