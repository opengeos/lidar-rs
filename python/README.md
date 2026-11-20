# lidar-rs (Python bindings)

Pure-Rust port of [`opengeos/lidar`](https://github.com/opengeos/lidar) for delineating nested surface depressions and mounts in digital elevation models (DEMs), powered by [`terrio`](https://github.com/opengeos/terrio) and [`richdem`](https://github.com/opengeos/richdem-rs) with **zero GDAL or C/C++ dependencies**.

## Installation

Build from source with [maturin](https://www.maturin.rs/):

```sh
cd python
maturin develop --release
```

## Quick Start

```python
import lidar_rs as lidar

med = lidar.MedianFilter("dem.tif", kernel_size=3, out_file="out/median.tif")
sink = lidar.ExtractSinks(med, min_size=1000, out_dir="out")
dep_id, dep_level = lidar.DelineateDepressions(
    sink, min_size=1000, min_depth=0.3, interval=0.3, out_dir="out", bool_level_shp=True
)
```
