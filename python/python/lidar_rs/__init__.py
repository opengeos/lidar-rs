"""Pure-Rust port of `opengeos/lidar` for Python, powered by `terrio` and `richdem`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from . import _lidar_rs

__version__: str = _lidar_rs.__version__


def _as_path_str(p: str | os.PathLike[str]) -> str:
    return os.fspath(p)


def MeanFilter(
    in_dem: str | os.PathLike[str] | np.ndarray,
    kernel_size: int = 3,
    out_file: str | os.PathLike[str] | None = None,
    no_data: float | None = None,
) -> str | np.ndarray:
    """Apply a moving-window mean filter to a DEM file or 2D NumPy array."""
    if isinstance(in_dem, np.ndarray):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr = np.ascontiguousarray(in_dem, dtype=np.float64)
        return _lidar_rs.mean_filter_array(arr, kernel_size, no_data)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.mean_filter_file(_as_path_str(in_dem), kernel_size, out_str, no_data)
    return out_str if out_str is not None else res


def MedianFilter(
    in_dem: str | os.PathLike[str] | np.ndarray,
    kernel_size: int = 3,
    out_file: str | os.PathLike[str] | None = None,
    no_data: float | None = None,
) -> str | np.ndarray:
    """Apply a moving-window median filter to a DEM file or 2D NumPy array."""
    if isinstance(in_dem, np.ndarray):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr = np.ascontiguousarray(in_dem, dtype=np.float64)
        return _lidar_rs.median_filter_array(arr, kernel_size, no_data)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.median_filter_file(_as_path_str(in_dem), kernel_size, out_str, no_data)
    return out_str if out_str is not None else res


def GaussianFilter(
    in_dem: str | os.PathLike[str] | np.ndarray,
    sigma: float = 1.0,
    out_file: str | os.PathLike[str] | None = None,
    no_data: float | None = None,
) -> str | np.ndarray:
    """Apply a 2D Gaussian filter to a DEM file or 2D NumPy array."""
    if isinstance(in_dem, np.ndarray):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr = np.ascontiguousarray(in_dem, dtype=np.float64)
        return _lidar_rs.gaussian_filter_array(arr, float(sigma), no_data)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.gaussian_filter_file(_as_path_str(in_dem), float(sigma), out_str, no_data)
    return out_str if out_str is not None else res


def FlipDEM(
    dem: str | os.PathLike[str] | np.ndarray,
    delta: float = 100.0,
    out_file: str | os.PathLike[str] | None = None,
    no_data: float | None = None,
) -> str | np.ndarray:
    """Flip a DEM upside-down so elevated features become depressions."""
    if isinstance(dem, np.ndarray):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr = np.ascontiguousarray(dem, dtype=np.float64)
        return _lidar_rs.flip_dem_array(arr, float(delta), no_data)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.flip_dem_file(_as_path_str(dem), float(delta), out_str, no_data)
    return out_str if out_str is not None else res


def extract_sinks(
    in_dem: str | os.PathLike[str],
    min_size: int,
    out_dir: str | os.PathLike[str],
    filled_dem: str | os.PathLike[str] | None = None,
    topology: str = "D8",
    keep_files: bool = True,
    compress: bool = False,
) -> dict[str, Any]:
    """Extract surface depressions (sinks) from a DEM, returning paths and attributes."""
    return _lidar_rs.extract_sinks(
        _as_path_str(in_dem),
        int(min_size),
        _as_path_str(out_dir),
        _as_path_str(filled_dem) if filled_dem is not None else None,
        topology,
        keep_files,
        compress,
    )


def ExtractSinks(
    in_dem: str | os.PathLike[str],
    min_size: int,
    out_dir: str | os.PathLike[str],
    filled_dem: str | os.PathLike[str] | None = None,
    keep_files: bool = True,
) -> str:
    """Drop-in replacement for `lidar.ExtractSinks`; returns the path to `sink.tif`."""
    if not keep_files:
        raise ValueError(
            "ExtractSinks returns sink_path which is deleted when keep_files=False; "
            "call extract_sinks(..., keep_files=False) for GeoPackage-only output"
        )
    res = extract_sinks(
        in_dem, min_size, out_dir, filled_dem=filled_dem, keep_files=keep_files
    )
    return str(res["sink_path"])


def delineate_depressions(
    in_sink: str | os.PathLike[str],
    min_size: int,
    min_depth: float,
    interval: float,
    out_dir: str | os.PathLike[str],
    bool_level_shp: bool = False,
    bool_comb_tif: bool = False,
    compress: bool = False,
) -> dict[str, Any]:
    """Delineate nested depressions using Wu et al. (2019) level-set method."""
    return _lidar_rs.delineate_depressions(
        _as_path_str(in_sink),
        int(min_size),
        float(min_depth),
        float(interval),
        _as_path_str(out_dir),
        bool_level_shp,
        bool_comb_tif,
        compress,
    )


def DelineateDepressions(
    in_sink: str | os.PathLike[str],
    min_size: int,
    min_depth: float,
    interval: float,
    out_dir: str | os.PathLike[str],
    bool_level_shp: bool = False,
) -> tuple[str, str]:
    """Drop-in replacement for `lidar.DelineateDepressions`; returns `(dep_id_path, dep_level_path)`."""
    res = delineate_depressions(
        in_sink, min_size, min_depth, interval, out_dir, bool_level_shp=bool_level_shp
    )
    return str(res["depression_id_path"]), str(res["depression_level_path"])


def delineate_mounts(
    in_dem: str | os.PathLike[str],
    min_size: int,
    min_height: float,
    interval: float,
    out_dir: str | os.PathLike[str],
    bool_shp: bool = False,
    compress: bool = False,
) -> dict[str, Any]:
    """Delineate nested hierarchy of elevated features (mounts)."""
    return _lidar_rs.delineate_mounts(
        _as_path_str(in_dem),
        int(min_size),
        float(min_height),
        float(interval),
        _as_path_str(out_dir),
        bool_shp,
        compress,
    )


def DelineateMounts(
    in_dem: str | os.PathLike[str],
    min_size: int,
    min_height: float,
    interval: float,
    out_dir: str | os.PathLike[str],
    bool_shp: bool = False,
) -> tuple[str, str]:
    """Drop-in replacement for `lidar.DelineateMounts`; returns `(mount_id_path, mount_level_path)`."""
    res = delineate_mounts(
        in_dem, min_size, min_height, interval, out_dir, bool_shp=bool_shp
    )
    return str(res["depression_id_path"]), str(res["depression_level_path"])


def polygonize(
    in_raster: str | os.PathLike[str],
    out_vector: str | os.PathLike[str],
    field_name: str = "id",
) -> int:
    """Polygonize a raster band into a vector file (Shapefile, GeoPackage, GeoJSON, etc.)."""
    return int(_lidar_rs.polygonize(_as_path_str(in_raster), _as_path_str(out_vector), field_name))


def mosaic(
    images: str | os.PathLike[str] | list[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    ext: str = ".tif",
) -> None:
    """Mosaic a list of rasters (or all raster files in a directory) into a single GeoTIFF."""
    out_resolved = Path(output).resolve()
    if isinstance(images, (str, os.PathLike)):
        directory = Path(images)
        ext_lower = ext.lower()
        inputs = sorted(
            str(p)
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() == ext_lower
            and p.resolve() != out_resolved
        )
    else:
        inputs = [_as_path_str(p) for p in images]
    _lidar_rs.mosaic(inputs, _as_path_str(output))


def reproject_image(
    image: str | os.PathLike[str],
    output: str | os.PathLike[str],
    dst_crs: str = "EPSG:4326",
    resampling: str = "nearest",
    resolution: tuple[float, float] | None = None,
) -> None:
    """Reproject a raster to `dst_crs`."""
    _lidar_rs.reproject_image(
        _as_path_str(image), _as_path_str(output), dst_crs, resampling, resolution
    )


def resample(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    resolution: float,
    resampling: str = "bilinear",
) -> None:
    """Resample a raster to a target pixel resolution."""
    _lidar_rs.resample(_as_path_str(src), _as_path_str(dst), float(resolution), resampling)


def clip_image(
    image: str | os.PathLike[str],
    mask: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> None:
    """Clip a raster by a polygon vector mask."""
    _lidar_rs.clip_image(_as_path_str(image), _as_path_str(mask), _as_path_str(output))


def image_to_cog(
    source: str | os.PathLike[str],
    dst_path: str | os.PathLike[str] | None = None,
    compression: str = "DEFLATE",
) -> str:
    """Convert a raster to a Cloud Optimized GeoTIFF (COG)."""
    src = Path(source)
    dst = Path(dst_path) if dst_path is not None else src.with_name(f"{src.stem}_cog.tif")
    _lidar_rs.image_to_cog(str(src), str(dst), compression)
    return str(dst)


def add_crs(
    filename: str | os.PathLike[str],
    epsg: int | str,
    output: str | os.PathLike[str] | None = None,
) -> None:
    """Assign a CRS (EPSG code, WKT, or PROJ string) to a raster."""
    crs_str = f"EPSG:{epsg}" if isinstance(epsg, int) else str(epsg)
    out_str = _as_path_str(output) if output is not None else None
    _lidar_rs.add_crs(_as_path_str(filename), crs_str, out_str)


def join_tables(
    in_shp: str | os.PathLike[str],
    in_csv: str | os.PathLike[str],
    out_shp: str | os.PathLike[str],
    vec_col: str = "id",
    csv_col: str = "region_id",
) -> int:
    """Join a CSV attribute table onto a vector file."""
    return int(
        _lidar_rs.join_tables(
            _as_path_str(in_shp), _as_path_str(in_csv), _as_path_str(out_shp), vec_col, csv_col
        )
    )


def csv_to_shp(
    in_csv: str | os.PathLike[str],
    out_shp: str | os.PathLike[str],
    latitude: str = "latitude",
    longitude: str = "longitude",
) -> int:
    """Convert a CSV file with latitude/longitude columns into a WGS 84 point vector file."""
    return int(
        _lidar_rs.csv_to_vector(_as_path_str(in_csv), _as_path_str(out_shp), latitude, longitude)
    )


csv_points_to_shp = csv_to_shp


def convert_lidar(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    point_format_id: int | None = None,
    minor_version: int | None = None,
) -> int:
    """Convert a LiDAR point cloud between formats (LAS, LAZ, COPC, PLY, E57)."""
    return int(
        _lidar_rs.convert_lidar(
            _as_path_str(source), _as_path_str(destination), point_format_id, minor_version
        )
    )


def lidar_to_dsm(
    filename: str | os.PathLike[str],
    output: str | os.PathLike[str],
    resolution: float = 1.0,
    radius: float | None = None,
    minz: float | None = None,
    maxz: float | None = None,
    stat: str = "max",
    method: str = "statistics",
    window_size: int = 0,
) -> tuple[int, int]:
    """Grid a LiDAR point cloud into a Digital Surface Model (DSM) GeoTIFF."""
    cols, rows = _lidar_rs.lidar_to_dsm(
        _as_path_str(filename),
        _as_path_str(output),
        float(resolution),
        radius,
        minz,
        maxz,
        stat,
        method,
        int(window_size),
    )
    return int(cols), int(rows)


# Snake-case aliases
mean_filter = MeanFilter
median_filter = MedianFilter
gaussian_filter = GaussianFilter
flip_dem = FlipDEM

__all__ = [
    "__version__",
    "MeanFilter",
    "MedianFilter",
    "GaussianFilter",
    "FlipDEM",
    "ExtractSinks",
    "DelineateDepressions",
    "DelineateMounts",
    "mean_filter",
    "median_filter",
    "gaussian_filter",
    "flip_dem",
    "extract_sinks",
    "delineate_depressions",
    "delineate_mounts",
    "polygonize",
    "mosaic",
    "reproject_image",
    "resample",
    "clip_image",
    "image_to_cog",
    "add_crs",
    "join_tables",
    "csv_to_shp",
    "csv_points_to_shp",
    "convert_lidar",
    "lidar_to_dsm",
]
