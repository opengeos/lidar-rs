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
    export_gpkg: bool = False,
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
        export_gpkg,
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
        path = Path(images)
        if path.is_file():
            inputs = [str(path)]
        elif path.is_dir():
            ext_lower = ext.lower()
            inputs = sorted(
                str(p)
                for p in path.iterdir()
                if p.is_file()
                and p.suffix.lower() == ext_lower
                and p.resolve() != out_resolved
            )
        else:
            raise ValueError(
                f"mosaic expected an existing file, directory, or list of raster paths; got {path}"
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
    classifications: list[int] | None = None,
    return_kinds: list[str] | None = None,
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
        [int(c) for c in classifications] if classifications is not None else None,
        list(return_kinds) if return_kinds is not None else None,
    )
    return int(cols), int(rows)


def lidar_to_dtm(
    filename: str | os.PathLike[str],
    output: str | os.PathLike[str],
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    radius: float | None = None,
    method: str = "linear",
    stat: str = "idw",
    window_size: int = 5,
) -> tuple[int, int]:
    """Generate a bare-earth DTM GeoTIFF from a LiDAR point cloud (preclassified, CSF, or PMF)."""
    cols, rows = _lidar_rs.lidar_to_dtm(
        _as_path_str(filename),
        _as_path_str(output),
        float(resolution),
        ground_method,
        int(ground_class),
        radius,
        method,
        stat,
        int(window_size),
    )
    return int(cols), int(rows)


def lidar_to_chm(
    filename: str | os.PathLike[str],
    output: str | os.PathLike[str],
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    radius: float | None = None,
    window_size: int = 5,
) -> tuple[int, int]:
    """Generate a Canopy Height Model (CHM = DSM - DTM) GeoTIFF from a LiDAR point cloud."""
    cols, rows = _lidar_rs.lidar_to_chm(
        _as_path_str(filename),
        _as_path_str(output),
        float(resolution),
        ground_method,
        int(ground_class),
        radius,
        int(window_size),
    )
    return int(cols), int(rows)


def delineate_catchments(
    in_dem: str | os.PathLike[str],
    in_sink: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    compress: bool = False,
) -> dict[str, Any]:
    """Delineate contributing catchments for extracted sinks (ports 2_Delineate_Catchment.py)."""
    return _lidar_rs.delineate_catchments(
        _as_path_str(in_dem),
        _as_path_str(in_sink),
        _as_path_str(out_dir),
        compress,
    )


def DelineateCatchments(
    in_dem: str | os.PathLike[str],
    in_sink: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> str:
    """ArcGIS-style wrapper for `delineate_catchments`; returns the catchment vector path."""
    return str(delineate_catchments(in_dem, in_sink, out_dir)["vector_path"])


def delineate_flow_paths(
    in_dem_filled: str | os.PathLike[str],
    in_sink: str | os.PathLike[str],
    out_flowpath: str | os.PathLike[str],
    rain_intensity_cm_h: float = 5.0,
    refined_dem: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Delineate overland spill flow paths between sinks (ports 3_Flow_Path.py)."""
    return _lidar_rs.delineate_flow_paths(
        _as_path_str(in_dem_filled),
        _as_path_str(in_sink),
        _as_path_str(out_flowpath),
        float(rain_intensity_cm_h),
        _as_path_str(refined_dem) if refined_dem is not None else None,
    )


def DelineateFlowPaths(
    in_dem_filled: str | os.PathLike[str],
    in_sink: str | os.PathLike[str],
    out_flowpath: str | os.PathLike[str],
    rain_intensity_cm_h: float = 5.0,
    refined_dem: str | os.PathLike[str] | None = None,
) -> str:
    """ArcGIS-style wrapper for `delineate_flow_paths`; returns the output flow-path vector path."""
    return str(
        delineate_flow_paths(
            in_dem_filled,
            in_sink,
            out_flowpath,
            rain_intensity_cm_h=rain_intensity_cm_h,
            refined_dem=refined_dem,
        )["flowpath_vector_path"]
    )


def delineate_catchment_hierarchy(
    in_dem: str | os.PathLike[str],
    in_level_dir: str | os.PathLike[str],
    out_catchment_img: str | os.PathLike[str],
    compress: bool = False,
) -> dict[str, Any]:
    """Delineate nested catchment hierarchy across depression levels (ports 5_Catchment_Hierarchy.py)."""
    return _lidar_rs.delineate_catchment_hierarchy(
        _as_path_str(in_dem),
        _as_path_str(in_level_dir),
        _as_path_str(out_catchment_img),
        compress,
    )


def CatchmentHierarchy(
    in_dem: str | os.PathLike[str],
    in_level_dir: str | os.PathLike[str],
    out_catchment_img: str | os.PathLike[str],
) -> str:
    """ArcGIS-style wrapper for `delineate_catchment_hierarchy`; returns `catchment_hir.tif` path."""
    return str(
        delineate_catchment_hierarchy(in_dem, in_level_dir, out_catchment_img)[
            "catchment_raster_path"
        ]
    )


def simulate_inundation(
    in_sink: str | os.PathLike[str],
    in_catchment: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    min_size: int = 1000,
    min_depth: float = 0.2,
    interval: float = 0.2,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    compress: bool = False,
) -> dict[str, Any]:
    """Simulate surface inundation dynamics over time (ports 6_Simulate Inundation.py)."""
    return _lidar_rs.simulate_inundation(
        _as_path_str(in_sink),
        _as_path_str(in_catchment),
        _as_path_str(out_dir),
        int(min_size),
        float(min_depth),
        float(interval),
        float(rain_intensity_cm_h),
        float(rain_duration_h),
        float(time_step_h),
        compress,
    )


def SimulateInundation(
    in_sink: str | os.PathLike[str],
    in_catchment: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    min_size: int = 1000,
    min_depth: float = 0.2,
    interval: float = 0.2,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
) -> list[str]:
    """ArcGIS-style wrapper for `simulate_inundation`; returns the list of `flood_*.tif` paths."""
    return list(
        simulate_inundation(
            in_sink,
            in_catchment,
            out_dir,
            min_size=min_size,
            min_depth=min_depth,
            interval=interval,
            rain_intensity_cm_h=rain_intensity_cm_h,
            rain_duration_h=rain_duration_h,
            time_step_h=time_step_h,
        )["frame_paths"]
    )


def animate_inundation(
    in_dem: str | os.PathLike[str],
    in_flood_dir: str | os.PathLike[str],
    out_gif: str | os.PathLike[str],
    frame_delay_ms: int = 200,
    ping_pong: bool = True,
    loops: int = 0,
    water_alpha: float = 0.55,
    max_dimension: int | None = None,
    time_step_h: float | None = 1.0,
    rain_intensity_cm_h: float | None = None,
    azimuth_deg: float = 135.0,
    elevation_deg: float = 45.0,
    show_scale_bar: bool = False,
    show_legend: bool = False,
) -> dict[str, Any]:
    """Render a shaded-relief inundation animated GIF (ports 7_Play_Animation.py)."""
    return _lidar_rs.animate_inundation(
        _as_path_str(in_dem),
        _as_path_str(in_flood_dir),
        _as_path_str(out_gif),
        int(frame_delay_ms),
        bool(ping_pong),
        int(loops),
        float(water_alpha),
        int(max_dimension) if max_dimension is not None else None,
        float(time_step_h) if time_step_h is not None else None,
        float(rain_intensity_cm_h) if rain_intensity_cm_h is not None else None,
        float(azimuth_deg),
        float(elevation_deg),
        bool(show_scale_bar),
        bool(show_legend),
    )


def export_inundation_frames_png(
    in_dem: str | os.PathLike[str],
    in_fill_time: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    num_steps: int = 10,
    time_step_h: float = 1.0,
    rain_intensity_cm_h: float | None = None,
    water_alpha: float = 0.55,
    max_dimension: int | None = None,
    show_scale_bar: bool = True,
    show_legend: bool = True,
    azimuth_deg: float = 135.0,
    elevation_deg: float = 45.0,
) -> list[str]:
    """Export numbered shaded-relief inundation PNG frames with optional scale bar and legend."""
    return list(
        _lidar_rs.export_inundation_frames_png(
            _as_path_str(in_dem),
            _as_path_str(in_fill_time),
            _as_path_str(out_dir),
            int(num_steps),
            float(time_step_h),
            float(rain_intensity_cm_h) if rain_intensity_cm_h is not None else None,
            float(water_alpha),
            int(max_dimension) if max_dimension is not None else None,
            bool(show_scale_bar),
            bool(show_legend),
            float(azimuth_deg),
            float(elevation_deg),
        )
    )


def breach_depressions(
    in_dem: str | os.PathLike[str] | np.ndarray,
    out_dem: str | os.PathLike[str] | None = None,
    mode: str = "selective",
    max_depth: float | None = 1.5,
    max_length: int | None = 50,
    fill_remaining: bool = False,
    compress: bool = False,
    no_data: float | None = None,
) -> str | np.ndarray:
    """Breach depressions using Lindsay (2016) on a GeoTIFF path or 2D NumPy array."""
    if isinstance(in_dem, np.ndarray):
        arr = np.ascontiguousarray(in_dem, dtype=np.float64)
        return _lidar_rs.breach_depressions_array(
            arr, mode, max_depth, max_length, bool(fill_remaining), no_data
        )
    if out_dem is None:
        raise ValueError("out_dem is required when in_dem is a file path")
    return str(
        _lidar_rs.breach_depressions(
            _as_path_str(in_dem),
            _as_path_str(out_dem),
            mode,
            max_depth,
            max_length,
            bool(fill_remaining),
            bool(compress),
        )
    )


def burn_culverts(
    in_dem: str | os.PathLike[str],
    culverts_vector: str | os.PathLike[str],
    out_dem: str | os.PathLike[str],
    half_width_cells: int = 0,
    depth_offset: float = 0.05,
    point_radius_cells: int = 5,
    compress: bool = False,
) -> int:
    """Burn polyline or point culvert vectors across road embankments into a DEM."""
    return int(
        _lidar_rs.burn_culverts(
            _as_path_str(in_dem),
            _as_path_str(culverts_vector),
            _as_path_str(out_dem),
            int(half_width_cells),
            float(depth_offset),
            int(point_radius_cells),
            bool(compress),
        )
    )


def scs_curve_number_runoff(
    cn_arr: np.ndarray,
    precip_mm: float,
    no_data: float | None = None,
) -> np.ndarray:
    """Convert an NRCS Curve Number 2D NumPy array and rainfall depth (mm) into runoff coefficients (0..1)."""
    arr = np.ascontiguousarray(cn_arr, dtype=np.float64)
    return _lidar_rs.scs_curve_number_runoff(arr, float(precip_mm), no_data)


def scs_type2_hyetograph(
    total_precip_mm: float,
    duration_h: float = 24.0,
    time_step_h: float = 1.0,
) -> list[tuple[float, float]]:
    """Generate an SCS Type II design storm hyetograph as `[(dt_h, intensity_cm_h), ...]`."""
    return list(
        _lidar_rs.scs_type2_hyetograph(
            float(total_precip_mm), float(duration_h), float(time_step_h)
        )
    )


def extract_sinks_array(
    dem_arr: np.ndarray,
    min_size: int = 1000,
    topology: str = "D8",
    no_data: float | None = None,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Extract surface sinks completely in memory from a 2D NumPy DEM array."""
    arr = np.ascontiguousarray(dem_arr, dtype=np.float64)
    return _lidar_rs.extract_sinks_array(arr, int(min_size), topology, no_data, float(cell_size))


def delineate_depressions_array(
    sink_arr: np.ndarray,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Delineate nested depression hierarchy completely in memory from a 2D NumPy sink array."""
    arr = np.ascontiguousarray(sink_arr, dtype=np.float64)
    return _lidar_rs.delineate_depressions_array(
        arr, int(min_size), float(min_depth), float(interval), no_data, float(cell_size)
    )


def simulate_inundation_array(
    dem_arr: np.ndarray,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    runoff_coeff_arr: np.ndarray | None = None,
    infiltration_arr: np.ndarray | None = None,
    antecedent_depth_arr: np.ndarray | None = None,
    hyetograph: list[tuple[float, float]] | None = None,
    breach_max_depth: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Run the complete hydrology and inundation pipeline in memory on a 2D NumPy DEM array."""
    arr = np.ascontiguousarray(dem_arr, dtype=np.float64)
    rc = np.ascontiguousarray(runoff_coeff_arr, dtype=np.float64) if runoff_coeff_arr is not None else None
    inf = np.ascontiguousarray(infiltration_arr, dtype=np.float64) if infiltration_arr is not None else None
    ant = (
        np.ascontiguousarray(antecedent_depth_arr, dtype=np.float64)
        if antecedent_depth_arr is not None
        else None
    )
    return _lidar_rs.simulate_inundation_array(
        arr,
        int(kernel_size),
        int(min_size),
        float(min_depth),
        float(interval),
        float(rain_intensity_cm_h),
        float(rain_duration_h),
        float(time_step_h),
        rc,
        inf,
        ant,
        hyetograph,
        breach_max_depth,
        no_data,
        float(cell_size),
    )


def run_pipeline(
    in_dem: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    breach_max_depth: float | None = None,
    breach_max_length: int = 40,
    culverts_vector: str | os.PathLike[str] | None = None,
    runoff_coeff_path: str | os.PathLike[str] | None = None,
    infiltration_path: str | os.PathLike[str] | None = None,
    antecedent_depth_path: str | os.PathLike[str] | None = None,
    hyetograph: list[tuple[float, float]] | None = None,
    stream_vector: str | os.PathLike[str] | None = None,
    use_f32: bool = False,
    save_rasters: bool = True,
    export_vectors: bool = True,
    export_graph: bool = True,
    compress: bool = False,
) -> dict[str, Any]:
    """Run the all-in-one in-memory hydrology pipeline from a raw DEM file to GeoPackage, CSV, GraphML, and rasters."""
    return _lidar_rs.run_pipeline(
        _as_path_str(in_dem),
        _as_path_str(out_dir),
        int(kernel_size),
        int(min_size),
        float(min_depth),
        float(interval),
        float(rain_intensity_cm_h),
        float(rain_duration_h),
        float(time_step_h),
        breach_max_depth,
        int(breach_max_length),
        _as_path_str(culverts_vector) if culverts_vector is not None else None,
        _as_path_str(runoff_coeff_path) if runoff_coeff_path is not None else None,
        _as_path_str(infiltration_path) if infiltration_path is not None else None,
        _as_path_str(antecedent_depth_path) if antecedent_depth_path is not None else None,
        hyetograph,
        _as_path_str(stream_vector) if stream_vector is not None else None,
        bool(use_f32),
        bool(save_rasters),
        bool(export_vectors),
        bool(export_graph),
        bool(compress),
    )


def delineate_depressions_tiled(
    in_dem: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    tile_size: int = 2048,
    overlap: int = 128,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    use_f32: bool = True,
) -> dict[str, Any]:
    """Delineate nested depressions across large DEMs using parallel overlapping tiles."""
    return _lidar_rs.delineate_depressions_tiled(
        _as_path_str(in_dem),
        _as_path_str(out_dir),
        int(tile_size),
        int(overlap),
        int(kernel_size),
        int(min_size),
        float(min_depth),
        float(interval),
        bool(use_f32),
    )


def PlayAnimation(
    in_dem: str | os.PathLike[str],
    in_flood_dir: str | os.PathLike[str],
    out_gif: str | os.PathLike[str] = "inundation.gif",
    frame_delay_ms: int = 200,
    ping_pong: bool = True,
    loops: int = 0,
    display: bool = False,
    **kwargs: Any,
) -> Any:
    """Render an inundation animation GIF; when `display=True` in Jupyter/Colab, returns `IPython.display.Image`."""
    res = animate_inundation(
        in_dem,
        in_flood_dir,
        out_gif,
        frame_delay_ms=frame_delay_ms,
        ping_pong=ping_pong,
        loops=loops,
        **kwargs,
    )
    if display:
        from IPython.display import Image

        return Image(filename=res["gif_path"])
    return str(res["gif_path"])


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
    "DelineateCatchments",
    "DelineateFlowPaths",
    "CatchmentHierarchy",
    "SimulateInundation",
    "PlayAnimation",
    "mean_filter",
    "median_filter",
    "gaussian_filter",
    "flip_dem",
    "extract_sinks",
    "extract_sinks_array",
    "delineate_depressions",
    "delineate_depressions_array",
    "delineate_depressions_tiled",
    "delineate_mounts",
    "delineate_catchments",
    "delineate_flow_paths",
    "delineate_catchment_hierarchy",
    "simulate_inundation",
    "simulate_inundation_array",
    "run_pipeline",
    "breach_depressions",
    "burn_culverts",
    "scs_curve_number_runoff",
    "scs_type2_hyetograph",
    "animate_inundation",
    "export_inundation_frames_png",
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
    "lidar_to_dtm",
    "lidar_to_chm",
]
