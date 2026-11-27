"""Pure-Rust port of `opengeos/lidar` for Python, powered by `terrio` and `richdem`."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _lidar_rs
from ._adapters import (
    array_to_xarray,
    connectivity_to_networkx,
    depressions_to_geodataframe,
    extract_spatial_array,
    is_spatial_array,
)
from ._types import (
    AnimateInundationResult,
    CatchmentHierarchyRecord,
    CatchmentHierarchyResult,
    CatchmentRecord,
    ConnectivityGraph,
    DelineateCatchmentsResult,
    DelineateFlowPathsResult,
    DelineateMountsResult,
    DelineationArrayResult,
    DelineationResult,
    DepressionRecord,
    Download3depResult,
    EmbankmentArtifactCandidate,
    ExtractSinksArrayResult,
    ExtractSinksResult,
    FederalDataResult,
    FlowPathRecord,
    FsmBasinSummary,
    FsmHydrographPoint,
    FsmMassBalance,
    Geotransform,
    GraphEdge,
    GraphNode,
    GroundClassifyResult,
    InundationArrayResult,
    InundationResult,
    InundationStepRecord,
    InundationStepSummaryRecord,
    MergeTreeNode,
    MergeTreeResult,
    NoaaAtlas14Table,
    PathLikeStr,
    PipelineResult,
    PointCloudDict,
    RatingCurve,
    RatingCurvePoint,
    RecessionCurvePoint,
    SinkDepressionRecord,
    SpatialArrayLike,
    SpillFlowPathRecord,
    TiledDelineationResult,
    TnmProductItem,
)

__version__: str = _lidar_rs.__version__


def _as_path_str(p: PathLikeStr) -> str:
    return os.fspath(p)


@overload
def MeanFilter(
    in_dem: PathLikeStr,
    kernel_size: int = 3,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def MeanFilter(
    in_dem: PathLikeStr,
    kernel_size: int,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def MeanFilter(
    in_dem: PathLikeStr | SpatialArrayLike,
    kernel_size: int = 3,
    out_file: None = None,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...


def MeanFilter(
    in_dem: PathLikeStr | SpatialArrayLike,
    kernel_size: int = 3,
    out_file: PathLikeStr | None = None,
    no_data: float | None = None,
) -> str | NDArray[np.float64]:
    """Apply a moving-window mean filter to a DEM file, 2D NumPy array, xarray DataArray, or rasterio dataset."""
    if is_spatial_array(in_dem) or not isinstance(in_dem, (str, os.PathLike)):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr, _gt, _crs, nd = extract_spatial_array(in_dem, no_data=no_data)
        return _lidar_rs.mean_filter_array(arr, kernel_size, nd)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.mean_filter_file(
        _as_path_str(in_dem), kernel_size, out_str, no_data
    )
    if out_str is not None:
        return out_str
    assert res is not None
    return res


@overload
def MedianFilter(
    in_dem: PathLikeStr,
    kernel_size: int = 3,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def MedianFilter(
    in_dem: PathLikeStr,
    kernel_size: int,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def MedianFilter(
    in_dem: PathLikeStr | SpatialArrayLike,
    kernel_size: int = 3,
    out_file: None = None,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...


def MedianFilter(
    in_dem: PathLikeStr | SpatialArrayLike,
    kernel_size: int = 3,
    out_file: PathLikeStr | None = None,
    no_data: float | None = None,
) -> str | NDArray[np.float64]:
    """Apply a moving-window median filter to a DEM file, 2D NumPy array, xarray DataArray, or rasterio dataset."""
    if is_spatial_array(in_dem) or not isinstance(in_dem, (str, os.PathLike)):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr, _gt, _crs, nd = extract_spatial_array(in_dem, no_data=no_data)
        return _lidar_rs.median_filter_array(arr, kernel_size, nd)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.median_filter_file(
        _as_path_str(in_dem), kernel_size, out_str, no_data
    )
    if out_str is not None:
        return out_str
    assert res is not None
    return res


@overload
def GaussianFilter(
    in_dem: PathLikeStr,
    sigma: float = 1.0,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def GaussianFilter(
    in_dem: PathLikeStr,
    sigma: float,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def GaussianFilter(
    in_dem: PathLikeStr | SpatialArrayLike,
    sigma: float = 1.0,
    out_file: None = None,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...


def GaussianFilter(
    in_dem: PathLikeStr | SpatialArrayLike,
    sigma: float = 1.0,
    out_file: PathLikeStr | None = None,
    no_data: float | None = None,
) -> str | NDArray[np.float64]:
    """Apply a 2D Gaussian filter to a DEM file, 2D NumPy array, xarray DataArray, or rasterio dataset."""
    if is_spatial_array(in_dem) or not isinstance(in_dem, (str, os.PathLike)):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr, _gt, _crs, nd = extract_spatial_array(in_dem, no_data=no_data)
        return _lidar_rs.gaussian_filter_array(arr, float(sigma), nd)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.gaussian_filter_file(
        _as_path_str(in_dem), float(sigma), out_str, no_data
    )
    if out_str is not None:
        return out_str
    assert res is not None
    return res


@overload
def bilateral_filter(
    in_dem: PathLikeStr,
    spatial_sigma: float = 1.5,
    range_sigma: float = 0.5,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def bilateral_filter(
    in_dem: PathLikeStr,
    spatial_sigma: float,
    range_sigma: float,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def bilateral_filter(
    in_dem: PathLikeStr | SpatialArrayLike,
    spatial_sigma: float = 1.5,
    range_sigma: float = 0.5,
    out_file: None = None,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...


def bilateral_filter(
    in_dem: PathLikeStr | SpatialArrayLike,
    spatial_sigma: float = 1.5,
    range_sigma: float = 0.5,
    out_file: PathLikeStr | None = None,
    no_data: float | None = None,
) -> str | NDArray[np.float64]:
    """Apply an edge-preserving 2D bilateral filter to a DEM file or 2D spatial array."""
    if is_spatial_array(in_dem) or not isinstance(in_dem, (str, os.PathLike)):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for array input; pass a raster file path or omit out_file"
            )
        arr, _gt, _crs, nd = extract_spatial_array(in_dem, no_data=no_data)
        return _lidar_rs.bilateral_filter_array(
            arr, float(spatial_sigma), float(range_sigma), nd
        )
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.bilateral_filter_file(
        _as_path_str(in_dem), float(spatial_sigma), float(range_sigma), out_str, no_data
    )
    if out_str is not None:
        return out_str
    assert res is not None
    return res


@overload
def feature_preserving_smoothing(
    in_dem: PathLikeStr,
    filter_size: int = 11,
    norm_diff: float = 15.0,
    num_iter: int = 3,
    max_z_change: float | None = None,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def feature_preserving_smoothing(
    in_dem: PathLikeStr,
    filter_size: int,
    norm_diff: float,
    num_iter: int,
    max_z_change: float | None,
    out_file: PathLikeStr,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def feature_preserving_smoothing(
    in_dem: PathLikeStr | SpatialArrayLike,
    filter_size: int = 11,
    norm_diff: float = 15.0,
    num_iter: int = 3,
    max_z_change: float | None = None,
    out_file: None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> NDArray[np.float64]: ...


def feature_preserving_smoothing(
    in_dem: PathLikeStr | SpatialArrayLike,
    filter_size: int = 11,
    norm_diff: float = 15.0,
    num_iter: int = 3,
    max_z_change: float | None = None,
    out_file: PathLikeStr | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str | NDArray[np.float64]:
    """Apply Lindsay et al. (2019) feature-preserving DEM smoothing to a file or 2D spatial array."""
    if is_spatial_array(in_dem) or not isinstance(in_dem, (str, os.PathLike)):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for array input; pass a raster file path or omit out_file"
            )
        arr, gt, resolved_crs, nd = extract_spatial_array(
            in_dem, geotransform=geotransform, crs=crs, no_data=no_data
        )
        return _lidar_rs.feature_preserving_smoothing_array(
            arr,
            int(filter_size),
            float(norm_diff),
            int(num_iter),
            float(max_z_change) if max_z_change is not None else None,
            nd,
            float(cell_size),
            gt,
            resolved_crs,
        )
    if geotransform is not None or crs is not None or cell_size != 1.0:
        raise ValueError(
            "cell_size, geotransform, and crs overrides are not supported when in_dem is a file path"
        )
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.feature_preserving_smoothing_file(
        _as_path_str(in_dem),
        int(filter_size),
        float(norm_diff),
        int(num_iter),
        float(max_z_change) if max_z_change is not None else None,
        out_str,
        no_data,
        bool(compress),
    )
    if out_str is not None:
        return out_str
    assert res is not None
    return res


@overload
def FlipDEM(
    dem: PathLikeStr,
    delta: float = 100.0,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def FlipDEM(
    dem: PathLikeStr,
    delta: float,
    out_file: PathLikeStr,
    no_data: float | None = None,
) -> str: ...


@overload
def FlipDEM(
    dem: PathLikeStr | SpatialArrayLike,
    delta: float = 100.0,
    out_file: None = None,
    no_data: float | None = None,
) -> NDArray[np.float64]: ...


def FlipDEM(
    dem: PathLikeStr | SpatialArrayLike,
    delta: float = 100.0,
    out_file: PathLikeStr | None = None,
    no_data: float | None = None,
) -> str | NDArray[np.float64]:
    """Flip a DEM upside-down so elevated features become depressions."""
    if is_spatial_array(dem) or not isinstance(dem, (str, os.PathLike)):
        if out_file is not None:
            raise ValueError(
                "out_file is not supported for NumPy array input; pass a raster file path or omit out_file"
            )
        arr, _gt, _crs, nd = extract_spatial_array(dem, no_data=no_data)
        return _lidar_rs.flip_dem_array(arr, float(delta), nd)
    out_str = _as_path_str(out_file) if out_file is not None else None
    res = _lidar_rs.flip_dem_file(_as_path_str(dem), float(delta), out_str, no_data)
    if out_str is not None:
        return out_str
    assert res is not None
    return res


def extract_sinks(
    in_dem: PathLikeStr,
    min_size: int,
    out_dir: PathLikeStr,
    filled_dem: PathLikeStr | None = None,
    topology: str = "D8",
    keep_files: bool = True,
    compress: bool = False,
) -> ExtractSinksResult:
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
    in_dem: PathLikeStr,
    min_size: int,
    out_dir: PathLikeStr,
    filled_dem: PathLikeStr | None = None,
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
    in_sink: PathLikeStr,
    min_size: int,
    min_depth: float,
    interval: float,
    out_dir: PathLikeStr,
    bool_level_shp: bool = False,
    bool_comb_tif: bool = False,
    compress: bool = False,
    export_gpkg: bool = False,
) -> DelineationResult:
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
        export_gpkg,
    )


def DelineateDepressions(
    in_sink: PathLikeStr,
    min_size: int,
    min_depth: float,
    interval: float,
    out_dir: PathLikeStr,
    bool_level_shp: bool = False,
) -> tuple[str, str]:
    """Drop-in replacement for `lidar.DelineateDepressions`; returns `(dep_id_path, dep_level_path)`."""
    res = delineate_depressions(
        in_sink, min_size, min_depth, interval, out_dir, bool_level_shp=bool_level_shp
    )
    return str(res["depression_id_path"]), str(res["depression_level_path"])


def delineate_mounts(
    in_dem: PathLikeStr,
    min_size: int,
    min_height: float,
    interval: float,
    out_dir: PathLikeStr,
    bool_shp: bool = False,
    compress: bool = False,
    export_gpkg: bool = False,
    bool_comb_tif: bool = False,
) -> DelineateMountsResult:
    """Delineate nested hierarchy of elevated features (mounts)."""
    return _lidar_rs.delineate_mounts(
        _as_path_str(in_dem),
        int(min_size),
        float(min_height),
        float(interval),
        _as_path_str(out_dir),
        bool_shp,
        compress,
        export_gpkg,
        bool_comb_tif,
    )


def DelineateMounts(
    in_dem: PathLikeStr,
    min_size: int,
    min_height: float,
    interval: float,
    out_dir: PathLikeStr,
    bool_shp: bool = False,
) -> tuple[str, str]:
    """Drop-in replacement for `lidar.DelineateMounts`; returns `(mount_id_path, mount_level_path)`."""
    res = delineate_mounts(
        in_dem, min_size, min_height, interval, out_dir, bool_shp=bool_shp
    )
    return str(res["depression_id_path"]), str(res["depression_level_path"])


def polygonize(
    in_raster: PathLikeStr,
    out_vector: PathLikeStr,
    field_name: str = "id",
) -> int:
    """Polygonize a raster band into a vector file (Shapefile, GeoPackage, GeoJSON, etc.)."""
    return int(
        _lidar_rs.polygonize(
            _as_path_str(in_raster), _as_path_str(out_vector), field_name
        )
    )


def mosaic(
    images: PathLikeStr | Sequence[PathLikeStr],
    output: PathLikeStr,
    ext: str = ".tif",
    *,
    resampling: str = "nearest",
    compress: bool = False,
    cog: bool = False,
) -> None:
    """Mosaic a list of rasters (or all raster files in a directory) into a single GeoTIFF or COG."""
    out_resolved = Path(output).resolve()
    if isinstance(images, (str, os.PathLike)):
        path = Path(images)
        if path.is_file():
            inputs = [str(path)]
        elif path.is_dir():
            ext_norm = ext if ext.startswith(".") else f".{ext}"
            ext_lower = ext_norm.lower()
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
    _lidar_rs.mosaic(inputs, _as_path_str(output), resampling, bool(compress), bool(cog))


def reproject_image(
    image: PathLikeStr,
    output: PathLikeStr,
    dst_crs: str = "EPSG:4326",
    resampling: str = "nearest",
    resolution: tuple[float, float] | None = None,
    *,
    compress: bool = False,
    cog: bool = False,
) -> None:
    """Reproject a raster to `dst_crs`."""
    _lidar_rs.reproject_image(
        _as_path_str(image),
        _as_path_str(output),
        dst_crs,
        resampling,
        resolution,
        bool(compress),
        bool(cog),
    )


def resample(
    src: PathLikeStr,
    dst: PathLikeStr,
    resolution: float,
    resampling: str = "bilinear",
    *,
    compress: bool = False,
    cog: bool = False,
) -> None:
    """Resample a raster to a target pixel resolution."""
    _lidar_rs.resample(
        _as_path_str(src),
        _as_path_str(dst),
        float(resolution),
        resampling,
        bool(compress),
        bool(cog),
    )


def clip_image(
    image: PathLikeStr,
    mask: PathLikeStr,
    output: PathLikeStr,
    *,
    compress: bool = False,
    cog: bool = False,
) -> None:
    """Clip a raster by a polygon vector mask."""
    _lidar_rs.clip_image(
        _as_path_str(image),
        _as_path_str(mask),
        _as_path_str(output),
        bool(compress),
        bool(cog),
    )


def image_to_cog(
    source: PathLikeStr,
    dst_path: PathLikeStr | None = None,
    compression: str = "DEFLATE",
) -> str:
    """Convert a raster to a Cloud Optimized GeoTIFF (COG)."""
    src = Path(source)
    dst = (
        Path(dst_path) if dst_path is not None else src.with_name(f"{src.stem}_cog.tif")
    )
    _lidar_rs.image_to_cog(str(src), str(dst), compression)
    return str(dst)


def add_crs(
    filename: PathLikeStr,
    epsg: int | str,
    output: PathLikeStr | None = None,
) -> None:
    """Assign a CRS (EPSG code, WKT, or PROJ string) to a raster."""
    crs_str = f"EPSG:{epsg}" if isinstance(epsg, int) else str(epsg)
    out_str = _as_path_str(output) if output is not None else None
    _lidar_rs.add_crs(_as_path_str(filename), crs_str, out_str)


def join_tables(
    in_shp: PathLikeStr,
    in_csv: PathLikeStr,
    out_shp: PathLikeStr,
    vec_col: str = "id",
    csv_col: str = "region_id",
) -> int:
    """Join a CSV attribute table onto a vector file."""
    return int(
        _lidar_rs.join_tables(
            _as_path_str(in_shp),
            _as_path_str(in_csv),
            _as_path_str(out_shp),
            vec_col,
            csv_col,
        )
    )


def csv_to_shp(
    in_csv: PathLikeStr,
    out_shp: PathLikeStr,
    latitude: str = "latitude",
    longitude: str = "longitude",
) -> int:
    """Convert a CSV file with latitude/longitude columns into a WGS 84 point vector file."""
    return int(
        _lidar_rs.csv_to_vector(
            _as_path_str(in_csv), _as_path_str(out_shp), latitude, longitude
        )
    )


csv_points_to_shp = csv_to_shp


def convert_lidar(
    source: PathLikeStr,
    destination: PathLikeStr,
    point_format_id: int | None = None,
    minor_version: int | None = None,
) -> int:
    """Convert a LiDAR point cloud between formats (LAS, LAZ, COPC, PLY, E57)."""
    return int(
        _lidar_rs.convert_lidar(
            _as_path_str(source),
            _as_path_str(destination),
            point_format_id,
            minor_version,
        )
    )


def _as_uint(
    arr: ArrayLike, dtype: type[np.uint8 | np.uint16], name: str
) -> NDArray[Any]:
    raw = np.asarray(arr).ravel()
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(
            f"{name} must contain integer values, got dtype {raw.dtype}"
        )
    info = np.iinfo(dtype)
    if raw.size > 0:
        lo = int(raw.min())
        hi = int(raw.max())
        if lo < info.min or hi > info.max:
            raise ValueError(
                f"{name} values must be in [{info.min}, {info.max}], got range [{lo}, {hi}]"
            )
    return np.ascontiguousarray(raw, dtype=dtype)


def read_lidar(
    path: PathLikeStr,
    bounds: tuple[float, float, float, float] | Sequence[float] | None = None,
    polygon_path: PathLikeStr | None = None,
    max_points: int | None = None,
    *,
    min_z: float | None = None,
    max_z: float | None = None,
    classifications: Sequence[int] | None = None,
    return_kinds: str | Sequence[str] | None = None,
    max_level: int | None = None,
    resolution: float | None = None,
) -> PointCloudDict:
    """Read a LiDAR point cloud (LAS, LAZ, COPC, PLY, E57, EPT) into a NumPy-backed dictionary."""
    norm_bounds = _normalize_bbox(bounds, names=("min_x", "min_y", "max_x", "max_y"))
    return _lidar_rs.read_lidar(
        _as_path_str(path),
        norm_bounds,
        _as_path_str(polygon_path) if polygon_path is not None else None,
        int(max_points) if max_points is not None else None,
        float(min_z) if min_z is not None else None,
        float(max_z) if max_z is not None else None,
        [int(c) for c in classifications] if classifications is not None else None,
        list((return_kinds,) if isinstance(return_kinds, str) else return_kinds)
        if return_kinds is not None
        else None,
        int(max_level) if max_level is not None else None,
        float(resolution) if resolution is not None else None,
    )


def write_lidar(
    path: PathLikeStr,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    classification: ArrayLike | None = None,
    intensity: ArrayLike | None = None,
    return_number: ArrayLike | None = None,
    number_of_returns: ArrayLike | None = None,
    crs: str | None = None,
) -> int:
    """Write 1D NumPy coordinate and attribute arrays to a LAS, LAZ, COPC, PLY, or E57 file."""
    x_arr = np.ascontiguousarray(x, dtype=np.float64).ravel()
    y_arr = np.ascontiguousarray(y, dtype=np.float64).ravel()
    z_arr = np.ascontiguousarray(z, dtype=np.float64).ravel()
    cls_arr = (
        _as_uint(classification, np.uint8, "classification")
        if classification is not None
        else None
    )
    int_arr = (
        _as_uint(intensity, np.uint16, "intensity")
        if intensity is not None
        else None
    )
    rn_arr = (
        _as_uint(return_number, np.uint8, "return_number")
        if return_number is not None
        else None
    )
    nr_arr = (
        _as_uint(number_of_returns, np.uint8, "number_of_returns")
        if number_of_returns is not None
        else None
    )
    return int(
        _lidar_rs.write_lidar(
            _as_path_str(path),
            x_arr,
            y_arr,
            z_arr,
            cls_arr,
            int_arr,
            rn_arr,
            nr_arr,
            crs,
        )
    )


def classify_ground(
    source: PathLikeStr | Mapping[str, Any],
    output: PathLikeStr | None = None,
    method: str = "csf",
    *,
    ground_class: int = 2,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
) -> GroundClassifyResult:
    """Classify ground points in a LiDAR file or `PointCloudDict` using tunable CSF, PMF, or SMRF parameters.

    Parameters
    ----------
    source:
        Path to a point cloud file (LAS, LAZ, COPC, PLY, E57) or an in-memory `PointCloudDict`
        returned by `read_lidar`.
    output:
        Optional output point cloud path to write the classified cloud to disk.
    method:
        Ground classification algorithm: `"csf"`, `"pmf"`, or `"smrf"`.
    ground_class:
        ASPRS classification code assigned to ground points (default `2`).
    cloth_resolution, rigidness, class_threshold, slope_smooth, iterations, time_step:
        Tunable Cloth Simulation Filter (CSF) parameters.
    max_window_size, slope, initial_distance, max_distance, cell_size:
        Tunable Progressive Morphological Filter (PMF) parameters.
    cell_size, slope, window_radius, elevation_threshold, elevation_scalar:
        Tunable Simple Morphological Filter (SMRF) parameters.
    """
    out_str = _as_path_str(output) if output is not None else None
    if isinstance(source, Mapping):
        x_arr = np.ascontiguousarray(source["x"], dtype=np.float64).ravel()
        y_arr = np.ascontiguousarray(source["y"], dtype=np.float64).ravel()
        z_arr = np.ascontiguousarray(source["z"], dtype=np.float64).ravel()
        cls_raw = source.get("classification")
        int_raw = source.get("intensity")
        rn_raw = source.get("return_number")
        nr_raw = source.get("number_of_returns")
        crs_val = source.get("crs")
        return _lidar_rs.classify_ground_array(
            x_arr,
            y_arr,
            z_arr,
            _as_uint(cls_raw, np.uint8, "classification")
            if cls_raw is not None
            else None,
            _as_uint(int_raw, np.uint16, "intensity")
            if int_raw is not None
            else None,
            _as_uint(rn_raw, np.uint8, "return_number")
            if rn_raw is not None
            else None,
            _as_uint(nr_raw, np.uint8, "number_of_returns")
            if nr_raw is not None
            else None,
            str(crs_val) if crs_val is not None else None,
            out_str,
            method,
            int(ground_class),
            float(cloth_resolution) if cloth_resolution is not None else None,
            int(rigidness) if rigidness is not None else None,
            float(class_threshold) if class_threshold is not None else None,
            bool(slope_smooth) if slope_smooth is not None else None,
            int(iterations) if iterations is not None else None,
            float(time_step) if time_step is not None else None,
            float(max_window_size) if max_window_size is not None else None,
            float(initial_distance) if initial_distance is not None else None,
            float(max_distance) if max_distance is not None else None,
            float(cell_size) if cell_size is not None else None,
            float(slope) if slope is not None else None,
            float(window_radius) if window_radius is not None else None,
            float(elevation_threshold) if elevation_threshold is not None else None,
            float(elevation_scalar) if elevation_scalar is not None else None,
        )

    return _lidar_rs.classify_ground_file(
        _as_path_str(source),
        out_str,
        method,
        int(ground_class),
        float(cloth_resolution) if cloth_resolution is not None else None,
        int(rigidness) if rigidness is not None else None,
        float(class_threshold) if class_threshold is not None else None,
        bool(slope_smooth) if slope_smooth is not None else None,
        int(iterations) if iterations is not None else None,
        float(time_step) if time_step is not None else None,
        float(max_window_size) if max_window_size is not None else None,
        float(initial_distance) if initial_distance is not None else None,
        float(max_distance) if max_distance is not None else None,
        float(cell_size) if cell_size is not None else None,
        float(slope) if slope is not None else None,
        float(window_radius) if window_radius is not None else None,
        float(elevation_threshold) if elevation_threshold is not None else None,
        float(elevation_scalar) if elevation_scalar is not None else None,
    )


def lidar_to_dsm(
    filename: PathLikeStr,
    output: PathLikeStr,
    resolution: float = 1.0,
    radius: float | None = None,
    minz: float | None = None,
    maxz: float | None = None,
    stat: str = "max",
    method: str = "statistics",
    window_size: int = 0,
    classifications: Sequence[int] | None = None,
    return_kinds: Sequence[str] | None = None,
    *,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]:
    """Grid a LiDAR point cloud into a Digital Surface Model (DSM) GeoTIFF or COG."""
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
        bool(compress),
        bool(cog),
    )
    return int(cols), int(rows)


def lidar_to_dtm(
    filename: PathLikeStr,
    output: PathLikeStr,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    radius: float | None = None,
    method: str = "linear",
    stat: str = "idw",
    window_size: int = 5,
    *,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]:
    """Generate a bare-earth DTM GeoTIFF (or COG) from a LiDAR point cloud."""
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
        float(cloth_resolution) if cloth_resolution is not None else None,
        int(rigidness) if rigidness is not None else None,
        float(class_threshold) if class_threshold is not None else None,
        bool(slope_smooth) if slope_smooth is not None else None,
        int(iterations) if iterations is not None else None,
        float(time_step) if time_step is not None else None,
        float(max_window_size) if max_window_size is not None else None,
        float(initial_distance) if initial_distance is not None else None,
        float(max_distance) if max_distance is not None else None,
        float(cell_size) if cell_size is not None else None,
        float(slope) if slope is not None else None,
        float(window_radius) if window_radius is not None else None,
        float(elevation_threshold) if elevation_threshold is not None else None,
        float(elevation_scalar) if elevation_scalar is not None else None,
        bool(remove_outliers),
        float(ror_radius) if ror_radius is not None else None,
        int(ror_min_neighbors) if ror_min_neighbors is not None else None,
        int(sor_k) if sor_k is not None else None,
        float(sor_sigma) if sor_sigma is not None else None,
        bool(compress),
        bool(cog),
    )
    return int(cols), int(rows)


def lidar_to_chm(
    filename: PathLikeStr,
    output: PathLikeStr,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    radius: float | None = None,
    window_size: int = 5,
    *,
    cloth_resolution: float | None = None,
    rigidness: int | None = None,
    class_threshold: float | None = None,
    slope_smooth: bool | None = None,
    iterations: int | None = None,
    time_step: float | None = None,
    max_window_size: float | None = None,
    initial_distance: float | None = None,
    max_distance: float | None = None,
    cell_size: float | None = None,
    slope: float | None = None,
    window_radius: float | None = None,
    elevation_threshold: float | None = None,
    elevation_scalar: float | None = None,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]:
    """Generate a Canopy Height Model (CHM = DSM - DTM) GeoTIFF (or COG) from a LiDAR point cloud."""
    cols, rows = _lidar_rs.lidar_to_chm(
        _as_path_str(filename),
        _as_path_str(output),
        float(resolution),
        ground_method,
        int(ground_class),
        radius,
        int(window_size),
        float(cloth_resolution) if cloth_resolution is not None else None,
        int(rigidness) if rigidness is not None else None,
        float(class_threshold) if class_threshold is not None else None,
        bool(slope_smooth) if slope_smooth is not None else None,
        int(iterations) if iterations is not None else None,
        float(time_step) if time_step is not None else None,
        float(max_window_size) if max_window_size is not None else None,
        float(initial_distance) if initial_distance is not None else None,
        float(max_distance) if max_distance is not None else None,
        float(cell_size) if cell_size is not None else None,
        float(slope) if slope is not None else None,
        float(window_radius) if window_radius is not None else None,
        float(elevation_threshold) if elevation_threshold is not None else None,
        float(elevation_scalar) if elevation_scalar is not None else None,
        bool(remove_outliers),
        float(ror_radius) if ror_radius is not None else None,
        int(ror_min_neighbors) if ror_min_neighbors is not None else None,
        int(sor_k) if sor_k is not None else None,
        float(sor_sigma) if sor_sigma is not None else None,
        bool(compress),
        bool(cog),
    )
    return int(cols), int(rows)


def lidar_to_chm_pit_free(
    filename: PathLikeStr,
    output: PathLikeStr,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    *,
    height_thresholds: Sequence[float] | None = None,
    max_triangle_edge: float | None = None,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
) -> tuple[int, int]:
    """Generate a Khosravipour et al. (2014) pit-free multi-threshold TIN CHM GeoTIFF (or COG)."""
    cols, rows = _lidar_rs.lidar_to_chm_pit_free(
        _as_path_str(filename),
        _as_path_str(output),
        float(resolution),
        ground_method,
        int(ground_class),
        [float(v) for v in height_thresholds]
        if height_thresholds is not None
        else None,
        float(max_triangle_edge) if max_triangle_edge is not None else None,
        bool(remove_outliers),
        float(ror_radius) if ror_radius is not None else None,
        int(ror_min_neighbors) if ror_min_neighbors is not None else None,
        int(sor_k) if sor_k is not None else None,
        float(sor_sigma) if sor_sigma is not None else None,
        bool(compress),
        bool(cog),
    )
    return int(cols), int(rows)


def lidar_to_dtm_tiled(
    filename: PathLikeStr,
    output: PathLikeStr,
    tile_size: float = 500.0,
    overlap: float = 25.0,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    method: str = "linear",
    stat: str = "idw",
    window_size: int = 5,
    *,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
    read_resolution: float | None = None,
    max_level: int | None = None,
) -> tuple[int, int]:
    """Generate a bare-earth DTM GeoTIFF (or streaming COG) from a large point cloud in overlapping tiles."""
    cols, rows = _lidar_rs.lidar_to_dtm_tiled(
        _as_path_str(filename),
        _as_path_str(output),
        float(tile_size),
        float(overlap),
        float(resolution),
        ground_method,
        int(ground_class),
        method,
        stat,
        int(window_size),
        bool(remove_outliers),
        float(ror_radius) if ror_radius is not None else None,
        int(ror_min_neighbors) if ror_min_neighbors is not None else None,
        int(sor_k) if sor_k is not None else None,
        float(sor_sigma) if sor_sigma is not None else None,
        bool(compress),
        bool(cog),
        float(read_resolution) if read_resolution is not None else None,
        int(max_level) if max_level is not None else None,
    )
    return int(cols), int(rows)


def lidar_to_chm_tiled(
    filename: PathLikeStr,
    output: PathLikeStr,
    tile_size: float = 500.0,
    overlap: float = 25.0,
    resolution: float = 1.0,
    ground_method: str = "csf",
    ground_class: int = 2,
    window_size: int = 5,
    *,
    remove_outliers: bool = False,
    ror_radius: float | None = None,
    ror_min_neighbors: int | None = None,
    sor_k: int | None = None,
    sor_sigma: float | None = None,
    compress: bool = False,
    cog: bool = False,
    read_resolution: float | None = None,
    max_level: int | None = None,
) -> tuple[int, int]:
    """Generate a CHM GeoTIFF (or streaming COG) from a large point cloud in overlapping tiles."""
    cols, rows = _lidar_rs.lidar_to_chm_tiled(
        _as_path_str(filename),
        _as_path_str(output),
        float(tile_size),
        float(overlap),
        float(resolution),
        ground_method,
        int(ground_class),
        int(window_size),
        bool(remove_outliers),
        float(ror_radius) if ror_radius is not None else None,
        int(ror_min_neighbors) if ror_min_neighbors is not None else None,
        int(sor_k) if sor_k is not None else None,
        float(sor_sigma) if sor_sigma is not None else None,
        bool(compress),
        bool(cog),
        float(read_resolution) if read_resolution is not None else None,
        int(max_level) if max_level is not None else None,
    )
    return int(cols), int(rows)


def stream_remote_pointcloud_aoi(
    source_url: str,
    *,
    bounds_wgs84: tuple[float, float, float, float] | Sequence[float] | None = None,
    bounds_native: tuple[float, float, float, float] | Sequence[float] | None = None,
    classifications: Sequence[int] | None = None,
    return_kinds: str | Sequence[str] | None = None,
    max_level: int | None = None,
    resolution: float | None = None,
    max_points: int | None = None,
) -> PointCloudDict:
    """Stream an AOI from a remote public HTTPS COPC or EPT point cloud dataset.

    Performs a lexical-only pre-flight check on `source_url` (DNS resolution, HTTP redirects,
    and child EPT manifest URLs are fetched directly by `terrio`; enforce egress controls at the
    deployment layer for untrusted URLs). Supply `bounds_wgs84`, `bounds_native`, or `max_points`
    to bound memory usage instead of streaming an entire remote dataset.
    """
    return _lidar_rs.stream_remote_pointcloud_aoi(
        str(source_url),
        _normalize_bbox(bounds_wgs84),
        _normalize_bbox(
            bounds_native, names=("min_x", "min_y", "max_x", "max_y")
        ),
        [int(c) for c in classifications] if classifications is not None else None,
        list((return_kinds,) if isinstance(return_kinds, str) else return_kinds)
        if return_kinds is not None
        else None,
        int(max_level) if max_level is not None else None,
        float(resolution) if resolution is not None else None,
        int(max_points) if max_points is not None else None,
    )


def delineate_catchments(
    in_dem: PathLikeStr,
    in_sink: PathLikeStr,
    out_dir: PathLikeStr,
    compress: bool = False,
) -> DelineateCatchmentsResult:
    """Delineate contributing catchments for extracted sinks (ports 2_Delineate_Catchment.py)."""
    return _lidar_rs.delineate_catchments(
        _as_path_str(in_dem),
        _as_path_str(in_sink),
        _as_path_str(out_dir),
        compress,
    )


def DelineateCatchments(
    in_dem: PathLikeStr,
    in_sink: PathLikeStr,
    out_dir: PathLikeStr,
) -> str:
    """ArcGIS-style wrapper for `delineate_catchments`; returns the catchment vector path."""
    return str(delineate_catchments(in_dem, in_sink, out_dir)["vector_path"])


def delineate_flow_paths(
    in_dem_filled: PathLikeStr,
    in_sink: PathLikeStr,
    out_flowpath: PathLikeStr,
    rain_intensity_cm_h: float = 5.0,
    refined_dem: PathLikeStr | None = None,
) -> DelineateFlowPathsResult:
    """Delineate overland spill flow paths between sinks (ports 3_Flow_Path.py)."""
    return _lidar_rs.delineate_flow_paths(
        _as_path_str(in_dem_filled),
        _as_path_str(in_sink),
        _as_path_str(out_flowpath),
        float(rain_intensity_cm_h),
        _as_path_str(refined_dem) if refined_dem is not None else None,
    )


def DelineateFlowPaths(
    in_dem_filled: PathLikeStr,
    in_sink: PathLikeStr,
    out_flowpath: PathLikeStr,
    rain_intensity_cm_h: float = 5.0,
    refined_dem: PathLikeStr | None = None,
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
    in_dem: PathLikeStr,
    in_level_dir: PathLikeStr,
    out_catchment_img: PathLikeStr,
    compress: bool = False,
) -> CatchmentHierarchyResult:
    """Delineate nested catchment hierarchy across depression levels (ports 5_Catchment_Hierarchy.py)."""
    return _lidar_rs.delineate_catchment_hierarchy(
        _as_path_str(in_dem),
        _as_path_str(in_level_dir),
        _as_path_str(out_catchment_img),
        compress,
    )


def CatchmentHierarchy(
    in_dem: PathLikeStr,
    in_level_dir: PathLikeStr,
    out_catchment_img: PathLikeStr,
) -> str:
    """ArcGIS-style wrapper for `delineate_catchment_hierarchy`; returns `catchment_hir.tif` path."""
    return str(
        delineate_catchment_hierarchy(in_dem, in_level_dir, out_catchment_img)[
            "catchment_raster_path"
        ]
    )


def simulate_inundation(
    in_sink: PathLikeStr,
    in_catchment: PathLikeStr,
    out_dir: PathLikeStr,
    min_size: int = 1000,
    min_depth: float = 0.2,
    interval: float = 0.2,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    compress: bool = False,
) -> InundationResult:
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
    in_sink: PathLikeStr,
    in_catchment: PathLikeStr,
    out_dir: PathLikeStr,
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
    in_dem: PathLikeStr,
    in_flood_dir: PathLikeStr,
    out_gif: PathLikeStr,
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
) -> AnimateInundationResult:
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
    in_dem: PathLikeStr,
    in_fill_time: PathLikeStr,
    out_dir: PathLikeStr,
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


@overload
def breach_depressions(
    in_dem: PathLikeStr,
    out_dem: PathLikeStr,
    mode: str = "selective",
    max_depth: float | None = None,
    max_length: int | None = None,
    fill_remaining: bool = False,
    compress: bool = False,
    no_data: float | None = None,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> str: ...


@overload
def breach_depressions(
    in_dem: SpatialArrayLike,
    out_dem: None = None,
    mode: str = "selective",
    max_depth: float | None = None,
    max_length: int | None = None,
    fill_remaining: bool = False,
    compress: bool = False,
    no_data: float | None = None,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> NDArray[np.float64]: ...


def breach_depressions(
    in_dem: PathLikeStr | SpatialArrayLike,
    out_dem: PathLikeStr | None = None,
    mode: str = "selective",
    max_depth: float | None = None,
    max_length: int | None = None,
    fill_remaining: bool = False,
    compress: bool = False,
    no_data: float | None = None,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> str | NDArray[np.float64]:
    """Breach depressions using Lindsay (2016) on a GeoTIFF path, 2D NumPy array, xarray DataArray, or rasterio dataset.

    Note: `geotransform` and `crs` apply only to in-memory spatial array inputs; passing them
    with a raster file path raises `ValueError` because spatial metadata is read from the file.
    """
    if is_spatial_array(in_dem) or not isinstance(in_dem, (str, os.PathLike)):
        if out_dem is not None:
            raise ValueError(
                "out_dem is not supported for NumPy array input; pass a raster file path or omit out_dem"
            )
        arr, gt, resolved_crs, nd = extract_spatial_array(
            in_dem, geotransform=geotransform, crs=crs, no_data=no_data
        )
        return _lidar_rs.breach_depressions_array(
            arr,
            mode,
            max_depth,
            max_length,
            bool(fill_remaining),
            nd,
            gt,
            resolved_crs,
        )
    if out_dem is None:
        raise ValueError("out_dem is required when in_dem is a file path")
    if geotransform is not None or crs is not None:
        raise ValueError(
            "geotransform and crs overrides are not supported when in_dem is a file path; "
            "geotransform and CRS are read from the input raster file"
        )
    return str(
        _lidar_rs.breach_depressions(
            _as_path_str(in_dem),
            _as_path_str(out_dem),
            mode,
            max_depth,
            max_length,
            bool(fill_remaining),
            bool(compress),
            no_data,
        )
    )


def burn_culverts(
    in_dem: PathLikeStr,
    culverts_vector: PathLikeStr,
    out_dem: PathLikeStr,
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
    cn_arr: SpatialArrayLike,
    precip_mm: float,
    no_data: float | None = None,
) -> NDArray[np.float64]:
    """Convert an NRCS Curve Number 2D array and rainfall depth (mm) into runoff coefficients (0..1)."""
    arr, _gt, _crs, nd = extract_spatial_array(cn_arr, no_data=no_data)
    return _lidar_rs.scs_curve_number_runoff(arr, float(precip_mm), nd)


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
    dem_arr: SpatialArrayLike,
    min_size: int = 1000,
    topology: str = "D8",
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> ExtractSinksArrayResult:
    """Extract surface sinks completely in memory from a 2D array, xarray DataArray, or rasterio dataset."""
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem_arr, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.extract_sinks_array(
        arr, int(min_size), topology, nd, float(cell_size), gt, resolved_crs
    )


def delineate_depressions_array(
    sink_arr: SpatialArrayLike,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> DelineationArrayResult:
    """Delineate nested depression hierarchy completely in memory from a 2D array, xarray DataArray, or rasterio dataset."""
    arr, gt, resolved_crs, nd = extract_spatial_array(
        sink_arr, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.delineate_depressions_array(
        arr,
        int(min_size),
        float(min_depth),
        float(interval),
        nd,
        float(cell_size),
        gt,
        resolved_crs,
    )


def delineate_mounts_array(
    dsm: SpatialArrayLike,
    min_size: int = 1000,
    min_height: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> DelineationArrayResult:
    """Delineate nested hierarchy of elevated features (mounts) completely in memory."""
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dsm, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.delineate_mounts_array(
        arr,
        int(min_size),
        float(min_height),
        float(interval),
        nd,
        float(cell_size),
        gt,
        resolved_crs,
    )


def compute_hillshade(
    dem: PathLikeStr | SpatialArrayLike,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    no_data: float | None = None,
    crs: str | None = None,
) -> NDArray[np.uint8]:
    """Compute an RGB shaded-relief array `(rows, cols, 3)` (`uint8`) from a DEM file or 2D spatial array.

    Note: `geotransform` and `crs` apply only to in-memory spatial array inputs; passing them
    with a raster file path raises `ValueError` because spatial metadata is read from the file.
    """
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None:
            raise ValueError(
                "geotransform and crs overrides are not supported when dem is a file path; "
                "geotransform and CRS are read from the input raster file"
            )
        return _lidar_rs.compute_hillshade_file(
            _as_path_str(dem),
            float(azimuth),
            float(altitude),
            float(z_factor),
            no_data,
        )
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.compute_hillshade_array(
        arr,
        float(azimuth),
        float(altitude),
        float(z_factor),
        float(cell_size),
        gt,
        nd,
        resolved_crs,
    )


def _check_aux_grid_alignment(
    name: str,
    aux_gt: Geotransform | None,
    aux_crs: str | None,
    dem_gt: Geotransform | None,
    dem_crs: str | None,
    *,
    dem_shape: tuple[int, int] | None = None,
    cell_size: float = 1.0,
) -> None:
    """Validate that an auxiliary spatial input's geotransform and CRS match the DEM."""
    effective_dem_gt = dem_gt
    if effective_dem_gt is None and dem_shape is not None:
        effective_dem_gt = (
            0.0,
            float(cell_size),
            0.0,
            float(dem_shape[0]) * float(cell_size),
            0.0,
            -float(cell_size),
        )
    if (
        aux_gt is not None
        and effective_dem_gt is not None
        and not np.allclose(aux_gt, effective_dem_gt, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(
            f"{name} geotransform {aux_gt} does not match DEM geotransform {effective_dem_gt}"
        )
    if aux_crs is not None:
        if dem_crs is None:
            raise ValueError(
                f"{name} carries CRS {aux_crs!r} but DEM has no CRS; provide matching crs on DEM"
            )
        if aux_crs.strip() != dem_crs.strip():
            raise ValueError(
                f"{name} CRS {aux_crs!r} does not match DEM CRS {dem_crs!r}"
            )


@overload
def dinf_flow_direction(
    dem: PathLikeStr,
    out_file: PathLikeStr,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def dinf_flow_direction(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> NDArray[np.float64]: ...


def dinf_flow_direction(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: PathLikeStr | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str | NDArray[np.float64]:
    """Compute Tarboton (1997) D-infinity flow direction (radians in `[0, 2*pi)`)."""
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None or cell_size != 1.0:
            raise ValueError(
                "cell_size, geotransform, and crs overrides are not supported when dem is a file path"
            )
        out_str = _as_path_str(out_file) if out_file is not None else None
        res = _lidar_rs.dinf_flow_direction_file(
            _as_path_str(dem), out_str, no_data, bool(compress)
        )
        if out_str is not None:
            return out_str
        assert res is not None
        return res
    if out_file is not None:
        raise ValueError("out_file is not supported for array input")
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.dinf_flow_direction_array(
        arr, nd, float(cell_size), gt, resolved_crs
    )


@overload
def dinf_flow_accumulation(
    dem: PathLikeStr,
    out_file: PathLikeStr,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def dinf_flow_accumulation(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> NDArray[np.float64]: ...


def dinf_flow_accumulation(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: PathLikeStr | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str | NDArray[np.float64]:
    """Compute Tarboton (1997) D-infinity flow accumulation (contributing area in cells)."""
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None or cell_size != 1.0:
            raise ValueError(
                "cell_size, geotransform, and crs overrides are not supported when dem is a file path"
            )
        out_str = _as_path_str(out_file) if out_file is not None else None
        res = _lidar_rs.dinf_flow_accumulation_file(
            _as_path_str(dem), out_str, no_data, bool(compress)
        )
        if out_str is not None:
            return out_str
        assert res is not None
        return res
    if out_file is not None:
        raise ValueError("out_file is not supported for array input")
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.dinf_flow_accumulation_array(
        arr, nd, float(cell_size), gt, resolved_crs
    )


@overload
def mfd_flow_accumulation(
    dem: PathLikeStr,
    exponent: float = 1.1,
    *,
    out_file: PathLikeStr,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def mfd_flow_accumulation(
    dem: PathLikeStr,
    exponent: float,
    out_file: PathLikeStr,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def mfd_flow_accumulation(
    dem: PathLikeStr | SpatialArrayLike,
    exponent: float = 1.1,
    out_file: None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> NDArray[np.float64]: ...


def mfd_flow_accumulation(
    dem: PathLikeStr | SpatialArrayLike,
    exponent: float = 1.1,
    out_file: PathLikeStr | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str | NDArray[np.float64]:
    """Compute Multiple Flow Direction (MFD; Quinn 1991 / Freeman 1991) flow accumulation."""
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None or cell_size != 1.0:
            raise ValueError(
                "cell_size, geotransform, and crs overrides are not supported when dem is a file path"
            )
        out_str = _as_path_str(out_file) if out_file is not None else None
        res = _lidar_rs.mfd_flow_accumulation_file(
            _as_path_str(dem), float(exponent), out_str, no_data, bool(compress)
        )
        if out_str is not None:
            return out_str
        assert res is not None
        return res
    if out_file is not None:
        raise ValueError("out_file is not supported for array input")
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.mfd_flow_accumulation_array(
        arr, float(exponent), nd, float(cell_size), gt, resolved_crs
    )


def burn_streams(
    in_dem: PathLikeStr,
    streams_vector: PathLikeStr,
    out_dem: PathLikeStr,
    *,
    depth: float = 1.0,
    width_cells: int = 0,
    buffer_cells: int = 0,
    smooth_drop: float = 0.0,
    compress: bool = False,
) -> int:
    """Burn stream polylines into a DEM GeoTIFF with optional AGREE-style smooth buffer."""
    return int(
        _lidar_rs.burn_streams(
            _as_path_str(in_dem),
            _as_path_str(streams_vector),
            _as_path_str(out_dem),
            float(depth),
            int(width_cells),
            int(buffer_cells),
            float(smooth_drop),
            bool(compress),
        )
    )


def delineate_depressions_merge_tree(
    sink_arr: SpatialArrayLike,
    dem_arr: SpatialArrayLike | None = None,
    *,
    min_size: int = 1000,
    min_depth: float = 0.3,
    topology: str = "D8",
    dem_is_refined: bool = False,
    max_rating_curve_points: int | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> MergeTreeResult:
    """Delineate exact bottom-up Union-Find merge-tree depressions and continuous rating curves."""
    arr, gt, resolved_crs, nd = extract_spatial_array(
        sink_arr, geotransform=geotransform, crs=crs, no_data=no_data
    )
    dem_np: NDArray[np.float64] | None = None
    if dem_arr is not None:
        dem_np, dgt, dcrs, _dnd = extract_spatial_array(dem_arr, no_data=nd)
        _check_aux_grid_alignment(
            "dem_arr",
            dgt,
            dcrs,
            gt,
            resolved_crs,
            dem_shape=(int(arr.shape[0]), int(arr.shape[1])),
            cell_size=float(cell_size),
        )
    return _lidar_rs.delineate_depressions_merge_tree(
        arr,
        dem_np,
        int(min_size),
        float(min_depth),
        topology,
        bool(dem_is_refined),
        int(max_rating_curve_points) if max_rating_curve_points is not None else None,
        nd,
        float(cell_size),
        gt,
        resolved_crs,
    )


def attenuate_fsm_hydrograph(
    hydrograph: Sequence[FsmHydrographPoint],
    mean_travel_time_h: float,
    *,
    dt_h: float = 0.25,
    num_reservoirs: int = 2,
    tail_duration_h: float | None = None,
) -> list[FsmHydrographPoint]:
    """Apply volume-conserving Nash linear-reservoir travel-time attenuation to an FSM hydrograph."""
    return list(
        _lidar_rs.attenuate_fsm_hydrograph(
            list(hydrograph),
            float(mean_travel_time_h),
            float(dt_h),
            int(num_reservoirs),
            float(tail_duration_h) if tail_duration_h is not None else None,
        )
    )


@overload
def compute_twi(
    dem: PathLikeStr,
    out_file: PathLikeStr,
    flow_acc: PathLikeStr | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def compute_twi(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: None = None,
    flow_acc: PathLikeStr | SpatialArrayLike | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> NDArray[np.float64]: ...


def compute_twi(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: PathLikeStr | None = None,
    flow_acc: PathLikeStr | SpatialArrayLike | None = None,
    *,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str | NDArray[np.float64]:
    """Compute the Topographic Wetness Index (`TWI = ln(a / tan(beta))`) from a DEM file or 2D spatial array.

    `flow_acc`, if provided, must be a cell-count flow-accumulation grid (not specific
    catchment area) aligned to `dem`. `cell_size`, `geotransform`, and `crs` apply only to
    array inputs. When `dem` is a file path and `out_file` is provided, writes the TWI
    GeoTIFF and returns `str(out_file)`. Otherwise returns a 2D `float64` NumPy array.
    """
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None or cell_size != 1.0:
            raise ValueError(
                "cell_size, geotransform, and crs overrides are not supported when dem is a file path; "
                "spatial metadata is read from the input raster file"
            )
        fa_str: str | None = None
        if flow_acc is not None:
            if not isinstance(flow_acc, (str, os.PathLike)):
                raise ValueError(
                    "flow_acc must be a raster file path when dem is a file path"
                )
            fa_str = _as_path_str(flow_acc)
        out_str = _as_path_str(out_file) if out_file is not None else None
        res = _lidar_rs.compute_twi_file(
            _as_path_str(dem), out_str, fa_str, no_data, bool(compress)
        )
        if out_str is not None:
            return out_str
        assert res is not None
        return res

    if out_file is not None:
        raise ValueError(
            "out_file is not supported for array input; pass a raster file path or omit out_file"
        )
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    fa_arr: NDArray[np.float64] | None = None
    if flow_acc is not None:
        if isinstance(flow_acc, (str, os.PathLike)):
            raise ValueError("flow_acc must be an array when dem is an array")
        fa_raw, fa_gt, fa_crs, fa_nd = extract_spatial_array(flow_acc)
        _check_aux_grid_alignment(
            "flow_acc",
            fa_gt,
            fa_crs,
            gt,
            resolved_crs,
            dem_shape=(int(arr.shape[0]), int(arr.shape[1])),
            cell_size=float(cell_size),
        )
        if fa_nd is not None and np.isfinite(fa_nd):
            fa_arr = np.array(fa_raw, dtype=np.float64, copy=True)
            fa_arr[fa_arr == float(fa_nd)] = np.nan
        else:
            fa_arr = fa_raw
    return _lidar_rs.compute_twi_array(
        arr, fa_arr, nd, float(cell_size), gt, resolved_crs
    )


@overload
def compute_hand(
    dem: PathLikeStr,
    out_file: PathLikeStr,
    *,
    stream_mask: PathLikeStr | None = None,
    acc_threshold_cells: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str: ...


@overload
def compute_hand(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: None = None,
    *,
    stream_mask: PathLikeStr | SpatialArrayLike | None = None,
    acc_threshold_cells: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> NDArray[np.float64]: ...


def compute_hand(
    dem: PathLikeStr | SpatialArrayLike,
    out_file: PathLikeStr | None = None,
    *,
    stream_mask: PathLikeStr | SpatialArrayLike | None = None,
    acc_threshold_cells: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    compress: bool = False,
) -> str | NDArray[np.float64]:
    """Compute Height Above Nearest Drainage (HAND) from a DEM file or 2D spatial array.

    Specify exactly one of `stream_mask` (binary raster/array where values > 0 mark drainage
    cells) or `acc_threshold_cells` (D8 flow-accumulation cell threshold to extract streams).
    `cell_size`, `geotransform`, and `crs` apply only to array inputs.
    """
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None or cell_size != 1.0:
            raise ValueError(
                "cell_size, geotransform, and crs overrides are not supported when dem is a file path; "
                "spatial metadata is read from the input raster file"
            )
        mask_str: str | None = None
        if stream_mask is not None:
            if not isinstance(stream_mask, (str, os.PathLike)):
                raise ValueError(
                    "stream_mask must be a raster file path when dem is a file path"
                )
            mask_str = _as_path_str(stream_mask)
        out_str = _as_path_str(out_file) if out_file is not None else None
        thresh = (
            float(acc_threshold_cells) if acc_threshold_cells is not None else None
        )
        res = _lidar_rs.compute_hand_file(
            _as_path_str(dem), out_str, mask_str, thresh, no_data, bool(compress)
        )
        if out_str is not None:
            return out_str
        assert res is not None
        return res

    if out_file is not None:
        raise ValueError(
            "out_file is not supported for array input; pass a raster file path or omit out_file"
        )
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    mask_u8: NDArray[np.uint8] | None = None
    if stream_mask is not None:
        if isinstance(stream_mask, (str, os.PathLike)):
            raise ValueError("stream_mask must be an array when dem is an array")
        m_raw, m_gt, m_crs, m_nd = extract_spatial_array(stream_mask)
        _check_aux_grid_alignment(
            "stream_mask",
            m_gt,
            m_crs,
            gt,
            resolved_crs,
            dem_shape=(int(arr.shape[0]), int(arr.shape[1])),
            cell_size=float(cell_size),
        )
        if m_nd is not None and np.isfinite(m_nd):
            mask_f64 = np.array(m_raw, dtype=np.float64, copy=True)
            mask_f64[mask_f64 == float(m_nd)] = 0.0
        else:
            mask_f64 = m_raw
        mask_u8 = np.ascontiguousarray(
            (np.isfinite(mask_f64) & (mask_f64 > 0.0)).astype(np.uint8)
        )
    thresh = float(acc_threshold_cells) if acc_threshold_cells is not None else None
    return _lidar_rs.compute_hand_array(
        arr, mask_u8, thresh, nd, float(cell_size), gt, resolved_crs
    )


def classify_embankment_artifacts(
    dem: PathLikeStr | SpatialArrayLike,
    depressions: DelineationArrayResult | Mapping[str, Any] | None = None,
    *,
    max_barrier_width_m: float = 60.0,
    min_exit_drop: float = 0.05,
    score_threshold: float = 0.55,
    leaves_only: bool = True,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> list[EmbankmentArtifactCandidate]:
    """Score delineated depressions to flag likely road/rail embankment impoundment artifacts.

    Accepts a DEM GeoTIFF path or a 2D spatial array (NumPy ndarray, xarray DataArray, or
    rasterio dataset). `cell_size`, `geotransform`, and `crs` apply only to array inputs.
    For array inputs, pass the dict returned by `delineate_depressions_array` as `depressions`
    to reuse an existing delineation, or omit `depressions` to extract sinks and delineate
    automatically.
    """
    if isinstance(dem, (str, os.PathLike)):
        if geotransform is not None or crs is not None or cell_size != 1.0:
            raise ValueError(
                "cell_size, geotransform, and crs overrides are not supported when dem is a file path; "
                "spatial metadata is read from the input raster file"
            )
        if depressions is not None:
            raise ValueError(
                "precomputed depressions apply only when dem is an array, not a file path"
            )
        return list(
            _lidar_rs.classify_embankment_artifacts_file(
                _as_path_str(dem),
                float(max_barrier_width_m),
                float(min_exit_drop),
                float(score_threshold),
                bool(leaves_only),
                int(min_size),
                float(min_depth),
                float(interval),
                no_data,
            )
        )

    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem, geotransform=geotransform, crs=crs, no_data=no_data
    )
    dep_id_arr: NDArray[np.int32] | None = None
    dep_list: list[DepressionRecord] | None = None
    if depressions is not None:
        dep_id_arr = np.ascontiguousarray(depressions["depression_id"], dtype=np.int32)
        dep_list = list(depressions["depressions"])
        dep_gt: Geotransform | None = None
        if depressions.get("geotransform") is not None:
            raw_gt = depressions["geotransform"]
            dep_gt = (
                float(raw_gt[0]),
                float(raw_gt[1]),
                float(raw_gt[2]),
                float(raw_gt[3]),
                float(raw_gt[4]),
                float(raw_gt[5]),
            )
        dep_crs = (
            str(depressions["crs"]) if depressions.get("crs") is not None else None
        )
        if gt is not None or resolved_crs is not None:
            _check_aux_grid_alignment(
                "depressions",
                dep_gt,
                dep_crs,
                gt,
                resolved_crs,
                dem_shape=(int(arr.shape[0]), int(arr.shape[1])),
                cell_size=float(cell_size),
            )
        if gt is None and dep_gt is not None:
            gt = dep_gt
        if resolved_crs is None and dep_crs is not None:
            resolved_crs = dep_crs

    return list(
        _lidar_rs.classify_embankment_artifacts_array(
            arr,
            dep_id_arr,
            dep_list,
            float(max_barrier_width_m),
            float(min_exit_drop),
            float(score_threshold),
            bool(leaves_only),
            int(min_size),
            float(min_depth),
            float(interval),
            nd,
            float(cell_size),
            gt,
            resolved_crs,
        )
    )


def compute_rating_curves(
    sink_arr: SpatialArrayLike,
    depressions: DelineationArrayResult | Mapping[str, Any] | None = None,
    interval: float = 0.3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    *,
    delineation_method: str = "level_set",
) -> list[RatingCurve]:
    """Compute exact stage-area-volume rating curves for every depression in `sink_arr`.

    Pass the dict returned by `delineate_depressions_array` as `depressions` to reuse an
    existing delineation, or omit `depressions` to delineate automatically before computing
    rating curves.
    """
    arr, gt, resolved_crs, nd = extract_spatial_array(
        sink_arr, geotransform=geotransform, crs=crs, no_data=no_data
    )
    dep_id_arr: NDArray[np.int32] | None = None
    dep_lvl_arr: NDArray[np.int32] | None = None
    dep_list: list[DepressionRecord] | None = None
    if depressions is not None:
        dep_id_arr = np.ascontiguousarray(depressions["depression_id"], dtype=np.int32)
        raw_lvl = depressions.get("depression_level")
        if raw_lvl is not None:
            dep_lvl_arr = np.ascontiguousarray(raw_lvl, dtype=np.int32)
        dep_list = list(depressions["depressions"])
        if gt is None and depressions.get("geotransform") is not None:
            raw_gt = depressions["geotransform"]
            gt = (
                float(raw_gt[0]),
                float(raw_gt[1]),
                float(raw_gt[2]),
                float(raw_gt[3]),
                float(raw_gt[4]),
                float(raw_gt[5]),
            )
        if resolved_crs is None and depressions.get("crs") is not None:
            resolved_crs = str(depressions["crs"])

    return _lidar_rs.compute_rating_curves(
        arr,
        dep_id_arr,
        dep_lvl_arr,
        dep_list,
        float(interval),
        int(min_size),
        float(min_depth),
        nd,
        float(cell_size),
        gt,
        resolved_crs,
        delineation_method,
    )


def build_connectivity_graph(
    dem_or_depressions: SpatialArrayLike | DelineationArrayResult | Mapping[str, Any],
    *,
    fill_time: SpatialArrayLike | None = None,
    catchment_records: Sequence[CatchmentHierarchyRecord] | None = None,
    spill_paths: Sequence[SpillFlowPathRecord] | None = None,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    kernel_size: int = 1,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    stream_vector: PathLikeStr | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    geotransform: Geotransform | None = None,
    crs: str | None = None,
) -> ConnectivityGraph:
    """Build the directed depression hierarchy and spill connectivity graph.

    Accepts either:
    - A 2D DEM array (NumPy ndarray, xarray DataArray, or rasterio dataset), running the
      in-memory delineation and spill-path pipeline to produce nodes and edges.
    - A precomputed delineation dict (from `delineate_depressions_array` or
      `simulate_inundation_array`), optionally paired with `fill_time`, `catchment_records`,
      and `spill_paths`.
    """
    stream_str = _as_path_str(stream_vector) if stream_vector is not None else None
    hyeto_list = [(float(dt), float(i)) for dt, i in hyetograph] if hyetograph is not None else None

    if isinstance(dem_or_depressions, Mapping):
        dep_id_arr = np.ascontiguousarray(
            dem_or_depressions["depression_id"], dtype=np.int32
        )
        raw_lvl = dem_or_depressions.get("depression_level")
        dep_lvl_arr = (
            np.ascontiguousarray(raw_lvl, dtype=np.int32)
            if raw_lvl is not None
            else None
        )
        dep_list = list(dem_or_depressions["depressions"])
        gt = geotransform
        if gt is None and dem_or_depressions.get("geotransform") is not None:
            raw_gt = dem_or_depressions["geotransform"]
            gt = (
                float(raw_gt[0]),
                float(raw_gt[1]),
                float(raw_gt[2]),
                float(raw_gt[3]),
                float(raw_gt[4]),
                float(raw_gt[5]),
            )
        resolved_crs = crs
        if resolved_crs is None and dem_or_depressions.get("crs") is not None:
            resolved_crs = str(dem_or_depressions["crs"])

        ft_src: Any = fill_time if fill_time is not None else dem_or_depressions.get("fill_time")
        ft_arr = (
            _extract_array_masking_nodata(ft_src) if ft_src is not None else None
        )
        cat_raw: Any = (
            catchment_records
            if catchment_records is not None
            else dem_or_depressions.get("catchment_records")
        )
        sp_raw: Any = (
            spill_paths
            if spill_paths is not None
            else dem_or_depressions.get("spill_paths")
        )
        cat_list: list[CatchmentHierarchyRecord] | None = (
            list(cat_raw) if cat_raw is not None else None
        )
        sp_list: list[SpillFlowPathRecord] | None = (
            list(sp_raw) if sp_raw is not None else None
        )
        return _lidar_rs.build_connectivity_graph_from_parts(
            dep_id_arr,
            dep_lvl_arr,
            dep_list,
            ft_arr,
            cat_list,
            sp_list,
            float(rain_intensity_cm_h),
            hyeto_list,
            stream_str,
            float(cell_size),
            gt,
            resolved_crs,
        )

    if (
        fill_time is not None
        or catchment_records is not None
        or spill_paths is not None
    ):
        raise ValueError(
            "fill_time, catchment_records, and spill_paths apply only when "
            "dem_or_depressions is a delineation dict, not a DEM array"
        )

    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem_or_depressions, geotransform=geotransform, crs=crs, no_data=no_data
    )
    return _lidar_rs.build_connectivity_graph_from_dem(
        arr,
        int(min_size),
        float(min_depth),
        float(interval),
        float(rain_intensity_cm_h),
        float(rain_duration_h),
        float(time_step_h),
        int(kernel_size),
        hyeto_list,
        stream_str,
        nd,
        float(cell_size),
        gt,
        resolved_crs,
    )


def _extract_array_masking_nodata(
    obj: SpatialArrayLike, *, fill: float = np.nan
) -> np.ndarray:
    """Extract a 2D array from a spatial adapter and replace any finite NoData sentinel with `fill`."""
    arr, _gt, _crs, nd = extract_spatial_array(obj)
    if nd is not None and np.isfinite(nd):
        masked = np.array(arr, dtype=np.float64, copy=True)
        masked[masked == float(nd)] = fill
        return masked
    return arr


def simulate_inundation_array(
    dem_arr: SpatialArrayLike,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    runoff_coeff_arr: SpatialArrayLike | None = None,
    infiltration_arr: SpatialArrayLike | None = None,
    antecedent_depth_arr: SpatialArrayLike | None = None,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    breach_max_depth: float | None = None,
    no_data: float | None = None,
    cell_size: float = 1.0,
    routing: str = "static",
    geotransform: Geotransform | None = None,
    crs: str | None = None,
    *,
    runoff_coeff_no_data: float | None = None,
    infiltration_no_data: float | None = None,
    antecedent_depth_no_data: float | None = None,
    delineation_method: str = "level_set",
    green_ampt: tuple[float, float, float] | None = None,
    compute_hazard: bool = False,
    recession_et_cm_h: float | None = None,
    recession_seepage_cm_h: float | None = None,
    recession_max_h: float = 168.0,
) -> InundationArrayResult:
    """Run the complete hydrology and inundation pipeline in memory on a 2D array, xarray DataArray, or rasterio dataset.

    Set `routing="fsm"` for dynamic Fill-Spill-Merge cascade mass-balance routing.
    Per-forcing NoData sentinels (from spatial adapters or explicit keyword overrides) are
    forwarded directly to Rust so `crates/lidar` owns forcing NoData semantics in one place
    (for `runoff_coeff_arr`, NoData cells fall back to the core default coefficient of `1.0`).
    """
    arr, gt, resolved_crs, nd = extract_spatial_array(
        dem_arr, geotransform=geotransform, crs=crs, no_data=no_data
    )
    rc: NDArray[np.float64] | None = None
    rc_nd: float | None = runoff_coeff_no_data
    if runoff_coeff_arr is not None:
        rc, _rc_gt, _rc_crs, rc_nd = extract_spatial_array(
            runoff_coeff_arr, no_data=runoff_coeff_no_data
        )
    inf: NDArray[np.float64] | None = None
    inf_nd: float | None = infiltration_no_data
    if infiltration_arr is not None:
        inf, _inf_gt, _inf_crs, inf_nd = extract_spatial_array(
            infiltration_arr, no_data=infiltration_no_data
        )
    ant: NDArray[np.float64] | None = None
    ant_nd: float | None = antecedent_depth_no_data
    if antecedent_depth_arr is not None:
        ant, _ant_gt, _ant_crs, ant_nd = extract_spatial_array(
            antecedent_depth_arr, no_data=antecedent_depth_no_data
        )
    ga_tuple = (
        (float(green_ampt[0]), float(green_ampt[1]), float(green_ampt[2]))
        if green_ampt is not None
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
        [(float(dt), float(i)) for dt, i in hyetograph]
        if hyetograph is not None
        else None,
        breach_max_depth,
        nd,
        float(cell_size),
        routing,
        gt,
        resolved_crs,
        rc_nd,
        inf_nd,
        ant_nd,
        delineation_method,
        ga_tuple,
        bool(compute_hazard),
        float(recession_et_cm_h) if recession_et_cm_h is not None else None,
        float(recession_seepage_cm_h) if recession_seepage_cm_h is not None else None,
        float(recession_max_h),
    )


def run_pipeline(
    in_dem: PathLikeStr,
    out_dir: PathLikeStr,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    rain_intensity_cm_h: float = 5.0,
    rain_duration_h: float = 50.0,
    time_step_h: float = 1.0,
    breach_max_depth: float | None = None,
    breach_max_length: int = 40,
    culverts_vector: PathLikeStr | None = None,
    runoff_coeff_path: PathLikeStr | None = None,
    infiltration_path: PathLikeStr | None = None,
    antecedent_depth_path: PathLikeStr | None = None,
    hyetograph: Sequence[tuple[float, float]] | None = None,
    stream_vector: PathLikeStr | None = None,
    use_f32: bool = False,
    save_rasters: bool = True,
    export_vectors: bool = True,
    export_graph: bool = True,
    compress: bool = False,
    routing: str = "static",
    *,
    delineation_method: str = "level_set",
) -> PipelineResult:
    """Run the all-in-one in-memory hydrology pipeline from a raw DEM file to GeoPackage, CSV, GraphML, and rasters.

    Set `routing="fsm"` for dynamic Fill-Spill-Merge cascade mass-balance routing.
    """
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
        _as_path_str(antecedent_depth_path)
        if antecedent_depth_path is not None
        else None,
        [(float(dt), float(i)) for dt, i in hyetograph]
        if hyetograph is not None
        else None,
        _as_path_str(stream_vector) if stream_vector is not None else None,
        bool(use_f32),
        bool(save_rasters),
        bool(export_vectors),
        bool(export_graph),
        bool(compress),
        routing,
        delineation_method,
    )


def delineate_depressions_tiled(
    in_dem: PathLikeStr,
    out_dir: PathLikeStr,
    tile_size: int = 2048,
    overlap: int = 128,
    kernel_size: int = 3,
    min_size: int = 1000,
    min_depth: float = 0.3,
    interval: float = 0.3,
    use_f32: bool = True,
    *,
    delineation_method: str = "level_set",
    write_rasters: bool = False,
    write_gpkg: bool = False,
    compress: bool = False,
) -> TiledDelineationResult:
    """Delineate nested depressions across large DEMs using parallel overlapping tiles.

    Note: `use_f32` applies to the eager/in-memory fallback path (used for legacy
    non-lazy formats such as ESRI ASCII `.asc` grids), while windowed GeoTIFF/COG
    inputs read one padded tile at a time directly into `f64`.
    """
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
        delineation_method,
        bool(write_rasters),
        bool(write_gpkg),
        bool(compress),
    )


def PlayAnimation(
    in_dem: PathLikeStr,
    in_flood_dir: PathLikeStr,
    out_gif: PathLikeStr = "inundation.gif",
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
        from IPython.display import Image  # type: ignore[import-not-found]

        return Image(filename=res["gif_path"])
    return str(res["gif_path"])


def synthesize_culverts(
    streams_vector: PathLikeStr,
    roads_vector: PathLikeStr,
    out_culverts: PathLikeStr,
    half_length_m: float = 25.0,
    snap_distance_m: float = 15.0,
    target_crs: str | None = None,
) -> int:
    """Synthesize deterministic road-stream culvert breach polylines by intersecting stream and road vectors."""
    return int(
        _lidar_rs.synthesize_culverts(
            _as_path_str(streams_vector),
            _as_path_str(roads_vector),
            _as_path_str(out_culverts),
            float(half_length_m),
            float(snap_distance_m),
            target_crs,
        )
    )


def _normalize_bbox(
    bbox: tuple[float, float, float, float] | Sequence[float] | None,
    names: tuple[str, str, str, str] = ("min_lon", "min_lat", "max_lon", "max_lat"),
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        labels = ", ".join(names)
        raise ValueError(
            f"bbox must have 4 elements ({labels}); got {len(bbox)}"
        )
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def query_3dep_products(
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    dem_product: str = "1m",
    deduplicate: bool = True,
) -> list[TnmProductItem]:
    """Query USGS 3DEP DEM or LiDAR Point Cloud products intersecting a region without downloading payloads."""
    return list(
        _lidar_rs.query_3dep_products(
            _normalize_bbox(bbox),
            huc,
            _as_path_str(region) if region is not None else None,
            dem_product,
            bool(deduplicate),
        )
    )


def download_3dep_dem(
    out_dir: PathLikeStr,
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    dem_product: str = "1m",
    mosaic_dem: bool = True,
    clip_to_region: bool = True,
    tile_cache_dir: PathLikeStr | None = None,
    overwrite: bool = False,
) -> Download3depResult:
    """Download, optionally mosaic, and clip USGS 3DEP DEM (or LPC) tiles for a specified region."""
    return _lidar_rs.download_3dep_dem(
        _as_path_str(out_dir),
        _normalize_bbox(bbox),
        huc,
        _as_path_str(region) if region is not None else None,
        dem_product,
        bool(mosaic_dem),
        bool(clip_to_region),
        _as_path_str(tile_cache_dir) if tile_cache_dir is not None else None,
        bool(overwrite),
    )


def download_wbd(
    out_path: PathLikeStr,
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    level: int = 12,
    overwrite: bool = False,
) -> int:
    """Download USGS Watershed Boundary Dataset (WBD) polygons for a specified region."""
    return int(
        _lidar_rs.download_wbd(
            _as_path_str(out_path),
            _normalize_bbox(bbox),
            huc,
            _as_path_str(region) if region is not None else None,
            int(level),
            bool(overwrite),
        )
    )


def download_nhd(
    out_path: PathLikeStr,
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    clip_to_region: bool = True,
    overwrite: bool = False,
) -> int:
    """Download USGS National Hydrography Dataset (NHD) flowlines for a specified region."""
    return int(
        _lidar_rs.download_nhd(
            _as_path_str(out_path),
            _normalize_bbox(bbox),
            huc,
            _as_path_str(region) if region is not None else None,
            bool(clip_to_region),
            bool(overwrite),
        )
    )


def download_nwi(
    out_path: PathLikeStr,
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    clip_to_region: bool = True,
    overwrite: bool = False,
) -> int:
    """Download USFWS National Wetlands Inventory (NWI) polygons for a specified region."""
    return int(
        _lidar_rs.download_nwi(
            _as_path_str(out_path),
            _normalize_bbox(bbox),
            huc,
            _as_path_str(region) if region is not None else None,
            bool(clip_to_region),
            bool(overwrite),
        )
    )


def download_roads(
    out_path: PathLikeStr,
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    clip_to_region: bool = True,
    overwrite: bool = False,
) -> int:
    """Download USGS National Transportation road and railway centerlines for a specified region."""
    return int(
        _lidar_rs.download_roads(
            _as_path_str(out_path),
            _normalize_bbox(bbox),
            huc,
            _as_path_str(region) if region is not None else None,
            bool(clip_to_region),
            bool(overwrite),
        )
    )


def fetch_noaa_atlas14(
    lat: float,
    lon: float,
    out_json: PathLikeStr | None = None,
    overwrite: bool = False,
) -> NoaaAtlas14Table:
    """Fetch NOAA Atlas 14 PDS precipitation depth frequency values (mm) for a WGS 84 point."""
    return _lidar_rs.fetch_noaa_atlas14(
        float(lat),
        float(lon),
        _as_path_str(out_json) if out_json is not None else None,
        bool(overwrite),
    )


def download_federal_data(
    out_dir: PathLikeStr,
    *,
    bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    huc: str | None = None,
    region: PathLikeStr | None = None,
    datasets: str | Sequence[str] = "all",
    dem_product: str = "1m",
    wbd_level: int = 12,
    mosaic_dem: bool = True,
    clip_to_region: bool = True,
    tile_cache_dir: PathLikeStr | None = None,
    overwrite: bool = False,
) -> FederalDataResult:
    """Download a bundle of open U.S. federal datasets (WBD, 3DEP DEM, NHD, NWI, roads, culverts, NOAA Atlas 14) for a region."""
    ds_spec = datasets if isinstance(datasets, str) else ",".join(str(d) for d in datasets)
    return _lidar_rs.download_federal_data(
        _as_path_str(out_dir),
        _normalize_bbox(bbox),
        huc,
        _as_path_str(region) if region is not None else None,
        ds_spec,
        dem_product,
        int(wbd_level),
        bool(mosaic_dem),
        bool(clip_to_region),
        _as_path_str(tile_cache_dir) if tile_cache_dir is not None else None,
        bool(overwrite),
    )


# Snake-case aliases
mean_filter = MeanFilter
median_filter = MedianFilter
gaussian_filter = GaussianFilter
flip_dem = FlipDEM

__all__ = [
    "AnimateInundationResult",
    "CatchmentHierarchy",
    "CatchmentHierarchyRecord",
    "CatchmentHierarchyResult",
    "CatchmentRecord",
    "ConnectivityGraph",
    "DelineateCatchments",
    "DelineateCatchmentsResult",
    "DelineateDepressions",
    "DelineateFlowPaths",
    "DelineateFlowPathsResult",
    "DelineateMounts",
    "DelineateMountsResult",
    "DelineationArrayResult",
    "DelineationResult",
    "DepressionRecord",
    "Download3depResult",
    "EmbankmentArtifactCandidate",
    "ExtractSinks",
    "ExtractSinksArrayResult",
    "ExtractSinksResult",
    "FederalDataResult",
    "FlipDEM",
    "FlowPathRecord",
    "FsmBasinSummary",
    "FsmHydrographPoint",
    "FsmMassBalance",
    "GaussianFilter",
    "Geotransform",
    "GraphEdge",
    "GraphNode",
    "GroundClassifyResult",
    "InundationArrayResult",
    "InundationResult",
    "InundationStepRecord",
    "InundationStepSummaryRecord",
    "MeanFilter",
    "MedianFilter",
    "MergeTreeNode",
    "MergeTreeResult",
    "NoaaAtlas14Table",
    "PathLikeStr",
    "PipelineResult",
    "PlayAnimation",
    "PointCloudDict",
    "RatingCurve",
    "RatingCurvePoint",
    "RecessionCurvePoint",
    "SimulateInundation",
    "SinkDepressionRecord",
    "SpatialArrayLike",
    "SpillFlowPathRecord",
    "TiledDelineationResult",
    "TnmProductItem",
    "__version__",
    "add_crs",
    "animate_inundation",
    "array_to_xarray",
    "attenuate_fsm_hydrograph",
    "bilateral_filter",
    "breach_depressions",
    "build_connectivity_graph",
    "burn_culverts",
    "burn_streams",
    "classify_embankment_artifacts",
    "classify_ground",
    "clip_image",
    "compute_hand",
    "compute_hillshade",
    "compute_rating_curves",
    "compute_twi",
    "connectivity_to_networkx",
    "convert_lidar",
    "csv_points_to_shp",
    "csv_to_shp",
    "delineate_catchment_hierarchy",
    "delineate_catchments",
    "delineate_depressions",
    "delineate_depressions_array",
    "delineate_depressions_merge_tree",
    "delineate_depressions_tiled",
    "delineate_flow_paths",
    "delineate_mounts",
    "delineate_mounts_array",
    "depressions_to_geodataframe",
    "dinf_flow_accumulation",
    "dinf_flow_direction",
    "download_3dep_dem",
    "download_federal_data",
    "download_nhd",
    "download_nwi",
    "download_roads",
    "download_wbd",
    "export_inundation_frames_png",
    "extract_sinks",
    "extract_sinks_array",
    "feature_preserving_smoothing",
    "fetch_noaa_atlas14",
    "flip_dem",
    "gaussian_filter",
    "image_to_cog",
    "join_tables",
    "lidar_to_chm",
    "lidar_to_chm_pit_free",
    "lidar_to_chm_tiled",
    "lidar_to_dsm",
    "lidar_to_dtm",
    "lidar_to_dtm_tiled",
    "mean_filter",
    "median_filter",
    "mfd_flow_accumulation",
    "mosaic",
    "polygonize",
    "query_3dep_products",
    "read_lidar",
    "reproject_image",
    "resample",
    "run_pipeline",
    "scs_curve_number_runoff",
    "scs_type2_hyetograph",
    "simulate_inundation",
    "simulate_inundation_array",
    "stream_remote_pointcloud_aoi",
    "synthesize_culverts",
    "write_lidar",
]
