"""Zero-hard-dependency duck-typed spatial adapters for `lidar_rs`."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._types import (
    ConnectivityGraph,
    DelineationArrayResult,
    DelineationResult,
    Geotransform,
    InundationArrayResult,
)


def _normalize_transform(transform: Any) -> Geotransform | None:
    """Convert an `affine.Affine`, rasterio transform, or 6-element GDAL sequence to a GDAL geotransform."""
    if transform is None:
        return None
    to_gdal = getattr(transform, "to_gdal", None)
    if callable(to_gdal):
        vals = to_gdal()
        if len(vals) == 6:
            return (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
                float(vals[4]),
                float(vals[5]),
            )
    if all(hasattr(transform, attr) for attr in ("a", "b", "c", "d", "e", "f")):
        return (
            float(transform.c),
            float(transform.a),
            float(transform.b),
            float(transform.f),
            float(transform.d),
            float(transform.e),
        )
    if isinstance(transform, Sequence) and len(transform) == 6:
        return (
            float(transform[0]),
            float(transform[1]),
            float(transform[2]),
            float(transform[3]),
            float(transform[4]),
            float(transform[5]),
        )
    raise ValueError(
        "geotransform must be a 6-element GDAL tuple (origin_x, pixel_w, rot_x, origin_y, rot_y, pixel_h) "
        "or an affine.Affine-compatible transform"
    )


def _normalize_crs(crs: Any) -> str | None:
    """Convert a rasterio/pyproj CRS object or string into a CRS string."""
    if crs is None:
        return None
    if isinstance(crs, str):
        cleaned = crs.strip()
        return cleaned if cleaned else None
    for method_name in ("to_string", "to_wkt"):
        method = getattr(crs, method_name, None)
        if callable(method):
            val = method()
            if isinstance(val, str) and val.strip():
                return val.strip()
    text = str(crs).strip()
    return text if text else None


def _is_rasterio_dataset(obj: Any) -> bool:
    return (
        not isinstance(obj, (str, bytes, os.PathLike, np.ndarray))
        and callable(getattr(obj, "read", None))
        and hasattr(obj, "transform")
        and hasattr(obj, "crs")
    )


def _is_xarray_dataarray(obj: Any) -> bool:
    return (
        not isinstance(obj, (str, bytes, os.PathLike, np.ndarray))
        and hasattr(obj, "values")
        and hasattr(obj, "dims")
        and hasattr(obj, "coords")
    )


def is_spatial_array(obj: Any) -> bool:
    """Return True if `obj` is a NumPy ndarray or a duck-typed xarray.DataArray / rasterio.DatasetReader."""
    return (
        isinstance(obj, np.ndarray)
        or _is_rasterio_dataset(obj)
        or _is_xarray_dataarray(obj)
    )


def _coord_array(coords: Any, names: tuple[str, ...]) -> NDArray[np.float64] | None:
    for name in names:
        try:
            present = name in coords
        except TypeError:
            present = hasattr(coords, name)
        if not present:
            continue
        try:
            coord_obj = coords[name]
        except (TypeError, KeyError):
            coord_obj = getattr(coords, name, None)
        if coord_obj is None:
            continue
        raw = getattr(coord_obj, "values", coord_obj)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 1 and arr.size >= 2:
            return arr
    return None


def _geotransform_from_xarray_coords(obj: Any) -> Geotransform | None:
    coords = getattr(obj, "coords", None)
    if coords is None:
        return None
    xs = _coord_array(coords, ("x", "lon", "longitude"))
    ys = _coord_array(coords, ("y", "lat", "latitude"))
    if xs is None or ys is None:
        return None
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    if not (np.isfinite(dx) and np.isfinite(dy)) or dx == 0.0 or dy == 0.0:
        return None
    origin_x = float(xs[0]) - 0.5 * dx
    origin_y = float(ys[0]) - 0.5 * dy
    return (origin_x, dx, 0.0, origin_y, 0.0, dy)


def _coerce_2d_float64(raw: Any) -> NDArray[np.float64]:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


_IDENTITY_GEOTRANSFORM: Geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def extract_spatial_array(
    obj: Any,
    *,
    geotransform: Geotransform | Sequence[float] | Any | None = None,
    crs: Any | None = None,
    no_data: float | None = None,
) -> tuple[NDArray[np.float64], Geotransform | None, str | None, float | None]:
    """Extract `(array, geotransform, crs, no_data)` from a NumPy array, xarray.DataArray, or rasterio.DatasetReader.

    Explicit `geotransform`, `crs`, and `no_data` arguments take precedence over metadata
    discovered on the input object. Any explicit `geotransform` or georeferenced input
    metadata takes precedence over `cell_size` in downstream `*_array` functions; an
    un-georeferenced identity transform `(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)` with no CRS is
    ignored so caller-supplied `cell_size` applies.
    """
    explicit_gt = geotransform is not None
    gt_out = _normalize_transform(geotransform) if explicit_gt else None
    crs_out = _normalize_crs(crs) if crs is not None else None
    nd_out = float(no_data) if no_data is not None else None

    if _is_rasterio_dataset(obj):
        arr = _coerce_2d_float64(obj.read(1))
        if crs_out is None:
            crs_out = _normalize_crs(getattr(obj, "crs", None))
        if gt_out is None:
            discovered = _normalize_transform(getattr(obj, "transform", None))
            if discovered is not None and (
                crs_out is not None or discovered != _IDENTITY_GEOTRANSFORM
            ):
                gt_out = discovered
        if nd_out is None:
            raw_nd = getattr(obj, "nodata", None)
            if raw_nd is not None:
                nd_out = float(raw_nd)
        return arr, gt_out, crs_out, nd_out

    if _is_xarray_dataarray(obj):
        arr = _coerce_2d_float64(obj.values)
        rio = getattr(obj, "rio", None)
        if rio is not None:
            if crs_out is None:
                crs_out = _normalize_crs(getattr(rio, "crs", None))
            if gt_out is None:
                rio_tf = getattr(rio, "transform", None)
                tf_val = rio_tf() if callable(rio_tf) else rio_tf
                if tf_val is not None:
                    discovered = _normalize_transform(tf_val)
                    if discovered is not None and (
                        crs_out is not None or discovered != _IDENTITY_GEOTRANSFORM
                    ):
                        gt_out = discovered
            if nd_out is None:
                raw_nd = getattr(rio, "nodata", None)
                if raw_nd is None:
                    raw_nd = getattr(rio, "encoded_nodata", None)
                if raw_nd is not None:
                    nd_out = float(raw_nd)

        attrs = getattr(obj, "attrs", None)
        if isinstance(attrs, Mapping):
            if crs_out is None and "crs" in attrs:
                crs_out = _normalize_crs(attrs["crs"])
            if gt_out is None:
                for key in ("geotransform", "transform"):
                    if key in attrs and attrs[key] is not None:
                        discovered = _normalize_transform(attrs[key])
                        if discovered is not None and (
                            crs_out is not None or discovered != _IDENTITY_GEOTRANSFORM
                        ):
                            gt_out = discovered
                        break
            if nd_out is None:
                for key in ("nodata", "_FillValue"):
                    if key in attrs and attrs[key] is not None:
                        nd_out = float(attrs[key])
                        break

        if gt_out is None:
            discovered = _geotransform_from_xarray_coords(obj)
            if discovered is not None and (
                crs_out is not None or discovered != _IDENTITY_GEOTRANSFORM
            ):
                gt_out = discovered

        return arr, gt_out, crs_out, nd_out

    return _coerce_2d_float64(obj), gt_out, crs_out, nd_out


def _pixel_corner_xy(
    gt: Geotransform, col: float, row: float
) -> tuple[float, float]:
    return (
        gt[0] + col * gt[1] + row * gt[2],
        gt[3] + col * gt[4] + row * gt[5],
    )


def _mask_to_geometry(
    mask: NDArray[np.bool_],
    gt: Geotransform,
    polygon_cls: Any,
    unary_union: Any,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> Any:
    """Convert a 2D boolean pixel mask into an exact Shapely geometry via horizontal run strips."""
    strips: list[Any] = []
    rows, cols = mask.shape
    for r in range(rows):
        row_mask = mask[r]
        if not np.any(row_mask):
            continue
        padded = np.empty(cols + 2, dtype=np.bool_)
        padded[0] = False
        padded[1:-1] = row_mask
        padded[-1] = False
        diff = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        gr = float(r + row_offset)
        gr1 = float(r + row_offset + 1)
        for c0, c1 in zip(starts.tolist(), ends.tolist()):
            gc0 = float(c0 + col_offset)
            gc1 = float(c1 + col_offset)
            p00 = _pixel_corner_xy(gt, gc0, gr)
            p10 = _pixel_corner_xy(gt, gc1, gr)
            p11 = _pixel_corner_xy(gt, gc1, gr1)
            p01 = _pixel_corner_xy(gt, gc0, gr1)
            strips.append(polygon_cls([p00, p10, p11, p01, p00]))
    if not strips:
        return polygon_cls()
    if len(strips) == 1:
        return strips[0]
    return unary_union(strips)


def _compute_id_bboxes(
    dep_id_arr: NDArray[Any],
) -> dict[int, tuple[int, int, int, int]]:
    """Compute `(r_min, r_max, c_min, c_max)` bounding boxes for all positive IDs in one vectorized pass."""
    rows, cols = np.nonzero(dep_id_arr > 0)
    if rows.size == 0:
        return {}
    ids = dep_id_arr[rows, cols]
    unique_ids, inv = np.unique(ids, return_inverse=True)
    n_ids = unique_ids.size
    r_min = np.full(n_ids, dep_id_arr.shape[0], dtype=np.intp)
    r_max = np.full(n_ids, -1, dtype=np.intp)
    c_min = np.full(n_ids, dep_id_arr.shape[1], dtype=np.intp)
    c_max = np.full(n_ids, -1, dtype=np.intp)
    np.minimum.at(r_min, inv, rows)
    np.maximum.at(r_max, inv, rows)
    np.minimum.at(c_min, inv, cols)
    np.maximum.at(c_max, inv, cols)
    return {
        int(uid): (int(r_min[i]), int(r_max[i]), int(c_min[i]), int(c_max[i]))
        for i, uid in enumerate(unique_ids)
    }


def _descendant_ids(
    root_id: int, children_map: dict[int, list[int]]
) -> set[int]:
    ids: set[int] = set()
    stack = [root_id]
    while stack:
        cur = stack.pop()
        if cur in ids:
            continue
        ids.add(cur)
        stack.extend(children_map.get(cur, ()))
    return ids


def depressions_to_geodataframe(
    result: DelineationArrayResult | DelineationResult | InundationArrayResult | Mapping[str, Any],
    crs: Any | None = None,
) -> Any:
    """Convert a delineation result dict into a `geopandas.GeoDataFrame`.

    Supports both in-memory results (containing `depression_id` and `depressions`) and
    file-based results (containing `gpkg_path` or `shapefile_path`). Requires optional
    `geopandas` and `shapely` packages.
    """
    try:
        import geopandas as gpd  # type: ignore[import-untyped,import-not-found]
        from shapely.geometry import Polygon  # type: ignore[import-untyped,import-not-found]
        from shapely.ops import unary_union  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise ImportError(
            "depressions_to_geodataframe requires geopandas and shapely; "
            "install them with `pip install geopandas shapely`"
        ) from exc

    resolved_crs = _normalize_crs(crs) if crs is not None else _normalize_crs(result.get("crs"))

    dep_id_arr = result.get("depression_id")
    if isinstance(dep_id_arr, np.ndarray):
        rows = dep_id_arr.shape[0]
        raw_gt = result.get("geotransform")
        gt: Geotransform = (
            _normalize_transform(raw_gt) if raw_gt is not None else None
        ) or (0.0, 1.0, 0.0, float(rows), 0.0, -1.0)
        depressions = list(result.get("depressions", []))
        children_map: dict[int, list[int]] = {
            int(d["id"]): [int(cid) for cid in d.get("children_ids", [])]
            for d in depressions
            if "id" in d
        }
        id_bboxes = _compute_id_bboxes(dep_id_arr)
        geometries: list[Any] = []
        records: list[dict[str, Any]] = []
        for dep in depressions:
            did = int(dep["id"])
            member_ids = _descendant_ids(did, children_map)
            member_bboxes = [id_bboxes[mid] for mid in member_ids if mid in id_bboxes]
            if not member_bboxes:
                geometries.append(Polygon())
            else:
                r0 = min(b[0] for b in member_bboxes)
                r1 = max(b[1] for b in member_bboxes)
                c0 = min(b[2] for b in member_bboxes)
                c1 = max(b[3] for b in member_bboxes)
                sub = dep_id_arr[r0 : r1 + 1, c0 : c1 + 1]
                if len(member_ids) == 1:
                    mask = sub == did
                else:
                    mask = np.isin(sub, list(member_ids))
                geometries.append(
                    _mask_to_geometry(
                        mask, gt, Polygon, unary_union, row_offset=r0, col_offset=c0
                    )
                )
            records.append(dict(dep))
        return gpd.GeoDataFrame(records, geometry=geometries, crs=resolved_crs)

    for key in ("gpkg_path", "shapefile_path"):
        path = result.get(key)
        if path:
            gdf = gpd.read_file(path)
            if resolved_crs is not None and getattr(gdf, "crs", None) is None:
                gdf = gdf.set_crs(resolved_crs)
            return gdf

    raise ValueError(
        "result must contain an in-memory 'depression_id' ndarray or a 'gpkg_path'/'shapefile_path'"
    )


def connectivity_to_networkx(
    graph: ConnectivityGraph | Mapping[str, Any],
    *,
    multi: bool = False,
) -> Any:
    """Convert a `ConnectivityGraph` dict into a NetworkX directed graph.

    By default (`multi=False`), returns a `networkx.DiGraph`. When both a `"spill"`
    and a `"merge"` edge connect the same `(from_id, to_id)` pair (for example when a
    child depression spills directly into its enclosing parent), attributes from both
    edges are merged without overwriting non-None `"spill"` attributes: `edge_type`
    becomes `"spill+merge"`, `edge_types` lists all merged edge types, type-prefixed
    attributes (e.g. `spill_merge_time_h`, `merge_merge_time_h`, `spill_spill_elev`,
    `merge_spill_elev`) preserve per-edge values, and `parallel_edges` holds the raw
    attribute dicts for all edges between `(u, v)`.

    Pass `multi=True` to return a `networkx.MultiDiGraph` where parallel `"spill"` and
    `"merge"` edges are kept as separate keyed edges (`key=edge_type`).

    Every node carries `min_elev`, `spill_elev`, `area`, `volume`, `level` (plus all
    other node fields), and every edge carries `pour_row`, `pour_col`, `spill_elev`
    (plus all other edge fields). Requires the optional `networkx` package.
    """
    try:
        import networkx as nx  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise ImportError(
            "connectivity_to_networkx requires networkx; "
            "install it with `pip install networkx`"
        ) from exc

    g = nx.MultiDiGraph() if multi else nx.DiGraph()
    if "rain_intensity_cm_h" in graph:
        g.graph["rain_intensity_cm_h"] = graph["rain_intensity_cm_h"]

    for node in graph.get("nodes", []):
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        attrs.setdefault("spill_elev", attrs.get("max_elev"))
        attrs.setdefault("min_elev", None)
        attrs.setdefault("area", None)
        attrs.setdefault("volume", None)
        attrs.setdefault("level", None)
        g.add_node(node_id, **attrs)

    existing_edges: dict[tuple[Any, Any], dict[str, Any]] = {}
    multi_key_counts: dict[tuple[Any, Any, str], int] = {}
    for edge in graph.get("edges", []):
        u = edge["from_id"]
        v = edge["to_id"]
        attrs = {k: val for k, val in edge.items() if k not in ("from_id", "to_id")}
        attrs.setdefault("pour_row", None)
        attrs.setdefault("pour_col", None)
        attrs.setdefault("spill_elev", None)

        if multi:
            edge_key = attrs.get("edge_type")
            if edge_key is not None:
                key_str = str(edge_key)
                count = multi_key_counts.get((u, v, key_str), 0) + 1
                multi_key_counts[(u, v, key_str)] = count
                unique_key = key_str if count == 1 else f"{key_str}#{count}"
                g.add_edge(u, v, key=unique_key, **attrs)
            else:
                g.add_edge(u, v, **attrs)
            continue

        pair = (u, v)
        if pair not in existing_edges:
            merged = dict(attrs)
            existing_edges[pair] = merged
            g.add_edge(u, v, **merged)
        else:
            prev = existing_edges[pair]
            prev_raw = prev.get("parallel_edges")
            if prev_raw is None:
                prev_snapshot = dict(prev)
                prev_types = [prev.get("edge_type", "unknown")]
                parallel_list: list[dict[str, Any]] = [prev_snapshot]
                # Preserve prefixed fields from the first edge on first collision.
                first_prefix = str(prev_types[0])
                for pk, pv in prev_snapshot.items():
                    if pk not in ("edge_type", "edge_types", "parallel_edges") and pv is not None:
                        prev[f"{first_prefix}_{pk}"] = pv
            else:
                prev_types = list(prev.get("edge_types", []))
                parallel_list = list(prev_raw)

            new_type = str(attrs.get("edge_type", "unknown"))
            parallel_list.append(dict(attrs))
            if new_type not in prev_types:
                prev_types.append(new_type)

            # Merge non-None attributes without clobbering existing non-None values;
            # store prefixed copies for the first incoming edge of each type so neither
            # distinct-type nor same-type parallel edges overwrite earlier attributes.
            for k, val in attrs.items():
                if k in ("edge_type", "edge_types", "parallel_edges"):
                    continue
                if val is not None:
                    prefixed = f"{new_type}_{k}"
                    if prefixed not in prev:
                        prev[prefixed] = val
                    if prev.get(k) is None or (k == "merge_time_h" and new_type == "merge"):
                        prev[k] = val

            prev["edge_type"] = "+".join(prev_types)
            prev["edge_types"] = prev_types
            prev["parallel_edges"] = parallel_list
            g.add_edge(u, v, **prev)

    return g


def array_to_xarray(
    arr: ArrayLike,
    geotransform: Geotransform | Sequence[float] | Any,
    crs: Any | None = None,
    name: str | None = None,
) -> Any:
    """Wrap a 2D (or 3D `(y, x, band)`) NumPy array as an `xarray.DataArray` with pixel-center coordinates.

    Requires the optional `xarray` package.
    """
    try:
        import xarray as xr  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise ImportError(
            "array_to_xarray requires xarray; install it with `pip install xarray`"
        ) from exc

    data = np.asarray(arr)
    if data.ndim not in (2, 3):
        raise ValueError(f"array_to_xarray expects a 2D or 3D array, got shape {data.shape}")

    gt = _normalize_transform(geotransform)
    if gt is None:
        raise ValueError("geotransform is required for array_to_xarray")

    rows, cols = data.shape[0], data.shape[1]
    attrs: dict[str, Any] = {"geotransform": gt}
    crs_str = _normalize_crs(crs)
    if crs_str is not None:
        attrs["crs"] = crs_str

    dims: tuple[str, ...] = ("y", "x") if data.ndim == 2 else ("y", "x", "band")
    if gt[2] == 0.0 and gt[4] == 0.0:
        x_coords = gt[0] + (np.arange(cols, dtype=np.float64) + 0.5) * gt[1]
        y_coords = gt[3] + (np.arange(rows, dtype=np.float64) + 0.5) * gt[5]
        coords: dict[str, Any] = {"y": y_coords, "x": x_coords}
    else:
        col_grid, row_grid = np.meshgrid(
            np.arange(cols, dtype=np.float64) + 0.5,
            np.arange(rows, dtype=np.float64) + 0.5,
        )
        x2d = gt[0] + col_grid * gt[1] + row_grid * gt[2]
        y2d = gt[3] + col_grid * gt[4] + row_grid * gt[5]
        coords = {"yc": (("y", "x"), y2d), "xc": (("y", "x"), x2d)}

    if data.ndim == 3:
        coords["band"] = np.arange(data.shape[2], dtype=np.int64)

    da = xr.DataArray(data, coords=coords, dims=dims, name=name, attrs=attrs)
    if crs_str is not None and hasattr(da, "rio"):
        write_crs = getattr(da.rio, "write_crs", None)
        if callable(write_crs):
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                write_crs(crs_str, inplace=True)
    return da


__all__ = [
    "array_to_xarray",
    "connectivity_to_networkx",
    "depressions_to_geodataframe",
    "extract_spatial_array",
    "is_spatial_array",
]
