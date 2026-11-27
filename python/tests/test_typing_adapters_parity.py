"""Tests for PEP 561 typing, spatial adapters, geotransform/CRS, and Rust-to-Python API parity."""

from __future__ import annotations

import ast
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import lidar_rs

TEST_DATA = Path(__file__).resolve().parents[2] / "test_data"


def _make_synthetic_point_cloud() -> dict[str, Any]:
    """Build a synthetic 12x12 ground grid (z ~ 100m) plus elevated canopy returns (z ~ 120m)."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    cls: list[int] = []
    inten: list[int] = []
    rn: list[int] = []
    nr: list[int] = []

    for r in range(12):
        for c in range(12):
            xs.append(500_000.0 + float(c))
            ys.append(4_000_000.0 + float(r))
            zs.append(100.0 + 0.05 * c)
            cls.append(1)
            inten.append(120 + r * 5 + c)
            rn.append(1)
            nr.append(1)

    # Elevated canopy points
    for r in range(4, 8):
        for c in range(4, 8):
            xs.append(500_000.0 + float(c) + 0.25)
            ys.append(4_000_000.0 + float(r) + 0.25)
            zs.append(120.0)
            cls.append(1)
            inten.append(450)
            rn.append(1)
            nr.append(1)

    return {
        "x": np.array(xs, dtype=np.float64),
        "y": np.array(ys, dtype=np.float64),
        "z": np.array(zs, dtype=np.float64),
        "classification": np.array(cls, dtype=np.uint8),
        "intensity": np.array(inten, dtype=np.uint16),
        "return_number": np.array(rn, dtype=np.uint8),
        "number_of_returns": np.array(nr, dtype=np.uint8),
        "crs": "EPSG:32615",
    }


def _make_two_basin_dem() -> np.ndarray:
    """Build a synthetic 30x30 DEM with two nested basins and a higher outer rim."""
    rr, cc = np.indices((30, 30), dtype=np.float64)
    dem = np.full((30, 30), 20.0, dtype=np.float64)
    d1 = np.hypot(rr - 10.0, cc - 10.0)
    d2 = np.hypot(rr - 18.0, cc - 18.0)
    basin1 = np.where(d1 < 6.0, 12.0 + d1 * 1.2, 20.0)
    basin2 = np.where(d2 < 6.0, 13.0 + d2 * 1.1, 20.0)
    dem = np.minimum(dem, np.minimum(basin1, basin2))
    # Outer enclosing bowl connecting both sub-basins
    d_outer = np.hypot(rr - 14.0, cc - 14.0)
    dem = np.minimum(dem, np.where(d_outer < 12.0, 16.5 + (d_outer / 12.0) * 3.0, 20.0))
    return dem


class _FakeAffine:
    """Minimal duck-typed affine.Affine transform."""

    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def to_gdal(self) -> tuple[float, float, float, float, float, float]:
        return (self.c, self.a, self.b, self.f, self.d, self.e)


class _FakeCRS:
    def __init__(self, code: str):
        self._code = code

    def to_string(self) -> str:
        return self._code


class _FakeRioAccessor:
    def __init__(self, transform: _FakeAffine, crs: _FakeCRS, nodata: float | None):
        self._transform = transform
        self.crs = crs
        self.nodata = nodata

    def transform(self) -> _FakeAffine:
        return self._transform


class _FakeDataArray:
    """Minimal duck-typed xarray.DataArray with optional `.rio` accessor."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        coords: dict[str, np.ndarray] | None = None,
        dims: tuple[str, ...] = ("y", "x"),
        attrs: dict[str, Any] | None = None,
        rio: _FakeRioAccessor | None = None,
    ):
        self.values = values
        self.coords = coords or {}
        self.dims = dims
        self.attrs = attrs or {}
        if rio is not None:
            self.rio = rio


class _FakeRasterioDataset:
    """Minimal duck-typed rasterio.DatasetReader."""

    def __init__(
        self,
        data: np.ndarray,
        transform: _FakeAffine,
        crs: _FakeCRS,
        nodata: float | None = None,
    ):
        self._data = data
        self.transform = transform
        self.crs = crs
        self.nodata = nodata

    def read(self, band: int = 1) -> np.ndarray:
        assert band == 1
        return self._data


# ---------------------------------------------------------------------------
# 1. PEP 561 typing, py.typed, _lidar_rs.pyi, __all__, and docstrings
# ---------------------------------------------------------------------------


def test_pep561_py_typed_and_stub_completeness():
    pkg_dir = Path(lidar_rs.__file__).resolve().parent
    py_typed = pkg_dir / "py.typed"
    stub_path = pkg_dir / "_lidar_rs.pyi"

    assert py_typed.is_file(), "PEP 561 py.typed marker must exist"
    assert stub_path.is_file(), "_lidar_rs.pyi type stub must exist"

    # Parse stub with ast to guarantee valid Python syntax
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    stub_funcs = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    stub_ann_names = {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "__version__" in stub_ann_names

    # Every public function exported by the compiled Rust module must exist in _lidar_rs.pyi
    ext_funcs = {
        name
        for name in dir(lidar_rs._lidar_rs)
        if not name.startswith("_") and callable(getattr(lidar_rs._lidar_rs, name))
    }
    missing = ext_funcs - stub_funcs
    extra = stub_funcs - ext_funcs
    assert not missing, f"Functions missing from _lidar_rs.pyi: {sorted(missing)}"
    assert not extra, f"Extra functions in _lidar_rs.pyi not in _lidar_rs: {sorted(extra)}"

    # Every symbol in lidar_rs.__all__ must resolve on the module
    for name in lidar_rs.__all__:
        assert hasattr(lidar_rs, name), f"{name!r} listed in __all__ but missing on lidar_rs"

    # Key TypedDict exports exist and declare expected runtime keys
    assert "id" in lidar_rs.DepressionRecord.__annotations__
    assert "stages" in lidar_rs.RatingCurve.__annotations__
    assert "nodes" in lidar_rs.ConnectivityGraph.__annotations__
    assert "x" in lidar_rs.PointCloudDict.__annotations__
    assert "ground_points" in lidar_rs.GroundClassifyResult.__annotations__


def test_delineate_depressions_tiled_docstring_notes_use_f32_semantics():
    doc = inspect.getdoc(lidar_rs.delineate_depressions_tiled) or ""
    assert "use_f32" in doc
    assert "fallback" in doc.lower() or "eager" in doc.lower()
    assert "f64" in doc


# ---------------------------------------------------------------------------
# 2. Geotransform & CRS support + duck-typed spatial adapters
# ---------------------------------------------------------------------------


def test_geotransform_and_crs_on_array_functions():
    dem = _make_two_basin_dem()
    gt_1m = (500_000.0, 1.0, 0.0, 4_000_030.0, 0.0, -1.0)
    gt_2m = (500_000.0, 2.0, 0.0, 4_000_060.0, 0.0, -2.0)

    s1 = lidar_rs.extract_sinks_array(dem, min_size=5, geotransform=gt_1m, crs="EPSG:32615")
    s2 = lidar_rs.extract_sinks_array(dem, min_size=5, geotransform=gt_2m, crs="EPSG:32615")
    assert s1["geotransform"] == gt_1m
    assert s2["geotransform"] == gt_2m
    assert s1["crs"] == "EPSG:32615"
    assert len(s1["depressions"]) >= 1
    assert s2["depressions"][0]["area"] == pytest.approx(
        s1["depressions"][0]["area"] * 4.0, rel=1e-6
    )
    assert s2["depressions"][0]["volume"] == pytest.approx(
        s1["depressions"][0]["volume"] * 4.0, rel=1e-6
    )

    d1 = lidar_rs.delineate_depressions_array(
        s1["sink"], min_size=5, min_depth=0.2, interval=0.5, geotransform=gt_1m, crs="EPSG:32615"
    )
    d2 = lidar_rs.delineate_depressions_array(
        s2["sink"], min_size=5, min_depth=0.2, interval=0.5, geotransform=gt_2m, crs="EPSG:32615"
    )
    assert d2["geotransform"] == gt_2m
    assert d2["crs"] == "EPSG:32615"
    assert d2["depressions"][0]["area"] == pytest.approx(
        d1["depressions"][0]["area"] * 4.0, rel=1e-6
    )

    sim = lidar_rs.simulate_inundation_array(
        dem,
        kernel_size=1,
        min_size=5,
        min_depth=0.2,
        interval=0.5,
        rain_intensity_cm_h=10.0,
        rain_duration_h=5.0,
        time_step_h=1.0,
        geotransform=gt_2m,
        crs="EPSG:32615",
    )
    assert sim["geotransform"] == gt_2m
    assert sim["crs"] == "EPSG:32615"
    assert len(sim["connectivity_graph"]["nodes"]) >= 1
    assert "spill_elev" in sim["connectivity_graph"]["nodes"][0]

    # Invalid geotransforms rejected cleanly
    with pytest.raises(ValueError, match="determinant"):
        lidar_rs.extract_sinks_array(dem, geotransform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        lidar_rs.extract_sinks_array(dem, geotransform=(float("nan"), 1.0, 0.0, 0.0, 0.0, -1.0))


def test_duck_typed_xarray_and_rasterio_input_adapters():
    dem = _make_two_basin_dem()
    aff = _FakeAffine(2.0, 0.0, 500_000.0, 0.0, -2.0, 4_000_060.0)
    expected_gt = (500_000.0, 2.0, 0.0, 4_000_060.0, 0.0, -2.0)

    # 1. Duck-typed xarray.DataArray with .rio accessor
    da_rio = _FakeDataArray(
        dem,
        rio=_FakeRioAccessor(aff, _FakeCRS("EPSG:32615"), -9999.0),
    )
    sinks_rio = lidar_rs.extract_sinks_array(da_rio, min_size=5)
    assert sinks_rio["geotransform"] == expected_gt
    assert sinks_rio["crs"] == "EPSG:32615"

    # 2. Duck-typed xarray.DataArray with plain x/y pixel-center coordinates (no .rio)
    xs = 500_000.0 + (np.arange(30, dtype=np.float64) + 0.5) * 2.0
    ys = 4_000_060.0 + (np.arange(30, dtype=np.float64) + 0.5) * (-2.0)
    da_coords = _FakeDataArray(
        dem,
        coords={"x": xs, "y": ys},
        attrs={"crs": "EPSG:32615"},
    )
    sinks_coords = lidar_rs.extract_sinks_array(da_coords, min_size=5)
    assert sinks_coords["geotransform"] == pytest.approx(expected_gt)
    assert sinks_coords["crs"] == "EPSG:32615"

    # 3. Duck-typed rasterio.DatasetReader
    rio_ds = _FakeRasterioDataset(dem, aff, _FakeCRS("EPSG:32615"), nodata=-9999.0)
    sinks_ds = lidar_rs.extract_sinks_array(rio_ds, min_size=5)
    assert sinks_ds["geotransform"] == expected_gt
    assert sinks_ds["crs"] == "EPSG:32615"

    # Explicit overrides win over adapter metadata
    override_gt = (10.0, 5.0, 0.0, 200.0, 0.0, -5.0)
    sinks_override = lidar_rs.extract_sinks_array(
        rio_ds, min_size=5, geotransform=override_gt, crs="EPSG:32614"
    )
    assert sinks_override["geotransform"] == override_gt
    assert sinks_override["crs"] == "EPSG:32614"

    # Polymorphic filters also accept duck-typed spatial objects
    filtered = lidar_rs.MeanFilter(da_rio, kernel_size=3)
    assert isinstance(filtered, np.ndarray) and filtered.shape == dem.shape
    hs = lidar_rs.compute_hillshade(rio_ds)
    assert hs.shape == (30, 30, 3) and hs.dtype == np.uint8

    # Un-georeferenced identity transform without CRS falls back to caller cell_size
    ident_aff = _FakeAffine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    ungeoref_ds = _FakeRasterioDataset(dem, ident_aff, crs=None, nodata=None)  # type: ignore[arg-type]
    sinks_ungeoref = lidar_rs.extract_sinks_array(ungeoref_ds, min_size=5, cell_size=3.0)
    assert sinks_ungeoref["geotransform"][1] == pytest.approx(3.0)
    assert sinks_ungeoref["geotransform"][5] == pytest.approx(-3.0)
    assert sinks_ungeoref["crs"] is None


def test_output_adapters_import_errors_and_conversion(monkeypatch: pytest.MonkeyPatch):
    dem = _make_two_basin_dem()
    gt = (500_000.0, 2.0, 0.0, 4_000_060.0, 0.0, -2.0)
    sinks = lidar_rs.extract_sinks_array(dem, min_size=5, geotransform=gt, crs="EPSG:32615")
    deps = lidar_rs.delineate_depressions_array(
        sinks["sink"], min_size=5, min_depth=0.2, interval=0.5, geotransform=gt, crs="EPSG:32615"
    )
    graph = lidar_rs.build_connectivity_graph(
        dem, min_size=5, min_depth=0.2, interval=0.5, geotransform=gt, crs="EPSG:32615"
    )

    # Clear ImportError when optional packages are missing
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "geopandas", None)
        with pytest.raises(ImportError, match="geopandas"):
            lidar_rs.depressions_to_geodataframe(deps)

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "networkx", None)
        with pytest.raises(ImportError, match="networkx"):
            lidar_rs.connectivity_to_networkx(graph)

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "xarray", None)
        with pytest.raises(ImportError, match="xarray"):
            lidar_rs.array_to_xarray(dem, gt, crs="EPSG:32615")

    # Lightweight stub modules to verify exact adapter contract even when optional deps aren't installed
    # 1. connectivity_to_networkx
    class _StubDiGraph:
        def __init__(self) -> None:
            self.graph: dict[str, Any] = {}
            self.nodes: dict[Any, dict[str, Any]] = {}
            self.edges: dict[tuple[Any, Any], dict[str, Any]] = {}

        def add_node(self, n: Any, **attrs: Any) -> None:
            self.nodes[n] = dict(attrs)

        def add_edge(self, u: Any, v: Any, **attrs: Any) -> None:
            self.edges[(u, v)] = dict(attrs)

    class _StubMultiDiGraph:
        def __init__(self) -> None:
            self.graph: dict[str, Any] = {}
            self.nodes: dict[Any, dict[str, Any]] = {}
            self.edges: dict[tuple[Any, Any, Any], dict[str, Any]] = {}

        def add_node(self, n: Any, **attrs: Any) -> None:
            self.nodes[n] = dict(attrs)

        def add_edge(self, u: Any, v: Any, key: Any = None, **attrs: Any) -> None:
            self.edges[(u, v, key)] = dict(attrs)

    fake_nx = types.ModuleType("networkx")
    fake_nx.DiGraph = _StubDiGraph  # type: ignore[attr-defined]
    fake_nx.MultiDiGraph = _StubMultiDiGraph  # type: ignore[attr-defined]
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "networkx", fake_nx)
        g = lidar_rs.connectivity_to_networkx(graph)
        assert isinstance(g, _StubDiGraph)
        assert len(g.nodes) == len(graph["nodes"]) >= 1
        first_node = next(iter(g.nodes.values()))
        for attr in ("min_elev", "spill_elev", "area", "volume", "level"):
            assert attr in first_node
        if graph["edges"]:
            first_edge = next(iter(g.edges.values()))
            for attr in ("pour_row", "pour_col", "spill_elev"):
                assert attr in first_edge

        # Parallel "spill" + "merge" edges between the same (u, v) pair must preserve both
        # attributes in DiGraph mode and stay distinct keyed edges when multi=True.
        parallel_graph = {
            "rain_intensity_cm_h": 5.0,
            "nodes": [
                {"id": 1, "min_elev": 1.0, "max_elev": 2.0, "area": 10.0, "volume": 5.0, "level": 1},
                {"id": 2, "min_elev": 0.5, "max_elev": 3.0, "area": 40.0, "volume": 30.0, "level": 2},
            ],
            "edges": [
                {
                    "from_id": 1,
                    "to_id": 2,
                    "edge_type": "spill",
                    "spill_time_h": 1.5,
                    "spill_runoff_mm": 75.0,
                    "merge_time_h": 4.0,
                    "spill_elev": 2.0,
                    "length_m": 12.5,
                    "start_xy": [10.0, 20.0],
                    "end_xy": [22.5, 20.0],
                    "dist_to_stream": 3.0,
                },
                {
                    "from_id": 1,
                    "to_id": 2,
                    "edge_type": "merge",
                    "spill_time_h": None,
                    "spill_runoff_mm": None,
                    "merge_time_h": 4.5,
                    "spill_elev": 2.1,
                    "length_m": 0.0,
                    "start_xy": None,
                    "end_xy": None,
                    "dist_to_stream": None,
                },
            ],
        }
        g_di = lidar_rs.connectivity_to_networkx(parallel_graph)
        assert isinstance(g_di, _StubDiGraph)
        merged_edge = g_di.edges[(1, 2)]
        assert merged_edge["edge_type"] == "spill+merge"
        assert merged_edge["edge_types"] == ["spill", "merge"]
        assert merged_edge["spill_time_h"] == 1.5
        assert merged_edge["spill_runoff_mm"] == 75.0
        assert merged_edge["length_m"] == 12.5
        assert merged_edge["spill_merge_time_h"] == 4.0
        assert merged_edge["merge_merge_time_h"] == 4.5
        assert merged_edge["merge_time_h"] == 4.5
        assert len(merged_edge["parallel_edges"]) == 2

        g_multi = lidar_rs.connectivity_to_networkx(parallel_graph, multi=True)
        assert isinstance(g_multi, _StubMultiDiGraph)
        assert (1, 2, "spill") in g_multi.edges
        assert (1, 2, "merge") in g_multi.edges
        assert g_multi.edges[(1, 2, "spill")]["length_m"] == 12.5
        assert g_multi.edges[(1, 2, "merge")]["merge_time_h"] == 4.5

        # Multiple same-type ("spill" + "spill") edges between the same (u, v) pair
        # must keep distinct keys in MultiDiGraph and preserve first spill_* in DiGraph.
        dup_spill_graph = {
            "rain_intensity_cm_h": 5.0,
            "nodes": parallel_graph["nodes"],
            "edges": [
                parallel_graph["edges"][0],
                {
                    "from_id": 1,
                    "to_id": 2,
                    "edge_type": "spill",
                    "spill_time_h": 2.0,
                    "spill_runoff_mm": 100.0,
                    "merge_time_h": 4.0,
                    "spill_elev": 2.2,
                    "length_m": 25.0,
                    "start_xy": [10.0, 21.0],
                    "end_xy": [35.0, 21.0],
                    "dist_to_stream": 5.0,
                },
            ],
        }
        g_dup_di = lidar_rs.connectivity_to_networkx(dup_spill_graph)
        dup_edge = g_dup_di.edges[(1, 2)]
        assert dup_edge["spill_length_m"] == 12.5
        assert len(dup_edge["parallel_edges"]) == 2
        assert dup_edge["parallel_edges"][1]["length_m"] == 25.0

        g_dup_multi = lidar_rs.connectivity_to_networkx(dup_spill_graph, multi=True)
        assert (1, 2, "spill") in g_dup_multi.edges
        assert (1, 2, "spill#2") in g_dup_multi.edges
        assert g_dup_multi.edges[(1, 2, "spill")]["length_m"] == 12.5
        assert g_dup_multi.edges[(1, 2, "spill#2")]["length_m"] == 25.0

    # 2. depressions_to_geodataframe
    class _StubPolygon:
        def __init__(self, coords: list[tuple[float, float]] | None = None) -> None:
            self.coords = coords or []

    class _StubGeoDataFrame:
        def __init__(
            self,
            data: list[dict[str, Any]],
            *,
            geometry: list[Any],
            crs: str | None = None,
        ) -> None:
            self.data = data
            self.geometry = geometry
            self.crs = crs

    fake_gpd = types.ModuleType("geopandas")
    fake_gpd.GeoDataFrame = _StubGeoDataFrame  # type: ignore[attr-defined]
    fake_shapely = types.ModuleType("shapely")
    fake_shapely_geom = types.ModuleType("shapely.geometry")
    fake_shapely_geom.Polygon = _StubPolygon  # type: ignore[attr-defined]
    fake_shapely_ops = types.ModuleType("shapely.ops")
    fake_shapely_ops.unary_union = lambda geoms: ("union", list(geoms))  # type: ignore[attr-defined]

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "geopandas", fake_gpd)
        mp.setitem(sys.modules, "shapely", fake_shapely)
        mp.setitem(sys.modules, "shapely.geometry", fake_shapely_geom)
        mp.setitem(sys.modules, "shapely.ops", fake_shapely_ops)
        gdf = lidar_rs.depressions_to_geodataframe(deps)
        assert isinstance(gdf, _StubGeoDataFrame)
        assert gdf.crs == "EPSG:32615"
        assert len(gdf.data) == len(deps["depressions"]) >= 1
        assert len(gdf.geometry) == len(deps["depressions"])

    # 3. array_to_xarray + exact round-trip back through extract_spatial_array
    class _StubDataArray:
        def __init__(
            self,
            data: np.ndarray,
            *,
            coords: dict[str, Any],
            dims: tuple[str, ...],
            name: str | None = None,
            attrs: dict[str, Any] | None = None,
        ) -> None:
            self.values = data
            self.coords = coords
            self.dims = dims
            self.name = name
            self.attrs = attrs or {}

    fake_xr = types.ModuleType("xarray")
    fake_xr.DataArray = _StubDataArray  # type: ignore[attr-defined]
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "xarray", fake_xr)
        da = lidar_rs.array_to_xarray(dem, gt, crs="EPSG:32615", name="dem")
        assert da.dims == ("y", "x")
        assert da.name == "dem"
        assert da.attrs["crs"] == "EPSG:32615"
        assert da.coords["x"][0] == pytest.approx(500_001.0)
        assert da.coords["y"][0] == pytest.approx(4_000_059.0)
        # Round-trip back into extract_sinks_array
        sinks_rt = lidar_rs.extract_sinks_array(da, min_size=5)
        assert sinks_rt["geotransform"] == pytest.approx(gt)
        assert sinks_rt["crs"] == "EPSG:32615"

        # Rotated geotransform uses 2D non-dimension coords ("yc", "xc") to avoid colliding with ("y", "x")
        rot_gt = (500_000.0, 2.0, 0.5, 4_000_060.0, -0.5, -2.0)
        da_rot = lidar_rs.array_to_xarray(dem, rot_gt, crs="EPSG:32615")
        assert da_rot.dims == ("y", "x")
        assert "yc" in da_rot.coords and "xc" in da_rot.coords
        assert "y" not in da_rot.coords and "x" not in da_rot.coords
        assert da_rot.coords["xc"][0] == ("y", "x")
        assert da_rot.coords["xc"][1].shape == dem.shape
        assert da_rot.coords["xc"][1][0, 0] == pytest.approx(500_000.0 + 0.5 * 2.0 + 0.5 * 0.5)
        assert da_rot.coords["yc"][1][0, 0] == pytest.approx(4_000_060.0 + 0.5 * (-0.5) + 0.5 * (-2.0))


# ---------------------------------------------------------------------------
# 3. Rust-to-Python API parity functions
# ---------------------------------------------------------------------------


def test_compute_rating_curves_and_build_connectivity_graph():
    dem_arr = lidar_rs.MedianFilter(TEST_DATA / "dem.tif", kernel_size=3)
    assert isinstance(dem_arr, np.ndarray)
    gt = (500_000.0, 1.0, 0.0, 4_000_400.0, 0.0, -1.0)

    sinks = lidar_rs.extract_sinks_array(dem_arr, min_size=1000, geotransform=gt, crs="EPSG:32614")
    deps = lidar_rs.delineate_depressions_array(
        sinks["sink"], min_size=1000, min_depth=0.3, interval=0.3, geotransform=gt, crs="EPSG:32614"
    )

    # Rating curves with precomputed delineation
    curves = lidar_rs.compute_rating_curves(sinks["sink"], deps, interval=0.3)
    assert len(curves) == len(deps["depressions"])
    c0 = curves[0]
    for key in ("id", "dep_id", "level", "region_id", "stages", "depths", "areas", "volumes"):
        assert key in c0
    assert len(c0["stages"]) >= 2
    assert c0["stages"] == sorted(c0["stages"])
    assert c0["areas"] == sorted(c0["areas"])
    assert c0["volumes"] == sorted(c0["volumes"])
    assert c0["volumes"][-1] > 0.0

    # Rating curves standalone (auto-delineate from sink_arr)
    curves_auto = lidar_rs.compute_rating_curves(
        sinks["sink"], interval=0.3, min_size=1000, min_depth=0.3, geotransform=gt
    )
    assert len(curves_auto) == len(curves)
    assert curves_auto[0]["volumes"][-1] == pytest.approx(c0["volumes"][-1])

    # Connectivity graph directly from DEM array
    graph = lidar_rs.build_connectivity_graph(
        dem_arr,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=20.0,
        time_step_h=2.0,
        geotransform=gt,
        crs="EPSG:32614",
    )
    assert len(graph["nodes"]) >= 2
    assert len(graph["edges"]) >= 1
    assert "spill_elev" in graph["nodes"][0]
    assert "pour_row" in graph["edges"][0] and "pour_col" in graph["edges"][0]
    spill_edges = [e for e in graph["edges"] if e["edge_type"] == "spill"]
    if spill_edges:
        assert spill_edges[0]["pour_row"] is not None
        assert spill_edges[0]["pour_col"] is not None

    # Connectivity graph from precomputed delineation dict
    graph_parts = lidar_rs.build_connectivity_graph(deps, rain_intensity_cm_h=5.0)
    assert len(graph_parts["nodes"]) == len(deps["depressions"])

    # Parts-only arguments rejected cleanly when passing a DEM array
    with pytest.raises(ValueError, match="delineation dict"):
        lidar_rs.build_connectivity_graph(dem_arr, fill_time=np.zeros_like(dem_arr))
    with pytest.raises(ValueError, match="delineation dict"):
        lidar_rs.build_connectivity_graph(dem_arr, catchment_records=[])
    with pytest.raises(ValueError, match="delineation dict"):
        lidar_rs.build_connectivity_graph(dem_arr, spill_paths=[])


def test_delineate_mounts_array_and_compute_hillshade():
    # Synthetic mound
    rr, cc = np.indices((25, 25), dtype=np.float64)
    dsm = np.full((25, 25), 10.0, dtype=np.float64)
    dist = np.hypot(rr - 12.0, cc - 12.0)
    dsm = np.where(dist < 7.0, 22.0 - dist * 1.5, dsm)
    gt = (1000.0, 2.0, 0.0, 2050.0, 0.0, -2.0)

    mounts = lidar_rs.delineate_mounts_array(
        dsm, min_size=10, min_height=0.5, interval=0.5, geotransform=gt, crs="EPSG:32615"
    )
    assert mounts["depression_id"].shape == (25, 25)
    assert mounts["depression_level"].shape == (25, 25)
    assert mounts["max_level"] >= 1
    assert len(mounts["depressions"]) >= 1
    assert mounts["geotransform"] == gt
    assert mounts["crs"] == "EPSG:32615"

    # Real DSM fixture in memory
    dsm_arr = lidar_rs.MeanFilter(TEST_DATA / "dsm.tif", kernel_size=1)
    assert isinstance(dsm_arr, np.ndarray)
    dsm_mounts = lidar_rs.delineate_mounts_array(
        dsm_arr, min_size=1000, min_height=0.3, interval=0.3
    )
    assert len(dsm_mounts["depressions"]) >= 2

    # Hillshade on array and on file
    hs_arr = lidar_rs.compute_hillshade(
        dsm, azimuth=315.0, altitude=45.0, z_factor=1.5, geotransform=gt
    )
    assert hs_arr.shape == (25, 25, 3)
    assert hs_arr.dtype == np.uint8

    hs_east = lidar_rs.compute_hillshade(
        dsm, azimuth=90.0, altitude=45.0, z_factor=1.5, geotransform=gt
    )
    assert not np.array_equal(hs_arr, hs_east)

    hs_file = lidar_rs.compute_hillshade(TEST_DATA / "dem.tif", azimuth=315.0, altitude=45.0)
    assert hs_file.shape == (400, 400, 3)
    assert hs_file.dtype == np.uint8

    with pytest.raises(ValueError, match="file path"):
        lidar_rs.compute_hillshade(TEST_DATA / "dem.tif", geotransform=gt)
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.compute_hillshade(TEST_DATA / "dem.tif", crs="EPSG:32615")
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.breach_depressions(TEST_DATA / "dem.tif", "unused.tif", geotransform=gt)
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.breach_depressions(TEST_DATA / "dem.tif", "unused.tif", crs="EPSG:32615")


def test_read_write_lidar_numpy_roundtrip_and_filtering(tmp_path: Path):
    pts = _make_synthetic_point_cloud()
    las_path = tmp_path / "roundtrip.las"
    laz_path = tmp_path / "roundtrip.laz"

    written = lidar_rs.write_lidar(
        las_path,
        pts["x"],
        pts["y"],
        pts["z"],
        classification=pts["classification"],
        intensity=pts["intensity"],
        return_number=pts["return_number"],
        number_of_returns=pts["number_of_returns"],
        crs=pts["crs"],
    )
    assert written == len(pts["x"])
    assert las_path.exists()

    cloud = lidar_rs.read_lidar(las_path)
    assert cloud["count"] == written
    np.testing.assert_allclose(cloud["x"], pts["x"], atol=1e-2)
    np.testing.assert_allclose(cloud["y"], pts["y"], atol=1e-2)
    np.testing.assert_allclose(cloud["z"], pts["z"], atol=1e-2)
    assert cloud["classification"] is not None
    np.testing.assert_array_equal(cloud["classification"], pts["classification"])
    assert cloud["intensity"] is not None
    np.testing.assert_array_equal(cloud["intensity"], pts["intensity"])
    assert cloud["return_number"] is not None
    np.testing.assert_array_equal(cloud["return_number"], pts["return_number"])
    assert cloud["number_of_returns"] is not None
    np.testing.assert_array_equal(cloud["number_of_returns"], pts["number_of_returns"])
    assert cloud["crs"] == "EPSG:32615"

    # LAZ roundtrip + bounds + max_points + polygon_path
    lidar_rs.write_lidar(laz_path, pts["x"], pts["y"], pts["z"], crs=pts["crs"])
    sub_bounds = lidar_rs.read_lidar(
        laz_path, bounds=(500_000.0, 4_000_000.0, 500_005.0, 4_000_005.0)
    )
    assert 0 < sub_bounds["count"] < written
    assert np.all(sub_bounds["x"] <= 500_005.0 + 1e-6)
    assert np.all(sub_bounds["y"] <= 4_000_005.0 + 1e-6)

    sub_capped = lidar_rs.read_lidar(laz_path, max_points=25)
    assert sub_capped["count"] <= 25

    # Bare string return_kinds should be treated as a single return kind, not split into chars
    sub_return = lidar_rs.read_lidar(las_path, return_kinds="only")
    assert sub_return["count"] == written

    poly_gj = tmp_path / "clip.geojson"
    poly_gj.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:32615"}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [500_002.0, 4_000_002.0],
                                    [500_006.0, 4_000_002.0],
                                    [500_006.0, 4_000_006.0],
                                    [500_002.0, 4_000_006.0],
                                    [500_002.0, 4_000_002.0],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sub_poly = lidar_rs.read_lidar(laz_path, polygon_path=poly_gj)
    assert 0 < sub_poly["count"] < written
    assert np.all((sub_poly["x"] >= 500_002.0 - 1e-6) & (sub_poly["x"] <= 500_006.0 + 1e-6))

    # Mismatched array lengths raise ValueError
    with pytest.raises(ValueError, match="identical lengths"):
        lidar_rs.write_lidar(tmp_path / "bad.las", pts["x"][:5], pts["y"], pts["z"])

    # Out-of-range and non-integer unsigned attribute casts raise ValueError
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        lidar_rs.write_lidar(
            tmp_path / "bad_u8_hi.las",
            pts["x"][:2],
            pts["y"][:2],
            pts["z"][:2],
            classification=[2, 258],
        )
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        lidar_rs.write_lidar(
            tmp_path / "bad_u8_neg.las",
            pts["x"][:2],
            pts["y"][:2],
            pts["z"][:2],
            classification=[-1, 2],
        )
    with pytest.raises(ValueError, match="integer"):
        lidar_rs.write_lidar(
            tmp_path / "bad_float.las",
            pts["x"][:2],
            pts["y"][:2],
            pts["z"][:2],
            classification=np.array([2.5, 1.0]),
        )
    with pytest.raises(ValueError, match=r"\[0, 65535\]"):
        lidar_rs.write_lidar(
            tmp_path / "bad_u16.las",
            pts["x"][:2],
            pts["y"][:2],
            pts["z"][:2],
            intensity=[100, 70_000],
        )
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        lidar_rs.classify_ground(
            {
                "x": pts["x"][:2],
                "y": pts["y"][:2],
                "z": pts["z"][:2],
                "classification": [2, 300],
            },
            method="csf",
        )

    # Bbox wording differentiates geographic federal queries vs Cartesian read_lidar bounds
    with pytest.raises(ValueError, match="min_lon, min_lat, max_lon, max_lat"):
        lidar_rs.query_3dep_products(bbox=(1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_x, min_y, max_x, max_y"):
        lidar_rs.read_lidar(las_path, bounds=(1.0, 2.0))  # type: ignore[arg-type]


def test_mask_to_geometry_and_bbox_slicing_exact_l_shape(monkeypatch: pytest.MonkeyPatch):
    from lidar_rs._adapters import _mask_to_geometry

    # L-shaped depression (id=2) with child (id=1) placed at offset (row=2..4, col=3..5)
    dep_id = np.zeros((7, 8), dtype=np.int32)
    dep_id[2, 3] = 1  # top-left pixel of L (child)
    dep_id[3, 3] = 2  # middle-left pixel of L
    dep_id[4, 3:6] = 2  # bottom horizontal arm (cols 3, 4, 5)
    gt = (100.0, 2.0, 0.0, 214.0, 0.0, -2.0)

    class _StubPolygon:
        def __init__(self, coords: list[tuple[float, float]] | None = None) -> None:
            self.coords = coords or []

    # Direct check of _mask_to_geometry strips with row/col offset
    sub_mask = dep_id[2:5, 3:6] > 0
    strips = _mask_to_geometry(
        sub_mask, gt, _StubPolygon, lambda gs: list(gs), row_offset=2, col_offset=3
    )
    assert len(strips) == 3
    # Row 2, col [3, 4) -> x in [106, 108], y in [208, 210]
    assert strips[0].coords == [
        (106.0, 210.0),
        (108.0, 210.0),
        (108.0, 208.0),
        (106.0, 208.0),
        (106.0, 210.0),
    ]
    # Row 3, col [3, 4) -> x in [106, 108], y in [206, 208]
    assert strips[1].coords == [
        (106.0, 208.0),
        (108.0, 208.0),
        (108.0, 206.0),
        (106.0, 206.0),
        (106.0, 208.0),
    ]
    # Row 4, col [3, 6) -> x in [106, 112], y in [204, 206]
    assert strips[2].coords == [
        (106.0, 206.0),
        (112.0, 206.0),
        (112.0, 204.0),
        (106.0, 204.0),
        (106.0, 206.0),
    ]

    # End-to-end depressions_to_geodataframe with compound parent (id=2 enclosing child id=1)
    class _StubGeoDataFrame:
        def __init__(
            self,
            data: list[dict[str, Any]],
            *,
            geometry: list[Any],
            crs: str | None = None,
        ) -> None:
            self.data = data
            self.geometry = geometry
            self.crs = crs

    fake_gpd = types.ModuleType("geopandas")
    fake_gpd.GeoDataFrame = _StubGeoDataFrame  # type: ignore[attr-defined]
    fake_shapely = types.ModuleType("shapely")
    fake_shapely_geom = types.ModuleType("shapely.geometry")
    fake_shapely_geom.Polygon = _StubPolygon  # type: ignore[attr-defined]
    fake_shapely_ops = types.ModuleType("shapely.ops")
    fake_shapely_ops.unary_union = lambda geoms: list(geoms)  # type: ignore[attr-defined]

    result = {
        "depression_id": dep_id,
        "geotransform": gt,
        "crs": "EPSG:32615",
        "depressions": [
            {"id": 1, "level": 1, "children_ids": []},
            {"id": 2, "level": 2, "children_ids": [1]},
        ],
    }
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "geopandas", fake_gpd)
        mp.setitem(sys.modules, "shapely", fake_shapely)
        mp.setitem(sys.modules, "shapely.geometry", fake_shapely_geom)
        mp.setitem(sys.modules, "shapely.ops", fake_shapely_ops)
        gdf = lidar_rs.depressions_to_geodataframe(result)
        # Child (id=1): single 1x1 pixel strip at row 2, col 3
        assert isinstance(gdf.geometry[0], _StubPolygon)
        assert gdf.geometry[0].coords == strips[0].coords
        # Parent (id=2): compound L-shape unioning child id=1 + parent id=2 -> all 3 strips
        assert len(gdf.geometry[1]) == 3
        assert [p.coords for p in gdf.geometry[1]] == [s.coords for s in strips]

    # When real shapely is importable, also verify exact polygon area and validity
    try:
        from shapely.geometry import Polygon as RealPolygon  # type: ignore[import-untyped,import-not-found]
        from shapely.ops import unary_union as real_union  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        return
    geom = _mask_to_geometry(
        sub_mask, gt, RealPolygon, real_union, row_offset=2, col_offset=3
    )
    assert geom.is_valid
    # 5 pixels * (2.0 * 2.0) = 20.0 map units squared
    assert geom.area == pytest.approx(20.0)


def test_classify_ground_and_tunable_dtm_chm_options(tmp_path: Path):
    pts = _make_synthetic_point_cloud()
    las_path = tmp_path / "forest.las"
    lidar_rs.write_lidar(
        las_path,
        pts["x"],
        pts["y"],
        pts["z"],
        classification=pts["classification"],
        intensity=pts["intensity"],
        return_number=pts["return_number"],
        number_of_returns=pts["number_of_returns"],
        crs=pts["crs"],
    )

    # 1. File-based CSF with all tunable CSF parameters
    csf_out = tmp_path / "forest_csf.las"
    res_csf = lidar_rs.classify_ground(
        las_path,
        csf_out,
        method="csf",
        cloth_resolution=1.5,
        rigidness=2,
        class_threshold=0.5,
        slope_smooth=True,
        iterations=300,
        time_step=0.65,
    )
    assert res_csf["method"] == "csf"
    assert res_csf["output_path"] == str(csf_out)
    assert csf_out.exists()
    assert res_csf["total_points"] == len(pts["x"])
    assert res_csf["ground_points"] > 0
    assert res_csf["non_ground_points"] > 0
    assert res_csf["ground_points"] + res_csf["non_ground_points"] == res_csf["total_points"]
    # Canopy points (z=120) must not be classified as ground (class 2)
    canopy_mask = pts["z"] > 110.0
    assert np.all(res_csf["classification"][canopy_mask] != 2)

    # 2. In-memory PointCloudDict with PMF tuning
    cloud = lidar_rs.read_lidar(las_path)
    res_pmf = lidar_rs.classify_ground(
        cloud,
        method="pmf",
        max_window_size=15.0,
        slope=0.5,
        initial_distance=0.2,
        max_distance=2.0,
        cell_size=1.0,
    )
    assert res_pmf["method"] == "pmf"
    assert res_pmf["output_path"] is None
    assert res_pmf["ground_points"] > 0
    assert np.all(res_pmf["classification"][canopy_mask] != 2)

    # 3. File-based SMRF without output file, with all tunable SMRF parameters
    res_smrf = lidar_rs.classify_ground(
        las_path,
        method="smrf",
        cell_size=1.0,
        slope=0.15,
        window_radius=12.0,
        elevation_threshold=0.5,
        elevation_scalar=1.25,
    )
    assert res_smrf["method"] == "smrf"
    assert res_smrf["ground_points"] > 0
    assert np.all(res_smrf["classification"][canopy_mask] != 2)

    # Validation errors on invalid parameters / method
    with pytest.raises(ValueError, match="rigidness"):
        lidar_rs.classify_ground(las_path, method="csf", rigidness=5)
    with pytest.raises(ValueError, match="csf"):
        lidar_rs.classify_ground(las_path, method="preclassified")

    # 4. Tunable CSF/PMF/SMRF parameters wired into lidar_to_dtm and lidar_to_chm
    dtm_csf = tmp_path / "dtm_csf.tif"
    cols, rows = lidar_rs.lidar_to_dtm(
        las_path,
        dtm_csf,
        resolution=1.0,
        ground_method="csf",
        cloth_resolution=1.5,
        rigidness=2,
        class_threshold=0.5,
        slope_smooth=True,
        iterations=300,
        time_step=0.65,
    )
    assert cols >= 10 and rows >= 10 and dtm_csf.exists()

    dtm_pmf = tmp_path / "dtm_pmf.tif"
    cols, rows = lidar_rs.lidar_to_dtm(
        las_path,
        dtm_pmf,
        resolution=1.0,
        ground_method="pmf",
        max_window_size=15.0,
        slope=0.5,
        initial_distance=0.2,
        max_distance=2.0,
        cell_size=1.0,
    )
    assert cols >= 10 and rows >= 10 and dtm_pmf.exists()

    chm_smrf = tmp_path / "chm_smrf.tif"
    cols, rows = lidar_rs.lidar_to_chm(
        las_path,
        chm_smrf,
        resolution=1.0,
        ground_method="smrf",
        cell_size=1.0,
        slope=0.15,
        window_radius=12.0,
        elevation_threshold=0.5,
        elevation_scalar=1.25,
    )
    assert cols >= 10 and rows >= 10 and chm_smrf.exists()


def test_audit_v06_python_forcing_nodata_types_and_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify spatial-adapter NoData preservation, FsmBasinSummary typing, and Sequence[str] datasets."""
    from collections import UserList
    from typing import get_type_hints

    # 1. FsmBasinSummary.spill_to_dep_id must allow None (unspilled basins).
    hints = get_type_hints(lidar_rs.FsmBasinSummary)
    assert type(None) in getattr(hints["spill_to_dep_id"], "__args__", ())

    # 2. simulate_inundation_array masks finite NoData sentinels from spatial-adapter forcing inputs.
    class _FakeRasterio:
        def __init__(self, data: np.ndarray, nodata: float) -> None:
            self._data = data
            self.nodata = nodata
            self.transform = (0.0, 1.0, 0.0, float(data.shape[0]), 0.0, -1.0)
            self.crs = "EPSG:32615"

        def read(self, band: int) -> np.ndarray:
            assert band == 1
            return self._data

    dem = np.full((9, 9), 15.0, dtype=np.float64)
    dem[3:6, 3:6] = 10.0
    dem[4, 4] = 8.0
    rc_data = np.full((9, 9), 0.5, dtype=np.float64)
    rc_data[0, 0] = -9999.0
    fake_rc = _FakeRasterio(rc_data, nodata=-9999.0)

    captured: dict[str, Any] = {}
    orig_sim = lidar_rs._lidar_rs.simulate_inundation_array

    def _spy_sim(*args: Any, **kwargs: Any) -> Any:
        captured["rc"] = args[8]
        captured["rc_nd"] = args[18]
        return orig_sim(*args, **kwargs)

    monkeypatch.setattr(lidar_rs._lidar_rs, "simulate_inundation_array", _spy_sim)
    res_rc = lidar_rs.simulate_inundation_array(
        dem,
        kernel_size=1,
        min_size=1,
        min_depth=0.5,
        interval=0.5,
        rain_intensity_cm_h=5.0,
        rain_duration_h=5.0,
        time_step_h=1.0,
        runoff_coeff_arr=fake_rc,
    )
    assert captured["rc"][0, 0] == pytest.approx(-9999.0), (
        "forcing array must be forwarded unmutated to Rust"
    )
    assert captured["rc_nd"] == pytest.approx(-9999.0), (
        "finite runoff_coeff NoData sentinel must be plumbed through PyO3 ABI"
    )
    assert captured["rc"][4, 4] == pytest.approx(0.5)
    assert res_rc["fill_time"].shape == dem.shape

    # 3. download_federal_data accepts arbitrary Sequence[str] (not just list/tuple).
    seen_ds: dict[str, str] = {}

    def _fake_download(*args: Any) -> dict[str, Any]:
        seen_ds["spec"] = args[4]
        return {}

    monkeypatch.setattr(lidar_rs._lidar_rs, "download_federal_data", _fake_download)
    lidar_rs.download_federal_data(
        "out",
        bbox=(-99.3, 47.0, -99.1, 47.2),
        datasets=UserList(["dem", "nhd"]),
    )
    assert seen_ds["spec"] == "dem,nhd"


def test_compute_twi_and_hand_array_and_file(tmp_path: Path):
    # Tilted V-shaped valley draining east along row 10: higher TWI and HAND == 0 in valley floor
    rr, cc = np.indices((21, 21), dtype=np.float64)
    dem = 100.0 + np.abs(rr - 10.0) * 1.5 + (20.0 - cc) * 0.4
    gt = (500_000.0, 2.0, 0.0, 4_000_042.0, 0.0, -2.0)

    twi = lidar_rs.compute_twi(dem, geotransform=gt, crs="EPSG:32615")
    assert twi.shape == dem.shape
    assert twi.dtype == np.float64
    assert np.all(np.isfinite(twi))
    # Valley bottom accumulates more flow and is wetter than ridge crest
    assert twi[10, 18] > twi[0, 2]

    # Custom flow_acc array is actually used and alters TWI values
    custom_acc = np.full_like(dem, 10.0)
    custom_acc[10, :] = 500.0
    twi_custom = lidar_rs.compute_twi(dem, flow_acc=custom_acc, geotransform=gt)
    assert twi_custom[10, 10] > twi_custom[2, 10]
    assert not np.allclose(twi_custom, twi)

    # Duck-typed xarray adapter works seamlessly
    aff = _FakeAffine(2.0, 0.0, 500_000.0, 0.0, -2.0, 4_000_042.0)
    da_rio = _FakeDataArray(dem, rio=_FakeRioAccessor(aff, _FakeCRS("EPSG:32615"), -9999.0))
    twi_da = lidar_rs.compute_twi(da_rio)
    np.testing.assert_allclose(twi_da, twi, rtol=1e-12)

    # HAND with explicit boolean stream mask along the valley center
    stream_mask = np.zeros(dem.shape, dtype=bool)
    stream_mask[10, :] = True
    hand_mask = lidar_rs.compute_hand(dem, stream_mask=stream_mask, geotransform=gt)
    assert hand_mask.shape == dem.shape
    assert np.allclose(hand_mask[10, :], 0.0)
    # Interior cells drain into row 10 (grid border row 0 drains off-grid -> NoData)
    assert hand_mask[2, 10] > hand_mask[5, 10] > 0.0

    # HAND with accumulation threshold
    hand_thresh = lidar_rs.compute_hand(dem, acc_threshold_cells=25.0, geotransform=gt)
    assert hand_thresh.shape == dem.shape
    assert hand_thresh[10, 18] == pytest.approx(0.0)
    assert hand_thresh[2, 10] > 0.0

    # File-based TWI and HAND (both returning array and writing GeoTIFF)
    twi_file_arr = lidar_rs.compute_twi(TEST_DATA / "dem.tif")
    assert twi_file_arr.shape == (400, 400)
    twi_out = tmp_path / "twi.tif"
    assert lidar_rs.compute_twi(TEST_DATA / "dem.tif", out_file=twi_out, compress=True) == str(
        twi_out
    )
    assert twi_out.exists()

    hand_file_arr = lidar_rs.compute_hand(TEST_DATA / "dem.tif", acc_threshold_cells=500.0)
    assert hand_file_arr.shape == (400, 400)
    hand_out = tmp_path / "hand.tif"
    assert lidar_rs.compute_hand(
        TEST_DATA / "dem.tif",
        out_file=hand_out,
        acc_threshold_cells=500.0,
        compress=True,
    ) == str(hand_out)
    assert hand_out.exists()

    # Validation errors
    with pytest.raises(ValueError, match="out_file"):
        lidar_rs.compute_twi(dem, out_file=tmp_path / "unused.tif")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.compute_twi(TEST_DATA / "dem.tif", geotransform=gt)
    with pytest.raises(ValueError, match="stream_mask or acc_threshold_cells"):
        lidar_rs.compute_hand(dem)
    with pytest.raises(ValueError, match="not both"):
        lidar_rs.compute_hand(dem, stream_mask=stream_mask, acc_threshold_cells=10.0)
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.compute_hand(TEST_DATA / "dem.tif", acc_threshold_cells=100.0, crs="EPSG:32615")
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.compute_twi(TEST_DATA / "dem.tif", cell_size=2.0)
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.compute_hand(TEST_DATA / "dem.tif", acc_threshold_cells=100.0, cell_size=2.0)
    with pytest.raises(ValueError, match="overwrite"):
        lidar_rs.compute_twi(TEST_DATA / "dem.tif", out_file=TEST_DATA / "dem.tif")
    with pytest.raises(ValueError, match="overwrite"):
        lidar_rs.compute_hand(
            TEST_DATA / "dem.tif", out_file=TEST_DATA / "dem.tif", acc_threshold_cells=500.0
        )

    # Misaligned spatial-adapter auxiliary grids are rejected (both georeferenced and plain NumPy DEMs)
    shifted_aff = _FakeAffine(2.0, 0.0, 500_100.0, 0.0, -2.0, 4_000_042.0)
    shifted_acc = _FakeDataArray(
        custom_acc, rio=_FakeRioAccessor(shifted_aff, _FakeCRS("EPSG:32615"), -9999.0)
    )
    with pytest.raises(ValueError, match="geotransform"):
        lidar_rs.compute_twi(da_rio, flow_acc=shifted_acc)
    with pytest.raises(ValueError, match="geotransform"):
        lidar_rs.compute_twi(dem, flow_acc=shifted_acc)
    shifted_mask = _FakeDataArray(
        stream_mask.astype(np.float64),
        rio=_FakeRioAccessor(shifted_aff, _FakeCRS("EPSG:32615"), 0.0),
    )
    with pytest.raises(ValueError, match="geotransform"):
        lidar_rs.compute_hand(da_rio, stream_mask=shifted_mask)
    with pytest.raises(ValueError, match="geotransform"):
        lidar_rs.compute_hand(dem, stream_mask=shifted_mask)


def test_classify_embankment_artifacts_array_and_file():
    # Synthetic sloping valley blocked by a narrow raised road embankment at col 24..25
    h, w = 50, 50
    rr, cc = np.indices((h, w), dtype=np.float64)
    dem = 100.0 + np.abs(rr - 25.0) * 0.15 + (w - cc) * 0.3
    # Raised straight embankment perpendicular to valley flow (crest above rim, lower downstream)
    dem[:, 24:26] = np.maximum(dem[:, 24:26], 112.0)
    gt = (500_000.0, 2.0, 0.0, 4_000_100.0, 0.0, -2.0)

    sinks = lidar_rs.extract_sinks_array(dem, min_size=10, geotransform=gt, crs="EPSG:32615")
    deps = lidar_rs.delineate_depressions_array(
        sinks["sink"],
        min_size=10,
        min_depth=0.2,
        interval=0.3,
        no_data=0.0,
        geotransform=gt,
        crs="EPSG:32615",
    )
    candidates = lidar_rs.classify_embankment_artifacts(dem, deps, max_barrier_width_m=40.0)
    assert len(candidates) >= 1
    first = candidates[0]
    for key in (
        "dep_id",
        "level",
        "area_m2",
        "max_depth",
        "elongatedness",
        "circularity",
        "compactness",
        "barrier_width_m",
        "barrier_height_ratio",
        "rim_linearity",
        "artifact_score",
        "is_likely_artifact",
        "suggested_breach",
    ):
        assert key in first
    assert any(c["is_likely_artifact"] for c in candidates)
    flagged = next(c for c in candidates if c["is_likely_artifact"])
    assert flagged["barrier_width_m"] is not None and flagged["barrier_width_m"] > 0.0
    assert flagged["suggested_breach"] is not None
    (x1, y1), (x2, y2) = flagged["suggested_breach"]
    assert all(np.isfinite([x1, y1, x2, y2]))

    # Auto-delineate mode (no precomputed depressions) matches candidate IDs, flags, scores, and breaches
    candidates_auto = lidar_rs.classify_embankment_artifacts(
        dem,
        min_size=10,
        min_depth=0.2,
        interval=0.3,
        max_barrier_width_m=40.0,
        geotransform=gt,
        crs="EPSG:32615",
    )
    by_id_pre = {c["dep_id"]: c for c in candidates}
    by_id_auto = {c["dep_id"]: c for c in candidates_auto}
    assert set(by_id_auto) == set(by_id_pre)
    for dep_id, pre in by_id_pre.items():
        auto = by_id_auto[dep_id]
        assert auto["is_likely_artifact"] == pre["is_likely_artifact"]
        assert auto["artifact_score"] == pytest.approx(pre["artifact_score"], rel=1e-9)
        assert auto["barrier_width_m"] == pytest.approx(pre["barrier_width_m"], rel=1e-9)
        if pre["suggested_breach"] is None:
            assert auto["suggested_breach"] is None
        else:
            assert auto["suggested_breach"] is not None
            np.testing.assert_allclose(auto["suggested_breach"], pre["suggested_breach"])

    # Misaligned precomputed depressions rejected when DEM carries a different geotransform
    shifted_deps = dict(deps, geotransform=(500_200.0, 2.0, 0.0, 4_000_100.0, 0.0, -2.0))
    with pytest.raises(ValueError, match="geotransform"):
        lidar_rs.classify_embankment_artifacts(dem, shifted_deps, geotransform=gt, crs="EPSG:32615")

    # File-based call on real DEM fixture
    file_cands = lidar_rs.classify_embankment_artifacts(
        TEST_DATA / "dem.tif", min_size=1000, min_depth=0.3, interval=0.3
    )
    assert isinstance(file_cands, list) and len(file_cands) >= 1

    # Validation errors on file path with incompatible overrides
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.classify_embankment_artifacts(TEST_DATA / "dem.tif", depressions=deps)
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.classify_embankment_artifacts(TEST_DATA / "dem.tif", geotransform=gt)
    with pytest.raises(ValueError, match="file path"):
        lidar_rs.classify_embankment_artifacts(TEST_DATA / "dem.tif", cell_size=2.0)


def test_v06_python_api_parity_filters_derivatives_merge_tree_fsm_and_pointcloud(
    tmp_path: Path,
) -> None:
    """Exercise all v0.6.0 Python parity functions and options end to end."""
    h, w = 24, 24
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dem = 110.0 - 0.05 * xx + 0.02 * yy
    d1 = np.hypot(xx - 8.0, yy - 12.0)
    d2 = np.hypot(xx - 16.0, yy - 12.0)
    dem -= np.where(d1 < 5.0, 2.0 * (1.0 - (d1 / 5.0) ** 2), 0.0)
    dem -= np.where(d2 < 4.5, 1.5 * (1.0 - (d2 / 4.5) ** 2), 0.0)
    gt = (500_000.0, 2.0, 0.0, 4_500_000.0 + h * 2.0, 0.0, -2.0)

    # 1. Bilateral & feature-preserving smoothing (array + file)
    bil = lidar_rs.bilateral_filter(dem, spatial_sigma=1.5, range_sigma=0.8)
    fps = lidar_rs.feature_preserving_smoothing(
        dem, filter_size=5, norm_diff=15.0, num_iter=2, geotransform=gt, crs="EPSG:32614"
    )
    assert bil.shape == dem.shape and fps.shape == dem.shape
    bil_file = tmp_path / "bil.tif"
    fps_file = tmp_path / "fps.tif"
    assert lidar_rs.bilateral_filter(TEST_DATA / "dem.tif", out_file=bil_file) == str(bil_file)
    assert lidar_rs.feature_preserving_smoothing(
        TEST_DATA / "dem.tif", filter_size=5, num_iter=1, out_file=fps_file, compress=True
    ) == str(fps_file)
    assert bil_file.exists() and fps_file.exists()

    # 2. D-inf flow direction / accumulation, MFD flow accumulation (array + file)
    breached = lidar_rs.breach_depressions(
        dem, mode="selective", max_depth=3.0, fill_remaining=True, geotransform=gt, crs="EPSG:32614"
    )
    dinf_dir = lidar_rs.dinf_flow_direction(breached, geotransform=gt, crs="EPSG:32614")
    dinf_acc = lidar_rs.dinf_flow_accumulation(breached, geotransform=gt, crs="EPSG:32614")
    mfd_acc = lidar_rs.mfd_flow_accumulation(
        breached, exponent=1.1, geotransform=gt, crs="EPSG:32614"
    )
    assert dinf_dir.shape == dem.shape
    assert np.nanmax(dinf_acc) >= 1.0
    assert np.nanmax(mfd_acc) >= 1.0

    # 3. Exact Merge-Tree delineation + merge_tree rating curves
    sinks = lidar_rs.extract_sinks_array(dem, min_size=4, geotransform=gt, crs="EPSG:32614")
    mt = lidar_rs.delineate_depressions_merge_tree(
        sinks["sink"],
        dem_arr=sinks["dem_refined"],
        min_size=4,
        min_depth=0.1,
        dem_is_refined=True,
        max_rating_curve_points=12,
        geotransform=gt,
        crs="EPSG:32614",
    )
    assert len(mt["nodes"]) >= 1
    assert len(mt["rating_curves"]) >= 2
    rc_mt = lidar_rs.compute_rating_curves(
        sinks["sink"],
        min_size=4,
        min_depth=0.1,
        geotransform=gt,
        crs="EPSG:32614",
        delineation_method="merge_tree",
    )
    assert len(rc_mt) >= 1

    # 4. FSM simulate_inundation_array with merge_tree, Green-Ampt, hazard, recession, hydrograph attenuation
    sim = lidar_rs.simulate_inundation_array(
        dem,
        kernel_size=1,
        min_size=4,
        min_depth=0.1,
        interval=0.2,
        rain_intensity_cm_h=8.0,
        rain_duration_h=6.0,
        time_step_h=2.0,
        routing="fsm",
        delineation_method="merge_tree",
        green_ampt=(0.5, 10.0, 0.3),
        compute_hazard=True,
        recession_et_cm_h=0.5,
        recession_seepage_cm_h=0.5,
        recession_max_h=24.0,
        geotransform=gt,
        crs="EPSG:32614",
    )
    assert sim["fsm_hydrograph"] is not None and len(sim["fsm_hydrograph"]) >= 2
    assert "fsm_hazard_max_depth" in sim and "fsm_recession_drain_time_h" in sim
    att = lidar_rs.attenuate_fsm_hydrograph(
        sim["fsm_hydrograph"], mean_travel_time_h=1.0, dt_h=0.5, num_reservoirs=2
    )
    assert len(att) >= 2

    # 5. Raster output options (compress + cog) + tiled merge-tree delineation + pipeline merge_tree
    reproj_cog = tmp_path / "reproj_cog.tif"
    lidar_rs.reproject_image(
        TEST_DATA / "dem.tif", reproj_cog, dst_crs="EPSG:4326", compress=True, cog=True
    )
    resamp_cog = tmp_path / "resamp_cog.tif"
    lidar_rs.resample(
        TEST_DATA / "dem.tif", resamp_cog, resolution=4.0, compress=True, cog=True
    )
    assert reproj_cog.exists() and resamp_cog.exists()

    tiled_out = tmp_path / "tiled_mt"
    td = lidar_rs.delineate_depressions_tiled(
        TEST_DATA / "dem.tif",
        tiled_out,
        tile_size=256,
        overlap=32,
        min_size=1000,
        delineation_method="merge_tree",
        write_rasters=True,
        write_gpkg=True,
        compress=True,
    )
    assert len(td["depressions"]) >= 1
    assert (tiled_out / "depression_id.tif").exists()

    # 6. Point-cloud pit-free CHM, tiled DTM/CHM, SOR/COG flags, and URL validation on remote COPC
    las_path = tmp_path / "pts.las"
    px, py_, pz, pcls = [], [], [], []
    for r in range(16):
        for c in range(16):
            px.append(500_000.0 + c)
            py_.append(4_500_000.0 + r)
            pz.append(100.0)
            pcls.append(2)
            if 4 <= r < 12 and 4 <= c < 12:
                px.append(500_000.0 + c + 0.2)
                py_.append(4_500_000.0 + r + 0.2)
                pz.append(110.0)
                pcls.append(5)
    lidar_rs.write_lidar(
        las_path,
        np.array(px),
        np.array(py_),
        np.array(pz),
        classification=np.array(pcls, dtype=np.uint8),
        crs="EPSG:32614",
    )
    chm_pf = tmp_path / "chm_pf.tif"
    dtm_tiled = tmp_path / "dtm_tiled.tif"
    chm_tiled = tmp_path / "chm_tiled.tif"
    cw, ch = lidar_rs.lidar_to_chm_pit_free(
        las_path,
        chm_pf,
        resolution=2.0,
        ground_method="preclassified",
        height_thresholds=[0.0, 2.0, 5.0],
        compress=True,
        cog=True,
    )
    dw, dh = lidar_rs.lidar_to_dtm_tiled(
        las_path,
        dtm_tiled,
        tile_size=8.0,
        overlap=2.0,
        resolution=2.0,
        ground_method="preclassified",
        sor_k=4,
        sor_sigma=2.5,
        compress=True,
        cog=True,
    )
    tcw, tch = lidar_rs.lidar_to_chm_tiled(
        las_path,
        chm_tiled,
        tile_size=8.0,
        overlap=2.0,
        resolution=2.0,
        ground_method="preclassified",
        compress=True,
    )
    assert cw > 0 and ch > 0 and chm_pf.exists()
    assert dw > 0 and dh > 0 and dtm_tiled.exists()
    assert tcw == dw and tch == dh and chm_tiled.exists()

    with pytest.raises(ValueError, match="https://"):
        lidar_rs.stream_remote_pointcloud_aoi("http://127.0.0.1/test.copc.laz")
