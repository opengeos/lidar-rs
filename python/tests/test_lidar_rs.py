"""Pytest suite for `lidar-rs` Python bindings."""

from pathlib import Path

import numpy as np
import pytest

import lidar_rs

TEST_DATA = Path(__file__).resolve().parents[2] / "test_data"


def test_array_filters():
    arr = np.full((7, 7), 10.0, dtype=np.float64)
    arr[3, 3] = 100.0
    med = lidar_rs.MedianFilter(arr, kernel_size=3)
    assert med.shape == (7, 7)
    assert med[3, 3] == pytest.approx(10.0)

    mean = lidar_rs.MeanFilter(arr, kernel_size=3)
    assert mean[3, 3] == pytest.approx(20.0)

    gau = lidar_rs.GaussianFilter(arr, sigma=1.0)
    assert gau.shape == (7, 7)
    assert gau[3, 3] > 10.0

    flipped = lidar_rs.FlipDEM(np.array([[10.0, 25.0]], dtype=np.float64), delta=100.0)
    assert flipped[0, 0] == pytest.approx(115.0)
    assert flipped[0, 1] == pytest.approx(100.0)

    with pytest.raises(ValueError, match="out_file"):
        lidar_rs.MeanFilter(arr, out_file="should_fail.tif")


def test_file_filters_and_pipeline(tmp_path: Path):
    dem = TEST_DATA / "dem.tif"
    med_path = tmp_path / "median.tif"
    out = lidar_rs.MedianFilter(dem, kernel_size=3, out_file=med_path)
    assert Path(out).exists()

    with pytest.raises(ValueError, match="keep_files"):
        lidar_rs.ExtractSinks(
            med_path, min_size=1000, out_dir=tmp_path, keep_files=False
        )

    sink_path = lidar_rs.ExtractSinks(med_path, min_size=1000, out_dir=tmp_path)
    assert Path(sink_path).exists()
    assert (tmp_path / "median.gpkg").exists()

    dep_id, dep_level = lidar_rs.DelineateDepressions(
        sink_path,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        out_dir=tmp_path,
        bool_level_shp=True,
    )
    assert Path(dep_id).exists()
    assert Path(dep_level).exists()
    assert (tmp_path / "depressions_info.csv").exists()
    level_shps = list((tmp_path / "shp-level").glob("Single_level_*.shp"))
    assert len(level_shps) >= 1


def test_delineate_mounts(tmp_path: Path):
    dsm = TEST_DATA / "dsm.tif"
    mount_id, mount_level = lidar_rs.DelineateMounts(
        dsm,
        min_size=1000,
        min_height=0.3,
        interval=0.3,
        out_dir=tmp_path,
        bool_shp=False,
    )
    assert Path(mount_id).exists()
    assert Path(mount_level).exists()


def test_raster_vector_helpers(tmp_path: Path):
    dem = TEST_DATA / "dem.tif"
    resampled = tmp_path / "res2.tif"
    lidar_rs.resample(dem, resampled, resolution=2.0)
    assert resampled.exists()

    cog = lidar_rs.image_to_cog(resampled)
    assert Path(cog).exists()

    pts_csv = tmp_path / "pts.csv"
    pts_csv.write_text("name,latitude,longitude\nA,35.96,-83.92\nB,39.74,-104.99\n")
    pts_shp = tmp_path / "pts.shp"
    n = lidar_rs.csv_to_shp(pts_csv, pts_shp)
    assert n == 2
    assert pts_shp.exists()


def test_inundation_hydrology_pipeline(tmp_path: Path):
    dem = TEST_DATA / "dem.tif"
    med_path = tmp_path / "median.tif"
    lidar_rs.MedianFilter(dem, kernel_size=3, out_file=med_path)

    sinks = lidar_rs.extract_sinks(med_path, min_size=1000, out_dir=tmp_path)
    refined_dem = sinks["refined_dem_path"]
    filled_dem = sinks["filled_dem_path"]
    region_tif = sinks["region_path"]
    sink_tif = sinks["sink_path"]

    cats = lidar_rs.delineate_catchments(
        refined_dem, region_tif, tmp_path / "catchments"
    )
    assert len(cats["records"]) >= 1
    assert Path(cats["watershed_path"]).exists()
    assert Path(cats["vector_path"]).exists()

    fp = lidar_rs.delineate_flow_paths(
        filled_dem,
        region_tif,
        tmp_path / "flowpath.shp",
        rain_intensity_cm_h=5.0,
        refined_dem=refined_dem,
    )
    assert Path(fp["flowpath_vector_path"]).exists()

    deps = lidar_rs.delineate_depressions(
        sink_tif,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        out_dir=tmp_path,
        bool_level_shp=True,
    )
    assert deps["max_level"] >= 2

    hir_tif = tmp_path / "hir" / "catchment_hir.tif"
    hir = lidar_rs.delineate_catchment_hierarchy(
        refined_dem, tmp_path / "shp-level", hir_tif
    )
    assert Path(hir["catchment_raster_path"]).exists()
    assert len(hir["records"]) >= 1

    sim_dir = tmp_path / "simulation"
    sim = lidar_rs.simulate_inundation(
        sink_tif,
        hir_tif,
        sim_dir,
        min_size=1000,
        min_depth=0.2,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=10.0,
        time_step_h=2.0,
    )
    assert len(sim["steps"]) == 5
    assert sim["steps"][-1]["inundated_cells"] > 0

    gif_path = tmp_path / "inundation.gif"
    anim = lidar_rs.animate_inundation(
        refined_dem,
        sim_dir,
        gif_path,
        frame_delay_ms=150,
        ping_pong=True,
        max_dimension=200,
        time_step_h=2.0,
        rain_intensity_cm_h=5.0,
    )
    assert anim["frame_count"] == 9
    assert Path(anim["gif_path"]).read_bytes()[:6] == b"GIF89a"


def test_numpy_in_memory_and_pipeline(tmp_path: Path):
    dem_path = TEST_DATA / "dem.tif"

    # Load DEM via MedianFilter into a 2D NumPy array
    dem_arr = lidar_rs.MedianFilter(dem_path, kernel_size=3)
    assert isinstance(dem_arr, np.ndarray)
    assert dem_arr.shape == (400, 400)

    # In-memory sink extraction + nested depression delineation (zero temp files)
    sinks_mem = lidar_rs.extract_sinks_array(dem_arr, min_size=1000)
    assert sinks_mem["sink"].shape == (400, 400)
    assert len(sinks_mem["depressions"]) >= 1

    deps_mem = lidar_rs.delineate_depressions_array(
        sinks_mem["sink"], min_size=1000, min_depth=0.3, interval=0.3
    )
    assert deps_mem["max_level"] >= 2
    first_dep = deps_mem["depressions"][0]
    for key in (
        "parent_id",
        "circularity",
        "depth_diam_ratio",
        "hypsometric_integral",
        "rim_slope_deg",
        "artifact_flag",
    ):
        assert key in first_dep

    # SCS Curve Number and Type II hyetograph
    cn = np.array([[70.0, 90.0]], dtype=np.float64)
    rc = lidar_rs.scs_curve_number_runoff(cn, 100.0)
    assert 0.2 < rc[0, 0] < rc[0, 1] < 1.0
    hyeto = lidar_rs.scs_type2_hyetograph(100.0, duration_h=24.0, time_step_h=1.0)
    assert len(hyeto) == 24
    assert sum(dt * i * 10.0 for dt, i in hyeto) == pytest.approx(100.0, rel=1e-3)

    # In-memory full inundation pipeline on NumPy array
    sim_mem = lidar_rs.simulate_inundation_array(
        dem_arr,
        kernel_size=1,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=20.0,
        time_step_h=2.0,
    )
    assert sim_mem["fill_time"].shape == (400, 400)
    assert len(sim_mem["steps"]) == 10
    assert len(sim_mem["rating_curves"]) >= 1
    assert sim_mem["graph_node_count"] >= 1

    # Selective breaching on array & file
    breached_arr = lidar_rs.breach_depressions(
        dem_arr, mode="selective", max_depth=0.5, max_length=15
    )
    assert isinstance(breached_arr, np.ndarray)
    assert breached_arr.shape == (400, 400)

    # All-in-one file pipeline + PNG frame export + tiled delineation
    pipe_dir = tmp_path / "pipe"
    pipe = lidar_rs.run_pipeline(
        dem_path,
        pipe_dir,
        kernel_size=3,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=10.0,
        time_step_h=2.0,
    )
    assert pipe["num_depressions"] >= 2
    assert Path(pipe["depressions_gpkg"]).exists()
    assert Path(pipe["spill_paths_gpkg"]).exists()
    assert Path(pipe["rating_curves_csv"]).exists()
    assert Path(pipe["connectivity_graph_json"]).exists()
    assert Path(pipe["connectivity_graphml"]).exists()

    pngs = lidar_rs.export_inundation_frames_png(
        dem_path,
        pipe_dir / "fill_time.tif",
        tmp_path / "frames",
        num_steps=3,
        time_step_h=2.0,
        rain_intensity_cm_h=5.0,
        max_dimension=200,
        show_scale_bar=True,
        show_legend=True,
    )
    assert len(pngs) == 3
    assert Path(pngs[0]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    tiled = lidar_rs.delineate_depressions_tiled(
        dem_path,
        tmp_path / "tiled",
        tile_size=200,
        overlap=50,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        use_f32=True,
    )
    assert tiled["num_tiles"] == 4
    assert len(tiled["depressions"]) >= 1

    # Verify mode="complete" breaches a deep (>1.5 m) wall to a lower outlet without hitting selective caps
    rr, cc = np.indices((15, 15))
    deep = np.full((15, 15), 20.0, dtype=np.float64)
    deep[:, 0] = 5.0  # lower outlet along west boundary
    dist = np.hypot(rr - 7, cc - 7)
    deep[dist <= 3.5] = 9.0 + dist[dist <= 3.5]
    breached_selective = lidar_rs.breach_depressions(deep, mode="selective")
    breached_complete = lidar_rs.breach_depressions(deep, mode="complete")
    # Selective default (max_depth=1.5) leaves the 10 m wall intact; complete carves through it.
    assert breached_selective[:, 1:4].min() == pytest.approx(20.0)
    assert breached_complete[:, 1:4].min() < 12.0


def test_fill_spill_merge_routing(tmp_path: Path):
    dem_path = TEST_DATA / "dem.tif"
    dem_arr = lidar_rs.MedianFilter(dem_path, kernel_size=3)

    # In-memory FSM routing on NumPy array
    sim_fsm = lidar_rs.simulate_inundation_array(
        dem_arr,
        kernel_size=1,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=50.0,
        time_step_h=2.0,
        routing="fsm",
    )
    mb = sim_fsm["fsm_mass_balance"]
    assert mb is not None
    assert mb["total_runoff_m3"] > 0.0
    assert abs(mb["mass_balance_error_m3"]) < 1e-6 * mb["total_runoff_m3"]
    assert sim_fsm["fsm_basins"] is not None
    assert len(sim_fsm["fsm_basins"]) == len(sim_fsm["depressions"])
    assert any(b["spilled_volume_m3"] > 0.0 for b in sim_fsm["fsm_basins"])
    assert any(b["merge_time_h"] is not None for b in sim_fsm["fsm_basins"])

    # File-based pipeline with FSM routing
    pipe_dir = tmp_path / "pipe_fsm"
    pipe = lidar_rs.run_pipeline(
        dem_path,
        pipe_dir,
        kernel_size=3,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=50.0,
        time_step_h=2.0,
        routing="fsm",
    )
    assert pipe["routing"] == "fsm"
    assert Path(pipe["fsm_basins_csv"]).exists()
    assert (
        abs(pipe["fsm_mass_balance"]["mass_balance_error_m3"])
        < 1e-6 * pipe["fsm_mass_balance"]["total_runoff_m3"]
    )

    with pytest.raises(ValueError, match="unknown routing mode"):
        lidar_rs.simulate_inundation_array(dem_arr, routing="invalid_mode")


def test_synthesize_culverts_and_federal_validation(tmp_path: Path):
    import json

    streams_gj = tmp_path / "streams.geojson"
    roads_gj = tmp_path / "roads.geojson"
    culverts_gpkg = tmp_path / "culverts.gpkg"

    streams_gj.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "test_creek"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-99.205, 47.100], [-99.195, 47.100]],
                        },
                    }
                ],
            }
        )
    )
    roads_gj.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "county_rd"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-99.200, 47.095], [-99.200, 47.105]],
                        },
                    }
                ],
            }
        )
    )

    n = lidar_rs.synthesize_culverts(
        streams_gj, roads_gj, culverts_gpkg, half_length_m=25.0, snap_distance_m=15.0
    )
    assert n == 1
    assert culverts_gpkg.exists()

    # Validate region argument checks without network calls
    with pytest.raises(ValueError, match="exactly one"):
        lidar_rs.query_3dep_products()
    with pytest.raises(ValueError, match="invalid HUC"):
        lidar_rs.query_3dep_products(huc="12345")
    with pytest.raises(ValueError, match="invalid longitude"):
        lidar_rs.download_federal_data(
            tmp_path / "bad", bbox=(-99.0, 47.0, -99.5, 47.5)
        )
    with pytest.raises(ValueError, match="unknown federal dataset"):
        lidar_rs.download_federal_data(
            tmp_path / "bad_ds", bbox=(-99.3, 47.0, -99.2, 47.1), datasets="nonexistent"
        )


def test_python_cli_and_mcp_server():
    import contextlib
    import json
    import queue
    import subprocess
    import sys
    import threading

    dem = str(TEST_DATA / "dem.tif")
    proc = subprocess.Popen(
        [sys.executable, "-m", "lidar_rs", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    out_q: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out_q.put(line)
        out_q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    replies = {}
    try:
        msgs = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "info", "arguments": {"args": [dem, "--json"]}},
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "preview",
                    "arguments": {"args": [dem, "--max-dim", "64"]},
                },
            },
        ]
        for m in msgs:
            proc.stdin.write(json.dumps(m) + "\n")
        proc.stdin.flush()

        for _ in range(4):
            line = out_q.get(timeout=30.0)
            assert line is not None, "unexpected EOF from lidar mcp"
            obj = json.loads(line)
            replies[obj["id"]] = obj
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        try:
            exit_code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise
        reader_thread.join(timeout=5)
    assert exit_code == 0

    assert replies[1]["result"]["serverInfo"]["name"] == "lidar-rs"
    tool_names = [t["name"] for t in replies[2]["result"]["tools"]]
    assert (
        "download" in tool_names
        and "pipeline" in tool_names
        and "preview" in tool_names
    )
    assert replies[3]["result"]["isError"] is False
    assert isinstance(json.loads(replies[3]["result"]["content"][0]["text"]), dict)
    assert replies[4]["result"]["isError"] is False
    imgs = [c for c in replies[4]["result"]["content"] if c["type"] == "image"]
    assert len(imgs) == 1 and imgs[0]["mimeType"] == "image/png"
    assert imgs[0]["data"].startswith("iVBORw0KGgo")


def test_file_filters_return_modes_and_non_contiguous(tmp_path: Path):
    dem = TEST_DATA / "dem.tif"
    mean_out = tmp_path / "mean.tif"
    gauss_out = tmp_path / "gauss.tif"
    flip_out = tmp_path / "flip.tif"

    assert lidar_rs.MeanFilter(dem, kernel_size=3, out_file=mean_out) == str(mean_out)
    assert lidar_rs.GaussianFilter(dem, sigma=1.0, out_file=gauss_out) == str(gauss_out)
    assert lidar_rs.FlipDEM(dem, delta=50.0, out_file=flip_out) == str(flip_out)
    assert mean_out.exists() and gauss_out.exists() and flip_out.exists()

    # Low-level file wrapper returns None when out_file is set (no wasted NumPy allocation)
    assert (
        lidar_rs._lidar_rs.mean_filter_file(str(dem), 3, str(tmp_path / "m2.tif"), None)
        is None
    )
    assert (
        lidar_rs._lidar_rs.median_filter_file(
            str(dem), 3, str(tmp_path / "med2.tif"), None
        )
        is None
    )
    assert (
        lidar_rs._lidar_rs.gaussian_filter_file(
            str(dem), 1.0, str(tmp_path / "g2.tif"), None
        )
        is None
    )
    assert (
        lidar_rs._lidar_rs.flip_dem_file(
            str(dem), 100.0, str(tmp_path / "f2.tif"), None
        )
        is None
    )

    # File-to-array mode returns ndarray
    mean_arr = lidar_rs.MeanFilter(dem, kernel_size=3)
    assert isinstance(mean_arr, np.ndarray)
    assert mean_arr.shape == (400, 400)

    # Non-contiguous (Fortran-ordered) array passed directly to Rust exercises from_py_array2 fallback
    base = np.arange(64, dtype=np.float64).reshape((8, 8))
    f_order = np.asfortranarray(base)
    assert not f_order.flags.c_contiguous
    c_out = lidar_rs._lidar_rs.mean_filter_array(base, 3, None)
    f_out = lidar_rs._lidar_rs.mean_filter_array(f_order, 3, None)
    np.testing.assert_allclose(c_out, f_out)


def test_delineate_mounts_snake_case_and_gpkg(tmp_path: Path):
    dsm = TEST_DATA / "dsm.tif"
    res = lidar_rs.delineate_mounts(
        dsm,
        min_size=1000,
        min_height=0.3,
        interval=0.3,
        out_dir=tmp_path,
        bool_shp=True,
        export_gpkg=True,
        bool_comb_tif=True,
    )
    assert Path(res["flipped_dem_path"]).exists()
    assert Path(res["depression_id_path"]).exists()
    assert Path(res["depression_level_path"]).exists()
    assert res["gpkg_path"] is not None and Path(res["gpkg_path"]).exists()
    assert len(res["level_shapefiles"]) >= 1
    assert all(Path(p).exists() for p in res["level_shapefiles"])
    assert len(res["depressions"]) >= 2


def test_raster_and_vector_operations_extended(tmp_path: Path):
    import json
    import shutil

    dem = TEST_DATA / "dem.tif"
    sinks_dir = tmp_path / "sinks"
    sinks = lidar_rs.extract_sinks(dem, min_size=1000, out_dir=sinks_dir)
    region_tif = sinks["region_path"]
    regions_csv = sinks["csv_path"]

    # polygonize
    poly_gj = tmp_path / "regions.geojson"
    n_poly = lidar_rs.polygonize(region_tif, poly_gj, field_name="id")
    assert n_poly >= 1 and poly_gj.exists()

    # join_tables
    joined_gpkg = tmp_path / "joined.gpkg"
    n_joined = lidar_rs.join_tables(
        poly_gj, regions_csv, joined_gpkg, vec_col="id", csv_col="region_id"
    )
    assert n_joined >= 1 and joined_gpkg.exists()

    # mosaic from directory with ext without leading dot ("tif")
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    shutil.copy2(dem, tiles_dir / "tile_a.tif")
    shutil.copy2(dem, tiles_dir / "tile_b.tif")
    mosaic_out = tmp_path / "mosaic_out.tif"
    lidar_rs.mosaic(tiles_dir, mosaic_out, ext="tif")
    assert mosaic_out.exists()

    # reproject_image
    reproj_out = tmp_path / "dem_4326.tif"
    lidar_rs.reproject_image(dem, reproj_out, dst_crs="EPSG:4326", resampling="nearest")
    assert reproj_out.exists()

    # add_crs (non-destructive output copy)
    crs_out = tmp_path / "dem_with_crs.tif"
    lidar_rs.add_crs(dem, 32614, output=crs_out)
    assert crs_out.exists()

    # clip_image using polygonized sink footprint mask
    clipped_out = tmp_path / "clipped.tif"
    lidar_rs.clip_image(dem, poly_gj, clipped_out)
    assert clipped_out.exists()

    # burn_culverts using a LineString spanning the first polygon's bounding box
    gj_data = json.loads(poly_gj.read_text())
    ring = gj_data["features"][0]["geometry"]["coordinates"][0]
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    mid_y = 0.5 * (min(ys) + max(ys))
    culvert_gj = tmp_path / "culvert.geojson"
    culvert_doc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[min(xs), mid_y], [max(xs), mid_y]],
                },
            }
        ],
    }
    if "crs" in gj_data:
        culvert_doc["crs"] = gj_data["crs"]
    culvert_gj.write_text(json.dumps(culvert_doc))
    burned_out = tmp_path / "dem_burned.tif"
    carved = lidar_rs.burn_culverts(dem, culvert_gj, burned_out)
    assert carved >= 1 and burned_out.exists()


def test_breach_depressions_file_and_validation(tmp_path: Path):
    dem = TEST_DATA / "dem.tif"
    out_breach = tmp_path / "breached.tif"
    ret = lidar_rs.breach_depressions(
        dem,
        out_breach,
        mode="selective",
        max_depth=0.5,
        max_length=15,
        no_data=-9999.0,
    )
    assert ret == str(out_breach)
    assert out_breach.exists()

    with pytest.raises(ValueError, match="out_dem is not supported"):
        lidar_rs.breach_depressions(np.zeros((8, 8), dtype=np.float64), out_dem=out_breach)

    with pytest.raises(ValueError, match="out_dem is required"):
        lidar_rs.breach_depressions(dem)


def test_simulate_inundation_array_forcing_and_graph():
    dem_arr = lidar_rs.MedianFilter(TEST_DATA / "dem.tif", kernel_size=3)
    rc_arr = np.full_like(dem_arr, 0.8)
    inf_arr = np.full_like(dem_arr, 0.5)
    ant_arr = np.full_like(dem_arr, 0.02)
    hyeto = lidar_rs.scs_type2_hyetograph(120.0, duration_h=10.0, time_step_h=2.0)

    res = lidar_rs.simulate_inundation_array(
        dem_arr,
        kernel_size=1,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=10.0,
        time_step_h=2.0,
        runoff_coeff_arr=rc_arr,
        infiltration_arr=inf_arr,
        antecedent_depth_arr=ant_arr,
        hyetograph=hyeto,
    )
    graph = res["connectivity_graph"]
    assert isinstance(graph, dict)
    assert len(graph["nodes"]) == res["graph_node_count"] >= 2
    assert len(graph["edges"]) == res["graph_edge_count"] >= 1
    assert "id" in graph["nodes"][0] and "volume" in graph["nodes"][0]
    assert "from_id" in graph["edges"][0] and "edge_type" in graph["edges"][0]
    assert "<graphml" in res["connectivity_graphml"]
    assert res["inundation_curve_csv"].startswith("step,time_hours,runoff_mm,")
    assert len(res["catchment_records"]) >= 1
    assert "cat_area" in res["catchment_records"][0]


def test_pointcloud_las_roundtrip(tmp_path: Path):
    import struct

    # Write a minimal valid LAS 1.2 (Point Data Record Format 0) file with ground + canopy points
    las_path = tmp_path / "synthetic.las"
    pts: list[tuple[float, float, float, int]] = []
    for row in range(6):
        for col in range(6):
            pts.append((float(col), float(row), 10.0 + 0.1 * col, 2))  # ground (class 2)
    pts.append((2.5, 2.5, 18.0, 5))  # canopy (class 5)

    scale = 0.001
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    header = bytearray(227)
    header[0:4] = b"LASF"
    struct.pack_into("<BB", header, 24, 1, 2)  # version 1.2
    struct.pack_into("<H", header, 94, 227)  # header size
    struct.pack_into("<I", header, 96, 227)  # offset to point data
    struct.pack_into("<I", header, 100, 0)  # number of VLRs
    struct.pack_into("<B", header, 104, 0)  # point data format 0
    struct.pack_into("<H", header, 105, 20)  # point record length
    struct.pack_into("<I", header, 107, len(pts))  # number of points
    struct.pack_into("<I", header, 111, len(pts))  # points by return 1
    struct.pack_into("<ddd", header, 131, scale, scale, scale)
    struct.pack_into("<ddd", header, 155, 0.0, 0.0, 0.0)
    struct.pack_into(
        "<dddddd", header, 179, max(xs), min(xs), max(ys), min(ys), max(zs), min(zs)
    )

    records = bytearray()
    for x, y, z, cls in pts:
        records += struct.pack(
            "<iiiHBBbBH",
            round(x / scale),
            round(y / scale),
            round(z / scale),
            100,  # intensity
            0b00001001,  # return 1 of 1
            cls,
            0,  # scan angle
            0,  # user data
            0,  # point source id
        )
    las_path.write_bytes(bytes(header) + bytes(records))

    laz_path = tmp_path / "synthetic.laz"
    count = lidar_rs.convert_lidar(las_path, laz_path)
    assert count == len(pts) and laz_path.exists()

    dsm_path = tmp_path / "dsm.tif"
    cols, rows = lidar_rs.lidar_to_dsm(laz_path, dsm_path, resolution=1.0)
    assert cols >= 5 and rows >= 5 and dsm_path.exists()

    dtm_path = tmp_path / "dtm.tif"
    d_cols, d_rows = lidar_rs.lidar_to_dtm(
        laz_path, dtm_path, resolution=1.0, ground_method="preclassified"
    )
    assert d_cols >= 5 and d_rows >= 5 and dtm_path.exists()

    chm_path = tmp_path / "chm.tif"
    c_cols, c_rows = lidar_rs.lidar_to_chm(
        laz_path, chm_path, resolution=1.0, ground_method="preclassified"
    )
    assert c_cols >= 5 and c_rows >= 5 and chm_path.exists()
