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
        try:
            proc.stdin.close()
        except Exception:
            pass
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
