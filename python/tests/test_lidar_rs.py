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
        lidar_rs.ExtractSinks(med_path, min_size=1000, out_dir=tmp_path, keep_files=False)

    sink_path = lidar_rs.ExtractSinks(med_path, min_size=1000, out_dir=tmp_path)
    assert Path(sink_path).exists()
    assert (tmp_path / "median.gpkg").exists()

    dep_id, dep_level = lidar_rs.DelineateDepressions(
        sink_path, min_size=1000, min_depth=0.3, interval=0.3, out_dir=tmp_path, bool_level_shp=True
    )
    assert Path(dep_id).exists()
    assert Path(dep_level).exists()
    assert (tmp_path / "depressions_info.csv").exists()
    level_shps = list((tmp_path / "shp-level").glob("Single_level_*.shp"))
    assert len(level_shps) >= 1


def test_delineate_mounts(tmp_path: Path):
    dsm = TEST_DATA / "dsm.tif"
    mount_id, mount_level = lidar_rs.DelineateMounts(
        dsm, min_size=1000, min_height=0.3, interval=0.3, out_dir=tmp_path, bool_shp=False
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

    cats = lidar_rs.delineate_catchments(refined_dem, region_tif, tmp_path / "catchments")
    assert len(cats["records"]) >= 1
    assert Path(cats["watershed_path"]).exists()
    assert Path(cats["vector_path"]).exists()

    fp = lidar_rs.delineate_flow_paths(
        filled_dem, region_tif, tmp_path / "flowpath.shp", rain_intensity_cm_h=5.0, refined_dem=refined_dem
    )
    assert Path(fp["flowpath_vector_path"]).exists()

    deps = lidar_rs.delineate_depressions(
        sink_tif, min_size=1000, min_depth=0.3, interval=0.3, out_dir=tmp_path, bool_level_shp=True
    )
    assert deps["max_level"] >= 2

    hir_tif = tmp_path / "hir" / "catchment_hir.tif"
    hir = lidar_rs.delineate_catchment_hierarchy(refined_dem, tmp_path / "shp-level", hir_tif)
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
