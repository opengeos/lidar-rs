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


def test_file_filters_and_pipeline(tmp_path: Path):
    dem = TEST_DATA / "dem.tif"
    med_path = tmp_path / "median.tif"
    out = lidar_rs.MedianFilter(dem, kernel_size=3, out_file=med_path)
    assert Path(out).exists()

    sink_path = lidar_rs.ExtractSinks(med_path, min_size=1000, out_dir=tmp_path)
    assert Path(sink_path).exists()
    assert (tmp_path / "median.gpkg").exists()

    dep_id, dep_level = lidar_rs.DelineateDepressions(
        sink_path, min_size=1000, min_depth=0.3, interval=0.3, out_dir=tmp_path, bool_level_shp=True
    )
    assert Path(dep_id).exists()
    assert Path(dep_level).exists()
    assert (tmp_path / "depressions_info.csv").exists()


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
