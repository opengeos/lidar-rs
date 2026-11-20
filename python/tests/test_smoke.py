"""Smoke test for `lidar-rs` scaffolding."""

import lidar_rs


def test_version():
    assert lidar_rs.__version__
