"""Pure-Rust port of `opengeos/lidar` for Python."""

from . import _lidar_rs

__version__: str = _lidar_rs.__version__

__all__ = ["__version__"]
