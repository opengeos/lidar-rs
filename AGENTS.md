# lidar-rs contributor instructions

## Scope and architecture

`lidar-rs` is a pure-Rust port of [`opengeos/lidar`](https://github.com/opengeos/lidar), implementing DEM filtering, sink extraction, the Wu et al. (2019) level-set algorithm for nested surface depression and mount delineation, and geospatial raster/vector/point-cloud utilities using [`terrio`](https://github.com/opengeos/terrio) and [`richdem`](https://github.com/opengeos/richdem-rs).

- Put all core algorithms in `crates/lidar` (published to crates.io as `lidar`); keep `crates/lidar-cli` and `python/` thin wrappers.
- Keep default builds 100% pure Rust with zero C/C++ or GDAL dependencies.
- Keep `python/LICENSE` and `python/NOTICE` synchronized with root `LICENSE` and `NOTICE` so wheel and sdist builds pass license verification.
- Preserve sample test fixtures under `test_data/` verbatim (`dem.tif`, `dsm.tif`, `sink.tif` from `opengeos/lidar`).
- `python/` is excluded from the root Cargo workspace so pure-Rust builds never require `libpython`. Minimum supported Python is **3.11** (`requires-python = ">=3.11"`, `abi3-py311`).

## Validation

Run the full gate matching `.github/workflows/ci.yml` before pushing:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p lidar --no-default-features --lib
cargo test --workspace --all-features
```

When touching `python/`, rebuild and test the Python bindings inside a virtualenv:

```sh
cd python && .venv/bin/maturin develop --release && .venv/bin/pytest -v
```

## GitHub workflow

- Submit feature and bugfix changes through feature branches and pull requests rather than pushing directly to `main`.
