# Installation

`lidar-rs` is 100% pure Rust with no GDAL or C/C++ system dependencies.

## Standalone CLI

```sh
# Linux and macOS
curl -LsSf https://lidar-rs.opengeos.org/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://lidar-rs.opengeos.org/install.ps1 | iex"

# Or via Cargo
cargo install lidar-cli
```

## Rust Library

Add `lidar` to your `Cargo.toml`:

```toml
[dependencies]
lidar = "0.2"
```

## Python Package

```sh
pip install lidar-rs
```

Or build from source using [maturin](https://www.maturin.rs/):

```sh
cd python
maturin develop --release
```
