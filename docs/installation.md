# Installation

`lidar-rs` is 100% pure Rust with **zero GDAL, PDAL, or C/C++ system dependencies**. Choose the interface that fits your workflow.

## Standalone CLI (`lidar`)

=== "Linux & macOS"

    ```sh
    curl -LsSf https://lidar-rs.opengeos.org/install.sh | sh
    ```

=== "Windows (PowerShell)"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://lidar-rs.opengeos.org/install.ps1 | iex"
    ```

=== "Cargo"

    ```sh
    cargo install lidar-cli
    ```

Verify the installation:

```sh
lidar --version
lidar --help
```

## Python Package (`lidar-rs`)

Pre-built ABI3 wheels support **Python 3.11+** on Linux, macOS, and Windows:

```sh
pip install lidar-rs
```

Or build from source with [maturin](https://www.maturin.rs/):

```sh
git clone https://github.com/opengeos/lidar-rs.git
cd lidar-rs/python
pip install maturin numpy
maturin develop --release
```

## Rust Library (`lidar`)

Add `lidar` to your `Cargo.toml` (requires Rust 1.90+):

```toml
[dependencies]
lidar = "0.4"
```

Default features enable `parallel` (Rayon), `pointcloud` (LAS/LAZ/COPC/PLY/E57 + CSF/PMF), and `serde` (JSON graph serialization). Disable unused features for lighter binaries:

```toml
[dependencies]
lidar = { version = "0.4", default-features = false, features = ["parallel"] }
```

## WebAssembly (`lidar-js`)

No installation is needed to try the [WebAssembly Playground](playground.md). To compile the browser or Node.js WASM package locally:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd js && wasm-pack build --target web --release --out-dir pkg -- --locked
```
