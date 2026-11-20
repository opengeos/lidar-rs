//! PyO3 bindings for `lidar-rs`.

use pyo3::prelude::*;

#[pymodule]
fn _lidar_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", lidar::VERSION)?;
    Ok(())
}
