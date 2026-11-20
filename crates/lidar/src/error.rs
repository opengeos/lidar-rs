//! Error types for `lidar-rs`.

use thiserror::Error;

/// Unified error type returned by `lidar-rs` operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Error from the underlying `richdem` crate.
    #[error("richdem error: {0}")]
    RichDem(#[from] richdem::Error),

    /// Error from the underlying `terrio` geospatial I/O crate.
    #[error("terrio error: {0}")]
    Terrio(#[from] terrio::Error),

    /// Standard filesystem / I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid argument or configuration parameter.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Raster or array dimension mismatch or overflow.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Parse error (CSV, CRS, numeric values, etc.).
    #[error("parse error: {0}")]
    Parse(String),

    /// Algorithm execution failure.
    #[error("algorithm error: {0}")]
    Algorithm(String),
}

/// Convenience `Result` alias for `lidar-rs`.
pub type Result<T> = std::result::Result<T, Error>;
