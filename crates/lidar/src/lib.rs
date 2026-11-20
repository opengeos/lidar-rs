//! Pure-Rust port of [`opengeos/lidar`](https://github.com/opengeos/lidar) for
//! delineating nested surface depressions and mounts in digital elevation models (DEMs).

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!VERSION.is_empty());
    }
}
