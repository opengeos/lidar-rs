//! Standalone `lidar` CLI binary.

use std::process::ExitCode;

fn main() -> ExitCode {
    let code = lidar_cli::run(std::env::args_os().collect());
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}
