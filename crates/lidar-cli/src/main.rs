//! Command-line interface for `lidar-rs`.

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "lidar",
    version,
    about = "Nested surface depression and mount delineation from DEMs in pure Rust"
)]
struct Cli {}

fn main() {
    let _ = Cli::parse();
    println!("lidar {}", lidar::VERSION);
}
