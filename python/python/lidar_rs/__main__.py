"""Command-line entry point for the ``lidar`` CLI and MCP server via Python."""

from __future__ import annotations

import signal
import sys


def main() -> int:
    """Runs the ``lidar`` CLI (including ``lidar mcp``) using the bundled Rust implementation."""
    from lidar_rs._lidar_rs import cli_main

    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # `lidar mcp` invokes each tool as `python -m lidar_rs <subcommand>`.
    return cli_main(["lidar", *sys.argv[1:]], [sys.executable, "-m", "lidar_rs"])


if __name__ == "__main__":
    sys.exit(main())
