"""
MuJoCo-Bench-IDO Webviz Launcher
=================================

Simple command-line entry point for starting the webviz server.

Usage:
  python webviz/run_webviz.py
  python -m webviz.server

Author: MuJoCo-Bench-IDO Webviz extension v0.4.4
"""

import os
import sys
from pathlib import Path

# ── Add project root to PYTHONPATH ──
PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also add webviz directory
WEBVIZ_DIR: str = str(Path(__file__).resolve().parent)
if WEBVIZ_DIR not in sys.path:
    sys.path.insert(0, WEBVIZ_DIR)


def main() -> None:
    """Launch the MuJoCo-Bench-IDO webviz server on port 8080.

    Uses uvicorn to run the FastAPI application defined in server.py.
    Prints the access URL on startup.
    """
    import uvicorn

    print("=" * 60)
    print("  MuJoCo-Bench-IDO Web Visualization Dashboard")
    print("  Version: v0.4.4")
    print("=" * 60)
    print()
    print(f"  Access the dashboard at: http://localhost:8080")
    print(f"  WebSocket endpoint:      ws://localhost:8080/ws/stream")
    print(f"  API tasks endpoint:      http://localhost:8080/api/tasks")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    uvicorn.run(
        "webviz.server:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
