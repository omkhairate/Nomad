#!/usr/bin/env python3
"""Plot GPU memory usage per frame.

This script reads the ``gpu_mem.csv`` log produced when the renderer is
run with ``MPT_RUNS_PATH`` set.  The CSV should contain ``frame`` and
``gpu_memory_mb`` columns.  An interactive Plotly plot is written to an
HTML file for inspection.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import plotly.graph_objects as go
import plotly.io as pio

# Prefer opening the result in a browser for interactive viewing
pio.renderers.default = "browser"


def _load_csv(path: Path) -> Tuple[List[int], List[float]]:
    frames: List[int] = []
    mem: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            mem.append(float(row["gpu_memory_mb"]))
    return frames, mem


def _create_figure(frames: List[int], mem: List[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=mem, mode="lines+markers"))
    fig.update_layout(
        title="GPU memory usage per frame",
        xaxis_title="Frame",
        yaxis_title="Memory (MB)",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GPU memory usage over frames"
    )
    parser.add_argument(
        "path", type=Path, help="CSV file from MPT_RUNS_PATH/gpu_mem.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("gpu_memory.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not automatically open the browser"
    )
    args = parser.parse_args()

    frames, mem = _load_csv(args.path)
    fig = _create_figure(frames, mem)
    fig.write_html(args.output, auto_open=not args.no_open)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
