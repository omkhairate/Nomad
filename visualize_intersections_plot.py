#!/usr/bin/env python3
"""Interactive visualisation of intersection counts and residency.

This script reads the same per-frame JSON dumps produced by the renderer as
``visualize_residency_html.py`` and ``visualize_intersections_html.py`` but
uses Plotly to produce an interactive heatmap.  The top subplot shows the number
of ray intersections per primitive and frame.  The bottom subplot displays the
renderer\'s residency decision (green for resident, red for offloaded).

Usage::

    python visualize_intersections_plot.py /path/to/frame_dumps

A browser window will open with the visualisation unless ``--no-open`` is
specified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Always try to open a browser for interactive viewing
pio.renderers.default = "browser"


def _process_frame(data: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
    """Normalise various frame dump formats and extract primitive data."""
    frame: Dict[str, Any] = {"frame": frame_index}
    if "primitives" in data:
        frame["primitives"] = data["primitives"]
    elif "tlas" in data and "blas" in data:
        # TLAS/BLAS info without primitive list: assume everything resident.
        frame["primitives"] = []
    elif "nodes" in data:
        frame["primitives"] = data.get("primitives", [])
    else:
        raise ValueError("Unsupported frame format")
    return frame


def _load_frames(path: Path) -> List[Dict[str, Any]]:
    """Return a list of frame dictionaries from ``path``."""
    frames: List[Dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for d in data:
                frames.append(_process_frame(d, len(frames)))
        else:
            frames.append(_process_frame(data, 0))
    else:
        for p in sorted(path.glob("*.json")):
            with p.open("r", encoding="utf-8") as f:
                frames.append(_process_frame(json.load(f), len(frames)))
    return frames


def _build_arrays(frames: List[Dict[str, Any]]) -> Tuple[List[List[int]], List[List[int]]]:
    """Return 2-D arrays for intersection counts and residency."""
    if not frames:
        return [], []
    frame_count = len(frames)
    max_prims = max(len(f.get("primitives", [])) for f in frames)
    counts = [[0 for _ in range(frame_count)] for _ in range(max_prims)]
    active = [[0 for _ in range(frame_count)] for _ in range(max_prims)]
    for f_idx, frame in enumerate(frames):
        prims = frame.get("primitives", [])
        for p_idx, prim in enumerate(prims):
            counts[p_idx][f_idx] = prim.get("lastIntersection", 0)
            active[p_idx][f_idx] = 1 if prim.get("active", True) else 0
    return counts, active


def _create_figure(
    counts: List[List[int]], active: List[List[int]], frames: List[Dict[str, Any]]
) -> go.Figure:
    """Create a Plotly figure visualising intersections and residency."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        subplot_titles=("Intersection count", "Residency"),
    )

    fig.add_trace(
        go.Heatmap(
            z=counts,
            colorscale="Viridis",
            colorbar=dict(title="Hits"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=active,
            colorscale=[[0, "#f44336"], [1, "#4caf50"]],
            colorbar=dict(title="Active"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        xaxis=dict(title="Frame"),
        xaxis2=dict(title="Frame"),
        yaxis=dict(title="Primitive"),
        yaxis2=dict(title="Primitive"),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive heatmap of intersection counts and residency"
    )
    parser.add_argument(
        "path", type=Path, help="Directory or JSON file containing frame dumps"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("intersections_plot.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the browser",
    )
    args = parser.parse_args()

    frames = _load_frames(args.path)
    if not frames:
        raise SystemExit("No frames were loaded")

    counts, active = _build_arrays(frames)
    fig = _create_figure(counts, active, frames)
    fig.write_html(args.output, auto_open=not args.no_open)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
