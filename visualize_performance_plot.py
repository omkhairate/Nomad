#!/usr/bin/env python3
"""Plot per-frame performance metrics with optional active node counts.

This script reads the ``perf.csv`` log produced when the renderer is run
with ``MPT_RUNS_PATH`` set. The CSV must contain a ``frame`` column and
may contain additional metrics such as ``fps`` or ``rays_per_second``.
If a directory or JSON file of acceleration-structure dumps is supplied,
the number of loaded BLAS nodes per frame is computed and plotted on a
secondary axis for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def _load_perf_csv(
    path: Path,
) -> Tuple[List[int], Dict[str, List[float]], List[int] | None, List[int] | None]:
    """Return frame numbers, metric columns, and node counts from ``path``."""
    frames: List[int] = []
    metrics: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            for key, value in row.items():
                if key == "frame":
                    continue
                metrics.setdefault(key, []).append(float(value))
    active = metrics.pop("active_nodes", None)
    offloaded = metrics.pop("offloaded_nodes", None)
    return frames, metrics, active, offloaded


# The following helpers are adapted from ``visualize_active_nodes_plot.py``
def _nodes_from_dump(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = data.get("blas", [])
    prims = data.get("primitives", [])
    status: List[Dict[str, Any]] = [None] * len(nodes)

    def compute(idx: int) -> bool:
        node = nodes[idx]
        count = node.get("count", 0)
        if count > 0:  # Leaf
            start = node.get("leftFirst", 0)
            end = start + count
            active = any(p.get("active", True) for p in prims[start:end])
        else:  # Internal
            left = node.get("leftFirst", 0)
            right = -count
            l_active = compute(left)
            r_active = compute(right)
            active = l_active or r_active
        status[idx] = {"loaded": active}
        return active

    if nodes:
        compute(0)
    return status


def _process_frame(data: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
    if "tlas" in data and "blas" in data:
        return {"frame": frame_index, "nodes": _nodes_from_dump(data)}
    if "nodes" in data:
        data.setdefault("frame", frame_index)
        return data
    raise ValueError("Unsupported frame format")


def _load_frames(path: Path) -> List[Dict[str, Any]]:
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


def _count_active_nodes(frames: List[Dict[str, Any]]) -> List[int]:
    return [sum(1 for n in f.get("nodes", []) if n.get("loaded", True)) for f in frames]


def _create_figure(
    frames: List[int],
    metrics: Dict[str, List[float]],
    active_nodes: List[int] | None,
    offloaded_nodes: List[int] | None,
) -> go.Figure:
    if active_nodes is not None or offloaded_nodes is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    for name, values in metrics.items():
        if active_nodes is not None or offloaded_nodes is not None:
            fig.add_trace(
                go.Scatter(x=frames, y=values, mode="lines+markers", name=name),
                secondary_y=False,
            )
        else:
            fig.add_trace(go.Scatter(x=frames, y=values, mode="lines+markers", name=name))

    if active_nodes is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(active_nodes))),
                y=active_nodes,
                mode="lines+markers",
                name="active_nodes",
            ),
            secondary_y=True,
        )
    if offloaded_nodes is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(offloaded_nodes))),
                y=offloaded_nodes,
                mode="lines+markers",
                name="offloaded_nodes",
            ),
            secondary_y=True,
        )
    if active_nodes is not None or offloaded_nodes is not None:
        fig.update_yaxes(title_text="Performance", secondary_y=False)
        fig.update_yaxes(title_text="Nodes", secondary_y=True)
    else:
        fig.update_yaxes(title_text="Value")

    fig.update_layout(title="Per-frame performance metrics", xaxis_title="Frame")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot performance metrics over frames"
    )
    parser.add_argument(
        "csv", type=Path, help="CSV file from MPT_RUNS_PATH/perf.csv"
    )
    parser.add_argument(
        "--as-path",
        type=Path,
        default=None,
        help=(
            "Directory or JSON file containing frame dumps for active node counts "
            "(e.g., MPT_RUNS_PATH/as)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not automatically open the browser"
    )
    args = parser.parse_args()

    frames, metrics, csv_active, csv_offloaded = _load_perf_csv(args.csv)
    active = csv_active
    offloaded = csv_offloaded
    if active is None and args.as_path is not None:
        active = _count_active_nodes(_load_frames(args.as_path))
    fig = _create_figure(frames, metrics, active, offloaded)
    fig.write_html(args.output, auto_open=not args.no_open)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
