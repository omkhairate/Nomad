#!/usr/bin/env python3
"""Plot per-frame performance metrics, one plot per metric.

This script reads the ``perf.csv`` log produced when the renderer is run
with ``MPT_RUNS_PATH`` set. The CSV must contain a ``frame`` column and
may contain additional metrics such as ``fps`` or ``rays_per_second``.
If a directory or JSON file of acceleration-structure dumps is supplied,
the number of loaded BLAS nodes per frame is computed and plotted as the
``active_nodes`` metric. Each metric is written to its own interactive
Plotly HTML file for easy comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def _load_perf_csv(path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    """Return frame numbers and metric columns from ``path``."""
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
    return frames, metrics


def _load_gpu_mem_csv(path: Path) -> Tuple[List[int], List[float]]:
    """Return frame numbers and GPU memory usage from ``path``."""
    frames: List[int] = []
    mem: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            mem.append(float(row["gpu_memory_mb"]))
    return frames, mem


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


def _create_metric_figure(
    frames: List[int], values: List[float], name: str
) -> go.Figure:
    """Return a Plotly line chart for a single metric."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=values, mode="lines+markers", name=name))
    fig.update_layout(
        title=f"{name} per frame",
        xaxis_title="Frame",
        yaxis_title=name,
    )
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
        "--gpu-mem-csv",
        type=Path,
        default=None,
        help="CSV file from MPT_RUNS_PATH/gpu_mem.csv to include GPU memory usage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance_plots"),
        help="Output directory for HTML files",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not automatically open the browser"
    )
    args = parser.parse_args()

    frames, metrics = _load_perf_csv(args.csv)
    if "active_nodes" not in metrics and args.as_path is not None:
        metrics["active_nodes"] = _count_active_nodes(_load_frames(args.as_path))
    if args.gpu_mem_csv is not None:
        mem_frames, mem = _load_gpu_mem_csv(args.gpu_mem_csv)
        if mem_frames != frames:
            print("Warning: frame mismatch between perf.csv and gpu_mem.csv")
        metrics["gpu_memory_mb"] = mem

    args.output.mkdir(parents=True, exist_ok=True)

    print("Metrics:", ", ".join(metrics.keys()))

    for name, values in metrics.items():
        fig = _create_metric_figure(frames, values, name)
        out_file = args.output / f"{name}.html"
        fig.write_html(out_file, auto_open=not args.no_open)
        print(f"Wrote {out_file} for metric '{name}'")


if __name__ == "__main__":
    main()
