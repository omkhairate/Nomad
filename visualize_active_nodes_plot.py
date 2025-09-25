#!/usr/bin/env python3
"""Plot active BLAS node counts across frames.

This utility reads per-frame JSON dumps produced by the renderer and
visualises how many BLAS nodes are resident in each frame.  A simple line
chart reveals when nodes become active or are offloaded during a
sequence.

It accepts either a directory of JSON files (one per frame) or a single
JSON file containing a list of frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import plotly.graph_objects as go
import plotly.io as pio

# Prefer opening the result in a browser for interactive viewing
pio.renderers.default = "browser"


def _nodes_from_dump(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return BLAS node information with computed ``loaded`` flags."""
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
    """Normalise various frame dump formats."""
    if "tlas" in data and "blas" in data:
        return {"frame": frame_index, "nodes": _nodes_from_dump(data)}
    if "nodes" in data:
        data.setdefault("frame", frame_index)
        return data
    raise ValueError("Unsupported frame format")


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


def _count_active_nodes(frames: List[Dict[str, Any]]) -> List[int]:
    """Return the number of loaded nodes per frame."""
    return [sum(1 for n in f.get("nodes", []) if n.get("loaded", True)) for f in frames]


def _create_figure(counts: List[int]) -> go.Figure:
    """Return a Plotly line chart of ``counts`` over frames."""
    frames = list(range(len(counts)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=counts, mode="lines+markers"))
    fig.update_layout(
        title="Active BLAS nodes per frame",
        xaxis_title="Frame",
        yaxis_title="Active nodes",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot active BLAS node counts over time"
    )
    parser.add_argument(
        "path", type=Path, help="Directory or JSON file containing frame dumps"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("active_nodes.html"),
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

    counts = _count_active_nodes(frames)
    fig = _create_figure(counts)
    fig.write_html(args.output, auto_open=not args.no_open)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

