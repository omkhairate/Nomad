#!/usr/bin/env python3
"""Animate TLAS/BLAS residency and ray hits across frames.

The renderer emits a JSON dump of the acceleration structure for every
frame.  Each dump contains TLAS and BLAS nodes together with per–primitive
activity and intersection counts.  This tool converts those dumps into an
interactive Plotly visualisation showing which BLAS nodes are resident
(``green``) or offloaded (``red``).  TLAS nodes are always shown in blue and
per–node ray hit counts are visualised as coloured markers at node centres.

The script accepts either a directory containing one JSON file per frame
or a single JSON file holding a list of frame objects.

Usage::

    python visualize_bvh.py /path/to/as_dumps

A browser window will open showing an animatable 3D view of the TLAS/BLAS
along with recorded ray intersections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# Force Plotly to open a browser window regardless of environment
pio.renderers.default = "browser"


def _nodes_from_dump(data):
    """Convert BLAS dump data into a list of node dictionaries."""
    nodes = data.get("blas", [])
    prims = data.get("primitives", [])
    status = [None] * len(nodes)

    def compute(idx):
        node = nodes[idx]
        count = node.get("count", 0)
        if count > 0:  # Leaf
            start = node.get("leftFirst", 0)
            end = start + count
            active = any(p.get("active", True) for p in prims[start:end])
            hits = sum(p.get("lastIntersection", 0) for p in prims[start:end])
        else:  # Internal
            left = node.get("leftFirst", 0)
            right = -count
            l_active, l_hits = compute(left)
            r_active, r_hits = compute(right)
            active = l_active or r_active
            hits = l_hits + r_hits
        status[idx] = {
            "min": node.get("min", [0, 0, 0]),
            "max": node.get("max", [0, 0, 0]),
            "loaded": active,
            "hits": hits,
        }
        return active, hits

    if nodes:
        compute(0)
    return status


def _process_frame(data, frame_index):
    """Normalise various frame dump formats."""
    if "tlas" in data and "blas" in data:
        return {
            "frame": frame_index,
            "tlas": data.get("tlas", []),
            "nodes": _nodes_from_dump(data),
        }
    if "nodes" in data:
        data.setdefault("frame", frame_index)
        return data
    raise ValueError("Unsupported frame format")


def _load_frames(path: Path):
    """Return a list of frame dictionaries from ``path``."""
    frames = []
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


def _box_edges(node, color: str):
    """Return Plotly line traces for a single bounding box."""
    mnx, mny, mnz = node["min"]
    mxx, mxy, mxz = node["max"]
    x = [mnx, mxx, mxx, mnx, mnx, mxx, mxx, mnx]
    y = [mny, mny, mxy, mxy, mny, mny, mxy, mxy]
    z = [mnz, mnz, mnz, mnz, mxz, mxz, mxz, mxz]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    traces = []
    for i, j in edges:
        traces.append(
            go.Scatter3d(
                x=[x[i], x[j]],
                y=[y[i], y[j]],
                z=[z[i], z[j]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )
    return traces


def _hit_markers(nodes):
    """Return scatter markers visualising per-node ray hits."""
    xs, ys, zs, colors, texts = [], [], [], [], []
    for n in nodes:
        hits = n.get("hits", 0)
        if hits <= 0:
            continue
        mn = n.get("min", [0, 0, 0])
        mx = n.get("max", [0, 0, 0])
        xs.append((mn[0] + mx[0]) / 2)
        ys.append((mn[1] + mx[1]) / 2)
        zs.append((mn[2] + mx[2]) / 2)
        colors.append(hits)
        texts.append(f"hits: {hits}")
    if not xs:
        return []
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=5, color=colors, colorscale="Viridis", colorbar=dict(title="Ray hits")),
            text=texts,
            showlegend=False,
        )
    ]


def _frame_traces(frame):
    traces = []
    for node in frame.get("tlas", []):
        traces.extend(_box_edges(node, "blue"))
    for node in frame.get("nodes", []):
        color = "green" if node.get("loaded", True) else "red"
        traces.extend(_box_edges(node, color))
    traces.extend(_hit_markers(frame.get("nodes", [])))
    return traces


def _visualize(frames):
    if not frames:
        raise SystemExit("No frames were loaded")

    fig = go.Figure(
        data=_frame_traces(frames[0]),
        frames=[
            go.Frame(data=_frame_traces(f), name=str(f.get("frame", i)))
            for i, f in enumerate(frames)
        ],
    )

    fig.update_layout(
        title="Acceleration Structure Residency",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                },
            ],
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[str(f.get("frame", i))], {"frame": {"duration": 0}, "mode": "immediate"}],
                    "label": str(f.get("frame", i)),
                    "method": "animate",
                }
                for i, f in enumerate(frames)
            ]
        }],
        height=700,
    )

    fig.show()


def main():
    parser = argparse.ArgumentParser(description="Animate TLAS/BLAS residency over time")
    parser.add_argument("path", type=Path, help="Directory or JSON file containing frame dumps")
    args = parser.parse_args()

    frames = _load_frames(args.path)
    _visualize(frames)


if __name__ == "__main__":
    main()
