#!/usr/bin/env python3
"""Generate all research plots from a single renderer run.

The renderer can log performance counters (``perf.csv``), GPU memory usage
(``gpu_mem.csv``), and per-frame acceleration-structure dumps. Previously a
collection of small scripts were required to visualise these outputs. This
utility consolidates those workflows: run it once and it will emit dedicated
Plotly HTML dashboards for every available data source.

Example::

    python visualize_metrics.py runs --output-dir figures

The script will look for ``perf.csv`` and ``gpu_mem.csv`` inside ``runs`` and
an ``as`` directory containing frame dumps. Each output is optional; missing
inputs are reported but do not abort the run.

Set ``MPT_RUNS_PATH`` before launching the renderer to capture ``as/frame_XXXX.json``
files alongside the CSV logs. Each primitive entry exposes ``active``,
``hitProbability``, and (when available) ``object`` metadata. The probability
values feed the new heatmap and per-object trend plots, which highlight how
the stochastic residency tracker cools primitives over time.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_perf_csv(path: Path) -> tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
    """Return frame numbers, non-node metrics, and node-related series."""

    frames: List[int] = []
    metrics: Dict[str, List[float]] = {}
    node_series: Dict[str, List[float]] = {}

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_value = row.get("frame")
            if frame_value is None:
                raise ValueError("perf.csv is missing a 'frame' column")
            frames.append(int(frame_value))
            for key, value in row.items():
                if key == "frame" or value in (None, ""):
                    continue
                try:
                    as_float = float(value)
                except ValueError:
                    continue
                metrics.setdefault(key, []).append(as_float)

    for name in ["active_nodes", "resident_nodes", "offloaded_nodes", "total_nodes"]:
        if name in metrics:
            node_series[name] = metrics.pop(name)

    return frames, metrics, node_series


def _load_gpu_mem_csv(path: Path) -> tuple[List[int], List[float]]:
    """Return frame numbers and GPU memory usage."""

    frames: List[int] = []
    mem: List[float] = []

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_value = row.get("frame")
            mem_value = row.get("gpu_memory_mb")
            if frame_value is None or mem_value is None:
                raise ValueError("gpu_mem.csv must contain 'frame' and 'gpu_memory_mb'")
            frames.append(int(frame_value))
            mem.append(float(mem_value))
    return frames, mem


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
    """Normalise various frame dump formats and extract primitive data."""

    if "tlas" in data and "blas" in data:
        return {
            "frame": frame_index,
            "nodes": _nodes_from_dump(data),
            "primitives": data.get("primitives", []),
        }
    if "nodes" in data:
        data.setdefault("frame", frame_index)
        data.setdefault("primitives", data.get("primitives", []))
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
        return frames

    if not path.exists():
        raise FileNotFoundError(path)

    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON dumps found in directory")

    for p in json_files:
        with p.open("r", encoding="utf-8") as f:
            frames.append(_process_frame(json.load(f), len(frames)))
    return frames


def _node_series_from_frames(frames: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Compute node residency metrics from frame dumps."""

    active_counts: List[int] = []
    total_counts: List[int] = []

    for frame in frames:
        nodes = frame.get("nodes", [])
        total = len(nodes)
        active = sum(1 for node in nodes if node.get("loaded", True))
        active_counts.append(active)
        total_counts.append(total)

    offloaded = [total - active for total, active in zip(total_counts, active_counts)]

    return {
        "active_nodes": active_counts,
        "resident_nodes": active_counts,  # Renderer treats loaded == resident
        "offloaded_nodes": offloaded,
        "total_nodes": total_counts,
    }


def _build_intersection_arrays(
    frames: List[Dict[str, Any]]
) -> tuple[List[List[int]], List[List[int]], List[List[Optional[float]]]]:
    """Return 2-D arrays for intersection counts, residency, and hit probability."""

    if not frames:
        return [], [], []

    frame_count = len(frames)
    max_prims = max(len(f.get("primitives", [])) for f in frames)

    counts = [[0 for _ in range(frame_count)] for _ in range(max_prims)]
    active = [[0 for _ in range(frame_count)] for _ in range(max_prims)]
    probabilities: List[List[Optional[float]]] = [
        [None for _ in range(frame_count)] for _ in range(max_prims)
    ]

    for f_idx, frame in enumerate(frames):
        prims = frame.get("primitives", [])
        for p_idx, prim in enumerate(prims):
            counts[p_idx][f_idx] = int(prim.get("lastIntersection", 0))
            active[p_idx][f_idx] = 1 if prim.get("active", True) else 0
            prob_value = prim.get("hitProbability")
            if prob_value is not None:
                try:
                    probabilities[p_idx][f_idx] = float(prob_value)
                except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                    probabilities[p_idx][f_idx] = None

    return counts, active, probabilities


def _has_probability_data(probabilities: List[List[Optional[float]]]) -> bool:
    """Return True if any probability samples are present."""

    return any(any(value is not None for value in row) for row in probabilities)


def _aggregate_object_probabilities(
    frames: List[Dict[str, Any]]
) -> Dict[int, List[Optional[float]]]:
    """Return average hit probability per object for each frame."""

    if not frames:
        return {}

    frame_count = len(frames)
    object_ids = set()
    for frame in frames:
        for prim in frame.get("primitives", []):
            obj = prim.get("object")
            prob = prim.get("hitProbability")
            if obj is None or prob is None:
                continue
            try:
                object_ids.add(int(obj))
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                continue

    series: Dict[int, List[Optional[float]]] = {
        obj: [None for _ in range(frame_count)] for obj in sorted(object_ids)
    }

    if not series:
        return series

    for f_idx, frame in enumerate(frames):
        totals: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for prim in frame.get("primitives", []):
            obj = prim.get("object")
            prob = prim.get("hitProbability")
            if obj is None or prob is None:
                continue
            try:
                obj_idx = int(obj)
                prob_value = float(prob)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                continue
            totals[obj_idx] = totals.get(obj_idx, 0.0) + prob_value
            counts[obj_idx] = counts.get(obj_idx, 0) + 1
        for obj_idx, total in totals.items():
            count = counts.get(obj_idx, 0)
            if count:
                series[obj_idx][f_idx] = total / count

    return series


def _has_object_probability_series(series: Mapping[int, List[Optional[float]]]) -> bool:
    """Return True if any object-level probability samples are present."""

    return any(any(value is not None for value in values) for values in series.values())


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _write_html(fig: go.Figure, output: Path, auto_open: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, auto_open=auto_open)
    print(f"Wrote {output}")


def _performance_figure(
    frames: List[int],
    metrics: Mapping[str, List[float]],
    node_series: Mapping[str, List[float]],
) -> go.Figure:
    has_nodes = bool(node_series)
    if has_nodes:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    for name, values in metrics.items():
        if has_nodes:
            fig.add_trace(
                go.Scatter(x=frames, y=values, mode="lines+markers", name=name),
                secondary_y=False,
            )
        else:
            fig.add_trace(go.Scatter(x=frames, y=values, mode="lines+markers", name=name))

    if has_nodes:
        for name, values in node_series.items():
            fig.add_trace(
                go.Scatter(
                    x=frames,
                    y=values,
                    mode="lines+markers",
                    name=name,
                ),
                secondary_y=True,
            )
        fig.update_yaxes(title_text="Performance", secondary_y=False)
        fig.update_yaxes(title_text="Nodes", secondary_y=True)
    else:
        fig.update_yaxes(title_text="Value")

    fig.update_layout(title="Per-frame performance metrics", xaxis_title="Frame")
    return fig


def _node_figure(frames: Iterable[int], series: Mapping[str, List[int]]) -> go.Figure:
    fig = go.Figure()
    frame_list = list(frames)
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=frame_list, y=values, mode="lines+markers", name=name))
    fig.update_layout(title="Acceleration-structure node residency", xaxis_title="Frame", yaxis_title="Nodes")
    return fig


def _gpu_memory_figure(frames: List[int], mem: List[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=mem, mode="lines+markers", name="gpu_memory_mb"))
    fig.update_layout(title="GPU memory usage per frame", xaxis_title="Frame", yaxis_title="Memory (MB)")
    return fig


def _intersection_figure(counts: List[List[int]], active: List[List[int]]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        subplot_titles=("Intersection count", "Residency"),
    )

    if counts:
        frame_count = len(counts[0]) if counts[0] else 0
        prim_indices = list(range(len(counts)))
        frame_indices = list(range(frame_count))
        fig.add_trace(
            go.Heatmap(
                z=counts,
                x=frame_indices,
                y=prim_indices,
                colorscale="Viridis",
                colorbar=dict(title="Hits"),
            ),
            row=1,
            col=1,
        )

    if active:
        frame_count = len(active[0]) if active[0] else 0
        prim_indices = list(range(len(active)))
        frame_indices = list(range(frame_count))
        fig.add_trace(
            go.Heatmap(
                z=active,
                x=frame_indices,
                y=prim_indices,
                colorscale=[[0, "#f44336"], [1, "#4caf50"]],
                colorbar=dict(title="Active"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=700)
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    fig.update_yaxes(title_text="Primitive", row=1, col=1)
    fig.update_yaxes(title_text="Primitive", row=2, col=1)
    return fig



def _probability_figure(probabilities: List[List[Optional[float]]]) -> go.Figure:
    fig = go.Figure()
    if probabilities:
        frame_count = len(probabilities[0]) if probabilities[0] else 0
        frame_indices = list(range(frame_count))
        prim_indices = list(range(len(probabilities)))
        fig.add_trace(
            go.Heatmap(
                z=probabilities,
                x=frame_indices,
                y=prim_indices,
                colorscale="Inferno",
                colorbar=dict(title="P(hit)"),
                zmin=0.0,
                zmax=1.0,
            )
        )
    fig.update_layout(
        title="Per-primitive hit probability",
        xaxis_title="Frame",
        yaxis_title="Primitive",
        height=500,
    )
    return fig


def _object_probability_figure(
    frames: Iterable[int], series: Mapping[int, List[Optional[float]]]
) -> go.Figure:
    fig = go.Figure()
    frame_list = list(frames)
    for obj_idx, values in series.items():
        if not any(value is not None for value in values):
            continue
        fig.add_trace(
            go.Scatter(
                x=frame_list,
                y=values,
                mode="lines+markers",
                name=f"Object {obj_idx}",
            )
        )
    fig.update_layout(
        title="Average hit probability per object",
        xaxis_title="Frame",
        yaxis_title="Probability",
    )
    fig.update_yaxes(range=[0.0, 1.0])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all renderer research plots from one command")
    parser.add_argument("runs", type=Path, help="Directory containing perf.csv, gpu_mem.csv, and acceleration-structure dumps")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory that will receive the generated HTML files")
    parser.add_argument("--as-path", type=Path, default=None, help="Override path to acceleration-structure dumps (defaults to <runs>/as)")
    parser.add_argument("--open", dest="auto_open", action="store_true", help="Open generated HTML files in the default browser")
    parser.add_argument("--no-open", dest="auto_open", action="store_false", help="Do not open the generated HTML files")
    parser.set_defaults(auto_open=False)
    args = parser.parse_args()

    outputs: Dict[str, Path] = {}

    perf_path = args.runs / "perf.csv"
    try:
        frames, metrics, node_series = _load_perf_csv(perf_path)
    except FileNotFoundError:
        print(f"Skipping performance plot: {perf_path} was not found")
    except Exception as exc:  # pragma: no cover - defensive reporting
        print(f"Skipping performance plot: failed to parse {perf_path}: {exc}")
    else:
        perf_fig = _performance_figure(frames, metrics, node_series)
        perf_output = args.output_dir / "performance.html"
        _write_html(perf_fig, perf_output, args.auto_open)
        outputs["performance"] = perf_output

    gpu_path = args.runs / "gpu_mem.csv"
    try:
        gpu_frames, gpu_mem = _load_gpu_mem_csv(gpu_path)
    except FileNotFoundError:
        print(f"Skipping GPU memory plot: {gpu_path} was not found")
    except Exception as exc:  # pragma: no cover - defensive reporting
        print(f"Skipping GPU memory plot: failed to parse {gpu_path}: {exc}")
    else:
        gpu_fig = _gpu_memory_figure(gpu_frames, gpu_mem)
        gpu_output = args.output_dir / "gpu_memory.html"
        _write_html(gpu_fig, gpu_output, args.auto_open)
        outputs["gpu_memory"] = gpu_output

    as_path = args.as_path if args.as_path is not None else args.runs / "as"
    if args.as_path is None and not as_path.exists():
        # Many historical captures stored JSON dumps directly in the runs folder.
        if any(args.runs.glob("*.json")):
            as_path = args.runs
    frames_for_nodes: Optional[List[Dict[str, Any]]] = None
    try:
        frames_for_nodes = _load_frames(as_path)
    except FileNotFoundError:
        print(f"Skipping acceleration-structure plots: {as_path} was not found or empty")
    except Exception as exc:  # pragma: no cover - defensive reporting
        print(f"Skipping acceleration-structure plots: failed to parse {as_path}: {exc}")

    if frames_for_nodes:
        node_series = _node_series_from_frames(frames_for_nodes)
        node_fig = _node_figure(range(len(frames_for_nodes)), node_series)
        node_output = args.output_dir / "active_nodes.html"
        _write_html(node_fig, node_output, args.auto_open)
        outputs["active_nodes"] = node_output

        counts, active, probabilities = _build_intersection_arrays(frames_for_nodes)
        if counts or active:
            intersection_fig = _intersection_figure(counts, active)
            intersection_output = args.output_dir / "intersections.html"
            _write_html(intersection_fig, intersection_output, args.auto_open)
            outputs["intersections"] = intersection_output

        if _has_probability_data(probabilities):
            probability_fig = _probability_figure(probabilities)
            probability_output = args.output_dir / "hit_probability.html"
            _write_html(probability_fig, probability_output, args.auto_open)
            outputs["hit_probability"] = probability_output

        object_series = _aggregate_object_probabilities(frames_for_nodes)
        if _has_object_probability_series(object_series):
            object_fig = _object_probability_figure(
                range(len(frames_for_nodes)), object_series
            )
            object_output = args.output_dir / "object_hit_probability.html"
            _write_html(object_fig, object_output, args.auto_open)
            outputs["object_hit_probability"] = object_output

    if not outputs:
        raise SystemExit("No plots were generated; ensure the runs directory contains logs")


if __name__ == "__main__":
    main()
