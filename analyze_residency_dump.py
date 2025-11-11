#!/usr/bin/env python3
"""Visualise stochastic residency metrics logged to renderer benchmark CSVs.

The renderer no longer emits ``as/frame_XXXX.json`` acceleration-structure
dumps, which means residency analysis must rely entirely on the CSV exports
produced in benchmark mode.  This script mirrors the CSV handling used by
``compare_runs.py`` so that the same log files can now drive single-run
diagnostics.  Point the tool at a benchmark CSV (or the directory containing
it) and the following artefacts will be generated when the relevant columns are
present:

* ``hit_probability_heatmap.png`` – a per-frame heatmap built from columns such
  as ``primitive_0_hit_probability``.  The script automatically discovers
  numbered primitive columns and paints them against the recorded frame index.
* ``object_hit_probability.png`` – line plots for columns named like
  ``object_12_hit_probability``.  The highest ``top-n`` object curves are drawn
  to highlight residency outliers across the run.
* ``object_hit_probability.csv`` – the numeric data backing the per-object plot
  (one column per object index) for downstream tooling.
* ``hit_probability_trends.png`` – aggregated metrics for
  ``avg_hit_probability``, ``p95_hit_probability``, and
  ``probability_threshold`` plotted against the frame index.
* ``probabilistic_toggles.png`` – a helper plot for the
  ``probabilistic_toggles`` column when it is present in the CSV.

Example::

    python analyze_residency_dump.py Benchmarks/metrics_20251016_101530.csv

If the provided path is a directory the script will search for a single CSV
inside it, preferring files with ``metrics`` in the name.  Set
``METALAPT_BENCH`` before launching the renderer to store benchmark logs in a
known location; the script defaults to that directory when no explicit path is
supplied.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


PrimitiveSeries = Dict[int, List[float]]
ObjectSeries = Dict[int, List[float]]

PRIMITIVE_COLUMN_RE = re.compile(
    r"(?:(?:primitive|prim)[_\-]?)(\d+)(?:[_\-]hit[_\-]?probability|[_\-]?probability)?$",
    re.IGNORECASE,
)
OBJECT_COLUMN_RE = re.compile(
    r"object[_\-]?(\d+)(?:[_\-]hit[_\-]?probability|[_\-]?probability)?$",
    re.IGNORECASE,
)


@dataclass
class RunMetrics:
    """Container for per-frame CSV metrics used by the residency analyser."""

    frames: List[int]
    metrics: Dict[str, List[float]]


def _load_csv(path: Path) -> RunMetrics:
    """Return frame numbers and numeric metric series from ``path``."""

    frames: List[int] = []
    metrics: Dict[str, List[float]] = {}

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        for row in reader:
            frame_value = row.get("frame")
            if frame_value in (None, ""):
                raise ValueError(f"Missing 'frame' value in {path}")
            frames.append(int(float(frame_value)))

            for key, value in row.items():
                if key == "frame" or value in (None, ""):
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                metrics.setdefault(key, []).append(numeric_value)

    valid_frame_count = len(frames)
    metrics = {
        key: values
        for key, values in metrics.items()
        if len(values) == valid_frame_count
    }
    return RunMetrics(frames=frames, metrics=metrics)


def _partition_metrics(metrics: Mapping[str, Sequence[float]]) -> tuple[
    PrimitiveSeries, ObjectSeries, Dict[str, List[float]]
]:
    """Split ``metrics`` into primitive, object, and remaining series."""

    primitive_series: PrimitiveSeries = {}
    object_series: ObjectSeries = {}
    remaining: Dict[str, List[float]] = {}

    for name, values in metrics.items():
        primitive_match = PRIMITIVE_COLUMN_RE.search(name)
        if primitive_match:
            index = int(primitive_match.group(1))
            primitive_series[index] = list(map(float, values))
            continue

        object_match = OBJECT_COLUMN_RE.search(name)
        if object_match:
            index = int(object_match.group(1))
            object_series[index] = list(map(float, values))
            continue

        remaining[name] = list(map(float, values))

    return primitive_series, object_series, remaining


def _probability_matrix(
    primitive_series: PrimitiveSeries, frame_count: int
) -> Optional[np.ndarray]:
    if not primitive_series:
        return None

    max_index = max(primitive_series)
    matrix = np.full((frame_count, max_index + 1), np.nan, dtype=float)
    for index, series in primitive_series.items():
        if len(series) != frame_count:
            continue
        matrix[:, index] = series
    if np.isnan(matrix).all():
        return None
    return matrix


def plot_heatmap(
    matrix: np.ndarray, frames: Sequence[int], output_dir: Path
) -> Path:
    extent = [0, matrix.shape[1], frames[0], frames[-1] + 1]

    plt.figure(figsize=(12, 6))
    img = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        cmap="viridis",
    )
    plt.colorbar(img, label="hitProbability")
    plt.xlabel("Primitive index")
    plt.ylabel("Frame")
    plt.title("Per-primitive hitProbability heatmap")
    plt.tight_layout()

    output_path = output_dir / "hit_probability_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_object_series(
    object_series: Mapping[int, Sequence[float]],
    frames: Sequence[int],
    output_dir: Path,
    top_n: int,
) -> Optional[Path]:
    if not object_series:
        return None

    frame_numbers = list(frames)
    candidates = []
    for object_index, series in object_series.items():
        finite_values = [value for value in series if not math.isnan(value)]
        if not finite_values or len(series) != len(frame_numbers):
            continue
        candidates.append((object_index, max(finite_values)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = [object_index for object_index, _ in candidates[:top_n]]

    plt.figure(figsize=(12, 6))
    for object_index in selected:
        series = object_series[object_index]
        plt.plot(frame_numbers, series, marker="o", label=f"object {object_index}")

    plt.xlabel("Frame")
    plt.ylabel("Average hitProbability")
    plt.title("Top object hitProbability trends")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = output_dir / "object_hit_probability.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def export_object_csv(
    object_series: Mapping[int, Sequence[float]],
    frames: Sequence[int],
    output_dir: Path,
) -> Optional[Path]:
    if not object_series:
        return None

    frame_numbers = list(frames)
    object_indices = sorted(object_series)
    if not object_indices:
        return None

    output_path = output_dir / "object_hit_probability.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        header = ["frame"] + [f"object_{idx}" for idx in object_indices]
        writer.writerow(header)
        for row_idx, frame_number in enumerate(frame_numbers):
            row = [frame_number]
            for object_index in object_indices:
                series = object_series[object_index]
                value = series[row_idx] if row_idx < len(series) else math.nan
                row.append("" if math.isnan(value) else f"{value:.6f}")
            writer.writerow(row)
    return output_path


def plot_probability_trends(
    frames: Sequence[int],
    metrics: Mapping[str, Sequence[float]],
    output_dir: Path,
) -> Optional[Path]:
    keys = [
        ("avg_hit_probability", "Average"),
        ("p95_hit_probability", "95th percentile"),
        ("probability_threshold", "Threshold"),
    ]
    available = [
        (name, label)
        for name, label in keys
        if name in metrics and len(metrics[name]) == len(frames)
    ]

    if not available:
        return None

    plt.figure(figsize=(12, 6))
    for name, label in available:
        plt.plot(frames, metrics[name], marker="o", label=label)

    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.title("Hit probability trends")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()

    output_path = output_dir / "hit_probability_trends.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_probabilistic_toggles(
    frames: Sequence[int],
    metrics: Mapping[str, Sequence[float]],
    output_dir: Path,
) -> Optional[Path]:
    key = "probabilistic_toggles"
    if key not in metrics or len(metrics[key]) != len(frames):
        return None

    plt.figure(figsize=(12, 4))
    plt.plot(frames, metrics[key], marker="s", color="tab:orange")
    plt.xlabel("Frame")
    plt.ylabel("Toggles")
    plt.title("Probabilistic residency toggles per frame")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = output_dir / "probabilistic_toggles.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _resolve_csv(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {path}")
        return path

    if not path.exists():
        raise FileNotFoundError(path)

    candidates = sorted(p for p in path.glob("*.csv") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {path}")

    preferred = [p for p in candidates if "metrics" in p.stem]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise ValueError(
            "Multiple metrics CSV files found; please specify the desired file explicitly"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple CSV files found; please specify the desired file explicitly"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Benchmark CSV or directory (defaults to METALAPT_BENCH)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots and CSV data will be written (defaults to the CSV directory)",
    )
    parser.add_argument(
        "--top-objects",
        type=int,
        default=8,
        help="Number of objects to include in the trend plot (default: 8)",
    )

    args = parser.parse_args()

    env_default = os.getenv("METALAPT_BENCH")
    if args.path is None:
        if not env_default:
            parser.error("Provide a CSV path or set METALAPT_BENCH.")
        base_path = Path(env_default).resolve()
    else:
        base_path = args.path.resolve()

    csv_path = _resolve_csv(base_path)
    run_metrics = _load_csv(csv_path)
    primitive_series, object_series, remaining_metrics = _partition_metrics(run_metrics.metrics)

    output_dir = args.output_dir.resolve() if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    matrix = _probability_matrix(primitive_series, len(run_metrics.frames))
    if matrix is not None:
        outputs["heatmap"] = plot_heatmap(matrix, run_metrics.frames, output_dir)
        print(f"Saved heatmap: {outputs['heatmap']}")
    else:
        print("No per-primitive probability columns found; skipping heatmap.")

    object_plot_path = plot_object_series(
        object_series, run_metrics.frames, output_dir, args.top_objects
    )
    if object_plot_path:
        outputs["object_plot"] = object_plot_path
        print(f"Saved object plot: {object_plot_path}")
    else:
        print("No object probability columns found; skipping object trend plot.")

    csv_output = export_object_csv(object_series, run_metrics.frames, output_dir)
    if csv_output:
        outputs["object_csv"] = csv_output
        print(f"Saved object data: {csv_output}")

    trend_path = plot_probability_trends(run_metrics.frames, remaining_metrics, output_dir)
    if trend_path:
        outputs["probability_trends"] = trend_path
        print(f"Saved probability trends: {trend_path}")

    toggles_path = plot_probabilistic_toggles(run_metrics.frames, remaining_metrics, output_dir)
    if toggles_path:
        outputs["probabilistic_toggles"] = toggles_path
        print(f"Saved probabilistic toggles: {toggles_path}")

    if not outputs:
        raise SystemExit(
            "No residency metrics were discovered in the CSV; ensure the log contains probability columns"
        )


if __name__ == "__main__":
    main()

