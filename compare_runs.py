#!/usr/bin/env python3
"""Compare two benchmark CSV files and emit per-metric Matplotlib plots.

The tool expects the CSVs to contain a ``frame`` column plus one or more
numeric metric columns.  For every metric that exists in both files a
``frame vs metric`` plot is written next to the CSVs.  This is useful when
benchmark runs are timestamped under ``benchmarks/runs1``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

Frames = List[int]
Series = Dict[str, List[float]]


def _load_csv(path: Path) -> Tuple[Frames, Series]:
    """Return frame numbers and numeric metric series from ``path``."""

    frames: Frames = []
    metrics: Series = {}

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        for row in reader:
            frame_value = row.get("frame")
            if frame_value is None or frame_value == "":
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

    # Drop metrics that do not have data for every frame
    valid_frame_count = len(frames)
    metrics = {
        key: values
        for key, values in metrics.items()
        if len(values) == valid_frame_count
    }
    return frames, metrics


def _metric_names(metrics_a: Series, metrics_b: Series) -> Iterable[str]:
    """Return the metrics shared by both CSV files."""

    shared = sorted(set(metrics_a) & set(metrics_b))
    if not shared:
        raise ValueError("No shared numeric metrics between the provided CSVs")
    return shared


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def plot_metrics(
    csv_a: Path,
    csv_b: Path,
    output_dir: Path,
    frames_a: Frames,
    frames_b: Frames,
    metrics_a: Series,
    metrics_b: Series,
) -> None:
    """Plot shared metrics between ``csv_a`` and ``csv_b``."""

    label_a = csv_a.stem
    label_b = csv_b.stem

    for metric in _metric_names(metrics_a, metrics_b):
        plt.figure(figsize=(10, 6))
        plt.plot(frames_a, metrics_a[metric], label=label_a, marker="o")
        plt.plot(frames_b, metrics_b[metric], label=label_b, marker="s")
        plt.xlabel("Frame")
        plt.ylabel(metric)
        plt.title(f"{metric} comparison")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        output_path = output_dir / f"{_safe_name(metric)}_comparison.png"
        plt.savefig(output_path, dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a", type=Path, help="First CSV file to compare")
    parser.add_argument("csv_b", type=Path, help="Second CSV file to compare")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated PNG files (defaults to the CSV directory)",
    )

    args = parser.parse_args()

    csv_a: Path = args.csv_a.resolve()
    csv_b: Path = args.csv_b.resolve()

    if not csv_a.is_file() or not csv_b.is_file():
        raise FileNotFoundError("Both CSV paths must point to files")

    frames_a, metrics_a = _load_csv(csv_a)
    frames_b, metrics_b = _load_csv(csv_b)

    output_dir = args.output_dir.resolve() if args.output_dir else csv_a.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(csv_a, csv_b, output_dir, frames_a, frames_b, metrics_a, metrics_b)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
