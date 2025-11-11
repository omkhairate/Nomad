#!/usr/bin/env python3
"""Visualise per-primitive hit probabilities from acceleration-structure dumps.

This tool complements ``compare_runs.py`` by focusing on the stochastic
residency tracker data that the renderer emits inside ``as/frame_XXXX.json``
files.  Point the script at a benchmark run directory (or directly at the
``as`` folder) and it will generate:

* ``hit_probability_heatmap.png`` – a frame-by-frame heatmap of each
  primitive's ``hitProbability`` value.  Bright regions correspond to
  primitives that stay hot or repeatedly receive hits, while dark regions show
  primitives that are cooling successfully.
* ``object_hit_probability.png`` – line plots of the average
  ``hitProbability`` per object, highlighting the objects with the highest
  observed probabilities so that residency anomalies are easy to spot.
* ``object_hit_probability.csv`` – the numeric data behind the per-object plot
  (one column per object index) for downstream tooling.

Capture guidance
================
1. Export ``METALAPT_BENCHMARK=/path/to/runs`` before launching the renderer.
   (Legacy automation that still sets ``MPT_RUNS_PATH`` continues to work as a
   fallback.)
2. Optionally bound the capture with ``MPT_MAX_FRAMES=300`` (or similar) to
   keep the dump series manageable.
3. After the run, you should have ``runs/<timestamp>/as/frame_XXXX.json``
   alongside CSV metrics.  Run this script against the directory to produce the
   heatmap and per-object summaries that help debug residency behaviour.
4. Interpret the new plots as follows: bright bands in the heatmap flag
   primitives whose ``hitProbability`` remains high (potential residency
   pressure), while diagonal fades show cooling behaviour.  The per-object
   trends highlight which scene instances accumulate probability so you can
   focus on problematic assets frame-by-frame.

Example::

    python analyze_residency_dump.py runs/2024-10-31_15-00-00 --output-dir plots

When multiple benchmark runs are stored inside ``runs/``, point to the specific
run directory (the script will automatically look for an ``as`` subdirectory).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PrimitiveRecord:
    """Normalised primitive data extracted from a frame dump."""

    index: int
    hit_probability: float
    object_index: Optional[int]
    active: bool


@dataclass
class FrameRecord:
    """Container for per-frame primitive information."""

    frame: int
    primitives: List[PrimitiveRecord]


def _coerce_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):  # bool is a subclass of int
            return int(value)
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _infer_frame_index(data: Mapping[str, object], path: Path, fallback: int) -> int:
    candidate = data.get("frame")
    frame_index = _coerce_int(candidate)
    if frame_index is not None:
        return frame_index
    match = re.search(r"(\d+)", path.stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return fallback


def _normalise_primitives(raw: Iterable[Mapping[str, object]]) -> List[PrimitiveRecord]:
    primitives: List[PrimitiveRecord] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        index = _coerce_int(entry.get("index"))
        if index is None or index < 0:
            continue
        hit_probability_raw = entry.get("hitProbability", 0.0)
        try:
            hit_probability = float(hit_probability_raw)
        except (TypeError, ValueError):
            hit_probability = 0.0
        object_index = _coerce_int(entry.get("object"))
        active = bool(entry.get("active", True))
        primitives.append(
            PrimitiveRecord(
                index=index,
                hit_probability=max(0.0, min(1.0, hit_probability)),
                object_index=object_index,
                active=active,
            )
        )
    primitives.sort(key=lambda prim: prim.index)
    return primitives


def _load_frame_file(path: Path, fallback_frame_index: int) -> List[FrameRecord]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, Mapping):
        primitives_raw = data.get("primitives")
        if not isinstance(primitives_raw, list):
            raise ValueError(f"Dump file {path} is missing a 'primitives' array")
        frame_index = _infer_frame_index(data, path, fallback_frame_index)
        primitives = _normalise_primitives(primitives_raw)
        return [FrameRecord(frame=frame_index, primitives=primitives)]
    if isinstance(data, list):
        frames: List[FrameRecord] = []
        for idx, entry in enumerate(data):
            if not isinstance(entry, Mapping):
                continue
            primitives_raw = entry.get("primitives")
            if not isinstance(primitives_raw, list):
                continue
            frame_index = _coerce_int(entry.get("frame"))
            if frame_index is None:
                frame_index = fallback_frame_index + idx
            frames.append(
                FrameRecord(
                    frame=frame_index,
                    primitives=_normalise_primitives(primitives_raw),
                )
            )
        if not frames:
            raise ValueError(f"Dump file {path} does not contain valid frame entries")
        return frames
    raise ValueError(f"Unsupported JSON structure in dump file {path}")


def load_frames(source: Path) -> List[FrameRecord]:
    """Return sorted frame records from ``source`` (file or directory)."""

    if source.is_file():
        return _load_frame_file(source, 0)

    if not source.exists():
        raise FileNotFoundError(source)

    json_paths = sorted(p for p in source.glob("*.json") if p.is_file())
    if not json_paths:
        raise FileNotFoundError(f"No JSON dumps found in {source}")

    frames: List[FrameRecord] = []
    for idx, path in enumerate(json_paths):
        frames.extend(_load_frame_file(path, idx))

    frames.sort(key=lambda frame: frame.frame)
    return frames


def probability_matrix(frames: Sequence[FrameRecord]) -> np.ndarray:
    """Build a ``frame_count x primitive_count`` array of hit probabilities."""

    if not frames:
        raise ValueError("No frames available")

    max_index = max((prim.index for frame in frames for prim in frame.primitives), default=-1)
    if max_index < 0:
        raise ValueError("Frames do not contain primitives")

    matrix = np.full((len(frames), max_index + 1), np.nan, dtype=float)
    for row, frame in enumerate(frames):
        for prim in frame.primitives:
            matrix[row, prim.index] = prim.hit_probability
    return matrix


def aggregate_by_object(frames: Sequence[FrameRecord]) -> Dict[int, List[float]]:
    """Return average hit probability per object for each frame."""

    object_series: Dict[int, List[float]] = {}
    for frame_idx, frame in enumerate(frames):
        for series in object_series.values():
            series.append(math.nan)

        per_object: MutableMapping[int, List[float]] = defaultdict(list)
        for prim in frame.primitives:
            if prim.object_index is None:
                continue
            per_object[prim.object_index].append(prim.hit_probability)

        for object_index, values in per_object.items():
            mean_value = float(np.mean(values)) if values else math.nan
            if object_index not in object_series:
                object_series[object_index] = [math.nan] * frame_idx + [mean_value]
            else:
                object_series[object_index][-1] = mean_value
    return object_series


def plot_heatmap(matrix: np.ndarray, frames: Sequence[FrameRecord], output_dir: Path) -> Path:
    frame_numbers = [frame.frame for frame in frames]
    extent = [0, matrix.shape[1], frame_numbers[0], frame_numbers[-1] + 1]

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
    frames: Sequence[FrameRecord],
    output_dir: Path,
    top_n: int,
) -> Optional[Path]:
    if not object_series:
        return None

    frame_numbers = [frame.frame for frame in frames]
    candidates = []
    for object_index, series in object_series.items():
        finite_values = [value for value in series if not math.isnan(value)]
        if not finite_values:
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
    object_series: Mapping[int, Sequence[float]], frames: Sequence[FrameRecord], output_dir: Path
) -> Optional[Path]:
    if not object_series:
        return None

    frame_numbers = [frame.frame for frame in frames]
    object_indices = sorted(object_series)

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


def resolve_source_path(path: Path) -> Path:
    if path.is_dir():
        as_dir = path / "as"
        return as_dir if as_dir.exists() else path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Run directory or JSON dump path (defaults to METALAPT_BENCHMARK)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots and CSV data will be written (defaults to the run directory)",
    )
    parser.add_argument(
        "--top-objects",
        type=int,
        default=8,
        help="Number of objects to include in the trend plot (default: 8)",
    )

    args = parser.parse_args()

    env_default = os.getenv("METALAPT_BENCHMARK") or os.getenv("MPT_RUNS_PATH")
    if args.path is None:
        if not env_default:
            parser.error(
                "Provide a run directory/JSON path or set METALAPT_BENCHMARK (or the legacy MPT_RUNS_PATH)."
            )
        base_path = Path(env_default).resolve()
    else:
        base_path = args.path.resolve()

    source = resolve_source_path(base_path)
    frames = load_frames(source)

    output_dir = args.output_dir.resolve() if args.output_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = probability_matrix(frames)
    heatmap_path = plot_heatmap(matrix, frames, output_dir)

    object_series = aggregate_by_object(frames)
    object_plot_path = plot_object_series(object_series, frames, output_dir, args.top_objects)
    csv_path = export_object_csv(object_series, frames, output_dir)

    print(f"Saved heatmap: {heatmap_path}")
    if object_plot_path:
        print(f"Saved object plot: {object_plot_path}")
    else:
        print("No object indices present in the dumps; skipping object trend plot.")
    if csv_path:
        print(f"Saved object data: {csv_path}")


if __name__ == "__main__":
    main()
