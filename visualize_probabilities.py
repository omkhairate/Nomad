"""Visualize probabilistic residency metrics recorded in benchmark CSV files."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

FrameProbabilities = Dict[int, float]
ProbabilityFrames = List[FrameProbabilities]
Totals = Dict[int, float]
Counts = Dict[int, int]


def _parse_probability_column(value: Optional[str]) -> FrameProbabilities:
    """Parse a probability column into ``index -> probability`` mappings."""

    if not value:
        return {}

    result: FrameProbabilities = {}
    entries = value.split(";")
    for entry in entries:
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        index_part, probability_part = entry.split(":", 1)
        try:
            index = int(index_part.strip())
            probability = float(probability_part.strip())
        except ValueError:
            continue
        result[index] = probability
    return result


def _load_probability_data(
    path: Path,
) -> Tuple[
    List[int],
    ProbabilityFrames,
    ProbabilityFrames,
    str,
    Totals,
    Counts,
    Totals,
    Counts,
]:
    frames: List[int] = []
    primitive_frames: ProbabilityFrames = []
    object_frames: ProbabilityFrames = []
    strategy = ""

    primitive_totals: Totals = {}
    primitive_counts: Counts = {}
    object_totals: Totals = {}
    object_counts: Counts = {}

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        for row in reader:
            frame_value = row.get("frame")
            if frame_value is None or frame_value == "":
                raise ValueError(f"Missing 'frame' value in {path}")
            frames.append(int(float(frame_value)))

            if not strategy:
                strategy = row.get("strategy", "") or ""

            primitive_probs = _parse_probability_column(
                row.get("primitive_probabilities")
            )
            object_probs = _parse_probability_column(row.get("object_probabilities"))

            primitive_frames.append(primitive_probs)
            object_frames.append(object_probs)

            for index, probability in primitive_probs.items():
                primitive_totals[index] = primitive_totals.get(index, 0.0) + probability
                primitive_counts[index] = primitive_counts.get(index, 0) + 1

            for index, probability in object_probs.items():
                object_totals[index] = object_totals.get(index, 0.0) + probability
                object_counts[index] = object_counts.get(index, 0) + 1

    return (
        frames,
        primitive_frames,
        object_frames,
        strategy,
        primitive_totals,
        primitive_counts,
        object_totals,
        object_counts,
    )


def _average_probabilities(totals: Totals, counts: Counts) -> Dict[int, float]:
    return {
        index: totals[index] / counts[index]
        for index in totals
        if counts.get(index, 0) > 0
    }


def _select_indices(
    explicit: Optional[Sequence[int]],
    top_count: Optional[int],
    totals: Totals,
    counts: Counts,
    default_top: int,
) -> List[int]:
    if explicit:
        unique = sorted({int(value) for value in explicit})
        return unique

    averages = _average_probabilities(totals, counts)
    if not averages:
        return []

    if top_count is None:
        top_count = default_top
    top_count = max(int(top_count), 0)
    if top_count == 0:
        return []

    sorted_indices = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    return [index for index, _ in sorted_indices[:top_count]]


def _build_series(
    frames: ProbabilityFrames, indices: Iterable[int]
) -> Dict[int, List[float]]:
    series: Dict[int, List[float]] = {index: [] for index in indices}
    for frame_data in frames:
        for index in series:
            series[index].append(frame_data.get(index, math.nan))
    return series


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _plot_series(
    frames: List[int],
    series: Dict[int, List[float]],
    output_dir: Path,
    prefix: str,
    strategy: str,
) -> None:
    if not series:
        return

    for index, values in series.items():
        plt.figure(figsize=(10, 6))
        plt.plot(frames, values, marker="o")
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.0)
        title = f"{prefix.title()} {index} probability"
        if strategy:
            title += f" ({strategy})"
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        filename = f"{_safe_name(prefix)}_{index}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved {prefix} {index} plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Metrics CSV file produced by the renderer")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated plots (defaults to the CSV directory)",
    )
    parser.add_argument(
        "--primitives",
        type=int,
        nargs="*",
        help="Specific primitive indices to plot (defaults to top entries by average probability)",
    )
    parser.add_argument(
        "--top-primitives",
        type=int,
        default=None,
        help="Number of primitives with the highest average probability to plot",
    )
    parser.add_argument(
        "--objects",
        type=int,
        nargs="*",
        help="Specific object indices to plot (defaults to top entries by average probability)",
    )
    parser.add_argument(
        "--top-objects",
        type=int,
        default=None,
        help="Number of objects with the highest average probability to plot",
    )

    args = parser.parse_args()

    csv_path: Path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV path is not a file: {csv_path}")

    (
        frames,
        primitive_frames,
        object_frames,
        strategy,
        primitive_totals,
        primitive_counts,
        object_totals,
        object_counts,
    ) = _load_probability_data(csv_path)

    output_dir = args.output_dir.resolve() if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    primitive_indices = _select_indices(
        args.primitives, args.top_primitives, primitive_totals, primitive_counts, default_top=5
    )
    object_indices = _select_indices(
        args.objects, args.top_objects, object_totals, object_counts, default_top=5
    )

    if primitive_indices:
        primitive_series = _build_series(primitive_frames, primitive_indices)
        _plot_series(frames, primitive_series, output_dir, "primitive", strategy)
    else:
        print("No primitive probability data available to plot.")

    if object_indices:
        object_series = _build_series(object_frames, object_indices)
        _plot_series(frames, object_series, output_dir, "object", strategy)
    else:
        print("No object probability data available to plot.")


if __name__ == "__main__":
    main()
