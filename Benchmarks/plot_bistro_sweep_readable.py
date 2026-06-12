#!/usr/bin/env python3
"""Generate reader-friendly Bistro sweep comparison plots.

This script focuses on readability:
- human-friendly labels
- explicit "Always Resident" baseline in black
- grouped plots by strategy family
- only the key memory/churn metrics
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "cpu_ms",
    "gpu_ms",
    "gpu_memory_mb",
    "gpu_geometry_mb",
    "resident_geometry_memory_mb",
    "strict_resident_geometry_memory_mb",
    "gpu_heaps_mb",
    "gpu_other_mb",
    "objects_offload_requested",
    "offload_requested_mb",
]

FAMILY_ORDER = [
    "distance",
    "rayhit",
    "screenspace",
    "energy",
    "probabilistic",
    "unified",
    "environment",
    "predictive",
]

FAMILY_COLORS = {
    "distance": "#1f77b4",
    "rayhit": "#d62728",
    "screenspace": "#2ca02c",
    "energy": "#ff7f0e",
    "probabilistic": "#9467bd",
    "unified": "#8c564b",
    "environment": "#17becf",
    "predictive": "#bcbd22",
    "baseline": "#111111",
}

VARIANT_SHADE = {
    "base": 1.00,
    "aggressive": 0.80,
    "conservative": 1.20,
}


@dataclass(frozen=True)
class RunInfo:
    scene_id: str
    label: str
    family: str
    variant: str
    metrics_csv: Path
    df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate readable Bistro sweep plots.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "Benchmarks" / "bistro_sweep_runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Benchmarks" / "comparative_bistro_baseline_readable",
    )
    parser.add_argument("--rolling", type=int, default=5)
    return parser.parse_args()


def human_label(scene_id: str) -> tuple[str, str, str]:
    if scene_id == "scene_bistro_test_v2":
        return "Always Resident", "baseline", "base"
    stem = scene_id.removeprefix("scene_bistro_test_v2_")
    parts = stem.split("_")
    family = parts[0]
    variant = "base"
    if len(parts) > 1:
        variant = parts[1]
    family_label = {
        "distance": "Distance",
        "rayhit": "Ray Hit",
        "screenspace": "Screen Space",
        "energy": "Energy",
        "probabilistic": "Probabilistic",
        "unified": "Unified",
        "environment": "Environment",
        "predictive": "Predictive",
    }.get(family, family.title())
    if variant == "base":
        return family_label, family, variant
    return f"{family_label} ({variant.title()})", family, variant


def shade_hex(color: str, factor: float) -> str:
    color = color.lstrip("#")
    rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    adjusted = []
    for c in rgb:
        if factor >= 1.0:
            value = c + (255 - c) * (factor - 1.0)
        else:
            value = c * factor
        adjusted.append(max(0, min(255, int(round(value)))))
    return "#" + "".join(f"{c:02x}" for c in adjusted)


def line_color(run: RunInfo) -> str:
    if run.family == "baseline":
        return FAMILY_COLORS["baseline"]
    base = FAMILY_COLORS.get(run.family, "#555555")
    return shade_hex(base, VARIANT_SHADE.get(run.variant, 1.0))


def finite_minmax(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def load_runs(root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = run_dir / "run_summary.json"
        complete = False
        if summary.exists():
            try:
                complete = bool(json.loads(summary.read_text()).get("complete"))
            except Exception:
                complete = False
        if not complete:
            continue
        metrics = sorted(run_dir.glob("metrics_*.csv"))
        if not metrics:
            continue
        parts = run_dir.name.split("_")
        scene_id = "_".join(parts[:-2]) if len(parts) >= 3 else run_dir.name
        label, family, variant = human_label(scene_id)
        df = pd.read_csv(metrics[0])
        for col in df.columns:
            if col not in {
                "strategy",
                "environment_depth_weights",
                "environment_depth_radii",
                "primitive_probabilities",
                "object_probabilities",
            }:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "offloaded_nodes" not in df.columns and {"total_nodes", "resident_nodes"}.issubset(df.columns):
            df["offloaded_nodes"] = pd.to_numeric(df["total_nodes"], errors="coerce") - pd.to_numeric(
                df["resident_nodes"], errors="coerce"
            )
        runs.append(RunInfo(scene_id, label, family, variant, metrics[0], df))
    return runs


def svg_line_plot(path: Path, title: str, y_label: str, runs: list[RunInfo], column: str, rolling: int) -> None:
    width, height = 1320, 760
    margin_l, margin_r, margin_t, margin_b = 90, 320, 65, 85
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_values = []
    y_values = []
    series = []
    for run in runs:
        x = safe_numeric(run.df, "frame").to_numpy(dtype=float)
        y = safe_numeric(run.df, column)
        if rolling > 1:
            y = y.rolling(window=rolling, min_periods=1).mean()
        y = y.to_numpy(dtype=float)
        x_values.extend(x[np.isfinite(x)])
        y_values.extend(y[np.isfinite(y)])
        series.append((run, x, y))

    x_lo, x_hi = finite_minmax(x_values)
    y_lo, y_hi = finite_minmax(y_values)

    def sx(v: float) -> float:
        return margin_l + (v - x_lo) / (x_hi - x_lo) * plot_w

    def sy(v: float) -> float:
        return margin_t + (y_hi - v) / (y_hi - y_lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<text x="{margin_l+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="14">Frame</text>',
        f'<text x="24" y="{margin_t+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {margin_t+plot_h/2})" font-family="Arial" font-size="14">{y_label}</text>',
    ]

    for tick in range(6):
        xv = x_lo + (x_hi - x_lo) * tick / 5
        yv = y_lo + (y_hi - y_lo) * tick / 5
        parts.append(f'<line x1="{sx(xv):.1f}" y1="{margin_t}" x2="{sx(xv):.1f}" y2="{margin_t+plot_h}" stroke="#eee"/>')
        parts.append(f'<text x="{sx(xv):.1f}" y="{margin_t+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.0f}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{sy(yv):.1f}" x2="{margin_l+plot_w}" y2="{sy(yv):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{sy(yv)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.1f}</text>')

    for idx, (run, xs, ys) in enumerate(series):
        points = []
        for xv, yv in zip(xs, ys):
            if np.isfinite(xv) and np.isfinite(yv):
                points.append(f"{sx(float(xv)):.1f},{sy(float(yv)):.1f}")
        if len(points) >= 2:
            stroke = line_color(run)
            width_px = 3.6 if run.family == "baseline" else 2.0
            dash = ' stroke-dasharray="8 5"' if run.variant == "aggressive" else ""
            if run.variant == "conservative":
                dash = ' stroke-dasharray="3 3"'
            parts.append(
                f'<polyline fill="none" stroke="{stroke}" stroke-width="{width_px}"{dash} points="' + " ".join(points) + '"/>'
            )
        ly = margin_t + 22 + idx * 22
        stroke = line_color(run)
        width_px = 3.6 if run.family == "baseline" else 2.6
        parts.append(f'<line x1="{margin_l+plot_w+25}" y1="{ly-4}" x2="{margin_l+plot_w+58}" y2="{ly-4}" stroke="{stroke}" stroke-width="{width_px}"/>')
        parts.append(f'<text x="{margin_l+plot_w+66}" y="{ly}" font-family="Arial" font-size="13">{run.label}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def grouped_runs(all_runs: list[RunInfo]) -> dict[str, list[RunInfo]]:
    baseline = [run for run in all_runs if run.family == "baseline"]
    groups: dict[str, list[RunInfo]] = {"all": all_runs}
    for family in FAMILY_ORDER:
        family_runs = [run for run in all_runs if run.family == family]
        if family_runs:
            groups[family] = baseline + family_runs
    return groups


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.input_root)
    if not runs:
        raise SystemExit(f"No completed runs found under {args.input_root}")

    groups = grouped_runs(runs)

    summary_rows = []
    for run in runs:
        row = {"label": run.label, "scene_id": run.scene_id, "csv_path": str(run.metrics_csv), "frames": len(run.df)}
        for col in KEY_COLUMNS:
            values = safe_numeric(run.df, col)
            row[f"{col}_mean"] = float(values.mean()) if not values.dropna().empty else np.nan
            row[f"{col}_max"] = float(values.max()) if not values.dropna().empty else np.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "comparison_summary.csv", index=False)

    # All-runs overview plots
    y_labels = {
        "cpu_ms": "CPU Time (ms)",
        "gpu_ms": "GPU Time (ms)",
        "gpu_memory_mb": "GPU Memory (MB)",
        "gpu_geometry_mb": "Geometry-Class Memory (MB)",
        "resident_geometry_memory_mb": "Resident Geometry Payload (MB)",
        "strict_resident_geometry_memory_mb": "Strict Resident Geometry Payload (MB)",
        "gpu_heaps_mb": "Heap Memory (MB)",
        "gpu_other_mb": "Other GPU Memory (MB)",
        "objects_offload_requested": "Offload Requests",
        "offload_requested_mb": "Requested Offload (MB)",
    }
    for col in KEY_COLUMNS:
        svg_line_plot(
            args.output_dir / f"{col}_all_runs.svg",
            f"All Completed Bistro Runs: {y_labels[col]}",
            y_labels[col],
            groups["all"],
            col,
            args.rolling,
        )

    # Family-specific plots for the key memory metrics.
    family_cols = [
        "cpu_ms",
        "gpu_ms",
        "gpu_memory_mb",
        "gpu_geometry_mb",
        "resident_geometry_memory_mb",
        "strict_resident_geometry_memory_mb",
        "gpu_heaps_mb",
        "gpu_other_mb",
    ]
    for family, family_runs in groups.items():
        if family == "all":
            continue
        for col in family_cols:
            svg_line_plot(
                args.output_dir / f"{family}_{col}.svg",
                f"{family.title()} Family vs Always Resident: {y_labels[col]}",
                y_labels[col],
                family_runs,
                col,
                args.rolling,
            )

    manifest = [
        "Readable Bistro sweep comparison plots",
        f"Input root: {args.input_root}",
        f"Output dir: {args.output_dir}",
        "",
        "Runs:",
    ]
    for run in runs:
        manifest.append(f"- {run.label}: {run.metrics_csv}")
    (args.output_dir / "comparison_manifest.txt").write_text("\n".join(manifest))

    print(f"Wrote plots to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
