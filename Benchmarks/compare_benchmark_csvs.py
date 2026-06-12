#!/usr/bin/env python3
"""Generate comparative residency/memory plots across multiple benchmark CSVs.

The script intentionally writes lightweight SVG/CSV outputs and avoids heavier
plotting dependencies so it works in the current local environment.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = [
    "gpu_memory_mb",
    "gpu_geometry_mb",
    "resident_geometry_memory_mb",
    "gpu_textures_mb",
    "scratch_memory_mb",
    "objects_onload_requested",
    "objects_offload_requested",
    "onload_requested_mb",
    "offload_requested_mb",
    "offloaded_nodes",
]

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass(frozen=True)
class InputRun:
    label: str
    csv_path: Path
    df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_out = repo_root / "Benchmarks" / "comparative_plots"

    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple renderer metrics CSVs and generate SVG plots for "
            "residency, memory, and churn columns."
        )
    )
    parser.add_argument(
        "csv",
        nargs="+",
        help="Metrics CSV paths to compare.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help=(
            "Optional label for a CSV, in the same order as the CSV arguments. "
            "If omitted, the filename stem is used."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Output directory for plots and summaries. Default: {default_out}",
    )
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        help=(
            "Column to plot. Can be passed multiple times. If omitted, a default "
            "residency/memory set is used."
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=["frame", "wall_seconds"],
        default="frame",
        help="Horizontal axis for plots. Default: frame",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=1,
        help=(
            "Optional rolling-average window for smoother plots. Use 1 to disable. "
            "Default: 1"
        ),
    )
    return parser.parse_args()


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


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def numeric_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            pass
    return df


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "offloaded_nodes" not in result.columns:
        if "total_nodes" in result.columns and "resident_nodes" in result.columns:
            total_nodes = pd.to_numeric(result["total_nodes"], errors="coerce")
            resident_nodes = pd.to_numeric(result["resident_nodes"], errors="coerce")
            result["offloaded_nodes"] = (total_nodes - resident_nodes).clip(lower=0)
    return result


def safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def load_runs(csv_paths: list[str], labels: list[str]) -> list[InputRun]:
    runs: list[InputRun] = []
    for index, csv_arg in enumerate(csv_paths):
        csv_path = Path(csv_arg).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metrics CSV not found: {csv_arg}")
        label = labels[index] if index < len(labels) else csv_path.stem
        df = ensure_derived_columns(numeric_df(csv_path))
        runs.append(InputRun(label=label, csv_path=csv_path, df=df))
    return runs


def svg_line_plot(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
) -> None:
    width = 1280
    height = 720
    margin_l, margin_r, margin_t, margin_b = 90, 310, 60, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_lo, x_hi = finite_minmax(v for _, xs, _, _ in series for v in xs)
    y_lo, y_hi = finite_minmax(v for _, _, ys, _ in series for v in ys)

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
        f'<text x="{margin_l+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>',
        f'<text x="24" y="{margin_t+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {margin_t+plot_h/2})" font-family="Arial" font-size="14">{y_label}</text>',
    ]

    for tick in range(6):
        x_val = x_lo + (x_hi - x_lo) * tick / 5
        y_val = y_lo + (y_hi - y_lo) * tick / 5
        parts.append(f'<line x1="{sx(x_val):.1f}" y1="{margin_t}" x2="{sx(x_val):.1f}" y2="{margin_t+plot_h}" stroke="#eee"/>')
        parts.append(f'<text x="{sx(x_val):.1f}" y="{margin_t+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">{x_val:.0f}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{sy(y_val):.1f}" x2="{margin_l+plot_w}" y2="{sy(y_val):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{sy(y_val)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.1f}</text>')

    for idx, (name, xs, ys, stroke) in enumerate(series):
        points = []
        for xv, yv in zip(xs, ys):
            if np.isfinite(xv) and np.isfinite(yv):
                points.append(f"{sx(float(xv)):.1f},{sy(float(yv)):.1f}")
        if len(points) >= 2:
            parts.append(
                f'<polyline fill="none" stroke="{stroke}" stroke-width="2.2" points="'
                + " ".join(points)
                + '"/>'
            )
        ly = margin_t + 24 + idx * 22
        parts.append(f'<line x1="{margin_l+plot_w+28}" y1="{ly-4}" x2="{margin_l+plot_w+55}" y2="{ly-4}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(f'<text x="{margin_l+plot_w+62}" y="{ly}" font-family="Arial" font-size="13">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def build_series(
    runs: list[InputRun],
    column: str,
    x_axis: str,
    rolling: int,
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    output = []
    for index, run in enumerate(runs):
        x = safe_numeric_series(run.df, x_axis).to_numpy(dtype=float)
        if x_axis == "frame" and np.isnan(x).all():
            x = np.arange(len(run.df), dtype=float)
        y = safe_numeric_series(run.df, column)
        if rolling > 1:
            y = y.rolling(window=rolling, min_periods=1).mean()
        output.append((run.label, x, y.to_numpy(dtype=float), color(index)))
    return output


def write_summary_csv(
    path: Path,
    runs: list[InputRun],
    columns: list[str],
) -> None:
    rows: list[dict[str, object]] = []
    for run in runs:
        row: dict[str, object] = {
            "label": run.label,
            "csv_path": str(run.csv_path),
            "frames": len(run.df),
        }
        for column in columns:
            values = safe_numeric_series(run.df, column)
            row[f"{column}_mean"] = float(values.mean()) if not values.dropna().empty else np.nan
            row[f"{column}_median"] = float(values.median()) if not values.dropna().empty else np.nan
            row[f"{column}_max"] = float(values.max()) if not values.dropna().empty else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    runs = load_runs(args.csv, args.label)
    columns = args.column if args.column else list(DEFAULT_COLUMNS)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for column in columns:
        series = build_series(runs, column, args.x_axis, max(1, args.rolling))
        y_label = "count" if "nodes" in column or "objects_" in column or "requests" in column else "MB"
        title = f"{column} comparison"
        out_path = out_dir / f"{sanitize_filename(column)}_comparison.svg"
        svg_line_plot(out_path, title, args.x_axis, y_label, series)

    summary_csv = out_dir / "comparison_summary.csv"
    write_summary_csv(summary_csv, runs, columns)

    manifest = out_dir / "comparison_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "Comparative benchmark plots generated for:",
                *[f"- {run.label}: {run.csv_path}" for run in runs],
                "",
                f"x-axis: {args.x_axis}",
                f"rolling window: {max(1, args.rolling)}",
                "columns:",
                *[f"- {column}" for column in columns],
            ]
        )
        + "\n"
    )

    print(f"Wrote plots to: {out_dir}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
