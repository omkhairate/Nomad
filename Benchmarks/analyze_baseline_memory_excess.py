#!/usr/bin/env python3
"""Explain why Bistro sweep runs exceed the always-resident baseline.

This script compares the latest run for each scene against an always-resident
baseline scene and generates:

- a summary CSV ranking runs by excess memory
- a per-run peak attribution CSV
- a Markdown report with the strongest explanations we can infer from the logs
- lightweight SVG plots so results are easy to inspect without extra deps
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")

MEMORY_COMPONENT_COLUMNS = [
    "gpu_heaps_mb",
    "gpu_geometry_mb",
    "resident_geometry_memory_mb",
    "gpu_renderer_mb",
    "gpu_textures_mb",
    "gpu_restir_mb",
    "gpu_staging_mb",
    "scratch_memory_mb",
    "gpu_other_mb",
]

CORRELATION_COLUMNS = [
    "gpu_heaps_mb",
    "gpu_geometry_mb",
    "resident_geometry_memory_mb",
    "gpu_renderer_mb",
    "gpu_textures_mb",
    "gpu_restir_mb",
    "gpu_staging_mb",
    "scratch_memory_mb",
    "gpu_other_mb",
    "objects_onload_requested",
    "objects_offload_requested",
    "onload_requested_mb",
    "offload_requested_mb",
    "blas_build_requests",
    "tlas_rebuilds",
    "tlas_refits",
    "active_nodes",
    "resident_nodes",
    "resident_objects",
    "gpu_power_mw",
    "cpu_power_mw",
    "combined_power_mw",
    "gpu_frequency_mhz",
    "cpu_frequency_mhz",
]


@dataclass(frozen=True)
class RunRecord:
    scene_id: str
    timestamp: str
    run_dir: Path
    metrics_csv: Path
    complete: bool
    strategy: str | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Explain memory excess over the always-resident baseline.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "Benchmarks" / "bistro_observer_sweep_runs_topdown_fixed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Benchmarks" / "baseline_memory_excess_analysis",
    )
    parser.add_argument("--baseline-scene", default="scene_bistro_test_v2")
    parser.add_argument("--include-incomplete", action="store_true")
    return parser.parse_args()


def parse_scene_id(run_dir: Path) -> tuple[str, str]:
    match = TIMESTAMP_RE.search(run_dir.name)
    if not match:
        return run_dir.name, ""
    return run_dir.name[: match.start()], match.group(1)


def load_strategy(metrics_csv: Path) -> str | None:
    try:
        df = pd.read_csv(metrics_csv, usecols=["strategy"])
    except Exception:
        return None
    if "strategy" not in df.columns or df.empty:
        return None
    value = str(df["strategy"].dropna().iloc[0]).strip()
    return value or None


def discover_runs(root: Path, include_incomplete: bool) -> list[RunRecord]:
    latest: dict[str, RunRecord] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = sorted(run_dir.glob("metrics_*.csv"))
        if not metrics:
            continue
        summary_path = run_dir / "run_summary.json"
        complete = False
        if summary_path.exists():
            try:
                complete = bool(json.loads(summary_path.read_text()).get("complete"))
            except Exception:
                complete = False
        if not include_incomplete and not complete:
            continue
        scene_id, timestamp = parse_scene_id(run_dir)
        record = RunRecord(
            scene_id=scene_id,
            timestamp=timestamp,
            run_dir=run_dir,
            metrics_csv=metrics[0],
            complete=complete,
            strategy=load_strategy(metrics[0]),
        )
        previous = latest.get(scene_id)
        if previous is None or record.timestamp > previous.timestamp:
            latest[scene_id] = record
    return [latest[key] for key in sorted(latest)]


def numeric_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col in {"strategy", "environment_depth_weights", "environment_depth_radii", "primitive_probabilities", "object_probabilities"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "offloaded_nodes" not in df.columns and {"total_nodes", "resident_nodes"}.issubset(df.columns):
        df["offloaded_nodes"] = pd.to_numeric(df["total_nodes"], errors="coerce").fillna(0) - pd.to_numeric(
            df["resident_nodes"], errors="coerce"
        ).fillna(0)
    return df


def merge_thermal(run: RunRecord, df: pd.DataFrame) -> pd.DataFrame:
    thermal_csv = run.run_dir / "thermal.csv"
    if not thermal_csv.is_file():
        return df
    try:
        thermal = pd.read_csv(thermal_csv)
    except Exception:
        return df
    if thermal.empty or "run_elapsed_seconds" not in thermal.columns or "wall_seconds" not in df.columns:
        return df
    if "error" in thermal.columns and thermal["error"].fillna("").astype(str).str.len().gt(0).all():
        return df
    thermal = thermal.copy()
    thermal["thermal_join_seconds"] = pd.to_numeric(thermal["run_elapsed_seconds"], errors="coerce")
    thermal = thermal.dropna(subset=["thermal_join_seconds"]).sort_values("thermal_join_seconds")
    if thermal.empty:
        return df
    merged = df.copy()
    wall = pd.to_numeric(merged["wall_seconds"], errors="coerce")
    merged["thermal_join_seconds"] = wall - wall.min()
    thermal_cols = [c for c in thermal.columns if c != "run_elapsed_seconds"]
    return pd.merge_asof(
        merged.sort_values("thermal_join_seconds"),
        thermal[["thermal_join_seconds", *thermal_cols]].sort_values("thermal_join_seconds"),
        on="thermal_join_seconds",
        direction="nearest",
    )


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


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


def color(index: int) -> str:
    palette = [
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
    return palette[index % len(palette)]


def svg_bar_plot(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width = max(1000, 80 * len(labels) + 240)
    height = 640
    margin_l, margin_r, margin_t, margin_b = 95, 30, 65, 220
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_lo, y_hi = finite_minmax([0.0] + values)
    y_lo = min(0.0, y_lo)

    def sy(v: float) -> float:
        return margin_t + (y_hi - v) / (y_hi - y_lo) * plot_h

    slot = plot_w / max(1, len(labels))
    bar_w = slot * 0.72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<text x="24" y="{margin_t+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {margin_t+plot_h/2})" font-family="Arial" font-size="14">{ylabel}</text>',
    ]
    for tick in range(6):
        y_val = y_lo + (y_hi - y_lo) * tick / 5
        y = sy(y_val)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l+plot_w}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.1f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_l + idx * slot + (slot - bar_w) / 2
        y = sy(max(value, 0.0))
        h = margin_t + plot_h - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color(idx)}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.1f}</text>')
        parts.append(
            f'<text x="{x+bar_w/2:.1f}" y="{height-190}" text-anchor="end" transform="rotate(-55 {x+bar_w/2:.1f} {height-190})" font-family="Arial" font-size="11">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_line_plot(path: Path, title: str, x: np.ndarray, series: list[tuple[str, np.ndarray, str]], y_label: str) -> None:
    width, height = 1240, 680
    margin_l, margin_r, margin_t, margin_b = 85, 260, 65, 75
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x_lo, x_hi = finite_minmax(x)
    y_lo, y_hi = finite_minmax(v for _, vals, _ in series for v in vals)

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
        f'<text x="{margin_l+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">frame</text>',
        f'<text x="22" y="{margin_t+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {margin_t+plot_h/2})" font-family="Arial" font-size="14">{y_label}</text>',
    ]
    for tick in range(6):
        xv = x_lo + (x_hi - x_lo) * tick / 5
        yv = y_lo + (y_hi - y_lo) * tick / 5
        parts.append(f'<line x1="{sx(xv):.1f}" y1="{margin_t}" x2="{sx(xv):.1f}" y2="{margin_t+plot_h}" stroke="#eee"/>')
        parts.append(f'<text x="{sx(xv):.1f}" y="{margin_t+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.0f}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{sy(yv):.1f}" x2="{margin_l+plot_w}" y2="{sy(yv):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{sy(yv)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.1f}</text>')
    for idx, (name, vals, stroke) in enumerate(series):
        pts = []
        for xv, yv in zip(x, vals):
            if np.isfinite(xv) and np.isfinite(yv):
                pts.append(f"{sx(float(xv)):.1f},{sy(float(yv)):.1f}")
        if len(pts) >= 2:
            parts.append(f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="' + " ".join(pts) + '"/>')
        ly = margin_t + 22 + idx * 22
        parts.append(f'<line x1="{margin_l+plot_w+25}" y1="{ly-4}" x2="{margin_l+plot_w+54}" y2="{ly-4}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(f'<text x="{margin_l+plot_w+62}" y="{ly}" font-family="Arial" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def summarize_run(run: RunRecord, baseline_df: pd.DataFrame, output_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    df = merge_thermal(run, numeric_df(run.metrics_csv))
    merged = pd.merge(
        df,
        baseline_df,
        on="frame",
        suffixes=("", "_baseline"),
        how="inner",
    )
    merged["gpu_memory_excess_mb"] = safe_numeric(merged, "gpu_memory_mb") - safe_numeric(merged, "gpu_memory_mb_baseline")
    merged["positive_gpu_memory_excess_mb"] = merged["gpu_memory_excess_mb"].clip(lower=0.0)

    peak_idx = int(merged["gpu_memory_excess_mb"].idxmax())
    peak_row = merged.loc[peak_idx]
    above = merged[merged["gpu_memory_excess_mb"] > 0.0].copy()
    if above.empty:
        above = merged.iloc[[]].copy()

    attribution: list[dict[str, object]] = []
    for component in MEMORY_COMPONENT_COLUMNS:
        base_col = f"{component}_baseline"
        if component not in merged.columns or base_col not in merged.columns:
            continue
        delta = float(peak_row.get(component, np.nan) - peak_row.get(base_col, np.nan))
        attribution.append({"component": component, "delta_mb_at_peak": delta})
    attribution_df = pd.DataFrame(attribution).sort_values("delta_mb_at_peak", ascending=False)
    attribution_df.to_csv(output_dir / f"{run.scene_id}_peak_component_deltas.csv", index=False)

    excess = merged["gpu_memory_excess_mb"]
    correlations: list[tuple[str, float]] = []
    for col in CORRELATION_COLUMNS:
        if col not in merged.columns:
            continue
        series = pd.to_numeric(merged[col], errors="coerce")
        valid = pd.concat([excess, series], axis=1).dropna()
        if len(valid) < 8:
            continue
        if valid.iloc[:, 1].nunique() <= 1 or valid.iloc[:, 0].nunique() <= 1:
            continue
        corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
        if np.isfinite(corr):
            correlations.append((col, corr))
    correlations.sort(key=lambda item: abs(item[1]), reverse=True)
    top_corr = correlations[:8]

    summary = {
        "scene_id": run.scene_id,
        "strategy": run.strategy or "",
        "rows_compared": len(merged),
        "peak_gpu_memory_mb": float(safe_numeric(merged, "gpu_memory_mb").max()),
        "baseline_peak_gpu_memory_mb": float(safe_numeric(merged, "gpu_memory_mb_baseline").max()),
        "mean_gpu_memory_mb": float(safe_numeric(merged, "gpu_memory_mb").mean()),
        "baseline_mean_gpu_memory_mb": float(safe_numeric(merged, "gpu_memory_mb_baseline").mean()),
        "peak_excess_mb": float(excess.max()),
        "mean_excess_mb": float(excess.mean()),
        "p95_excess_mb": float(excess.quantile(0.95)),
        "frames_above_baseline": int((excess > 0.0).sum()),
        "fraction_above_baseline": float((excess > 0.0).mean()),
        "area_above_baseline_mb_frames": float(merged["positive_gpu_memory_excess_mb"].sum()),
        "peak_excess_frame": int(peak_row["frame"]),
        "peak_gpu_memory_baseline_frame_value_mb": float(peak_row.get("gpu_memory_mb_baseline", np.nan)),
        "peak_gpu_memory_frame_value_mb": float(peak_row.get("gpu_memory_mb", np.nan)),
        "peak_delta_gpu_heaps_mb": float(peak_row.get("gpu_heaps_mb", np.nan) - peak_row.get("gpu_heaps_mb_baseline", np.nan)),
        "peak_delta_gpu_geometry_mb": float(peak_row.get("gpu_geometry_mb", np.nan) - peak_row.get("gpu_geometry_mb_baseline", np.nan)),
        "peak_delta_resident_geometry_memory_mb": float(
            peak_row.get("resident_geometry_memory_mb", np.nan) - peak_row.get("resident_geometry_memory_mb_baseline", np.nan)
        ),
        "peak_delta_gpu_renderer_mb": float(peak_row.get("gpu_renderer_mb", np.nan) - peak_row.get("gpu_renderer_mb_baseline", np.nan)),
        "peak_delta_gpu_textures_mb": float(peak_row.get("gpu_textures_mb", np.nan) - peak_row.get("gpu_textures_mb_baseline", np.nan)),
        "peak_delta_gpu_restir_mb": float(peak_row.get("gpu_restir_mb", np.nan) - peak_row.get("gpu_restir_mb_baseline", np.nan)),
        "peak_delta_scratch_memory_mb": float(
            peak_row.get("scratch_memory_mb", np.nan) - peak_row.get("scratch_memory_mb_baseline", np.nan)
        ),
        "peak_offload_requested_mb": float(peak_row.get("offload_requested_mb", np.nan)),
        "peak_onload_requested_mb": float(peak_row.get("onload_requested_mb", np.nan)),
        "peak_objects_offload_requested": float(peak_row.get("objects_offload_requested", np.nan)),
        "peak_objects_onload_requested": float(peak_row.get("objects_onload_requested", np.nan)),
        "top_correlation_1": top_corr[0][0] if len(top_corr) > 0 else "",
        "top_correlation_1_value": top_corr[0][1] if len(top_corr) > 0 else np.nan,
        "top_correlation_2": top_corr[1][0] if len(top_corr) > 1 else "",
        "top_correlation_2_value": top_corr[1][1] if len(top_corr) > 1 else np.nan,
        "top_correlation_3": top_corr[2][0] if len(top_corr) > 2 else "",
        "top_correlation_3_value": top_corr[2][1] if len(top_corr) > 2 else np.nan,
    }

    interesting_cols = [
        "frame",
        "gpu_memory_excess_mb",
        "gpu_memory_mb",
        "gpu_memory_mb_baseline",
        "gpu_heaps_mb",
        "gpu_heaps_mb_baseline",
        "gpu_geometry_mb",
        "gpu_geometry_mb_baseline",
        "resident_geometry_memory_mb",
        "resident_geometry_memory_mb_baseline",
        "gpu_renderer_mb",
        "gpu_renderer_mb_baseline",
        "gpu_textures_mb",
        "gpu_textures_mb_baseline",
        "objects_onload_requested",
        "objects_offload_requested",
        "onload_requested_mb",
        "offload_requested_mb",
        "blas_build_requests",
        "tlas_rebuilds",
        "tlas_refits",
    ]
    top_excess_frames = merged.sort_values("gpu_memory_excess_mb", ascending=False)[interesting_cols].head(12)
    top_excess_frames.to_csv(output_dir / f"{run.scene_id}_top_excess_frames.csv", index=False)
    return summary, merged


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.input_root, args.include_incomplete)
    if not runs:
        raise SystemExit(f"No runs discovered under {args.input_root}")

    baseline_run = next((run for run in runs if run.scene_id == args.baseline_scene), None)
    if baseline_run is None:
        raise SystemExit(f"Could not find baseline scene '{args.baseline_scene}' under {args.input_root}")

    baseline_df = merge_thermal(baseline_run, numeric_df(baseline_run.metrics_csv))
    summaries: list[dict[str, object]] = []
    merged_runs: dict[str, pd.DataFrame] = {}
    for run in runs:
        if run.scene_id == args.baseline_scene:
            continue
        summary, merged = summarize_run(run, baseline_df, args.output_dir)
        summaries.append(summary)
        merged_runs[run.scene_id] = merged

    summary_df = pd.DataFrame(summaries).sort_values("peak_excess_mb", ascending=False)
    summary_path = args.output_dir / "baseline_memory_excess_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    labels = summary_df["scene_id"].tolist()
    svg_bar_plot(
        args.output_dir / "summary_peak_excess_mb.svg",
        "Peak GPU Memory Above Always-Resident Baseline",
        labels,
        summary_df["peak_excess_mb"].tolist(),
        "MB",
    )
    svg_bar_plot(
        args.output_dir / "summary_mean_excess_mb.svg",
        "Mean GPU Memory Above Always-Resident Baseline",
        labels,
        summary_df["mean_excess_mb"].tolist(),
        "MB",
    )
    svg_bar_plot(
        args.output_dir / "summary_area_above_baseline.svg",
        "Area Above Always-Resident Baseline",
        labels,
        summary_df["area_above_baseline_mb_frames"].tolist(),
        "MB * frame",
    )

    base_frame = safe_numeric(baseline_df, "frame").to_numpy(dtype=float)
    for idx, (scene_id, merged) in enumerate(merged_runs.items()):
        series = [
            ("baseline gpu_memory_mb", safe_numeric(merged, "gpu_memory_mb_baseline").to_numpy(dtype=float), "#7f7f7f"),
            (f"{scene_id} gpu_memory_mb", safe_numeric(merged, "gpu_memory_mb").to_numpy(dtype=float), color(idx)),
        ]
        svg_line_plot(
            args.output_dir / f"{scene_id}_vs_baseline_gpu_memory.svg",
            f"{scene_id} vs baseline GPU memory",
            base_frame[: len(merged)],
            series,
            "MB",
        )
        excess_series = [
            ("gpu_memory_excess_mb", safe_numeric(merged, "gpu_memory_excess_mb").to_numpy(dtype=float), "#d62728"),
            (
                "gpu_heaps_mb delta",
                (safe_numeric(merged, "gpu_heaps_mb") - safe_numeric(merged, "gpu_heaps_mb_baseline")).to_numpy(dtype=float),
                "#1f77b4",
            ),
            (
                "gpu_geometry_mb delta",
                (safe_numeric(merged, "gpu_geometry_mb") - safe_numeric(merged, "gpu_geometry_mb_baseline")).to_numpy(dtype=float),
                "#2ca02c",
            ),
            (
                "gpu_renderer_mb delta",
                (safe_numeric(merged, "gpu_renderer_mb") - safe_numeric(merged, "gpu_renderer_mb_baseline")).to_numpy(dtype=float),
                "#9467bd",
            ),
        ]
        svg_line_plot(
            args.output_dir / f"{scene_id}_excess_breakdown.svg",
            f"{scene_id} excess vs component deltas",
            base_frame[: len(merged)],
            excess_series,
            "MB",
        )

    report_lines = [
        "# Baseline Memory Excess Analysis",
        "",
        f"- Input root: `{args.input_root}`",
        f"- Baseline scene: `{args.baseline_scene}`",
        f"- Baseline strategy: `{baseline_run.strategy or 'unknown'}`",
        f"- Baseline CSV: `{baseline_run.metrics_csv}`",
        "",
        "## Main Ranking",
        "",
        "| Scene | Strategy | Peak excess MB | Mean excess MB | Frames above baseline | Strongest correlates |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary_df.iterrows():
        corr_parts = []
        for i in range(1, 4):
            key = row.get(f"top_correlation_{i}", "")
            val = row.get(f"top_correlation_{i}_value", np.nan)
            if isinstance(key, str) and key:
                corr_parts.append(f"{key} ({val:+.3f})")
        report_lines.append(
            f"| {row['scene_id']} | {row['strategy']} | {row['peak_excess_mb']:.1f} | {row['mean_excess_mb']:.1f} | {int(row['frames_above_baseline'])} | "
            + ", ".join(corr_parts)
            + " |"
        )

    report_lines.extend(
        [
            "",
            "## Peak-Frame Explanations",
            "",
        ]
    )
    for _, row in summary_df.iterrows():
        report_lines.extend(
            [
                f"### {row['scene_id']}",
                "",
                f"- Peak excess: `{row['peak_excess_mb']:.1f} MB` at frame `{int(row['peak_excess_frame'])}`",
                f"- Run GPU memory at peak: `{row['peak_gpu_memory_frame_value_mb']:.1f} MB` vs baseline `{row['peak_gpu_memory_baseline_frame_value_mb']:.1f} MB`",
                f"- Peak deltas: `gpu_heaps_mb {row['peak_delta_gpu_heaps_mb']:+.1f}`, `gpu_geometry_mb {row['peak_delta_gpu_geometry_mb']:+.1f}`, `resident_geometry_memory_mb {row['peak_delta_resident_geometry_memory_mb']:+.1f}`, `gpu_renderer_mb {row['peak_delta_gpu_renderer_mb']:+.1f}`, `gpu_textures_mb {row['peak_delta_gpu_textures_mb']:+.1f}`, `gpu_restir_mb {row['peak_delta_gpu_restir_mb']:+.1f}`, `scratch_memory_mb {row['peak_delta_scratch_memory_mb']:+.1f}`",
                f"- Churn at peak: `objects_onload_requested={row['peak_objects_onload_requested']:.0f}`, `objects_offload_requested={row['peak_objects_offload_requested']:.0f}`, `onload_requested_mb={row['peak_onload_requested_mb']:.1f}`, `offload_requested_mb={row['peak_offload_requested_mb']:.1f}`",
                "",
            ]
        )
    report_path = args.output_dir / "baseline_memory_excess_report.md"
    report_path.write_text("\n".join(report_lines))

    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote plots to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
