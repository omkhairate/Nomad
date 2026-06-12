#!/usr/bin/env python3
"""Analyze Bistro sweep CSVs and generate lightweight SVG plots.

The script intentionally avoids matplotlib so it works in the local environment
with only pandas/numpy available.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


@dataclass(frozen=True)
class RunCsv:
    scene_id: str
    timestamp: str
    run_dir: Path
    metrics_csv: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Benchmarks" / "bistro_sweep_runs"
    default_out = repo_root / "Benchmarks" / "bistro_sweep_analysis"
    parser = argparse.ArgumentParser(description="Analyze Bistro sweep metrics and write SVG plots.")
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--all-runs", action="store_true", help="Analyze every metrics CSV instead of the latest one per scene.")
    parser.add_argument("--top-events", type=int, default=12)
    return parser.parse_args()


def parse_scene_id(run_dir: Path) -> tuple[str, str]:
    match = TIMESTAMP_RE.search(run_dir.name)
    if not match:
        return run_dir.name, ""
    return run_dir.name[: match.start()], match.group(1)


def discover_runs(input_root: Path, all_runs: bool) -> list[RunCsv]:
    runs: list[RunCsv] = []
    for metrics in sorted(input_root.glob("*/metrics_*.csv")):
        scene_id, timestamp = parse_scene_id(metrics.parent)
        runs.append(RunCsv(scene_id, timestamp, metrics.parent, metrics))
    if all_runs:
        return runs
    latest: dict[str, RunCsv] = {}
    for run in runs:
        previous = latest.get(run.scene_id)
        if previous is None or run.timestamp > previous.timestamp:
            latest[run.scene_id] = run
    return [latest[key] for key in sorted(latest)]


def numeric_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col in {"strategy", "environment_depth_weights", "environment_depth_radii"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


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
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return palette[index % len(palette)]


def svg_line_plot(
    path: Path,
    title: str,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    width: int = 1180,
    height: int = 620,
    y_label: str = "MB",
) -> None:
    margin_l, margin_r, margin_t, margin_b = 80, 260, 60, 70
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
        x_val = x_lo + (x_hi - x_lo) * tick / 5
        y_val = y_lo + (y_hi - y_lo) * tick / 5
        parts.append(f'<line x1="{sx(x_val):.1f}" y1="{margin_t}" x2="{sx(x_val):.1f}" y2="{margin_t+plot_h}" stroke="#eee"/>')
        parts.append(f'<text x="{sx(x_val):.1f}" y="{margin_t+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">{x_val:.0f}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{sy(y_val):.1f}" x2="{margin_l+plot_w}" y2="{sy(y_val):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{sy(y_val)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.0f}</text>')

    for idx, (name, vals, stroke) in enumerate(series):
        points = []
        for xv, yv in zip(x, vals):
            if np.isfinite(xv) and np.isfinite(yv):
                points.append(f'{sx(float(xv)):.1f},{sy(float(yv)):.1f}')
        if len(points) >= 2:
            dash = ' stroke-dasharray="7 5"' if 'cap' in name.lower() else ''
            parts.append(f'<polyline fill="none" stroke="{stroke}" stroke-width="2"{dash} points="' + ' '.join(points) + '"/>')
        ly = margin_t + 24 + idx * 22
        parts.append(f'<line x1="{margin_l+plot_w+28}" y1="{ly-4}" x2="{margin_l+plot_w+55}" y2="{ly-4}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(f'<text x="{margin_l+plot_w+62}" y="{ly}" font-family="Arial" font-size="13">{name}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def svg_bar_plot(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width = max(900, 65 * len(labels) + 220)
    height = 620
    margin_l, margin_r, margin_t, margin_b = 85, 30, 65, 190
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_lo, y_hi = finite_minmax([0.0] + values)
    y_lo = min(0.0, y_lo)

    def sy(v: float) -> float:
        return margin_t + (y_hi - v) / (y_hi - y_lo) * plot_h

    bar_w = plot_w / max(1, len(labels)) * 0.72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<text x="22" y="{margin_t+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {margin_t+plot_h/2})" font-family="Arial" font-size="14">{ylabel}</text>',
    ]
    for tick in range(6):
        y_val = y_lo + (y_hi - y_lo) * tick / 5
        parts.append(f'<line x1="{margin_l}" y1="{sy(y_val):.1f}" x2="{margin_l+plot_w}" y2="{sy(y_val):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{sy(y_val)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.0f}</text>')
    slot = plot_w / max(1, len(labels))
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin_l + i * slot + (slot - bar_w) / 2
        y = sy(max(value, 0.0))
        h = margin_t + plot_h - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color(i)}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.0f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-170}" text-anchor="end" transform="rotate(-55 {x+bar_w/2:.1f} {height-170})" font-family="Arial" font-size="11">{label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


THERMAL_PRESSURE_LEVELS = {
    "nominal": 0.0,
    "fair": 1.0,
    "serious": 2.0,
    "critical": 3.0,
}


def merge_thermal_samples(run: RunCsv, df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    thermal_csv = run.run_dir / "thermal.csv"
    if not thermal_csv.is_file():
        return df, False, "missing"
    try:
        thermal = pd.read_csv(thermal_csv)
    except Exception as exc:
        return df, False, f"read_error:{exc}"
    if thermal.empty:
        return df, False, "empty"
    if "error" in thermal.columns and thermal["error"].fillna("").astype(str).str.len().gt(0).all():
        return df, False, "error:" + str(thermal["error"].dropna().iloc[0])
    if "run_elapsed_seconds" not in thermal.columns or "wall_seconds" not in df.columns:
        return df, False, "missing_time_columns"

    merged = df.copy()
    metrics_time = pd.to_numeric(merged["wall_seconds"], errors="coerce")
    merged["thermal_join_seconds"] = metrics_time - metrics_time.min()
    thermal = thermal.copy()
    thermal["thermal_join_seconds"] = pd.to_numeric(thermal["run_elapsed_seconds"], errors="coerce")
    thermal = thermal.dropna(subset=["thermal_join_seconds"]).sort_values("thermal_join_seconds")
    if thermal.empty:
        return df, False, "no_valid_samples"

    keep_cols = ["thermal_join_seconds"]
    for col in ["thermal_pressure", "cpu_power_mw", "gpu_power_mw", "ane_power_mw", "combined_power_mw", "cpu_frequency_mhz", "gpu_frequency_mhz", "ane_frequency_mhz"]:
        if col in thermal.columns:
            keep_cols.append(col)
    thermal = thermal[keep_cols]
    aligned = pd.merge_asof(
        merged.sort_values("thermal_join_seconds"),
        thermal,
        on="thermal_join_seconds",
        direction="nearest",
        tolerance=5.0,
    ).sort_index()
    if "thermal_pressure" in aligned.columns:
        aligned["thermal_pressure_level"] = (
            aligned["thermal_pressure"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map(THERMAL_PRESSURE_LEVELS)
        )
    return aligned, True, "ok"


def summarize_run(run: RunCsv, df: pd.DataFrame, top_events: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    total_mem = safe_series(df, "gpu_memory_mb")
    geom = safe_series(df, "gpu_geometry_mb")
    heaps = safe_series(df, "gpu_heaps_mb")
    resident_geom = safe_series(df, "resident_geometry_memory_mb")
    textures = safe_series(df, "gpu_textures_mb")
    cpu = safe_series(df, "cpu_ms")
    gpu = safe_series(df, "gpu_ms")
    onload = safe_series(df, "objects_onload_requested")
    offload = safe_series(df, "objects_offload_requested")
    cap = safe_series(df, "total_memory_cap_mb")

    deltas = total_mem.diff().fillna(0)
    spike_threshold = max(50.0, float(deltas.std(skipna=True) * 3.0))
    spike_frames = df.loc[deltas.abs() >= spike_threshold, "frame"] if "frame" in df else pd.Series([], dtype=float)

    summary = {
        "scene_id": run.scene_id,
        "timestamp": run.timestamp,
        "metrics_csv": str(run.metrics_csv),
        "frames": len(df),
        "strategy": str(df["strategy"].dropna().iloc[0]) if "strategy" in df and not df["strategy"].dropna().empty else run.scene_id,
        "peak_gpu_memory_mb": float(total_mem.max()),
        "median_gpu_memory_mb": float(total_mem.median()),
        "final_gpu_memory_mb": float(total_mem.iloc[-1]) if len(total_mem) else np.nan,
        "peak_gpu_geometry_mb": float(geom.max()),
        "peak_gpu_heaps_mb": float(heaps.max()),
        "peak_resident_geometry_mb": float(resident_geom.max()),
        "peak_gpu_textures_mb": float(textures.max()),
        "p95_cpu_ms": float(cpu.quantile(0.95)),
        "p95_gpu_ms": float(gpu.quantile(0.95)),
        "total_onload_requested": float(onload.sum()),
        "total_offload_requested": float(offload.sum()),
        "spike_threshold_mb": spike_threshold,
        "spike_or_drop_count": int(len(spike_frames)),
        "cap_mb": float(cap.max()) if cap.notna().any() else np.nan,
    }

    component_cols = [
        "gpu_geometry_mb", "gpu_heaps_mb", "resident_geometry_memory_mb",
        "gpu_textures_mb", "gpu_scratch_mb", "gpu_staging_mb", "gpu_other_mb",
        "objects_onload_requested", "objects_offload_requested", "tlas_rebuilds",
        "blas_build_requests", "active_objects", "resident_objects",
    ]
    events = []
    top_delta_indices = deltas.abs().sort_values(ascending=False).head(top_events).index
    for idx in top_delta_indices:
        row = df.loc[idx]
        event = {
            "scene_id": run.scene_id,
            "frame": int(row.get("frame", idx)),
            "wall_seconds": float(row.get("wall_seconds", np.nan)),
            "gpu_memory_mb": float(total_mem.loc[idx]),
            "gpu_memory_delta_mb": float(deltas.loc[idx]),
        }
        for col in component_cols:
            if col in df.columns:
                values = safe_series(df, col)
                event[col] = float(values.loc[idx])
                event[f"delta_{col}"] = float(values.diff().fillna(0).loc[idx])
        events.append(event)
    return summary, events




def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No rows."
    view = df[columns].copy()
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                if np.isnan(value):
                    cells.append("nan")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def corr_table(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict[str, float]:
    if target_col not in df.columns:
        return {}
    target = pd.to_numeric(df[target_col], errors="coerce")
    result = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
            result[col] = np.nan
        else:
            result[col] = float(target.corr(values))
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = args.output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.input_root, args.all_runs)
    if not runs:
        print(f"No metrics CSVs found under {args.input_root}", flush=True)
        return 2

    summaries: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    temp_columns_seen: set[str] = set()

    feature_cols = [
        "objects_onload_requested", "objects_offload_requested",
        "onload_requested_mb", "offload_requested_mb",
        "blas_build_requests", "tlas_rebuilds", "tlas_refits",
        "active_objects", "resident_objects", "active_nodes", "resident_nodes",
        "gpu_geometry_mb", "gpu_heaps_mb", "resident_geometry_memory_mb",
        "gpu_textures_mb", "gpu_staging_mb", "camera_motion_metric",
        "over_memory_cap", "total_over_memory_cap", "total_memory_cap_denials",
        "thermal_pressure_level", "cpu_power_mw", "gpu_power_mw", "combined_power_mw",
        "cpu_frequency_mhz", "gpu_frequency_mhz",
    ]

    for run in runs:
        df = numeric_df(run.metrics_csv)
        df, thermal_available, thermal_status = merge_thermal_samples(run, df)
        temp_columns_seen.update(
            col for col in df.columns
            if any(token in col.lower() for token in ["temp", "thermal", "fan", "power"])
        )
        summary, events = summarize_run(run, df, args.top_events)
        summary["thermal_available"] = thermal_available
        summary["thermal_status"] = thermal_status
        summaries.append(summary)
        all_events.extend(events)

        x = safe_series(df, "frame").to_numpy()
        cap = safe_series(df, "total_memory_cap_mb").replace(0, np.nan).to_numpy()
        series = [
            ("total gpu", safe_series(df, "gpu_memory_mb").to_numpy(), "#111111"),
            ("geometry", safe_series(df, "gpu_geometry_mb").to_numpy(), "#1f77b4"),
            ("resident geometry", safe_series(df, "resident_geometry_memory_mb").to_numpy(), "#2ca02c"),
            ("heaps", safe_series(df, "gpu_heaps_mb").to_numpy(), "#ff7f0e"),
            ("textures", safe_series(df, "gpu_textures_mb").to_numpy(), "#9467bd"),
            ("total cap", cap, "#d62728"),
        ]
        svg_line_plot(
            per_run_dir / f"{run.scene_id}_memory.svg",
            f"{run.scene_id}: memory components",
            x,
            series,
        )
        event_series = [
            ("onload count", safe_series(df, "objects_onload_requested").to_numpy(), "#2ca02c"),
            ("offload count", safe_series(df, "objects_offload_requested").to_numpy(), "#d62728"),
            ("TLAS rebuilds", safe_series(df, "tlas_rebuilds").to_numpy(), "#1f77b4"),
            ("BLAS builds", safe_series(df, "blas_build_requests").to_numpy(), "#ff7f0e"),
        ]
        svg_line_plot(
            per_run_dir / f"{run.scene_id}_events.svg",
            f"{run.scene_id}: residency events",
            x,
            event_series,
            y_label="count",
        )

        corrs = corr_table(df, "gpu_memory_mb", feature_cols)
        for feature, value in corrs.items():
            corr_rows.append({"scene_id": run.scene_id, "target": "gpu_memory_mb", "feature": feature, "pearson": value})
        delta_df = df.copy()
        delta_df["delta_gpu_memory_mb"] = safe_series(df, "gpu_memory_mb").diff()
        for col in feature_cols:
            if col in df.columns:
                delta_df[f"delta_{col}"] = safe_series(df, col).diff()
        delta_features = [f"delta_{col}" for col in feature_cols if f"delta_{col}" in delta_df.columns]
        delta_corrs = corr_table(delta_df, "delta_gpu_memory_mb", delta_features)
        for feature, value in delta_corrs.items():
            corr_rows.append({"scene_id": run.scene_id, "target": "delta_gpu_memory_mb", "feature": feature, "pearson": value})

    summary_df = pd.DataFrame(summaries).sort_values("scene_id")
    events_df = pd.DataFrame(all_events)
    corr_df = pd.DataFrame(corr_rows)

    summary_csv = args.output_dir / "sweep_summary.csv"
    events_csv = args.output_dir / "top_memory_spikes_and_drops.csv"
    corr_csv = args.output_dir / "memory_correlations.csv"
    summary_df.to_csv(summary_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)

    labels = summary_df["scene_id"].astype(str).tolist()
    svg_bar_plot(args.output_dir / "summary_peak_gpu_memory.svg", "Peak total GPU memory by run", labels, summary_df["peak_gpu_memory_mb"].tolist(), "MB")
    svg_bar_plot(args.output_dir / "summary_spike_drop_counts.svg", "Detected memory spike/drop frames", labels, summary_df["spike_or_drop_count"].tolist(), "count")
    svg_bar_plot(args.output_dir / "summary_total_offloads.svg", "Total offload requests by run", labels, summary_df["total_offload_requested"].tolist(), "objects")
    svg_bar_plot(args.output_dir / "summary_p95_gpu_ms.svg", "P95 GPU timing by run", labels, summary_df["p95_gpu_ms"].tolist(), "ms")
    svg_bar_plot(args.output_dir / "summary_p95_cpu_ms.svg", "P95 CPU timing by run", labels, summary_df["p95_cpu_ms"].tolist(), "ms")

    report = args.output_dir / "analysis_notes.md"
    biggest = summary_df.sort_values("peak_gpu_memory_mb", ascending=False).head(6)
    most_spiky = summary_df.sort_values("spike_or_drop_count", ascending=False).head(6)
    event_top = events_df.reindex(events_df["gpu_memory_delta_mb"].abs().sort_values(ascending=False).index).head(12) if not events_df.empty else events_df
    thermal_count = int(summary_df.get("thermal_available", pd.Series(dtype=bool)).fillna(False).sum())
    thermal_status_series = summary_df.get("thermal_status", pd.Series(dtype=str)).fillna("").astype(str)
    error_statuses = sorted({value for value in thermal_status_series if value.startswith("error:")})
    temp_note = (
        "Thermal CSV files were present, but no valid samples were merged. "
        + (f"Observed logger errors: {', '.join(error_statuses)}. " if error_statuses else "")
        + "So direct temperature/thermal correlation is not possible from these runs."
        if thermal_count == 0 else
        f"Thermal logger samples were merged for {thermal_count} run(s). Columns found: {', '.join(sorted(temp_columns_seen))}."
    )
    report.write_text(
        "# Bistro Sweep Analysis\n\n"
        f"Input root: `{args.input_root}`\n\n"
        f"Runs analyzed: `{len(runs)}`\n\n"
        "## Temperature Correlation\n\n"
        f"{temp_note}\n\n"
        "## Highest Peak GPU Memory\n\n"
        + markdown_table(biggest, ["scene_id", "peak_gpu_memory_mb", "peak_gpu_geometry_mb", "peak_gpu_heaps_mb", "peak_resident_geometry_mb", "total_offload_requested"])
        + "\n\n## Most Spike/Drop Frames\n\n"
        + markdown_table(most_spiky, ["scene_id", "spike_or_drop_count", "spike_threshold_mb", "peak_gpu_memory_mb", "total_onload_requested", "total_offload_requested"])
        + "\n\n## Largest Single-Frame Memory Changes\n\n"
        + (markdown_table(event_top, ["scene_id", "frame", "wall_seconds", "gpu_memory_mb", "gpu_memory_delta_mb", "delta_gpu_geometry_mb", "delta_gpu_heaps_mb", "delta_resident_geometry_memory_mb", "objects_onload_requested", "objects_offload_requested", "tlas_rebuilds"]) if not event_top.empty else "No events detected.")
        + "\n\n## Output Files\n\n"
        f"- Summary CSV: `{summary_csv}`\n"
        f"- Top spikes/drops CSV: `{events_csv}`\n"
        f"- Correlations CSV: `{corr_csv}`\n"
        f"- Per-run plots: `{per_run_dir}`\n"
    )

    print(f"Analyzed {len(runs)} run(s).")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {events_csv}")
    print(f"Wrote: {corr_csv}")
    print(f"Wrote plots under: {args.output_dir}")
    print(f"Wrote notes: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
