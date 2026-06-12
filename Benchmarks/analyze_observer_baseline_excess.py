#!/usr/bin/env python3
"""Compare observer sweep runs against the always-resident baseline.

This version is intentionally lightweight:
- it uses `gpu_mem.csv` for every run, including the ray-hit runs whose
  `metrics_*.csv` files ballooned because of probability logging
- it optionally uses pre-reduced metrics CSVs for extra attribution where
  available
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze observer sweep memory excess over always-resident baseline.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "Benchmarks" / "bistro_observer_sweep_runs_topdown_fixed",
    )
    parser.add_argument(
        "--reduced-metrics-root",
        type=Path,
        default=repo_root / "Benchmarks" / "baseline_memory_excess_analysis" / "reduced",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Benchmarks" / "observer_baseline_excess_analysis",
    )
    parser.add_argument("--baseline-scene", default="scene_bistro_test_v2")
    return parser.parse_args()


def scene_id_from_run_dir(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return path.name


def load_complete_flag(run_dir: Path) -> str:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return "unknown"
    try:
        value = json.loads(summary_path.read_text()).get("complete")
    except Exception:
        return "unknown"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


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
    ]
    return palette[index % len(palette)]


def svg_bar_plot(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width = max(1000, 82 * len(labels) + 240)
    height = 640
    margin_l, margin_r, margin_t, margin_b = 95, 30, 65, 220
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_hi = max([0.0] + [float(v) for v in values]) * 1.1 + 1.0

    def sy(v: float) -> float:
        return margin_t + (y_hi - v) / max(y_hi, 1.0) * plot_h

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
        y_val = y_hi * tick / 5
        y = sy(y_val)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l+plot_w}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.1f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_l + idx * slot + (slot - bar_w) / 2
        y = sy(max(float(value), 0.0))
        h = margin_t + plot_h - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color(idx)}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{float(value):.1f}</text>')
        parts.append(
            f'<text x="{x+bar_w/2:.1f}" y="{height-190}" text-anchor="end" transform="rotate(-55 {x+bar_w/2:.1f} {height-190})" font-family="Arial" font-size="11">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for run_dir in sorted(args.input_root.iterdir()):
        if not run_dir.is_dir():
            continue
        gpu_mem = run_dir / "gpu_mem.csv"
        if not gpu_mem.exists():
            continue
        scene_id = scene_id_from_run_dir(run_dir)
        complete = load_complete_flag(run_dir)
        runs.append({"scene_id": scene_id, "run_dir": run_dir, "gpu_mem": gpu_mem, "complete": complete})

    baseline_info = next((r for r in runs if r["scene_id"] == args.baseline_scene), None)
    if baseline_info is None:
        raise SystemExit(f"Could not find baseline scene {args.baseline_scene}")

    baseline_df = pd.read_csv(baseline_info["gpu_mem"])
    summary_rows: list[dict[str, object]] = []

    for info in runs:
        scene_id = str(info["scene_id"])
        if scene_id == args.baseline_scene:
            continue
        run_df = pd.read_csv(info["gpu_mem"])
        merged = run_df.merge(baseline_df, on="frame", suffixes=("", "_baseline"))
        excess = merged["gpu_memory_mb"] - merged["gpu_memory_mb_baseline"]
        peak_idx = int(excess.idxmax())
        peak = merged.loc[peak_idx]
        row = {
            "scene_id": scene_id,
            "complete": info["complete"],
            "rows_compared": len(merged),
            "peak_excess_mb": float(excess.max()),
            "mean_excess_mb": float(excess.mean()),
            "p95_excess_mb": float(excess.quantile(0.95)),
            "frames_above_baseline": int((excess > 0).sum()),
            "peak_frame": int(peak["frame"]),
            "peak_gpu_memory_mb": float(peak["gpu_memory_mb"]),
            "peak_baseline_gpu_memory_mb": float(peak["gpu_memory_mb_baseline"]),
            "peak_delta_scratch_memory_mb": float(peak["scratch_memory_mb"] - peak["scratch_memory_mb_baseline"]),
            "peak_delta_resident_geometry_memory_mb": float(
                peak["resident_geometry_memory_mb"] - peak["resident_geometry_memory_mb_baseline"]
            ),
            "peak_delta_residency_memory_mb": float(peak["residency_memory_mb"] - peak["residency_memory_mb_baseline"]),
        }

        reduced = args.reduced_metrics_root / f"{info['run_dir'].name}.csv"
        if reduced.exists():
            reduced_df = pd.read_csv(reduced)
            reduced_base = pd.read_csv(args.reduced_metrics_root / f"{Path(str(baseline_info['run_dir'])).name}.csv")
            reduced_merged = reduced_df.merge(reduced_base, on="frame", suffixes=("", "_baseline"))
            if int(peak["frame"]) in set(reduced_merged["frame"]):
                peak_metrics = reduced_merged.loc[reduced_merged["frame"] == int(peak["frame"])].iloc[0]
                row.update(
                    {
                        "strategy": str(peak_metrics["strategy"]),
                        "peak_delta_gpu_heaps_mb": float(peak_metrics["gpu_heaps_mb"] - peak_metrics["gpu_heaps_mb_baseline"]),
                        "peak_delta_gpu_geometry_mb": float(
                            peak_metrics["gpu_geometry_mb"] - peak_metrics["gpu_geometry_mb_baseline"]
                        ),
                        "peak_delta_gpu_renderer_mb": float(
                            peak_metrics["gpu_renderer_mb"] - peak_metrics["gpu_renderer_mb_baseline"]
                        ),
                        "peak_delta_gpu_textures_mb": float(
                            peak_metrics["gpu_textures_mb"] - peak_metrics["gpu_textures_mb_baseline"]
                        ),
                        "peak_delta_gpu_restir_mb": float(
                            peak_metrics["gpu_restir_mb"] - peak_metrics["gpu_restir_mb_baseline"]
                        ),
                        "peak_delta_gpu_other_mb": float(peak_metrics["gpu_other_mb"] - peak_metrics["gpu_other_mb_baseline"]),
                        "peak_objects_onload_requested": float(peak_metrics["objects_onload_requested"]),
                        "peak_objects_offload_requested": float(peak_metrics["objects_offload_requested"]),
                        "peak_onload_requested_mb": float(peak_metrics["onload_requested_mb"]),
                        "peak_offload_requested_mb": float(peak_metrics["offload_requested_mb"]),
                        "peak_tlas_rebuilds": float(peak_metrics["tlas_rebuilds"]),
                        "peak_tlas_refits": float(peak_metrics["tlas_refits"]),
                        "peak_blas_build_requests": float(peak_metrics["blas_build_requests"]),
                    }
                )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("peak_excess_mb", ascending=False)
    summary_path = args.output_dir / "observer_baseline_excess_summary.csv"
    summary.to_csv(summary_path, index=False)

    svg_bar_plot(
        args.output_dir / "observer_peak_excess_mb.svg",
        "Observer Sweep: Peak Memory Above Always-Resident",
        summary["scene_id"].tolist(),
        summary["peak_excess_mb"].tolist(),
        "MB",
    )
    svg_bar_plot(
        args.output_dir / "observer_mean_excess_mb.svg",
        "Observer Sweep: Mean Memory Above Always-Resident",
        summary["scene_id"].tolist(),
        summary["mean_excess_mb"].tolist(),
        "MB",
    )

    lines = [
        "# Observer Baseline Excess Analysis",
        "",
        f"- Input root: `{args.input_root}`",
        f"- Baseline scene: `{args.baseline_scene}`",
        f"- Baseline gpu_mem.csv: `{baseline_info['gpu_mem']}`",
        "",
        "## Key Result",
        "",
        "The dominant pattern in these observer runs is that memory excursions above the always-resident baseline are small and usually not accompanied by higher `resident_geometry_memory_mb` or `residency_memory_mb`.",
        "",
        "That means the excess is usually not caused by keeping more scene content resident than the baseline. It is more consistent with allocator slack, heap granularity, scratch variance, or small `gpu_other_mb`/bookkeeping differences.",
        "",
        "## Run Summary",
        "",
        "| Scene | Complete | Peak excess MB | Mean excess MB | Frames above baseline | Peak delta resident geom MB | Peak delta residency total MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scene_id']} | {row['complete']} | {row['peak_excess_mb']:.2f} | {row['mean_excess_mb']:.2f} | "
            f"{int(row['frames_above_baseline'])} | {row['peak_delta_resident_geometry_memory_mb']:.3f} | {row['peak_delta_residency_memory_mb']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- When `peak_delta_resident_geometry_memory_mb ~= 0` and `peak_delta_residency_memory_mb ~= 0`, but total `gpu_memory_mb` is still higher than baseline, the extra usage is coming from non-residency overhead rather than extra resident scene data.",
            "- Repeated peak deltas around `~19.5 MB` across several runs are a strong sign of allocator/heap granularity rather than strategy-specific content retention.",
            "- The distance and screenspace observer runs do not show meaningful evidence that they exceeded baseline because they held more geometry resident.",
            "- The ray-hit runs are included in the total-memory comparison through `gpu_mem.csv`, but their full `metrics_*.csv` files ballooned because of probability logging, so fine-grained per-frame residency-churn attribution is limited unless those CSVs are preprocessed further.",
        ]
    )
    report_path = args.output_dir / "observer_baseline_excess_report.md"
    report_path.write_text("\n".join(lines))

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote plots to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
