#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime.
    plt = None


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


DEFAULT_SIGNALS = [
    "objects_onload_requested",
    "objects_offload_requested",
    "onload_requested_mb",
    "offload_requested_mb",
    "blas_build_requests",
    "tlas_rebuilds",
    "tlas_refits",
]

DEFAULT_REGRESSION_FEATURES = [
    "objects_onload_requested",
    "objects_offload_requested",
    "tlas_rebuilds",
    "gpu_geometry_mb",
]

TIMING_REQUIRED_COLUMNS = {
    "cpu_ms": ["frame", "cpu_ms", *DEFAULT_SIGNALS],
    "gpu_ms": ["frame", "gpu_ms", *DEFAULT_SIGNALS],
}


def safe_corr(lhs: pd.Series, rhs: pd.Series) -> float:
    paired = pd.concat([lhs, rhs], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    if paired.iloc[:, 0].nunique() <= 1 or paired.iloc[:, 1].nunique() <= 1:
        return float("nan")
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        raise SystemExit("Missing required column: frame")
    return df.sort_values("frame").reset_index(drop=True)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def lag_profile(timing: pd.Series, signal: pd.Series, max_lag: int) -> list[float]:
    return [safe_corr(timing, signal.shift(lag)) for lag in range(max_lag + 1)]


def best_lag(profile: list[float]) -> tuple[int, float]:
    best_index = 0
    best_value = float("nan")
    for index, value in enumerate(profile):
        if pd.isna(value):
            continue
        if pd.isna(best_value) or abs(value) > abs(best_value):
            best_index = index
            best_value = value
    return best_index, best_value


def segment_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    on = df["objects_onload_requested"] > 0
    off = df["objects_offload_requested"] > 0
    return {
        "none": ~(on | off),
        "onload_only": on & ~off,
        "offload_only": ~on & off,
        "both": on & off,
    }


def describe_series(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
    }


def winsorize_upper(series: pd.Series, quantile: float) -> pd.Series:
    cap = float(series.quantile(quantile))
    return series.clip(upper=cap)


def trim_upper(series: pd.Series, quantile: float) -> pd.Series:
    cap = float(series.quantile(quantile))
    return series.loc[series <= cap]


def format_stat_block(label: str, stats: dict[str, float]) -> str:
    if stats["count"] == 0:
        return f"{label:12s}: n=0"
    return (
        f"{label:12s}: n={stats['count']:4d} "
        f"mean={stats['mean']:9.3f} median={stats['median']:9.3f} "
        f"p95={stats['p95']:9.3f} p99={stats['p99']:9.3f} std={stats['std']:9.3f}"
    )


def print_segment_stats(df: pd.DataFrame, timing_col: str) -> None:
    print(f"\n=== Segment Stats ({timing_col}) ===")
    for name, mask in segment_masks(df).items():
        print(format_stat_block(name, describe_series(df.loc[mask, timing_col])))


def print_outlier_stats(df: pd.DataFrame, timing_col: str, quantile: float) -> None:
    print(f"\n=== Outlier Handling ({timing_col}, top {(1.0 - quantile) * 100:.1f}% threshold) ===")
    raw = df[timing_col].dropna()
    winsorized = winsorize_upper(raw, quantile)
    trimmed = trim_upper(raw, quantile)
    print(format_stat_block("raw", describe_series(raw)))
    print(format_stat_block("winsorized", describe_series(winsorized)))
    print(format_stat_block("trimmed", describe_series(trimmed)))


def fit_regression(df: pd.DataFrame, target_col: str, features: list[str]) -> tuple[np.ndarray, float, int]:
    required = [target_col, *features]
    subset = df[required].dropna()
    if subset.empty:
        return np.array([]), float("nan"), 0

    y = subset[target_col].to_numpy(dtype=float)
    x = subset[features].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x), dtype=float), x])
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    prediction = x @ coefficients
    residual_sum = float(np.sum((y - prediction) ** 2))
    total_sum = float(np.sum((y - y.mean()) ** 2))
    if total_sum <= 0.0:
        r_squared = float("nan")
    else:
        r_squared = 1.0 - residual_sum / total_sum
    return coefficients, r_squared, len(subset)


def print_regression(df: pd.DataFrame, target_col: str, features: list[str]) -> None:
    coefficients, r_squared, sample_count = fit_regression(df, target_col, features)
    print(f"\n=== Regression Model ({target_col}) ===")
    if sample_count == 0 or coefficients.size == 0:
        print("insufficient data")
        return
    print(f"n={sample_count} R^2={r_squared: .4f}")
    print(f"intercept                 : {coefficients[0]: .4f}")
    for feature, value in zip(features, coefficients[1:]):
        print(f"{feature:25s}: {value: .4f}")


def print_lag_summary(df: pd.DataFrame, timing_col: str, signals: list[str], max_lag: int) -> dict[str, list[float]]:
    print(f"\n=== Same-frame Pearson Correlation with {timing_col} ===")
    profiles: dict[str, list[float]] = {}
    for signal in signals:
        profile = lag_profile(df[timing_col], df[signal], max_lag)
        profiles[signal] = profile
        print(f"{signal:28s}: {profile[0]: .4f}")

    print(f"\n=== Lagged Correlation: corr({timing_col}[t], signal[t-k]) ===")
    for signal in signals:
        lag, value = best_lag(profiles[signal])
        print(f"{signal:28s}: best={value: .4f} at lag={lag}")
    return profiles


def save_lag_profile_csv(
    profiles: dict[str, list[float]],
    output_path: Path,
    max_lag: int,
) -> None:
    payload = {"lag": list(range(max_lag + 1))}
    payload.update(profiles)
    pd.DataFrame(payload).to_csv(output_path, index=False)


def _save_svg_line_plot(
    profiles: dict[str, list[float]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    width = 1000
    height = 600
    margin_left = 80
    margin_right = 30
    margin_top = 60
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for profile in profiles.values() for value in profile if not pd.isna(value)]
    if not all_values:
        all_values = [0.0]
    min_y = min(min(all_values), 0.0)
    max_y = max(max(all_values), 0.0)
    if math.isclose(min_y, max_y):
        max_y += 1.0
        min_y -= 1.0
    padding = 0.05 * (max_y - min_y)
    min_y -= padding
    max_y += padding

    max_x = max((len(profile) - 1 for profile in profiles.values()), default=0)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    def map_x(index: int) -> float:
        if max_x <= 0:
            return margin_left + plot_width / 2.0
        return margin_left + (index / max_x) * plot_width

    def map_y(value: float) -> float:
        return margin_top + (max_y - value) / (max_y - min_y) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{_svg_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" stroke-width="1"/>',
        f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="16" font-family="Arial">{_svg_escape(x_label)}</text>',
        f'<text x="20" y="{height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 20 {height / 2:.1f})">{_svg_escape(y_label)}</text>',
    ]

    zero_y = map_y(0.0)
    lines.append(
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + plot_width}" y2="{zero_y:.2f}" stroke="#666" stroke-width="1" stroke-dasharray="4 4"/>'
    )

    for tick in range(max_x + 1):
        x = map_x(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 5}" stroke="black" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 22}" text-anchor="middle" font-size="12" font-family="Arial">{tick}</text>')

    for fraction in np.linspace(0.0, 1.0, 6):
        value = max_y - fraction * (max_y - min_y)
        y = map_y(value)
        lines.append(f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="black" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{value:.2f}</text>')

    legend_y = margin_top
    legend_x = margin_left + plot_width - 180
    for index, (label, profile) in enumerate(profiles.items()):
        color = colors[index % len(colors)]
        points = []
        for lag, value in enumerate(profile):
            if pd.isna(value):
                continue
            points.append(f"{map_x(lag):.2f},{map_y(value):.2f}")
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
            )
        lines.append(f'<rect x="{legend_x}" y="{legend_y + index * 20 - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 18}" y="{legend_y + index * 20}" font-size="12" font-family="Arial">{_svg_escape(label)}</text>'
        )

    lines.append("</svg>")
    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def save_lag_profile_plot(
    profiles: dict[str, list[float]],
    output_path: Path,
    title: str,
) -> Path:
    if plt is None:
        return _save_svg_line_plot(profiles, output_path, title, "Lag (frames)", "Pearson correlation")

    figure, axis = plt.subplots(figsize=(10, 6))
    for signal, profile in profiles.items():
        axis.plot(range(len(profile)), profile, marker="o", linewidth=1.8, label=signal)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axis.set_xlabel("Lag (frames)")
    axis.set_ylabel("Pearson correlation")
    axis.set_title(title)
    axis.set_xlim(0, max(len(next(iter(profiles.values()), [])) - 1, 0))
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def detect_temperature_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"(temp|temperature|thermal)", re.IGNORECASE)
    return [column for column in df.columns if pattern.search(column)]


def merge_temperature_data(
    df: pd.DataFrame,
    temperature_csv: str | None,
    key_column: str,
    temperature_column: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    if temperature_csv is None:
        return df, detect_temperature_columns(df)

    temperature_df = pd.read_csv(temperature_csv)
    if key_column not in temperature_df.columns:
        raise SystemExit(f"Temperature CSV missing key column: {key_column}")

    candidate_columns = [
        column
        for column in temperature_df.columns
        if column != key_column and pd.api.types.is_numeric_dtype(temperature_df[column])
    ]
    if temperature_column is not None:
        if temperature_column not in temperature_df.columns:
            raise SystemExit(f"Temperature CSV missing temperature column: {temperature_column}")
        candidate_columns = [temperature_column]
    elif not candidate_columns:
        raise SystemExit("Temperature CSV has no numeric temperature-like columns to merge")

    renamed = {
        column: column if column not in df.columns else f"temp_external_{column}"
        for column in candidate_columns
    }
    merged = df.merge(
        temperature_df[[key_column, *candidate_columns]].rename(columns=renamed),
        on=key_column,
        how="left",
    )
    return merged, list(renamed.values())


def print_temperature_analysis(
    df: pd.DataFrame,
    max_lag: int,
    explicit_temp_columns: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    temp_columns = list(dict.fromkeys([*(explicit_temp_columns or []), *detect_temperature_columns(df)]))
    if not temp_columns:
        print(
            "\n=== Temperature / Thermal Analysis ===\n"
            "No temperature or thermal columns found. "
            "Throttling analysis requires a temperature column in the metrics CSV "
            "or an external CSV merged with --temperature-csv."
        )
        return {}

    print("\n=== Temperature / Thermal Analysis ===")
    results: dict[str, dict[str, list[float]]] = {}
    for temp_column in temp_columns:
        results[temp_column] = {}
        for timing_col in ("cpu_ms", "gpu_ms"):
            if timing_col not in df.columns:
                continue
            profile = lag_profile(df[timing_col], df[temp_column], max_lag)
            lag, value = best_lag(profile)
            results[temp_column][timing_col] = profile
            print(
                f"{temp_column:28s} vs {timing_col:6s}: "
                f"same={profile[0]: .4f} best={value: .4f} at lag={lag}"
            )
    return results


def print_timing_relationships(df: pd.DataFrame, max_lag: int) -> dict[str, list[float]] | None:
    if "cpu_ms" not in df.columns or "gpu_ms" not in df.columns:
        return None

    print("\n=== CPU / GPU Timing Relationship ===")
    same_frame = safe_corr(df["cpu_ms"], df["gpu_ms"])
    diff_corr = safe_corr(df["cpu_ms"].diff(), df["gpu_ms"].diff())
    window = min(max_lag + 1, max(len(df) // 10, 4))
    rolling_cpu = df["cpu_ms"].rolling(window=window, min_periods=window).std()
    rolling_gpu = df["gpu_ms"].rolling(window=window, min_periods=window).std()
    rolling_corr = safe_corr(rolling_cpu, rolling_gpu)

    cpu_leads = lag_profile(df["cpu_ms"], df["gpu_ms"], max_lag)
    gpu_leads = lag_profile(df["gpu_ms"], df["cpu_ms"], max_lag)
    cpu_lag, cpu_value = best_lag(cpu_leads)
    gpu_lag, gpu_value = best_lag(gpu_leads)

    print(f"same-frame corr(cpu_ms, gpu_ms): {same_frame: .4f}")
    print(f"corr(diff(cpu_ms), diff(gpu_ms)): {diff_corr: .4f}")
    print(f"corr(rolling std, window={window}) : {rolling_corr: .4f}")
    print(f"best corr(cpu_ms[t], gpu_ms[t-k]): {cpu_value: .4f} at lag={cpu_lag}")
    print(f"best corr(gpu_ms[t], cpu_ms[t-k]): {gpu_value: .4f} at lag={gpu_lag}")

    return {
        "cpu_vs_gpu": cpu_leads,
        "gpu_vs_cpu": gpu_leads,
    }


def save_relationship_plot(profiles: dict[str, list[float]], output_path: Path, title: str) -> Path:
    if plt is None:
        return _save_svg_line_plot(profiles, output_path, title, "Lag (frames)", "Pearson correlation")
    figure, axis = plt.subplots(figsize=(10, 6))
    for label, profile in profiles.items():
        axis.plot(range(len(profile)), profile, marker="o", linewidth=1.8, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axis.set_xlabel("Lag (frames)")
    axis.set_ylabel("Pearson correlation")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def default_plot_dir(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.parent / f"{csv_path.stem}_analysis"


def sanitize_stem(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()


def analyze_single_timing(
    df: pd.DataFrame,
    csv_path: str | Path,
    timing_col: str,
    max_lag: int,
    outlier_quantile: float,
    plot_dir: str | Path | None = None,
    signals: list[str] | None = None,
) -> dict[str, list[float]]:
    ensure_columns(df, TIMING_REQUIRED_COLUMNS[timing_col])
    chosen_signals = [signal for signal in (signals or DEFAULT_SIGNALS) if signal in df.columns]
    output_dir = Path(plot_dir) if plot_dir is not None else default_plot_dir(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = print_lag_summary(df, timing_col, chosen_signals, max_lag)
    print_segment_stats(df, timing_col)
    print_outlier_stats(df, timing_col, outlier_quantile)

    regression_features = [feature for feature in DEFAULT_REGRESSION_FEATURES if feature in df.columns]
    print_regression(df, timing_col, regression_features)

    stem = sanitize_stem(timing_col)
    csv_output = output_dir / f"{stem}_lag_profile.csv"
    png_output = output_dir / f"{stem}_lag_profile.png"
    save_lag_profile_csv(profiles, csv_output, max_lag)
    actual_plot = save_lag_profile_plot(profiles, png_output, f"{timing_col} lag profile vs residency activity")
    print(f"Saved lag profile CSV: {csv_output}")
    print(f"Saved lag profile plot: {actual_plot}")
    return profiles


def analyze_combined(
    csv_path: str | Path,
    max_lag: int,
    outlier_quantile: float,
    plot_dir: str | Path | None = None,
    temperature_csv: str | None = None,
    temperature_key: str = "frame",
    temperature_column: str | None = None,
) -> None:
    df = load_metrics(csv_path)
    df, merged_temp_columns = merge_temperature_data(df, temperature_csv, temperature_key, temperature_column)
    output_dir = Path(plot_dir) if plot_dir is not None else default_plot_dir(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "cpu_ms" in df.columns:
        analyze_single_timing(df, csv_path, "cpu_ms", max_lag, outlier_quantile, output_dir)
    if "gpu_ms" in df.columns:
        analyze_single_timing(df, csv_path, "gpu_ms", max_lag, outlier_quantile, output_dir)

    relation_profiles = print_timing_relationships(df, max_lag)
    if relation_profiles:
        relation_plot = output_dir / "cpu_gpu_lag_relationship.png"
        relation_csv = output_dir / "cpu_gpu_lag_relationship.csv"
        actual_relation_plot = save_relationship_plot(relation_profiles, relation_plot, "CPU/GPU lag relationship")
        save_lag_profile_csv(relation_profiles, relation_csv, max_lag)
        print(f"Saved CPU/GPU relationship CSV: {relation_csv}")
        print(f"Saved CPU/GPU relationship plot: {actual_relation_plot}")

    temperature_profiles = print_temperature_analysis(df, max_lag, merged_temp_columns)
    for temp_column, profiles in temperature_profiles.items():
        if not profiles:
            continue
        stem = sanitize_stem(temp_column)
        temp_plot = output_dir / f"{stem}_timing_lag_relationship.png"
        temp_csv = output_dir / f"{stem}_timing_lag_relationship.csv"
        actual_temp_plot = save_relationship_plot(profiles, temp_plot, f"{temp_column} lag relationship")
        save_lag_profile_csv(profiles, temp_csv, max_lag)
        print(f"Saved temperature relationship CSV: {temp_csv}")
        print(f"Saved temperature relationship plot: {actual_temp_plot}")

    if temperature_csv is not None and merged_temp_columns:
        print("\nMerged temperature columns:")
        for column in merged_temp_columns:
            print(f"  - {column}")
