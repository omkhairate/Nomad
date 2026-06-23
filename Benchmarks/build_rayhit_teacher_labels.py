#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["scene_variant", "clip_id", "clip_start_frame", "clip_end_frame"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_session = (
        repo_root
        / "Benchmarks"
        / "neural_cgvqm_pipeline_bistro_v2"
        / "scene_bistro_test_v2_20260606_123721"
    )
    default_features = default_session / "features" / "scene_bistro_test_v2_20260606_123721" / "neural_object_features.csv"
    default_output = default_session / "active_learning" / "teacher" / "rayhit_teacher_labels.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Build a cheap rayhit-teacher dataset from neural_object_features.csv. "
            "The target is a clip-local rayhit importance score that bakes in a mild cost penalty, "
            "so the neural model learns a ranking that resembles a practical residency policy."
        )
    )
    parser.add_argument("--features-csv", type=Path, default=default_features)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--target-column",
        default="teacher_rayhit_percentile",
        choices=["teacher_rayhit_percentile", "teacher_rayhit_efficiency", "teacher_rayhit_raw"],
        help="Which derived teacher target to expose as the training target.",
    )
    parser.add_argument(
        "--cost-exponent",
        type=float,
        default=0.5,
        help="Penalty applied to estimated bytes when building the efficiency target. Default: sqrt(cost).",
    )
    parser.add_argument(
        "--visibility-weight",
        type=float,
        default=0.25,
        help="Bonus multiplier from visible_frame_fraction. Keeps visible objects slightly favored.",
    )
    parser.add_argument(
        "--hit-probability-weight",
        type=float,
        default=0.25,
        help="Bonus multiplier from mean_hit_probability. Helps teacher targets stay closer to what the rayhit policy values.",
    )
    parser.add_argument(
        "--min-bytes",
        type=float,
        default=1.0,
        help="Lower clamp for estimated object bytes before cost normalization.",
    )
    return parser.parse_args()


def clip_percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values), dtype=np.float64), index=values.index)
    ranks = values.rank(method="average", pct=True)
    return ranks.astype(np.float64)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.features_csv).copy()

    required = [
        "mean_object_rayhit_score",
        "visible_frame_fraction",
        "mean_hit_probability",
        "mean_estimated_object_bytes",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Features CSV is missing required columns: {missing}")

    for col in required + ["total_object_hits", "total_object_rays_tested", "object_id", "primitive_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rayhit_raw = np.maximum(df["mean_object_rayhit_score"].to_numpy(dtype=np.float64), 0.0)
    visibility_bonus = 1.0 + np.maximum(df["visible_frame_fraction"].to_numpy(dtype=np.float64), 0.0) * max(args.visibility_weight, 0.0)
    probability_bonus = 1.0 + np.maximum(df["mean_hit_probability"].to_numpy(dtype=np.float64), 0.0) * max(args.hit_probability_weight, 0.0)
    estimated_bytes = np.maximum(df["mean_estimated_object_bytes"].to_numpy(dtype=np.float64), max(args.min_bytes, 1.0e-6))
    cost_term = np.power(estimated_bytes, max(args.cost_exponent, 0.0))
    teacher_efficiency = (rayhit_raw * visibility_bonus * probability_bonus) / np.maximum(cost_term, 1.0e-6)

    df["teacher_rayhit_raw"] = rayhit_raw
    df["teacher_rayhit_efficiency"] = teacher_efficiency
    df["teacher_rayhit_percentile"] = (
        df.groupby(GROUP_COLUMNS, dropna=False)["teacher_rayhit_efficiency"]
        .transform(clip_percentile)
        .astype(np.float64)
    )
    df["target"] = df[args.target_column]
    df["teacher_target_column"] = args.target_column

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    summary = {
        "features_csv": str(args.features_csv.resolve()),
        "output": str(args.output.resolve()),
        "target_column": args.target_column,
        "row_count": int(len(df)),
        "clip_count": int(df[[c for c in GROUP_COLUMNS if c in df.columns]].drop_duplicates().shape[0]),
        "cost_exponent": args.cost_exponent,
        "visibility_weight": args.visibility_weight,
        "hit_probability_weight": args.hit_probability_weight,
        "target_min": float(df[args.target_column].min()),
        "target_max": float(df[args.target_column].max()),
        "target_mean": float(df[args.target_column].mean()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Rayhit teacher labels ready")
    print(f"  features csv: {args.features_csv}")
    print(f"  output: {args.output}")
    print(f"  target column: {args.target_column}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
