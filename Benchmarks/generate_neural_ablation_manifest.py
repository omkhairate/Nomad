#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a manageable per-clip object subset for CGVQM ablation runs "
            "from neural_object_features.csv."
        )
    )
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-visible", type=int, default=8)
    parser.add_argument("--top-hit", type=int, default=6)
    parser.add_argument("--top-toggle", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def weighted_candidate_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.35 * df["visible_frame_fraction"].fillna(0.0)
        + 0.20 * df["mean_visible_coverage"].fillna(0.0)
        + 0.20 * df["total_object_hits"].fillna(0.0).rank(pct=True)
        + 0.15 * df["total_object_rays_tested"].fillna(0.0).rank(pct=True)
        + 0.10 * df["toggle_count"].fillna(0.0).rank(pct=True)
    )


def choose_subset(group: pd.DataFrame, rng: random.Random, args: argparse.Namespace) -> pd.DataFrame:
    selected_ids: set[int] = set()

    def take_top(frame: pd.DataFrame, column: str, count: int) -> None:
        if count <= 0 or column not in frame.columns:
            return
        ordered = frame.sort_values(column, ascending=False)
        for object_id in ordered["object_id"].head(count):
            selected_ids.add(int(object_id))

    take_top(group, "visible_frame_fraction", args.top_visible)
    take_top(group, "total_object_hits", args.top_hit)
    take_top(group, "toggle_count", args.top_toggle)

    remaining = [int(v) for v in group["object_id"].tolist() if int(v) not in selected_ids]
    rng.shuffle(remaining)
    for object_id in remaining[: max(0, args.random_controls)]:
        selected_ids.add(object_id)

    subset = group[group["object_id"].isin(selected_ids)].copy()
    subset["selection_score"] = weighted_candidate_score(subset)
    subset = subset.sort_values(
        ["selection_score", "visible_frame_fraction", "total_object_hits"],
        ascending=False,
    )
    return subset


def main() -> int:
    args = parse_args()
    output = (
        args.output
        if args.output is not None
        else args.features_csv.with_name("neural_ablation_manifest.csv")
    )

    df = pd.read_csv(args.features_csv)
    for col in [
        "object_id",
        "clip_id",
        "visible_frame_fraction",
        "mean_visible_coverage",
        "total_object_hits",
        "total_object_rays_tested",
        "toggle_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    rng = random.Random(args.seed)
    picked = []
    group_cols = ["scene_variant", "strategy", "clip_id", "clip_start_frame", "clip_end_frame"]
    for _, group in df.groupby(group_cols, dropna=False):
        picked.append(choose_subset(group, rng, args))

    out_df = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()
    out_df.to_csv(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
