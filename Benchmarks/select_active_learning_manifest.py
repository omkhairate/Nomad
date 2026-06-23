#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from train_unified_neural_baseline import V1_FEATURES, fit_ridge, pick_all_nonconstant_numeric_features
from train_unified_neural_mlp import TinyMLP, set_seed as set_torch_seed


KEY_COLUMNS = ["scene_variant", "strategy", "clip_id", "object_id"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_session = (
        repo_root
        / "Benchmarks"
        / "neural_cgvqm_pipeline_bistro_v2"
        / "scene_bistro_test_v2_20260606_123721"
    )
    default_features = default_session / "features" / "scene_bistro_test_v2_20260606_123721" / "neural_object_features.csv"
    default_labels = default_session / "cgvqm" / "neural_training_labels_50.csv"
    default_output = default_session / "features" / "scene_bistro_test_v2_20260606_123721" / "neural_ablation_manifest_active_learning.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Score unlabeled object/clip candidates for the next CGVQM ablation round. "
            "The selector learns from existing labels, estimates uncertainty and "
            "model disagreement, then emits a manifest compatible with run_neural_ablation_batch.py."
        )
    )
    parser.add_argument("--features-csv", type=Path, default=default_features)
    parser.add_argument("--labels-csv", type=Path, default=default_labels)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--candidates-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--target", default="delta_cgvqm")
    parser.add_argument("--feature-set", choices=["v1", "all_nonconstant"], default="v1")
    parser.add_argument("--group-column", default="clip_id")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--per-clip-limit", type=int, default=10)
    parser.add_argument("--committee-seeds", default="7,17,27,37,47")
    parser.add_argument("--hidden-dims", default="16,8")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--log1p-target", action="store_true", default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization for the linear reference model.")
    parser.add_argument("--uncertainty-weight", type=float, default=0.45)
    parser.add_argument("--disagreement-weight", type=float, default=0.25)
    parser.add_argument("--impact-weight", type=float, default=0.20)
    parser.add_argument("--visibility-weight", type=float, default=0.10)
    parser.add_argument("--min-visible-fraction", type=float, default=0.0)
    parser.add_argument("--min-total-hits", type=float, default=0.0)
    parser.add_argument(
        "--drop-empty-candidates",
        action="store_true",
        default=True,
        help="Drop candidates with no visibility, no hits, and no rays tested. Default: on",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Include already labeled rows in the scored candidate table. They are still excluded from the final manifest unless explicitly requested.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def standardize_train_eval(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_eval - mean) / std, mean, std


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_key_strings(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in KEY_COLUMNS:
        if col in df.columns:
            series = df[col].astype(str)
        else:
            series = pd.Series([""] * len(df), index=df.index, dtype=str)
        parts.append(series)
    out = parts[0]
    for series in parts[1:]:
        out = out + "|" + series
    return out


def choose_features(df: pd.DataFrame, feature_set: str, target: str, group_col: str) -> list[str]:
    if feature_set == "v1":
        return [feature for feature in V1_FEATURES if feature in df.columns]
    return pick_all_nonconstant_numeric_features(df, target, group_col)


def prepare_xy(df: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return x, y


def bootstrap_indices(count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 1:
        return np.arange(count)
    return rng.integers(0, count, size=count)


def split_fit_validation(indices: np.ndarray, rng: np.random.Generator, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) <= 5:
        return indices, indices
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_idx = shuffled[:val_count]
    fit_idx = shuffled[val_count:]
    if len(fit_idx) == 0:
        fit_idx = val_idx
    return fit_idx, val_idx


def train_mlp_committee(
    x_train: np.ndarray,
    y_train_true: np.ndarray,
    x_eval: np.ndarray,
    *,
    seeds: Iterable[int],
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    log1p_target: bool,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    from train_unified_neural_mlp import train_one_fold

    x_train_std, x_eval_std, _, _ = standardize_train_eval(x_train, x_eval)
    if log1p_target:
        y_train_fit = np.log1p(np.maximum(y_train_true, 0.0))
    else:
        y_train_fit = y_train_true.copy()

    predictions = []
    member_info: list[dict[str, float]] = []
    base_indices = np.arange(len(x_train_std))

    for seed in seeds:
        set_torch_seed(seed)
        rng = np.random.default_rng(seed)
        boot_idx = bootstrap_indices(len(base_indices), rng)
        fit_idx, val_idx = split_fit_validation(boot_idx, rng)
        model, train_info = train_one_fold(
            x_fit=x_train_std[fit_idx],
            y_fit=y_train_fit[fit_idx],
            x_val=x_train_std[val_idx],
            y_val=y_train_fit[val_idx],
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            x_eval_t = torch.tensor(x_eval_std, dtype=torch.float32, device=device)
            pred_fit = model(x_eval_t).detach().cpu().numpy()
        y_pred = np.expm1(pred_fit) if log1p_target else pred_fit
        y_pred = np.maximum(y_pred, 0.0)
        predictions.append(y_pred)
        member_info.append(
            {
                "seed": float(seed),
                "best_epoch": float(train_info["best_epoch"]),
                "best_val_loss": float(train_info["best_val_loss"]),
            }
        )

    return np.vstack(predictions), member_info


def train_linear_reference(
    x_train: np.ndarray,
    y_train_true: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float,
    log1p_target: bool,
) -> np.ndarray:
    x_train_std, x_eval_std, _, _ = standardize_train_eval(x_train, x_eval)
    y_fit = np.log1p(np.maximum(y_train_true, 0.0)) if log1p_target else y_train_true.copy()
    weights, intercept = fit_ridge(x_train_std, y_fit, alpha)
    pred_fit = x_eval_std @ weights + intercept
    y_pred = np.expm1(pred_fit) if log1p_target else pred_fit
    return np.maximum(y_pred, 0.0)


def safe_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(numeric) == 0:
        return numeric
    return numeric.rank(pct=True, method="average")


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(numeric) == 0:
        return numeric
    lo = float(numeric.min())
    hi = float(numeric.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(numeric), dtype=np.float64), index=series.index)
    return (numeric - lo) / (hi - lo)


def score_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    scored = df.copy()
    scored["uncertainty_norm"] = normalize_series(scored["pred_std"])
    scored["disagreement_norm"] = normalize_series(scored["linear_mlp_gap"])
    scored["impact_norm"] = normalize_series(scored["pred_mean"])
    visibility_prior = (
        0.60 * safe_rank(scored["visible_frame_fraction"])
        + 0.20 * safe_rank(scored["total_object_hits"])
        + 0.10 * safe_rank(scored["total_object_rays_tested"])
        + 0.10 * safe_rank(scored["mean_visible_coverage"])
    )
    scored["visibility_prior"] = visibility_prior
    scored["active_learning_score"] = (
        args.uncertainty_weight * scored["uncertainty_norm"]
        + args.disagreement_weight * scored["disagreement_norm"]
        + args.impact_weight * scored["impact_norm"]
        + args.visibility_weight * scored["visibility_prior"]
    )
    return scored


def filter_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if args.min_visible_fraction > 0.0 and "visible_frame_fraction" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["visible_frame_fraction"], errors="coerce").fillna(0.0) >= args.min_visible_fraction]
    if args.min_total_hits > 0.0 and "total_object_hits" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["total_object_hits"], errors="coerce").fillna(0.0) >= args.min_total_hits]
    if args.drop_empty_candidates:
        vis = pd.to_numeric(filtered.get("visible_frame_fraction", 0.0), errors="coerce").fillna(0.0)
        hits = pd.to_numeric(filtered.get("total_object_hits", 0.0), errors="coerce").fillna(0.0)
        rays = pd.to_numeric(filtered.get("total_object_rays_tested", 0.0), errors="coerce").fillna(0.0)
        filtered = filtered[(vis > 0.0) | (hits > 0.0) | (rays > 0.0)]
    return filtered


def pick_manifest_rows(df: pd.DataFrame, batch_size: int, per_clip_limit: int) -> pd.DataFrame:
    ordered = df.sort_values(
        ["active_learning_score", "pred_std", "linear_mlp_gap", "pred_mean", "visible_frame_fraction", "total_object_hits"],
        ascending=False,
    )
    if batch_size <= 0:
        return ordered.iloc[0:0].copy()

    if per_clip_limit <= 0 or "clip_id" not in ordered.columns:
        return ordered.head(batch_size).copy()

    picked_rows = []
    clip_counts: dict[int, int] = {}
    for _, row in ordered.iterrows():
        clip_id = int(row.get("clip_id", -1))
        count = clip_counts.get(clip_id, 0)
        if count >= per_clip_limit:
            continue
        picked_rows.append(row)
        clip_counts[clip_id] = count + 1
        if len(picked_rows) >= batch_size:
            break
    if not picked_rows:
        return ordered.iloc[0:0].copy()
    return pd.DataFrame(picked_rows).reset_index(drop=True)


def describe_reason(row: pd.Series) -> str:
    reasons = []
    if row.get("uncertainty_norm", 0.0) >= 0.75:
        reasons.append("high_uncertainty")
    if row.get("disagreement_norm", 0.0) >= 0.75:
        reasons.append("model_disagreement")
    if row.get("impact_norm", 0.0) >= 0.75:
        reasons.append("high_predicted_impact")
    if row.get("visibility_prior", 0.0) >= 0.75:
        reasons.append("high_visibility")
    return ",".join(reasons) if reasons else "mixed_signal"


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_torch_seed(args.seed)

    features_df = pd.read_csv(args.features_csv)
    labels_df = pd.read_csv(args.labels_csv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates_out = args.candidates_out.resolve() if args.candidates_out else output.with_name(output.stem + "_candidates.csv")
    summary_out = args.summary_out.resolve() if args.summary_out else output.with_name(output.stem + "_summary.json")

    feature_names = choose_features(labels_df, args.feature_set, args.target, args.group_column)
    missing = [name for name in feature_names if name not in features_df.columns]
    if missing:
        raise RuntimeError(f"Features CSV is missing required columns: {missing}")
    if args.target not in labels_df.columns:
        raise RuntimeError(f"Labels CSV is missing target column: {args.target}")

    labeled_keys = set(build_key_strings(labels_df).tolist())
    feature_keys = build_key_strings(features_df)
    features_df = features_df.copy()
    features_df["candidate_key"] = feature_keys
    features_df["is_labeled"] = features_df["candidate_key"].isin(labeled_keys)

    scored_pool = features_df.copy() if args.include_labeled else features_df[~features_df["is_labeled"]].copy()
    scored_pool = filter_candidates(scored_pool, args)
    if scored_pool.empty:
        raise RuntimeError("No unlabeled candidates remain after filtering.")

    x_train, y_train = prepare_xy(labels_df, feature_names, args.target)
    x_eval = scored_pool[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    committee_seeds = parse_int_list(args.committee_seeds)
    if not committee_seeds:
        raise RuntimeError("At least one committee seed is required.")
    hidden_dims = parse_int_list(args.hidden_dims)
    if not hidden_dims:
        raise RuntimeError("At least one hidden layer dimension is required.")

    device = resolve_device(args.device)
    committee_preds, member_info = train_mlp_committee(
        x_train,
        y_train,
        x_eval,
        seeds=committee_seeds,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        log1p_target=args.log1p_target,
        device=device,
    )
    linear_pred = train_linear_reference(
        x_train,
        y_train,
        x_eval,
        alpha=args.alpha,
        log1p_target=args.log1p_target,
    )

    scored_pool["pred_mean"] = committee_preds.mean(axis=0)
    scored_pool["pred_std"] = committee_preds.std(axis=0)
    scored_pool["pred_min"] = committee_preds.min(axis=0)
    scored_pool["pred_max"] = committee_preds.max(axis=0)
    scored_pool["linear_pred"] = linear_pred
    scored_pool["linear_mlp_gap"] = np.abs(scored_pool["pred_mean"] - scored_pool["linear_pred"])
    scored_pool["committee_size"] = len(committee_seeds)
    scored_pool = score_candidates(scored_pool, args)
    scored_pool["selection_score"] = scored_pool["active_learning_score"]
    scored_pool["acquisition_reason"] = scored_pool.apply(describe_reason, axis=1)

    manifest_df = pick_manifest_rows(scored_pool, args.batch_size, args.per_clip_limit)
    manifest_df.to_csv(output, index=False)

    scored_candidates = scored_pool.sort_values("active_learning_score", ascending=False)
    scored_candidates.to_csv(candidates_out, index=False)

    summary = {
        "features_csv": str(args.features_csv.resolve()),
        "labels_csv": str(args.labels_csv.resolve()),
        "output_manifest": str(output),
        "candidates_csv": str(candidates_out),
        "target": args.target,
        "feature_set": args.feature_set,
        "features": feature_names,
        "labeled_count": int(len(labels_df)),
        "candidate_count": int(len(scored_pool)),
        "selected_count": int(len(manifest_df)),
        "committee_seeds": committee_seeds,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "device": str(device),
        "weights": {
            "uncertainty": args.uncertainty_weight,
            "disagreement": args.disagreement_weight,
            "impact": args.impact_weight,
            "visibility": args.visibility_weight,
        },
        "committee_members": member_info,
        "top_selected": manifest_df[
            [
                "scene_variant",
                "clip_id",
                "object_id",
                "active_learning_score",
                "pred_mean",
                "pred_std",
                "linear_mlp_gap",
                "acquisition_reason",
            ]
        ].to_dict(orient="records"),
    }
    summary_out.write_text(json.dumps(summary, indent=2))

    print("Active learning manifest ready")
    print(f"  labels: {args.labels_csv}")
    print(f"  unlabeled candidates scored: {len(scored_pool)}")
    print(f"  selected: {len(manifest_df)}")
    print(f"  output manifest: {output}")
    print(f"  scored candidates: {candidates_out}")
    print(f"  summary: {summary_out}")
    if not manifest_df.empty:
        preview = manifest_df[
            ["clip_id", "object_id", "active_learning_score", "pred_mean", "pred_std", "linear_mlp_gap", "acquisition_reason"]
        ].head(10)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("\nTop selections:")
            print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
