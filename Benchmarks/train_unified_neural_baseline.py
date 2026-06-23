#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


V1_FEATURES = [
    "total_object_hits",
    "mean_object_rayhit_score",
    "mean_object_importance",
    "mean_hit_probability",
    "total_object_rays_tested",
    "mean_distance",
    "min_distance",
    "mean_visible_coverage",
    "max_visible_coverage",
    "primitive_count",
    "mean_estimated_object_bytes",
]


@dataclass(frozen=True)
class FoldResult:
    fold_name: str
    train_count: int
    test_count: int
    mae: float
    rmse: float
    corr: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_csv = (
        repo_root
        / "Benchmarks"
        / "neural_cgvqm_pipeline_bistro_v2"
        / "scene_bistro_test_v2_20260606_123721"
        / "cgvqm"
        / "neural_training_labels_50.csv"
    )
    default_out = default_csv.with_name("unified_neural_baseline")

    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight baseline regressor for UnifiedNeural using only "
            "numpy/pandas. The default evaluation is leave-one-clip-out because "
            "the current dataset is organized by clip."
        )
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--target", default="delta_cgvqm")
    parser.add_argument(
        "--feature-set",
        choices=["v1", "all_nonconstant"],
        default="v1",
        help="Choose the curated UnifiedNeural v1 set or all non-constant numeric features.",
    )
    parser.add_argument(
        "--group-column",
        default="clip_id",
        help="Column used for leave-one-group-out evaluation. Default: clip_id",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength. Default: 1.0",
    )
    parser.add_argument(
        "--log1p-target",
        action="store_true",
        help="Fit on log1p(target) and invert with expm1 at prediction time.",
    )
    return parser.parse_args()


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pick_all_nonconstant_numeric_features(df: pd.DataFrame, target: str, group_col: str) -> list[str]:
    blocked = {
        target,
        group_col,
        "object_id",
        "forced_object_off",
        "clip_start_frame",
        "clip_end_frame",
        "clip_frame_count",
        "baseline_cgvqm_score",
        "ablation_cgvqm_score",
        "window_padding_frames",
        "captured_frame_count",
    }
    features: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        features.append(col)
    return features


def prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return x, y


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    xtx = x_train.T @ x_train
    reg = np.eye(xtx.shape[0], dtype=np.float64) * alpha
    weights = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    intercept = float(y_train.mean() - x_train.mean(axis=0) @ weights)
    return weights, intercept


def predict_ridge(
    x: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> np.ndarray:
    return x @ weights + intercept


def evaluate_leave_one_group_out(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    group_col: str,
    alpha: float,
    log1p_target: bool,
) -> tuple[list[FoldResult], pd.DataFrame, pd.DataFrame]:
    x_all, y_all = prepare_matrix(df, features, target)
    predictions: list[dict[str, float | int]] = []
    folds: list[FoldResult] = []

    for group_value in sorted(df[group_col].dropna().unique().tolist()):
        test_mask = df[group_col] == group_value
        train_mask = ~test_mask

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        x_train = x_all[train_mask.to_numpy()]
        y_train = y_all[train_mask.to_numpy()]
        x_test = x_all[test_mask.to_numpy()]
        y_test = y_all[test_mask.to_numpy()]

        if log1p_target:
            y_train_fit = np.log1p(np.maximum(y_train, 0.0))
        else:
            y_train_fit = y_train

        x_train_std, x_test_std, _, _ = standardize_train_test(x_train, x_test)
        weights, intercept = fit_ridge(x_train_std, y_train_fit, alpha)
        pred_fit = predict_ridge(x_test_std, weights, intercept)
        y_pred = np.expm1(pred_fit) if log1p_target else pred_fit
        y_pred = np.maximum(y_pred, 0.0)

        folds.append(
            FoldResult(
                fold_name=f"{group_col}={group_value}",
                train_count=len(train_df),
                test_count=len(test_df),
                mae=mae(y_test, y_pred),
                rmse=rmse(y_test, y_pred),
                corr=pearson_corr(y_test, y_pred),
            )
        )

        for row_index, truth, pred in zip(test_df.index.tolist(), y_test.tolist(), y_pred.tolist()):
            predictions.append(
                {
                    "row_index": int(row_index),
                    group_col: int(group_value),
                    "y_true": float(truth),
                    "y_pred": float(pred),
                    "abs_error": float(abs(truth - pred)),
                }
            )

    pred_df = pd.DataFrame(predictions).sort_values("row_index")

    x_std, _, mean, std = standardize_train_test(x_all, x_all)
    y_fit = np.log1p(np.maximum(y_all, 0.0)) if log1p_target else y_all
    full_weights, full_intercept = fit_ridge(x_std, y_fit, alpha)
    coef_df = pd.DataFrame(
        {
            "feature": features,
            "weight": full_weights,
            "abs_weight": np.abs(full_weights),
            "train_mean": mean,
            "train_std": std,
        }
    ).sort_values("abs_weight", ascending=False)
    return folds, pred_df, coef_df


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set == "v1":
        features = [feature for feature in V1_FEATURES if feature in df.columns]
    else:
        features = pick_all_nonconstant_numeric_features(df, args.target, args.group_column)

    folds, pred_df, coef_df = evaluate_leave_one_group_out(
        df=df,
        features=features,
        target=args.target,
        group_col=args.group_column,
        alpha=args.alpha,
        log1p_target=args.log1p_target,
    )

    folds_df = pd.DataFrame([fold.__dict__ for fold in folds])
    summary = {
        "csv": str(args.csv),
        "target": args.target,
        "feature_set": args.feature_set,
        "features": features,
        "group_column": args.group_column,
        "alpha": args.alpha,
        "log1p_target": args.log1p_target,
        "row_count": int(len(df)),
        "fold_count": int(len(folds_df)),
        "mean_mae": float(folds_df["mae"].mean()),
        "mean_rmse": float(folds_df["rmse"].mean()),
        "mean_corr": float(folds_df["corr"].dropna().mean()) if folds_df["corr"].notna().any() else None,
    }

    folds_path = args.output_dir / "fold_metrics.csv"
    preds_path = args.output_dir / "predictions.csv"
    coefs_path = args.output_dir / "feature_weights.csv"
    summary_path = args.output_dir / "summary.json"

    folds_df.to_csv(folds_path, index=False)
    pred_df.to_csv(preds_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("UnifiedNeural baseline training summary")
    print(f"  csv: {args.csv}")
    print(f"  output dir: {args.output_dir}")
    print(f"  feature set: {args.feature_set}")
    print(f"  features: {', '.join(features)}")
    print(f"  mean MAE: {summary['mean_mae']:.4f}")
    print(f"  mean RMSE: {summary['mean_rmse']:.4f}")
    if summary["mean_corr"] is not None:
        print(f"  mean corr: {summary['mean_corr']:.4f}")
    print(f"  fold metrics: {folds_path}")
    print(f"  predictions: {preds_path}")
    print(f"  feature weights: {coefs_path}")
    print(f"  summary json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
