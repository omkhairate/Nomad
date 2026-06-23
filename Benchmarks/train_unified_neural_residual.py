#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_unified_neural_baseline import mae, pearson_corr, rmse


KEY_COLUMNS = ["scene_variant", "strategy", "clip_id", "object_id"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_session = (
        repo_root
        / "Benchmarks"
        / "neural_cgvqm_pipeline_bistro_v2"
        / "scene_bistro_test_v2_20260606_123721"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual UnifiedNeural model: first fit an affine mapping from the rayhit teacher "
            "target to CGVQM labels, then train a tiny MLP only on the remaining signed residual."
        )
    )
    parser.add_argument("--labels-csv", type=Path, default=default_session / "cgvqm" / "neural_training_labels_50.csv")
    parser.add_argument("--teacher-csv", type=Path, default=default_session / "active_learning" / "teacher" / "rayhit_teacher_labels.csv")
    parser.add_argument("--teacher-column", default="teacher_rayhit_percentile")
    parser.add_argument("--target", default="delta_cgvqm")
    parser.add_argument("--group-column", default="clip_id")
    parser.add_argument("--output-dir", type=Path, default=default_session / "active_learning" / "residual_model")
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--hidden-dims", default="16,8")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def best_join_keys(labels_df: pd.DataFrame, teacher_df: pd.DataFrame) -> list[str]:
    keys = [col for col in KEY_COLUMNS if col in labels_df.columns and col in teacher_df.columns]
    if not keys:
        raise RuntimeError(f"Could not find any shared key columns from {KEY_COLUMNS}.")
    return keys


def fit_affine_teacher(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    scale = float(coeffs[0])
    bias = float(coeffs[1])
    return scale, bias


def main() -> int:
    args = parse_args()
    labels_df = pd.read_csv(args.labels_csv).copy()
    teacher_df = pd.read_csv(args.teacher_csv).copy()
    join_keys = best_join_keys(labels_df, teacher_df)

    if args.teacher_column not in teacher_df.columns:
        raise RuntimeError(f"Teacher CSV is missing column: {args.teacher_column}")
    if args.target not in labels_df.columns:
        raise RuntimeError(f"Labels CSV is missing target column: {args.target}")

    teacher_subset = teacher_df[join_keys + [args.teacher_column]].drop_duplicates(subset=join_keys, keep="last")
    merged = labels_df.merge(teacher_subset, on=join_keys, how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping labeled rows found between labels CSV and teacher CSV.")
    if len(merged) != len(labels_df):
        print(f"Warning: joined {len(merged)} of {len(labels_df)} labeled rows with teacher targets.")

    teacher_values = pd.to_numeric(merged[args.teacher_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    target_values = pd.to_numeric(merged[args.target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    scale, bias = fit_affine_teacher(teacher_values, target_values)
    teacher_baseline = teacher_values * scale + bias
    residual_target = target_values - teacher_baseline

    merged["teacher_baseline_prediction"] = teacher_baseline
    merged["residual_target"] = residual_target

    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual_csv = args.output_dir / "residual_training_labels.csv"
    merged.to_csv(residual_csv, index=False)

    trainer = Path(__file__).resolve().with_name("train_unified_neural_mlp.py")
    train_cmd = [
        sys.executable,
        str(trainer),
        "--csv",
        str(residual_csv),
        "--target",
        "residual_target",
        "--group-column",
        args.group_column,
        "--output-dir",
        str(args.output_dir),
        "--hidden-dims",
        args.hidden_dims,
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--no-log1p-target",
        "--allow-negative-target",
    ]
    if args.init_model is not None:
        train_cmd.extend(["--init-model", str(args.init_model.resolve())])

    print("Residual training")
    print(f"  joined labels: {len(merged)}")
    print(f"  teacher column: {args.teacher_column}")
    print(f"  affine teacher mapping: target ~= {scale:.6f} * teacher + {bias:.6f}")
    print("  command:", " ".join(train_cmd))
    completed = subprocess.run(train_cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Residual model training failed with exit code {completed.returncode}")

    teacher_summary_path = args.teacher_csv.with_suffix(".summary.json")
    teacher_summary = {}
    if teacher_summary_path.is_file():
        try:
            teacher_summary = json.loads(teacher_summary_path.read_text())
        except Exception:
            teacher_summary = {}

    checkpoint_path = args.output_dir / "full_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["residual_teacher"] = {
        "kind": "rayhit_efficiency_percentile_v1",
        "teacher_column": args.teacher_column,
        "scale": scale,
        "bias": bias,
        "cost_exponent": float(teacher_summary.get("cost_exponent", 0.5)),
        "visibility_weight": float(teacher_summary.get("visibility_weight", 0.25)),
        "hit_probability_weight": float(teacher_summary.get("hit_probability_weight", 0.25)),
    }
    torch.save(checkpoint, checkpoint_path)

    predictions_csv = args.output_dir / "predictions.csv"
    pred_df = pd.read_csv(predictions_csv).sort_values("row_index").reset_index(drop=True)
    merged_reset = merged.reset_index(drop=True)
    if len(pred_df) != len(merged_reset):
        raise RuntimeError("Prediction row count does not match merged residual label table.")

    pred_df = pred_df.rename(columns={"y_true": "residual_true", "y_pred": "residual_pred"})
    eval_df = merged_reset.join(pred_df[["residual_true", "residual_pred", "abs_error"]])
    eval_df["teacher_baseline_prediction"] = pd.to_numeric(eval_df["teacher_baseline_prediction"], errors="coerce").fillna(0.0)
    eval_df[args.target] = pd.to_numeric(eval_df[args.target], errors="coerce").fillna(0.0)
    eval_df["residual_model_prediction"] = pd.to_numeric(eval_df["residual_pred"], errors="coerce").fillna(0.0)
    eval_df["final_prediction"] = eval_df["teacher_baseline_prediction"] + eval_df["residual_model_prediction"]

    y_true = eval_df[args.target].to_numpy(dtype=np.float64)
    baseline_pred = eval_df["teacher_baseline_prediction"].to_numpy(dtype=np.float64)
    final_pred = eval_df["final_prediction"].to_numpy(dtype=np.float64)

    summary = {
        "labels_csv": str(args.labels_csv.resolve()),
        "teacher_csv": str(args.teacher_csv.resolve()),
        "teacher_column": args.teacher_column,
        "target": args.target,
        "group_column": args.group_column,
        "joined_row_count": int(len(eval_df)),
        "teacher_scale": scale,
        "teacher_bias": bias,
        "teacher_baseline_metrics": {
            "mae": mae(y_true, baseline_pred),
            "rmse": rmse(y_true, baseline_pred),
            "corr": pearson_corr(y_true, baseline_pred),
        },
        "teacher_plus_residual_metrics": {
            "mae": mae(y_true, final_pred),
            "rmse": rmse(y_true, final_pred),
            "corr": pearson_corr(y_true, final_pred),
        },
        "init_model": str(args.init_model.resolve()) if args.init_model else None,
        "residual_teacher": checkpoint["residual_teacher"],
    }

    eval_csv = args.output_dir / "residual_eval_predictions.csv"
    summary_json = args.output_dir / "residual_summary.json"
    eval_df.to_csv(eval_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2))

    print("Residual evaluation summary")
    print(f"  teacher baseline MAE: {summary['teacher_baseline_metrics']['mae']:.4f}")
    print(f"  teacher baseline RMSE: {summary['teacher_baseline_metrics']['rmse']:.4f}")
    print(f"  teacher baseline corr: {summary['teacher_baseline_metrics']['corr']:.4f}")
    print(f"  teacher+residual MAE: {summary['teacher_plus_residual_metrics']['mae']:.4f}")
    print(f"  teacher+residual RMSE: {summary['teacher_plus_residual_metrics']['rmse']:.4f}")
    print(f"  teacher+residual corr: {summary['teacher_plus_residual_metrics']['corr']:.4f}")
    print(f"  residual labels csv: {residual_csv}")
    print(f"  residual eval csv: {eval_csv}")
    print(f"  residual summary json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
