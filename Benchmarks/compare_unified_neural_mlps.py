#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CONFIGS = [
    "8",
    "16,8",
    "32,16",
    "32,16,8",
]


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
    default_root = default_csv.with_name("unified_neural_mlp_comparison")

    parser = argparse.ArgumentParser(
        description="Compare several tiny MLP architectures for UnifiedNeural."
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--output-root", type=Path, default=default_root)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help='Hidden-layer configs, e.g. "8" "16,8" "32,16"',
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run_model(args: argparse.Namespace, hidden_dims: str) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    trainer = repo_root / "Benchmarks" / "train_unified_neural_mlp.py"
    tag = hidden_dims.replace(",", "x")
    output_dir = args.output_root / f"mlp_{tag}"
    cmd = [
        sys.executable,
        str(trainer),
        "--csv",
        str(args.csv),
        "--output-dir",
        str(output_dir),
        "--hidden-dims",
        hidden_dims,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Model run failed for config {hidden_dims} with exit code {completed.returncode}")
    summary = json.loads((output_dir / "summary.json").read_text())
    summary["config"] = hidden_dims
    summary["output_dir"] = str(output_dir)
    return summary


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for hidden_dims in args.configs:
        rows.append(run_model(args, hidden_dims))

    df = pd.DataFrame(rows)
    df = df.sort_values(["mean_rmse", "mean_mae", "mean_corr"], ascending=[True, True, False])
    summary_csv = args.output_root / "comparison.csv"
    summary_json = args.output_root / "comparison.json"
    df.to_csv(summary_csv, index=False)
    summary_json.write_text(df.to_json(orient="records", indent=2))

    print("\nTiny MLP comparison")
    print(df[["config", "mean_mae", "mean_rmse", "mean_corr"]].to_string(index=False))
    print(f"\ncomparison csv: {summary_csv}")
    print(f"comparison json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
