#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = (
        repo_root
        / "Benchmarks"
        / "neural_cgvqm_pipeline_bistro_v2"
        / "scene_bistro_test_v2_20260606_123721"
        / "cgvqm"
        / "unified_neural_mlp_comparison"
        / "mlp_16x8"
        / "full_model.pt"
    )
    default_output = default_model.with_suffix(".json")
    parser = argparse.ArgumentParser(
        description="Export a trained tiny UnifiedNeural MLP checkpoint to a renderer-friendly JSON file."
    )
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--output", type=Path, default=default_output)
    return parser.parse_args()


def tensor_to_nested_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    state = checkpoint["state_dict"]

    layer_indices = []
    for key in state.keys():
        if key.endswith(".weight"):
            # keys like net.0.weight, net.3.weight, net.6.weight
            layer_indices.append(int(key.split(".")[1]))
    layer_indices = sorted(layer_indices)

    layers = []
    for index, layer_id in enumerate(layer_indices):
        weight = state[f"net.{layer_id}.weight"]
        bias = state[f"net.{layer_id}.bias"]
        is_last = index == len(layer_indices) - 1
        layers.append(
            {
                "type": "dense",
                "activation": "linear" if is_last else "relu",
                "weight_rows": int(weight.shape[0]),
                "weight_cols": int(weight.shape[1]),
                "weights": tensor_to_nested_list(weight),
                "bias": tensor_to_nested_list(bias),
            }
        )

    export = {
        "format": "unified_neural_mlp_v1",
        "description": "Tiny MLP exported for renderer-side UnifiedNeural inference.",
        "target": checkpoint["target"],
        "target_transform": "exp_m1_of_linear_output" if checkpoint.get("log1p_target", False) else "identity",
        "input_normalization": {
            "features": checkpoint["features"],
            "mean": checkpoint["mean"],
            "std": checkpoint["std"],
        },
        "network": {
            "hidden_dims": checkpoint["hidden_dims"],
            "dropout": checkpoint.get("dropout", 0.0),
            "layers": layers,
        },
        "training": {
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "allow_negative_target": checkpoint.get("allow_negative_target", False),
        },
    }

    if "residual_teacher" in checkpoint:
        export["residual_teacher"] = checkpoint["residual_teacher"]
    else:
        residual_summary_path = args.model.with_name("residual_summary.json")
        if checkpoint.get("target") == "residual_target" and residual_summary_path.is_file():
            residual_summary = json.loads(residual_summary_path.read_text())
            residual_teacher = residual_summary.get("residual_teacher")
            if not residual_teacher:
                teacher_csv = residual_summary.get("teacher_csv")
                teacher_column = residual_summary.get("teacher_column", "teacher_rayhit_percentile")
                teacher_summary = {}
                if teacher_csv:
                    teacher_summary_path = Path(teacher_csv).with_suffix(".summary.json")
                    if teacher_summary_path.is_file():
                        teacher_summary = json.loads(teacher_summary_path.read_text())
                residual_teacher = {
                    "kind": "rayhit_efficiency_percentile_v1",
                    "teacher_column": teacher_column,
                    "scale": residual_summary.get("teacher_scale", 1.0),
                    "bias": residual_summary.get("teacher_bias", 0.0),
                    "cost_exponent": teacher_summary.get("cost_exponent", 0.5),
                    "visibility_weight": teacher_summary.get("visibility_weight", 0.25),
                    "hit_probability_weight": teacher_summary.get("hit_probability_weight", 0.25),
                }
            export["residual_teacher"] = residual_teacher

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
