#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = (
        repo_root
        / "Benchmarks"
        / "neural_cgvqm_pipeline_bistro_v2"
        / "scene_bistro_test_v2_20260606_123721"
        / "cgvqm"
    )
    parser = argparse.ArgumentParser(
        description="Create prediction comparison plots for the linear baseline and tiny UnifiedNeural MLP."
    )
    parser.add_argument("--labels", type=Path, default=root / "neural_training_labels_50.csv")
    parser.add_argument(
        "--linear-predictions",
        type=Path,
        default=root / "unified_neural_baseline_v1" / "predictions.csv",
    )
    parser.add_argument(
        "--mlp-predictions",
        type=Path,
        default=root / "unified_neural_mlp_comparison" / "mlp_16x8" / "predictions.csv",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=root / "prediction_comparison_linear_vs_mlp16x8.html",
    )
    return parser.parse_args()


def attach_predictions(labels: pd.DataFrame, predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pred = predictions.rename(
        columns={
            "y_true": f"{prefix}_y_true",
            "y_pred": f"{prefix}_y_pred",
            "abs_error": f"{prefix}_abs_error",
        }
    )
    return labels.merge(pred[["row_index", f"{prefix}_y_true", f"{prefix}_y_pred", f"{prefix}_abs_error"]], left_index=True, right_on="row_index")


def main() -> int:
    args = parse_args()
    labels = pd.read_csv(args.labels).reset_index(drop=True)
    linear_pred = pd.read_csv(args.linear_predictions)
    mlp_pred = pd.read_csv(args.mlp_predictions)

    merged = attach_predictions(labels, linear_pred, "linear")
    merged = attach_predictions(merged, mlp_pred, "mlp")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "True vs Linear Prediction",
            "True vs MLP(16,8) Prediction",
            "Absolute Error by Sample",
            "Prediction Error by Clip",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=merged["delta_cgvqm"],
            y=merged["linear_y_pred"],
            mode="markers+text",
            text=[f"c{c}-o{o}" for c, o in zip(merged["clip_id"], merged["object_id"])],
            textposition="top center",
            name="Linear",
            marker=dict(color="#1f77b4", size=8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["delta_cgvqm"],
            y=merged["mlp_y_pred"],
            mode="markers+text",
            text=[f"c{c}-o{o}" for c, o in zip(merged["clip_id"], merged["object_id"])],
            textposition="top center",
            name="MLP 16,8",
            marker=dict(color="#d62728", size=8),
        ),
        row=1,
        col=2,
    )

    max_val = max(
        merged["delta_cgvqm"].max(),
        merged["linear_y_pred"].max(),
        merged["mlp_y_pred"].max(),
    )
    for row, col in [(1, 1), (1, 2)]:
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["linear_abs_error"],
            mode="lines+markers",
            name="Linear abs error",
            marker=dict(color="#1f77b4"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["mlp_abs_error"],
            mode="lines+markers",
            name="MLP abs error",
            marker=dict(color="#d62728"),
        ),
        row=2,
        col=1,
    )

    clip_stats = (
        merged.groupby("clip_id")[["linear_abs_error", "mlp_abs_error"]]
        .mean()
        .reset_index()
    )
    fig.add_trace(
        go.Bar(
            x=clip_stats["clip_id"].astype(str),
            y=clip_stats["linear_abs_error"],
            name="Linear mean abs error",
            marker_color="#1f77b4",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=clip_stats["clip_id"].astype(str),
            y=clip_stats["mlp_abs_error"],
            name="MLP mean abs error",
            marker_color="#d62728",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="UnifiedNeural Prediction Comparison: Linear vs Tiny MLP (16,8)",
        template="plotly_white",
        height=900,
        width=1300,
        barmode="group",
    )
    fig.update_xaxes(title_text="True delta_cgvqm", row=1, col=1)
    fig.update_yaxes(title_text="Predicted delta_cgvqm", row=1, col=1)
    fig.update_xaxes(title_text="True delta_cgvqm", row=1, col=2)
    fig.update_yaxes(title_text="Predicted delta_cgvqm", row=1, col=2)
    fig.update_xaxes(title_text="Sample index", row=2, col=1)
    fig.update_yaxes(title_text="Absolute error", row=2, col=1)
    fig.update_xaxes(title_text="Clip id", row=2, col=2)
    fig.update_yaxes(title_text="Mean absolute error", row=2, col=2)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output_html, include_plotlyjs="inline")
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
