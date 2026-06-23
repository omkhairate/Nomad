#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_unified_neural_baseline import V1_FEATURES, mae, pearson_corr, rmse


@dataclass(frozen=True)
class FoldResult:
    fold_name: str
    train_count: int
    test_count: int
    mae: float
    rmse: float
    corr: float


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
    default_out = default_csv.with_name("unified_neural_mlp_v1")

    parser = argparse.ArgumentParser(
        description="Train a tiny MLP baseline for UnifiedNeural using leave-one-clip-out evaluation."
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--target", default="delta_cgvqm")
    parser.add_argument("--group-column", default="clip_id")
    parser.add_argument("--hidden-dims", default="32,16")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log1p-target", action="store_true", default=True)
    parser.add_argument(
        "--no-log1p-target",
        action="store_false",
        dest="log1p_target",
        help="Disable log1p(target) fitting. Use this for signed residual targets.",
    )
    parser.add_argument(
        "--allow-negative-target",
        action="store_true",
        help=(
            "Allow signed targets and signed predictions. Use this for residual learning. "
            "When omitted, predictions are clamped to be nonnegative."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint from a previous teacher-pretrain or fine-tune run. "
            "When compatible, matching layers are loaded before training."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(x_train)
    if count <= 5:
        return x_train, y_train, x_train, y_train
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    val_count = max(1, int(round(count * val_fraction)))
    val_indices = indices[:val_count]
    fit_indices = indices[val_count:]
    if len(fit_indices) == 0:
        fit_indices = val_indices
    return (
        x_train[fit_indices],
        y_train[fit_indices],
        x_train[val_indices],
        y_train[val_indices],
    )


def train_one_fold(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    init_state_dict: dict[str, torch.Tensor] | None,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[TinyMLP, dict[str, float]]:
    model = TinyMLP(x_fit.shape[1], hidden_dims, dropout).to(device)
    if init_state_dict:
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in init_state_dict.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        if compatible:
            model_state.update(compatible)
            model.load_state_dict(model_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    x_fit_t = torch.tensor(x_fit, dtype=torch.float32, device=device)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    stalled = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_fit_t)
        loss = loss_fn(pred, y_fit_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stalled = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stalled += 1
            if stalled >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "best_epoch": float(best_epoch)}


def load_init_state_dict(path: Path | None, device: torch.device) -> dict[str, torch.Tensor] | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint at {path} does not contain a state_dict.")
    return state_dict


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    init_state_dict = load_init_state_dict(args.init_model.resolve(), device) if args.init_model else None

    df = pd.read_csv(args.csv)
    features = [feature for feature in V1_FEATURES if feature in df.columns]
    x_all = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y_all = pd.to_numeric(df[args.target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if args.log1p_target and args.allow_negative_target:
        raise RuntimeError("--log1p-target and --allow-negative-target cannot be used together.")
    if args.log1p_target:
        y_fit_all = np.log1p(np.maximum(y_all, 0.0))
    else:
        y_fit_all = y_all.copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds: list[FoldResult] = []
    predictions: list[dict[str, float | int]] = []

    for group_value in sorted(df[args.group_column].dropna().unique().tolist()):
        test_mask = df[args.group_column] == group_value
        train_mask = ~test_mask

        x_train = x_all[train_mask.to_numpy()]
        x_test = x_all[test_mask.to_numpy()]
        y_train_fit = y_fit_all[train_mask.to_numpy()]
        y_test_true = y_all[test_mask.to_numpy()]

        x_train_std, x_test_std, mean, std = standardize_train_test(x_train, x_test)
        x_fit_std, y_fit_std, x_val_std, y_val_std = split_train_validation(
            x_train_std,
            y_train_fit,
            seed=args.seed + int(group_value),
        )
        model, train_info = train_one_fold(
            x_fit=x_fit_std,
            y_fit=y_fit_std,
            x_val=x_val_std,
            y_val=y_val_std,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            init_state_dict=init_state_dict,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            x_test_t = torch.tensor(x_test_std, dtype=torch.float32, device=device)
            pred_fit = model(x_test_t).detach().cpu().numpy()
        y_pred = np.expm1(pred_fit) if args.log1p_target else pred_fit
        if not args.allow_negative_target:
            y_pred = np.maximum(y_pred, 0.0)

        folds.append(
            FoldResult(
                fold_name=f"{args.group_column}={group_value}",
                train_count=int(train_mask.sum()),
                test_count=int(test_mask.sum()),
                mae=mae(y_test_true, y_pred),
                rmse=rmse(y_test_true, y_pred),
                corr=pearson_corr(y_test_true, y_pred),
            )
        )

        for row_index, truth, pred in zip(df[test_mask].index.tolist(), y_test_true.tolist(), y_pred.tolist()):
            predictions.append(
                {
                    "row_index": int(row_index),
                    args.group_column: int(group_value),
                    "y_true": float(truth),
                    "y_pred": float(pred),
                    "abs_error": float(abs(truth - pred)),
                    "best_epoch": train_info["best_epoch"],
                    "best_val_loss": train_info["best_val_loss"],
                }
            )

    folds_df = pd.DataFrame([fold.__dict__ for fold in folds])
    pred_df = pd.DataFrame(predictions).sort_values("row_index")

    x_std_all, _, mean_all, std_all = standardize_train_test(x_all, x_all)
    x_fit_all, y_fit_sub, x_val_all, y_val_all = split_train_validation(
        x_std_all,
        y_fit_all,
        seed=args.seed,
    )
    full_model, full_info = train_one_fold(
        x_fit=x_fit_all,
        y_fit=y_fit_sub,
        x_val=x_val_all,
        y_val=y_val_all,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        init_state_dict=init_state_dict,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
    )

    model_path = args.output_dir / "full_model.pt"
    torch.save(
        {
            "state_dict": full_model.state_dict(),
            "features": features,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "mean": mean_all.tolist(),
            "std": std_all.tolist(),
            "target": args.target,
            "log1p_target": args.log1p_target,
            "allow_negative_target": args.allow_negative_target,
            "best_epoch": full_info["best_epoch"],
            "best_val_loss": full_info["best_val_loss"],
            "init_model": str(args.init_model.resolve()) if args.init_model else None,
        },
        model_path,
    )

    summary = {
        "csv": str(args.csv),
        "target": args.target,
        "features": features,
        "group_column": args.group_column,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "log1p_target": args.log1p_target,
        "allow_negative_target": args.allow_negative_target,
        "init_model": str(args.init_model.resolve()) if args.init_model else None,
        "row_count": int(len(df)),
        "mean_mae": float(folds_df["mae"].mean()),
        "mean_rmse": float(folds_df["rmse"].mean()),
        "mean_corr": float(folds_df["corr"].dropna().mean()) if folds_df["corr"].notna().any() else None,
    }

    folds_path = args.output_dir / "fold_metrics.csv"
    preds_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.json"
    folds_df.to_csv(folds_path, index=False)
    pred_df.to_csv(preds_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("UnifiedNeural tiny MLP summary")
    print(f"  csv: {args.csv}")
    print(f"  output dir: {args.output_dir}")
    print(f"  features: {', '.join(features)}")
    print(f"  hidden dims: {hidden_dims}")
    print(f"  device: {device}")
    print(f"  mean MAE: {summary['mean_mae']:.4f}")
    print(f"  mean RMSE: {summary['mean_rmse']:.4f}")
    if summary["mean_corr"] is not None:
        print(f"  mean corr: {summary['mean_corr']:.4f}")
    print(f"  fold metrics: {folds_path}")
    print(f"  predictions: {preds_path}")
    print(f"  model: {model_path}")
    print(f"  summary json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
