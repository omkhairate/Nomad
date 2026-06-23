#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_SCENE = "scene_bistro_test_v2.xml"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_app_path() -> Path:
    return (
        repo_root()
        / "build"
        / "Build"
        / "Products"
        / "Release"
        / "Nomad Path Tracer.app"
        / "Contents"
        / "MacOS"
        / "Nomad Path Tracer"
    )


def default_session_root() -> Path:
    return repo_root() / "Benchmarks" / "neural_cgvqm_pipeline_bistro_v2" / "scene_bistro_test_v2_20260606_123721"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Practical UnifiedNeural active-learning orchestrator. "
            "It can bootstrap an initial manifest, run an uncertainty-driven ablation round, "
            "merge labels, retrain the model, and export renderer-ready weights."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap-manifest",
        help="Create an initial heuristic ablation manifest from feature logs.",
    )
    add_session_args(bootstrap)
    bootstrap.add_argument("--features-csv", type=Path, default=None)
    bootstrap.add_argument("--output", type=Path, default=None)
    bootstrap.add_argument("--top-visible", type=int, default=8)
    bootstrap.add_argument("--top-hit", type=int, default=6)
    bootstrap.add_argument("--top-toggle", type=int, default=4)
    bootstrap.add_argument("--random-controls", type=int, default=2)
    bootstrap.add_argument("--seed", type=int, default=7)
    bootstrap.add_argument("--dry-run", action="store_true")

    active = subparsers.add_parser(
        "active-round",
        help="Run one full active-learning round: select, ablate, label, merge, retrain, export.",
    )
    add_session_args(active)
    active.add_argument("--features-csv", type=Path, default=None)
    active.add_argument(
        "--master-labels-csv",
        type=Path,
        default=None,
        help="Canonical merged label CSV. Default: <session-root>/active_learning/labels_master.csv",
    )
    active.add_argument(
        "--seed-labels-csv",
        action="append",
        type=Path,
        default=[],
        help="Existing label CSV(s) used to seed or refresh the master label table.",
    )
    active.add_argument("--round-name", default=None)
    active.add_argument("--app", type=Path, default=default_app_path())
    active.add_argument("--scene", default=DEFAULT_SCENE)
    active.add_argument("--ablation-root", type=Path, default=None)
    active.add_argument("--baseline-run-dir", type=Path, default=None)
    active.add_argument("--baseline-root", type=Path, default=None)
    active.add_argument("--max-frames", type=int, default=500)
    active.add_argument("--capture-interval", type=int, default=4)
    active.add_argument("--video-fps", type=float, default=15.0)
    active.add_argument("--window-mode", choices=["clip", "full"], default="clip")
    active.add_argument("--window-padding-frames", type=int, default=48)
    active.add_argument("--representation", choices=["denoised", "raw"], default="denoised")
    active.add_argument("--no-backfill-denoise", action="store_true")
    active.add_argument("--no-defer-oidn", action="store_true")
    active.add_argument("--batch-size", type=int, default=30)
    active.add_argument("--per-clip-limit", type=int, default=10)
    active.add_argument("--feature-set", choices=["v1", "all_nonconstant"], default="v1")
    active.add_argument("--group-column", default="clip_id")
    active.add_argument("--committee-seeds", default="7,17,27,37,47")
    active.add_argument("--selector-hidden-dims", default="16,8")
    active.add_argument("--selector-dropout", type=float, default=0.10)
    active.add_argument("--selector-lr", type=float, default=1e-3)
    active.add_argument("--selector-weight-decay", type=float, default=1e-4)
    active.add_argument("--selector-epochs", type=int, default=400)
    active.add_argument("--selector-patience", type=int, default=60)
    active.add_argument("--selector-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    active.add_argument("--selector-alpha", type=float, default=1.0)
    active.add_argument("--uncertainty-weight", type=float, default=0.45)
    active.add_argument("--disagreement-weight", type=float, default=0.25)
    active.add_argument("--impact-weight", type=float, default=0.20)
    active.add_argument("--visibility-weight", type=float, default=0.10)
    active.add_argument("--min-visible-fraction", type=float, default=0.0)
    active.add_argument("--min-total-hits", type=float, default=0.0)
    active.add_argument("--keep-empty-candidates", action="store_true")
    active.add_argument("--cgvqm-python", type=Path, default=Path(sys.executable))
    active.add_argument("--cgvqm-script", type=Path, default=repo_root() / "Benchmarks" / "cgvqm" / "run_metric.py")
    active.add_argument("--cgvqm-type", choices=["cgvqm2", "cgvqm5"], default="cgvqm2")
    active.add_argument("--cgvqm-device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    active.add_argument("--patch-scale", type=int, default=4)
    active.add_argument("--patch-pool", choices=["mean", "max"], default="mean")
    active.add_argument("--train-hidden-dims", default="16,8")
    active.add_argument("--train-dropout", type=float, default=0.10)
    active.add_argument("--train-lr", type=float, default=1e-3)
    active.add_argument("--train-weight-decay", type=float, default=1e-4)
    active.add_argument("--train-epochs", type=int, default=600)
    active.add_argument("--train-patience", type=int, default=80)
    active.add_argument("--train-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    active.add_argument("--train-init-model", type=Path, default=None)
    active.add_argument("--skip-existing", action="store_true")
    active.add_argument("--dry-run", action="store_true")
    active.add_argument("--seed", type=int, default=7)

    retrain = subparsers.add_parser(
        "retrain-export",
        help="Retrain and export the model from an existing merged label CSV.",
    )
    add_session_args(retrain)
    retrain.add_argument("--labels-csv", type=Path, default=None)
    retrain.add_argument("--output-dir", type=Path, default=None)
    retrain.add_argument("--init-model", type=Path, default=None)
    retrain.add_argument("--train-hidden-dims", default="16,8")
    retrain.add_argument("--train-dropout", type=float, default=0.10)
    retrain.add_argument("--train-lr", type=float, default=1e-3)
    retrain.add_argument("--train-weight-decay", type=float, default=1e-4)
    retrain.add_argument("--train-epochs", type=int, default=600)
    retrain.add_argument("--train-patience", type=int, default=80)
    retrain.add_argument("--train-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    retrain.add_argument("--seed", type=int, default=7)
    retrain.add_argument("--dry-run", action="store_true")

    teacher = subparsers.add_parser(
        "teacher-pretrain",
        help="Build a rayhit teacher dataset, pretrain the tiny MLP on it, and export renderer-ready weights.",
    )
    add_session_args(teacher)
    teacher.add_argument("--features-csv", type=Path, default=None)
    teacher.add_argument("--teacher-csv", type=Path, default=None)
    teacher.add_argument("--output-dir", type=Path, default=None)
    teacher.add_argument(
        "--target-column",
        default="teacher_rayhit_percentile",
        choices=["teacher_rayhit_percentile", "teacher_rayhit_efficiency", "teacher_rayhit_raw"],
    )
    teacher.add_argument("--cost-exponent", type=float, default=0.5)
    teacher.add_argument("--visibility-weight", type=float, default=0.25)
    teacher.add_argument("--hit-probability-weight", type=float, default=0.25)
    teacher.add_argument("--train-hidden-dims", default="16,8")
    teacher.add_argument("--train-dropout", type=float, default=0.10)
    teacher.add_argument("--train-lr", type=float, default=1e-3)
    teacher.add_argument("--train-weight-decay", type=float, default=1e-4)
    teacher.add_argument("--train-epochs", type=int, default=600)
    teacher.add_argument("--train-patience", type=int, default=80)
    teacher.add_argument("--train-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    teacher.add_argument("--seed", type=int, default=7)
    teacher.add_argument("--dry-run", action="store_true")

    residual = subparsers.add_parser(
        "residual-train",
        help="Train a teacher-plus-residual model: affine rayhit teacher baseline plus signed neural correction.",
    )
    add_session_args(residual)
    residual.add_argument("--labels-csv", type=Path, default=None)
    residual.add_argument("--teacher-csv", type=Path, default=None)
    residual.add_argument("--teacher-column", default="teacher_rayhit_percentile")
    residual.add_argument("--output-dir", type=Path, default=None)
    residual.add_argument("--init-model", type=Path, default=None)
    residual.add_argument("--train-hidden-dims", default="16,8")
    residual.add_argument("--train-dropout", type=float, default=0.10)
    residual.add_argument("--train-lr", type=float, default=1e-3)
    residual.add_argument("--train-weight-decay", type=float, default=1e-4)
    residual.add_argument("--train-epochs", type=int, default=600)
    residual.add_argument("--train-patience", type=int, default=80)
    residual.add_argument("--train-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    residual.add_argument("--seed", type=int, default=7)
    residual.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-root", type=Path, default=default_session_root())


def run_checked(command: list[str | Path], *, cwd: Path, dry_run: bool) -> None:
    text = " ".join(str(part) for part in command)
    print(f"  command: {text}")
    if dry_run:
        return
    completed = subprocess.run([str(part) for part in command], cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {text}")


def latest_matching_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def resolve_features_csv(session_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise RuntimeError(f"Features CSV not found: {path}")
        return path
    path = latest_matching_file(session_root / "features", "*/neural_object_features.csv")
    if path is None:
        raise RuntimeError(
            f"Could not find neural_object_features.csv under {session_root / 'features'}. "
            "Run the feature logging pass first or pass --features-csv explicitly."
        )
    return path.resolve()


def resolve_master_labels_csv(session_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return (session_root / "active_learning" / "labels_master.csv").resolve()


def resolve_teacher_csv(session_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return (session_root / "active_learning" / "teacher" / "rayhit_teacher_labels.csv").resolve()


def ensure_merged_labels(master_csv: Path, seed_csvs: Iterable[Path], *, dry_run: bool) -> Path:
    seed_paths = [path.resolve() for path in seed_csvs]
    if master_csv.is_file() and not seed_paths:
        return master_csv

    if not master_csv.is_file() and not seed_paths:
        raise RuntimeError(
            f"No master labels CSV exists at {master_csv}. Provide --seed-labels-csv to initialize the active-learning loop."
        )

    merge_inputs: list[Path] = []
    if master_csv.is_file():
        merge_inputs.append(master_csv)
    merge_inputs.extend(seed_paths)
    for path in merge_inputs:
        if not path.is_file():
            raise RuntimeError(f"Label CSV not found: {path}")

    if dry_run:
        print(f"  dry-run: would merge {len(merge_inputs)} label CSV(s) into {master_csv}")
        return master_csv

    import pandas as pd

    frames = []
    for path in merge_inputs:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    keep_cols = [col for col in ["scene_variant", "strategy", "clip_id", "object_id"] if col in merged.columns]
    if not keep_cols:
        raise RuntimeError("Could not merge labels because the key columns are missing.")
    merged = merged.drop_duplicates(subset=keep_cols, keep="last").reset_index(drop=True)
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(master_csv, index=False)
    return master_csv


def next_round_dir(session_root: Path, round_name: str | None) -> Path:
    active_root = session_root / "active_learning"
    active_root.mkdir(parents=True, exist_ok=True)
    if round_name:
        return active_root / round_name

    existing_ids = []
    for path in active_root.glob("round_*"):
        suffix = path.name.removeprefix("round_")
        if suffix.isdigit():
            existing_ids.append(int(suffix))
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    return active_root / f"round_{next_id:03d}"


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def copy_latest_artifact(source: Path, destination: Path, *, dry_run: bool) -> None:
    print(f"  promote: {source} -> {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def cmd_bootstrap_manifest(args: argparse.Namespace) -> int:
    session_root = args.session_root.resolve()
    features_csv = resolve_features_csv(session_root, args.features_csv)
    output = args.output.resolve() if args.output else (session_root / "active_learning" / "bootstrap" / "bootstrap_manifest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        repo_root() / "Benchmarks" / "generate_neural_ablation_manifest.py",
        features_csv,
        "--output",
        output,
        "--top-visible",
        str(args.top_visible),
        "--top-hit",
        str(args.top_hit),
        "--top-toggle",
        str(args.top_toggle),
        "--random-controls",
        str(args.random_controls),
        "--seed",
        str(args.seed),
    ]

    print("Bootstrap manifest")
    print(f"  session root: {session_root}")
    print(f"  features csv: {features_csv}")
    print(f"  output: {output}")
    run_checked(command, cwd=repo_root(), dry_run=args.dry_run)
    return 0


def cmd_active_round(args: argparse.Namespace) -> int:
    session_root = args.session_root.resolve()
    features_csv = resolve_features_csv(session_root, args.features_csv)
    master_labels_csv = ensure_merged_labels(
        resolve_master_labels_csv(session_root, args.master_labels_csv),
        args.seed_labels_csv,
        dry_run=args.dry_run,
    )
    round_dir = next_round_dir(session_root, args.round_name)
    selector_dir = round_dir / "selector"
    cgvqm_dir = round_dir / "cgvqm"
    model_dir = round_dir / "model"
    ablation_root = args.ablation_root.resolve() if args.ablation_root else (session_root / "ablations")

    manifest_csv = selector_dir / "active_manifest.csv"
    candidates_csv = selector_dir / "active_candidates.csv"
    selector_summary_json = selector_dir / "active_summary.json"
    round_scores_csv = cgvqm_dir / "cgvqm_scores_round.csv"
    round_labels_csv = cgvqm_dir / "neural_training_labels_round.csv"
    exported_json = model_dir / "unified_neural_mlp.json"
    latest_root = session_root / "active_learning" / "latest"

    selector_dir.mkdir(parents=True, exist_ok=True)
    cgvqm_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    ablation_root.mkdir(parents=True, exist_ok=True)

    print("Active learning round")
    print(f"  session root: {session_root}")
    print(f"  round dir: {round_dir}")
    print(f"  features csv: {features_csv}")
    print(f"  master labels: {master_labels_csv}")
    print(f"  ablation root: {ablation_root}")

    selector_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "select_active_learning_manifest.py",
        "--features-csv",
        features_csv,
        "--labels-csv",
        master_labels_csv,
        "--output",
        manifest_csv,
        "--candidates-out",
        candidates_csv,
        "--summary-out",
        selector_summary_json,
        "--feature-set",
        args.feature_set,
        "--group-column",
        args.group_column,
        "--batch-size",
        str(args.batch_size),
        "--per-clip-limit",
        str(args.per_clip_limit),
        "--committee-seeds",
        args.committee_seeds,
        "--hidden-dims",
        args.selector_hidden_dims,
        "--dropout",
        str(args.selector_dropout),
        "--lr",
        str(args.selector_lr),
        "--weight-decay",
        str(args.selector_weight_decay),
        "--epochs",
        str(args.selector_epochs),
        "--patience",
        str(args.selector_patience),
        "--device",
        args.selector_device,
        "--alpha",
        str(args.selector_alpha),
        "--uncertainty-weight",
        str(args.uncertainty_weight),
        "--disagreement-weight",
        str(args.disagreement_weight),
        "--impact-weight",
        str(args.impact_weight),
        "--visibility-weight",
        str(args.visibility_weight),
        "--min-visible-fraction",
        str(args.min_visible_fraction),
        "--min-total-hits",
        str(args.min_total_hits),
        "--seed",
        str(args.seed),
    ]
    if not args.keep_empty_candidates:
        selector_cmd.append("--drop-empty-candidates")

    print("\nStep 1: Select informative candidates")
    run_checked(selector_cmd, cwd=repo_root(), dry_run=args.dry_run)

    ablation_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "run_neural_ablation_batch.py",
        manifest_csv,
        "--app",
        args.app,
        "--output-root",
        ablation_root,
        "--ablation-scene",
        args.scene,
        "--max-frames",
        str(args.max_frames),
        "--capture-exr",
        "--capture-interval",
        str(args.capture_interval),
    ]
    if not args.no_defer_oidn:
        ablation_cmd.append("--defer-oidn")

    print("\nStep 2: Run only the selected ablations")
    run_checked(ablation_cmd, cwd=repo_root(), dry_run=args.dry_run)

    labels_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "run_neural_cgvqm_labels.py",
        "--session-root",
        session_root,
        "--manifest-csv",
        manifest_csv,
        "--scene",
        args.scene,
        "--ablation-root",
        ablation_root,
        "--max-frames",
        str(args.max_frames),
        "--capture-interval",
        str(args.capture_interval),
        "--video-fps",
        str(args.video_fps),
        "--window-mode",
        args.window_mode,
        "--window-padding-frames",
        str(args.window_padding_frames),
        "--representation",
        args.representation,
        "--cgvqm-python",
        args.cgvqm_python,
        "--cgvqm-script",
        args.cgvqm_script,
        "--cgvqm-type",
        args.cgvqm_type,
        "--cgvqm-device",
        args.cgvqm_device,
        "--patch-scale",
        str(args.patch_scale),
        "--patch-pool",
        args.patch_pool,
        "--labels-csv-out",
        round_labels_csv,
        "--scores-csv-out",
        round_scores_csv,
    ]
    if args.baseline_run_dir is not None:
        labels_cmd.extend(["--baseline-run-dir", args.baseline_run_dir])
    elif args.baseline_root is not None:
        labels_cmd.extend(["--baseline-root", args.baseline_root])
    else:
        labels_cmd.extend(["--app", args.app])
    if not args.no_backfill_denoise:
        labels_cmd.append("--backfill-denoise")
    if args.skip_existing:
        labels_cmd.append("--skip-existing")

    print("\nStep 3: Score the selected ablations with CGVQM")
    run_checked(labels_cmd, cwd=repo_root(), dry_run=args.dry_run)

    merge_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "merge_neural_label_csvs.py",
        master_labels_csv,
        round_labels_csv,
        "--output",
        master_labels_csv,
    ]

    print("\nStep 4: Merge new labels into the master training table")
    run_checked(merge_cmd, cwd=repo_root(), dry_run=args.dry_run)

    train_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "train_unified_neural_mlp.py",
        "--csv",
        master_labels_csv,
        "--output-dir",
        model_dir,
        "--hidden-dims",
        args.train_hidden_dims,
        "--dropout",
        str(args.train_dropout),
        "--lr",
        str(args.train_lr),
        "--weight-decay",
        str(args.train_weight_decay),
        "--epochs",
        str(args.train_epochs),
        "--patience",
        str(args.train_patience),
        "--device",
        args.train_device,
        "--seed",
        str(args.seed),
        "--log1p-target",
    ]
    if args.train_init_model is not None:
        train_cmd.extend(["--init-model", args.train_init_model.resolve()])

    print("\nStep 5: Retrain the tiny MLP on the expanded dataset")
    run_checked(train_cmd, cwd=repo_root(), dry_run=args.dry_run)

    export_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "export_unified_neural_model.py",
        "--model",
        model_dir / "full_model.pt",
        "--output",
        exported_json,
    ]

    print("\nStep 6: Export renderer-ready model weights")
    run_checked(export_cmd, cwd=repo_root(), dry_run=args.dry_run)

    if not args.dry_run:
        copy_latest_artifact(master_labels_csv, latest_root / "labels_master.csv", dry_run=False)
        copy_latest_artifact(manifest_csv, latest_root / "active_manifest.csv", dry_run=False)
        copy_latest_artifact(candidates_csv, latest_root / "active_candidates.csv", dry_run=False)
        copy_latest_artifact(model_dir / "full_model.pt", latest_root / "full_model.pt", dry_run=False)
        copy_latest_artifact(exported_json, latest_root / "unified_neural_mlp.json", dry_run=False)

        summary = {
            "round_dir": str(round_dir),
            "features_csv": str(features_csv),
            "master_labels_csv": str(master_labels_csv),
            "manifest_csv": str(manifest_csv),
            "candidates_csv": str(candidates_csv),
            "round_scores_csv": str(round_scores_csv),
            "round_labels_csv": str(round_labels_csv),
            "model_dir": str(model_dir),
            "exported_json": str(exported_json),
            "selected_count": count_csv_rows(manifest_csv),
            "new_label_count": count_csv_rows(round_labels_csv),
            "master_label_count": count_csv_rows(master_labels_csv),
            "batch_size_requested": args.batch_size,
            "per_clip_limit": args.per_clip_limit,
            "scene": args.scene,
            "window_mode": args.window_mode,
            "representation": args.representation,
        }
        (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))

        print("\nRound complete.")
        print(f"  selected candidates: {summary['selected_count']}")
        print(f"  newly scored labels: {summary['new_label_count']}")
        print(f"  master labels now: {summary['master_label_count']}")
        print(f"  exported model: {exported_json}")
        print(f"  round summary: {round_dir / 'round_summary.json'}")
    else:
        print("\nDry run complete.")

    return 0


def cmd_retrain_export(args: argparse.Namespace) -> int:
    session_root = args.session_root.resolve()
    labels_csv = args.labels_csv.resolve() if args.labels_csv else resolve_master_labels_csv(session_root, None)
    output_dir = args.output_dir.resolve() if args.output_dir else (session_root / "active_learning" / "retrain_export")
    exported_json = output_dir / "unified_neural_mlp.json"

    print("Retrain and export")
    print(f"  labels csv: {labels_csv}")
    print(f"  output dir: {output_dir}")

    train_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "train_unified_neural_mlp.py",
        "--csv",
        labels_csv,
        "--output-dir",
        output_dir,
        "--hidden-dims",
        args.train_hidden_dims,
        "--dropout",
        str(args.train_dropout),
        "--lr",
        str(args.train_lr),
        "--weight-decay",
        str(args.train_weight_decay),
        "--epochs",
        str(args.train_epochs),
        "--patience",
        str(args.train_patience),
        "--device",
        args.train_device,
        "--seed",
        str(args.seed),
        "--log1p-target",
    ]
    if args.init_model is not None:
        train_cmd.extend(["--init-model", args.init_model.resolve()])
    run_checked(train_cmd, cwd=repo_root(), dry_run=args.dry_run)

    export_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "export_unified_neural_model.py",
        "--model",
        output_dir / "full_model.pt",
        "--output",
        exported_json,
    ]
    run_checked(export_cmd, cwd=repo_root(), dry_run=args.dry_run)
    return 0


def cmd_teacher_pretrain(args: argparse.Namespace) -> int:
    session_root = args.session_root.resolve()
    features_csv = resolve_features_csv(session_root, args.features_csv)
    teacher_csv = resolve_teacher_csv(session_root, args.teacher_csv)
    output_dir = args.output_dir.resolve() if args.output_dir else (session_root / "active_learning" / "teacher" / "model")
    exported_json = output_dir / "unified_neural_mlp.json"

    teacher_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Teacher pretrain")
    print(f"  session root: {session_root}")
    print(f"  features csv: {features_csv}")
    print(f"  teacher csv: {teacher_csv}")
    print(f"  output dir: {output_dir}")

    build_teacher_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "build_rayhit_teacher_labels.py",
        "--features-csv",
        features_csv,
        "--output",
        teacher_csv,
        "--target-column",
        args.target_column,
        "--cost-exponent",
        str(args.cost_exponent),
        "--visibility-weight",
        str(args.visibility_weight),
        "--hit-probability-weight",
        str(args.hit_probability_weight),
    ]
    print("\nStep 1: Build cheap rayhit teacher labels")
    run_checked(build_teacher_cmd, cwd=repo_root(), dry_run=args.dry_run)

    train_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "train_unified_neural_mlp.py",
        "--csv",
        teacher_csv,
        "--target",
        args.target_column,
        "--output-dir",
        output_dir,
        "--hidden-dims",
        args.train_hidden_dims,
        "--dropout",
        str(args.train_dropout),
        "--lr",
        str(args.train_lr),
        "--weight-decay",
        str(args.train_weight_decay),
        "--epochs",
        str(args.train_epochs),
        "--patience",
        str(args.train_patience),
        "--device",
        args.train_device,
        "--seed",
        str(args.seed),
    ]
    print("\nStep 2: Pretrain the tiny MLP on the rayhit teacher target")
    run_checked(train_cmd, cwd=repo_root(), dry_run=args.dry_run)

    export_cmd = [
        sys.executable,
        repo_root() / "Benchmarks" / "export_unified_neural_model.py",
        "--model",
        output_dir / "full_model.pt",
        "--output",
        exported_json,
    ]
    print("\nStep 3: Export renderer-ready teacher-pretrained weights")
    run_checked(export_cmd, cwd=repo_root(), dry_run=args.dry_run)

    if not args.dry_run:
        summary = {
            "features_csv": str(features_csv),
            "teacher_csv": str(teacher_csv),
            "target_column": args.target_column,
            "output_dir": str(output_dir),
            "exported_json": str(exported_json),
        }
        (output_dir / "teacher_pretrain_summary.json").write_text(json.dumps(summary, indent=2))
        print("\nTeacher pretrain complete.")
        print(f"  teacher csv: {teacher_csv}")
        print(f"  checkpoint: {output_dir / 'full_model.pt'}")
        print(f"  exported json: {exported_json}")
    else:
        print("\nDry run complete.")
    return 0


def cmd_residual_train(args: argparse.Namespace) -> int:
    session_root = args.session_root.resolve()
    labels_csv = args.labels_csv.resolve() if args.labels_csv else (session_root / "cgvqm" / "neural_training_labels_50.csv")
    teacher_csv = resolve_teacher_csv(session_root, args.teacher_csv)
    output_dir = args.output_dir.resolve() if args.output_dir else (session_root / "active_learning" / "residual_model")

    print("Residual train")
    print(f"  session root: {session_root}")
    print(f"  labels csv: {labels_csv}")
    print(f"  teacher csv: {teacher_csv}")
    print(f"  output dir: {output_dir}")

    command = [
        sys.executable,
        repo_root() / "Benchmarks" / "train_unified_neural_residual.py",
        "--labels-csv",
        labels_csv,
        "--teacher-csv",
        teacher_csv,
        "--teacher-column",
        args.teacher_column,
        "--output-dir",
        output_dir,
        "--hidden-dims",
        args.train_hidden_dims,
        "--dropout",
        str(args.train_dropout),
        "--lr",
        str(args.train_lr),
        "--weight-decay",
        str(args.train_weight_decay),
        "--epochs",
        str(args.train_epochs),
        "--patience",
        str(args.train_patience),
        "--device",
        args.train_device,
        "--seed",
        str(args.seed),
    ]
    if args.init_model is not None:
        command.extend(["--init-model", args.init_model.resolve()])

    run_checked(command, cwd=repo_root(), dry_run=args.dry_run)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "bootstrap-manifest":
        return cmd_bootstrap_manifest(args)
    if args.command == "active-round":
        return cmd_active_round(args)
    if args.command == "retrain-export":
        return cmd_retrain_export(args)
    if args.command == "teacher-pretrain":
        return cmd_teacher_pretrain(args)
    if args.command == "residual-train":
        return cmd_residual_train(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
