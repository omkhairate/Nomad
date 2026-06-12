#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SCENE = "scene_bistro_test_v2.xml"
SCORE_PATTERN = re.compile(r"CGVQM score:\s*([0-9.+\-eE]+)")
JOB_NAME_PATTERN = re.compile(
    r"^(?P<scene>.+)__always_resident__clip(?P<clip>\d+)__obj(?P<object>\d+)$"
)


@dataclass(frozen=True)
class ManifestEntry:
    row: dict[str, str]
    clip_id: int
    object_id: int
    clip_start_frame: int
    clip_end_frame: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_cgvqm_script = repo_root / "Benchmarks" / "cgvqm" / "run_metric.py"
    default_denoise_script = repo_root / "Benchmarks" / "denoise_exr_batch.py"
    default_sweep_script = repo_root / "Benchmarks" / "run_bistro_sweep.py"

    parser = argparse.ArgumentParser(
        description=(
            "Create CGVQM-ready labels for rigorous AlwaysResident ablation runs: "
            "optionally capture a baseline reference run, backfill denoise outputs, "
            "encode clip videos, score them with CGVQM, and merge labels back into "
            "a training CSV."
        )
    )
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--app", type=Path, default=None)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--capture-interval", type=int, default=4)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument(
        "--window-mode",
        choices=["clip", "full"],
        default="clip",
        help="Score either the clip window from the manifest or the full run.",
    )
    parser.add_argument(
        "--window-padding-frames",
        type=int,
        default=0,
        help="Extra rendered frames to include on both sides of the clip window.",
    )
    parser.add_argument(
        "--representation",
        choices=["denoised", "raw"],
        default="denoised",
        help="Which EXR representation to encode into videos.",
    )
    parser.add_argument(
        "--backfill-denoise",
        action="store_true",
        help="Run deferred denoise on baseline/ablation runs that are missing denoised EXRs.",
    )
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing videos/scores when present.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--clip-id", action="append", type=int, default=[])
    parser.add_argument("--object-id", action="append", type=int, default=[])
    parser.add_argument("--cgvqm-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cgvqm-script", type=Path, default=default_cgvqm_script)
    parser.add_argument("--cgvqm-type", choices=["cgvqm2", "cgvqm5"], default="cgvqm2")
    parser.add_argument("--cgvqm-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--patch-scale", type=int, default=4)
    parser.add_argument("--patch-pool", choices=["mean", "max"], default="mean")
    parser.add_argument(
        "--ffmpeg-vf",
        default="format=rgb24",
        help="Filter graph used when encoding EXR frames into videos.",
    )
    parser.add_argument("--ffmpeg-codec", default="libx264rgb")
    parser.add_argument("--ffmpeg-crf", default="0")
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--denoise-script", type=Path, default=default_denoise_script)
    parser.add_argument("--sweep-script", type=Path, default=default_sweep_script)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str], cwd: Path, *, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("  command:", " ".join(str(part) for part in command))
    if dry_run:
        return None
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(str(part) for part in command)}\n"
            f"{completed.stdout}"
        )
    return completed


def latest_complete_run(root: Path, scene_name: str) -> Path | None:
    if not root.exists():
        return None
    scene_stem = Path(scene_name).stem
    candidates: list[tuple[str, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        summary_path = path / "run_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if not summary.get("complete"):
            continue
        scene_path = summary.get("scene_path", "")
        if Path(scene_path).stem != scene_stem:
            continue
        metrics_csv = summary.get("metrics_csv")
        if metrics_csv and not Path(metrics_csv).is_file():
            continue
        candidates.append((path.name, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_manifest_csv(session_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    feature_root = session_root / "features"
    manifests = sorted(feature_root.glob("*/neural_ablation_manifest.csv"))
    if not manifests:
        raise RuntimeError(f"No neural_ablation_manifest.csv found under {feature_root}")
    return manifests[-1]


def load_manifest_entries(manifest_csv: Path) -> list[ManifestEntry]:
    with manifest_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    entries: list[ManifestEntry] = []
    for row in rows:
        entries.append(
            ManifestEntry(
                row=row,
                clip_id=int(row["clip_id"]),
                object_id=int(row["object_id"]),
                clip_start_frame=int(row["clip_start_frame"]),
                clip_end_frame=int(row["clip_end_frame"]),
            )
        )
    return entries


def filter_entries(entries: list[ManifestEntry], args: argparse.Namespace) -> list[ManifestEntry]:
    selected = entries
    if args.clip_id:
        wanted = set(args.clip_id)
        selected = [entry for entry in selected if entry.clip_id in wanted]
    if args.object_id:
        wanted = set(args.object_id)
        selected = [entry for entry in selected if entry.object_id in wanted]
    if args.start_index:
        selected = selected[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def ensure_baseline_run(
    args: argparse.Namespace,
    session_root: Path,
    repo_root: Path,
) -> Path:
    if args.baseline_run_dir is not None:
        return args.baseline_run_dir.resolve()

    baseline_root = args.baseline_root.resolve() if args.baseline_root else (session_root / "baseline_capture")
    baseline_root.mkdir(parents=True, exist_ok=True)
    existing = latest_complete_run(baseline_root, args.scene)
    if existing is not None:
        return existing

    if args.app is None:
        raise RuntimeError(
            "No complete baseline capture run exists yet, and --app was not provided to create one."
        )

    capture_cmd = [
        args.cgvqm_python,
        args.sweep_script,
        "--app",
        args.app,
        "--scene",
        args.scene,
        "--output-root",
        baseline_root,
        "--max-frames",
        str(args.max_frames),
        "--capture-exr",
        "--capture-interval",
        str(args.capture_interval),
        "--defer-oidn",
    ]
    run_checked(capture_cmd, repo_root, dry_run=args.dry_run)
    if args.dry_run:
        raise RuntimeError("Dry run requested before baseline capture could be resolved.")
    latest = latest_complete_run(baseline_root, args.scene)
    if latest is None:
        raise RuntimeError(f"Baseline capture under {baseline_root} did not produce a completed run.")
    return latest


def parse_frame_index(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    if not match:
        raise RuntimeError(f"Could not parse frame number from {path.name}")
    return int(match.group(1))


def list_frame_map(run_dir: Path, representation: str) -> dict[int, Path]:
    frames_dir = run_dir / "frames"
    if not frames_dir.is_dir():
        raise RuntimeError(f"Expected frames directory at {frames_dir}")

    mapping: dict[int, Path] = {}
    if representation == "denoised":
        for path in sorted(frames_dir.glob("frame_*_denoised.exr")):
            mapping[parse_frame_index(path)] = path
        if not mapping:
            raise RuntimeError(f"No denoised EXR frames found in {frames_dir}")
        return mapping

    for path in sorted(frames_dir.glob("frame_*.exr")):
        stem = path.stem
        if stem.endswith("_albedo") or stem.endswith("_normal") or stem.endswith("_denoised"):
            continue
        mapping[parse_frame_index(path)] = path
    if not mapping:
        raise RuntimeError(f"No raw color EXR frames found in {frames_dir}")
    return mapping


def ensure_representation(
    run_dir: Path,
    representation: str,
    denoise_script: Path,
    repo_root: Path,
    *,
    dry_run: bool,
    backfill_denoise: bool,
) -> None:
    if representation != "denoised":
        return
    frames_dir = run_dir / "frames"
    has_denoised = any(frames_dir.glob("frame_*_denoised.exr"))
    if has_denoised:
        return
    if not backfill_denoise:
        raise RuntimeError(
            f"Run {run_dir} is missing denoised EXRs. Re-run with --backfill-denoise or use --representation raw."
        )
    cmd = [sys.executable, str(denoise_script), "--skip-existing", str(run_dir)]
    run_checked(cmd, repo_root, dry_run=dry_run)


def select_window_indices(
    mapping: dict[int, Path],
    entry: ManifestEntry,
    *,
    window_mode: str,
    window_padding_frames: int,
) -> list[int]:
    if window_mode == "full":
        return sorted(mapping.keys())

    start_frame = entry.clip_start_frame - window_padding_frames
    end_frame = entry.clip_end_frame + window_padding_frames
    selected = [idx for idx in sorted(mapping.keys()) if start_frame <= idx <= end_frame]
    if not selected:
        raise RuntimeError(
            f"No captured frames found in requested window {start_frame}..{end_frame} "
            f"for clip {entry.clip_id}, object {entry.object_id}."
        )
    return selected


def encode_video(
    frame_paths: list[Path],
    output_path: Path,
    *,
    video_fps: float,
    ffmpeg_vf: str,
    ffmpeg_codec: str,
    ffmpeg_crf: str,
    ffmpeg_preset: str,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if skip_existing:
            return
        output_path.unlink()

    with tempfile.TemporaryDirectory(prefix="mpt_cgvqm_frames_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        seq_dir = temp_dir / "seq"
        seq_dir.mkdir()
        for index, source in enumerate(frame_paths):
            staged = seq_dir / f"img_{index:06d}.exr"
            shutil.copy2(source, staged)

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(video_fps),
            "-i",
            str(seq_dir / "img_%06d.exr"),
            "-vf",
            ffmpeg_vf,
            "-c:v",
            ffmpeg_codec,
            "-crf",
            ffmpeg_crf,
            "-preset",
            ffmpeg_preset,
            str(output_path),
        ]
        run_checked(command, output_path.parent, dry_run=dry_run)


def run_cgvqm(
    cgvqm_python: Path,
    cgvqm_script: Path,
    *,
    dist_video: Path,
    ref_video: Path,
    metric_type: str,
    device: str,
    patch_scale: int,
    patch_pool: str,
    dry_run: bool,
) -> float:
    command = [
        cgvqm_python,
        cgvqm_script,
        "--dist",
        dist_video,
        "--ref",
        ref_video,
        "--type",
        metric_type,
        "--device",
        device,
        "--patch-scale",
        str(patch_scale),
        "--patch-pool",
        patch_pool,
    ]
    completed = run_checked(command, cgvqm_script.parent, dry_run=dry_run)
    if completed is None:
        return 0.0
    match = SCORE_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse CGVQM score from output for {dist_video.name}:\n{completed.stdout}"
        )
    return float(match.group(1))


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def probe_cgvqm_environment(cgvqm_python: Path, cgvqm_script: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    command = [cgvqm_python, cgvqm_script, "--help"]
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cgvqm_script.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CGVQM environment probe failed. The metric dependencies are likely missing.\n"
            f"Command: {' '.join(str(part) for part in command)}\n"
            f"Output:\n{completed.stdout}\n"
            "Tip: create a dedicated environment and install the CGVQM requirements "
            f"from {cgvqm_script.parent / 'requirements.txt'}."
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    session_root = args.session_root.resolve()
    manifest_csv = resolve_manifest_csv(session_root, args.manifest_csv)
    entries = filter_entries(load_manifest_entries(manifest_csv), args)
    if not entries:
        print("No manifest rows matched the requested filters.", file=sys.stderr)
        return 2

    ablation_root = session_root / "ablations"
    output_root = session_root / "cgvqm"
    videos_root = output_root / "videos"
    labels_csv = output_root / "neural_training_labels.csv"
    scores_csv = output_root / "cgvqm_scores.csv"

    baseline_run_dir = ensure_baseline_run(args, session_root, repo_root)
    ensure_representation(
        baseline_run_dir,
        args.representation,
        args.denoise_script,
        repo_root,
        dry_run=args.dry_run,
        backfill_denoise=args.backfill_denoise,
    )
    baseline_map = list_frame_map(baseline_run_dir, args.representation)

    if not args.encode_only:
        probe_cgvqm_environment(args.cgvqm_python, args.cgvqm_script, dry_run=args.dry_run)

    score_rows: list[dict[str, str | int | float]] = []
    merged_rows: list[dict[str, str | int | float]] = []
    baseline_cache: dict[tuple[int, int, int], tuple[Path, float]] = {}
    skipped_missing_jobs = 0
    skipped_incomplete_jobs = 0

    print("CGVQM labeling run")
    print(f"  session root: {session_root}")
    print(f"  manifest: {manifest_csv}")
    print(f"  baseline run: {baseline_run_dir}")
    print(f"  ablation root: {ablation_root}")
    print(f"  entries: {len(entries)}")

    for index, entry in enumerate(entries, start=1):
        job_name = f"{Path(args.scene).stem}__always_resident__clip{entry.clip_id:04d}__obj{entry.object_id:05d}"
        job_dir = ablation_root / job_name
        if not job_dir.is_dir():
            skipped_missing_jobs += 1
            print(f"[{index}/{len(entries)}] skipping missing ablation job: {job_name}", file=sys.stderr)
            continue
        run_dirs = sorted([path for path in job_dir.iterdir() if path.is_dir()])
        if not run_dirs:
            skipped_missing_jobs += 1
            print(f"[{index}/{len(entries)}] skipping empty ablation job directory: {job_name}", file=sys.stderr)
            continue
        run_dir = run_dirs[-1]
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            skipped_incomplete_jobs += 1
            print(f"[{index}/{len(entries)}] skipping run without summary: {run_dir}", file=sys.stderr)
            continue
        summary = json.loads(summary_path.read_text())
        if not summary.get("complete"):
            skipped_incomplete_jobs += 1
            print(f"[{index}/{len(entries)}] skipping incomplete ablation run: {run_dir}", file=sys.stderr)
            continue

        ensure_representation(
            run_dir,
            args.representation,
            args.denoise_script,
            repo_root,
            dry_run=args.dry_run,
            backfill_denoise=args.backfill_denoise,
        )
        ablation_map = list_frame_map(run_dir, args.representation)

        window_key = (
            entry.clip_id,
            entry.clip_start_frame if args.window_mode == "clip" else 0,
            entry.clip_end_frame if args.window_mode == "clip" else args.max_frames - 1,
        )
        if window_key not in baseline_cache:
            baseline_indices = select_window_indices(
                baseline_map,
                entry,
                window_mode=args.window_mode,
                window_padding_frames=args.window_padding_frames,
            )
            baseline_video = videos_root / f"baseline_clip{entry.clip_id:04d}_{args.representation}.mkv"
            encode_video(
                [baseline_map[idx] for idx in baseline_indices],
            baseline_video,
            video_fps=args.video_fps,
            ffmpeg_vf=args.ffmpeg_vf,
            ffmpeg_codec=args.ffmpeg_codec,
            ffmpeg_crf=args.ffmpeg_crf,
            ffmpeg_preset=args.ffmpeg_preset,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
            baseline_score = 100.0 if args.encode_only else run_cgvqm(
                args.cgvqm_python,
                args.cgvqm_script,
                dist_video=baseline_video,
                ref_video=baseline_video,
                metric_type=args.cgvqm_type,
                device=args.cgvqm_device,
                patch_scale=args.patch_scale,
                patch_pool=args.patch_pool,
                dry_run=args.dry_run,
            )
            baseline_cache[window_key] = (baseline_video, baseline_score)

        baseline_video, baseline_score = baseline_cache[window_key]
        ablation_indices = select_window_indices(
            ablation_map,
            entry,
            window_mode=args.window_mode,
            window_padding_frames=args.window_padding_frames,
        )
        distorted_video = videos_root / f"ablation_clip{entry.clip_id:04d}_obj{entry.object_id:05d}_{args.representation}.mkv"
        encode_video(
            [ablation_map[idx] for idx in ablation_indices],
            distorted_video,
            video_fps=args.video_fps,
            ffmpeg_vf=args.ffmpeg_vf,
            ffmpeg_codec=args.ffmpeg_codec,
            ffmpeg_crf=args.ffmpeg_crf,
            ffmpeg_preset=args.ffmpeg_preset,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

        if args.encode_only:
            ablation_score = 0.0
        else:
            ablation_score = run_cgvqm(
                args.cgvqm_python,
                args.cgvqm_script,
                dist_video=distorted_video,
                ref_video=baseline_video,
                metric_type=args.cgvqm_type,
                device=args.cgvqm_device,
                patch_scale=args.patch_scale,
                patch_pool=args.patch_pool,
                dry_run=args.dry_run,
            )

        delta = baseline_score - ablation_score
        score_row: dict[str, str | int | float] = {
            "scene_variant": entry.row.get("scene_variant", args.scene),
            "clip_id": entry.clip_id,
            "clip_start_frame": entry.clip_start_frame,
            "clip_end_frame": entry.clip_end_frame,
            "object_id": entry.object_id,
            "job_name": job_name,
            "baseline_run_dir": str(baseline_run_dir),
            "ablation_run_dir": str(run_dir),
            "reference_video": str(baseline_video),
            "distorted_video": str(distorted_video),
            "representation": args.representation,
            "window_mode": args.window_mode,
            "window_padding_frames": args.window_padding_frames,
            "captured_frame_count": len(ablation_indices),
            "baseline_cgvqm_score": baseline_score,
            "ablation_cgvqm_score": ablation_score,
            "delta_cgvqm": delta,
        }
        score_rows.append(score_row)

        merged_row: dict[str, str | int | float] = dict(entry.row)
        merged_row.update(score_row)
        merged_rows.append(merged_row)
        print(
            f"[{index}/{len(entries)}] clip {entry.clip_id} object {entry.object_id}: "
            f"baseline={baseline_score:.4f} ablation={ablation_score:.4f} delta={delta:.4f}"
        )

    if not args.dry_run:
        write_csv(scores_csv, score_rows)
        write_csv(labels_csv, merged_rows)

    print("\nCGVQM labeling complete.")
    print(f"  baseline run: {baseline_run_dir}")
    print(f"  scores csv: {scores_csv}")
    print(f"  training labels csv: {labels_csv}")
    print(f"  scored entries: {len(score_rows)}")
    print(f"  skipped missing jobs: {skipped_missing_jobs}")
    print(f"  skipped incomplete jobs: {skipped_incomplete_jobs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
