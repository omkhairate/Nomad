#!/usr/bin/env python3
"""
Batch-run the Bistro residency-study scenes one by one and collect their CSVs.

Why this script exists:
- The renderer loads ``scene.xml`` at startup rather than taking a scene path
  on the command line.
- Benchmark CSVs are written to ``Benchmarks/metrics_*.csv`` in the repo root.
- Additional per-frame logs (``perf.csv``, ``gpu_mem.csv``, AS dumps) can be
  redirected into a dedicated run directory using ``MPT_RUNS_PATH``.

This script temporarily swaps ``scene.xml`` with each Bistro study variant,
launches the app, waits for it to terminate, then copies the newly created
benchmark CSV into the run directory for that scene.

Resume behavior:
- The newest run folder for each scene acts as that scene's checkpoint.
- If the newest run has a successful summary and a copied metrics CSV, that
  scene is skipped on the next invocation.
- If the newest run is incomplete or failed, that scene is rerun and the sweep
  continues from there.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_BISTRO_OBSERVER_POSITION = "0,0,52"
DEFAULT_BISTRO_OBSERVER_LOOK_AT = "0,0,3"
DEFAULT_BISTRO_OBSERVER_UP = "0,1,0"
DEFAULT_BISTRO_OBSERVER_FOV = 70.0


@dataclass(frozen=True)
class RunResult:
    scene_path: Path
    run_dir: Path
    metrics_csv: Path | None
    return_code: int
    duration_seconds: float


@dataclass(frozen=True)
class ExistingRunState:
    scene_path: Path
    run_dir: Path
    metrics_csv: Path | None
    return_code: int | None
    complete: bool


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output = repo_root / "Benchmarks" / "bistro_sweep_runs"
    default_app = (
        repo_root
        / "build"
        / "Build"
        / "Products"
        / "Release"
        / "Nomad Path Tracer.app"
        / "Contents"
        / "MacOS"
        / "Nomad Path Tracer"
    )

    parser = argparse.ArgumentParser(
        description="Run the Bistro study scenes sequentially and collect CSV outputs."
    )
    parser.add_argument(
        "--app",
        type=Path,
        default=default_app,
        help=(
            "Path to the renderer executable. Defaults to the usual Xcode Release "
            "bundle location under build/Build/Products/Release."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Directory where per-scene run folders will be created.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Value passed to MPT_MAX_FRAMES for each run. Default: 500",
    )
    parser.add_argument(
        "--capture-exr",
        action="store_true",
        help="Enable EXR capture during the sweep (disabled by default).",
    )
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=4,
        help="Frame capture interval when --capture-exr is enabled. Default: 4",
    )
    parser.add_argument(
        "--log-neural-features",
        action="store_true",
        help="Write clip-level object features for neural residency training.",
    )
    parser.add_argument(
        "--log-unified-neural-scores",
        action="store_true",
        help=(
            "Write per-frame UnifiedNeural score breakdowns "
            "(heuristic/neural/blended) to unified_neural_scores.csv."
        ),
    )
    parser.add_argument(
        "--neural-clip-length",
        type=int,
        default=16,
        help="Clip length used when --log-neural-features is enabled. Default: 16",
    )
    parser.add_argument(
        "--force-object-off",
        type=int,
        default=None,
        help="Force one object id off for the whole run (used for ablation data generation).",
    )
    parser.add_argument(
        "--defer-oidn",
        action="store_true",
        help=(
            "Capture raw EXRs during the sweep and run OIDN only after the "
            "entire sweep finishes successfully."
        ),
    )
    parser.add_argument(
        "--thermal-log",
        action="store_true",
        help="Run Benchmarks/thermal_logger.py beside each scene and write thermal.csv into each run folder.",
    )
    parser.add_argument(
        "--thermal-interval-ms",
        type=int,
        default=2000,
        help="Thermal logger sampling interval in milliseconds. Default: 2000",
    )
    parser.add_argument(
        "--include-colored",
        action="store_true",
        help="Include the colorized always-resident Bistro variant in the run list.",
    )
    parser.add_argument(
        "--decision-observer",
        action="store_true",
        help=(
            "Force observer-camera capture with residency preview materials so "
            "saved frames show onload/offload decisions from an elevated side view."
        ),
    )
    parser.add_argument(
        "--observer-position",
        default=DEFAULT_BISTRO_OBSERVER_POSITION,
        help=(
            "Observer camera position as x,y,z when --decision-observer is enabled. "
            f"Default: {DEFAULT_BISTRO_OBSERVER_POSITION}"
        ),
    )
    parser.add_argument(
        "--observer-look-at",
        default=DEFAULT_BISTRO_OBSERVER_LOOK_AT,
        help=(
            "Observer camera look-at point as x,y,z when --decision-observer is enabled. "
            f"Default: {DEFAULT_BISTRO_OBSERVER_LOOK_AT}"
        ),
    )
    parser.add_argument(
        "--observer-up",
        default=DEFAULT_BISTRO_OBSERVER_UP,
        help=(
            "Observer camera up vector as x,y,z when --decision-observer is enabled. "
            f"Default: {DEFAULT_BISTRO_OBSERVER_UP}"
        ),
    )
    parser.add_argument(
        "--observer-fov",
        type=float,
        default=DEFAULT_BISTRO_OBSERVER_FOV,
        help=(
            "Observer camera vertical field of view in degrees when --decision-observer "
            f"is enabled. Default: {DEFAULT_BISTRO_OBSERVER_FOV}"
        ),
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help=(
            "Explicit scene XML filename or path to run. Can be passed multiple times. "
            "If omitted, all scene_bistro_test_v2*.xml variants are used."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan without launching the executable.",
    )
    return parser.parse_args()


def find_scene_variants(repo_root: Path, include_colored: bool) -> list[Path]:
    scene_dir = repo_root / "Nomad Path Tracer"
    candidates = sorted(scene_dir.glob("scene_bistro_test_v2*.xml"))

    filtered: list[Path] = []
    for path in candidates:
        stem = path.stem
        if "colored" in stem and not include_colored:
            continue
        filtered.append(path)

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem == "scene_bistro_test_v2":
            return (0, stem)
        group_order = [
            "distance",
            "rayhit",
            "screenspace",
            "energy",
            "probabilistic",
            "unified",
            "environment",
            "predictive",
        ]
        for index, group in enumerate(group_order, start=1):
            if group in stem:
                return (index, stem)
        return (99, stem)

    return sorted(filtered, key=sort_key)


def resolve_explicit_scenes(repo_root: Path, scene_args: Sequence[str]) -> list[Path]:
    scene_dir = repo_root / "Nomad Path Tracer"
    resolved: list[Path] = []
    for value in scene_args:
        candidate = Path(value)
        if not candidate.is_absolute():
            direct = repo_root / candidate
            scene_local = scene_dir / candidate
            if direct.is_file():
                candidate = direct
            elif scene_local.is_file():
                candidate = scene_local
        if not candidate.is_file():
            raise FileNotFoundError(f"Scene XML not found: {value}")
        resolved.append(candidate.resolve())
    return resolved


def build_run_list(
    repo_root: Path, scene_args: Sequence[str], include_colored: bool
) -> list[Path]:
    if scene_args:
        return resolve_explicit_scenes(repo_root, scene_args)
    return find_scene_variants(repo_root, include_colored)


def benchmark_dirs(repo_root: Path) -> list[Path]:
    return [
        repo_root / "Benchmarks",
        repo_root / "Nomad Path Tracer" / "Benchmarks",
    ]


def list_metrics_csvs(benchmarks_dirs: Sequence[Path]) -> set[Path]:
    paths: set[Path] = set()
    for benchmarks_dir in benchmarks_dirs:
        if not benchmarks_dir.exists():
            continue
        paths.update(path.resolve() for path in benchmarks_dir.glob("metrics_*.csv"))
    return paths


def newest_new_metrics_csv(before: set[Path], after: set[Path]) -> Path | None:
    new_files = [path for path in after if path not in before]
    if not new_files:
        return None
    return max(new_files, key=lambda path: path.stat().st_mtime)


def scene_xml_paths(repo_root: Path) -> list[Path]:
    """Return every scene.xml location the app might consult at startup.

    We intentionally write both:
    - repo-root `scene.xml`, which older builds try first because the app is
      launched with cwd set to the repo root
    - app-local `Nomad Path Tracer/scene.xml`, which newer builds may use via
      fallback logic

    Keeping both in sync makes the sweep robust across builds and avoids
    silently running the wrong strategy.
    """

    return [
        repo_root / "scene.xml",
        repo_root / "Nomad Path Tracer" / "scene.xml",
    ]


def inject_observer_camera(
    scene_text: str,
    position: str,
    look_at: str,
    up: str,
    vertical_fov: float,
) -> str:
    observer_line = (
        f'  <ObserverCamera position="{position}" lookAt="{look_at}" '
        f'up="{up}" verticalFov="{vertical_fov:g}" aperture="0" />'
    )

    lines = scene_text.splitlines()
    replaced = False
    updated_lines: list[str] = []
    for line in lines:
        if "<ObserverCamera" in line:
            if not replaced:
                updated_lines.append(observer_line)
                replaced = True
            continue
        updated_lines.append(line)

    if not replaced:
        inserted = False
        for index, line in enumerate(updated_lines):
            if "<Scene" in line:
                updated_lines.insert(index + 1, observer_line)
                inserted = True
                break
        if not inserted:
            updated_lines.insert(0, observer_line)

    return "\n".join(updated_lines) + "\n"


def write_scene_override(
    target_scene_xmls: Sequence[Path],
    variant_path: Path,
    *,
    decision_observer: bool,
    observer_position: str,
    observer_look_at: str,
    observer_up: str,
    observer_fov: float,
) -> None:
    scene_text = variant_path.read_text()
    if decision_observer:
        scene_text = inject_observer_camera(
            scene_text,
            position=observer_position,
            look_at=observer_look_at,
            up=observer_up,
            vertical_fov=observer_fov,
        )
    for target_scene_xml in target_scene_xmls:
        target_scene_xml.write_text(scene_text)


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def run_summary_path(run_dir: Path) -> Path:
    return run_dir / "run_summary.json"


def write_run_summary(result: RunResult) -> None:
    summary = {
        "scene_path": str(result.scene_path),
        "run_dir": str(result.run_dir),
        "metrics_csv": str(result.metrics_csv) if result.metrics_csv else None,
        "return_code": result.return_code,
        "duration_seconds": result.duration_seconds,
        "complete": result.return_code == 0 and result.metrics_csv is not None,
    }
    run_summary_path(result.run_dir).write_text(json.dumps(summary, indent=2))


def parse_run_dir_name(run_dir: Path) -> tuple[str, str]:
    parts = run_dir.name.rsplit("_", 2)
    if len(parts) != 3:
        return run_dir.name, ""
    scene_id, date_part, time_part = parts
    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        return scene_id, f"{date_part}_{time_part}"
    return run_dir.name, ""


def scene_run_dirs(output_root: Path, scene: Path) -> list[Path]:
    matches = []
    for path in output_root.iterdir() if output_root.exists() else []:
        if not path.is_dir():
            continue
        scene_id, _timestamp = parse_run_dir_name(path)
        if scene_id == scene.stem:
            matches.append(path)
    return sorted(matches, key=lambda path: parse_run_dir_name(path)[1] or path.name)


def inspect_existing_run(scene: Path, output_root: Path) -> ExistingRunState | None:
    run_dirs = scene_run_dirs(output_root, scene)
    if not run_dirs:
        return None

    run_dir = run_dirs[-1]
    summary_file = run_summary_path(run_dir)
    if summary_file.is_file():
        try:
            summary = json.loads(summary_file.read_text())
            metrics_value = summary.get("metrics_csv")
            metrics_csv = Path(metrics_value) if metrics_value else None
            return_code = summary.get("return_code")
            complete = bool(summary.get("complete", False))
            if metrics_csv is not None and not metrics_csv.is_file():
                complete = False
            return ExistingRunState(
                scene_path=scene,
                run_dir=run_dir,
                metrics_csv=metrics_csv,
                return_code=return_code,
                complete=complete,
            )
        except Exception:
            pass

    metrics_candidates = sorted(run_dir.glob("metrics_*.csv"))
    metrics_csv = metrics_candidates[-1] if metrics_candidates else None
    complete = metrics_csv is not None
    return ExistingRunState(
        scene_path=scene,
        run_dir=run_dir,
        metrics_csv=metrics_csv,
        return_code=0 if complete else None,
        complete=complete,
    )


def resume_run_list(scenes: Sequence[Path], output_root: Path) -> tuple[list[Path], list[ExistingRunState]]:
    completed: list[ExistingRunState] = []
    pending: list[Path] = []
    resume_started = False

    for scene in scenes:
        existing = inspect_existing_run(scene, output_root)
        if not resume_started:
            if existing is not None and existing.complete:
                completed.append(existing)
                continue
            resume_started = True
        pending.append(scene)

    return pending, completed


def run_scene(
    repo_root: Path,
    app_path: Path,
    variant_path: Path,
    output_root: Path,
    max_frames: int,
    capture_exr: bool,
    capture_interval: int,
    defer_oidn: bool,
    log_neural_features: bool,
    log_unified_neural_scores: bool,
    neural_clip_length: int,
    force_object_off: int | None,
    thermal_log: bool,
    thermal_interval_ms: int,
    decision_observer: bool,
) -> RunResult:
    benchmarks_dirs = benchmark_dirs(repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{variant_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Clear sticky debug/capture vars that may have been left in the shell from
    # earlier experiments. This keeps ordinary sweep runs deterministic unless
    # the corresponding command-line flags intentionally re-enable them below.
    for key in (
        "MPT_OBSERVER_CAPTURE",
        "MPT_RESIDENCY_PREVIEW_ONLY",
        "MPT_FORCE_OBJECT_OFF",
        "MPT_DISABLE_OFFLINE_OIDN",
        "MPT_LOG_NEURAL_FEATURES",
        "MPT_LOG_UNIFIED_NEURAL_SCORES",
        "MPT_NEURAL_CLIP_LENGTH",
    ):
        env.pop(key, None)

    env["METALPT_BENCH"] = "1"
    env["METALPT_BENCH_LOG_PROBABILITIES"] = "0"
    env["MPT_MAX_FRAMES"] = str(max_frames)
    env["MPT_RUNS_PATH"] = str(run_dir)
    env["MPT_SCENE_VARIANT"] = variant_path.name
    env["MPT_CAPTURE_EXR"] = "1" if capture_exr else "0"
    env["MPT_CAPTURE_INTERVAL"] = str(max(1, capture_interval))
    env["MPT_LOG_NEURAL_FEATURES"] = "1" if log_neural_features else "0"
    env["MPT_LOG_UNIFIED_NEURAL_SCORES"] = (
        "1" if log_unified_neural_scores else "0"
    )
    env["MPT_NEURAL_CLIP_LENGTH"] = str(max(1, neural_clip_length))
    if force_object_off is not None and force_object_off >= 0:
        env["MPT_FORCE_OBJECT_OFF"] = str(force_object_off)
    if defer_oidn:
        env["MPT_DISABLE_OFFLINE_OIDN"] = "1"
    if decision_observer:
        env["MPT_OBSERVER_CAPTURE"] = "1"
        env["MPT_RESIDENCY_PREVIEW_ONLY"] = "1"
        env["MPT_DISABLE_OFFLINE_OIDN"] = "1"

    thermal_proc: subprocess.Popen | None = None
    if thermal_log:
        thermal_script = repo_root / "Benchmarks" / "thermal_logger.py"
        thermal_proc = subprocess.Popen(
            [
                sys.executable,
                str(thermal_script),
                "--out",
                str(run_dir / "thermal.csv"),
                "--raw-log",
                str(run_dir / "thermal_raw.log"),
                "--interval-ms",
                str(max(250, thermal_interval_ms)),
            ],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    before = list_metrics_csvs(benchmarks_dirs)
    launch_cwd = repo_root / "Nomad Path Tracer"
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(app_path)],
            cwd=launch_cwd,
            env=env,
            check=False,
        )
    finally:
        if thermal_proc is not None and thermal_proc.poll() is None:
            thermal_proc.terminate()
            try:
                thermal_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                thermal_proc.kill()
                thermal_proc.wait()
    duration_seconds = time.monotonic() - start
    after = list_metrics_csvs(benchmarks_dirs)

    new_metrics = newest_new_metrics_csv(before, after)
    copied_metrics: Path | None = None
    if new_metrics is not None:
        copied_metrics = run_dir / new_metrics.name
        shutil.copy2(new_metrics, copied_metrics)

    result = RunResult(
        scene_path=variant_path,
        run_dir=run_dir,
        metrics_csv=copied_metrics,
        return_code=completed.returncode,
        duration_seconds=duration_seconds,
    )
    write_run_summary(result)
    return result


def run_deferred_oidn(repo_root: Path, run_dirs: Sequence[Path]) -> None:
    if not run_dirs:
        return
    script = repo_root / "Benchmarks" / "denoise_exr_batch.py"
    cmd = [sys.executable, str(script), "--skip-existing"]
    cmd.extend(str(run_dir) for run_dir in run_dirs)
    print("\nRunning deferred OIDN denoise over captured EXRs...", flush=True)
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Deferred OIDN post-process failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {completed.returncode}"
        )


class SudoKeepAlive:
    def __init__(self, interval_seconds: float = 60.0) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def worker() -> None:
            while not self._stop.wait(self._interval_seconds):
                subprocess.run(["sudo", "-n", "-v"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self._thread = threading.Thread(target=worker, name="sudo-keepalive", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def thermal_sudo_ready() -> bool:
    probe = subprocess.run(["sudo", "-n", "true"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return probe.returncode == 0


def print_run_plan(scenes: Iterable[Path]) -> None:
    print("Bistro sweep run list:")
    for index, scene in enumerate(scenes, start=1):
        print(f"  {index:02d}. {scene.name}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    target_scenes = scene_xml_paths(repo_root)
    default_output_root = repo_root / "Benchmarks" / "bistro_sweep_runs"
    output_root = args.output_root.resolve()
    if args.defer_oidn and not args.capture_exr:
        print("--defer-oidn requires --capture-exr.", file=sys.stderr)
        return 2
    if args.decision_observer and output_root == default_output_root.resolve():
        output_root = (repo_root / "Benchmarks" / "bistro_observer_sweep_runs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = build_run_list(repo_root, args.scene, args.include_colored)
    if not scenes:
        print("No Bistro scene variants matched the selection.", file=sys.stderr)
        return 2

    resumed_scenes, completed_runs = resume_run_list(scenes, output_root)

    print_run_plan(scenes)
    print(f"\nExecutable: {args.app}")
    print(f"Output root: {output_root}")
    print(f"Max frames: {args.max_frames}")
    print(f"EXR capture: {'on' if args.capture_exr else 'off'}")
    print(f"Deferred OIDN: {'on' if args.defer_oidn else 'off'}")
    print(f"Neural feature logging: {'on' if args.log_neural_features else 'off'}")
    print(
        "UnifiedNeural score logging: "
        f"{'on' if args.log_unified_neural_scores else 'off'}"
    )
    if args.log_neural_features:
        print(f"Neural clip length: {args.neural_clip_length}")
    if args.force_object_off is not None:
        print(f"Forced object off: {args.force_object_off}")
    print(f"Thermal log: {'on' if args.thermal_log else 'off'}")
    print(f"Decision observer: {'on' if args.decision_observer else 'off'}")
    if args.decision_observer:
        print(
            "Observer camera: "
            f"position={args.observer_position} "
            f"lookAt={args.observer_look_at} "
            f"up={args.observer_up} "
            f"verticalFov={args.observer_fov:g}"
        )
    print(f"Already complete: {len(completed_runs)}")
    print(f"Remaining: {len(resumed_scenes)}")

    if completed_runs:
        print("Completed scenes detected from previous runs:")
        for state in completed_runs:
            csv_name = state.metrics_csv.name if state.metrics_csv else "missing"
            print(f"  - {state.scene_path.name} ({csv_name})")

    if resumed_scenes and len(resumed_scenes) != len(scenes):
        print("Resuming from:")
        for scene in resumed_scenes:
            print(f"  - {scene.name}")

    if args.dry_run:
        return 0

    scenes = resumed_scenes
    if not scenes:
        print("Nothing to do. All selected scenes already have completed runs.")
        return 0

    if args.thermal_log and not thermal_sudo_ready():
        print(
            "Thermal logging was requested, but sudo credentials are not currently cached.\n"
            "Please run `sudo -v` in this terminal immediately before starting the sweep.",
            file=sys.stderr,
        )
        return 2

    if not args.app.is_file():
        print(
            "Renderer executable not found.\n"
            f"Expected at: {args.app}\n"
            "Pass --app /absolute/path/to/'Nomad Path Tracer' to point at the built executable.",
            file=sys.stderr,
        )
        return 2

    metallib_path = args.app.parent / "default.metallib"
    if not metallib_path.is_file():
        print(
            "Renderer support library not found.\n"
            f"Expected Metal library beside executable at: {metallib_path}\n"
            "The renderer calls `newDefaultLibrary()`, so the executable must sit next to "
            "`default.metallib` from the same build output.\n"
            "Rebuild the app or point --app at a build directory that still contains both files.",
            file=sys.stderr,
        )
        return 2

    original_scene_backups: dict[Path, bytes | None] = {
        path: path.read_bytes() if path.exists() else None for path in target_scenes
    }
    results: list[RunResult] = []
    sudo_keepalive = SudoKeepAlive() if args.thermal_log else None

    try:
        if sudo_keepalive is not None:
            sudo_keepalive.start()
        for index, scene in enumerate(scenes, start=1):
            print(f"\n[{index}/{len(scenes)}] Running {scene.name} ...", flush=True)
            write_scene_override(
                target_scenes,
                scene,
                decision_observer=args.decision_observer,
                observer_position=args.observer_position,
                observer_look_at=args.observer_look_at,
                observer_up=args.observer_up,
                observer_fov=args.observer_fov,
            )
            result = run_scene(
                repo_root=repo_root,
                app_path=args.app.resolve(),
                variant_path=scene,
                output_root=output_root,
                max_frames=args.max_frames,
                capture_exr=args.capture_exr,
                capture_interval=args.capture_interval,
                defer_oidn=args.defer_oidn,
                log_neural_features=args.log_neural_features,
                log_unified_neural_scores=args.log_unified_neural_scores,
                neural_clip_length=args.neural_clip_length,
                force_object_off=args.force_object_off,
                thermal_log=args.thermal_log,
                thermal_interval_ms=args.thermal_interval_ms,
                decision_observer=args.decision_observer,
            )
            results.append(result)

            status = "ok" if result.return_code == 0 else f"failed({result.return_code})"
            csv_name = result.metrics_csv.name if result.metrics_csv else "missing"
            print(
                f"  status={status} duration={format_seconds(result.duration_seconds)} "
                f"metrics={csv_name}",
                flush=True,
            )

            if result.return_code != 0:
                print(
                    f"Stopping after failure in {scene.name}. "
                    f"Run artifacts remain in {result.run_dir}",
                    file=sys.stderr,
                )
                return result.return_code

        if args.capture_exr and args.defer_oidn:
            run_deferred_oidn(
                repo_root,
                [result.run_dir for result in results if result.return_code == 0],
            )

    finally:
        if sudo_keepalive is not None:
            sudo_keepalive.stop()
        for target_scene, original_scene_backup in original_scene_backups.items():
            if original_scene_backup is not None:
                target_scene.write_bytes(original_scene_backup)
            elif target_scene.exists():
                target_scene.unlink()

    print("\nSweep complete.\n")
    for result in results:
        csv_path = result.metrics_csv if result.metrics_csv else Path("<missing>")
        print(f"- {result.scene_path.name}: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
