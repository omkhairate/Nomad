#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path


BENCHMARKS_DIR = Path("/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks")
BACKUP_DIR = BENCHMARKS_DIR / "backup_ready_20260504"

KEEP_EXTENSIONS = {".csv", ".json", ".log", ".md", ".txt", ".svg", ".png"}
TOP_LEVEL_COPY_EXTENSIONS = {".py", ".md", ".log"}
MAX_TOP_LEVEL_METRICS_BYTES = 100 * 1024 * 1024
MAX_STRIPPED_FILE_BYTES = 100 * 1024 * 1024


ANALYSIS_DIRS = [
    "baseline_memory_excess_analysis",
    "bistro_sweep_analysis",
    "bistro_sweep_analysis_vs_baseline",
    "comparative_bistro_baseline_all",
    "comparative_bistro_baseline_readable",
    "comparison_test",
    "distancexalwaysbistro",
    "distancexalwaysbistrofinal",
    "metrics_20260320_114433_analysis",
    "observer_baseline_excess_analysis",
    "rayhitxalwaysbistro",
    "rayhitxalwaysbistrofinal",
]

RUN_DIRS_TO_STRIP = [
    "alwaysresidentbistro09",
    "alwaysresidentbistrorestir09",
    "bistro_baseline_clean",
    "bistro_baseline_clean_v2",
    "bistro_distance_newtrim",
    "bistro_observer_sweep_runs",
    "bistro_observer_sweep_runs_topdown_fixed",
    "bistro_sweep_runs",
    "distancebistro09",
    "distancebistrorestir09",
    "frames",
    "rayhitbistro09",
    "rayhitrestir09",
    "run1",
    "run2",
    "run3",
    "run4",
    "run5",
]

DELETE_TOP_LEVEL_GLOBS = [
    "metrics_*.csv",
]

CGVQM_HEAVY_SUBDIRS = [".venv", ".torch", ".git", "media", "__pycache__"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def move_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def copy_tree_filtered(src: Path, dst: Path, keep_extensions: set[str]) -> int:
    copied = 0
    if not src.exists():
        return copied
    for file_path in src.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in keep_extensions:
            continue
        try:
            if file_path.stat().st_size > MAX_STRIPPED_FILE_BYTES:
                continue
        except OSError:
            continue
        relative = file_path.relative_to(src)
        target = dst / relative
        copy_file(file_path, target)
        copied += 1
    return copied


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prepare_backup() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    scripts_dir = BACKUP_DIR / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    for file_path in BENCHMARKS_DIR.iterdir():
        if not file_path.is_file():
            continue
        if file_path.name == ".DS_Store":
            continue
        if file_path.suffix.lower() in TOP_LEVEL_COPY_EXTENSIONS:
            copy_file(file_path, scripts_dir / file_path.name)
        elif file_path.name.startswith("metrics_") and file_path.suffix.lower() == ".csv":
            if file_path.stat().st_size <= MAX_TOP_LEVEL_METRICS_BYTES:
                copy_file(file_path, BACKUP_DIR / "top_level_metrics" / file_path.name)

    for dirname in ANALYSIS_DIRS:
        src = BENCHMARKS_DIR / dirname
        if src.exists():
            move_tree(src, BACKUP_DIR / "analysis" / dirname)

    for dirname in RUN_DIRS_TO_STRIP:
        src = BENCHMARKS_DIR / dirname
        if src.exists():
            copy_tree_filtered(src, BACKUP_DIR / "run_data" / dirname, KEEP_EXTENSIONS)

    cgvqm_src = BENCHMARKS_DIR / "cgvqm"
    if cgvqm_src.exists():
        cgvqm_dst = BACKUP_DIR / "tools" / "cgvqm"
        for item in cgvqm_src.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(cgvqm_src)
            if any(part in CGVQM_HEAVY_SUBDIRS for part in relative.parts):
                continue
            copy_file(item, cgvqm_dst / relative)


def cleanup_originals() -> None:
    for dirname in RUN_DIRS_TO_STRIP:
        remove_path(BENCHMARKS_DIR / dirname)

    for pattern in DELETE_TOP_LEVEL_GLOBS:
        for file_path in BENCHMARKS_DIR.glob(pattern):
            if file_path.is_file():
                remove_path(file_path)

    for dirname in ANALYSIS_DIRS:
        remove_path(BENCHMARKS_DIR / dirname)

    cgvqm_dir = BENCHMARKS_DIR / "cgvqm"
    if cgvqm_dir.exists():
        for subdir in CGVQM_HEAVY_SUBDIRS:
            remove_path(cgvqm_dir / subdir)

    remove_path(BENCHMARKS_DIR / "__pycache__")
    remove_path(BENCHMARKS_DIR / ".DS_Store")


def report_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def main() -> None:
    prepare_backup()
    cleanup_originals()
    backup_bytes = report_size(BACKUP_DIR)
    print(f"Backup folder ready: {BACKUP_DIR}")
    print(f"Backup size bytes: {backup_bytes}")


if __name__ == "__main__":
    main()
