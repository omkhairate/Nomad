#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_app = (
        repo_root
        / 'build'
        / 'Build'
        / 'Products'
        / 'Release'
        / 'Nomad Path Tracer.app'
        / 'Contents'
        / 'MacOS'
        / 'Nomad Path Tracer'
    )
    default_output = repo_root / 'Benchmarks' / 'neural_ablation_runs'

    parser = argparse.ArgumentParser(
        description=(
            'Run a batch of object-ablation Bistro jobs from a '
            'neural_ablation_manifest.csv file.'
        )
    )
    parser.add_argument('manifest_csv', type=Path)
    parser.add_argument('--app', type=Path, default=default_app)
    parser.add_argument('--output-root', type=Path, default=default_output)
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--capture-exr', action='store_true')
    parser.add_argument('--capture-interval', type=int, default=4)
    parser.add_argument('--defer-oidn', action='store_true')
    parser.add_argument('--decision-observer', action='store_true')
    parser.add_argument('--observer-position', default='0,0,52')
    parser.add_argument('--observer-look-at', default='0,0,3')
    parser.add_argument('--observer-up', default='0,1,0')
    parser.add_argument('--observer-fov', type=float, default=70.0)
    parser.add_argument('--log-neural-features', action='store_true')
    parser.add_argument('--neural-clip-length', type=int, default=16)
    parser.add_argument('--thermal-log', action='store_true')
    parser.add_argument('--thermal-interval-ms', type=int, default=2000)
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--scene', action='append', default=[])
    parser.add_argument(
        '--ablation-scene',
        default=None,
        help=(
            'Override the scene used for ablation runs. Useful for running '
            'always-resident reference ablations such as scene_bistro_test_v2.xml '
            "while still selecting objects from another strategy's manifest."
        ),
    )
    parser.add_argument('--strategy', action='append', default=[])
    parser.add_argument('--clip-id', action='append', type=int, default=[])
    parser.add_argument('--object-id', action='append', type=int, default=[])
    parser.add_argument('--min-selection-score', type=float, default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def normalize_scene_name(value: str) -> str:
    return Path(value).name


def ablation_scene_name(args: argparse.Namespace, row: pd.Series) -> str:
    if args.ablation_scene:
        return normalize_scene_name(args.ablation_scene)
    return normalize_scene_name(str(row['scene_variant']))


def job_slug(args: argparse.Namespace, row: pd.Series) -> str:
    scene_stem = Path(ablation_scene_name(args, row)).stem
    clip_id = int(row['clip_id'])
    object_id = int(row['object_id'])
    strategy = str(row.get('strategy', 'unknown')).strip().lower().replace(' ', '_')
    return f'{scene_stem}__{strategy}__clip{clip_id:04d}__obj{object_id:05d}'


def complete_run_exists(job_root: Path) -> bool:
    if not job_root.exists():
        return False
    for summary_path in job_root.glob('*/run_summary.json'):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if summary.get('complete') and summary.get('return_code') == 0 and summary.get('metrics_csv'):
            return True
    return False


def build_command(args: argparse.Namespace, row: pd.Series, job_root: Path) -> list[str]:
    script = Path(__file__).resolve().with_name('run_bistro_sweep.py')
    command = [
        sys.executable,
        str(script),
        '--app',
        str(args.app),
        '--scene',
        ablation_scene_name(args, row),
        '--output-root',
        str(job_root),
        '--max-frames',
        str(args.max_frames),
        '--force-object-off',
        str(int(row['object_id'])),
    ]
    if args.capture_exr:
        command.append('--capture-exr')
        command.extend(['--capture-interval', str(args.capture_interval)])
    if args.defer_oidn:
        command.append('--defer-oidn')
    if args.decision_observer:
        command.append('--decision-observer')
        command.extend(['--observer-position', args.observer_position])
        command.extend(['--observer-look-at', args.observer_look_at])
        command.extend(['--observer-up', args.observer_up])
        command.extend(['--observer-fov', str(args.observer_fov)])
    if args.log_neural_features:
        command.append('--log-neural-features')
        command.extend(['--neural-clip-length', str(args.neural_clip_length)])
    if args.thermal_log:
        command.append('--thermal-log')
        command.extend(['--thermal-interval-ms', str(args.thermal_interval_ms)])
    return command


def write_job_metadata(args: argparse.Namespace, job_root: Path, row: pd.Series) -> None:
    job_root.mkdir(parents=True, exist_ok=True)
    payload = {
        'manifest_scene_variant': str(row['scene_variant']),
        'ablation_scene_variant': ablation_scene_name(args, row),
        'strategy': str(row.get('strategy', '')),
        'clip_id': int(row['clip_id']),
        'clip_start_frame': int(row.get('clip_start_frame', 0)),
        'clip_end_frame': int(row.get('clip_end_frame', 0)),
        'object_id': int(row['object_id']),
        'selection_score': float(row.get('selection_score', 0.0)),
        'visible_frame_fraction': float(row.get('visible_frame_fraction', 0.0)),
        'total_object_hits': float(row.get('total_object_hits', 0.0)),
        'total_object_rays_tested': float(row.get('total_object_rays_tested', 0.0)),
        'toggle_count': float(row.get('toggle_count', 0.0)),
    }
    (job_root / 'ablation_job.json').write_text(json.dumps(payload, indent=2))


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'scene_variant' not in df.columns or 'object_id' not in df.columns or 'clip_id' not in df.columns:
        raise ValueError('Manifest must include scene_variant, clip_id, and object_id columns.')
    for column in ['object_id', 'clip_id', 'clip_start_frame', 'clip_end_frame']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
    for column in ['selection_score', 'visible_frame_fraction', 'total_object_hits', 'total_object_rays_tested', 'toggle_count']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0)
    return df


def filter_manifest(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if args.scene:
        wanted = {normalize_scene_name(value) for value in args.scene}
        filtered = filtered[filtered['scene_variant'].map(lambda v: normalize_scene_name(str(v))) .isin(wanted)]
    if args.strategy:
        wanted = {value.strip().lower() for value in args.strategy}
        filtered = filtered[filtered['strategy'].fillna('').map(lambda v: str(v).strip().lower()).isin(wanted)]
    if args.clip_id:
        wanted = set(args.clip_id)
        filtered = filtered[filtered['clip_id'].isin(wanted)]
    if args.object_id:
        wanted = set(args.object_id)
        filtered = filtered[filtered['object_id'].isin(wanted)]
    if args.min_selection_score is not None and 'selection_score' in filtered.columns:
        filtered = filtered[filtered['selection_score'] >= args.min_selection_score]

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, asc in [
        ('scene_variant', True),
        ('clip_id', True),
        ('selection_score', False),
        ('visible_frame_fraction', False),
        ('total_object_hits', False),
        ('object_id', True),
    ]:
        if column in filtered.columns:
            sort_columns.append(column)
            ascending.append(asc)
    if sort_columns:
        filtered = filtered.sort_values(sort_columns, ascending=ascending, kind='stable')

    filtered = filtered.iloc[max(0, args.start_index) :]
    if args.limit is not None:
        filtered = filtered.head(max(0, args.limit))
    return filtered.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    manifest_csv = args.manifest_csv.resolve()
    if not manifest_csv.is_file():
        print(f'Manifest not found: {manifest_csv}', file=sys.stderr)
        return 2

    df = filter_manifest(load_manifest(manifest_csv), args)
    if df.empty:
        print('No ablation jobs matched the current filters.', file=sys.stderr)
        return 2

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'Ablation manifest: {manifest_csv}')
    print(f'Jobs selected: {len(df)}')
    print(f'Output root: {output_root}')
    print(f'Capture EXR: {"on" if args.capture_exr else "off"}')
    if args.ablation_scene:
        print(f'Ablation scene override: {normalize_scene_name(args.ablation_scene)}')
    print(f'Deferred OIDN: {"on" if args.defer_oidn else "off"}')
    print(f'Continue on error: {"yes" if args.continue_on_error else "no"}')

    failures = 0
    for index, row in df.iterrows():
        slug = job_slug(args, row)
        job_root = output_root / slug
        scene_name = normalize_scene_name(str(row['scene_variant']))
        run_scene_name = ablation_scene_name(args, row)
        object_id = int(row['object_id'])
        clip_id = int(row['clip_id'])
        score = float(row.get('selection_score', 0.0))

        print(
            f'[{index + 1}/{len(df)}] manifest={scene_name} run={run_scene_name} '
            f'clip={clip_id} obj={object_id} score={score:.4f}'
        )

        if complete_run_exists(job_root):
            print(f'  skipping existing complete ablation at {job_root}')
            continue

        command = build_command(args, row, job_root)
        write_job_metadata(args, job_root, row)
        if args.dry_run:
            print('  dry-run:', ' '.join(command))
            continue

        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
        if completed.returncode != 0:
            failures += 1
            print(
                f'  failed with exit code {completed.returncode}; artifacts kept in {job_root}',
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return completed.returncode

    if args.dry_run:
        print('Dry run complete.')
        return 0

    if failures:
        print(f'Ablation batch finished with {failures} failed job(s).', file=sys.stderr)
        return 1

    print('Ablation batch complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
