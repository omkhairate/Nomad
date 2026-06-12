#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_SCENE = 'scene_bistro_test_v2.xml'


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_app = repo_root / 'build' / 'Build' / 'Products' / 'Release' / 'Nomad Path Tracer.app' / 'Contents' / 'MacOS' / 'Nomad Path Tracer'
    default_root = repo_root / 'Benchmarks' / 'neural_cgvqm_pipeline'

    parser = argparse.ArgumentParser(
        description=(
            'Run the rigorous CGVQM ablation prep pipeline in one command: '
            'always-resident feature logging, manifest generation, then '
            'always-resident object ablation batch.'
        )
    )
    parser.add_argument('--app', type=Path, default=default_app)
    parser.add_argument('--pipeline-root', type=Path, default=default_root)
    parser.add_argument(
        '--run-name',
        default=None,
        help=(
            'Optional subdirectory name under --pipeline-root for this pipeline run. '
            'If omitted, a timestamped directory is created automatically.'
        ),
    )
    parser.add_argument(
        '--reuse-root',
        action='store_true',
        help='Reuse --pipeline-root directly instead of creating a fresh timestamped subdirectory.',
    )
    parser.add_argument('--scene', default=DEFAULT_SCENE, help='Always-resident Bistro scene to use for both feature logging and ablation. Default: scene_bistro_test_v2.xml')
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--neural-clip-length', type=int, default=16)
    parser.add_argument('--top-visible', type=int, default=8)
    parser.add_argument('--top-hit', type=int, default=6)
    parser.add_argument('--top-toggle', type=int, default=4)
    parser.add_argument('--random-controls', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--capture-exr', action='store_true')
    parser.add_argument('--capture-interval', type=int, default=4)
    parser.add_argument('--defer-oidn', action='store_true', default=True, help='Use deferred OIDN during ablation runs. Default: on')
    parser.add_argument('--no-defer-oidn', action='store_false', dest='defer_oidn', help='Disable deferred OIDN for ablation runs.')
    parser.add_argument('--decision-observer', action='store_true')
    parser.add_argument('--observer-position', default='0,0,52')
    parser.add_argument('--observer-look-at', default='0,0,3')
    parser.add_argument('--observer-up', default='0,1,0')
    parser.add_argument('--observer-fov', type=float, default=70.0)
    parser.add_argument('--thermal-log', action='store_true')
    parser.add_argument('--thermal-interval-ms', type=int, default=2000)
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--clip-id', action='append', type=int, default=[])
    parser.add_argument('--object-id', action='append', type=int, default=[])
    parser.add_argument('--min-selection-score', type=float, default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, *, dry_run: bool) -> None:
    print('  command:', ' '.join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        cmd_text = ' '.join(command)
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {cmd_text}')


def parse_run_dir_name(run_dir: Path) -> tuple[str, str]:
    parts = run_dir.name.rsplit('_', 2)
    if len(parts) != 3:
        return run_dir.name, ''
    scene_id, date_part, time_part = parts
    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        return scene_id, f'{date_part}_{time_part}'
    return run_dir.name, ''


def latest_complete_feature_run(feature_root: Path, scene_name: str) -> Path | None:
    scene_stem = Path(scene_name).stem
    if not feature_root.exists():
        return None

    candidates: list[tuple[str, Path]] = []
    for path in feature_root.iterdir():
        if not path.is_dir():
            continue
        run_scene_stem, timestamp = parse_run_dir_name(path)
        if run_scene_stem != scene_stem:
            continue
        summary_path = path / 'run_summary.json'
        features_path = path / 'neural_object_features.csv'
        if not summary_path.is_file() or not features_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if not summary.get('complete'):
            continue
        candidates.append((timestamp or path.name, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_session_root(args: argparse.Namespace) -> Path:
    pipeline_root = args.pipeline_root.resolve()
    if args.reuse_root:
        return pipeline_root
    if args.run_name:
        return pipeline_root / args.run_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scene_stem = Path(args.scene).stem
    return pipeline_root / f'{scene_stem}_{timestamp}'


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    session_root = resolve_session_root(args)
    feature_root = session_root / 'features'
    ablation_root = session_root / 'ablations'
    feature_root.mkdir(parents=True, exist_ok=True)
    ablation_root.mkdir(parents=True, exist_ok=True)

    sweep_script = repo_root / 'Benchmarks' / 'run_bistro_sweep.py'
    manifest_script = repo_root / 'Benchmarks' / 'generate_neural_ablation_manifest.py'
    ablation_script = repo_root / 'Benchmarks' / 'run_neural_ablation_batch.py'

    print('Rigorous neural CGVQM pipeline')
    print(f'  scene: {args.scene}')
    print(f'  session root: {session_root}')
    print(f'  feature root: {feature_root}')
    print(f'  ablation root: {ablation_root}')

    feature_cmd = [
        sys.executable,
        str(sweep_script),
        '--app',
        str(args.app),
        '--scene',
        args.scene,
        '--output-root',
        str(feature_root),
        '--max-frames',
        str(args.max_frames),
        '--log-neural-features',
        '--neural-clip-length',
        str(args.neural_clip_length),
    ]
    if args.decision_observer:
        feature_cmd.append('--decision-observer')
        feature_cmd.extend(['--observer-position', args.observer_position])
        feature_cmd.extend(['--observer-look-at', args.observer_look_at])
        feature_cmd.extend(['--observer-up', args.observer_up])
        feature_cmd.extend(['--observer-fov', str(args.observer_fov)])
    if args.thermal_log:
        feature_cmd.append('--thermal-log')
        feature_cmd.extend(['--thermal-interval-ms', str(args.thermal_interval_ms)])
    if args.dry_run:
        feature_cmd.append('--dry-run')

    print('\nStep 1: Generate always-resident clip features')
    run_command(feature_cmd, repo_root, dry_run=args.dry_run)

    if args.dry_run:
        print('\nDry run complete. No files were generated.')
        return 0

    feature_run_dir = latest_complete_feature_run(feature_root, args.scene)
    if feature_run_dir is None:
        print('Could not find a completed feature run with neural_object_features.csv.', file=sys.stderr)
        return 2

    features_csv = feature_run_dir / 'neural_object_features.csv'
    manifest_csv = feature_run_dir / 'neural_ablation_manifest.csv'

    manifest_cmd = [
        sys.executable,
        str(manifest_script),
        str(features_csv),
        '--output',
        str(manifest_csv),
        '--top-visible',
        str(args.top_visible),
        '--top-hit',
        str(args.top_hit),
        '--top-toggle',
        str(args.top_toggle),
        '--random-controls',
        str(args.random_controls),
        '--seed',
        str(args.seed),
    ]

    print('\nStep 2: Generate ablation manifest')
    run_command(manifest_cmd, repo_root, dry_run=False)

    ablation_cmd = [
        sys.executable,
        str(ablation_script),
        str(manifest_csv),
        '--app',
        str(args.app),
        '--output-root',
        str(ablation_root),
        '--ablation-scene',
        args.scene,
        '--max-frames',
        str(args.max_frames),
    ]
    if args.capture_exr:
        ablation_cmd.append('--capture-exr')
        ablation_cmd.extend(['--capture-interval', str(args.capture_interval)])
    if args.defer_oidn:
        ablation_cmd.append('--defer-oidn')
    if args.decision_observer:
        ablation_cmd.append('--decision-observer')
        ablation_cmd.extend(['--observer-position', args.observer_position])
        ablation_cmd.extend(['--observer-look-at', args.observer_look_at])
        ablation_cmd.extend(['--observer-up', args.observer_up])
        ablation_cmd.extend(['--observer-fov', str(args.observer_fov)])
    if args.thermal_log:
        ablation_cmd.append('--thermal-log')
        ablation_cmd.extend(['--thermal-interval-ms', str(args.thermal_interval_ms)])
    if args.continue_on_error:
        ablation_cmd.append('--continue-on-error')
    if args.limit is not None:
        ablation_cmd.extend(['--limit', str(args.limit)])
    if args.start_index:
        ablation_cmd.extend(['--start-index', str(args.start_index)])
    for clip_id in args.clip_id:
        ablation_cmd.extend(['--clip-id', str(clip_id)])
    for object_id in args.object_id:
        ablation_cmd.extend(['--object-id', str(object_id)])
    if args.min_selection_score is not None:
        ablation_cmd.extend(['--min-selection-score', str(args.min_selection_score)])

    print('\nStep 3: Run always-resident object ablations')
    run_command(ablation_cmd, repo_root, dry_run=False)

    print('\nPipeline complete.')
    print(f'  session root: {session_root}')
    print(f'  feature run: {feature_run_dir}')
    print(f'  features csv: {features_csv}')
    print(f'  manifest csv: {manifest_csv}')
    print(f'  ablation root: {ablation_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
