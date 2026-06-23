#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCORE_PATTERN = re.compile(r"CGVQM score:\s*([0-9.+\-eE]+)")
FRAME_PATTERN = re.compile(r"frame_(\d+)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Align two benchmark runs by frame index, encode matched EXR sequences into videos, "
            "and compute CGVQM between them."
        )
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--test-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--representation", choices=["denoised", "raw"], default="denoised")
    parser.add_argument("--frame-shift", type=int, default=0,
                        help="Alignment convention: reference_frame = test_frame + frame_shift")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--ffmpeg-vf", default="format=rgb24")
    parser.add_argument("--ffmpeg-codec", default="libx264rgb")
    parser.add_argument("--ffmpeg-crf", default="0")
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--cgvqm-python", type=Path, default=repo_root / 'Benchmarks' / 'cgvqm' / '.venv' / 'bin' / 'python')
    parser.add_argument("--cgvqm-script", type=Path, default=repo_root / 'Benchmarks' / 'cgvqm' / 'run_metric.py')
    parser.add_argument("--cgvqm-type", choices=["cgvqm2", "cgvqm5"], default="cgvqm2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--patch-scale", type=int, default=4)
    parser.add_argument("--patch-pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--emap-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_frame_index(path: Path) -> int:
    match = FRAME_PATTERN.search(path.name)
    if not match:
        raise RuntimeError(f"Could not parse frame index from {path}")
    return int(match.group(1))


def list_frame_map(run_dir: Path, representation: str) -> dict[int, Path]:
    frames_dir = run_dir / 'frames'
    if not frames_dir.is_dir():
        raise RuntimeError(f"Expected frames directory at {frames_dir}")
    mapping: dict[int, Path] = {}
    if representation == 'denoised':
        for path in sorted(frames_dir.glob('frame_*_denoised.exr')):
            mapping[parse_frame_index(path)] = path
    else:
        for path in sorted(frames_dir.glob('frame_*.exr')):
            stem = path.stem
            if stem.endswith('_albedo') or stem.endswith('_normal') or stem.endswith('_denoised'):
                continue
            mapping[parse_frame_index(path)] = path
    if not mapping:
        raise RuntimeError(f"No {representation} frames found in {frames_dir}")
    return mapping


def run_checked(command: list[str | Path], cwd: Path) -> subprocess.CompletedProcess[str]:
    print('  command:', ' '.join(str(x) for x in command))
    completed = subprocess.run(
        [str(x) for x in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(str(x) for x in command)}\n{completed.stdout}"
        )
    return completed


def encode_video(frame_paths: list[Path], output_path: Path, *, fps: float, vf: str, codec: str, crf: str, preset: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with tempfile.TemporaryDirectory(prefix='aligned_quality_') as temp_dir_str:
        seq_dir = Path(temp_dir_str) / 'seq'
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(frame_paths):
            dst = seq_dir / f'img_{i:06d}.exr'
            shutil.copy2(src, dst)
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(seq_dir / 'img_%06d.exr'),
            '-vf', vf,
            '-c:v', codec,
            '-crf', crf,
            '-preset', preset,
            str(output_path),
        ]
        run_checked(command, output_path.parent)


def main() -> int:
    args = parse_args()
    reference_map = list_frame_map(args.reference_run.resolve(), args.representation)
    test_map = list_frame_map(args.test_run.resolve(), args.representation)

    pairs: list[tuple[int, int]] = []
    for test_frame in sorted(test_map.keys()):
        ref_frame = test_frame + args.frame_shift
        if ref_frame in reference_map:
            pairs.append((test_frame, ref_frame))
    if args.limit is not None:
        pairs = pairs[:max(args.limit, 0)]
    if not pairs:
        raise RuntimeError('No aligned frame pairs found.')

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    test_video = output_root / f'{args.test_run.name}_{args.representation}.mkv'
    ref_video = output_root / f'{args.reference_run.name}_{args.representation}.mkv'
    emap_out = args.emap_out.resolve() if args.emap_out else output_root / f'{args.test_run.name}_vs_{args.reference_run.name}_emap.mkv'
    pairs_csv = output_root / 'aligned_frame_pairs.csv'
    summary_json = output_root / 'aligned_cgvqm_summary.json'
    stdout_txt = output_root / 'cgvqm_stdout.txt'

    encode_video([test_map[t] for t, _ in pairs], test_video, fps=args.fps, vf=args.ffmpeg_vf, codec=args.ffmpeg_codec, crf=args.ffmpeg_crf, preset=args.ffmpeg_preset)
    encode_video([reference_map[r] for _, r in pairs], ref_video, fps=args.fps, vf=args.ffmpeg_vf, codec=args.ffmpeg_codec, crf=args.ffmpeg_crf, preset=args.ffmpeg_preset)

    command = [
        args.cgvqm_python,
        args.cgvqm_script,
        '--dist', test_video,
        '--ref', ref_video,
        '--type', args.cgvqm_type,
        '--device', args.device,
        '--patch-scale', str(args.patch_scale),
        '--patch-pool', args.patch_pool,
        '--emap-out', emap_out,
    ]
    completed = run_checked(command, args.cgvqm_script.parent)
    stdout_txt.write_text(completed.stdout)
    match = SCORE_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError(f"Could not parse CGVQM score from output:\n{completed.stdout}")
    score = float(match.group(1))

    with pairs_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sequence_index', 'test_frame', 'reference_frame'])
        writer.writeheader()
        for index, (test_frame, ref_frame) in enumerate(pairs):
            writer.writerow({'sequence_index': index, 'test_frame': test_frame, 'reference_frame': ref_frame})

    summary = {
        'frame_alignment': f'reference_frame = test_frame + {args.frame_shift}',
        'paired_frame_count': len(pairs),
        'fps': args.fps,
        'representation': args.representation,
        'metric': args.cgvqm_type,
        'device': args.device,
        'score_text': completed.stdout,
        'score': score,
        'test_video': str(test_video),
        'reference_video': str(ref_video),
        'emap_video': str(emap_out),
        'pairs_csv': str(pairs_csv),
    }
    summary_json.write_text(__import__('json').dumps(summary, indent=2))

    print(f'CGVQM score: {score:.4f}/100')
    print(f'Saved error-map video: {emap_out}')
    print(f'Saved summary: {summary_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
