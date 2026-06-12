#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameTriplet:
    color: Path
    albedo: Path
    normal: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise captured EXR triplets after a sweep or scene run completes."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories or frame directories to scan for frame_*.exr triplets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original color EXR instead of writing *_denoised.exr files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose denoised output already exists.",
    )
    return parser.parse_args()


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise SystemExit(f"Required tool '{name}' was not found on PATH.")
    return resolved


def iter_frame_triplets(path: Path, overwrite: bool, skip_existing: bool) -> list[FrameTriplet]:
    matches: list[FrameTriplet] = []
    search_roots: list[Path]
    if path.is_dir():
        search_roots = [path]
    else:
        search_roots = [path.parent]

    seen: set[Path] = set()
    for root in search_roots:
        for color in sorted(root.rglob("frame_*.exr")):
            if color in seen:
                continue
            seen.add(color)
            stem = color.stem
            if stem.endswith("_albedo") or stem.endswith("_normal") or stem.endswith("_denoised"):
                continue
            albedo = color.with_name(f"{stem}_albedo{color.suffix}")
            normal = color.with_name(f"{stem}_normal{color.suffix}")
            if not albedo.is_file() or not normal.is_file():
                continue
            output = color if overwrite else color.with_name(f"{stem}_denoised{color.suffix}")
            if skip_existing and output.is_file():
                continue
            matches.append(FrameTriplet(color=color, albedo=albedo, normal=normal, output=output))
    return matches


def run_checked(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def normalize_pfm_to_little_endian(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.readline()
        if magic not in {b"PF\n", b"Pf\n"}:
            raise RuntimeError(f"Unsupported PFM header in {path}: {magic!r}")

        dims = handle.readline()
        while dims.startswith(b"#"):
            dims = handle.readline()
        try:
            width_str, height_str = dims.split()
            width = int(width_str)
            height = int(height_str)
        except ValueError as exc:
            raise RuntimeError(f"Invalid PFM dimensions in {path}: {dims!r}") from exc

        scale_line = handle.readline()
        try:
            scale = float(scale_line.strip())
        except ValueError as exc:
            raise RuntimeError(f"Invalid PFM scale in {path}: {scale_line!r}") from exc

        pixel_bytes = handle.read()

    if scale < 0:
        return

    channels = 3 if magic == b"PF\n" else 1
    float_count = width * height * channels
    pixels = array("f")
    pixels.frombytes(pixel_bytes)
    if len(pixels) != float_count:
        raise RuntimeError(
            f"Unexpected pixel count in {path}: expected {float_count}, got {len(pixels)}"
        )

    if sys.byteorder == "little":
        pixels.byteswap()

    little_scale = -abs(scale) if scale != 0.0 else -1.0
    with path.open("wb") as handle:
        handle.write(magic)
        handle.write(f"{width} {height}\n".encode("ascii"))
        handle.write(f"{little_scale}\n".encode("ascii"))
        if sys.byteorder == "big":
            pixels.byteswap()
        handle.write(pixels.tobytes())


def denoise_triplet(ffmpeg: str, oidn: str, triplet: FrameTriplet) -> None:
    with tempfile.TemporaryDirectory(prefix="mpt_oidn_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        hdr_pfm = temp_dir / "color.pfm"
        alb_pfm = temp_dir / "albedo.pfm"
        nrm_pfm = temp_dir / "normal.pfm"
        out_pfm = temp_dir / "denoised.pfm"

        run_checked([ffmpeg, "-y", "-i", str(triplet.color), str(hdr_pfm)])
        run_checked([ffmpeg, "-y", "-i", str(triplet.albedo), str(alb_pfm)])
        run_checked([ffmpeg, "-y", "-i", str(triplet.normal), str(nrm_pfm)])
        normalize_pfm_to_little_endian(hdr_pfm)
        normalize_pfm_to_little_endian(alb_pfm)
        normalize_pfm_to_little_endian(nrm_pfm)
        run_checked(
            [
                oidn,
                "--hdr",
                str(hdr_pfm),
                "--alb",
                str(alb_pfm),
                "--nrm",
                str(nrm_pfm),
                "-o",
                str(out_pfm),
            ]
        )
        run_checked([ffmpeg, "-y", "-i", str(out_pfm), str(triplet.output)])


def main() -> int:
    args = parse_args()
    ffmpeg = require_tool("ffmpeg")
    oidn = require_tool("oidnDenoise")

    triplets: list[FrameTriplet] = []
    for path in args.paths:
        triplets.extend(iter_frame_triplets(path.resolve(), args.overwrite, args.skip_existing))

    if not triplets:
        print("No EXR triplets found for deferred denoising.")
        return 0

    failures = 0
    for index, triplet in enumerate(triplets, start=1):
        print(f"[{index}/{len(triplets)}] Denoising {triplet.color}")
        try:
            denoise_triplet(ffmpeg, oidn, triplet)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  failed: {exc}", file=sys.stderr)

    if failures:
        print(f"Deferred denoise completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"Deferred denoise complete for {len(triplets)} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
