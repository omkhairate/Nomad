"""Compare EXR images with SSIM and export results to an Excel file.

This script matches EXR images in a reference directory with images of the
same name in a test directory, computes the Structural Similarity (SSIM)
index for each pair, and writes the results to an Excel workbook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import OpenEXR  # type: ignore
import Imath  # type: ignore
import pandas as pd
from skimage.metrics import structural_similarity


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare EXR images found in two directories using SSIM and write "
            "the scores to an Excel file."
        )
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Directory containing the reference EXR images.",
    )
    parser.add_argument(
        "test",
        type=Path,
        help="Directory containing the test EXR images to compare against the reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ssim_scores.xlsx"),
        help="Destination Excel file (default: ssim_scores.xlsx).",
    )
    parser.add_argument(
        "--include-alpha",
        action="store_true",
        help="Include alpha/opacity channels in the comparison when present.",
    )
    return parser.parse_args()


def list_exr_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".exr":
            if entry.name in files:
                raise RuntimeError(f"Duplicate EXR filename detected: {entry.name}")
            files[entry.name] = entry
    return files


def get_channel_names(path: Path) -> List[str]:
    handle = OpenEXR.InputFile(str(path))
    try:
        return list(handle.header()["channels"].keys())
    finally:
        del handle


def order_channels(channels: Iterable[str], include_alpha: bool) -> List[str]:
    canonical_order = ["R", "G", "B", "A", "Y", "L", "U", "V"]
    remaining = []
    selected: List[str] = []
    alpha_names = {"A", "ALPHA"}

    for channel in channels:
        name = channel.upper()
        if not include_alpha and name in alpha_names:
            continue
        remaining.append(channel)

    for name in canonical_order:
        for channel in remaining:
            if channel.upper() == name:
                selected.append(channel)
    for channel in remaining:
        if channel not in selected:
            selected.append(channel)
    return selected


def load_exr_channels(path: Path, channels: Sequence[str]) -> np.ndarray:
    handle = OpenEXR.InputFile(str(path))
    header = handle.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    data = np.empty((height, width, len(channels)), dtype=np.float32)
    try:
        for index, channel in enumerate(channels):
            if channel not in header["channels"]:
                raise RuntimeError(f"Channel '{channel}' not present in {path}")
            raw = handle.channel(channel, pixel_type)
            array = np.frombuffer(raw, dtype=np.float32)
            if array.size != width * height:
                raise RuntimeError(
                    f"Unexpected channel size for '{channel}' in {path}: {array.size}"
                )
            data[:, :, index] = array.reshape((height, width))
    finally:
        del handle
    return data


def compute_ssim(reference: np.ndarray, test: np.ndarray) -> float:
    if reference.shape != test.shape:
        raise RuntimeError("Reference and test images must have matching shapes")

    data_range = float(
        max(reference.max(), test.max()) - min(reference.min(), test.min())
    )
    if data_range == 0:
        data_range = 1.0

    channel_axis = -1 if reference.ndim == 3 and reference.shape[2] > 1 else None
    return float(
        structural_similarity(
            reference,
            test,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )


def main() -> int:
    args = parse_arguments()

    if not args.reference.is_dir():
        print(f"Reference path is not a directory: {args.reference}", file=sys.stderr)
        return 1
    if not args.test.is_dir():
        print(f"Test path is not a directory: {args.test}", file=sys.stderr)
        return 1

    reference_files = list_exr_files(args.reference)
    test_files = list_exr_files(args.test)

    results: list[dict[str, float | str]] = []

    for filename, ref_path in sorted(reference_files.items()):
        test_path = test_files.get(filename)
        if test_path is None:
            print(f"Warning: missing test EXR for '{filename}'", file=sys.stderr)
            continue

        ref_channels = get_channel_names(ref_path)
        test_channels = get_channel_names(test_path)
        common_channels = set(ref_channels) & set(test_channels)
        if not common_channels:
            print(
                f"Warning: no common channels for '{filename}', skipping",
                file=sys.stderr,
            )
            continue

        channel_order = order_channels(common_channels, include_alpha=args.include_alpha)
        if not channel_order:
            print(
                f"Warning: no channels selected for '{filename}', skipping",
                file=sys.stderr,
            )
            continue

        ref_data = load_exr_channels(ref_path, channel_order)
        test_data = load_exr_channels(test_path, channel_order)

        try:
            score = compute_ssim(ref_data, test_data)
        except Exception as exc:  # noqa: BLE001 - surface the error to the user
            print(f"Error computing SSIM for '{filename}': {exc}", file=sys.stderr)
            continue

        results.append({"file": filename, "ssim": score})

    if not results:
        print("No SSIM scores were computed. Nothing to write.", file=sys.stderr)
        return 1

    df = pd.DataFrame(results)
    df.sort_values("file", inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output, index=False)
    print(f"Wrote {len(df)} SSIM scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
