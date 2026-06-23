#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["scene_variant", "strategy", "clip_id", "object_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one or more neural CGVQM label CSVs and drop duplicate object/clip entries."
    )
    parser.add_argument("csv", nargs="+", type=Path, help="Input label CSV files in oldest-to-newest order.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = []
    for path in args.csv:
        df = pd.read_csv(path)
        df["source_csv"] = str(path.resolve())
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    keep_cols = [col for col in KEY_COLUMNS if col in merged.columns]
    if not keep_cols:
        raise RuntimeError(f"None of the key columns {KEY_COLUMNS} were found in the input CSVs.")

    merged = merged.drop_duplicates(subset=keep_cols, keep="last").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print("Merged neural label CSVs")
    print(f"  inputs: {len(args.csv)}")
    print(f"  output: {args.output.resolve()}")
    print(f"  rows: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
