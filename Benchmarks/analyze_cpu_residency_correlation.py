#!/usr/bin/env python3
import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CPU timing correlation with residency onload/offload activity.")
    parser.add_argument("csv", help="Path to metrics CSV")
    parser.add_argument("--max-lag", type=int, default=10,
                        help="Max lag (frames) for lagged correlation")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [
        "frame", "cpu_ms", "objects_onload_requested", "objects_offload_requested",
        "onload_requested_mb", "offload_requested_mb", "blas_build_requests",
        "tlas_rebuilds", "tlas_refits"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df.sort_values("frame").reset_index(drop=True)

    signals = [
        "objects_onload_requested",
        "objects_offload_requested",
        "onload_requested_mb",
        "offload_requested_mb",
        "blas_build_requests",
        "tlas_rebuilds",
        "tlas_refits",
    ]

    print("=== Same-frame Pearson Correlation with cpu_ms ===")
    for col in signals:
        corr = df["cpu_ms"].corr(df[col])
        print(f"{col:28s}: {corr: .4f}")

    print("\n=== Lagged Correlation: corr(cpu_ms[t], signal[t-k]) ===")
    for col in signals:
        best_k = 0
        best = df["cpu_ms"].corr(df[col])
        for k in range(1, args.max_lag + 1):
            lag_corr = df["cpu_ms"].corr(df[col].shift(k))
            if pd.notna(lag_corr) and (pd.isna(best) or abs(lag_corr) > abs(best)):
                best = lag_corr
                best_k = k
        print(f"{col:28s}: best={best: .4f} at lag={best_k}")

    print("\n=== Segment Stats (cpu_ms) ===")
    on = df["objects_onload_requested"] > 0
    off = df["objects_offload_requested"] > 0
    segments = {
        "none": ~(on | off),
        "onload_only": on & ~off,
        "offload_only": ~on & off,
        "both": on & off,
    }
    for name, mask in segments.items():
        sub = df.loc[mask, "cpu_ms"]
        if sub.empty:
            print(f"{name:12s}: n=0")
            continue
        print(
            f"{name:12s}: n={len(sub):4d} mean={sub.mean():7.3f} "
            f"median={sub.median():7.3f} p95={sub.quantile(0.95):7.3f}")


if __name__ == "__main__":
    main()
