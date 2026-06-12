#!/usr/bin/env python3
import argparse

from timing_analysis_common import analyze_single_timing, load_metrics, merge_temperature_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GPU timing correlation with residency activity, outliers, regression, and lag plots."
    )
    parser.add_argument("csv", help="Path to metrics CSV")
    parser.add_argument("--max-lag", type=int, default=15, help="Max lag (frames) for lag profiles")
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used for winsorized/trimmed outlier handling",
    )
    parser.add_argument("--plot-dir", default=None, help="Directory for lag-profile plots/CSVs")
    parser.add_argument("--temperature-csv", default=None, help="Optional external CSV with temperature samples")
    parser.add_argument(
        "--temperature-key",
        default="frame",
        help="Join key shared by the metrics CSV and the external temperature CSV",
    )
    parser.add_argument(
        "--temperature-column",
        default=None,
        help="Optional explicit temperature column name from the external temperature CSV",
    )
    args = parser.parse_args()

    df = load_metrics(args.csv)
    df, _ = merge_temperature_data(df, args.temperature_csv, args.temperature_key, args.temperature_column)
    analyze_single_timing(
        df=df,
        csv_path=args.csv,
        timing_col="gpu_ms",
        max_lag=args.max_lag,
        outlier_quantile=args.outlier_quantile,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
