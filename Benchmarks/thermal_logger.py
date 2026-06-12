#!/usr/bin/env python3
"""Log macOS thermal/power samples to CSV while a benchmark run executes.

On Apple Silicon, direct temperature sensors are not normally available to an
unprivileged process. This script therefore uses ``powermetrics`` thermal and
power samplers. ``powermetrics`` requires administrator privileges, so the
intended workflow is:

    sudo -v
    python3 Benchmarks/run_bistro_sweep.py --thermal-log ...

The sweep launcher starts this logger with ``sudo -n`` so it will never hang
waiting for a password. If sudo has not been authenticated first, the logger
writes a one-line error log and exits.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

POWER_RE = re.compile(r"^(CPU|GPU|ANE|Combined)\s+Power:\s+([0-9.]+)\s*(mW|W)", re.I)
FREQ_RE = re.compile(r"^(CPU|GPU|ANE).*?(?:frequency|freq).*?:\s+([0-9.]+)\s*(MHz|GHz)", re.I)
THERMAL_RE = re.compile(r"Thermal pressure:\s*([^\n]+)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log macOS powermetrics thermal samples to CSV.")
    parser.add_argument("--out", type=Path, required=True, help="CSV path to write.")
    parser.add_argument("--interval-ms", type=int, default=2000, help="powermetrics interval in ms.")
    parser.add_argument("--no-sudo", action="store_true", help="Run powermetrics directly instead of via sudo -n.")
    parser.add_argument("--raw-log", type=Path, default=None, help="Optional raw powermetrics text log.")
    return parser.parse_args()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="milliseconds")


def mw(value: float, unit: str) -> float:
    return value * 1000.0 if unit.lower() == "w" else value


def mhz(value: float, unit: str) -> float:
    return value * 1000.0 if unit.lower() == "ghz" else value


def write_error(out: Path, message: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index", "wall_unix_seconds", "wall_iso", "run_elapsed_seconds",
                "thermal_pressure", "cpu_power_mw", "gpu_power_mw", "ane_power_mw",
                "combined_power_mw", "cpu_frequency_mhz", "gpu_frequency_mhz",
                "ane_frequency_mhz", "error",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "sample_index": 0,
            "wall_unix_seconds": f"{time.time():.6f}",
            "wall_iso": now_iso(),
            "run_elapsed_seconds": "0.000000",
            "error": message,
        })


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_file = args.raw_log.open("w") if args.raw_log else None

    cmd = [
        "powermetrics",
        "--samplers", "thermal,cpu_power,gpu_power,ane_power",
        "--show-extra-power-info",
        "-i", str(max(250, args.interval_ms)),
        "--buffer-size", "1",
    ]
    if not args.no_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n", *cmd]

    fieldnames = [
        "sample_index", "wall_unix_seconds", "wall_iso", "run_elapsed_seconds",
        "thermal_pressure", "cpu_power_mw", "gpu_power_mw", "ane_power_mw",
        "combined_power_mw", "cpu_frequency_mhz", "gpu_frequency_mhz",
        "ane_frequency_mhz", "error",
    ]

    start = time.monotonic()
    sample_index = 0
    current = {name: "" for name in fieldnames}
    current["sample_index"] = sample_index

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        write_error(args.out, f"failed_to_start_powermetrics: {exc}")
        return 1

    stop_requested = False

    def request_stop(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def emit(writer: csv.DictWriter) -> None:
        nonlocal sample_index, current
        row = dict(current)
        row["sample_index"] = sample_index
        row["wall_unix_seconds"] = f"{time.time():.6f}"
        row["wall_iso"] = now_iso()
        row["run_elapsed_seconds"] = f"{time.monotonic() - start:.6f}"
        writer.writerow(row)
        sample_index += 1
        current = {name: "" for name in fieldnames}
        current["sample_index"] = sample_index

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        assert proc.stdout is not None
        saw_any_output = False
        for line in proc.stdout:
            saw_any_output = True
            if raw_file:
                raw_file.write(line)
                raw_file.flush()
            stripped = line.strip()
            if not stripped:
                if any(current.get(key) for key in fieldnames if key not in {"sample_index", "error"}):
                    emit(writer)
                    f.flush()
                if stop_requested:
                    break
                continue

            thermal = THERMAL_RE.search(stripped)
            if thermal:
                current["thermal_pressure"] = thermal.group(1).strip()

            power = POWER_RE.search(stripped)
            if power:
                key = f"{power.group(1).lower()}_power_mw"
                current[key] = f"{mw(float(power.group(2)), power.group(3)):.3f}"

            freq = FREQ_RE.search(stripped)
            if freq:
                key = f"{freq.group(1).lower()}_frequency_mhz"
                current[key] = f"{mhz(float(freq.group(2)), freq.group(3)):.3f}"

            lowered = stripped.lower()
            if "must be invoked as the superuser" in lowered or "a password is required" in lowered:
                current["error"] = stripped
                emit(writer)
                f.flush()
                proc.terminate()
                break

        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        if not saw_any_output and sample_index == 0:
            current["error"] = f"powermetrics exited without output, return_code={proc.returncode}"
            emit(writer)

    if raw_file:
        raw_file.close()
    return 0 if proc.returncode in (0, -signal.SIGTERM, -signal.SIGINT, None) else int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
