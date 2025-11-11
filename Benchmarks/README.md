# Benchmarks and Frame Capture Controls

The path tracer can emit additional metrics and frame captures while running benchmarks. Configure the behavior with the following environment variables before launching the app:

- `MPT_RUNS_PATH` (default: unset) – directory where run artifacts such as acceleration structure dumps and CSV metrics are written.
- `MPT_MAX_FRAMES` (default: unset) – stop rendering after the specified number of frames when keyframes are present.
- `MPT_CAPTURE_EXR` (default: `0`) – set to `1`, `true`, or `yes` to enable EXR frame capture from the renderer. Captured frames are written to the `Benchmarks/frames` directory next to benchmark logs.
- `MPT_CAPTURE_INTERVAL` (default: `4`) – capture every _n_ frames when EXR capture is enabled. Values less than `1` are ignored and treated as `1`.

These variables can be exported in your shell or added to any run scripts you maintain for benchmarking sessions.

## Benchmark CSV columns

Benchmark exports now include stochastic residency metrics alongside the existing memory and activation counters:

- `avg_hit_probability` – arithmetic mean of the per-primitive hit probabilities tracked during the frame.
- `p95_hit_probability` – 95th percentile of the same per-primitive hit probabilities, highlighting upper-tail engagement.
- `probability_threshold` – the active residency threshold applied by the probabilistic strategy for the frame.
- `probabilistic_toggles` – number of primitives that flipped residency state due to the probabilistic strategy during the frame.

These columns complement the existing residency memory statistics in `*_memory_mb` and allow the plotting tools to chart probability-driven residency behavior when comparing runs.

## Comparing EXR captures

Use `compare_exr_ssim.cpp` to quantify the visual difference between two EXR frames with the Structural Similarity Index (SSIM). Build the tool with a C++17 compiler:

```bash
cd MetalPathtracer
g++ -std=c++17 compare_exr_ssim.cpp -o compare_exr_ssim
```

The executable expects two EXR files that share the same resolution. By default it evaluates the RGB channels (ignoring alpha):

```bash
./compare_exr_ssim Benchmarks/frames/frame_000000.exr Benchmarks/frames/frame_000006.exr
```

Use `--include-alpha` to keep the alpha channel in the computation or `--luminance` to reduce RGB data to a single luminance channel prior to computing SSIM. The SSIM kernel can be adjusted with `--window-size` (odd integer) and `--sigma` if you need to tune the spatial weighting.

## Latest benchmark snapshot

- `metrics_20251015_150120.csv` captures the baseline renderer before the TLAS leaf caching changes.
- `metrics_20251016_101530.csv` reflects the post-optimization run and shows an average GPU frame time that is roughly 15% lower.
