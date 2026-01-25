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

Residency memory reporting now splits the working set into the actual budget components:

- `resident_geometry_memory_mb` – geometry residency tracked by the active strategy (distance, energy, etc.).
- `resident_texture_memory_mb` – texture and accumulation targets managed by the texture residency pool.
- `restir_memory_mb` – ReSTIR buffers that scale with resolution when sampling is enabled.
- `residency_memory_mb` – sum of the above resident components, which is the budget that the residency strategies and caps are meant to manage. Geometry strategy tweaks primarily affect the geometry slice, while texture eviction and ReSTIR toggles influence the other two components.

## Residency cap configuration for the study

The renderer now enforces a **single** residency budget using `totalGpuMemoryCapMB`. Geometry allocations and texture evictions both reference this unified cap (which includes resident content + scratch), and the legacy per-pool caps are synced to the same value for reporting/compatibility.

For the study runs reported in the paper/appendix, we fixed the caps to a single value to avoid ambiguity:

- `totalGpuMemoryCapMB = 4096` MB (authoritative)
- `textureResidencyMemoryCapMB = 4096` MB (synced)
- `geometryResidencyMemoryCapMB = 4096` MB (synced)

If you override the cap in a scene XML, report `totalGpuMemoryCapMB` explicitly; the renderer will align the texture and geometry caps to that value.

## Environment-hit scene attributes

Scenes that opt into the `environment` residency strategy can now tune how aggressively the renderer combats environment-map leaks. The `<Scene>` root accepts the following optional attributes:

- `envTargetFraction` – fraction of total primitives that should remain active, even when the escape rate is low. Acts as a soft floor before the escape threshold kicks in. Defaults to `0.0` to preserve the previous "escape only" behavior.
- `envEscapeThreshold` – maximum allowed average escape probability across the active set. Lower values keep more geometry resident to occlude environment rays.
- `envMinActive` – minimum number of primitives that stay active regardless of the fraction/escape targets. Aliased to the older `environmentMinActive` attribute.
- `envToggleBudget` – maximum number of primitives the environment strategy can flip per frame (`environmentToggleBudget` is still accepted).
- `envDepthRadii` / `envDepthWeights` – comma-separated lists that bias which objects are prioritized. Each radius entry (world-space distance from the camera) pairs with a weight multiplier at the same index. Objects whose bounding-sphere centers fall within a radius use the corresponding weight when ranking environment leaks. When omitted, objects are ranked purely by their measured hit probability.

All legacy `environment*` attribute names continue to work, and any `env*` overrides take precedence when both are present. Example configuration:

```xml
<Scene residencyStrategy="environment"
       envTargetFraction="0.25"
       envEscapeThreshold="0.35"
       envMinActive="48"
       envToggleBudget="96"
       envDepthRadii="15,30,45"
       envDepthWeights="1.5,1.0,0.5">
    ...
</Scene>
```

This snippet keeps at least 25% of the scene active, clamps the escape rate to 35%, and prioritizes geometry within 15 units of the camera 1.5× higher than distant occluders.

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
