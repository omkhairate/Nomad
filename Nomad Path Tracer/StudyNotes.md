# Memory budget study notes

## Single-budget policy (totalGpuMemoryCapMB)

The study now treats `totalGpuMemoryCapMB` as the **single authoritative budget**. Texture and geometry allocations check the total cap before allocating; if the next allocation would exceed the cap, the renderer blocks the allocation and schedules residency evictions until the total budget is satisfied. This keeps `scene_bistro_test_v2_distance.xml` results aligned with a single-budget policy in both runtime logs and benchmark exports.

## Eviction-stall fallback

To avoid getting stuck in a perpetual eviction loop, the renderer tracks a minimum resident footprint estimate:

```
mandatory buffers + essential geometry
```

When the configured total cap stays below this footprint for several frames, the renderer:

1. Logs a warning that the total cap is below the minimum footprint.
2. Forces an accumulation reset.
3. Temporarily relaxes the effective total cap to the minimum footprint value.

The benchmark CSV exposes the minimum footprint, relaxed cap, and eviction-stall status to make the fallback visible when comparing study runs.
