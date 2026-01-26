# Nomad Path Tracer runtime controls

## Path tracing tile work budget

Long-running path tracing commands can trigger GPU timeout errors when the pixel/sample workload is too large for a single command buffer. The renderer now exposes a runtime-configurable knob to cap the amount of tile work recorded per command:

- **Environment variable:** set `MPT_MAX_TILE_WORK` to the maximum number of pixel/sample operations per command buffer. Example: `MPT_MAX_TILE_WORK=20000`.
- **Scene XML:** add `maxTileWorkPerCommand="<value>"` (or the alias `maxTileWork`) to the `<Scene>` root to bake a limit into a scene file.
- **Default:** when unset, the renderer uses a conservative default budget.
- **AlwaysResident override (opt-in):** set `MPT_ALWAYS_RESIDENT_TILE_WORK_HALF=1` to halve the default budget when the `AlwaysResident` residency strategy is active.
- **Benchmarking tip:** set `MPT_MAX_TILE_WORK` (or `maxTileWorkPerCommand` in the scene) to lock the same budget across residency strategies and avoid `MPT_ALWAYS_RESIDENT_TILE_WORK_HALF`.

The renderer clamps the budget so each command can still process at least one full tile, based on the current tile dimensions and sample count.

## Geometry residency memory cap

The `geometryResidencyMemoryCapMB` scene parameter remains a hard ceiling. Geometry residency allocations (including streaming uploads, prewarm passes, and rebuilds) are rejected if the next allocation would exceed the cap; the renderer queues the request and triggers eviction until enough space is available to retry.

For the memory-budget study, `totalGpuMemoryCapMB` is now the single authoritative (hard) budget. Both texture and geometry allocations check the total cap before proceeding; if the next allocation would exceed the cap, the renderer blocks the allocation and evicts residency from textures and geometry until the total budget is satisfied.

To prevent eviction scheduling deadlocks, the renderer estimates a minimum resident footprint (history + mandatory buffers + essential geometry). If the configured total cap remains below this footprint for several frames, it logs a warning, disables history, forces an accumulation reset, and temporarily relaxes the effective total cap to the minimum footprint. Benchmark metrics report the minimum footprint, relaxed cap, and whether the frame is in eviction-stall mode.

See `StudyNotes.md` for the study-specific memory budget notes and recommended interpretation of the new metrics.
