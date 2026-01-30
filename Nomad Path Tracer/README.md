# Nomad Path Tracer runtime controls

## Path tracing tile work budget

Long-running path tracing commands can trigger GPU timeout errors when the pixel/sample workload is too large for a single command buffer. The renderer now exposes a runtime-configurable knob to cap the amount of tile work recorded per command:

- **Environment variable:** set `MPT_MAX_TILE_WORK` to the maximum number of pixel/sample operations per command buffer. Example: `MPT_MAX_TILE_WORK=20000`.
- **Scene XML:** add `maxTileWorkPerCommand="<value>"` (or the alias `maxTileWork`) to the `<Scene>` root to bake a limit into a scene file.
- **Default:** when unset, the renderer uses a conservative default budget.
- **AlwaysResident override (opt-in):** set `MPT_ALWAYS_RESIDENT_TILE_WORK_HALF=1` to halve the default budget when the `AlwaysResident` residency strategy is active.
- **Benchmarking tip:** set `MPT_MAX_TILE_WORK` (or `maxTileWorkPerCommand` in the scene) to lock the same budget across residency strategies and avoid `MPT_ALWAYS_RESIDENT_TILE_WORK_HALF`.

The renderer clamps the budget so each command can still process at least one full tile, based on the current tile dimensions and sample count.

### Extra guardrails for GPU timeout errors

On lower-power Apple Silicon (e.g., M1), GPU command buffer aborts often indicate a single command recorded too much work for the driver watchdog. In addition to `maxTileWorkPerCommand`, you can further cap batch size and reduce per-sample workload:

- **Environment variable:** set `MPT_MAX_TILES_PER_COMMAND` to cap the number of tiles recorded per command buffer. This limit is applied in addition to the adaptive budget.
- **Scene XML:** set `restirEnabled="false"` on the `<Scene>` root to disable ReSTIR sampling if you need a quick way to reduce per-sample work.
- **Scene XML:** reduce `maxRayDepth` and/or `maxSamplesPerPixel` for smaller per-tile workloads.
- **Scene XML:** lower `width`/`height` (render resolution) to shrink the total tile count and reduce the number of command buffers dispatched per frame.
- **Scene XML:** lower `maxTileWorkPerCommand` below the default `128*128*4` (e.g., `32*32*4` on M1-class GPUs) to cap per-command work more aggressively.

These controls stack with the tile work budget and are the most reliable way to eliminate GPU watchdog aborts without globally halving GPU work.

## Geometry residency memory cap

The `geometryResidencyMemoryCapMB` scene parameter remains a hard ceiling. Geometry residency allocations (including streaming uploads, prewarm passes, and rebuilds) are rejected if the next allocation would exceed the cap; the renderer queues the request and triggers eviction until enough space is available to retry.

For the memory-budget study, `totalGpuMemoryCapMB` is now the single authoritative (hard) budget. Both texture and geometry allocations check the total cap before proceeding; if the next allocation would exceed the cap, the renderer blocks the allocation and evicts residency from textures and geometry until the total budget is satisfied.

To prevent eviction scheduling deadlocks, the renderer estimates a minimum resident footprint (mandatory buffers + essential geometry). If the configured total cap remains below this footprint for several frames, it logs a warning, disables texture history, and temporarily relaxes the effective total cap to the minimum footprint. Benchmark metrics report the minimum footprint, relaxed cap, and whether the frame is in eviction-stall mode.

See `StudyNotes.md` for the study-specific memory budget notes and recommended interpretation of the new metrics.
