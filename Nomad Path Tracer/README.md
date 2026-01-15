# Nomad Path Tracer runtime controls

## Path tracing tile work budget

Long-running path tracing commands can trigger GPU timeout errors when the pixel/sample workload is too large for a single command buffer. The renderer now exposes a runtime-configurable knob to cap the amount of tile work recorded per command:

- **Environment variable:** set `MPT_MAX_TILE_WORK` to the maximum number of pixel/sample operations per command buffer. Example: `MPT_MAX_TILE_WORK=20000`.
- **Scene XML:** add `maxTileWorkPerCommand="<value>"` (or the alias `maxTileWork`) to the `<Scene>` root to bake a limit into a scene file.
- **Default:** when unset, the renderer uses a conservative default budget. For `AlwaysResident` residency, the default budget is halved to keep command buffers short while more geometry remains resident.

The renderer clamps the budget so each command can still process at least one full tile, based on the current tile dimensions and sample count.

## ReSTIR baseline mode

To keep ReSTIR temporal reuse stable for baseline comparisons, you can disable texture history eviction:

- **Scene XML:** add `restirBaselineMode="true"` (or `restirBaseline="true"`) to the `<Scene>` root.
- **Behavior:** no history eviction; maintains stable ReSTIR temporal reuse for baseline comparisons.
