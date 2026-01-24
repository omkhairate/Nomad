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

## ReSTIR sampling toggle

ReSTIR sampling can be enabled without changing the residency strategy, which is useful for A/B comparisons.

- **Scene XML:** add `restirSampling="true"` (or `restirSamplingEnabled="true"`) to the `<Scene>` root.
- **Behavior:** when enabled, ReSTIR sampling runs regardless of the configured residency strategy.
- **Legacy scenes:** `residencyStrategy="restir"` still enables ReSTIR sampling automatically.

## Bistro test scenes with ReSTIR sampling enabled

The `scene_bistro_test_v2_*_restir_on.xml` variants mirror their corresponding bistro test scenes but explicitly enable ReSTIR sampling via `restirSampling="true"` on the `<Scene>` root. Use these for A/B comparisons that isolate the sampling toggle from other settings.

- `scene_bistro_test_v2_distance_restir_on.xml`
- `scene_bistro_test_v2_energy_restir_on.xml`
- `scene_bistro_test_v2_environment_restir_on.xml`
- `scene_bistro_test_v2_predictive_restir_on.xml`
- `scene_bistro_test_v2_probabilistic_restir_on.xml`
- `scene_bistro_test_v2_rayhit_restir_on.xml`
- `scene_bistro_test_v2_restir_restir_on.xml`
- `scene_bistro_test_v2_screenspace_restir_on.xml`
- `scene_bistro_test_v2_unified_restir_on.xml`
