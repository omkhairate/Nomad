# Renderer Module Guide

This folder now separates the renderer by responsibility so the residency work can evolve without turning `Renderer.cpp` back into a single monolith.

## File Roles

- `Renderer.cpp`: core renderer lifecycle, scene setup, GPU resource orchestration, TLAS/BLAS management, benchmark logging, and general frame flow.
- `RendererResidency.cpp`: residency-strategy execution and per-frame object activation/deactivation decisions.
- `RendererUnifiedNeural.cpp`: unified-neural model loading, feature-vector construction, and runtime prediction helpers.
- `RendererFrameCapture.cpp`: EXR capture, deferred denoise helpers, and frame-output post-processing.
- `RendererTextures.cpp`: texture-slot lifecycle, texture residency, material textures, and environment texture management.
- `Renderer.h`: shared renderer state and declarations consumed by the module `.cpp` files.

## Editing Guidelines

- Put strategy-selection and residency heuristics in `RendererResidency.cpp`.
- Put neural scoring or model-format changes in `RendererUnifiedNeural.cpp`.
- Put capture / EXR / denoise changes in `RendererFrameCapture.cpp`.
- Put texture-slot or material/environment texture work in `RendererTextures.cpp`.
- Keep cross-cutting renderer state in `Renderer.h`, but keep implementation local to the most specific module possible.

## Why This Split Exists

The residency experiments now touch several independent concerns:

- runtime object residency decisions
- neural scoring
- capture / evaluation instrumentation
- core renderer and GPU setup

Keeping those concerns in separate translation units makes it much easier to reason about regressions and benchmark-specific changes.
