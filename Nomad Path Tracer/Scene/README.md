# Scene Module Guide

This folder is split so scene loading, material parsing, and acceleration-structure construction can be edited independently.

## File Roles

- `Scene.cpp`: scene state, scene-level helpers, and non-BVH scene behavior.
- `SceneBVH.cpp`: scene BVH construction and related acceleration-structure preparation.
- `SceneLoader.cpp`: XML scene loading and high-level import orchestration.
- `MaterialUtils.cpp` / `MaterialUtils.h`: material parsing helpers shared by scene-loading code.
- `TextureLoader.cpp`: texture import and upload preparation.

## Editing Guidelines

- Put XML parsing flow in `SceneLoader.cpp`.
- Put material parsing details in `MaterialUtils.cpp`.
- Put BVH-building logic in `SceneBVH.cpp`.
- Avoid re-growing `Scene.cpp` with loader- or BVH-specific code unless the state truly belongs to the `Scene` object itself.

## Why This Split Exists

The scene pipeline now serves both rendering and benchmark/data-collection workflows. Separating loading, materials, and BVH construction makes those pieces easier to test and reuse without pulling unrelated scene code along with them.
