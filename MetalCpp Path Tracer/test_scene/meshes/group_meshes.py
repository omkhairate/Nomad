#!/usr/bin/env python3
"""Group identical PLY meshes by hashing their raw contents.

This script scans the sibling PLY files in the meshes directory, computes a
SHA-256 hash of each file's contents, and groups files that share the same hash.
The result is saved as a JSON mapping used to reuse canonical geometry when
converting meshes or updating scene descriptions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def hash_file(path: Path) -> str:
    """Return the SHA-256 hash for the given file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_mesh_groups(mesh_dir: Path) -> List[Dict[str, object]]:
    """Group meshes that share identical file contents."""
    groups: Dict[str, List[Path]] = {}

    for mesh_path in sorted(mesh_dir.glob("*.ply")):
        if mesh_path.is_dir():
            continue
        mesh_hash = hash_file(mesh_path)
        groups.setdefault(mesh_hash, []).append(mesh_path)

    mesh_groups: List[Dict[str, object]] = []
    for mesh_hash, mesh_paths in groups.items():
        # sort paths to pick a stable canonical mesh name
        mesh_paths.sort()
        canonical = mesh_paths[0].name
        instances = [path.name for path in mesh_paths]
        mesh_groups.append(
            {
                "hash": mesh_hash,
                "canonical": canonical,
                "instances": instances,
            }
        )

    mesh_groups.sort(key=lambda group: group["canonical"])  # type: ignore[index]
    return mesh_groups


def main() -> None:
    mesh_dir = Path(__file__).resolve().parent
    conversion_dir = mesh_dir.parent / "mesh_conversions"
    conversion_dir.mkdir(parents=True, exist_ok=True)

    mesh_groups = build_mesh_groups(mesh_dir)

    output_path = conversion_dir / "mesh_map.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"groups": mesh_groups}, handle, indent=2)
        handle.write("\n")

    print(f"Wrote mesh mapping to {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
