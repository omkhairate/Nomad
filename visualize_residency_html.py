#!/usr/bin/env python3
"""Visualise BLAS *primitive* residency over frames as a simple HTML grid.

The renderer can dump per-frame JSON describing the TLAS/BLAS state and the
status of individual primitives.  This script consumes either a directory of
JSON files or a single JSON file containing a list of frames and emits a small
HTML heatmap.  Each row represents a primitive, each column a frame.  Cells are
green while the primitive is resident and red once it has been offloaded.  Hover
over a cell to see the primitive index along with the number of frames since it
was last hit and the intersection count that led to the decision.

This approach avoids heavy dependencies and produces a compact, fast-to-render
visualisation suitable for quick inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def _process_frame(data: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
    """Normalise various frame dump formats and extract primitive data."""
    frame: Dict[str, Any] = {"frame": frame_index}
    if "primitives" in data:
        frame["primitives"] = data["primitives"]
    elif "tlas" in data and "blas" in data:
        # Frame contains TLAS/BLAS info but no explicit primitive list.
        # In that case we assume everything is resident.
        frame["primitives"] = []
    elif "nodes" in data:
        frame["primitives"] = data.get("primitives", [])
    else:
        raise ValueError("Unsupported frame format")
    return frame


def _load_frames(path: Path) -> List[Dict[str, Any]]:
    """Return a list of frame dictionaries from ``path``."""
    frames: List[Dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for d in data:
                frames.append(_process_frame(d, len(frames)))
        else:
            frames.append(_process_frame(data, 0))
    else:
        for p in sorted(path.glob("*.json")):
            with p.open("r", encoding="utf-8") as f:
                frames.append(_process_frame(json.load(f), len(frames)))
    return frames


def _write_html(frames: List[Dict[str, Any]], output: Path) -> None:
    """Write residency information to ``output`` as an HTML file."""
    if not frames:
        raise SystemExit("No frames were loaded")

    max_prims = max(len(f.get("primitives", [])) for f in frames)

    header_cells = ['<th class="primitive">Prim</th>'] + [
        f'<th class="frame">{f["frame"]}</th>' for f in frames
    ]
    header = "<tr>" + "".join(header_cells) + "</tr>"

    rows: List[str] = []
    for prim_idx in range(max_prims):
        cells: List[str] = []
        for frame in frames:
            prims = frame.get("primitives", [])
            if prim_idx < len(prims):
                prim = prims[prim_idx]
                active = prim.get("active", True)
                last_hit = prim.get("lastIntersection", 0)
                inactive = prim.get("inactiveFrames", 0)
                title = (
                    f"primitive {prim_idx}: lastIntersection={last_hit}, "
                    f"inactiveFrames={inactive}"
                )
            else:
                active = False
                title = f"primitive {prim_idx}: missing"
            cls = "loaded" if active else "offloaded"
            cells.append(f'<td class="{cls}" title="{title}"></td>')
        row = f"<tr><th class='primitive'>{prim_idx}</th>{''.join(cells)}</tr>"
        rows.append(row)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<style>
  table.residency {{ border-collapse: collapse; }}
  table.residency td {{ width: 6px; height: 6px; padding: 0; }}
  table.residency th.primitive {{ text-align: right; padding-right: 4px; }}
  table.residency th.frame {{ width: 6px; padding: 0; writing-mode: vertical-rl; font-size: 8px; }}
  td.loaded {{ background: #4caf50; }}
  td.offloaded {{ background: #f44336; }}
</style>
</head>
<body>
<table class='residency'>
<thead>
{header}
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a primitive residency heatmap")
    parser.add_argument("path", type=Path, help="Directory or JSON file containing frame dumps")
    parser.add_argument("--output", type=Path, default=Path("residency.html"), help="Output HTML file")
    args = parser.parse_args()

    frames = _load_frames(args.path)
    _write_html(frames, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
