"""Generate a heatmap of per-primitive ray intersection counts.

This utility reads the same per-frame JSON dumps produced by the renderer as
``visualize_residency_html.py`` but colours each cell based on the number of
intersections that primitive saw in a given frame.  Darker red indicates a
higher intersection count.  The resulting HTML file provides a compact overview
of how work is distributed across primitives over time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _process_frame(data: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
    """Normalise various frame dump formats and extract primitive data."""
    frame: Dict[str, Any] = {"frame": frame_index}
    if "primitives" in data:
        frame["primitives"] = data["primitives"]
    elif "tlas" in data and "blas" in data:
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
    """Write intersection count information to ``output`` as an HTML file."""
    if not frames:
        raise SystemExit("No frames were loaded")

    max_prims = max(len(f.get("primitives", [])) for f in frames)
    max_count = max(
        (prim.get("lastIntersection", 0) for f in frames for prim in f.get("primitives", [])),
        default=0,
    )

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
                count = prims[prim_idx].get("lastIntersection", 0)
            else:
                count = 0
            intensity = 0 if max_count == 0 else int(255 * count / max_count)
            title = f"primitive {prim_idx}: lastIntersection={count}"
            cells.append(
                f'<td style="background: rgb({intensity},0,0)" title="{title}"></td>'
            )
        row = f"<tr><th class='primitive'>{prim_idx}</th>{''.join(cells)}</tr>"
        rows.append(row)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<style>
  table.intersections {{ border-collapse: collapse; }}
  table.intersections td {{ width: 6px; height: 6px; padding: 0; }}
  table.intersections th.primitive {{ text-align: right; padding-right: 4px; }}
  table.intersections th.frame {{ width: 6px; padding: 0; writing-mode: vertical-rl; font-size: 8px; }}
</style>
</head>
<body>
<table class='intersections'>
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
    parser = argparse.ArgumentParser(
        description="Generate a ray intersection count heatmap"
    )
    parser.add_argument(
        "path", type=Path, help="Directory or JSON file containing frame dumps"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("intersections.html"),
        help="Output HTML file",
    )
    args = parser.parse_args()

    frames = _load_frames(args.path)
    _write_html(frames, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

