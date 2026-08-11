"""Fallback reading-order heuristic for engines with no built-in order head.

Buckets regions into rows by vertical overlap, then sorts each row
left-to-right and rows top-to-bottom. Not column-aware beyond that, but
reasonable for typical single/multi-column pages.
"""

from __future__ import annotations

from doc2md.models import Region


def assign_reading_order(regions: list[Region], overlap_frac: float = 0.5) -> list[Region]:
    if not regions:
        return regions

    ordered = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))
    rows: list[list[Region]] = []
    for region in ordered:
        y0, y1 = region.bbox[1], region.bbox[3]
        height = max(1, y1 - y0)
        placed = False
        for row in rows:
            row_y0 = min(r.bbox[1] for r in row)
            row_y1 = max(r.bbox[3] for r in row)
            overlap = min(y1, row_y1) - max(y0, row_y0)
            if overlap > overlap_frac * min(height, max(1, row_y1 - row_y0)):
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])

    rows.sort(key=lambda row: min(r.bbox[1] for r in row))
    result: list[Region] = []
    for row in rows:
        row.sort(key=lambda r: r.bbox[0])
        result.extend(row)

    for i, region in enumerate(result, start=1):
        region.order_index = i
    return result
