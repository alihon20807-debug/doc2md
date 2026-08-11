"""Suppress near-duplicate regions that survive an engine's own NMS.

RT-DETR's one-to-one matching is IoU-based, so a coarse box that tightly contains
several finer same-bucket boxes (e.g. a paragraph-level "text" region overlapping
its own text lines) can survive even though it's a near-duplicate: containment is
high but IoU is low because the box areas differ. The Docling engine has been
observed to emit these, causing the same content to be transcribed twice.
"""

from __future__ import annotations

from doc2md.models import Region

DEFAULT_CONTAINMENT_THRESHOLD = 0.8


def _area(bbox: tuple[int, int, int, int]) -> int:
    xmin, ymin, xmax, ymax = bbox
    return max(0, xmax - xmin) * max(0, ymax - ymin)


def suppress_contained_duplicates(
    regions: list[Region], containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD
) -> list[Region]:
    """Drop lower-confidence regions that substantially overlap an
    already-kept region of the same bucket, in either size direction.

    Uses intersection-over-smaller-area (rather than plain IoU or one-sided
    containment) so it catches duplicates regardless of which box happens to
    be larger - e.g. a coarse paragraph box vs. one of its own text lines,
    which can have a low IoU (very different areas) despite one fully
    containing the other.
    """
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        xmin, ymin, xmax, ymax = region.bbox
        area = _area(region.bbox)
        suppressed = False
        for other in kept:
            if other.bucket is not region.bucket:
                continue
            oxmin, oymin, oxmax, oymax = other.bbox
            ix0, iy0 = max(xmin, oxmin), max(ymin, oymin)
            ix1, iy1 = min(xmax, oxmax), min(ymax, oymax)
            intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            smaller_area = min(area, _area(other.bbox))
            if smaller_area > 0 and intersection / smaller_area >= containment_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(region)
    return kept
