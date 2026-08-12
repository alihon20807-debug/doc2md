"""Merge consecutive, same-column line regions into one region per paragraph.

Layout detectors (verified on mineru) often emit one box per visual *line*,
not per paragraph - a short line ("Trailer", "Header" used as a one-word
sub-label within a bulleted breakdown) produces a small bbox for a mundane
reason (the text itself is short), not because anything is wrong with the
detection. Each region still costs a full separate VLM round-trip regardless
of how little text it holds, so a 5-line paragraph pays 5x the fixed
per-request overhead (network round-trip, prompt prefill, crop/encode) for
the same content one merged region would produce. Unlike a size/confidence
based drop filter (investigated and rejected - see docs/OPTIMIZATION_PLAN.md),
merging loses no content: it only changes how many VLM calls the same total
text is packaged into.
"""

from __future__ import annotations

from doc2md.config import Settings
from doc2md.models import Bucket, Region


def _can_merge(a: Region, b: Region, settings: Settings) -> bool:
    """True if `b` is plausibly the next line of the same paragraph as `a`.

    Only ever merges TEXT_LIKE regions with the *same* raw label - never
    TABLE/PICTURE (which need their own dedicated prompts), and never across
    labels (a title merged into body text would apply the wrong prompt to
    part of the merged content, since text_prompt() branches on label).
    """
    if a.bucket is not Bucket.TEXT_LIKE or b.bucket is not Bucket.TEXT_LIKE:
        return False
    if a.label != b.label:
        return False

    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox

    a_height = max(1, ay1 - ay0)
    vertical_gap = by0 - ay1
    if vertical_gap < 0 or vertical_gap > settings.region_merge_max_gap_frac * a_height:
        return False

    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    min_width = min(ax1 - ax0, bx1 - bx0)
    aligned_start = abs(bx0 - ax0) <= settings.region_merge_x_align_px
    overlapping = min_width > 0 and x_overlap / min_width >= 0.5
    if not (aligned_start or overlapping):
        return False

    return True


def _merge_group(group: list[Region]) -> Region:
    if len(group) == 1:
        return group[0]
    x0 = min(r.bbox[0] for r in group)
    y0 = min(r.bbox[1] for r in group)
    x1 = max(r.bbox[2] for r in group)
    y1 = max(r.bbox[3] for r in group)
    first = group[0]
    return Region(
        page_no=first.page_no,
        label=first.label,
        bucket=first.bucket,
        bbox=(x0, y0, x1, y1),
        score=min(r.score for r in group),  # report the weakest link's confidence, not an average
        order_index=first.order_index,
    )


def merge_adjacent_line_regions(regions: list[Region], settings: Settings) -> list[Region]:
    """Merges reading-order-consecutive same-label TEXT_LIKE regions that are
    vertically stacked with a small gap and horizontally aligned/overlapping -
    i.e. genuinely stacked lines of one visual block, not two unrelated
    regions that merely ended up reading-order-adjacent (e.g. two side-by-side
    columns' first rows, which this rejects via the horizontal-alignment
    check).

    `regions` must already be in reading order (order_index assigned) -
    merging only ever considers strictly consecutive list entries, so it is
    conservative about multi-column layouts by construction: it will not
    bridge across a region from a different column that happens to sit
    between two same-column lines in reading order.
    """
    if not regions:
        return regions

    merged: list[Region] = []
    group: list[Region] = [regions[0]]
    for region in regions[1:]:
        if _can_merge(group[-1], region, settings):
            group.append(region)
        else:
            merged.append(_merge_group(group))
            group = [region]
    merged.append(_merge_group(group))

    for i, region in enumerate(merged, start=1):
        region.order_index = i
    return merged
