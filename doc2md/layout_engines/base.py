from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from doc2md.classify import bucket_for_label
from doc2md.config import Settings
from doc2md.layout_engines.dedup import suppress_contained_duplicates
from doc2md.layout_engines.merge import merge_adjacent_line_regions
from doc2md.layout_engines.reading_order import assign_reading_order
from doc2md.models import Bucket, PageImage, Region


class LayoutEngine(ABC):
    """Detects layout regions on a page, returned in reading order."""

    @abstractmethod
    def detect(self, page: PageImage) -> list[Region]: ...


def make_region(
    page_no: int,
    raw_label: str,
    label_map: dict[str, Bucket],
    skip_labels: frozenset[str],
    bbox: tuple[int, int, int, int],
    score: float,
    order_index: int = 0,
) -> Region | None:
    """Builds a Region from a raw detector label, or None if it maps to SKIP.

    Replaces the `bucket = bucket_for_label(...); if bucket is not SKIP:
    regions.append(Region(...))` pattern previously duplicated in every engine.
    """
    bucket = bucket_for_label(raw_label, label_map, skip_labels)
    if bucket is Bucket.SKIP:
        return None
    return Region(
        page_no=page_no,
        label=raw_label,
        bucket=bucket,
        bbox=bbox,
        score=score,
        order_index=order_index,
    )


def finalize_regions(regions: list[Region], settings: Settings) -> list[Region]:
    """Suppresses near-duplicate boxes and assigns reading order.

    Used by every engine except MinerU, whose PP-DocLayoutV2 model has its
    own built-in reading-order head.
    """
    deduped = suppress_contained_duplicates(
        regions, containment_threshold=settings.region_containment_threshold
    )
    ordered = assign_reading_order(deduped, overlap_frac=settings.reading_order_overlap_frac)
    if settings.region_merge_enabled:
        ordered = merge_adjacent_line_regions(ordered, settings)
    return ordered


def safe_detect(
    engine_name: str,
    page: PageImage,
    settings: Settings,
    label_map: dict[str, Bucket],
    detect_fn: Callable[[], list[Region]],
) -> list[Region]:
    """Runs `detect_fn`, giving every engine the same failure contract:

    a detection failure (or an engine returning zero regions) is logged once
    and falls back to a single full-page region, instead of either aborting
    the whole document or silently dropping the page's content.
    """
    try:
        regions = detect_fn()
    except Exception as exc:  # noqa: BLE001
        print(f"doc2md: {engine_name} failed on page {page.page_no}: {exc}", file=sys.stderr)
        return full_page_fallback_region(page, settings, label_map)
    if not regions:
        return full_page_fallback_region(page, settings, label_map)
    return regions


def resolve_source_pdf(page: PageImage) -> Path | None:
    """Resolves a page back to its source PDF file, if any.

    Prefers `page.source_path` (set by render.py for real PDF inputs, an
    absolute path - this is what actually makes this resolvable regardless
    of the process's current working directory). Falls back to a
    stem-based, cwd-relative guess for inputs with no resolved source_path
    (e.g. a directory-of-page-images input with a same-named sibling PDF,
    which render.py has no path for since it never opens one). Returns None
    if no such PDF can be found.
    """
    if page.source_path is not None and page.source_path.exists() and page.source_path.suffix.lower() == ".pdf":
        return page.source_path

    possible_pdf = Path(page.source_name)
    if not possible_pdf.suffix and possible_pdf.with_suffix(".pdf").exists():
        possible_pdf = possible_pdf.with_suffix(".pdf")
    if possible_pdf.exists() and possible_pdf.suffix.lower() == ".pdf":
        return possible_pdf
    return None


def _content_bbox(page: PageImage, padding: int) -> tuple[int, int, int, int]:
    """Finds the bounding box of actual non-blank ink on the page, padded.

    Falls back to the full page if no content is found (a genuinely blank
    page) or if the detected content already spans nearly the whole page.
    Used by `full_page_fallback_region()` so a page with a small cluster of
    real text on an otherwise blank canvas doesn't get sent to the VLM as
    one giant mostly-empty crop - a real, confirmed cause of VLM
    hallucination (see docs/OPTIMIZATION_PLAN.md's "pre-existing
    hallucination bug" entry: a 2666x1500 page whose only real content was a
    228x188px cluster of text, ~5% of the page area, produced a fabricated
    "CONFIDENTIALITY NOTICE" paragraph with no basis in the source).
    """
    from PIL import ImageOps

    gray = page.image.convert("L")
    inverted = ImageOps.invert(gray)
    thresholded = inverted.point(lambda p: 255 if p > 20 else 0)
    bbox = thresholded.getbbox()
    if bbox is None:
        return (0, 0, page.width, page.height)
    x0, y0, x1, y1 = bbox
    if (x1 - x0) * (y1 - y0) >= 0.9 * page.width * page.height:
        return (0, 0, page.width, page.height)
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(page.width, x1 + padding),
        min(page.height, y1 + padding),
    )


def full_page_fallback_region(page: PageImage, settings: Settings, label_map: dict[str, Bucket], label: str = "text") -> list[Region]:
    """Synthesizes one region covering the page's actual content, used when
    an engine detects nothing - tightened to the real content bounding box
    rather than always the literal full page (see `_content_bbox()`)."""
    bucket = bucket_for_label(label, label_map, settings.skip_labels)
    if bucket is Bucket.SKIP:
        return []
    return [
        Region(
            page_no=page.page_no,
            label=label,
            bucket=bucket,
            bbox=_content_bbox(page, settings.crop_padding_px),
            score=1.0,
            order_index=0,
        )
    ]
