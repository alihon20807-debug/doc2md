from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from doc2md.classify import bucket_for_label
from doc2md.config import Settings
from doc2md.layout_engines.dedup import suppress_contained_duplicates
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
    return assign_reading_order(deduped, overlap_frac=settings.reading_order_overlap_frac)


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


def full_page_fallback_region(page: PageImage, settings: Settings, label_map: dict[str, Bucket], label: str = "text") -> list[Region]:
    """Synthesizes one full-page region, used when an engine detects nothing."""
    bucket = bucket_for_label(label, label_map, settings.skip_labels)
    if bucket is Bucket.SKIP:
        return []
    return [
        Region(
            page_no=page.page_no,
            label=label,
            bucket=bucket,
            bbox=(0, 0, page.width, page.height),
            score=1.0,
            order_index=0,
        )
    ]
