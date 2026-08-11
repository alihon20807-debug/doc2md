from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from doc2md.classify import bucket_for_label
from doc2md.config import Settings
from doc2md.models import Bucket, PageImage, Region


class LayoutEngine(ABC):
    """Detects layout regions on a page, returned in reading order."""

    @abstractmethod
    def detect(self, page: PageImage) -> list[Region]: ...


def resolve_source_pdf(page: PageImage) -> Path | None:
    """Resolves a page's `source_name` back to its source PDF file, if any.

    Handles inputs where `source_name` lacks a `.pdf` suffix (e.g. a
    directory-of-page-images input) by also checking for a sibling PDF with
    the suffix appended. Returns None if no such PDF exists.
    """
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
