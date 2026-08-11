"""Core data structures shared across the doc2md pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image


class Bucket(str, Enum):
    """How a detected region should be routed for content extraction."""

    TEXT_LIKE = "text_like"
    TABLE = "table"
    PICTURE = "picture"
    SKIP = "skip"


@dataclass
class PageImage:
    """A single rasterized page ready for layout detection."""

    page_no: int  # 1-indexed
    image: Image.Image
    source_name: str  # stem of the input file, used for asset naming
    source_path: Path | None = None  # resolved path to the source PDF, if the input was one

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass
class Region:
    """A single layout-detected bounding box on a page."""

    page_no: int
    label: str
    bucket: Bucket
    bbox: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax, pixel coords on the page image
    score: float
    order_index: int  # reading-order rank from the layout model, 1-indexed


@dataclass
class RegionResult:
    """The markdown content produced for one region."""

    region: Region
    markdown: str
    asset_path: str | None = None  # relative path to a saved crop, if this region embeds an image


@dataclass
class DocumentResult:
    """The fully assembled conversion output for one input document."""

    source_path: str
    markdown: str
    asset_dir: str | None
    page_results: dict[int, list[RegionResult]] = field(default_factory=dict)
