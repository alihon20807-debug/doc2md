"""Cropping detected regions out of a page image, and saving embeddable assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from doc2md.models import PageImage, Region


def crop_region(page: PageImage, region: Region, padding: int) -> Image.Image:
    x0, y0, x1, y1 = region.bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(page.width, x1 + padding)
    y1 = min(page.height, y1 + padding)
    return page.image.crop((x0, y0, x1, y1))


def region_asset_filename(region: Region) -> str:
    return f"p{region.page_no:03d}_{region.order_index:03d}_{region.label}.png"


def save_region_crop(crop: Image.Image, asset_dir: Path, region: Region) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / region_asset_filename(region)
    crop.save(path, format="PNG")
    return path
