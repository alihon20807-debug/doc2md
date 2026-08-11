"""Orchestrates the full render -> layout -> crop -> VLM -> assemble pipeline asynchronously."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from doc2md.config import Settings
from doc2md.crop import crop_region, save_region_crop
from doc2md.layout_engines import get_layout_engine
from doc2md.markdown_builder import build_markdown
from doc2md.models import Bucket, DocumentResult, PageImage, Region, RegionResult
from doc2md.ocr_engines import OCREngine, get_ocr_engine
from doc2md.prompts import FORMULA_LABELS, NOT_DIAGRAM_SENTINEL, PICTURE_PROMPT, TABLE_PROMPT, TITLE_LABELS, text_prompt
from doc2md.render import render_input_async
from doc2md.vlm_client import AsyncOpenRouterClient, AsyncVLLMClient, VLMClient, create_vlm_client

_MERMAID_KEYWORDS = (
    "graph ",
    "flowchart",
    "sequencediagram",
    "classdiagram",
    "statediagram",
    "erdiagram",
    "gantt",
    "mindmap",
    "journey",
)


def _extract_mermaid_block(reply: str) -> str | None:
    """Return a fenced Mermaid block for `reply`, or None if the VLM said NOT_DIAGRAM."""
    text = reply.strip()
    if not text or text == NOT_DIAGRAM_SENTINEL:
        return None
    if "```" in text:
        return text
    lowered = text.lower()
    if "mermaid" in lowered or any(lowered.startswith(kw) for kw in _MERMAID_KEYWORDS):
        return f"```mermaid\n{text}\n```"
    return None


def _ocr_eligible(region: Region) -> bool:
    """Titles and formulas always go to the VLM: OCR can't produce a Markdown
    heading or LaTeX, only plain recognized text."""
    return region.label not in TITLE_LABELS and region.label not in FORMULA_LABELS


async def _process_region_async(
    page: PageImage,
    region: Region,
    vlm: AsyncVLLMClient | AsyncOpenRouterClient | VLMClient,
    ocr: OCREngine | None,
    asset_dir: Path,
    settings: Settings,
    executor: ThreadPoolExecutor,
) -> RegionResult:
    loop = asyncio.get_running_loop()
    crop = await loop.run_in_executor(executor, crop_region, page, region, settings.crop_padding_px)

    if region.bucket is Bucket.TABLE:
        content = await vlm.ask(crop, TABLE_PROMPT, bucket=region.bucket)
        return RegionResult(region=region, markdown=content)

    if region.bucket is Bucket.PICTURE:
        reply = await vlm.ask(crop, PICTURE_PROMPT, bucket=region.bucket)
        mermaid_block = _extract_mermaid_block(reply)
        if mermaid_block is not None:
            return RegionResult(region=region, markdown=mermaid_block)

        asset_path = await loop.run_in_executor(executor, save_region_crop, crop, asset_dir, region)
        rel_path = f"{asset_dir.name}/{asset_path.name}"
        return RegionResult(
            region=region,
            markdown=f"![{region.label}]({rel_path})",
            asset_path=rel_path,
        )

    # Bucket.TEXT_LIKE
    if settings.ocr_fast_path and ocr is not None and _ocr_eligible(region):
        try:
            ocr_text = await loop.run_in_executor(executor, ocr.recognize, crop)
        except Exception:  # noqa: BLE001
            ocr_text = ""
        if ocr_text.strip():
            return RegionResult(region=region, markdown=ocr_text)
        # Empty/failed OCR result - fall through to the VLM below rather
        # than returning nothing for this region.

    content = await vlm.ask(crop, text_prompt(region.label), bucket=region.bucket)
    return RegionResult(region=region, markdown=content)


async def _process_region_safe_async(
    page: PageImage,
    region: Region,
    vlm: AsyncVLLMClient | AsyncOpenRouterClient | VLMClient,
    ocr: OCREngine | None,
    asset_dir: Path,
    settings: Settings,
    executor: ThreadPoolExecutor,
) -> RegionResult:
    """Like `_process_region`, but a single region's failure (VLM timeout, transient
    server error, etc.) becomes an inline error note instead of aborting the whole document.
    """
    try:
        return await _process_region_async(page, region, vlm, ocr, asset_dir, settings, executor)
    except Exception as exc:  # noqa: BLE001
        return RegionResult(
            region=region,
            markdown=f"> **doc2md: failed to extract this {region.label} region ({exc})**",
        )


async def _detect_page_safe(detector, page: PageImage, executor: ThreadPoolExecutor) -> list[Region]:
    """Runs layout detection for one page; a detection failure is logged and
    treated as zero regions for that page rather than aborting the whole
    document. Every built-in engine already catches its own internal
    failures (doc2md.layout_engines.base.safe_detect) and falls back to a
    full-page region instead of raising - this is a second layer of defense
    for anything that still escapes that (e.g. a future custom engine).
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(executor, detector.detect, page)
    except Exception as exc:  # noqa: BLE001
        print(f"doc2md: layout detection failed on page {page.page_no}: {exc}", file=sys.stderr)
        return []


def _checkpoint_path(output_dir: Path, source_name: str) -> Path:
    return output_dir / f"{source_name}.doc2md_progress.json"


def _region_to_dict(region: Region) -> dict:
    return {
        "page_no": region.page_no,
        "label": region.label,
        "bucket": region.bucket.value,
        "bbox": list(region.bbox),
        "score": region.score,
        "order_index": region.order_index,
    }


def _region_from_dict(data: dict) -> Region:
    return Region(
        page_no=data["page_no"],
        label=data["label"],
        bucket=Bucket(data["bucket"]),
        bbox=tuple(data["bbox"]),
        score=data["score"],
        order_index=data["order_index"],
    )


def _checkpoint_fingerprint(input_path: Path, settings: Settings, total_pages: int) -> dict:
    return {
        "input_path": str(input_path.resolve()),
        "layout_engine": settings.layout_engine,
        "render_dpi": settings.render_dpi,
        "total_pages": total_pages,
    }


def _load_checkpoint(
    checkpoint_path: Path, fingerprint: dict
) -> dict[int, list[RegionResult]]:
    if not checkpoint_path.exists():
        return {}
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("fingerprint") != fingerprint:
        print(
            f"doc2md: ignoring stale progress checkpoint at {checkpoint_path} "
            "(input/engine/dpi/page-count changed since it was written)",
            file=sys.stderr,
        )
        return {}
    try:
        return {
            int(page_no): [
                RegionResult(
                    region=_region_from_dict(item["region"]),
                    markdown=item["markdown"],
                    asset_path=item.get("asset_path"),
                )
                for item in results
            ]
            for page_no, results in data["pages"].items()
        }
    except (KeyError, TypeError, ValueError):
        print(f"doc2md: ignoring corrupt progress checkpoint at {checkpoint_path}", file=sys.stderr)
        return {}


def _save_checkpoint(
    checkpoint_path: Path, fingerprint: dict, page_results: dict[int, list[RegionResult]]
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "pages": {
            str(page_no): [
                {
                    "region": _region_to_dict(result.region),
                    "markdown": result.markdown,
                    "asset_path": result.asset_path,
                }
                for result in results
            ]
            for page_no, results in page_results.items()
        },
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


async def convert_document_async(input_path: Path, output_dir: Path, settings: Settings) -> DocumentResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = await render_input_async(input_path, dpi=settings.render_dpi)
    if not pages:
        raise ValueError(f"No pages found in {input_path}")
    source_name = pages[0].source_name
    asset_dir = output_dir / f"{source_name}{settings.assets_dirname_suffix}"
    md_path = output_dir / f"{source_name}.md"
    checkpoint_path = _checkpoint_path(output_dir, source_name)
    fingerprint = _checkpoint_fingerprint(input_path, settings, len(pages))

    vlm = create_vlm_client(settings)
    ocr = get_ocr_engine(settings) if settings.ocr_fast_path else None
    executor: ThreadPoolExecutor | None = None
    pbar: tqdm | None = None
    try:
        await vlm.health_check()

        detector = get_layout_engine(settings)
        # Sized independently of vlm_max_concurrency: this pool does CPU-bound
        # crop/detect work, not the I/O-bound VLM request fan-out, so it
        # shouldn't scale past the machine's actual core count.
        cpu_workers = min(max(1, settings.vlm_max_concurrency), os.cpu_count() or 4)
        executor = ThreadPoolExecutor(max_workers=cpu_workers)

        page_results: dict[int, list[RegionResult]] = (
            _load_checkpoint(checkpoint_path, fingerprint) if settings.resume else {}
        )
        if page_results:
            print(
                f"doc2md: resuming from checkpoint - {len(page_results)}/{len(pages)} "
                f"pages already done",
                file=sys.stderr,
            )
        pending_pages = [page for page in pages if page.page_no not in page_results]

        # Only detect layout for pages that still need it - already-checkpointed
        # pages skip this (potentially GPU-bound) step entirely on resume.
        page_regions: dict[int, list[Region]] = {}
        for page in pending_pages:
            page_regions[page.page_no] = await _detect_page_safe(detector, page, executor)

        already_done_regions = sum(len(results) for results in page_results.values())
        pending_regions = sum(len(regions) for regions in page_regions.values())

        semaphore = asyncio.Semaphore(max(1, settings.vlm_max_concurrency))
        pbar = tqdm(
            total=already_done_regions + pending_regions,
            initial=already_done_regions,
            desc="Extracting regions (async vLLM)",
        )

        async def _worker_task(page: PageImage, region: Region) -> tuple[int, RegionResult]:
            async with semaphore:
                res = await _process_region_safe_async(page, region, vlm, ocr, asset_dir, settings, executor)
                pbar.update(1)
                return region.order_index, res

        async def _page_task(page: PageImage) -> tuple[int, list[RegionResult]]:
            regions = page_regions[page.page_no]
            raw_results = await asyncio.gather(*[_worker_task(page, region) for region in regions])
            order_dict = dict(raw_results)
            return page.page_no, [order_dict[idx] for idx in sorted(order_dict)]

        # All pages' region tasks are scheduled up front and share one
        # semaphore, so cross-page concurrency is unchanged from a single
        # flat gather() - but checkpointing after each page completes (via
        # as_completed, rather than once after everything finishes) means a
        # crash mid-run only loses whichever pages hadn't finished yet.
        page_tasks = [asyncio.ensure_future(_page_task(page)) for page in pending_pages]
        for coro in asyncio.as_completed(page_tasks):
            page_no, sorted_results = await coro
            page_results[page_no] = sorted_results
            _save_checkpoint(checkpoint_path, fingerprint, page_results)

        markdown = build_markdown(page_results)
        md_path.write_text(markdown, encoding="utf-8")
        checkpoint_path.unlink(missing_ok=True)

        return DocumentResult(
            source_path=str(input_path),
            markdown=markdown,
            asset_dir=str(asset_dir) if asset_dir.exists() else None,
            page_results=page_results,
        )
    finally:
        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=False)
        await vlm.close()


def convert_document(input_path: Path, output_dir: Path, settings: Settings) -> DocumentResult:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(convert_document_async(input_path, output_dir, settings))
    return asyncio.run(convert_document_async(input_path, output_dir, settings))

