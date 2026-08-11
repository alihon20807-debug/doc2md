"""Orchestrates the full render -> layout -> crop -> VLM -> assemble pipeline asynchronously."""

from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from doc2md.config import Settings
from doc2md.crop import crop_region, save_region_crop
from doc2md.layout_engines import get_layout_engine
from doc2md.markdown_builder import build_markdown
from doc2md.models import Bucket, DocumentResult, PageImage, Region, RegionResult
from doc2md.prompts import NOT_DIAGRAM_SENTINEL, PICTURE_PROMPT, TABLE_PROMPT, text_prompt
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


async def _process_region_async(
    page: PageImage,
    region: Region,
    vlm: AsyncVLLMClient | AsyncOpenRouterClient | VLMClient,
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
    content = await vlm.ask(crop, text_prompt(region.label), bucket=region.bucket)
    return RegionResult(region=region, markdown=content)


async def _process_region_safe_async(
    page: PageImage,
    region: Region,
    vlm: AsyncVLLMClient | AsyncOpenRouterClient | VLMClient,
    asset_dir: Path,
    settings: Settings,
    executor: ThreadPoolExecutor,
) -> RegionResult:
    """Like `_process_region`, but a single region's failure (VLM timeout, transient
    server error, etc.) becomes an inline error note instead of aborting the whole document.
    """
    try:
        return await _process_region_async(page, region, vlm, asset_dir, settings, executor)
    except Exception as exc:  # noqa: BLE001
        return RegionResult(
            region=region,
            markdown=f"> **doc2md: failed to extract this {region.label} region ({exc})**",
        )


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
    try:
        await vlm.health_check()

        detector = get_layout_engine(settings)
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=max(1, settings.vlm_max_concurrency))

        page_regions: dict[int, list[Region]] = {}
        for page in pages:
            page_regions[page.page_no] = await loop.run_in_executor(executor, detector.detect, page)

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
        already_done_regions = sum(len(results) for results in page_results.values())
        pending_regions = sum(len(page_regions[page.page_no]) for page in pending_pages)

        semaphore = asyncio.Semaphore(max(1, settings.vlm_max_concurrency))
        pbar = tqdm(
            total=already_done_regions + pending_regions,
            initial=already_done_regions,
            desc="Extracting regions (async vLLM)",
        )

        async def _worker_task(page: PageImage, region: Region) -> tuple[int, int, RegionResult]:
            async with semaphore:
                res = await _process_region_safe_async(page, region, vlm, asset_dir, settings, executor)
                pbar.update(1)
                return page.page_no, region.order_index, res

        tasks = []
        for page in pending_pages:
            for region in page_regions[page.page_no]:
                tasks.append(_worker_task(page, region))

        if tasks:
            raw_results = await asyncio.gather(*tasks)

            # Group results by page
            page_region_map: dict[int, dict[int, RegionResult]] = {}
            for page_no, order_idx, res in raw_results:
                page_region_map.setdefault(page_no, {})[order_idx] = res

            for page_no, order_dict in page_region_map.items():
                sorted_results = [order_dict[idx] for idx in sorted(order_dict)]
                page_results[page_no] = sorted_results

            _save_checkpoint(checkpoint_path, fingerprint, page_results)

        pbar.close()
        executor.shutdown(wait=False)

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

