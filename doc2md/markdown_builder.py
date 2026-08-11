"""Assemble per-region Markdown fragments into a single document."""

from __future__ import annotations

from doc2md.models import RegionResult


def build_markdown(page_results: dict[int, list[RegionResult]]) -> str:
    blocks: list[str] = []
    for page_no in sorted(page_results):
        for result in page_results[page_no]:
            content = result.markdown.strip()
            if content:
                blocks.append(content)
    return "\n\n".join(blocks) + "\n"
