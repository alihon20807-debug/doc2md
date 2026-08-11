"""Prompt templates sent to the local VLM (Gemma via llama.cpp) per region bucket."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous OCR and document-transcription assistant. You are given a single "
    "cropped region from a scanned document page. Respond with ONLY the requested Markdown "
    "content for that region - no preamble, no explanations, no surrounding commentary."
)

TITLE_LABELS = {"doc_title", "paragraph_title"}
FORMULA_LABELS = {"display_formula", "inline_formula", "formula_number"}

NOT_DIAGRAM_SENTINEL = "NOT_DIAGRAM"

_DEFAULT_TEXT_PROMPT = (
    "Transcribe the text in this image exactly as written, preserving reading order, line "
    "breaks between distinct items, and any obvious list/bullet structure as Markdown. Do not "
    "translate, summarize, or add anything not present in the image. Output only the "
    "transcribed Markdown text. If the region is blank or contains no legible text, output "
    "nothing."
)

_TITLE_PROMPT = (
    "This image is a document title or section heading. Transcribe its exact text as a "
    "Markdown heading (use '#' for a document title, '##' for a section/paragraph title). "
    "Output only the heading line."
)

_FORMULA_PROMPT = (
    "This image contains a mathematical formula. Transcribe it as LaTeX. Wrap a "
    "standalone/display formula in $$...$$ and an inline formula in $...$. Output only the "
    "LaTeX expression, nothing else."
)

TABLE_PROMPT = (
    "This image is a table. Transcribe it as a GitHub-flavored Markdown table, preserving row "
    "and column structure as closely as possible. If a cell is empty, leave it blank. Output "
    "only the Markdown table, nothing else."
)

PICTURE_PROMPT = (
    "This image is a figure from a document. Decide whether it is a structured diagram "
    "(flowchart, sequence diagram, state machine, org chart, tree, simple graph/network) that "
    "can be faithfully redrawn as a Mermaid diagram.\n\n"
    "- If YES: respond with ONLY a fenced Mermaid code block (```mermaid ... ```) that "
    "reproduces its structure and labels as closely as possible. Pick the most appropriate "
    "Mermaid diagram type (flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, etc.).\n"
    f"- If NO (it is a photo, screenshot, plot/chart with continuous data, illustration, or "
    f"anything you cannot faithfully represent structurally): respond with EXACTLY the single "
    f"word {NOT_DIAGRAM_SENTINEL} and nothing else."
)


def text_prompt(label: str) -> str:
    if label in TITLE_LABELS:
        return _TITLE_PROMPT
    if label in FORMULA_LABELS:
        return _FORMULA_PROMPT
    return _DEFAULT_TEXT_PROMPT
