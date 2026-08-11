"""Map a layout engine's labels to a handling bucket (text / table / picture / skip)."""

from __future__ import annotations

from doc2md.models import Bucket

# Full PP-DocLayoutV2 (MinerU engine) label set -> bucket. Any label not listed
# here defaults to TEXT_LIKE (safe fallback: it gets transcribed rather than
# silently dropped).
PP_DOCLAYOUT_LABEL_TO_BUCKET: dict[str, Bucket] = {
    "abstract": Bucket.TEXT_LIKE,
    "algorithm": Bucket.TEXT_LIKE,
    "aside_text": Bucket.TEXT_LIKE,
    "chart": Bucket.PICTURE,
    "content": Bucket.TEXT_LIKE,
    "display_formula": Bucket.TEXT_LIKE,
    "doc_title": Bucket.TEXT_LIKE,
    "figure_title": Bucket.TEXT_LIKE,
    "footer": Bucket.SKIP,
    "footer_image": Bucket.PICTURE,
    "footnote": Bucket.TEXT_LIKE,
    "formula_number": Bucket.TEXT_LIKE,
    "header": Bucket.SKIP,
    "header_image": Bucket.PICTURE,
    "image": Bucket.PICTURE,
    "inline_formula": Bucket.TEXT_LIKE,
    "number": Bucket.SKIP,
    "paragraph_title": Bucket.TEXT_LIKE,
    "reference": Bucket.TEXT_LIKE,
    "reference_content": Bucket.TEXT_LIKE,
    "seal": Bucket.PICTURE,
    "table": Bucket.TABLE,
    "text": Bucket.TEXT_LIKE,
    "vertical_text": Bucket.TEXT_LIKE,
    "vision_footnote": Bucket.TEXT_LIKE,
}

# DocLayNet-style label set (IBM Docling engine, docling-layout-heron) -> bucket.
DOCLING_LABEL_TO_BUCKET: dict[str, Bucket] = {
    "Caption": Bucket.TEXT_LIKE,
    "Checkbox-Selected": Bucket.TEXT_LIKE,
    "Checkbox-Unselected": Bucket.TEXT_LIKE,
    "Code": Bucket.TEXT_LIKE,
    "Document Index": Bucket.TEXT_LIKE,
    "Footnote": Bucket.TEXT_LIKE,
    "Form": Bucket.TEXT_LIKE,
    "Formula": Bucket.TEXT_LIKE,
    "Key-Value Region": Bucket.TEXT_LIKE,
    "List-item": Bucket.TEXT_LIKE,
    "Page-footer": Bucket.SKIP,
    "Page-header": Bucket.SKIP,
    "Picture": Bucket.PICTURE,
    "Section-header": Bucket.TEXT_LIKE,
    "Table": Bucket.TABLE,
    "Text": Bucket.TEXT_LIKE,
    "Title": Bucket.TEXT_LIKE,
}

# Microsoft MarkItDown layout label set -> bucket.
MARKITDOWN_LABEL_TO_BUCKET: dict[str, Bucket] = {
    "Caption": Bucket.TEXT_LIKE,
    "Code": Bucket.TEXT_LIKE,
    "Footnote": Bucket.TEXT_LIKE,
    "Form": Bucket.TEXT_LIKE,
    "List-item": Bucket.TEXT_LIKE,
    "Page-footer": Bucket.SKIP,
    "Page-header": Bucket.SKIP,
    "Picture": Bucket.PICTURE,
    "Section-header": Bucket.TEXT_LIKE,
    "Table": Bucket.TABLE,
    "Text": Bucket.TEXT_LIKE,
    "Title": Bucket.TEXT_LIKE,
    "figure": Bucket.PICTURE,
    "footer": Bucket.SKIP,
    "header": Bucket.SKIP,
    "image": Bucket.PICTURE,
    "paragraph": Bucket.TEXT_LIKE,
    "picture": Bucket.PICTURE,
    "table": Bucket.TABLE,
    "title": Bucket.TEXT_LIKE,
    "text": Bucket.TEXT_LIKE,
}

# DocLayout-YOLO label set -> bucket.
DOCLAYOUT_YOLO_LABEL_TO_BUCKET: dict[str, Bucket] = {
    "title": Bucket.TEXT_LIKE,
    "plain text": Bucket.TEXT_LIKE,
    "header": Bucket.SKIP,
    "footer": Bucket.SKIP,
    "page_number": Bucket.SKIP,
    "table": Bucket.TABLE,
    "figure": Bucket.PICTURE,
    # Formulas go to TEXT_LIKE (-> LaTeX transcription via prompts.FORMULA_LABELS),
    # matching every other engine - not PICTURE, which would send them through
    # the diagram/Mermaid decision instead of LaTeX transcription.
    "isolate_formula": Bucket.TEXT_LIKE,
    "formula": Bucket.TEXT_LIKE,
    "caption": Bucket.TEXT_LIKE,
    "abandon": Bucket.SKIP,
}


def bucket_for_label(label: str, mapping: dict[str, Bucket], skip_labels: frozenset[str]) -> Bucket:
    if label in skip_labels:
        return Bucket.SKIP
    return mapping.get(label, Bucket.TEXT_LIKE)
