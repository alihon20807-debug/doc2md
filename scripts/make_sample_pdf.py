"""Generate a small synthetic PDF (title, paragraph, table, flowchart diagram)
for end-to-end smoke-testing the doc2md pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

OUT_PATH = Path(__file__).resolve().parent.parent / "sample_docs" / "test.pdf"


def make_flowchart_image() -> Image.Image:
    """Render Start -> Process -> End as a raster image (not vector shapes),
    so the layout model sees one 'picture'/'chart' blob rather than separate
    text boxes - this is what exercises the diagram-handling path."""
    img = Image.new("RGB", (700, 220), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    box_w, box_h, gap = 160, 70, 60
    y0 = 75
    labels = ["Start", "Process", "End"]
    centers = []
    for i, label in enumerate(labels):
        x0 = 20 + i * (box_w + gap)
        x1, y1 = x0 + box_w, y0 + box_h
        draw.rectangle([x0, y0, x1, y1], outline="black", width=3)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (box_w - tw) / 2, y0 + (box_h - th) / 2), label, fill="black", font=font)
        centers.append((x0, x1, (y0 + y1) / 2))
    for i in range(len(centers) - 1):
        _, x1, ymid = centers[i]
        x0_next, _, _ = centers[i + 1]
        draw.line([(x1, ymid), (x0_next, ymid)], fill="black", width=3)
        draw.polygon([(x0_next, ymid), (x0_next - 12, ymid - 8), (x0_next - 12, ymid + 8)], fill="black")
    return img


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    width, height = LETTER
    c = canvas.Canvas(str(OUT_PATH), pagesize=LETTER)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 90, "Sample Report")

    # Paragraph
    c.setFont("Helvetica", 11)
    paragraph = [
        "This is a short sample document used to verify the doc2md pipeline end to end.",
        "It contains a title, a paragraph of body text, a small table, and a simple",
        "flowchart diagram so each region-handling path can be exercised.",
    ]
    text_y = height - 130
    for line in paragraph:
        c.drawString(72, text_y, line)
        text_y -= 16

    # Table
    table_top = text_y - 40
    rows = [["Name", "Score", "Passed"], ["Alice", "92", "Yes"], ["Bob", "58", "No"]]
    col_widths = [120, 80, 80]
    row_h = 22
    c.setFont("Helvetica-Bold", 10)
    x = 72
    y = table_top
    for r_idx, row in enumerate(rows):
        x = 72
        for c_idx, cell in enumerate(row):
            c.rect(x, y - row_h, col_widths[c_idx], row_h, stroke=1, fill=0)
            c.setFont("Helvetica-Bold" if r_idx == 0 else "Helvetica", 10)
            c.drawString(x + 6, y - row_h + 7, str(cell))
            x += col_widths[c_idx]
        y -= row_h

    # Flowchart diagram: Start -> Process -> End, embedded as a raster image
    diagram = make_flowchart_image()
    diagram_w, diagram_h = 350, 110
    diagram_top = y - 40
    c.drawImage(
        ImageReader(diagram),
        72,
        diagram_top - diagram_h,
        width=diagram_w,
        height=diagram_h,
    )

    c.showPage()
    c.save()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
