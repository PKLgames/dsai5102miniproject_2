"""Raster preview of the generated PPTX (no LibreOffice on this machine).

Draws shape fills, pictures and wrapped text with DejaVu Sans, which is slightly
wider than Arial, so the preview is a pessimistic check.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).resolve().parent.parent
PPTX = ROOT / "slides" / "DSAI5102_Group16_Slides.pptx"
OUT = ROOT / "slides" / "preview"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fonts import dejavu_paths                                   # noqa: E402

FONT_DIR = dejavu_paths()[0].parent
DPI = 110
EMU = 914400.0

CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size_pt: float, bold: bool) -> ImageFont.FreeTypeFont:
    key = (int(size_pt * DPI / 72), bold)
    if key not in CACHE:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        CACHE[key] = ImageFont.truetype(str(FONT_DIR / name), key[0],
                                        layout_engine=ImageFont.Layout.BASIC)
    return CACHE[key]


def rgb(color, default=(0, 0, 0)):
    try:
        return tuple(int(str(color), 16) >> s & 0xFF for s in (16, 8, 0))
    except Exception:
        return default


def wrap(text, f, maxw):
    lines, cur = [], ""
    for word in text.split(" "):
        cand = f"{cur} {word}".strip()
        if f.getlength(cand) <= maxw or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_text_frame(draw, tf, x, y, w, h):
    cy = y
    for i, p in enumerate(tf.paragraphs):
        runs = [(r.text, (r.font.size.pt if r.font.size else 18), bool(r.font.bold))
                for r in p.runs if r.text]
        if not runs:
            continue
        size = max(r[1] for r in runs)
        gap = p.space_before.pt if p.space_before else 0.0
        cy += gap * DPI / 72 if i else 0
        # wrap by accumulating run widths (bold leading runs measured bold)
        words = []
        for text, sz, bold in runs:
            for word in text.split(" "):
                if word:
                    words.append((word, sz, bold))
        lines, cur, curw = [], [], 0.0
        space_w = lambda sz, b: font(sz, b).getlength(" ")
        for word, sz, bold in words:
            fnt = font(sz, bold)
            ww = fnt.getlength(word)
            add = ww + (space_w(sz, bold) if cur else 0)
            if curw + add <= w or not cur:
                cur.append((word, sz, bold))
                curw += add
            else:
                lines.append(cur)
                cur = [(word, sz, bold)]
                curw = ww
        if cur:
            lines.append(cur)
        for line in lines:
            lh = max(l[1] for l in line) * 1.21 * DPI / 72
            if cy + lh > y + h + 2:
                draw.rectangle([x, cy, x + w, cy + lh], outline=(255, 0, 0))
            cx = x
            for j, (word, sz, bold) in enumerate(line):
                fnt = font(sz, bold)
                seg = (" " if j else "") + word
                draw.text((cx, cy), seg, font=fnt, fill=(20, 20, 20))
                cx += fnt.getlength(seg)
            cy += lh


def render():
    OUT.mkdir(parents=True, exist_ok=True)
    prs = Presentation(PPTX)
    sw = int(prs.slide_width / EMU * DPI)
    sh = int(prs.slide_height / EMU * DPI)
    for n, slide in enumerate(prs.slides, 1):
        img = Image.new("RGB", (sw, sh), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for shape in slide.shapes:
            x = shape.left / EMU * DPI
            y = shape.top / EMU * DPI
            w = shape.width / EMU * DPI
            h = shape.height / EMU * DPI
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                import io
                pic = Image.open(io.BytesIO(shape.image.blob))
                pic = pic.convert("RGB").resize((max(1, int(w)), max(1, int(h))))
                img.paste(pic, (int(x), int(y)))
                continue
            if shape.has_table if hasattr(shape, "has_table") else False:
                tbl = shape.table
                cy = y
                for i, row in enumerate(tbl.rows):
                    rh = row.height / EMU * DPI
                    cx = x
                    for j, cell in enumerate(row.cells):
                        cw = tbl.columns[j].width / EMU * DPI
                        fill = rgb(cell.fill.fore_color.rgb, (255, 255, 255)) if cell.fill.type == 1 else (255, 255, 255)
                        draw.rectangle([cx, cy, cx + cw, cy + rh], fill=fill, outline=(210, 215, 222))
                        if cell.text_frame.paragraphs[0].runs:
                            draw_text_frame(draw, cell.text_frame, cx + 5, cy + 4, cw - 10, rh - 6)
                        cx += cw
                    cy += rh
                continue
            # filled shape
            try:
                if shape.fill.type == 1:
                    draw.rounded_rectangle([x, y, x + w, y + h], radius=4,
                                           fill=rgb(shape.fill.fore_color.rgb))
            except Exception:
                pass
            if shape.has_text_frame and shape.text_frame.text.strip():
                draw_text_frame(draw, shape.text_frame, x, y, w, h)
        img.save(OUT / f"s{n:02d}.png")
    print("rendered", n, "slides ->", OUT)


if __name__ == "__main__":
    render()
