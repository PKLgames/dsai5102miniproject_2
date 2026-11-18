"""Build the DSAI5102 Format-1 slide deck (English, 16:9) with python-pptx.

Run with the interpreter that has python-pptx:
  /home/pklgames/coding/dsai5102MiniProject/.venv/bin/python slides/build_slides.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
OUT = ROOT / "slides" / "DSAI5102_Group16_Slides.pptx"

FONT_DIR = Path(
    "/home/pklgames/coding/dsai5104report/.venv/lib/python3.10/site-packages"
    "/matplotlib/mpl-data/fonts/ttf"
)
F_REG = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 100)
F_BOLD = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 100)

INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x5A, 0x64, 0x72)
ACCENT = RGBColor(0x1F, 0x6F, 0xB4)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
AMBER = RGBColor(0xC7, 0x7F, 0x1A)
RULE = RGBColor(0xD9, 0xDE, 0xE5)
PALE = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = 13.333, 7.5
ML = 0.78                    # left margin
CW = W - 2 * ML              # content width
FONT = "Arial"

WARNINGS: list[str] = []


# ---------------------------------------------------------------- text metric
def text_width_in(text: str, size_pt: float, bold: bool = False) -> float:
    font = F_BOLD if bold else F_REG
    px = font.getlength(text) / 100.0 * size_pt   # 100 px em -> points
    return px / 72.0


def wrapped_height_in(runs: list[tuple[str, bool]], size_pt: float, width_in: float) -> float:
    """Greedy wrap of one paragraph, returned in inches."""
    space = 0.0
    lines = 1
    for text, bold in runs:
        for word in text.split(" "):
            if not word:
                continue
            w = text_width_in(word, size_pt, bold)
            if space + w <= width_in or space == 0.0:
                space += w + text_width_in(" ", size_pt, bold)
            else:
                lines += 1
                space = w + text_width_in(" ", size_pt, bold)
    return lines * size_pt * 1.21 / 72.0


# ------------------------------------------------------------------ primitives
def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(slide, x, y, w, h, text="", size=18, bold=False, color=INK,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    r = p.add_run()
    r.text = text
    f = r.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color
    return tb


def rule(slide, x, y, w, h=0.035, color=RULE):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def panel(slide, x, y, w, h, fill=PALE, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    sh.adjustments[0] = 0.04
    return sh


def bullets(slide, x, y, w, h, items, size=17, gap=9, lead_color=INK, body_color=INK,
            bullet_char="\u2022", indent=0.26, check=True, tag=""):
    """items: list of (lead, body) or plain strings. lead is bold."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    used = 0.0
    for i, item in enumerate(items):
        lead, body = (item, "") if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.02
        p.space_before = Pt(0 if i == 0 else gap)
        runs = [(f"{bullet_char}  ", True)] if bullet_char else []
        if lead:
            runs.append((lead, True))
        if body:
            runs.append(((" " if lead else "") + body, False))
        for text, bold in runs:
            r = p.add_run()
            r.text = text
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = lead_color if (bold and lead and text != f"{bullet_char}  ") else body_color
        used += wrapped_height_in(runs, size, w - indent) + (0 if i == 0 else gap / 72.0)
    if check and used > h + 0.02:
        WARNINGS.append(f"{tag}: bullets overflow {used:.2f}in > {h:.2f}in (size {size})")
    return tb


def picture(slide, name, x, y, w, h, check=True, tag=""):
    path = FIGS / name
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    left = x + (w - pw) / 2
    top = y + (h - ph) / 2
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(pw), Inches(ph))
    return pw, ph


def footer(slide, num, total, label="DSAI5102 \u00b7 Group 16"):
    rule(slide, ML, 6.86, CW, 0.012, RULE)
    box(slide, ML, 6.95, 6.0, 0.3, label, 10.5, False, MUTED)
    box(slide, W - ML - 1.2, 6.95, 1.2, 0.3, f"{num} / {total}", 10.5, False, MUTED,
        align=PP_ALIGN.RIGHT)


def heading(slide, kicker, title, sub=None):
    if kicker:
        box(slide, ML, 0.50, CW, 0.3, kicker.upper(), 11.5, True, ACCENT)
    th = wrapped_height_in([(title, True)], 28, CW)
    box(slide, ML, 0.80, CW, th + 0.06, title, 28, True, INK, spacing=0.96)
    rule(slide, ML, 0.80 + th + 0.06, 1.15, 0.045, ACCENT)
    if sub:
        box(slide, ML, 1.88, CW - 0.4, 0.4, sub, 15, False, MUTED, italic=True)


def table(slide, x, y, w, rows, col_w, head_size=13, body_size=13, row_h=0.36):
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * n_r))
    tbl = shape.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(int(Inches(w) * cw))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if i == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(head_size if i == 0 else body_size)
            r.font.bold = i == 0
            r.font.color.rgb = INK if i == 0 else INK
            if i == 0:
                continue
    return shape


# --------------------------------------------------------------------- slides
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    T = 14

    # 1 ----------------------------------------------------------------- title
    s = blank(prs)
    rule(s, ML, 1.02, 1.3, 0.05, ACCENT)
    box(s, ML, 1.24, CW, 0.34,
        "DSAI5102 PRINCIPLES OF DATA SCIENCE  \u00b7  MINI-PROJECT, FORMAT 1", 12, True, ACCENT)
    box(s, ML, 1.66, 10.6, 1.72, "How many indicators does development monitoring need?",
        40, True, INK, spacing=0.96)
    box(s, ML, 3.24, 9.9, 0.5, "A minimal, stable development index and three policy archetypes",
        20, False, MUTED, italic=True)
    rule(s, ML, 3.95, CW - 1.0, 0.012, RULE)
    box(s, ML, 4.15, 6.0, 0.34, "Group 16", 13.5, True, INK)
    box(s, ML, 4.50, 8.4, 0.34, "Chen Wanchuan  \u00b7  Lu Yuqing  \u00b7  Yuen Lok Yan", 15, False, INK)
    box(s, ML, 5.05, 10.2, 0.32,
        "World Bank World Development Indicators  \u00b7  198 economies  \u00b7  28 indicators  \u00b7  2005\u20132022",
        12.5, False, MUTED)
    box(s, ML, 5.60, 10.2, 0.32,
        "All results reproducible from the released code: 8 scripts, under 15 minutes, laptop CPU",
        12.5, False, MUTED)
    footer(s, 1, T, "DSAI5102 \u00b7 Group 16 \u00b7 Format 1")

    # 2 --------------------------------------------------------------- problem
    s = blank(prs)
    heading(s, "The problem", "Monitoring is indicator-rich and data-poor at the same time")
    bullets(s, ML, 1.94, 6.05, 3.5, [
        ("The apparatus keeps growing.",
         "The SDG framework defines 231 unique indicators; the HDI uses 3, the MPI 10. "
         "Each added dimension is justified — and each one widens the gap between the index and the data that exist."),
        ("The gap is not random.",
         "Missingness is concentrated in the countries the monitoring is meant to guide. "
         "An index that needs 28 series works best where it is needed least."),
    ], size=15.5, gap=13)
    panel(s, ML, 5.55, 6.05, 1.05, PALE)
    box(s, ML + 0.22, 5.68, 5.6, 0.8,
        "34.5% of country-years have no tertiary enrolment, 27.8% no secondary enrolment, "
        "22.6% no energy use — the capacity-intensive series are the sparse ones.",
        12, False, INK)
    picture(s, "S1_missing.png", 6.95, 1.92, 5.75, 3.45, tag="S1")
    box(s, 6.95, 5.66, 5.75, 0.6,
        "Share of country-years with no observation, before any imputation. "
        "Panel coverage otherwise sits near 94% in every year.", 11, False, MUTED, italic=True)
    footer(s, 2, T)

    # 3 ------------------------------------------------------------------ data
    s = blank(prs)
    heading(s, "Data and questions", "A country-level profile, and three questions about how little it takes")
    bullets(s, ML, 1.94, 5.85, 4.4, [
        ("Source.", "World Bank WDI via the official API, CC BY 4.0 — 1,498 series in the collection, "
                    "28 used here; frozen response kept with a SHA-256."),
        ("Panel.", "198 reporting entities \u00d7 28 indicators \u00d7 2005\u20132022 = 3,517 country-years."),
        ("Unit of analysis.", "country means over 2005\u20132022 \u2014 the object of policy, and far better observed than any single year."),
        ("Imputation.", "k-NN (k = 5) at country level; 198 of 5,544 cells (3.6%)."),
        ("Uptake.", "The WDI record is cited ~6,800 times in OpenAlex, the PCA wealth-index papers "
                    "~7,100 and ~3,800 \u2014 a well-trodden instrument, not a novelty."),
    ], size=14.2, gap=10)
    panel(s, 6.95, 1.92, 5.6, 3.62, PALE)
    box(s, 7.20, 2.06, 5.1, 0.3, "THREE QUESTIONS", 11.5, True, ACCENT)
    bullets(s, 7.20, 2.44, 5.1, 3.0, [
        ("RQ1  Is there one dominant axis \u2014 and is it stable?", "One component, four perturbation tests."),
        ("RQ2  What is the smallest set that reproduces it and still predicts?", "Fidelity against usefulness: two criteria, two answers."),
        ("RQ3  How many kinds of country are there, and who is moving?", "A typology chosen on stability, not on separation."),
    ], size=14, gap=12, bullet_char="")
    panel(s, ML, 5.86, CW, 0.78, PALE)
    box(s, ML + 0.24, 5.98, CW - 0.6, 0.6,
        "Six domains  \u00b7  Economy 6  \u00b7  Human development 7  \u00b7  Urban & energy 5  \u00b7  "
        "Environment 5  \u00b7  Digital 3  \u00b7  Agriculture & food 2", 13, False, INK)
    footer(s, 3, T)

    # 4 --------------------------------------------------------------- methods
    s = blank(prs)
    heading(s, "Method", "Seven steps, one of which is the discipline that makes the rest credible")
    steps = [
        ("M1", "Index", "PCA on the 198 \u00d7 28 country matrix"),
        ("M2", "Label-blind\nshortlist", "sparse PCA, \u03b1 chosen at the knee"),
        ("M3", "Label-aware\nshortlist", "greedy forward selection, nested CV"),
        ("M4", "Classify", "income groups, 3 models, 5\u00d75 CV"),
        ("M5", "Archetypes", "K-means k = 2\u20266, bootstrap ARI"),
        ("M6", "Dynamics", "two windows, rank persistence"),
        ("M7", "Robustness", "6 checks that could overturn a claim"),
    ]
    x = ML
    bw, gap = 1.60, 0.115
    for i, (code, name, desc) in enumerate(steps):
        fill = ACCENT if i in (1, 2) else PALE
        fg = WHITE if i in (1, 2) else INK
        panel(s, x, 1.96, bw, 1.05, fill)
        box(s, x + 0.10, 2.05, bw - 0.2, 0.26, code, 11, True, fg if i in (1, 2) else ACCENT)
        box(s, x + 0.10, 2.32, bw - 0.2, 0.55, name, 12.5, True, fg)
        box(s, x, 3.10, bw, 0.95, desc, 10.5, False, MUTED)
        if i < len(steps) - 1:
            box(s, x + bw + 0.005, 2.40, gap + 0.10, 0.3, "\u203a", 15, True, RULE)
        x += bw + gap
    panel(s, ML, 4.10, CW, 0.92, PALE)
    box(s, ML + 0.22, 4.23, CW - 0.5, 0.7,
        "The credibility step is M3/M7: selecting indicators on the full sample and then reporting "
        "cross-validated accuracy inflates the result. We report both protocols and measure the gap "
        "instead of assuming it away.", 13.5, False, INK)
    bullets(s, ML, 5.32, 5.7, 1.25, [
        ("Stability is the primary criterion for both the index and the typology.", ""),
        ("Everything is seeded; B = 200 bootstraps; 8 scripts, no GPU.", ""),
    ], size=12.5, gap=5)
    panel(s, 6.95, 5.28, 5.6, 1.12, PALE)
    box(s, 7.15, 5.40, 5.2, 0.9,
        "One full re-run: 8 scripts, under 15 minutes on a laptop CPU; every number in the report "
        "is regenerated from the released artefacts.", 12, False, INK)
    footer(s, 4, T)


    # 5 -------------------------------------------------------- why these models
    s = blank(prs)
    heading(s, "Method  \u00b7  why these models",
            "Every step is chosen so it remains defensible without labels")
    rows = [
        ["Model (role)", "What it buys", "What it costs", "Used this way elsewhere"],
        ["PCA (index)", "unsupervised, variance-optimal, exactly reproducible",
         "linear; dense weights are hard to defend", "DHS wealth indices; an OECD/JRC option"],
        ["Sparse PCA (blind shortlist)", "names a shortlist and never sees a label",
         "non-convex; the set moves under resampling \u2014 we measure it", "wherever a component must be read"],
        ["Forward selection + nested CV (aware shortlist)", "targets the practical goal; gives a leak-free estimate",
         "greedy; unstable across folds", "standard when prediction is the goal"],
        ["LogReg / RF / HistGB / MLP (classification)", "three biases: a result across all three is robust",
         "trees unstable at n = 198; the label is administrative; the MLP is 0.05 worse", "Random Forest is the default baseline"],
        ["K-means, GMM check (archetypes)", "simple, and describable to a policy audience",
         "k not identified; boundary countries unstable", "stability-based k without labels"],
    ]
    table(s, ML, 1.88, CW, rows, [0.235, 0.275, 0.265, 0.225],
          head_size=12, body_size=10.5, row_h=0.62)
    panel(s, ML, 5.86, CW, 0.62, PALE)
    box(s, ML + 0.22, 5.94, CW - 0.5, 0.5,
        "Rejected, and why:  rotated factor analysis (the rotation is not identified)  \u00b7  autoencoders "
        "(expressive but not identifiable)  \u00b7  hierarchical/PAM clustering (implies a tree the data do not "
        "support)  \u00b7  tuned boosting (n = 198 is too small to tune honestly).",
        11.5, False, INK)
    footer(s, 5, T)

    # 5 ---------------------------------------------------------------- result 1
    s = blank(prs)
    heading(s, "Result 1 \u00b7 the index", "One axis carries 47.9% of the variance \u2014 but the ordering is what to trust")
    picture(s, "S2_index.png", ML, 1.92, 8.15, 4.72, tag="S2")
    bullets(s, 9.15, 1.94, 3.75, 4.3, [
        ("PC1 = 47.9%", "95% bootstrap CI 44.4\u201351.6%; PC1+PC2 = 56.6%."),
        ("Robust to its inputs.", "\u03c1 \u2265 0.9986 after deleting any one indicator; \u2265 0.9974 across bootstrap resamples."),
        ("Robust to time.", "\u03c1 \u2265 0.9826 between four non-overlapping windows."),
        ("And it is not just income.", "\u03c1 = 0.95 with GDP per capita, but 5% of rank variance is education and health."),
    ], size=13.5, gap=10)
    footer(s, 6, T)

    # 6 ---------------------------------------------------------------- result 2
    s = blank(prs)
    heading(s, "Result 2 \u00b7 how few", "Twelve indicators reproduce the ranking \u2014 three appear to, and that is selection bias")
    picture(s, "F3_selection.png", ML, 1.92, 8.5, 3.62, tag="F3")
    bullets(s, 9.55, 1.96, 3.35, 4.2, [
        ("Label-blind: 12.", "sparse PCA at \u03b1* = 0.35, \u03c1 = 0.9935 with the full index; 6 of 12 survive >90% of bootstraps."),
        ("Label-aware: 3.", "GDP per capita, mobile subscriptions, GDP growth \u2014 \u03c1 = 0.917."),
        ("They barely overlap.", "Jaccard 0.14: 2 shared indicators out of 22."),
    ], size=13, gap=11)
    panel(s, ML, 5.72, CW, 0.86, PALE)
    box(s, ML + 0.24, 5.84, CW - 0.6, 0.66,
        "Fidelity asks which directions carry the variance; usefulness asks which series draw the class boundary. "
        "They are different questions, and the disagreement is the finding.", 13, False, INK)
    footer(s, 7, T)

    # 7 ---------------------------------------------------------------- result 3
    s = blank(prs)
    heading(s, "Result 3 \u00b7 does the compression survive?", "Compressing to 12 costs almost nothing; compressing to 3 costs something measurable")
    picture(s, "F4_classification.png", ML, 1.92, 8.5, 3.55, tag="F4")
    bullets(s, 9.55, 1.96, 3.35, 4.3, [
        ("All 28 versus 12.", "Macro-F1 0.834 against 0.821 \u2014 a gap inside one fold SD (\u2248 0.07)."),
        ("The naive protocol lies.", "0.849 balanced accuracy at k = 3 becomes 0.803 when selection moves inside the fold."),
        ("Where it fails.", "Lower-middle recall 0.75, low 0.73 \u2014 the boundary cases."),
    ], size=13, gap=10)
    table(s, ML, 5.62, 7.4, [
        ["Shortlist", "Indicators", "Balanced acc.", "Macro-F1"],
        ["All 28", "28", "0.828", "0.834"],
        ["Label-blind", "12", "0.816", "0.821"],
        ["Label-aware (honest)", "3 / 12", "0.803 / 0.825", "0.809 / 0.836"],
    ], col_w=[0.34, 0.20, 0.23, 0.23], head_size=11.5, body_size=11.5, row_h=0.28)
    box(s, 8.95, 5.72, 3.9, 0.9,
        "Random Forest, repeated stratified 5-fold \u00d7 5. Selection inside the fold for the honest row.",
        10.5, False, MUTED, italic=True)
    footer(s, 8, T)

    # 8 ---------------------------------------------------------------- result 4
    s = blank(prs)
    heading(s, "Result 4 \u00b7 archetypes", "Three stable types \u2014 chosen on reproducibility, not on how well they separate")
    picture(s, "F5_archetypes.png", ML, 1.90, 8.3, 4.35, tag="F5")
    bullets(s, 9.35, 1.94, 3.55, 4.4, [
        ("k = 3, not k = 4.", "Bootstrap ARI 0.953 (k=3) vs 0.823 (k=4); k=2 ties on stability but tells us nothing new."),
        ("Advanced (n=66).", "Gaps: unemployment, forest cover, water withdrawal."),
        ("Mature middle (n=73).", "The digital gap: fixed broadband, internet users."),
        ("Emerging frontier (n=59).", "Electricity access, secondary enrolment, life expectancy."),
        ("But not canonical.", "A Gaussian mixture moves 27 of 198 countries across the middle boundary (ARI 0.58)."),
    ], size=12, gap=7)
    footer(s, 9, T)

    # 9 ---------------------------------------------------------------- result 5
    s = blank(prs)
    heading(s, "Result 5 \u00b7 dynamics", "The ranking persists at \u03c1 = 0.986 \u2014 and the exceptions are real crises")
    picture(s, "F6_dynamics.png", ML, 1.94, 8.6, 3.45, tag="F6")
    bullets(s, 9.65, 1.96, 3.25, 4.3, [
        ("Persistent.", "26% of countries move more than 10 rank places, 5% more than 20."),
        ("Gains.", "China +32, Georgia +26, Costa Rica +24, Peru +20."),
        ("Losses.", "Venezuela \u221234, Aruba \u221224, Syria \u221222, Libya \u221221."),
        ("Not told about them.", "No crisis, sanction or conflict variable enters the pipeline."),
    ], size=12.5, gap=9)
    panel(s, ML, 5.52, CW, 1.0, PALE)
    box(s, ML + 0.24, 5.65, CW - 0.6, 0.8,
        "A noisy index does not locate a crisis. The two development collapses of the period \u2014 Venezuela and Syria \u2014 "
        "are the largest negative movements on the list, and the Chinese catch-up is the largest positive one.", 13, False, INK)
    footer(s, 10, T)

    # 10 ------------------------------------------------------------ robustness
    s = blank(prs)
    heading(s, "Robustness", "Six checks, each of which could have overturned a claim")
    table(s, ML, 1.90, CW, [
        ["Check", "Result", "Reading"],
        ["Gaussian mixture instead of K-means", "ARI 0.582", "Both extremes identical; 27 of 198 change side at the advanced/middle boundary"],
        ["K-means in all 28 dimensions", "ARI 0.938", "Same partition: the 2-D display is not the partition"],
        ["Shortlist at \u03b1 = 0.25 / 0.50", "14 / 11 kept, Jaccard 0.86 / 0.92", "Stable to the penalty"],
        ["Shortlist without East Asia & Pacific", "10 kept, Jaccard 0.83", "Not an East-Asian catch-up story"],
        ["Index on PC1\u2013PC3 vs all 28", "macro-F1 0.761 vs 0.834", "The projection preserves order and destroys class separation"],
    ], col_w=[0.34, 0.24, 0.42], head_size=12.5, body_size=12.5, row_h=0.56)
    panel(s, ML, 5.62, CW, 0.92, PALE)
    box(s, ML + 0.24, 5.74, CW - 0.6, 0.7,
        "The last row is the honest caveat: a single axis is a good ordering device and a poor classification device. "
        "The report keeps both results instead of reporting the flattering one.", 13, False, INK)
    footer(s, 11, T)


    # 12 ------------------------------------------------- context and next steps
    s = blank(prs)
    heading(s, "Context", "The question is already asked \u2014 by committee, not by test")
    bullets(s, ML, 1.92, 7.15, 3.9, [
        ("Indices are the operating instrument.",
         "National human-development reports run on the HDI, poverty plans on the MPI, budget debates on "
         "the SDG index \u2014 each a weighted summary chosen by judgement."),
        ("Capacity is the binding constraint.",
         "Serajuddin et al. (2015): dozens of countries cannot measure poverty trends at all, so a "
         "28-series index is least available where targeting matters most."),
        ("The provider filters too.",
         "WDI\u2019s 1,498 series are a curated subset, admitted on coverage and relevance \u2014 the "
         "\u201cfull\u201d set is already an editorial product."),
    ], size=14.2, gap=14)
    panel(s, 8.35, 1.92, 4.2, 3.05, PALE)
    box(s, 8.58, 2.06, 3.75, 0.3, "WHERE IT GOES WRONG", 11.5, True, ACCENT)
    box(s, 8.58, 2.42, 3.75, 2.6,
        "Ratings that multiply without a selection discipline stop agreeing: ESG scores from major "
        "providers correlate at roughly half, largely because they pick different indicators "
        "(Berg et al., 2022).  Disagreement should be about measurement error \u2014 not about which "
        "twenty-eight series each analyst happened to prefer.  Our result is that failure mode run "
        "backwards: hold the menu fixed, and measure what the choice costs.", 12, False, INK)
    panel(s, ML, 5.52, CW, 1.2, PALE)
    box(s, ML + 0.24, 5.62, CW - 0.6, 0.3, "WHAT WE WOULD DO NEXT", 11.5, True, ACCENT)
    box(s, ML + 0.24, 5.94, CW - 0.6, 0.7,
        "widen the menu to the 231 SDG indicators  \u00b7  repeat the index on a second provider (UN SDG "
        "database) to test the single-source ceiling  \u00b7  put error bars on the ranks (Bayesian PCA)  \u00b7  "
        "test the 3-indicator screen prospectively against the next WDI release", 12, False, INK)
    footer(s, 12, T)

    # 11 ----------------------------------------------------------- implication
    s = blank(prs)
    heading(s, "Implication", "A division of labour, not a reduction: three to screen, twelve to place, the rest to diagnose")
    cards = [
        ("3", "SCREEN", "GDP per capita, mobile subscriptions, GDP growth",
         "\u03c1 = 0.917 \u00b7 balanced accuracy 0.80 (honest protocol) \u00b7 all three \u2264 1.7% missing, "
         "so it can be computed where monitoring is hardest.", ACCENT),
        ("12", "PLACE", "The label-blind sparse-PCA shortlist",
         "\u03c1 = 0.9935 \u00b7 macro-F1 0.821 against 0.834 for all 28 \u00b7 six members stable in >90% of resamples.", TEAL),
        ("28", "DIAGNOSE", "The full indicator set",
         "Worth 0.073 macro-F1 over the 12 for classification, and the only tier that says what a country is short of.", AMBER),
    ]
    x = ML
    cw, cg = 3.78, 0.245
    for num, label, sub, body, col in cards:
        panel(s, x, 1.94, cw, 3.40, PALE)
        rule(s, x, 1.94, cw, 0.05, col)
        box(s, x + 0.24, 2.12, 2.0, 0.80, num, 40, True, col)
        box(s, x + 0.24, 2.96, cw - 0.5, 0.3, label, 12.5, True, INK)
        box(s, x + 0.24, 3.24, cw - 0.5, 0.62, sub, 12.5, False, MUTED, italic=True)
        box(s, x + 0.24, 3.96, cw - 0.5, 1.25, body, 12.0, False, INK, spacing=1.04)
        x += cw + cg
    panel(s, ML, 5.62, CW, 0.72, PALE)
    box(s, ML + 0.24, 5.71, CW - 0.6, 0.55,
        "What a ministry should stop asking for: all 28 series before it is willing to place a country on a line.",
        14, True, INK)
    footer(s, 13, T)

    # 12 ------------------------------------------------------------ conclusion
    s = blank(prs)
    heading(s, "Conclusion", "The ordering of the world is not fragile")
    bullets(s, ML, 1.92, 6.1, 3.9, [
        ("One axis, stable.", "47.9% of the variance; \u03c1 \u2265 0.9986 to deleting any indicator, \u2265 0.9826 across windows."),
        ("Twelve suffice for fidelity.", "\u03c1 = 0.9935 and macro-F1 0.821 against 0.834 \u2014 and the choice never sees a label."),
        ("Three suffice for a screen, at 0.80 not 0.85.", "The gap between the naive and the honest protocol is itself a result."),
        ("Three archetypes, targeted differently.", "Digital infrastructure is the middle type's gap; electrification, schooling and survival are the frontier's."),
    ], size=14, gap=10)
    panel(s, 7.05, 1.90, 5.5, 2.55, PALE)
    box(s, 7.28, 2.04, 5.05, 0.3, "LIMITATIONS WE STATE", 11.5, True, ACCENT)
    bullets(s, 7.28, 2.40, 5.05, 1.95, [
        "The answer is conditional on the 28-indicator menu.",
        "Income groups are an administrative label, not ground truth.",
        "One provider: shared measurement error inflates unidimensionality.",
        "n = 198, and 3.6% of cells are imputed toward regional peers.",
    ], size=11.5, gap=5)
    panel(s, 7.05, 4.62, 5.5, 1.72, PALE)
    box(s, 7.28, 4.74, 5.05, 0.3, "STATEMENT OF INDIVIDUAL CONTRIBUTION", 11.5, True, ACCENT)
    box(s, 7.28, 5.08, 5.05, 1.1,
        "Chen Wanchuan (26108204G) 33.3% \u00b7 Lu Yuqing 33.3% \u00b7 Yuen Lok Yan 33.3%. "
        "Equal split; four streams with a primary owner each; all three authors reviewed and re-ran the full pipeline.",
        11.5, False, INK)
    box(s, ML, 5.95, 5.9, 0.8, "Thank you \u2014 questions welcome.", 20, True, INK)
    box(s, ML, 6.34, 5.9, 0.4, "Data and code released with the report; CC BY 4.0 source data.", 11.5, False, MUTED)
    footer(s, 14, T)

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"[done] {path}")
    if WARNINGS:
        print("\n[layout warnings]")
        for w in WARNINGS:
            print("  -", w)
    else:
        print("[layout] all measured text boxes fit")
