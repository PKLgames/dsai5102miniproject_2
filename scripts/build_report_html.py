"""Typeset the report to PDF with pandoc + WeasyPrint (no LibreOffice needed).

The primary builder is ``build_report.py`` (pandoc -> DOCX -> LibreOffice -> PDF), which
reproduces the Word styling. This script is the fallback for machines without LibreOffice:
it renders the same Markdown straight to HTML and prints it with WeasyPrint.

Math. The report uses ``\\(...\\)`` for inline symbols and ``$$...$$`` for two display
equations. WeasyPrint has no MathML support, so both are translated to Unicode text here;
nothing else about the analysis changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pypandoc
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "DSAI5102_Group16_Report.md"
OUT = SRC.with_suffix(".pdf")

INLINE = {
    "w": "w",
    "u": "u",
    "Z": "Z",
    "a": "a",
    "b": "b",
    "c": "c",
    "P": "P",
    "R": "R",
    r"\alpha": "α",
    r"\ell_1": "ℓ₁",
    r"C = 4": "C = 4",
    r"10^{-8}": "10⁻⁸",
}

DISPLAY = {
    r"\min_{u,w\,:\, \parallel u \parallel_{2} \leq 1}\ \frac{1}{2} \parallel Z - uw^{\top} \parallel_{F}^{2} + \alpha \parallel w \parallel_{1}":
        "min over <i>u</i>, <i>w</i> with ‖<i>u</i>‖₂ ≤ 1 of "
        "½ ‖<i>Z</i> − <i>u w</i>ᵀ‖²<sub>F</sub> + α ‖<i>w</i>‖₁",
    r"\text{Balanced accuracy} = \frac{1}{C}\sum_{c=1}^{C} R_c, \quad \text{Macro F1} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 P_c R_c}{P_c + R_c}":
        "Balanced accuracy = (1/<i>C</i>) Σ<sub><i>c</i></sub> <i>R<sub>c</sub></i>"
        " &nbsp;&nbsp; Macro F1 = (1/<i>C</i>) Σ<sub><i>c</i></sub> "
        "2<i>P<sub>c</sub>R<sub>c</sub></i> / (<i>P<sub>c</sub></i> + <i>R<sub>c</sub></i>)",
}

CSS = """
@page { size: Letter; margin: 0.9in 0.95in 0.8in 0.95in;
        @bottom-center { content: counter(page); font: 10pt "Nimbus Roman", serif; color: #333; } }
body { font-family: "Nimbus Roman", "Times New Roman", serif; font-size: 11pt;
       line-height: 1.45; text-align: justify; hyphens: auto; }
h1 { font-size: 15pt; text-align: center; margin: 0 0 6pt 0; line-height: 1.3; }
h2 { font-size: 12.5pt; margin: 14pt 0 6pt 0; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 4pt 0; page-break-after: avoid; }
p { margin: 0 0 6pt 0; text-indent: 0.3in; orphans: 2; widows: 2; }
h1 + p, h2 + p, h3 + p, blockquote p { text-indent: 0; }
p.caption { font-size: 9.5pt; text-align: justify; text-indent: 0; margin: 2pt 0 10pt 0; }
p.math { text-align: center; text-indent: 0; margin: 8pt 0; font-size: 11pt; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 4pt 0 2pt 0; }
th, td { padding: 2.4pt 5pt; text-align: left; vertical-align: top; }
thead th { border-top: 0.9pt solid #000; border-bottom: 0.6pt solid #000; }
tbody tr:last-child td { border-bottom: 0.9pt solid #000; }
td:nth-child(n+2), th:nth-child(n+2) { text-align: right; }
figure { margin: 10pt 0 0 0; page-break-inside: avoid; text-align: center; }
img { max-width: 100%; }
figcaption { display: none; }
.refs p { text-indent: -0.45in; margin-left: 0.45in; text-align: left; margin-bottom: 5pt; }
em, i { font-style: italic; }
strong { font-weight: bold; }
"""


def translate_math(md: str) -> str:
    """Replace the LaTeX with Unicode/HTML that WeasyPrint can typeset."""

    def plain(tex: str) -> str:
        t = tex.strip()
        t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
        for a, b in ((r"\\max", "max"), (r"\\min", "min"), (r"\\quad", " "),
                     (r"\\,", ""), (r"\\cdot", "·"), (r"\\leq", "≤"),
                     (r"\\geq", "≥"), (r"\\sum", "Σ"), (r"\\alpha", "α"),
                     (r"\\ell_1", "ℓ₁"), (r"\\top", "ᵀ"),
                     (r"\\parallel", "‖")):
            t = re.sub(a, b, t)
        t = re.sub(r"\^\{(\w+)\}", r"^\1", t)
        t = t.replace("{", "").replace("}", "").replace("\\", "")
        return t

    for tex, repl in DISPLAY.items():
        md = re.sub(r"\$\$\s*" + re.escape(tex) + r"\s*\$\$",
                    lambda m: "\n<para-math>" + repl + "</para-math>\n", md)
    md = re.sub(r"\$\$(.+?)\$\$",
                lambda m: "\n<para-math>" + plain(m.group(1)) + "</para-math>\n",
                md, flags=re.S)

    def inline(m: re.Match) -> str:
        key = m.group(1).strip()
        return INLINE.get(key) or plain(key)

    return re.sub(r"\\\((.+?)\\\)", inline, md, flags=re.S)


def main() -> None:
    md = SRC.read_text()
    title, _, body = md.partition("\n")
    body = translate_math(body)

    html_body = pypandoc.convert_text(
        body, "html5",
        format="markdown+pipe_tables+raw_html+tex_math_single_backslash-smart",
        extra_args=["--wrap=none"],
    )
    heading = f"<h1>{title.lstrip('# ').strip()}</h1>"
    html_body = heading + html_body
    html_body = re.sub(r"<p>(Figure\s+\d+\.)", r'<p class="caption">\1', html_body)
    html_body = re.sub(r"<p>(Table\s+[A-Z]?\d+\.)", r'<p class="caption">\1', html_body)
    html_body = re.sub(r"<para-math>(.*?)</para-math>", r'<p class="math">\1</p>', html_body, flags=re.S)

    refs = re.search(r'(<h2 id="references">References</h2>)(.*?)(<h2 id="appendix)', html_body, re.S)
    if refs:
        inner = re.sub(r"<p>", '<p class="ref">', refs.group(2))
        html_body = html_body.replace(
            refs.group(0), refs.group(1) + '<div class="refs">' + inner + "</div>" + refs.group(3))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title.lstrip('# ').strip()}</title><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    HTML(string=html, base_url=str(SRC.parent)).write_pdf(str(OUT))
    print(f"[built] {OUT}")


if __name__ == "__main__":
    main()
