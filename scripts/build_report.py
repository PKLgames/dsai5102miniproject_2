"""Build the submission PDF from the canonical Markdown report."""
from pathlib import Path
import shutil
import subprocess
import tempfile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

report_dir = Path(__file__).resolve().parents[1] / "report"
source = report_dir / "DSAI5102_Group16_Report.md"
title_line, _, body = source.read_text().partition("\n")
title = title_line.removeprefix("# ")
office = shutil.which("soffice") or shutil.which("libreoffice")
if office is None:
    raise SystemExit("Install LibreOffice and add soffice to PATH.")

with tempfile.TemporaryDirectory() as temp_dir:
    working = Path(temp_dir)
    content = working / "report.md"
    content.write_text(body)
    docx_path = working / source.with_suffix(".docx").name
    subprocess.run(
        ["pandoc", str(content), "--from=markdown+tex_math_single_backslash-smart",
         "--metadata", f"title={title}", "-o", str(docx_path)],
        cwd=report_dir, check=True,
    )
    doc = Document(docx_path)
    for style in doc.styles:
        style.font.name = "Liberation Serif"
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().rFonts
        for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn(f"w:{attribute}"), None)
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Body Text"].font.size = Pt(11)
    doc.styles["Title"].font.size = Pt(19)
    doc.styles["Title"].font.bold = True
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.85)
        section.top_margin = section.bottom_margin = Inches(0.8)
        footer = section.footer.paragraphs[0]
        footer.alignment = 1
        page = OxmlElement("w:fldSimple")
        page.set(qn("w:instr"), "PAGE")
        footer._p.append(page)
    section_starts = [
        paragraph for paragraph in doc.paragraphs
        if paragraph.style.name == "Heading 2"
        and (paragraph.text == "References"
             or paragraph.text.startswith("Appendix A ")
             or paragraph.text.startswith("Statement of Individual"))
    ]
    if len(section_starts) != 3:
        raise ValueError("Expected References, Appendix A and Statement headings in the report.")
    for paragraph in section_starts:
        paragraph.paragraph_format.page_break_before = True

    contribution = doc.tables[-1]
    for column, width in zip(contribution.columns, (1.25, 1.05, 3.45, 1.05)):
        column.width = Inches(width)
        for cell in column.cells:
            cell.width = Inches(width)
    doc.save(docx_path)
    subprocess.run(
        [office, f"-env:UserInstallation={(working / 'profile').as_uri()}",
         "--headless", "--convert-to", "pdf", "--outdir", str(working), str(docx_path)],
        check=True,
    )
    source.with_suffix(".pdf").write_bytes(docx_path.with_suffix(".pdf").read_bytes())
