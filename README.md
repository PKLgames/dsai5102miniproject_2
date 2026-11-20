# DSAI5102 Mini-Project · Group 16

**Reducing Development Indicators While Preserving Country Rankings and Income Classification**

- Report: [PDF](report/DSAI5102_Group16_Report.pdf) and [Markdown](report/DSAI5102_Group16_Report.md).
- Presentation: [PowerPoint](slides/DSAI5102_Group16_Slides.pptx) and its [PDF](slides/DSAI5102_Group16_Slides.pdf).
- Analytical inputs: the frozen WDI panel and economy metadata in `results/`.

## Reproduce the reported analysis

From the project root, use Python 3.12 and the pinned analysis dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/analyse.py
python scripts/make_figures.py
python scripts/add_figures.py
python scripts/export_tables.py
python scripts/repeat_splits.py
```

The scripts run offline from the frozen inputs. The main run uses one
five-fold split and the repeated run evaluates ten split seeds. Results are
written to `results/`. The Markdown report uses relative links to the original
figure filenames. The links above use the original submission filenames.
Place the final report and presentation under those names when merging.

To build the V4 PDF, install Pandoc, LibreOffice and the Liberation Serif
font, then run `python scripts/build_report.py` with
`report/DSAI5102_Group16_Report_V4.md` present.

`results/file_checksums.json` records SHA-256 digests for the current
analysis inputs and outputs. Paths are relative to `results/`.
