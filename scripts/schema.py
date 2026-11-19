"""Shared configuration: indicator schema, paths, plotting style, table writer.

The dataset definition exists in exactly one place so that every script and the
report agree on what an "indicator" is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths ----
PROJECT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT / "data" / "raw"
DATA_PROC = PROJECT / "data" / "processed"
FIGS = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics"
TABLES = PROJECT / "results" / "tables"
for d in (DATA_RAW, DATA_PROC, FIGS, METRICS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- schema ----
# code, label (report-ready), domain, unit, log-transform?
INDICATORS = [
    # economy
    ("NY.GDP.PCAP.CD",      "GDP per capita",        "Economy",            "current US$",   True),
    ("NY.GNP.PCAP.PP.CD",   "GNI per capita PPP",    "Economy",            "intl $",        True),
    ("NY.GDP.MKTP.KD.ZG",   "GDP growth",            "Economy",            "%/year",        False),
    ("NE.TRD.GNFS.ZS",      "Trade openness",        "Economy",            "% of GDP",      False),
    ("SL.UEM.TOTL.ZS",      "Unemployment",          "Economy",            "% labour force", False),
    ("BX.KLT.DINV.WD.GD.ZS", "FDI inflow",           "Economy",            "% of GDP",      False),
    # human development
    ("SP.DYN.LE00.IN",      "Life expectancy",       "Human development",  "years",         False),
    ("SH.DYN.MORT",         "Under-5 mortality",     "Human development",  "per 1,000",     True),
    ("SH.STA.MMRT",         "Maternal mortality",    "Human development",  "per 100,000",   True),
    ("SE.SEC.ENRR",         "Secondary enrolment",   "Human development",  "% gross",       False),
    ("SE.TER.ENRR",         "Tertiary enrolment",    "Human development",  "% gross",       False),
    ("SH.XPD.CHEX.GD.ZS",   "Health spending",       "Human development",  "% of GDP",      False),
    ("SH.IMM.MEAS",         "Measles immunisation",  "Human development",  "% age 12-23m",  False),
    # urban & energy
    ("SP.URB.TOTL.IN.ZS",   "Urbanisation",          "Urban & energy",     "% of population", False),
    ("EG.USE.PCAP.KG.OE",   "Energy use per capita", "Urban & energy",     "kg oil equiv.", True),
    ("EG.ELC.ACCS.ZS",      "Electricity access",    "Urban & energy",     "% of population", False),
    ("EG.FEC.RNEW.ZS",      "Renewable energy share", "Urban & energy",    "% of final energy", False),
    ("EG.ELC.RNEW.ZS",      "Renewable electricity", "Urban & energy",     "% of electricity", False),
    # environment
    ("EN.GHG.CO2.PC.CE.AR5", "CO2 emissions per capita", "Environment",    "t CO2e/capita", True),
    ("EN.ATM.PM25.MC.M3",   "PM2.5 exposure",        "Environment",        "ug/m3",         True),
    ("AG.LND.FRST.ZS",      "Forest cover",          "Environment",        "% of land",     False),
    ("ER.H2O.FWTL.ZS",      "Water withdrawal",      "Environment",        "% of resources", True),
    ("AG.LND.AGRI.ZS",      "Agricultural land",     "Environment",        "% of land",     False),
    # digital
    ("IT.NET.USER.ZS",      "Internet users",        "Digital",            "% of population", False),
    ("IT.CEL.SETS.P2",      "Mobile subscriptions",  "Digital",            "per 100 people", True),
    ("IT.NET.BBND.P2",      "Fixed broadband",       "Digital",            "per 100 people", True),
    # agriculture & food
    ("AG.YLD.CREL.KG",      "Cereal yield",          "Agriculture & food", "kg/ha",         True),
    ("SN.ITK.DEFC.ZS",      "Undernourishment",      "Agriculture & food", "% of population", False),
]

CODES = [i[0] for i in INDICATORS]
LABELS = [i[1] for i in INDICATORS]
DOMAIN = {i[1]: i[2] for i in INDICATORS}
UNIT = {i[1]: i[3] for i in INDICATORS}
LOG = {i[1]: i[4] for i in INDICATORS}
CODE2LABEL = {i[0]: i[1] for i in INDICATORS}
DOMAINS = ["Economy", "Human development", "Urban & energy",
           "Environment", "Digital", "Agriculture & food"]

# analysis window and quality thresholds (identical to the preprocessing run)
YEAR_MIN, YEAR_MAX = 2005, 2022
MAX_COL_MISSING = 0.40
MIN_ROW_COVERAGE = 0.60
KNN_K = 5

# Okabe-Ito colour-blind-safe palette
PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermilion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}
INCOME_ORDER = ["High income", "Upper middle income",
                "Lower middle income", "Low income"]
INCOME_COLORS = {
    "High income": "#0072B2", "Upper middle income": "#009E73",
    "Lower middle income": "#E69F00", "Low income": "#D55E00",
}
DOMAIN_COLORS = dict(zip(DOMAINS, ["#0072B2", "#009E73", "#E69F00",
                                   "#56B4E9", "#CC79A7", "#D55E00"]))


def setup_matplotlib():
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    return plt


def save(fig, name):
    pdf, png = FIGS / f"{name}.pdf", FIGS / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    return pdf


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_raw_panel() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "wdi_panel_long.csv")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_raw_units, Z_standardised) with a (iso3, year) MultiIndex."""
    X = pd.read_csv(DATA_PROC / "dataset.csv", index_col=["iso3", "year"])
    Z = pd.read_csv(DATA_PROC / "dataset_std.csv", index_col=["iso3", "year"])
    return X, Z


def load_countries() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "countries.csv").set_index("iso3")


def load_country_meta() -> pd.DataFrame:
    m = pd.read_csv(DATA_RAW / "countries.csv").set_index("iso3")
    m["region"] = m["region"].str.strip()
    m["income"] = m["income"].str.strip()
    return m


def income_short(s):
    """'High income' -> 'High'; works on a Series or a single string."""
    if hasattr(s, "str"):
        return s.astype(str).str.replace(" income", "", regex=False)
    return str(s).replace(" income", "")


# ------------------------------------------------------------- tables -----
def write_table(df: pd.DataFrame, stem: str, caption: str, label: str,
                align: str | None = None, note: str = "",
                float_fmt: str = "%.3f") -> None:
    """Write the same table as LaTeX (results/tables/<stem>.tex) and CSV."""
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    cols = list(df.columns)
    if align is None:
        align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{%s}" % caption,
        "\\label{%s}" % label,
        "\\small",
        "\\begin{tabular}{%s}" % align,
        "\\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v}")
            elif isinstance(v, (float, np.floating)):
                cells.append(float_fmt % v)
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\vspace{2pt}\\parbox{0.92\\linewidth}{\\footnotesize %s}" % note]
    lines += ["\\end{table}"]
    (TABLES / f"{stem}.tex").write_text("\n".join(lines) + "\n")


def x_escape(df: pd.DataFrame) -> pd.DataFrame:
    """Escape underscores for LaTeX cells (kept minimal: only our own strings)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.replace("_", r"\_", regex=False)
    return out
