"""Shared configuration: indicator schema, paths, plotting style, table writer.

The dataset definition exists in exactly one place so that every script and the
report agree on what an "indicator" is.
"""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------- paths ----
PROJECT = Path(__file__).resolve().parents[1]
FIGS = PROJECT / "results" / "figures"
TABLES = PROJECT / "results" / "tables"
for d in (FIGS, TABLES):
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
