# Data provenance

**Source.** World Bank, *World Development Indicators* (WDI), retrieved 2026-10-29 12:40 UTC.

**Endpoint.** `https://api.worldbank.org/v2/country/all/indicator/<CODE>?date=2000:2022&format=json`

**Licence.** World Bank Open Data, CC BY 4.0.

**Coverage.** 217 countries/territories; 28 indicators; years 2000:2022; 4,991 country-year rows.

**Dataset checksum (SHA-256).** `7f9ac6a982a34954b607ed939c8434cad3690ded1ae791637f9dd9f227c3f1c5`

## Indicators

| WDI code | Indicator | Domain | Non-missing |
|---|---|---|---|
| `NY.GDP.PCAP.CD` | GDP per capita | Economy | 97% |
| `NY.GNP.PCAP.PP.CD` | GNI per capita PPP | Economy | 89% |
| `NY.GDP.MKTP.KD.ZG` | GDP growth | Economy | 95% |
| `NE.TRD.GNFS.ZS` | Trade openness | Economy | 82% |
| `SL.UEM.TOTL.ZS` | Unemployment | Economy | 86% |
| `BX.KLT.DINV.WD.GD.ZS` | FDI inflow | Economy | 89% |
| `SP.DYN.LE00.IN` | Life expectancy | Human development | 100% |
| `SH.DYN.MORT` | Under-5 mortality | Human development | 90% |
| `SH.STA.MMRT` | Maternal mortality | Human development | 89% |
| `SE.SEC.ENRR` | Secondary enrolment | Human development | 67% |
| `SE.TER.ENRR` | Tertiary enrolment | Human development | 59% |
| `SH.XPD.CHEX.GD.ZS` | Health spending | Human development | 87% |
| `SH.IMM.MEAS` | Measles immunisation | Human development | 89% |
| `SP.URB.TOTL.IN.ZS` | Urbanisation | Urban & energy | 100% |
| `EG.USE.PCAP.KG.OE` | Energy use per capita | Urban & energy | 71% |
| `EG.ELC.ACCS.ZS` | Electricity access | Urban & energy | 99% |
| `EG.FEC.RNEW.ZS` | Renewable energy share | Urban & energy | 94% |
| `EG.ELC.RNEW.ZS` | Renewable electricity | Urban & energy | 89% |
| `EN.GHG.CO2.PC.CE.AR5` | CO2 emissions per capita | Environment | 94% |
| `EN.ATM.PM25.MC.M3` | PM2.5 exposure | Environment | 92% |
| `AG.LND.FRST.ZS` | Forest cover | Environment | 97% |
| `ER.H2O.FWTL.ZS` | Water withdrawal | Environment | 79% |
| `AG.LND.AGRI.ZS` | Agricultural land | Environment | 96% |
| `IT.NET.USER.ZS` | Internet users | Digital | 91% |
| `IT.CEL.SETS.P2` | Mobile subscriptions | Digital | 94% |
| `IT.NET.BBND.P2` | Fixed broadband | Digital | 80% |
| `AG.YLD.CREL.KG` | Cereal yield | Agriculture & food | 82% |
| `SN.ITK.DEFC.ZS` | Undernourishment | Agriculture & food | 73% |

## Note on originality

This is not a dataset shipped with a textbook or tutorial. The indicator set, the country set, the year window and all cleaning rules were chosen for this project; the collection is scripted in `src/00_download.py`, so the raw panel can be regenerated and verified from the checksum above.
