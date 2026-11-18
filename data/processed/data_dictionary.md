# Data dictionary

Window 2005-2022; 3,517 country-year rows; 28 indicators; k-NN imputation (k=5).

| Indicator | Domain | Unit | Transform | Missing before |
|---|---|---|---|---|
| Cereal yield | Agriculture & food | kg/ha | log1p + winsor(1,99) + z | 8.7% |
| Undernourishment | Agriculture & food | % of population | winsor(1,99) + z | 15.1% |
| Fixed broadband | Digital | per 100 people | log1p + winsor(1,99) + z | 6.5% |
| Internet users | Digital | % of population | winsor(1,99) + z | 3.7% |
| Mobile subscriptions | Digital | per 100 people | log1p + winsor(1,99) + z | 1.2% |
| FDI inflow | Economy | % of GDP | winsor(1,99) + z | 3.8% |
| GDP growth | Economy | %/year | winsor(1,99) + z | 1.7% |
| GDP per capita | Economy | current US$ | log1p + winsor(1,99) + z | 1.2% |
| GNI per capita PPP | Economy | intl $ | log1p + winsor(1,99) + z | 4.0% |
| Trade openness | Economy | % of GDP | winsor(1,99) + z | 12.8% |
| Unemployment | Economy | % labour force | winsor(1,99) + z | 6.9% |
| Agricultural land | Environment | % of land | winsor(1,99) + z | 0.3% |
| CO2 emissions per capita | Environment | t CO2e/capita | log1p + winsor(1,99) + z | 2.6% |
| Forest cover | Environment | % of land | winsor(1,99) + z | 0.8% |
| PM2.5 exposure | Environment | ug/m3 | log1p + winsor(1,99) + z | 1.6% |
| Water withdrawal | Environment | % of resources | log1p + winsor(1,99) + z | 10.9% |
| Health spending | Human development | % of GDP | winsor(1,99) + z | 3.8% |
| Life expectancy | Human development | years | winsor(1,99) + z | 0.0% |
| Maternal mortality | Human development | per 100,000 | log1p + winsor(1,99) + z | 1.7% |
| Measles immunisation | Human development | % age 12-23m | winsor(1,99) + z | 2.3% |
| Secondary enrolment | Human development | % gross | winsor(1,99) + z | 27.8% |
| Tertiary enrolment | Human development | % gross | winsor(1,99) + z | 34.5% |
| Under-5 mortality | Human development | per 1,000 | log1p + winsor(1,99) + z | 2.2% |
| Electricity access | Urban & energy | % of population | winsor(1,99) + z | 0.2% |
| Energy use per capita | Urban & energy | kg oil equiv. | log1p + winsor(1,99) + z | 22.6% |
| Renewable electricity | Urban & energy | % of electricity | winsor(1,99) + z | 8.2% |
| Renewable energy share | Urban & energy | % of final energy | winsor(1,99) + z | 4.2% |
| Urbanisation | Urban & energy | % of population | winsor(1,99) + z | 0.0% |