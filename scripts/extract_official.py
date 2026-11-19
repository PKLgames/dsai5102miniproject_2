import zipfile,pandas as pd,sys,json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'));import schema as C
p=R;z=zipfile.ZipFile(sys.argv[1])
meta=pd.read_csv(z.open('WDICountry.csv')).set_index('Country Code')
meta=meta[meta.Region.notna()]
parts=[]
for d in pd.read_csv(z.open('WDICSV.csv'),chunksize=15000):
 d=d[d['Indicator Code'].isin(C.CODES)&d['Country Code'].isin(meta.index)]
 parts.append(d[['Country Code','Indicator Code']+[str(x) for x in range(2005,2023)]])
d=pd.concat(parts);print('Codes',len(d['Indicator Code'].unique()),'economies',d['Country Code'].nunique())
long=d.melt(id_vars=['Country Code','Indicator Code'],var_name='year',value_name='value')
x=long.pivot(index=['Country Code','year'],columns='Indicator Code',values='value').reindex(columns=C.CODES)
x.columns=C.LABELS;x.index.names=['iso3','year'];x=x.sort_index()
x.to_csv(p/'results/official_panel.csv');meta.to_csv(p/'results/country_metadata.csv')
print('raw shape',x.shape,'rowfilter',sum(x.notna().mean(axis=1)>=.6));print(x.isna().mean().round(3).to_dict())
