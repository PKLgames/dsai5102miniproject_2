"""Additional report figures computed from the frozen official inputs and held-out predictions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'));import schema as C
s=json.loads((R/'results/analysis.json').read_text());plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','font.family':'DejaVu Sans'})
fullnames={'GDP per capita':'Gross domestic product per capita','GNI per capita PPP':'Gross national income per capita at purchasing power parity','GDP growth':'Gross domestic product growth','FDI inflow':'Foreign direct investment inflow','CO2 emissions per capita':'Carbon dioxide emissions per capita','PM2.5 exposure':'Fine particulate matter exposure'}
raw=pd.read_csv(R/'results/official_panel.csv');keep=raw[C.LABELS].notna().mean(axis=1)>=.6
before=raw[C.LABELS].isna().mean()*100;after=raw.loc[keep,C.LABELS].isna().mean()*100;order=before.sort_values().tail(14).index
fig,ax=plt.subplots(figsize=(10,6));y=np.arange(len(order))
ax.barh(y+.18,before[order],height=.34,color='#aebbc5',label='Before coverage filter (3,906 economy-years)')
ax.barh(y-.18,after[order],height=.34,color='#286387',label='After coverage filter (3,517 economy-years)')
import textwrap
plotnames={**fullnames,'GNI per capita PPP':'National income per capita at purchasing power parity'}
ax.set_yticks(y,['\n'.join(textwrap.wrap(plotnames.get(x,x),31)) for x in order],fontsize=11);ax.set_xlabel('Missing observations (%)');ax.legend(frameon=False,loc='lower right',fontsize=8);ax.set_xlim(0,45);fig.tight_layout();fig.savefig(R/'results/figures/missingness.png',dpi=220);plt.close(fig)
pd.DataFrame({'before_pct':before,'after_pct':after}).to_csv(R/'results/figure_missingness.csv')
# A precise diagram of objectives, with the prediction branch kept distinct.
fig,ax=plt.subplots(figsize=(10,5.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text,color='#edf3f6'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.012',facecolor=color,edgecolor='#526875',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526875','lw':1.1})
box(.23,.86,.54,.11,'Official indicators and economy metadata\n2005–2022 annual observations')
box(.23,.66,.54,.12,'Coverage rules and transformed economy means\nOne profile per economy')
arrow((.5,.86),(.5,.78))
branches=[(.015,'Ranking question','Prepare all profiles\nFit principal components\nSelect sparse indicators','Rank fidelity and\nconditional stability'),(.352,'Classification question','Split economies into folds\nLearn preparation and selection\nwithin training data','Held-out income-group\npredictions'),(.689,'Profiles and change question','Fit two-component display\nCompare K-means partitions\nRefit separate time windows','Group profiles and\nrelative rank changes')]
for x,heading,method,outcome in branches:
 cx=x+.148
 box(x,.43,.296,.12,heading,color='#dce9ee');arrow((.5,.66),(cx,.55))
 box(x,.205,.296,.17,method);arrow((cx,.43),(cx,.375))
 box(x,.02,.296,.13,outcome,color='#f4eee7');arrow((cx,.205),(cx,.15))
fig.tight_layout();fig.savefig(R/'results/figures/workflow.png',dpi=220);plt.close(fig)
# Row-normalised confusion matrices; counts shown to keep the sample sizes explicit.
d=pd.read_csv(R/'results/held_out_predictions.csv');labels=['High income','Upper middle income','Lower middle income','Low income'];ticks=['High','Upper\nmiddle','Lower\nmiddle','Low']
fig,axes=plt.subplots(1,2,figsize=(10,4.2));cms={}
for ax,key,title in zip(axes,['full_forest','sparse_forest'],['All 28 indicators','Training-fold sparse selection']):
 cm=confusion_matrix(d.actual,d[key],labels=labels);pct=cm/cm.sum(axis=1,keepdims=True)*100;cms[key]=cm.tolist();im=ax.imshow(pct,vmin=0,vmax=100,cmap='Blues')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{cm[i,j]}\n({pct[i,j]:.0f}%)',ha='center',va='center',color='white' if pct[i,j]>55 else '#172c39',fontsize=12)
 ax.set_xticks(range(4),ticks);ax.set_yticks(range(4),ticks);ax.set(xlabel='Predicted income group',ylabel='Recorded income group',title=title)
fig.tight_layout();fig.savefig(R/'results/figures/confusion.png',dpi=220);plt.close(fig)
(R/'results/figure_confusion.json').write_text(json.dumps({'labels':labels,'counts':cms,'n':len(d),'normalisation':'within each recorded class'},indent=2))
# Recreate both window rankings from the documented preparation, without rerunning models.
ns={'__file__':str(R/'scripts/analyse.py')};exec((R/'scripts/analyse.py').read_text().split('X=means.to_numpy();')[0],ns);panel,Prep=ns['raw'],ns['Prep'];win={}
for tag,years in [('early',range(2005,2010)),('late',range(2015,2020))]:
 m=panel[panel.year.isin(years)].groupby('iso3')[C.LABELS].mean();m=m[m.notna().mean(axis=1)>=.6];z=Prep().fit_transform(m);p=PCA().fit(z);scores=p.transform(z)[:,0]*(1 if p.components_[0,0]>0 else -1);win[tag]=pd.Series(scores,index=m.index)
a,b=win['early'].align(win['late'],join='inner');ranks=pd.DataFrame({'early':a.rank(ascending=False),'late':b.rank(ascending=False)});ranks['gain']=ranks.early-ranks.late;ranks.to_csv(R/'results/figure_window_ranks.csv')
selected=list(s['movers']);meta=pd.read_csv(R/'results/country_metadata.csv',index_col=0);fig,ax=plt.subplots(figsize=(9,4.2));
for j,code in enumerate(selected):
 e,l=ranks.loc[code,['early','late']];assert int(e-l)==int(s['movers'][code]);ax.plot([e,l],[j,j],color='#aebbc5',lw=3,zorder=1);ax.scatter(e,j,c='#286387',s=60,zorder=2);ax.scatter(l,j,c='#b56b44',s=60,zorder=3);ax.text(max(e,l)+4,j,f'{int(e-l):+d}',va='center',fontsize=9)
ax.scatter([],[],c='#286387',label='2005–2009');ax.scatter([],[],c='#b56b44',label='2015–2019');ax.set_yticks(range(len(selected)),['Venezuela' if i=='VEN' else meta.loc[i,'Short Name'] for i in selected]);ax.invert_yaxis();ax.set(xlabel='Rank among the 196 common economies (1 = highest)',xlim=(0,205));ax.legend(frameon=False,ncol=2,loc='lower left');ax.set_ylim(len(selected)-.2,-.6);fig.tight_layout();fig.savefig(R/'results/figures/rank_changes.png',dpi=220);plt.close(fig)
print('Four figures generated; held-out counts and rank changes verified.')
