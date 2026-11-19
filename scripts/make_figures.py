from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
R=Path(__file__).resolve().parents[1];s=json.loads((R/'results/analysis.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
z=pd.read_csv(R/'results/country_standardised.csv',index_col=0)
fig,ax=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[1,1.5]});ax[0].bar(range(1,6),100*np.array(s['evr']),color='#2d678c');ax[0].set(xlabel='Component',ylabel='Explained variance (%)');v=pd.Series(s['loadings']).sort_values();v=pd.concat([v.head(5),v.tail(5)]);ax[1].barh(v.index,v.values,color=['#cb805b' if x<0 else '#2d678c' for x in v]);ax[1].axvline(0,c='black',lw=.7);ax[1].set_xlabel('First component loading');fig.tight_layout();fig.savefig(R/'results/figures/structure.png',dpi=200);plt.close(fig)
c=pd.DataFrame(s['curve']);fig,ax=plt.subplots(figsize=(8,3.5));ax.plot(c['size'],c.rho,'o-',c='#2d678c');ax.axhline(.99,ls='--',c='gray');ax.scatter([len(s['selected'])],[s['rho']],s=100,facecolors='none',edgecolors='#cb805b',lw=2);ax.set(xlabel='Indicators retained',ylabel='Rank correlation with full index',ylim=(min(.98,c.rho.min()-.005),1.002));fig.tight_layout();fig.savefig(R/'results/figures/selection.png',dpi=200);plt.close(fig)
P=PCA(2).fit_transform(z);P[:,0]*=1 if np.corrcoef(P[:,0],z.iloc[:,0])[0,1]>0 else -1;lab=np.array([s['cluster_membership'][i] for i in z.index]);fig,ax=plt.subplots(figsize=(8,4));
for k in range(s['cluster_k']):ax.scatter(P[lab==k,0],P[lab==k,1],label=f'Group {k+1}',s=20,alpha=.8)
ax.set(xlabel='First component score',ylabel='Second component score');ax.legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(R/'results/figures/groups.png',dpi=200);plt.close(fig)
