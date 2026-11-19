import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,warnings,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA,SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,silhouette_score,adjusted_rand_score
from sklearn.cluster import KMeans
import schema as C
warnings.filterwarnings('ignore',category=RuntimeWarning)
R=Path(__file__).resolve().parents[1];D=R/'results';D.mkdir(exist_ok=True)
raw=pd.read_csv(D/'official_panel.csv');meta=pd.read_csv(D/'country_metadata.csv',index_col=0)
raw=raw[raw[C.LABELS].notna().mean(axis=1)>=.6].copy()
for c in C.LABELS:
 if C.LOG[c]:raw[c]=np.log1p(raw[c])
means=raw.groupby('iso3')[C.LABELS].mean();eligible=means.notna().mean(axis=1)>=.6
means=means[eligible];meta=meta.reindex(means.index)
class Prep(BaseEstimator,TransformerMixin):
 def fit(self,X,y=None):
  X=np.asarray(X);self.lo_=np.nanquantile(X,.01,axis=0);self.hi_=np.nanquantile(X,.99,axis=0)
  self.scale_=StandardScaler().fit(np.clip(X,self.lo_,self.hi_));a=self.scale_.transform(np.clip(X,self.lo_,self.hi_))
  self.imp_=KNNImputer(n_neighbors=5,keep_empty_features=True).fit(a)
  self.final_=StandardScaler().fit(self.imp_.transform(a));return self
 def transform(self,X):return self.final_.transform(self.imp_.transform(self.scale_.transform(np.clip(np.asarray(X),self.lo_,self.hi_))))
def pc(Z):
 p=PCA().fit(Z);sc=p.transform(Z);v=p.components_[0];sgn=1 if v[0]>0 else -1
 return p,sc[:,0]*sgn,v*sgn
ALPHA=[.05,.1,.2,.35,.5,.75,1,1.5,2,3,5,8,12]
def select(Z,details=False):
 rows=[];supports=[];full=pc(Z)[1]
 for a in ALPHA:
  w=SparsePCA(n_components=1,alpha=a,random_state=0,max_iter=400,tol=1e-6).fit(Z).components_[0];ind=np.flatnonzero(abs(w)>1e-8)
  rho=abs(spearmanr(pc(Z[:,ind])[1],full).statistic) if len(ind)>1 else abs(spearmanr(Z[:,ind[0]],full).statistic) if len(ind) else 0
  rows.append({'alpha':a,'size':len(ind),'rho':float(rho)});supports.append(ind)
 # Predeclared fidelity target, descriptive rather than a minimum over all subsets.
 valid=[i for i,r in enumerate(rows) if r['rho']>=.99]
 i=min(valid,key=lambda i:(rows[i]['size'],-rows[i]['alpha'])) if valid else max(range(len(rows)),key=lambda i:rows[i]['rho'])
 return (supports[i],rows,i) if details else supports[i]
X=means.to_numpy();Z=Prep().fit_transform(X);p,score,v=pc(Z);chosen,curve,ci=select(Z,True)
print('N',len(X),'variance',p.explained_variance_ratio_[:3],'chosen',len(chosen),flush=True)
res={'n':len(X),'rows':len(raw),'missing':int(means.isna().sum().sum()),'classes':meta['Income Group'].value_counts().to_dict(),'evr':p.explained_variance_ratio_[:5].tolist(),'selected':[C.LABELS[j] for j in chosen],'alpha':curve[ci]['alpha'],'curve':curve,'rho':curve[ci]['rho'],'loadings':dict(zip(C.LABELS,v.tolist()))}
means.to_csv(D/'country_means_before_imputation.csv');pd.DataFrame(Z,index=means.index,columns=C.LABELS).to_csv(D/'country_standardised.csv')
rank=pd.DataFrame({'country':meta['Short Name'],'score':score,'rank':pd.Series(score,index=means.index).rank(ascending=False).astype(int)});rank.to_csv(D/'rankings.csv')
rng=np.random.default_rng(19);boot=[];rankrho=[];freq=np.zeros(28)
for b in range(200):
 ids=rng.integers(len(X),size=len(X));zb=Z[ids];pb,sb,vb=pc(zb);boot.append(pb.explained_variance_ratio_[0]);rankrho.append(spearmanr(score,Z@vb).statistic)
 wb=SparsePCA(n_components=1,alpha=res['alpha'],random_state=0,max_iter=400,tol=1e-6).fit(zb).components_[0];freq+=abs(wb)>1e-8
res['bootstrap_ci']=np.quantile(boot,[.025,.975]).tolist();res['bootstrap_rho_range']=[float(min(rankrho)),float(max(rankrho))];res['selection_frequency']=dict(zip(C.LABELS,(freq/200).tolist()))
res['leave_one_rho_min']=float(min(abs(spearmanr(score,pc(np.delete(Z,j,axis=1))[1]).statistic) for j in range(28)))
print('bootstrap done',flush=True)
y=meta['Income Group'].to_numpy();classified=pd.notna(y)&(y!='Not classified');Xc=X[classified];yc=y[classified]
outer=StratifiedKFold(5,shuffle=True,random_state=7);preds={k:np.empty(len(yc),dtype=object) for k in ['full_forest','sparse_forest','full_logistic','sparse_logistic','three_forest','without_income','fixed_nine_forest','fixed_nine_logistic']};folds=[];selections=[]
def forest():return RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=1)
nonincome=[j for j in range(28) if j not in [0,1]]
for fold,(tr,te) in enumerate(outer.split(Xc,yc)):
 prep=Prep().fit(Xc[tr]);zt=prep.transform(Xc[tr]);ze=prep.transform(Xc[te]);ix=select(zt)
 # The full-sample nine-indicator candidate is evaluated unchanged in every fold:
 # same outer split, same within-fold preprocessing, same forest size.
 pp=Prep().fit(Xc[tr][:,chosen]);model=forest().fit(pp.transform(Xc[tr][:,chosen]),yc[tr]);preds['fixed_nine_forest'][te]=model.predict(pp.transform(Xc[te][:,chosen]))
 pp=Prep().fit(Xc[tr][:,chosen]);lg=LogisticRegression(max_iter=3000).fit(pp.transform(Xc[tr][:,chosen]),yc[tr]);preds['fixed_nine_logistic'][te]=lg.predict(pp.transform(Xc[te][:,chosen]))
 row={'fold':fold,'selected_count':len(ix)}
 for key,cols,model in [('full_forest',np.arange(28),forest()),('sparse_forest',ix,forest()),('full_logistic',np.arange(28),LogisticRegression(max_iter=3000)),('sparse_logistic',ix,LogisticRegression(max_iter=3000))]:
  pp=Prep().fit(Xc[tr][:,cols]);model.fit(pp.transform(Xc[tr][:,cols]),yc[tr]);pr=model.predict(pp.transform(Xc[te][:,cols]));preds[key][te]=pr;row[key]=float(f1_score(yc[te],pr,average='macro'))
 # Income exclusion sensitivity refits all preprocessing without those columns.
 pp=Prep().fit(Xc[tr][:,nonincome]);model=forest().fit(pp.transform(Xc[tr][:,nonincome]),yc[tr]);preds['without_income'][te]=model.predict(pp.transform(Xc[te][:,nonincome]))
 # Every inner fold relearns preprocessing using only that inner training partition.
 chosen3=[];inner=StratifiedKFold(3,shuffle=True,random_state=11)
 for step in range(3):
  cand=[]
  for j in range(28):
   if j in chosen3:continue
   cols=chosen3+[j]
   clf=make_pipeline(Prep(),RandomForestClassifier(n_estimators=60,random_state=0,n_jobs=1))
   val=cross_val_score(clf,Xc[tr][:,cols],yc[tr],cv=inner,scoring='balanced_accuracy').mean();cand.append((val,j))
  chosen3.append(max(cand,key=lambda a:(a[0],-a[1]))[1])
 pp=Prep().fit(Xc[tr][:,chosen3]);model=forest().fit(pp.transform(Xc[tr][:,chosen3]),yc[tr]);preds['three_forest'][te]=model.predict(pp.transform(Xc[te][:,chosen3]))
 selections.append({'fold':fold,'sparse':[C.LABELS[j] for j in ix],'three':[C.LABELS[j] for j in chosen3]});folds.append(row);print('fold complete',fold,flush=True)
res['classification']={k:{'macro_f1':float(f1_score(yc,pr,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(yc,pr)),'accuracy':float(accuracy_score(yc,pr))} for k,pr in preds.items()};res['folds']=folds;res['fold_selections']=selections;res['classified_n']=len(yc)
pd.DataFrame({'iso3':means.index[classified],'actual':yc,**preds}).to_csv(D/'held_out_predictions.csv',index=False)
print('classification',res['classification'],flush=True)
P=p.transform(Z)[:,:2];cl=[]
for k in range(2,7):
 labels=KMeans(k,n_init=20,random_state=0).fit_predict(P);st=[]
 for b in range(200):
  ids=rng.integers(len(X),size=len(X));km=KMeans(k,n_init=10,random_state=b).fit(P[ids]);st.append(adjusted_rand_score(labels,km.predict(P)))
 cl.append({'k':k,'silhouette':float(silhouette_score(P,labels)),'stability':float(np.mean(st))})
res['clusters']=cl;best=max(cl,key=lambda a:a['silhouette'])['k'];res['cluster_k']=best
labels=KMeans(best,n_init=20,random_state=0).fit_predict(P);res['cluster_sizes']=pd.Series(labels).value_counts().sort_index().tolist()
pd.DataFrame(Z,columns=C.LABELS).groupby(labels).mean().to_csv(D/'cluster_profiles.csv');res['cluster_membership']=dict(zip(means.index,labels.tolist()))
# Window comparisons refit preprocessing and principal directions separately.
windows={}
for name,years in [('early',range(2005,2010)),('late',range(2015,2020))]:
 m=raw[raw.year.isin(years)].groupby('iso3')[C.LABELS].mean();m=m[m.notna().mean(axis=1)>=.6];zz=Prep().fit_transform(m);ss=pc(zz)[1];windows[name]=pd.Series(ss,index=m.index).rank(ascending=False)
a,b=windows['early'].align(windows['late'],join='inner');res['window_n']=len(a);res['window_rho']=float(spearmanr(a,b).statistic)
# Ranks are recomputed over the common economies to avoid changing comparison populations.
res['window_rho']=float(spearmanr(a.rank(),b.rank()).statistic);movers=(a.rank()-b.rank()).sort_values(ascending=False);res['movers']={i:float(j) for i,j in pd.concat([movers.head(3),movers.tail(3)]).items()}
zippath=Path(sys.argv[1] if len(sys.argv)>1 else '/workspace/scratch/8e9ddc941f75/upload/WDI_CSV.zip')
# The zip itself is not versioned; record the hash when it is available.
prevpath=D/'analysis.json';prev=json.loads(prevpath.read_text()) if prevpath.exists() else {}
res['input_zip_sha256']=hashlib.sha256(zippath.read_bytes()).hexdigest() if zippath.exists() else prev.get('input_zip_sha256')
(D/'analysis.json').write_text(json.dumps(res,indent=2));print('ALL COMPLETE',flush=True)
if __name__=='__main__':pass
