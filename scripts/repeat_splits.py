"""Repeat the head-to-head comparison across several outer split seeds.

Three arms, identical inside each seed and fold:
  full      - all 28 indicators
  fixed9    - the frozen sparse support selected once on the full sample
  reselect  - sparse support re-estimated inside the training fold
All preprocessing is refitted on the training economies of the fold.
"""
import os, json, time
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA, SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
warnings.filterwarnings('ignore', category=RuntimeWarning)

R = Path(__file__).resolve().parents[1]
D = R / 'results'
sys.path.insert(0, str(R / 'scripts'))
import schema as C

raw = pd.read_csv(D / 'official_panel.csv')
meta = pd.read_csv(D / 'country_metadata.csv', index_col=0)
raw = raw[raw[C.LABELS].notna().mean(axis=1) >= .6].copy()
for c in C.LABELS:
    if C.LOG[c]:
        raw[c] = np.log1p(raw[c])
means = raw.groupby('iso3')[C.LABELS].mean()
means = means[means.notna().mean(axis=1) >= .6]
meta = meta.reindex(means.index)


class Prep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lo_ = np.nanquantile(X, .01, axis=0)
        self.hi_ = np.nanquantile(X, .99, axis=0)
        self.scale_ = StandardScaler().fit(np.clip(X, self.lo_, self.hi_))
        a = self.scale_.transform(np.clip(X, self.lo_, self.hi_))
        self.imp_ = KNNImputer(n_neighbors=5, keep_empty_features=True).fit(a)
        self.final_ = StandardScaler().fit(self.imp_.transform(a))
        return self

    def transform(self, X):
        return self.final_.transform(self.imp_.transform(self.scale_.transform(np.clip(np.asarray(X), self.lo_, self.hi_))))


def pc(Z):
    p = PCA().fit(Z)
    sc = p.transform(Z)
    v = p.components_[0]
    sgn = 1 if v[0] > 0 else -1
    return p, sc[:, 0] * sgn, v * sgn


ALPHA = [.05, .1, .2, .35, .5, .75, 1, 1.5, 2, 3, 5, 8, 12]


def select(Z):
    rows, supports, full = [], [], pc(Z)[1]
    for a in ALPHA:
        w = SparsePCA(n_components=1, alpha=a, random_state=0, max_iter=400, tol=1e-6).fit(Z).components_[0]
        ind = np.flatnonzero(abs(w) > 1e-8)
        rho = abs(spearmanr(pc(Z[:, ind])[1], full).statistic) if len(ind) > 1 else abs(spearmanr(Z[:, ind[0]], full).statistic) if len(ind) else 0
        rows.append({'alpha': a, 'size': len(ind), 'rho': float(rho)})
        supports.append(ind)
    valid = [i for i, r in enumerate(rows) if r['rho'] >= .99]
    i = min(valid, key=lambda i: (rows[i]['size'], -rows[i]['alpha'])) if valid else max(range(len(rows)), key=lambda i: rows[i]['rho'])
    return supports[i]


X = means.to_numpy()
Z = Prep().fit_transform(X)
chosen = select(Z)
print('full-sample sparse support:', len(chosen), [C.LABELS[j] for j in chosen], flush=True)

y = meta['Income Group'].to_numpy()
mask = pd.notna(y) & (y != 'Not classified')
Xc, yc = X[mask], y[mask]

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
def forest():
    return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1)

out = []
for seed in SEEDS:
    t0 = time.time()
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    preds = {k: np.empty(len(yc), dtype=object) for k in ('full', 'fixed9', 'reselect')}
    sizes = []
    for tr, te in outer.split(Xc, yc):
        ix = select(Prep().fit(Xc[tr]).transform(Xc[tr]))
        sizes.append(len(ix))
        for key, cols in (('full', np.arange(28)), ('fixed9', chosen), ('reselect', ix)):
            pp = Prep().fit(Xc[tr][:, cols])
            m = forest().fit(pp.transform(Xc[tr][:, cols]), yc[tr])
            preds[key][te] = m.predict(pp.transform(Xc[te][:, cols]))
    row = {'seed': seed, 'sizes': sizes}
    for key, pr in preds.items():
        row[key + '_f1'] = float(f1_score(yc, pr, average='macro'))
        row[key + '_bal'] = float(balanced_accuracy_score(yc, pr))
    row['seconds'] = round(time.time() - t0, 1)
    out.append(row)
    print(json.dumps(row), flush=True)

d = pd.DataFrame(out)
print('\n=== macro-F1 across', len(d), 'split seeds ===')
for arm in ('full', 'fixed9', 'reselect'):
    print(f"{arm:9s} mean {d[arm+'_f1'].mean():.4f}  sd {d[arm+'_f1'].std(ddof=1):.4f}  min {d[arm+'_f1'].min():.4f}  max {d[arm+'_f1'].max():.4f}")
diff = d['fixed9_f1'] - d['reselect_f1']
print(f"\nfixed9 - reselect: mean {diff.mean():+.4f}  sd {diff.std(ddof=1):.4f}  "
      f"min {diff.min():+.4f}  max {diff.max():+.4f}  fixed9 wins {(diff>0).sum()}/{len(diff)}")
d.to_csv(D/'repeat_splits.csv', index=False)
json.dump({'per_seed': out,
           'summary': {arm: {'mean': float(d[arm+'_f1'].mean()), 'sd': float(d[arm+'_f1'].std(ddof=1)),
                             'min': float(d[arm+'_f1'].min()), 'max': float(d[arm+'_f1'].max())}
                       for arm in ('full', 'fixed9', 'reselect')},
           'paired': {'mean': float(diff.mean()), 'sd': float(diff.std(ddof=1)),
                      'min': float(diff.min()), 'max': float(diff.max()), 'wins': int((diff > 0).sum())},
           'seeds': SEEDS, 'support_sizes': [r['sizes'] for r in out]},
          open(D/'repeat_splits.json', 'w'), indent=1)
