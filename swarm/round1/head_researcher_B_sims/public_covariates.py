"""Public data only. (1) Does the leave-one-benchmark-out subject ability predict near-zero pairs (the 'pitcher'
covariate)? (2) Item-level bimodality: share of items solved by no / all subjects per benchmark (items with >=8 subjects)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ''))
from paiec.data import load_responses, SUBJECT_FIELDS
from paiec.hier import fit_subject_prior
from paiec.subjects import name_key, predict_attr

df = load_responses()
pairs = df.groupby(['bench', 'pair']).agg(k=('label', 'sum'), n=('label', 'size'), items=('item_id', 'nunique')).reset_index()
pairs = pairs[(pairs['items'] >= 80) & (pairs.bench != 'swe_rebench')]
first = df.drop_duplicates('pair').set_index('pair')
rows = []
for b in sorted(pairs.bench.unique()):
    prior = fit_subject_prior(df[df.bench != b])
    for _, r in pairs[pairs.bench == b].iterrows():
        subj = {f: first.at[r['pair'], f] for f in SUBJECT_FIELDS}
        key = name_key(subj)
        if key and key in prior['names']:
            th, var = prior['names'][key]; src = 'name'
        else:
            th, var = predict_attr(prior['attr']['coef'], subj), prior['attr']['var']; src = 'attr'
        rows.append(dict(bench=b, rate=r['k'] / r['n'], theta=th, src=src))
d = pd.DataFrame(rows)
d['hard'] = d.rate < 0.05
def auc(score, lab):
    s1, s0 = score[lab], score[~lab]
    if len(s1) == 0 or len(s0) == 0: return float('nan')
    return float(((s1[:, None] < s0[None, :]).mean() + 0.5 * (s1[:, None] == s0[None, :]).mean()))
print('pairs', len(d), 'hard(<5%)', int(d.hard.sum()), 'by source', d.groupby('src').size().to_dict())
for b, g in d.groupby('bench'):
    print(f"  {b:18s} n={len(g):3d} hard={int(g.hard.sum()):2d}  AUC(low theta -> hard)={auc(g.theta.values, g.hard.values):.2f}  "
          f"corr(theta, logit rate)={np.corrcoef(g.theta, np.log((g.rate+0.01)/(1.01-g.rate)))[0,1]:.2f}")
print(f"  pooled AUC={auc(d.theta.values, d.hard.values):.2f}")
# rate quintiles by theta tercile
d['tt'] = pd.qcut(d.theta, 3, labels=['low', 'mid', 'high'])
print(d.groupby('tt', observed=True).agg(n=('rate', 'size'), frac_hard=('hard', 'mean'), mean_rate=('rate', 'mean')).round(3))

# item-level bimodality
it = df[df.bench != 'swe_rebench'].groupby(['bench', 'item_id']).agg(subj=('subject_id', 'nunique'), m=('label', 'mean')).reset_index()
it = it[it.subj >= 8]
print('\nitem solve-rate distribution (items answered by >=8 subjects)')
for b, g in it.groupby('bench'):
    print(f"  {b:18s} items={len(g):5d}  solved by none={np.mean(g.m == 0):.2f}  by all={np.mean(g.m == 1):.2f}  "
          f"<10%={np.mean(g.m < 0.1):.2f}  >90%={np.mean(g.m > 0.9):.2f}")
