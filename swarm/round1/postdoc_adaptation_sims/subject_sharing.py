"""Is there pair-level signal shared across pairs with the same base model (canonical name) or the same harness?
Public data only. Residual of a pair = logit(rate) - benchmark mean - lam*theta, theta from fit_subject_prior on
benchmarks OTHER than the ones being compared (leave-two-out), lam = 1."""
import sys, itertools
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from paiec.data import load_responses, SUBJECT_FIELDS
from paiec.hier import fit_subject_prior
from paiec.subjects import name_key, predict_attr

df = load_responses()
B = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
pairs = df.groupby("pair").agg(bench=("bench", "first"), k=("label", "sum"), n=("label", "size"), nit=("item_id", "nunique"))
first = df.drop_duplicates("pair").set_index("pair")
pairs = pairs[pairs.nit >= 80]
pairs["logit"] = np.log((pairs.k + 0.5) / (pairs.n - pairs.k + 0.5))
pairs["key"] = [name_key({f: first.at[p, f] for f in SUBJECT_FIELDS}) for p in pairs.index]
pairs["harness"] = [first.at[p, "harness"] for p in pairs.index]

def theta_of(prior, subj):
    key = name_key(subj)
    if key and key in prior["names"]:
        return prior["names"][key][0]
    return predict_attr(prior["attr"]["coef"], subj)

# (a) cross-benchmark, same canonical name
rows = []
for b1, b2 in itertools.combinations(B, 2):
    prior = fit_subject_prior(df[~df.bench.isin([b1, b2])])
    sub = pairs[pairs.bench.isin([b1, b2])].copy()
    sub["theta"] = [theta_of(prior, {f: first.at[p, f] for f in SUBJECT_FIELDS}) for p in sub.index]
    sub["res"] = sub.logit - sub.groupby("bench").logit.transform("mean") - sub.theta
    sub["res"] -= sub.groupby("bench").res.transform("mean")
    g = sub[sub.key != ""].groupby(["key", "bench"]).res.mean().unstack()
    g = g.dropna()
    if len(g) >= 4:
        r = np.corrcoef(g[b1], g[b2])[0, 1]
        rows.append((b1, b2, len(g), r, float(sub.res.std())))
print("(a) cross-benchmark residual correlation for the same canonical name (theta from the other benchmarks):")
for b1, b2, n, r, sd in rows:
    print(f"  {b1:>18} x {b2:<18} names {n:3d}  corr {r:+.3f}  (residual sd {sd:.2f})")
ns = np.array([x[2] for x in rows]); rs = np.array([x[3] for x in rows])
z = np.arctanh(np.clip(rs, -0.99, 0.99)); w = ns - 3
print(f"  pooled (Fisher z, weights n-3): corr {np.tanh(np.sum(w * z) / np.sum(w)):+.3f}, total names {ns.sum()}, se(z) {1/np.sqrt(np.sum(w)):.3f}")

# (b) within-benchmark: same name (different subject), same harness (different name)
prior_all = {b: fit_subject_prior(df[df.bench != b]) for b in B}
print("\n(b) within-benchmark residual correlation between two pairs (theta from other benchmarks):")
for b in B:
    sub = pairs[pairs.bench == b].copy()
    sub["theta"] = [theta_of(prior_all[b], {f: first.at[p, f] for f in SUBJECT_FIELDS}) for p in sub.index]
    sub["res"] = sub.logit - sub.logit.mean() - sub.theta
    sub["res"] -= sub.res.mean()
    idx = list(sub.index)
    same_name, same_h, other = [], [], []
    for i, j in itertools.combinations(idx, 2):
        a, c = sub.loc[i], sub.loc[j]
        prod = (a.res * c.res)
        if a.key and a.key == c.key:
            same_name.append(prod)
        elif a.harness and a.harness == c.harness:
            same_h.append(prod)
        else:
            other.append(prod)
    v = sub.res.var()
    f = lambda xs: f"{np.mean(xs) / v:+.2f} (n={len(xs)})" if xs else "n/a"
    print(f"  {b:>18}: pairs {len(sub)}, resid sd {np.sqrt(v):.2f}; corr same name {f(same_name)}; same harness/diff name {f(same_h)}; other {f(other)}")
