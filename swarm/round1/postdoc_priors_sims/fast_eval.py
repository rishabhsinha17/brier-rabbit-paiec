"""Fast exact LOBO screen of level / subject priors on the 220 public eligible pairs (each alone on its benchmark).

For every held-out benchmark b: fit_subject_prior on the other benchmarks (exactly as experiments/paired.py does), take
each pair's (theta, var_theta) from the variant's _theta() (HierModel or a plug-in such as SubjCalHier), and compute the
exact expected Brier at budgets 0/1/3/7/15/31 with fastpair.predict_table (bit-exact replica of HierModel.predict for an
alone pair without item_features groups) under Binomial(n, r) own labels, r = the pair's observed rate.

Variant spec keys: any HierModel / SubjCalHier params, "class" (plug-in whose _theta is used), "mix_b" (mixture-weight
tilt: logit pi_s = logit(mix_pi) + mix_b * lam * theta_s), "q0" (solve m0 so that a theta=0 subject's B0 prediction
equals q0, other params fixed).
Pool transforms (robustness to the hidden pool): "shift" (logit shift of every pair rate), "nz" (fraction of pairs
replaced by near-zero pairs, rate 0.015, chosen with probability tilted by -nz_b * theta).
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

from fastpair import BUDGETS, expected_scores, predict_table, predict_table_exact
from paiec.data import SUBJECT_FIELDS
from paiec.hier import HierModel, fit_subject_prior

LIVE = json.loads((ROOT / "submission_src" / "v3b_noM_n_m0n20" / "params.json").read_text())
_CACHE = {}


def public_pairs():
    """DataFrame of eligible public pairs with bench, pair, r, subject dict, plus per-fold LOBO priors (cached)."""
    if "pairs" in _CACHE:
        return _CACHE["pairs"], _CACHE["priors"]
    from light import load_light
    df = load_light()
    g = df.groupby(["bench", "pair"]).agg(n=("item_id", "nunique"), k=("label", "sum"), m=("label", "size")).reset_index()
    el = g[(g.n >= 80) & (g.bench != "swe_rebench")].copy()
    first = df.drop_duplicates("pair").set_index("pair")
    el["subject"] = [{f: first.at[p, f] for f in SUBJECT_FIELDS} for p in el.pair]
    el["r"] = el.k / el.m
    priors = {b: fit_subject_prior(df[df.bench != b]) for b in sorted(el.bench.unique())}
    _CACHE["pairs"], _CACHE["priors"] = el.reset_index(drop=True), priors
    return _CACHE["pairs"], priors


def _cls(spec):
    path = spec.get("class")
    if not path:
        return HierModel
    mod, name = path.split(":")
    return getattr(importlib.import_module(mod), name)


def solve_m0(params, q0, exact=False):
    """m0 such that the theta=0 (var q0_var) budget-0 prediction equals q0 (bisection)."""
    lo, hi = -8.0, 4.0
    fn = predict_table_exact if exact else predict_table
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q = fn(dict(params, m0=mid), [0.0], [params.get("q0_var", 0.25)])[0, 0]
        lo, hi = (mid, hi) if q < q0 else (lo, mid)
    return 0.5 * (lo + hi)


def thetas(spec, el, priors):
    cls = _cls(spec)
    params = dict(LIVE, **{k: v for k, v in spec.items() if k not in ("class",)})
    th = np.zeros(len(el))
    tv = np.zeros(len(el))
    for b, prior in priors.items():
        mdl = cls(prior, params)
        for i in np.where(el.bench.values == b)[0]:
            th[i], tv[i] = mdl._theta(el.subject.iloc[i])
    return th, tv, params


def transform_pool(el, th_ref, pool, seed=0):
    r = el.r.values.astype(float).copy()
    if pool.get("scale"):  # compress/expand logit spread around the pooled mean logit (narrower or wider pool)
        rc = np.clip(r, 1e-3, 1 - 1e-3)
        lg = np.log(rc / (1 - rc))
        lg = lg.mean() + pool["scale"] * (lg - lg.mean())
        r = 1 / (1 + np.exp(-lg))
    if pool.get("shift"):
        lg = np.log(np.clip(r, 1e-3, 1 - 1e-3) / (1 - np.clip(r, 1e-3, 1 - 1e-3))) + pool["shift"]
        r = 1 / (1 + np.exp(-lg))
    nz = pool.get("nz", 0.0)
    if nz > 0:
        rng = np.random.default_rng(seed)
        # replace a fraction nz of pairs (within each benchmark) by near-zero pairs, tilted toward low theta
        logit_w = -pool.get("nz_b", 0.0) * (th_ref - th_ref.mean())
        for b in el.bench.unique():
            idx = np.where(el.bench.values == b)[0]
            w = np.exp(logit_w[idx]); w /= w.sum()
            m = int(round(nz * len(idx)))
            pick = rng.choice(idx, m, replace=False, p=w) if m else []
            r[pick] = pool.get("nz_rate", 0.015)
    return r


def evaluate(specs: dict, pool: dict | None = None, ref: str | None = None, verbose=True):
    el, priors = public_pairs()
    pool = pool or {}
    names = list(specs)
    ref = ref or names[0]
    th0, _, _ = thetas({}, el, priors)
    r = transform_pool(el, th0, pool)
    res = {}
    for name, spec in specs.items():
        th, tv, params = thetas(spec, el, priors)
        exact = bool(spec.get("exact", False))
        if "q0" in spec:
            params = dict(params, m0=solve_m0(params, spec["q0"], exact))
        pi_over = None
        if spec.get("mix_b"):
            base = params.get("mix_pi", 0.0)
            lg = math.log(base / (1 - base)) + spec["mix_b"] * params["lam"] * th
            pi_over = 1 / (1 + np.exp(-lg))
        q = (predict_table_exact if exact else predict_table)(params, th, tv, pi_over)
        res[name] = (expected_scores(q, r), params)
    out = []
    for name in names:
        sc = res[name][0]
        row = {"model": name, "m0": round(res[name][1]["m0"], 3)}
        for key in ["alc", *BUDGETS]:
            v = pd.Series(sc[key]).groupby(el.bench.values).mean().mean()
            d = pd.Series(sc[key] - res[ref][0][key]).groupby(el.bench.values)
            dm = d.mean().mean()
            se = math.sqrt(float((d.var() / d.size()).sum())) / len(d.size())
            row[f"{key}"] = v
            row[f"d_{key}"] = dm
            row[f"se_{key}"] = se
        for b in sorted(el.bench.unique()):
            mask = el.bench.values == b
            row[f"d_alc_{b[:5]}"] = float(np.mean(sc["alc"][mask] - res[ref][0]["alc"][mask]))
        out.append(row)
    T = pd.DataFrame(out)
    if verbose:
        show = ["model", "m0", "alc", "d_alc", "se_alc", "d_0", "d_1", "d_3", "d_7", "d_15", "d_31"] + \
               [c for c in T.columns if c.startswith("d_alc_")]
        S = T[show].copy()
        for c in show[2:]:
            S[c] = S[c] * (1e3 if c.startswith(("d_", "se_")) else 1)
        print(f"pool={pool}  (d_* and se in 1e-3; bench-equal macro over {len(el)} public pairs)")
        print(S.round(3).to_string(index=False))
    return T


if __name__ == "__main__":
    specs = json.loads(sys.argv[1])
    pool = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    evaluate(specs, pool)
