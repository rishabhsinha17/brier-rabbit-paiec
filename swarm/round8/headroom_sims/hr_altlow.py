"""Robustness of the headroom / mixture result to the regime family's FIXED near-zero mode (round 8).

The round-4 grid puts the low mode at logit-normal(logit 0.06, 0.5^2), chosen at random within each benchmark (no link
to theta). This script rebuilds the design grid for an alternative low mode (rate, sd) and/or a theta-tilted choice of
the near-zero pairs (P(pick) proportional to exp(-tilt * theta) within each benchmark: near-zero pairs are the weaker
subjects; public AUC of theta for near-zero is 0.72), refits the design posterior on the same live evidence, and
reports (i) the log marginal likelihood vs the base family (how strongly the live summaries prefer each low mode) and
(ii) the candidates vs W20 under the refit posterior. Nothing in swarm/round4 is modified (the variant grid lives in
cache/alt_<tag>/; ld_calc.CACHE is redirected in this process only).

usage: R hr_altlow.py <rate> <sd> <tilt> [tag]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

import hr_common as hc
import ld_calc as lc
import fastregime as fr

t0 = time.time()
rate, sd, tilt = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else f"r{rate}_s{sd}_t{tilt}"
out_dir = hc.CACHE / f"alt_{tag}"
out_dir.mkdir(exist_ok=True)
s = hc.setup()
el, th = s["el"], s["th"]


def transform(pool, seed=0):
    r = el.r.values.astype(float).copy()
    rc = np.clip(r, 1e-3, 1 - 1e-3)
    lg = np.log(rc / (1 - rc))
    if pool.get("scale", 1.0) != 1.0:
        lg = lg.mean() + pool["scale"] * (lg - lg.mean())
    lg = lg + pool.get("shift", 0.0)
    r = 1 / (1 + np.exp(-lg))
    pi = pool.get("pi_low", 0.0)
    if pi > 0:
        rng = np.random.default_rng(seed)
        for b in np.unique(el.bench.values):
            idx = np.where(el.bench.values == b)[0]
            m = int(round(pi * len(idx)))
            if not m:
                continue
            if tilt == 0.0:          # exactly fastregime.transform's draw (reproduces the base grid)
                pick = rng.choice(idx, m, replace=False)
            else:
                p = np.exp(-tilt * (th[idx] - th[idx].mean()))
                p /= p.sum()
                pick = rng.choice(idx, m, replace=False, p=p)
            qs = (np.arange(m) + 0.5) / m
            vals = math.log(rate / (1 - rate)) + sd * norm.ppf(qs)
            r[pick] = 1 / (1 + np.exp(-vals[rng.permutation(m)]))
    return r


grid_file = out_dir / "calc_grid.npz"
pools = lc.pools()
for p_ in pools:
    p_["mu_low"], p_["sd_low"] = math.log(rate / (1 - rate)), sd
R = np.stack([transform(pl) for pl in pools])
if not grid_file.exists():
    Q, keys = lc.load_tables()
    wts = s["wts"]
    E = np.zeros((len(pools), Q.shape[0], 6), np.float32)
    desc = np.zeros((len(pools), 3))
    Q2 = Q * Q
    for ip in range(len(pools)):
        r = R[ip]
        pmf = fr.pmf_table(r)
        A1 = wts[:, None] * pmf
        G = np.einsum("tpk,pk->tk", Q2, A1) - 2.0 * np.einsum("tpk,pk->tk", Q, A1 * r[:, None])
        const = float(wts @ (r * r + r * (1 - r)))
        for j, n in enumerate(lc.BUDGETS):
            E[ip, :, j] = G[:, fr.IDX[n]].sum(1) + const
        desc[ip] = (wts @ r, wts @ (r * (1 - r)), wts @ (r < 0.05))
    np.savez_compressed(grid_file, E=E, desc=desc, pools=np.array([json.dumps(p) for p in pools]),
                        keys=np.array([json.dumps(list(k)) for k in keys]))
    print(f"grid built {time.time() - t0:.0f}s", flush=True)
# refit the design posterior on this grid (redirect ld_calc's cache in this process only)
lc.CACHE = out_dir
import hr_live  # noqa: E402
import ld_post as lp  # noqa: E402
hr_live.register_family()
P = lp.Posterior(hc.LIVE_NPZ).fit()
logZ = float(np.log(np.exp(P.LL - P.LL.max()).sum()) + P.LL.max() - math.log(P.LL.size))
m = P.marginals()
W = P.W.sum(2)
W = W / W.sum()
print(f"[{tag}] log marginal likelihood {logZ:.2f}; MAP chi2 {m['map']['chi2']:.1f} + B31 {m['map']['chi2_b31']:.1f}; "
      f"E[d] {m['E[d]']:.2f} P(H_pub) {m['P(d<=0.85)']:.2f}; pi_low {dict((k, round(v, 2)) for k, v in m['pi_low'].items())}; "
      f"shift {dict((k, round(v, 2)) for k, v in m['shift'].items() if v > 0.02)}; pool mean r {m['pool_mean_r']:.3f} "
      f"P(r<5%) {m['pool_p_lt05']:.3f}", flush=True)
# candidates under the variant pools
hc._S.update(R=R, PMF=np.stack([fr.pmf_table(r) for r in R]), pools=pools)
import hr_fast as hf  # noqa: E402
CE = hf.CellEval(w=W)
specs = {"W20": {"m0": -2.0, "V": 5.25}, "REF": {"m0": -2.0, "V": 2.69}, "NLs": {"m0": -1.5, "V": 4.5},
         "M150": {"m0": -1.5, "V": 5.25},
         "MXs": {"m0": -0.5, "V": 1.5, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55},
         "MXh": {"m0": -0.75, "V": 2.0, "mix_pi": 0.25, "mix_m": -2.9, "mix_s": 0.55},
         "MXd": {"m0": -1.0, "V": 1.5, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55}}
L = math.log(rate / (1 - rate)) / 0.936          # matched hard component for this low mode (level scale)
specs["MXm"] = {"m0": -0.5, "V": 1.5, "mix_pi": 0.3, "mix_m": round(L, 2), "mix_s": round(max(sd, 0.2), 2)}
A = {k: CE.alc(CE.eval_spec(v)) for k, v in specs.items()}
line = []
for k in specs:
    d = A[k] - A["W20"]
    line.append(f"{k} {1e3 * (CE.wc @ d):+.2f} ({(CE.wc * (d < -1e-3)).sum():.2f})")
print(f"[{tag}] vs W20 x1e-3 (P<-1e-3 over cells): " + "; ".join(line) + f"  [MXm = mixture matched to this mode: "
      f"mix_m {specs['MXm']['mix_m']}, mix_s {specs['MXm']['mix_s']}]")
with (hc.CACHE / "altlow_results.jsonl").open("a") as fh:
    fh.write(json.dumps({"tag": tag, "rate": rate, "sd": sd, "tilt": tilt, "logZ": logZ, "Ed": m["E[d]"],
                         "Hpub": m["P(d<=0.85)"], "pi_low": {str(k): v for k, v in m["pi_low"].items()},
                         "vsW20": {k: float(CE.wc @ (A[k] - A["W20"])) for k in specs},
                         "P_lt_m1e3": {k: float((CE.wc * ((A[k] - A["W20"]) < -1e-3)).sum()) for k in specs}}) + "\n")
print(f"total {time.time() - t0:.0f}s")
