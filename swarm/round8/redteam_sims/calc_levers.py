"""Screen pair-level levers with the exact alone-pair calculus (public world d 0, and the round-4 live posterior).

Each lever maps W20's prediction table q (pairs x (k, n) states) to q'. Reported: expected Brier contrast lever - W20
per budget and ALC (x1e-3), bracket mean, posterior mean and SD across posterior cells (the SD is the spread over
plausible pools, not a sampling SE). Levers:
  zero_b{b}     k = 0 states with n >= 1: logit q' = logit q - b * log(1 + n) / log(32)   (zero-run decay)
  slip{s}       q' = (1 - s) q                                                         (slip, doctawho42 0.01)
  floor{f}      q' = max(q, f)                                                           (probability floor)
  ext{a}_n{m}   n >= m: logit q' = (1 + a * log2(n + 1) / 5) logit q                      (late extremization)
  mix...        level prior mixed with a near-zero component N(mu, sd^2) at weight pi (hier.py mix_* exactly)
  oracle_k0     per-state additive correction on k = 0 states only (in-world oracle: an upper bound)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calc_state as cs  # noqa: E402

BUD = cs.BUD
K, N = cs.K, cs.N
LOGIT = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))  # noqa: E731
SIG = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def lever_fns():
    out = {}
    for b in (0.5, 1.0):
        out[f"zero_b{b}"] = (lambda q, b=b: np.where((K == 0) & (N >= 1), SIG(LOGIT(q) - b * np.log1p(N) / math.log(32)), q))
    for s in (0.005, 0.01, 0.02):
        out[f"slip{s}"] = (lambda q, s=s: (1 - s) * q)
    for f in (0.01, 0.02):
        out[f"floor{f}"] = (lambda q, f=f: np.maximum(q, f))
    for a, m in ((0.05, 7), (0.10, 7), (0.10, 15)):
        out[f"ext{a}_n{m}"] = (lambda q, a=a, m=m: np.where(N >= m, SIG((1 + a * np.log2(N + 1) / 5) * LOGIT(q)), q))
    return out


MIX = {"mix20_m275_s05": {"mix_pi": 0.20, "mix_m": math.log(0.06 / 0.94), "mix_s": 0.5},
       "mix10_m275_s05": {"mix_pi": 0.10, "mix_m": math.log(0.06 / 0.94), "mix_s": 0.5},
       "mix20_m35_s07": {"mix_pi": 0.20, "mix_m": -3.5, "mix_s": 0.7}}


def contrast(e1, e0):
    d = {n: 1e3 * (e1[n] - e0[n]) for n in BUD}
    d["alc"] = 1e3 * (e1["alc"] - e0["alc"])
    return d


def oracle_k0(q, r, w):
    st = cs.state_table(q, r, w)
    q2 = q.copy()
    sel = (K == 0) & (N >= 1)
    q2[:, sel] = np.clip(q[:, sel] - st["bias"][None, sel], 1e-4, 1 - 1e-4)
    return q2


if __name__ == "__main__":
    el, th, tv = cs.setup()
    w = cs.fr.bench_weights(el)
    fns = lever_fns()
    worlds = {"public": [({"shift": 0.0, "scale": 1.0, "pi_low": 0.0}, 0.0, 1.0)], "live": cs.posterior_cells()}
    for wname, cells in worlds.items():
        res = {}
        for pool, d, wt in cells:
            pool = dict(pool)
            pool.setdefault("mu_low", math.log(0.06 / 0.94))
            pool.setdefault("sd_low", 0.5)
            r = cs.fr.transform(el, pool)
            for br in (0, 1):
                q0 = cs.table_cached(cs.W20, th, tv, d, br)
                e0 = cs.expected(q0, r, w)
                cand = {nm: f(q0) for nm, f in fns.items()}
                cand["oracle_k0"] = oracle_k0(q0, r, w)
                for nm, mx in MIX.items():
                    cand[nm] = cs.table_cached(cs.W20, th, tv, d, br, tag=nm, **mx)
                for nm, q1 in cand.items():
                    c = contrast(cs.expected(q1, r, w), e0)
                    res.setdefault(nm, []).append((0.5 * wt, c))
        print(f"\n=== {wname} ({len(cells)} cells): lever - W20, x1e-3 (mean; SD over cells)")
        for nm, lst in res.items():
            wts = np.array([x[0] for x in lst])
            wts = wts / wts.sum()
            line = []
            for key in ("alc", 0, 1, 3, 7, 15, 31):
                v = np.array([x[1][key] for x in lst])
                m = float(wts @ v)
                sd = float(np.sqrt(max(wts @ (v - m) ** 2, 0.0)))
                line.append(f"{key}:{m:+.3f}" + (f"({sd:.2f})" if key == "alc" else ""))
            print(f"  {nm:18s} " + " ".join(line))
        sys.stdout.flush()
