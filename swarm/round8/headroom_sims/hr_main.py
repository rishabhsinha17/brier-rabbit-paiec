"""Main headroom table (round 8): every predictor's expected Brier per kept posterior cell, posterior summaries vs W20,
per-regime breakdowns, and the public-pool local proxy (d = 0, public rates: what a public LOBO replay should show).

Writes cache/main_cells.npz (per-cell E for every predictor) and prints the tables used in headroom.md.
usage: R swarm/round8/headroom_sims/hr_main.py
"""
from __future__ import annotations

import json
import time

import numpy as np

import hr_common as hc
import hr_fast as hf

t0 = time.time()
CE = hf.CellEval()
s = hc.setup()
thS, tvS = hc.thetas_S()
CES = hf.CellEval(th=thS, tv=tvS)

CANDS = {
    "REF": {"m0": -2.0, "V": 2.69},
    "W20": {"m0": -2.0, "V": 5.25},
    "NL* normal Laplace (m0 -1.5, V 4.5)": {"m0": -1.5, "V": 4.5},
    "NE* normal exact (m0 -1.25, V 2.69)": {"m0": -1.25, "V": 2.69, "exact": True},
    "MX* mixture hier (m0 -0.5, V 1.5, pi .3, -2.9/.55)": {"m0": -0.5, "V": 1.5, "mix_pi": 0.3, "mix_m": -2.9,
                                                           "mix_s": 0.55},
    "ME* mixture exact (m0 -0.5, V 1.0, pi .3, -2.9/.55)": {"m0": -0.5, "V": 1.0, "mix_pi": 0.3, "mix_m": -2.9,
                                                            "mix_s": 0.55, "exact": True},
    "MXw mixture on W20 (m0 -2.0, V 5.25, pi .3)": {"m0": -2.0, "V": 5.25, "mix_pi": 0.3, "mix_m": -2.9,
                                                    "mix_s": 0.55},
    "MXd mixture d-hedge (m0 -1.0, V 1.5, pi .3)": {"m0": -1.0, "V": 1.5, "mix_pi": 0.3, "mix_m": -2.9,
                                                    "mix_s": 0.55},
}
E = {k: CE.eval_spec(v) for k, v in CANDS.items()}
ES = {k + " +S": CES.eval_spec(v) for k, v in CANDS.items() if k in ("W20", "REF") or k.startswith("MX*")}
print(f"candidates done {time.time() - t0:.0f}s", flush=True)

# Bayes predictors (tables per true d; bracket-free)
bayes = {}
for mode in ("model", "pooled"):
    z = np.load(hc.CACHE / f"bayes_{mode}.npz")
    Q, QO = z["Q"], z["QO"]
    bayes[f"Bayes[{mode}] posterior-mixture prior"] = CE.eval_tables(lambda d, br, Q=Q: Q[list(hc.DS).index(d)])
    # oracle-cell: per pool table, d-free; evaluate pool by pool
    Eo = np.zeros((len(CE.cells), 6))
    for i in np.unique(CE.cells[:, 0]):
        ci = np.flatnonzero(CE.cells[:, 0] == i)
        q = QO[i]
        A1 = CE.wts[:, None] * CE.PMF[i]
        A2 = A1 * CE.R[i][:, None]
        g = (q * q * A1).sum(0) - 2.0 * (q * A2).sum(0)
        for j, n in enumerate(hc.BUD):
            Eo[ci, j] = g[hf.fp_idx(n)].sum() + CE.const[i]
    bayes[f"Oracle-cell[{mode}] (knows the true cell)"] = Eo
E.update(bayes)
E.update(ES)
print(f"bayes done {time.time() - t0:.0f}s", flush=True)

w = CE.wc
a = {k: CE.alc(v) for k, v in E.items()}
ref = a["W20"]
refS = a["W20 +S"]


def summ(k):
    base = refS if k.endswith("+S") and k != "W20 +S" else ref
    d = a[k] - base
    m = w @ d
    sd = np.sqrt(w @ (d - m) ** 2)
    return m, sd, (w * (d < -1e-3)).sum(), (w * (d < 0)).sum()


print("\n== Posterior-mean expected Brier (x1e-3; alone-pair calculus, bracket mean; kept cells %d)" % len(w))
print("%-55s %7s | %6s %6s %6s %6s %6s %6s" % ("predictor", "ALC", "B0", "B1", "B3", "B7", "B15", "B31"))
for k, v in E.items():
    m = w @ v
    print("%-55s %7.2f | %s" % (k, 1e3 * (m @ hc.WB), " ".join("%6.2f" % (1e3 * x) for x in m)))
print("\n== Contrast vs W20 (x1e-3; '+S' rows vs W20+S); posterior mean, SD over cells, P(< -1e-3), P(< 0)")
hb = summ("Bayes[model] posterior-mixture prior")[0]
for k in E:
    m, sd, p1, p0 = summ(k)
    base = E["W20 +S"] if (k.endswith("+S") and k != "W20 +S") else E["W20"]
    mb = w @ (E[k] - base)
    cap = m / hb if hb < 0 else float("nan")
    print("%-55s ALC %+6.2f (%4.2f) P<-1e-3 %.2f P<0 %.2f captured %5.0f%% | %s" % (
        k, 1e3 * m, 1e3 * sd, p1, p0, 100 * cap, " ".join("%+6.2f" % (1e3 * x) for x in mb)))

# per-regime breakdown (pi_low, shift, scale, d) of selected contrasts vs W20
pools = s["pools"]
keys_sel = ["REF", "NL* normal Laplace (m0 -1.5, V 4.5)", "NE* normal exact (m0 -1.25, V 2.69)",
            "MX* mixture hier (m0 -0.5, V 1.5, pi .3, -2.9/.55)", "MXd mixture d-hedge (m0 -1.0, V 1.5, pi .3)",
            "Bayes[model] posterior-mixture prior", "Oracle-cell[model] (knows the true cell)"]
short = ["REF", "NL*", "NE*", "MX*", "MXd", "Bayes", "Oracle"]
for dim, vals in (("pi_low", (0.0, 0.1, 0.2, 0.3, 0.4)), ("shift", (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)),
                  ("scale", (0.8, 1.0, 1.2, 1.4))):
    print(f"\n== by {dim}: posterior mass, contrast vs W20 (x1e-3, conditional posterior mean)")
    print("  %-8s %6s  " % (dim, "mass") + " ".join("%7s" % x for x in short))
    for v in vals:
        msk = np.array([pools[i][dim] == v for i in CE.cells[:, 0]])
        if w[msk].sum() < 1e-3:
            continue
        ww = w * msk / w[msk].sum()
        print("  %-8s %6.3f  " % (v, w[msk].sum()) + " ".join("%+7.2f" % (1e3 * (ww @ (a[k] - ref))) for k in keys_sel))
print("\n== by subject offset d")
print("  %-8s %6s  " % ("d", "mass") + " ".join("%7s" % x for x in short))
for jd in sorted(set(CE.cells[:, 1])):
    msk = CE.cells[:, 1] == jd
    if w[msk].sum() < 1e-3:
        continue
    ww = w * msk / w[msk].sum()
    print("  %-8s %6.3f  " % (hc.DS[jd], w[msk].sum()) + " ".join("%+7.2f" % (1e3 * (ww @ (a[k] - ref))) for k in keys_sel))

# local proxy: public pool (shift 0, scale 1, pi 0) at d = 0, our model's thetas
ip = [i for i, pl in enumerate(pools) if pl["shift"] == 0 and pl["scale"] == 1.0 and pl["pi_low"] == 0][0]
jd0 = list(hc.DS).index(0.0)


class _One(hf.CellEval):
    def __init__(self, th, tv):
        super().__init__(th=th, tv=tv)
        self.cells = np.array([[ip, jd0]])
        self.wc = np.array([1.0])
        self.ds = [jd0]


L1 = _One(s["th"], s["tv"])
print("\n== Local proxy (public pool, d = 0; what public LOBO should show), x1e-3 vs W20 / vs REF")
lp20 = L1.alc(L1.eval_spec(CANDS["W20"]))[0]
lpref = L1.alc(L1.eval_spec(CANDS["REF"]))[0]
for k, v in CANDS.items():
    e = L1.eval_spec(v)
    print("  %-55s vs W20 %+6.2f  vs REF %+6.2f | B0 %+6.2f B1 %+6.2f B3 %+6.2f" % (
        k, 1e3 * (L1.alc(e)[0] - lp20), 1e3 * (L1.alc(e)[0] - lpref),
        *[1e3 * (e[0, j] - L1.eval_spec(CANDS["W20"])[0, j]) for j in (0, 1, 2)]))
np.savez_compressed(hc.CACHE / "main_cells.npz", cells=CE.cells, w=w, names=np.array(list(E)),
                    E=np.stack([E[k] for k in E]), specs=json.dumps(CANDS))
print(f"\ntotal {time.time() - t0:.0f}s")
