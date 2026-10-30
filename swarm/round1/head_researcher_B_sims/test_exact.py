"""Sanity test of paiec.ext.hrB_exact.ExactAloneHier against an independent exact computation (no groups) and with groups."""
import sys, math
import numpy as np
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ''))
from paiec.hier import HierModel
from paiec.ext.hrB_exact import ExactAloneHier
P = dict(m0=-1.5, tau_mu=1.2, lam=1.0, tau_delta=1.0, sigma_i=0.6, sigma_g=1.0, tau_phi=0.5, tau_theta=0.5, damped_newton=True)
gx, gw = np.polynomial.hermite_e.hermegauss(60); gw = gw / gw.sum()
def pbar(L, s): return (1/(1+np.exp(-(L[:, None] + s*gx[None, :])))) @ gw
def exact(k, n, m, v, s):
    L = np.linspace(m - 9*math.sqrt(v), m + 9*math.sqrt(v), 4001)
    lp = -0.5*(L-m)**2/v; pb = np.clip(pbar(L, s), 1e-300, 1-1e-16)
    lp = lp + k*np.log(pb) + (n-k)*np.log1p(-pb); w = np.exp(lp-lp.max()); w /= w.sum()
    return float(w @ pb)
subj = {"normalized_name": "zz-a", "provider": "x"}
def it(i, f=""): return {"item_content": f"q{i}", "item_features": f, "interactors": "", "benchmark_id": "b1"}
v = 1.2**2 + 1.0**2 + 0.5**2 + 0.25
for n in (0, 1, 3, 7):
    for k in sorted({0, n // 2, n}):
        lab = [[[subj, it(i)], 1 if i < k else 0] for i in range(n)]
        e = exact(k, n, -1.5, v, 0.6)
        h = HierModel({}, P).predict([subj, it(999)], lab)
        x = ExactAloneHier({}, P).predict([subj, it(999)], lab)
        print(f"n={n} k={k}: independent exact {e:.4f} | ExactAloneHier {x:.4f} | HierModel {h:.4f}")
# with item groups: 7 labels, 4 in group lang=a (all 0), 3 in group lang=b (all 1)
lab = [[[subj, it(i, "lang=a")], 0] for i in range(4)] + [[[subj, it(10 + i, "lang=b")], 1] for i in range(3)]
for f in ("lang=a", "lang=b", "lang=c", ""):
    h = HierModel({}, P).predict([subj, it(999, f)], lab)
    x = ExactAloneHier({}, P).predict([subj, it(999, f)], lab)
    print(f"groups target {f or '(none)':7s}: ExactAloneHier {x:.4f} | HierModel {h:.4f}")
# mixture prior on
Pm = dict(P, mix_pi=0.25, mix_m=-4.0, mix_s=0.7, m0=-1.0)
for n, k in ((0, 0), (1, 0), (1, 1), (3, 0), (7, 0), (7, 2)):
    lab = [[[subj, it(i)], 1 if i < k else 0] for i in range(n)]
    h = HierModel({}, Pm).predict([subj, it(999)], lab)
    x = ExactAloneHier({}, Pm).predict([subj, it(999)], lab)
    print(f"mixture n={n} k={k}: ExactAloneHier {x:.4f} | HierModel(mix) {h:.4f}")
