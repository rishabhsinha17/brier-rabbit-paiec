"""Decompose the Laplace under-reaction: (a) code-consistent exact (ability variance 0.25 fixed, not learnable;
level prior var 2.69) vs (b) fully exact (ability variance learnable, level var 2.94) vs HierModel (ability_param
False / True). Pair alone on its benchmark, m0=-1.5, sigma_i=0.6."""
import sys, math
import numpy as np
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ''))
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + '/swarm/round1/head_researcher_B_sims'))
from check_laplace import P, pbar
from paiec.hier import HierModel

gx, gw = np.polynomial.hermite_e.hermegauss(60); gw = gw / gw.sum()

def exact(k, n, m, v, s_lab, s_pred):
    L = np.linspace(m - 9*math.sqrt(v), m + 9*math.sqrt(v), 4001)
    lp = -0.5*(L-m)**2/v
    pb = np.clip(pbar(L, s_lab), 1e-300, 1-1e-16)
    lp = lp + k*np.log(pb) + (n-k)*np.log1p(-pb)
    w = np.exp(lp - lp.max()); w /= w.sum()
    return float(w @ pbar(L, s_pred))

def hm(k, n, m0, ability_param):
    mdl = HierModel({}, dict(P, m0=m0, ability_param=ability_param))
    subj = {"normalized_name": "zz-unknown", "provider": "x"}
    lab = [[[subj, {"item_content": f"q{i}", "item_features": "", "interactors": "", "benchmark_id": "b"}], 1 if i < k else 0] for i in range(n)]
    return mdl.predict([subj, {"item_content": "new", "item_features": "", "interactors": "", "benchmark_id": "b"}], lab)

m0 = -1.5
v_lvl = P['tau_mu']**2 + P['tau_phi']**2 + P['tau_delta']**2
v_ab = P['lam']**2*P['tau_theta']**2
s_i = P['sigma_i']
L = np.linspace(m0 - 9*math.sqrt(v_lvl+v_ab), m0 + 9*math.sqrt(v_lvl+v_ab), 4001)
wL = np.exp(-0.5*(L-m0)**2/(v_lvl+v_ab)); wL /= wL.sum(); pbL = pbar(L, s_i)
tot = {"code-consistent exact": np.zeros(6), "HierModel ability_param=False": np.zeros(6), "HierModel ability_param=True": np.zeros(6)}
for bi, n in enumerate((0, 1, 3, 7, 15, 31)):
    for k in range(n+1):
        pk = float(wL @ (math.comb(n, k)*pbL**k*(1-pbL)**(n-k)))
        full = exact(k, n, m0, v_lvl + v_ab, s_i, s_i)
        cc = exact(k, n, m0, v_lvl, s_i, math.sqrt(s_i**2 + v_ab))
        a0 = hm(k, n, m0, False); a1 = hm(k, n, m0, True)
        tot["code-consistent exact"][bi] += pk*(cc-full)**2
        tot["HierModel ability_param=False"][bi] += pk*(a0-full)**2
        tot["HierModel ability_param=True"][bi] += pk*(a1-full)**2
        if n in (3, 7) and k in (0, n):
            print(f"n={n} k={k}: fully-exact {full:.4f} | code-consistent exact {cc:.4f} | HM(False) {a0:.4f} | HM(True) {a1:.4f}")
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
for name, v in tot.items():
    print(f"{name:32s} excess vs fully-exact Bayes: ALC {1e3*float(W@v):.3f}e-3 | " + " ".join(f"{1e3*x:.3f}" for x in v))
