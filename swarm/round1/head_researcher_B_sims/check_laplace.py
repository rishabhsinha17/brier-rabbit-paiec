"""Check A: Laplace+probit (as in paiec.hier.HierModel) vs exact quadrature, single pair alone on its benchmark.
Effective prior for a new pair's level: N(m0, tau_mu^2+tau_phi^2+tau_delta^2+lam^2*tau_theta^2); items eps~N(0,sigma_i^2).
Exact: level grid x GH over eps; per-label marginal pbar(L)=E_eps[sigmoid(L+eps)]; predictive E[pbar(L)|k,n].
Excess Brier of an approximation q = E_data[(q - p_exact)^2] under the model's own prior predictive.
Also Check E: B0 prediction under a t(3) level prior: exact vs the code's variance-matched probit approximation."""
import sys, json, math
import numpy as np
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ''))
from paiec.hier import HierModel

P = dict(m0=-2.0, tau_mu=1.2, lam=1.0, tau_delta=1.0, sigma_i=0.6, sigma_g=1.0, tau_phi=0.5, tau_theta=0.5, damped_newton=True)
gx, gw = np.polynomial.hermite_e.hermegauss(60); gw = gw / gw.sum()

def pbar(L, s):
    return (1/(1+np.exp(-(L[:, None] + s*gx[None, :])))) @ gw

def exact_pred(k, n, m, v, s):
    L = np.linspace(m - 9*math.sqrt(v), m + 9*math.sqrt(v), 4001)
    lp = -0.5*(L-m)**2/v
    pb = np.clip(pbar(L, s), 1e-300, 1-1e-16)
    lp = lp + k*np.log(pb) + (n-k)*np.log1p(-pb)
    w = np.exp(lp - lp.max()); w /= w.sum()
    return float(w @ pb), w, pb

def model_pred(m0, k, n, seed=0):
    p = dict(P, m0=m0)
    mdl = HierModel({}, p)
    subj = {"normalized_name": "zz-unknown", "provider": "x"}
    lab = []
    for i in range(n):
        item = {"item_content": f"q{i}", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}
        lab.append([[subj, item], 1 if i < k else 0])
    target = {"item_content": "new", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}
    return mdl.predict([subj, target], lab)

def _main():
    out = {}
    for m0 in (-2.0, -1.5, -1.0):
        v = P['tau_mu']**2 + P['tau_phi']**2 + P['tau_delta']**2 + P['lam']**2*P['tau_theta']**2
        s = P['sigma_i']
        rows = []
        for n in (0, 1, 3, 7, 15, 31):
            # prior predictive of k under the model itself
            L = np.linspace(m0 - 9*math.sqrt(v), m0 + 9*math.sqrt(v), 4001)
            wL = np.exp(-0.5*(L-m0)**2/v); wL /= wL.sum()
            pb = pbar(L, s)
            from math import comb
            pk = np.array([float(wL @ (comb(n, k)*pb**k*(1-pb)**(n-k))) for k in range(n+1)])
            exc, maxd = 0.0, 0.0
            for k in range(n+1):
                pe, _, _ = exact_pred(k, n, m0, v, s)
                pm = model_pred(m0, k, n)
                exc += pk[k]*(pm-pe)**2
                maxd = max(maxd, abs(pm-pe)) if pk[k] > 1e-3 else maxd
            rows.append((n, exc, maxd))
        out[m0] = rows
        print(f"m0={m0}: excess Brier of Laplace+probit vs exact (prior-predictive weighted), max|dp| over k with P(k)>1e-3")
        for n, e, d in rows:
            print(f"   n={n:2d}  excess={e:.6f}  max|dp|={d:.4f}")

    # Check E: t(3) prior on mu_b, B0 prediction
    nu = 3.0
    for m0 in (-2.0, -1.5, -1.0):
        rest = P['tau_phi']**2 + P['tau_delta']**2 + P['lam']**2*P['tau_theta']**2 + P['sigma_i']**2
        tau = P['tau_mu']
        # code's approximation: variance-matched normal then probit
        var_code = tau**2*nu/(nu-2) + rest
        p_code = 1/(1+math.exp(-m0/math.sqrt(1+math.pi*var_code/8)))
        # exact: t = z*sqrt(nu/W), W~chi2_nu ; integrate over W by quadrature on a fine grid
        W = np.linspace(1e-4, 40, 40001); from scipy.stats import chi2
        fw = chi2.pdf(W, nu); fw /= fw.sum()
        var_cond = tau**2*nu/W + rest
        pc = 1/(1+np.exp(-m0/np.sqrt(1+np.pi*var_cond/8)))
        p_exact = float(fw @ pc)
        var_norm = tau**2 + rest
        p_norm = 1/(1+math.exp(-m0/math.sqrt(1+math.pi*var_norm/8)))
        print(f"B0 m0={m0}: normal-prior {p_norm:.4f} | t3 exact {p_exact:.4f} | t3 as coded (variance-matched) {p_code:.4f}")

if __name__ == "__main__":
    _main()
