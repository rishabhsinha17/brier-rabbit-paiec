"""Pair alone on its benchmark, level L ~ N(m0, v), labels on distinct items, item effect eps ~ N(0, s^2).
Compare predictive for a new item: (1) exact quadrature; (2) 'collapsed' Laplace: drop per-item eps, use attenuated
likelihood sigmoid(c L), c = 1/sqrt(1+pi s^2/8), Laplace on L, probit-moderated prediction with var(L)+s^2;
(3) collapsed likelihood + exact 1-D quadrature on L (no Laplace). Excess Brier vs (1), prior-predictive weighted."""
import math
import numpy as np
gx, gw = np.polynomial.hermite_e.hermegauss(60); gw = gw / gw.sum()
def pbar(L, s): return (1/(1+np.exp(-(L[:, None] + s*gx[None, :])))) @ gw
def sig(x): return 1/(1+np.exp(-x))
m0, v, s = -1.5, 2.94, 0.6
c = 1/math.sqrt(1+math.pi*s*s/8)
L = np.linspace(m0-9*math.sqrt(v), m0+9*math.sqrt(v), 6001)
prior = np.exp(-0.5*(L-m0)**2/v); prior /= prior.sum()
pbL = pbar(L, s)
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
acc = {"collapsed+Laplace+probit": np.zeros(6), "collapsed+1D quadrature": np.zeros(6)}
for bi, n in enumerate((0, 1, 3, 7, 15, 31)):
    for k in range(n+1):
        pk = float(prior @ (math.comb(n, k)*pbL**k*(1-pbL)**(n-k)))
        lp = np.log(prior) + k*np.log(pbL) + (n-k)*np.log1p(-pbL)
        w = np.exp(lp-lp.max()); w /= w.sum(); exact = float(w @ pbL)
        # collapsed likelihood
        q = sig(c*L)
        lq = np.log(prior) + k*np.log(q) + (n-k)*np.log1p(-q)
        wq = np.exp(lq-lq.max()); wq /= wq.sum(); quad = float(wq @ pbL)
        # Laplace on L with collapsed likelihood (Newton)
        x = m0
        for _ in range(50):
            p_ = 1/(1+math.exp(-c*x))
            g = c*(k - n*p_) - (x-m0)/v
            h = c*c*n*p_*(1-p_) + 1/v
            x += g/h
        var = 1/h
        lap = 1/(1+math.exp(-x/math.sqrt(1+math.pi*(var + s*s)/8)))
        acc["collapsed+Laplace+probit"][bi] += pk*(lap-exact)**2
        acc["collapsed+1D quadrature"][bi] += pk*(quad-exact)**2
        if n == 7 and k in (0, 7):
            print(f"n=7 k={k}: exact {exact:.4f} | collapsed+Laplace {lap:.4f} | collapsed+quad {quad:.4f}")
for kname, val in acc.items():
    print(f"{kname:26s} ALC excess {1e3*float(W@val):.3f}e-3 | " + " ".join(f"{1e3*x:.3f}" for x in val))
