import sys, math
from pathlib import Path; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
import numpy as np
from unit_checks import *
def lg(p): return math.log(p/(1-p))
def exact_unl(n_b, k_b, y_j, m0=-2.0, sx=2.0, n_a=0, k_a=0):
    # same as exact_ref but the target item is a DIFFERENT unseen item (partner's label on j still in evidence)
    kappa = 1.0 / math.sqrt(1.0 + math.pi * sx * sx / 8.0)
    gx, gw = np.polynomial.hermite_e.hermegauss(25); gw = gw / gw.sum()
    sd_mu, sd_d = 1.2 / kappa, math.sqrt(1.25) / kappa
    mu = m0 / kappa + sd_mu * gx[:, None, None]
    LA = mu + sd_d * gx[None, :, None]; LB = mu + sd_d * gx[None, None, :]
    W = gw[:, None, None] * gw[None, :, None] * gw[None, None, :]
    ux, uw = np.polynomial.hermite_e.hermegauss(60); uw = uw / uw.sum(); u = sx * ux
    def p_item(L): return np.tensordot(1.0 / (1.0 + np.exp(-(L[..., None] + u))), uw, axes=([-1], [0]))
    pB, pA = p_item(LB), p_item(LA)
    logw = np.log(W) + k_b * np.log(pB) + (n_b - k_b) * np.log1p(-pB) + k_a * np.log(pA) + (n_a - k_a) * np.log1p(-pA)
    pBj = pB if y_j else 1 - pB
    w = np.exp(logw - logw.max()) * pBj
    return float((w * pA).sum() / w.sum())
print(" n  y | shift vs own unlabelled item (logit): base headA cav nocav map | exact ; unl: base exact")
for n in (1, 3, 7, 15, 31):
    for y in (1, 0):
        k = int(round(0.3 * (n - 1))); ka = int(round(0.3 * n))
        spec = [("A", 1000 + i, int(i < ka)) for i in range(n)] + [("B", 2000 + i, int(i < k)) for i in range(n - 1)] + [("B", 99, y)]
        ev = evidence(spec); ms = models()
        unl = ms["base"].predict([subj("A"), item(98)], ev)
        v = {nm: lg(ms[nm].predict([subj("A"), item(99)], ev)) - lg(unl) for nm in ("base", "headA_x2", "cav_x2", "cav_x2_nocav", "map_x2")}
        ex = exact_ref(n - 1, k, y, n_a=n, k_a=ka); exu = exact_unl(n - 1, k, y, n_a=n, k_a=ka)
        print(f"{n:2d}  {y} | " + " ".join(f"{v[nm]:+.3f}" for nm in v) + f" | {lg(ex)-lg(exu):+.3f} ; unl {unl:.3f} {exu:.3f}")
