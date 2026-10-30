"""Residuals of every observed arm (selection-set live contrast vs REF) against the posterior-mean calculus prediction,
x1e-3, with the observation SD (1.2 x joint bootstrap). Arm-level only."""
import numpy as np
import hr_common as hc

P = hc.posterior()
W = P.W.sum(2)
mu = np.einsum("ij,ijo->o", W, P.MU)
sd = np.sqrt(np.diag(P.S))
arms = []
for a, s in P.labels:
    if a not in arms:
        arms.append(a)
print("%-26s %s" % ("arm", "  ".join(f"{s:>22s}" for s in ("b0", "b1", "b3", "b7"))))
print("%-26s %s" % ("", "  ".join(f"{'obs  calc  resid(sd)':>22s}" for s in range(4))))
for a in arms:
    cells = []
    for s in ("b0", "b1", "b3", "b7"):
        try:
            o = P.labels.index((a, s))
            cells.append(f"{1e3*P.y[o]:+6.1f} {1e3*mu[o]:+5.1f} {1e3*(P.y[o]-mu[o]):+5.1f}({1e3*sd[o]:4.1f})")
        except ValueError:
            cells.append(" " * 22)
    print("%-26s %s" % (a, "  ".join(cells)))
