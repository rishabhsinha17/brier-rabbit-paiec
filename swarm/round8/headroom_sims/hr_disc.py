"""Discrepancy-aware posterior of the mixture candidates vs W20 (round 8).

The design's own model: a level-family arm's true live contrast vs REF = calculus + GP discrepancy over
(m0, log V, tau_mu^2/V) learned from the live family arms (ld_post.arm_post). A mixture arm is not a family arm, so:
  MX - W20 = [anchor - REF] - [W20 - REF]          (joint GP posterior per regime cell, incl. discrepancy)
           + [calc(MX) - calc(anchor)]              (shape term: the near-zero component, cell by cell)
           + eta,  eta ~ N(0, (kappa * |shape term|)^2 + floor^2)
anchor = the family arm with the mixture's normal component (same m0, tau_mu, tau_delta, tau_phi)."""
import time
import numpy as np
import hr_common as hc
import hr_full as hfu

t0 = time.time()
P = hc.posterior()
fam = {"W20": {"m0": -2.0, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5},
       "aMXs": {"m0": -0.5, "tau_mu": 1.07, "tau_delta": 0.33, "tau_phi": 0.5},
       "aMXh": {"m0": -0.75, "tau_mu": 1.2, "tau_delta": 0.557, "tau_phi": 0.5},
       "M150": {"m0": -1.5, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5},
       "NLs": {"m0": -1.5, "tau_mu": 1.80, "tau_delta": 1.0, "tau_phi": 0.5}}
for spec in fam.values():          # pre-warm off-grid tables: ld_calc.Grid.budgets() indexes self.E before
    P.G.tix(None, 0.0, 0, spec)   # tix() extends it, so the FIRST use of an off-grid arm raises IndexError
names, cells, w, means, covs = P.arm_post(fam, secs=("alc",), max_cells=600)
print(f"arm_post {len(cells)} cells, {time.time() - t0:.0f}s")
Eful = {n: hfu.full(n) for n in ("MXs", "MXh")}
Eful["aMXs"] = hfu.full("aMXs", {"m0": -0.5, "V": 1.07 ** 2 + 0.25 + 0.33 ** 2})
Eful["aMXh"] = hfu.full("aMXh", {"m0": -0.75, "V": 1.44 + 0.25 + 0.557 ** 2})
shape = {k: np.array([(Eful[k][i, j] - Eful["a" + k][i, j]) @ hc.WB for (i, j, _) in cells]) for k in ("MXs", "MXh")}
rng = np.random.default_rng(8)
n = 40000
ci = rng.choice(len(w), size=n, p=w)
nc = len(names)
Z = np.zeros((n, nc))
for c in np.unique(ci):
    idx = np.flatnonzero(ci == c)
    C = covs[c] + 1e-14 * np.eye(nc)
    L = np.linalg.cholesky(C)
    Z[idx] = means[c][:, 0][None, :] + rng.standard_normal((len(idx), nc)) @ L.T
iW = names.index("W20")
print("calc-only shape term (MX - anchor), posterior mean x1e-3: MXs %+.2f  MXh %+.2f" % (
    1e3 * (w @ shape["MXs"]), 1e3 * (w @ shape["MXh"])))
for nm in ("M150", "NLs", "aMXs", "aMXh"):
    d = Z[:, names.index(nm)] - Z[:, iW]
    print("%-5s (family arm, GP) - W20: mean %+6.2f sd %5.2f  P<-1e-3 %.2f  P<0 %.2f" % (
        nm, 1e3 * d.mean(), 1e3 * d.std(), (d < -1e-3).mean(), (d < 0).mean()))
print("mixture candidates vs W20 with the anchor's GP discrepancy + shape discrepancy eta:")
for k in ("MXs", "MXh"):
    base = Z[:, names.index("a" + k)] - Z[:, iW] + shape[k][ci]
    for kappa, floor in ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 0.5e-3), (1.0, 1.0e-3)):
        eta = rng.standard_normal(n) * np.sqrt((kappa * shape[k][ci]) ** 2 + floor ** 2)
        d = base + eta
        print("  %s kappa %.1f floor %.1f: mean %+6.2f sd %5.2f  P<-1e-3 %.2f  P<0 %.2f  EI(vs W20) %.2f" % (
            k, kappa, 1e3 * floor, 1e3 * d.mean(), 1e3 * d.std(), (d < -1e-3).mean(), (d < 0).mean(),
            1e3 * np.maximum(0, -d).mean()))
print(f"total {time.time() - t0:.0f}s")
