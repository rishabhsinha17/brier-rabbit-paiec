"""Posterior-matched NORMAL level prior with the design's own discrepancy model: every round-4 grid arm (84) plus the
calculus-optimal off-grid normal arms, via ld_post.arm_post (GP), joint draws -> contrast vs W20."""
import numpy as np
import hr_common as hc
import ld_calc as lc

P = hc.posterior()
fam = lc.family_arms()
extra = {"NLs m-1.50 V4.5 (tm1.80 td1.0)": {"m0": -1.5, "tau_mu": 1.80, "tau_delta": 1.0, "tau_phi": 0.5},
         "NLs' m-1.50 V4.5 (tm2.0 td0.5)": {"m0": -1.5, "tau_mu": 2.0, "tau_delta": 0.5, "tau_phi": 0.5},
         "m-1.75 V5.25 (tm2.0 td1.0)": {"m0": -1.75, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5}}
fam.update(extra)
for spec in extra.values():
    P.G.tix(None, 0.0, 0, spec)          # pre-warm off-grid tables (ld_calc first-use IndexError)
names, cells, w, means, covs = P.arm_post(fam, secs=("alc",), max_cells=500)
S, D = P.summarize(names, w, means, covs, secs=("alc",), ndraw=20000, seed=8)
iW = names.index(lc.arm_name(-2.0, 2.0, 1.0))
rel = D[:, :, 0] - D[:, iW:iW + 1, 0]
order = np.argsort(rel.mean(0))
print("normal level arms, ALC vs W20 x1e-3 incl. GP discrepancy: mean (sd) P(<-1e-3) P(<0) | P(best incl REF)")
for c in list(order[:10]) + [names.index(lc.arm_name(-1.5, 1.2, 1.0)), names.index(lc.arm_name(-1.5, 2.0, 1.0)),
                             names.index(lc.arm_name(-2.5, 2.5, 0.6)), names.index(lc.arm_name(-2.25, 2.5, 1.4))]:
    r = rel[:, c]
    print("  %-34s %+6.2f (%4.2f)  %.2f  %.2f | %.3f" % (names[c], 1e3 * r.mean(), 1e3 * r.std(), (r < -1e-3).mean(),
                                                        (r < 0).mean(), S[names[c]]["p_best"]))
print("REF P(best) %.3f" % S["_REF"]["p_best"])
