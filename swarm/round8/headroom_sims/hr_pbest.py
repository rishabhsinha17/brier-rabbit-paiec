"""Joint posterior (design GP for family arms; anchor + shape term for the mixtures) of the live shortlist arms and the
mixture candidates: P(best), expected improvement over W20 and over the best live arm (x1e-3, ALC vs REF)."""
import numpy as np
import hr_common as hc
import hr_full as hfu

P = hc.posterior()
fam = {"W20": {"m0": -2.0, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5},
       "L15": {"m0": -1.5, "tau_mu": 1.2, "tau_delta": 1.0, "tau_phi": 0.5},
       "M150": {"m0": -1.5, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5},
       "m250_d06": {"m0": -2.5, "tau_mu": 2.5, "tau_delta": 0.6, "tau_phi": 0.5},
       "m225_d14": {"m0": -2.25, "tau_mu": 2.5, "tau_delta": 1.4, "tau_phi": 0.5},
       "aMXs": {"m0": -0.5, "tau_mu": 1.07, "tau_delta": 0.33, "tau_phi": 0.5},
       "aMXh": {"m0": -0.75, "tau_mu": 1.233, "tau_delta": 0.48, "tau_phi": 0.5}}
for spec in fam.values():
    P.G.tix(None, 0.0, 0, spec)
names, cells, w, means, covs = P.arm_post(fam, secs=("alc",), max_cells=600)
Ef = {"MXs": hfu.full("MXs"), "MXh": hfu.full("MXh"), "aMXs": hfu.full("aMXs", {"m0": -0.5, "V": 1.07 ** 2 + 0.25 + 0.33 ** 2}),
      "aMXh": hfu.full("aMXh2", {"m0": -0.75, "V": 1.233 ** 2 + 0.25 + 0.48 ** 2})}
shape = {k: np.array([(Ef[k][i, j] - Ef["a" + k][i, j]) @ hc.WB for (i, j, _) in cells]) for k in ("MXs", "MXh")}
rng = np.random.default_rng(81)
n = 40000
ci = rng.choice(len(w), size=n, p=w)
Z = np.zeros((n, len(names)))
for c in np.unique(ci):
    idx = np.flatnonzero(ci == c)
    L = np.linalg.cholesky(covs[c] + 1e-14 * np.eye(len(names)))
    Z[idx] = means[c][:, 0][None, :] + rng.standard_normal((len(idx), len(names))) @ L.T
live = ["W20", "L15", "M150", "m250_d06", "m225_d14"]
cols = {k: Z[:, names.index(k)] for k in live}
for kappa in (0.0, 0.5):
    X = dict(cols)
    for k in ("MXs", "MXh"):
        X[k] = Z[:, names.index("a" + k)] + shape[k][ci] + kappa * np.abs(shape[k][ci]) * rng.standard_normal(n)
    X["REF"] = np.zeros(n)
    keys = list(X)
    A = np.stack([X[k] for k in keys], 1)
    best = np.bincount(A.argmin(1), minlength=len(keys)) / n
    best_live = np.minimum.reduce([X[k] for k in live + ["REF"]])
    print(f"kappa {kappa}: arm mean vs REF (sd) | P(best among all) | EI over W20 | EI over best live arm")
    for j, k in enumerate(keys):
        print("  %-9s %+6.2f (%4.2f) | P(best) %.3f | EI/W20 %.2f | EI/best-live %.2f" % (
            k, 1e3 * X[k].mean(), 1e3 * X[k].std(), best[j], 1e3 * np.maximum(0, X["W20"] - X[k]).mean(),
            1e3 * np.maximum(0, best_live - X[k]).mean()))

# ---- behaviour-nearest anchor: M150 (W20@-1.5) moves like MXs on zero-heavy sequences (out_behavior.txt), and it has
#      live runs, so its GP posterior is tight. MX = M150 + calc(MX - M150) + eta.
EM = hfu.full("M150")
for k in ("MXs", "MXh"):
    sh = np.array([(Ef[k][i, j] - EM[i, j]) @ hc.WB for (i, j, _) in cells])
    for kappa in (0.0, 0.5, 1.0):
        x = Z[:, names.index("M150")] + sh[ci] + kappa * np.abs(sh[ci]) * rng.standard_normal(n)
        d = x - Z[:, names.index("W20")]
        allx = np.stack([Z[:, names.index(a)] for a in live] + [x, np.zeros(n)], 1)
        pb = (allx.argmin(1) == len(live)).mean()
        print("M150-anchored %s kappa %.1f: vs W20 mean %+6.2f sd %5.2f P<-1e-3 %.2f P<0 %.2f | P(best vs live+REF) %.2f "
              "EI/best-live %.2f" % (k, kappa, 1e3 * d.mean(), 1e3 * d.std(), (d < -1e-3).mean(), (d < 0).mean(), pb,
                                     1e3 * np.maximum(0, np.minimum.reduce([Z[:, names.index(a)] for a in live] + [np.zeros(n)]) - x).mean()))
