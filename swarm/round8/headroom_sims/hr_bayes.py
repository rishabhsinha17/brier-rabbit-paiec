"""(c) The Bayes-optimal alone-pair predictor that knows the regime distribution (round 8, headroom designer).

Information set: the pair's subject-prior theta (as our model computes it on a hidden subject, theta_eff = theta + d)
and its own k of n labels; the prediction minimising expected Brier is E[r | theta_eff, k, n].

Prior = the round-4 design POSTERIOR MIXTURE over regime cells c = (pool i, offset d'):
  p(x | theta_eff) = sum_c w_c [ (1 - pi_i) REG_i(x | theta' = theta_eff - d') + pi_i N(x; logit .06, .5^2) ],
  x = logit of the pair rate. REG_i is the regime's regular component: the 220 public pair rates transformed by
  (shift, scale) as fastregime.transform does, made conditional on theta by a residual shift with slope beta_i:
  atoms a_j + beta_i * theta', a_j = x_j^i - beta_i * theta_j (leave-one-pair-out: the target's own atom is removed;
  benchmark-equal atom weights; Gaussian kernel sd KSD on each atom).
  beta mode 'model' : beta_i = 1.0 (our model's lam: the predictor uses theta exactly as HierModel does; isolates the
                      level block)
  beta mode 'pooled': beta_i = scale_i * 1.250 (the bench-equal pooled public slope, i.e. the calculus truth's own
                      conditional slope: adds the theta-slope headroom, which is NOT a level-block parameter)
Posterior predictive by exact summation on an x grid. Also the ORACLE-CELL Bayes (knows the true cell; per pool) for
the value-of-regime-information split.

Output: cache/bayes_<mode>.npz with Q (nD, P, 63) for every true d, and QO (nPool, P, 63) oracle-cell tables.
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round8/headroom_sims/hr_bayes.py [model|pooled]
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import hr_common as hc

XG = np.arange(-12.0, 9.0 + 1e-9, 0.03)          # logit-rate grid
RG = 1.0 / (1.0 + np.exp(-XG))
KSD = 0.2                                        # kernel sd on each atom (logit)
LOW_MU, LOW_SD = math.log(0.06 / 0.94), 0.5
BETA_POOLED = 1.250
DD = np.round(np.arange(-3.0, 3.0001, 0.25), 3)  # d_true - d' offsets (index 12 = 0)


def lik_matrix():
    """B[g, kn] = r^k (1-r)^(n-k) (binomial coefficient cancels), computed in logs."""
    lr1 = np.log(np.clip(RG, 1e-300, 1.0))
    lr0 = np.log(np.clip(1.0 - RG, 1e-300, 1.0))
    L = hc.K_ARR[None, :] * lr1[:, None] + (hc.N_ARR - hc.K_ARR)[None, :] * lr0[:, None]
    return L


def norm_pdf_grid(mu, sd):
    z = (XG - mu) / sd
    v = np.exp(-0.5 * z * z)
    return v / v.sum()


def regular_atoms(pool):
    """Transformed regular logits of all 220 public pairs (no near-zero replacement), as fastregime.transform."""
    s = hc.setup()
    r = np.clip(s["el"].r.values.astype(float), 1e-3, 1 - 1e-3)
    lg = np.log(r / (1 - r))
    if pool.get("scale", 1.0) != 1.0:
        lg = lg.mean() + pool["scale"] * (lg - lg.mean())
    return lg + pool.get("shift", 0.0)


def reg_density(pool, beta, dshift):
    """Regular-component prior on XG for every target pair p at theta' = theta_p + dshift: (P, G), rows sum to 1,
    leave-one-pair-out, bench-equal atom weights, Gaussian kernel KSD."""
    s = hc.setup()
    th, w = s["th"], s["wts"]
    x = regular_atoms(pool)
    a = x - beta * th                                    # intercepts
    loc = a[None, :] + beta * (th[:, None] + dshift)     # (P targets, P atoms)
    wt = np.tile(w[None, :], (len(th), 1))
    np.fill_diagonal(wt, 0.0)                            # leave-one-pair-out
    wt /= wt.sum(1, keepdims=True)
    # density on the grid: sum_j wt_j phi((XG - loc_j)/KSD); do it by binning onto the grid then smoothing (exact
    # enough at grid step 0.03 vs KSD 0.2)
    G = len(XG)
    idx = np.clip(np.rint((loc - XG[0]) / (XG[1] - XG[0])).astype(int), 0, G - 1)
    H = np.zeros((len(th), G))
    rows = np.repeat(np.arange(len(th)), len(th))
    np.add.at(H, (rows, idx.ravel()), wt.ravel())
    k = np.arange(-int(4 * KSD / 0.03), int(4 * KSD / 0.03) + 1) * 0.03
    ker = np.exp(-0.5 * (k / KSD) ** 2)
    ker /= ker.sum()
    H = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, H)
    return H / H.sum(1, keepdims=True)


def predictive(prior, LL):
    """prior (..., G) -> q (..., 63) posterior predictive E[r | k, n]."""
    top = LL.max(0, keepdims=True)
    Lk = np.exp(LL - top)                                # (G, 63) scaled likelihoods (per-column scale cancels)
    num = prior @ (Lk * RG[:, None])
    den = prior @ Lk
    return num / np.maximum(den, 1e-300)


def build(mode="model", mass=0.9995):
    t0 = time.time()
    w = hc.cell_weights()                                # (nPool, nD)
    s = hc.setup()
    pools = s["pools"]
    nP = len(s["th"])
    LL = lik_matrix()
    low = norm_pdf_grid(LOW_MU, LOW_SD)
    # cells kept
    flat = np.sort(w.ravel())[::-1]
    thr = flat[min(np.searchsorted(np.cumsum(flat), mass), len(flat) - 1)]
    keep = w >= thr
    wk = np.where(keep, w, 0.0)
    wk /= wk.sum()
    print(f"cells kept {int(keep.sum())} of {keep.size} (mass {w[keep].sum():.4f})", flush=True)
    # group pools by (shift, scale)
    groups = {}
    for i, pl in enumerate(pools):
        groups.setdefault((pl["shift"], pl["scale"]), []).append(i)
    nD = len(hc.DS)
    prior = np.zeros((nD, nP, len(XG)))
    QO = np.full((len(pools), nP, len(hc.KN)), np.nan)
    low_mass = np.zeros(nD)
    for (sh, sc), idxs in groups.items():
        wg = wk[idxs]                                     # (members, nD)
        if wg.sum() <= 0:
            continue
        beta = 1.0 if mode == "model" else sc * BETA_POOLED
        pool0 = {"shift": sh, "scale": sc}
        # which offsets dd are needed: t - d' for kept cells
        need = set()
        for jd in range(nD):
            if wg[:, jd].sum() > 0:
                for t in range(nD):
                    need.add(t - jd)
        H = {}
        for o in sorted(need):
            H[o] = reg_density(pool0, beta, 0.25 * o)
        C = {}                                            # (t, offset) -> regular-component weight
        for m, i in enumerate(idxs):
            pi = pools[i]["pi_low"]
            for jd in range(nD):
                if wg[m, jd] <= 0:
                    continue
                for t in range(nD):
                    C[(t, t - jd)] = C.get((t, t - jd), 0.0) + wg[m, jd] * (1 - pi)
                low_mass += wg[m, jd] * pi
        for (t, o), c in C.items():
            prior[t] += c * H[o]
        # oracle-cell Bayes for every kept pool of this group (knows the cell: theta' = theta_p)
        H0 = H[0] if 0 in H else reg_density(pool0, beta, 0.0)
        for m, i in enumerate(idxs):
            if wg[m].sum() > 0:
                pi = pools[i]["pi_low"]
                QO[i] = predictive((1 - pi) * H0 + pi * low[None, :], LL)
        print(f"  group shift {sh:+.2f} scale {sc:.1f}: {len(need)} offsets, {time.time() - t0:.1f}s", flush=True)
    prior += low_mass[:, None, None] * low[None, None, :]
    Q = predictive(prior, LL)                            # (nD, P, 63)
    out = hc.CACHE / f"bayes_{mode}.npz"
    np.savez_compressed(out, Q=Q, QO=QO, keep=keep, w=wk)
    print(f"saved {out} ({time.time() - t0:.1f}s)")
    return Q, QO


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "model")
