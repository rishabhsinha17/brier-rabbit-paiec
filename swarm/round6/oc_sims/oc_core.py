"""Generative model of the live formative data and the pre-registered estimators and rules (swarm round 6).

WORLD (one simulated hidden pool):
  - 7 benchmark strata with the stratum model's sizes N_b and second-pair probabilities d_b (round-4 design: ours +
    public connectors, 285 pairs; a 244-pair ours-only variant is available). Each stratum draws its pairs from one
    public benchmark at the regime cell's transformed rates (ld_sim.World logic, exact alone-pair calculus tables).
  - Per arm and pair: the ALONE contrast vs REF at budgets 0/1/3/7/15/31 (deterministic, exact calculus on the
    pair's own label path and evaluation mean), optional extra per-(arm, pair) item/group noise (fixed per pair),
    then (null) centred within every stratum so that every benchmark's mean contrast is exactly 0, plus a constant
    effect profile per arm. ALC = 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31.
  - Shared appearances (the stratum got two pairs in the run): + a pair-level pooling shift g_p (common to all arms)
    + per-appearance partner noise e (independent per appearance; B0 has neither: it is condition-free).
  - Holdout: a fixed 22.6% of pairs (Bernoulli per pair; the pinned hash plays this role live).
RUN: one pair per stratum plus a second with probability d_b (8.7 pairs, ~40% shared appearances).
ESTIMATORS (tools/daily_report.py / tools/swarm_evalsci.py, ported to arrays):
  - paircov FE: cells = (pair, condition, arm) means; unit = pair; common 'shared' covariate except at B0
    (swarm_evalsci.fe_within, the same function the tools call);
  - CR1 pair-cluster sandwich on raw sightings exactly as daily_report.cr1_section;
  - SE_h = max(1.2 x bootstrap, CR1, FLOOR/sqrt(K)); the stratified pair bootstrap is replaced by the CR0 sandwich
    (CR1 without its small-sample factor), validated against the real bootstrap in validate_se.py;
  - K = distinct pairs seen by both the arm and REF (any condition) in the subset.
Nothing here reads live data.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LIB = REPO / "swarm" / "round4" / "design_sims"
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(REPO / "tools"))
import ld_calc as lc  # noqa: E402
import swarm_evalsci as ev  # noqa: E402

SEC = ("alc", "b0", "b1", "b3", "b7", "b15", "b31")
SI = {s: i for i, s in enumerate(SEC)}
WB = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
STRATA_N = (11, 13, 19, 26, 32, 42, 142)
STRATA_D = (0.284, 0.409, 0.136, 0.159, 0.159, 0.227, 0.352)
STRATA_N_OURS = (11, 13, 19, 22, 31, 40, 108)
STRATA_D_OURS = (0.297, 0.432, 0.081, 0.243, 0.189, 0.189, 0.297)
HOLDOUT_SHARE = 0.226
FLOOR = np.array([0.0147, 0.055, 0.0487, 0.0132, 0.0076, 0.0043, 0.0039])   # QUEUE round-3 floors, SEC order
BOOT_INFLATE = 1.2
BOOT_PROXY = 0.97      # stratified pair bootstrap SE ~= 0.97 x CR0 (validate_se.py: 210 cells, 3 worlds, 150 reps)
# partner model (Brier units, budgets 0/1/3/7/15/31), from within-arm replicates of our logs (live_targets.json):
POOL_MEAN = np.array([0.0, -0.010, -0.010, -0.005, -0.002, -0.002])
POOL_SD = np.array([0.0, 0.010, 0.008, 0.005, 0.002, 0.002])
PARTNER_SD = np.array([0.0, 0.025, 0.012, 0.005, 0.0025, 0.0025])
REF_SPEC = {"m0": -2.0, "tau_mu": 1.2, "tau_delta": 1.0, "tau_phi": lc.TAU_PHI}


def with_alc(b):
    """(..., 6 budgets) -> (..., 7 sections) with ALC first."""
    return np.concatenate([(b @ WB)[..., None], b], axis=-1)


class World:
    """One simulated pool. arms: list of family specs (REF excluded); cell = (pool i, d j, lam k) of the posterior."""

    def __init__(self, T, cell, arms, rng, *, effects=None, center=True, scale=None, extra_sd=None,
                 strata=(STRATA_N, STRATA_D), partner=True, holdout_share=HOLDOUT_SHARE, comp=(), target_sd=None,
                 superpop=False):
        i, j, _ = cell
        d = lc.DS[j]
        r_pub = lc.fr.transform(T.el, T.pools[i])
        N = np.array(strata[0])
        self.strata_d = np.array(strata[1], float)
        self.N = N
        pub_idx, stratum, brs = [], [], []
        for b, n in enumerate(N):
            pb = T.benches[rng.integers(len(T.benches))]
            pub_idx.append(rng.choice(T.by_bench[pb], size=n, replace=True))
            stratum.append(np.full(n, b))
            brs.append(np.full(n, rng.integers(2)))
        self.pub = np.concatenate(pub_idx)
        self.stratum = np.concatenate(stratum)
        br = np.concatenate(brs)
        P = len(self.pub)
        self.P = P
        r = r_pub[self.pub]
        self.rate = r
        m = rng.integers(44, 114, size=P)
        ybar = rng.binomial(m, r) / m
        labels = rng.random((P, 31)) < r[:, None]
        cum = np.concatenate([np.zeros((P, 1), int), np.cumsum(labels, axis=1)], axis=1)
        kb = cum[:, list(lc.BUDGETS)]
        kn = kb + np.array(lc.BUDGETS)[None, :]
        self.holdout = rng.random(P) < holdout_share

        def brier_tab(tix_fn):
            out = np.zeros((P, 6))
            for bb in (0, 1):
                sel = br == bb
                if not sel.any():
                    continue
                q = T.Q[tix_fn(bb)][self.pub[sel]]
                qq = np.take_along_axis(q, kn[sel], axis=1)
                out[sel] = (qq - ybar[sel, None]) ** 2
            return out

        b_ref = brier_tab(lambda bb: T.tix(REF_SPEC, d, bb))
        A = len(arms)
        self.A = A
        Cb = np.zeros((A, P, 6))
        for a, spec in enumerate(arms):
            Cb[a] = brier_tab(lambda bb, s=spec: T.tix(s, d, bb)) - b_ref
        self.raw_budget_sd = Cb.std(1, ddof=1)                       # calculus-only per-pair SD (A, 6)
        if extra_sd is not None:                                     # fixed per-(arm, pair) item/group noise
            Cb = Cb + rng.standard_normal((A, P, 6)) * np.asarray(extra_sd)[None, None, :]
        if center:
            for b in range(len(N)):
                sel = self.stratum == b
                Cb[:, sel, :] -= Cb[:, sel, :].mean(1, keepdims=True)
        if scale is not None:
            Cb = Cb * np.asarray(scale)[None, None, :]
        if target_sd is not None:                                    # every arm's alone SD set to target (stress)
            sd = Cb.std(1, ddof=1)
            Cb = Cb * (np.asarray(target_sd)[None, :] / np.maximum(sd, 1e-12))[:, None, :]
        if superpop:                    # the pool is itself a sample: each stratum's pool mean deviates from the
            for b in range(len(N)):     # population mean (0 under the null) by one pair's contrast / sqrt(N_b)
                sel = np.flatnonzero(self.stratum == b)
                j = sel[rng.integers(len(sel))]
                Cb[:, sel, :] += Cb[:, [j], :] / math.sqrt(len(sel))
        if effects is not None:
            Cb = Cb + np.asarray(effects)[:, None, :]
        self.C = with_alc(Cb)                                         # (A, P, 7) alone contrasts
        self.truth = self.C.mean(1)                                   # pair-equal pool truth (A, 7)
        share = (1 + self.strata_d) / (1 + self.strata_d).sum()
        self.truth_bal = np.stack([self.C[:, self.stratum == b, :].mean(1) for b in range(len(N))], 1)
        self.truth_bal = np.einsum("b,abs->as", share, self.truth_bal)  # draw-weighted (benchmark-balanced) truth
        self.partner = partner
        g = POOL_MEAN[None, :] + rng.standard_normal((P, 6)) * POOL_SD[None, :]
        self.G = with_alc(g)                                          # pooling shift per pair (P, 7)
        self.by_stratum = [np.flatnonzero(self.stratum == b) for b in range(len(N))]
        # connector arms (public competitor replicas; only B0 is used, for the calibration test)
        self.comp_b0 = {}
        for name in comp:
            tb = brier_tab(lambda bb, nm=name: T.idx[(nm, bb, 0.0, 0.0)] if nm != "pub_doctawho42_hiership"
                           else T.idx[(nm, bb, 0.0, round(float(d), 3))])
            self.comp_b0[name] = tb[:, 0] - b_ref[:, 0]

    def draw_run(self, rng):
        picks, shared = [], []
        for b, idx in enumerate(self.by_stratum):
            if rng.random() < self.strata_d[b] and len(idx) > 1:
                two = rng.choice(idx, size=2, replace=False)
                picks.extend(two)
                shared.extend([True, True])
            else:
                picks.append(idx[rng.integers(len(idx))])
                shared.append(False)
        return np.array(picks), np.array(shared)


class Data:
    """Appearance log: arm index (-1 = REF, 0..A-1 arms, A.. connectors), pair, shared flag, 7 section scores."""

    def __init__(self, world: World, cap: int = 12000):
        self.w = world
        self.arm = np.empty(cap, np.int16)
        self.pair = np.empty(cap, np.int32)
        self.sh = np.empty(cap, bool)
        self.run = np.empty(cap, np.int32)
        self.Y = np.empty((cap, 7))
        self.n = 0
        self.n_runs = 0
        self.runs = np.zeros(world.A, int)
        self.runs_ref = 0

    def _grow(self, k):
        if self.n + k <= len(self.arm):
            return
        new = max(2 * len(self.arm), self.n + k)
        for nm in ("arm", "pair", "sh", "run", "Y"):
            old = getattr(self, nm)
            arr = np.empty((new,) + old.shape[1:], old.dtype)
            arr[:self.n] = old[:self.n]
            setattr(self, nm, arr)

    def upload(self, a: int, rng, y_fn=None):
        p, sh = self.w.draw_run(rng)
        k = len(p)
        self._grow(k)
        sl = slice(self.n, self.n + k)
        self.arm[sl] = a
        self.pair[sl] = p
        self.sh[sl] = sh
        self.run[sl] = self.n_runs
        if y_fn is not None:
            Y = y_fn(p)
        else:
            Y = np.zeros((k, 7)) if a < 0 else self.w.C[a, p].copy()
        if self.w.partner and sh.any():
            e = np.zeros((int(sh.sum()), 6))
            e[:, 1:] = rng.standard_normal((int(sh.sum()), 5)) * PARTNER_SD[None, 1:]
            Y[sh] = Y[sh] + self.w.G[p[sh]] + with_alc(e)
        self.Y[sl] = Y
        self.n += k
        self.n_runs += 1
        if a < 0:
            self.runs_ref += 1
        elif a < self.w.A:
            self.runs[a] += 1
        return p, sh


# ---------------------------------------------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------------------------------------------
def _gsum(codes, X, n_groups):
    """Group sums of the rows of X (n, m) by integer codes, via a sparse indicator product (fast np.add.at)."""
    M = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_groups, len(codes)))
    return np.asarray(M @ X)


def _cells(arm, pair, sh, Y, n_arms_tot):
    """Cell means per (pair, cond, arm). Returns pair codes, arm codes (0 = REF, 1.. = arm+1), shared flag, Y."""
    a = arm.astype(np.int64) + 1
    key = (pair.astype(np.int64) * 2 + sh.astype(np.int64)) * (n_arms_tot + 1) + a
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    Ym = _gsum(inv, Y, len(uk)) / cnt[:, None]
    ca = (uk % (n_arms_tot + 1)).astype(np.int64)
    rest = uk // (n_arms_tot + 1)
    csh = (rest % 2).astype(bool)
    cp = rest // 2
    return cp, ca, csh, Ym


def paircov(arm, pair, sh, Y, n_arms_tot, b0_col=1):
    """paircov FE (FE2.fit('paircov')) for every arm vs REF: (n_arms_tot+1, ncol) with row 0 = REF (0). Column
    b0_col (B0) is refitted without the shared covariate (B0 is condition-free); None skips that."""
    cp, ca, csh, Ym = _cells(arm, pair, sh, Y, n_arms_tot)
    beta = ev.fe_within(cp, ca, Ym, n_arms_tot + 1, 0, csh.astype(float)[:, None])[0]
    if b0_col is not None and b0_col < Ym.shape[1]:
        beta[:, b0_col] = ev.fe_within(cp, ca, Ym[:, [b0_col]], n_arms_tot + 1, 0)[0][:, 0]
    return beta


def cr_se(arm, pair, sh, Y, n_arms_tot, cov: bool):
    """CR1 and CR0 pair-cluster sandwich SEs of the dummy-variable OLS on raw sightings (daily_report.cr1_section)
    for all sections at once (same design for every section). Returns (cr1, cr0) arrays (n_arms_tot+1, ncol)."""
    out1 = np.full((n_arms_tot + 1, Y.shape[1]), np.nan)
    out0 = out1.copy()
    if len(arm) == 0:
        return out1, out0
    a = arm.astype(np.int64) + 1
    _, ui = np.unique(pair, return_inverse=True)
    pa = np.unique(ui * (n_arms_tot + 1) + a)
    nun = np.bincount(pa // (n_arms_tot + 1), minlength=int(ui.max()) + 1)
    keep = nun[ui] > 1
    if not keep.any():
        return out1, out0
    a, ui, Yk = a[keep], ui[keep], Y[keep]
    shk = sh[keep].astype(float)
    ui = np.unique(ui, return_inverse=True)[1]
    nu = int(ui.max()) + 1
    est = [int(x) for x in np.unique(a) if x != 0]
    if not est:
        return out1, out0
    cols = [(a == x).astype(float) for x in est] + ([shk] if cov else [])
    X = np.column_stack(cols)
    n = len(ui)
    M = sp.csr_matrix((np.ones(n), (ui, np.arange(n))), shape=(nu, n))
    cnt = np.asarray(M.sum(1)).ravel()
    Xd = X - (np.asarray(M @ X) / cnt[:, None])[ui]
    Yd = Yk - (np.asarray(M @ Yk) / cnt[:, None])[ui]
    ok = np.abs(Xd).sum(0) > 1e-12
    Xo = Xd[:, ok]
    sol, *_ = np.linalg.lstsq(Xo, Yd, rcond=None)
    R = Yd - Xo @ sol
    XtX_inv = np.linalg.pinv(Xo.T @ Xo)
    kpar = Xo.shape[1] + nu
    f1 = nu / max(nu - 1, 1) * (n - 1) / max(n - kpar, 1)
    names = [x for x, o in zip(est + (["cov"] if cov else []), ok) if o]
    p = Xo.shape[1]
    Z = (Xo[:, :, None] * R[:, None, :]).reshape(n, -1)
    Sall = np.asarray(M @ Z).reshape(nu, p, Y.shape[1])
    for s_ in range(Y.shape[1]):
        Ss = Sall[:, :, s_]
        V0 = XtX_inv @ (Ss.T @ Ss) @ XtX_inv
        d0 = np.sqrt(np.clip(np.diag(V0), 0, None))
        for ii, nm in enumerate(names):
            if nm == "cov":
                continue
            out0[nm, s_] = d0[ii]
            out1[nm, s_] = d0[ii] * math.sqrt(f1)
    return out1, out0


def k_direct(arm, pair, n_arms_tot, P):
    """Distinct pairs seen by both REF and each arm (any condition) -> (n_arms_tot+1,) with K[0] = REF's own count."""
    seen = np.zeros((n_arms_tot + 1, P), bool)
    seen[arm.astype(np.int64) + 1, pair] = True
    K = (seen & seen[0][None]).sum(1)
    return K


def estimate(D: Data, rows: np.ndarray, n_arms_tot: int | None = None):
    """Effect, SE_h, z and K for every arm on the given appearance rows (paircov; B0 without the covariate)."""
    n_arms_tot = D.w.A if n_arms_tot is None else n_arms_tot
    arm, pair, sh, Y = D.arm[rows], D.pair[rows], D.sh[rows], D.Y[rows]
    beta = paircov(arm, pair, sh, Y, n_arms_tot)
    cr1, cr0 = cr_se(arm, pair, sh, Y, n_arms_tot, cov=True)
    cr1b, cr0b = cr_se(arm, pair, sh, Y[:, [1]], n_arms_tot, cov=False)
    cr1[:, 1], cr0[:, 1] = cr1b[:, 0], cr0b[:, 0]
    K = k_direct(arm, pair, n_arms_tot, D.w.P)
    floor = FLOOR[None, :] / np.sqrt(np.maximum(K, 1))[:, None]
    # bootstrap proxy: the cell-based pair bootstrap is ~0.97 x CR0 at screening run counts (validate_se.py) and
    # ~0.78 x CR0 at final-look counts (cell_boot vs CR0, 7 arms with 30-90 runs, REF 290): interpolate in arm runs
    runs = np.concatenate([[D.runs_ref], D.runs, np.zeros(max(n_arms_tot - D.w.A, 0))])[:n_arms_tot + 1]
    fac = BOOT_PROXY - 0.19 * np.clip((runs - 30) / 70.0, 0, 1)
    seh = np.fmax(np.fmax(BOOT_INFLATE * fac[:, None] * np.nan_to_num(cr0), np.nan_to_num(cr1)), floor)
    z = beta / seh
    return {"est": beta[1:], "se": seh[1:], "z": z[1:], "K": K[1:], "cr1": cr1[1:], "cr0": cr0[1:]}


def cell_boot(D: Data, rows: np.ndarray, reps: int = 100, seed: int = 0, n_arms_tot: int | None = None):
    """Stratified pair bootstrap of the cell-based paircov FE (FE2.boot('paircov') as daily_report runs it):
    resample pairs with replacement within benchmark strata, duplicates become distinct units. Returns the bootstrap
    SD (n_arms, 7) for the non-REF arms."""
    n_arms_tot = D.w.A if n_arms_tot is None else n_arms_tot
    cp, ca, csh, Ym = _cells(D.arm[rows], D.pair[rows], D.sh[rows], D.Y[rows], n_arms_tot)
    order = np.argsort(cp, kind="stable")
    cp, ca, csh, Ym = cp[order], ca[order], csh[order], Ym[order]
    up, start, cnt = np.unique(cp, return_index=True, return_counts=True)
    strat = D.w.stratum[up]
    rng = np.random.default_rng(seed)
    out = np.full((reps, n_arms_tot + 1, Ym.shape[1]), np.nan)
    by_b = [np.flatnonzero(strat == b) for b in np.unique(strat)]
    for r in range(reps):
        pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_b])
        c = cnt[pick]
        idx = np.repeat(start[pick], c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
        unit = np.repeat(np.arange(len(pick)), c)
        beta = ev.fe_within(unit, ca[idx], Ym[idx], n_arms_tot + 1, 0, csh[idx].astype(float)[:, None])[0]
        beta[:, 1] = ev.fe_within(unit, ca[idx], Ym[idx][:, [1]], n_arms_tot + 1, 0)[0][:, 0]
        out[r] = beta
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(out, axis=0, ddof=1)
    return sd[1:]


def estimate_exact(D: Data, rows: np.ndarray, reps: int = 100, seed: int = 0):
    """estimate() with SE_h = max(1.2 x cell-based stratified pair bootstrap, CR1, floor/sqrt(K)): daily_report's
    formula with the real bootstrap (used at the final look)."""
    e = estimate(D, rows)
    bs = cell_boot(D, rows, reps=reps, seed=seed)
    floor = FLOOR[None, :] / np.sqrt(np.maximum(e["K"], 1))[:, None]
    seh = np.fmax(np.fmax(BOOT_INFLATE * np.nan_to_num(bs), np.nan_to_num(e["cr1"])), floor)
    e2 = dict(e)
    e2["se"], e2["z"], e2["boot"] = seh, e["est"] / seh, bs
    return e2


def subset_rows(D: Data, which: str):
    n = D.n
    if which == "all":
        return np.arange(n)
    h = D.w.holdout[D.pair[:n]]
    return np.flatnonzero(~h) if which == "sel" else np.flatnonzero(h)


# ---------------------------------------------------------------------------------------------------------------
# pre-registered rules
# ---------------------------------------------------------------------------------------------------------------
DROP_ALC, DROP_Z, FINALIST_Z = 0.0035, 2.0, -1.0
SCREEN_RUNS, FINALIST_RUNS = 6, 30
LOBO_MAX = 0.001


def screen_one(e, z):
    """QUEUE screen on one arm's selection-set estimates: 'drop' / 'finalist' / 'continue'."""
    if not np.isfinite(e[0]):
        return "n/a"
    if e[0] > DROP_ALC or any(np.isfinite(z[SI[s]]) and z[SI[s]] > DROP_Z for s in ("b0", "b1", "b3")):
        return "drop"
    if e[0] < 0 and z[0] <= FINALIST_Z:
        return "finalist"
    return "continue"


def balanced_and_lobo(D: Data, arms_idx):
    """Benchmark-balanced ALC (per-stratum paircov, draw-share weights, renormalised over covered strata) and the
    leave-one-stratum-out maximum, on all data, for the given arms."""
    n = D.n
    arm, pair, sh, Y = D.arm[:n], D.pair[:n], D.sh[:n], D.Y[:n, :1]
    st = D.w.stratum[pair]
    nb = len(D.w.N)
    share = (1 + D.w.strata_d) / (1 + D.w.strata_d).sum()
    per = np.full((nb, D.w.A), np.nan)
    lobo = np.full((nb, D.w.A), np.nan)
    for b in range(nb):
        m = st == b
        bb = paircov(arm[m], pair[m], sh[m], Y[m], D.w.A, b0_col=None)
        per[b] = bb[1:, 0]
        mm = ~m
        lo = paircov(arm[mm], pair[mm], sh[mm], Y[mm], D.w.A, b0_col=None)
        lobo[b] = lo[1:, 0]
    out = {}
    for a in arms_idx:
        ok = np.isfinite(per[:, a])
        bal = float((share[ok] * per[ok, a]).sum() / share[ok].sum()) if ok.any() else np.nan
        lm = float(np.nanmax(lobo[:, a])) if np.isfinite(lobo[:, a]).any() else np.nan
        out[a] = (bal, lm)
    return out


def switch_decision(D: Data, eligible, n_final: int, est_all=None, est_hold=None):
    """The 10-30 checklist on every eligible arm; returns (shipped arm or -1, per-arm item outcomes)."""
    thr = -1.64 if n_final <= 1 else -1.8
    est_all = est_all or estimate(D, subset_rows(D, "all"))
    est_hold = est_hold or estimate(D, subset_rows(D, "hold"))
    bl = balanced_and_lobo(D, eligible) if eligible else {}
    items, best, best_e = {}, -1, np.inf
    for a in eligible:
        e, z = est_all["est"][a], est_all["z"][a]
        h = est_hold["est"][a, 0]
        it = {"a": bool(np.isfinite(h) and h <= 0), "b": bool(np.isfinite(z[0]) and z[0] <= thr),
              "c": bool(np.isfinite(bl[a][0]) and bl[a][0] <= 0), "d": bool(np.isfinite(bl[a][1]) and bl[a][1] <= LOBO_MAX),
              "e": bool(np.all(np.nan_to_num(z[1:], nan=99.0) <= DROP_Z)), "tie": bool(abs(e[0]) >= 0.001)}
        items[a] = it
        if all(it.values()) and e[0] < best_e:
            best, best_e = a, e[0]
    return best, items
