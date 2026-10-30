"""Evaluation-science analysis of our own formative feedback (swarm round 1, postdoc 3).

usage:  nice -n 19 taskpolicy -b ../.venv/bin/python tools/swarm_evalsci.py [pool|noise|fe|agree|replay|robust|power|all]

pool    recurrence of anonymous pair IDs across runs, capture-recapture estimates of the hidden pool size (unstratified
        and stratified by benchmark) and a parametric bootstrap of the stratified total
noise   within-arm replicate differences (determinism, partner noise, pooling gain), between-arm matched spread,
        between-pair noise after B31 adjustment, snapshot drift check
fe      exact two-way fixed effects (least squares on pool-aware cells), pair-cluster bootstrap SE (default 1000 reps),
        the ratio to tools/feedback.py approx_se, and a benchmark jackknife (stability only; benchmarks stay unnamed)
agree   local paired LOBO effects (unweighted and reweighted) vs live effects from the pool-aware FE and the
        pair+covariate FE, with SE = max(stratified bootstrap, sd/sqrt(K)) and agreement statistics
replay  reweights local LOBO pairs so the reference model's (B0, B31) distribution matches the live GLOBAL one and
        re-scores the local paired effects under each weighting mode
robust  arm effects under four estimators: pool-aware, pair+covariate, benchmark-balanced, pool-weighted
power   analytic matched-pair yield from the stratified pool model, SE and minimum detectable effect per design

--select / --holdout (added swarm round 3; anywhere on the command line, any subcommand): restrict the live data to
        the selection set / the pinned 25% hash holdout of QUEUE.md (holdout_flag below). Without them nothing changes.

Library use: FE2(w).fit("paircov") is the recommended primary live estimator (unit = pair, common 'shared' covariate,
B0 without covariate); live_estimates(w) adds the stratified bootstrap and the SE floor. holdout_mask(w) flags the
held-out rows of any wide frame. The daily decision report built on these is tools/daily_report.py.

Compliance: anonymous IDs are used only as opaque join keys. Every printed quantity is a global count, a
distribution over pairs, or an arm-level comparison; no ID is mapped to an identity and no pair's labels or base rate
are reconstructed. Nothing here feeds per-ID behaviour into a model.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from feedback import ledger_arms, load  # noqa: E402

REF = "v3b_noM_n_m20"
SECTIONS = ["alc", "b0", "b1", "b3", "b7", "b15", "b31"]
RNG = np.random.default_rng(20261006)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def live_wide() -> pd.DataFrame:
    """One row per (submission, pair) with all sections, the arm, and the pool-aware condition."""
    df = load()
    w = df.pivot_table(index=["submission", "subject_id", "benchmark_id"], columns="section",
                       values="score").reset_index()
    n_u = df[df.section == "alc"].set_index(["submission", "subject_id", "benchmark_id"])["n_unique"]
    w["n_unique"] = pd.MultiIndex.from_frame(w[["submission", "subject_id", "benchmark_id"]]).map(n_u.to_dict())
    arm_of = ledger_arms()
    w["arm"] = w["submission"].map(arm_of)
    w["key"] = w["subject_id"] + "|" + w["benchmark_id"]
    n_same = w.groupby(["submission", "benchmark_id"])["subject_id"].transform("nunique")
    w["cond"] = np.where(n_same > 1, "shared", "alone")
    w["pkey"] = w["key"] + "|" + w["cond"]
    return w


# ----------------------------------------------------------------------------------------------------------------
# pinned 25% hash holdout (added in swarm round 3; additive, nothing else in this file uses it unless --select or
# --holdout is given)
# ----------------------------------------------------------------------------------------------------------------
HOLDOUT_SALT = "brier-rabbit-holdout"
_SUBJECT_ID = re.compile(r"subject_\d+")
_BENCHMARK_ID = re.compile(r"benchmark_\d+")


def holdout_flag(subject_id: str, benchmark_id: str) -> bool:
    """The QUEUE.md holdout, pinned: sha256('brier-rabbit-holdout|subject_<id>|benchmark_<id>') with both IDs exactly as
    the scoring logs print them (prefixes included; tools/feedback.py keeps them), held out iff the first two hex digits
    are < '40' (25% in expectation). Hashing bare digits would move ~1/3 of appearances, so any other ID spelling is an
    error rather than a silent reshuffle. Opaque join keys only: the hash never maps an ID to an identity."""
    if not (_SUBJECT_ID.fullmatch(str(subject_id)) and _BENCHMARK_ID.fullmatch(str(benchmark_id))):
        raise ValueError(f"holdout_flag needs IDs as printed in the logs (subject_<digits>, benchmark_<digits>); got "
                         f"{type(subject_id).__name__}/{type(benchmark_id).__name__} in another format")
    return hashlib.sha256(f"{HOLDOUT_SALT}|{subject_id}|{benchmark_id}".encode()).hexdigest()[:2] < "40"


def holdout_mask(w: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned with w (True = held-out pair) from its subject_id / benchmark_id columns."""
    pairs = w[["subject_id", "benchmark_id"]].drop_duplicates()
    flag = {(s, b): holdout_flag(s, b) for s, b in pairs.itertuples(index=False)}
    return pd.Series([flag[(s, b)] for s, b in zip(w["subject_id"], w["benchmark_id"])], index=w.index, dtype=bool)


def _pop_holdout_switch() -> str | None:
    """Remove --select / --holdout from sys.argv (so the positional parsing in main() is unchanged) and return the
    chosen subset, or None when neither flag is present."""
    flags = [a for a in sys.argv[1:] if a in ("--select", "--holdout")]
    if not flags:
        return None
    if len(set(flags)) > 1:
        raise SystemExit("use either --select or --holdout, not both")
    sys.argv[:] = [a for a in sys.argv if a not in ("--select", "--holdout")]
    return flags[0][2:]


# ----------------------------------------------------------------------------------------------------------------
# 1. pool size
# ----------------------------------------------------------------------------------------------------------------
def mt_mle(sizes: list[int], s_obs: int, n_max: int | None = None):
    """Model M_t with fixed sample sizes (each run is a simple random sample of n_t units from N):
    L(N) = N!/(N-S)! / prod_t C(N, n_t). Returns the MLE and a 95% profile-likelihood interval."""
    n_max = n_max or 60 * max(s_obs, 1)
    grid = np.arange(s_obs, n_max + 1)
    lg = np.vectorize(math.lgamma)
    ll = lg(grid + 1) - lg(grid - s_obs + 1)
    for n in sizes:
        ll = ll - (lg(grid + 1) - math.lgamma(n + 1) - lg(grid - n + 1))
    i = int(np.argmax(ll))
    ok = grid[ll >= ll[i] - 1.92]
    return int(grid[i]), int(ok.min()), int(ok.max()), bool(ok.max() == n_max)


def chao_mth(f: dict, t: int) -> float:
    s = sum(f.values())
    f1, f2 = f.get(1, 0), f.get(2, 0)
    return s + (t - 1) / t * f1 * (f1 - 1) / (2 * (f2 + 1))


def jackknife(f: dict, t: int) -> tuple[float, float]:
    s = sum(f.values())
    f1, f2 = f.get(1, 0), f.get(2, 0)
    j1 = s + f1 * (t - 1) / t
    j2 = s + f1 * (2 * t - 3) / t - f2 * (t - 2) ** 2 / (t * (t - 1))
    return j1, j2


def betabin_mh(counts: np.ndarray, t: int):
    """Model M_h with Beta(a, b) capture probabilities: zero-truncated beta-binomial ML on the capture counts.
    N_hat = S / (1 - P(0)). Returns N_hat, a, b, mean capture probability per run."""
    from scipy.optimize import minimize
    from scipy.special import betaln, gammaln

    k = counts[counts > 0]

    def nll(z):
        a, b = np.exp(z)
        logp = gammaln(t + 1) - gammaln(k + 1) - gammaln(t - k + 1) + betaln(k + a, t - k + b) - betaln(a, b)
        p0 = np.exp(betaln(a, t + b) - betaln(a, b))
        return -(logp.sum() - len(k) * np.log1p(-p0))

    best = None
    for z0 in ([0.0, 2.0], [-1.0, 1.0], [1.0, 4.0], [0.5, 3.0]):
        r = minimize(nll, np.array(z0), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(best.x)
    p0 = math.exp(betaln(a, t + b) - betaln(a, b))  # P(never captured in t runs)
    return len(k) / (1 - p0), a, b, a / (a + b)


def pool_stats(w: pd.DataFrame, verbose: bool = True) -> dict:
    runs = sorted(w["submission"].unique())
    t = len(runs)
    sets = [set(w.loc[w.submission == r, "key"]) for r in runs]
    sizes = [len(s) for s in sets]
    caps = w.groupby("key")["submission"].nunique()
    f = caps.value_counts().sort_index().to_dict()
    s_obs = int(caps.size)
    # pooled Lincoln-Petersen (ratio of sums over run pairs) and Schnabel
    num = den = 0
    for i in range(t):
        for j in range(i + 1, t):
            num += sizes[i] * sizes[j]
            den += len(sets[i] & sets[j])
    lp = num / den if den else float("inf")
    marked, s_num, s_den = set(), 0.0, 0
    for s in sets:
        s_num += len(s) * len(marked)
        s_den += len(s & marked)
        marked |= s
    schnabel = s_num / (s_den + 1)
    mt, mt_lo, mt_hi, mt_open = mt_mle(sizes, s_obs)
    j1, j2 = jackknife(f, t)
    bb = betabin_mh(caps.reindex(caps.index).to_numpy(), t)
    # stratified by benchmark (heterogeneity is mostly between benchmarks: every run contains every benchmark)
    strat = []
    for b, g in w.groupby("benchmark_id"):
        bsets = [set(g.loc[g.submission == r, "key"]) for r in runs]
        bsizes = [len(s) for s in bsets if s]
        bcaps = g.groupby("key")["submission"].nunique()
        bf = bcaps.value_counts().sort_index().to_dict()
        m, lo, hi, op = mt_mle(bsizes, int(bcaps.size), n_max=40 * int(bcaps.size))
        strat.append({"s_obs": int(bcaps.size), "draws": int(sum(bsizes)), "mt": m, "lo": lo, "hi": hi, "open": op,
                      "chao": chao_mth(bf, len(bsizes)), "j1": jackknife(bf, len(bsizes))[0],
                      "sightings_per_run": float(np.mean(bsizes))})
    st = pd.DataFrame(strat).sort_values("mt").reset_index(drop=True)
    out = {"runs": t, "appearances": int(len(w)), "S_obs": s_obs, "f": f, "pairs_per_run": float(np.mean(sizes)),
           "benchmarks": int(w["benchmark_id"].nunique()),
           "bench_per_run": float(w.groupby("submission")["benchmark_id"].nunique().mean()),
           "subjects": int(w["subject_id"].nunique()),
           "subjects_in_multiple_benchmarks": int((w.groupby("subject_id")["benchmark_id"].nunique() > 1).sum()),
           "shared_fraction": float((w["cond"] == "shared").mean()),
           "lp_pooled": lp, "schnabel": schnabel, "mt": (mt, mt_lo, mt_hi, mt_open), "chao_mth": chao_mth(f, t),
           "jack1": j1, "jack2": j2, "betabin": bb, "strat": st,
           "strat_total_mt": int(st["mt"].sum()), "strat_total_lo": int(st["lo"].sum()),
           "strat_total_hi": int(st["hi"].sum()), "strat_total_chao": float(st["chao"].sum()),
           "strat_total_j1": float(st["j1"].sum())}
    if verbose:
        print("== POOL (global counts only)")
        print(f"runs {t}, pair appearances {len(w)}, distinct pairs {s_obs}, pairs/run {np.mean(sizes):.2f}, "
              f"benchmarks {out['benchmarks']} (per run {out['bench_per_run']:.2f}), distinct subjects {out['subjects']}, "
              f"subjects seen in >1 benchmark {out['subjects_in_multiple_benchmarks']}")
        print(f"capture-frequency counts f_k (k = #runs a pair appeared in): {f}")
        print(f"share of appearances where another pair of the same benchmark is in the draw: {out['shared_fraction']:.3f}")
        print(f"pooled Lincoln-Petersen {lp:.0f} | Schnabel {schnabel:.0f} | M_t MLE {mt} (95% PL {mt_lo}-{mt_hi}"
              f"{', open' if mt_open else ''}) | Chao M_th {out['chao_mth']:.0f} | jackknife J1 {j1:.0f} J2 {j2:.0f} | "
              f"beta-binomial M_h {bb[0]:.0f} (a={bb[1]:.2f}, b={bb[2]:.2f}, mean p/run {bb[3]:.3f})")
        print(f"stratified by benchmark (sum over {len(st)} strata): M_t {out['strat_total_mt']} "
              f"(sum of 95% PL bounds {out['strat_total_lo']}-{out['strat_total_hi']}), Chao {out['strat_total_chao']:.0f}, "
              f"J1 {out['strat_total_j1']:.0f}")
        print("per-stratum estimates, sorted, unlabeled: M_t", st["mt"].tolist(), "| observed", st["s_obs"].tolist(),
              "| pairs drawn per run", [round(x, 2) for x in st["sightings_per_run"]])
    return out


def pool_ci(w: pd.DataFrame, sims: int = 300) -> dict:
    """Parametric bootstrap for the stratified pool total: simulate the draw (one pair per stratum per run plus a
    second pair w.p. d_b) from the fitted stratum sizes, re-estimate each stratum by M_t, sum. Global count only."""
    st = stratum_model(w)
    t = w["submission"].nunique()
    totals = []
    for _ in range(sims):
        tot = 0
        for _, r in st.iterrows():
            n_b, d = int(r["N"]), float(r["d"])
            seen, sizes = set(), []
            for _r in range(t):
                m = 2 if RNG.random() < d else 1
                draw = RNG.choice(n_b, size=min(m, n_b), replace=False)
                sizes.append(len(draw))
                seen.update(draw.tolist())
            tot += mt_mle(sizes, len(seen), n_max=40 * max(len(seen), 1))[0]
        totals.append(tot)
    totals = np.array(totals)
    point = int(st["N"].sum())
    q = np.quantile(totals, [0.05, 0.25, 0.5, 0.75, 0.95])
    out = {"point": point, "median": float(q[2]), "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]),
           "q95": float(q[4]), "share_capped": float(np.mean(totals >= 0.9 * 40 * st["seen"].sum()))}
    print(f"\n== POOL TOTAL parametric bootstrap ({sims} sims from the fitted strata): point {point}; re-estimates "
          f"median {q[2]:.0f}, IQR {q[1]:.0f}-{q[3]:.0f}, 5-95% {q[0]:.0f}-{q[4]:.0f} (right tail = large strata with "
          f"few recaptures; the M_t MLE is unstable there)")
    return out


# ----------------------------------------------------------------------------------------------------------------
# 2. exact two-way fixed effects
# ----------------------------------------------------------------------------------------------------------------
def cells_of(w: pd.DataFrame, unit: str = "pkey") -> pd.DataFrame:
    """Cell means per (pool-aware unit, arm) for every section, plus the underlying pair key and benchmark."""
    d = w.dropna(subset=["arm"])
    c = d.groupby([unit, "arm"])[SECTIONS].mean().reset_index()
    first = d.drop_duplicates(unit).set_index(unit)
    c["key"] = c[unit].map(first["key"])
    c["bench"] = c[unit].map(first["benchmark_id"])
    return c.rename(columns={unit: "unit"})


def fe_core(units: np.ndarray, arms: np.ndarray, Y: np.ndarray, n_arms: int, ref: int) -> np.ndarray:
    """Least squares Y[c] = alpha[unit c] + beta[arm c], beta[ref] = 0, for several columns of Y at once, with the
    unit effects absorbed (within-unit demeaning gives the normal equations G beta = h on the arms only).
    Requires at most one cell per (unit, arm). Units seen under one arm drop out; arms not connected to ref -> NaN."""
    out = np.full((n_arms, Y.shape[1]), np.nan)
    out[ref] = 0.0
    if len(units) == 0:
        return out
    uniq_u, ui = np.unique(units, return_inverse=True)
    B = np.zeros((len(uniq_u), n_arms))
    B[ui, arms] = 1.0
    n_u = B.sum(1)
    keep = n_u > 1
    if not keep.any():
        return out
    cm = keep[ui]
    remap = np.full(len(uniq_u), -1)
    remap[keep] = np.arange(int(keep.sum()))
    ui2, arms2, Y2 = remap[ui[cm]], arms[cm], Y[cm]
    B, n_u = B[keep], n_u[keep]
    C = (B.T @ B) > 0  # arms sharing a unit
    reach = np.zeros(n_arms, bool)
    reach[ref] = True
    while True:
        new = reach | C[reach].any(0)
        if (new == reach).all():
            break
        reach = new
    est = np.flatnonzero(reach & (np.arange(n_arms) != ref))
    if len(est) == 0:
        return out
    Wb = B / n_u[:, None]
    G = np.diag(B.sum(0)) - B.T @ Wb
    S_u = np.zeros((len(n_u), Y.shape[1]))
    np.add.at(S_u, ui2, Y2)
    H = np.zeros((n_arms, Y.shape[1]))
    np.add.at(H, arms2, Y2)
    h = H - Wb.T @ S_u
    sol, *_ = np.linalg.lstsq(G[np.ix_(est, est)], h[est], rcond=None)
    out[est] = sol
    return out


class FE:
    """Precomputed cell arrays so that refits (bootstrap, jackknife) are fast."""

    def __init__(self, w: pd.DataFrame, ref: str = REF, unit: str = "pkey"):
        self.c = cells_of(w, unit)
        self.arms = sorted(self.c["arm"].unique())
        self.ai = {a: i for i, a in enumerate(self.arms)}
        self.ref = self.ai[ref]
        self.u_codes, self.u_names = pd.factorize(self.c["unit"])
        self.a_codes = self.c["arm"].map(self.ai).to_numpy()
        self.Y = self.c[SECTIONS].to_numpy(float)
        self.keys = self.c["key"].unique()
        self.rows_of_key = {k: np.flatnonzero(self.c["key"].to_numpy() == k) for k in self.keys}

    def fit(self, rows: np.ndarray | None = None, unit_codes: np.ndarray | None = None) -> pd.DataFrame:
        rows = np.arange(len(self.c)) if rows is None else rows
        u = self.u_codes[rows] if unit_codes is None else unit_codes
        b = fe_core(u, self.a_codes[rows], self.Y[rows], len(self.arms), self.ref)
        return pd.DataFrame(b, index=self.arms, columns=SECTIONS)

    def boot(self, reps: int = 400) -> list[pd.DataFrame]:
        outs = []
        nu = len(self.u_names)
        for _ in range(reps):
            pick = RNG.choice(len(self.keys), size=len(self.keys), replace=True)
            rows, ucodes = [], []
            for i, k in enumerate(pick):
                r = self.rows_of_key[self.keys[k]]
                rows.append(r)
                ucodes.append(self.u_codes[r] + i * nu)  # duplicates become distinct units
            rows = np.concatenate(rows)
            outs.append(self.fit(rows, np.concatenate(ucodes)))
        return outs

    def without_bench(self, bench: str) -> pd.DataFrame:
        return self.fit(np.flatnonzero(self.c["bench"].to_numpy() != bench))


def fe_exact(w: pd.DataFrame, section: str, ref: str = REF, unit: str = "pkey") -> pd.Series:
    return FE(w, ref, unit).fit()[section]


def fe_resid_sd(w: pd.DataFrame, section: str, ref: str = REF) -> tuple[float, int]:
    """Residual SD of the cell-level FE fit with the correct degrees of freedom (cells - units - arms + 1)."""
    c = cells_of(w)
    c = c[c.groupby("unit")["arm"].transform("nunique") > 1]
    beta = fe_exact(w, section, ref)
    c = c[c["arm"].map(beta).notna()]
    r = c[section] - c["arm"].map(beta)
    resid = r - r.groupby(c["unit"]).transform("mean")
    dof = len(c) - c["unit"].nunique() - (c["arm"].nunique() - 1)
    return float(np.sqrt((resid ** 2).sum() / max(dof, 1))), int(dof)


def shared_counts(w: pd.DataFrame, ref: str = REF) -> pd.Series:
    """Per arm: number of pool-aware units it shares with the reference arm directly."""
    ref_units = set(w.loc[w.arm == ref, "pkey"])
    return w.groupby("arm")["pkey"].apply(lambda s: len(set(s) & ref_units))


def fe_bootstrap(w: pd.DataFrame, sections, reps: int = 400, ref: str = REF) -> dict:
    """Pair-cluster bootstrap: resample underlying (subject, benchmark) keys with replacement; duplicates become
    distinct units. Returns {section: DataFrame(effect, se, lo90, hi90, boot_na)} plus {"_draws": {section: arms x reps}}
    so that contrasts between any two arms get a proper (correlated) bootstrap SE."""
    fe = FE(w, ref)
    point = fe.fit()
    draws = fe.boot(reps)
    out = {"_draws": {}}
    for s in sections:
        bd = pd.concat([d[s] for d in draws], axis=1)
        out["_draws"][s] = bd
        out[s] = pd.DataFrame({"effect": point[s], "se": bd.std(axis=1), "lo90": bd.quantile(0.05, axis=1),
                               "hi90": bd.quantile(0.95, axis=1), "boot_na": bd.isna().mean(axis=1)})
    return out


def contrast(boot: dict, section: str, arm: str, base: str) -> tuple[float, float]:
    """FE contrast arm - base with its bootstrap SE (both relative to the same reference in every draw)."""
    e = boot[section]["effect"]
    d = boot["_draws"][section]
    if arm not in e.index or base not in e.index:
        return np.nan, np.nan
    diff = (d.loc[arm] - d.loc[base]).dropna()
    return float(e[arm] - e[base]), float(diff.std()) if len(diff) > 10 else np.nan


def bench_jackknife(w: pd.DataFrame, section: str, ref: str = REF) -> pd.DataFrame:
    """Leave-one-benchmark-out FE (stability only). Reports jackknife SE and the range; benchmarks stay unnamed."""
    fe = FE(w, ref)
    point = fe.fit()[section]
    jd = pd.concat([fe.without_bench(b)[section] for b in w["benchmark_id"].unique()], axis=1)
    nb = jd.notna().sum(axis=1)
    se = np.sqrt((nb - 1) / nb * ((jd.sub(jd.mean(axis=1), axis=0)) ** 2).sum(axis=1))
    return pd.DataFrame({"effect": point, "jk_se": se, "jk_min": jd.min(axis=1), "jk_max": jd.max(axis=1),
                         "jk_n": nb})


def fe_within(unit_codes: np.ndarray, arm_codes: np.ndarray, Y: np.ndarray, n_arms: int, ref: int,
              Z: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """General two-way FE with optional covariates Z (cells x q); unit effects absorbed by within-unit demeaning, so
    several cells per (unit, arm) are allowed (e.g. unit = pair, cells split alone/shared, Z = shared indicator).
    Returns (beta arms x sections with NaN for arms not connected to ref, gamma q x sections)."""
    q = 0 if Z is None else Z.shape[1]
    out = np.full((n_arms, Y.shape[1]), np.nan)
    out[ref] = 0.0
    gam = np.full((q, Y.shape[1]), np.nan)
    if len(unit_codes) == 0:
        return out, gam
    uniq, ui, cnt = np.unique(unit_codes, return_inverse=True, return_counts=True)
    keep = cnt[ui] > 1
    if not keep.any():
        return out, gam
    ui = np.unique(ui[keep], return_inverse=True)[1]
    arms, Y = arm_codes[keep], Y[keep]
    Z = None if Z is None else Z[keep]
    nu = int(ui.max()) + 1
    B = np.zeros((nu, n_arms))
    B[ui, arms] = 1.0
    multi = B.sum(1) > 1
    C = (B[multi].T @ B[multi]) > 0
    reach = np.zeros(n_arms, bool)
    reach[ref] = True
    while True:
        new = reach | C[reach].any(0)
        if (new == reach).all():
            break
        reach = new
    est = np.flatnonzero(reach & (np.arange(n_arms) != ref))
    col = np.full(n_arms, -1)
    col[est] = np.arange(len(est))
    X = np.zeros((len(ui), len(est) + q))
    m = col[arms] >= 0
    X[np.flatnonzero(m), col[arms][m]] = 1.0
    if q:
        X[:, len(est):] = Z
    cnt_u = np.bincount(ui, minlength=nu).astype(float)
    Xm = np.zeros((nu, X.shape[1]))
    np.add.at(Xm, ui, X)
    Ym = np.zeros((nu, Y.shape[1]))
    np.add.at(Ym, ui, Y)
    Xd = X - (Xm / cnt_u[:, None])[ui]
    Yd = Y - (Ym / cnt_u[:, None])[ui]
    # an arm column that is identically zero after demeaning is not identified
    ok = np.abs(Xd).sum(0) > 1e-12
    coef = np.full((X.shape[1], Y.shape[1]), np.nan)
    if ok.any():
        sol, *_ = np.linalg.lstsq(Xd[:, ok], Yd, rcond=None)
        coef[ok] = sol
    out[est] = coef[:len(est)]
    gam = coef[len(est):]
    return out, gam


class FE2:
    """Robust variants of the arm comparison on the same live data:
    - 'poolaware' : unit = pair x {alone, shared} (the repo's definition), one cell per (unit, arm)
    - 'paircov'   : unit = pair, cells split by condition, common 'shared' covariate (doubles the matched yield)
    - balanced     : per-benchmark FE averaged with draw-share weights (what a draw's mean score weights)
    - poolweighted : per-benchmark FE averaged with weights ~ estimated subjects per benchmark (pair-equal pool)"""

    def __init__(self, w: pd.DataFrame, ref: str = REF, strata_weights: dict | None = None):
        d = w.dropna(subset=["arm"]).copy()
        c = d.groupby(["key", "cond", "arm"])[SECTIONS].mean().reset_index()
        c["bench"] = c["key"].map(d.drop_duplicates("key").set_index("key")["benchmark_id"])
        self.c = c
        self.arms = sorted(c["arm"].unique())
        self.ai = {a: i for i, a in enumerate(self.arms)}
        self.ref = self.ai[ref]
        self.a = c["arm"].map(self.ai).to_numpy()
        self.Y = c[SECTIONS].to_numpy(float)
        self.kcode, self.knames = pd.factorize(c["key"])
        self.pcode = pd.factorize(c["key"] + "|" + c["cond"])[0]
        self.Z = (c["cond"] == "shared").to_numpy(float)[:, None]
        self.bench = c["bench"].to_numpy()
        self.benches = sorted(set(self.bench))
        per_run = d.groupby(["submission", "benchmark_id"]).size().groupby(level=1).mean()
        share = per_run / per_run.sum()
        self.share = share.to_dict()
        self.poolw = strata_weights or self.share
        self.key_bench = c.drop_duplicates("key").set_index("key")["bench"]
        self.rows_of_key = {k: np.flatnonzero(self.kcode == i) for i, k in enumerate(self.knames)}

    def _fit(self, rows, kcode, pcode, how):
        a, Y, Z, bench = self.a[rows], self.Y[rows], self.Z[rows], self.bench[rows]
        if how == "poolaware":
            return fe_within(pcode, a, Y, len(self.arms), self.ref)[0]
        if how == "paircov":
            return self._paircov(kcode, a, Y, Z)
        weights = self.share if how == "balanced" else self.poolw
        acc = np.zeros((len(self.arms), Y.shape[1]))
        wsum = np.zeros((len(self.arms), Y.shape[1]))
        for b in self.benches:
            m = bench == b
            if not m.any():
                continue
            beta = self._paircov(kcode[m], a[m], Y[m], Z[m])
            ok = ~np.isnan(beta)
            acc[ok] += weights[b] * beta[ok]
            wsum[ok] += weights[b]
        out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
        self.last_cover = wsum / sum(weights.values())
        return out

    def _paircov(self, kcode, a, Y, Z):
        """unit = pair with a common 'shared' covariate; B0 has no labels so it is condition-free: no covariate."""
        beta = fe_within(kcode, a, Y, len(self.arms), self.ref, Z)[0]
        j0 = SECTIONS.index("b0")
        beta[:, j0] = fe_within(kcode, a, Y[:, [j0]], len(self.arms), self.ref)[0][:, 0]
        return beta

    def fit(self, how: str) -> pd.DataFrame:
        rows = np.arange(len(self.c))
        res = self._fit(rows, self.kcode, self.pcode, how)
        df = pd.DataFrame(res, index=self.arms, columns=SECTIONS)
        if how in ("balanced", "poolweighted"):
            df["weight_covered"] = self.last_cover[:, 0]
        return df

    def boot(self, how: str, reps: int = 200, stratified: bool = True) -> list[np.ndarray]:
        outs = []
        nk = len(self.knames)
        keys_by_b = {b: [i for i, k in enumerate(self.knames) if self.key_bench[k] == b] for b in self.benches}
        for _ in range(reps):
            if stratified:
                pick = np.concatenate([RNG.choice(v, size=len(v), replace=True) for v in keys_by_b.values()])
            else:
                pick = RNG.choice(nk, size=nk, replace=True)
            rows, kc, pc = [], [], []
            for j, i in enumerate(pick):
                r = self.rows_of_key[self.knames[i]]
                rows.append(r)
                kc.append(np.full(len(r), j))
                pc.append(self.pcode[r] + (j + 1) * 100000)
            rows = np.concatenate(rows)
            outs.append(self._fit(rows, np.concatenate(kc), np.concatenate(pc), how))
        return outs


def robust_table(w: pd.DataFrame, pool: dict, reps: int = 150, verbose: bool = True) -> pd.DataFrame:
    st = {}
    for b, g in w.groupby("benchmark_id"):
        caps = g.groupby("key")["submission"].nunique()
        sizes = [x for x in (len(set(g.loc[g.submission == r, "key"])) for r in sorted(w.submission.unique())) if x]
        st[b] = mt_mle(sizes, int(caps.size), n_max=40 * int(caps.size))[0]
    tot = sum(st.values())
    fe2 = FE2(w, strata_weights={b: v / tot for b, v in st.items()})
    rows = []
    for how in ("poolaware", "paircov", "balanced", "poolweighted"):
        point = fe2.fit(how)
        bs = np.stack(fe2.boot(how, reps=reps))
        se = np.nanstd(bs, axis=0, ddof=1)
        lo = np.nanquantile(bs, 0.05, axis=0)
        hi = np.nanquantile(bs, 0.95, axis=0)
        for i, arm in enumerate(fe2.arms):
            for j, s in enumerate(SECTIONS):
                if s not in ("alc", "b0", "b1", "b3"):
                    continue
                rows.append({"how": how, "arm": arm, "section": s, "effect": point.iloc[i, j], "se": se[i, j],
                             "lo90": lo[i, j], "hi90": hi[i, j],
                             "covered": point["weight_covered"].iloc[i] if "weight_covered" in point else np.nan})
    t = pd.DataFrame(rows)
    if verbose:
        print("\n== ROBUST ARM EFFECTS vs", REF, "(stratified pair bootstrap;",
              "'covered' = share of benchmark weight where the arm is identified)")
        piv = t[t.section == "alc"].pivot_table(index="arm", columns="how", values=["effect", "se"])
        print(piv.round(4).to_string())
        cov = t[(t.section == "alc") & t.how.isin(["balanced"])].set_index("arm")["covered"]
        print("benchmark-weight covered (balanced):", cov.round(2).to_dict())
        for s in ("b0", "b1", "b3"):
            piv = t[t.section == s].pivot_table(index="arm", columns="how", values="effect")
            print(f"\n{s} effects by estimator:")
            print(piv.round(4).to_string())
    return t


# ----------------------------------------------------------------------------------------------------------------
# 3. noise structure
# ----------------------------------------------------------------------------------------------------------------
def noise(w: pd.DataFrame, verbose: bool = True) -> dict:
    rows = []
    for arm, g in w.groupby("arm"):
        for key, h in g.groupby("key"):
            if len(h) < 2:
                continue
            h = h.sort_values("submission")
            for i in range(len(h)):
                for j in range(i + 1, len(h)):
                    a, b = h.iloc[i], h.iloc[j]
                    kind = "alone/alone" if a.cond == b.cond == "alone" else (
                        "shared/shared" if a.cond == b.cond == "shared" else "alone/shared")
                    sh, al = (a, b) if a.cond == "shared" else (b, a)
                    diff = {s: (sh[s] - al[s]) if kind == "alone/shared" else (b[s] - a[s]) for s in SECTIONS}
                    rows.append({"arm": arm, "kind": kind, **diff})
    rep = pd.DataFrame(rows)
    out = {"replicates": rep}
    if verbose:
        print("\n== WITHIN-ARM REPLICATES (same arm, same pair, two runs)")
        if rep.empty:
            print("none")
        else:
            for kind, g in rep.groupby("kind"):
                stat = "mean(shared-alone)" if kind == "alone/shared" else "mean |diff|"
                vals = g[SECTIONS].mean() if kind == "alone/shared" else g[SECTIONS].abs().mean()
                print(f"{kind:14s} n={len(g):3d}  {stat}: " + " ".join(f"{s} {vals[s]:+.4f}" for s in SECTIONS))
                if kind == "alone/shared":
                    se = g[SECTIONS].std(ddof=1) / np.sqrt(len(g))
                    print(" " * 20 + "se: " + " ".join(f"{s} {se[s]:.4f}" for s in SECTIONS))
            arms_m = rep.groupby(["arm", "kind"]).size().unstack(fill_value=0)
            print(arms_m.to_string())
    return out


def ancova_noise(w: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """How much of a pair's score level does its own (arm-invariant) B31 explain? Within each arm, regress the section
    score on [1, B31, B31^2, shared]; pool the residual SD over arms. If it is close to the matched-difference SD, an
    UNMATCHED sighting is almost as informative as a matched one (combine within- and between-pair information)."""
    rows = []
    for s in ("alc", "b0", "b1", "b3", "b7"):
        rss, dof = 0.0, 0
        for arm, g in w.groupby("arm"):
            g = g.dropna(subset=[s, "b31"])
            if len(g) < 6:
                continue
            X = np.column_stack([np.ones(len(g)), g["b31"], g["b31"] ** 2, (g["cond"] == "shared").astype(float)])
            coef, *_ = np.linalg.lstsq(X, g[s].to_numpy(), rcond=None)
            r = g[s].to_numpy() - X @ coef
            rss += float((r ** 2).sum())
            dof += len(g) - X.shape[1]
        raw = w.groupby("arm")[s].std().mean()
        rows.append({"section": s, "raw_sd_within_arm": raw, "resid_sd_given_b31": math.sqrt(rss / max(dof, 1)), "dof": dof})
    t = pd.DataFrame(rows)
    if verbose:
        print("\n== BETWEEN-PAIR NOISE AFTER B31 ADJUSTMENT (within arm; pooled over arms with >= 6 sightings)")
        print(t.round(4).to_string(index=False))
    return t


def drift_check(w: pd.DataFrame, verbose: bool = True) -> dict:
    """Snapshot drift detector (global counts only): a pair's evaluation size and its B0 under the same arm must not
    change between runs unless the evaluation snapshot changed."""
    n_change = int((w.groupby("key")["n_unique"].nunique() > 1).sum()) if "n_unique" in w else -1
    b0 = w.groupby(["arm", "key"])["b0"].agg(["min", "max", "size"])
    b0 = b0[b0["size"] > 1]
    out = {"pairs_with_changed_eval_size": n_change, "same_arm_b0_repeats": int(len(b0)),
           "same_arm_b0_max_abs_change": float((b0["max"] - b0["min"]).max()) if len(b0) else float("nan")}
    if verbose:
        print("\n== DRIFT CHECK", out)
    return out


def matched_spread(w: pd.DataFrame, min_k: int = 3, verbose: bool = True) -> pd.DataFrame:
    """For every pair of arms: matched pool-aware units, mean and SD of the per-unit difference."""
    cell = w.groupby(["pkey", "arm"])[SECTIONS].mean().reset_index()
    arms = sorted(cell["arm"].unique())
    rows = []
    for i, a in enumerate(arms):
        ca = cell[cell.arm == a].set_index("pkey")
        for b in arms[i + 1:]:
            cb = cell[cell.arm == b].set_index("pkey")
            common = ca.index.intersection(cb.index)
            if len(common) < min_k:
                continue
            d = cb.loc[common, SECTIONS] - ca.loc[common, SECTIONS]
            row = {"a": a, "b": b, "k": len(common)}
            for s in SECTIONS:
                row[f"{s}_mean"] = d[s].mean()
                row[f"{s}_sd"] = d[s].std(ddof=1)
            rows.append(row)
    ms = pd.DataFrame(rows)
    if verbose and not ms.empty:
        print(f"\n== BETWEEN-ARM MATCHED DIFFERENCES (arm pairs with >= {min_k} matched pool-aware units)")
        show = ms[["a", "b", "k"] + [f"{s}_{x}" for s in ("alc", "b0", "b1", "b3") for x in ("mean", "sd")]]
        print(show.round(4).to_string(index=False))
        wts = ms["k"] - 1
        pooled = {s: float(np.sqrt((wts * ms[f"{s}_sd"] ** 2).sum() / wts.sum())) for s in SECTIONS}
        print("pooled SD of a matched per-pair arm difference (dof-weighted):",
              {s: round(v, 4) for s, v in pooled.items()})
    return ms


# ----------------------------------------------------------------------------------------------------------------
# 4. power: simulate matched yields for upload designs
# ----------------------------------------------------------------------------------------------------------------
def stratum_model(w: pd.DataFrame) -> pd.DataFrame:
    """Per benchmark stratum (kept unlabeled): estimated number of subjects N_b (M_t MLE within the stratum) and the
    per-run probability d_b that the stratum receives a second pair. Every run draws >= 1 pair from every stratum."""
    runs = sorted(w.submission.unique())
    rows = []
    for b, g in w.groupby("benchmark_id"):
        caps = g.groupby("key")["submission"].nunique()
        sizes = [x for x in (len(set(g.loc[g.submission == r, "key"])) for r in runs) if x]
        m, lo, hi, op = mt_mle(sizes, int(caps.size), n_max=40 * int(caps.size))
        rows.append({"N": m, "N_lo": lo, "N_hi": hi, "seen": int(caps.size),
                     "d": float(np.mean([x > 1 for x in sizes]) * len(sizes) / len(runs))})
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def expected_matches(strata: pd.DataFrame, n_a: float, n_b: float, ncol: str = "N") -> dict:
    """Analytic expected matched units between an arm with n_a runs and one with n_b runs.
    Per run a given subject of stratum b is seen alone w.p. (1-d)/N and shared w.p. 2d/N."""
    k_pa = k_any = 0.0
    per_b = []
    for _, r in strata.iterrows():
        N, d = max(float(r[ncol]), float(r["seen"])), float(r["d"])
        pa1, ps1, pan1 = (1 - d) / N, min(2 * d / N, 1.0), min((1 + d) / N, 1.0)
        A = lambda p, n: 1 - (1 - p) ** n  # noqa: E731
        kp = N * (A(pa1, n_a) * A(pa1, n_b) + A(ps1, n_a) * A(ps1, n_b))
        ka = N * A(pan1, n_a) * A(pan1, n_b)
        k_pa += kp
        k_any += ka
        per_b.append((N, ka))
    return {"k_poolaware": k_pa, "k_any": k_any, "per_b": per_b}


def power(w: pd.DataFrame, pool: dict, ms: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    strata = stratum_model(w)
    sd_fe = {s: math.sqrt(2) * fe_resid_sd(w, s)[0] for s in ("alc", "b0", "b1", "b3")}
    wts = ms["k"] - 1
    sd_ms = {s: float(np.sqrt((wts * ms[f"{s}_sd"] ** 2).sum() / wts.sum())) for s in ("alc", "b0", "b1", "b3")}
    sd = {s: max(sd_fe[s], sd_ms[s]) for s in sd_fe}  # conservative
    share = (1 + strata["d"]) / (1 + strata["d"]).sum()
    rows = []
    scen = {"M_t": strata.assign(Nx=strata["N"]),
            "large strata x2": strata.assign(Nx=np.where(strata["N"] > 40, 2 * strata["N"], strata["N"]))}
    for name, stt in scen.items():
        for n_ref in (10, 20, 40, 80):
            for n in (5, 10, 20, 40, 60):
                e = expected_matches(stt, n_ref, n, ncol="Nx")
                k_pa, k_any = e["k_poolaware"], e["k_any"]
                # benchmark-balanced SE: each stratum's mean difference from its own matched pairs, finite pool
                var_bal = 0.0
                for (N, kb), sh in zip(e["per_b"], share):
                    kb = max(kb, 0.3)
                    var_bal += sh ** 2 * sd["alc"] ** 2 / kb * max(1 - kb / N, 0.0)
                rows.append({"pool": name, "N": int(stt["Nx"].sum()), "n_ref": n_ref, "n_arm": n,
                             "K_poolaware": round(k_pa, 1), "K_pair": round(k_any, 1),
                             "se_alc_pa": sd["alc"] / math.sqrt(max(k_pa, 0.5)),
                             "se_alc_pair": sd["alc"] / math.sqrt(max(k_any, 0.5)),
                             "se_alc_pair_fpc": sd["alc"] / math.sqrt(max(k_any, 0.5)) * math.sqrt(max(1 - k_any / stt["Nx"].sum(), 0)),
                             "se_alc_balanced_fpc": math.sqrt(var_bal),
                             "se_b0": sd["b0"] / math.sqrt(max(k_any, 0.5)),
                             "se_b1_pa": sd["b1"] / math.sqrt(max(k_pa, 0.5))})
    pw = pd.DataFrame(rows)
    pw["mde80_alc_pair"] = 2.80 * pw["se_alc_pair"]
    pw["mde80_alc_bal"] = 2.80 * pw["se_alc_balanced_fpc"]
    if verbose:
        print("\n== POWER (analytic matched yield from the stratified pool model)")
        print("strata (unlabeled, sorted by N): N", strata["N"].tolist(), "| 95% PL", list(zip(strata["N_lo"], strata["N_hi"])),
              "| seen", strata["seen"].tolist(), "| P(second pair per run)", strata["d"].round(2).tolist())
        print("SD of a matched per-pair difference: FE-residual based", {k: round(v, 4) for k, v in sd_fe.items()},
              "| arm-pair based", {k: round(v, 4) for k, v in sd_ms.items()}, "| used (max)", {k: round(v, 4) for k, v in sd.items()})
        print(pw.round(4).to_string(index=False))
    return pw


# ----------------------------------------------------------------------------------------------------------------
# 5. local vs live
# ----------------------------------------------------------------------------------------------------------------
LOCAL_MAP = [
    # live arm, live ref, [(csv, model, ref_model, sign)] chain, note
    ("v1_m05", REF, [("results/paired_damped.csv", "v1_tuned", "n20D")], "v1 names prior m0 -0.5 (undamped)"),
    ("v1_m15", REF, [("results/paired_m0n15.csv", "v1_m0n15", "v1_tuned"),
                     ("results/paired_damped.csv", "v1_tuned", "n20D")], "chain via v1_tuned"),
    ("v2_noM_m15", REF, [("results/paired_damped.csv", "n15", "n20D")], "subject prior, undamped, m0 -1.5"),
    ("v2_noM_m10", REF, [("results/paired_damped.csv", "n10D", "n20D")], "approx: local arm is damped"),
    ("v3_noM_t3_m15", REF, [("results/paired_damped.csv", "t15D", "n20D")], "t(3) level prior"),
    ("v3b_noM_n_m15", REF, [("results/paired_damped.csv", "n15D", "n20D")], "m0 -1.5"),
    ("v3b_noM_n_m10", REF, [("results/paired_damped.csv", "n10D", "n20D")], "m0 -1.0"),
    ("v4_mix25_m10", REF, [("results/paired_mix.csv", "mix25_m10", "base")], "mixture pi .25 (live pending)"),
    ("v4_mix35_m10", REF, [("results/paired_mix.csv", "mix35_m10", "base")], "mixture pi .35 (live pending)"),
]
LOCAL_TAU_M = [("v2_Mtop_m15", "v2_noM_m15"), ("v2_Mtop_m10", "v2_noM_m10"), ("v2_Min_m15", "v2_noM_m15")]
_CSV = {}


def _csv(path):
    if path not in _CSV:
        _CSV[path] = pd.read_csv(REPO / path)
    return _CSV[path]


def local_effect(csv: str, model: str, ref: str, weights: pd.Series | None = None, k_mix: float | None = None) -> dict:
    """Paired local effect model - ref per section: mean over pairs within (bench, k), then k-mix, then bench-equal.
    k_mix = weight on k=1 (None = equal, the repo's macro). Optional per-row weights (index aligned to the ref rows)."""
    d = _csv(csv)
    idx = ["bench", "k", "run", "pair"]
    a = d[d.model == model].set_index(idx)
    b = d[d.model == ref].set_index(idx)
    common = a.index.intersection(b.index)
    diff = (a.loc[common, SECTIONS] - b.loc[common, SECTIONS])
    if weights is not None:
        wv = weights.reindex(common).fillna(0.0)
    else:
        wv = pd.Series(1.0, index=common)
    out = {}
    for s in SECTIONS:
        cells = []
        for (bench, k), g in diff[s].groupby(level=[0, 1]):
            ww = wv.loc[g.index]
            if ww.sum() <= 0:
                continue
            m = float((g * ww).sum() / ww.sum())
            v = float(((ww / ww.sum()) ** 2 * (g - m) ** 2).sum())
            cells.append((bench, k, m, v))
        c = pd.DataFrame(cells, columns=["bench", "k", "m", "v"])
        if k_mix is None:
            per_b = c.groupby("bench").agg(m=("m", "mean"), v=("v", lambda x: x.sum() / len(x) ** 2))
        else:
            c["wk"] = np.where(c["k"] == 1, k_mix, 1 - k_mix)
            per_b = c.groupby("bench").apply(lambda g: pd.Series({"m": (g.m * g.wk).sum() / g.wk.sum(),
                                                                  "v": ((g.wk / g.wk.sum()) ** 2 * g.v).sum()}),
                                             include_groups=False)
        out[s] = (float(per_b["m"].mean()), float(np.sqrt(per_b["v"].sum()) / len(per_b)))
    return out


def chain_effect(chain, **kw) -> dict:
    tot = {s: [0.0, 0.0] for s in SECTIONS}
    for csv, model, ref in chain:
        e = local_effect(csv, model, ref, **kw)
        for s in SECTIONS:
            tot[s][0] += e[s][0]
            tot[s][1] = math.sqrt(tot[s][1] ** 2 + e[s][1] ** 2)
    return {s: tuple(v) for s, v in tot.items()}


def l2o_tau_m(section: str, tau: float = 0.8) -> tuple[float, float]:
    d = _csv("results/l2o_shared_level.csv")
    if section not in d.columns:
        return (np.nan, np.nan)
    a = d[d.tau_M == tau].set_index(["pair_bench", "combo", "run"])[section]
    b = d[d.tau_M == 0.0].set_index(["pair_bench", "combo", "run"])[section]
    diff = (a - b).dropna()
    per_b = diff.groupby(level=0).agg(["mean", "var", "size"])
    return float(per_b["mean"].mean()), float(np.sqrt((per_b["var"] / per_b["size"]).sum()) / len(per_b))


def agree(w: pd.DataFrame, boot: dict, k_mix: float | None, verbose: bool = True) -> pd.DataFrame:
    rows = []
    sh = shared_counts(w)
    for arm, ref, chain, note in LOCAL_MAP:
        loc = chain_effect(chain, k_mix=k_mix)
        for s in ("alc", "b0", "b1", "b3", "b7"):
            live = boot[s].loc[arm] if arm in boot[s].index else None
            rows.append({"contrast": f"{arm} - {ref}", "section": s, "local": loc[s][0], "local_se": loc[s][1],
                         "live": (live["effect"] if live is not None else np.nan),
                         "live_se": (live["se"] if live is not None else np.nan),
                         "shared": int(sh.get(arm, 0)), "note": note})
    # shared-level contrasts: live FE difference of two arms vs local leave-two-out tau_M 0.8 - 0
    for arm, base in LOCAL_TAU_M:
        for s in ("alc", "b0", "b1", "b3"):
            live, lse = contrast(boot, s, arm, base)
            m, se = l2o_tau_m(s)
            rows.append({"contrast": f"{arm} - {base}", "section": s, "local": m, "local_se": se, "live": live,
                         "live_se": lse, "shared": np.nan, "note": "local = leave-two-out tau_M 0.8 vs 0"})
    t = pd.DataFrame(rows)
    t["sign_agree"] = np.where(t["live"].isna(), "", np.where(np.sign(t["local"]) == np.sign(t["live"]), "yes", "NO"))
    t["live_z"] = t["live"] / t["live_se"]
    if verbose:
        mix = "equal k=1/k=2 (repo macro)" if k_mix is None else f"weight {k_mix:.3f} on k=1 (live share alone)"
        print(f"\n== LOCAL vs LIVE (local composition: {mix})")
        print(t.round(4).to_string(index=False))
        for s in ("alc", "b0", "b1", "b3"):
            g = t[(t.section == s) & t.live.notna() & t.local.notna()]
            if len(g) >= 3:
                rho = g[["local", "live"]].corr(method="spearman").iloc[0, 1]
                r = g[["local", "live"]].corr().iloc[0, 1]
                slope = float(np.polyfit(g["local"], g["live"], 1)[0])
                agree_n = int((g.sign_agree == "yes").sum())
                print(f"{s}: n={len(g)} sign agreement {agree_n}/{len(g)}, Pearson {r:+.2f}, Spearman {rho:+.2f}, "
                      f"OLS slope live~local {slope:+.2f}")
    return t


def live_estimates(w: pd.DataFrame, reps: int = 400) -> pd.DataFrame:
    """Live arm effects vs REF from the pool-aware FE and from the pair+covariate FE, with stratified pair-bootstrap
    SEs, direct matched counts, and a model-based SE floor sd_d / sqrt(K) (the bootstrap is degenerate for K <= 2)."""
    fe2 = FE2(w)
    ref_units = set(w.loc[w.arm == REF, "pkey"])
    ref_keys = set(w.loc[w.arm == REF, "key"])
    k_pa = w.groupby("arm")["pkey"].apply(lambda x: len(set(x) & ref_units))
    k_any = w.groupby("arm")["key"].apply(lambda x: len(set(x) & ref_keys))
    sd_floor = {"alc": 0.0147, "b0": 0.0417, "b1": 0.0487, "b3": 0.0106, "b7": 0.0053}
    rows = []
    for how, kk in (("poolaware", k_pa), ("paircov", k_any)):
        point = fe2.fit(how)
        bs = np.stack(fe2.boot(how, reps=reps))
        se = pd.DataFrame(np.nanstd(bs, axis=0, ddof=1), index=fe2.arms, columns=SECTIONS)
        for arm in fe2.arms:
            for sec in ("alc", "b0", "b1", "b3", "b7"):
                k = int(kk.get(arm, 0))
                floor = sd_floor[sec] / math.sqrt(max(k, 1))
                rows.append({"how": how, "arm": arm, "section": sec, "live": point.loc[arm, sec],
                             "live_se_boot": se.loc[arm, sec], "k_direct": k,
                             "live_se": max(se.loc[arm, sec] if np.isfinite(se.loc[arm, sec]) else 0.0, floor)
                             if arm != REF else 0.0})
    return pd.DataFrame(rows)


def agree2(w: pd.DataFrame, k_mix: float, reps: int = 400, verbose: bool = True) -> pd.DataFrame:
    live = live_estimates(w, reps)
    dist = live_global_dist(w)
    rows = []
    for arm, ref, chain, note in LOCAL_MAP:
        for mode in ("none", "joint"):
            tot = {sec: [0.0, 0.0] for sec in SECTIONS}
            for csv, model, refm in chain:
                wv = None
                if mode != "none" and not csv.endswith("paired_m0n15.csv"):
                    wv = replay_weights(csv, "n20D" if "paired_damped" in csv else "base", dist, mode)
                e = local_effect(csv, model, refm, weights=wv, k_mix=k_mix)
                for sec in SECTIONS:
                    tot[sec][0] += e[sec][0]
                    tot[sec][1] = math.sqrt(tot[sec][1] ** 2 + e[sec][1] ** 2)
            for sec in ("alc", "b0", "b1", "b3", "b7"):
                r = {"arm": arm, "mode": mode, "section": sec, "local": tot[sec][0], "local_se": tot[sec][1], "note": note}
                for how in ("poolaware", "paircov"):
                    m = live[(live.how == how) & (live.arm == arm) & (live.section == sec)]
                    r[f"{how}"] = float(m["live"].iloc[0]) if len(m) else np.nan
                    r[f"{how}_se"] = float(m["live_se"].iloc[0]) if len(m) else np.nan
                    r[f"{how}_k"] = int(m["k_direct"].iloc[0]) if len(m) else 0
                rows.append(r)
    t = pd.DataFrame(rows)
    if verbose:
        print("\n== LOCAL (unweighted / joint-reweighted) vs LIVE (pool-aware FE / pair+covariate FE); SE = max(bootstrap, floor)")
        show = t.pivot_table(index=["arm", "section"], columns="mode", values="local").rename(columns={"none": "local", "joint": "local_rw"})
        first = t[t["mode"] == "none"].set_index(["arm", "section"])
        for c in ("local_se", "poolaware", "poolaware_se", "poolaware_k", "paircov", "paircov_se", "paircov_k"):
            show[c] = first[c]
        print(show.round(4).to_string())
        print("\nagreement statistics (arms with a live estimate; z uses live SE with floor and local SE):")
        for sec in ("alc", "b0", "b1", "b3"):
            for mode in ("none", "joint"):
                for how in ("poolaware", "paircov"):
                    g = t[(t.section == sec) & (t["mode"] == mode) & t[how].notna()]
                    if len(g) < 3:
                        continue
                    err = g["local"] - g[how]
                    z = err / np.sqrt(g[f"{how}_se"] ** 2 + g["local_se"] ** 2)
                    sign = int((np.sign(g["local"]) == np.sign(g[how])).sum())
                    wls = float(np.sum(g["local"] * g[how] / g[f"{how}_se"] ** 2) / np.sum(g["local"] ** 2 / g[f"{how}_se"] ** 2))
                    print(f"  {sec:4s} local={mode:5s} live={how:9s} n={len(g)} sign {sign}/{len(g)}  "
                          f"mean(local-live) {err.mean():+.4f}  chi2 {float((z ** 2).sum()):.1f} on {len(g)} dof  "
                          f"WLS slope live~local (no intercept) {wls:+.2f}")
    return t


# ----------------------------------------------------------------------------------------------------------------
# 6. reweighted local replay
# ----------------------------------------------------------------------------------------------------------------
B0_EDGES = [0.0, 0.08, 0.16, 0.24, 0.32, 10.0]
B31_EDGES = [0.0, 0.06, 0.12, 0.18, 10.0]


def live_global_dist(w: pd.DataFrame) -> dict:
    """GLOBAL distributions over distinct pairs: B31 pooled over every run (B31 is arm-invariant: FE ~ 0), B0 only
    from the reference arm (same model as the local reference). Returned as normalised histograms."""
    b31 = w.groupby("key")["b31"].mean()
    ref = w[w.arm == REF].groupby("key")[["b0", "b31"]].mean()
    h31 = np.histogram(b31, bins=B31_EDGES)[0]
    h0 = np.histogram(ref["b0"], bins=B0_EDGES)[0]
    h2 = np.histogram2d(ref["b0"], ref["b31"], bins=[B0_EDGES, B31_EDGES])[0]
    return {"b31_all": b31, "ref": ref, "h31": h31 / h31.sum(), "h0": h0 / h0.sum(), "h2": h2 / h2.sum(),
            "n31": int(b31.size), "n_ref": int(len(ref))}


def replay_weights(csv: str, ref_model: str, live: dict, mode: str) -> pd.Series:
    d = _csv(csv)
    r = d[d.model == ref_model].set_index(["bench", "k", "run", "pair"])
    if mode == "b31":
        lb = np.clip(np.digitize(r["b31"], B31_EDGES) - 1, 0, len(B31_EDGES) - 2)
        loc_h = np.bincount(lb, minlength=len(B31_EDGES) - 1) / len(lb)
        ratio = np.where(loc_h > 0, live["h31"] / np.maximum(loc_h, 1e-12), 0.0)
        wv = ratio[lb]
    elif mode == "b0":
        lb = np.clip(np.digitize(r["b0"], B0_EDGES) - 1, 0, len(B0_EDGES) - 2)
        loc_h = np.bincount(lb, minlength=len(B0_EDGES) - 1) / len(lb)
        ratio = np.where(loc_h > 0, live["h0"] / np.maximum(loc_h, 1e-12), 0.0)
        wv = ratio[lb]
    else:  # joint, coarse
        i0 = np.clip(np.digitize(r["b0"], B0_EDGES) - 1, 0, len(B0_EDGES) - 2)
        i1 = np.clip(np.digitize(r["b31"], B31_EDGES) - 1, 0, len(B31_EDGES) - 2)
        loc_h = np.zeros((len(B0_EDGES) - 1, len(B31_EDGES) - 1))
        np.add.at(loc_h, (i0, i1), 1.0)
        loc_h /= loc_h.sum()
        ratio = np.where(loc_h > 0, live["h2"] / np.maximum(loc_h, 1e-12), 0.0)
        wv = ratio[i0, i1]
    return pd.Series(wv, index=r.index)


def ess(wv: pd.Series) -> float:
    return float(wv.sum() ** 2 / (wv ** 2).sum()) if (wv > 0).any() else 0.0


def replay(w: pd.DataFrame, boot: dict, k_mix: float, verbose: bool = True) -> pd.DataFrame:
    live = live_global_dist(w)
    if verbose:
        print("\n== GLOBAL live distributions (distinct pairs; histograms only)")
        print(f"B31 over {live['n31']} pairs, bins {B31_EDGES[:-1]}+: {np.round(live['h31'], 3).tolist()}; "
              f"mean {live['b31_all'].mean():.4f}, quartiles {np.round(live['b31_all'].quantile([.25, .5, .75]).to_numpy(), 4).tolist()}")
        print(f"B0 under {REF} over {live['n_ref']} pairs, bins {B0_EDGES[:-1]}+: {np.round(live['h0'], 3).tolist()}; "
              f"mean {live['ref']['b0'].mean():.4f}")
        for csv, refm in (("results/paired_damped.csv", "n20D"), ("results/paired_mix.csv", "base")):
            d = _csv(csv)
            r = d[d.model == refm]
            h31 = np.histogram(r["b31"], bins=B31_EDGES)[0] / len(r)
            h0 = np.histogram(r["b0"], bins=B0_EDGES)[0] / len(r)
            print(f"local {csv.split('/')[-1]}:{refm}  B31 hist {np.round(h31, 3).tolist()} mean {r['b31'].mean():.4f} | "
                  f"B0 hist {np.round(h0, 3).tolist()} mean {r['b0'].mean():.4f} | n={len(r)}")
    rows = []
    modes = ("none", "b31", "b0", "joint")
    for arm, ref, chain, note in LOCAL_MAP:
        for mode in modes:
            tot = {s: [0.0, 0.0] for s in SECTIONS}
            ess_min = np.inf
            for csv, model, refm in chain:
                refmodel = "n20D" if "paired_damped" in csv else ("base" if "paired_mix" in csv else refm)
                wv = None
                if mode != "none":
                    if csv.endswith("paired_m0n15.csv"):
                        wv = None  # no m0=-2 reference model in this file; leave unweighted
                    else:
                        wv = replay_weights(csv, refmodel, live, mode)
                        ess_min = min(ess_min, ess(wv) / len(wv))
                e = local_effect(csv, model, refm, weights=wv, k_mix=k_mix)
                for s in SECTIONS:
                    tot[s][0] += e[s][0]
                    tot[s][1] = math.sqrt(tot[s][1] ** 2 + e[s][1] ** 2)
            for s in ("alc", "b0", "b1", "b3"):
                live_e = boot[s].loc[arm, "effect"] if arm in boot[s].index else np.nan
                live_se = boot[s].loc[arm, "se"] if arm in boot[s].index else np.nan
                rows.append({"arm": arm, "mode": mode, "section": s, "local": tot[s][0], "local_se": tot[s][1],
                             "live": live_e, "live_se": live_se, "ess_frac": (ess_min if mode != "none" else 1.0)})
    t = pd.DataFrame(rows)
    if verbose:
        piv = t.pivot_table(index=["arm", "section"], columns="mode", values="local").reindex(columns=list(modes))
        piv["live"] = t.drop_duplicates(["arm", "section"]).set_index(["arm", "section"])["live"]
        piv["live_se"] = t.drop_duplicates(["arm", "section"]).set_index(["arm", "section"])["live_se"]
        print("\n== REWEIGHTED LOCAL EFFECTS vs LIVE FE (local effect under each weighting)")
        print(piv.round(4).to_string())
        print("\nESS fraction of the weights per mode (min over files in the chain):",
              t.groupby("mode")["ess_frac"].min().round(3).to_dict())
        print("\nAgreement by weighting mode (arms with live estimates):")
        for s in ("alc", "b0", "b1", "b3"):
            for mode in modes:
                g = t[(t.section == s) & (t["mode"] == mode) & t.live.notna()]
                if len(g) < 3:
                    continue
                err = g["local"] - g["live"]
                z = err / np.sqrt(g["live_se"] ** 2 + g["local_se"] ** 2)
                sign = int((np.sign(g.local) == np.sign(g.live)).sum())
                r = g[["local", "live"]].corr().iloc[0, 1]
                print(f"  {s:4s} {mode:6s} n={len(g)} sign {sign}/{len(g)}  MAE {err.abs().mean():.4f}  "
                      f"mean(local-live) {err.mean():+.4f}  chi2/n {float((z ** 2).mean()):.2f}  Pearson {r:+.2f}")
    return t


# ----------------------------------------------------------------------------------------------------------------
def main() -> None:
    subset = _pop_holdout_switch()  # None unless --select / --holdout was given (round 3, additive)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    w = live_wide()
    w = w[w["arm"].notna()].copy()
    if subset is not None:
        held = holdout_mask(w)
        w = (w[~held] if subset == "select" else w[held]).copy()
        print(f"[{subset} set: {w['key'].nunique()} distinct pairs, {len(w)} appearances; holdout = "
              f"sha256('{HOLDOUT_SALT}|subject_<id>|benchmark_<id>')[:2] < '40']")
    pool = pool_stats(w, verbose=what in ("pool", "all", "power"))
    if what in ("pool", "all"):
        pool_ci(w, sims=int(sys.argv[2]) if (what == "pool" and len(sys.argv) > 2) else 300)
    ms = None
    if what in ("noise", "all", "power"):
        noise(w)
        ms = matched_spread(w, min_k=2)
        ancova_noise(w)
        drift_check(w)
    if what in ("fe", "all", "agree", "replay"):
        secs = ("alc", "b0", "b1", "b3", "b7", "b15", "b31")
        boot = fe_bootstrap(w, secs, reps=int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
        sh = shared_counts(w)
        print(f"\n== EXACT FIXED EFFECTS vs {REF} (pair-cluster bootstrap)")
        tab = pd.DataFrame({s: boot[s]["effect"] for s in secs})
        tab["direct_shared"] = sh
        tab["runs"] = w.groupby("arm")["submission"].nunique()
        tab["alc_se"] = boot["alc"]["se"]
        tab["alc_lo90"] = boot["alc"]["lo90"]
        tab["alc_hi90"] = boot["alc"]["hi90"]
        for s in ("b0", "b1", "b3"):
            tab[f"{s}_se"] = boot[s]["se"]
        print(tab.round(4).to_string())
        for s in ("alc", "b1"):
            sd, dof = fe_resid_sd(w, s)
            print(f"FE residual SD ({s}) with correct dof: {sd:.4f} (dof {dof}); a 2-arm matched difference has "
                  f"SD ~ {math.sqrt(2) * sd:.4f}")
        jk = bench_jackknife(w, "alc")
        jk1 = bench_jackknife(w, "b1")
        tool_se = {}
        try:  # what tools/feedback.py fe prints as approx_se, for comparison with the bootstrap
            from feedback import fixed_effects, ledger_arms as _la, load as _ld
            arm_of = dict(sorted(_la().items(), key=lambda kv: kv[1] != REF))
            tool_se = fixed_effects(_ld(), arm_of, "alc", iters=60).set_index("arm")["approx_se"].to_dict()
        except Exception as exc:  # pragma: no cover
            print("tool approx_se unavailable:", exc)
        if tool_se:
            ratio = (boot["alc"]["se"] / pd.Series(tool_se)).replace([np.inf, -np.inf], np.nan).dropna()
            ratio = ratio[ratio.index != REF]
            print(f"bootstrap SE / tools/feedback.py approx_se (ALC): median {ratio.median():.2f}, "
                  f"range {ratio.min():.2f}-{ratio.max():.2f} over {len(ratio)} arms")
        print("\nbenchmark jackknife (ALC and B1; leave one benchmark out; benchmarks unnamed):")
        print(pd.concat([jk.add_prefix("alc_"), jk1[["effect", "jk_se"]].add_prefix("b1_")], axis=1).round(4).to_string())
        k_mix = float((w["cond"] == "alone").mean())
        if what in ("all", "agree"):
            agree2(w, k_mix)
        if what in ("all", "replay"):
            replay(w, boot, k_mix)
    if what in ("robust", "all"):
        robust_table(w, pool, reps=int(sys.argv[2]) if (what == "robust" and len(sys.argv) > 2) else 150)
    if what in ("power", "all"):
        power(w, pool, ms)


if __name__ == "__main__":
    main()
