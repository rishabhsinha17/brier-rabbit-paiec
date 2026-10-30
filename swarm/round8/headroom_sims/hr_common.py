"""Shared machinery for the round-8 headroom study (Bayes-optimal headroom of the level block).

Everything is the round-2/round-4 exact alone-pair calculus on the 220 public eligible pairs (LOBO subject priors,
benchmark-equal), under the round-4 design posterior over pool regimes x subject offset d x discrepancy scale, refit on
the CURRENT selection-set evidence (hr_live.py -> cache/live_hr.npz). No live data enters the tables; live data enter
only the posterior weights over regime cells (and the GP discrepancy of family arms, via ld_post).

Conventions
- tables: q(k, n) arrays of shape (P=220, 63) in fastpair.KN order, one per (true d, bracket).
- E[pool, d, bracket, budget]: benchmark-equal expected Brier of a predictor in regime cell (pool, d).
- the hidden-subject map is theta_eff = theta + d (slope 1), as in the design.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "swarm" / "round4" / "design_sims", ROOT / "swarm" / "round2" / "adjudicator_sims",
          ROOT / "swarm" / "round1" / "postdoc_priors_sims", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import adj_calc as ac  # noqa: E402
import fastpair as fp  # noqa: E402
import fastregime as fr  # noqa: E402
import ld_calc as lc  # noqa: E402
import ld_post as lp  # noqa: E402
import regime_test as rt  # noqa: E402

CACHE = HERE / "cache"
LIVE_NPZ = CACHE / "live_hr.npz"
BUD = fp.BUDGETS
WB = np.array([fp.WEIGHTS[n] for n in BUD])
KN = fp.KN
K_ARR, N_ARR = fp.K_ARR, fp.N_ARR
DS = lc.DS
REFP = dict(rt.LIVE)          # REF params (v3b_noM_n_m0n20)
W20P = dict(REFP, tau_mu=2.0)

_S = {}


def setup():
    """el, theta, var_theta, bench weights, per-pool rates and pmf tables (cached in memory)."""
    if _S:
        return _S
    S = rt.setup()
    el, th, tv = S["el"], np.asarray(S["th"], float), np.asarray(S["tv"], float)
    wts = fr.bench_weights(el)
    pools = lc.pools()
    R = np.stack([fr.transform(el, pl) for pl in pools])            # (nPool, P)
    PMF = np.stack([fr.pmf_table(r) for r in R])                     # (nPool, P, 63)
    _S.update(el=el, th=th, tv=tv, wts=wts, pools=pools, R=R, PMF=PMF, S=S)
    return _S


def evaluate(Q):
    """Q: (..., P, 63) tables -> E (nPool, ..., 6): bench-equal expected Brier at each budget for every pool.
    E Brier = sum_k pmf (q - r)^2 + r(1 - r)."""
    s = setup()
    wts, R, PMF = s["wts"], s["R"], s["PMF"]
    Q = np.asarray(Q, float)
    lead = Q.shape[:-2]
    Qf = Q.reshape(-1, Q.shape[-2], Q.shape[-1])                      # (T, P, 63)
    A1 = wts[None, :, None] * PMF                                      # (nPool, P, 63)
    A2 = A1 * R[:, :, None]
    G = np.einsum("tpk,ipk->itk", Qf * Qf, A1) - 2.0 * np.einsum("tpk,ipk->itk", Qf, A2)
    const = (wts[None, :] * (R * R + R * (1 - R))).sum(1)             # (nPool,)
    out = np.zeros((len(R), Qf.shape[0], len(BUD)))
    for j, n in enumerate(BUD):
        out[:, :, j] = G[:, :, fr.IDX[n]].sum(-1) + const[:, None]
    return out.reshape((len(R),) + lead + (len(BUD),))


def alc(E):
    return (E * WB).sum(-1)


# ---- predictor tables ------------------------------------------------------------------------------------------
def family_tables(params, th=None, tv=None, exact=False):
    """Our model's alone-pair tables for every true d and bracket: (nD, 2, P, 63).
    params: HierModel params (normal prior, optional HierModel-style mixture). exact=True: ExactAloneHier replica
    (fastpair.predict_table_exact; mixture handled exactly)."""
    s = setup()
    th = s["th"] if th is None else th
    tv = s["tv"] if tv is None else tv
    P = len(th)
    thd = np.concatenate([th + d for d in DS])
    tvd = np.tile(tv, len(DS))
    out = np.zeros((len(DS), 2, P, len(KN)))
    for br in (0, 1):
        p = dict(params, extra_V=float(br))
        q = fp.predict_table_exact(p, thd, tvd) if exact else predict_table_hier(p, thd, tvd)
        out[:, br] = q.reshape(len(DS), P, len(KN))
    return out


def predict_table_hier(p, theta, th_var):
    """fastpair.predict_table, but with the mixture weight computed EXACTLY as paiec/hier.py does in bracket 1:
    hier.py's v_norm excludes the item_features group variance (extra_V), while fastpair adds it. Identical when
    mix_pi = 0 or extra_V = 0."""
    if not p.get("mix_pi", 0.0) or not p.get("extra_V", 0.0):
        return fp.predict_table(p, theta, th_var)
    theta = np.asarray(theta, float)[:, None]
    th_var = np.asarray(th_var, float)[:, None]
    lam = p["lam"]
    # normal part, unclipped (group variance included), exactly as fastpair.predict_table computes it
    V = p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2 + p.get("extra_V", 0.0)
    L, VL = fp.laplace_level(K_ARR[None, :], N_ARR[None, :], p["m0"] + lam * theta, V, p["sigma_i"] ** 2)
    q = fp._sig(L / np.sqrt(1.0 + math.pi * (VL + p["sigma_i"] ** 2 + lam ** 2 * th_var) / 8.0))
    pi = np.clip(p["mix_pi"], 1e-9, 1 - 1e-9)
    m_norm = p["m0"] + lam * theta
    v_norm = p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2 + lam ** 2 * th_var
    lh = fp.log_marginal(K_ARR[None, :], N_ARR[None, :], p["mix_m"], p["mix_s"] ** 2)
    ln = fp._ln_interp(np.broadcast_to(m_norm, (q.shape[0], 1)), np.broadcast_to(v_norm, (q.shape[0], 1)))
    w_hard = 1.0 / (1.0 + np.exp(np.clip(math.log((1 - pi) / pi) + ln - lh, -700, 700)))
    w_hard = np.where(N_ARR[None, :] > 0, w_hard, pi)
    p_hard = fp.post_pred(K_ARR, N_ARR, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)[None, :]
    q = w_hard * p_hard + (1 - w_hard) * q          # hier.py clips once, at the end
    return np.clip(q, p.get("clip", 0.005), 1 - p.get("clip", 0.005))


# ---- posterior -------------------------------------------------------------------------------------------------
_P = {}


def posterior(**kw):
    """Round-4 design posterior refit on cache/live_hr.npz (new family arms registered from the ledger)."""
    key = tuple(sorted(kw.items()))
    if key in _P:
        return _P[key]
    import hr_live
    hr_live.register_family()
    P = lp.Posterior(LIVE_NPZ, **kw).fit()
    _P[key] = P
    return P


def cell_weights(P=None, **kw):
    """(nPool, nD) posterior weights (summed over the discrepancy scale). With P None: cached on disk per evidence
    file and posterior options (cache/W_<opts>.npz), refit when live_hr.npz is newer."""
    if P is not None:
        return P.W.sum(2)
    tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items())) or "base"
    f = CACHE / f"W_{tag}.npz"
    if f.exists() and f.stat().st_mtime > LIVE_NPZ.stat().st_mtime:
        return np.load(f)["W"].sum(2)
    P = posterior(**kw)
    np.savez_compressed(f, W=P.W)
    return P.W.sum(2)


def wmean(x, w):
    """Posterior mean over cells of x[pool, d, ...] with weights w[pool, d]."""
    return np.tensordot(w, x, axes=([0, 1], [0, 1]))


def wsd(x, w):
    m = wmean(x, w)
    return np.sqrt(np.maximum(wmean(x * x, w) - m * m, 0.0))


def thetas_variant(spec: dict, tag: str):
    """(theta, var_theta) of the 220 public pairs under a subject-prior variant (e.g. SubjCalHier attr_scale 1.56),
    LOBO priors exactly as the calculus (fast_eval.thetas); cached in cache/theta_<tag>.npz."""
    f = CACHE / f"theta_{tag}.npz"
    if f.exists():
        z = np.load(f)
        return z["th"], z["tv"]
    import fast_eval as fe
    el, priors = fe.public_pairs()
    th, tv, _ = fe.thetas(spec, el, priors)
    s = setup()
    assert len(th) == len(s["th"])
    np.savez_compressed(f, th=th, tv=tv)
    return th, tv


S_SPEC = {"class": "paiec.ext.pd1_subjcal:SubjCalHier", "attr_scale": 1.56, "attr_year_max": 2026.4}


def thetas_S():
    return thetas_variant(S_SPEC, "S_a156")
