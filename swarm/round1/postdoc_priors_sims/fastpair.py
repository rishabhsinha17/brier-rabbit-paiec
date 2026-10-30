"""Vectorized exact replica of HierModel.predict for ONE pair alone on its benchmark (no item_features group, no partner).

Reduction (exact for the joint-mode Laplace): the likelihood touches mu_b, phi_s, delta_sb only through their sum L, and
each labeled item carries one label, so at the mode all success items share one item effect e1 and all failure items
share e0. Newton on (L, e1, e0) with multiplicities k and n-k gives the same mode as HierModel's (n+3)-dim fit, and the
Schur complement gives Var(L) = 1 / (1/V + k w1/(1 + s2 w1) + (n-k) w0/(1 + s2 w0)), w = p(1-p) at the mode.
Prediction (ability_param False): sigmoid((m_L) / sqrt(1 + pi (V_L + sigma_i^2 + lam^2 var_theta) / 8)), where the prior
of L is N(m0 + lam theta, tau_mu^2 + tau_phi^2 + tau_delta^2); optional zero-inflated mixture exactly as hier.py.
Validated against HierModel.predict in validate() (max abs diff printed).
"""
from __future__ import annotations

import math

import numpy as np

_GX, _GW = np.polynomial.hermite_e.hermegauss(48)
_GLW = np.log(_GW / np.sqrt(2.0 * np.pi))
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
KN = [(k, n) for n in BUDGETS for k in range(n + 1)]
K_ARR = np.array([k for k, n in KN], float)
N_ARR = np.array([n for k, n in KN], float)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def laplace_level(k, n, m, V, s2, iters=60):
    """Posterior mean and variance of L (vectorized over broadcast arrays k, n, m, V)."""
    k, n, m, V = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float), np.asarray(m, float), np.asarray(V, float))
    L = m.copy()
    e1 = np.zeros_like(L)
    e0 = np.zeros_like(L)
    f = n - k

    def logpost(L, e1, e0):
        a, b = L + e1, L + e0
        return (-k * np.logaddexp(0.0, -a) - f * np.logaddexp(0.0, b) - 0.5 * (L - m) ** 2 / V
                - 0.5 * (k * e1 ** 2 + f * e0 ** 2) / s2)

    cur = logpost(L, e1, e0)
    for _ in range(iters):
        p1, p0 = _sig(L + e1), _sig(L + e0)
        w1, w0 = p1 * (1 - p1), p0 * (1 - p0)
        gL = k * (1 - p1) - f * p0 - (L - m) / V
        g1 = (1 - p1) - e1 / s2  # per success item
        g0 = -p0 - e0 / s2  # per failure item
        # Newton step on the full symmetric system, reduced: item blocks are diagonal (one item each)
        d1 = w1 + 1 / s2
        d0 = w0 + 1 / s2
        A = k * w1 + f * w0 + 1 / V
        # eliminate items: (A - k w1^2/d1 - f w0^2/d0) dL = gL - k w1 g1/d1 - f w0 g0/d0
        S = A - k * w1 ** 2 / d1 - f * w0 ** 2 / d0
        dL = (gL - k * w1 * g1 / d1 - f * w0 * g0 / d0) / S
        de1 = (g1 - w1 * dL) / d1
        de0 = (g0 - w0 * dL) / d0
        # damped (step halving) like HierModel(damped_newton=True): never lower the log posterior
        scale = np.ones_like(L)
        for _h in range(30):
            cand = logpost(L + scale * dL, e1 + scale * de1, e0 + scale * de0)
            bad = cand < cur - 1e-10
            if not np.any(bad):
                break
            scale = np.where(bad, scale * 0.5, scale)
        L, e1, e0 = L + scale * dL, e1 + scale * de1, e0 + scale * de0
        cur = logpost(L, e1, e0)
        if np.max(np.abs(scale * dL)) < 1e-10:
            break
    p1, p0 = _sig(L + e1), _sig(L + e0)
    w1, w0 = p1 * (1 - p1), p0 * (1 - p0)
    VL = 1.0 / (1.0 / V + k * w1 / (1 + s2 * w1) + f * w0 / (1 + s2 * w0))
    return L, VL


def log_marginal(k, n, m, v):
    """Vectorized hier._log_marginal: log P(k of n | logit level ~ N(m, v)) (no binomial coefficient)."""
    k, n, m, v = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float), np.asarray(m, float), np.asarray(v, float))
    eta = m[..., None] + np.sqrt(np.maximum(v, 1e-9))[..., None] * _GX
    ll = -k[..., None] * np.logaddexp(0.0, -eta) - (n - k)[..., None] * np.logaddexp(0.0, eta) + _GLW
    top = ll.max(axis=-1, keepdims=True)
    return top[..., 0] + np.log(np.exp(ll - top).sum(axis=-1))


def post_pred(k, n, m, v, item_var):
    """Vectorized hier._posterior_predictive."""
    k, n, m, v = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float), np.asarray(m, float), np.asarray(v, float))
    eta = m[..., None] + np.sqrt(np.maximum(v, 1e-9))[..., None] * _GX
    ll = -k[..., None] * np.logaddexp(0.0, -eta) - (n - k)[..., None] * np.logaddexp(0.0, eta) + _GLW
    w = np.exp(ll - ll.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return (w / (1.0 + np.exp(-eta / math.sqrt(1.0 + math.pi * item_var / 8.0)))).sum(axis=-1)


_LN_CACHE = {}
_MGRID = np.arange(-14.0, 8.0 + 1e-9, 0.02)


def _ln_interp(m, v):
    """log_marginal(K, N, m, v) for arrays m (P,1), v (P,1) via per-v tables on an m-grid (linear interpolation;
    error < 1e-3 in log-likelihood units for n <= 31). Exact path when the grid is not applicable."""
    m = np.asarray(m, float).reshape(-1)
    v = np.asarray(v, float).reshape(-1)
    out = np.empty((len(m), len(KN)))
    for vv in np.unique(np.round(v, 9)):
        key = float(vv)
        if key not in _LN_CACHE:
            _LN_CACHE[key] = log_marginal(K_ARR[None, :], N_ARR[None, :], _MGRID[:, None], key)
        T = _LN_CACHE[key]
        sel = np.where(np.round(v, 9) == vv)[0]
        pos = np.clip((m[sel] - _MGRID[0]) / 0.02, 0, len(_MGRID) - 1.000001)
        i0 = np.floor(pos).astype(int)
        fr = (pos - i0)[:, None]
        out[sel] = T[i0] * (1 - fr) + T[i0 + 1] * fr
    return out


def predict_table(p, theta, th_var, pi_override=None, exact=False):
    """q(k, n) for every (k, n) in KN; theta/th_var arrays of shape (P,) -> output (P, len(KN)).
    p: params dict (HierModel names). pi_override: optional (P,) array of per-pair mixture weights."""
    theta = np.asarray(theta, float)[:, None]
    th_var = np.asarray(th_var, float)[:, None]
    lam = p["lam"]
    # extra_V: optional fixed variance confounded with the level (e.g. sigma_g^2 of a single item_features group)
    V = p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2 + p.get("extra_V", 0.0)
    m = p["m0"] + lam * theta
    L, VL = laplace_level(K_ARR[None, :], N_ARR[None, :], m, V, p["sigma_i"] ** 2)
    var = VL + p["sigma_i"] ** 2 + lam ** 2 * th_var
    q = _sig(L / np.sqrt(1.0 + math.pi * var / 8.0))
    pi = p.get("mix_pi", 0.0) if pi_override is None else np.asarray(pi_override, float)[:, None]
    if np.any(np.asarray(pi) > 0):
        pi = np.clip(np.broadcast_to(pi, q.shape), 1e-9, 1 - 1e-9)
        m_norm = m
        v_norm = V + lam ** 2 * th_var
        lh = log_marginal(K_ARR[None, :], N_ARR[None, :], p["mix_m"], p["mix_s"] ** 2)
        if exact:
            ln = log_marginal(K_ARR[None, :], N_ARR[None, :], m_norm, v_norm)
        else:
            ln = _ln_interp(np.broadcast_to(m_norm, (q.shape[0], 1)), np.broadcast_to(v_norm, (q.shape[0], 1)))
        w_hard = 1.0 / (1.0 + np.exp(np.clip(np.log((1 - pi) / pi) + ln - lh, -700, 700)))
        w_hard = np.where(N_ARR[None, :] > 0, w_hard, pi)
        p_hard = post_pred(K_ARR, N_ARR, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)[None, :]
        q = w_hard * p_hard + (1 - w_hard) * q
    return np.clip(q, p.get("clip", 0.005), 1 - p.get("clip", 0.005))


def expected_scores(q, r):
    """q (P, len(KN)), r (P,) true pair rates -> dict budget -> (P,) expected Brier; plus 'alc'."""
    from scipy.stats import binom
    r = np.asarray(r, float)
    out = {}
    for n in BUDGETS:
        idx = [i for i, (k, nn) in enumerate(KN) if nn == n]
        ks = np.arange(n + 1)
        pk = binom.pmf(ks[None, :], n, r[:, None])
        qq = q[:, idx]
        out[n] = np.sum(pk * ((qq - r[:, None]) ** 2 + (r * (1 - r))[:, None]), axis=1)
    out["alc"] = sum(WEIGHTS[n] * out[n] for n in BUDGETS)
    return out


def validate(params_list, n_cases=40, seed=0):
    """Compare predict_table with HierModel.predict on synthetic single pairs; returns max abs difference."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from paiec.hier import HierModel
    rng = np.random.default_rng(seed)
    worst = 0.0
    subj = {"normalized_name": "probe-x"}
    for params in params_list:
        for _ in range(n_cases):
            th, tv = float(rng.normal(0, 0.6)), float(rng.uniform(0.1, 0.6))
            mdl = HierModel({}, params)
            mdl._theta = lambda s, th=th, tv=tv: (th, tv)
            n = int(rng.choice(BUDGETS))
            k = int(rng.integers(0, n + 1))
            it = lambda j: {"item_content": f"i{j}", "item_features": "", "interactors": "", "benchmark_id": "bx"}
            lab = [[[subj, it(j)], int(j < k)] for j in range(n)]
            ref = mdl.predict([subj, it(999)], lab)
            q = predict_table(dict(mdl.p), [th], [tv])[0, KN.index((k, n))]
            worst = max(worst, abs(ref - q))
    return worst


def predict_table_exact(p, theta, th_var, pi_override=None, grid=241):
    """Exact 1-D quadrature posterior predictive for an alone pair without groups (head_researcher_B's ExactAloneHier,
    vectorized): prior L ~ N(m0 + lam theta, tau_mu^2 + tau_phi^2 + tau_delta^2 + lam^2 var_theta) [optionally mixed
    with N(mix_m, mix_s^2) at weight pi], per-label likelihood sigmoid(c L), c = 1/sqrt(1 + pi sigma_i^2 / 8), new-item
    prediction E[sigmoid(L / sqrt(1 + pi sigma_i^2 / 8))]."""
    theta = np.asarray(theta, float)[:, None]
    th_var = np.asarray(th_var, float)[:, None]
    m = p["m0"] + p["lam"] * theta  # (P,1)
    V = p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2 + p["lam"] ** 2 * th_var + p.get("extra_V", 0.0)
    sd = np.sqrt(V)
    lo = np.minimum(m - 7 * sd, p.get("mix_m", -4.0) - 7 * p.get("mix_s", 0.7))
    hi = m + 7 * sd
    u = np.linspace(0.0, 1.0, grid)[None, :]
    L = lo + (hi - lo) * u  # (P, G)
    logprior = -0.5 * (L - m) ** 2 / V - 0.5 * np.log(V)
    pi = p.get("mix_pi", 0.0) if pi_override is None else np.asarray(pi_override, float)[:, None]
    if np.any(np.asarray(pi) > 0):
        pi = np.clip(np.broadcast_to(pi, (L.shape[0], 1)), 1e-9, 1 - 1e-9)
        lh = -0.5 * (L - p["mix_m"]) ** 2 / p["mix_s"] ** 2 - math.log(p["mix_s"])
        logprior = np.logaddexp(np.log(1 - pi) + logprior, np.log(pi) + lh)
    c = 1.0 / math.sqrt(1.0 + math.pi * p["sigma_i"] ** 2 / 8.0)
    lp1 = -np.logaddexp(0.0, -c * L)  # log sigmoid(cL)
    lp0 = -np.logaddexp(0.0, c * L)
    newp = _sig(L / math.sqrt(1.0 + math.pi * p["sigma_i"] ** 2 / 8.0))
    out = np.empty((L.shape[0], len(KN)))
    for j, (k, n) in enumerate(KN):
        lp = logprior + k * lp1 + (n - k) * lp0
        w = np.exp(lp - lp.max(axis=1, keepdims=True))
        out[:, j] = (w * newp).sum(axis=1) / w.sum(axis=1)
    return np.clip(out, p.get("clip", 0.005), 1 - p.get("clip", 0.005))
