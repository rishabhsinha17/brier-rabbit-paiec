"""Fast alone-pair tables for candidate level priors (round 8, headroom designer).

Our model's Laplace level fit depends on a pair only through (m = m0 + lam*theta_eff, V = tau_mu^2 + tau_phi^2 +
tau_delta^2 [+ extra_V for the item-group bracket]); theta's own variance tv enters only the final moderation. So (L, VL)
are tabulated once per V on an m-grid and interpolated (checked against fastpair.predict_table in hr_check1.py).
Exact inference (ExactAloneHier / fastpair.predict_table_exact) depends on (m, V + lam^2 tv): tabulated per V for the 7
distinct tv values of the public pairs. HierModel's mixture weight needs log_marginal(k, n, m, V + lam^2 tv): same.

Candidate spec (dict): m0, V (pair-level prior variance without the group bracket), lam (1), sigma_i (0.6),
  mix_pi / mix_m / mix_s (HierModel zero-inflated component; 0 = off), exact (False: Laplace + probit as HierModel;
  True: ExactAloneHier replica, the mixture then enters the prior exactly).
"""
from __future__ import annotations

import math

import numpy as np

import hr_common as hc  # noqa: I001  (sets sys.path)
import fastpair as fp

MG = np.round(np.arange(-9.0, 7.0 + 1e-9, 0.025), 4)
K, N = hc.K_ARR, hc.N_ARR
_LAP, _EX, _LN = {}, {}, {}


def _key(*a):
    return tuple(round(float(x), 6) for x in a)


def laplace_grid(V, s2=0.36):
    k = _key(V, s2)
    if k not in _LAP:
        L, VL = fp.laplace_level(K[None, :], N[None, :], MG[:, None], V, s2)
        _LAP[k] = (L, VL)
    return _LAP[k]


def _interp_rows(T, m):
    """T (G, 63) tabulated on MG; m (R,) -> (R, 63) linear interpolation."""
    pos = np.clip((m - MG[0]) / (MG[1] - MG[0]), 0, len(MG) - 1.000001)
    i0 = np.floor(pos).astype(int)
    f = (pos - i0)[:, None]
    return T[i0] * (1 - f) + T[i0 + 1] * f


def tv_values():
    s = hc.setup()
    return np.unique(np.round(s["tv"], 9))


def exact_grid(V_tot, spec):
    """predict_table_exact on the m-grid for prior variance V_tot (incl. lam^2 tv and bracket), optional exact mixture."""
    mix = (spec.get("mix_pi", 0.0) or 0.0)
    k = _key(V_tot, spec.get("sigma_i", 0.6), mix, spec.get("mix_m", -4.0), spec.get("mix_s", 0.7))
    if k not in _EX:
        p = {"m0": 0.0, "lam": 1.0, "tau_mu": 0.0, "tau_phi": 0.0, "tau_delta": 0.0, "extra_V": V_tot,
             "sigma_i": spec.get("sigma_i", 0.6), "clip": 0.0}
        if mix > 0:
            p.update(mix_pi=mix, mix_m=spec["mix_m"], mix_s=spec["mix_s"])
        _EX[k] = fp.predict_table_exact(p, MG, np.zeros_like(MG))
    return _EX[k]


def lnmarg_grid(v):
    k = _key(v)
    if k not in _LN:
        _LN[k] = fp.log_marginal(K[None, :], N[None, :], MG[:, None], v)
    return _LN[k]


def table(spec, d, br, th, tv):
    """(P, 63) q(k, n) of a candidate for true offset d and bracket br (0 none / 1 one level-confounded group)."""
    lam = spec.get("lam", 1.0)
    s2 = spec.get("sigma_i", 0.6) ** 2
    V = spec["V"] + float(br)
    m = spec["m0"] + lam * (th + d)
    clip = spec.get("clip", 0.005)
    if spec.get("exact", False):
        out = np.empty((len(th), len(hc.KN)))
        for u in np.unique(np.round(tv, 9)):
            sel = np.round(tv, 9) == u
            T = exact_grid(V + lam ** 2 * u, spec)
            out[sel] = _interp_rows(T, m[sel])
        return np.clip(out, clip, 1 - clip)
    L, VL = laplace_grid(V, s2)
    Lr, VLr = _interp_rows(L, m), _interp_rows(VL, m)
    var = VLr + s2 + lam ** 2 * tv[:, None]
    q = 1.0 / (1.0 + np.exp(-np.clip(Lr / np.sqrt(1.0 + math.pi * var / 8.0), -35, 35)))
    pi = spec.get("mix_pi", 0.0) or 0.0
    if pi > 0:
        pi = min(max(pi, 1e-9), 1 - 1e-9)
        ln = np.empty_like(q)
        for u in np.unique(np.round(tv, 9)):
            sel = np.round(tv, 9) == u
            ln[sel] = _interp_rows(lnmarg_grid(spec["V"] + lam ** 2 * u), m[sel])   # hier.py: no group variance
        lh = fp.log_marginal(K, N, spec["mix_m"], spec["mix_s"] ** 2)[None, :]
        w_hard = 1.0 / (1.0 + np.exp(np.clip(math.log((1 - pi) / pi) + ln - lh, -700, 700)))
        w_hard = np.where(N[None, :] > 0, w_hard, pi)
        p_hard = fp.post_pred(K, N, spec["mix_m"], spec["mix_s"] ** 2, s2)[None, :]
        q = w_hard * p_hard + (1 - w_hard) * q
    return np.clip(q, clip, 1 - clip)


class CellEval:
    """Expected Brier of candidates in the kept posterior cells (pool i, true offset d)."""

    def __init__(self, mass=0.9995, th=None, tv=None, w=None):
        s = hc.setup()
        self.th = s["th"] if th is None else th
        self.tv = s["tv"] if tv is None else tv
        w = hc.cell_weights() if w is None else w
        flat = np.sort(w.ravel())[::-1]
        thr = flat[min(np.searchsorted(np.cumsum(flat), mass), len(flat) - 1)]
        self.keep = w >= thr
        self.cells = np.argwhere(self.keep)                 # (C, 2): pool, d index
        wc = w[self.keep]
        self.wc = wc / wc.sum()
        self.ds = sorted(set(self.cells[:, 1]))
        self.R, self.PMF, self.wts = s["R"], s["PMF"], s["wts"]
        self.const = (self.wts[None, :] * self.R).sum(1)

    def eval_tables(self, tabfun):
        """tabfun(d, br) -> (P, 63). Returns E (C, 6) bracket-mean expected Brier per kept cell."""
        E = np.zeros((len(self.cells), 6))
        for jd in self.ds:
            d = hc.DS[jd]
            ci = np.flatnonzero(self.cells[:, 1] == jd)
            pools = self.cells[ci, 0]
            acc = np.zeros((len(ci), 63))
            A1 = self.wts[None, :, None] * self.PMF[pools]              # (c, P, 63)
            A2 = A1 * self.R[pools][:, :, None]
            for br in (0, 1):
                q = tabfun(d, br)                                       # (P, 63)
                acc += 0.5 * (np.einsum("pk,cpk->ck", q * q, A1) - 2.0 * np.einsum("pk,cpk->ck", q, A2))
            for j, n in enumerate(hc.BUD):
                E[ci, j] = acc[:, fp_idx(n)].sum(1) + self.const[pools]
        return E

    def eval_spec(self, spec):
        return self.eval_tables(lambda d, br: table(spec, d, br, self.th, self.tv))

    def mean(self, E):
        return self.wc @ E

    def alc(self, E):
        return E @ hc.WB


_FPIDX = {}


def fp_idx(n):
    if n not in _FPIDX:
        _FPIDX[n] = np.array([i for i, (k, nn) in enumerate(hc.KN) if nn == n])
    return _FPIDX[n]


def W20_SPEC():
    return {"m0": -2.0, "V": 2.0 ** 2 + 0.5 ** 2 + 1.0 ** 2}


def REF_SPEC():
    return {"m0": -2.0, "V": 1.2 ** 2 + 0.5 ** 2 + 1.0 ** 2}


# ---- disk cache of the tabulated grids (shared across processes) ------------------------------------------------
_DISK = hc.CACHE / "fast_grids.pkl"


def load_disk():
    import pickle
    if _DISK.exists():
        try:
            d = pickle.loads(_DISK.read_bytes())
            for k, v in d.get("lap", {}).items():
                _LAP.setdefault(k, v)
            for k, v in d.get("ex", {}).items():
                _EX.setdefault(k, v)
            for k, v in d.get("ln", {}).items():
                _LN.setdefault(k, v)
        except Exception:
            pass


def save_disk():
    import pickle
    load_disk()          # merge with what other runs saved
    _DISK.write_bytes(pickle.dumps({"lap": _LAP, "ex": _EX, "ln": _LN}))


load_disk()
