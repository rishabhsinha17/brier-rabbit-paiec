"""Similarity-weighted cross-pair pooling (postdoc_adaptation, swarm round 1).

Public-data evidence (swarm/round1/postdoc_adaptation_sims/subject_sharing.py):
  - Same base model (canonical name) on two DIFFERENT benchmarks: the pair-level residuals after benchmark mean and the
    public ability prior (theta fit on the remaining benchmarks) correlate at r = +0.49 (72 names over the 6 benchmark
    pairs; Fisher-z SE 0.14). HierModel keys phi_s by the full subject dict, and in the hidden pool no subject ID ever
    spans two benchmarks, so this signal is never used even when the same base model appears twice in a draw.
  - Same harness string, different model, SAME benchmark (public Multi-SWE-bench, 12 harnesses): residual r = +0.82
    (401 pair-pairs), versus -0.13 for pairs differing in both. Same model with a different harness: -0.05.
Change (variance-preserving, so B0 and every pair without a coincidence are unchanged):
  logit p = mu_b + lam*theta_s + nu_name(s) + h_{b,harness(s)} + delta_sb + gamma_bg + eps_bi
  nu_name ~ N(0, tau_name^2) keyed by the canonical name (subjects.name_key; falls back to the full subject dict),
  h_{b,harness} ~ N(0, tau_h^2) keyed by (benchmark, harness string), only for a non-empty harness,
  delta_sb ~ N(0, tau_delta^2 + tau_phi^2 - tau_name^2 - [tau_h^2 if harness]) (floored at 0.05^2),
so the total prior variance of every pair level equals HierModel's tau_phi^2 + tau_delta^2. Two pairs pool through
nu_name when their base model matches (on any benchmark) and through h when they share a harness on a benchmark.
Defaults tau_name = 0.75, tau_h = 0.6 (from the public correlations: 0.49 * 1.15 ~ 0.56 = 0.75^2). Requires tau_M = 0,
nu_mu = 0 (normal level prior) and uses the damped Newton solver; the zero-inflated mixture is applied as in HierModel.
Numpy only; never raises.
"""
from __future__ import annotations

import math

import numpy as np

from paiec.hier import HierModel, _entries, _groups, _item_key, _log_marginal, _posterior_predictive, _sk

try:
    from paiec.subjects import name_key
except ImportError:  # pragma: no cover
    name_key = None


class SimilarityPoolModel(HierModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self.p.setdefault("tau_name", 0.75)
        self.p.setdefault("tau_h", 0.6)
        self.p.setdefault("pool_name", True)
        self.p.setdefault("pool_harness", True)

    # ----- structure -----------------------------------------------------------------------------------
    def _keys(self, subject, b):
        """(name key, harness key or None, pair-specific delta variance)."""
        p = self.p
        s = _sk(subject)
        nk = None
        if p.get("pool_name", True) and name_key is not None:
            try:
                nk = name_key(subject) or None
            except Exception:
                nk = None
        harness = str(subject.get("harness", "") or "").strip() if p.get("pool_harness", True) else ""
        hk = ("hb", b, harness) if harness else None
        total = p["tau_phi"] ** 2 + p["tau_delta"] ** 2
        tn = min(p["tau_name"] ** 2, total - 0.0025)
        th2 = min(p["tau_h"] ** 2, max(total - tn - 0.0025, 0.0)) if hk else 0.0
        dvar = max(total - tn - th2, 0.0025)
        return ("nm", nk if nk else "s:" + s), tn, hk, th2, ("delta", s, b), dvar

    def _cols(self, subject, item, b):
        p = self.p
        nm, tn, hk, th2, dk, dvar = self._keys(subject, b)
        th, th_var = self._theta(subject)
        if p["ability_param"]:
            nm_prior = (p["lam"] * th, tn + p["lam"] ** 2 * th_var)
            off, off_var = 0.0, 0.0
        else:
            nm_prior = (0.0, tn)
            off, off_var = p["lam"] * th, p["lam"] ** 2 * th_var
        cols = [(("mu", b), p["m0"], p["tau_mu"] ** 2), (nm, *nm_prior), (dk, 0.0, dvar)]
        if hk:
            cols.append((hk, 0.0, th2))
        for g in _groups(item, p.get("all_feature_tokens", False)):
            cols.append((("gamma", b, g), 0.0, p["sigma_g"] ** 2))
        cols.append((("eps", b, _item_key(item)), 0.0, p["sigma_i"] ** 2))
        return cols, off, off_var

    # ----- fit (damped Newton on the exact log posterior, normal priors) --------------------------------
    def _fit_posterior(self, labeled):
        p = self.p
        index, pm, pv = {}, [], []
        rows, ys, offs = [], [], []
        counts = {}
        for subject, item, label in _entries(labeled):
            b = str(item.get("benchmark_id", ""))
            cols, off, _ = self._cols(subject, item, b)
            r = []
            for name, m, v in cols:
                if name not in index:
                    index[name] = len(pm)
                    pm.append(m)
                    pv.append(v)
                r.append(index[name])
            rows.append(r)
            ys.append(label)
            offs.append(off)
            key = (_sk(subject), b)
            kk, nn = counts.get(key, (0, 0))
            counts[key] = (kk + int(label), nn + 1)
        n = len(pm)
        if n == 0:
            return {"index": {}, "mean": np.zeros(0), "cov": np.zeros((0, 0)), "counts": {}}
        m0 = np.asarray(pm)
        prec0 = 1.0 / np.asarray(pv)
        X = np.zeros((len(rows), n))
        for i, r in enumerate(rows):
            X[i, r] = 1.0
        y = np.asarray(ys)
        off = np.asarray(offs)

        def log_post(v):
            e = np.clip(X @ v + off, -35.0, 35.0)
            return float(np.sum(y * e - np.logaddexp(0.0, e))) - 0.5 * float(np.sum(prec0 * (v - m0) ** 2))

        w = m0.copy()
        current = log_post(w)
        for _ in range(int(p.get("max_iters", 60))):
            eta = np.clip(X @ w + off, -35.0, 35.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            grad = X.T @ (y - mu) - prec0 * (w - m0)
            H = (X.T * (mu * (1 - mu))) @ X + np.diag(prec0)
            step = np.linalg.solve(H, grad)
            scale = 1.0
            while True:
                cand = w + scale * step
                value = log_post(cand)
                if value >= current - 1e-10 or scale < 1e-4:
                    break
                scale *= 0.5
            moved = float(np.max(np.abs(cand - w)))
            w, current = cand, value
            if moved < 1e-7:
                break
        eta = np.clip(X @ w + off, -35.0, 35.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        H = (X.T * (mu * (1 - mu))) @ X + np.diag(prec0)
        return {"index": index, "mean": w, "cov": np.linalg.inv(H), "counts": counts}

    # ----- prediction ------------------------------------------------------------------------------------
    def predict(self, input, labeled=None) -> float:
        p = self.p
        if float(p.get("tau_M", 0.0)) > 0 or float(p.get("nu_mu", 0.0)) > 0:
            return super().predict(input, labeled)
        try:
            key = self._evidence_key(labeled)
            if key != self._key:
                self._fit, self._key = self._fit_posterior(labeled), key
            fit = self._fit
            subject, item = input
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            cols, mean, var = self._cols(subject, item, b)
            known = []
            for name, m, v in cols:
                if name in fit["index"]:
                    known.append(fit["index"][name])
                else:
                    mean += m
                    var += v
            if known:
                mean += float(fit["mean"][known].sum())
                var += float(fit["cov"][np.ix_(known, known)].sum())
            prob = 1.0 / (1.0 + math.exp(-mean / math.sqrt(1.0 + math.pi * var / 8.0)))
            if p.get("mix_pi", 0.0) > 0:  # identical to HierModel's zero-inflated pair component
                th, th_var = self._theta(subject)
                k, n_lab = fit.get("counts", {}).get((s, b), (0, 0))
                pi = p["mix_pi"]
                if n_lab:
                    m_norm = p["m0"] + p["lam"] * th
                    v_norm = p["tau_mu"] ** 2 + p["tau_delta"] ** 2 + p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var
                    lh = _log_marginal(k, n_lab, p["mix_m"], p["mix_s"] ** 2)
                    ln = _log_marginal(k, n_lab, m_norm, v_norm)
                    w_hard = 1.0 / (1.0 + math.exp(math.log((1 - pi) / pi) + ln - lh))
                    p_hard = _posterior_predictive(k, n_lab, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
                else:
                    w_hard = pi
                    p_hard = _posterior_predictive(0, 0, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
                prob = w_hard * p_hard + (1.0 - w_hard) * prob
            if not math.isfinite(prob):
                return super().predict(input, labeled)
        except Exception:
            try:
                return super().predict(input, labeled)
            except Exception:
                prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))
