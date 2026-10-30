"""Two-scale cross-subject item pooling (head researcher A, swarm round 1).

Motivation (literature + public data):
- Hardy et al. 2026 (arXiv 2610.00651, Table 8): on 29,923 HAL agent rollouts the task-within-benchmark SD is 2.41
  logits while task x model is 0.88 and task x scaffold 1.03, so item difficulty is mostly SHARED across subjects.
- Public measurement-db (crossed Rasch fit, this round): item SD 2.5-3.1 logits, cross-subject ICC 0.66-0.75 on
  Multi-SWE-bench, REAL and ResearchCodeBench.
- HierModel uses one item variance sigma_i (tuned 0.6, ICC ~0.10) for two jobs: moderating unseen items and pooling
  other subjects' labels on the same item. A pair's own acquisition and evaluation pools are disjoint, so the item
  effect of an evaluation item is only ever learned from OTHER subjects. This class keeps the base model for every
  item nobody else has labeled and, for items that other subjects labeled, applies a 1-D exact posterior over the
  shared item effect on the conditional (median-item) scale with prior SD sigma_x.

Mechanism for an evaluation item j of subject A on benchmark b with labels y_s from other subjects s:
  L_m(s) = marginal-scale level of pair (s, b) from the fitted posterior (mu_b + phi_s + delta_sb + gamma + lam*theta)
  kappa  = 1 / sqrt(1 + pi * sigma_x^2 / 8);   L_c = L_m / kappa   (inverse probit approximation)
  u_j ~ N(0, sigma_x^2);  p(u_j | y) on a Gauss-Hermite grid, each likelihood term moderated by its level variance
  p_item  = E[sigmoid((L_c(A) + u_j) / sqrt(1 + pi v_c(A) / 8)) | y],  p_prior = the same under the prior of u_j
  final normal-component prediction = sigmoid(logit(p_base_noeps) + shrink * (logit(p_item) - logit(p_prior)))
so with no informative partner labels the prediction equals the base model's unseen-item prediction exactly.
The optional zero-inflated pair mixture of HierModel is applied afterwards, unchanged.
Numpy only; never raises (falls back to the base prediction).
"""
from __future__ import annotations

import math

import numpy as np

from paiec.hier import HierModel, _entries, _groups, _item_key, _log_marginal, _posterior_predictive, _sk

_GX, _GW = np.polynomial.hermite_e.hermegauss(40)
_GLW = np.log(_GW / np.sqrt(2.0 * np.pi))


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


class ItemPoolModel(HierModel):
    """HierModel with cross-subject item pooling at a separately chosen item variance (sigma_x)."""

    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self.p.setdefault("sigma_x", 2.0)
        self.p.setdefault("pool_shrink", 1.0)
        self._lab_key = None
        self._lab_index = {}

    # ----- helpers ------------------------------------------------------------------------------
    def _label_index(self, labeled):
        """(benchmark_id, item_key) -> list of (subject_key, subject_dict, y); rebuilt when the evidence changes."""
        if self._lab_key != self._key:
            index = {}
            for subject, item, y in _entries(labeled):
                b = str(item.get("benchmark_id", ""))
                index.setdefault((b, _item_key(item)), []).append((_sk(subject), subject, y))
            self._lab_index, self._lab_key = index, self._key
        return self._lab_index

    def _level(self, fit, subject, b, item):
        """Marginal-scale level mean/var of pair (subject, b) for this item, excluding the item effect eps."""
        p = self.p
        s = _sk(subject)
        th, th_var = self._theta(subject)
        shared = p["tau_M"] > 0
        names = [("mu", b), ("phi", s), ("delta", s, b)]
        if shared:
            names.append(("M",))
        for g in _groups(item, p.get("all_feature_tokens", False)):
            names.append(("gamma", b, g))
        if p["ability_param"]:
            phi_prior = (p["lam"] * th, p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var)
            mean, var = 0.0, 0.0
        else:
            phi_prior = (0.0, p["tau_phi"] ** 2)
            mean, var = p["lam"] * th, (p["lam"] ** 2) * th_var
        nu = float(p.get("nu_mu", 0.0))
        mu_var = p["tau_mu"] ** 2 * (nu / (nu - 2.0) if nu > 2 else (3.0 if nu > 0 else 1.0))
        prior = {"mu": (0.0 if shared else p["m0"], mu_var), "phi": phi_prior,
                 "delta": (0.0, p["tau_delta"] ** 2), "gamma": (0.0, p["sigma_g"] ** 2), "M": (p["m0"], p["tau_M"] ** 2)}
        known = [fit["index"][nm] for nm in names if nm in fit["index"]]
        for nm in names:
            if nm not in fit["index"]:
                mpr, vpr = prior[nm[0]]
                mean += mpr
                var += vpr
        if known:
            mean += float(fit["mean"][known].sum())
            var += float(fit["cov"][np.ix_(known, known)].sum())
        return mean, var

    # ----- prediction -----------------------------------------------------------------------------
    def predict(self, input, labeled=None) -> float:
        p = self.p
        try:
            key = self._evidence_key(labeled)
            if key != self._key:
                self._fit, self._key = self._fit_posterior(labeled), key
            fit = self._fit
            subject, item = input
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            partners = [(sk, sd, y) for sk, sd, y in self._label_index(labeled).get((b, _item_key(item)), []) if sk != s]
            if not partners:
                return super().predict(input, labeled)
            m_a, v_a = self._level(fit, subject, b, item)
            # Base prediction for an item nobody labeled (same formula as HierModel: prior item variance sigma_i^2).
            p_base = 1.0 / (1.0 + math.exp(-m_a / math.sqrt(1.0 + math.pi * (v_a + p["sigma_i"] ** 2) / 8.0)))
            sx = float(p["sigma_x"])
            kappa = 1.0 / math.sqrt(1.0 + math.pi * sx * sx / 8.0)
            u = sx * _GX
            logw = _GLW.copy()
            for _, sd, y in partners:
                m_s, v_s = self._level(fit, sd, b, item)
                z = (m_s / kappa + u) / math.sqrt(1.0 + math.pi * (v_s / kappa ** 2) / 8.0)
                logw = logw + (-np.logaddexp(0.0, -z) if y else -np.logaddexp(0.0, z))
            w = np.exp(logw - logw.max())
            w /= w.sum()
            pw = np.exp(_GLW - _GLW.max())
            pw /= pw.sum()
            za = (m_a / kappa + u) / math.sqrt(1.0 + math.pi * (v_a / kappa ** 2) / 8.0)
            sig = 1.0 / (1.0 + np.exp(-za))
            p_item, p_prior = float(np.sum(w * sig)), float(np.sum(pw * sig))
            prob = 1.0 / (1.0 + math.exp(-(_logit(p_base) + p["pool_shrink"] * (_logit(p_item) - _logit(p_prior)))))
            if p.get("mix_pi", 0.0) > 0:  # identical to HierModel's zero-inflated pair component
                th, th_var = self._theta(subject)
                k, n_lab = fit.get("counts", {}).get((s, b), (0, 0))
                pi = p["mix_pi"]
                if n_lab:
                    m_norm = (p["m0"] if not p["ability_param"] else p["m0"] + p["lam"] * th) + (0.0 if p["ability_param"] else p["lam"] * th)
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
