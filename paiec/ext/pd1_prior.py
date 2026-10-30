"""Postdoc 1 (priors), swarm round 1: level-prior block with (a) an ability-tilted hard-mixture weight and (b) an
optional matched-B0 parametrization, on top of the source-calibrated subject prior (pd1_subjcal.SubjCalHier).

(a) Tilted mixture weight ("pitcher covariate", Brown 2008; head_researcher_A idea 6, head_researcher_B idea 3):
        logit pi_s = logit(mix_pi) + mix_b * lam * theta_s
    theta_s is the (calibrated) transferred ability. Public LOBO: logistic slope of 1[pair rate < 5%] on theta is -1.41
    with benchmark fixed effects (-1.52 pooled; AUC 0.72), so mix_b = -1.4 is the data-based default when enabled.
    mix_b = 0 reproduces HierModel's constant weight exactly.
(b) Matched B0 (q0): if q0 > 0, m0 is solved at construction so that the budget-0 prediction for a reference subject
    (theta = q0_theta, var = q0_var, item with an item_features group when q0_group) equals q0. This lets live arms
    change the SHAPE of the level prior (mixture weight, tau split) at a fixed B0 level, or the level at a fixed shape,
    so each live arm moves one readout: B0 for the level, B1-B7 for the shape.
Everything else is HierModel.predict unchanged (the predict body is copied with the one-line pi change; core untouched).
Numpy only; never raises.
"""
from __future__ import annotations

import math

import numpy as np

from paiec.ext.pd1_subjcal import SubjCalHier
from paiec.hier import _groups, _item_key, _log_marginal, _posterior_predictive, _sk

PRIOR_DEFAULTS = {"mix_b": 0.0, "q0": 0.0, "q0_theta": 0.0, "q0_var": 0.25, "q0_group": True}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


class PriorCalHier(SubjCalHier):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        for k, v in PRIOR_DEFAULTS.items():
            self.p.setdefault(k, v)
        if float(self.p.get("q0", 0.0) or 0.0) > 0:
            try:
                self.p["m0"] = self._solve_m0(float(self.p["q0"]))
            except Exception:
                pass

    # ----- level block -----------------------------------------------------------------------
    def _pi(self, theta: float) -> float:
        p = self.p
        base = float(p.get("mix_pi", 0.0))
        if base <= 0:
            return 0.0
        b = float(p.get("mix_b", 0.0) or 0.0)
        if b == 0.0:
            return base
        lg = math.log(base / (1.0 - base)) + b * p["lam"] * theta
        return min(0.98, max(1e-4, _sigmoid(lg)))

    def _b0(self, m0: float) -> float:
        """Budget-0 prediction of HierModel for the reference subject (mirrors predict() with an empty fit)."""
        p = self.p
        th, tv = float(p["q0_theta"]), float(p["q0_var"])
        var = p["lam"] ** 2 * tv + p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2 + p["sigma_i"] ** 2
        if p.get("q0_group", True):
            var += p["sigma_g"] ** 2
        prob = _sigmoid((m0 + p["lam"] * th) / math.sqrt(1.0 + math.pi * var / 8.0))
        pi = self._pi(th)
        if pi > 0:
            p_hard = _posterior_predictive(0, 0, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
            prob = pi * p_hard + (1.0 - pi) * prob
        return prob

    def _solve_m0(self, q0: float) -> float:
        lo, hi = -10.0, 6.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self._b0(mid) < q0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ----- prediction (HierModel.predict with the tilted weight) -------------------------------
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
            th, th_var = self._theta(subject)
            shared = p["tau_M"] > 0
            names = [("mu", b), ("phi", s), ("delta", s, b)]
            if shared:
                names.append(("M",))
            for g in _groups(item, p.get("all_feature_tokens", False)):
                names.append(("gamma", b, g))
            names.append(("eps", b, _item_key(item)))
            if p["ability_param"]:
                phi_prior = (p["lam"] * th, p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var)
                mean, var = 0.0, 0.0
            else:
                phi_prior = (0.0, p["tau_phi"] ** 2)
                mean, var = p["lam"] * th, (p["lam"] ** 2) * th_var
            nu = float(p.get("nu_mu", 0.0))
            mu_var = p["tau_mu"] ** 2 * (nu / (nu - 2.0) if nu > 2 else (3.0 if nu > 0 else 1.0))
            prior = {"mu": (0.0 if shared else p["m0"], mu_var), "phi": phi_prior,
                     "delta": (0.0, p["tau_delta"] ** 2), "gamma": (0.0, p["sigma_g"] ** 2),
                     "eps": (0.0, p["sigma_i"] ** 2), "M": (p["m0"], p["tau_M"] ** 2)}
            known = [fit["index"][nm] for nm in names if nm in fit["index"]]
            for nm in names:
                if nm not in fit["index"]:
                    mpr, vpr = prior[nm[0]]
                    mean += mpr
                    var += vpr
            if known:
                mean += float(fit["mean"][known].sum())
                var += float(fit["cov"][np.ix_(known, known)].sum())
            prob = 1.0 / (1.0 + math.exp(-mean / math.sqrt(1.0 + math.pi * var / 8.0)))
            pi = self._pi(th)
            if pi > 0:
                k, n_lab = fit.get("counts", {}).get((s, b), (0, 0))
                if n_lab:
                    m_norm = p["m0"] + p["lam"] * th
                    v_norm = p["tau_mu"] ** 2 + p["tau_delta"] ** 2 + p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var
                    lh = _log_marginal(k, n_lab, p["mix_m"], p["mix_s"] ** 2)
                    ln = _log_marginal(k, n_lab, m_norm, v_norm)
                    w_hard = 1.0 / (1.0 + math.exp(max(-700.0, min(700.0, math.log((1 - pi) / pi) + ln - lh))))
                    p_hard = _posterior_predictive(k, n_lab, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
                else:
                    w_hard = pi
                    p_hard = _posterior_predictive(0, 0, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
                prob = w_hard * p_hard + (1.0 - w_hard) * prob
        except Exception:
            prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))
