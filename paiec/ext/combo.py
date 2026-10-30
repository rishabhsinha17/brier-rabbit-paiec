"""Composite plug-in (swarm round 3, composer): the local-only improvements that pass their local rules, each behind its
own switch, stacked on whichever level block (m0, tau_mu, tau_delta, tau_phi, mixture) wins live.

Components (every switch off = HierModel bit for bit; each switch alone = that plug-in's own class bit for bit):
  subjcal   (False)  paiec.ext.pd1_subjcal.SubjCalHier._theta: attribute-ridge abilities (unknown names) x attr_scale
                     (ship 1.56) with the release-year feature clamped at attr_year_max (ship 2026.4); name abilities
                     unchanged at name_r = name_s2 = 1.
  pool_item (False)  paiec.ext.pool_cavity.CavityPoolModel: cavity-corrected cross-subject pooling on evaluation items
                     another subject labelled, shared item effect u ~ N(0, sigma_x^2) (ship sigma_x 3.0, the pick of
                     pdpool_bench_k2's pre-registered rule). Everything else is the active base prediction.
  pool_sim  (False)  paiec.ext.pdadapt_simpool.SimilarityPoolModel with the skeptic's patch P2: the name and
                     (benchmark, harness) effects get SHARES rho_name (0.45) and rho_h (0.288) of the pair variance
                     tau_phi^2 + tau_delta^2 instead of absolute SDs, so the split survives a level-block change (at
                     cur_final it equals the absolute defaults 0.75 / 0.6). rho_name / rho_h None = absolute tau_name /
                     tau_h. Needs tau_M = 0 and nu_mu = 0 (otherwise the HierModel structure is used).
  plain params       any HierModel parameter (sigma_g, sigma_i, ...) passes through unchanged.
  m0 matching        q0 > 0 (as paiec.ext.pd1_prior.PriorCalHier): m0 is solved at construction so that the budget-0
                     prediction of a reference subject (theta q0_theta, var q0_var, one item_features group if q0_group)
                     equals q0. q0_match "base" (with q0 0) instead keeps B0 fixed across the subject-prior change: the
                     reference subject is an unknown name with UNcalibrated attribute ability (q0_theta, q0_var), the
                     target is its budget-0 prediction under the configured level block with subjcal off, and m0 is
                     re-solved with its calibrated theta (attr_scale * q0_theta), i.e. m0' = m0 - lam (attr_scale - 1)
                     q0_theta. The exact screen (swarm/round3/composer.md section 4) says leave q0_match off: on
                     public-like unknown names SubjCal changes the spread, not the level. Use an explicit q0 to hold
                     B0 when a plain parameter that moves B0 through moderation (sigma_g, sigma_i) changes and the L15
                     B0 test says H_cal.

Composition, not copies: ComboModel's MRO is ComboModel -> SubjCalHier -> CavityPoolModel -> _ActiveStructure ->
SimilarityPoolModel -> HierModel. _ActiveStructure routes the fit and the base prediction to SimilarityPoolModel when
pool_sim is on and to HierModel otherwise, so CavityPoolModel's own super().predict lands on the active structure, and
the cavity helpers (_row, _level_vec) read the active structure's columns. Numpy only; never raises (falls back to the
active base prediction, then to sigmoid(m0)). Ship with tools/make_submission.py --plugin paiec.ext.combo:ComboModel,
which vendors the three component modules first.
"""
from __future__ import annotations

import math

from paiec.ext.pd1_subjcal import SubjCalHier
from paiec.ext.pdadapt_simpool import SimilarityPoolModel
from paiec.ext.pool_cavity import CavityPoolModel
from paiec.hier import HierModel, _posterior_predictive

COMBO_DEFAULTS = {
    "subjcal": False, "pool_item": False, "pool_sim": False,
    "rho_name": 0.45, "rho_h": 0.288,
    "q0": 0.0, "q0_match": "", "q0_theta": 0.0, "q0_var": 0.25, "q0_group": True,
}


class _ActiveStructure(SimilarityPoolModel):
    """Fit and base prediction of the active pair structure: SimilarityPoolModel's (name / harness / pair split of the
    pair deviation) when pool_sim is on, HierModel's own otherwise (bit for bit)."""

    def _sim_active(self) -> bool:
        p = self.p
        try:
            return bool(p.get("pool_sim", False)) and float(p.get("tau_M", 0.0) or 0.0) <= 0.0 \
                and float(p.get("nu_mu", 0.0) or 0.0) <= 0.0
        except (TypeError, ValueError):
            return False

    def _fit_posterior(self, labeled):
        if self._sim_active():
            return SimilarityPoolModel._fit_posterior(self, labeled)
        return HierModel._fit_posterior(self, labeled)

    def predict(self, input, labeled=None) -> float:
        if self._sim_active():
            return SimilarityPoolModel.predict(self, input, labeled)
        return HierModel.predict(self, input, labeled)


class ComboModel(SubjCalHier, CavityPoolModel, _ActiveStructure):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)  # SubjCalHier -> CavityPoolModel -> SimilarityPoolModel -> HierModel
        p = self.p
        for k, v in COMBO_DEFAULTS.items():
            p.setdefault(k, v)
        try:  # P2: pooled variances as shares of the pair variance (fixed at construction, like the taus)
            total = float(p["tau_phi"]) ** 2 + float(p["tau_delta"]) ** 2
            if p.get("rho_name") is not None:
                p["tau_name"] = math.sqrt(max(float(p["rho_name"]) * total, 0.0))
            if p.get("rho_h") is not None:
                p["tau_h"] = math.sqrt(max(float(p["rho_h"]) * total, 0.0))
        except Exception:
            pass
        self.m0_level_block = p.get("m0")  # m0 as configured, before any matching (provenance)
        try:
            q0 = float(p.get("q0", 0.0) or 0.0)
            th, tv = float(p["q0_theta"]), float(p["q0_var"])  # explicit q0: the theta the model uses (pd1_prior)
            if q0 <= 0.0 and str(p.get("q0_match", "") or "") == "base":
                th_c, tv_c = self._ref_theta()
                if (th_c, tv_c) != (th, tv):  # nothing to match when the reference subject's theta is unchanged
                    q0 = self._b0(float(p["m0"]), th, tv)  # plain level block, uncalibrated reference theta
                    th, tv = th_c, tv_c
            if 0.0 < q0 < 1.0:
                p["m0"] = self._solve_m0(q0, th, tv)
        except Exception:
            pass

    # ----- switches ------------------------------------------------------------------------------------
    def _theta(self, subject):
        if self.p.get("subjcal", False):
            return SubjCalHier._theta(self, subject)
        return HierModel._theta(self, subject)

    def predict(self, input, labeled=None) -> float:
        try:
            if self.p.get("pool_item", False):
                return CavityPoolModel.predict(self, input, labeled)
            return _ActiveStructure.predict(self, input, labeled)
        except Exception:
            try:
                return _ActiveStructure.predict(self, input, labeled)
            except Exception:
                p = self.p
                return float(min(1 - p["clip"], max(p["clip"], 1.0 / (1.0 + math.exp(-p["m0"])))))

    # ----- cavity helpers on the active structure ------------------------------------------------------
    def _level_vec(self, fit, subject, b, item):
        """(fit columns, prior mean, prior variance) of the pair level for this item, excluding eps."""
        if not self._sim_active():
            return CavityPoolModel._level_vec(self, fit, subject, b, item)
        spec, mean, var = self._cols(subject, item, b)
        cols = []
        for name, m, v in spec:
            if name[0] == "eps":
                continue
            j = fit["index"].get(name)
            if j is None:
                mean += m
                var += v
            else:
                cols.append(j)
        return cols, mean, var

    def _row(self, fit, subject, item, b):
        """Fit columns and offset of one evidence label (the row the active _fit_posterior built for it)."""
        if not self._sim_active():
            return CavityPoolModel._row(self, fit, subject, item, b)
        spec, off, _ = self._cols(subject, item, b)
        return [fit["index"][name] for name, _, _ in spec], off

    # ----- matched B0 (q0) -----------------------------------------------------------------------------
    def _ref_theta(self):
        """q0_match 'base': the reference subject is an unknown name whose UNcalibrated attribute ability is
        (q0_theta, q0_var); this is its (theta, var) as the composite sees it (attribute calibration when subjcal is on;
        the release-year clamp does not apply to a subject given by its theta)."""
        p = self.p
        th, tv = float(p["q0_theta"]), float(p["q0_var"])
        if p.get("subjcal", False):
            th = float(p.get("attr_scale", 1.0)) * th
            clip = float(p.get("attr_clip", 0.0) or 0.0)
            if clip > 0:
                th = max(-clip, min(clip, th))
            tv = float(p.get("attr_var_mult", 1.0)) * tv
        return th, tv

    def _b0(self, m0: float, th: float, tv: float) -> float:
        """Budget-0 prediction for a subject with transferred ability (th, tv): HierModel.predict on an empty evidence
        set (normal level prior, no shared level), mixed with the hard component when mix_pi > 0. The pool_sim split
        preserves the pair variance, so it applies to every structure."""
        p = self.p
        var = p["lam"] ** 2 * tv + p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2 + p["sigma_i"] ** 2
        if p.get("q0_group", True):
            var += p["sigma_g"] ** 2
        z = (m0 + p["lam"] * th) / math.sqrt(1.0 + math.pi * var / 8.0)
        prob = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
        pi = float(p.get("mix_pi", 0.0) or 0.0)
        if pi > 0:
            p_hard = _posterior_predictive(0, 0, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
            prob = pi * p_hard + (1.0 - pi) * prob
        return prob

    def _solve_m0(self, q0: float, th: float, tv: float) -> float:
        lo, hi = -10.0, 6.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self._b0(mid, th, tv) < q0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
