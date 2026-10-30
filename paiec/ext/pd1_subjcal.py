"""Postdoc 1 (priors), swarm round 1: source-calibrated subject prior.

Why (public LOBO, swarm/round1/postdoc_priors.md section 2): the transferred ability theta comes from two sources with
opposite miscalibration.
- Canonical-name abilities (fit_subject_prior: theta = sum of centred logits / (k + 1), k = benchmarks seen) transfer
  WEAKER than the model assumes: within-benchmark slope of the held-out centred pair logit on theta is 0.65 (n=123
  pairs; 1.24 / 0.52 / 0.89 / 0.44 on MathArena / Multi-SWE-bench / REAL / ResearchCodeBench).
- Attribute-ridge abilities (unknown names) transfer STRONGER: slope 1.67 (n=97; 1.33 / 2.21 / 2.66 / 0.99). The ridge
  is fitted on name abilities that are already shrunk by k/(k+1) (mostly k=1, factor 1/2), so its predictions are on a
  compressed scale.
Refitting the ridge on UNSHRUNK targets scales its predictions by 1.56-1.60 in every LOBO fold. In the exact pair-level
screen (swarm/round1/postdoc_priors_sims/fast_eval.py) attr_scale=1.56 lowers ALC by 0.4-0.85e-3 on all four held-out
benchmarks under every pool scenario tried, while name_r > 1 (shrinking name abilities) costs +0.1-0.2e-3 and inflating
the attribute variance (attr_var_mult > 1) gives back part of the gain. Recommended: attr_scale 1.56, attr_year_max 2026.4,
name_r = name_s2 = attr_var_mult = 1 (attr_clip off).

What: _theta() is replaced (consistently used by the fit, the predictive variance and the mixture branch):
  name source:  k = round(1/var - 1);  theta' = theta * (k + 1) / (k + name_r)
                var'   = name_s2 * name_r / (k + name_r)        (name_r = name_s2 = 1 reproduces the original exactly)
  attr source:  theta' = attr_scale * theta_attr;   var' = attr_var_mult * var_attr
  optional:     attr_year_max (e.g. 2026.4 = last public release date): the release-year feature is clamped to it, so
                abilities never extrapolate in time (an unknown OpenAI model released 2026-10 would otherwise get +0.99
                after scaling vs +0.76 at the clamp); attr_clip bounds |theta_attr'| (0 = off).
Defaults reproduce HierModel bit for bit. Numpy only; never raises (falls back to the parent's value).
"""
from __future__ import annotations

from paiec.hier import HierModel

try:
    from paiec.subjects import attributes, name_key, predict_attr
except ImportError:  # pragma: no cover
    pass

SUBJCAL_DEFAULTS = {"name_r": 1.0, "name_s2": 1.0, "attr_scale": 1.0, "attr_var_mult": 1.0, "attr_clip": 0.0,
                    "attr_year_max": 0.0, "attr_year_mult": None}


class SubjCalHier(HierModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        for k, v in SUBJCAL_DEFAULTS.items():
            self.p.setdefault(k, v)
        self._theta_cache = {}

    def _theta_source(self, subject):
        """(theta, var, source) with source in {'name', 'attr', 'none'} before calibration."""
        prior = self.abilities
        if "names" in prior:
            key = name_key(subject)
            if key and key in prior["names"]:
                th, var = prior["names"][key]
                return float(th), float(var), "name"
            attr = prior.get("attr")
            if attr:
                ymax = float(self.p.get("attr_year_max", 0.0) or 0.0)
                if ymax > 0:  # clamp the release-year feature to the training range (no extrapolation in time)
                    x = attributes(subject)
                    x[1] = min(x[1], ymax - 2025.0)
                    return float(sum(c * v for c, v in zip(attr["coef"], x))), float(attr["var"]), "attr"
                return float(predict_attr(attr["coef"], subject)), float(attr["var"]), "attr"
            return 0.0, self.p["tau_theta"] ** 2, "none"
        th, var = super()._theta(subject)
        return float(th), float(var), "name" if th != 0.0 else "none"

    def _theta(self, subject):
        p = self.p
        try:
            th, var, src = self._theta_source(subject)
            if src == "name":
                r = float(p["name_r"])
                k = max(1.0, round(1.0 / max(var, 1e-9) - 1.0))
                th = th * (k + 1.0) / (k + r)
                var = float(p["name_s2"]) * r / (k + r)
            elif src == "attr":
                th = float(p["attr_scale"]) * th
                ym = p.get("attr_year_mult")
                if ym is not None:  # release-year term at ym x its base coefficient (round-3 shift study, SY1.0)
                    x1 = attributes(subject)[1]
                    ymax = float(p.get("attr_year_max", 0.0) or 0.0)
                    if ymax > 0:
                        x1 = min(x1, ymax - 2025.0)
                    th += (float(ym) - float(p["attr_scale"])) * float(self.abilities["attr"]["coef"][1]) * x1
                var = float(p["attr_var_mult"]) * var
                clip = float(p.get("attr_clip", 0.0) or 0.0)
                if clip > 0:
                    th = max(-clip, min(clip, th))
            return th, var
        except Exception:
            return super()._theta(subject)
