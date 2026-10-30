"""ShiftNPHier: SubjCalHier (attr x attr_scale, release-year clamp) with two optional subject-prior changes from the
round-3 shift study (swarm/round3/subject_shift.md):

  attr_coef       list of 18 floats (paiec.subjects.attributes order) or None: replaces the prior's attribute ridge,
                  e.g. the provider-free refit in swarm/round3/shift_sims/np_attr_coef.json (variant 'NP')
  attr_var        float or None: replaces the ridge residual variance (default: keep the prior's)
  attr_year_mult  float or None: effective multiplier of the release-year term while the rest of the attribute
                  prediction keeps attr_scale (variant 'SY'; A156+SY1.0 = x1.56 dispersion fix without amplified
                  vintage extrapolation)
Defaults reproduce SubjCalHier (and HierModel at SubjCal defaults) bit for bit. Never raises: a malformed override is
ignored. To ship: copy this file to paiec/ext/shift_np.py, then
  tools/make_submission.py <name> --plugin paiec.ext.shift_np:ShiftNPHier --params '{<level block>, "attr_scale": 1.56,
      "attr_year_max": 2026.4, ...}'
"""
from __future__ import annotations

from paiec.ext.pd1_subjcal import SubjCalHier

try:
    from paiec.subjects import attributes
except ImportError:  # pragma: no cover
    pass


class ShiftNPHier(SubjCalHier):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self.p.setdefault("attr_year_mult", None)
        try:
            coef = self.p.get("attr_coef")
            if coef is not None and isinstance(self.abilities, dict) and "names" in self.abilities:
                coef = [float(c) for c in coef]
                old = self.abilities.get("attr") or {}
                if len(coef) == len(old.get("coef", coef)):
                    var = self.p.get("attr_var")
                    var = float(var) if var is not None else float(old.get("var", 0.25))
                    self.abilities = dict(self.abilities, attr={"coef": coef, "var": var})
        except Exception:
            pass

    def _theta(self, subject):
        th, var = super()._theta(subject)
        f = self.p.get("attr_year_mult")
        if f is None:
            return th, var
        try:
            _, _, src = self._theta_source(subject)
            if src != "attr":
                return th, var
            x1 = attributes(subject)[1]
            ymax = float(self.p.get("attr_year_max", 0.0) or 0.0)
            if ymax > 0:
                x1 = min(x1, ymax - 2025.0)
            coef1 = float(self.abilities["attr"]["coef"][1])
            return th + (float(f) - float(self.p["attr_scale"])) * coef1 * x1, var
        except Exception:
            return th, var
