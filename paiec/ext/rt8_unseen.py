"""Round-8 red team: unseen-effect moderation at prediction time (plug-in over the shipping composite).

Why (swarm/round8/redteam.md): on public LOBO the composite over-predicts near-zero pairs and under-predicts easy pairs
at late budgets well beyond an unbiased per-pair estimate (W20+SCP bias at B31: near-zero +0.050 vs the 31-label
empirical mean -0.001; easy -0.075 vs -0.026). The joint-mode fit puts the pair level roughly on the marginal scale
(each acquired item's eps and each group's gamma absorb part of the labels), and the prediction for an evaluation item
whose item_features group (or item) has no evidence then adds the full prior variance of that unseen effect
(sigma_g^2 + sigma_i^2) as probit moderation, pulling extreme pairs toward 0.5 a second time.

What: when the target pair already has >= ug_nmin of its own labels, the prior variance of UNSEEN group effects is
multiplied by ug_mult and that of unseen item effects by ui_mult in the prediction only (the fit, the evidence cache and
every seen effect are untouched). B0 (no own labels) is unchanged by construction. ug_mult = ui_mult = 1 reproduces
paiec.ext.combo.ComboModel bit for bit. Numpy only; never raises (falls back to ComboModel).
"""
from __future__ import annotations

import math

from paiec.ext.combo import ComboModel
from paiec.hier import _entries, _sk

RT8_DEFAULTS = {"ug_mult": 1.0, "ui_mult": 1.0, "ug_nmin": 1}


class UnseenModCombo(ComboModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        for k, v in RT8_DEFAULTS.items():
            self.p.setdefault(k, v)
        self._own_key = None
        self._own = {}

    def _own_count(self, labeled, s, b):
        if self._own_key != self._key:
            own = {}
            for subject, item, _y in _entries(labeled):
                k = (_sk(subject), str(item.get("benchmark_id", "")))
                own[k] = own.get(k, 0) + 1
            self._own, self._own_key = own, self._key
        return self._own.get((s, b), 0)

    def predict(self, input, labeled=None) -> float:
        p = self.p
        try:
            ug, ui = float(p.get("ug_mult", 1.0)), float(p.get("ui_mult", 1.0))
            if ug == 1.0 and ui == 1.0:
                return ComboModel.predict(self, input, labeled)
            key = self._evidence_key(labeled)
            if key != self._key:  # fit with the configured variances before anything is modified
                self._fit, self._key = self._fit_posterior(labeled), key
            subject, item = input
            if self._own_count(labeled, _sk(subject), str(item.get("benchmark_id", ""))) < int(p.get("ug_nmin", 1)):
                return ComboModel.predict(self, input, labeled)
            sg, si = p["sigma_g"], p["sigma_i"]
            try:
                p["sigma_g"], p["sigma_i"] = sg * math.sqrt(max(ug, 0.0)), si * math.sqrt(max(ui, 0.0))
                return ComboModel.predict(self, input, labeled)
            finally:
                p["sigma_g"], p["sigma_i"] = sg, si
        except Exception:
            return ComboModel.predict(self, input, labeled)
