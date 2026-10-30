"""Mixture-consistent cross-pair pooling ("BCM"), postdoc_adaptation, swarm round 1.

Problem (HierModel with the zero-inflated pair mixture, mix_pi > 0):
  - The normal-component prediction comes from ONE Laplace fit in which every pair is normal. A near-zero ("hard")
    partner on the same benchmark therefore drags the shared benchmark level mu_b down, and its zeros also inform
    item effects. Example (swarm/round1/postdoc_adaptation_sims/contamination.py, mix_pi .35, m0 -1): target 1/1 alone
    -> 0.510; same target with a 0/1 partner -> 0.470, 0/7 partner -> 0.396.
  - The hard weight of a pair uses only its OWN counts against the PRIOR normal component, so a partner that shows the
    benchmark is easy (or hard) never changes how likely the target is to be hard.
  Which of these is right depends on whether hardness is a pair property (contamination is wrong) or a benchmark
  property (the partner's zeros are informative). This class makes the mixture and the pooling one model and lets one
  global knob decide.

Model: each pair p on benchmark b has z_p in {hard, normal}. Normal pairs share mu_b exactly as in HierModel; hard
pairs have their own level h_p ~ N(mix_m, mix_s^2) and carry no information about mu_b, groups or items. The z_p of
the pairs on one benchmark are exchangeable with marginal P(hard) = mix_pi and intra-benchmark correlation hard_rho
(beta-binomial). hard_rho = 0: independent pair-level hardness; hard_rho -> 1: the whole benchmark is hard or not.
  P(z | y) ∝ P(z) * prod_{p hard} M_h(k_p, n_p) * J(normal set), where J integrates the shared mu_b exactly
  (Gauss-Hermite over mu_b, inner 1-D marginal per pair; the same pair-level marginals the base mixture uses).
  Prediction for target T = sum_z P(z | y) * [z_T hard ? p_hard(T) : normal prediction from a Laplace fit on the
  labels of the normal pairs only].
For a pair alone on its benchmark this is identical to HierModel (same weights, same fit). Requires tau_M = 0.
Cost: 2^K - 1 extra fits per benchmark with K labeled pairs (K <= 2 in formative draws -> 2 extra small fits);
benchmarks with more than bcm_max_pairs pairs fall back to HierModel. Numpy only; never raises.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from paiec.hier import HierModel, _entries, _groups, _item_key, _log_marginal, _posterior_predictive, _sk

_MX, _MW = np.polynomial.hermite_e.hermegauss(40)
_MLW = np.log(_MW / np.sqrt(2.0 * np.pi))


def _lse(a) -> float:
    a = np.asarray(a, dtype=float)
    top = float(np.max(a))
    return top + float(np.log(np.sum(np.exp(a - top))))


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class MixturePoolModel(HierModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self.p.setdefault("hard_rho", 0.0)
        self.p.setdefault("bcm_max_pairs", 4)
        self._bkey = None
        self._bcache = {}

    # ----- helpers ----------------------------------------------------------------------------------
    def _normal_pred(self, fit, subject, item) -> float:
        """HierModel's normal-component (no mixture) prediction from a given fit."""
        p = self.p
        b = str(item.get("benchmark_id", ""))
        s = _sk(subject)
        th, th_var = self._theta(subject)
        names = [("mu", b), ("phi", s), ("delta", s, b)]
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
        prior = {"mu": (p["m0"], mu_var), "phi": phi_prior, "delta": (0.0, p["tau_delta"] ** 2),
                 "gamma": (0.0, p["sigma_g"] ** 2), "eps": (0.0, p["sigma_i"] ** 2)}
        index = fit["index"]
        known = [index[nm] for nm in names if nm in index]
        for nm in names:
            if nm not in index:
                mpr, vpr = prior[nm[0]]
                mean += mpr
                var += vpr
        if known:
            mean += float(fit["mean"][known].sum())
            var += float(fit["cov"][np.ix_(known, known)].sum())
        return 1.0 / (1.0 + math.exp(-mean / math.sqrt(1.0 + math.pi * var / 8.0)))

    def _pair_offset(self, subject):
        p = self.p
        th, th_var = self._theta(subject)
        # Pair level given mu_b: lam*theta + phi + delta (ability_param folds lam*theta into phi's prior mean).
        return p["lam"] * th, p["tau_delta"] ** 2 + p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var

    def _log_J(self, members) -> float:
        """log of ∫ N(mu; m0, tau_mu^2) prod_p M(k_p, n_p | mu + off_p, v_p) dmu  (members: list of (k, n, off, v))."""
        p = self.p
        if not members:
            return 0.0
        if len(members) == 1:
            k, n, off, v = members[0]
            return _log_marginal(k, n, p["m0"] + off, p["tau_mu"] ** 2 + v)
        mus = p["m0"] + p["tau_mu"] * _MX
        acc = _MLW.copy()
        for k, n, off, v in members:
            if n == 0:
                continue
            acc = acc + np.array([_log_marginal(k, n, m + off, v) for m in mus])
        return _lse(acc)

    def _config_prior(self, K: int, h: int) -> float:
        pi = float(self.p["mix_pi"])
        rho = float(self.p.get("hard_rho", 0.0))
        if rho <= 1e-6:
            return h * math.log(pi) + (K - h) * math.log(1.0 - pi)
        rho = min(rho, 0.999)
        a = pi * (1.0 - rho) / rho
        b = (1.0 - pi) * (1.0 - rho) / rho
        return _log_beta(a + h, b + K - h) - _log_beta(a, b)

    def _bench_state(self, labeled, b):
        """Per-benchmark pairs, counts, configuration posterior and per-configuration fits (cached per evidence)."""
        if self._bkey != self._key:
            self._bcache, self._bkey = {}, self._key
        if b in self._bcache:
            return self._bcache[b]
        p = self.p
        pairs, subj_of, counts = [], {}, {}
        for subject, item, y in _entries(labeled):
            if str(item.get("benchmark_id", "")) != b:
                continue
            s = _sk(subject)
            if s not in subj_of:
                subj_of[s] = subject
                pairs.append(s)
            k, n = counts.get(s, (0, 0))
            counts[s] = (k + int(y), n + 1)
        state = {"pairs": pairs, "subj": subj_of, "counts": counts, "fits": {}, "post": None}
        if 2 <= len(pairs) <= int(p.get("bcm_max_pairs", 4)):
            K = len(pairs)
            info = {}
            for s in pairs:
                off, v = self._pair_offset(subj_of[s])
                k, n = counts[s]
                info[s] = (k, n, off, v, _log_marginal(k, n, p["mix_m"], p["mix_s"] ** 2))
            post = {}
            for mask in itertools.product((0, 1), repeat=K):  # 1 = normal
                normal = frozenset(s for s, z in zip(pairs, mask) if z)
                h = K - len(normal)
                lw = self._config_prior(K, h)
                lw += sum(info[s][4] for s in pairs if s not in normal)
                lw += self._log_J([info[s][:4] for s in pairs if s in normal])
                post[normal] = lw
            top = max(post.values())
            z = sum(math.exp(v - top) for v in post.values())
            state["post"] = {k: math.exp(v - top) / z for k, v in post.items()}
        self._bcache[b] = state
        return state

    def _fit_subset(self, labeled, b, normal, state):
        """Laplace fit on the labels of the normal pairs of benchmark b (other benchmarks' labels of those subjects kept)."""
        if normal in state["fits"]:
            return state["fits"][normal]
        if normal == frozenset(state["pairs"]):
            fit = self._fit
        else:
            keep = []
            for entry in labeled or []:
                try:
                    (subject, item), _ = entry
                    s = _sk(subject)
                    if str(item.get("benchmark_id", "")) == b:
                        if s in normal:
                            keep.append(entry)
                    elif s in normal:
                        keep.append(entry)
                except (TypeError, ValueError):
                    continue
            fit = self._fit_posterior(keep)
        state["fits"][normal] = fit
        return fit

    # ----- prediction --------------------------------------------------------------------------------
    def predict(self, input, labeled=None) -> float:
        p = self.p
        if float(p.get("mix_pi", 0.0)) <= 0 or float(p.get("tau_M", 0.0)) > 0:
            return super().predict(input, labeled)
        try:
            key = self._evidence_key(labeled)
            if key != self._key:
                self._fit, self._key = self._fit_posterior(labeled), key
            subject, item = input
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            state = self._bench_state(labeled, b)
            others = [q for q in state["pairs"] if q != s]
            if not others or state["post"] is None or s not in state["pairs"]:
                return super().predict(input, labeled)
            k, n = state["counts"][s]
            p_hard = _posterior_predictive(k, n, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2)
            prob = 0.0
            for normal, w in state["post"].items():
                if w < 1e-6:
                    continue
                if s in normal:
                    prob += w * self._normal_pred(self._fit_subset(labeled, b, normal, state), subject, item)
                else:
                    prob += w * p_hard
            tot = sum(w for w in state["post"].values() if w >= 1e-6)
            prob = prob / tot if tot > 0 else super().predict(input, labeled)
            if not math.isfinite(prob):
                return super().predict(input, labeled)
        except Exception:
            try:
                return super().predict(input, labeled)
            except Exception:
                prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))
