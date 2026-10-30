"""Head researcher B prototype: exact (quadrature) posterior predictive for a pair that is alone on its benchmark.

Why: the joint-mode Laplace + probit output of HierModel under-reacts to a pair's own labels (too high after zeros,
too low after ones). For a pair alone on its benchmark (m0=-1.5): 0/7 labels -> 0.116 vs exact 0.089; the error costs
~0.6e-3 ALC under the model's own prior (swarm/round1/head_researcher_B.md, Check A).

What: when no other subject has labels on the target's benchmark, the pair's level L = mu_b + phi_s + delta_sb + lam*theta
and the item-feature group effects gamma_g are the only latent quantities touched by its labels. Given L the groups are
independent, so the exact posterior is a 1-D grid over L times one Gauss-Hermite quadrature per group:
    p(L | y) ∝ prior(L) * prod_g  ∫ N(gamma; 0, sigma_g^2) prod_{i in g} Bern(y_i | sigmoid(c (L + gamma))) dgamma,
with c = 1/sqrt(1 + pi sigma_i^2 / 8) the item-effect attenuation (each acquired item carries one label). The prior on L is
the hierarchical normal N(m0 + lam*theta + E[phi_s], tau_mu^2 + tau_delta^2 + Var[phi_s] + lam^2 var_theta), optionally
mixed with the hard component N(mix_m, mix_s^2) with weight mix_pi (exact posterior weights, no heuristic blending).
Pairs that share their benchmark with another subject fall back to the parent (Laplace) prediction, so cross-subject
item pooling is unchanged. Requires tau_M = 0, nu_mu = 0, all_feature_tokens = False (the current live configuration).
"""
from __future__ import annotations

import math

import numpy as np

from paiec.hier import HierModel, _entries, _group, _sk

_GX, _GW = np.polynomial.hermite_e.hermegauss(24)
_GW = _GW / _GW.sum()


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class ExactAloneHier(HierModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self._pair_cache = {}

    def _pair_posterior(self, labeled, s, b, subject):
        p = self.p
        own, other_on_b, other_pairs_of_s = [], False, False
        for subj, item, y in _entries(labeled):
            sb = str(item.get("benchmark_id", ""))
            ss = _sk(subj)
            if sb == b and ss == s:
                own.append((_group(item), y))
            elif sb == b:
                other_on_b = True
            elif ss == s:
                other_pairs_of_s = True
        if other_on_b:
            return None
        th, th_var = self._theta(subject)
        m_c = p["m0"] + p["lam"] * th
        v_c = p["tau_mu"] ** 2 + p["tau_delta"] ** 2 + p["lam"] ** 2 * th_var
        fit = self._fit
        j = fit["index"].get(("phi", s)) if fit else None
        if other_pairs_of_s and j is not None:
            m_c += float(fit["mean"][j])
            v_c += float(fit["cov"][j, j])
        else:
            v_c += p["tau_phi"] ** 2
        pi = float(p.get("mix_pi", 0.0))
        sd_c = math.sqrt(v_c)
        lo = min(m_c - 7 * sd_c, p["mix_m"] - 7 * p["mix_s"]) if pi > 0 else m_c - 7 * sd_c
        hi = m_c + 7 * sd_c
        L = np.linspace(lo, hi, 241)
        logprior = -0.5 * (L - m_c) ** 2 / v_c - 0.5 * math.log(v_c)
        if pi > 0:
            lh = -0.5 * (L - p["mix_m"]) ** 2 / p["mix_s"] ** 2 - math.log(p["mix_s"])
            logprior = np.logaddexp(math.log(1 - pi) + logprior, math.log(pi) + lh)
        c = 1.0 / math.sqrt(1.0 + math.pi * p["sigma_i"] ** 2 / 8.0)
        counts = {}
        for g, y in own:
            k, n = counts.get(g, (0, 0))
            counts[g] = (k + int(y), n + 1)
        loglik = np.zeros_like(L)
        group_w = {}
        gam = p["sigma_g"] * _GX
        for g, (k, n) in counts.items():
            if g == "":
                eta = c * L
                loglik += -k * np.logaddexp(0.0, -eta) - (n - k) * np.logaddexp(0.0, eta)
                continue
            eta = c * (L[:, None] + gam[None, :])  # (grid, quad)
            ll = -k * np.logaddexp(0.0, -eta) - (n - k) * np.logaddexp(0.0, eta) + np.log(_GW)[None, :]
            top = ll.max(axis=1, keepdims=True)
            loglik += (top[:, 0] + np.log(np.exp(ll - top).sum(axis=1)))
            w = np.exp(ll - top)
            group_w[g] = w / w.sum(axis=1, keepdims=True)  # posterior over gamma_g given L
        lp = logprior + loglik
        post = np.exp(lp - lp.max())
        post /= post.sum()
        return {"L": L, "post": post, "group_w": group_w, "c": c}

    def predict(self, input, labeled=None) -> float:
        p = self.p
        try:
            key = self._evidence_key(labeled)
            if key != self._key:
                self._fit, self._key = self._fit_posterior(labeled), key
                self._pair_cache = {}
            subject, item = input
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            if (s, b) not in self._pair_cache:
                self._pair_cache[(s, b)] = self._pair_posterior(labeled, s, b, subject)
            pp = self._pair_cache[(s, b)]
            if pp is None:  # benchmark shared with another subject: keep the Laplace model (item pooling)
                return super().predict(input, labeled)
            L, post, c = pp["L"], pp["post"], pp["c"]
            g = _group(item)
            if g and g in pp["group_w"]:
                gam = p["sigma_g"] * _GX
                per_L = (pp["group_w"][g] * _sig(c * (L[:, None] + gam[None, :]))).sum(axis=1)
            else:
                extra = p["sigma_g"] ** 2 if g else 0.0
                per_L = _sig(L / math.sqrt(1.0 + math.pi * (p["sigma_i"] ** 2 + extra) / 8.0))
            prob = float(post @ per_L)
        except Exception:
            prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))
