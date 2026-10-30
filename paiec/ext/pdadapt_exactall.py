"""Exact (quadrature) posterior predictive for EVERY pair, shared or alone (postdoc_adaptation, swarm round 1).

Extends head_researcher_B's ExactAloneHier (paiec/ext/hrB_exact.py), which only covers pairs alone on their
benchmark. Why it matters: under the model's own prior the Laplace + probit predictive's excess Brier ALC is 0.70e-3
for a pair that shares its benchmark with one partner, vs 0.62e-3 for an alone pair
(swarm/round1/postdoc_adaptation_sims/shared_laplace_excess.py); 42% of live pair appearances are shared.

Method for target pair T = (subject s, benchmark b):
  1. Cavity: Laplace fit on the labels of the OTHER subjects on benchmark b only (no partner -> the prior). It gives
     Gaussian posteriors for the benchmark level mu_b, the group effects gamma_bg and the item effects eps_bi that
     partners have touched. (Cross-covariances in the cavity are dropped.)
  2. Prior for T's level L = mu_b + lam*theta + phi_s + delta_sb:
       normal component N(E_cav[mu_b] + lam*theta, Var_cav[mu_b] + tau_phi^2 + tau_delta^2 + lam^2 var_theta),
       mixed with the hard atom N(mix_m, mix_s^2) at weight mix_pi when the mixture is on (exact weights).
  3. Likelihood of T's own label on item i in group g: sigmoid(c_i (L + gamma_g + m_i)), c_i = 1/sqrt(1 + pi v_i / 8)
     with (m_i, v_i) the cavity posterior of eps_bi (prior (0, sigma_i^2) if no partner labelled item i); gamma_g is
     integrated exactly per group by Gauss-Hermite around its cavity posterior (prior N(0, sigma_g^2) if unseen).
  4. 1-D grid posterior over L; predictive for an evaluation item j integrates L, gamma_{g_j} (posterior given L when
     T has labels in g_j) and eps_bj (cavity) exactly, i.e. partner labels on item j still pool (on the sigma_i scale).
For an alone pair (no partner on b) this reproduces ExactAloneHier exactly (same grid and quadrature).
Requires tau_M = 0, nu_mu = 0, all_feature_tokens = False. Cost: one small cavity fit per shared pair per evidence
set plus a 241 x 24 grid per group. Numpy only; never raises (falls back to HierModel).
"""
from __future__ import annotations

import math

import numpy as np

from paiec.hier import HierModel, _entries, _group, _item_key, _sk

_GX, _GW = np.polynomial.hermite_e.hermegauss(24)
_GW = _GW / _GW.sum()
_LGW = np.log(_GW)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class ExactAllHier(HierModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self._pair_cache = {}
        self._cav_cache = {}

    # ----- cavity --------------------------------------------------------------------------------------
    def _cavity(self, labeled, s, b):
        key = (s, b)
        if key in self._cav_cache:
            return self._cav_cache[key]
        others = []
        for entry in labeled or []:
            try:
                (subject, item), _ = entry
                if str(item.get("benchmark_id", "")) == b and _sk(subject) != s:
                    others.append(entry)
            except (TypeError, ValueError):
                continue
        fit = self._fit_posterior(others) if others else None
        self._cav_cache[key] = fit
        return fit

    @staticmethod
    def _post(fit, name, prior):
        if fit is None or name not in fit["index"]:
            return prior
        j = fit["index"][name]
        return float(fit["mean"][j]), float(max(fit["cov"][j, j], 1e-9))

    # ----- exact pair posterior -----------------------------------------------------------------------
    def _pair_posterior(self, labeled, s, b, subject):
        p = self.p
        own = []
        for subj, item, y in _entries(labeled):
            if str(item.get("benchmark_id", "")) == b and _sk(subj) == s:
                own.append((_group(item), _item_key(item), y))
        cav = self._cavity(labeled, s, b)
        th, th_var = self._theta(subject)
        m_mu, v_mu = self._post(cav, ("mu", b), (p["m0"], p["tau_mu"] ** 2))
        m_c = m_mu + p["lam"] * th
        v_c = v_mu + p["tau_delta"] ** 2 + p["lam"] ** 2 * th_var + p["tau_phi"] ** 2
        pi = float(p.get("mix_pi", 0.0))
        sd_c = math.sqrt(v_c)
        lo = min(m_c - 7 * sd_c, p["mix_m"] - 7 * p["mix_s"]) if pi > 0 else m_c - 7 * sd_c
        hi = m_c + 7 * sd_c
        L = np.linspace(lo, hi, 241)
        logprior = -0.5 * (L - m_c) ** 2 / v_c - 0.5 * math.log(v_c)
        if pi > 0:
            lh = -0.5 * (L - p["mix_m"]) ** 2 / p["mix_s"] ** 2 - math.log(p["mix_s"])
            logprior = np.logaddexp(math.log(1 - pi) + logprior, math.log(pi) + lh)
        by_group = {}
        for g, ik, y in own:
            m_i, v_i = self._post(cav, ("eps", b, ik), (0.0, p["sigma_i"] ** 2))
            by_group.setdefault(g, []).append((m_i, 1.0 / math.sqrt(1.0 + math.pi * v_i / 8.0), y))
        loglik = np.zeros_like(L)
        group_w, group_nodes = {}, {}
        for g, labs in by_group.items():
            if g == "":
                for m_i, c_i, y in labs:
                    eta = c_i * (L + m_i)
                    loglik += -np.logaddexp(0.0, -eta) if y else -np.logaddexp(0.0, eta)
                continue
            m_g, v_g = self._post(cav, ("gamma", b, g), (0.0, p["sigma_g"] ** 2))
            gam = m_g + math.sqrt(v_g) * _GX
            ll = np.tile(_LGW[None, :], (len(L), 1))
            for m_i, c_i, y in labs:
                eta = c_i * (L[:, None] + gam[None, :] + m_i)
                ll = ll + (-np.logaddexp(0.0, -eta) if y else -np.logaddexp(0.0, eta))
            top = ll.max(axis=1, keepdims=True)
            loglik += top[:, 0] + np.log(np.exp(ll - top).sum(axis=1))
            w = np.exp(ll - top)
            group_w[g] = w / w.sum(axis=1, keepdims=True)
            group_nodes[g] = gam
        lp = logprior + loglik
        post = np.exp(lp - lp.max())
        post /= post.sum()
        return {"L": L, "post": post, "group_w": group_w, "group_nodes": group_nodes, "cav": cav}

    # ----- prediction ------------------------------------------------------------------------------------
    def predict(self, input, labeled=None) -> float:
        p = self.p
        if p["tau_M"] > 0 or float(p.get("nu_mu", 0.0)) > 0 or p.get("all_feature_tokens", False):
            return super().predict(input, labeled)
        try:
            key = self._evidence_key(labeled)
            if key != self._key:
                self._fit, self._key = self._fit_posterior(labeled), key
                self._pair_cache, self._cav_cache = {}, {}
            subject, item = input
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            if (s, b) not in self._pair_cache:
                self._pair_cache[(s, b)] = self._pair_posterior(labeled, s, b, subject)
            pp = self._pair_cache[(s, b)]
            L, post, cav = pp["L"], pp["post"], pp["cav"]
            m_j, v_j = self._post(cav, ("eps", b, _item_key(item)), (0.0, p["sigma_i"] ** 2))
            g = _group(item)
            if g and g in pp["group_w"]:
                c_j = 1.0 / math.sqrt(1.0 + math.pi * v_j / 8.0)
                gam = pp["group_nodes"][g]
                per_L = (pp["group_w"][g] * _sig(c_j * (L[:, None] + gam[None, :] + m_j))).sum(axis=1)
            else:
                m_g, v_g = (self._post(cav, ("gamma", b, g), (0.0, p["sigma_g"] ** 2)) if g else (0.0, 0.0))
                per_L = _sig((L + m_g + m_j) / math.sqrt(1.0 + math.pi * (v_j + v_g) / 8.0))
            prob = float(post @ per_L)
            if not math.isfinite(prob):
                return super().predict(input, labeled)
        except Exception:
            try:
                return super().predict(input, labeled)
            except Exception:
                prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))
