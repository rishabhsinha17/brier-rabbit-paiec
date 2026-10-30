"""Likelihood power (alpha) on the Laplace fit: a cheap, all-pairs correction of the Laplace under-reaction
(postdoc_adaptation, swarm round 1).

Evidence: under the model's own prior the joint-mode Laplace + probit predictive under-reacts to a pair's labels
(head_researcher_B Check A: effective likelihood power 0.84 / 0.77 / 0.69 / 0.59 at n = 1 / 3 / 7 / 15; ALC excess
0.62e-3 for alone pairs). swarm/round1/postdoc_adaptation_sims/shared_laplace_excess.py shows the same error is LARGER
for pairs that share their benchmark (0.70e-3; e.g. 0/7 + partner 0/7: 0.087 vs exact 0.055), which the exact
alone-pair predictor (paiec/ext/hrB_exact.py) does not cover. Raising the likelihood to a power alpha > 1 inside the
Laplace fit sharpens the update for every pair, shared or alone, at no extra cost.

Change: log posterior = alpha * sum_labels loglik + log prior (gradient and Hessian scaled accordingly). The
zero-inflated mixture weights use the same tempered pair-level marginals (alpha on the Bernoulli terms), so both
components stay consistent. alpha = 1 reproduces HierModel exactly. Requires tau_M = 0, nu_mu = 0. Numpy only.
"""
from __future__ import annotations

import math

import numpy as np

from paiec.hier import _GH_LOGW, _GH_X, HierModel, _entries, _groups, _item_key, _sk


def _log_marginal_pow(k, n, m, v, alpha):
    eta = m + np.sqrt(max(v, 1e-9)) * _GH_X
    ll = alpha * (-k * np.logaddexp(0.0, -eta) - (n - k) * np.logaddexp(0.0, eta))
    a = ll + _GH_LOGW
    top = float(np.max(a))
    return top + float(np.log(np.sum(np.exp(a - top))))


def _post_pred_pow(k, n, m, v, item_var, alpha):
    eta = m + np.sqrt(max(v, 1e-9)) * _GH_X
    ll = alpha * (-k * np.logaddexp(0.0, -eta) - (n - k) * np.logaddexp(0.0, eta))
    a = ll + _GH_LOGW
    w = np.exp(a - np.max(a))
    w /= w.sum()
    return float(np.sum(w / (1.0 + np.exp(-eta / np.sqrt(1.0 + np.pi * item_var / 8.0)))))


class PowerHier(HierModel):
    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self.p.setdefault("lik_power", 1.0)

    def _fit_posterior(self, labeled):
        p = self.p
        alpha = float(p.get("lik_power", 1.0))
        if abs(alpha - 1.0) < 1e-12 or p["tau_M"] > 0 or float(p.get("nu_mu", 0.0)) > 0:
            return super()._fit_posterior(labeled)
        index, prior_mean, prior_var = {}, [], []

        def param(name, mean, var):
            if name not in index:
                index[name] = len(prior_mean)
                prior_mean.append(mean)
                prior_var.append(var)
            return index[name]

        rows, ys, offsets, counts = [], [], [], {}
        for subject, item, label in _entries(labeled):
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            th, th_var = self._theta(subject)
            phi_prior = (p["lam"] * th, p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var) if p["ability_param"] else (0.0, p["tau_phi"] ** 2)
            cols = [param(("mu", b), p["m0"], p["tau_mu"] ** 2), param(("phi", s), *phi_prior),
                    param(("delta", s, b), 0.0, p["tau_delta"] ** 2)]
            for g in _groups(item, p.get("all_feature_tokens", False)):
                cols.append(param(("gamma", b, g), 0.0, p["sigma_g"] ** 2))
            cols.append(param(("eps", b, _item_key(item)), 0.0, p["sigma_i"] ** 2))
            rows.append(cols)
            ys.append(label)
            kk, nn = counts.get((s, b), (0, 0))
            counts[(s, b)] = (kk + int(label), nn + 1)
            offsets.append(0.0 if p["ability_param"] else p["lam"] * th)
        n = len(prior_mean)
        if n == 0:
            return {"index": {}, "mean": np.zeros(0), "cov": np.zeros((0, 0)), "counts": {}}
        m0 = np.asarray(prior_mean)
        prec0 = 1.0 / np.asarray(prior_var)
        X = np.zeros((len(rows), n))
        for r, cols in enumerate(rows):
            X[r, cols] = 1.0
        y = np.asarray(ys)
        off = np.asarray(offsets)

        def log_post(v):
            e = np.clip(X @ v + off, -35.0, 35.0)
            return alpha * float(np.sum(y * e - np.logaddexp(0.0, e))) - 0.5 * float(np.sum(prec0 * (v - m0) ** 2))

        w = m0.copy()
        current = log_post(w)
        for _ in range(int(p.get("max_iters", 60))):
            eta = np.clip(X @ w + off, -35.0, 35.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            grad = alpha * (X.T @ (y - mu)) - prec0 * (w - m0)
            H = alpha * ((X.T * (mu * (1 - mu))) @ X) + np.diag(prec0)
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
        H = alpha * ((X.T * (mu * (1 - mu))) @ X) + np.diag(prec0)
        return {"index": index, "mean": w, "cov": np.linalg.inv(H), "counts": counts}

    def predict(self, input, labeled=None) -> float:
        p = self.p
        alpha = float(p.get("lik_power", 1.0))
        if abs(alpha - 1.0) < 1e-12 or p.get("mix_pi", 0.0) <= 0:
            return super().predict(input, labeled)  # mixture-free path: only the fit changes
        # Mixture path: normal component from the tempered fit, weights from tempered pair marginals.
        try:
            saved = p["mix_pi"]
            p["mix_pi"] = 0.0
            try:
                prob = super().predict(input, labeled)
            finally:
                p["mix_pi"] = saved
            subject, item = input
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            th, th_var = self._theta(subject)
            k, n_lab = (self._fit or {}).get("counts", {}).get((s, b), (0, 0))
            pi = p["mix_pi"]
            if n_lab:
                m_norm = p["m0"] + p["lam"] * th
                v_norm = p["tau_mu"] ** 2 + p["tau_delta"] ** 2 + p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var
                lh = _log_marginal_pow(k, n_lab, p["mix_m"], p["mix_s"] ** 2, alpha)
                ln = _log_marginal_pow(k, n_lab, m_norm, v_norm, alpha)
                w_hard = 1.0 / (1.0 + math.exp(math.log((1 - pi) / pi) + ln - lh))
                p_hard = _post_pred_pow(k, n_lab, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2, alpha)
            else:
                w_hard = pi
                p_hard = _post_pred_pow(0, 0, p["mix_m"], p["mix_s"] ** 2, p["sigma_i"] ** 2, alpha)
            prob = w_hard * p_hard + (1.0 - w_hard) * prob
            if not math.isfinite(prob):
                raise ValueError
        except Exception:
            try:
                return super().predict(input, labeled)
            except Exception:
                prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))
