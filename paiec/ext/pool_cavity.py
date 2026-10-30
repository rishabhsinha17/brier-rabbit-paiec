"""Cavity cross-subject item pooling (postdoc_pooling, swarm round 2).

Why HierModel under-pools. One item variance sigma_i (0.6) does three jobs in one joint Laplace fit:
  (a) it is the prior of an eps for every label of the target's OWN acquisition items, which are never evaluated, so
      a large sigma_i only lets those eps absorb the pair's own labels (joint-mode / PQL shrinkage: the pair level
      learns more slowly; tune_hier_v3: sigma_i 1.8 cost +0.0009, 2.4 cost +0.0027);
  (b) it moderates every evaluation item nobody labelled (sigma(m / sqrt(1 + pi (v + sigma_i^2) / 8)));
  (c) it is the only channel through which a partner's label on the target's evaluation item moves that prediction.
Tuning on a replay where (c) is rare (pair-keyed item subsample, k <= 2) therefore picks a small sigma_i for (a) and
(b), and (c) is left at an implied cross-subject ICC of about 0.10 (public crossed-Rasch ICC 0.66-0.75, item SD
2.5-3.1 logits).

This class keeps HierModel bit-for-bit for every prediction whose item no OTHER subject has labelled, and for an
evaluation item j that other subjects labelled it replaces the weak eps_j pooling by an exact 1-D posterior over a
shared item effect u_j ~ N(0, sigma_x^2) on the conditional (item-level) scale:
  1. Cavity. All labels on item j (same benchmark_id and item content) are removed from the fitted Laplace posterior by
     a rank-m Woodbury downdate (approximate leave-these-out: Sigma_cav = Sigma + Sigma X' (H^-1 - X Sigma X')^-1 X Sigma,
     w_cav = w - Sigma_cav X' (y - p)). Without it each partner's level already contains its own label on j, which
     explains part of that label away (an under-update that is largest at B1-B7, where a partner has few labels).
  2. Every labeller s of j gets its marginal-scale level (mu_b + phi_s + delta_sb + gamma + lam*theta_s, no eps) as
     cavity mean/variance (m_s, v_s), mapped to the conditional scale by L_c = L_m / kappa, v_c = v_m / kappa^2,
     kappa = 1 / sqrt(1 + pi sigma_x^2 / 8) (inverse probit approximation). Its likelihood term is
     sigma((m_s/kappa + u) / sqrt(1 + pi v_s / (8 kappa^2))): the partner's level uncertainty is integrated, which is
     what keeps a B1 partner (level barely known) from moving the item much.
  3. The target's own cavity level gives p_item = E[sigma(L_A,c + u) | labels on j] and p_prior (same under the prior
     of u), each on a 40-node Gauss-Hermite grid; the normal-component prediction is
     sigma(logit(p_base) + pool_shrink * (logit(p_item) - logit(p_prior))), p_base = HierModel's prediction for an
     unlabelled item (level without eps_j, variance + sigma_i^2). So with no informative labels it equals the base.
The optional zero-inflated pair mixture of HierModel is applied afterwards, unchanged.

Items are matched on the full text (sha1 of item_content within a benchmark_id), not on HierModel._item_key (length +
first/last 2048 characters, which merges 3 of 212 public ResearchCodeBench items). Labels by the target itself on the
same text (duplicated item texts, e.g. MathArena competition aliases) join the item posterior only when another subject
also labelled that text; otherwise the item is left to HierModel (identity contract).

Params: sigma_x (2.0; 2.5 recommended), pool_shrink (1.0), pool_cavity (True; False uses the full-fit partner levels,
as headA_itempool.ItemPoolModel does), pool_mode ("exact"; "map" = aghasalim-style point offset: MAP of b_j with prior
sd sigma_x given plug-in partner logits, added to the base logit, no moderation), pool_max_labels (64 per item).
Numpy only; never raises (falls back to HierModel, then to sigma(m0)).

Local evidence (swarm/round2/postdoc_pooling.md): benchmark-keyed replay, k=2, ALC vs cur_final at sigma_x 2.0/3.0:
RCB -0.0008/-0.0009, Multi-SWE -0.0009/-0.0010, MathArena -0.0012/-0.0014, REAL +0.0000/+0.0002 per shared pair;
identical to the base at B0 and for every pair alone on its benchmark.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from paiec.hier import HierModel, _entries, _groups, _item_key, _log_marginal, _posterior_predictive, _sk

_GX, _GW = np.polynomial.hermite_e.hermegauss(40)
_GLW = np.log(_GW / np.sqrt(2.0 * np.pi))
_PW = np.exp(_GLW - _GLW.max())
_PW = _PW / _PW.sum()


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


def _logsig(z):
    return -np.logaddexp(0.0, -z)


def _full_key(item) -> str:
    """Pooling key: sha1 of the FULL item text. HierModel._item_key hashes length + first/last 2048 characters, which
    merges different long items that share a preamble and an ending (3 of 212 ResearchCodeBench items collide)."""
    return hashlib.sha1(str(item.get("item_content", "")).encode("utf-8", "replace")).hexdigest()


class CavityPoolModel(HierModel):
    """HierModel with cavity-corrected cross-subject item pooling at a separate item variance sigma_x."""

    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        p = self.p
        p.setdefault("sigma_x", 2.0)
        p.setdefault("pool_shrink", 1.0)
        p.setdefault("pool_cavity", True)
        p.setdefault("pool_mode", "exact")
        p.setdefault("pool_max_labels", 64)
        self._lab_key = None
        self._lab_index = {}
        self._shift_cache = {}

    # ----- evidence helpers ---------------------------------------------------------------------------
    def _label_index(self, labeled):
        """(benchmark_id, item_key) -> list of (subject_key, subject, item, y); rebuilt when the evidence changes."""
        if self._lab_key != self._key:
            index = {}
            for subject, item, y in _entries(labeled):
                index.setdefault((str(item.get("benchmark_id", "")), _full_key(item)), []).append(
                    (_sk(subject), subject, item, y))
            self._lab_index, self._lab_key, self._shift_cache = index, self._key, {}
        return self._lab_index

    def _level_vec(self, fit, subject, b, item):
        """(column indices, prior mean, prior variance) of the pair level for this item, excluding eps.

        Mirrors HierModel.predict: components present in the fit are read from it, the rest contribute their prior."""
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
        cols = []
        for nm in names:
            j = fit["index"].get(nm)
            if j is None:
                mpr, vpr = prior[nm[0]]
                mean += mpr
                var += vpr
            else:
                cols.append(j)
        return cols, mean, var

    def _row(self, fit, subject, item, b):
        """Fit columns and offset of one evidence label (the row HierModel._fit_posterior built for it)."""
        p = self.p
        s = _sk(subject)
        names = [("mu", b), ("phi", s), ("delta", s, b)]
        if p["tau_M"] > 0:
            names.append(("M",))
        for g in _groups(item, p.get("all_feature_tokens", False)):
            names.append(("gamma", b, g))
        names.append(("eps", b, _item_key(item)))
        cols = [fit["index"][nm] for nm in names]
        off = 0.0 if p["ability_param"] else p["lam"] * self._theta(subject)[0]
        return cols, off

    # ----- the item shift -------------------------------------------------------------------------------
    def _shift(self, fit, subject, item, b, labs):
        """(logit shift for the target on this item, target cavity level mean, var) or None."""
        p = self.p
        w, S = fit["mean"], fit["cov"]
        labs = labs[: int(p.get("pool_max_labels", 64))]
        rows = [self._row(fit, sd, it, b) for _, sd, it, _ in labs]
        ys = np.array([float(y) for _, _, _, y in labs])
        eta = np.array([float(w[c].sum()) + off for c, off in rows])
        pr = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        h = np.maximum(pr * (1.0 - pr), 1e-9)
        m = len(rows)
        # Sigma X_J' (n x m) and G = X_J Sigma X_J' (m x m)
        SX = np.stack([S[:, c].sum(axis=1) for c, _ in rows], axis=1)
        Minv = None
        if p.get("pool_cavity", True):
            G = np.array([[float(SX[c, k].sum()) for k in range(m)] for c, _ in rows])
            M = np.diag(1.0 / h) - G
            M = 0.5 * (M + M.T)
            try:
                np.linalg.cholesky(M)
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                Minv = None
        rh = (ys - pr) / h

        def level(sd, it):  # level of subject sd on its own labelled copy `it` of the text (groups can differ)
            cols, mean, var = self._level_vec(fit, sd, b, it)
            if cols:
                mean += float(w[cols].sum())
                var += float(S[np.ix_(cols, cols)].sum())
                if Minv is not None:
                    t = SX[cols].sum(axis=0)
                    mean -= float(t @ Minv @ rh)
                    var += float(t @ Minv @ t)
            return mean, max(var, 1e-9)

        sx = float(p["sigma_x"])
        m_a, v_a = level(subject, item)
        if p.get("pool_mode", "exact") == "map":
            # aghasalim-style: MAP offset given plug-in partner logits (full fit, no cavity), no moderation.
            fs = np.array([self._plain_level(fit, sd, b, it)[0] for _, sd, it, _ in labs])
            x = 0.0
            for _ in range(20):
                q = 1.0 / (1.0 + np.exp(-np.clip(fs + x, -35.0, 35.0)))
                g = float(np.sum(ys - q)) - x / sx ** 2
                hh = -float(np.sum(q * (1.0 - q))) - 1.0 / sx ** 2
                step = g / hh
                x -= step
                if abs(step) < 1e-9:
                    break
            return x, m_a, v_a
        kappa = 1.0 / math.sqrt(1.0 + math.pi * sx * sx / 8.0)
        u = sx * _GX
        logw = _GLW.copy()
        for (_, sd, it, _), y in zip(labs, ys):
            m_s, v_s = level(sd, it)
            z = (m_s / kappa + u) / math.sqrt(1.0 + math.pi * (v_s / kappa ** 2) / 8.0)
            logw = logw + (_logsig(z) if y > 0.5 else _logsig(-z))
        wt = np.exp(logw - logw.max())
        wt /= wt.sum()
        za = (m_a / kappa + u) / math.sqrt(1.0 + math.pi * (v_a / kappa ** 2) / 8.0)
        sig = 1.0 / (1.0 + np.exp(-za))
        p_item, p_prior = float(np.sum(wt * sig)), float(np.sum(_PW * sig))
        return _logit(p_item) - _logit(p_prior), m_a, v_a

    def _plain_level(self, fit, subject, b, item):
        cols, mean, var = self._level_vec(fit, subject, b, item)
        if cols:
            mean += float(fit["mean"][cols].sum())
            var += float(fit["cov"][np.ix_(cols, cols)].sum())
        return mean, var

    # ----- prediction -----------------------------------------------------------------------------------
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
            ik = _full_key(item)
            labs = self._label_index(labeled).get((b, ik), [])
            if not any(sk != s for sk, _, _, _ in labs):
                return super().predict(input, labeled)
            ck = (s, b, ik, tuple(_groups(item, p.get("all_feature_tokens", False))))  # P4: target group sets the shift
            if ck not in self._shift_cache:
                self._shift_cache[ck] = self._shift(fit, subject, item, b, labs)
            shift = self._shift_cache[ck][0]
            m_full, v_full = self._plain_level(fit, subject, b, item)
            p_base = 1.0 / (1.0 + math.exp(-m_full / math.sqrt(1.0 + math.pi * (v_full + p["sigma_i"] ** 2) / 8.0)))
            prob = 1.0 / (1.0 + math.exp(-(_logit(p_base) + p["pool_shrink"] * shift)))
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
