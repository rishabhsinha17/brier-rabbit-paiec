"""v1: hierarchical logistic model with a Laplace posterior refit from `labeled` on every new evidence set.

logit P(correct | subject s, benchmark b, item i) =
    mu_b                  benchmark level, prior N(m0, tau_mu^2)
  + lam * theta_s         transferred general ability (training estimate; 0 for unknown names)
  + phi_s                 subject offset shared across benchmarks in the draw, prior N(0, tau_phi^2)
  + delta_sb              pair deviation, prior N(0, tau_delta^2)
  + gamma_bg              item_features group effect inside the benchmark, prior N(0, sigma_g^2)
  + eps_bi                item effect learned from other subjects' labels on the same item, prior N(0, sigma_i^2)

Prediction is the posterior predictive mean via the probit approximation sigma(m / sqrt(1 + pi v / 8)), where v
includes the prior variance of every effect that has no evidence yet (e.g. an unseen item's eps_bi).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

try:  # inside a submission the helpers are vendored into the same file
    from .subjects import attributes, name_key, predict_attr, ridge_fit
except ImportError:  # pragma: no cover
    pass

DEFAULTS = {
    "m0": -0.5, "tau_mu": 1.0, "lam": 0.6, "tau_theta": 0.5, "tau_phi": 0.5,
    "tau_delta": 0.7, "sigma_g": 0.5, "sigma_i": 1.2, "newton_iters": 12, "clip": 0.005,
    # Optional third level: benchmark levels share an unknown mean M ~ N(m0, tau_M^2) that every benchmark's labels
    # in the draw update. 0 disables it (then mu_b ~ N(m0, tau_mu^2) as before).
    "tau_M": 0.0,
    # When true, subject ability is a fitted random effect (prior mean lam*theta, var tau_phi^2 + lam^2 var_theta)
    # that revealed labels update; when false (v1/v2 zips) it is a fixed offset whose variance is never reduced.
    "ability_param": False,
    # Student-t prior (nu degrees of freedom) on each benchmark level mu_b instead of a normal; 0 keeps the normal.
    # Heavy tails let labels override the prior quickly for outlier benchmarks.
    "nu_mu": 0.0,
    # Use every k=v token of item_features as its own group effect (prior sigma_g each); False = first token only.
    "all_feature_tokens": False,
    # Zero-inflated pair level ("hard" component): with prior probability mix_pi a pair is near-impossible, its level
    # ~ N(mix_m, mix_s^2). The pair's own labels give the posterior weight via 1-D quadrature; the prediction mixes the
    # hard component with the full hierarchical model. mix_pi = 0 disables it.
    "mix_pi": 0.0, "mix_m": -4.0, "mix_s": 0.7,
}

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(48)
_GH_LOGW = np.log(_GH_W / np.sqrt(2.0 * np.pi))


def _log_marginal(k: int, n: int, m: float, v: float) -> float:
    """log P(k successes in n Bernoulli trials | logit level ~ N(m, v)) via Gauss-Hermite quadrature."""
    eta = m + np.sqrt(max(v, 1e-9)) * _GH_X
    ll = -k * np.logaddexp(0.0, -eta) - (n - k) * np.logaddexp(0.0, eta)
    a = ll + _GH_LOGW
    top = float(np.max(a))
    return top + float(np.log(np.sum(np.exp(a - top))))


def _posterior_predictive(k: int, n: int, m: float, v: float, item_var: float) -> float:
    eta = m + np.sqrt(max(v, 1e-9)) * _GH_X
    ll = -k * np.logaddexp(0.0, -eta) - (n - k) * np.logaddexp(0.0, eta)
    a = ll + _GH_LOGW
    w = np.exp(a - np.max(a))
    w /= w.sum()
    return float(np.sum(w / (1.0 + np.exp(-eta / np.sqrt(1.0 + np.pi * item_var / 8.0)))))


def _sk(subject) -> str:
    return json.dumps(subject, sort_keys=True, separators=(",", ":"), default=str)


def _group(item) -> str:
    feats = str(item.get("item_features", "") or "")
    # Use the first k=v token as the group (e.g. lang=python, website=staynb, paper=...); "" if none.
    return feats.split(";")[0].strip() if feats else ""


def _groups(item, all_tokens: bool) -> list:
    if not all_tokens:
        g = _group(item)
        return [g] if g else []
    feats = str(item.get("item_features", "") or "")
    return [t.strip() for t in feats.split(";") if t.strip()]


def _entries(labeled):
    """Yield (subject, item, y) for well-formed evidence entries; skip anything malformed."""
    for entry in labeled or []:
        try:
            (subject, item), label = entry
            if isinstance(subject, dict) and isinstance(item, dict):
                yield subject, item, 1.0 if int(label) else 0.0
        except (TypeError, ValueError):
            continue


def _item_key(item) -> str:
    # Head + tail + length: cheap for 100k-character items, collision-safe in practice.
    text = str(item.get("item_content", ""))
    return hashlib.sha1(f"{len(text)}|{text[:2048]}|{text[-2048:]}".encode()).hexdigest()


class HierModel:
    def __init__(self, abilities: dict | None = None, params: dict | None = None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.abilities = abilities or {}  # normalized_name -> (theta, var)
        self._key = None
        self._fit = None

    # ----- evidence -> posterior -------------------------------------------------------------
    def _theta(self, subject):
        prior = self.abilities
        if "names" in prior:  # subject prior from fit_subject_prior: canonical names + attribute regression
            key = name_key(subject)
            if key and key in prior["names"]:
                return tuple(prior["names"][key])
            attr = prior.get("attr")
            if attr:
                return predict_attr(attr["coef"], subject), attr["var"]
            return 0.0, self.p["tau_theta"] ** 2
        name = str(subject.get("normalized_name", "") or "")
        if name in prior:
            return prior[name]
        return 0.0, self.p["tau_theta"] ** 2

    def _evidence_key(self, labeled) -> str:
        h = hashlib.sha1()
        for subject, item, y in _entries(labeled):
            h.update(_sk(subject).encode())
            h.update(str(item.get("benchmark_id", "")).encode())
            h.update(_item_key(item).encode())
            h.update(b"1" if y else b"0")
        return h.hexdigest()

    def _fit_posterior(self, labeled):
        p = self.p
        index, prior_mean, prior_var = {}, [], []

        def param(name, mean, var):
            if name not in index:
                index[name] = len(prior_mean)
                prior_mean.append(mean)
                prior_var.append(var)
            return index[name]

        rows, ys, offsets = [], [], []
        counts = {}
        shared = p["tau_M"] > 0
        for subject, item, label in _entries(labeled):
            b = str(item.get("benchmark_id", ""))
            s = _sk(subject)
            th, th_var = self._theta(subject)
            if p["ability_param"]:
                phi_prior = (p["lam"] * th, p["tau_phi"] ** 2 + p["lam"] ** 2 * th_var)
            else:
                phi_prior = (0.0, p["tau_phi"] ** 2)
            cols = [
                param(("mu", b), 0.0 if shared else p["m0"], p["tau_mu"] ** 2),
                param(("phi", s), *phi_prior),
                param(("delta", s, b), 0.0, p["tau_delta"] ** 2),
            ]
            if shared:
                cols.append(param(("M",), p["m0"], p["tau_M"] ** 2))
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
        nu = float(p.get("nu_mu", 0.0))
        t_idx = np.array([j for name, j in index.items() if name[0] == "mu"], dtype=int) if nu > 0 else np.zeros(0, int)
        t_var = np.asarray(prior_var)[t_idx] if nu > 0 else None
        X = np.zeros((len(rows), n))
        for r, cols in enumerate(rows):
            X[r, cols] = 1.0
        y = np.asarray(ys)
        off = np.asarray(offsets)
        w = m0.copy()
        if not p.get("damped_newton", False):
            # Original undamped Newton (kept bit-exact for the v1/v2 zips). It can diverge when labels strongly
            # contradict the prior; new submissions set damped_newton.
            for _ in range(int(p["newton_iters"])):
                if nu > 0 and len(t_idx):
                    prec0[t_idx] = (nu + 1.0) / (nu * t_var + (w[t_idx] - m0[t_idx]) ** 2)
                eta = X @ w + off
                mu = 1.0 / (1.0 + np.exp(-eta))
                grad = X.T @ (y - mu) - prec0 * (w - m0)
                H = (X.T * (mu * (1 - mu))) @ X + np.diag(prec0)
                step = np.linalg.solve(H, grad)
                w = w + step
                if np.max(np.abs(step)) < 1e-6:
                    break
            eta = X @ w + off
            mu = 1.0 / (1.0 + np.exp(-eta))
            H = (X.T * (mu * (1 - mu))) @ X + np.diag(prec0)
            return {"index": index, "mean": w, "cov": np.linalg.inv(H), "counts": counts}

        base_prec = prec0.copy()
        normal = np.ones(n, dtype=bool)
        normal[t_idx] = False

        def log_post(v):
            e = np.clip(X @ v + off, -35.0, 35.0)
            d = v - m0
            val = float(np.sum(y * e - np.logaddexp(0.0, e))) - 0.5 * float(np.sum(base_prec[normal] * d[normal] ** 2))
            if len(t_idx):
                val -= 0.5 * (nu + 1.0) * float(np.sum(np.log1p(d[t_idx] ** 2 / (nu * t_var))))
            return val

        current = log_post(w)
        for _ in range(int(p.get("max_iters", 60))):
            if len(t_idx):  # scale-mixture (IRLS) precision for the t prior
                prec0[t_idx] = (nu + 1.0) / (nu * t_var + (w[t_idx] - m0[t_idx]) ** 2)
            eta = np.clip(X @ w + off, -35.0, 35.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            grad = X.T @ (y - mu) - prec0 * (w - m0)
            H = (X.T * (mu * (1 - mu))) @ X + np.diag(prec0)
            step = np.linalg.solve(H, grad)
            scale = 1.0
            while True:  # step halving: never accept a step that lowers the log posterior
                cand = w + scale * step
                value = log_post(cand)
                if value >= current - 1e-10 or scale < 1e-4:
                    break
                scale *= 0.5
            moved = float(np.max(np.abs(cand - w)))
            w, current = cand, value
            if moved < 1e-7:
                break
        if len(t_idx):
            prec0[t_idx] = (nu + 1.0) / (nu * t_var + (w[t_idx] - m0[t_idx]) ** 2)
        eta = np.clip(X @ w + off, -35.0, 35.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        H = (X.T * (mu * (1 - mu))) @ X + np.diag(prec0)
        return {"index": index, "mean": w, "cov": np.linalg.inv(H), "counts": counts}

    # ----- prediction --------------------------------------------------------------------------
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
            if p.get("mix_pi", 0.0) > 0:
                k, n_lab = fit.get("counts", {}).get((s, b), (0, 0))
                pi = p["mix_pi"]
                if n_lab:
                    # Pair-level 1-D marginal likelihoods under each component (prior only; consistent for both).
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
        except Exception:
            prob = 1.0 / (1.0 + math.exp(-p["m0"]))
        return float(min(1 - p["clip"], max(p["clip"], prob)))


def fit_abilities(train_df, shrink: float = 1.0) -> dict:
    """General ability per normalized_name: centred logit of smoothed pair rates, averaged over benchmarks.

    Returns name -> (theta, var). var shrinks with the number of benchmarks the name appears in.
    """
    rates = train_df.groupby(["bench", "pair", "normalized_name"])["label"].agg(["sum", "count"]).reset_index()
    rates = rates[rates["normalized_name"] != ""]
    rates["logit"] = np.log((rates["sum"] + 0.5) / (rates["count"] - rates["sum"] + 0.5))
    rates["centred"] = rates["logit"] - rates.groupby("bench")["logit"].transform("mean")
    per_name = rates.groupby(["normalized_name", "bench"])["centred"].mean().reset_index()
    out = {}
    for name, g in per_name.groupby("normalized_name"):
        k = len(g)
        theta = float(g["centred"].sum() / (k + shrink))
        out[name] = (theta, float(1.0 / (k + shrink)))
    return out


def fit_subject_prior(train_df, shrink: float = 1.0, alpha: float = 1.0) -> dict:
    """Abilities keyed by canonical name (recovers empty normalized_name from source_model_name), plus a ridge
    regression from subject attributes to ability for models never seen in training."""
    from .data import SUBJECT_FIELDS

    pairs = train_df.drop_duplicates("pair")[["bench", "pair", *SUBJECT_FIELDS]].copy()
    pairs["key"] = [name_key({f: r[f] for f in SUBJECT_FIELDS}) for _, r in pairs.iterrows()]
    rates = train_df.groupby("pair")["label"].agg(["sum", "count"])
    pairs = pairs.join(rates, on="pair")
    pairs = pairs[pairs["key"] != ""]
    pairs["logit"] = np.log((pairs["sum"] + 0.5) / (pairs["count"] - pairs["sum"] + 0.5))
    pairs["centred"] = pairs["logit"] - pairs.groupby("bench")["logit"].transform("mean")
    per = pairs.groupby(["key", "bench"])["centred"].mean().reset_index()
    names = {}
    for key, g in per.groupby("key"):
        k = len(g)
        names[key] = (float(g["centred"].sum() / (k + shrink)), float(1.0 / (k + shrink)))
    reps = pairs.drop_duplicates("key").set_index("key")
    X, y, w = [], [], []
    for key, (theta, var) in names.items():
        subj = {f: reps.at[key, f] for f in SUBJECT_FIELDS}
        X.append(attributes(subj))
        y.append(theta)
        w.append(1.0 / var)
    coef, rvar = ridge_fit(X, y, w, alpha=alpha)
    return {"names": names, "attr": {"coef": coef, "var": rvar}}
