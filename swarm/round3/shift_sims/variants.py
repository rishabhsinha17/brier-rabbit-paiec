"""Subject-prior variants for the shift study. Every variant maps (fold prior, subject) -> (theta, var, src, k).

  REF        HierModel._theta (canonical-name ability, else attribute ridge)
  A156       SubjCalHier: attr theta x1.56, release-year feature clamped at 2026.4 (round-1 E3, ship-ready)
  NR<r>      name shrink: theta (k+1)/(k+r), var r/(k+r)   (SubjCalHier name_r; r=1 is REF)
  NP         attribute ridge refit WITHOUT the 11 provider dummies (benchmark-specific in public data)
  PS<a>      provider dummies penalised with ridge alpha a (others alpha 1)
  AV<m>      attribute variance x m (moderation hedge for unknown names)
  SY<f>      release-year coefficient at f x its base value while the rest of the ridge keeps 'scale'
             (A156+SY1.0 = the x1.56 dispersion fix without amplifying vintage extrapolation)
Variants compose with '+', e.g. 'A156+NR2+NP'. The refit reproduces fit_subject_prior's ridge exactly when nothing is
dropped (checked in selftest()).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import shiftlib as sl  # noqa: E402
from paiec.data import SUBJECT_FIELDS  # noqa: E402
from paiec.subjects import attributes, name_key  # noqa: E402

PROV_IDX = list(range(7, 18))


def ridge_design(train_df):
    """X, y, w exactly as fit_subject_prior builds them (names table first)."""
    pairs = train_df.drop_duplicates("pair")[["bench", "pair", *SUBJECT_FIELDS]].copy()
    pairs["key"] = [name_key({f: r[f] for f in SUBJECT_FIELDS}) for _, r in pairs.iterrows()]
    rates = train_df.groupby("pair")["label"].agg(["sum", "count"])
    pairs = pairs.join(rates, on="pair")
    pairs = pairs[pairs["key"] != ""]
    pairs["logit"] = np.log((pairs["sum"] + 0.5) / (pairs["count"] - pairs["sum"] + 0.5))
    pairs["centred"] = pairs["logit"] - pairs.groupby("bench")["logit"].transform("mean")
    per = pairs.groupby(["key", "bench"])["centred"].mean().reset_index()
    reps = pairs.drop_duplicates("key").set_index("key")
    X, y, w = [], [], []
    for key, g in per.groupby("key"):
        k = len(g)
        X.append(attributes({f: reps.at[key, f] for f in SUBJECT_FIELDS}))
        y.append(float(g["centred"].sum() / (k + 1.0)))
        w.append(k + 1.0)
    return np.array(X), np.array(y), np.array(w)


def ridge_alpha(X, y, w, alpha_vec):
    """Weighted ridge with a per-column penalty (alpha 0 = unpenalised; np.inf = column dropped)."""
    keep = np.isfinite(alpha_vec)
    Xk = X[:, keep]
    A = (Xk.T * w) @ Xk + np.diag(np.asarray(alpha_vec)[keep])
    b = (Xk.T * w) @ y
    ck = np.linalg.solve(A, b)
    coef = np.zeros(X.shape[1])
    coef[keep] = ck
    resid = y - X @ coef
    var = max(float((w * resid ** 2).sum() / w.sum()), 1e-3)
    return coef, var


class Fold:
    """Per-fold prior plus lazily refitted ridge variants."""

    def __init__(self, prior, train_df):
        self.prior = prior
        self.train_df = train_df
        self._design = None
        self._coef = {}

    def coef(self, tag):
        if tag == "REF":
            return np.asarray(self.prior["attr"]["coef"]), float(self.prior["attr"]["var"])
        if tag not in self._coef:
            if self._design is None:
                self._design = ridge_design(self.train_df)
            X, y, w = self._design
            alpha = np.ones(X.shape[1])
            alpha[0] = 0.0
            if tag == "NP":
                alpha[PROV_IDX] = np.inf
            elif tag.startswith("PS"):
                alpha[PROV_IDX] = float(tag[2:])
            self._coef[tag] = ridge_alpha(X, y, w, alpha)
        return self._coef[tag]


def parse(spec):
    parts = spec.split("+") if spec else ["REF"]
    o = {"scale": 1.0, "ymax": 0.0, "name_r": 1.0, "ridge": "REF", "avar": 1.0, "ymult": None}
    for p in parts:
        if p == "REF":
            continue
        if p == "A156":
            o["scale"], o["ymax"] = 1.56, 2026.4
        elif p.startswith("NR"):
            o["name_r"] = float(p[2:])
        elif p == "NP" or p.startswith("PS"):
            o["ridge"] = p
        elif p.startswith("AV"):
            o["avar"] = float(p[2:])
        elif p.startswith("YM"):  # explicit year clamp, e.g. YM2025.5
            o["ymax"] = float(p[2:])
        elif p.startswith("SC"):  # explicit attr scale, e.g. SC1.3
            o["scale"] = float(p[2:])
        elif p.startswith("SY"):  # effective multiplier of the release-year coefficient (others keep 'scale')
            o["ymult"] = float(p[2:])
        else:
            raise ValueError(p)
    return o


def theta(fold: Fold, subj, spec="REF"):
    o = parse(spec)
    pr = fold.prior
    key = name_key(subj)
    if key and key in pr["names"]:
        th, var = pr["names"][key]
        k = max(1.0, round(1.0 / max(var, 1e-9) - 1.0))
        r = o["name_r"]
        return th * (k + 1.0) / (k + r), r / (k + r), "name", k
    coef, var = fold.coef(o["ridge"])
    x = attributes(subj)
    if o["ymax"]:
        x[1] = min(x[1], o["ymax"] - 2025.0)
    t = o["scale"] * float(np.dot(coef, x))
    if o["ymult"] is not None:
        t += (o["ymult"] - o["scale"]) * float(coef[1]) * x[1]
    return t, o["avar"] * var, "attr", 0


def selftest():
    df = sl.load()
    P = sl.lobo_priors(df)
    worst = 0.0
    for b in sl.LOBO:
        tr = df[df.bench != b]
        X, y, w = ridge_design(tr)
        a = np.ones(X.shape[1]); a[0] = 0.0
        c, v = ridge_alpha(X, y, w, a)
        worst = max(worst, float(np.max(np.abs(c - np.asarray(P[b]["attr"]["coef"])))), abs(v - P[b]["attr"]["var"]))
    print(f"selftest: max |refit - fit_subject_prior| over 4 folds = {worst:.2e}")
    return worst


if __name__ == "__main__":
    selftest()
