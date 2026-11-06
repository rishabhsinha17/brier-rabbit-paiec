"""Round 12 candidates F0 and F2 (submissions/QUEUE.md, round-12 amendment item 4): the subject prior on the full public
database, in the exact format of paiec.hier.fit_subject_prior, from the counts of swarm/round12/pair_rates.py.

F0 is fit_subject_prior's estimator. F2 solves the ridge objective F0 approximates in one step,
sum (y_ij - theta_i - beta_j)^2 + sum theta_i^2 over name i and benchmark j, so a benchmark's subject mix does not bias theta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paiec.data import SUBJECT_FIELDS  # noqa: E402
from paiec.subjects import attributes, ridge_fit  # noqa: E402

ATTR_SCALE = 1.56  # params.json attr_scale of the shipped composite
MIN_NAMES = 3  # item 3: a benchmark needs at least 3 named subjects to enter a fit
SIX_FIRST = {"matharena": 0, "multi_swebench": 1, "real_webagents": 2, "researchcodebench": 3, "swe_rebench": 4}


def prepare(rows: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    """Named pairs of the benchmarks that qualify, with the logit of each smoothed pair rate."""
    p = rows[(rows["key"] != "") & (rows["bench"] != exclude)].copy()
    names_in = p.groupby("bench")["key"].transform("nunique")
    p = p[names_in >= MIN_NAMES].copy()
    p["order"] = p["bench"].map(SIX_FIRST).fillna(9)
    p = p.sort_values(["order", "bench"], kind="stable")
    p["logit"] = np.log((p["s"] + 0.5) / (p["n"] - p["s"] + 0.5))
    return p


def per_name_bench(p: pd.DataFrame) -> pd.DataFrame:
    return p.groupby(["key", "bench"], sort=False)["logit"].mean().reset_index()


def f0_theta(p: pd.DataFrame, shrink: float = 1.0) -> dict:
    d = p.copy()
    d["centred"] = d["logit"] - d.groupby("bench")["logit"].transform("mean")
    per = d.groupby(["key", "bench"])["centred"].mean().reset_index()
    return {key: (float(g["centred"].sum() / (len(g) + shrink)), float(1.0 / (len(g) + shrink)))
            for key, g in per.groupby("key")}


def f2_theta(p: pd.DataFrame, shrink: float = 1.0, tol: float = 1e-9, max_iter: int = 5000) -> dict:
    d = per_name_bench(p)
    keys, ki = np.unique(d["key"].to_numpy(), return_inverse=True)
    benches, bi = np.unique(d["bench"].to_numpy(), return_inverse=True)
    y = d["logit"].to_numpy(float)
    k_i = np.bincount(ki, minlength=len(keys)).astype(float)
    n_b = np.bincount(bi, minlength=len(benches)).astype(float)
    theta, beta = np.zeros(len(keys)), np.bincount(bi, y, len(benches)) / n_b
    for _ in range(max_iter):
        new_theta = np.bincount(ki, y - beta[bi], len(keys)) / (k_i + shrink)
        new_beta = np.bincount(bi, y - new_theta[ki], len(benches)) / n_b
        done = max(np.abs(new_theta - theta).max(), np.abs(new_beta - beta).max()) < tol
        theta, beta = new_theta, new_beta
        if done:
            break
    else:
        raise RuntimeError("F2 did not converge")
    return {str(key): (float(t), float(1.0 / (k + shrink))) for key, t, k in zip(keys, theta, k_i)}


def fit(rows: pd.DataFrame, which: str, exclude: str | None = None, alpha: float = 1.0) -> dict:
    p = prepare(rows, exclude)
    names = (f0_theta if which == "F0" else f2_theta)(p)
    reps = p.drop_duplicates("key").set_index("key")
    X, y, w = [], [], []
    for key in sorted(names):
        theta, var = names[key]
        X.append(attributes({f: reps.at[key, f] for f in SUBJECT_FIELDS}))
        y.append(theta)
        w.append(1.0 / var)
    coef, rvar = ridge_fit(X, y, w, alpha=alpha)
    return {"names": {k: list(v) for k, v in names.items()}, "attr": {"coef": coef, "var": rvar}}


def coverage(ref: dict, full: dict, which: str) -> dict:
    """Round-12b candidates. E: the reference prior plus the names only the full data cover. E2: also F2's theta for the
    reference prior's names, keeping their var. The attribute ridge stays the reference one in both."""
    names = {k: list(v) for k, v in ref["names"].items()}
    for key, value in full["names"].items():
        if key not in names:
            names[key] = list(value)
        elif which == "E2":
            names[key] = [value[0], names[key][1]]
    if which == "E3":  # round-12c: E2 with every full-data theta times the constant S applies to attribute abilities
        names = {k: list(v) for k, v in ref["names"].items()}
        for key, value in full["names"].items():
            names[key] = [ATTR_SCALE * value[0], names[key][1] if key in names else value[1]]
    return {"names": names, "attr": ref["attr"]}
