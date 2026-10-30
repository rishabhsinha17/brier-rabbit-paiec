"""Shared helpers for the round-3 subject-shift study (public data only).

load()            light response frame (no item text) for the 5 binary benchmarks, same normalisation as paiec.data
pair_table(df)    one row per (bench, subject) pair: subject dict, k, m, n distinct items, rate, smoothed logit, flags
prior(df_train)   paiec.hier.fit_subject_prior on a training frame (the exact function the submission uses)
theta(prior, s)   (theta, var, source) exactly as HierModel._theta (source 'name' | 'attr' | 'none')
ATTR_NAMES        names of the 18 attribute features of paiec.subjects.attributes
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "swarm" / "round1" / "postdoc_priors_sims"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paiec.data import SUBJECT_FIELDS  # noqa: E402
from paiec.hier import fit_subject_prior  # noqa: E402
from paiec.subjects import PROVIDERS, attributes, name_key, predict_attr, _year  # noqa: E402

ATTR_NAMES = ["const", "year", "noyear", "effort", "think", "small", "large"] + [f"prov_{p}" for p in PROVIDERS]
LOBO = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
CACHE = HERE / "_cache_df.pkl"


def load():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    from light import load_light
    df = load_light()
    CACHE.write_bytes(pickle.dumps(df))
    return df


def pair_table(df):
    g = df.groupby(["bench", "pair"]).agg(n=("item_id", "nunique"), k=("label", "sum"), m=("label", "size")).reset_index()
    first = df.drop_duplicates("pair").set_index("pair")
    g["subject"] = [{f: first.at[p, f] for f in SUBJECT_FIELDS} for p in g.pair]
    g["r"] = g.k / g.m
    g["lg"] = np.log((g.k + 0.5) / (g.m - g.k + 0.5))
    g["eligible"] = (g.n >= 80) & (g.bench != "swe_rebench")
    g["key"] = [name_key(s) for s in g.subject]
    A = np.array([attributes(s) for s in g.subject])
    for j, nm in enumerate(ATTR_NAMES):
        g[nm] = A[:, j]
    g["rel"] = [_year(s.get("release_date", "")) for s in g.subject]
    g["acc"] = [_year(s.get("access_date", "")) for s in g.subject]
    g["harness"] = [s.get("harness", "") for s in g.subject]
    g["eff_raw"] = [s.get("reasoning_effort", "") for s in g.subject]
    return g


def prior(df_train, **kw):
    return fit_subject_prior(df_train, **kw)


def theta(pr, subj):
    key = name_key(subj)
    if key and key in pr["names"]:
        th, v = pr["names"][key]
        return float(th), float(v), "name"
    a = pr.get("attr")
    if a:
        return float(predict_attr(a["coef"], subj)), float(a["var"]), "attr"
    return 0.0, 0.25, "none"


def lobo_priors(df, benches=LOBO):
    path = HERE / "_cache_lobo_priors.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = {b: fit_subject_prior(df[df.bench != b]) for b in benches}
    out["ALL"] = fit_subject_prior(df)
    path.write_bytes(pickle.dumps(out))
    return out
