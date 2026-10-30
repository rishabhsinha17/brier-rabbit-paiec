"""Shared helpers for the round-2 plug-in review (skeptic_plugins). Public data only; no core file is touched.

build_cache(): one-time load of the public data -> _cache.pkl next to this file:
  raw[b]    = paiec.protocol.index_pairs(df[df.bench == b])      (identical structures to paiec.evaluate.FoldData)
  priors[b] = paiec.hier.fit_subject_prior(df[df.bench != b])    (LOBO subject prior, as experiments/paired.py)
item_content strings are interned so the pickle stays small; values are unchanged.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache.pkl")
BENCHES = ["real_webagents", "researchcodebench", "matharena", "multi_swebench"]
BASE = json.load(open(os.path.join(ROOT, "submission_src/v3b_noM_n_m0n20/params.json")))  # cur_final 963675
MIX25 = dict(BASE, m0=-1.0, mix_pi=0.25, mix_m=-4.0, mix_s=0.7)                              # live arm v4_mix25_m0n10
D06 = json.load(open(os.path.join(ROOT, "submission_src/v5_d06_m0n169/params.json")))      # live arm 964154


def build_cache():
    from paiec.data import load_responses
    from paiec.hier import fit_subject_prior
    from paiec.protocol import index_pairs

    t0 = time.time()
    df = load_responses()
    t1 = time.time()
    canon, raw = {}, {}
    for b in BENCHES:
        rps = index_pairs(df[df.bench == b])
        for rp in rps:
            for it in rp["items"].values():
                c = it["item"]["item_content"]
                it["item"]["item_content"] = canon.setdefault(c, c)
        raw[b] = rps
    priors = {b: fit_subject_prior(df[df.bench != b]) for b in BENCHES}
    with open(CACHE, "wb") as fh:
        pickle.dump({"raw": raw, "priors": priors}, fh, protocol=4)
    print(f"cache built: load {t1 - t0:.0f}s, total {time.time() - t0:.0f}s -> {CACHE}")


_C = {}


def load_cache():
    if not _C:
        with open(CACHE, "rb") as fh:
            _C.update(pickle.load(fh))
    return _C["raw"], _C["priors"]


def draws(bench: str, k: int, runs: int, seed0: int = 0, subsample: str = "pair"):
    """Exactly paiec.evaluate.FoldData.draws on the cached raw pairs (subsample "pair" = every existing result)."""
    from paiec.evaluate import FoldData

    raw, _ = load_cache()
    fd = FoldData.__new__(FoldData)
    fd.raw = raw
    fd.subsample = subsample
    return fd.draws(bench, k, runs, seed0) or []


def make(cls_path: str | None, prior, params: dict):
    import importlib

    from paiec.hier import HierModel

    if not cls_path:
        return HierModel(prior, dict(params))
    mod, name = cls_path.split(":")
    return getattr(importlib.import_module(mod), name)(prior, dict(params))


PLUGINS = {
    "subjcal": "paiec.ext.pd1_subjcal:SubjCalHier",
    "prior": "paiec.ext.pd1_prior:PriorCalHier",
    "power": "paiec.ext.pdadapt_power:PowerHier",
    "simpool": "paiec.ext.pdadapt_simpool:SimilarityPoolModel",
    "itempool": "paiec.ext.headA_itempool:ItemPoolModel",
    "hrb": "paiec.ext.hrB_exact:ExactAloneHier",
    "exactall": "paiec.ext.pdadapt_exactall:ExactAllHier",
    "bcm": "paiec.ext.pdadapt_bcm:MixturePoolModel",
}


if __name__ == "__main__":
    build_cache()
