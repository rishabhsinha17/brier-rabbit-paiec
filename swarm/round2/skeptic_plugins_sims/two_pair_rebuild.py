"""Rebuild round-1 two-pair prediction tables (swarm/round1/postdoc_adaptation_sims/two_pair_tables.py logic) from the
CURRENT plug-in code into ./tables, and compare with the round-1 tables of the same name. Score them afterwards with
  TPE_TAB=swarm/round2/skeptic_plugins_sims/tables TPE_SCEN=compact python swarm/round1/postdoc_adaptation_sims/two_pair_exact.py cur_m20
usage: python two_pair_rebuild.py name1 name2 ...   (names from SPECS)
"""
from __future__ import annotations

import importlib
import os
import sys
import time

sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np  # noqa: E402

from paiec.hier import HierModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")
R1 = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round1/postdoc_adaptation_sims/tables")
BUD = (0, 1, 3, 7, 15, 31)
CUR = {"tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5, "tau_theta": 0.5,
       "damped_newton": True, "m0": -2.0}
MIX = dict(CUR, m0=-1.0, mix_pi=0.25, mix_m=-4.0, mix_s=0.7)
SPECS = {
    "cur_m20": (None, CUR),
    "pow_m20_a1.3": ("paiec.ext.pdadapt_power:PowerHier", dict(CUR, lik_power=1.3)),
    "exall_m20": ("paiec.ext.pdadapt_exactall:ExactAllHier", CUR),
    "mix25_m10": (None, MIX),
    "pow13_mix25_m10": ("paiec.ext.pdadapt_power:PowerHier", dict(MIX, lik_power=1.3)),
    "d06_m169": (None, dict(CUR, tau_delta=0.6, tau_phi=0.3, m0=-1.693)),
    "pow13_d06_m169": ("paiec.ext.pdadapt_power:PowerHier", dict(CUR, tau_delta=0.6, tau_phi=0.3, m0=-1.693, lik_power=1.3)),
    "cur_m25": (None, dict(CUR, m0=-2.5)),
    "pow13_m25": ("paiec.ext.pdadapt_power:PowerHier", dict(CUR, m0=-2.5, lik_power=1.3)),
}


def subj(name):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}


def item(i):
    return {"item_content": f"task {i}", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}


T, P = subj("unknown-target-x"), subj("unknown-partner-y")


def lab(s, k, n, off):
    return [[[s, item(off + i)], int(i < k)] for i in range(n)]


def build(name):
    cls_path, params = SPECS[name]
    cls = HierModel
    if cls_path:
        mod, c = cls_path.split(":")
        cls = getattr(importlib.import_module(mod), c)
    m = cls({}, dict(params))
    tabs = {}
    for n in BUD:
        q = np.zeros((n + 1, n + 1))
        for kT in range(n + 1):
            for kP in range(n + 1):
                q[kT, kP] = m.predict([T, item(10 ** 6)], lab(T, kT, n, 0) + lab(P, kP, n, 1000))
        tabs[f"n{n}"] = q
    for n in BUD:
        tabs[f"a{n}"] = np.array([m.predict([T, item(10 ** 6)], lab(T, k, n, 0)) for k in range(n + 1)])
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{name}.npz"), **tabs)
    return tabs


def main():
    for name in sys.argv[1:]:
        t0 = time.time()
        tabs = build(name)
        r1 = os.path.join(R1, f"{name}.npz")
        if os.path.exists(r1):
            old = dict(np.load(r1))
            d = max(float(np.max(np.abs(tabs[k] - old[k]))) for k in tabs)
            print(f"{name}: rebuilt in {time.time() - t0:.0f}s; max |current code - round-1 table| = {d:.2e}", flush=True)
        else:
            print(f"{name}: rebuilt in {time.time() - t0:.0f}s (no round-1 table)", flush=True)


if __name__ == "__main__":
    main()
