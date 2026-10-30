"""Worst-case refit and B31-checkpoint cost with ~300 labels, on a synthetic draw shaped like a formative draw:
10 pairs on 7 benchmarks (3 benchmarks with two subjects, partners overlapping on half their items), 30 labels per pair,
10 item_features groups per benchmark, item_content of 0.1-100 KB (ResearchCodeBench-like on two benchmarks),
5 name-known and 5 unknown subjects (shipped subject prior). Simulates a platform worker: a fresh model per
checkpoint, one refit, then 30 evaluation items of every pair (300 calls; partners labeled all 30 for 3 pairs).

usage: nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/timing_refit.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, MIX25, PLUGINS, ROOT, make  # noqa: E402

import numpy as np  # noqa: E402

PRIOR = json.load(open(os.path.join(ROOT, "submission_src/v3b_noM_n_m0n20/subject_prior.json")))
KNOWN = sorted(PRIOR["names"])[:5]


def build(seed=0, per_pair=30):
    rng = random.Random(seed)
    benches = [f"benchmark_{100000 + 7919 * j}" for j in range(7)]
    big = {benches[0], benches[1]}
    subjects = [{"normalized_name": KNOWN[j], "provider": "", "release_date": "2025-03-01", "access_date": "",
                 "harness": "", "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""} for j in range(5)]
    subjects += [{"normalized_name": f"Unseen Model {j}", "provider": "OpenAI", "release_date": "2026-0%d-01" % (j + 1),
                  "access_date": "", "harness": "AgentX", "reasoning_effort": "high", "harness_version": "",
                  "subject_features_extra": ""} for j in range(5)]
    layout = [0, 0, 1, 1, 2, 2, 3, 4, 5, 6]  # benchmark of each subject's pair
    items = {}
    for b in benches:
        n = 120
        pad = "x" * (100_000 if b in big else 200)
        items[b] = [{"item_content": f"{b} problem {i} " + pad + f" end {i}", "item_features": f"grp=g{i % 10}",
                     "interactors": "", "benchmark_id": b} for i in range(n)]
    labeled, evals = [], []
    for si, s in enumerate(subjects):
        b = benches[layout[si]]
        pool = list(range(120))
        rng.shuffle(pool)
        acq, ev = pool[:60], pool[60:]
        if si % 2 == 1 and layout[si] == layout[si - 1]:  # partner: half of its acquisition = target's eval items
            prev_ev = evals[-1][1]
            acq = [prev_ev[j] for j in range(30)] + acq[:30]
        rate = 0.05 + 0.9 * rng.random()
        for i in acq[:per_pair]:
            labeled.append([[s, items[b][i]], int(rng.random() < rate)])
        evals.append((s, ev, b))
    calls = [[s, items[b][i]] for s, ev, b in evals for i in ev[:30]]
    return labeled, calls


CONFIGS = {
    "HierModel cur_final": (None, BASE),
    "HierModel mix25": (None, MIX25),
    "subjcal a1.56": ("subjcal", dict(BASE, attr_scale=1.56, attr_year_max=2026.4)),
    "prior d06_q25": ("prior", dict(BASE, q0=0.25, tau_mu=1.2, tau_delta=0.6, tau_phi=0.3)),
    "prior tilt mix25": ("prior", dict(MIX25, mix_b=-1.4)),
    "power 1.3": ("power", dict(BASE, lik_power=1.3)),
    "power 1.3 mix25": ("power", dict(MIX25, lik_power=1.3)),
    "simpool": ("simpool", dict(BASE)),
    "itempool x2.0": ("itempool", dict(BASE, sigma_x=2.0)),
    "hrb": ("hrb", dict(BASE)),
    "exactall": ("exactall", dict(BASE)),
    "exactall mix25": ("exactall", dict(MIX25)),
    "bcm rho.3 mix25": ("bcm", dict(MIX25, hard_rho=0.3)),
}


def main():
    labeled, calls = build()
    print(f"synthetic B31 checkpoint: {len(labeled)} labels, {len(calls)} evaluation calls")
    print("CPU time. model | first call (refit) ms | amortized ms/call over the checkpoint | max call ms | checkpoint total wall s")
    for name, (key, params) in CONFIGS.items():
        reps = []
        for rep in range(2):
            m = make(PLUGINS.get(key), PRIOR, params)
            ts = []
            t_all = time.perf_counter()
            for inp in calls:
                t = time.process_time()  # CPU time: background QoS on a loaded laptop inflates wall time ~2x
                m.predict(inp, labeled)
                ts.append(time.process_time() - t)
            reps.append((ts[0], np.mean(ts), np.max(ts), time.perf_counter() - t_all))
        first, amort, mx, tot = (min(r[i] for r in reps) for i in range(4))
        print(f"{name:20s} | {1e3 * first:8.1f} | {1e3 * amort:6.3f} | {1e3 * mx:8.1f} | {tot:6.2f}", flush=True)
    # where the per-call time of the base model goes: the evidence hash of 300 entries on every call
    from paiec.hier import HierModel
    m = HierModel(PRIOR, BASE)
    t = time.process_time()
    for _ in range(50):
        m._evidence_key(labeled)
    print(f"HierModel._evidence_key on {len(labeled)} entries: {1e3 * (time.process_time() - t) / 50:.3f} ms CPU per call")


if __name__ == "__main__":
    main()
