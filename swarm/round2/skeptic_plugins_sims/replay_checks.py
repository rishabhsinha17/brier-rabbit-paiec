"""Replay-based checks of the paiec/ext plug-ins on public LOBO draws (streaming_alc_v1 replica, default random
acquisition, so the evidence sequence does not depend on the predictor and can be recorded once).

modes:
  identity : each plug-in at NEUTRAL settings vs paiec.hier.HierModel on the same calls (bit-exact count, max |diff|),
             split by k and by B0 / B>=1; also the plug-ins with no neutral setting, to measure where they differ.
  purity   : each plug-in at its RECOMMENDED settings: (A) one instance reused across every draw (paired.py style)
             vs (B) a fresh instance per checkpoint with the calls in reverse order on deep copies (platform style:
             workers restart between checkpoints) vs (C) one instance fed 240 calls in random order across all
             checkpoints of all draws (stale-cache test). Any difference = state leaking across calls.
             --save/--compare write/compare A's predictions (run twice under different PYTHONHASHSEED).
  timing   : amortized ms per call and worst single call (= refit) per model on the k=2 replays.

usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/replay_checks.py identity
"""
from __future__ import annotations

import copy
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, D06, MIX25, PLUGINS, draws, load_cache, make  # noqa: E402

import numpy as np  # noqa: E402

REPLAYS = [("real_webagents", 1, 2), ("real_webagents", 2, 2), ("researchcodebench", 2, 1), ("matharena", 2, 1),
           ("multi_swebench", 2, 1)]


def record(bench, k, runs, seed0=0):
    from paiec.protocol import run_protocol

    out = []
    for r, pairs in draws(bench, k, runs, seed0):
        cps = []

        def rec(inp, lab, cps=cps):
            if not cps or cps[-1][0] is not lab:
                cps.append((lab, []))
            cps[-1][1].append(inp)
            return 0.5

        run_protocol(pairs, rec, seed=r)
        out.append({"bench": bench, "k": k, "run": r, "cps": cps})
    return out


def all_records():
    recs = {}
    for bench, k, runs in REPLAYS:
        recs.setdefault(bench, []).extend(record(bench, k, runs))
    return recs


def calls(recs_b):
    """Flat list of (draw index, checkpoint index, k, labeled, input)."""
    out = []
    for di, d in enumerate(recs_b):
        for ci, (lab, inps) in enumerate(d["cps"]):
            for inp in inps:
                out.append((di, ci, d["k"], lab, inp))
    return out


def run_seq(model, cl, timer=None):
    preds = np.empty(len(cl))
    for j, (_, _, _, lab, inp) in enumerate(cl):
        t = time.perf_counter()
        preds[j] = model.predict(inp, lab)
        if timer is not None:
            timer.append(time.perf_counter() - t)
    return preds


IDENTITY = {
    "BASE": (BASE, [
        ("subjcal defaults", "subjcal", {}),
        ("prior defaults", "prior", {}),
        ("power alpha=1", "power", {"lik_power": 1.0}),
        ("simpool neutral", "simpool", {"pool_name": False, "pool_harness": False, "tau_name": BASE.get("tau_phi", 0.5)}),
        ("simpool defaults", "simpool", {}),
        ("itempool defaults", "itempool", {}),
        ("itempool shrink=0", "itempool", {"pool_shrink": 0.0}),
        ("hrb", "hrb", {}),
        ("exactall", "exactall", {}),
        ("bcm (mix off)", "bcm", {}),
    ]),
    "MIX25": (MIX25, [
        ("subjcal defaults", "subjcal", {}),
        ("prior defaults (mix_b 0)", "prior", {}),
        ("power alpha=1", "power", {"lik_power": 1.0}),
        ("simpool neutral", "simpool", {"pool_name": False, "pool_harness": False, "tau_name": MIX25.get("tau_phi", 0.5)}),
        ("bcm rho 0", "bcm", {}),
        ("itempool defaults", "itempool", {}),
    ]),
    "D06": (D06, [
        ("subjcal defaults", "subjcal", {}),
        ("simpool neutral", "simpool", {"pool_name": False, "pool_harness": False, "tau_name": D06["tau_phi"]}),
        ("simpool defaults", "simpool", {}),
    ]),
}


def identity(recs):
    _, priors = load_cache()
    rows = []
    for gname, (params, cases) in IDENTITY.items():
        for bench, recs_b in recs.items():
            cl = calls(recs_b)
            ks = np.array([c[2] for c in cl])
            b0 = np.array([c[1] == 0 for c in cl])
            ref = run_seq(make(None, priors[bench], params), cl)
            for label, key, extra in cases:
                pv = run_seq(make(PLUGINS[key], priors[bench], dict(params, **extra)), cl)
                d = np.abs(pv - ref)
                for kk in (1, 2):
                    for part, mask in (("B0", b0), ("B>=1", ~b0)):
                        m = (ks == kk) & mask
                        if m.any():
                            rows.append((gname, label, bench, kk, part, int(m.sum()), int((d[m] == 0).sum()), float(d[m].max())))
        print(f"[identity] group {gname} done", flush=True)
    import pandas as pd

    df = pd.DataFrame(rows, columns=["group", "case", "bench", "k", "part", "n", "n_exact", "max_abs"])
    agg = df.groupby(["group", "case", "k", "part"]).agg(n=("n", "sum"), n_exact=("n_exact", "sum"), max_abs=("max_abs", "max"))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(agg.to_string())
    return df


RECOMMENDED = {
    "subjcal a1.56": ("subjcal", dict(BASE, attr_scale=1.56, attr_year_max=2026.4)),
    "prior d06_q25": ("prior", dict(BASE, q0=0.25, tau_mu=1.2, tau_delta=0.6, tau_phi=0.3)),
    "prior tilt mix25": ("prior", dict(MIX25, mix_b=-1.4)),
    "power 1.3": ("power", dict(BASE, lik_power=1.3)),
    "power 1.3 mix25": ("power", dict(MIX25, lik_power=1.3)),
    "simpool": ("simpool", dict(BASE)),
    "itempool x2.0": ("itempool", dict(BASE, sigma_x=2.0)),
    "itempool x2.0 mix25": ("itempool", dict(MIX25, sigma_x=2.0)),
    "hrb": ("hrb", dict(BASE)),
    "exactall": ("exactall", dict(BASE)),
    "exactall mix25": ("exactall", dict(MIX25)),
    "bcm rho.3 mix25": ("bcm", dict(MIX25, hard_rho=0.3)),
}


def purity(recs, save=None, compare=None, only_a=False):
    _, priors = load_cache()
    rng = random.Random(7)
    saved = {}
    for name, (key, params) in RECOMMENDED.items():
        worst_b, worst_c, n_b, n_c, mutated = 0.0, 0.0, 0, 0, 0
        for bench, recs_b in recs.items():
            cl = calls(recs_b)
            a = run_seq(make(PLUGINS[key], priors[bench], params), cl)
            saved[f"{name}|{bench}"] = a
            if only_a:
                continue
            # (B) fresh instance per checkpoint, reversed order, deep copies; also detects argument mutation
            groups = {}
            for j, c in enumerate(cl):
                groups.setdefault((c[0], c[1]), []).append(j)
            for (_, _), idx in groups.items():
                m = make(PLUGINS[key], priors[bench], params)
                for j in reversed(idx):
                    lab, inp = cl[j][3], cl[j][4]
                    lab2, inp2 = copy.deepcopy(lab), copy.deepcopy(inp)
                    v = m.predict(inp2, lab2)
                    if lab2 != lab or inp2 != inp:
                        mutated += 1
                    worst_b = max(worst_b, abs(v - a[j]))
                    n_b += 1
            # (C) one instance, random interleaving across checkpoints and draws
            m = make(PLUGINS[key], priors[bench], params)
            for j in rng.sample(range(len(cl)), min(240 // len(recs) + 1, len(cl))):
                v = m.predict(cl[j][4], cl[j][3])
                worst_c = max(worst_c, abs(v - a[j]))
                n_c += 1
        if not only_a:
            print(f"[purity] {name:22s} fresh-per-checkpoint reversed: n={n_b} max|diff|={worst_b:.1e} | "
                  f"random interleave: n={n_c} max|diff|={worst_c:.1e} | argument mutations: {mutated}", flush=True)
    if save:
        np.savez(save, **saved)
    if compare:
        other = np.load(compare)
        worst = max(float(np.max(np.abs(other[k] - v))) for k, v in saved.items())
        print(f"[purity] vs {compare} (other PYTHONHASHSEED): max|diff| over {len(saved)} model-bench arrays = {worst:.1e}")


def timing(recs):
    _, priors = load_cache()
    models = {"HierModel (cur_final)": (None, BASE), "HierModel mix25": (None, MIX25)}
    models.update(RECOMMENDED)
    print("model | calls | amortized ms/call | p50 ms | max single call ms (k=2 replays, one instance per bench)")
    for name, (key, params) in models.items():
        timer, n = [], 0
        for bench, recs_b in recs.items():
            cl = [c for c in calls(recs_b) if c[2] == 2]
            run_seq(make(PLUGINS.get(key), priors[bench], params), cl, timer)
            n += len(cl)
        t = np.array(timer) * 1e3
        print(f"{name:24s} | {n:5d} | {t.mean():6.3f} | {np.median(t):6.3f} | {t.max():7.1f}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "identity"
    t0 = time.time()
    recs = all_records()
    n = sum(len(calls(v)) for v in recs.values())
    print(f"recorded {n} predict calls over {sum(len(v) for v in recs.values())} draws in {time.time() - t0:.0f}s", flush=True)
    if mode == "identity":
        identity(recs)
    elif mode == "purity":
        save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
        cmp_ = sys.argv[sys.argv.index("--compare") + 1] if "--compare" in sys.argv else None
        purity(recs, save, cmp_, only_a="--only-a" in sys.argv)
    elif mode == "timing":
        timing(recs)
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
