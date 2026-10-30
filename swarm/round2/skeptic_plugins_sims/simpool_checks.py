"""SimilarityPoolModel stress checks (public data only).
1. Variance split under cur_final vs the in-flight d06 level block (tau_name/tau_h are ABSOLUTE SDs, so they eat a
   larger share of a narrower pair variance).
2. How often two pairs of ONE benchmark share a canonical name (simpool pools them through nu_name, although round 1
   measured within-benchmark same-name residual correlation -0.05 on Multi-SWE-bench and +0.14 on MathArena).
3. Replay: k=2 draws made of two same-name pairs on one benchmark, base vs simpool (cur_final and d06 levels), plus
   name-unrelated controls.
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/simpool_checks.py
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, D06, PLUGINS, load_cache, make  # noqa: E402

import numpy as np  # noqa: E402

from paiec.protocol import BUDGETS, materialize, pair_alc, run_protocol, unit  # noqa: E402
from paiec.subjects import name_key  # noqa: E402


def split(params):
    m = make(PLUGINS["simpool"], {}, params)
    subj_h = {"normalized_name": "X", "harness": "AgentX"}
    subj_n = {"normalized_name": "X", "harness": ""}
    _, tn, _, th2, _, dvar = m._keys(subj_h, "b")
    _, tn0, _, _, _, dvar0 = m._keys(subj_n, "b")
    tot = params.get("tau_phi", 0.5) ** 2 + params["tau_delta"] ** 2
    return (f"total pair var {tot:.3f}: with harness tn {tn:.4f} th2 {th2:.4f} dvar {dvar:.4f} | no harness tn {tn0:.4f} "
            f"dvar {dvar0:.4f} | implied prior corr: same name {tn / tot:.2f}, same bench+harness {th2 / tot:.2f}, "
            f"same name+bench+harness {(tn + th2) / tot:.2f}")


def main():
    raw, priors = load_cache()
    print("1. variance split")
    print("   cur_final:", split(dict(BASE)))
    print("   d06      :", split(dict(D06)))
    print("   d06 with tau_name 0.47 (=sqrt(0.49*0.45)), tau_h 0.3:", split(dict(D06, tau_name=0.47, tau_h=0.3)))
    print("\n2. same canonical name within one benchmark (public eligible pairs)")
    same_pairs = {}
    for b, rps in raw.items():
        keys = [name_key(rp["subject"]) for rp in rps]
        harn = [str(rp["subject"].get("harness", "") or "") for rp in rps]
        tot, same, same_diffh = 0, [], 0
        for i, j in itertools.combinations(range(len(rps)), 2):
            tot += 1
            if keys[i] and keys[i] == keys[j]:
                same.append((i, j))
                same_diffh += harn[i] != harn[j]
        same_pairs[b] = same
        print(f"   {b:18s} pairs {len(rps):3d} | pair-pairs {tot:5d} | same name {len(same):4d} ({100 * len(same) / max(tot, 1):.1f}%),"
              f" of which different harness {same_diffh}")
    print("\n3. replay: two same-name pairs on one benchmark (12 draws per benchmark max), ALC diff simpool - base (x1e-3)")
    for label, params in (("cur_final", BASE), ("d06", D06)):
        for b in ("multi_swebench", "matharena"):
            cand = sorted(same_pairs[b], key=lambda ij: unit("same", b, ij[0], ij[1]))[:12]
            if not cand:
                continue
            base = make(None, priors[b], params)
            sp = make(PLUGINS["simpool"], priors[b], params)
            diffs, per_b = [], {x: [] for x in BUDGETS}
            for r, (i, j) in enumerate(cand):
                pairs = materialize([raw[b][i], raw[b][j]], size_range=(88, 226), seed=r)
                ra = run_protocol(pairs, base.predict, seed=r)
                rb = run_protocol(pairs, sp.predict, seed=r)
                for key in ra:
                    diffs.append(pair_alc(rb[key]) - pair_alc(ra[key]))
                    for x in BUDGETS:
                        per_b[x].append(rb[key][x]["brier"] - ra[key][x]["brier"])
            d = np.array(diffs)
            print(f"   {label:9s} {b:15s} n={len(d):2d} pair-draws: ALC {1e3 * d.mean():+.2f} +- {1e3 * d.std() / np.sqrt(len(d)):.2f}"
                  f" | B1 {1e3 * np.mean(per_b[1]):+.2f} B3 {1e3 * np.mean(per_b[3]):+.2f} B7 {1e3 * np.mean(per_b[7]):+.2f}"
                  f" B15 {1e3 * np.mean(per_b[15]):+.2f} B31 {1e3 * np.mean(per_b[31]):+.2f}", flush=True)


if __name__ == "__main__":
    main()
