"""Compute-queue job (swarm round 6, statistician): more replicates of the pipeline's operating characteristics.

Runs worlds of several scenarios round-robin in PAIEC_WORKERS processes (default 2) until a wall-clock budget is
spent, then exits. Each worker appends to its own file cache/oc_<scenario>_<policy>_<model>_<strata>_q<k>.jsonl;
world seeds start at 90,000,000 so they never collide with the direct runs (chunk seed x 100000 + world, chunk seeds < 900). Summaries:
  R swarm/round6/oc_sims/summarize_oc.py      (reads every cache/oc_*.jsonl, de-duplicated by world seed)
usage: python swarm/round6/oc_sims/oc_queue_job.py [minutes]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCENARIOS = [  # (scenario, policy, model, strata), in priority order
    ("null", "P1plan", "floorsup", "r4"), ("one:0:-0.002", "P1plan", "cal", "r4"),
    ("one:0:-0.003", "P1plan", "cal", "r4"), ("null", "P1plan", "calsup", "r4"),
    ("one:0:-0.001", "P1plan", "cal", "r4"), ("one:0:+0.002", "P1plan", "cal", "r4"),
    ("all:+0.002", "P1plan", "cal", "r4"), ("one:0:-0.005", "P1plan", "cal", "r4"),
    ("one:1:-0.003", "P1plan", "cal", "r4"), ("all:+0.001", "P1plan", "cal", "r4"),
    ("one:0:-0.002", "P1plan", "floorsup", "r4"), ("one:0:-0.003", "P1plan", "floorsup", "r4"),
]


def worker(k: int, n_workers: int, deadline: float):
    import numpy as np
    import oc_core as oc
    import oc_pipeline as op
    import oc_setup
    import ld_sim as ls
    from run_oc import CAL, FLOOR_B, effects_for
    S = oc_setup.load()
    T = ls.Tables()
    arms = [S["cands"][n] for n in S["names"]]
    i = 0
    while time.time() < deadline:
        scn, pol, model, strata = SCENARIOS[(i * n_workers + k) % len(SCENARIOS)]
        it = (i * n_workers + k) // len(SCENARIOS)
        wseed = 90_000_000 + 10_000 * SCENARIOS.index((scn, pol, model, strata)) + it
        rng = np.random.default_rng(wseed)
        rng0 = np.random.default_rng(wseed + 17)
        c = int(rng0.choice(len(S["w"]), p=S["w"]))
        st = (oc.STRATA_N, oc.STRATA_D) if strata == "r4" else (oc.STRATA_N_OURS, oc.STRATA_D_OURS)
        E = effects_for(scn, len(arms))
        kw = {"effects": E, "strata": st, "extra_sd": CAL}
        if model.startswith("floor"):
            kw["target_sd"] = FLOOR_B
        if model.endswith("sup"):
            kw["superpop"] = True
        W = oc.World(T, S["cells"][c], arms, rng, **kw)
        rec = op.simulate(W, S, pol, rng, seed=wseed)
        rec.update({"wseed": wseed, "cell": [int(x) for x in S["cells"][c]], "scenario": scn, "policy": pol,
                    "model": model, "strata": strata, "sd_alc": W.C[:, :, 0].std(1, ddof=1).tolist(),
                    "pop_truth": (E @ oc.WB).tolist()})
        tag = scn.replace(":", "_").replace("-", "m").replace(".", "p")
        with (HERE / "cache" / f"oc_{tag}_{pol}_{model}_{strata}_q{k}.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        i += 1
    print(f"worker {k}: {i} worlds", flush=True)


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 36.0
    n = max(1, int(os.environ.get("PAIEC_WORKERS", "2")))
    deadline = time.time() + 60 * minutes
    ps = [mp.Process(target=worker, args=(k, n, deadline)) for k in range(n)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print(f"done after {(time.time() - deadline) / 60 + minutes:.1f} min")


if __name__ == "__main__":
    main()
