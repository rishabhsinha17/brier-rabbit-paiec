"""Operating characteristics of the live decision pipeline: simulated worlds x policy (swarm round 6, statistician).

usage: R swarm/round6/oc_sims/run_oc.py N_WORLDS SEED SCENARIO [POLICY] [MODEL] [STRATA] [--max-seconds S]
  SCENARIO  null | one:<arm>:<delta>[:<profile>] | all:<delta>[:<profile>]
            arm = index in the 10-07 shortlist (0 W20, 1 L15, 2 W20@-1.5, 3 m-2.50_mu2.5_d0.6, 4 m-2.25_mu2.5_d1.4,
            5 m-1.50_mu2.5_d0.6, 6 m-2.00_mu2.0_d0.6); delta = true ALC effect vs REF (e.g. -0.003);
            profile = w20 (default; W20's pre-registered H_pub budget profile) | flat | level
  POLICY    P1plan (default; the adopted adaptive design) | P1fast (200 posterior draws) | P2stat (static rotation)
  MODEL     cal (default; calculus contrasts + item/group + partner noise calibrated to the live matched SDs) |
            floor (stress: every arm's per-pair SD at the QUEUE floors) | nocal (calculus + partner noise only);
            suffix 'sup' (calsup, floorsup): superpopulation null, the pool's stratum means are themselves a sample
            (each deviates from the population value by one pair's contrast / sqrt(N_b)); default: the pool is the
            population (every stratum mean exactly at the population value)
  STRATA    r4 (default; 285 pairs, ours + connectors) | ours (244 pairs, ours only)
Appends one JSON record per world to cache/oc_<scenario>_<policy>_<model>_<strata>.jsonl (the world seed is stored, so
chunks with different SEEDs never overlap). Each call is meant to stay under 2 minutes (--max-seconds stops early).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oc_core as oc  # noqa: E402
import oc_pipeline as op  # noqa: E402
import oc_setup  # noqa: E402
import ld_sim as ls  # noqa: E402

CAL = np.array([0.012, 0.008, 0.008, 0.003, 0.001, 0.0005])
FLOOR_B = oc.FLOOR[1:]


def effects_for(scn: str, A: int):
    E = np.zeros((A, 6))
    if scn == "null":
        return E
    parts = scn.split(":")
    if parts[0] == "one":
        a, delta = int(parts[1]), float(parts[2])
        prof = op.UNIT_PROFILES[parts[3] if len(parts) > 3 else "w20"]
        E[a] = delta * prof
    elif parts[0] == "all":
        delta = float(parts[1])
        prof = op.UNIT_PROFILES[parts[2] if len(parts) > 2 else "w20"]
        E[:] = delta * prof[None, :]
    else:
        raise ValueError(scn)
    return E


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    maxs = 110.0
    for a in sys.argv[1:]:
        if a.startswith("--max-seconds="):
            maxs = float(a.split("=")[1])
    nw, seed, scn = int(args[0]), int(args[1]), args[2]
    policy = args[3] if len(args) > 3 else "P1plan"
    model = args[4] if len(args) > 4 else "cal"
    strata = args[5] if len(args) > 5 else "r4"
    t0 = time.time()
    S = oc_setup.load()
    T = ls.Tables()
    arms = [S["cands"][n] for n in S["names"]]
    A = len(arms)
    E = effects_for(scn, A)
    st = (oc.STRATA_N, oc.STRATA_D) if strata == "r4" else (oc.STRATA_N_OURS, oc.STRATA_D_OURS)
    tag = scn.replace(":", "_").replace("-", "m").replace(".", "p")
    out = HERE / "cache" / f"oc_{tag}_{policy}_{model}_{strata}.jsonl"
    rng0 = np.random.default_rng(seed * 1000003 + 17)
    done = 0
    with out.open("a") as fh:
        for iw in range(nw):
            if time.time() - t0 > maxs:
                break
            wseed = seed * 100000 + iw
            rng = np.random.default_rng(wseed)
            c = int(rng0.choice(len(S["w"]), p=S["w"]))
            kw = {"effects": E, "strata": st}
            if model.startswith("cal"):
                kw["extra_sd"] = CAL
            elif model.startswith("floor"):
                kw["extra_sd"] = CAL
                kw["target_sd"] = FLOOR_B
            if model.endswith("sup"):
                kw["superpop"] = True
            W = oc.World(T, S["cells"][c], arms, rng, **kw)
            rec = op.simulate(W, S, policy, rng, seed=wseed)
            rec.update({"wseed": wseed, "cell": [int(x) for x in S["cells"][c]], "scenario": scn, "policy": policy,
                        "model": model, "strata": strata, "sd_alc": W.C[:, :, 0].std(1, ddof=1).tolist(),
                        "pop_truth": (E @ oc.WB).tolist()})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
    print(f"{done} worlds of {scn} / {policy} / {model} / {strata} in {time.time() - t0:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()
