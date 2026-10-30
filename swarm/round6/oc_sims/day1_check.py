"""Sanity check: the in-simulation allocation (ld_policy.P1 on a simulated world at the 10-06 state) against the
real 2026-10-07 plan (8 W20, 8 REF, 4 W20@-1.5, 4 m-2.50_mu2.5_d0.6, 3 L15, 3 m-2.25_mu2.5_d1.4). usage: R day1_check.py"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oc_core as oc  # noqa: E402
import oc_pipeline as op  # noqa: E402
import oc_setup  # noqa: E402
import ld_sim as ls  # noqa: E402
from run_oc import CAL  # noqa: E402

S = oc_setup.load()
T = ls.Tables()
arms = [S["cands"][n] for n in S["names"]]
tot = Counter()
for seed in range(6):
    rng = np.random.default_rng(seed)
    c = int(rng.choice(len(S["w"]), p=S["w"]))
    W = oc.World(T, S["cells"][c], arms, rng, extra_sd=CAL)
    D = oc.Data(W)
    for _ in range(10):
        D.upload(-1, rng)
    for a, k in ((0, 4), (1, 5)):
        for _ in range(k):
            D.upload(a, rng)
    R = op.Run(W, D, S["sd_arm"])
    pol, post = op.make_policy("P1plan", S, seed)
    R.day = 1
    ups = pol.plan(R, 1)
    cnt = Counter("REF" if u < 0 else S["tags"][u] for u in ups)
    tot.update(cnt)
    print(f"world {seed}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda t: -t[1])) +
          f" | planned finals {dict(zip(S['tags'], pol.plan_final.tolist()))}, REF {pol.ref_final}")
print("mean over worlds: " + ", ".join(f"{k} {v / 6:.1f}" for k, v in sorted(tot.items(), key=lambda t: -t[1])))
