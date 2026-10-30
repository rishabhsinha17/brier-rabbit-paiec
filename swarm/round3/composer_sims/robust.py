"""Robustness of ComboModel: fuzz (never raises, finite float in (0,1), no argument mutation) and purity (prediction
independent of call order and of instance reuse), reusing the skeptic's generators and recorded calls.

usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round3/composer_sims/robust.py fuzz|purity
"""
from __future__ import annotations

import copy
import math
import os
import random
import sys
import time

ROOT = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "swarm/round2/skeptic_plugins_sims"))
from common import BASE, MIX25, load_cache  # noqa: E402

import numpy as np  # noqa: E402

from paiec.ext.combo import ComboModel  # noqa: E402

ALL = {"subjcal": True, "attr_scale": 1.56, "attr_year_max": 2026.4, "pool_item": True, "sigma_x": 2.5, "pool_sim": True}
CONFIGS = {
    "all off": dict(BASE),
    "all on REF": dict(BASE, **ALL),
    "all on W20": dict(BASE, tau_mu=2.0, **ALL),
    "all on mix25": dict(MIX25, **ALL),
    "all on q0_match": dict(BASE, q0_match="base", q0_theta=0.5, **ALL),
    "all on tau_M (sim off)": dict(BASE, tau_M=0.5, **ALL),
}


def fuzz():
    import fuzz as F  # the skeptic's generators

    t0 = time.time()
    failures, n, slow = [], 0, []
    for cname, params in CONFIGS.items():
        model = ComboModel(F.PRIOR, params)
        for lname, lab in F.bad_labeled():
            for iname, inp in F.bad_inputs():
                if lname.startswith("1000 entries") and iname not in ("good", "unknown benchmark", "None", "item fields NaN"):
                    continue
                lab_c, inp_c = copy.deepcopy(lab), copy.deepcopy(inp)
                t = time.time()
                try:
                    v = model.predict(inp, lab)
                    ok = isinstance(v, float) and math.isfinite(v) and 0.0 < v < 1.0
                    if not ok:
                        failures.append((cname, lname, iname, f"bad value {v!r}"))
                except Exception as e:  # noqa: BLE001
                    failures.append((cname, lname, iname, f"raised {type(e).__name__}: {e}"))
                if repr(lab_c) != repr(lab) or repr(inp_c) != repr(inp):
                    failures.append((cname, lname, iname, "argument mutated"))
                dt = time.time() - t
                if dt > 0.25:
                    slow.append((cname, lname, iname, round(dt, 2)))
                n += 1
        print(f"[fuzz] {cname:24s} done ({n} calls so far, {len(failures)} failures)", flush=True)
    print(f"[fuzz] {n} calls, {len(failures)} failures, {len(slow)} calls > 0.25 s wall; {time.time() - t0:.0f}s")
    for f in failures[:20]:
        print("  FAIL", f)
    for s in slow[:8]:
        print("  slow", s)


def purity():
    from replay_checks import all_records, calls, run_seq

    _, priors = load_cache()
    recs = all_records()
    rng = random.Random(11)
    for cname in ("all on REF", "all on mix25", "all on q0_match"):
        params = CONFIGS[cname]
        worst_b = worst_c = 0.0
        n_b = n_c = mutated = 0
        for bench, recs_b in recs.items():
            cl = calls(recs_b)
            a = run_seq(ComboModel(priors[bench], params), cl)
            groups = {}
            for j, c in enumerate(cl):
                groups.setdefault((c[0], c[1]), []).append(j)
            for idx in groups.values():  # fresh instance per checkpoint, reversed order, deep copies
                m = ComboModel(priors[bench], params)
                for j in reversed(idx):
                    lab, inp = cl[j][3], cl[j][4]
                    lab2, inp2 = copy.deepcopy(lab), copy.deepcopy(inp)
                    v = m.predict(inp2, lab2)
                    mutated += int(lab2 != lab or inp2 != inp)
                    worst_b = max(worst_b, abs(v - a[j]))
                    n_b += 1
            m = ComboModel(priors[bench], params)  # one instance, random interleave across checkpoints and draws
            for j in rng.sample(range(len(cl)), min(80, len(cl))):
                worst_c = max(worst_c, abs(m.predict(cl[j][4], cl[j][3]) - a[j]))
                n_c += 1
        print(f"[purity] {cname:18s} fresh-per-checkpoint reversed n={n_b} max|diff|={worst_b:.1e} | random interleave "
              f"n={n_c} max|diff|={worst_c:.1e} | argument mutations {mutated}", flush=True)


if __name__ == "__main__":
    {"fuzz": fuzz, "purity": purity}[sys.argv[1]]()
