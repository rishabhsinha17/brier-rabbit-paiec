"""Runtime of ComboModel on the skeptic's synthetic B31 checkpoint (300 labels, 10 pairs on 7 benchmarks, 3 shared,
100 KB item texts on two benchmarks, 5 known + 5 unknown subjects, the unknown ones sharing a harness string).
CPU time, fresh model per checkpoint (platform-like), best of 2 repetitions.
usage: nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round3/composer_sims/timing.py"""
import os, sys, time
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round2/skeptic_plugins_sims"))
import numpy as np
from timing_refit import PRIOR, build
from common import BASE
from paiec.ext.combo import ComboModel
from paiec.hier import HierModel
ALL = {"subjcal": True, "attr_scale": 1.56, "attr_year_max": 2026.4, "pool_item": True, "sigma_x": 3.0, "pool_sim": True}
CONF = {"HierModel REF": (HierModel, BASE), "combo all off": (ComboModel, BASE),
        "combo S": (ComboModel, dict(BASE, subjcal=True, attr_scale=1.56, attr_year_max=2026.4)),
        "combo C": (ComboModel, dict(BASE, pool_item=True, sigma_x=3.0)),
        "combo P": (ComboModel, dict(BASE, pool_sim=True)),
        "combo SCP": (ComboModel, dict(BASE, **ALL)),
        "combo SCP W20": (ComboModel, dict(BASE, tau_mu=2.0, **ALL))}
labeled, calls = build()
print(f"synthetic B31 checkpoint: {len(labeled)} labels, {len(calls)} calls; CPU ms: first call (refit) | amortized | max")
for name, (cls, params) in CONF.items():
    reps = []
    for _ in range(2):
        m = cls(PRIOR, dict(params))
        ts = []
        for inp in calls:
            t = time.process_time()
            m.predict(inp, labeled)
            ts.append(time.process_time() - t)
        reps.append((ts[0], np.mean(ts), np.max(ts)))
    print(f"{name:16s} | {1e3 * min(r[0] for r in reps):7.1f} | {1e3 * min(r[1] for r in reps):6.2f} | {1e3 * min(r[2] for r in reps):7.1f}", flush=True)
