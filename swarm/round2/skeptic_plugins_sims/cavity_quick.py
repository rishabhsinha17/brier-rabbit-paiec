"""Bonus battery for paiec/ext/pool_cavity.py (CavityPoolModel, round-2 file that appeared during this review):
identity vs HierModel (k=1 and B0 must be exact), purity (fresh instance per checkpoint, reversed; random interleave),
caught-exception audit, fuzz, and the 300-label timing. Same harness as the other checks.
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/cavity_quick.py
"""
from __future__ import annotations

import copy
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import BASE, MIX25, load_cache, make  # noqa: E402

common.PLUGINS["cavity"] = "paiec.ext.pool_cavity:CavityPoolModel"
import fallback_audit as fa  # noqa: E402
import fuzz  # noqa: E402
import timing_refit  # noqa: E402
from replay_checks import all_records, calls, run_seq  # noqa: E402

import numpy as np  # noqa: E402

CLS = common.PLUGINS["cavity"]


def main():
    print("pool_cavity.py mtime:", time.ctime(os.path.getmtime((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/paiec/ext/pool_cavity.py"))))
    _, priors = load_cache()
    recs = all_records()
    for label, params in (("cur_final sx2.0", dict(BASE)), ("cur_final sx2.5", dict(BASE, sigma_x=2.5)), ("mix25 sx2.0", dict(MIX25))):
        stats = {}
        worst_b, worst_c = 0.0, 0.0
        fa.EVENTS.clear()
        for bench, rb in recs.items():
            cl = calls(rb)
            ref = run_seq(make(None, priors[bench], params), cl)
            sys.settrace(fa._global)
            try:
                a = run_seq(make(CLS, priors[bench], params), cl)
            finally:
                sys.settrace(None)
            for j, c in enumerate(cl):
                key = (c[2], "B0" if c[1] == 0 else "B>=1")
                n, ne, mx = stats.get(key, (0, 0, 0.0))
                stats[key] = (n + 1, ne + int(a[j] == ref[j]), max(mx, abs(a[j] - ref[j])))
            groups = {}
            for j, c in enumerate(cl):
                groups.setdefault((c[0], c[1]), []).append(j)
            for idx in groups.values():
                m = make(CLS, priors[bench], params)
                for j in reversed(idx):
                    v = m.predict(copy.deepcopy(cl[j][4]), copy.deepcopy(cl[j][3]))
                    worst_b = max(worst_b, abs(v - a[j]))
            m = make(CLS, priors[bench], params)
            for j in random.Random(3).sample(range(len(cl)), min(60, len(cl))):
                worst_c = max(worst_c, abs(m.predict(cl[j][4], cl[j][3]) - a[j]))
        print(f"[{label}] identity vs HierModel: " + "; ".join(f"k={k} {part}: {ne}/{n} exact, max {mx:.2e}" for (k, part), (n, ne, mx) in sorted(stats.items())))
        print(f"[{label}] purity: fresh-per-checkpoint reversed max|diff| {worst_b:.1e}; random interleave {worst_c:.1e};"
              f" caught exceptions: {dict(fa.EVENTS) or 'none'}", flush=True)
    # fuzz with the generic generators
    fails, n = [], 0
    for params in (dict(BASE), dict(MIX25)):
        model = make(CLS, fuzz.PRIOR, params)
        for lname, lab in fuzz.bad_labeled():
            for iname, inp in fuzz.bad_inputs():
                if lname.startswith("1000 entries") and iname not in ("good", "unknown benchmark"):
                    continue
                try:
                    v = model.predict(inp, lab)
                    ok = isinstance(v, float) and math.isfinite(v) and 0.0 < v < 1.0
                except Exception as e:  # noqa: BLE001
                    ok, v = False, repr(e)
                n += 1
                if not ok:
                    fails.append((lname, iname, v))
    print(f"fuzz: {n} calls, failures {len(fails)} {fails[:5]}")
    labeled, cl = timing_refit.build()
    for label, params in (("HierModel", None), ("cavity sx2.0", dict(BASE))):
        m = make(CLS if params else None, timing_refit.PRIOR, params or dict(BASE))
        ts = []
        for inp in cl:
            t = time.process_time()
            m.predict(inp, labeled)
            ts.append(time.process_time() - t)
        print(f"timing 300 labels {label:13s}: first call {1e3 * ts[0]:.1f} ms CPU, amortized {1e3 * np.mean(ts):.2f} ms, max {1e3 * max(ts):.1f} ms")


if __name__ == "__main__":
    main()
