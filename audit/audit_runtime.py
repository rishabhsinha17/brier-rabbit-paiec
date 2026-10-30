"""Runtime audit of a shipped composite zip (default v6_combo_w20; the three v6_combo zips share all code).

usage (from brier_rabbit/, quick, background QoS):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/release/audit/audit_runtime.py budgets  [--name Z]
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/release/audit/audit_runtime.py scaling  --labels 372,744,1488
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/release/audit/audit_runtime.py pool     [--concurrency 4]
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/release/audit/audit_runtime.py order

budgets  formative-shaped draw (9 pairs / 7 benchmarks, 2 shared, one benchmark with 100 KB texts) and a dense draw
         (12 pairs on one benchmark): per checkpoint, a fresh module (workers restart between checkpoints), the first
         call (refit) and the per-call cost with and without the worker's copy.deepcopy(labeled).
scaling  one refit at L labels (dense pairs, 31 labels each) in a fresh subprocess: CPU time and peak RSS.
pool     the starting kit's own worker pool (prediction_workers.parallel_predictions) on the formative B31 checkpoint at
         concurrency 1 and N: wall time and bit-identity with the in-process predictions.
order    forward / reversed / shuffled call order on fresh modules: bit-identity (determinism across worker splits).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")


def fresh(name, tag):
    mod, d = common.shipped(name, module_name=f"pm_{tag}_{time.monotonic_ns()}")
    return mod, d


def checkpoint(name, labeled, inputs, deep=True):
    """One platform checkpoint in-process: fresh module, every call gets a deepcopy of labeled like the worker."""
    t_imp = time.perf_counter()
    mod, _ = fresh(name, "ck")
    t_imp = time.perf_counter() - t_imp
    times, dc_times, out = [], [], []
    for inp in inputs:
        t0 = time.perf_counter()
        lab = copy.deepcopy(labeled) if deep else labeled
        t1 = time.perf_counter()
        p = mod.predict(copy.deepcopy(inp), lab)
        t2 = time.perf_counter()
        dc_times.append(t1 - t0)
        times.append(t2 - t1)
        out.append(p)
        assert isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0
    return t_imp, times, dc_times, out


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def cmd_budgets(args):
    _, d = common.shipped(args.name)
    names = common.prior_names(d)
    shapes = {"formative 9 pairs / 7 benchmarks": dict(n_pairs=9, n_bench=7),
              "dense 12 pairs on 1 benchmark": dict(n_bench=1, layout=[0] * 12, big_bench=0, n_items=80)}
    print(f"{args.name}: per checkpoint, fresh module; ms per call (model only | worker deepcopy of labeled)")
    print("shape | budget | labels | calls | import ms | first call (refit) ms | median | p99 | max | deepcopy median"
          " | checkpoint s (1 worker) | x4 workers est s")
    for shape, kw in shapes.items():
        for budget in (0, 1, 3, 7, 15, 31):
            labeled, inputs = common.draw(names, budget=budget, seed=args.seed, **kw)
            t_imp, ts, dcs, _ = checkpoint(args.name, labeled, inputs)
            tot = t_imp + sum(ts) + sum(dcs)
            est4 = t_imp + ts[0] + (sum(ts[1:]) + sum(dcs)) / 4
            print(f"{shape} | B{budget} | {len(labeled)} | {len(inputs)} | {1e3 * t_imp:.0f} | {1e3 * ts[0]:.1f} | "
                  f"{1e3 * q(ts[1:], 0.5):.2f} | {1e3 * q(ts[1:], 0.99):.2f} | {1e3 * max(ts[1:]):.1f} | "
                  f"{1e3 * q(dcs, 0.5):.2f} | {tot:.2f} | {est4:.2f}", flush=True)
    print(f"peak RSS of this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MB (macOS bytes)")


SCALING_CHILD = r'''
import json, resource, sys, time
sys.path.insert(0, sys.argv[1])
import common
name, n_pairs, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mod, d = common.shipped(name)
names = common.prior_names(d)
n_bench = max(1, n_pairs // 4)
labeled, inputs = common.draw(names, n_pairs=n_pairs, n_bench=n_bench, budget=31, seed=seed, big_bench=0, n_items=80)
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.process_time(); w0 = time.perf_counter()
p = mod.predict(inputs[0], labeled)
t1 = time.process_time(); w1 = time.perf_counter()
p2 = mod.predict(inputs[1], labeled)
t2 = time.perf_counter()
fit = mod._MODEL._fit
print(json.dumps({"labels": len(labeled), "params": int(len(fit["mean"])), "refit_cpu_s": t1 - t0,
                  "refit_wall_s": w1 - w0, "next_call_ms": 1e3 * (t2 - w1),
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                  "rss_before_mb": rss0 / 2**20, "p": p}))
'''


def cmd_scaling(args):
    print(f"{args.name}: one refit at B31 with dense pairs (31 labels each, 4 pairs per benchmark), fresh subprocess")
    print("labels | parameters | refit CPU s | refit wall s | next call ms | peak RSS MB | RSS before fit MB")
    for L in [int(x) for x in args.labels.split(",")]:
        n_pairs = max(1, L // 31)
        r = subprocess.run([sys.executable, "-c", SCALING_CHILD, str(Path(__file__).resolve().parent), args.name,
                            str(n_pairs), str(args.seed)], capture_output=True, text=True, timeout=args.timeout)
        if r.returncode:
            print(f"{L}: FAILED {r.stderr[-800:]}")
            continue
        j = json.loads(r.stdout.strip().splitlines()[-1])
        print(f"{j['labels']} | {j['params']} | {j['refit_cpu_s']:.2f} | {j['refit_wall_s']:.2f} | "
              f"{j['next_call_ms']:.1f} | {j['peak_rss_mb']:.0f} | {j['rss_before_mb']:.0f}", flush=True)


def cmd_pool(args):
    sys.path.insert(0, str(common.KIT / "tools"))
    from prediction_workers import parallel_predictions
    _, d = common.shipped(args.name)
    names = common.prior_names(d)
    labeled, inputs = common.draw(names, budget=31, seed=args.seed)
    _, _, _, ref = checkpoint(args.name, labeled, inputs)
    labeled_json = json.loads(json.dumps(labeled))  # the coordinator sends JSON
    for conc in sorted({1, args.concurrency}):
        t0 = time.perf_counter()
        got = [None] * len(inputs)
        for i, p in parallel_predictions(d / "model.py", inputs, labeled_json, concurrency=conc,
                                         deadline=time.monotonic() + 600, per_call_timeout=120):
            got[i] = p
        wall = time.perf_counter() - t0
        diff = max(abs(a - b) for a, b in zip(got, ref))
        print(f"kit worker pool, concurrency {conc}: {len(inputs)} calls, {len(labeled)} labels, wall {wall:.2f}s, "
              f"max |pool - in-process| = {diff:.2e}", flush=True)


def cmd_order(args):
    _, d = common.shipped(args.name)
    names = common.prior_names(d)
    worst = 0.0
    for kw in (dict(n_pairs=9, n_bench=7), dict(n_bench=1, layout=[0] * 12, big_bench=0, n_items=80)):
        labeled, inputs = common.draw(names, budget=31, seed=args.seed, **kw)
        _, _, _, fwd = checkpoint(args.name, labeled, inputs, deep=False)
        rev_in = list(reversed(inputs))
        _, _, _, rev = checkpoint(args.name, labeled, rev_in, deep=False)
        rev = list(reversed(rev))
        perm = list(range(len(inputs)))
        random.Random(1).shuffle(perm)
        _, _, _, shf = checkpoint(args.name, labeled, [inputs[i] for i in perm], deep=False)
        unshf = [None] * len(inputs)
        for j, i in enumerate(perm):
            unshf[i] = shf[j]
        # a worker that serves only every 4th call (one of 4 workers)
        _, _, _, part = checkpoint(args.name, labeled, inputs[2::4], deep=False)
        d_rev = max(abs(a - b) for a, b in zip(fwd, rev))
        d_shf = max(abs(a - b) for a, b in zip(fwd, unshf))
        d_part = max(abs(a - b) for a, b in zip(fwd[2::4], part))
        worst = max(worst, d_rev, d_shf, d_part)
        print(f"{len(labeled)} labels, {len(inputs)} calls: max diff reversed {d_rev:.1e}, shuffled {d_shf:.1e}, "
              f"one-of-4-workers subset {d_part:.1e}", flush=True)
    print("order-invariant (bit-identical)" if worst == 0.0 else f"ORDER-DEPENDENT: worst {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["budgets", "scaling", "pool", "order"])
    ap.add_argument("--name", default="v6_combo_w20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="372,744,1488")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=110)
    args = ap.parse_args()
    {"budgets": cmd_budgets, "scaling": cmd_scaling, "pool": cmd_pool, "order": cmd_order}[args.cmd](args)


if __name__ == "__main__":
    main()
