"""Small targeted checks (public data / synthetic):
A. simpool within-benchmark same-name draws at the d06 level block: defaults vs the variance-share patch (SimPoolCorr).
B. item-key collisions: distinct item_content vs distinct paiec.hier._item_key per public benchmark, and items sharing
   identical content (ItemPoolModel would pool every partner label of colliding items onto one item effect).
C. hrB_exact latent double count: a subject that also has labels on another benchmark gets phi's posterior from the
   FULL fit (which already contains the pair's own labels) and then multiplies the own labels in again.
D. BCM configuration prior sums to 1 (beta-binomial exchangeable hardness).
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/misc_checks.py
"""
from __future__ import annotations

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, D06, PLUGINS, load_cache, make  # noqa: E402
from simpool_xname import SimPoolCorr  # noqa: E402

import numpy as np  # noqa: E402

from paiec.hier import HierModel, _item_key  # noqa: E402
from paiec.protocol import BUDGETS, materialize, pair_alc, run_protocol, unit  # noqa: E402
from paiec.subjects import name_key  # noqa: E402


def check_a():
    raw, priors = load_cache()
    print("A. within-benchmark same-name draws, ALC diff vs base x1e-3 (12 draws = 24 pair-draws per row)")
    for label, params in (("cur_final", BASE), ("d06", D06)):
        for b in ("multi_swebench", "matharena"):
            rps = raw[b]
            keys = [name_key(rp["subject"]) for rp in rps]
            cand = [(i, j) for i, j in itertools.combinations(range(len(rps)), 2) if keys[i] and keys[i] == keys[j]]
            cand = sorted(cand, key=lambda ij: unit("same", b, ij[0], ij[1]))[:12]
            models = {"base": HierModel(priors[b], dict(params)),
                      "sp": make(PLUGINS["simpool"], priors[b], dict(params)),
                      "sp_corr": SimPoolCorr(priors[b], dict(params))}
            res = {n: [] for n in models}
            for r, (i, j) in enumerate(cand):
                pairs = materialize([rps[i], rps[j]], size_range=(88, 226), seed=r)
                for n, m in models.items():
                    out = run_protocol(pairs, m.predict, seed=r)
                    res[n] += [pair_alc(out[k]) for k in sorted(out)]
            ref = np.array(res["base"])
            line = " | ".join(f"{n} {1e3 * (np.array(v) - ref).mean():+.2f} +- {1e3 * (np.array(v) - ref).std() / np.sqrt(len(ref)):.2f}"
                              for n, v in res.items() if n != "base")
            print(f"   {label:9s} {b:15s} {line}", flush=True)


def check_b():
    raw, _ = load_cache()
    print("B. item-key collisions (public eligible pairs)")
    for b, rps in raw.items():
        contents, keys = {}, {}
        for rp in rps:
            for iid, it in rp["items"].items():
                c = str(it["item"]["item_content"])
                contents.setdefault(c, set()).add(iid)
                keys.setdefault(_item_key(it["item"]), set()).add(c)
        ids = set().union(*contents.values())
        dup_content = sum(len(v) - 1 for v in contents.values() if len(v) > 1)
        key_coll = sum(len(v) - 1 for v in keys.values() if len(v) > 1)
        empty = sum(1 for c in contents if not c.strip())
        print(f"   {b:18s} item ids {len(ids):5d} | distinct contents {len(contents):5d} (ids sharing content: {dup_content}) |"
              f" distinct item keys {len(keys):5d} (key collisions between different contents: {key_coll}) | empty content {empty}")


def check_c():
    print("C. hrB_exact double count when the subject also has labels on another benchmark")
    from paiec.ext.hrB_exact import ExactAloneHier

    prm = dict(BASE)
    S = {"normalized_name": "zz-unknown", "provider": ""}

    def it(i, b):
        return {"item_content": f"q{b}{i}", "item_features": "", "interactors": "", "benchmark_id": b}

    own = [[[S, it(i, "b1")], int(i < 1)] for i in range(7)]          # 1/7 on b1
    for extra_name, extra in (("none", []), ("1 label (1) on b2", [[[S, it(0, "b2")], 1]]),
                              ("1 label (0) on b2", [[[S, it(0, "b2")], 0]])):
        lab = own + extra
        h, x = HierModel({}, prm).predict([S, it(99, "b1")], lab), ExactAloneHier({}, prm).predict([S, it(99, "b1")], lab)
        print(f"   own 1/7 on b1, other-benchmark evidence {extra_name:18s}: HierModel {h:.4f} | ExactAloneHier {x:.4f}")


def check_d():
    m = make(PLUGINS["bcm"], {}, dict(BASE, mix_pi=0.25))
    worst = 0.0
    for rho in (0.0, 0.3, 0.6, 0.9):
        m.p["hard_rho"] = rho
        for K in (2, 3, 4):
            tot = sum(math.exp(m._config_prior(K, sum(z))) for z in itertools.product((0, 1), repeat=K))
            worst = max(worst, abs(tot - 1.0))
    print(f"D. BCM configuration prior: max |sum over configurations - 1| = {worst:.1e}")


if __name__ == "__main__":
    check_b()
    check_c()
    check_d()
    check_a()
