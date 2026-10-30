"""Robustness fuzz for every paiec/ext plug-in: malformed input / labeled must never raise and must return a finite
float strictly inside (0, 1). Also checks that predict() never mutates its arguments. Synthetic data + the shipped
subject prior (submission_src/v3b_noM_n_m0n20/subject_prior.json); no public item data needed.

usage: nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/fuzz.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, MIX25, PLUGINS, ROOT, make  # noqa: E402

PRIOR = json.load(open(os.path.join(ROOT, "submission_src/v3b_noM_n_m0n20/subject_prior.json")))

CONFIGS = {  # recommended / queued settings of each plug-in, with and without the hard mixture
    "base": (None, {}),
    "subjcal a1.56": ("subjcal", {"attr_scale": 1.56, "attr_year_max": 2026.4}),
    "prior q0.25 d06": ("prior", {"q0": 0.25, "tau_mu": 1.2, "tau_delta": 0.6, "tau_phi": 0.3}),
    "prior tilt mix": ("prior", dict(MIX25, mix_b=-1.4)),
    "power 1.3": ("power", {"lik_power": 1.3}),
    "power 1.3 mix": ("power", dict(MIX25, lik_power=1.3)),
    "simpool": ("simpool", {}),
    "simpool mix": ("simpool", dict(MIX25)),
    "itempool x2": ("itempool", {"sigma_x": 2.0}),
    "itempool x2 mix": ("itempool", dict(MIX25, sigma_x=2.0)),
    "hrb": ("hrb", {}),
    "hrb mix": ("hrb", dict(MIX25)),
    "exactall": ("exactall", {}),
    "exactall mix": ("exactall", dict(MIX25)),
    "bcm rho.3 mix": ("bcm", dict(MIX25, hard_rho=0.3)),
}

NAN = float("nan")


def subj(name="GPT-4o", **kw):
    d = {"normalized_name": name, "provider": "OpenAI", "release_date": "2024-05-13", "access_date": "",
         "harness": "", "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
    d.update(kw)
    return d


def item(i, b="benchmark_111111", g="lang=py", content=None):
    return {"item_content": content if content is not None else f"task {b} {i}", "item_features": g,
            "interactors": "", "benchmark_id": b}


S1, S2, S3 = subj(), subj("Totally New Model 9", provider="Acme", release_date="2026-09-01", harness="AgentX"), \
    subj("claude-3.7-sonnet", provider="Anthropic", harness="AgentX")
GOOD = [S1, item(1)]


def normal_evidence(n=7):
    lab = []
    for j in range(n):
        lab.append([[S1, item(100 + j)], j % 3 == 0])
        lab.append([[S2, item(100 + j)], int(j % 2 == 0)])           # same items: item pooling paths
        lab.append([[S3, item(200 + j, "benchmark_222222")], 0])    # other benchmark
    return lab


def bad_inputs():
    yield "None", None
    yield "empty list", []
    yield "[None, None]", [None, None]
    yield "[{}, {}]", [{}, {}]
    yield "subject ok, item {}", [S1, {}]
    yield "subject {}, item ok", [{}, item(1)]
    yield "item fields None", [S1, {"benchmark_id": None, "item_content": None, "item_features": None, "interactors": None}]
    yield "item fields NaN", [S1, {"benchmark_id": NAN, "item_content": NAN, "item_features": NAN, "interactors": NAN}]
    yield "subject fields None", [{k: None for k in S1}, item(1)]
    yield "subject fields NaN", [{k: NAN for k in S1}, item(1)]
    yield "subject numeric/nested", [{"normalized_name": 7, "provider": ["x"], "release_date": {"a": 1}, "harness": 3.5}, item(1)]
    yield "subject is str", ["GPT-4o", item(1)]
    yield "3-tuple input", [S1, item(1), "extra"]
    yield "tuple input", (S1, item(1))
    yield "weird features", [S1, item(1, g=" ;; =;lang;=x;a=b=c ; ")]
    yield "features only spaces", [S1, item(1, g="   ")]
    yield "huge content", [S1, item(1, content="x" * 2_000_000)]
    yield "non-str content", [S1, item(1, content=12345)]
    yield "unknown benchmark", [S1, item(1, b="benchmark_999999")]
    yield "release date garbage", [subj("Unseen Thing", release_date="20xx-yy"), item(1)]
    yield "future release", [subj("Unseen Thing", provider="OpenAI", release_date="2031-12-01"), item(1)]
    yield "target item in own labels", [S1, item(100)]
    yield "good", GOOD


def bad_labeled():
    ev = normal_evidence()
    yield "None", None
    yield "[]", []
    yield "[None]", [None]
    yield "[[None, 1]]", [[None, 1]]
    yield "label None", [[[S1, item(5)], None]]
    yield "label '1'", [[[S1, item(5)], "1"]]
    yield "label 'x'", [[[S1, item(5)], "x"]]
    yield "label 2 / -1", [[[S1, item(5)], 2], [[S1, item(6)], -1]]
    yield "label NaN", [[[S1, item(5)], NAN]]
    yield "label True/1.0", [[[S1, item(5)], True], [[S1, item(6)], 1.0]]
    yield "label [1]", [[[S1, item(5)], [1]]]
    yield "item None in entry", [[[S1, None], 1]]
    yield "subject str in entry", [[["GPT-4o", item(5)], 1]]
    yield "wrong nesting", [[[[S1, item(5)]], 1], [S1, item(5), 1], [[S1, item(6)], 1, "extra"]]
    yield "unknown benchmarks only", [[[S2, item(j, "benchmark_%d" % (300000 + j))], j % 2] for j in range(20)]
    yield "duplicates mixed", [[[S1, item(5)], y] for y in (0, 1, 0, 1, 1)] * 3
    yield "dup target item for partner", ev + [[[S2, item(1)], 1]] * 10
    yield "all ones 300 (one pair)", [[[S1, item(j)], 1] for j in range(300)]
    yield "all zeros 300 (one pair)", [[[S1, item(j)], 0] for j in range(300)]
    yield "same subject 2 benches", ev + [[[S1, item(j, "benchmark_222222")], 1] for j in range(7)]
    yield "NaN fields in entries", [[[{k: NAN for k in S1}, {"benchmark_id": NAN, "item_content": NAN, "item_features": NAN}], 1]] + ev
    yield "normal", ev
    rng = random.Random(0)
    big = []
    for j in range(1000):  # 9 subjects x 7 benchmarks, ~470 distinct items: refit on ~560 parameters
        s = subj(f"Model {j % 9}", harness=["", "A", "B"][j % 3])
        big.append([[s, item(j % 67, "benchmark_%d" % (400000 + j % 7), g=f"g={j % 5}")], int(rng.random() < 0.3)])
    yield "1000 entries 9 subj x 7 bench", big


def main():
    t0 = time.time()
    failures, n_calls, slow = [], 0, []
    for cname, (key, extra) in CONFIGS.items():
        params = dict(BASE, **extra)
        model = make(PLUGINS.get(key), PRIOR, params)
        for lname, lab in bad_labeled():
            for iname, inp in bad_inputs():
                if lname.startswith("1000 entries") and iname not in ("good", "unknown benchmark", "None", "item fields NaN"):
                    continue
                lab_c, inp_c = copy.deepcopy(lab), copy.deepcopy(inp)
                t = time.time()
                try:
                    v = model.predict(inp, lab)
                    err = None
                except Exception as e:  # noqa: BLE001
                    v, err = None, repr(e)[:120]
                dt = time.time() - t
                n_calls += 1
                bad = err or not isinstance(v, float) or isinstance(v, bool) or not math.isfinite(v) or not 0.0 < v < 1.0
                mut = False
                try:
                    mut = (repr(lab_c) != repr(lab)) or (repr(inp_c) != repr(inp))
                except Exception:  # noqa: BLE001
                    pass
                if bad or mut:
                    failures.append((cname, lname, iname, err or v, "MUTATED" if mut else ""))
                if dt > 0.25:
                    slow.append((round(dt, 3), cname, lname, iname))
        print(f"{cname:18s} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n{n_calls} fuzz calls; failures: {len(failures)}")
    for f in failures[:60]:
        print("  FAIL", f)
    slow.sort(reverse=True)
    print(f"calls slower than 250 ms: {len(slow)}")
    for s in slow[:25]:
        print("  SLOW", s)


if __name__ == "__main__":
    main()
