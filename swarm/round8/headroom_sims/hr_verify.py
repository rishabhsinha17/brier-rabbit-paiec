"""Verification of the round-8 mixture candidates (params-only arms: HierModel's existing mixture branch).

1. Calculus exactness: the alone-pair calculus table (hr_fast.table) == paiec.hier.HierModel.predict on synthetic
   single pairs, items without item_features (bracket 0) and all items in one item_features group (bracket 1).
2. Never-raise fuzz: HierModel and the shipping composite ComboModel (S+C+P switches on) at the candidate params on
   malformed evidence / inputs: always a float strictly inside (0, 1), no exception, no argument mutation.
3. Identity at neutral settings: mix_pi = 0 at the candidate's normal component == the plain family arm (HierModel)
   bit for bit, and ComboModel with every switch off == HierModel bit for bit at the candidate params, on synthetic
   multi-pair draws (alone and shared benchmarks, groups, unseen and partner-labelled items).
usage: R swarm/round8/headroom_sims/hr_verify.py
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np

import hr_common as hc  # noqa: F401  (sets sys.path)
import hr_fast as hf
from paiec.ext.combo import ComboModel
from paiec.hier import HierModel

HERE = Path(__file__).resolve().parent
t0 = time.time()
CANDS = {n: json.loads((HERE / f"params_{n}.json").read_text()) for n in ("mxs", "mxh")}
COMBO = {n: json.loads((HERE / f"params_{n}_combo.json").read_text()) for n in ("mxs", "mxh")}
rng = np.random.default_rng(0)
subj = {"normalized_name": "probe-x"}

# ---- 1. calculus exactness ---------------------------------------------------------------------------------------
for name, p in CANDS.items():
    spec = {"m0": p["m0"], "V": p["tau_mu"] ** 2 + p["tau_phi"] ** 2 + p["tau_delta"] ** 2, "mix_pi": p["mix_pi"],
            "mix_m": p["mix_m"], "mix_s": p["mix_s"], "sigma_i": p["sigma_i"]}
    worst = {0: 0.0, 1: 0.0}
    for case in range(120):
        th, tv = float(rng.normal(0, 0.6)), float(rng.choice([0.1477, 0.25, 0.2793, 0.3333, 0.5]))
        n = int(rng.choice(hc.BUD))
        k = int(rng.integers(0, n + 1))
        for br in (0, 1):
            feat = "grp=a" if br else ""
            it = lambda j, f=feat: {"item_content": f"i{j}", "item_features": f, "interactors": "", "benchmark_id": "bx"}
            mdl = HierModel({}, p)
            mdl._theta = lambda s, th=th, tv=tv: (th, tv)
            lab = [[[subj, it(j)], int(j < k)] for j in range(n)]
            ref = mdl.predict([subj, it(999)], lab)
            q = hf.table(spec, 0.0, br, np.array([th]), np.array([tv]))[0, hc.KN.index((k, n))]
            worst[br] = max(worst[br], abs(ref - q))
    print(f"[1] {name}: calculus vs HierModel.predict, max |diff| bracket0 {worst[0]:.2e} bracket1 {worst[1]:.2e}")

# ---- 2. never-raise fuzz ------------------------------------------------------------------------------------------
from paiec.hier import fit_subject_prior  # noqa: E402
from paiec.data import load_responses  # noqa: E402
df = load_responses()
prior = fit_subject_prior(df[df.bench != "real_webagents"])
good_item = {"item_content": "x" * 50, "item_features": "lang=py", "interactors": "", "benchmark_id": "b1"}
good_subj = {"normalized_name": "gpt-4o", "provider": "openai", "release_date": "2024-05-13"}
bad_labeled = [None, [], [None], [[None, None]], [[[good_subj, good_item], "x"]], [[[good_subj, None], 1]],
               [[[None, good_item], 1]], [[[good_subj, good_item], 1, 2]], [[[{}, {}], 1]], "garbage", 17,
               [[[good_subj, dict(good_item, item_content=None)], 1]], [[[good_subj, dict(good_item, benchmark_id=None)], 0]],
               [[[dict(good_subj, release_date="not-a-date"), good_item], 1]] * 3,
               [[[good_subj, dict(good_item, item_content=f"i{j}")], j % 2] for j in range(40)]]
bad_inputs = [None, [], [good_subj], [good_subj, None], [None, good_item], [good_subj, good_item], "x",
              [dict(good_subj, normalized_name=None), good_item], [good_subj, dict(good_item, item_features=None)],
              [{"provider": 3}, {"benchmark_id": 5}]]
for name in CANDS:
    for label, mk in (("HierModel", lambda: HierModel(prior, CANDS[name])),
                      ("ComboModel S+C+P", lambda: ComboModel(prior, COMBO[name]))):
        n_calls, bad = 0, 0
        for lab in bad_labeled:
            for inp in bad_inputs:
                m = mk()
                lab_c, inp_c = copy.deepcopy(lab), copy.deepcopy(inp)
                try:
                    v = m.predict(inp, lab)
                    ok = isinstance(v, float) and math.isfinite(v) and 0.0 < v < 1.0
                except Exception:
                    ok = False
                if lab != lab_c or inp != inp_c:
                    ok = False
                n_calls += 1
                bad += (not ok)
        print(f"[2] {name} {label}: {n_calls} fuzz calls, failures {bad}")


# ---- 3. identity at neutral settings --------------------------------------------------------------------------------
def draw_calls(seed):
    """Synthetic multi-pair evidence: 4 benchmarks, 2 of them shared by 2-3 subjects, groups, partner-labelled items."""
    r = np.random.default_rng(seed)
    names = ["gpt-4o", "claude-3-5-sonnet", "llama-3-70b", "unknown-model-z", "qwen2.5-72b"]
    subs = [{"normalized_name": nm, "provider": "p", "release_date": "2025-0%d-01" % (1 + i)} for i, nm in enumerate(names)]
    lab, calls = [], []
    for b in range(4):
        who = r.choice(len(subs), size=(1 if b < 2 else int(r.integers(2, 4))), replace=False)
        items = [{"item_content": f"b{b}_i{j}", "item_features": ("t=%d" % (j % 3)) if b % 2 else "",
                  "interactors": "", "benchmark_id": f"bench{b}"} for j in range(30)]
        for s in who:
            for j in r.choice(30, size=int(r.integers(0, 12)), replace=False):
                lab.append([[subs[s], items[j]], int(r.random() < 0.4)])
            for j in r.choice(30, size=4, replace=False):
                calls.append([subs[s], items[j]])
    return lab, calls


for name, p in CANDS.items():
    fam = {k: v for k, v in p.items() if not k.startswith("mix_")}
    n, ex1, ex2, w1, w2 = 0, 0, 0, 0.0, 0.0
    for seed in range(12):
        lab, calls = draw_calls(seed)
        for nested in (lab[: len(lab) // 2], lab):
            a0, a1 = HierModel(prior, dict(p, mix_pi=0.0)), HierModel(prior, fam)
            c0, h0 = ComboModel(prior, dict(p)), HierModel(prior, dict(p))
            for inp in calls:
                x, y = a0.predict(inp, nested), a1.predict(inp, nested)
                u, v = c0.predict(inp, nested), h0.predict(inp, nested)
                n += 1
                ex1 += x == y
                ex2 += u == v
                w1, w2 = max(w1, abs(x - y)), max(w2, abs(u - v))
    print(f"[3] {name}: mix_pi 0 == family arm: {ex1}/{n} bit-exact (max {w1:.1e}); "
          f"ComboModel switches off == HierModel: {ex2}/{n} bit-exact (max {w2:.1e})")
print(f"total {time.time() - t0:.0f}s")
