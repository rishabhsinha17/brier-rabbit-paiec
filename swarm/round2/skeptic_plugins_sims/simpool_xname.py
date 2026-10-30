"""Re-run of the simpool smoke test cited in round 1 (l2o_names.py --modes xname, REAL x RCB: -0.0005 +- 0.0003 on 24
pair-draws) from the cached pairs and a leave-two-out subject prior (light loader, no item text), plus:
  - SimPoolCorr: tau_name / tau_h given as SHARES of the pair variance (rho_name 0.45, rho_h 0.288 reproduce the
    cur_final split exactly), so the pooling strength survives a level-block change such as d06;
  - the same comparisons at the in-flight d06 level block (tau_phi 0.3, tau_delta 0.6, m0 -1.693).
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/simpool_xname.py
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round1/postdoc_priors_sims"))
from common import BASE, D06, load_cache  # noqa: E402

import numpy as np  # noqa: E402

from paiec.ext.pdadapt_simpool import SimilarityPoolModel  # noqa: E402
from paiec.hier import HierModel, fit_subject_prior  # noqa: E402
from paiec.protocol import BUDGETS, materialize, pair_alc, run_protocol, unit  # noqa: E402
from paiec.subjects import name_key  # noqa: E402


class SimPoolCorr(SimilarityPoolModel):
    """Patch under test: pooled variances as shares of tau_phi^2 + tau_delta^2."""

    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        self.p.setdefault("rho_name", 0.45)
        self.p.setdefault("rho_h", 0.288)
        total = self.p["tau_phi"] ** 2 + self.p["tau_delta"] ** 2
        self.p["tau_name"] = math.sqrt(self.p["rho_name"] * total)
        self.p["tau_h"] = math.sqrt(self.p["rho_h"] * total)


def main():
    from light import load_light

    t0 = time.time()
    raw, _ = load_cache()
    held = ("real_webagents", "researchcodebench")
    df = load_light()
    prior = fit_subject_prior(df[~df.bench.isin(held)])
    print(f"leave-two-out prior built in {time.time() - t0:.0f}s ({len(prior['names'])} names)", flush=True)
    names = {b: {} for b in held}
    for b in held:
        for rp in raw[b]:
            nk = name_key(rp["subject"])
            if nk:
                names[b].setdefault(nk, []).append(rp)
    common = sorted(set(names[held[0]]) & set(names[held[1]]))
    draws = []
    for nm in common:
        pick = [sorted(names[b][nm], key=lambda rp: unit("xn", 0, rp["key"]))[0] for b in held]
        draws.append(pick)
    print(f"{len(common)} same-name draws (REAL x RCB)")
    variants = {
        "cur_final": (BASE, {"base": (HierModel, {}), "sp": (SimilarityPoolModel, {}),
                             "sp_small": (SimilarityPoolModel, {"tau_name": 0.5}), "sp_corr": (SimPoolCorr, {})}),
        "d06": (D06, {"base": (HierModel, {}), "sp": (SimilarityPoolModel, {}), "sp_corr": (SimPoolCorr, {})}),
    }
    if "--decouple" in sys.argv:
        print("subject dicts decoupled across benchmarks (hidden-pool-like)")
    for vname, (params, models) in variants.items():
        ms = {n: c(prior, dict(params, **o)) for n, (c, o) in models.items()}
        res = {n: [] for n in ms}
        for picks in draws:
            pairs = []
            for rp in picks:
                pairs += materialize([rp], size_range=(88, 226), seed=0)
            if "--decouple" in sys.argv:  # hidden-pool-like: no subject dict spans benchmarks (access_date feeds no prior)
                for pr in pairs:
                    pr.subject = dict(pr.subject, access_date="decoupled-" + pr.bench)
            for n, m in ms.items():
                out = run_protocol(pairs, m.predict, seed=0)
                for key in sorted(out):
                    res[n].append([pair_alc(out[key])] + [out[key][b]["brier"] for b in BUDGETS])
        ref = np.array(res["base"])
        for n in ms:
            if n == "base":
                continue
            d = np.array(res[n]) - ref
            print(f"  [{vname}] {n:9s} n={len(d)} ALC {1e3 * d[:, 0].mean():+.2f} +- {1e3 * d[:, 0].std() / np.sqrt(len(d)):.2f} | "
                  + " ".join(f"B{b} {1e3 * d[:, j + 1].mean():+.2f}" for j, b in enumerate(BUDGETS)), flush=True)
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
