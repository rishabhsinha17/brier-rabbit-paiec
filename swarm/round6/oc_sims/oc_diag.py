"""Why the final test is conservative: decomposition of the full-data ALC z of finalists (swarm round 6).

For every finalist in the exact-SE records of a scenario: the realised estimation error (estimate - the arm's truth on
the pool, or - the population value for the 'sup' models), the real cell-level bootstrap SE, CR1 (raw sightings),
the floor 14.7e-3/sqrt(K), and which part binds in SE_h. usage: R swarm/round6/oc_sims/oc_diag.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    recs = {}
    for f in sorted(glob.glob(str(HERE / "cache" / "oc_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("final_se") != "exact":
                continue
            recs.setdefault((r["scenario"], r["model"]), {})[r["wseed"]] = r
    print("| scenario | model | finalist-worlds | err SD vs pool truth | err SD vs population | mean boot SE | mean CR1 | "
          "mean floor | mean SE_h | binds boot/CR1/floor | K median |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (scn, model), d in sorted(recs.items()):
        e_pool, e_pop, bs, c1, fl, seh, bind, Ks = [], [], [], [], [], [], [], []
        for r in d.values():
            pop = r.get("pop_truth") or [0.0] * len(r["truth"])
            for a in r["eligible"]:
                est = r["full_est"][a][0]
                if est is None or not np.isfinite(est):
                    continue
                K = r["K_full"][a]
                b = 1.2 * r["full_boot"][a][0]
                c = r["full_cr1"][a][0]
                f = 0.0147 / math.sqrt(max(K, 1))
                s = max(b, c, f)
                e_pool.append(est - r["truth"][a])
                e_pop.append(est - pop[a])
                bs.append(r["full_boot"][a][0])
                c1.append(c)
                fl.append(f)
                seh.append(s)
                bind.append(int(np.argmax([b, c, f])))
                Ks.append(K)
        if not e_pool:
            continue
        bc = np.bincount(bind, minlength=3) / len(bind)
        print(f"| {scn} | {model} | {len(e_pool)} | {1e3 * np.std(e_pool):.2f} | {1e3 * np.std(e_pop):.2f} | "
              f"{1e3 * np.mean(bs):.2f} | {1e3 * np.mean(c1):.2f} | {1e3 * np.mean(fl):.2f} | {1e3 * np.mean(seh):.2f} | "
              f"{bc[0]:.2f}/{bc[1]:.2f}/{bc[2]:.2f} | {np.median(Ks):.0f} |")


if __name__ == "__main__":
    main()
