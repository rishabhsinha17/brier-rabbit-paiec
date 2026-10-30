"""Extra summaries for the report (swarm round 6): screens by reason, null z tails and pool truths of population-null
switches, K of finalists, commits. Exact-SE records only. usage: R swarm/round6/oc_sims/oc_extra.py"""
from __future__ import annotations

import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
TAGS = ["W20", "L15", "W15", "m250", "m225", "S6", "S7"]


def load():
    d = defaultdict(dict)
    for f in sorted(glob.glob(str(HERE / "cache" / "oc_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("final_se") != "exact":
                continue
            d[(r["scenario"], r["model"], r["strata"])][r["wseed"]] = r
    return {k: list(v.values()) for k, v in d.items()}


def main():
    G = load()
    print("== screens: drops per started arm (runs >= 6) and reasons")
    for k, rows in sorted(G.items()):
        runs = np.array([r["runs"] for r in rows])
        drop = np.array([r["dropped"] for r in rows])
        st = runs >= 6
        why = Counter()
        for r in rows:
            for w in r["drop_why"]:
                for p in (w.split("+") if w else []):
                    why[p] += 1
        per_arm = ", ".join(f"{TAGS[a]} {drop[st[:, a], a].mean():.2f}" for a in range(7) if st[:, a].any())
        n = int(st.sum())
        p = drop[st].mean()
        print(f"  {k}: {len(rows)} worlds, {n} started arm-worlds, drop {p:.3f} ± {math.sqrt(p * (1 - p) / n):.3f}; "
              f"reasons {dict(why)}; per arm {per_arm}")
    print("\n== null scenarios: finalists' z and population-null switches")
    for k, rows in sorted(G.items()):
        if k[0] != "null":
            continue
        zs = np.array([r["full_z"][a] for r in rows for a in r["eligible"]])
        Ks = np.array([r["K_full"][a] for r in rows for a in r["eligible"]])
        pt = np.array([r["truth"] for r in rows]) * 1e3
        sw = [r for r in rows if r["ship"] >= 0]
        print(f"  {k}: {len(rows)} worlds; finalists {len(zs)}: z mean {zs.mean():+.2f}, SD {zs.std(ddof=1):.2f}, "
              f"P(z <= -1.8) {(zs <= -1.8).mean():.4f} (normal with same mean/SD {norm.cdf((-1.8 - zs.mean()) / zs.std()):.4f}); "
              f"K min {Ks.min()}, median {np.median(Ks):.0f}; SD of pool truth by arm {np.round(pt.std(0), 2).tolist()}")
        for r in sw:
            a = r["ship"]
            print(f"     switch -> {TAGS[a]}: pool truth {1e3 * r['truth'][a]:+.2f} (best arm on the pool "
                  f"{1e3 * min(r['truth']):+.2f}), full est {1e3 * r['full_alc'][a]:+.2f}, z {r['full_z'][a]:+.2f}, "
                  f"finalists {r['n_final']}")
    print("\n== alternatives: effect arm K_full, commits")
    for k, rows in sorted(G.items()):
        if not k[0].startswith("one:"):
            continue
        a = int(k[0].split(":")[1])
        Ks = np.array([r["K_full"][a] for r in rows])
        com = [r for r in rows if r["commit_day"] >= 0]
        right = sum(1 for r in com if r["commit"][0] == a)
        shipped = sum(1 for r in com if r["ship"] == a)
        days = [r["commit_day"] for r in com]
        print(f"  {k}: effect arm K_full min {Ks.min()}, median {np.median(Ks):.0f}, max {Ks.max()}; commits {len(com)}/"
              f"{len(rows)}, leader = effect arm {right}, shipped it {shipped}, day median "
              f"{np.median(days) if days else float('nan'):.0f}")
    allK = np.array([r["K_full"][a] for rows in G.values() for r in rows for a in r["eligible"]])
    print(f"\nall finalists: K min {allK.min()}, 1% {np.quantile(allK, .01):.0f}, median {np.median(allK):.0f}, max "
          f"{allK.max()}; worlds with exact final SE: {sum(len(v) for v in G.values())}")


if __name__ == "__main__":
    main()
