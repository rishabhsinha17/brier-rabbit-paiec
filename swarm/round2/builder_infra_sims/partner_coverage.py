"""B31 partner-label coverage on k=2 public draws: pair-keyed vs benchmark-keyed item subsample (round 2 builder).

usage: nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/builder_infra_sims/partner_coverage.py [runs]
For each public benchmark with >= 2 pairs and runs r = 0..runs-1, FoldData picks the same two pairs in both modes; the
table reports the mean over pairs of paiec.evaluate.partner_coverage (share of a pair's unique evaluation items that
the co-drawn pair has a revealed label for after 31 labels each, default sha256 acquisition), the Jaccard overlap of
the two kept item sets, and the same coverage with no subsample at all ("full": complete public groups, which are larger
than platform groups, so their per-item reveal rate is lower).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paiec.data import BINARY_BENCHMARKS, load_responses  # noqa: E402
from paiec.evaluate import FoldData, partner_coverage  # noqa: E402
from paiec.protocol import materialize, unit  # noqa: E402


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    t0 = time.time()
    df = load_responses()
    fd = FoldData(df, BINARY_BENCHMARKS)
    rows = []
    for bench in BINARY_BENCHMARKS:
        raw = fd.raw[bench]
        if len(raw) < 2:
            continue
        stat = {m: {"cov": [], "jac": []} for m in ("pair", "bench", "full")}
        for mode in ("pair", "bench"):
            fd.subsample = mode
            for r, pairs in fd.draws(bench, 2, runs):
                stat[mode]["cov"].extend(partner_coverage(pairs, seed=r).values())
                a, b = (set(p.items) for p in pairs)
                stat[mode]["jac"].append(len(a & b) / len(a | b))
                if mode == "pair":  # same two pairs, complete groups
                    keys = sorted((rp["key"] for rp in raw), key=lambda key: unit("pick", bench, 2, r, key))[:2]
                    full = materialize([rp for rp in raw if rp["key"] in set(keys)], size_range=None, seed=r)
                    stat["full"]["cov"].extend(partner_coverage(full, seed=r).values())
                    a, b = (set(p.items) for p in full)
                    stat["full"]["jac"].append(len(a & b) / len(a | b))
        fd.subsample = "pair"
        sizes = [len(rp["items"]) for rp in raw]
        rows.append((bench, len(raw), int(np.median(sizes)), *(np.mean(stat[m][x]) for m in ("pair", "bench", "full")
                                                               for x in ("cov", "jac")),
                     np.std(np.array(stat["bench"]["cov"]) - np.array(stat["pair"]["cov"]), ddof=1)
                     / np.sqrt(len(stat["pair"]["cov"])), len(stat["pair"]["cov"])))
    print(f"k=2 draws, runs={runs}; coverage = mean over pairs of the share of eval items with a partner label at B31")
    print(f"{'bench':18s} {'pairs':>5s} {'med_items':>9s} | {'cov_pair':>8s} {'jac_pair':>8s} | {'cov_bench':>9s} "
          f"{'jac_bench':>9s} | {'cov_full':>8s} {'jac_full':>8s} | {'se(diff)':>8s} {'n':>4s}")
    for b, n, med, cp, jp, cb, jb, cf, jf, se, n_obs in rows:
        print(f"{b:18s} {n:5d} {med:9d} | {cp:8.3f} {jp:8.3f} | {cb:9.3f} {jb:9.3f} | {cf:8.3f} {jf:8.3f} | "
              f"{se:8.3f} {n_obs:4d}")
    print(f"# {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
