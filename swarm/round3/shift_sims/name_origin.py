"""Section 1d: transfer of name abilities by ORIGIN of the name's evidence (LOBO, public data only).

For held-out benchmark b, a name-source pair's theta is the shrunk mean centred logit over the other benchmarks the
name appears on. Split by which benchmarks supplied it: MathArena only, non-MathArena only, mixed. Report the
within-benchmark slope of lg on theta and the B0 offset (+ = under-predicted) at the benchmark-optimal level.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calib_lobo as cl  # noqa: E402
import shiftlib as sl  # noqa: E402


def main():
    pd.set_option("display.width", 220)
    df = sl.load()
    pt = sl.pair_table(df)
    benches_of = pt[pt.key != ""].groupby("key").bench.apply(set)
    for scale, ymax, lab in ((1.0, 0.0, "base"), (1.56, 2026.4, "a156+clamp")):
        g = cl.build(scale, ymax)
        orig = []
        for _, r in g.iterrows():
            if r.src != "name":
                orig.append("attr")
                continue
            other = benches_of.get(r.key, set()) - {r.bench}
            if other == {"matharena"}:
                orig.append("name:MA-only")
            elif "matharena" in other:
                orig.append("name:MA+other")
            else:
                orig.append("name:non-MA")
        g["origin"] = orig
        print(f"\n== {lab}: by origin of the transferred ability (all LOBO pairs; off at the benchmark-optimal level)")
        print(cl.table(g, "origin").round(3).to_string(index=False))
        h = g[g.bench != "matharena"]
        print(f"-- excluding MathArena as the held-out benchmark (hidden-like: non-math targets), n={len(h)}")
        t = cl.table(h, "origin")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
