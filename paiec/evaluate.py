"""Leave-one-benchmark-out replay helpers shared by experiments."""
from __future__ import annotations

import numpy as np

from .protocol import BUDGETS, check_subsample, index_pairs, materialize, pair_alc, run_protocol, unit

SIZE_RANGE = (88, 226)  # platform eval pools were 44-113 unique items per pair


class FoldData:
    """Per-benchmark indexed pairs, built once and reused across models and seeds.

    subsample="pair" (default) reproduces every existing result; "bench" keys the item subsample on the benchmark so
    co-drawn pairs of one benchmark keep the same items where they overlap (see protocol.materialize)."""

    def __init__(self, df, benches, subsample: str = "pair"):
        self.df = df
        self.subsample = check_subsample(subsample)
        self.raw = {b: index_pairs(df[df.bench == b]) for b in benches}

    def draws(self, bench: str, k: int, runs: int, seed0: int = 0):
        raw = self.raw[bench]
        if len(raw) < k:
            return
        n = runs if len(raw) > 1 else min(runs, 10)
        for r in range(seed0, seed0 + n):
            keys = sorted((rp["key"] for rp in raw), key=lambda key: unit("pick", bench, k, r, key))[:k]
            chosen = materialize([rp for rp in raw if rp["key"] in set(keys)], size_range=SIZE_RANGE, seed=r,
                                 subsample=self.subsample)
            yield r, chosen


def replay(fold_data: FoldData, bench: str, predict, ks=(1, 2), runs: int = 20, seed0: int = 0, hook=None) -> dict:
    """Mean pair ALC and per-budget Brier on draws from a held-out benchmark."""
    out = {}
    for k in ks:
        alcs, per_budget = [], {b: [] for b in BUDGETS}
        for r, pairs in fold_data.draws(bench, k, runs, seed0) or []:
            res = run_protocol(pairs, predict, acquisition_function=hook, seed=r)
            for pr in res.values():
                alcs.append(pair_alc(pr))
                for b in BUDGETS:
                    per_budget[b].append(pr[b]["brier"])
        if alcs:
            out[k] = {"alc": float(np.mean(alcs)), "se": float(np.std(alcs) / np.sqrt(len(alcs))),
                      "n_pairs": len(alcs), "per_budget": {b: float(np.mean(v)) for b, v in per_budget.items()}}
    return out


def partner_coverage(pairs, seed: int, budget: int = 31) -> dict:
    """Cross-subject pooling opportunity of one draw: for every pair with a co-drawn pair of the same benchmark, the
    share of its unique evaluation items that such a partner has a revealed label for once every pair holds `budget`
    labels under the default (sha256 selection) acquisition. Pairs alone on their benchmark are omitted."""
    import json

    snaps = {}

    def predict(input, labeled=None):  # every call at one checkpoint sees the same evidence list
        snaps.setdefault(len(labeled or []), list(labeled or []))
        return 0.5

    run_protocol(pairs, predict, seed=seed)
    n_at = sum(min(budget, len(p.acq)) for p in pairs)
    where = {json.dumps([p.subject, p.items[i]["item"]], sort_keys=True, default=str): (p.key, i)
             for p in pairs for i in p.acq}
    revealed: dict = {}
    for inp, _ in snaps.get(n_at, []):
        key, item_id = where[json.dumps(inp, sort_keys=True, default=str)]
        revealed.setdefault(key, set()).add(item_id)
    out = {}
    for p in pairs:
        partners = [q for q in pairs if q.bench == p.bench and q.key != p.key]
        if partners and p.eval:
            seen = set().union(*(revealed.get(q.key, set()) for q in partners))
            out[p.key] = sum(i in seen for i in p.eval) / len(p.eval)
    return out
