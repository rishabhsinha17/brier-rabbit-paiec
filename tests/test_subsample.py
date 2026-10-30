"""protocol.materialize subsample modes: 'pair' (default) must stay bit-exact; 'bench' must raise cross-pair overlap."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paiec.evaluate import FoldData, partner_coverage  # noqa: E402
from paiec.protocol import build_pairs, index_pairs, materialize  # noqa: E402
from test_protocol import synthetic_df  # noqa: E402

# sha256 digests of the pre-change materialize() (2026-10-06, before the subsample option existed) on the frame below.
GOLDEN = {
    "sized_seed0": "2fd5a5bcee85e000efcee7a928c0a9c137a70df4f2409556870dcc8c8755d820",
    "full_seed0": "f82f39c65b600ece30b3ba320d4258b9685766c72ad42e197596b753a82f7da0",
    "sized_seed1": "ab2bbf896af3653a34df1c0b7f14b86cc82a1b88d68ba5c504fe5a8041583992",
    "full_seed1": "d890add867e370a0e38dc9f00f6b7d95196e24a68d38b2c251ca1abcad882ebd",
    "sized_seed2": "70d9611358ba9eebee1569b36acf33e99057ba7d9e55b4b8e76c025176912b41",
    "full_seed2": "3bed04ff21d98a51f210574086e9b216eb7a731d72e0deebd6dd6a5e5e26da1d",
    "sized_seed3": "9bf59876bee0bf63b8bc4ded7e0e62c013eb665b99a4561674d623dae2c40286",
    "full_seed3": "dab9d369e62f56d4a597e1e17f24a877989f1bbc8c4b6b2deb3a56652488e16c",
}
GOLDEN_BUILD_SEED7 = "cbd9525057c77739ef455721b32482f590ba6c8565bb2e509168668bf25c301d"
SIZE = (88, 226)


def two_bench_df():
    return pd.concat([synthetic_df(n_pairs=3, n_items=(300, 260, 240), bench="bench_a"),
                      synthetic_df(n_pairs=2, n_items=(230, 120), bench="bench_b")], ignore_index=True)


def digest(pairs) -> str:
    h = hashlib.sha256()
    for p in pairs:
        h.update(json.dumps([p.key, p.bench, p.subject_id, sorted(p.items), p.acq, p.eval]).encode())
    return h.hexdigest()


def jaccard(a, b) -> float:
    a, b = set(a.items), set(b.items)
    return len(a & b) / len(a | b)


def test_default_is_bit_exact_with_pre_change_materialize():
    raw = index_pairs(two_bench_df())
    for seed in range(4):
        for kw in ({}, {"subsample": "pair"}):
            assert digest(materialize(raw, size_range=SIZE, seed=seed, **kw)) == GOLDEN[f"sized_seed{seed}"]
            assert digest(materialize(raw, seed=seed, **kw)) == GOLDEN[f"full_seed{seed}"]
    assert digest(build_pairs(two_bench_df(), size_range=SIZE, seed=7)) == GOLDEN_BUILD_SEED7


def test_bench_mode_only_changes_the_subsample():
    """Without size_range nothing is dropped, so both modes give the same pairs: split and order stay pair-keyed."""
    raw = index_pairs(two_bench_df())
    for seed in range(3):
        assert digest(materialize(raw, seed=seed, subsample="bench")) == digest(materialize(raw, seed=seed))
    for p in materialize(raw, size_range=SIZE, seed=0, subsample="bench"):
        assert len(p.acq) == (p.n_items + 1) // 2 and set(p.acq).isdisjoint(p.eval)
        assert set(p.acq) | set(p.eval) == set(p.items)


def test_bench_mode_keeps_identical_subsets_and_raises_overlap():
    raw = index_pairs(synthetic_df(n_pairs=2, n_items=(300,), bench="bench_a"))  # same 300 items for both subjects
    jac = {"pair": [], "bench": []}
    for seed in range(10):
        for mode in jac:
            a, b = materialize(raw, size_range=SIZE, seed=seed, subsample=mode)
            jac[mode].append(jaccard(a, b))
    assert all(j == 1.0 for j in jac["bench"])
    assert np.mean(jac["pair"]) < 0.7  # independent random subsets of 300 items


def test_bench_mode_nests_common_items_when_item_sets_differ():
    df = synthetic_df(n_pairs=2, n_items=(300,), bench="bench_a")
    df = df[~((df.subject_id == "s0") & (df.item_id.str[1:].astype(int) >= 250))]  # s0: i0..i249
    df = df[~((df.subject_id == "s1") & (df.item_id.str[1:].astype(int) < 50))]  # s1: i50..i299
    raw = index_pairs(df)
    nested = {"pair": 0, "bench": 0}
    for seed in range(10):
        for mode in nested:
            a, b = materialize(raw, size_range=SIZE, seed=seed, subsample=mode)
            common = set(raw[0]["items"]) & set(raw[1]["items"])
            ka, kb = set(a.items) & common, set(b.items) & common
            nested[mode] += ka <= kb or kb <= ka
    assert nested["bench"] == 10
    assert nested["pair"] == 0


def test_fold_data_k2_bench_mode_has_higher_overlap_and_partner_coverage():
    df = synthetic_df(n_pairs=2, n_items=(300,), bench="bench_a")
    stats = {}
    for mode in ("pair", "bench"):
        fd = FoldData(df, ["bench_a"], subsample=mode)
        jac, cov = [], []
        for r, pairs in fd.draws("bench_a", 2, 6):
            assert len(pairs) == 2
            jac.append(jaccard(*pairs))
            cov.extend(partner_coverage(pairs, seed=r).values())
        stats[mode] = (np.mean(jac), np.mean(cov))
    assert stats["bench"][0] > stats["pair"][0]
    assert stats["bench"][1] > stats["pair"][1]


def test_partner_coverage_is_zero_without_shared_items_and_skips_alone_pairs():
    df = pd.concat([synthetic_df(n_pairs=1, n_items=(120,), bench="bench_a"),
                    synthetic_df(n_pairs=1, n_items=(120,), bench="bench_b")], ignore_index=True)
    pairs = build_pairs(df, seed=0)
    assert partner_coverage(pairs, seed=0) == {}  # each pair is alone on its benchmark
    df2 = synthetic_df(n_pairs=2, n_items=(120,), bench="bench_a")
    df2.loc[df2.subject_id == "s1", "item_id"] = "x" + df2.loc[df2.subject_id == "s1", "item_id"]
    cov = partner_coverage(build_pairs(df2, seed=0), seed=0)
    assert set(cov) == {"bench_a|s0", "bench_a|s1"} and all(v == 0.0 for v in cov.values())


def test_invalid_subsample_mode_raises():
    raw = index_pairs(synthetic_df(n_pairs=1, n_items=(100,)))
    with pytest.raises(ValueError):
        materialize(raw, subsample="benchmark")
    with pytest.raises(ValueError):
        FoldData(synthetic_df(n_pairs=1, n_items=(100,)), ["bench_a"], subsample="item")
