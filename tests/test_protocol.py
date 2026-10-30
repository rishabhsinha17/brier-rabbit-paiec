import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiec.data import SUBJECT_FIELDS  # noqa: E402
from paiec.protocol import BUDGETS, WEIGHTS, MAX_LABELS, build_pairs, pair_alc, run_protocol, summarize  # noqa: E402


def synthetic_df(n_pairs=3, n_items=(80, 101, 150), runs=1, bench="bench_a"):
    rows = []
    for p in range(n_pairs):
        for i in range(n_items[p % len(n_items)]):
            for t in range(runs):
                row = {f: "" for f in SUBJECT_FIELDS}
                row.update(normalized_name=f"Model {p}", bench=bench, benchmark_id="benchmark_123456",
                           pair=f"{bench}|s{p}", subject_id=f"s{p}", item_id=f"i{i}", trial=t,
                           label=(i + t + p) % 2, item_content=f"question {i}", item_features="", interactors="")
                rows.append(row)
    return pd.DataFrame(rows)


def test_weights_are_the_trapezoid_on_log2():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12
    assert [WEIGHTS[b] for b in BUDGETS] == [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]


def test_split_is_half_with_extra_to_acquisition_and_disjoint():
    pairs = build_pairs(synthetic_df(), seed=0)
    for p in pairs:
        assert len(p.acq) == (p.n_items + 1) // 2
        assert set(p.acq).isdisjoint(p.eval)
        assert set(p.acq) | set(p.eval) == set(p.items)


def test_ineligible_pairs_dropped():
    pairs = build_pairs(synthetic_df(n_items=(79, 80, 81)), seed=0)
    assert sorted(p.n_items for p in pairs) == [80, 81]


def test_random_acquisition_is_nested_and_reaches_cap():
    pairs = build_pairs(synthetic_df(), seed=1)
    seen = {}

    def predict(input, labeled=None):
        n = len(labeled or [])
        seen.setdefault(n, 0)
        return 0.5

    res = run_protocol(pairs, predict, seed=1)
    for p in pairs:
        used = [res[p.key][b]["labels_used"] for b in BUDGETS]
        assert used == [min(b, len(p.acq)) for b in BUDGETS]
    # evidence size seen by predict() equals the sum of caps across pairs at each checkpoint
    expected = {sum(min(b, len(p.acq)) for p in pairs) for b in BUDGETS}
    assert set(seen) == expected


def test_evaluation_items_never_revealed():
    pairs = build_pairs(synthetic_df(), seed=2)
    # Item text is shared across subjects here, so the forbidden unit is the (subject, item) combination.
    eval_inputs = {json.dumps([p.subject, p.items[i]["item"]], sort_keys=True) for p in pairs for i in p.eval}
    revealed = []

    def predict(input, labeled=None):
        for (subject, item), _ in labeled or []:
            assert json.dumps([subject, item], sort_keys=True) not in eval_inputs
            revealed.append(1)
        return 0.5

    res = run_protocol(pairs, predict, seed=2)
    assert summarize(res)["alc"] == pytest.approx(0.25)


def test_constant_predictor_has_flat_curve():
    pairs = build_pairs(synthetic_df(runs=3), seed=3)
    res = run_protocol(pairs, lambda input, labeled=None: 0.3, seed=3)
    for r in res.values():
        assert pair_alc(r) == pytest.approx(r[0]["brier"])
        assert r[0]["rows"] == 3 * r[0]["unique"]


def test_stream_hook_accept_all_takes_candidates_in_order():
    pairs = build_pairs(synthetic_df(n_pairs=1, n_items=(100,)), seed=4)
    taken = []

    def hook(input, prediction=None, labeled=None, context=None):
        assert prediction == 0.5 and context["max_labels"] == MAX_LABELS
        taken.append(input[1]["item_content"])
        return True

    run_protocol(pairs, lambda input, labeled=None: 0.5, acquisition_function=hook, seed=4)
    p = pairs[0]
    assert taken == [p.items[i]["item"]["item_content"] for i in p.acq[:31]]


def test_hook_without_prediction_parameter_gets_none_and_must_return_bool():
    pairs = build_pairs(synthetic_df(n_pairs=1, n_items=(100,)), seed=5)
    calls = {"predict": 0}

    def predict(input, labeled=None):
        calls["predict"] += 1
        return 0.5

    def hook(input, *, labeled=None, context=None):
        return context["items_remaining"] <= context["labels_remaining"] or len(labeled) % 2 == 0

    run_protocol(pairs, predict, acquisition_function=hook, seed=5)
    assert calls["predict"] == 6 * len(pairs[0].eval)  # no extra predictor calls for the hook

    with pytest.raises(ValueError):
        run_protocol(pairs, predict, acquisition_function=lambda input, prediction=None, labeled=None, context=None: 1)


def test_deterministic_given_seed():
    pairs = build_pairs(synthetic_df(), seed=6)
    model = lambda input, labeled=None: 0.25 + 0.5 * (len(labeled or []) % 2)  # noqa: E731
    a = summarize(run_protocol(pairs, model, seed=6))
    b = summarize(run_protocol(pairs, model, seed=6))
    assert a == b
