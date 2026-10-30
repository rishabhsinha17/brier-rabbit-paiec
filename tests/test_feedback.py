"""tools/feedback.py: corrected fixed-effects SE (fe_exact) and the public reference arms."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import feedback  # noqa: E402


def synthetic_cells(seed=0, n_units=40, arms=("ref", "a", "b", "c")):
    rng = np.random.default_rng(seed)
    effect = {"ref": 0.0, "a": 0.01, "b": -0.02, "c": 0.005}
    rows = []
    for u in range(n_units):
        alpha = rng.normal(0.2, 0.05)
        for a in rng.choice(arms, size=int(rng.integers(2, len(arms) + 1)), replace=False):
            rows.append({"pair": f"u{u}", "arm": str(a), "score": alpha + effect[a] + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def dummy_ols(cell, ref):
    """Brute force: score ~ unit dummies + non-ref arm dummies, classical SE with dof n - p."""
    multi = cell[cell.groupby("pair")["arm"].transform("nunique") > 1]
    units = sorted(multi["pair"].unique())
    est = sorted(set(multi["arm"]) - {ref})
    X = np.zeros((len(multi), len(units) + len(est)))
    for r, (u, a) in enumerate(zip(multi["pair"], multi["arm"])):
        X[r, units.index(u)] = 1.0
        if a != ref:
            X[r, len(units) + est.index(a)] = 1.0
    y = multi["score"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    k = len(units)
    return dict(zip(est, coef[k:])), dict(zip(est, np.sqrt(np.diag(cov)[k:])))


def test_fe_exact_matches_dummy_variable_ols():
    cell = synthetic_cells()
    beta, se, dof = feedback.fe_exact(cell, "ref")
    b2, se2 = dummy_ols(cell, "ref")
    for a in b2:
        assert beta[a] == pytest.approx(b2[a], abs=1e-12)
        assert se[a] == pytest.approx(se2[a], rel=1e-9)
    assert beta["ref"] == 0.0 and se["ref"] == 0.0


def test_matched_pair_se_is_sqrt2_sigma_over_sqrt_k():
    """Two arms on K units, one cell each: SE = sd(diff)/sqrt(K) = sqrt(2) s / sqrt(K), not s / sqrt(K)."""
    rng = np.random.default_rng(1)
    K = 25
    base = rng.normal(0.2, 0.05, K)
    d = rng.normal(0.01, 0.02, K)
    cell = pd.DataFrame({"pair": [f"u{i}" for i in range(K)] * 2, "arm": ["ref"] * K + ["x"] * K,
                         "score": np.concatenate([base, base + d])})
    beta, se, dof = feedback.fe_exact(cell, "ref")
    assert beta["x"] == pytest.approx(d.mean(), abs=1e-12)
    assert se["x"] == pytest.approx(d.std(ddof=1) / np.sqrt(K), rel=1e-9)
    assert dof == K - 1


def test_fixed_effects_effects_unchanged_and_approx_se_is_fe_exact():
    cell = synthetic_cells(seed=3)
    df = pd.DataFrame({"submission": cell["arm"] + "_sub", "section": "alc",
                       "subject_id": cell["pair"], "benchmark_id": "benchmark_1", "score": cell["score"]})
    arm_of = {f"{a}_sub": a for a in ("ref", "a", "b", "c")}
    out = feedback.fixed_effects(df, arm_of, "alc", pool_aware=False).set_index("arm")
    beta, se, _ = feedback.fe_exact(cell, "ref")
    for a in ("a", "b", "c"):
        assert out.loc[a, "effect_vs_ref"] == pytest.approx(beta[a], abs=1e-4)
        assert out.loc[a, "approx_se"] == pytest.approx(round(se[a], 4), abs=1e-12)


def test_public_selection_parsing(monkeypatch):
    monkeypatch.delenv("PAIEC_FEEDBACK_PUBLIC", raising=False)
    assert feedback._public_default(None) is None
    assert feedback._public_default(False) is None and feedback._public_default(True) == "all"
    assert feedback._public_default("pub_x, pub_y") == {"pub_x", "pub_y"}
    monkeypatch.setenv("PAIEC_FEEDBACK_PUBLIC", "1")
    assert feedback._public_default(None) == "all"
    monkeypatch.setenv("PAIEC_FEEDBACK_PUBLIC", "0")
    assert feedback._public_default(None) is None


@pytest.mark.skipif(not (feedback.PUBLIC / "arms.csv").exists(), reason="public logs not fetched")
def test_public_logs_parse_into_the_long_format(monkeypatch):
    monkeypatch.delenv("PAIEC_FEEDBACK_PUBLIC", raising=False)
    pub = feedback.load_public()
    arms = feedback.public_arms()
    assert set(pub.columns) == {"submission", "section", "subject_id", "benchmark_id", "n_unique", "score", "ece"}
    assert pub["submission"].nunique() == len(arms) == 55
    assert set(pub["section"]) == {"alc", "b0", "b1", "b3", "b7", "b15", "b31"}
    counts = pd.Series(arms).value_counts().to_dict()
    assert counts == {"pub_aghasalim_v3C": 15, "pub_aghasalim_v3A": 14, "pub_aghasalim_v3B": 14, "pub_aghasalim_v2": 5,
                      "pub_aghasalim_v1": 4, "pub_doctawho42_hiership": 2, "pub_doctawho42_legacyP": 1}
    only = feedback.load_public({"pub_doctawho42_hiership"})
    assert only["submission"].nunique() == 2
    with pytest.raises(ValueError):
        feedback.public_arms({"pub_nobody"})
    ours = feedback.load()
    assert not ours["submission"].str.startswith("pub_").any()  # default unchanged without the flag or env
    both = feedback.load(include_public=True)
    assert len(both) == len(ours) + len(pub)
    assert set(feedback.ledger_arms(include_public=True).values()) >= set(arms.values())
