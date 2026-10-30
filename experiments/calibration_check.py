"""Is the model systematically over/under-confident at some budget? Fit a per-budget logit map a*logit(p)+c on
three held-out benchmarks and test it on the fourth (cross-fitted), reporting the Brier change per budget."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiec.data import load_responses  # noqa: E402
from paiec.evaluate import FoldData  # noqa: E402
from paiec.hier import HierModel, fit_subject_prior  # noqa: E402
from paiec.protocol import BUDGETS, run_protocol  # noqa: E402

BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
PARAMS = {"m0": -1.0, "tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "nu_mu": 3,
          "damped_newton": True}


def main():
    df = load_responses()
    folds = FoldData(df, BENCHES)
    rec = {}
    for b in BENCHES:
        model = HierModel(fit_subject_prior(df[df.bench != b]), PARAMS)
        rows = []
        for k in (1, 2):
            for r, pairs in folds.draws(b, k, 20):
                run_protocol(pairs, model.predict, seed=r, record=rows)
        rec[b] = np.array([(bud, p, y) for _, bud, p, y in rows], dtype=float)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    print("budget  brier_raw  brier_cal  (fit on 3 benchmarks, tested on the 4th; pair-weighting ignored)")
    for bud in BUDGETS:
        raw, cal = [], []
        for test in BENCHES:
            tr = np.vstack([rec[b][rec[b][:, 0] == bud] for b in BENCHES if b != test])
            te = rec[test][rec[test][:, 0] == bud]

            def loss(v, d=tr):
                q = 1 / (1 + np.exp(-(v[0] * lg(d[:, 1]) + v[1])))
                return np.mean((q - d[:, 2]) ** 2)
            a, c = minimize(loss, [1.0, 0.0], method="Nelder-Mead").x
            q = 1 / (1 + np.exp(-(a * lg(te[:, 1]) + c)))
            raw.append(np.mean((te[:, 1] - te[:, 2]) ** 2))
            cal.append(np.mean((q - te[:, 2]) ** 2))
        print(f"B{bud:<5d} {np.mean(raw):.4f}     {np.mean(cal):.4f}")


if __name__ == "__main__":
    main()
