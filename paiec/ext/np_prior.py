"""Provider-free subject prior ('NP', swarm round 3 subject-shift study).

Same as paiec.hier.fit_subject_prior, but the attribute ridge is fit with the provider one-hot columns zeroed, so
their coefficients are exactly 0 and the remaining coefficients are the reduced-feature ridge fit. Rationale
(swarm/round3/subject_shift.md): provider effects flip sign across public benchmarks, and on a temporal hold-out
(prior fit on older subjects, scored on newer ones) A156+NP beats A156 by about 2e-3 ALC at both cut-offs
(results/shift_temporal_replay.csv). Fit-time only: the shipped model reads the coefficients from subject_prior.json,
so no predict-time code changes.
"""
from __future__ import annotations

import numpy as np

from paiec.hier import fit_subject_prior
from paiec.subjects import PROVIDERS, attributes, name_key, ridge_fit

N_BASE = len(attributes({})) - len(PROVIDERS)  # const, year, noyear, effort, think, small, large


def fit_subject_prior_np(train_df, shrink: float = 1.0, alpha: float = 1.0) -> dict:
    from paiec.data import SUBJECT_FIELDS

    prior = fit_subject_prior(train_df, shrink=shrink, alpha=alpha)  # names part is unchanged
    pairs = train_df.drop_duplicates("pair")[["bench", "pair", *SUBJECT_FIELDS]].copy()
    pairs["key"] = [name_key({f: r[f] for f in SUBJECT_FIELDS}) for _, r in pairs.iterrows()]
    pairs = pairs[pairs["key"] != ""]
    reps = pairs.drop_duplicates("key").set_index("key")
    X, y, w = [], [], []
    for key, (theta, var) in prior["names"].items():
        x = attributes({f: reps.at[key, f] for f in SUBJECT_FIELDS})
        X.append([*x[:N_BASE], *([0.0] * (len(x) - N_BASE))])
        y.append(theta)
        w.append(1.0 / var)
    coef, rvar = ridge_fit(X, y, w, alpha=alpha)
    prior["attr"] = {"coef": [float(c) for c in np.asarray(coef, dtype=float)], "var": float(rvar)}
    return prior
