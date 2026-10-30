"""Section 1e: which attributes carry residual B0 mis-calibration? Regress the LOBO B0 offset (lg - theta, centred per
benchmark) on the attribute vector with benchmark fixed effects; under a calibrated prior every coefficient is ~0.
Positive = subjects with that attribute are under-predicted. HC1 robust SEs. Base prior and SubjCal (a156+clamp)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calib_lobo as cl  # noqa: E402
import shiftlib as sl  # noqa: E402


def ols_hc1(X, y):
    XtX = X.T @ X
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    inv = np.linalg.pinv(XtX)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    n, k = X.shape
    cov = inv @ meat @ inv * n / max(n - k, 1)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def main():
    pd.set_option("display.width", 200)
    feats = ["year", "noyear", "effort", "think", "small", "large", "prov_OpenAI", "prov_Anthropic", "prov_Google",
             "prov_DeepSeek", "prov_Alibaba", "prov_Meta", "prov_xAI"]
    out = {}
    for scale, ymax, lab in ((1.0, 0.0, "base"), (1.56, 2026.4, "a156")):
        g = cl.build(scale, ymax)
        g["is_attr"] = (g.src == "attr").astype(float)
        B = pd.get_dummies(g.bench).astype(float).values
        for subset, h in (("all", g), ("attr-source", g[g.src == "attr"]), ("name-source", g[g.src == "name"])):
            Bh = pd.get_dummies(h.bench).astype(float).values
            use = [f for f in feats if h[f].abs().sum() > 0 and h[f].std() > 0]
            X = np.c_[h[use].values, Bh]
            beta, se = ols_hc1(X, h.off.values)
            out[(lab, subset)] = pd.Series({f: f"{b:+.2f} ({s:.2f})" for f, b, s in zip(use, beta, se)})
    T = pd.DataFrame(out)
    print("LOBO B0 offset regressed on attributes + benchmark FE: coef (HC1 SE), logits; + = under-predicted")
    print(T.fillna("").to_string())


if __name__ == "__main__":
    main()
