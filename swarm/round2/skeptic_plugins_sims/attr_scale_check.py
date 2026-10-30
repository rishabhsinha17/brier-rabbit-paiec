"""Is attr_scale (uniform x1.56 of the ridge prediction, intercept included) the same as the unshrunk-target refit?
Uses the shipped all-public subject prior (names -> (theta, var)) and public subjects.parquet only (no item data)."""
import json, sys
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from paiec.data import DATA_ROOT, BINARY_BENCHMARKS, SUBJECT_FIELDS, _text
from paiec.subjects import attributes, name_key, ridge_fit
pr = json.load(open("submission_src/v3b_noM_n_m0n20/subject_prior.json"))
coef = np.array(pr["attr"]["coef"]); names = pr["names"]
labels = ["icpt", "year-2025", "no_year", "effort/4", "think", "small", "large"] + ["prov_" + p for p in
          ("OpenAI", "Anthropic", "Google", "DeepSeek", "Alibaba", "Meta", "xAI", "ByteDance", "Mistral", "Moonshot", "Zhipu")]
print("shipped attr coef:", ", ".join(f"{l}={c:+.3f}" for l, c in zip(labels, coef)), f"| resid var {pr['attr']['var']:.3f}")
reps = {}
for b in BINARY_BENCHMARKS:
    s = pd.read_parquet(DATA_ROOT / b / "subjects.parquet")
    for _, r in s.iterrows():
        d = {f: _text(r.get(f)).strip() for f in SUBJECT_FIELDS}
        k = name_key(d)
        if k in names and k not in reps:
            reps[k] = d
X, y_sh, y_un, w_sh, w_un = [], [], [], [], []
for key, (th, var) in names.items():
    if key not in reps:
        continue
    k = round(1 / var - 1)
    X.append(attributes(reps[key])); y_sh.append(th); w_sh.append(1 / var)
    y_un.append(th * (k + 1) / k); w_un.append(k / (k + 1))
Xa = np.array(X)
c_sh, v_sh = ridge_fit(X, y_sh, w_sh, alpha=1.0)
c_un, v_un = ridge_fit(X, y_un, w_un, alpha=1.0)
old, new = Xa @ np.array(c_sh), Xa @ np.array(c_un)
slope, icpt = np.polyfit(old, new, 1)
print(f"{len(X)} names with attributes; refit check: shipped coef reproduced max|diff| {np.max(np.abs(np.array(c_sh) - coef)):.2e}")
print(f"unshrunk refit vs shrunk: new = {icpt:+.3f} + {slope:.3f} * old   (through-origin ratio {np.sum(old*new)/np.sum(old*old):.3f})")
print(f"mean pred: shrunk {old.mean():+.3f}, x1.56 {1.56*old.mean():+.3f}, unshrunk refit {new.mean():+.3f}; resid var shrunk {v_sh:.3f} unshrunk {v_un:.3f}")
print(f"intercept: shrunk {c_sh[0]:+.3f} x1.56 {1.56*c_sh[0]:+.3f} unshrunk {c_un[0]:+.3f}; year coef shrunk {c_sh[1]:+.3f} x1.56 {1.56*c_sh[1]:+.3f} unshrunk {c_un[1]:+.3f}")
rms = np.sqrt(np.mean((1.56 * old - new) ** 2))
print(f"RMS(1.56*old - unshrunk refit) over names = {rms:.3f} logits; corr(old,new) = {np.corrcoef(old, new)[0,1]:.3f}")
# what a recent unknown subject gets (year clamp at 2026.4 vs not), e.g. an OpenAI 2026-09 model
for date in ("2025-06-01", "2026-04-01", "2026-09-01"):
    s = {"normalized_name": "Unseen Model X", "provider": "OpenAI", "release_date": date, "reasoning_effort": "high",
         "subject_features_extra": ""}
    x = attributes(s); xc = list(x); xc[1] = min(xc[1], 2026.4 - 2025.0)
    print(f"  unknown OpenAI high-effort {date}: theta x1 {np.dot(coef, x):+.3f} | x1.56 {1.56*np.dot(coef, x):+.3f} | x1.56 clamp {1.56*np.dot(coef, xc):+.3f} | unshrunk refit {np.dot(c_un, x):+.3f}")
