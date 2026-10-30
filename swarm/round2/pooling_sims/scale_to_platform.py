"""Scale matched-mode replay gains to the platform's partner coverage.

For each benchmark and arm: per-budget Brier gain vs base (mean over pairs) divided by the replay's partner-label
coverage of eval rows at that budget = gain per covered row. Platform coverage for a shared pair with complete item
overlap (eval pool = N/2, independent per-pair splits, partner's B labels uniform over its acquisition pool) is
c_B = B / N; N = 112 (median eval pool 56). Expected platform ALC change per shared pair = sum_B w_B * gain_B * c_B."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = {1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
N = float(sys.argv[1]) if len(sys.argv) > 1 else 112.0
files = sys.argv[2:] or sorted(glob.glob(str(HERE / "out" / "q*matched*.csv")))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df["mode"] == "matched"].copy()
df["model"] = df["model"].str.replace(r"_x(\d)$", r"_x\1.0", regex=True)  # cav_x2 == cav_x2.0
df = df[~df["model"].str.startswith("si")]  # sigma_i arms change every item, not only covered rows
df = df.drop_duplicates(["bench", "run", "pair", "model"])
rows = []
for (bench, model), g in df.groupby(["bench", "model"]):
    if model == "base":
        continue
    ref = df[(df.bench == bench) & (df.model == "base")].set_index(["run", "pair"])
    g = g.set_index(["run", "pair"])
    common = g.index.intersection(ref.index)
    if not len(common):
        continue
    line = {"bench": bench, "model": model, "pairs": len(common),
            "replay_dALC": float((g.loc[common, "alc"] - ref.loc[common, "alc"]).mean())}
    plat = 0.0
    for b in W:
        gain = float((g.loc[common, f"b{b}"] - ref.loc[common, f"b{b}"]).mean())
        cov = float(ref.loc[common, f"cov{b}"].mean())
        per = gain / cov if cov > 0 else 0.0
        line[f"perRow_b{b}"] = per
        plat += W[b] * per * (b / N)
    line["platform_dALC_shared"] = plat
    rows.append(line)
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"N = {N:.0f} (platform coverage c_B = B/N: " + ", ".join(f"B{b} {b / N:.3f}" for b in W) + ")")
print(out.round(5).to_string(index=False))
for model, g in out.groupby("model"):
    print(f"{model}: bench-equal platform dALC per shared pair {g.platform_dALC_shared.mean():+.5f} over {len(g)} benches")

print("\nExpected platform Brier change per shared pair by budget (bench-equal mean of per-row gain x B/N):")
for model, g in out.groupby("model"):
    if len(g) < 4:
        continue
    parts = {b: float(g[f"perRow_b{b}"].mean()) * b / N for b in W}
    print(f"{model}: " + "  ".join(f"B{b} {v:+.5f}" for b, v in parts.items())
          + f"  -> ALC {sum(W[b] * v for b, v in parts.items()):+.5f}; range over benches "
          f"[{g.platform_dALC_shared.min():+.5f}, {g.platform_dALC_shared.max():+.5f}]")
