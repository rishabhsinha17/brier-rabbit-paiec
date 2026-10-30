"""Q10: REAL placeholder-like subjects ('Test', 'Update Eval', ...): coverage, success, duplicate response vectors;
MathArena cells whose trials do not start at 1; Multi-SWE 'MSWE-Agent' spelling."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
s = pd.read_parquet(ROOT / "real_webagents" / "subjects.parquet")
r = pd.read_parquet(ROOT / "real_webagents" / "response.parquet")
st = r.groupby("subject_id").agg(n=("response", "size"), items=("item_id", "nunique"), mean=("response", "mean"), maxtrial=("trial", "max"))
s = s.merge(st, left_on="subject_id", right_index=True)
print(s[["display_name", "normalized_name", "provider", "release_date", "n", "items", "mean", "maxtrial"]].sort_values("display_name").to_string())
# identical per-item outcome vectors between subjects (trial-1 only), on the common items
p = r[r.trial == 1].pivot_table(index="subject_id", columns="item_id", values="response")
names = s.set_index("subject_id").display_name
ids = list(p.index); dup = []
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        x, y = p.loc[ids[a]], p.loc[ids[b]]
        m = x.notna() & y.notna()
        if m.sum() >= 50 and (x[m] == y[m]).mean() > 0.97:
            dup.append((names[ids[a]], names[ids[b]], int(m.sum()), round(float((x[m] == y[m]).mean()), 3)))
print("\nREAL subject pairs with >97% identical trial-1 outcomes on >=50 common items:", dup)
# MathArena cells not starting at trial 1
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
mr = pd.read_parquet(ROOT / "matharena" / "response.parquet")
ms = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")
c = mr.groupby(["subject_id", "item_id"]).trial.agg(["min", "max", "count"]).reset_index()
c = c[c["min"] != 1].merge(it[["item_id", "content"]], on="item_id").merge(ms[["subject_id", "display_name"]], on="subject_id")
c["text"] = c.content.str[:40]
print("\nMathArena cells whose trials do not start at 1:", len(c))
print(c.groupby(["display_name", "text"]).size().to_string())
m = pd.read_parquet(ROOT / "multi_swebench" / "subjects.parquet")
print("\nMulti-SWE subjects with harness 'MSWE-Agent':", m[m.harness == "MSWE-Agent"][["display_name", "normalized_name", "access_date"]].to_dict("records"),
      "| with 'MSWE-agent':", int((m.harness == "MSWE-agent").sum()))
