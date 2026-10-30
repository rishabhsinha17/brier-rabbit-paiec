"""Q14: items whose content is a system prompt or log header: are their labels plausible? Compare each subject's success
on those items with (a) the same subject's success on the same (competition, problem_idx) under a normal prompt and
(b) all subjects' success on that problem."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["comp"] = F.map(lambda d: d.get("competition", "")); it["idx"] = F.map(lambda d: int(d.get("problem_idx", -1)))
it["sys"] = it.content.str.startswith(("You are ChatGPT", "Date: 20"))
r = pd.read_parquet(ROOT / "matharena" / "response.parquet").merge(it[["item_id", "comp", "idx", "sys"]], on="item_id")
s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")[["subject_id", "display_name"]]
r = r.merge(s, on="subject_id")
prob_all = r[~r.sys].groupby(["comp", "idx"]).response.mean()
for name in ["GPT OSS 20B (high)", "GPT OSS 120B (high)", "GPT-5 (high)"]:
    x = r[(r.display_name == name)]
    sysr = x[x.sys]
    cell = sysr.groupby(["comp", "idx"]).response.mean()
    own_norm = x[~x.sys].groupby(["comp", "idx"]).response.mean()
    both = cell.index.intersection(own_norm.index)
    others = prob_all.reindex(cell.index)
    print(f"{name}: sys/header items {sysr.item_id.nunique()}, responses {len(sysr)}, success {sysr.response.mean():.3f} | "
          f"all subjects on the same problems {others.mean():.3f} (corr per problem {np.corrcoef(cell.values, others.values)[0,1]:.2f}) | "
          f"same subject, same problems, normal prompt: {len(both)} problems, success {own_norm.reindex(both).mean() if len(both) else float('nan'):.3f} "
          f"vs header {cell.reindex(both).mean() if len(both) else float('nan'):.3f} | normal-prompt items of this subject: "
          f"{x[~x.sys].item_id.nunique()} (success {x[~x.sys].response.mean():.3f})")
    print("   competitions of sys/header items:", sysr.groupby("comp").item_id.nunique().to_dict())
