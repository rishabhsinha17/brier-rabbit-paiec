"""Q15 (our pipeline): rubric-criterion rows kept by the value filter response in {0,1}; per-item response_scale in
grading_criterion; effect on eligible MathArena pairs."""
import json, sys
from collections import Counter
from pathlib import Path
import pandas as pd
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
sc = it.grading_criterion.map(lambda g: json.dumps(json.loads(g).get("response_scale"), sort_keys=True))
print("response_scale values in grading_criterion:", Counter(sc).most_common(5))
it["discrete01"] = sc == json.dumps({"kind": "discrete", "values": [0.0, 1.0]}, sort_keys=True)
r = pd.read_parquet(ROOT / "matharena" / "response.parquet").merge(it[["item_id", "discrete01"]], on="item_id")
b = r[r.response.isin([0.0, 1.0])]
print("rows with response in {0,1}:", len(b), "| of these on items whose scale is NOT discrete {0,1}:", int((~b.discrete01).sum()))
def eligible(df):
    c = df.groupby("subject_id").item_id.nunique()
    return int((c >= 80).sum()), int(df.item_id.nunique())
print("eligible MathArena pairs / item IDs: value filter", eligible(b), "| scale filter", eligible(b[b.discrete01]))
