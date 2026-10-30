"""Public-data facts quoted in the paper's Data section (measurement-db bc8204d8, binary responses only)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from paiec.data import load_responses, SUBJECT_FIELDS  # noqa: E402
from paiec.protocol import index_pairs  # noqa: E402
from paiec.subjects import name_key  # noqa: E402

df = load_responses()
print("rows", len(df), "benchmarks", df.bench.nunique())
for b, g in df.groupby("bench"):
    raw = index_pairs(g)
    sizes = sorted(len(r["items"]) for r in raw)
    med = sizes[len(sizes) // 2] if sizes else 0
    print(f"{b}: responses {len(g)}, items {g.item_id.nunique()}, distinct texts {g.item_content.nunique()}, "
          f"subjects {g.subject_id.nunique()}, pairs(all) {g.pair.nunique()}, eligible pairs {len(raw)}, "
          f"median items/eligible pair {med}, mean label {g.label.mean():.3f}, "
          f"empty normalized_name subjects {g[g.normalized_name == ''].subject_id.nunique()}")
pairs = df.drop_duplicates("pair")
keys = [name_key({f: r[f] for f in SUBJECT_FIELDS}) for _, r in pairs.iterrows()]
pairs = pairs.assign(key=keys)
raw_names = pairs[pairs.normalized_name != ""].normalized_name.nunique()
k = pairs[pairs.key != ""]
per = k.groupby("key").bench.nunique()
print(f"canonical names {per.size}; on 1 benchmark {(per == 1).sum()}; on >=2 {(per >= 2).sum()}; "
      f"raw normalized_name values {raw_names}; pairs with empty normalized_name but recovered key "
      f"{((pairs.normalized_name == '') & (pairs.key != '')).sum()}")
elig = sum(len(index_pairs(g)) for b, g in df.groupby("bench") if b != "swe_rebench")
print("eligible pairs excluding swe_rebench", elig)
