"""Q17: which MathArena labels are registry keys in map_model_registry.json on GitHub main cf680c65 (local copy fetched
2026-10-04; main has no newer commits as of 2026-10-06), and how the empty-name labels differ from existing keys."""
import json, re
from pathlib import Path
import pandas as pd
REG = json.load(open((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../research/a3_data/raw/gh_measurement-db/scripts/build_measurement_tables/map_model_registry.json")))
s = pd.read_parquet((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db/matharena/subjects.parquet"))
low = {}
for k in REG:
    low.setdefault(re.sub(r"[^a-z0-9]", "", k.lower()), []).append(k)
s["in_registry"] = s.display_name.isin(REG)
s["named"] = s.normalized_name.notna()
print(pd.crosstab(s.named, s.in_registry))
miss = s[~s.named]
near = {}
for lab in miss.display_name:
    key = re.sub(r"[^a-z0-9]", "", lab.lower())
    base = re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", lab).lower())
    hits = low.get(key, []) or low.get(base, [])
    near[lab] = [(h, REG[h]["model"]) for h in hits][:2]
print("empty-name labels with a registry key that matches after lowercasing/removing punctuation or the '(...)' qualifier:",
      sum(bool(v) for v in near.values()), "of", len(near))
for lab, v in near.items():
    print(f"  {lab!r:46s} -> {v}")
