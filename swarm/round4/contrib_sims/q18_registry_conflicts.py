"""Q18: alias conflicts in map_model_registry.json (main cf680c65) for labels used by the six public benchmarks:
registry keys that normalize to the same string (lowercase, drop org prefix and punctuation) but map to different
canonical models."""
import json, re
from pathlib import Path
import pandas as pd
REG = json.load(open((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../research/a3_data/raw/gh_measurement-db/scripts/build_measurement_tables/map_model_registry.json")))
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
def norm(k):
    k = k.lower().split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", k)
groups = {}
for k, v in REG.items():
    groups.setdefault(norm(k), set()).add(v["model"])
print("registry keys", len(REG), "| normalized keys mapping to >1 canonical model:", sum(len(v) > 1 for v in groups.values()))
used = []
for b in ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]:
    s = pd.read_parquet(ROOT / b / "subjects.parquet")
    for lab, nn in zip(s.display_name, s.normalized_name):
        used.append((b, lab, nn))
conf = []
for b, lab, nn in used:
    g = groups.get(norm(lab), set())
    if len(g) > 1 or (nn is not None and g and nn not in g):
        conf.append((b, lab, nn, sorted(g)))
print("labels used by the public benchmarks whose normalized form is ambiguous in the registry:", len(conf))
for c in conf:
    print("  ", c)
print("\nexplicit check: 'x-ai/grok-4-fast' ->", REG.get("x-ai/grok-4-fast", {}).get("model"), "| 'grok-4-fast' ->", REG.get("grok-4-fast", {}).get("model"))
