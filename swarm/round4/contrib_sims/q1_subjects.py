"""Q1: subject-table completeness on measurement-db bc8204d8 (public data, read-only)."""
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
BENCH = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
for b in BENCH:
    s = pd.read_parquet(ROOT / b / "subjects.parquet")
    print(f"== {b}: {len(s)} subjects; columns {list(s.columns)}")
    for c in s.columns:
        nnull = s[c].isna().sum() + (s[c].astype(str).str.strip() == "").sum()
        print(f"   {c:24s} null/empty {nnull:4d}  distinct {s[c].nunique(dropna=True)}")
s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")
print(s.head(8).to_string())
emp = s[s.normalized_name.isna() | (s.normalized_name.astype(str).str.strip() == "")]
print("\nMathArena empty normalized_name:", len(emp))
print("  of these, provider null:", emp.provider.isna().sum(), " release_date null:", emp.release_date.isna().sum())
print("  non-empty normalized_name rows provider null:", s.loc[~s.index.isin(emp.index), "provider"].isna().sum(),
      " release_date null:", s.loc[~s.index.isin(emp.index), "release_date"].isna().sum())
print(emp[["display_name", "reasoning_effort", "subject_features_extra"]].to_string())
