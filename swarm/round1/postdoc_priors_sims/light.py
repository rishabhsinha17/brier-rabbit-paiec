"""Light public-data loader (no item text): one row per response with subject fields; same normalisation as paiec.data."""
import sys, math
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import pandas as pd, numpy as np
from paiec.data import DATA_ROOT, BINARY_BENCHMARKS, SUBJECT_FIELDS, _text

def load_light(benches=BINARY_BENCHMARKS):
    frames = []
    for b in benches:
        s = pd.read_parquet(DATA_ROOT / b / "subjects.parquet")
        r = pd.read_parquet(DATA_ROOT / b / "response.parquet", columns=["subject_id", "item_id", "trial", "response"])
        r = r[r["response"].isin([0.0, 1.0])]
        it = pd.read_parquet(DATA_ROOT / b / "items.parquet", columns=["item_id", "item_features"])
        df = r.merge(s, on="subject_id", how="left").merge(it, on="item_id", how="left")
        df["bench"] = b
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for f in SUBJECT_FIELDS:
        df[f] = df[f].map(lambda v: _text(v).strip())
    df["item_features"] = df["item_features"].map(lambda v: _text(v).strip())
    df["label"] = df["response"].astype("int8")
    df["pair"] = df["bench"] + "|" + df["subject_id"].astype(str)
    return df
