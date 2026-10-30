"""Load the public measurement-db release into platform-shaped records.

Every response row becomes one target: the subject dict, the item dict and a 0/1 label,
built the same way the starting kit's run_smoke_test.py builds runtime inputs.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "measurement-db"

# Benchmarks with item-level 0/1 responses. MathArena is "mixed": we keep only its 0/1 rows
# (final-answer problems); MMDocRAG is fractional and stays out of evaluation.
BINARY_BENCHMARKS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
ALL_BENCHMARKS = BINARY_BENCHMARKS + ("mmdocrag",)

SUBJECT_FIELDS = (
    "normalized_name", "provider", "release_date", "access_date",
    "harness", "reasoning_effort", "harness_version", "subject_features_extra",
)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def benchmark_alias(name: str) -> str:
    """Stable six-digit anonymous alias, shaped like the platform's benchmark IDs."""
    digest = int(hashlib.sha256(f"brier-rabbit:{name}".encode()).hexdigest(), 16)
    return f"benchmark_{100000 + digest % 900000}"


def load_responses(benchmarks=BINARY_BENCHMARKS, root: Path = DATA_ROOT, binary_only: bool = True) -> pd.DataFrame:
    """One row per recorded response with runtime-shaped subject and item fields."""
    frames = []
    for bench in benchmarks:
        subjects = pd.read_parquet(root / bench / "subjects.parquet")
        items = pd.read_parquet(root / bench / "items.parquet", columns=["item_id", "content", "item_features"])
        resp = pd.read_parquet(
            root / bench / "response.parquet",
            columns=["subject_id", "item_id", "trial", "test_condition", "interactors", "response"],
        )
        if binary_only:
            resp = resp[resp["response"].isin([0.0, 1.0])]
        df = resp.merge(items, on="item_id", how="left").merge(subjects, on="subject_id", how="left")
        df["bench"] = bench
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Same normalisation as the kit's _build_input: subject fields, item_features and
    # interactors are stripped; item_content is passed through unchanged.
    for field in SUBJECT_FIELDS:
        df[field] = df[field].map(lambda v: _text(v).strip())
    df["item_content"] = df["content"].map(_text)
    df["item_features"] = df["item_features"].map(lambda v: _text(v).strip())
    df["interactors"] = df["interactors"].map(lambda v: _text(v).strip())
    df["benchmark_id"] = df["bench"].map(benchmark_alias)
    df["label"] = df["response"].astype("int8")
    df["pair"] = df["bench"] + "|" + df["subject_id"].astype(str)
    keep = ["bench", "benchmark_id", "pair", "subject_id", "display_name", "item_id", "trial",
            "test_condition", "label", "item_content", "item_features", "interactors", *SUBJECT_FIELDS]
    return df[keep]


def subject_dict(row) -> dict:
    return {field: row[field] for field in SUBJECT_FIELDS}


def item_dict(row) -> dict:
    return {
        "item_content": row["item_content"],
        "item_features": row["item_features"],
        "interactors": row["interactors"],
        "benchmark_id": row["benchmark_id"],
    }
