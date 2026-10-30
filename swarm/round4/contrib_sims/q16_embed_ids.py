"""Q16: measurement-db-embed (public, ungated; copies fetched 2026-10-04) vs measurement-db bc8204d8 item IDs: does the
official baseline's check (embed item_id -> sha1(current content) == content_sha1) pass?"""
import hashlib
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
EMB = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../research/a3_data/raw"))
for b in ["real_webagents", "researchcodebench", "swe_rebench"]:
    e = pd.read_parquet(EMB / f"embed_{b}.parquet", columns=["item_id", "content_sha1"])
    it = pd.read_parquet(ROOT / b / "items.parquet", columns=["item_id", "content"])
    it["sha1"] = it.content.map(lambda c: hashlib.sha1(str(c).encode()).hexdigest())
    cur = dict(zip(it.item_id, it.sha1))
    idmatch = e.item_id.isin(it.item_id).sum()
    ok = sum(cur.get(i) == s for i, s in zip(e.item_id, e.content_sha1))
    bysha = e.content_sha1.isin(it.sha1).sum()
    print(f"{b:18s} embed rows {len(e):5d} | current items {len(it):5d} | embed item_id found in current items {idmatch:5d} | "
          f"baseline check passes for {ok:5d} rows | embed content_sha1 found among current sha1(content) {bysha:5d} | "
          f"current items with no embedding (by sha1) {int((~it.sha1.isin(e.content_sha1)).sum())}")
