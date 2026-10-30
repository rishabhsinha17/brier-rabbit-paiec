"""Q19: cross-benchmark duplicate item texts (content_hash and exact content) among the six public benchmarks; also how
content_hash relates to the raw content (normalization)."""
import hashlib, itertools, unicodedata
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
B = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
it = {b: pd.read_parquet(ROOT / b / "items.parquet", columns=["item_id", "content", "content_hash"]) for b in B}
for a, b in itertools.combinations(B, 2):
    h = set(it[a].content_hash) & set(it[b].content_hash)
    c = set(it[a].content) & set(it[b].content)
    if h or c:
        print(f"{a} x {b}: shared content_hash {len(h)}, shared exact content {len(c)}")
print("pairs not listed share nothing")
# what normalization does content_hash use?
for b in B:
    d = it[b]
    raw = d.content.map(lambda c: hashlib.sha256(c.encode()).hexdigest()[:16])
    strip = d.content.map(lambda c: hashlib.sha256(c.strip().encode()).hexdigest()[:16])
    nfc = d.content.map(lambda c: hashlib.sha256(unicodedata.normalize("NFC", c).strip().encode()).hexdigest()[:16])
    print(f"{b:18s} content_hash == sha256(content)[:16]: {(raw == d.content_hash).mean():.3f} | of strip(): {(strip == d.content_hash).mean():.3f} | NFC+strip: {(nfc == d.content_hash).mean():.3f}")
