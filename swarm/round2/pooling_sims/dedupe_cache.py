"""Share one item dict per (benchmark, item_id) across pairs in cache/replay_cache.pkl (shrinks it ~5-10x)."""
import pickle, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
t0 = time.time()
path = HERE / "cache" / "replay_cache.pkl"
with open(path, "rb") as f:
    cache = pickle.load(f)
print(f"loaded in {time.time() - t0:.1f}s")
for b, raw in cache["raw"].items():
    shared, texts = {}, {}
    for rp in raw:
        for iid, rec in rp["items"].items():
            it = rec["item"]
            key = (iid, it["item_content"], it["item_features"], it["interactors"], it["benchmark_id"])
            if key not in shared:
                txt = texts.setdefault(it["item_content"], it["item_content"])
                shared[key] = {"item_content": txt, "item_features": it["item_features"],
                               "interactors": it["interactors"], "benchmark_id": it["benchmark_id"]}
            rec["item"] = shared[key]
    print(b, "distinct item dicts", len(shared), "distinct texts", len(texts))
with open(path, "wb") as f:
    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved in {time.time() - t0:.1f}s, size {path.stat().st_size / 1e6:.0f} MB")
