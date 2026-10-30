"""Build a pickle cache of everything the pooling replay needs (public data only), so quick checks skip the slow
parquet load: raw eligible pairs per benchmark (protocol.index_pairs) and the LOBO subject priors per held-out benchmark
(exactly what experiments/paired.build_models fits from df[df.bench != bench])."""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")


def main() -> None:
    from paiec.data import load_responses
    from paiec.hier import fit_abilities, fit_subject_prior
    from paiec.protocol import index_pairs

    t0 = time.time()
    df = load_responses()
    print(f"loaded {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
    out = {"raw": {}, "subject_prior": {}, "abilities": {}}
    for b in BENCHES:
        out["raw"][b] = index_pairs(df[df.bench == b])
        train = df[df.bench != b]
        out["subject_prior"][b] = fit_subject_prior(train)
        out["abilities"][b] = fit_abilities(train)
        print(b, len(out["raw"][b]), f"{time.time() - t0:.0f}s", flush=True)
    with open(HERE / "cache" / "replay_cache.pkl", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
