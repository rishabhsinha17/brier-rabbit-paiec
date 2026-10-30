"""Round 8 red team: reconstruct, for every pair-draw of results/cal_preds_bench.parquet, the label evidence seen.

The saved predictions hold evaluation rows only. Acquisition under the default (sha256) rule does not depend on the
model, so the revealed label sequence of every pair at every budget is recovered by replaying run_protocol with a
dummy predict (no model fit). Only label sequences and item-level structure are kept (no subject fields).
Public data only.

Output: swarm/round8/redteam_sims/pairinfo_bench.parquet (one row per bench, k, run, pair) and
        swarm/round8/redteam_sims/evidence_bench.parquet (one row per revealed label: bench, k, run, pair, pos, item, y).
Usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round8/redteam_sims/reconstruct.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paiec.data import BINARY_BENCHMARKS, load_responses  # noqa: E402
from paiec.evaluate import FoldData  # noqa: E402
from paiec.protocol import run_protocol  # noqa: E402

OUT = Path(__file__).resolve().parent
BENCHES = ("real_webagents", "researchcodebench", "multi_swebench", "matharena")


def main(runs: int = 18, ks=(1, 2), subsample: str = "bench") -> None:
    t0 = time.time()
    df = load_responses()
    folds = FoldData(df, BINARY_BENCHMARKS, subsample=subsample)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    info, ev = [], []
    for bench in BENCHES:
        for k in ks:
            for r, pairs in folds.draws(bench, k, runs, 0) or []:
                snaps = {}

                def predict(inp, labeled=None, snaps=snaps):
                    snaps.setdefault(len(labeled or []), list(labeled or []))
                    return 0.5

                run_protocol(pairs, predict, seed=r)
                final = snaps[max(snaps)]
                # run_protocol appends, for each cap, every pair's new labels in pair order (each pair reaches the cap:
                # n_acq >= 40 here), so the evidence list splits by position; item content is checked against p.acq.
                seqs = {p.key: [] for p in pairs}
                pos, prev = 0, 0
                for cap in (1, 3, 7, 15, 31):
                    for p in pairs:
                        take = min(cap, len(p.acq)) - min(prev, len(p.acq))
                        acq_text = {p.items[i]["item"]["item_content"]: i for i in p.acq}
                        for (_s, item), y in final[pos:pos + take]:
                            i = acq_text.get(item.get("item_content", ""))
                            if i is None:
                                raise RuntimeError(f"evidence order mismatch {bench} k={k} r={r} {p.key}")
                            seqs[p.key].append((i, int(y)))
                        pos += take
                    prev = cap
                if pos != len(final):
                    raise RuntimeError(f"evidence length mismatch {bench} k={k} r={r}")
                for p in pairs:
                    feats = [p.items[i]["item"].get("item_features", "") for i in p.eval]
                    ys = [y for i in p.eval for y in p.items[i]["labels"]]
                    acq_ys = [y for _, y in seqs[p.key]]
                    info.append({
                        "bench": bench, "k": k, "run": r, "pair": p.key,
                        "n_eval_items": len(p.eval), "n_eval_rows": len(ys), "eval_rate": float(np.mean(ys)),
                        "n_acq": len(p.acq), "acq_labels": "".join(str(y) for y in acq_ys),
                        "feat_share": float(np.mean([bool(f) for f in feats])),
                        "n_groups": len({str(f).split(";")[0].strip() for f in feats if f}),
                        "multi_run_share": float(np.mean([len(p.items[i]["labels"]) > 1 for i in p.eval])),
                    })
                    for pos, (item_id, y) in enumerate(seqs[p.key]):
                        ev.append({"bench": bench, "k": k, "run": r, "pair": p.key, "pos": pos,
                                   "item": str(item_id), "y": y})
        print(f"{bench} done {time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(info).to_parquet(OUT / f"pairinfo_{subsample}.parquet", index=False)
    pd.DataFrame(ev).to_parquet(OUT / f"evidence_{subsample}.parquet", index=False)
    print(f"{len(info)} pair-draws, {len(ev)} labels, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
