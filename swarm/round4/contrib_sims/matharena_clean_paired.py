"""Run experiments/paired.py on a cleaned MathArena view (swarm round 4, contributions drafter; NOT enqueued).

Why: MathArena items are keyed by text in HierModel (eps) and in the cavity pool (C). On the public tables, 37.6% of the
items in an eligible MathArena pair share their exact text with another item of the same pair (AIME copies released
under two competition labels, the one generic Kangaroo text, proof rubric criteria, system-prompt contents), so a
pair's own revealed labels reach its evaluation items, and partner evidence on AIME is counted twice. This wrapper
removes the data defects and re-runs the same comparison, so the PI can check whether local MathArena effects
(e.g. combo_v1 SCP -2.55e-3 on MathArena) survive.

usage (same arguments as experiments/paired.py, plus --clean):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round4/contrib_sims/matharena_clean_paired.py \
      --clean sys,misaligned,aimedup,proof --benches matharena --runs 18 --ks 1,2 --subsample bench \
      --ref base --out results/combo_clean_ma.csv --extra '<combo_v1 JSON>'
--clean takes a comma list of: sys (202 system-prompt/header items), misaligned (19 GLM 4.6 / Grok 3 Mini cells whose
prompt states another problem), aimedup (second copies of AIME attempts, i.e. the aime_2025/aime_2026 rows of
(subject, text) cells that also appear under aime_2025_I/II or aime_2026_I), proof (rubric-criterion items whose
response_scale is interval), kangaroo (all Kangaroo items, which share one generic text; off unless asked).
Only MathArena rows change; the other benchmarks are untouched. Workers get the cleaned frame through paired._init.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
DATA = ROOT.parent / "data" / "measurement-db" / "matharena"
SYS_PROMPT = "You are ChatGPT, a large language model trained by OpenAI."
HEADER = r"^Date: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\nID: [0-9a-f]+\n$"
KANGAROO = "You are given a Math Kangaroo problem"


def _feats(value) -> dict:
    out = {}
    for part in str(value or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def matharena_drop_rows(df, steps: set) -> "pd.Series":
    """Boolean mask over df rows (any benchmark) to DROP for the requested cleaning steps."""
    import pandas as pd

    it = pd.read_parquet(DATA / "items.parquet")
    it["competition"] = it.item_features.map(lambda v: _feats(v).get("competition"))
    it["problem_idx"] = it.item_features.map(lambda v: int(_feats(v).get("problem_idx", -1)))
    it["scale"] = it.grading_criterion.map(lambda g: json.loads(g)["response_scale"]["kind"])
    is_ma = df.bench == "matharena"
    drop = pd.Series(False, index=df.index)
    sys_ids = set(it[it.content.str.startswith(SYS_PROMPT) | it.content.str.match(HEADER)].item_id)
    if "sys" in steps:
        drop |= is_ma & df.item_id.isin(sys_ids)
    if "proof" in steps:
        drop |= is_ma & df.item_id.isin(set(it[it.scale == "interval"].item_id))
    if "kangaroo" in steps:
        drop |= is_ma & df.item_id.isin(set(it[it.content.str.startswith(KANGAROO)].item_id))
    fa = it[(it.scale == "discrete") & ~it.item_id.isin(sys_ids) & ~it.content.str.startswith(KANGAROO)]
    ma = df[is_ma].merge(fa[["item_id", "competition", "problem_idx"]], on="item_id")
    if "misaligned" in steps:
        ma = ma.assign(suffix=ma.item_content.str.strip().str[-120:])
        maj = (ma.groupby(["competition", "problem_idx", "suffix"]).subject_id.nunique().reset_index()
               .sort_values("subject_id", ascending=False).drop_duplicates(["competition", "problem_idx"]))
        own = {(c, i): x for c, i, x in zip(maj.competition, maj.problem_idx, maj.suffix)}
        majority_texts = {(c, x) for (c, i), x in own.items()}
        # misaligned: the prompt is not this index's majority statement but is another index's majority statement
        bad = [own.get((c, i)) != x and (c, x) in majority_texts
               for c, i, x in zip(ma.competition, ma.problem_idx, ma.suffix)]
        bad_cells = set(zip(ma[bad].subject_id, ma[bad].item_id))
        drop |= is_ma & pd.Series([(s, i) in bad_cells for s, i in zip(df.subject_id, df.item_id)], index=df.index)
    if "aimedup" in steps:
        aime = ma[ma.competition.str.startswith("aime")]
        labels = aime.groupby(["subject_id", "item_content"]).competition.agg(set)
        dup = labels[labels.map(len) > 1].index
        combined = {"aime_2025", "aime_2026"}
        rows = aime.set_index(["subject_id", "item_content"]).loc[dup].reset_index()
        bad_cells = set(zip(rows[rows.competition.isin(combined)].subject_id, rows[rows.competition.isin(combined)].item_id))
        drop |= is_ma & pd.Series([(s, i) in bad_cells for s, i in zip(df.subject_id, df.item_id)], index=df.index)
    return drop


_STEPS: set = set()


def clean_init(subsample: str = "pair", steps: tuple = ()):
    """Replacement for paired._init: same as the original, with MathArena rows removed."""
    import paired
    from paiec.data import BINARY_BENCHMARKS, load_responses
    from paiec.evaluate import FoldData

    df = load_responses()
    df = df[~matharena_drop_rows(df, set(steps))].reset_index(drop=True)
    paired._STATE["df"] = df
    paired._STATE["folds"] = FoldData(df, BINARY_BENCHMARKS, subsample=subsample)


def main() -> None:
    import functools
    import paired

    argv = sys.argv[1:]
    steps = ()
    if "--clean" in argv:
        k = argv.index("--clean")
        steps = tuple(s for s in argv[k + 1].split(",") if s)
        del argv[k:k + 2]
    sys.argv = [sys.argv[0]] + argv
    print(f"MathArena cleaning steps: {list(steps) or 'none'}", flush=True)
    # paired.main passes initargs=(subsample,); bind the steps so workers clean identically.
    paired._init = functools.partial(clean_init, steps=steps)
    paired.main()


if __name__ == "__main__":
    main()
