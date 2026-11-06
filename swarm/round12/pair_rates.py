"""Round 12: per-pair binary response counts from measurement-db revision 54074453 (release == public), the input of the
full-database subject prior. Rules: submissions/QUEUE.md, round-12 amendment item 3. Reads response tables only through
subject_id, item_id and response; writes swarm/round12/pair_rates.pkl and swarm/round12/bench_table.csv.

usage (from brier_rabbit/): nice -n 19 ../.venv/bin/python swarm/round12/pair_rates.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paiec.data import BINARY_BENCHMARKS, SUBJECT_FIELDS, _text, load_responses  # noqa: E402
from paiec.subjects import name_key  # noqa: E402

MAIN = ROOT.parent / "data" / "measurement-db-main"
OUT = ROOT / "swarm" / "round12"
SIX = ("matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
MIN_PAIR_N = 20  # item 3


def direction(scale) -> str | None:
    try:
        return json.loads(scale).get("direction")
    except Exception:
        return None


def qualifies(row) -> tuple[bool, str]:
    """Item 3 and the operational reading for mixed benchmarks."""
    if row["release"] != "public":
        return False, "withheld"
    if row["response_type"] == "binary":
        d = direction(row["response_scale"])
        return d == "higher_is_better", f"binary direction {d}"
    if row["response_type"] == "mixed":
        meta = (MAIN / row["dir"] / "metadata.yaml").read_text()
        bad = re.findall(r"direction:\s*(lower_is_better|unordered)", meta)
        return not bad, "mixed, no opposing direction" if not bad else "mixed with " + bad[0]
    return False, f"type {row['response_type']}"


def one_bench(bench: str) -> tuple[pd.DataFrame | None, dict]:
    path = MAIN / bench / "formatted_tables" / "responses.parquet"
    t = pq.read_table(path, columns=["subject_id", "item_id", "response"])
    total = t.num_rows
    mask = pc.is_in(t["response"], value_set=pa.array([0.0, 1.0], type=t["response"].type))
    t = t.filter(mask)
    info = {"rows": total, "binary_rows": t.num_rows}
    if total == 0 or t.num_rows / total < 0.5:
        return None, {**info, "why": "under half the rows are 0 or 1"}
    g = t.group_by("subject_id").aggregate([("response", "count"), ("response", "sum"), ("item_id", "count_distinct")])
    d = g.to_pandas().rename(columns={"response_count": "n", "response_sum": "s", "item_id_count_distinct": "n_items"})
    subj = pd.read_parquet(MAIN / bench / "formatted_tables" / "subjects.parquet")
    for f in SUBJECT_FIELDS:
        subj[f] = subj[f].map(lambda v: _text(v).strip()) if f in subj else ""
    d = d.merge(subj[["subject_id", *SUBJECT_FIELDS]], on="subject_id", how="left")
    d["bench"] = bench
    return d, info


def six_rows() -> pd.DataFrame:
    """The original benchmarks at bc8204d8, through the loader every shipped prior used."""
    df = load_responses()
    pairs = df.drop_duplicates("pair")[["bench", "pair", "subject_id", *SUBJECT_FIELDS]]
    agg = df.groupby("pair").agg(n=("label", "count"), s=("label", "sum"), n_items=("item_id", "nunique"))
    return pairs.join(agg, on="pair").drop(columns="pair")


def main() -> None:
    cat = pd.read_pickle(OUT / "catalog_benchmarks.pkl")
    table, frames = [], [six_rows()]
    for _, row in cat.sort_values("dir").iterrows():
        bench = row["dir"]
        if bench in SIX:
            table.append({"bench": bench, "kept": bench in BINARY_BENCHMARKS, "why": "original six, bc8204d8 tables"})
            continue
        ok, why = qualifies(row)
        rec = {"bench": bench, "kept": False, "why": why, "response_type": row["response_type"]}
        if ok:
            if not (MAIN / bench / "formatted_tables" / "responses.parquet").exists():
                rec["why"] = "responses not downloaded"
            else:
                d, info = one_bench(bench)
                rec.update(info)
                if d is not None:
                    rec["kept"] = True
                    frames.append(d)
                else:
                    rec["why"] = info["why"]
        table.append(rec)
        print(bench, rec.get("rows"), rec.get("kept"), flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["n"] >= MIN_PAIR_N].copy()
    keys = {}
    for r in rows.drop_duplicates(list(SUBJECT_FIELDS))[list(SUBJECT_FIELDS)].itertuples(index=False):
        keys[tuple(r)] = name_key(dict(zip(SUBJECT_FIELDS, r)))
    rows["key"] = [keys[tuple(v)] for v in rows[list(SUBJECT_FIELDS)].itertuples(index=False)]
    rows.to_pickle(OUT / "pair_rates.pkl")
    pd.DataFrame(table).to_csv(OUT / "bench_table.csv", index=False)
    named = rows[rows["key"] != ""]
    print("benchmarks kept:", rows["bench"].nunique(), "| pairs:", len(rows), "| named pairs:", len(named),
          "| names:", named["key"].nunique())


if __name__ == "__main__":
    main()
