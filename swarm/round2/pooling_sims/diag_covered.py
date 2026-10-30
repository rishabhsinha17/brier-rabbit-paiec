"""Diagnostics restricted to evaluation rows whose item carries another subject's label (all arms are identical on the
other rows). For each arm: Brier on covered rows by budget, and calibration split by the partner's outcome (mean
prediction vs mean outcome after a partner success / failure). Usage: diag_covered.py <bench> <mode> <runs> <seed0>
'<extra json>'."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay import build_models, draws  # noqa: E402
from paiec.hier import _item_key, _sk  # noqa: E402
from paiec.protocol import run_protocol  # noqa: E402


class Probe:
    def predict(self, input, labeled=None):
        subject, item = input
        s, b, ik = _sk(subject), str(item.get("benchmark_id", "")), _item_key(item)
        ys = [y for (subj, it), y in labeled or [] if str(it.get("benchmark_id", "")) == b and _sk(subj) != s
              and _item_key(it) == ik]
        if not ys:
            return 0.0
        return 0.75 if np.mean(ys) >= 0.5 else 0.25


def main():
    bench, mode, runs, seed0, spec = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), json.loads(sys.argv[5])
    k = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    rows = []
    for r, pairs in draws(bench, k, runs, mode, seed0):
        rec = []
        run_protocol(pairs, Probe().predict, seed=r, record=rec)
        for name, model in build_models(bench, spec).items():
            mr = []
            run_protocol(pairs, model.predict, seed=r, record=mr)
            for (key, budget, flag, y), (_, _, prob, _) in zip(rec, mr):
                if flag > 0:
                    rows.append({"run": r, "pair": key, "budget": budget, "partner_y": int(flag > 0.5), "model": name,
                                 "p": prob, "y": y})
    df = pd.DataFrame(rows)
    out = HERE / "out" / f"diag_{bench}_{mode}_{seed0}_{runs}.csv"
    df.to_csv(out, index=False)
    df["se"] = (df.p - df.y) ** 2
    print(f"{bench} {mode} runs {seed0}-{seed0 + runs - 1}: covered rows per model {len(df) // max(df.model.nunique(), 1)}")
    print("Brier on covered rows by budget:")
    print(df.pivot_table(index="model", columns="budget", values="se", aggfunc="mean").round(4).to_string())
    print("calibration on covered rows (all budgets): mean p | mean y, by partner outcome:")
    t = df.groupby(["model", "partner_y"]).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).unstack()
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
