"""Rerun one queued replay exactly as it ran for the report, from results/JOB_ARGS.json.

usage (from the repository root, data at ../data/measurement-db):
  python scripts/run_job.py --list               # job names, status and output
  python scripts/run_job.py combo_v1 --print     # show the command only
  PAIEC_WORKERS=2 python scripts/run_job.py combo_v1 [--out results/combo_v1_rerun.csv]

--out redirects the output CSV so the saved one is kept for comparison (`cmp` or pandas); draws are seeded, so a
rerun on the same data and environment reproduces the saved file.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    jobs = {j["name"]: j for j in json.loads((ROOT / "results" / "JOB_ARGS.json").read_text())["jobs"]}
    if args.list or not args.name:
        for j in jobs.values():
            out = j["args"][j["args"].index("--out") + 1] if "--out" in j["args"] else "-"
            print(f"{j['name']:24s} {j['status']:10s} {j['args'][0]:52s} -> {out}")
        return 0
    cmd = list(jobs[args.name]["args"])
    if args.out:
        cmd[cmd.index("--out") + 1] = args.out
    print(" ".join(shlex.quote(c) for c in [sys.executable, *cmd]))
    if args.print:
        return 0
    env = dict(os.environ)
    env.setdefault("PAIEC_WORKERS", "2")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(v, "1")
    return subprocess.call([sys.executable, *cmd], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
