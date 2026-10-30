"""Machine-readable submission ledger (submissions/ledger.csv) and its markdown view (submissions/LEDGER.md).

usage:
  python tools/ledger.py add ID ZIP --arm ARM --note "..."     # after an upload
  python tools/ledger.py score ID STATUS ALC PAIRS              # after results
  python tools/ledger.py final ID --why "..."                   # set cur_final
  python tools/ledger.py render                                 # rewrite LEDGER.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
from pathlib import Path

SUB = Path(__file__).resolve().parents[1] / "submissions"
CSV = SUB / "ledger.csv"
FIELDS = ["id", "uploaded_utc", "zip", "sha256", "arm", "status", "alc", "pairs", "note"]
FINAL = SUB / "cur_final.txt"


def read() -> list[dict]:
    if not CSV.exists():
        return []
    with CSV.open() as fh:
        return list(csv.DictReader(fh))


def write(rows: list[dict]) -> None:
    with CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    render()


def render() -> None:
    rows = read()
    final = FINAL.read_text().strip() if FINAL.exists() else "(none)"
    out = ["# Submission ledger (generated from ledger.csv by tools/ledger.py)", "", "## cur_final", final, "",
           "| ID | uploaded (UTC) | zip | sha256 | arm | status | live ALC | pairs | note |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['id']} | {r['uploaded_utc']} | {r['zip']} | `{r['sha256'][:16]}` | {r['arm']} | {r['status']} | "
                   f"{r['alc']} | {r['pairs']} | {r['note']} |")
    (SUB / "LEDGER.md").write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("add"); a.add_argument("id"); a.add_argument("zip"); a.add_argument("--arm", default="")
    a.add_argument("--note", default=""); a.add_argument("--when", default="")
    s = sp.add_parser("score"); s.add_argument("id"); s.add_argument("status"); s.add_argument("alc"); s.add_argument("pairs")
    f = sp.add_parser("final"); f.add_argument("id"); f.add_argument("--why", required=True)
    sp.add_parser("render")
    args = ap.parse_args()
    rows = read()
    if args.cmd == "add":
        z = SUB / args.zip
        sha = hashlib.sha256(z.read_bytes()).hexdigest() if z.exists() else ""
        when = args.when or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        rows.append({"id": args.id, "uploaded_utc": when, "zip": args.zip, "sha256": sha, "arm": args.arm,
                     "status": "Submitted", "alc": "", "pairs": "", "note": args.note})
        write(rows)
    elif args.cmd == "score":
        for r in rows:
            if r["id"] == args.id:
                r.update(status=args.status, alc=args.alc, pairs=args.pairs)
        write(rows)
    elif args.cmd == "final":
        r = next(r for r in rows if r["id"] == args.id)
        FINAL.write_text(f"**{r['id']}** ({r['zip']}, arm {r['arm']}, status {r['status']}, live ALC {r['alc']} on "
                         f"{r['pairs']} pairs). Why: {args.why}")
        render()
    else:
        render()


if __name__ == "__main__":
    main()
