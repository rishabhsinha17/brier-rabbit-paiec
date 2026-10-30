"""Final-selection rule (b): the live composite guard of submissions/QUEUE.md (round-5 rules, as made precise by the
composite readiness amendment of 2026-10-16).

  cd <repo>
  PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python tools/composite_guard.py COMPOSITE PLAIN [--reps N]
  e.g. tools/composite_guard.py v6_combo_w20 v5_w20_m20      (ledger arm names or daily_report aliases)
       tools/composite_guard.py v6_combo_w20_c1 v5_w20_m20 --pool v6_combo_w20   (a c1 build and its parent together)

Prints
1. every ledger run of the composite and how the amended failure clause reads it: scored (Finished or *-with-scores),
   dead before scoring (ignored, replaced), failed without scores (a composite failure if a plain run uploaded within one
   hour of it finished; a platform failure otherwise), or still in flight;
2. the paircov contrast COMPOSITE - PLAIN on our own runs of the two arms (PLAIN is the reference), on the selection set
   (holdout excluded, decisive) and on all pairs, with SE_h exactly as daily_report (max of 1.2 x stratified pair
   bootstrap, CR1 pair cluster, floor SD / sqrt(K)) and K = direct matches;
3. the verdict. Rule (b) trips on: selection-set ALC > +3.5e-3, or a composite failure, or two composite runs ending
   without scores whatever the cause. The other screen clauses (B0/B1/B3 worse by > 2 SE_h) are printed for
   information. Exceptions or time / memory limits in a run's logs are not visible here: read those logs by hand.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_report as dr  # noqa: E402

SCORED = ("Finished",)
DEAD_PREFIX = ("Dead", "Submitted", "Submitting", "Preparing")
NO_SCORES = ("Failed", "Failed-no-scores")
WINDOW = timedelta(hours=1)


def arm_name(a: str) -> str:
    rev = {v: k for k, v in dr.ALIAS.items()}
    return rev.get(a, a)


def when(r: dict) -> datetime | None:
    try:
        return datetime.strptime(r["uploaded_utc"], "%Y-%m-%d %H:%M")
    except (KeyError, ValueError):
        return None


def scored(status: str) -> bool:
    return status in SCORED or status.endswith("-with-scores")


def read_status(ledger: list[dict], comps: set[str], have: set[str]) -> tuple[list[tuple[str, str, str]], int, int]:
    """[(id, status, reading)], number of composite failures, number of composite runs ending without scores."""
    out, fails, noscore = [], 0, 0
    plain_rows = [r for r in ledger if r.get("arm") and r["arm"] not in comps]
    for r in ledger:
        if r.get("arm") not in comps:
            continue
        st = r.get("status", "") or ""
        if scored(st) or (not st and r["id"] in have):
            out.append((r["id"], st or "feedback saved", "scored"))
        elif st in NO_SCORES:
            noscore += 1
            t0 = when(r)
            near = [p for p in plain_rows if t0 and (t := when(p)) and abs(t - t0) <= WINDOW]
            ok = [p for p in near if scored(p.get("status", "") or "")]
            if ok:
                fails += 1
                out.append((r["id"], st, f"COMPOSITE FAILURE ({len(ok)} of {len(near)} plain runs within 1 h scored)"))
            else:
                out.append((r["id"], st, f"platform failure ({len(near)} plain runs within 1 h, none scored)"))
        elif st.startswith(DEAD_PREFIX):
            out.append((r["id"], st, "dead before scoring: ignored, replace it"))
        else:
            out.append((r["id"], st or "-", "in flight"))
    return out, fails, noscore


def contrast(comp: str, plain: str, reps: int, seed: int, pooled: tuple[str, ...] = ()):
    w_ours, _, _, _ = dr.load_data("none", None)
    w = w_ours[w_ours["arm"].isin([comp, plain, *pooled])].copy()
    w["arm"] = w["arm"].replace({x: comp for x in pooled})  # identical predictions: one arm
    if not (set(w["arm"]) >= {comp, plain}):
        return None
    w["arm"] = w["arm"].replace({plain: dr.REF})
    tables = []
    for subset, ws in (("select", w[~w["holdout"]]), ("all", w)):
        kt = dr.k_table(ws)
        tables.append(dr.estimate(ws, "paircov", reps, seed, kt, subset))
    import pandas as pd
    return pd.concat(tables, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("composite")
    ap.add_argument("plain")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--pool", action="append", default=[], metavar="ARM",
                    help="count the runs of ARM as the composite too (a build with identical predictions, "
                         "e.g. the parent of a c1 build)")
    a = ap.parse_args(argv)
    comp, plain = arm_name(a.composite), arm_name(a.plain)
    pooled = tuple(arm_name(x) for x in a.pool)
    tag_c, tag_p = dr.ALIAS.get(comp, comp), dr.ALIAS.get(plain, plain)
    also = f" (pooled with {', '.join(pooled)})" if pooled else ""
    print(f"== COMPOSITE GUARD (final-selection rule (b)): {tag_c} {comp}{also} vs plain {tag_p} {plain}")

    ledger = dr.read_ledger()
    have = {f.stem for f in dr.fb.FEEDBACK.glob("*.json")} | {f.stem for f in dr.fb.FEEDBACK.glob("*.txt")}
    runs, fails, noscore = read_status(ledger, {comp, *pooled}, have)
    print(f"\n1. composite runs in the ledger: {len(runs)}")
    for rid, st, reading in runs:
        print(f"   {rid}  {st:28s} {reading}")

    t = contrast(comp, plain, a.reps, a.seed, pooled)
    print("\n2. paircov contrast composite - plain (plain = reference), x1e-3: effect (SE_h, binds) z  [K direct]")
    alc = None
    if t is None or t.empty:
        print("   not identified: one of the two arms has no saved feedback")
    else:
        for subset in ("select", "all"):
            cells = []
            for s in dr.SECTIONS:
                r = dr.row(t, subset, "paircov", comp, s)
                if r is None:
                    cells.append(f"{s.upper()} n/a")
                    continue
                if subset == "select" and s == "alc":
                    alc = r
                cells.append(f"{s.upper()} {r['effect'] * 1e3:+.1f} ({r['se_h'] * 1e3:.1f}{r['binds'][0]}) "
                             f"z {r['z']:+.2f} [K {r['K']}]")
            label = "selection set" if subset == "select" else "all pairs   "
            print(f"   {label}: " + "; ".join(cells[:1]))
            print("                  " + "; ".join(cells[1:4]))
            print("                  " + "; ".join(cells[4:]))
        verdict, why = dr.screen(t, "paircov", comp, "select")
        print(f"   full screen on the selection set (information only): {verdict} ({why})")

    print("\n3. rule (b) reading")
    reasons = []
    if alc is not None and alc["effect"] > dr.DROP_ALC:
        reasons.append(f"selection-set ALC {alc['effect'] * 1e3:+.1f} > +3.5")
    if fails:
        reasons.append(f"{fails} composite failure(s)")
    if noscore >= 2:
        reasons.append(f"{noscore} composite runs ended without scores")
    n_scored = sum(1 for _, _, rd in runs if rd == "scored")
    if reasons:
        print(f"   TRIPS: {'; '.join(reasons)} -> ship the plain level-winner zip")
    elif alc is None:
        print("   NOT READ: no scored composite run matched with the plain arm yet")
    else:
        print(f"   holds: ALC {alc['effect'] * 1e3:+.1f} <= +3.5 on {n_scored} scored composite run(s), "
              f"{noscore} without scores, no composite failure")
    print("   reminder: read the logs of every composite run that ended without scores for an exception or a limit")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
