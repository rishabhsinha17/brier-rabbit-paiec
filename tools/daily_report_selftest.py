"""Self-test of tools/daily_report.py: decision-rule branches on synthetic estimate tables, plus an end-to-end run on a
scratch copy of the logs with a known injected effect. Touches nothing outside a temporary directory.

usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python tools/daily_report_selftest.py   (~30-60 s)

End-to-end design: two fake arms are added in a temporary copy of submissions/feedback + ledger.csv, built from REF's
own logs (same pairs, same draws): W20 = REF + d (B1 +10e-3, so ALC +2e-3) and TWIN = REF + 0. Two arms whose cells sit
in exactly the same units and differ by a constant have least-squares effects that differ by exactly that constant, so
W20 - TWIN must equal the injection to rounding in every estimator, subset and budget; the rule outputs must follow.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import daily_report as dr  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def fake_table(rows: list[dict]) -> pd.DataFrame:
    base = {"se_boot": 0.0, "boot_na": 0.0, "se_cr1": 0.0, "floor": 0.0, "binds": "floor", "K": 10, "K_any": 10,
            "K_alone": 5, "K_shared": 5}
    out = []
    for r in rows:
        d = {**base, **r}
        d["z"] = d["effect"] / d["se_h"]
        out.append(d)
    return pd.DataFrame(out)


def rule_branches() -> None:
    A = "armX"
    mk = lambda sec, e, se, subset="select", est="paircov": {"subset": subset, "est": est, "arm": A,  # noqa: E731
                                                              "section": sec, "effect": e, "se_h": se}
    t = fake_table([mk("alc", 0.004, 0.002)])
    check("screen DROP on ALC > +3.5e-3", dr.screen(t, "paircov", A)[0] == "DROP")
    t = fake_table([mk("alc", 0.001, 0.002), mk("b1", 0.011, 0.005)])
    check("screen DROP on B1 z > 2", dr.screen(t, "paircov", A)[0] == "DROP", dr.screen(t, "paircov", A)[1])
    t = fake_table([mk("alc", 0.001, 0.002), mk("b7", 0.050, 0.005)])
    check("screen ignores B7 for the drop rule", dr.screen(t, "paircov", A)[0] == "CONTINUE")
    t = fake_table([mk("alc", -0.003, 0.0025)])
    check("screen FINALIST at ALC < 0, z <= -1", dr.screen(t, "paircov", A)[0] == "FINALIST")
    t = fake_table([mk("alc", -0.001, 0.0025)])
    check("screen CONTINUE at z -0.4", dr.screen(t, "paircov", A)[0] == "CONTINUE")
    t = fake_table([mk("alc", 0.0035, 0.0025)])
    check("screen CONTINUE at exactly +3.5e-3", dr.screen(t, "paircov", A)[0] == "CONTINUE")
    check("screen n/a without evidence", dr.screen(fake_table([mk("alc", 0.0, 1.0)]).iloc[0:0], "paircov", A)[0] == "n/a")

    c = lambda e, se: dr.calib(fake_table([mk("b0", e, se, est="alone")]), A)  # noqa: E731
    check("calib H_pub", c(-0.020, 0.005)["out"] == "H_pub")
    check("calib H_cal", c(0.010, 0.005)["out"] == "H_cal")
    r = c(-0.005, 0.005)
    check("calib UNDECIDED (leans H_pub)", r["out"] == "UNDECIDED" and "H_pub" in r["act"], f"z {r['z']:+.2f}")
    r = c(0.0, 0.005)
    check("calib UNDECIDED (leans H_cal)", r["out"] == "UNDECIDED" and "H_cal" in r["act"], f"z {r['z']:+.2f}")
    check("calib uses the alone+P B0 only", dr.calib(fake_table([mk("b0", -0.02, 0.005)]), A) is None)

    lines = dr.w20_readout(fake_table([mk("b1", -0.006, 0.003), mk("b1", -0.004, 0.004, est="alone")]), A)
    check("W20 readout supports widening, closest H_pub", "supports widening" in lines[0] and "closest H_pub" in lines[0],
          lines[0])
    lines = dr.w20_readout(fake_table([mk("b1", 0.006, 0.003)]), A)
    check("W20 readout against widening, closest R1", "does not support" in lines[0] and "closest R1" in lines[0])

    rob = {"bal": pd.Series({A: -0.001}), "cover": pd.Series({A: 0.9}), "pw": pd.Series({A: 0.0}),
           "lobo_max": pd.Series({A: 0.0005})}
    loc = {A: {"unw": -0.001, "rw": 0.001, "bench_max": 0.002}}
    good = [mk("alc", -0.002, 0.004, "holdout"), mk("alc", -0.005, 0.0025, "all")] + \
           [mk(s, 0.001, 0.002, "all") for s in dr.BUDGETS]
    items, _ = dr.switch_check(fake_table(good), A, rob, loc, 1)
    check("switch all pass (z -2.0 <= -1.64)", all(ok for _, ok, _ in items), str([(k, ok) for k, ok, _ in items]))
    items, _ = dr.switch_check(fake_table(good), A, rob, loc, 2)
    check("switch (b) passes with two finalists at z -2.0 <= -1.8", dict((k, ok) for k, ok, _ in items)["b"] is True)
    mid = [good[0], mk("alc", -0.0017, 0.001, "all")] + good[2:]  # z -1.7
    b1 = dict((k, ok) for k, ok, _ in dr.switch_check(fake_table(mid), A, rob, loc, 1)[0])["b"]
    b2 = dict((k, ok) for k, ok, _ in dr.switch_check(fake_table(mid), A, rob, loc, 2)[0])["b"]
    check("switch (b) at z -1.7: passes with one finalist, fails with two", b1 is True and b2 is False)
    bad = [mk("alc", 0.001, 0.004, "holdout"), mk("alc", -0.004, 0.0025, "all")] + \
          [mk(s, 0.0055, 0.0025, "all") if s == "b3" else mk(s, 0.0, 0.002, "all") for s in dr.BUDGETS]
    rob2 = {**rob, "bal": pd.Series({A: 0.0002}), "lobo_max": pd.Series({A: 0.0015})}
    loc2 = {A: {"unw": 0.0025, "rw": 0.0, "bench_max": 0.006}}
    items = dict((k, ok) for k, ok, _ in dr.switch_check(fake_table(bad), A, rob2, loc2, 1)[0])
    check("switch fails a,b,c,d,e,f", not any(items.values()), str(items))
    items = dict((k, ok) for k, ok, _ in dr.switch_check(fake_table(good[1:]), A, rob, {}, 1)[0])
    check("switch: missing holdout / local guard count as not passed", items["a"] is None and items["f"] is None)

    # round-6 amendment (final look): K_full >= 60 -> (b) and (e) use max(1.2 x boot, CR1), no SD/sqrt(K) floor
    def fl(k_any: int, b7_effect: float = 0.0):
        alc = {**mk("alc", -0.003, 0.0025, "all"), "se_boot": 0.001, "se_cr1": 0.0014, "K_any": k_any}  # floored z -1.2
        bud = [{**mk(s, b7_effect if s == "b7" else 0.0, 0.004, "all"), "se_boot": 0.001, "se_cr1": 0.0012,
                "K_any": k_any} for s in dr.BUDGETS]
        return fake_table([good[0], alc] + bud)
    it, ex = dr.switch_check(fl(80), A, rob, loc, 2, final_look=True)
    zb = -0.003 / 0.0014
    check("final look: (b) uses max(1.2 boot, CR1) at K_full >= 60", dict((k, ok) for k, ok, _ in it)["b"] is True
          and abs(ex["z_b"] - zb) < 1e-12 and ex["nofloor"], f"z_b {ex['z_b']:+.3f} vs {zb:+.3f}")
    it, ex = dr.switch_check(fl(59), A, rob, loc, 2, final_look=True)
    check("final look: K_full 59 keeps the floored SE_h", dict((k, ok) for k, ok, _ in it)["b"] is False
          and not ex["nofloor"] and abs(ex["z_b"] + 1.2) < 1e-12, f"z_b {ex['z_b']:+.3f}")
    it, ex = dr.switch_check(fl(80), A, rob, loc, 2)
    check("no final look: floored SE_h even at K_full 80", dict((k, ok) for k, ok, _ in it)["b"] is False
          and not ex["nofloor"])
    it, _ = dr.switch_check(fl(80, b7_effect=0.0030), A, rob, loc, 2, final_look=True)  # floored z 0.75, no floor 2.5
    it0, _ = dr.switch_check(fl(80, b7_effect=0.0030), A, rob, loc, 2)
    check("final look: (e) also drops the floor (B7 +3e-3: fails without floor, passes with it)",
          dict((k, ok) for k, ok, _ in it)["e"] is False and dict((k, ok) for k, ok, _ in it0)["e"] is True)

    # round-5 confirmatory set: first seen in our run >= first_id, never in a public log
    wo = pd.DataFrame({"key": ["p1", "p1", "p2", "p3", "p3", "p4"],
                       "submission": ["100", "205", "210", "90", "220", "230"]})
    wp = pd.concat([wo, pd.DataFrame({"key": ["p4"], "submission": ["pub_x:1"]})], ignore_index=True)
    fk = dr.fresh_keys(wo, wp, 200)
    check("fresh pairs: first seen at/after the cut and in no public log", fk == {"p2"}, str(sorted(fk)))


def end_to_end(reps: int = 60) -> None:
    import feedback as fb

    d_b1 = 0.010
    tmp = Path(tempfile.mkdtemp(prefix="daily_report_selftest_"))
    try:
        fbdir = tmp / "submissions" / "feedback"
        shutil.copytree(fb.FEEDBACK, fbdir)
        ledger = pd.read_csv(fb.FEEDBACK.parent / "ledger.csv", dtype=str)
        ref_ids = [i for i in ledger.loc[ledger.arm == dr.REF, "id"] if (fbdir / f"{i}.json").exists()]
        new_rows = []
        for n, rid in enumerate(ref_ids):
            rec = json.loads((fbdir / f"{rid}.json").read_text())
            for arm, d, base_id in (("selftest_w20", d_b1, 990000), ("selftest_twin", 0.0, 991000)):
                fake = {**rec, "id": base_id + n, "p": [[k, nn, alc + 0.2 * d, b0, b1 + d, *rest]
                                                         for k, nn, alc, b0, b1, *rest in rec["p"]]}
                (fbdir / f"{base_id + n}.json").write_text(json.dumps(fake))
                new_rows.append({"id": str(base_id + n), "arm": arm, "status": "Finished",
                                 "uploaded_utc": "2026-10-06 00:00", "note": "selftest"})
        pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True).to_csv(tmp / "submissions" / "ledger.csv",
                                                                              index=False)
        saved = fb.FEEDBACK
        fb.FEEDBACK = fbdir
        dr.SD_ARM["selftest_w20"] = dr.SD_ARM.get(dr.W20_ARM, {})
        old_w20 = dr.W20_ARM
        try:
            dr.W20_ARM = "selftest_w20"
            w_ours, w_pub, _, _ = dr.load_data("rec", None)
            tabs = []
            for name, f in {"select": lambda w: w[~w["holdout"]], "holdout": lambda w: w[w["holdout"]],
                            "all": lambda w: w}.items():
                kt = dr.k_table(f(w_ours))
                tabs.append(dr.estimate(f(w_ours).copy(), "paircov", reps, 1, kt, name))
                tabs.append(dr.estimate(f(w_pub).copy(), "alone", reps, 3, kt, name))
            t = pd.concat(tabs, ignore_index=True).set_index(["subset", "est", "arm", "section"])
            worst = 0.0
            for (subset, est) in {(s, e) for s, e, _, _ in t.index}:
                for s in dr.SECTIONS:
                    k1, k2 = (subset, est, "selftest_w20", s), (subset, est, "selftest_twin", s)
                    if k1 in t.index and k2 in t.index and np.isfinite(t.loc[k1, "effect"]):
                        inj = d_b1 if s == "b1" else (0.2 * d_b1 if s == "alc" else 0.0)
                        worst = max(worst, abs(t.loc[k1, "effect"] - t.loc[k2, "effect"] - inj))
            check("end-to-end: W20 - TWIN = injection in every estimator/subset/budget", worst < 1e-12,
                  f"max |error| {worst:.1e}")
            twin = t.loc[("select", "paircov", "selftest_twin", "b1"), "effect"]
            check("end-to-end: exact REF twin has ~0 effect", abs(twin) < 1e-12, f"twin B1 {twin:+.1e}")
            tt = t.reset_index()
            lines = dr.w20_readout(tt, "selftest_w20")
            check("end-to-end: W20 readout says against widening", "does not support" in lines[0], lines[0])
            out, why = dr.screen(tt, "paircov", "selftest_w20")
            b1 = t.loc[("select", "paircov", "selftest_w20", "b1")]
            check("end-to-end: screen reacts to B1 z", (out == "DROP") == (b1["z"] > 2), f"{out}: {why}")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dr.main(["--reps", str(reps), "--arms", "selftest_w20,selftest_twin,L15", "--final-look"])
            txt = buf.getvalue()
            check("end-to-end: full report runs and lists both fake arms",
                  "selftest_w20" in txt and "selftest_twin" in txt and "== 4. DECISIONS" in txt)
            check("end-to-end: --final-look prints the finalist count and the fresh-pairs section",
                  "final look: finalists" in txt and "== 5. CONFIRMATORY FRESH-PAIRS ESTIMATE" in txt)
        finally:
            fb.FEEDBACK = saved
            dr.W20_ARM = old_w20
            dr.SD_ARM.pop("selftest_w20", None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    rule_branches()
    end_to_end()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
