"""Section 1c: theta that the LIVE prior (fit on all public data, as shipped) assigns to plausible hidden-pool cohorts,
relative to the public-LOBO pairs the level calculus is calibrated on.

Cohorts are built from public subject records only (no hidden information):
  PUB_LOBO   the 220 eligible public pairs with their LOBO theta (the calculus reference; H_pub)
  REC_name   public subject records released >= 2025.5 (or >= 2026.0), live-prior theta (their names are in the table)
  REC_attr   the same records forced through the attribute ridge (name unseen), base / x1.56 / x1.56 + clamp 2026.4
  FRONT      the 15 highest-theta canonical names of the live prior (a frontier-only pool)
Also: release-date extrapolation of a generic unknown subject per provider at 2025.0 .. 2026.9.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402
from paiec.data import SUBJECT_FIELDS  # noqa: E402

ROOT = sl.ROOT


def attr_theta(pr, subj, scale=1.0, ymax=0.0):
    x = sl.attributes(subj)
    if ymax:
        x[1] = min(x[1], ymax - 2025.0)
    return scale * float(np.dot(pr["attr"]["coef"], x))


def main():
    pd.set_option("display.width", 250)
    df = sl.load()
    P = sl.lobo_priors(df)
    ALL = P["ALL"]
    shipped = json.loads((ROOT / "submission_src" / "v3b_noM_n_m0n20" / "subject_prior.json").read_text())
    same = all(abs(ALL["names"][k][0] - shipped["names"][k][0]) < 1e-9 for k in shipped["names"]) and \
        np.allclose(ALL["attr"]["coef"], shipped["attr"]["coef"])
    print(f"live prior check: refit on all public data == shipped subject_prior.json: {same} "
          f"({len(shipped['names'])} names)")
    pt = sl.pair_table(df)
    el = pt[pt.eligible & pt.bench.isin(sl.LOBO)].copy()
    lobo = np.array([sl.theta(P[b], s)[0] for b, s in zip(el.bench, el.subject)])
    lobo_w = el.groupby("bench").bench.transform(lambda s: 1.0 / len(s)).values
    ref_pairw, ref_benchw = lobo.mean(), float(np.sum(lobo * lobo_w) / lobo_w.sum())
    print(f"PUB_LOBO reference: mean theta pair-weighted {ref_pairw:+.3f}, bench-equal {ref_benchw:+.3f}, sd {lobo.std():.3f}")
    # every public subject record (one per bench x subject); dedupe by (key, effort, harness) to avoid double counting
    rec = pt.copy()
    rec["th_live"] = [sl.theta(ALL, s)[0] for s in rec.subject]
    rec["src_live"] = [sl.theta(ALL, s)[2] for s in rec.subject]
    rec["th_attr"] = [attr_theta(ALL, s) for s in rec.subject]
    rec["th_attr156"] = [attr_theta(ALL, s, 1.56) for s in rec.subject]
    rec["th_attr156c"] = [attr_theta(ALL, s, 1.56, 2026.4) for s in rec.subject]
    rows = []
    for lab, mask in (("all public records", rec.rel.notna() | rec.rel.isna()),
                      ("released < 2025.0", rec.rel < 2025.0),
                      ("released 2025.0-2025.5", (rec.rel >= 2025.0) & (rec.rel < 2025.5)),
                      ("released >= 2025.5", rec.rel >= 2025.5),
                      ("released >= 2026.0", rec.rel >= 2026.0),
                      ("undated", rec.rel.isna())):
        h = rec[mask]
        rows.append({"cohort": lab, "n": len(h), "name_share": (h.src_live == "name").mean(),
                     "live(names)": h.th_live.mean(), "sd_live": h.th_live.std(),
                     "attr_base": h.th_attr.mean(), "attr_x1.56": h.th_attr156.mean(),
                     "attr_x1.56_clamp": h.th_attr156c.mean(), "sd_attr156": h.th_attr156.std()})
    C = pd.DataFrame(rows)
    for c in ("live(names)", "attr_base", "attr_x1.56", "attr_x1.56_clamp"):
        C[f"d_{c}"] = C[c] - ref_pairw
    print("\nmean theta by cohort under the live prior (names as known; or forced through the attribute ridge), and the"
          " offset vs PUB_LOBO (pair-weighted):")
    print(C.round(3).to_string(index=False))
    names = pd.Series({k: v[0] for k, v in ALL["names"].items()}).sort_values(ascending=False)
    k_of = pd.Series({k: round(1 / v[1] - 1) for k, v in ALL["names"].items()})
    rel_of = rec.groupby("key").rel.max()
    top = pd.DataFrame({"theta": names, "k": k_of.reindex(names.index), "latest_release": rel_of.reindex(names.index)})
    print("\ntop-20 canonical names in the live prior (theta, benchmarks seen k, latest release of any record):")
    print(top.head(20).round(3).to_string())
    print(f"\nFRONT (top-15 names) mean theta {names.head(15).mean():+.3f}; top-30 {names.head(30).mean():+.3f}; "
          f"all names {names.mean():+.3f} (sd {names.std():.3f})")
    print("\nname theta vs latest release year (names with a dated record): ")
    t2 = top.dropna()
    for lo, hi in ((2023, 2025.0), (2025.0, 2025.5), (2025.5, 2026.0), (2026.0, 2027)):
        h = t2[(t2.latest_release >= lo) & (t2.latest_release < hi)]
        print(f"   latest release {lo}-{hi}: n={len(h)} mean theta {h.theta.mean():+.3f} sd {h.theta.std():.3f}")
    print("\ngeneric unknown subject (only provider + release date set), attribute theta base / x1.56 / x1.56 clamp:")
    for prov in ("OpenAI", "Anthropic", "Google", "DeepSeek", "Alibaba", ""):
        line = []
        for d in ("2025-01-01", "2025-07-01", "2026-01-01", "2026-06-01", "2026-10-01"):
            s = {f: "" for f in SUBJECT_FIELDS}
            s["release_date"], s["provider"] = d, prov
            line.append(f"{d[:7]} {attr_theta(ALL, s):+.2f}/{attr_theta(ALL, s, 1.56):+.2f}/{attr_theta(ALL, s, 1.56, 2026.4):+.2f}")
        print(f"   {prov or '(none)':10s} " + " | ".join(line))
    s = {f: "" for f in SUBJECT_FIELDS}
    s.update(release_date="2026-03-01", provider="OpenAI", reasoning_effort="xhigh", harness="codex")
    s2 = dict(s, reasoning_effort="low")
    print(f"   effort xhigh vs low (OpenAI 2026-03): {attr_theta(ALL, s):+.2f} vs {attr_theta(ALL, s2):+.2f} (base)")
    rec.drop(columns=["subject"]).to_csv(Path(__file__).with_name("composition_records.csv"), index=False)


if __name__ == "__main__":
    main()
