"""Parse our own Codabench "Scoring output" logs and compare submissions on matching anonymous pair IDs.

usage:
  python tools/feedback.py summary                  # one line per saved log: ALC and per-budget means
  python tools/feedback.py compare A_ID B_ID [...]  # paired differences on (subject_id, benchmark_id) both arms saw
  python tools/feedback.py fe [section] [ref_arm] [--public]    # pool-aware two-way FE, arms from ledger.csv
  python tools/feedback.py boot [section] [ref_arm] [--public]  # pair bootstrap of the same FE
  python tools/feedback.py public                   # summary of the public reference arms only

Logs live in submissions/feedback/<submission_id>.txt (pasted verbatim from LOGS -> Scoring output).
The kit README endorses comparing "matching subject-benchmark IDs within the same snapshot". IDs are only ever used
as opaque join keys: never mapped to identities, never used to tune per-ID behaviour.

--public (or PAIEC_FEEDBACK_PUBLIC=1 in the environment, which also reaches tools/swarm_evalsci.py through load() and
ledger_arms()) adds other teams' public formative logs as reference arms (pub_<owner>_<arm>): MIT-licensed files in
research/public_logs/<owner>/, mapped to arms by research/public_logs/arms.csv (sources in each PROVENANCE.md). They
chain our arms through extra matched pairs; they are used only for arm comparisons, never per pair. Acknowledge the
owners (aghasalim, doctawho42) if a decision or the report uses them. --public=ARM1,ARM2 (or
PAIEC_FEEDBACK_PUBLIC=ARM1,ARM2) keeps only those public arms, e.g. for a leave-one-public-arm-out check.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEEDBACK = Path(__file__).resolve().parents[1] / "submissions" / "feedback"
PUBLIC = Path(__file__).resolve().parents[1] / "research" / "public_logs"
ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
BUDGETS = (0, 1, 3, 7, 15, 31)


def _public_default(include_public):
    """None (no public arms), "all", or a set of public arm labels. include_public=None reads PAIEC_FEEDBACK_PUBLIC
    ("" or "0": none; "1"/"all"/"true": all; otherwise a comma-separated list of arm labels)."""
    if include_public is None:
        include_public = os.environ.get("PAIEC_FEEDBACK_PUBLIC", "")
    if isinstance(include_public, str):
        v = include_public.strip()
        if v.lower() in ("", "0", "false", "no"):
            return None
        return "all" if v.lower() in ("1", "all", "true", "yes") else {a.strip() for a in v.split(",") if a.strip()}
    if isinstance(include_public, (set, list, tuple, frozenset)):
        return set(include_public) or None
    return "all" if include_public else None


def parse(text: str, submission_id: str) -> pd.DataFrame:
    rows, section = [], None
    for line in text.splitlines():
        s = line.strip()
        if s == "ALC summary":
            section = "alc"
        elif s.startswith("Label budget:"):
            section = f"b{int(s.split(':')[1])}"
        m = ROW.match(s)
        if m and section:
            rows.append({"submission": submission_id, "section": section, "subject_id": m[1], "benchmark_id": m[2],
                         "n_unique": int(m[3]), "score": float(m[4]), "ece": float(m[5])})
    return pd.DataFrame(rows)


def parse_json(rec: dict) -> pd.DataFrame:
    """Compact rows from tools/fetch_feedback.js: [subj/bench, n, alc, b0, b1, b3, b7, b15, b31, ece]."""
    rows = []
    for key, n, alc, *budgets_and_ece in rec.get("p", []):
        subj, bench = key.split("/")
        values = dict(zip(["alc", *[f"b{b}" for b in BUDGETS]], [alc, *budgets_and_ece[:6]]))
        for section, score in values.items():
            rows.append({"submission": str(rec["id"]), "section": section, "subject_id": f"subject_{subj}",
                         "benchmark_id": f"benchmark_{bench}", "n_unique": int(n), "score": float(score),
                         "ece": float(budgets_and_ece[6]) if section == "alc" else float("nan")})
    return pd.DataFrame(rows)


def load(ids=None, include_public=None) -> pd.DataFrame:
    """Our saved logs in long format (one row per submission, section and pair). `ids` filters our submissions;
    include_public (True, "all", or a set / comma list of public arm labels; default: env PAIEC_FEEDBACK_PUBLIC)
    appends the selected public logs listed in research/public_logs/arms.csv."""
    import json

    frames, seen = [], set()
    for f in sorted(FEEDBACK.glob("*.txt")):
        if ids and f.stem not in ids:
            continue
        frames.append(parse(f.read_text(), f.stem))
        seen.add(f.stem)
    for f in sorted(FEEDBACK.glob("*.json")):
        if (ids and f.stem not in ids) or f.stem in seen:
            continue
        frames.append(parse_json(json.loads(f.read_text())))
    sel = _public_default(include_public)
    if sel:
        frames.append(load_public(sel))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _public_index(arms="all") -> list[dict]:
    import csv

    path = PUBLIC / "arms.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if arms != "all":
        unknown = set(arms) - {r["arm"] for r in rows}
        if unknown:
            raise ValueError(f"unknown public arms {sorted(unknown)}; see research/public_logs/arms.csv")
        rows = [r for r in rows if r["arm"] in arms]
    return rows


def load_public(arms="all") -> pd.DataFrame:
    """Other teams' public formative logs in the same long format as load(); submission = 'pub_<owner>:<id>'.
    `arms`: "all" or a set of arm labels to keep."""
    frames = [parse((PUBLIC / r["log"]).read_text(), r["submission"]) for r in _public_index(arms)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def public_arms(arms="all") -> dict:
    """{public submission: reference-arm label}, e.g. 'pub_aghasalim:959137' -> 'pub_aghasalim_v3A'."""
    return {r["submission"]: r["arm"] for r in _public_index(arms)}


def arms(df: pd.DataFrame, arm_of: dict) -> pd.DataFrame:
    """Pool pair results per arm. 'adj' = ALC - B31: B31 is nearly model-independent (pair difficulty), so this
    removes most draw-composition noise. Also reports early-budget deltas vs each pair's B31 floor."""
    wide = df.pivot_table(index=["submission", "subject_id", "benchmark_id"], columns="section", values="score").reset_index()
    wide["arm"] = wide["submission"].map(arm_of)
    wide = wide.dropna(subset=["arm"])
    wide["adj"] = wide["alc"] - wide["b31"]
    for b in (0, 1, 3):
        wide[f"d{b}"] = wide[f"b{b}"] - wide["b31"]
    g = wide.groupby("arm")
    out = g[["alc", "adj", "d0", "d1", "d3", "b31"]].mean()
    out["adj_se"] = g["adj"].std(ddof=1) / np.sqrt(g.size())
    out["pairs"] = g.size()
    out["subs"] = g["submission"].nunique()
    return out.sort_values("adj").round(4)


def fe_exact(cell: pd.DataFrame, ref: str) -> tuple[dict, dict, int]:
    """Exact least squares for score[c] = alpha[unit c] + beta[arm c], beta[ref] = 0, on cells with one row per
    (unit, arm) (columns pair, arm, score), and its classical standard errors Var(beta) = s^2 G^-1: G is the arm block
    of the design after absorbing the unit effects and s^2 = RSS / (cells - units - estimated arms), all on units seen
    under 2+ arms connected to ref. One arm matched directly to ref on K units gives SE = sqrt(2) s / sqrt(K): the
    contrast carries the noise of both arms. Returns ({arm: beta}, {arm: se}, dof); unconnected arms are absent."""
    arms_of = cell.groupby("pair")["arm"].apply(set).to_dict()
    units_of = cell.groupby("arm")["pair"].apply(set).to_dict()
    reach, todo = {ref}, [ref]
    while todo:
        for u in units_of.get(todo.pop(), ()):
            for b in arms_of[u] - reach:
                reach.add(b)
                todo.append(b)
    c = cell[cell["arm"].isin(reach)]
    c = c[c.groupby("pair")["arm"].transform("nunique") > 1]
    est = sorted(set(c["arm"]) - {ref})
    if not est:
        return {ref: 0.0}, {ref: 0.0}, 0
    ui, units = pd.factorize(c["pair"])
    col = {a: j for j, a in enumerate(est)}
    X = np.zeros((len(c), len(est)))
    rows = [(r, col[a]) for r, a in enumerate(c["arm"]) if a in col]
    X[[r for r, _ in rows], [j for _, j in rows]] = 1.0
    y = c["score"].to_numpy(float)
    cnt = np.bincount(ui).astype(float)
    xm = np.zeros((len(units), len(est)))
    np.add.at(xm, ui, X)
    xd = X - (xm / cnt[:, None])[ui]
    yd = y - (np.bincount(ui, weights=y) / cnt)[ui]
    gram = xd.T @ xd
    beta = np.linalg.solve(gram, xd.T @ yd)
    resid = yd - xd @ beta
    dof = len(c) - len(units) - len(est)
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    se = np.sqrt(np.maximum(s2 * np.diag(np.linalg.inv(gram)), 0.0))
    return ({ref: 0.0, **dict(zip(est, beta.tolist()))}, {ref: 0.0, **dict(zip(est, se.tolist()))}, dof)


def fixed_effects(df: pd.DataFrame, arm_of: dict, section: str = "alc", iters: int = 200,
                  pool_aware: bool = True) -> pd.DataFrame:
    """Two-way fixed effects: score[pair, arm] = pair effect + arm effect. Pairs seen under several arms identify the
    arm differences; with default acquisition a pair's evaluation items and revealed labels are identical across
    submissions, so these comparisons are nearly free of draw noise. Effects are relative to the first arm given.

    approx_se (fixed 2026-10-06, swarm round 2): the classical least-squares SE of the effect from fe_exact. The old
    value, sigma / sqrt(pairs_shared) with dof = cells - arms, ignored the unit parameters in the dof and the second
    arm's noise in a contrast, and understated the pair-bootstrap SE by a median factor of 2.05 (postdoc_evalsci).
    Effects keep their alternating-projection values (unchanged; within ~1e-4 of fe_exact)."""
    d = df[df.section == section].copy()
    d["arm"] = d["submission"].map(arm_of)
    d = d.dropna(subset=["arm"])
    d["pair"] = d["subject_id"] + "|" + d["benchmark_id"]
    if pool_aware:
        # A pair scores better when another pair of its benchmark shares the draw (cross-subject item pooling), so
        # only compare a pair with itself under the same condition: alone on its benchmark vs sharing it.
        n_same = d.groupby(["submission", "benchmark_id"])["subject_id"].transform("nunique")
        d["pair"] = d["pair"] + np.where(n_same > 1, "|shared", "|alone")
    cell = d.groupby(["pair", "arm"])["score"].mean().reset_index()
    arms_list = list(dict.fromkeys(arm_of.values()))
    beta = {a: 0.0 for a in arms_list}
    alpha = cell.groupby("pair")["score"].mean().to_dict()
    for _ in range(iters):
        r = cell["score"] - cell["arm"].map(beta)
        alpha = r.groupby(cell["pair"]).mean().to_dict()
        r = cell["score"] - cell["pair"].map(alpha)
        beta = r.groupby(cell["arm"]).mean().to_dict()
        ref = beta[arms_list[0]]
        beta = {a: v - ref for a, v in beta.items()}
    multi = cell.groupby("pair")["arm"].transform("nunique") > 1
    se = fe_exact(cell, arms_list[0])[1]
    rows = []
    for a in arms_list:
        n_shared = int(cell[(cell["arm"] == a) & multi]["pair"].nunique())
        effect = round(beta[a], 4) if (n_shared or a == arms_list[0]) else float("nan")  # unidentified without overlap
        rows.append({"arm": a, "effect_vs_" + arms_list[0]: effect, "pairs": int((cell["arm"] == a).sum()),
                     "pairs_shared": n_shared, "approx_se": round(se.get(a, float("nan")), 4)})
    return pd.DataFrame(rows)


def fe_bootstrap(df: pd.DataFrame, arm_of: dict, section: str = "alc", reps: int = 300, seed: int = 0) -> pd.DataFrame:
    """Pair-level bootstrap of the fixed-effects arm differences: resample distinct (subject, benchmark) pairs with
    replacement, refit, and report the mean, standard error and 90% interval of each arm's effect."""
    rng = np.random.default_rng(seed)
    d = df[df.section == section].copy()
    d["key"] = d["subject_id"] + "|" + d["benchmark_id"]
    keys = d["key"].unique()
    ref = list(dict.fromkeys(arm_of.values()))[0]
    col = f"effect_vs_{ref}"
    draws = []
    for _ in range(reps):
        pick = rng.choice(keys, size=len(keys), replace=True)
        parts = []
        for i, k in enumerate(pick):  # relabel duplicates so resampled pairs stay distinct units
            part = d[d["key"] == k].copy()
            part["subject_id"] = part["subject_id"] + f"#{i}"
            parts.append(part)
        fe = fixed_effects(pd.concat(parts), arm_of, section, iters=60).set_index("arm")[col]
        draws.append(fe)
    boot = pd.concat(draws, axis=1)
    point = fixed_effects(df, arm_of, section).set_index("arm")
    out = pd.DataFrame({"effect": point[col], "boot_se": boot.std(axis=1),
                        "lo90": boot.quantile(0.05, axis=1), "hi90": boot.quantile(0.95, axis=1),
                        "shared": point["pairs_shared"]})
    return out.round(4)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index=["submission", "subject_id", "benchmark_id"], columns="section", values="score")
    out = wide.groupby(level=0).mean()
    out["pairs"] = wide.groupby(level=0).size()
    return out[["pairs", "alc", *[f"b{b}" for b in BUDGETS if f"b{b}" in out.columns]]].round(4)


def compare(df: pd.DataFrame, arms: list[list[str]]) -> None:
    """arms: lists of submission IDs per model. Averages each arm per pair, then pairs arms on shared pair IDs."""
    wide = df.pivot_table(index=["submission", "subject_id", "benchmark_id"], columns="section", values="score")
    per_arm = []
    for i, ids in enumerate(arms):
        part = wide[wide.index.get_level_values(0).isin(ids)].groupby(level=[1, 2]).mean()
        per_arm.append(part.add_prefix(f"arm{i}_"))
    joined = per_arm[0].join(per_arm[1], how="inner")
    print(f"pairs seen by both arms: {len(joined)}")
    if joined.empty:
        return
    for col in ["alc", *[f"b{b}" for b in BUDGETS]]:
        a, b = joined.get(f"arm0_{col}"), joined.get(f"arm1_{col}")
        if a is None or b is None:
            continue
        d = b - a
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  {col:4s} arm0 {a.mean():.4f}  arm1 {b.mean():.4f}  diff(arm1-arm0) {d.mean():+.4f} ± {se:.4f}")


def ledger_arms(include_public=None) -> dict:
    """{our submission id: arm} from ledger.csv; with include_public (default: env PAIEC_FEEDBACK_PUBLIC) the public
    reference arms are appended after ours, so our first arm stays the reference unless a ref is given."""
    import csv

    path = FEEDBACK.parent / "ledger.csv"
    with path.open() as fh:
        out = {r["id"]: r["arm"] for r in csv.DictReader(fh) if r["arm"]}
    sel = _public_default(include_public)
    if sel:
        out.update(public_arms(sel))
    return out


def main() -> None:
    flags = [a for a in sys.argv if a == "--public" or a.startswith("--public=")]  # anywhere on the command line
    public = (("all" if flags[-1] == "--public" else flags[-1].split("=", 1)[1]) if flags else None)  # None: env
    argv = [a for a in sys.argv if a not in flags]
    if len(argv) < 2 or argv[1] == "summary":
        print(summary(load(include_public=public)).to_string())
    elif argv[1] == "public":  # the public reference arms alone: logs, pairs and mean scores per arm
        sel = _public_default(public) or "all"
        s = summary(load_public(sel))
        s.insert(0, "arm", s.index.map(public_arms(sel)))
        print(s.groupby("arm").agg(logs=("pairs", "size"), pairs=("pairs", "sum"), alc=("alc", "mean")).round(4))
    elif argv[1] == "compare":
        groups = [g.split(",") for g in argv[2:4]]
        compare(load(include_public=public), groups)
    elif argv[1] == "fe":  # fe [section] [ref_arm] -> arms from ledger.csv; or explicit ARM=id1,id2 ...
        section = argv[2] if len(argv) > 2 and "=" not in argv[2] else "alc"
        specs = [a for a in argv[2:] if "=" in a]
        if specs:
            arm_of = {sid: spec.split("=")[0] for spec in specs for sid in spec.split("=")[1].split(",")}
            sel = _public_default(public)
            if sel:
                arm_of.update(public_arms(sel))
        else:
            arm_of = ledger_arms(include_public=public)
            ref = argv[3] if len(argv) > 3 else None
            if ref:  # put the reference arm first
                arm_of = dict(sorted(arm_of.items(), key=lambda kv: kv[1] != ref))
        print(fixed_effects(load(include_public=public), arm_of, section).to_string(index=False))
    elif argv[1] == "boot":  # boot [section] [ref_arm]
        section = argv[2] if len(argv) > 2 else "alc"
        ref = argv[3] if len(argv) > 3 else None
        arm_of = ledger_arms(include_public=public)
        if ref:
            arm_of = dict(sorted(arm_of.items(), key=lambda kv: kv[1] != ref))
        print(fe_bootstrap(load(include_public=public), arm_of, section).to_string())
    elif argv[1] == "arms":  # arms ARM=id1,id2 ARM2=id3 ...
        arm_of = {sid: spec.split("=")[0] for spec in argv[2:] for sid in spec.split("=")[1].split(",")}
        print(arms(load(include_public=public), arm_of).to_string())


if __name__ == "__main__":
    main()
