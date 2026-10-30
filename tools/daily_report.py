"""Daily live decision report: every pre-registered statistic of submissions/QUEUE.md and of the adjudicator's
section 3.2 (swarm/round2/adjudicator_level.md) for every live arm vs REF, in one command (swarm round 3).

Run it after every batch of live results, once the new logs are saved in submissions/feedback and their rows are in
submissions/ledger.csv (the arm column names the arm; a new arm needs nothing else):

  cd <repo>
  PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python tools/daily_report.py

  about 45-65 s wall (about 21 s CPU) at background QoS on the loaded laptop with the default 500 reps; --reps 200
  halves it. Options (all optional):
  --reps N            stratified pair-bootstrap reps per fit (default 500; 9 fits per run; printed with the runtime)
  --seed S            bootstrap seed (default 20261006). Every fit reseeds, so numbers do not depend on call order.
  --local CSV[:ref=MODEL,ARM=MODEL,...]   local LOBO guard (switch item f), repeatable. Default: results/adj_wide.csv,
                      results/pd1_priors.csv and results/paired_mix.csv with the arm -> model maps in LOCAL_MAP (first
                      CSV that maps an arm wins). Example for a new arm: --local results/adj_wide.csv:ref=base,v5_w20_m15=w20_m15
  --arms A,B          arms to report (ledger names or aliases such as L15,W20). Default: every arm with feedback that
                      is not historical (v1_/v2_/v3_ and v3b m0 -1.0). --all-arms adds the historical ones.
  --public SET        connector arms of the alone FE: rec (default, the 6 pre-registered public arms), all (adds
                      LegacyP = the adjudicator's round-2 set), none, or a comma list of pub_* labels
  --finalists N       finalists competing for the switch (z <= -1.64 for 1, <= -1.8 for 2+). Default: the number of arms
                      whose screen outcome is FINALIST, at least 1.
  --upto ID           only our submissions with id <= ID (reproduces an earlier snapshot; --upto 963848 --public all is
                      the adjudicator's round-2 data)
  --csv PATH          also write every estimate (subset x estimator x arm x section) as a long CSV
  --validate          cross-checks vs tools/swarm_evalsci.py and the adjudicator's reference code (adds ~20 s)
  --floor-b15b31      NOT pre-registered: also floor B15/B31 at the live matched SDs in FLOOR_SD_B15B31 (switch
                      item (e) uses every budget, and B15/B31 SEs are bootstrap/CR1 only, so small-K noise can fail it)
  --final-look        the 10-30 final look (round-6 amendment): arms with K_full >= 60 direct matches use SE_h =
                      max(1.2 x boot, CR1) in switch items (b) and (e) (z marked 'n'); without --finalists, the
                      finalist count is every non-dropped arm with >= 30 runs (round-4 item 7)
  --fresh-after ID    the round-5 confirmatory estimate (section 5) on pairs first seen in our run ID or later and in
                      no public log; --final-look turns it on with ID = FRESH_FIRST_ID (first upload after the
                      round-5 snapshot)
  --project R:A,...   planning aid: expected direct matches K for REF at R runs and an arm at A runs (stratified pool
                      model of swarm_evalsci.power), the floor-bound SE_h they imply, the chance that the L15 calibration
                      test and the screen decide, and the live matched per-pair SDs next to the QUEUE floors

Pre-registered rules implemented (QUEUE.md revision of 2026-10-06 09:10 UTC; adjudicator 3.2; round-1 postdoc_evalsci 2.4/4)
- Holdout: sha256('brier-rabbit-holdout|subject_<id>|benchmark_<id>'), IDs exactly as the logs print them, held out iff
  hex[:2] < '40' (tools/swarm_evalsci.holdout_flag). Screen, calibration test and W20 readout use the SELECTION set only;
  the holdout set is used only for switch item (a). Switch items (b)-(e) use all pairs.
- Estimators. paircov = tools/swarm_evalsci FE2(ours).fit("paircov"): unit = pair, common shared covariate (none for
  B0), ours only (PRIMARY, as QUEUE.md says). alone+P = alone-cell FE with public connectors: B1..B31 and ALC from cells
  where the pair was alone on its benchmark (FE2(alone cells).fit("poolaware")), B0 from every cell (paircov B0),
  ours + the public arms of --public. Both are reported everywhere; decisions use paircov and flag disagreement.
- SE_h = max(1.2 x stratified pair bootstrap SE, CR1 pair-cluster sandwich SE, SD/sqrt(max(K,1))), where
  SD = max(SD_ARM[arm][section], FLOOR_SD[section]) (the QUEUE floors are never undercut; B15/B31 have no floor) and
  K = direct matches with REF: any condition for paircov and for B0, alone-alone for alone+P B1..B31/ALC.
- Screen (after 6 runs; provisional before): DROP if ALC > +3.5e-3 or any of B0/B1/B3 > +2 SE_h; FINALIST if ALC < 0
  and z <= -1.0; otherwise CONTINUE (+10 runs). Stop any arm at SE_h(ALC) <= 0.0015 or 40 runs.
- L15 calibration test (B0, alone+P = every cell co-sighted with REF or through connectors), threshold -3.2e-3:
  B0 <= -3.2e-3 and z <= -1 (z vs -3.2e-3) -> H_pub (run W20@-1.5 next, keep L15 on the continue track);
  B0 >= -3.2e-3 and z >= +1 -> H_cal (stop L15, m0 -2.0 stays, judge W20 on its own screen);
  otherwise UNDECIDED (two more L15 runs, then decide on the point estimate).
- W20 mechanism readout: B1 <= 0 supports widening; predictions H_pub -6.4, H_cal +0.4, R1 +6.0 (x1e-3).
- Switch cur_final only if (a) holdout ALC <= 0, (b) full-data z <= -1.64 (one finalist) / -1.8 (two), (c) benchmark-
  balanced ALC <= 0, (d) leave-one-benchmark-out max <= +0.001, (e) no budget worse than REF by > 2 SE_h, (f) local LOBO
  macro not worse than +0.002 unweighted (repo macro: bench-equal, k-equal) and reweighted (swarm_evalsci 'joint' live
  reweighting, k mixed at the live alone share). Ties (|ALC| < 0.001) go to the simpler model. Shown, not enforced
  (round-1 extras): pool-weighted <= +0.001, balanced coverage >= 0.85, no public benchmark locally worse than +0.005.

Compliance: anonymous IDs are opaque join keys; the holdout hash only partitions them. Everything printed is an
arm-level contrast or a global count; no pair is listed, no label or base rate is reconstructed, nothing feeds a model.
Public logs (aghasalim, doctawho42; MIT) are used only as connector arms: acknowledge them if a decision uses them.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import hashlib
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import swarm_evalsci as ev  # noqa: E402
import feedback as fb  # noqa: E402  (module attributes, so tests can repoint fb.FEEDBACK)

REF = ev.REF
SECTIONS = list(ev.SECTIONS)  # alc b0 b1 b3 b7 b15 b31
BUDGETS = ["b0", "b1", "b3", "b7", "b15", "b31"]
PUB_REC = ["pub_aghasalim_v1", "pub_aghasalim_v2", "pub_aghasalim_v3A", "pub_aghasalim_v3B", "pub_aghasalim_v3C",
           "pub_doctawho42_hiership"]

# ---- pre-registered constants: edit here only ------------------------------------------------------------------
FLOOR_SD = {"alc": 0.0147, "b0": 0.055, "b1": 0.0487, "b3": 0.0132, "b7": 0.0076,  # QUEUE.md round-3 amendment
            "b15": 0.0043, "b31": 0.0039}  # (2026-10-06 ~10:30 UTC): B0/B3/B7 raised to live SDs, B15/B31 added
SD_ARM = {  # adjudicator 3.1: per-pair SD of the realised matched contrast (Monte Carlo, S1, no group)
    "v3b_noM_n_m15": {"alc": 0.0110, "b0": 0.0349},   # L15
    "v5_w20_m20": {"alc": 0.0129, "b0": 0.0204},      # W20
    "v5_d06_m169": {"alc": 0.00405, "b0": 0.0134},    # d06
}
BOOT_INFLATE = 1.2
SCREEN_RUNS, DROP_ALC, DROP_BUDGETS, DROP_Z, FINALIST_Z = 6, 0.0035, ("b0", "b1", "b3"), 2.0, -1.0
STOP_SE, STOP_RUNS = 0.0015, 40
CALIB_ARM, CALIB_THR = "v3b_noM_n_m15", -0.0032
W20_ARM, W20_PRED = "v5_w20_m20", {"H_pub": -0.0064, "H_cal": 0.0004, "R1": 0.0060}
SWITCH_Z = {1: -1.64, 2: -1.8}
LOBO_MAX, BUDGET_Z, LOCAL_MAX, TIE = 0.001, 2.0, 0.002, 0.001
FRESH_FIRST_ID = 965689  # first upload after the round-5 snapshot QUEUE_20261006T135127Z (2026-10-07 00:01 UTC);
                         # every public connector log was fetched before it (2026-10-06 ~08:20 UTC)
FINAL_K, FINAL_RUNS = 60, 30  # round-6 amendment: no SE floor in (b)/(e) at K_full >= 60 at the final look;
                              # finalists = non-dropped arms with >= 30 runs (round-4 item 7)
PW_MAX, COVER_MIN, LOCAL_BENCH_MAX = 0.001, 0.85, 0.005  # round-1 extras, shown only
FLOOR_SD_B15B31 = {"b15": 0.0043, "b31": 0.0039}  # live matched per-pair SD, ours vs REF, any condition (49 dof)
ALIAS = {"v3b_noM_n_m20": "REF", "v3b_noM_n_m15": "L15", "v5_w20_m20": "W20", "v5_d06_m169": "d06",
         "v3b_noM_n_m25": "M25", "v4_mix25_m10": "MIX25", "v4_mix35_m10": "MIX35",
         "v6_m150_mu20_d10": "W20@-1.5", "v6_m250_mu25_d06": "WLA", "v6_m225_mu25_d14": "WLB",
         "v6_combo_w20": "W20+SCP", "v6_combo_ref": "REF+SCP", "v6_combo_l15": "L15+SCP",
         "v6_m150_mu20_d06": "W15d6", "v8_mxs": "MXs", "v8_mxs_combo": "MXs+SCP",
         "v6_combo_m150_mu20_d10": "W20@-1.5+SCP", "v6_combo_m150_mu20_d06": "W15d6+SCP",
         "v6_m150_mu16_d14": "W15m16", "v6_combo_m150_mu16_d14": "W15m16+SCP",
         "v6_combo_w20_c1": "W20+SCPc1", "v6_combo_m150_mu16_d14_c1": "W15m16+SCPc1"}
PLAN_NOTE = {"v5_d06_m169": "no further uploads (adj 3.3)", "v3b_noM_n_m25": "dropped (adj 3.3)",
             "v4_mix25_m10": "dropped (adj 3.3)", "v4_mix35_m10": "dropped (adj 3.3)"}
HISTORICAL = ("v1_", "v2_", "v3_", "v3b_noM_n_m10")
LOCAL_MAP = {  # csv basename -> (local reference model, {live arm: local model})
    "adj_wide.csv": ("base", {"v3b_noM_n_m15": "n15D", "v5_w20_m20": "w20_m20", "v5_w20_m15": "w20_m15",
                              "v6_m150_mu20_d10": "w20_m15"}),
    "ld_guard_1007.csv": ("base", {"v6_m250_mu25_d06": "m250_mu25_d06", "v6_m225_mu25_d14": "m225_mu25_d14"}),
    "ld_guard_1008.csv": ("base", {"v6_m150_mu20_d06": "m150_mu20_d06"}),
    "ld_guard_1010.csv": ("base", {"v6_m150_mu16_d14": "m150_mu16_d14"}),
    "hr_mix_guard.csv": ("base", {"v8_mxs": "mxs", "v8_mxs_combo": "mxs_scp"}),
    "scp_add_1008.csv": ("base", {"v6_combo_m150_mu20_d10": "m150_mu20_d10_scp",
                                  "v6_combo_m150_mu20_d06": "m150_mu20_d06_scp"}),
    "scp_add_1015.csv": ("base", {"v6_combo_m150_mu16_d14": "m150_mu16_d14_scp"}),
    "combo_v1.csv": ("base", {"v6_combo_ref": "SCP_combo", "v6_combo_w20": "w20_SCP", "v6_combo_l15": "l15_SCP"}),
    "pd1_priors.csv": ("base", {"v5_d06_m169": "d06_q25", "v4_mix25_m10": "mix25_m10"}),
    "paired_mix.csv": ("base", {"v4_mix25_m10": "mix25_m10", "v4_mix35_m10": "mix35_m10"}),
    "paired_damped.csv": ("n20D", {"v3b_noM_n_m15": "n15D", "v3b_noM_n_m10": "n10D", "v2_noM_m15": "n15",
                                   "v3_noM_t3_m15": "t15D"}),
}
EXTRA_FLOOR: dict = {}  # filled only by --floor-b15b31
HUB_ARM = "pub_aghasalim_v3A"  # the best-covered connector (adjudicator 1.3 hub)
DEFAULT_LOCAL = ["results/adj_wide.csv", "results/pd1_priors.csv", "results/paired_mix.csv",
                 "results/ld_guard_1007.csv", "results/combo_v1.csv", "results/ld_guard_1008.csv",
                 "results/hr_mix_guard.csv", "results/scp_add_1008.csv", "results/ld_guard_1010.csv",
                 "results/scp_add_1015.csv"]
EST_LABEL = {"paircov": "paircov", "alone": "alone+P"}
STALE_HOURS = 3.0  # typical upload -> feedback is ~1.3 h
BIND_MARK = {"boot": "b", "cr1": "c", "floor": "f"}


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
def wide_from(df: pd.DataFrame, arm_of: dict) -> pd.DataFrame:
    """swarm_evalsci.live_wide() on any long frame (same columns and condition rule), plus the holdout flag."""
    w = df.pivot_table(index=["submission", "subject_id", "benchmark_id"], columns="section",
                       values="score").reset_index()
    n_u = df[df.section == "alc"].set_index(["submission", "subject_id", "benchmark_id"])["n_unique"]
    w["n_unique"] = pd.MultiIndex.from_frame(w[["submission", "subject_id", "benchmark_id"]]).map(n_u.to_dict())
    w["arm"] = w["submission"].map(arm_of)
    w["key"] = w["subject_id"] + "|" + w["benchmark_id"]
    n_same = w.groupby(["submission", "benchmark_id"])["subject_id"].transform("nunique")
    w["cond"] = np.where(n_same > 1, "shared", "alone")
    w["pkey"] = w["key"] + "|" + w["cond"]
    w = w[w["arm"].notna()].copy()
    w["holdout"] = ev.holdout_mask(w).to_numpy()
    return w


def public_selection(spec: str):
    s = spec.strip().lower()
    if s in ("none", "0", ""):
        return None
    if s == "rec":
        return set(PUB_REC)
    if s == "all":
        return "all"
    return {a.strip() for a in spec.split(",") if a.strip()}


def load_data(public_spec: str, upto: int | None):
    """(ours-only wide frame, ours + public connectors wide frame, our arm map). The environment variable
    PAIEC_FEEDBACK_PUBLIC is ignored on purpose: the ours-only estimator must stay ours-only."""
    ours = fb.load(include_public="0")
    if upto is not None:
        ours = ours[ours["submission"].astype(int) <= upto]
    arm_of = fb.ledger_arms(include_public="0")
    w_ours = wide_from(ours, arm_of)
    sel = public_selection(public_spec)
    if sel is None:
        return w_ours, w_ours.copy(), arm_of, []
    pub = fb.load_public(sel)
    pmap = fb.public_arms(sel)
    w_pub = wide_from(pd.concat([ours, pub], ignore_index=True), {**arm_of, **pmap})
    return w_ours, w_pub, arm_of, sorted(set(pmap.values()))


def read_ledger() -> list[dict]:
    with (fb.FEEDBACK.parent / "ledger.csv").open() as fh:  # the file fb.ledger_arms reads
        return list(csvmod.DictReader(fh))


def in_flight(upto: int | None) -> list[str]:
    have = {f.stem for f in fb.FEEDBACK.glob("*.txt")} | {f.stem for f in fb.FEEDBACK.glob("*.json")}
    now = datetime.now(timezone.utc)
    out = []
    for r in read_ledger():
        if r["id"] in have or not r.get("arm") or (upto is not None and int(r["id"]) > upto):
            continue
        try:
            age = (now - datetime.strptime(r["uploaded_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
            hours = age.total_seconds() / 3600
        except ValueError:
            hours = float("nan")
        tag = ALIAS.get(r["arm"], r["arm"])
        stale = " STALE?" if hours > STALE_HOURS else ""
        out.append(f"{r['id']} {tag} ({hours:.1f} h, {r.get('status', '')}{stale})")
    return out


# ---------------------------------------------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------------------------------------------
def k_table(w: pd.DataFrame, ref: str = REF) -> pd.DataFrame:
    """Direct matches with REF (distinct pair keys): any condition, alone-alone, shared-shared; runs and pairs."""
    r = w[w.arm == ref]
    ref_any, ref_al, ref_sh = set(r.key), set(r.loc[r.cond == "alone", "key"]), set(r.loc[r.cond == "shared", "key"])
    rows = {}
    for arm, g in w.groupby("arm"):
        rows[arm] = {"K_any": len(set(g.key) & ref_any),
                     "K_alone": len(set(g.loc[g.cond == "alone", "key"]) & ref_al),
                     "K_shared": len(set(g.loc[g.cond == "shared", "key"]) & ref_sh),
                     "pairs": int(g.key.nunique()), "runs": int(g.submission.nunique())}
    return pd.DataFrame.from_dict(rows, orient="index")


def _boot(fe2: ev.FE2, how: str, reps: int, seed: int) -> np.ndarray:
    ev.RNG = np.random.default_rng(seed)  # FE2.boot draws from the module RNG: reseed for order-free results
    if reps <= 1:
        return np.full((max(reps, 0), len(fe2.arms), len(SECTIONS)), np.nan)
    return np.stack(fe2.boot(how, reps=reps))


def fit_paircov(w: pd.DataFrame, reps: int, seed: int):
    fe2 = ev.FE2(w)
    return fe2.fit("paircov")[SECTIONS], _boot(fe2, "paircov", reps, seed), fe2.arms


def fit_alone(w: pd.DataFrame, reps: int, seed: int):
    """Alone-cell FE (B1..B31, ALC) with B0 from every cell, exactly as live_evidence.export_draws builds it."""
    fe2 = ev.FE2(w)
    pc = fe2.fit("paircov")
    wa = w[w["cond"] == "alone"]
    bs = np.full((max(reps, 0), len(fe2.arms), len(SECTIONS)), np.nan)
    if REF in set(wa["arm"]):
        fa = ev.FE2(wa)
        point = fa.fit("poolaware").reindex(fe2.arms)[SECTIONS].copy()
        ba = _boot(fa, "poolaware", reps, seed)
        for i, a in enumerate(fe2.arms):
            if a in fa.arms and len(ba):
                bs[:, i, :] = ba[:, fa.arms.index(a), :]
    else:
        point = pd.DataFrame(np.nan, index=fe2.arms, columns=SECTIONS)
    point["b0"] = pc["b0"]
    bc = _boot(fe2, "paircov", reps, seed + 1)
    j0 = SECTIONS.index("b0")
    if len(bc):
        bs[:, :, j0] = bc[:, :, j0]
    return point, bs, fe2.arms


def cr1_section(w: pd.DataFrame, how: str, section: str, ref: str = REF) -> pd.Series:
    """CR1 pair-cluster sandwich SE of the dummy-variable OLS on raw sightings (unit = pair; paircov adds the shared
    covariate except at B0; 'alone' keeps alone cells except at B0). Port of live_evidence.fe_ols + sandwich."""
    d = w.dropna(subset=["arm", section])
    if how == "alone" and section != "b0":
        d = d[d["cond"] == "alone"]
    cov = (d["cond"] == "shared").to_numpy(float)[:, None] if (how == "paircov" and section != "b0") else None
    _, ui = np.unique(d["key"].to_numpy(), return_inverse=True)
    arms = sorted(d["arm"].unique())
    if ref not in arms:
        return pd.Series(dtype=float)
    ai = {a: i for i, a in enumerate(arms)}
    a_codes = d["arm"].map(ai).to_numpy()
    keep = pd.Series(a_codes).groupby(ui).nunique().to_numpy()[ui] > 1
    d, ui, a_codes = d[keep], ui[keep], a_codes[keep]
    cov = None if cov is None else cov[keep]
    if len(d) == 0:
        return pd.Series(dtype=float)
    ui = np.unique(ui, return_inverse=True)[1]
    B = np.zeros((ui.max() + 1, len(arms)))
    B[ui, a_codes] = 1
    C = (B.T @ B) > 0
    reach = np.zeros(len(arms), bool)
    reach[ai[ref]] = True
    while True:
        new = reach | C[reach].any(0)
        if (new == reach).all():
            break
        reach = new
    est = [a for a in arms if reach[ai[a]] and a != ref]
    if not est:  # no arm connected to REF (sparse subsets such as the fresh-pairs set)
        return pd.Series(dtype=float)
    m = d["arm"].isin(est + [ref]).to_numpy()
    d, ui = d[m], ui[m]
    cov = None if cov is None else cov[m]
    ui = np.unique(ui, return_inverse=True)[1]
    nu = int(ui.max()) + 1
    cols = [(d["arm"].to_numpy() == a).astype(float) for a in est] + ([] if cov is None else [cov[:, 0]])
    if not est or not cols:
        return pd.Series(dtype=float)
    X = np.column_stack(cols)
    y = d[section].to_numpy(float)
    cnt = np.bincount(ui, minlength=nu).astype(float)
    Xm = np.zeros((nu, X.shape[1]))
    np.add.at(Xm, ui, X)
    Xd = X - (Xm / cnt[:, None])[ui]
    yd = y - (np.bincount(ui, weights=y, minlength=nu) / cnt)[ui]
    ok = np.abs(Xd).sum(0) > 1e-12
    names = [n for n, o in zip(est + (["shared"] if cov is not None else []), ok) if o]
    Xo = Xd[:, ok]
    sol, *_ = np.linalg.lstsq(Xo, yd, rcond=None)
    resid = yd - Xo @ sol
    XtX_inv = np.linalg.pinv(Xo.T @ Xo)
    ids, inv = np.unique(d["key"].to_numpy(), return_inverse=True)
    G = len(ids)
    S = np.zeros((G, Xo.shape[1]))
    np.add.at(S, inv, Xo * resid[:, None])
    n, k = Xo.shape[0], Xo.shape[1] + nu
    V = G / max(G - 1, 1) * (n - 1) / max(n - k, 1) * XtX_inv @ (S.T @ S) @ XtX_inv
    se = pd.Series(np.sqrt(np.clip(np.diag(V), 0, None)), index=names)
    return se.drop("shared", errors="ignore")


def estimate(w: pd.DataFrame, how: str, reps: int, seed: int, kt: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Long table for one (subset, estimator): effect, bootstrap SE, CR1, floor, SE_h, which part binds, K."""
    if REF not in set(w["arm"]):
        return pd.DataFrame()
    point, bs, arms = (fit_paircov if how == "paircov" else fit_alone)(w, reps, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se_b = np.nanstd(bs, axis=0, ddof=1) if len(bs) > 1 else np.full((len(arms), len(SECTIONS)), np.nan)
        na = np.isnan(bs).mean(axis=0) if len(bs) else np.full((len(arms), len(SECTIONS)), np.nan)
    cr1 = {s: cr1_section(w, how, s) for s in SECTIONS}
    rows = []
    for i, arm in enumerate(arms):
        if arm == REF:
            continue
        public = str(arm).startswith("pub_")  # connector arms: kept for the hub check, no floor (K is ours only)
        kr = kt.loc[arm] if arm in kt.index else pd.Series({"K_any": 0, "K_alone": 0, "K_shared": 0})
        for j, s in enumerate(SECTIONS):
            e = float(point.loc[arm, s])
            k = int(kr["K_any"] if (how == "paircov" or s == "b0") else kr["K_alone"])
            sd = max(SD_ARM.get(arm, {}).get(s, 0.0), FLOOR_SD.get(s, 0.0), EXTRA_FLOOR.get(s, 0.0))
            parts = {"boot": BOOT_INFLATE * se_b[i, j] if np.isfinite(se_b[i, j]) else 0.0,
                     "cr1": float(cr1[s].get(arm, np.nan)) if arm in cr1[s].index else 0.0,
                     "floor": sd / math.sqrt(max(k, 1)) if (sd > 0 and not public) else 0.0}
            parts = {p: (v if np.isfinite(v) else 0.0) for p, v in parts.items()}
            seh = max(parts.values())
            rows.append({"subset": subset, "est": how, "arm": arm, "section": s, "effect": e,
                         "se_boot": se_b[i, j], "boot_na": na[i, j], "se_cr1": parts["cr1"], "floor": parts["floor"],
                         "se_h": seh if np.isfinite(e) else np.nan, "binds": max(parts, key=parts.get),
                         "z": e / seh if (np.isfinite(e) and seh > 0) else np.nan, "K": k,
                         "K_any": int(kr["K_any"]), "K_alone": int(kr["K_alone"]), "K_shared": int(kr["K_shared"])})
    return pd.DataFrame(rows)


def strata_weights(w: pd.DataFrame) -> dict:
    """Pool-weighted weights exactly as swarm_evalsci.robust_table: M_t pool-size estimate per benchmark stratum."""
    st = {}
    runs = sorted(w.submission.unique())
    for b, g in w.groupby("benchmark_id"):
        caps = g.groupby("key")["submission"].nunique()
        sizes = [x for x in (len(set(g.loc[g.submission == r, "key"])) for r in runs) if x]
        st[b] = ev.mt_mle(sizes, int(caps.size), n_max=40 * int(caps.size))[0]
    tot = sum(st.values())
    return {b: v / tot for b, v in st.items()}


def robustness(w_full: pd.DataFrame) -> dict:
    """Full-data primary-estimator checks: benchmark-balanced (+ coverage), pool-weighted, leave-one-benchmark-out."""
    bal = ev.FE2(w_full).fit("balanced")
    pw = ev.FE2(w_full, strata_weights=strata_weights(w_full)).fit("poolweighted")
    lobo = {}
    for i, b in enumerate(sorted(w_full["benchmark_id"].unique())):
        sub = w_full[w_full["benchmark_id"] != b]
        if REF in set(sub["arm"]):
            lobo[f"bench{i}"] = ev.FE2(sub).fit("paircov")["alc"]  # benchmarks stay unnamed
    lobo = pd.DataFrame(lobo)
    return {"bal": bal["alc"], "cover": bal["weight_covered"], "pw": pw["alc"],
            "lobo_max": lobo.max(axis=1), "lobo_min": lobo.min(axis=1), "lobo_n": lobo.notna().sum(axis=1)}


# ---------------------------------------------------------------------------------------------------------------
# local LOBO guard (switch item f)
# ---------------------------------------------------------------------------------------------------------------
def parse_local(specs: list[str] | None) -> list[tuple[str, str, dict]]:
    out = []
    for spec in (specs or DEFAULT_LOCAL):
        path, _, mapping = spec.partition(":")
        ref_model, amap = LOCAL_MAP.get(Path(path).name, ("base", {}))
        amap = dict(amap)
        if mapping:
            for kv in mapping.split(","):
                k, _, v = kv.partition("=")
                if k.strip() == "ref":
                    ref_model = v.strip()
                elif k.strip():
                    amap[k.strip()] = v.strip()
        out.append((path, ref_model, amap))
    return out


def local_bench_effects(d: pd.DataFrame, model: str, ref: str) -> pd.Series:
    """Per public benchmark, unweighted repo macro ingredients: mean over pairs within (bench, k), then k-equal."""
    idx = ["bench", "k", "run", "pair"]
    a = d[d.model == model].set_index(idx)["alc"]
    b = d[d.model == ref].set_index(idx)["alc"]
    common = a.index.intersection(b.index)
    cell = (a.loc[common] - b.loc[common]).groupby(level=[0, 1]).mean()
    return cell.groupby(level=0).mean()


def local_guard(specs, w_full: pd.DataFrame) -> tuple[dict, list[str]]:
    dist = ev.live_global_dist(w_full)  # REF B0 and pooled B31 histograms over distinct pairs (global summaries)
    k_mix = float((w_full["cond"] == "alone").mean())
    in_ledger = {r["arm"] for r in read_ledger() if r.get("arm")}
    res, notes = {}, []
    for path, ref_model, amap in specs:
        full = (REPO / path) if not Path(path).is_absolute() else Path(path)
        if not full.exists():
            notes.append(f"{path}: missing (job not finished?)")
            continue
        d = pd.read_csv(full)
        models = set(d["model"].unique())
        if ref_model not in models:
            notes.append(f"{path}: reference model {ref_model!r} absent")
            continue
        used = []
        for arm, model in amap.items():
            if arm in res or model not in models:
                continue
            unw, unw_se = ev.local_effect(str(full), model, ref_model)["alc"]
            wv = ev.replay_weights(str(full), ref_model, dist, "joint")
            rw, rw_se = ev.local_effect(str(full), model, ref_model, weights=wv, k_mix=k_mix)["alc"]
            per_b = local_bench_effects(d, model, ref_model)
            res[arm] = {"csv": Path(path).name, "model": model, "ref": ref_model, "unw": unw, "unw_se": unw_se,
                        "rw": rw, "rw_se": rw_se, "bench_max": float(per_b.max()), "ess": ev.ess(wv) / len(wv)}
            if arm in in_ledger:
                used.append(f"{ALIAS.get(arm, arm)}={model} {unw * 1e3:+.2f}±{unw_se * 1e3:.2f} / "
                            f"rw {rw * 1e3:+.2f}±{rw_se * 1e3:.2f} (ESS {res[arm]['ess']:.2f})")
        notes.append(f"{path} (ref {ref_model}, {d['run'].nunique()} runs, k {sorted(d['k'].unique().tolist())}): "
                     f"{'; '.join(used) if used else 'no new ledger arm mapped'}")
    return res, notes


# ---------------------------------------------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------------------------------------------
def row(t: pd.DataFrame, subset: str, est: str, arm: str, section: str) -> dict | None:
    m = t[(t.subset == subset) & (t.est == est) & (t.arm == arm) & (t.section == section)]
    if m.empty or not np.isfinite(m["effect"].iloc[0]):
        return None
    return m.iloc[0].to_dict()


def screen(t: pd.DataFrame, est: str, arm: str, subset: str = "select") -> tuple[str, str]:
    a = row(t, subset, est, arm, "alc")
    if a is None:
        return "n/a", "not identified vs REF"
    if a["effect"] > DROP_ALC:
        return "DROP", f"ALC {a['effect'] * 1e3:+.1f} > +3.5"
    bad = [(s, r["z"]) for s in DROP_BUDGETS if (r := row(t, subset, est, arm, s)) is not None and r["z"] > DROP_Z]
    if bad:
        return "DROP", ", ".join(f"{s.upper()} z {z:+.1f} > +2" for s, z in bad)
    if a["effect"] < 0 and a["z"] <= FINALIST_Z:
        return "FINALIST", f"ALC < 0 with z {a['z']:+.2f} <= -1"
    return "CONTINUE", f"ALC {a['effect'] * 1e3:+.1f} <= +3.5"


def calib(t: pd.DataFrame, arm: str | None = None) -> dict | None:
    arm = arm or CALIB_ARM
    r = row(t, "select", "alone", arm, "b0")
    if r is None:
        return None
    z = (r["effect"] - CALIB_THR) / r["se_h"]
    lean = "H_pub" if r["effect"] < CALIB_THR else "H_cal"
    if r["effect"] <= CALIB_THR and z <= -1:
        out, act = "H_pub", "run W20@-1.5 (tau_mu 2.0, m0 -1.5) next; keep L15 on the continue track"
    elif r["effect"] >= CALIB_THR and z >= 1:
        out, act = "H_cal", "stop L15 (m0 -2.0 stays); judge W20 on its own screen"
    else:
        out, act = "UNDECIDED", f"two more L15 runs, then decide on the point estimate (now leans {lean})"
    p = row(t, "select", "paircov", arm, "b0")
    return {"b0": r["effect"], "se": r["se_h"], "K": r["K"], "z": z, "out": out, "act": act, "binds": r["binds"],
            "pc": p}


def w20_readout(t: pd.DataFrame, arm: str | None = None) -> list[str]:
    arm = arm or W20_ARM
    lines = []
    for est in ("paircov", "alone"):
        r = row(t, "select", est, arm, "b1")
        if r is None:
            lines.append(f"{EST_LABEL[est]:8s} B1 n/a (no matched evidence yet)")
            continue
        zs = {h: (r["effect"] - p) / r["se_h"] for h, p in W20_PRED.items()}
        best = min(zs, key=lambda h: abs(zs[h]))
        verdict = "supports widening" if r["effect"] <= 0 else "does not support widening"
        lines.append(f"{EST_LABEL[est]:8s} B1 {r['effect'] * 1e3:+.1f} (SE_h {r['se_h'] * 1e3:.1f}, K {r['K']}): "
                     f"{verdict}; z vs H_pub {zs['H_pub']:+.1f}, H_cal {zs['H_cal']:+.1f}, R1 {zs['R1']:+.1f} "
                     f"(closest {best})")
    return lines


def z_nofloor(r: dict) -> float:
    """z with SE_h = max(1.2 x bootstrap, CR1), i.e. without the SD/sqrt(K) floor (round-6 amendment, final look)."""
    seb = BOOT_INFLATE * r["se_boot"] if np.isfinite(r["se_boot"]) else 0.0
    se = max(seb, r["se_cr1"] if np.isfinite(r["se_cr1"]) else 0.0)
    return r["effect"] / se if se > 0 else np.nan


def fresh_keys(w_ours: pd.DataFrame, w_pub: pd.DataFrame, first_id: int) -> set:
    """Round-5 confirmatory set: pairs whose first appearance in any of our runs is in run id >= first_id and that
    appear in no public connector log (all of which predate the round-5 snapshot)."""
    sub = pd.to_numeric(w_ours["submission"], errors="coerce")
    first = sub.groupby(w_ours["key"]).min()
    pub_keys = set(w_pub.loc[w_pub["submission"].astype(str).str.startswith("pub_"), "key"])
    return set(first.index[first >= first_id]) - pub_keys


def switch_check(t, arm, rob, loc, n_final, final_look: bool = False) -> tuple[list[tuple[str, bool | None, str]], dict]:
    thr = SWITCH_Z[2] if n_final >= 2 else SWITCH_Z[1]
    items = []
    h = row(t, "holdout", "paircov", arm, "alc")
    items.append(("a", None if h is None else bool(h["effect"] <= 0),
                  "holdout n/a" if h is None else f"holdout {h['effect'] * 1e3:+.1f} (K {h['K_any']})"))
    f = row(t, "all", "paircov", arm, "alc")
    # round-6 amendment: at the final look, an arm with K_full >= FINAL_K direct matches uses no floor in (b) and (e)
    nofloor = bool(final_look and f is not None and f["K_any"] >= FINAL_K)
    zf = lambda r: z_nofloor(r) if nofloor else r["z"]  # noqa: E731
    items.append(("b", None if f is None else bool(zf(f) <= thr),
                  "full n/a" if f is None else f"z {zf(f):+.2f} vs {thr}" + (" (no floor)" if nofloor else "")))
    bal, cov = rob["bal"].get(arm, np.nan), rob["cover"].get(arm, np.nan)
    items.append(("c", None if not np.isfinite(bal) else bool(bal <= 0),
                  f"balanced {bal * 1e3:+.1f} (cover {cov:.2f})" if np.isfinite(bal) else "balanced n/a"))
    lm = rob["lobo_max"].get(arm, np.nan)
    items.append(("d", None if not np.isfinite(lm) else bool(lm <= LOBO_MAX),
                  f"LOBO max {lm * 1e3:+.1f}" if np.isfinite(lm) else "LOBO n/a"))
    zs = {s: zf(r) for s in BUDGETS if (r := row(t, "all", "paircov", arm, s)) is not None}
    worst = max(zs, key=zs.get) if zs else None
    items.append(("e", None if worst is None else bool(zs[worst] <= BUDGET_Z),
                  f"worst {worst.upper()} z {zs[worst]:+.1f}" if worst else "budgets n/a"))
    g = loc.get(arm)
    items.append(("f", None if g is None else bool(g["unw"] <= LOCAL_MAX and g["rw"] <= LOCAL_MAX),
                  "local n/a" if g is None else f"local {g['unw'] * 1e3:+.2f}/{g['rw'] * 1e3:+.2f}"))
    extras = {"pw": rob["pw"].get(arm, np.nan), "cover": cov,
              "bench_max": g["bench_max"] if g is not None else np.nan, "full": f, "nofloor": nofloor,
              "z_b": zf(f) if f is not None else np.nan}
    return items, extras


# ---------------------------------------------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------------------------------------------
def cell(r: dict | None, width: int = 13) -> str:
    """effect(SE_h + letter of the binding SE part: b = 1.2 x bootstrap, c = CR1, f = SD/sqrt(K) floor), x1e-3."""
    if r is None:
        return f"{'n/a':>{width}}"
    return f"{r['effect'] * 1e3:+.1f}({r['se_h'] * 1e3:.1f}{BIND_MARK.get(r['binds'], '')})".rjust(width)


def arm_order(arms_with_data: list[str], args) -> list[str]:
    """Arms to report: --arms, else every ledger arm (most recent upload first) that has feedback or is waiting for
    its first feedback, without the historical ones unless --all-arms; L15, W20, d06 lead."""
    inv = {v: k for k, v in ALIAS.items()}
    if args.arms:
        want = [inv.get(a.strip(), a.strip()) for a in args.arms.split(",") if a.strip()]
        return [a for a in want if a != REF]
    ledger = [r for r in read_ledger() if r.get("arm") and (args.upto is None or int(r["id"]) <= args.upto)]
    seen = []
    for r in reversed(ledger):  # most recently uploaded first
        a = r["arm"]
        if a not in seen and a != REF:
            seen.append(a)
    if not args.all_arms:
        seen = [a for a in seen if not a.startswith(HISTORICAL)]
    first = [a for a in ("v3b_noM_n_m15", "v5_w20_m20", "v5_d06_m169") if a in seen]
    return first + [a for a in seen if a not in first]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--local", action="append")
    ap.add_argument("--arms")
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--public", default="rec")
    ap.add_argument("--finalists", type=int)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--floor-b15b31", action="store_true")
    ap.add_argument("--project")
    ap.add_argument("--final-look", action="store_true")
    ap.add_argument("--fresh-after", type=int)
    args = ap.parse_args(argv)
    t0, c0 = time.time(), time.process_time()
    EXTRA_FLOOR.clear()
    if args.floor_b15b31:
        EXTRA_FLOOR.update(FLOOR_SD_B15B31)
    pd.set_option("display.width", 250)

    w_ours, w_pub, arm_of, pub_arms = load_data(args.public, args.upto)
    if REF not in set(w_ours["arm"]):
        print(f"REF arm {REF} has no feedback in this snapshot ({w_ours['submission'].nunique()} logs): nothing to "
              f"compare. Change REF in tools/swarm_evalsci.py only together with QUEUE.md.")
        return 2
    subsets = {"select": lambda w: w[~w["holdout"]], "holdout": lambda w: w[w["holdout"]], "all": lambda w: w}
    tabs, kts = [], {}
    for si, (name, f) in enumerate(subsets.items()):
        wo, wp = f(w_ours).copy(), f(w_pub).copy()
        kts[name] = k_table(wo)
        tabs.append(estimate(wo, "paircov", args.reps, args.seed + 10 * si, kts[name], name))
        tabs.append(estimate(wp, "alone", args.reps, args.seed + 10 * si + 2, k_table(wp), name))  # same K for ours
    tabs = [x for x in tabs if len(x)]
    if not tabs:
        print("no arm besides REF has feedback in this snapshot: nothing to compare")
        return 2
    t = pd.concat(tabs, ignore_index=True)
    rob = robustness(w_ours)
    loc, loc_notes = local_guard(parse_local(args.local), w_ours)
    runs = w_ours.groupby("arm")["submission"].nunique()
    arms = arm_order(sorted(set(w_ours["arm"])), args)

    # ---- 0. header
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"== DAILY LIVE REPORT {now}  (tools/daily_report.py; rules: submissions/QUEUE.md + adjudicator 3.2)")
    ours_logs = w_ours["submission"].nunique()
    pub_logs = w_pub.loc[w_pub["submission"].str.startswith("pub_"), "submission"].nunique()
    print(f"data: ours {ours_logs} logs{'' if args.upto is None else f' (id <= {args.upto})'}, REF {int(runs.get(REF, 0))}"
          f" runs; public connectors {pub_logs} logs ({', '.join(a.replace('pub_', '') for a in pub_arms) or 'none'})")
    allw = w_pub
    print(f"pairs: {allw['key'].nunique()} distinct, {len(allw)} appearances ({(allw['cond'] == 'shared').mean():.0%} "
          f"shared); ours {w_ours['key'].nunique()} distinct, {len(w_ours)} appearances")
    drift = ev.drift_check(allw, verbose=False)
    print(f"drift: {drift['pairs_with_changed_eval_size']} pairs changed eval size; {drift['same_arm_b0_repeats']} "
          f"same-arm B0 repeats, max |change| {drift['same_arm_b0_max_abs_change']:.6f}"
          + ("   <-- SNAPSHOT CHANGED: stop and investigate" if drift["pairs_with_changed_eval_size"] or
             (drift["same_arm_b0_max_abs_change"] or 0) > 1e-9 else ""))
    hk = allw.drop_duplicates("key")
    ho, href = w_ours.drop_duplicates("key"), w_ours[w_ours.arm == REF]
    print(f"holdout sha256('{ev.HOLDOUT_SALT}|subject_<id>|benchmark_<id>')[:2] < '40': {int(hk.holdout.sum())}/{len(hk)}"
          f" distinct pairs ({hk.holdout.mean():.1%}), {allw.holdout.mean():.1%} of appearances; ours "
          f"{int(ho.holdout.sum())}/{len(ho)} pairs ({ho.holdout.mean():.1%}); REF appearances {href.holdout.mean():.1%} "
          f"({int(href.holdout.sum())}/{len(href)})")
    fl = in_flight(args.upto)
    print("in flight (ledger rows without feedback): " + ("; ".join(fl) if fl else "none"))
    print(f"bootstrap: {args.reps} stratified pair reps per fit (9 fits, seed {args.seed}); SE_h = max({BOOT_INFLATE}"
          f" x boot, CR1 pair-cluster, SD/sqrt(max(K,1))), SD = max(arm SD, QUEUE floor); B15/B31 "
          + ("floored at the live SDs (--floor-b15b31, NOT pre-registered)" if EXTRA_FLOOR else "have no floor"))

    # ---- 1. selection table
    print("\n== 1. SELECTION SET (holdout excluded). arm - REF, x1e-3: effect(SE_h, binding part b=1.2xboot c=CR1 "
          "f=floor); z = ALC/SE_h;\n   K = direct matches with REF any/alone/shared; paircov = ours only (PRIMARY), "
          "alone+P = alone cells + public connectors (B0 all cells)")
    hdr = f"{'alias':5s} {'arm':15s} {'runs':>4s} {'K a/al/sh':>9s}  {'est':8s}{'ALC':>13s} {'z':>5s}" + \
          "".join(f"{s.upper():>13s}" for s in BUDGETS)
    print(hdr)
    for arm in arms:
        kr = kts["select"].loc[arm] if arm in kts["select"].index else None
        kstr = f"{kr['K_any']}/{kr['K_alone']}/{kr['K_shared']}" if kr is not None else "0/0/0"
        lead0 = f"{ALIAS.get(arm, ''):5s} {arm:15s} {int(runs.get(arm, 0)):>4d} {kstr:>9s}  "
        if arm not in kts["all"].index:
            print(lead0 + "no feedback yet")
            continue
        for j, est in enumerate(("paircov", "alone")):
            r = row(t, "select", est, arm, "alc")
            z = f"{r['z']:+5.1f}" if r is not None else "  n/a"
            print((lead0 if j == 0 else " " * len(lead0)) + f"{EST_LABEL[est]:8s}{cell(r)} {z}" +
                  "".join(cell(row(t, "select", est, arm, s)) for s in BUDGETS))

    # ---- 2. holdout + full data + switch inputs
    n_final = args.finalists
    outcomes = {arm: screen(t, "paircov", arm) for arm in arms}
    if n_final is None and args.final_look:  # round-4 item 7: every non-dropped arm with >= FINAL_RUNS runs
        fin = [arm for arm, o in outcomes.items() if o[0] != "DROP" and runs.get(arm, 0) >= FINAL_RUNS]
        n_final = max(1, len(fin))
        print(f"\nfinal look: finalists (not dropped, >= {FINAL_RUNS} runs) = {len(fin)}: {', '.join(fin) or 'none'}; "
              f"(b)/(e) without the SD/sqrt(K) floor for arms with K_full >= {FINAL_K} (round-6 amendment). The QUEUE "
              f"rule says pass --finalists explicitly; check this count against the drop history in QUEUE.md.")
    if n_final is None:
        n_final = max(1, sum(o[0] == "FINALIST" and runs.get(arm, 0) >= SCREEN_RUNS for arm, o in outcomes.items()))
    thr = SWITCH_Z[2] if n_final >= 2 else SWITCH_Z[1]
    print(f"\n== 2. SWITCH CHECKLIST (primary paircov; (a) on the holdout set, (b)-(e) on all pairs; ALC x1e-3; "
          f"finalists {n_final} -> (b) z <= {thr};\n   checks: + pass, - fail, ? n/a (counts as not passed), ! round-1 "
          f"caveat (c: benchmark coverage < {COVER_MIN}; f: a public benchmark locally worse than "
          f"+{LOCAL_BENCH_MAX * 1e3:.0f}e-3)")
    print(f"{'alias':5s} {'arm':15s} {'(a) hold(SE_h)':>15s} {'K':>3s} {'(b) full(SE_h)':>15s} {'K':>3s} {'z':>6s} "
          f"{'alone+P full':>13s} {'(c) bal cover':>13s} {'pw':>6s} {'(d)LOBOmax':>10s} {'(e) worst z':>13s} "
          f"{'(f) unw/rw':>12s} {'bmax':>5s}  checks")

    def num(x, fmt="{:+.1f}", scale=1e3):
        return fmt.format(x * scale) if (x is not None and np.isfinite(x)) else "n/a"

    checks = {}
    for arm in arms:
        items, ex = switch_check(t, arm, rob, loc, n_final, args.final_look)
        checks[arm] = (items, ex)
        g = loc.get(arm)
        locstr = f"{num(g['unw'], '{:+.2f}')}/{num(g['rw'], '{:+.2f}')}" if g else "n/a"
        mark = {"c": "!" if (np.isfinite(ex["cover"]) and ex["cover"] < COVER_MIN) else "",
                "f": "!" if (g and g["bench_max"] > LOCAL_BENCH_MAX) else ""}
        flags = " ".join(k + ("+" if ok else "?" if ok is None else "-") + mark.get(k, "") for k, ok, _ in items)
        if arm not in kts["all"].index:
            print(f"{ALIAS.get(arm, ''):5s} {arm:15s} {'no feedback yet':>15s} {'':>3s} {'':>15s} {'':>3s} {'':>6s} "
                  f"{'':>13s} {'':>13s} {'':>6s} {'':>10s} {'':>13s} {locstr:>12s} "
                  f"{num(g['bench_max'] if g else None):>5s}  {flags}")
            continue
        h = row(t, "holdout", "paircov", arm, "alc")
        f = ex["full"]
        bal = rob["bal"].get(arm, np.nan)
        balstr = f"{num(bal)} {ex['cover']:.2f}" if np.isfinite(bal) else "n/a"
        zstr = num(ex["z_b"] if f else None, "{:+.2f}", 1) + ("n" if ex["nofloor"] else "")
        kh, kf = (h["K_any"] if h else 0), (f["K_any"] if f else 0)
        wz = items[4][2].replace("worst ", "")
        print(f"{ALIAS.get(arm, ''):5s} {arm:15s} {cell(h, 15)} {kh:>3d} {cell(f, 15)} {kf:>3d} {zstr:>6s} "
              f"{cell(row(t, 'all', 'alone', arm, 'alc'))} {balstr:>13s} {num(ex['pw']):>6s} "
              f"{num(rob['lobo_max'].get(arm, np.nan)):>10s} {wz:>13s} {locstr:>12s} "
              f"{num(g['bench_max'] if g else None):>5s}  {flags}")
    for n in loc_notes:
        print(f"  local guard: {n}")

    # ---- 3. rule tests
    print("\n== 3. RULE TESTS (selection set)")
    c = calib(t)
    if c is None:
        print(f"L15 calibration ({CALIB_ARM}, B0 alone+P vs {CALIB_THR * 1e3:+.1f}): n/a (no matched evidence)")
    else:
        pc = c["pc"]
        print(f"L15 calibration ({CALIB_ARM}, runs {int(runs.get(CALIB_ARM, 0))}): B0 {c['b0'] * 1e3:+.1f} "
              f"(SE_h {c['se'] * 1e3:.1f}, binds {c['binds']}, K_any {c['K']}), z vs {CALIB_THR * 1e3:+.1f} = "
              f"{c['z']:+.2f} -> {c['out']}: {c['act']}")
        if pc is not None:
            print(f"   ours-only paircov B0 {pc['effect'] * 1e3:+.1f} (SE_h {pc['se_h'] * 1e3:.1f}); predictions "
                  f"H_pub -13.1, H_cal +6.7; separation needs K_any ~12 (adjudicator 3.1)")
    hub = [(s, row(t, "select", "alone", HUB_ARM, s)) for s in ("alc", "b0", "b1", "b3", "b7")]
    if all(r is not None for _, r in hub):
        print(f"hub check (alone+P, selection; REF - {HUB_ARM.replace('pub_', '')}, SE = max(1.2 boot, CR1)): " +
              ", ".join(f"{s.upper()} {-r['effect'] * 1e3:+.1f} ({r['se_h'] * 1e3:.1f})" for s, r in hub) +
              "; round 2 (all pairs): ALC -1.7 (4.8), B1 -1.6 (9.9), B3 -2.6 (5.1). A drift here moves every alone+P "
              "contrast together.")
    print(f"W20 mechanism readout ({W20_ARM}, runs {int(runs.get(W20_ARM, 0))}; B1 <= 0 supports widening):")
    for line in w20_readout(t):
        print("   " + line)

    # ---- 4. decisions
    print("\n== 4. DECISIONS (screen on the selection set with the primary estimator; alone+P shown when it disagrees)")
    flight = in_flight(args.upto)
    for arm in arms:
        n = int(runs.get(arm, 0))
        n_fl = sum(1 for x in flight if x.split()[1] in (arm, ALIAS.get(arm, arm)) and "STALE" not in x)
        items, ex = checks[arm]
        note = f" [{PLAN_NOTE[arm]}]" if arm in PLAN_NOTE else ""
        tag = f"{ALIAS.get(arm, '-'):5s} {arm:15s}"
        if arm not in kts["all"].index:
            ids = [x.split()[0] for x in flight if x.split()[1] in (arm, ALIAS.get(arm, arm))]
            g = loc.get(arm)
            gs = (f"; local guard {g['unw'] * 1e3:+.2f}/{g['rw'] * 1e3:+.2f} (unw/rw) "
                  f"{'passes' if max(g['unw'], g['rw']) <= LOCAL_MAX else 'FAILS'} (f)") if g else ""
            print(f"{tag} awaiting first feedback (in flight: {', '.join(ids) or 'none'}){gs}{note}")
            continue
        out, why = outcomes[arm]
        out2, why2 = screen(t, "alone", arm)
        a = row(t, "select", "paircov", arm, "alc")
        prov = (f" provisional ({n}/{SCREEN_RUNS} runs" + (f", +{n_fl} in flight" if n_fl else "") + ")"
                if n < SCREEN_RUNS else (f" ({n} runs, +{n_fl} in flight)" if n_fl else f" ({n} runs)"))
        dis = f"; alone+P says {out2} ({why2})" if out2 != out else ""
        stop = ""
        if a is not None and (a["se_h"] <= STOP_SE or n >= STOP_RUNS):
            stop = f"; STOP rule reached (SE_h {a['se_h'] * 1e3:.2f}, {n} runs)"
        fails = [k for k, ok, _ in items if ok is not True]
        sw = "SWITCH-ELIGIBLE" if not fails else "switch no (fails " + ",".join(fails) + ")"
        if not fails and ex["full"] is not None and abs(ex["full"]["effect"]) < TIE:
            sw += " but tie |ALC| < 1e-3 -> simpler model"
        extra = ""
        if arm == CALIB_ARM and c is not None:
            extra = f"; calib {c['out']}"
        if arm == W20_ARM:
            r1 = row(t, "select", "paircov", arm, "b1")
            extra = f"; B1 {r1['effect'] * 1e3:+.1f} ({'supports' if r1['effect'] <= 0 else 'against'} widening)" \
                if r1 is not None else ""
        az = f"{a['effect'] * 1e3:+.1f}±{a['se_h'] * 1e3:.1f} z {a['z']:+.2f}" if a is not None else "n/a"
        print(f"{tag} SCREEN {out}{prov}: {why} [ALC {az}]{dis}{extra}{stop}; {sw}{note}")

    # ---- 5. confirmatory fresh-pairs estimate (round-5 rule; reported at the final look, labelled confirmatory)
    if args.final_look or args.fresh_after is not None:
        fid = args.fresh_after if args.fresh_after is not None else FRESH_FIRST_ID
        fk = fresh_keys(w_ours, w_pub, fid)
        wf = w_ours[w_ours["key"].isin(fk)].copy()
        print(f"\n== 5. CONFIRMATORY FRESH-PAIRS ESTIMATE (round-5 rule; all pairs incl. holdout; paircov): pairs first "
              f"seen in our run id >= {fid} and in no public log: {len(fk)} pairs, {len(wf)} appearances, REF "
              f"{int((wf['arm'] == REF).sum())} appearances")
        tf = estimate(wf, "paircov", args.reps, args.seed + 50, k_table(wf), "fresh") if len(wf) else pd.DataFrame()
        for arm in arms:
            r = row(tf, "fresh", "paircov", arm, "alc") if len(tf) else None
            if r is None or not np.isfinite(r["effect"]):
                print(f"{ALIAS.get(arm, '-'):5s} {arm:15s} no fresh matched evidence")
                continue
            zn = z_nofloor(r)
            print(f"{ALIAS.get(arm, '-'):5s} {arm:15s} ALC {r['effect'] * 1e3:+.1f} (SE_h {r['se_h'] * 1e3:.1f}, z "
                  f"{r['z']:+.2f}; no floor z {zn:+.2f}; K {r['K_any']}) "
                  + " ".join(f"{s.upper()} {rr['effect'] * 1e3:+.1f}" for s in BUDGETS
                             if (rr := row(tf, "fresh", "paircov", arm, s)) is not None))

    print(f"\nruntime {time.time() - t0:.1f} s wall, {time.process_time() - c0:.1f} s CPU; reps {args.reps}")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv} ({len(t)} rows)")
    if args.project:
        project(args.project, w_ours)
    if args.validate:
        validate(args, w_ours, w_pub, t)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# planning aid (--project): how fast K grows and when the pre-registered rules can decide
# ---------------------------------------------------------------------------------------------------------------
def expected_k(st: pd.DataFrame, n_ref: int, n_arm: int) -> tuple[float, float]:
    """E[K_any], E[K_alone] from swarm_evalsci's stratum model: per run a subject of stratum b appears alone w.p.
    (1-d)/N and in any condition w.p. (1+d)/N (the same model as swarm_evalsci.expected_matches)."""
    ka = kal = 0.0
    for _, r in st.iterrows():
        N, d = max(float(r["N"]), float(r["seen"])), float(r["d"])
        pa, pn = (1 - d) / N, min((1 + d) / N, 1.0)
        kal += N * (1 - (1 - pa) ** n_ref) * (1 - (1 - pa) ** n_arm)
        ka += N * (1 - (1 - pn) ** n_ref) * (1 - (1 - pn) ** n_arm)
    return ka, kal


def project(spec: str, w_ours: pd.DataFrame) -> None:
    from scipy.stats import norm

    st = ev.stratum_model(w_ours)
    sel = 1.0 - float(w_ours.drop_duplicates("key")["holdout"].mean())
    kt = k_table(w_ours)
    print(f"\n== PROJECTION (stratum model: N {st['N'].tolist()}, sum {int(st['N'].sum())}; selection share "
          f"{sel:.2f}; SE_h taken at its floor, a lower bound; normal approximation)")
    for arm in (CALIB_ARM, W20_ARM):
        if arm in kt.index and REF in kt.index:
            ka, kal = expected_k(st, int(kt.loc[REF, "runs"]), int(kt.loc[arm, "runs"]))
            print(f"   model check {ALIAS.get(arm, arm)}: K_any {kt.loc[arm, 'K_any']} (model {ka:.1f}), K_alone "
                  f"{kt.loc[arm, 'K_alone']} (model {kal:.1f}) at REF {kt.loc[REF, 'runs']} / arm {kt.loc[arm, 'runs']} runs")
    sd_b0 = max(SD_ARM.get(CALIB_ARM, {}).get("b0", 0.0), FLOOR_SD["b0"])
    gap = (CALIB_THR - (-0.0131), 0.0067 - CALIB_THR)  # distance of H_pub / H_cal predictions from the threshold
    print(f"{'REF':>4s} {'arm':>4s} {'K_any':>6s} {'K_alone':>7s} {'K_sel':>6s} {'ALC floor':>9s} {'B0 floor':>8s} "
          f"{'P(calib z-call)':>15s} {'P(calib point ok)':>17s} {'P(drop|0)':>9s} {'P(drop|-3.1)':>12s} "
          f"{'P(final|-3.1)':>13s}")
    for part in spec.split(","):
        nr, _, na = part.partition(":")
        ka, kal = expected_k(st, int(nr), int(na))
        ks = ka * sel
        se_a = FLOOR_SD["alc"] / math.sqrt(max(ks, 1))
        se_0 = sd_b0 / math.sqrt(max(ks, 1))
        p_call = norm.cdf(min(gap) / se_0 - 1)
        p_point = norm.cdf(min(gap) / se_0)
        print(f"{int(nr):>4d} {int(na):>4d} {ka:>6.1f} {kal:>7.1f} {ks:>6.1f} {se_a * 1e3:>9.1f} {se_0 * 1e3:>8.1f} "
              f"{p_call:>15.2f} {p_point:>17.2f} {1 - norm.cdf(DROP_ALC / se_a):>9.2f} "
              f"{1 - norm.cdf((DROP_ALC + 0.0031) / se_a):>12.2f} {norm.cdf(0.0031 / se_a - 1):>13.2f}")
    S = SECTIONS
    r = w_ours[w_ours.arm == REF].groupby("key")[S].mean()
    acc = {s: [0.0, 0] for s in S}
    for arm, g in w_ours[w_ours.arm != REF].groupby("arm"):
        dlt = (g.groupby("key")[S].mean() - r).dropna()
        if len(dlt) >= 2:
            for s in S:
                acc[s][0] += (len(dlt) - 1) * float(dlt[s].var(ddof=1))
                acc[s][1] += len(dlt) - 1
    print("   live matched per-pair SD vs REF (ours, any condition, dof %d), x1e-3: " % acc["alc"][1] +
          ", ".join(f"{s.upper()} {math.sqrt(v / n) * 1e3:.1f}" + (f" (floor {FLOOR_SD[s] * 1e3:.1f})" if s in FLOOR_SD
                                                                   else "") for s, (v, n) in acc.items() if n))


# ---------------------------------------------------------------------------------------------------------------
# validation (--validate): numbers must match the existing tools exactly where the computation is the same
# ---------------------------------------------------------------------------------------------------------------
def _adj_module():
    import importlib.util
    path = REPO / "swarm" / "round2" / "adjudicator_sims" / "live_evidence.py"
    spec = importlib.util.spec_from_file_location("live_evidence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adj_name(a: str) -> str:
    m = {"ag_v1": "pub_aghasalim_v1", "ag_v2": "pub_aghasalim_v2", "ag_v3A": "pub_aghasalim_v3A",
         "ag_v3B": "pub_aghasalim_v3B", "ag_v3C": "pub_aghasalim_v3C", "dw_hiership": "pub_doctawho42_hiership",
         "dw_legacyP": "pub_doctawho42_legacyP"}
    return m.get(a, a)


def validate(args, w_ours: pd.DataFrame, w_pub: pd.DataFrame, t: pd.DataFrame) -> None:
    print("\n== VALIDATION")
    saved_env = os.environ.pop("PAIEC_FEEDBACK_PUBLIC", None)
    try:
        # V1 swarm_evalsci: same data, same primary estimator, same floors
        lw = ev.live_wide()
        lw = lw[lw["arm"].notna()].copy()
        if args.upto is not None:
            lw = lw[lw["submission"].astype(int) <= args.upto]
        p_tool = ev.FE2(lw).fit("paircov")[SECTIONS]
        p_mine = ev.FE2(w_ours).fit("paircov")[SECTIONS]
        print(f"V1 paircov point (all pairs, ours) vs swarm_evalsci FE2(live_wide()): max |diff| "
              f"{np.nanmax(np.abs((p_tool - p_mine.reindex(p_tool.index)).to_numpy())):.2e} over {p_tool.size} cells")
        le_tab = ev.live_estimates(lw, reps=50)
        le_tab = le_tab[(le_tab.how == "paircov") & (le_tab.arm != REF)]
        mine = t[(t.subset == "all") & (t.est == "paircov")].set_index(["arm", "section"])
        diffs = []
        for _, r in le_tab.iterrows():
            if (r.arm, r.section) in mine.index and r.section in FLOOR_SD and r.arm not in SD_ARM:
                diffs.append(abs(mine.loc[(r.arm, r.section), "floor"] -
                                 FLOOR_SD[r.section] / math.sqrt(max(r.k_direct, 1))))
        print(f"V1 floors sd/sqrt(K) vs swarm_evalsci.live_estimates (arms without an arm SD): max |diff| "
              f"{max(diffs) if diffs else float('nan'):.2e} over {len(diffs)} cells")
        # V2 holdout string and the adjudicator's export_draws(holdout='select') on the same data
        le = _adj_module()
        df, arm_of = le.load_all(True)
        if args.upto is not None:
            df = df[~df["submission"].str.fullmatch(r"\d+") | (pd.to_numeric(df["submission"], errors="coerce")
                                                                 .fillna(0) <= args.upto) |
                    (df["team"] != "ours")]
        aw = le.wide(df, arm_of)
        his = aw.apply(lambda r: hashlib.sha256(("brier-rabbit-holdout|" + r.subject_id + "|" + r.benchmark_id)
                                                .encode()).hexdigest()[:2] < "40", axis=1)
        mine_h = ev.holdout_mask(aw)
        print(f"V2 holdout mask vs the adjudicator's expression: {int((his == mine_h).sum())}/{len(aw)} rows agree")
        bare = aw.apply(lambda r: hashlib.sha256(("brier-rabbit-holdout|" + r.subject_id.replace("subject_", "") + "|" +
                                                  r.benchmark_id.replace("benchmark_", "")).encode())
                        .hexdigest()[:2] < "40", axis=1)
        print(f"   (bare-digit hashing would flip {(bare != mine_h).mean():.1%} of appearances: the prefixed form is pinned)")
        if args.upto is None and args.public == "all":
            ref_out = le.export_draws(reps=2, include_competitors=True, path=None, holdout="select")
            mine_sel = t[(t.subset == "select") & (t.est == "alone")]
            dmax, n = 0.0, 0
            for i, a in enumerate(ref_out["arms"]):
                a2 = _adj_name(str(a))
                for j, s in enumerate(ref_out["sections"]):
                    m = mine_sel[(mine_sel.arm == a2) & (mine_sel.section == s)]
                    if len(m) and np.isfinite(ref_out["alone_point"][i, j]):
                        dmax = max(dmax, abs(float(m["effect"].iloc[0]) - ref_out["alone_point"][i, j]))
                        n += 1
            print(f"V2 alone+P selection-set point vs export_draws(holdout='select')['alone_point']: max |diff| "
                  f"{dmax:.2e} over {n} cells")
        else:
            print("V2 export_draws comparison needs --public all and no --upto (the adjudicator's arm set)")
        # V3 adjudicator's published round-2 table (live_fe_all.csv) on his snapshot
        if args.upto == 963848 and args.public == "all":
            ref = pd.read_csv(REPO / "swarm" / "round2" / "adjudicator_sims" / "live_fe_all.csv")
            ref["arm"] = ref["arm"].map(_adj_name)
            mine_all = t[(t.subset == "all") & (t.est == "alone")].set_index(["arm", "section"])
            g = ref[ref.how == "alone"].set_index(["arm", "section"])
            common = g.index.intersection(mine_all.index)
            de = np.abs(g.loc[common, "effect"] - mine_all.loc[common, "effect"]).max()
            dc = np.abs(g.loc[common, "se_cl_pair"] - mine_all.loc[common, "se_cr1"]).max()
            dk = int((g.loc[common, "k_direct"] != mine_all.loc[common, "K"]).sum())
            rb = (mine_all.loc[common, "se_boot"] / g.loc[common, "se_boot"]).replace([np.inf, -np.inf], np.nan)
            print(f"V3 vs live_fe_all.csv (alone, {len(common)} cells): effect max |diff| {de:.2e}; CR1 max |diff| "
                  f"{dc:.2e}; K mismatches {dk}; boot SE ratio mine/his median {rb.median():.3f} "
                  f"(IQR {rb.quantile(.25):.3f}-{rb.quantile(.75):.3f}; his 300 reps, mine {args.reps})")
            # paircov with connectors, for the effect and CR1 columns
            wp = w_pub
            pc = ev.FE2(wp).fit("paircov")
            g2 = ref[ref.how == "paircov"].set_index(["arm", "section"])
            d2 = max(abs(pc.loc[a, s] - g2.loc[(a, s), "effect"]) for a, s in g2.index if a in pc.index)
            c2 = max(abs(cr1_section(wp, "paircov", s).get(a, np.nan) - g2.loc[(a, s), "se_cl_pair"])
                     for a, s in g2.index if a in pc.index)
            print(f"V3 vs live_fe_all.csv (paircov with connectors): effect max |diff| {d2:.2e}; CR1 max |diff| {c2:.2e}")
            npz = REPO / "swarm" / "round2" / "adjudicator_sims" / "draws_select.npz"
            if npz.exists():  # his export_draws(holdout='select') output on this snapshot
                z = np.load(npz, allow_pickle=True)
                sel = w_pub[~w_pub["holdout"]]
                pc_sel = ev.FE2(sel).fit("paircov")
                mine_sel = t[(t.subset == "select") & (t.est == "alone")].set_index(["arm", "section"])
                da = dp = 0.0
                n = 0
                ka = kn = 0
                kt_sel = k_table(sel)
                for i, a in enumerate(z["arms"]):
                    a2 = _adj_name(str(a))
                    for j, s in enumerate(z["sections"]):
                        if np.isfinite(z["alone_point"][i, j]) and (a2, s) in mine_sel.index:
                            da = max(da, abs(mine_sel.loc[(a2, s), "effect"] - z["alone_point"][i, j]))
                            n += 1
                        if np.isfinite(z["paircov_point"][i, j]) and a2 in pc_sel.index:
                            dp = max(dp, abs(pc_sel.loc[a2, s] - z["paircov_point"][i, j]))
                    if a2 in kt_sel.index:
                        ka += int(kt_sel.loc[a2, "K_alone"] != z["k_alone"][i])
                        kn += int(kt_sel.loc[a2, "K_any"] != z["k_any"][i])
                print(f"V3 vs draws_select.npz (export_draws holdout='select'): alone point max |diff| {da:.2e} over "
                      f"{n} cells; paircov-with-connectors point max |diff| {dp:.2e}; K_alone mismatches {ka}, "
                      f"K_any mismatches {kn}")
        else:
            print("V3 (adjudicator's round-2 table) needs --upto 963848 --public all")
        # V4 local guard against swarm_evalsci.local_effect (unweighted macro = mean of the per-bench effects)
        spec = parse_local(args.local)
        for path, ref_model, amap in spec:
            full = REPO / path
            if not full.exists():
                continue
            d = pd.read_csv(full)
            for arm, model in amap.items():
                if model in set(d.model):
                    m1 = ev.local_effect(str(full), model, ref_model)["alc"][0]
                    m2 = float(local_bench_effects(d, model, ref_model).mean())
                    print(f"V4 local macro {Path(path).name}:{model}: local_effect {m1 * 1e3:+.3f} vs per-bench mean "
                          f"{m2 * 1e3:+.3f} (diff {abs(m1 - m2):.1e})")
    finally:
        if saved_env is not None:
            os.environ["PAIEC_FEEDBACK_PUBLIC"] = saved_env


if __name__ == "__main__":
    sys.exit(main())
