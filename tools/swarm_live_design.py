"""Adaptive live design for the level block: the next UTC day's uploads (swarm round 4, live-design postdoc).

Run it once a day after the evening batch has reported (and after tools/daily_report.py):

  cd <repo>
  R() { PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python "$@"; }
  R tools/swarm_live_design.py --refresh-only        # step 1 (30-100 s): joint bootstrap of the new evidence
  R tools/swarm_live_design.py                       # step 2 (about 20 s): plan tomorrow (UTC) from submissions/
  R tools/swarm_live_design.py --day 2026-10-08      # plan a given UTC day
  R tools/swarm_live_design.py --arm=-1.75,2.2,0.8   # also report the posterior of any level arm (m0,tau_mu,tau_delta)
  R tools/swarm_live_design.py --must=-2.0/1.6/1.4   # force an arm into the shortlist (use '=' before a negative m0)
  The joint bootstrap is refreshed only when our feedback or the modelled arm list changed (--refresh forces it).

What it does (library: swarm/round4/design_sims/, report: swarm/round4/live_design.md)
 1. Live evidence, SELECTION SET ONLY (the QUEUE.md hash holdout is never read): competitor connector arms through the
    alone+P estimator, our level arms through the primary paircov estimator (ours only), the global
    benchmark-balanced B31, all with one joint stratified pair bootstrap (ld_data).
 2. Posterior over hidden-pool regimes (shift x spread x near-zero share at 6%) x our subject-prior offset d x a
    discrepancy scale, by the round-2 adjudicator's exact alone-pair calculus plus a GP discrepancy over the level
    family (ld_post). Prints P(H_pub: d <= 0.85) and the pool summaries.
 3. Posterior ALC effect vs REF of every level arm on the grid m0 {-2.5..-1.0} x tau_mu {1.2,1.6,2.0,2.5} x
    tau_delta {0.6,1.0,1.4} (tau_phi 0.5) and of every live family arm; off-grid arms on request (--arm).
 4. Shortlist by greedy batch expected improvement (active live arms are always in), then the plan: FINAL run counts
    per arm and REF that minimise the expected true ALC of the arm the pre-registered switch checklist will ship on
    10-30 (finalist >= 30 runs; holdout <= 0; z <= -1.64 one finalist / -1.8 two or more; no B0/B1/B3 z > 2), with
    REF absorbing every upload the arms do not use (ld_design.optimise_plan; validated in ld_policy/run_sim).
 5. Tomorrow's slice of that plan (new arms first get their 6 screening runs), exact params and build commands.
Pre-registered rules stay in force: daily_report.py's screens drop arms (pass them with --drop), the switch is decided
only by its checklist, new arms freeze after 2026-10-28 23:59 UTC, 10-29/10-30 carry only REF and <= 2 finalists.

Compliance: arm-level contrasts and global summaries only; anonymous IDs are opaque join keys; the holdout is not
read; public logs (aghasalim, doctawho42; MIT) are connectors only. No upload, no network.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "swarm" / "round4" / "design_sims"
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(REPO / "tools"))
import ld_calc as lc  # noqa: E402
import ld_data as ldt  # noqa: E402
import ld_design as ldd  # noqa: E402
import ld_post as lp  # noqa: E402

FEEDBACK = REPO / "submissions" / "feedback"
LEDGER = REPO / "submissions" / "ledger.csv"
SRC = REPO / "submission_src"
CACHE = LIB / "cache"
LIVE_NPZ = CACHE / "live_design_now.npz"
FAMILY_FIXED = {"lam": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_theta": 0.5, "tau_M": 0.0, "nu_mu": 0,
                "damped_newton": True}
ALIAS = {"v3b_noM_n_m20": "REF", "v3b_noM_n_m15": "L15", "v5_w20_m20": "W20", "v3b_noM_n_m25": "M25",
         "v3b_noM_n_m10": "M10", "v5_d06_m169": "d06"}
# arms with no further uploads under QUEUE.md (round-2 adjudication) - their data still enter the posterior
PREREG_DROPPED = {"v3b_noM_n_m25", "v3b_noM_n_m10", "v5_d06_m169", "v4_mix25_m10", "v4_mix35_m10"}
FREEZE = date(2026, 10, 28)
LAST = date(2026, 10, 30)
DAILY = 30
SECS = ("alc", "b0", "b1", "b3")
SD_EXTRA = np.array([7.0e-3, 0.0, 15.0e-3, 6.0e-3])     # per-pair partner/group noise (matches live SDs)


# ---------------------------------------------------------------------------------------------------------------
# ledger state
# ---------------------------------------------------------------------------------------------------------------
def family_spec(params: dict, folder: Path | None = None):
    """Level-family spec of a build, or None. Strict: damped Newton stated explicitly, normal level prior (nu_mu 0),
    no shared level, no mixture, no plug-in class, subjects prior shipped (v1 names-prior and v2 undamped builds are
    NOT family arms even when their m0/tau values coincide)."""
    if "class" in params or params.get("mix_pi"):
        return None
    if params.get("damped_newton") is not True or params.get("nu_mu", 0) != 0 or params.get("tau_M", 0.0) != 0.0:
        return None
    for k, v in FAMILY_FIXED.items():
        if k in params and params[k] != v:
            return None
    if folder is not None and (not (folder / "subject_prior.json").exists() or (folder / "abilities.json").exists()):
        return None
    try:
        return {"m0": float(params["m0"]), "tau_mu": float(params["tau_mu"]),
                "tau_delta": float(params["tau_delta"]), "tau_phi": float(params.get("tau_phi", lc.TAU_PHI))}
    except KeyError:
        return None


def ledger_state():
    have = {f.stem for f in FEEDBACK.glob("*.txt")} | {f.stem for f in FEEDBACK.glob("*.json")}
    arms = {}
    with LEDGER.open() as fh:
        for r in csv.DictReader(fh):
            a = (r.get("arm") or "").strip()
            if not a:
                continue
            st = (r.get("status") or "")
            d = arms.setdefault(a, {"done": 0, "inflight": 0, "zip": r.get("zip", ""), "spec": None})
            if r["id"] in have:
                d["done"] += 1
            elif not st.lower().startswith(("dead", "fail")):
                d["inflight"] += 1
            stem = Path(r.get("zip", "")).stem
            pj = SRC / stem / "params.json"
            if d["spec"] is None and pj.exists():
                d["spec"] = family_spec(json.loads(pj.read_text()), pj.parent)
    return arms, sorted(have)


def spec_name(s):
    return lc.arm_name(s["m0"], s["tau_mu"], s["tau_delta"]) + ("" if abs(s["tau_phi"] - lc.TAU_PHI) < 1e-9
                                                                 else f"_p{s['tau_phi']:.1f}")


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default=None, help="UTC day to plan, YYYY-MM-DD (default: tomorrow UTC)")
    ap.add_argument("--uploads", type=int, default=DAILY)
    ap.add_argument("--reps", type=int, default=800, help="joint bootstrap reps when the evidence is refreshed "
                    "(800: about 30-60 s; 1500 for a report-grade covariance)")
    ap.add_argument("--refresh", action="store_true", help="force the joint bootstrap refresh")
    ap.add_argument("--refresh-only", action="store_true", help="refresh the live evidence and stop (run the plan as a "
                    "second call when the machine is loaded, so each call stays under 2 minutes)")
    ap.add_argument("--ndraw", type=int, default=3000)
    ap.add_argument("--shortlist", type=int, default=7)
    ap.add_argument("--max-active", type=int, default=5)
    ap.add_argument("--must", default="-1.5/2.0/1.0", help="comma list of 'm0/tau_mu/tau_delta' triples forced into "
                    "the shortlist (default: W20@-1.5, the pre-registered conditional arm; the plan still decides its "
                    "runs). Active live arms are always in.")
    ap.add_argument("--drop", default="", help="comma list of ledger arms dropped by today's daily_report screen")
    ap.add_argument("--keep", default="", help="ledger arms to keep even if the screen replica below says DROP")
    ap.add_argument("--pending", default="", help="runs queued for today but not yet in the ledger, e.g. "
                    "'v3b_noM_n_m20:1,v5_w20_m20:1,v3b_noM_n_m15:1' (counted as in flight)")
    ap.add_argument("--arm", action="append", default=[], help="extra arm m0,tau_mu,tau_delta[,tau_phi] to report")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--json", default=None, help="also write the allocation as JSON here")
    ap.add_argument("--guard-job", default=None, help="write a compute-queue job spec (local LOBO guard, switch item f) "
                    "for every planned arm without one to this path (NOT enqueued; copy it to compute_queue/pending/)")
    a = ap.parse_args(argv)
    t0 = time.time()
    now = datetime.now(timezone.utc)
    day = date.fromisoformat(a.day) if a.day else (now + timedelta(days=1)).date()
    rng = np.random.default_rng(a.seed)

    # ---- 1. state and live evidence ----------------------------------------------------------------------------
    arms, have = ledger_state()
    for tok in [t for t in a.pending.split(",") if t.strip()]:
        nm, _, k = tok.partition(":")
        arms.setdefault(nm.strip(), {"done": 0, "inflight": 0, "zip": "", "spec": None})["inflight"] += int(k or 1)
    fam_live = {n: d for n, d in arms.items() if d["spec"] is not None}
    for n, d in fam_live.items():
        s = d["spec"]
        lc.LIVE_FAMILY[n] = (s["m0"], s["tau_mu"], s["tau_delta"], s["tau_phi"])
    our_arms = sorted(n for n in fam_live if n != lc.REF and fam_live[n]["done"] > 0) + list(lc.OTHER_OURS)
    key_now = {"feedback": sorted(have), "our_arms": our_arms}
    key_file = CACHE / "live_design_now.key.json"
    refresh = (a.refresh or not LIVE_NPZ.exists() or not key_file.exists()
               or json.loads(key_file.read_text()) != key_now)
    if refresh:
        print(f"refreshing live evidence ({a.reps} joint bootstrap reps) ...", flush=True)
        CACHE.mkdir(exist_ok=True)
        ldt.build(a.reps, LIVE_NPZ, our_arms=our_arms)
        key_file.write_text(json.dumps(key_now))
    if a.refresh_only:
        print(f"live evidence {'refreshed' if refresh else 'already current'}: {LIVE_NPZ} ({time.time() - t0:.1f}s)")
        return
    z = np.load(LIVE_NPZ, allow_pickle=True)
    stamp = ast.literal_eval(str(z["stamp"]))

    # ---- 2. posterior -------------------------------------------------------------------------------------------
    P = lp.Posterior(LIVE_NPZ).fit()
    m = P.marginals()
    sens = {}
    for key, kw in (("no B31 constraint", {"use_b31": False}), ("no competitor arms", {"use_comp": False})):
        sens[key] = lp.Posterior(LIVE_NPZ, **kw).fit().marginals()

    print(f"== LIVE DESIGN for UTC {day} (tools/swarm_live_design.py; {now:%Y-%m-%d %H:%M} UTC)")
    print(f"evidence: {stamp['n_logs_ours']} of our logs (max id {stamp['max_sub']}), selection set only, "
          f"{stamp['reps']} joint bootstrap reps; observations {P.nobs}; our modelled arms {', '.join(ALIAS.get(x, x) for x in stamp.get('our_arms', []))}")
    mp = m["map"]
    print(f"fit: MAP chi2 {mp['chi2']:.1f} + B31 {mp['chi2_b31']:.1f} on {mp['n_obs']} obs (lam {mp['lam']}); "
          f"discrepancy scale lam {{{', '.join(f'{k}: {v:.2f}' for k, v in m['lam'].items())}}}")
    print(f"pool: mean r {m['pool_mean_r']:.3f}, mean r(1-r) {m['pool_mean_rr']:.3f}, P(r<5%) {m['pool_p_lt05']:.3f}; "
          f"shift {{{', '.join(f'{k:+.2f}: {v:.2f}' for k, v in m['shift'].items())}}}")
    print(f"      spread {{{', '.join(f'{k}: {v:.2f}' for k, v in m['scale'].items())}}}; near-zero share "
          f"{{{', '.join(f'{k}: {v:.2f}' for k, v in m['pi_low'].items())}}}")
    hp = "P(d<=0.85)"
    print(f"subject-prior offset d: E {m['E[d]']:.2f}; P(H_pub: d <= 0.85) {m[hp]:.2f}  "
          f"[sensitivity: " + "; ".join(f"{k} {v[hp]:.2f}" for k, v in sens.items()) + "]")

    # ---- screen replica (pre-registered QUEUE.md screen on the selection set, paircov ours-only; SE_h approximated
    #      as max(1.2 x joint-bootstrap SE, floor/sqrt(K)); daily_report.py is the official decision) -------------
    import io
    import pandas as pd
    kt = pd.read_json(io.StringIO(str(z["kt"]))).set_index("index")
    full, ours, Fd = z["full"], list(z["ours"]), z["F"]
    auto_drop, screen_lines = set(), []
    keep = {x.strip() for x in a.keep.split(",") if x.strip()}
    for i, ln in enumerate(ours):
        if ln not in fam_live or ln in PREREG_DROPPED or ln not in kt.index:
            continue
        runs = int(arms[ln]["done"])
        k = int(kt.loc[ln, "K_any"])
        est = dict(zip(ldt.SECS, full[i]))
        bse = dict(zip(ldt.SECS, np.nanstd(Fd[:, i, :], axis=0, ddof=1)))
        z_ = {s_: est[s_] / max(1.2 * bse[s_], ldd.FLOOR.get(s_, 0.0) / math.sqrt(max(k, 1)))
              for s_ in ("alc", "b0", "b1", "b3")}
        out = ("drop" if (est["alc"] > ldd.DROP_ALC or any(z_[s_] > ldd.DROP_Z for s_ in ("b0", "b1", "b3")))
               else "finalist" if (est["alc"] < 0 and z_["alc"] <= ldd.FINALIST_Z) else "continue")
        prov = runs < ldd.SCREEN_RUNS
        screen_lines.append(f"{ALIAS.get(ln, ln)} runs {runs} K {k}: ALC {1e3 * est['alc']:+.1f} (z {z_['alc']:+.2f}), "
                            f"B0 z {z_['b0']:+.1f}, B1 z {z_['b1']:+.1f}, B3 z {z_['b3']:+.1f} -> "
                            f"{out.upper()}{' (provisional < 6 runs)' if prov else ''}")
        if out == "drop" and not prov and ln not in keep:
            auto_drop.add(ln)
    print("screen replica (selection set; official: tools/daily_report.py): " + (" | ".join(screen_lines) or "none"))

    # ---- 3. candidates and shortlist ----------------------------------------------------------------------------
    fam = lc.family_arms()
    live_spec = {}
    for n, d in fam_live.items():
        if n == lc.REF:
            continue
        nm = spec_name(d["spec"])
        if nm not in fam and n in PREREG_DROPPED:      # off-grid arms with no further uploads are not candidates
            continue
        fam.setdefault(nm, dict(d["spec"]))
        if nm in live_spec and live_spec[nm] != n:
            print(f"WARNING: ledger arms {live_spec[nm]} and {n} have the same level params ({nm}); the design uses "
                  f"{n}. Use one ledger arm name per params set so the FE pools them.")
        live_spec[nm] = n
    for t in a.arm:
        v = [float(x) for x in t.split(",")]
        s = {"m0": v[0], "tau_mu": v[1], "tau_delta": v[2], "tau_phi": v[3] if len(v) > 3 else lc.TAU_PHI}
        fam.setdefault(spec_name(s), s)
    names, cells, w, means, covs = P.arm_post(fam, secs=("alc",), max_cells=500)
    S_all, Dall = P.summarize(names, w, means, covs, secs=("alc",), ndraw=20000, seed=a.seed)
    dropped_names = {x.strip() for x in a.drop.split(",") if x.strip()} | PREREG_DROPPED | auto_drop
    active_live = [nm for nm, ln in live_spec.items()
                   if ln not in dropped_names and (arms[ln]["done"] + arms[ln]["inflight"]) > 0]
    must = active_live + [spec_name({"m0": float(x.split("/")[0]), "tau_mu": float(x.split("/")[1]),
                                     "tau_delta": float(x.split("/")[2]), "tau_phi": lc.TAU_PHI})
                          for x in a.must.split(",") if x.strip()]
    pool_names = [n for n in names if live_spec.get(n) not in dropped_names]
    idx = [names.index(n) for n in pool_names]
    short = [n for n, _ in ldd.greedy_qei(pool_names, Dall[:, idx, 0], must=[x for x in must if x in pool_names],
                                          k=a.shortlist)]
    cands = {n: fam[n] for n in short}
    _, cells, w, means, covs = P.arm_post(cands, secs=SECS, max_cells=400)
    S_short, D = P.summarize(short, w, means, covs, secs=SECS, ndraw=a.ndraw, seed=a.seed + 1)

    # ---- 4. plan ------------------------------------------------------------------------------------------------
    n_now, k_sel, est_sel, dropped = [], [], [], []
    for n in short:
        ln = live_spec.get(n)
        d = arms.get(ln, {"done": 0, "inflight": 0}) if ln else {"done": 0, "inflight": 0}
        n_now.append(d["done"] + d["inflight"])
        if ln and ln in ours and ln in kt.index:
            k_sel.append(int(kt.loc[ln, "K_any"]))
            est_sel.append(full[ours.index(ln)][[0, 1, 2, 3]])
        else:
            k_sel.append(0)
            est_sel.append(np.zeros(4))
        dropped.append(bool(ln and ln in dropped_names))
    n_now = np.array(n_now)
    ref = arms.get(lc.REF, {"done": 0, "inflight": 0})
    runs_ref = ref["done"] + ref["inflight"]
    days_any = max((FREEZE - day).days + 1, 0)
    days_all = max((LAST - day).days + 1, 0)
    remaining_any = DAILY * days_any
    remaining = DAILY * days_all
    sd = sd_arms(cands, short, cells, w)
    plan, ref_final, val, p_ship = ldd.optimise_plan(D, sd, n_now, runs_ref, np.array(k_sel), np.stack(est_sel),
                                                     np.array(dropped), remaining, remaining_any, days_any,
                                                     np.random.default_rng(a.seed + 2), max_active=a.max_active)
    # comparison plans
    iW20 = short.index(spec_name({"m0": -2.0, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5})) \
        if spec_name({"m0": -2.0, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5}) in short else None
    iL15 = short.index(spec_name({"m0": -1.5, "tau_mu": 1.2, "tau_delta": 1.0, "tau_phi": 0.5})) \
        if spec_name({"m0": -1.5, "tau_mu": 1.2, "tau_delta": 1.0, "tau_phi": 0.5}) in short else None
    comps = []
    if iW20 is not None and iL15 is not None:
        _, v_rot, _ = ldd.fixed_plan_value(D, sd, n_now, runs_ref, k_sel, np.stack(est_sel), [iW20, iL15], 1 / 3,
                                           remaining, np.random.default_rng(a.seed + 2))
        comps.append(("fixed rotation REF/W20/L15 (1/3 each) to 10-30", v_rot))
    if iW20 is not None:
        _, v_w, _ = ldd.fixed_plan_value(D, sd, n_now, runs_ref, k_sel, np.stack(est_sel), [iW20], 1 / 2,
                                         remaining, np.random.default_rng(a.seed + 2))
        comps.append(("REF/W20 only (1/2 each)", v_w))

    # ---- 5. tomorrow's slice ------------------------------------------------------------------------------------
    alloc = day_alloc(day, a.uploads, plan, n_now, ref_final, runs_ref, dropped, short, S_short)

    # ---- print ----------------------------------------------------------------------------------------------------
    print(f"\n== POSTERIOR: level arms vs REF (ALC x1e-3; mixture over regime cells; P(best) among the {len(names)} "
          f"arms + REF)")
    order = sorted(names, key=lambda n: S_all[n]["mean"]["alc"])
    show = [n for n in order[:12]] + [n for n in names if n in live_spec and n not in order[:12]]
    for n in show:
        s = S_all[n]
        tag = ALIAS.get(live_spec.get(n, ""), live_spec.get(n, ""))
        print(f"  {n:24s} {tag:8s} mean {1e3 * s['mean']['alc']:+5.2f}  sd {1e3 * s['sd']['alc']:4.2f}  "
              f"P(<0) {s['p_neg']:.2f}  P(best) {s['p_best']:.3f}")
    for t in a.arm:
        v = [float(x) for x in t.split(",")]
        s_ = {"m0": v[0], "tau_mu": v[1], "tau_delta": v[2], "tau_phi": v[3] if len(v) > 3 else lc.TAU_PHI}
        s = S_all[spec_name(s_)]
        print(f"  requested {spec_name(s_):24s} mean {1e3 * s['mean']['alc']:+5.2f} sd {1e3 * s['sd']['alc']:4.2f} "
              f"P(<0) {s['p_neg']:.2f}")
    print(f"  REF P(best) {S_all['_REF']['p_best']:.3f}")
    print(f"\n== SHORTLIST (greedy batch expected improvement; active live arms forced in) and PLAN to 10-30")
    ip_pub = [i for i, pl in enumerate(P.G.pools) if pl["shift"] == 0 and pl["scale"] == 1.0 and pl["pi_low"] == 0][0]
    print(f"  {'arm':24s} {'live':8s} {'ALC':>6s} {'sd':>5s} {'B0':>6s} {'B1':>6s} {'B3':>6s}  runs  -> final  "
          f"P(ship)  local-proxy")
    for i, n in enumerate(short):
        s = S_short[n]
        tag = ALIAS.get(live_spec.get(n, ""), live_spec.get(n, "") or "new")
        proxy = P.G.contrast(None, 0.0, fam=fam[n])[ip_pub].mean(0)[0]
        print(f"  {n:24s} {tag:8s} {1e3 * s['mean']['alc']:+6.2f} {1e3 * s['sd']['alc']:5.2f} "
              f"{1e3 * s['mean']['b0']:+6.1f} {1e3 * s['mean']['b1']:+6.1f} {1e3 * s['mean']['b3']:+6.1f}  "
              f"{n_now[i]:4d} -> {plan[i]:5d}  {p_ship[i]:.2f}     {1e3 * proxy:+5.2f}"
              f"{'  DROPPED' if dropped[i] else ''}")
    print("  (local-proxy = exact alone-pair calculus on the public LOBO pool, d 0: a screen for switch item (f); the "
          "switch itself needs the real local guard job, see --guard-job)")
    print(f"  REF{'':29s} runs {runs_ref} -> {ref_final}; P(no switch) {p_ship[-1]:.2f}")
    print(f"  preposterior expected true ALC of the shipped arm (x1e-3): this plan {1e3 * val:+.2f}" +
          "".join(f"; {k} {1e3 * v:+.2f}" for k, v in comps))
    print(f"\n== ALLOCATION for UTC {day}: {sum(alloc.values())} uploads")
    for n, k in sorted(alloc.items(), key=lambda t: -t[1]):
        if n == "REF":
            print(f"  {k:2d} x REF  v3b_noM_n_m0n20.zip (m0 -2.0, tau_mu 1.2, tau_delta 1.0, tau_phi 0.5)")
            continue
        s = fam[n]
        ln = live_spec.get(n)
        build = (f"existing {arms[ln]['zip']}" if ln else f"NEW build {build_name(s)}")
        print(f"  {k:2d} x {n:24s} m0 {s['m0']:+.2f} tau_mu {s['tau_mu']:.1f} tau_delta {s['tau_delta']:.1f} tau_phi "
              f"{s['tau_phi']:.1f}  [{build}]")
    new_arms = [n for n in alloc if n != "REF" and not live_spec.get(n)]
    if new_arms:
        print("  NEW arms: build, verify and zip each before its first upload; ledger arm name = build name:")
        for n in new_arms:
            nm = build_name(fam[n])
            print(f"    R tools/make_submission.py {nm} --params '{json.dumps(build_params(fam[n]))}'")
            print(f"    R tools/verify_shipped.py submission_src/{nm} && R tools/build_submission.py submission_src/{nm} "
                  f"--note 'live design arm {n}'")
    order = interleave(alloc)
    tag_of = {n: (ALIAS.get(live_spec.get(n, ""), "") or build_name(fam[n]) if n != "REF" else "REF") for n in alloc}
    print("  upload order (round-robin, REF spread evenly; batches of <= 4 in flight): "
          + ", ".join(tag_of[x] for x in order))
    with_runs = [i for i in range(len(short)) if not dropped[i] and n_now[i] > 0] or [0]
    lead = min(with_runs, key=lambda i: S_short[short[i]]["mean"]["alc"])
    pb = p_best_in(D, [i for i in range(len(short)) if not dropped[i] and (n_now[i] > 0 or plan[i] > n_now[i])])
    print(f"\n== COMMIT CHECK: leader (arms with live runs) {short[lead]} posterior P(best among planned shortlist arms + REF) "
          f"{pb.get(lead, 0.0):.2f} (commit needs >= 0.80 AND daily_report's switch checklist with z <= -2.5 "
          f"before 10-28)")
    print(f"runtime {time.time() - t0:.1f}s")
    if a.guard_job:
        extra = {"base": {"prior": "subjects", "damped_newton": True, "m0": -2.0}}
        guarded = {"m-1.50_mu1.2_d1.0", "m-2.00_mu2.0_d1.0", "m-1.50_mu2.0_d1.0", "m-2.50_mu2.5_d1.0"}  # adj_wide
        for i, n in enumerate(short):
            if plan[i] > n_now[i] and n not in guarded:
                sp = fam[n]
                extra[build_name(sp)[3:]] = {"prior": "subjects", "damped_newton": True, "m0": sp["m0"],
                                             "tau_mu": sp["tau_mu"], "tau_delta": sp["tau_delta"],
                                             "tau_phi": sp["tau_phi"]}
        job = {"name": f"ld_guard_{day:%m%d}", "priority": 6, "requested_by": "live design (swarm round 4)",
               "why": "Local LOBO guard (QUEUE switch item f) for the level arms the adaptive live design plans to run; "
                      "same draws as results/adj_wide.csv (runs 20, seed 0, k 1,2) so base chains exactly.",
               "args": ["experiments/paired.py", "--runs", "20", "--ks", "1,2", "--ref", "base", "--out",
                        f"results/ld_guard_{day:%m%d}.csv", "--extra", json.dumps(extra)]}
        Path(a.guard_job).write_text(json.dumps(job, indent=1))
        print(f"guard job spec written to {a.guard_job} ({len(extra) - 1} arms; not enqueued)")
    if a.json:
        Path(a.json).write_text(json.dumps({"day": str(day), "alloc": alloc,
                                            "params": {n: fam[n] for n in alloc if n != "REF"},
                                            "plan": {n: int(p) for n, p in zip(short, plan)}, "ref_final": ref_final,
                                            "value": val}, indent=1))


def interleave(alloc: dict) -> list:
    """Round-robin over arms (largest count first) with REF slots spread evenly through the day."""
    arms = {k: v for k, v in alloc.items() if k != "REF" and v > 0}
    seq = []
    left = dict(arms)
    while any(left.values()):
        for k in sorted(left, key=lambda k: -arms[k]):
            if left[k] > 0:
                seq.append(k)
                left[k] -= 1
    n_ref = alloc.get("REF", 0)
    total = len(seq) + n_ref
    out, r = [], 0
    for i in range(total):
        want_ref = round((i + 1) * n_ref / max(total, 1)) - r
        if (want_ref > 0 and r < n_ref) or not seq:
            out.append("REF")
            r += 1
        else:
            out.append(seq.pop(0))
    return out


def build_name(s):
    return f"v6_m{abs(s['m0']) * 100:03.0f}_mu{s['tau_mu'] * 10:02.0f}_d{s['tau_delta'] * 10:02.0f}"


def build_params(s):
    p = json.loads((SRC / "v3b_noM_n_m0n20" / "params.json").read_text())
    p.update({"m0": s["m0"], "tau_mu": s["tau_mu"], "tau_delta": s["tau_delta"], "tau_phi": s["tau_phi"]})
    return p


def p_best_in(D, idx):
    X = np.concatenate([D[:, idx, 0], np.zeros((len(D), 1))], axis=1)
    b = np.bincount(np.argmin(X, axis=1), minlength=len(idx) + 1) / len(D)
    return {i: float(b[j]) for j, i in enumerate(idx)}


def sd_arms(cands, short, cells, w, n=6, seed=5):
    """Per-pair SD of each arm's matched contrast: exact calculus on simulated pools of posterior cells, plus the
    partner/group noise that brings the calculus SD to the live matched SD (as in the validation simulator)."""
    import ld_sim as ls
    T = ls.Tables()
    rng = np.random.default_rng(seed)
    arms = [cands[k] for k in short]
    ok = []
    for s_ in arms:
        try:
            T.tix(s_, 0.0, 0)
            ok.append(True)
        except KeyError:                            # off-grid arm: no cached simulator tables
            ok.append(False)
    sub = [s_ for s_, o in zip(arms, ok) if o]
    res = np.tile(np.array([ldd.LIVE_SD[s_] for s_ in SECS]), (len(arms), 1))
    if sub:
        out = []
        for _ in range(n):
            c = rng.choice(len(w), p=w)
            Wd = ls.World(T, cells[c], np.zeros((len(sub), 4)), sub, rng)
            out.append(Wd.C.std(1, ddof=1))
        res[np.array(ok)] = np.sqrt(np.mean(np.square(out), 0) + SD_EXTRA[None] ** 2)
    return res


def day_alloc(day, n, plan, n_now, ref_final, runs_ref, dropped, short, S):
    """The day's slice: identical to ld_policy.slice_day (validated); 10-29/10-30: REF + <= 2 finalists."""
    import ld_policy as lpol
    if day > LAST:
        return {}
    if day > FREEZE:
        fin = [i for i in range(len(short)) if not dropped[i] and n_now[i] >= ldd.FINALIST_RUNS]
        fin = sorted(fin, key=lambda i: S[short[i]]["mean"]["alc"])[:2]
        out = {"REF": n - (n // 3) * len(fin)}
        for i in fin:
            out[short[i]] = n // 3
        return out
    need = {i: max(int(plan[i] - n_now[i]), 0) for i in range(len(short)) if not dropped[i]}
    days_left = max((FREEZE - day).days + 1, 1)
    lpol_day = lpol.LAST_ANY - days_left + 1          # map to the simulator's day index (same pacing)
    ups = lpol.slice_day(n, lpol_day, need, n_now, max(ref_final - runs_ref, 0))
    out = {}
    for u in ups:
        k = "REF" if u < 0 else short[u]
        out[k] = out.get(k, 0) + 1
    return out


if __name__ == "__main__":
    main()
