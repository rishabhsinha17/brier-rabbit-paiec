"""The live decision pipeline on a simulated world (swarm round 6, statistician).

Timeline (UTC): state of 2026-10-06 13:50 (REF 10 runs, W20 4, L15 5, all simulated from the world), then
days 1-22 = 10-07..10-28 with 30 uploads each, days 23-24 = 10-29/10-30 (REF + the two best finalists).
Each day, in this order (as operated: daily_report, then swarm_live_design):
  1. pre-registered screens on the selection set (paircov, SE_h, round-3 floors) for every arm at >= 6 runs and
     every +10 runs: DROP (final) if ALC > +3.5e-3 or any of B0/B1/B3 z > +2;
  2. allocation of the day's 30 uploads by the policy:
       P1plan  the round-4 adaptive plan (ld_policy.P1, the validated in-sim proxy of tools/swarm_live_design.py:
               posterior from the 10-07 evidence, preposterior plan re-optimised every 3 days, <= 5 active arms,
               REF >= 20%, new arms get their 6 screening runs at once);
       P2stat  static rotation over REF + the five 10-07 arms (non-adaptive comparator);
     uploads to dropped arms go to REF;
  3. optional interim commit (end of day, before 10-28): every switch item on the current data with full-data
     z <= -2.5, holdout <= 0, and posterior P(leader best) >= 0.8 -> uploads go to REF + leader + runner-up.
Final look (after 10-30): the switch checklist on every non-dropped arm with >= 30 runs: (a) holdout ALC <= 0,
(b) full-data z <= -1.64 (one finalist) / -1.8 (two or more), (c) benchmark-balanced ALC <= 0, (d) leave-one-
benchmark-out max <= +0.001, (e) no budget z > 2, (f) local guard (passes for every shortlist arm: not simulated),
tie |ALC| < 1e-3 -> no switch; the passing arm with the lowest full-data ALC ships.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oc_core as oc  # noqa: E402
import ld_policy as lpol  # noqa: E402

LAST_ANY = lpol.LAST_ANY      # 22 = 10-28
FINAL_BOOT_REPS = 100         # final look: real cell-based stratified pair bootstrap (daily_report uses 500)
N_DAY = 30
UNIT_PROFILES = {             # budget effect profiles with ALC = 1 (B0 B1 B3 B7 B15 B31)
    "w20": np.array([8.5, 6.4, 3.1, 1.25, 0.35, 0.3]) / 3.10,      # W20's pre-registered H_pub prediction
    "flat": np.ones(6),
    "level": np.array([13.1, 6.0, 2.2, 0.6, 0.2, 0.1]) / 3.09,    # L15's H_pub prediction (B0-heavy)
}


class _St:
    def __init__(self, D):
        self.D = D

    @property
    def runs(self):
        return self.D.runs

    @property
    def runs_ref(self):
        return self.D.runs_ref


class Run:
    """What the round-4 policies read (R.A, R.st.runs, R.st.runs_ref, R.dropped, R.stopped, R.est, R.new_meas,
    R.apply_screens), on top of the simulated appearance log, with the REAL estimators for screens."""

    def __init__(self, world, data, sd_arm):
        self.w, self.D, self.A = world, data, world.A
        self.st = _St(data)
        self.sd_arm = np.asarray(sd_arm)
        self.dropped = np.zeros(self.A, bool)
        self.stopped = np.zeros(self.A, bool)
        self.screened_at = np.zeros(self.A, int)
        self.drop_day = np.full(self.A, -1)
        self.drop_why = [""] * self.A
        self.day = 0
        self._cache = {}
        self.past_match = self.matched()

    def matched(self):
        n = self.D.n
        seen = np.zeros((self.A + 1, self.w.P), bool)
        arm = self.D.arm[:n].astype(int)
        m = arm < self.A
        seen[arm[m] + 1, self.D.pair[:n][m]] = True
        return seen[1:] & seen[0][None]

    def estimates(self, subset):
        key = (subset, self.D.n)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = oc.estimate(self.D, oc.subset_rows(self.D, subset))
        return self._cache[key]

    def est(self, a, subset="sel"):
        e = self.estimates(subset)
        v = e["est"][a, :4]
        k = int(e["K"][a])
        if not np.isfinite(v[0]):
            return np.full(4, np.nan), 0, self.sd_arm[a]
        return v, k, self.sd_arm[a]

    def new_meas(self, a):
        """Direct matched ALC contrast on selection pairs first matched after the simulation start (as ld_policy)."""
        n = self.D.n
        arm, pair, y = self.D.arm[:n], self.D.pair[:n], self.D.Y[:n, 0]
        P = self.w.P

        def pm(who):
            m = arm == who
            s = np.bincount(pair[m], weights=y[m], minlength=P)
            c = np.bincount(pair[m], minlength=P)
            return s, c
        sa, ca = pm(a)
        sr, cr = pm(-1)
        m = (ca > 0) & (cr > 0) & ~self.w.holdout & ~self.past_match[a]
        k = int(m.sum())
        if k < 3:
            return None
        x = sa[m] / ca[m] - sr[m] / cr[m]
        return float(x.mean()), max(float(x.var(ddof=1)), (0.7 * self.sd_arm[a, 0]) ** 2) / k

    def apply_screens(self):
        for a in range(self.A):
            if self.dropped[a] or self.D.runs[a] < oc.SCREEN_RUNS:
                continue
            due = self.screened_at[a] == 0 or self.D.runs[a] >= self.screened_at[a] + 10
            if not due:
                continue
            e = self.estimates("sel")
            out = oc.screen_one(e["est"][a], e["z"][a])
            self.screened_at[a] = self.D.runs[a]
            if out == "drop":
                self.dropped[a] = True
                self.drop_day[a] = self.day
                why = []
                if e["est"][a, 0] > oc.DROP_ALC:
                    why.append("alc")
                why += [s for s in ("b0", "b1", "b3") if e["z"][a, oc.SI[s]] > oc.DROP_Z]
                self.drop_why[a] = "+".join(why)


def commit_check(R: Run, post, rng, ndraw=400):
    """Round-4 amendment item 6 on the current data: returns (leader, runner_up) or None."""
    D = R.D
    live = [a for a in range(R.A) if not R.dropped[a] and D.runs[a] > 0]
    if not live:
        return None
    es = R.estimates("sel")                     # cheap precheck: full-data z <= -2.5 needs a strong selection z
    if np.nanmin(np.where(np.isin(np.arange(R.A), live), es["z"][:, 0], np.nan)) > -1.8:
        return None
    ea = R.estimates("all")
    order = sorted(live, key=lambda a: ea["est"][a, 0] if np.isfinite(ea["est"][a, 0]) else np.inf)
    lead = order[0]
    if not (np.isfinite(ea["z"][lead, 0]) and ea["z"][lead, 0] <= -2.5):
        return None
    if not np.all(np.nan_to_num(ea["z"][lead, 1:], nan=99.0) <= oc.DROP_Z):
        return None
    eh = R.estimates("hold")
    if not (np.isfinite(eh["est"][lead, 0]) and eh["est"][lead, 0] <= 0):
        return None
    bal, lm = oc.balanced_and_lobo(D, [lead])[lead]
    if not (bal <= 0 and lm <= oc.LOBO_MAX):
        return None
    if post is not None:
        Dr = post.draws(ndraw, rng)[:, :, 0]
        cand = [a for a in range(R.A) if not R.dropped[a]]
        X = np.concatenate([Dr[:, cand], np.zeros((len(Dr), 1))], axis=1)
        pb = np.bincount(np.argmin(X, axis=1), minlength=len(cand) + 1) / len(Dr)
        if pb[cand.index(lead)] < 0.8:
            return None
    return lead, (order[1] if len(order) > 1 else None)


def make_policy(name, S, seed):
    if name.startswith("P1"):
        post = lpol.SimPosterior(S["w"], S["means"], S["covs"])
        nd = 200 if name == "P1fast" else 400
        return lpol.P1(post, S["sd_arm"], ndraw=nd, seed=seed), post
    if name == "P2stat":
        return lpol.P2(range(5)), lpol.SimPosterior(S["w"], S["means"], S["covs"])
    raise ValueError(name)


def simulate(world, S, policy: str, rng, *, init_runs=None, init_ref=None, commit=True, seed=0):
    init_runs = init_runs if init_runs is not None else {0: S["runs_now"]["W20"], 1: S["runs_now"]["L15"]}
    init_ref = init_ref if init_ref is not None else S["ref_runs_now"]
    D = oc.Data(world)
    for _ in range(init_ref):
        D.upload(-1, rng)
    for a, k in init_runs.items():
        for _ in range(k):
            D.upload(a, rng)
    R = Run(world, D, S["sd_arm"])
    pol, post = make_policy(policy, S, seed)
    committed, commit_day = None, -1
    crng = np.random.default_rng(seed + 99991)
    for day in range(1, LAST_ANY + 3):
        R.day = day
        if committed is not None and day <= LAST_ANY:
            R.apply_screens()
            members = [a for a in committed if a is not None and not R.dropped[a]]
            ups = lpol.rotation(N_DAY, [-1] + members)
        elif committed is not None:
            R.apply_screens()
            fin = [a for a in range(R.A) if not R.dropped[a] and D.runs[a] >= oc.FINALIST_RUNS]
            fin = sorted(fin, key=lambda a: R.est(a, "sel")[0][0])[:2]
            ups = lpol.rotation(N_DAY, [-1] + fin) if fin else [-1] * N_DAY
        else:
            ups = pol.plan(R, day)
        for a in list(ups)[:N_DAY]:
            if a >= 0 and (R.dropped[a] or R.stopped[a]):
                a = -1
            D.upload(int(a), rng)
        if commit and committed is None and day < LAST_ANY:
            if policy.startswith("P2"):
                idx, zz, vv = [], [], []
                for a in range(R.A):
                    mv = R.new_meas(a)
                    if mv is not None:
                        idx.append(a)
                        zz.append(mv[0])
                        vv.append(mv[1])
                post.update(idx, np.array(zz), np.array(vv))
            c = commit_check(R, post, crng)
            if c is not None:
                committed, commit_day = c, day
    R.day = LAST_ANY + 3
    R.apply_screens()
    eligible = [a for a in range(R.A) if not R.dropped[a] and D.runs[a] >= oc.FINALIST_RUNS]
    ea = oc.estimate_exact(D, oc.subset_rows(D, "all"), reps=FINAL_BOOT_REPS, seed=seed)
    eh = oc.estimate(D, oc.subset_rows(D, "hold"))
    ship, items = oc.switch_decision(D, eligible, len(eligible), est_all=ea, est_hold=eh)
    es = R.estimates("sel")
    return {"ship": int(ship), "eligible": eligible, "n_final": len(eligible),
            "runs": D.runs.tolist(), "runs_ref": int(D.runs_ref), "dropped": R.dropped.tolist(),
            "drop_day": R.drop_day.tolist(), "drop_why": R.drop_why, "commit_day": commit_day,
            "commit": None if committed is None else [int(x) if x is not None else -1 for x in committed],
            "truth": world.truth[:, 0].tolist(), "truth_bal": world.truth_bal[:, 0].tolist(),
            "full_alc": ea["est"][:, 0].tolist(), "full_z": ea["z"][:, 0].tolist(), "K_full": ea["K"].tolist(),
            "hold_alc": eh["est"][:, 0].tolist(), "sel_alc": es["est"][:, 0].tolist(),
            "full_zmax_budget": np.nanmax(np.nan_to_num(ea["z"][:, 1:], nan=-99.0), axis=1).tolist(),
            "full_est": ea["est"].tolist(), "full_cr1": ea["cr1"].tolist(), "full_cr0": ea["cr0"].tolist(),
            "full_boot": ea["boot"].tolist(), "final_se": "exact",
            "hold_K": eh["K"].tolist(), "hold_se": eh["se"][:, 0].tolist(),
            "items": {int(a): it for a, it in items.items()}}
