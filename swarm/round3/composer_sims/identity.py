"""Identity battery for paiec/ext/combo.py (ComboModel) on recorded public LOBO predict calls.

Each switch setting is compared, call by call, with the class it must reproduce bit for bit:
  all off            -> paiec.hier.HierModel                       (5 level blocks: REF, W20, L15, MIX25, D06)
  subjcal            -> pd1_subjcal.SubjCalHier                    (attr_scale 1.56, attr_year_max 2026.4)
  pool_item          -> pool_cavity.CavityPoolModel                 (sigma_x 2.5)
  pool_sim (rho)     -> pdadapt_simpool.SimilarityPoolModel defaults at cur_final-type taus (0.75 / 0.6), and the
                        skeptic's P2 SimPoolCorr (shares) at D06
  pool_sim (rho None)-> SimilarityPoolModel with the same absolute taus
  subjcal+pool_item  -> class (SubjCalHier, CavityPoolModel)        (plain MRO composite, HierModel structure)
  subjcal+pool_sim   -> class (SubjCalHier, SimilarityPoolModel)
  all three          -> no reference class (new code path): on calls WITHOUT partner labels on the target item it must
                        equal subjcal+pool_sim exactly; on calls WITH them it differs by design (shift reported).
Calls: the skeptic's 7 recorded draws (swarm/round2/skeptic_plugins_sims/replay_checks.py REPLAYS, pair subsample).
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round3/composer_sims/identity.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "swarm/round2/skeptic_plugins_sims"))
from common import BASE, D06, MIX25, load_cache  # noqa: E402
from replay_checks import all_records, calls, run_seq  # noqa: E402

import numpy as np  # noqa: E402

from paiec.ext.combo import ComboModel  # noqa: E402
from paiec.ext.pd1_subjcal import SubjCalHier  # noqa: E402
from paiec.ext.pdadapt_simpool import SimilarityPoolModel  # noqa: E402
from paiec.ext.pool_cavity import CavityPoolModel, _full_key  # noqa: E402
from paiec.hier import HierModel, _sk  # noqa: E402

W20 = dict(BASE, tau_mu=2.0)
L15 = dict(BASE, m0=-1.5)
SC = {"attr_scale": 1.56, "attr_year_max": 2026.4}
CAV = {"sigma_x": 2.5}


class SubjCav(SubjCalHier, CavityPoolModel):
    pass


class SubjSim(SubjCalHier, SimilarityPoolModel):
    pass


class SimCorr(SimilarityPoolModel):
    """The skeptic's P2 test class (swarm/round2/skeptic_plugins_sims/simpool_xname.py)."""

    def __init__(self, abilities=None, params=None):
        super().__init__(abilities, params)
        total = self.p["tau_phi"] ** 2 + self.p["tau_delta"] ** 2
        self.p["tau_name"] = math.sqrt(0.45 * total)
        self.p["tau_h"] = math.sqrt(0.288 * total)


def combo(params, **sw):
    return lambda prior: ComboModel(prior, dict(params, **sw))


def ref(cls, params, **extra):
    return lambda prior: cls(prior, dict(params, **extra))


CASES = []
for name, lb in (("REF", BASE), ("W20", W20), ("L15", L15), ("MIX25", MIX25), ("D06", D06)):
    CASES.append((name, "all off = HierModel", combo(lb), ref(HierModel, lb)))
    CASES.append((name, "subjcal = SubjCalHier", combo(lb, subjcal=True, **SC), ref(SubjCalHier, lb, **SC)))
    CASES.append((name, "pool_item = CavityPoolModel", combo(lb, pool_item=True, **CAV), ref(CavityPoolModel, lb, **CAV)))
    if name != "D06":
        CASES.append((name, "pool_sim(rho) = SimPool defaults", combo(lb, pool_sim=True), ref(SimilarityPoolModel, lb)))
    CASES.append((name, "pool_sim(rho) = SimPoolCorr(P2)", combo(lb, pool_sim=True), ref(SimCorr, lb)))
    if name in ("REF", "D06"):
        CASES.append((name, "pool_sim(rho None) = SimPool abs", combo(lb, pool_sim=True, rho_name=None, rho_h=None),
                      ref(SimilarityPoolModel, lb)))
    CASES.append((name, "subjcal+pool_item = (SubjCal,Cavity)", combo(lb, subjcal=True, pool_item=True, **SC, **CAV),
                  ref(SubjCav, lb, **SC, **CAV)))
    CASES.append((name, "subjcal+pool_sim = (SubjCal,SimPool)", combo(lb, subjcal=True, pool_sim=True, **SC),
                  ref(SubjSim, lb, **SC) if name != "D06" else ref(SubjSim, lb, tau_name=math.sqrt(0.45 * 0.45),
                                                                     tau_h=math.sqrt(0.288 * 0.45), **SC)))


def partner_mask(cl):
    """True where another subject labelled the target item's full text (the cavity path)."""
    out = np.zeros(len(cl), bool)
    for j, (_, _, _, lab, inp) in enumerate(cl):
        s, it = inp
        b, ik, sk = str(it.get("benchmark_id", "")), _full_key(it), _sk(s)
        for (s2, it2), _ in lab:
            if str(it2.get("benchmark_id", "")) == b and _sk(s2) != sk and _full_key(it2) == ik:
                out[j] = True
                break
    return out


def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None  # level blocks to run, e.g. REF or W20,L15
    t0 = time.time()
    recs = all_records()
    _, priors = load_cache()
    cls_by_bench = {b: calls(r) for b, r in recs.items()}
    print(f"recorded {sum(len(v) for v in cls_by_bench.values())} calls in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for lbname, label, mk_a, mk_b in CASES:
        if only and lbname not in only:
            continue
        n, n_exact, worst = 0, 0, 0.0
        for bench, cl in cls_by_bench.items():
            a = run_seq(mk_a(priors[bench]), cl)
            b = run_seq(mk_b(priors[bench]), cl)
            d = np.abs(a - b)
            n += len(d)
            n_exact += int((d == 0).sum())
            worst = max(worst, float(d.max()))
        rows.append((lbname, label, n, n_exact, worst))
        print(f"{lbname:6s} {label:40s} n={n} bit-exact={n_exact} max|diff|={worst:.1e}", flush=True)
    # all three on: equal to subjcal+pool_sim off the cavity path, different (by design) on it
    for lbname, lb in (("REF", BASE), ("W20", W20), ("D06", D06)):
        if only and lbname not in only:
            continue
        n_off, ex_off, w_off, n_on, shifts = 0, 0, 0.0, 0, []
        for bench, cl in cls_by_bench.items():
            mask = partner_mask(cl)
            a = run_seq(ComboModel(priors[bench], dict(lb, subjcal=True, pool_item=True, pool_sim=True, **SC, **CAV)), cl)
            b = run_seq(SubjSim(priors[bench], dict(lb, **SC, tau_name=math.sqrt(0.45 * (lb["tau_phi"] ** 2 + lb["tau_delta"] ** 2)),
                                                   tau_h=math.sqrt(0.288 * (lb["tau_phi"] ** 2 + lb["tau_delta"] ** 2)))), cl)
            c = run_seq(SubjCav(priors[bench], dict(lb, **SC, **CAV)), cl)
            h = run_seq(HierModel(priors[bench], dict(lb)), cl)
            d = np.abs(a - b)
            n_off += int((~mask).sum())
            ex_off += int((d[~mask] == 0).sum())
            w_off = max(w_off, float(d[~mask].max()) if (~mask).any() else 0.0)
            n_on += int(mask.sum())
            lg = lambda x: np.log(x / (1 - x))
            if mask.any():
                shifts.append(np.c_[lg(a[mask]) - lg(b[mask]), lg(c[mask]) - lg(h[mask])])
        S = np.vstack(shifts) if shifts else np.zeros((0, 2))
        cr = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1]) if len(S) > 2 else float("nan")
        print(f"{lbname:6s} all three: off-cavity n={n_off} bit-exact={ex_off} max|diff|={w_off:.1e} vs subjcal+pool_sim; "
              f"cavity calls n={n_on}: logit shift (all3 - subjcal+sim) mean {S[:, 0].mean():+.3f} sd {S[:, 0].std():.3f}, "
              f"(subjcal+cav - HierModel) mean {S[:, 1].mean():+.3f} sd {S[:, 1].std():.3f}, corr {cr:.3f}", flush=True)
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
