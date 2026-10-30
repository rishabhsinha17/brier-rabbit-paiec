"""Where does W20 lose, by pair-level evidence state (k successes of n own labels)? Exact alone-pair calculus.

Uses the round-2 adjudicator / round-4 live-design machinery unchanged (imported): 220 public eligible pairs with LOBO
subject priors, pool regimes = transforms of the public pair rates (shift, spread, share of a ~6% mode), hidden-subject
offset d, two item-group brackets; the round-4 posterior over (pool, d, lam) from live_design_now.npz (arm-level
contrasts and one global B31 summary; nothing per pair). Everything here is pool-level math on public pairs.

For a prediction table q (pairs x states) and true rates r, at budget n:
  P(k|n)        = sum_p w_p Binom(k; n, r_p)                        (w = benchmark-equal pair weights)
  bias(k, n)    = E[q - r | k, n]
  gain_state    = P(k|n) bias(k,n)^2  = Brier reduction from the best additive correction per state (oracle)
  Bayes oracle  q*(k, n) = E[r | k, n] (pair-independent; ignores theta)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "swarm" / "round4" / "design_sims", ROOT / "swarm" / "round2" / "adjudicator_sims",
          ROOT / "swarm" / "round1" / "postdoc_priors_sims"):
    sys.path.insert(0, str(p))
import adj_calc as ac  # noqa: E402
import fastregime as fr  # noqa: E402
import regime_test as rt  # noqa: E402

BUD = (0, 1, 3, 7, 15, 31)
WB = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
KN = ac.KN
K = np.array([k for k, n in KN], float)
N = np.array([n for k, n in KN], float)
IDX = fr.IDX
W20 = {"m0": -2.0, "tau_mu": 2.0, "tau_delta": 1.0, "tau_phi": 0.5}
REF = {"m0": -2.0, "tau_mu": 1.2, "tau_delta": 1.0, "tau_phi": 0.5}


def setup():
    S = rt.setup()
    return S["el"][["bench", "r"]].copy(), np.asarray(S["th"], float), np.asarray(S["tv"], float)


def table(level: dict, th, tv, d: float, br: int, **extra):
    p = dict(rt.LIVE, **level, extra_V=float(br), **extra)
    return ac.predict_table(p, th + d, tv)


def expected(q, r, w):
    """Benchmark-weighted expected Brier per budget (dict) and ALC."""
    pmf = fr.pmf_table(r)
    sq = (q - r[:, None]) ** 2 + (r * (1 - r))[:, None]
    out = {n: float(w @ (pmf[:, IDX[n]] * sq[:, IDX[n]]).sum(1)) for n in BUD}
    out["alc"] = sum(WB[n] * out[n] for n in BUD)
    return out


def state_table(q, r, w):
    """Per (k, n): P, E[q], E[r], bias, oracle gain P*bias^2; plus Bayes-oracle regret per budget."""
    pmf = fr.pmf_table(r)
    A = w[:, None] * pmf                       # (P, 63)
    P = A.sum(0)
    Eq = (A * q).sum(0) / np.maximum(P, 1e-300)
    Er = (A * r[:, None]).sum(0) / np.maximum(P, 1e-300)
    bias = Eq - Er
    # Bayes oracle (pair-independent) regret: sum_p A (q - r)^2 - A (Er - r)^2
    reg = (A * ((q - r[:, None]) ** 2 - (Er[None, :] - r[:, None]) ** 2)).sum(0)
    return {"P": P, "Eq": Eq, "Er": Er, "bias": bias, "gain": P * bias ** 2, "regret": reg}


def posterior_cells(mass=0.99, max_cells=400):
    cache = HERE / "post_cells.json"
    if cache.exists():
        import json
        return [tuple(x) for x in json.loads(cache.read_text())]
    import ld_post as lp
    post = lp.Posterior(ROOT / "swarm" / "round4" / "design_sims" / "cache" / "live_design_now.npz")
    post.fit()
    Wpd = post.W.sum(2)
    flat = Wpd.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    n = int(min(np.searchsorted(cum, mass) + 1, max_cells))
    cells = [np.unravel_index(o, Wpd.shape) for o in order[:n]]
    wts = flat[order[:n]] / flat[order[:n]].sum()
    import json
    import ld_calc as lc
    out = [(post.G.pools[i], float(lc.DS[j]), float(wt)) for (i, j), wt in zip(cells, wts)]
    cache.write_text(json.dumps(out))
    return out


_TAB = {}


def table_cached(level: dict, th, tv, d: float, br: int, tag: str = "", **extra):
    key = (tag, tuple(sorted(level.items())), round(d, 3), br, tuple(sorted(extra.items())))
    if key not in _TAB:
        _TAB[key] = table(level, th, tv, d, br, **extra)
    return _TAB[key]


if __name__ == "__main__":
    el, th, tv = setup()
    w = fr.bench_weights(el)
    print(f"{len(el)} public pairs", flush=True)
    worlds = {"public (d 0)": [({"shift": 0.0, "scale": 1.0, "pi_low": 0.0}, 0.0, 1.0)]}
    worlds["live posterior"] = posterior_cells()
    print(f"live posterior cells: {len(worlds['live posterior'])}", flush=True)
    for name, cells in worlds.items():
        agg = None
        e_tot = {n: 0.0 for n in BUD}
        for pool, d, wt in cells:
            pool = dict(pool)
            pool.setdefault("mu_low", math.log(0.06 / 0.94))
            pool.setdefault("sd_low", 0.5)
            r = fr.transform(el, pool)
            for br in (0, 1):
                q = table_cached(W20, th, tv, d, br)
                st = state_table(q, r, w)
                e = expected(q, r, w)
                for n in BUD:
                    e_tot[n] += 0.5 * wt * e[n]
                if agg is None:
                    agg = {k: 0.5 * wt * v for k, v in st.items()}
                else:
                    for k2, v in st.items():
                        agg[k2] = agg[k2] + 0.5 * wt * v
        print(f"\n=== {name}: W20 expected Brier x1e-3:", {n: round(1e3 * v, 2) for n, v in e_tot.items()})
        print(" n  k   P(k|n)  E[q]   E[r]   bias   gain_x1e3  regret_vs_Bayes_x1e3")
        for n in (1, 3, 7, 15, 31):
            for i in IDX[n]:
                k = int(K[i])
                if agg["P"][i] < 0.01 and k > 2:
                    continue
                print(f"{n:2d} {k:2d}  {agg['P'][i]:.3f}  {agg['Eq'][i]:.3f}  {agg['Er'][i]:.3f}  {agg['bias'][i]:+.3f}"
                      f"  {1e3 * agg['gain'][i]:7.3f}  {1e3 * agg['regret'][i]:7.3f}")
            print(f"   budget {n}: oracle gain sum {1e3 * agg['gain'][IDX[n]].sum():.3f}, Bayes regret sum "
                  f"{1e3 * agg['regret'][IDX[n]].sum():.3f}")
        sys.stdout.flush()
