"""Parametric bootstrap of the round-2 regime test's delta chi-square (swarm round 6, statistician).

The statistic is exactly the adjudicator's (swarm/round2/adjudicator_sims/regime_test3.py / regime_subject.py):
52 hub-referenced alone-FE contrasts (13 arms x B0/B1/B3/B7 vs aghasalim v3A, draws_all.npz), covariance = joint
bootstrap covariance shrunk 50% to its diagonal + diag(model discrepancy^2) of the regime, plus the global
benchmark-balanced B31 term; total = chi2 + chi2_B31; delta = total(regime) - min over the grid.
Reference distributions (B replicates): data generated from a regime's prediction with the covariance the statistic
assumes (primary) or with the unshrunk bootstrap covariance (sensitivity), B31 from its stated SD; every replicate
is refitted over the same grid.
  prep-public            cache mu/md/B31 for the 315-point grid + named regimes (public-LOBO calibration)
  prep-cal I J           cache the subject-calibration cells (slope, offset) number I..J-1 of the 20 (regime_subject)
  boot [B]               run every test from the caches
Nothing live is read except the adjudicator's saved arm-level observations (draws_all.npz).
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADJ = HERE.parents[1] / "round2" / "adjudicator_sims"
sys.path.insert(0, str(ADJ))
import regime_test as rt  # noqa: E402
import regime_test3 as r3  # noqa: E402
import regime_variants as rv  # noqa: E402
import regime_subject as rs  # noqa: E402

CACHE = HERE / "cache"
SECS = ["b0", "b1", "b3", "b7"]
NAMED_KEYS = ["public", "R1 narrow s.8/-.5", "R1 narrow s.7/-.5", "R2 zi nz.35", "R2 zi nz.45", "R3 wide s1.3",
              "R3 wide s1.3/-.5"]
CAL_PAIRS = list(itertools.product(rs.SLOPES, rs.OFFSETS))      # 20 (slope, offset) cells


def data():
    v = rv.V(ADJ / "draws_all.npz", "ag_v3A", rt.TEST_ARMS + [rt.REF], SECS)
    ok = ~np.isnan(v.X).any(axis=1)
    S_raw = np.cov(v.X[ok].T)
    S = 0.5 * S_raw + 0.5 * np.diag(np.diag(S_raw))
    return v, S, S_raw


def pools_public():
    out = []
    for p in rv.grid_pools():
        out.append(("grid", p))
    for k in NAMED_KEYS:
        out.append((k, dict(r3.NAMED[k], _low_rate=0.015)))
    return out


def mu_block(v, pools):
    MU, MD, B31, names = [], [], [], []
    for name, pool in pools:
        pr = v.pred({k: x for k, x in pool.items() if not k.startswith("_")}, v.arms)
        mu, md = v.mu_md(pr, v.arms, SECS)
        MU.append(mu)
        MD.append(md)
        B31.append((pr[0]["_ref_b31"] + pr[1]["_ref_b31"]) / 2)
        names.append(name if name != "grid" else
                     f"sh{pool['shift']:+.2f} sc{pool['scale']} pi{pool['pi_low']}@{pool.get('_low_rate', 0.015)}")
    return np.array(MU), np.array(MD), np.array(B31), names


def prep_public(i0=0, i1=10_000):
    t0 = time.time()
    v, S, S_raw = data()
    pl = pools_public()
    MU, MD, B31, names = mu_block(v, pl[i0:i1])
    np.savez(CACHE / f"regime_public_{i0:04d}.npz", MU=MU, MD=MD, B31=B31, names=np.array(names), y=v.y, S=S,
             S_raw=S_raw)
    print(f"public calibration: regimes {i0}..{min(i1, len(pl)) - 1} of {len(pl)} in {time.time() - t0:.1f}s")


def load_public():
    fs = sorted(CACHE.glob("regime_public_*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    out = {k: np.concatenate([z[k] for z in zs]) for k in ("MU", "MD", "B31")}
    out["names"] = sum([list(z["names"]) for z in zs], [])
    for k in ("y", "S", "S_raw"):
        out[k] = zs[0][k]
    return out


def prep_cal(i0, i1):
    t0 = time.time()
    v, S, S_raw = data()
    base = v.S["tables"]
    pools = [(nm, dict(p, _low_rate=0.06)) for nm, p in rs.pools()]
    for ci in range(i0, min(i1, len(CAL_PAIRS))):
        s, d = CAL_PAIRS[ci]
        f = CACHE / f"regime_cal_{ci:02d}.npz"
        if f.exists():
            continue
        v.S["tables"] = rs.tables_map(v.S, s, d)
        MU, MD, B31, names = mu_block(v, pools)
        np.savez(f, MU=MU, MD=MD, B31=B31, names=np.array([f"s{s} d{d} {n}" for n in names]), slope=s, offset=d)
        v.S["tables"] = base
        print(f"  cal cell {ci} (slope {s}, offset {d}): {len(names)} pools, {time.time() - t0:.1f}s", flush=True)


def totals(Y, b31obs, MU, MD, B31, S):
    """total chi2 of every regime (rows of MU) for every replicate (rows of Y) -> (B, G)."""
    G = len(MU)
    out = np.empty((len(Y), G))
    sd31 = math.sqrt(r3.B31_BAL[1] ** 2 + r3.B31_MD ** 2)
    for g in range(G):
        C = S + np.diag(MD[g] ** 2)
        Ci = np.linalg.inv(C)
        E = Y - MU[g][None, :]
        out[:, g] = np.einsum("bi,ij,bj->b", E, Ci, E) + ((B31[g] - r3.B31_OFFSET - b31obs) / sd31) ** 2
    return out


def gen(mu, md, b31, S_gen, B, rng):
    L = np.linalg.cholesky(S_gen + np.diag(md ** 2))
    Y = mu[None, :] + rng.standard_normal((B, len(mu))) @ L.T
    b = b31 - r3.B31_OFFSET + rng.standard_normal(B) * math.sqrt(r3.B31_BAL[1] ** 2 + r3.B31_MD ** 2)
    return Y, b


def boot(B=2000, seed=20261006):
    rng = np.random.default_rng(seed)
    z = load_public()
    MU, MD, B31, names, y, S, S_raw = z["MU"], z["MD"], z["B31"], list(z["names"]), z["y"], z["S"], z["S_raw"]
    assert len(names) == len(pools_public()), f"public caches incomplete: {len(names)} of {len(pools_public())}"
    obs = totals(y[None, :], r3.B31_BAL[0], MU, MD, B31, S)[0]
    ng = sum(1 for n in names if n.startswith("sh"))
    gmin = obs[:ng].min()
    ib = int(np.argmin(obs[:ng]))
    print(f"observed: best grid point {names[ib]} total {gmin:.1f} on {len(y)} contrasts + B31 "
          f"(chi2 tail on 53 dof {chi2_sf(gmin, 53):.4f}; on 50 dof {chi2_sf(gmin, 50):.4f})")
    res = {}
    for gen_name, S_gen in (("shrunk (as the statistic)", S), ("unshrunk bootstrap", S_raw)):
        print(f"\n-- generating covariance: {gen_name}; {B} replicates per null")
        # (i) goodness of fit at the best grid point, accounting for the grid search
        Y, b = gen(MU[ib], MD[ib], B31[ib], S_gen, B, rng)
        T = totals(Y, b, MU[:ng], MD[:ng], B31[:ng], S)
        mins = T.min(1)
        p = (1 + (mins >= gmin).sum()) / (B + 1)
        print(f"  GOF, data from the best grid point: min-total median {np.median(mins):.1f}, 95% {np.quantile(mins, .95):.1f},"
              f" 99% {np.quantile(mins, .99):.1f}; observed {gmin:.1f} -> p = {p:.4f}")
        res[(gen_name, "gof_best")] = p
        # (ii) each named regime as the null: delta = total(regime) - min(grid U {regime})
        for k in NAMED_KEYS:
            j = names.index(k)
            d_obs = obs[j] - min(gmin, obs[j])
            Y, b = gen(MU[j], MD[j], B31[j], S_gen, B, rng)
            idx = list(range(ng)) + [j]
            T = totals(Y, b, MU[idx], MD[idx], B31[idx], S)
            dstar = T[:, -1] - T.min(1)
            p = (1 + (dstar >= d_obs).sum()) / (B + 1)
            print(f"  null {k:20s}: observed delta {d_obs:6.1f}; reference median {np.median(dstar):5.1f}, 95% "
                  f"{np.quantile(dstar, .95):5.1f}, 99% {np.quantile(dstar, .99):5.1f}, max {dstar.max():5.1f} -> p = {p:.4f}")
            res[(gen_name, k)] = (d_obs, p, float(np.quantile(dstar, .95)))
    # (iii) the subject-calibration (H_cal) fit, if cached
    cal_files = sorted(CACHE.glob("regime_cal_*.npz"))
    if len(cal_files) == len(CAL_PAIRS):
        boot_cal(cal_files, y, S, S_raw, B, rng)
    else:
        print(f"\n(subject-calibration caches: {len(cal_files)} of {len(CAL_PAIRS)}; run prep-cal first)")
    return res


def scaled(B=400, seed=20261007):
    """Misfit-calibrated variant (quasi-likelihood dispersion): the best grid point misfits (GOF p ~0.004), so scale the
    covariance by c = observed min-total / median min-total of data generated at the best point; the reference
    distribution of delta is scale-free, the observed delta is divided by c."""
    rng = np.random.default_rng(seed)
    z = load_public()
    MU, MD, B31, names, y, S = z["MU"], z["MD"], z["B31"], list(z["names"]), z["y"], z["S"]
    obs = totals(y[None, :], r3.B31_BAL[0], MU, MD, B31, S)[0]
    ng = sum(1 for n in names if n.startswith("sh"))
    gmin, ib = obs[:ng].min(), int(np.argmin(obs[:ng]))
    Y, b = gen(MU[ib], MD[ib], B31[ib], S, B, rng)
    med = np.median(totals(Y, b, MU[:ng], MD[:ng], B31[:ng], S).min(1))
    c = gmin / med
    print(f"dispersion c = {gmin:.1f} / {med:.1f} = {c:.2f} ({B} replicates)")
    for k in NAMED_KEYS:
        j = names.index(k)
        d_obs = (obs[j] - min(gmin, obs[j])) / c
        Y, b = gen(MU[j], MD[j], B31[j], S, B, rng)
        idx = list(range(ng)) + [j]
        T = totals(Y, b, MU[idx], MD[idx], B31[idx], S)
        dstar = T[:, -1] - T.min(1)
        p = (1 + (dstar >= d_obs).sum()) / (B + 1)
        print(f"  null {k:20s}: delta / c = {d_obs:5.1f}; reference 95% {np.quantile(dstar, .95):4.1f} -> p = {p:.4f}")


def boot_cal(files, y, S, S_raw, B, rng):
    blocks = [np.load(f, allow_pickle=True) for f in files]
    MU = np.concatenate([b["MU"] for b in blocks])
    MD = np.concatenate([b["MD"] for b in blocks])
    B31 = np.concatenate([b["B31"] for b in blocks])
    names = sum([list(b["names"]) for b in blocks], [])
    slope = np.concatenate([np.full(len(b["MU"]), float(b["slope"])) for b in blocks])
    off = np.concatenate([np.full(len(b["MU"]), float(b["offset"])) for b in blocks])
    obs = totals(y[None, :], r3.B31_BAL[0], MU, MD, B31, S)[0]
    pubm = (slope == 1.0) & (off == 0.0)
    ib = int(np.argmin(obs))
    ip = int(np.flatnonzero(pubm)[np.argmin(obs[pubm])])
    d_obs = obs[ip] - obs[ib]
    print(f"\n-- subject calibration (slope x offset x pool, {len(names)} cells): best {names[ib]} total {obs[ib]:.1f} "
          f"(chi2 tail on 53-5 dof {chi2_sf(obs[ib], 48):.3f}); best at public calibration {names[ip]} total {obs[ip]:.1f};"
          f" delta {d_obs:.1f}")
    for gen_name, S_gen in (("shrunk (as the statistic)", S), ("unshrunk bootstrap", S_raw)):
        # GOF at the best H_cal cell
        Y, b = gen(MU[ib], MD[ib], B31[ib], S_gen, B, rng)
        T = totals(Y, b, MU, MD, B31, S)
        mins = T.min(1)
        p = (1 + (mins >= obs[ib]).sum()) / (B + 1)
        print(f"  [{gen_name}] GOF, data from the best H_cal cell: min-total median {np.median(mins):.1f}, 95% "
              f"{np.quantile(mins, .95):.1f}; observed {obs[ib]:.1f} -> p = {p:.4f}")
        # H_pub (slope 1, offset 0) as the null: delta = min over public-calibration cells - min over all
        Y, b = gen(MU[ip], MD[ip], B31[ip], S_gen, B, rng)
        T = totals(Y, b, MU, MD, B31, S)
        dstar = T[:, pubm].min(1) - T.min(1)
        p = (1 + (dstar >= d_obs).sum()) / (B + 1)
        print(f"  [{gen_name}] null = public calibration (H_pub): delta reference median {np.median(dstar):.1f}, 95% "
              f"{np.quantile(dstar, .95):.1f}, 99% {np.quantile(dstar, .99):.1f}; observed {d_obs:.1f} -> p = {p:.4f}")


def chi2_sf(x, df):
    from scipy.stats import chi2
    return float(chi2.sf(x, df))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "boot"
    CACHE.mkdir(exist_ok=True)
    if cmd == "prep-public":
        prep_public(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "prep-cal":
        prep_cal(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "scaled":
        scaled(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    else:
        boot(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
