"""Synthetic checks of paiec/ext/pool_cavity.py (no data needed, seconds).

1. Identity: with no other-subject label on the target item the plug-in equals HierModel bit for bit.
2. Head A smoke case: A has 1/3, partner B 3/10 and B labelled item j: base / headA ItemPool / cavity / map.
3. Budget sweep of the B1-B7 regime: partner with n labels (rate 30%) labelled the target item; how far each design
   moves A's prediction after a partner success or failure, against an exact 2-D reference computed by brute force
   (partner level and item effect integrated jointly on a grid, target level fixed at its fitted mean).
4. Never raises on malformed input.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from paiec.ext.headA_itempool import ItemPoolModel  # noqa: E402
from paiec.ext.pool_cavity import CavityPoolModel  # noqa: E402
from paiec.hier import HierModel  # noqa: E402

LIVE = {"m0": -2.0, "tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5,
        "tau_theta": 0.5, "damped_newton": True}


def subj(name):
    return {"normalized_name": name, "provider": "x", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}


def item(i, bench="benchmark_1"):
    return {"item_content": f"question {i}", "item_features": "", "interactors": "", "benchmark_id": bench}


def evidence(spec):
    """spec: list of (subject name, item index, y)."""
    return [[[subj(s), item(i)], y] for s, i, y in spec]


def models(**kw):
    p = dict(LIVE, **kw)
    return {"base": HierModel({}, p), "headA_x2": ItemPoolModel({}, dict(p, sigma_x=2.0)),
            "cav_x2": CavityPoolModel({}, dict(p, sigma_x=2.0)),
            "cav_x2_nocav": CavityPoolModel({}, dict(p, sigma_x=2.0, pool_cavity=False)),
            "cav_x1.5": CavityPoolModel({}, dict(p, sigma_x=1.5)),
            "map_x2": CavityPoolModel({}, dict(p, sigma_x=2.0, pool_mode="map"))}


def check_identity():
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(20):
        spec = [("A", int(i), int(rng.random() < 0.3)) for i in rng.choice(50, 5, replace=False)]
        spec += [("B", int(i), int(rng.random() < 0.4)) for i in rng.choice(np.arange(100, 150), 7, replace=False)]
        ev = evidence(spec)
        ms = models()
        for tgt in ("A", "B", "C"):
            for j in (60, 61, 199):  # items nobody labelled
                ref = ms["base"].predict([subj(tgt), item(j)], ev)
                for name, m in ms.items():
                    worst = max(worst, abs(m.predict([subj(tgt), item(j)], ev) - ref))
        # own-labelled item of A predicted for A (no partner): must equal base too
        j = spec[0][1]
        ref = ms["base"].predict([subj("A"), item(j)], ev)
        for name, m in ms.items():
            worst = max(worst, abs(m.predict([subj("A"), item(j)], ev) - ref))
    print(f"1. identity without partner labels: max |plug-in - base| = {worst:.2e}")


def check_smoke():
    print("2. A 1/3, B 3/10, B labelled item 99 (y), predict A on 99:")
    for y in (1, 0):
        spec = [("A", 1, 1), ("A", 2, 0), ("A", 3, 0)] + [("B", 10 + i, int(i < 2)) for i in range(9)] + [("B", 99, y)]
        if y == 0:
            spec = [("A", 1, 1), ("A", 2, 0), ("A", 3, 0)] + [("B", 10 + i, int(i < 3)) for i in range(9)] + [("B", 99, 0)]
        ev = evidence(spec)
        out = {n: m.predict([subj("A"), item(99)], ev) for n, m in models().items()}
        unl = models()["base"].predict([subj("A"), item(98)], ev)
        print(f"   B {'solved' if y else 'failed'}: unlabelled {unl:.3f} | " + " ".join(f"{n} {v:.3f}" for n, v in out.items()))


def exact_ref(n_b, k_b, y_j, m0=-2.0, sx=2.0, n_a=0, k_a=0):
    """Brute-force reference on the conditional scale for the simplest two-pair model: benchmark level mu (sd 1.2),
    pair deviations dA, dB (sd sqrt(1 + .25)), item effects u ~ N(0, sx^2) for EVERY item (own labels integrated
    item by item, i.e. the item-level model the plug-in approximates), partner labels: k_b of n_b on other items plus
    y_j on item j; target labels: k_a of n_a. Returns P(A solves j). Grid over (mu, dA, dB) is too big, so mu+dB and
    mu+dA are integrated with mu shared via a 3-D grid of 25^3 points."""
    kappa = 1.0 / math.sqrt(1.0 + math.pi * sx * sx / 8.0)
    gx, gw = np.polynomial.hermite_e.hermegauss(25)
    gw = gw / gw.sum()
    # marginal-scale priors converted to the conditional scale
    sd_mu, sd_d = 1.2 / kappa, math.sqrt(1.25) / kappa
    mu = m0 / kappa + sd_mu * gx[:, None, None]
    LA = mu + sd_d * gx[None, :, None]
    LB = mu + sd_d * gx[None, None, :]
    W = gw[:, None, None] * gw[None, :, None] * gw[None, None, :]
    ux, uw = np.polynomial.hermite_e.hermegauss(60)
    uw = uw / uw.sum()
    u = sx * ux

    def p_item(L):  # E_u sigmoid(L + u) for unseen items
        return np.tensordot(1.0 / (1.0 + np.exp(-(L[..., None] + u))), uw, axes=([-1], [0]))

    pB, pA = p_item(LB), p_item(LA)
    logw = np.log(W) + k_b * np.log(pB) + (n_b - k_b) * np.log1p(-pB) + k_a * np.log(pA) + (n_a - k_a) * np.log1p(-pA)
    # item j: joint over u with partner's label and target prediction
    sB = 1.0 / (1.0 + np.exp(-(LB[..., None] + u)))
    sA = 1.0 / (1.0 + np.exp(-(LA[..., None] + u)))
    lik = sB if y_j else 1.0 - sB
    num = np.tensordot(lik * sA, uw, axes=([-1], [0]))
    den = np.tensordot(lik, uw, axes=([-1], [0]))
    w = np.exp(logw - logw.max()) * den
    return float((w * num / den).sum() / w.sum())


def check_budget_sweep():
    print("3. partner with n labels (30% rate), A with n labels (30%), partner labelled target item j:")
    print("   n  y | base   headA  cav_x2 nocav  map_x2 | exact(sx=2, item-level model)")
    for n in (1, 3, 7, 15, 31):
        for y in (1, 0):
            k = int(round(0.3 * (n - 1)))
            ka = int(round(0.3 * n))
            spec = [("A", 1000 + i, int(i < ka)) for i in range(n)]
            spec += [("B", 2000 + i, int(i < k)) for i in range(n - 1)] + [("B", 99, y)]
            ev = evidence(spec)
            ms = models()
            v = {nm: ms[nm].predict([subj("A"), item(99)], ev) for nm in ("base", "headA_x2", "cav_x2", "cav_x2_nocav", "map_x2")}
            ex = exact_ref(n - 1, k, y, n_a=n, k_a=ka)
            print(f"  {n:2d}  {y} | " + "  ".join(f"{v[nm]:.3f}" for nm in v) + f" | {ex:.3f}")


def check_never_raises():
    m = CavityPoolModel({}, dict(LIVE))
    bad = [None, [], [[None, 1]], [[[{}, {}], "x"]], [[[subj("A"), item(1)], 1], 5]]
    outs = []
    for ev in bad:
        for inp in ([subj("A"), item(1)], [None, None], "junk", [subj("B"), {"benchmark_id": None}]):
            outs.append(m.predict(inp, ev))
    ok = all(0.0 <= o <= 1.0 and math.isfinite(o) for o in outs)
    print(f"4. malformed inputs: {len(outs)} calls, all finite in [0,1]: {ok}")


if __name__ == "__main__":
    check_identity()
    check_smoke()
    check_budget_sweep()
    check_never_raises()
