"""Check the Woodbury cavity of CavityPoolModel against an actual refit without the labels on the target item.
Partner level (mean, var) for the target item under: full fit, Woodbury cavity, refit-without-item-labels."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import numpy as np  # noqa: E402

from unit_checks import LIVE, evidence, item, subj  # noqa: E402
from paiec.ext.pool_cavity import CavityPoolModel, _full_key  # noqa: E402

print("  n  y | partner level full (m, v) | Woodbury cavity (m, v) | refit without item-j labels (m, v)")
for n in (1, 3, 7, 15, 31):
    for y in (1, 0):
        k = int(round(0.3 * (n - 1)))
        ka = int(round(0.3 * n))
        spec = [("A", 1000 + i, int(i < ka)) for i in range(n)] + [("B", 2000 + i, int(i < k)) for i in range(n - 1)] + [("B", 99, y)]
        ev = evidence(spec)
        m = CavityPoolModel({}, dict(LIVE, sigma_x=2.0))
        m.predict([subj("A"), item(99)], ev)  # fit
        fit = m._fit
        b = item(99)["benchmark_id"]
        full = m._plain_level(fit, subj("B"), b, item(99))
        # Woodbury: replicate the cavity math for partner B
        labs = m._label_index(ev)[(b, _full_key(item(99)))]
        rows = [m._row(fit, sd, it, b) for _, sd, it, _ in labs]
        w, S = fit["mean"], fit["cov"]
        ys = np.array([float(yy) for _, _, _, yy in labs])
        eta = np.array([float(w[c].sum()) + off for c, off in rows])
        pr = 1 / (1 + np.exp(-eta))
        h = pr * (1 - pr)
        SX = np.stack([S[:, c].sum(axis=1) for c, _ in rows], axis=1)
        G = np.array([[float(SX[c, kk].sum()) for kk in range(len(rows))] for c, _ in rows])
        Minv = np.linalg.inv(np.diag(1 / h) - G)
        cols, mean, var = m._level_vec(fit, subj("B"), b, item(99))
        t = SX[cols].sum(axis=0)
        wc = (mean + float(w[cols].sum()) - float(t @ Minv @ ((ys - pr) / h)),
              var + float(S[np.ix_(cols, cols)].sum()) + float(t @ Minv @ t))
        ev2 = [e for e in ev if e[0][1]["item_content"] != item(99)["item_content"]]
        m2 = CavityPoolModel({}, dict(LIVE, sigma_x=2.0))
        m2.predict([subj("A"), item(98)], ev2)
        ref = m2._plain_level(m2._fit, subj("B"), b, item(99))
        print(f" {n:2d}  {y} | ({full[0]:+.3f}, {full[1]:.3f}) | ({wc[0]:+.3f}, {wc[1]:.3f}) | ({ref[0]:+.3f}, {ref[1]:.3f})")
