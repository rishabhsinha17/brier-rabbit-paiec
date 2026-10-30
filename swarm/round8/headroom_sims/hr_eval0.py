"""First headroom readout: posterior-mean expected Brier per budget of REF, W20, W20+S, and the Bayes predictors."""
import time
import numpy as np
import hr_common as hc

t0 = time.time()
w = hc.cell_weights()
s = hc.setup()
# sanity: base thetas reproduce
thS, tvS = hc.thetas_S()
th0 = s["th"]
print("S thetas: changed pairs %d of %d, max |dtheta| %.3f" % (int((np.abs(thS - th0) > 1e-12).sum()), len(th0), np.abs(thS - th0).max()))
rows = {}
def brmean(E):
    return E.mean(2)
rows["REF"] = brmean(hc.evaluate(hc.family_tables(hc.REFP)))
rows["W20"] = brmean(hc.evaluate(hc.family_tables(hc.W20P)))
rows["REF+S"] = brmean(hc.evaluate(hc.family_tables(hc.REFP, thS, tvS)))
rows["W20+S"] = brmean(hc.evaluate(hc.family_tables(hc.W20P, thS, tvS)))
for mode in ("model", "pooled"):
    z = np.load(hc.CACHE / f"bayes_{mode}.npz")
    rows[f"Bayes[{mode}]"] = hc.evaluate(z["Q"])                # (nPool, nD, 6)
    QO = z["QO"]
    ok = ~np.isnan(QO[:, 0, 0])
    EO = np.full((len(QO), len(hc.DS), 6), np.nan)
    Eall = hc.evaluate(np.nan_to_num(QO[ok], nan=0.5))         # (nPool, nOk, 6)
    for jj, i in enumerate(np.flatnonzero(ok)):
        EO[i] = Eall[i, jj][None, :]
    rows[f"Oracle-cell[{mode}]"] = EO
print(f"computed in {time.time() - t0:.1f}s")
wz = np.where(np.isnan(rows["Oracle-cell[model]"][:, :, 0]), 0, w); wz /= wz.sum()
print(f"weight on cells with oracle tables: {np.where(np.isnan(rows['Oracle-cell[model]'][:, :, 0]), 0, w).sum():.4f}")
hdr = "%-20s %8s | %7s %7s %7s %7s %7s %7s" % ("predictor", "ALC", "B0", "B1", "B3", "B7", "B15", "B31")
print("posterior-mean expected Brier x1e-3 (bench-equal, alone-pair calculus, bracket mean for our model)")
print(hdr)
for k, E in rows.items():
    m = hc.wmean(np.nan_to_num(E), wz)
    print("%-20s %8.2f | %s" % (k, 1e3 * (m * hc.WB).sum(), " ".join("%7.2f" % (1e3 * v) for v in m)))
print("\ncontrasts vs W20 (x1e-3): posterior mean (sd over cells)")
for k, E in rows.items():
    D = np.nan_to_num(E) - rows["W20"]
    a = (D * hc.WB).sum(-1)
    m = hc.wmean(D, wz)
    print("%-20s ALC %+7.2f (%5.2f) | %s" % (k, 1e3 * hc.wmean(a, wz), 1e3 * hc.wsd(a, wz), " ".join("%+7.2f" % (1e3 * v) for v in m)))
np.savez_compressed(hc.CACHE / "eval0.npz", **{k.replace("[", "_").replace("]", ""): v for k, v in rows.items()}, w=w)
