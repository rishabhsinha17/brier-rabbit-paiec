"""Search 1: posterior-matched NORMAL level prior, Laplace (HierModel) and exact (ExactAloneHier) inference, on an
(m0, V) grid, alone-pair calculus under the current design posterior (calculus only; no discrepancy).
usage: R hr_search1.py laplace|exact [V list]"""
import sys
import time
import numpy as np
import hr_common as hc
import hr_fast as hf

mode = sys.argv[1] if len(sys.argv) > 1 else "laplace"
VS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2.69, 3.5, 4.5, 5.25, 6.5, 8.0, 10.0]
M0S = np.round(np.arange(-3.0, -0.49, 0.25), 3)
t0 = time.time()
CE = hf.CellEval()
E20 = CE.eval_spec(hf.W20_SPEC())
a20 = CE.alc(E20)
out = hc.CACHE / f"search1_{mode}.csv"
rows = []
for V in VS:
    for m0 in M0S:
        spec = {"m0": float(m0), "V": V, "exact": mode == "exact"}
        E = CE.eval_spec(spec)
        D = E - E20
        a = CE.alc(E) - a20
        mean = CE.mean(D)
        rows.append([mode, m0, V, CE.wc @ a, np.sqrt(CE.wc @ (a - CE.wc @ a) ** 2), (CE.wc * (a < -1e-3)).sum(),
                     (CE.wc * (a < 0)).sum(), *mean])
    print(f"V {V}: done {time.time() - t0:.0f}s; best m0 {M0S[np.argmin([r[3] for r in rows[-len(M0S):]])]:+.2f} "
          f"ALC {1e3 * min(r[3] for r in rows[-len(M0S):]):+.2f}", flush=True)
    hf.save_disk()
import csv
new = not out.exists()
with out.open("a") as fh:
    wr = csv.writer(fh)
    if new:
        wr.writerow(["mode", "m0", "V", "dALC", "sd_cells", "P_lt_m1e3", "P_lt0", "dB0", "dB1", "dB3", "dB7", "dB15", "dB31"])
    wr.writerows(rows)
rows.sort(key=lambda r: r[3])
print("best 8 (x1e-3 vs W20, calculus only):")
for r in rows[:8]:
    print("  %s m0 %+.2f V %5.2f  dALC %+6.2f (sd %4.2f) P(<-1e-3) %.2f P(<0) %.2f | B0 %+6.2f B1 %+6.2f B3 %+6.2f B7 %+6.2f B15 %+6.2f B31 %+6.2f"
          % (r[0], r[1], r[2], 1e3 * r[3], 1e3 * r[4], r[5], r[6], *[1e3 * x for x in r[7:]]))
print(f"total {time.time() - t0:.0f}s")
