"""Search 2: posterior-matched TWO-COMPONENT level prior (near-zero component + regular normal component).
mode 'hier'  : HierModel's existing machinery (mix_pi, mix_m, mix_s; Laplace normal component, 1-D weight).
mode 'exact' : ExactAloneHier replica (mixture inside the exact 1-D prior).
Grid given as JSON lists: R hr_search2.py hier '{"m0":[...],"V":[...],"pi":[...],"mm":[...],"ms":[...]}' [tag]"""
import csv
import itertools
import json
import sys
import time
import numpy as np
import hr_common as hc
import hr_fast as hf

mode = sys.argv[1]
g = json.loads(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else ""
t0 = time.time()
CE = hf.CellEval()
E20 = CE.eval_spec(hf.W20_SPEC())
a20 = CE.alc(E20)
rows = []
combos = list(itertools.product(g["m0"], g["V"], g["pi"], g["mm"], g["ms"]))
for n, (m0, V, pi, mm, ms) in enumerate(combos):
    spec = {"m0": m0, "V": V, "mix_pi": pi, "mix_m": mm, "mix_s": ms, "exact": mode == "exact"}
    E = CE.eval_spec(spec)
    a = CE.alc(E) - a20
    mean = CE.mean(E - E20)
    rows.append([mode, m0, V, pi, mm, ms, CE.wc @ a, np.sqrt(CE.wc @ (a - CE.wc @ a) ** 2),
                 (CE.wc * (a < -1e-3)).sum(), (CE.wc * (a < 0)).sum(), *mean])
    if time.time() - t0 > 100:
        print(f"time budget hit after {n + 1} of {len(combos)} combos", flush=True)
        break
hf.save_disk()
out = hc.CACHE / f"search2_{mode}{tag}.csv"
new = not out.exists()
with out.open("a") as fh:
    wr = csv.writer(fh)
    if new:
        wr.writerow(["mode", "m0", "V", "pi", "mix_m", "mix_s", "dALC", "sd_cells", "P_lt_m1e3", "P_lt0",
                     "dB0", "dB1", "dB3", "dB7", "dB15", "dB31"])
    wr.writerows(rows)
rows.sort(key=lambda r: r[6])
print(f"{len(rows)} combos in {time.time() - t0:.0f}s; best 10 (x1e-3 vs W20, calculus only):")
for r in rows[:10]:
    print("  m0 %+.2f V %4.2f pi %.2f mm %+.2f ms %.2f  dALC %+6.2f (sd %4.2f) P(<-1e-3) %.2f P(<0) %.2f | B0 %+6.2f B1 %+6.2f B3 %+6.2f B7 %+6.2f B15 %+6.2f B31 %+6.2f"
          % (r[1], r[2], r[3], r[4], r[5], 1e3 * r[6], 1e3 * r[7], r[8], r[9], *[1e3 * x for x in r[10:]]))
