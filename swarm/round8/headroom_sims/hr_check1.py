"""Check 1: hr_fast tables == fastpair / hr_common reference tables (Laplace, HierModel mixture in both brackets,
exact normal, exact mixture); CellEval posterior mean == hr_common.evaluate."""
import math
import time
import numpy as np
import hr_common as hc
import fastpair as fp
import hr_fast as hf

t0 = time.time()
s = hc.setup(); th, tv = s["th"], s["tv"]
cases = {
    "REF": ({"m0": -2.0, "V": 2.69}, dict(hc.REFP)),
    "W20": ({"m0": -2.0, "V": 5.25}, dict(hc.W20P)),
    "mix": ({"m0": -1.5, "V": 4.0, "mix_pi": 0.25, "mix_m": -2.9, "mix_s": 0.55},
            dict(hc.REFP, m0=-1.5, tau_mu=math.sqrt(4.0 - 1.25), mix_pi=0.25, mix_m=-2.9, mix_s=0.55)),
}
for name, (spec, prm) in cases.items():
    worst = 0
    for d in (0.0, 0.75):
        for br in (0, 1):
            a = hf.table(spec, d, br, th, tv)
            b = hc.predict_table_hier(dict(prm, extra_V=float(br)), th + d, tv)
            worst = max(worst, np.abs(a - b).max())
    print(f"{name}: max |fast - ref| = {worst:.2e}")
# exact
for name, spec in (("exact W20", {"m0": -2.0, "V": 5.25, "exact": True}),
                   ("exact mix", {"m0": -1.5, "V": 4.0, "exact": True, "mix_pi": 0.25, "mix_m": -2.9, "mix_s": 0.55})):
    prm = dict(hc.REFP, m0=spec["m0"], tau_mu=math.sqrt(spec["V"] - 1.25))
    if spec.get("mix_pi"):
        prm.update(mix_pi=spec["mix_pi"], mix_m=spec["mix_m"], mix_s=spec["mix_s"])
    worst = 0
    for d in (0.0, 0.75):
        for br in (0, 1):
            a = hf.table(spec, d, br, th, tv)
            b = fp.predict_table_exact(dict(prm, extra_V=float(br)), th + d, tv)
            worst = max(worst, np.abs(a - b).max())
    print(f"{name}: max |fast - ref| = {worst:.2e}")
print(f"tables {time.time() - t0:.1f}s")
CE = hf.CellEval()
E = CE.eval_spec(hf.W20_SPEC())
m_fast = CE.mean(E)
Eref = hc.evaluate(hc.family_tables(hc.W20P)).mean(2)          # (nPool, nD, 6)
w = hc.cell_weights()
wk = np.where(CE.keep, w, 0); wk /= wk.sum()
m_ref = hc.wmean(Eref, wk)
print("CellEval W20 mean vs evaluate: max diff %.2e; ALC %.4f" % (np.abs(m_fast - m_ref).max(), m_fast @ hc.WB))
print(f"cells {len(CE.cells)}, d indices {CE.ds}; total {time.time() - t0:.1f}s")
