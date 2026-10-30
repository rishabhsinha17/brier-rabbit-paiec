"""Calculus prediction of the public LOBO replay (alone pairs, public rates, d = 0) per held-out benchmark, x1e-3 vs
REF (base): what the paired.py guard should show at k = 1 (pair-keyed or bench-keyed subsample; groups bracketed)."""
import numpy as np
import hr_common as hc
import hr_fast as hf
import fastregime as fr

s = hc.setup()
el, th, tv, wts = s["el"], s["th"], s["tv"], s["wts"]
r = el.r.values.astype(float)
pmf = fr.pmf_table(r)
specs = {"REF": {"m0": -2.0, "V": 2.69}, "W20": {"m0": -2.0, "V": 5.25}, "L15": {"m0": -1.5, "V": 2.69},
         "M150": {"m0": -1.5, "V": 5.25},
         "MXs": {"m0": -0.5, "V": 1.5038, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55},
         "MXh": {"m0": -0.75, "V": 2.0007, "mix_pi": 0.25, "mix_m": -2.9, "mix_s": 0.55}}
E = {}
for k, sp in specs.items():
    acc = np.zeros((len(r), 6))
    for br in (0, 1):
        q = hf.table(sp, 0.0, br, th, tv)
        sq = (q - r[:, None]) ** 2 + (r * (1 - r))[:, None]
        for j, n in enumerate(hc.BUD):
            acc[:, j] += 0.5 * (pmf[:, hf.fp_idx(n)] * sq[:, hf.fp_idx(n)]).sum(1)
    E[k] = acc
benches = sorted(set(el.bench))
print("%-6s %s | %8s  %s" % ("arm", " ".join("%10s" % b[:10] for b in benches), "macro", "B0/B1/B3/B7 macro"))
for k in specs:
    d = E[k] - E["REF"]
    per = [1e3 * (d[el.bench.values == b] @ hc.WB).mean() for b in benches]
    mb = [1e3 * np.mean([d[el.bench.values == b, j].mean() for b in benches]) for j in range(4)]
    print("%-6s %s | %+8.2f  %s" % (k, " ".join("%+10.2f" % x for x in per), np.mean(per), " ".join("%+6.1f" % x for x in mb)))
