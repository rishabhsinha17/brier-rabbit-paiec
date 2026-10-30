"""Reference arms in the same calculus/posterior: doctawho42's actual hier-ship predictor (round-2 replica, his theta =
our theta, d applies) and the 'doctawho42-direction' family arms the round-8 standing analyst guards, vs W20 and MXs."""
import numpy as np
import hr_common as hc
import hr_fast as hf

P = hc.posterior()
CE = hf.CellEval()
G = P.G
E = {}
# doctawho42 replica: grid tables per d (same for both brackets) -> per kept cell
Edw = np.zeros((len(CE.cells), 6))
for c, (i, j) in enumerate(CE.cells):
    Edw[c] = G.budgets("pub_doctawho42_hiership", hc.DS[j])[i].mean(0)
E["doctawho42 hier-ship (exact, his prior)"] = Edw
specs = {"W20": {"m0": -2.0, "V": 5.25},
         "m250_mu25_d14 (grid)": {"m0": -2.5, "V": 6.25 + 0.25 + 1.96},
         "m200_mu20_d14 (grid)": {"m0": -2.0, "V": 4.0 + 0.25 + 1.96},
         "m250_mu25_d24 (off-grid)": {"m0": -2.5, "V": 6.25 + 0.25 + 5.76},
         "m250_mu25_d24 exact": {"m0": -2.5, "V": 6.25 + 0.25 + 5.76, "exact": True},
         "MXs": {"m0": -0.5, "V": 1.5038, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55}}
for k, v in specs.items():
    E[k] = CE.eval_spec(v)
a20 = CE.alc(E["W20"])
print("%-42s %7s %6s %6s %6s | %s" % ("arm", "vsW20", "sd", "P<-1e-3", "P<0", "B0 B1 B3 B7 B15 B31 (x1e-3 vs W20)"))
for k in E:
    d = CE.alc(E[k]) - a20
    m = CE.wc @ d
    mb = CE.mean(E[k] - E["W20"])
    print("%-42s %+7.2f %6.2f %6.2f %6.2f | %s" % (k, 1e3 * m, 1e3 * np.sqrt(CE.wc @ (d - m) ** 2), (CE.wc * (d < -1e-3)).sum(),
          (CE.wc * (d < 0)).sum(), " ".join("%+6.2f" % (1e3 * x) for x in mb)))
