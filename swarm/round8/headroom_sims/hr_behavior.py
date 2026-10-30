"""q(k/n) for an unknown subject (theta 0, var 0.25), alone pair, no item group: how each arm moves."""
import numpy as np
import hr_common as hc
import hr_fast as hf
specs = {"REF": {"m0": -2.0, "V": 2.69}, "W20": {"m0": -2.0, "V": 5.25}, "M150 (W20@-1.5)": {"m0": -1.5, "V": 5.25},
         "MXs": {"m0": -0.5, "V": 1.5038, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55},
         "MXh": {"m0": -0.75, "V": 2.0007, "mix_pi": 0.25, "mix_m": -2.9, "mix_s": 0.55}}
cols = [(0, 0), (0, 1), (1, 1), (0, 3), (1, 3), (3, 3), (0, 7), (2, 7), (0, 15), (0, 31), (5, 31)]
print("%-16s %s" % ("arm", " ".join("%6s" % f"{k}/{n}" for k, n in cols)))
for k_, sp in specs.items():
    q = hf.table(sp, 0.0, 0, np.array([0.0]), np.array([0.25]))[0]
    print("%-16s %s" % (k_, " ".join("%6.3f" % q[hc.KN.index(c)] for c in cols)))
