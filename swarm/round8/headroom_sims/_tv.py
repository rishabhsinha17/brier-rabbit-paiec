import numpy as np, hr_common as hc
s = hc.setup(); th, tv = s["th"], s["tv"]
print("theta range %.2f %.2f; tv unique %d: %s" % (th.min(), th.max(), len(np.unique(np.round(tv, 6))), np.unique(np.round(tv, 4))[:30]))
thS, tvS = hc.thetas_S()
print("S theta range %.2f %.2f; tv unique %d" % (thS.min(), thS.max(), len(np.unique(np.round(tvS, 6)))))
w = hc.cell_weights()
print("d marginal", np.round(w.sum(0), 3))
