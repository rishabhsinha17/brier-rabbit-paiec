"""d-profile (conditional on the subject offset d) of candidates vs W20, calculus, base posterior over pools given d."""
import numpy as np
import hr_common as hc
import hr_full as hfu
W = hc.cell_weights(); W = W / W.sum()
names = ["REF", "L15", "M150", "NLs", "MXs", "MXh", "MXd"]
E = {n: hfu.full(n) for n in names + ["W20"]}
E["Bayes"] = hfu.full_tables("Bayes_model", None) if (hc.CACHE / "Efull_Bayes_model.npz").exists() else None
a20 = E["W20"] @ hc.WB
print("%5s %6s | %s" % ("d", "mass", " ".join("%7s" % n for n in names)))
for j, d in enumerate(hc.DS):
    m = W[:, j].sum()
    if m < 0.002:
        continue
    wj = W[:, j] / m
    print("%5.2f %6.3f | %s" % (d, m, " ".join("%+7.2f" % (1e3 * (wj @ (E[n][:, j] @ hc.WB - a20[:, j]))) for n in names)))
