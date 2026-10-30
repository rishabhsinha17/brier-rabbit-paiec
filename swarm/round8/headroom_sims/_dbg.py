import numpy as np, hr_common as hc, ld_calc as lc
G = lc.Grid()
print("E", G.E.shape, "keys", len(G.keys), "idx", len(G.idx), "max idx", max(G.idx.values()))
fam = {"m0": -0.5, "tau_mu": 1.07, "tau_delta": 0.33, "tau_phi": 0.5}
try:
    print(G.tix(None, 0.0, 0, fam))
except Exception as e:
    print("ERR", repr(e))
print("E", G.E.shape, "keys", len(G.keys), "idx", len(G.idx), "max idx", max(G.idx.values()))
