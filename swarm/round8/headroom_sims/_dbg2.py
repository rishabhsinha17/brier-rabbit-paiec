import hr_common as hc, ld_calc as lc
G = lc.Grid()
fam = {"m0": -1.75, "tau_mu": 2.2, "tau_delta": 0.8, "tau_phi": 0.5}
try:
    G.budgets(None, 0.0, fam)
    print("budgets ok")
except IndexError as e:
    print("IndexError on first use of an off-grid arm:", e)
print("second call:", G.budgets(None, 0.0, fam).shape)
