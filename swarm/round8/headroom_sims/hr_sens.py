"""Posterior variants -> regime/d marginals and candidate contrasts vs W20 (calculus only, x1e-3)."""
import sys
import time
import numpy as np
import hr_common as hc
import hr_full as hfu
import ld_post as lp

t0 = time.time()
names = ["REF", "L15", "M150", "NLs", "NEs", "MXs", "MXh", "MXd", "MEs"]
E = {n: hfu.full(n) for n in names + ["W20"]}
E["Oracle"] = hfu.oracle_full("model")
WB = hc.WB
a20 = E["W20"] @ WB
variants = {"base": {}, "nob31": {"use_b31": False}, "nocomp": {"use_comp": False}, "lam1": {"lams": (1.0,)},
            "old_1006": None}
print("%-9s %5s %5s %5s | %s" % ("posterior", "E[d]", "Hpub", "pi.3", " ".join("%13s" % n for n in names + ["Oracle"])))
for vn, kw in variants.items():
    if kw is None:
        import hr_live
        hr_live.register_family()
        P = lp.Posterior(hc.ROOT / "swarm" / "round4" / "design_sims" / "cache" / "live_design_now.npz").fit()
        W = P.W.sum(2)
    else:
        W = hc.cell_weights(**kw)
    W = W / W.sum()
    wd = W.sum(0)
    Ed = float(wd @ np.array(hc.DS))
    hp = float(wd[np.array(hc.DS) <= 0.85].sum())
    pools = hc.setup()["pools"]
    p3 = float(sum(W[i].sum() for i, pl in enumerate(pools) if pl["pi_low"] >= 0.3))
    cells = []
    for n in names + ["Oracle"]:
        d = np.nan_to_num(E[n] @ WB - a20)
        m = float((W * d).sum())
        p1 = float((W * (d < -1e-3)).sum())
        cells.append(f"{1e3*m:+6.2f} ({p1:.2f})")
    print("%-9s %5.2f %5.2f %5.2f | %s" % (vn, Ed, hp, p3, " ".join("%13s" % c for c in cells)), flush=True)
print("cells: posterior-mean ALC contrast vs W20 x1e-3 (P(contrast < -1e-3) over regime cells)")
print(f"total {time.time() - t0:.0f}s")
