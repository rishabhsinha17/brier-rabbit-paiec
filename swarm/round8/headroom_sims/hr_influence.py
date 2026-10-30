"""What drives the d (subject offset) posterior now? Refit the design posterior dropping one arm's observations (or
the B31 constraint / the competitor block) and report E[d], P(H_pub), and MXs / MXh / M150 / L15 vs W20 (calculus)."""
import time
import numpy as np
import hr_common as hc
import hr_full as hfu
import ld_post as lp
import ld_calc as lc
import hr_live

hr_live.register_family()


class DropPost(lp.Posterior):
    def __init__(self, npz, drop=(), **kw):
        self._drop = set(drop)
        super().__init__(npz, **kw)
        if self._drop:
            keep = [i for i, (a, s) in enumerate(self.labels) if a not in self._drop]
            self.labels = [self.labels[i] for i in keep]
            self.y = self.y[keep]
            self.S = self.S[np.ix_(keep, keep)]
            self._build_obs_pred()
            self._build_disc()


t0 = time.time()
E = {n: hfu.full(n) for n in ["W20", "MXs", "MXh", "M150", "L15", "REF"]}
a20 = E["W20"] @ hc.WB
P0 = hc.posterior()
arms = []
for a, s in P0.labels:
    if a not in arms:
        arms.append(a)
print("%-26s %5s %5s | %7s %7s %7s %7s %7s" % ("dropped", "E[d]", "Hpub", "REF", "L15", "M150", "MXh", "MXs"))
for drop in [()] + [(a,) for a in arms] + [tuple(a for a in arms if a.startswith("v6_")),
                                          tuple(a for a in arms if a.startswith("pub_"))]:
    P = DropPost(hc.LIVE_NPZ, drop=drop).fit()
    W = P.W.sum(2)
    W = W / W.sum()
    wd = W.sum(0)
    Ed = float(wd @ np.array(hc.DS))
    hp = float(wd[np.array(hc.DS) <= 0.85].sum())
    vals = [1e3 * float((W * (E[n] @ hc.WB - a20)).sum()) for n in ["REF", "L15", "M150", "MXh", "MXs"]]
    lab = "none" if not drop else (drop[0] if len(drop) == 1 else ("all v6 arms" if drop[0].startswith("v6") else "all competitors"))
    print("%-26s %5.2f %5.2f | %s" % (lab, Ed, hp, " ".join("%+7.2f" % v for v in vals)), flush=True)
print(f"total {time.time() - t0:.0f}s")
