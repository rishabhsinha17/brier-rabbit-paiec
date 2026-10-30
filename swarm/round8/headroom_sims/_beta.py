import numpy as np, hr_common as hc
s = hc.setup(); el, th, tv, w = s["el"], s["th"], s["tv"], s["wts"]
r = np.clip(el.r.values.astype(float), 1e-3, 1-1e-3); x = np.log(r/(1-r))
b = el.bench.values
print("benches", {k: int((b==k).sum()) for k in np.unique(b)})
X = np.c_[np.ones_like(th), th]; Wd = np.diag(w)
coef = np.linalg.solve(X.T@Wd@X, X.T@Wd@x); res = x - X@coef
print("pooled bench-equal: alpha %.3f beta %.3f resid sd %.3f total sd %.3f" % (coef[0], coef[1], np.sqrt(w@res**2), np.sqrt(w@(x - w@x)**2)))
# within-bench slope
xc = x.copy(); thc = th.copy()
for k in np.unique(b):
    m = b==k; xc[m] -= np.average(x[m]); thc[m] -= np.average(th[m])
bw = (w*thc*xc).sum()/(w*thc*thc).sum()
print("within-bench beta %.3f" % bw)
for k in np.unique(b):
    m = b==k
    print(k, "n", m.sum(), "mean logit %.2f sd %.2f  mean theta %.2f sd %.2f  slope %.2f" % (x[m].mean(), x[m].std(), th[m].mean(), th[m].std(), np.polyfit(th[m], x[m], 1)[0]))
print("theta sd %.3f, tv mean %.3f" % (np.sqrt(w@(th - w@th)**2), w@tv))
