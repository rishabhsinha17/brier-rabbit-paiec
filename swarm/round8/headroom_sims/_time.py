import time, numpy as np, hr_common as hc, fastpair as fp
s = hc.setup(); th, tv = s["th"], s["tv"]
m_eff = np.round(np.arange(-3.5, 1.25001, 0.125), 3)
thb = np.concatenate([th + m for m in m_eff]); tvb = np.tile(tv, len(m_eff))
p = dict(hc.REFP, m0=0.0, tau_mu=0.0, tau_phi=0.0, tau_delta=0.0, extra_V=5.25)
t0 = time.time(); q = fp.predict_table(p, thb, tvb); print("laplace rows", len(thb), "%.1fs" % (time.time()-t0))
t0 = time.time(); q2 = fp.predict_table_exact(p, thb, tvb); print("exact rows", len(thb), "%.1fs" % (time.time()-t0))
