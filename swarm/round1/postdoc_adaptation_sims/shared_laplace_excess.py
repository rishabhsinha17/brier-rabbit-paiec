"""Laplace+probit excess Brier for a pair that SHARES its benchmark with one partner, under the model's own prior
(cur_final normal model, unknown subjects). Exact = 2-D grid over (mu_b, L_T) with the partner level integrated;
per-label likelihood sigmoid(c L), c = 1/sqrt(1 + pi sigma_i^2 / 8) (distinct items), predictive sigmoid(c L_T).
Excess = E[(q_laplace - q_exact)^2] under the prior predictive, ALC-weighted. Same for alone pairs (1-D) for reference."""
import sys
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np
from scipy.stats import binom

TABD = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round1/postdoc_adaptation_sims/tables/")
NAMES = sys.argv[1:] or ["cur_m20"]
TS = {nm: np.load(TABD + nm + ".npz") for nm in NAMES}
t = TS[NAMES[0]]
m0, tau_mu, v_e, sig_i = -2.0, 1.2, 0.5 ** 2 + 1.0 ** 2 + 0.5 ** 2, 0.6
c = 1 / np.sqrt(1 + np.pi * sig_i ** 2 / 8)
BUD = (1, 3, 7, 15, 31); W = {1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
mu = np.linspace(m0 - 6 * tau_mu, m0 + 6 * tau_mu, 161); pmu = np.exp(-0.5 * ((mu - m0) / tau_mu) ** 2); pmu /= pmu.sum()
L = np.linspace(m0 - 6 * np.sqrt(tau_mu ** 2 + v_e), m0 + 6 * np.sqrt(tau_mu ** 2 + v_e), 241)
K = np.exp(-0.5 * (L[None, :] - mu[:, None]) ** 2 / v_e)          # N(L; mu, v_e) up to a constant  (mu x L)
K /= K.sum(axis=1, keepdims=True)
s = 1 / (1 + np.exp(-c * L))
rng = np.random.default_rng(0)
NS = 40000
mus = rng.normal(m0, tau_mu, NS); pT = 1 / (1 + np.exp(-c * (mus + rng.normal(0, np.sqrt(v_e), NS))))
pP = 1 / (1 + np.exp(-c * (mus + rng.normal(0, np.sqrt(v_e), NS))))
tot = {nm: [0.0, 0.0, []] for nm in NAMES}
tot_sh = tot_al = 0.0
for n in BUD:
    k = np.arange(n + 1)
    lik = s[None, :] ** k[:, None] * (1 - s[None, :]) ** (n - k)[:, None]   # (k x L)
    g = (K @ lik.T)                                   # (mu x kP): partner marginal given mu
    q_ex = np.zeros((n + 1, n + 1))
    for kT in range(n + 1):
        wT = K * lik[kT][None, :]                     # (mu x L)
        num = (wT * s[None, :]).sum(axis=1); den = wT.sum(axis=1)
        for kP in range(n + 1):
            wm = pmu * g[:, kP]
            q_ex[kT, kP] = np.sum(wm * num) / np.sum(wm * den)
    # alone exact
    pl = (pmu @ K)                                    # prior over L
    a_ex = np.array([np.sum(pl * lik[kT] * s) / np.sum(pl * lik[kT]) for kT in range(n + 1)])
    bT = binom.pmf(k[None, :], n, pT[:, None]); bP = binom.pmf(k[None, :], n, pP[:, None])
    d2 = (t[f"n{n}"] - q_ex) ** 2
    ex_sh = np.mean(np.sum((bT @ d2) * bP, axis=1)); ex_al = np.mean(bT @ ((t[f"a{n}"] - a_ex) ** 2))
    tot_sh += W[n] * ex_sh; tot_al += W[n] * ex_al
    for nm, tt in TS.items():
        e_sh = np.mean(np.sum((bT @ ((tt[f"n{n}"] - q_ex) ** 2)) * bP, axis=1)); e_al = np.mean(bT @ ((tt[f"a{n}"] - a_ex) ** 2))
        tot[nm][0] += W[n] * e_sh; tot[nm][1] += W[n] * e_al; tot[nm][2].append(f"n{n}: {1e3*e_sh:.2f}/{1e3*e_al:.2f}")
    i = n // 2
    print(f"n={n:2d}: excess shared {1e3 * ex_sh:.3f}e-3, alone {1e3 * ex_al:.3f}e-3 | e.g. kT=0,kP=0: laplace {t[f'n{n}'][0, 0]:.3f} exact {q_ex[0, 0]:.3f}; kT=n,kP=n: {t[f'n{n}'][n, n]:.3f} vs {q_ex[n, n]:.3f}; kT=0,kP=n: {t[f'n{n}'][0, n]:.3f} vs {q_ex[0, n]:.3f}")
print(f"ALC-weighted excess: shared pairs {1e3 * tot_sh:.3f}e-3, alone pairs {1e3 * tot_al:.3f}e-3")
for nm, (a, b, parts) in tot.items():
    print(f"{nm:>14}: ALC-weighted excess vs exact (x1e-3) shared {1e3*a:.3f}, alone {1e3*b:.3f} | per budget shared/alone: " + "  ".join(parts))
