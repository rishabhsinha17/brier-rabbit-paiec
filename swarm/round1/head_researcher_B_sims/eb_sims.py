"""Zero-cost synthetic checks for head_researcher_B.md (pure numpy, no data, no feedback used).

Reduction used throughout: a pair alone on its benchmark, no item side information. Given the pair's mean accuracy p
(over its item pool), labels on distinct random items are iid Bernoulli(p), so the Brier-optimal prediction for every
unseen item after k successes in n labels is E[p | k, n] = M(k+1, n-k) / M(k, n-k), M(a, b) = E_G[p^a (1-p)^b].
Excess Brier of a working prior Q at budget n = sum_k P_G(k | n) (E_Q[p|k,n] - E_G[p|k,n])^2 (item-level irreducible
noise is common to all predictors and cancels). ALC weights 0.1/0.2/0.2/0.2/0.2/0.1.

All "true" pools below are HYPOTHETICAL scenarios chosen to bracket the qualitative live finding ("many near-zero pairs
plus mid/high pairs"); none is estimated from formative feedback.
"""
import math
import numpy as np

BUD = (0, 1, 3, 7, 15, 31)
WTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
GRID = np.linspace(1e-5, 1 - 1e-5, 6001)       # p grid
LOGIT = np.log(GRID / (1 - GRID))
GX, GW = np.polynomial.hermite_e.hermegauss(40)
GW = GW / GW.sum()


def comb(n, k):
    return math.comb(n, k)


def density_logitnormal(m, s):
    """Density over the p-grid of p = E_eps[sigmoid(L + eps)] would need a transform; we approximate the item-averaged
    accuracy by sigmoid(L / sqrt(1 + pi*s_item^2/8)) (probit moderation) -> logit-normal with shrunk location/scale."""
    w = np.exp(-0.5 * ((LOGIT - m) / s) ** 2) / (GRID * (1 - GRID))
    return w / w.sum()


def model_prior(m0, v_level=2.94, s_item=0.6):
    """Our HierModel's implied prior on pair mean accuracy for an unknown pair (k=1 composition):
    level ~ N(m0, v_level); item-averaged accuracy ~= sigmoid(level * c), c = 1/sqrt(1+pi s_item^2/8)."""
    c = 1.0 / math.sqrt(1 + math.pi * s_item ** 2 / 8)
    return density_logitnormal(m0 * c, math.sqrt(v_level) * c)


def beta_density(a, b):
    w = np.exp((a - 1) * np.log(GRID) + (b - 1) * np.log1p(-GRID))
    return w / w.sum()


def mix(*parts):
    tot = sum(w for w, _ in parts)
    out = sum(w * d for w, d in parts) / tot
    return out / out.sum()


def post_mean_table(g, n):
    """E_g[p | k, n] for k = 0..n and P_g(k | n)."""
    lik = np.array([comb(n, k) * GRID ** k * (1 - GRID) ** (n - k) for k in range(n + 1)])  # (n+1, grid)
    pk = lik @ g
    pm = (lik * GRID) @ g / np.maximum(pk, 1e-300)
    return pm, pk


def tempered_post_mean(g, n, alpha):
    lik = np.array([GRID ** k * (1 - GRID) ** (n - k) for k in range(n + 1)]) ** alpha
    num = (lik * GRID) @ g
    den = lik @ g
    return num / np.maximum(den, 1e-300)


def regret(g_true, q, tempered_alpha=None):
    out = []
    for n in BUD:
        pm_t, pk = post_mean_table(g_true, n)
        if tempered_alpha is None:
            pm_q, _ = post_mean_table(q, n)
        else:
            pm_q = tempered_post_mean(q, n, tempered_alpha)
        out.append(float(pk @ (pm_q - pm_t) ** 2))
    return np.array(out)


def bayes_learnable(g_true):
    """E[Var(p | k, n)]: the part of Brier the oracle cannot remove at budget n, beyond item noise (for scale)."""
    out = []
    for n in BUD:
        pm, pk = post_mean_table(g_true, n)
        ep2 = sum(pk[k] * 0 for k in range(n + 1))
        lik = np.array([comb(n, k) * GRID ** k * (1 - GRID) ** (n - k) for k in range(n + 1)])
        e2 = (lik * GRID ** 2) @ g_true / np.maximum(lik @ g_true, 1e-300)
        out.append(float(pk @ (e2 - pm ** 2)))
    return np.array(out)


def alc(v):
    return float(np.dot(WTS, v))


def moments(g, kmax=4):
    return [float(g @ GRID ** j) for j in range(1, kmax + 1)]


def beta_match(g):
    m1, m2 = moments(g, 2)
    var = m2 - m1 ** 2
    s = m1 * (1 - m1) / var - 1
    return beta_density(m1 * s, (1 - m1) * s)


def logitnormal_match(g):
    """Find (m, s) of a logit-normal matching the first two moments of g (grid search)."""
    m1, m2 = moments(g, 2)
    best = None
    for m in np.linspace(-6, 3, 181):
        for s in np.linspace(0.05, 5, 100):
            d = density_logitnormal(m, s)
            a, b = moments(d, 2)
            err = (a - m1) ** 2 + (b - m2) ** 2
            if best is None or err < best[0]:
                best = (err, m, s)
    return density_logitnormal(best[1], best[2]), best[1], best[2]


HARD = beta_density(0.4, 20.0)  # mean 0.0196; 'near-zero' pairs (hypothetical)
SCEN = {
    "S1 bimodal 40% hard, rest logit N(-0.2,1.2)": mix((0.40, HARD), (0.60, density_logitnormal(-0.2, 1.2))),
    "S2 bimodal 25% hard, rest logit N(-0.5,1.2)": mix((0.25, HARD), (0.75, density_logitnormal(-0.5, 1.2))),
    "S3 bimodal 55% hard, rest logit N(0.3,1.0)": mix((0.55, HARD), (0.45, density_logitnormal(0.3, 1.0))),
    "S4 unimodal logit N(-1.2,1.6)": density_logitnormal(-1.2, 1.6),
}


def working_priors():
    w = {}
    for m0 in (-2.5, -2.0, -1.5, -1.0):
        w[f"normal m0={m0}"] = model_prior(m0)
    hard_comp = model_prior(-4.0, v_level=0.49)  # mix_m=-4, mix_s=0.7 (approx incl. item moderation)
    for pi, m0 in ((0.25, -1.0), (0.35, -1.0), (0.25, -1.5)):
        w[f"mix pi={pi} m0={m0}"] = mix((pi, hard_comp), (1 - pi, model_prior(m0)))
    w["3-comp robust: .25 hard/.60 m0=-1/.15 vague N(0,3^2)"] = mix(
        (0.25, hard_comp), (0.60, model_prior(-1.0)), (0.15, density_logitnormal(0.0, 3.0)))
    return w


def section_B():
    print("=" * 100)
    print("Check B: excess Brier (x1e-3) of working priors vs oracle Bayes, per budget and ALC-weighted")
    W = working_priors()
    for sname, g in SCEN.items():
        m = moments(g)
        print(f"\n{sname}: mean {m[0]:.3f}, sd {math.sqrt(m[1]-m[0]**2):.3f}; oracle E[Var(p|data)] x1e-3 by budget:",
              " ".join(f"{1e3*x:.1f}" for x in bayes_learnable(g)))
        rows = dict(W)
        rows["oracle-2-moment Beta"] = beta_match(g)
        ln, mm, ss = logitnormal_match(g)
        rows[f"oracle-2-moment logit-normal ({mm:.2f},{ss:.2f})"] = ln
        for name, q in rows.items():
            r = regret(g, q)
            print(f"   {name:58s} ALC {1e3*alc(r):6.2f} | " + " ".join(f"B{b}:{1e3*x:6.2f}" for b, x in zip(BUD, r)))


def section_D():
    print("=" * 100)
    print("Check D: power posterior (likelihood^alpha) with a misspecified unimodal prior (normal m0=-2.0)")
    q = model_prior(-2.0)
    for sname, g in SCEN.items():
        line = []
        for a in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            r = regret(g, q, tempered_alpha=a)
            line.append(f"a={a}: ALC {1e3*alc(r):5.2f} (B1 {1e3*r[1]:5.2f} B3 {1e3*r[2]:5.2f} B7 {1e3*r[3]:5.2f})")
        print(f"  {sname}\n     " + "\n     ".join(line))
    print("  well-specified sanity (truth = prior):", f"{1e3*alc(regret(q, q, 0.75)):.3f}",
          f"{1e3*alc(regret(q, q, 1.0)):.3f}", f"{1e3*alc(regret(q, q, 1.5)):.3f}")


def section_C(rng, n_draws=4000, n_pairs=9):
    """In-draw EB for the hard fraction pi (shared across the pairs of one draw) vs a fixed pi_a.
    Component shapes are known (isolates the value of learning pi). Labels nested across budgets."""
    print("=" * 100)
    print("Check C: draw-level learning of the hard-component weight pi (9 pairs per draw)")
    norm = density_logitnormal(-0.3, 1.2)
    hard = HARD
    cdf_h, cdf_n = np.cumsum(hard), np.cumsum(norm)
    pigrid = np.linspace(0.005, 0.995, 199)

    def comp_tables(n):
        lik = np.array([GRID ** k * (1 - GRID) ** (n - k) for k in range(n + 1)])
        Lh, Ln = lik @ hard, lik @ norm
        Eh, En = (lik * GRID) @ hard / Lh, (lik * GRID) @ norm / Ln
        return Lh, Ln, Eh, En

    tabs = {n: comp_tables(n) for n in BUD}
    settings = [("pool pi=0.40 fixed", lambda: 0.40), ("pool pi=0.25 fixed", lambda: 0.25),
                ("draw pi ~ Beta mean .35 sd .15", lambda: rng.beta(3.15, 5.85))]
    methods = [("fixed pi_a=0.25", 0.25, None), ("fixed pi_a=0.40", 0.40, None),
               ("hier Beta(c=4) mean .25", 0.25, 4.0), ("hier Beta(c=4) mean .40", 0.40, 4.0),
               ("hier Beta(c=10) mean .25", 0.25, 10.0), ("hier Beta(c=10) mean .40", 0.40, 10.0)]
    for sname, pi_fn in settings:
        acc = {m[0]: np.zeros(len(BUD)) for m in methods}
        acc["oracle (true draw pi)"] = np.zeros(len(BUD))
        for _ in range(n_draws):
            pi_t = pi_fn()
            z = rng.random(n_pairs) < pi_t
            u = rng.random(n_pairs)
            p = np.where(z, GRID[np.searchsorted(cdf_h, u).clip(0, len(GRID) - 1)],
                         GRID[np.searchsorted(cdf_n, u).clip(0, len(GRID) - 1)])
            y = (rng.random((n_pairs, 31)) < p[:, None]).astype(int)
            for bi, n in enumerate(BUD):
                k = y[:, :n].sum(1)
                Lh, Ln, Eh, En = tabs[n]
                lh, ln, eh, en = Lh[k], Ln[k], Eh[k], En[k]
                truth_pred = None
                for name, pa, c in methods + [("oracle (true draw pi)", pi_t, None)]:
                    if c is None:
                        wh = pa * lh / (pa * lh + (1 - pa) * ln)
                    else:
                        a, b = c * pa, c * (1 - pa)
                        lp = (a - 1) * np.log(pigrid) + (b - 1) * np.log1p(-pigrid)
                        mj = pigrid[:, None] * lh[None, :] + (1 - pigrid[:, None]) * ln[None, :]  # (grid, pairs)
                        lp = lp + np.log(mj).sum(1)
                        post = np.exp(lp - lp.max()); post /= post.sum()
                        # leave-one-in exact: P(z_j=1|all) = sum_pi post(pi) * pi*lh_j/m_j(pi)
                        wh = (post[:, None] * (pigrid[:, None] * lh[None, :] / mj)).sum(0)
                    pred = wh * eh + (1 - wh) * en
                    if name == "oracle (true draw pi)":
                        truth_pred = pred
                    acc[name][bi] += np.mean((pred - p) ** 2)  # Brier excess over known-p (item noise cancels)
        print(f"\n  {sname}: mean excess Brier over known-p predictor (x1e-3), per budget, and ALC")
        base = acc["oracle (true draw pi)"] / n_draws
        for name, v in acc.items():
            v = v / n_draws
            print(f"     {name:28s} ALC {1e3*alc(v):6.2f} (vs oracle {1e3*(alc(v)-alc(base)):+6.2f}) | "
                  + " ".join(f"B{b}:{1e3*x:5.1f}" for b, x in zip(BUD, v)))


def section_C2(rng, n_draws=6000):
    """Benchmark-level vs pair-level hard component. Draw = 9 pairs over 7 benchmarks (two benchmarks host 2 pairs).
    Truth: hardness is a BENCHMARK property (both pairs on a hard benchmark are hard). Pair-level mixture uses only the
    pair's own labels; benchmark-level mixture shares the indicator across pairs of the same benchmark."""
    print("=" * 100)
    print("Check C2: where the mixture lives (benchmark-level hardness truth), pi=0.35")
    norm = density_logitnormal(-0.3, 1.2)
    hard = HARD
    cdf_h, cdf_n = np.cumsum(hard), np.cumsum(norm)
    pi = 0.35
    bench = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6])
    shared = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
    tabs = {}
    for n in BUD:
        lik = np.array([GRID ** k * (1 - GRID) ** (n - k) for k in range(n + 1)])
        tabs[n] = (lik @ hard, lik @ norm, (lik * GRID) @ hard / (lik @ hard), (lik * GRID) @ norm / (lik @ norm))
    acc = {"pair-level": np.zeros((2, len(BUD))), "benchmark-level": np.zeros((2, len(BUD)))}
    for _ in range(n_draws):
        zb = rng.random(7) < pi
        z = zb[bench]
        u = rng.random(9)
        p = np.where(z, GRID[np.searchsorted(cdf_h, u).clip(0, len(GRID) - 1)],
                     GRID[np.searchsorted(cdf_n, u).clip(0, len(GRID) - 1)])
        y = (rng.random((9, 31)) < p[:, None]).astype(int)
        for bi, n in enumerate(BUD):
            k = y[:, :n].sum(1)
            Lh, Ln, Eh, En = tabs[n]
            lh, ln, eh, en = Lh[k], Ln[k], Eh[k], En[k]
            w_pair = pi * lh / (pi * lh + (1 - pi) * ln)
            # benchmark-level: product of pair likelihoods within benchmark
            lhb = np.ones(7); lnb = np.ones(7)
            for j in range(9):
                lhb[bench[j]] *= lh[j]; lnb[bench[j]] *= ln[j]
            w_b = (pi * lhb / (pi * lhb + (1 - pi) * lnb))[bench]
            for name, w in (("pair-level", w_pair), ("benchmark-level", w_b)):
                pred = w * eh + (1 - w) * en
                e = (pred - p) ** 2
                acc[name][0, bi] += e[shared].mean()
                acc[name][1, bi] += e[~shared].mean()
    for name, v in acc.items():
        v = v / n_draws
        print(f"   {name:16s} shared-benchmark pairs ALC {1e3*alc(v[0]):6.2f} | "
              + " ".join(f"B{b}:{1e3*x:5.1f}" for b, x in zip(BUD, v[0])) + f" || solo pairs ALC {1e3*alc(v[1]):6.2f}")


if __name__ == "__main__":
    rng = np.random.default_rng(20261005)
    section_B()
    section_D()
    section_C(rng)
    section_C2(rng)
