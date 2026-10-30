"""Per-budget logistic recalibration of saved predictions: in-sample oracle and leave-one-benchmark-out (LOBO) cross-fit.
p' = sigmoid(a_b * logit(p) + c_b); fitted by minimising the pair-weighted Brier (rows weighted 1/n_rows of their
pair-draw, matching the pair-equal metric). Macro = bench-equal, k-averaged mean pair ALC (paired.py's convention)."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose import load
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}

def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def fit_ac(z, y, w, free_c=True):
    def f(th):
        a, c = th[0], (th[1] if free_c else 0.0)
        q = 1 / (1 + np.exp(-(a * z + c)))
        return float(np.sum(w * (q - y) ** 2))
    th = minimize(f, [1.0, 0.0] if free_c else [1.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10}).x
    return th[0], (th[1] if free_c else 0.0)

def macro(df, col):
    e = df.assign(e2=(df[col] - df.label) ** 2)
    pb = e.groupby(["bench", "k", "run", "pair", "budget"], observed=True).e2.mean().reset_index()
    pb["w"] = pb.budget.map(W)
    alc = (pb.e2 * pb.w).groupby([pb.bench, pb.k, pb.run, pb.pair]).sum().reset_index(name="alc")
    return alc.groupby(["bench", "k"]).alc.mean().groupby("bench").mean().mean(), alc

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "w20_SCP"
    free_c = (sys.argv[2] != "noc") if len(sys.argv) > 2 else True
    df = load(models=(model,))
    df["z"] = lg(df.pred.values)
    df["w"] = 1.0 / df.groupby(["bench", "k", "run", "pair", "budget"], observed=True).label.transform("size")
    m0, alc0 = macro(df, "pred")
    print(model, "free_c" if free_c else "a only", "macro", round(m0, 5))
    # in-sample oracle per budget
    df["p_in"] = df.pred
    for b in W:
        s = df.budget == b
        a, c = fit_ac(df.z[s].values, df.label[s].values, df.w[s].values, free_c)
        df.loc[s, "p_in"] = 1 / (1 + np.exp(-(a * df.z[s] + c)))
        print(f"  in-sample B{b}: a {a:.3f} c {c:+.3f}")
    # LOBO cross-fit per budget
    df["p_cf"] = df.pred
    for hb in df.bench.unique():
        for b in W:
            tr = (df.budget == b) & (df.bench != hb)
            te = (df.budget == b) & (df.bench == hb)
            a, c = fit_ac(df.z[tr].values, df.label[tr].values, df.w[tr].values, free_c)
            df.loc[te, "p_cf"] = 1 / (1 + np.exp(-(a * df.z[te] + c)))
            if b in (7, 15, 31):
                print(f"  cf hold {hb} B{b}: a {a:.3f} c {c:+.3f}")
    for col in ("p_in", "p_cf"):
        m, alc = macro(df, col)
        d = alc.alc - alc0.alc
        # per-budget macro deltas
        e = df.assign(e0=(df.pred - df.label) ** 2, e1=(df[col] - df.label) ** 2)
        pb = e.groupby(["bench", "k", "run", "pair", "budget"], observed=True)[["e0", "e1"]].mean().reset_index()
        pb["d"] = pb.e1 - pb.e0
        perb = pb.groupby(["budget", "bench", "k"]).d.mean().groupby(["budget", "bench"]).mean().groupby("budget").mean()
        perbench = alc.assign(d=d).groupby(["bench", "k"]).d.mean().groupby("bench").mean()
        # draw-clustered SE of the macro delta: bootstrap over runs within bench
        rng = np.random.default_rng(0)
        dd = alc.assign(d=d)
        reps = []
        for _ in range(300):
            parts = []
            for bn, g in dd.groupby("bench"):
                runs = g.run.unique()
                pick = rng.choice(runs, len(runs))
                gg = pd.concat([g[g.run == r] for r in pick])
                parts.append(gg.groupby("k").d.mean().mean())
            reps.append(np.mean(parts))
        print(f"{col}: macro {m:.5f} delta {1e3*(m-m0):+.3f}e-3 (SE {1e3*np.std(reps):.3f}); per budget x1e-3:",
              {int(k): round(v * 1e3, 2) for k, v in perb.items()}, "per bench:", {k: round(v * 1e3, 2) for k, v in perbench.items()})
