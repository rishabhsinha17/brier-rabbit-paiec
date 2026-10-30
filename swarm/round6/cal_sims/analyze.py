"""Calibration of saved replay predictions (swarm round 6): Brier (bit-exact against the chained pair scores), ECE with
equal-mass bins, the binned Murphy decomposition, the platform-style ECE, draw-clustered bootstrap SEs, and reliability
diagrams per budget.

Input: the <stem>_<mode>.parquet written by replay_preds.py (one row per scored evaluation row: model, bench, k, run,
pair, item, copy, budget, pred, label, seq).

Definitions (all per budget; "scope" = all benchmarks pooled, or one benchmark pooled over k):
- Pair weighting (default): every pair-draw has total weight 1 and its scored rows share it equally, so the weighted
  Brier equals the mean of pair Brier scores, i.e. paired.py's "per-budget Brier (pooled over pairs)". --weighting row
  gives every scored row weight 1 instead (sensitivity only).
- Equal-mass bins: predictions sorted, each distinct value assigned by the midpoint of its cumulative weight to one of
  --bins bins (ties never split), so each bin holds about 1/bins of the scope's weight.
- ECE = sum_k W_k |pbar_k - obar_k| / W over those bins.
- Murphy (binned): Brier = REL - RES + UNC + R with REL = sum_k W_k (pbar_k - obar_k)^2 / W, RES = sum_k W_k (obar_k -
  obar)^2 / W, UNC = obar (1 - obar), and R = WBV - 2 WBC the within-bin terms (Stephenson et al. 2008), which vanish
  when every bin holds a single forecast value. The identity is checked to 1e-12 for every cell.
- Platform-style ECE: the leaderboard's diagnostic (ten fixed bins [0, .1), ..., [.9, 1], count-weighted, computed
  within each pair, then the unweighted mean over pairs); recomputed from the rows with paiec.protocol.ece.
- ALC column: the ALC-weighted sum over budgets (0.1/0.2/0.2/0.2/0.2/0.1); Brier, REL, RES and R are linear in the
  budget weights, so the ALC row decomposes the pooled Brier ALC the same way.
- SEs: draw-clustered bootstrap (resample draws, i.e. (benchmark, k, run), with replacement within each benchmark x k
  stratum; the same replicates for every model, so differences are paired). Bins are re-formed in every replicate.

Usage (repo root; about 1 CPU-minute for a full mode at --boot 200, i.e. a few wall-clock minutes at background
QoS on a busy laptop; --boot 50 --no-figs measured 17 s CPU on a 0.94M-row file):
  ../.venv/bin/python swarm/round6/cal_sims/analyze.py --preds results/cal_preds_bench.parquet \
      --ref-scores results/combo_v1.csv --outdir swarm/round6/cal_sims/out_bench
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paiec.protocol import BUDGETS, WEIGHTS, ece as platform_ece, pair_alc  # noqa: E402

LABELS = {"base": "base (REF)", "SCP_combo": "REF+SCP", "w20_SCP": "W20+SCP", "smoothed_k2": "smoothed mean (k2)",
          "empirical_mean": "empirical mean"}
# Reference categorical palette (dataviz skill), slots 1-5 in fixed order; slots 1-3 validated all-pairs (light).
COLORS = {"base": "#2a78d6", "SCP_combo": "#eb6834", "w20_SCP": "#1baf7a", "smoothed_k2": "#eda100",
          "empirical_mean": "#e87ba4"}
MARKERS = {"base": "o", "SCP_combo": "s", "w20_SCP": "^", "smoothed_k2": "D", "empirical_mean": "v"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STATS = ("brier", "ece", "rel", "res", "unc", "resid", "wbv", "wbc", "mean_pred", "base_rate", "sd_pred")
LINEAR = ("brier", "ece", "rel", "res", "unc", "resid", "mean_pred", "base_rate")  # ALC-weighted sums reported


def label(m: str) -> str:
    return LABELS.get(m, m)


# ----------------------------------------------------------------------------------------------------------------
# loading and the bit-exact pair-level recomputation
# ----------------------------------------------------------------------------------------------------------------
def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")
    need = {"model", "bench", "k", "run", "pair", "item", "copy", "budget", "pred", "label", "seq"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    for c in ("model", "bench", "pair", "item"):
        df[c] = df[c].astype("category")
    # run_protocol's record order inside every (model, bench, k, run); (budget, pair) groups are contiguous in it
    return df.sort_values(["model", "bench", "k", "run", "seq"], kind="stable").reset_index(drop=True)


def _codes(s: pd.Series) -> np.ndarray:
    return s.cat.codes.to_numpy() if isinstance(s.dtype, pd.CategoricalDtype) else s.to_numpy()


def _platform_floor(ps: np.ndarray, rng, sims: int) -> float:
    """Expected platform-style ECE of a perfectly calibrated forecaster with these predictions: every scored row's
    label drawn as Bernoulli(pred), ten fixed bins, mean over `sims` draws (ECE = sum_b |sum p - sum y| / n)."""
    idx = np.minimum((ps * 10).astype(int), 9)
    onehot = np.zeros((len(ps), 10))
    onehot[np.arange(len(ps)), idx] = 1.0
    ysim = (rng.random((sims, len(ps))) < ps).astype(float)
    return float(np.mean(np.abs(ysim @ onehot - ps @ onehot).sum(axis=1)) / len(ps))


def pair_level(df: pd.DataFrame, sims: int = 0, seed: int = 0) -> pd.DataFrame:
    """Per (pair-draw, model): Brier and platform ECE per budget, recomputed exactly as run_protocol does
    (np.mean over the record-ordered rows; paiec.protocol.ece), and the ALC via paiec.protocol.pair_alc. With sims > 0
    also the platform-ECE noise floor f0..f31 (see _platform_floor)."""
    keys = ["model", "bench", "k", "run", "budget", "pair"]
    codes = np.column_stack([_codes(df[c]) for c in keys])
    change = np.ones(len(df), dtype=bool)
    change[1:] = (codes[1:] != codes[:-1]).any(axis=1)
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(df))
    p = df["pred"].to_numpy(np.float64)
    y = df["label"].to_numpy(np.float64)
    items = _codes(df["item"])
    rec = []
    rng = np.random.default_rng(seed)
    for s, e in zip(starts, ends):
        ps, ys = p[s:e], y[s:e]
        rec.append((float(np.mean((ps - ys) ** 2)), platform_ece(ps, ys), e - s, len(np.unique(items[s:e])),
                    _platform_floor(ps, rng, sims) if sims > 0 else np.nan))
    head = df.iloc[starts][keys].reset_index(drop=True)
    head[["brier", "ece", "rows", "unique", "floor"]] = pd.DataFrame(
        rec, columns=["brier", "ece", "rows", "unique", "floor"])
    for c in ("model", "bench", "pair"):
        head[c] = head[c].astype(str)
    # unstack, not pivot_table(dropna=False): the latter reindexes to the cartesian product of the index levels
    wide = head.set_index(["bench", "k", "run", "pair", "model", "budget"])[["brier", "ece", "floor"]].unstack("budget")
    out = pd.DataFrame(index=wide.index)
    for b in BUDGETS:
        out[f"b{b}"] = wide[("brier", b)]
    out.insert(0, "alc", [pair_alc({b: {"brier": row[f"b{b}"]} for b in BUDGETS}) for _, row in out.iterrows()])
    for b in BUDGETS:
        out[f"e{b}"] = wide[("ece", b)]
    out["ece_alc"] = sum(WEIGHTS[b] * out[f"e{b}"] for b in BUDGETS)
    for b in BUDGETS:
        out[f"f{b}"] = wide[("floor", b)]
    out["floor_alc"] = sum(WEIGHTS[b] * out[f"f{b}"] for b in BUDGETS)
    rows = head[head.budget == 0].set_index(["bench", "k", "run", "pair", "model"])
    out["rows"] = rows["rows"]
    out["unique"] = rows["unique"]
    return out.reset_index()


def compare(ps: pd.DataFrame, ref_path: Path, cols: list[str]) -> list[str]:
    """Bitwise comparison with a reference pair-score CSV on overlapping (bench, k, run, pair, model) keys."""
    if not ref_path.exists():
        return [f"{ref_path.name}: not found, skipped"]
    ref = pd.read_csv(ref_path, float_precision="round_trip")
    cols = [c for c in cols if c in ref.columns]
    keys = ["bench", "k", "run", "pair", "model"]
    m = ps[keys + cols].merge(ref[keys + cols], on=keys, how="left", suffixes=("", "_ref"))
    out = []
    for model, g in m.groupby("model", sort=False):
        hit = g[cols[0] + "_ref"].notna().to_numpy()
        if not hit.any():
            out.append(f"{ref_path.name}: {model}: not in the reference file")
            continue
        a = g.loc[hit, cols].to_numpy(float)
        b = g.loc[hit, [c + "_ref" for c in cols]].to_numpy(float)
        exact = int((a == b).all(axis=1).sum())
        out.append(f"{ref_path.name}: {model}: {hit.sum()}/{len(g)} pair-draws matched on {len(cols)} columns, "
                   f"bit-identical {exact}/{hit.sum()}, max|diff| {np.abs(a - b).max():.3g}"
                   + ("  EXACT" if exact == hit.sum() else "  MISMATCH"))
    return out


def chained_aggregates(ps: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """paired.py's two printed aggregates: macro ALC (bench-equal, k-averaged) and per-budget Brier pooled over pairs."""
    wide = ps.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values="alc")
    macro = wide.groupby(level=[0, 1]).mean().groupby(level=0).mean().mean()
    budgets = ps.groupby("model")[[f"b{b}" for b in BUDGETS]].mean()
    return macro, budgets


def compare_aggregates(ps: pd.DataFrame, ref_path: Path) -> list[str]:
    """The same two aggregates from the reference CSV on the same pair-draws and models, against ours."""
    if not ref_path.exists():
        return []
    keys = ["bench", "k", "run", "pair", "model"]
    ref = pd.read_csv(ref_path, float_precision="round_trip")
    ref = ref.merge(ps[keys].astype({"k": "int64", "run": "int64"}), on=keys, how="inner")
    if ref.empty:
        return []
    ours = ps.astype({"k": "int64", "run": "int64"}).merge(ref[keys], on=keys, how="inner")
    m_r, b_r = chained_aggregates(ref)
    m_o, b_o = chained_aggregates(ours)
    models = sorted(set(m_r.index))
    dm = float(np.abs(m_o[models] - m_r[models]).max())
    db = float(np.abs(b_o.loc[models].to_numpy(float) - b_r.loc[models].to_numpy(float)).max())
    return [f"{ref_path.name}: aggregates on the same {len(ref) // max(len(models), 1)} pair-draws ({', '.join(models)}): "
            f"macro ALC max|diff| {dm:.3g}, per-budget pooled Brier max|diff| {db:.3g}"]


# ----------------------------------------------------------------------------------------------------------------
# calibration statistics on item-aggregated weights
# ----------------------------------------------------------------------------------------------------------------
def item_table(df: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """One row per (model, budget, pair-draw, evaluation item): prediction, recorded runs n, positives s, weights."""
    g = df.groupby(["model", "budget", "bench", "k", "run", "pair", "item"], observed=True, sort=False)
    t = g.agg(n=("label", "size"), s=("label", "sum"), p=("pred", "first"), pmin=("pred", "min"),
              pmax=("pred", "max")).reset_index()
    spread = float((t["pmax"] - t["pmin"]).abs().max())
    if spread != 0.0:
        raise SystemExit(f"recorded runs of one item carry different predictions (max spread {spread})")
    if weighting == "pair":
        rows = t.groupby(["model", "budget", "bench", "k", "run", "pair"], observed=True)["n"].transform("sum")
        t["w"] = t["n"] / rows
        t["v"] = t["s"] / rows
    else:
        t["w"] = t["n"].astype(float)
        t["v"] = t["s"].astype(float)
    for c in ("model", "bench", "pair"):
        t[c] = t[c].astype(str)
    return t.drop(columns=["pmin", "pmax"])


def binned(u: np.ndarray, Wu: np.ndarray, Vu: np.ndarray, nbins: int):
    """Statistics of one cell from distinct sorted forecasts u with total weight Wu and weighted positives Vu."""
    W = float(Wu.sum())
    cum = np.cumsum(Wu)
    b = np.minimum(((cum - 0.5 * Wu) * (nbins / W)).astype(np.int64), nbins - 1)
    Wb = np.bincount(b, Wu, nbins)
    Vb = np.bincount(b, Vu, nbins)
    Sb = np.bincount(b, Wu * u, nbins)
    ok = Wb > 0
    safe = np.where(ok, Wb, 1.0)
    pbar = np.where(ok, Sb / safe, 0.0)
    obar = np.where(ok, Vb / safe, 0.0)
    rate = float(Vb.sum() / W)
    dp = u - pbar[b]
    st = {
        "brier": float(np.sum(Wu * u * u - 2.0 * u * Vu + Vu) / W),
        "ece": float(np.sum(Wb * np.abs(pbar - obar)) / W),
        "rel": float(np.sum(Wb * (pbar - obar) ** 2) / W),
        "res": float(np.sum(Wb * (obar - rate) ** 2) / W),
        "unc": rate * (1.0 - rate),
        "wbv": float(np.sum(Wu * dp * dp) / W),
        "wbc": float(np.sum(dp * (Vu - Wu * obar[b])) / W),
        "mean_pred": float(Sb.sum() / W),
        "base_rate": rate,
    }
    st["resid"] = st["wbv"] - 2.0 * st["wbc"]
    st["sd_pred"] = float(np.sqrt(max(np.sum(Wu * (u - st["mean_pred"]) ** 2) / W, 0.0)))
    return st, (np.where(ok, pbar, np.nan), np.where(ok, obar, np.nan), Wb / W, b)


def bootstrap_matrix(clusters: pd.DataFrame, reps: int, seed: int) -> np.ndarray:
    """(reps x clusters) multiplicities: draws resampled with replacement within each (bench, k) stratum."""
    rng = np.random.default_rng(seed)
    M = np.zeros((reps, len(clusters)), dtype=np.float64)
    for _, idx in clusters.groupby(["bench", "k"]).indices.items():
        n = len(idx)
        draws = rng.integers(0, n, size=(reps, n))
        for r in range(reps):
            M[r, idx] = np.bincount(draws[r], minlength=n)
    return M


def calibrate(items: pd.DataFrame, scopes: list[str], models: list[str], nbins: int, M: np.ndarray | None,
              cl_index: dict, extra_bins: tuple = (), sims: int = 0, seed: int = 0):
    """Point estimates, bootstrap replicates and reliability bins for every (scope, model, budget). With sims > 0
    also the noise floor of the pooled ECE and REL: their mean over `sims` label draws of a perfectly calibrated
    forecaster (every recorded run ~ Bernoulli(pred)) with the same predictions, weights and bins."""
    point, reps, bins, sens = {}, {}, [], []
    rng = np.random.default_rng(seed)
    cl = np.array([cl_index[(b, k, r)] for b, k, r in zip(items["bench"], items["k"], items["run"])])
    groups = items.groupby(["model", "budget"], sort=False).indices
    for scope in scopes:
        in_scope = np.ones(len(items), dtype=bool) if scope == "all" else (items["bench"].to_numpy() == scope)
        for m in models:
            for bud in BUDGETS:
                idx = groups.get((m, bud))
                if idx is None:
                    continue
                idx = idx[in_scope[idx]]
                if not len(idx):
                    continue
                p = items["p"].to_numpy()[idx]
                w = items["w"].to_numpy()[idx]
                v = items["v"].to_numpy()[idx]
                n = items["n"].to_numpy()[idx]
                c = cl[idx]
                u, inv = np.unique(p, return_inverse=True)
                st, (pb, ob, share, bmap) = binned(u, np.bincount(inv, w, len(u)), np.bincount(inv, v, len(u)), nbins)
                gap = st["brier"] - (st["rel"] - st["res"] + st["unc"] + st["resid"])
                if abs(gap) > 1e-12:
                    raise SystemExit(f"Murphy identity off by {gap:.3g} at {scope} {m} B{bud}")
                st["n_pairs"] = int(len(set(zip(items["bench"].to_numpy()[idx], items["k"].to_numpy()[idx],
                                                items["run"].to_numpy()[idx], items["pair"].to_numpy()[idx]))))
                st["n_items"] = int(len(idx))
                st["n_rows"] = int(n.sum())
                if sims > 0:
                    Wu = np.bincount(inv, w, len(u))
                    fl = [binned(u, Wu, np.bincount(inv, rng.binomial(n, p) * (w / n), len(u)), nbins)[0]
                          for _ in range(sims)]
                    st["ece_floor"] = float(np.mean([f["ece"] for f in fl]))
                    st["rel_floor"] = float(np.mean([f["rel"] for f in fl]))
                point[(scope, m, bud)] = st
                cnt = np.bincount(bmap, np.bincount(inv, minlength=len(u)).astype(float), nbins)
                rws = np.bincount(bmap, np.bincount(inv, n.astype(float), len(u)), nbins)
                for j in range(nbins):
                    if share[j] > 0:
                        bins.append({"scope": scope, "model": m, "budget": bud, "bin": j, "weight_share": share[j],
                                     "mean_pred": pb[j], "obs_rate": ob[j], "items": int(cnt[j]), "rows": int(rws[j])})
                for nb in extra_bins:
                    s2, _ = binned(u, np.bincount(inv, w, len(u)), np.bincount(inv, v, len(u)), nb)
                    sens.append({"scope": scope, "model": m, "budget": bud, "bins": nb,
                                 **{k: s2[k] for k in ("ece", "rel", "res", "resid")}})
                if M is not None:
                    R = np.empty((len(M), len(STATS)))
                    for r in range(len(M)):
                        mult = M[r, c]
                        s_r, _ = binned(u, np.bincount(inv, w * mult, len(u)), np.bincount(inv, v * mult, len(u)),
                                        nbins)
                        R[r] = [s_r[k] for k in STATS]
                    reps[(scope, m, bud)] = R
    return point, reps, pd.DataFrame(bins), pd.DataFrame(sens)


def long_table(point: dict, reps: dict, ps: pd.DataFrame, M: np.ndarray | None, cl_index: dict, base: str):
    """One row per (scope, model, budget in BUDGETS + 'alc'): point stats, SEs, differences vs base with SEs, and the
    platform-style ECE (mean over pairs) with its SE."""
    rows = []
    scopes = sorted({k[0] for k in point}, key=lambda s: (s != "all", s))
    models = list(dict.fromkeys(k[1] for k in point))
    pcl = np.array([cl_index[(b, k, r)] for b, k, r in zip(ps["bench"], ps["k"], ps["run"])])
    for scope in scopes:
        for m in models:
            if not all((scope, m, b) in point for b in BUDGETS):
                continue
            sel = ((ps["model"] == m) & ((ps["bench"] == scope) if scope != "all" else True)).to_numpy()
            ecols = [f"e{b}" for b in BUDGETS] + ["ece_alc"]
            E = ps.loc[sel, ecols].to_numpy(float)
            F = ps.loc[sel, [f"f{b}" for b in BUDGETS] + ["floor_alc"]].to_numpy(float) if "f0" in ps else None
            Ew = None if M is None else (M[:, pcl[sel]] @ E) / M[:, pcl[sel]].sum(axis=1, keepdims=True)
            for j, bud in enumerate(list(BUDGETS) + ["alc"]):
                if bud == "alc":
                    st = {k: sum(WEIGHTS[b] * point[(scope, m, b)][k] for b in BUDGETS) for k in LINEAR}
                    st.update({k: point[(scope, m, 0)][k] for k in ("n_pairs", "n_items", "n_rows")})
                    wvec = np.array([WEIGHTS[b] for b in BUDGETS])
                    R = None if M is None else np.tensordot(np.stack([reps[(scope, m, b)] for b in BUDGETS]), wvec,
                                                            axes=([0], [0]))
                    Rb = None if (M is None or m == base) else np.tensordot(
                        np.stack([reps[(scope, base, b)] for b in BUDGETS]), wvec, axes=([0], [0]))
                    pb = {k: sum(WEIGHTS[b] * point[(scope, base, b)][k] for b in BUDGETS) for k in LINEAR} \
                        if (scope, base, 0) in point else None
                else:
                    st = dict(point[(scope, m, bud)])
                    R = None if M is None else reps[(scope, m, bud)]
                    Rb = None if (M is None or m == base or (scope, base, bud) not in reps) else reps[(scope, base, bud)]
                    pb = point.get((scope, base, bud))
                row = {"scope": scope, "model": m, "budget": bud, "n_pairs": st["n_pairs"], "n_items": st["n_items"],
                       "n_rows": st["n_rows"]}
                for k in LINEAR + (("sd_pred",) if bud != "alc" else ()):
                    row[k] = st[k]
                    if R is not None:
                        row[k + "_se"] = float(R[:, STATS.index(k)].std(ddof=1))
                    if pb is not None and m != base:
                        row["d_" + k] = st[k] - pb[k]
                        if Rb is not None:
                            row["d_" + k + "_se"] = float((R[:, STATS.index(k)] - Rb[:, STATS.index(k)]).std(ddof=1))
                row["ece_platform"] = float(E[:, j].mean())
                if F is not None and not np.isnan(F[:, j]).all():
                    row["ece_platform_floor"] = float(F[:, j].mean())
                for k in ("ece_floor", "rel_floor"):
                    if bud == "alc" and all(k in point[(scope, m, b)] for b in BUDGETS):
                        row[k] = sum(WEIGHTS[b] * point[(scope, m, b)][k] for b in BUDGETS)
                    elif bud != "alc" and k in st:
                        row[k] = st[k]
                if Ew is not None:
                    row["ece_platform_se"] = float(Ew[:, j].std(ddof=1))
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)


# Coinciding curves stay visible: series are drawn in plot order with decreasing stroke and marker size (a wide
# first stroke shows as a halo around the later ones wherever they overlap).
LAYER_LW = (3.2, 2.0, 1.2, 1.0, 1.0)
LAYER_MS = (8.5, 6.5, 4.8, 4.5, 4.5)


def _layer(i: int) -> tuple[float, float]:
    return LAYER_LW[min(i, len(LAYER_LW) - 1)], LAYER_MS[min(i, len(LAYER_MS) - 1)]


def reliability_panel(ax, bins: pd.DataFrame, lt: pd.DataFrame, models: list[str], bud: int, legend: bool = True):
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, zorder=1, label="perfect calibration")
    for i, m in enumerate(models):
        b = bins[(bins.scope == "all") & (bins.model == m) & (bins.budget == bud)].sort_values("bin")
        if b.empty:
            continue
        r = lt[(lt.scope == "all") & (lt.model == m) & (lt.budget == bud)].iloc[0]
        se = f" ± {r['ece_se']:.3f}" if "ece_se" in r and pd.notna(r.get("ece_se")) else ""
        lw, ms = _layer(i)
        ax.plot(b.mean_pred, b.obs_rate, color=COLORS.get(m, INK2), linewidth=lw, marker=MARKERS.get(m, "o"),
                markersize=ms, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3 + i,
                label=f"{label(m)}  ECE {r['ece']:.3f}{se}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"B{bud}", color=INK, fontsize=10, loc="left")
    if legend:
        ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK)


def hist_panel(ax, items: pd.DataFrame, models: list[str], bud: int):
    _style(ax)
    edges = np.linspace(0, 1, 21)
    for i, m in enumerate(models):
        t = items[(items.model == m) & (items.budget == bud)]
        if t.empty:
            continue
        h, _ = np.histogram(t["p"], bins=edges, weights=t["w"])
        ax.step(edges, np.append(h, h[-1]) / h.sum() * 100, where="post", color=COLORS.get(m, INK2),
                linewidth=_layer(i)[0], zorder=3 + i)
    ax.set_xlim(0, 1)
    ax.set_ylabel("% of weight", color=INK2, fontsize=8)
    ax.set_xlabel("predicted probability (prediction distribution, 0.05 bins)", color=INK2, fontsize=8)


def _footer(fig, text: str, width: int) -> None:
    import textwrap

    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(text, width)), color=MUTED, fontsize=6, va="bottom")


def make_figures(outdir: Path, bins: pd.DataFrame, lt: pd.DataFrame, items: pd.DataFrame, models: list[str],
                 title: str, base: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    files = []
    for bud in BUDGETS:  # one PNG per budget: reliability on top, prediction distribution (sharpness) below
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.8, 6.6), gridspec_kw={"height_ratios": [3.2, 1]})
        reliability_panel(a1, bins, lt, models, bud)
        a1.set_title(f"Reliability at budget {bud}", color=INK, fontsize=10, loc="left")
        a1.set_xlabel("mean prediction in bin (10 equal-mass bins)", color=INK2, fontsize=8)
        a1.set_ylabel("observed success rate", color=INK2, fontsize=8)
        hist_panel(a2, items, models, bud)
        _footer(fig, title, 95)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        f = outdir / f"reliability_b{bud}.png"
        fig.savefig(f, dpi=200)
        plt.close(fig)
        files.append(f.name)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.8))
    for ax, bud in zip(axes.ravel(), BUDGETS):
        reliability_panel(ax, bins, lt, models, bud)
    for ax in axes[1]:
        ax.set_xlabel("mean prediction in bin", color=INK2, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("observed success rate", color=INK2, fontsize=8)
    fig.suptitle("Reliability by label budget (10 equal-mass bins, pair-weighted)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    _footer(fig, title, 200)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=2.0)
    f = outdir / "reliability_grid.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    files.append(f.name)
    # Top row: ECE / REL / RES by budget (one axis per panel). Bottom row: each other model minus base, with +-1
    # paired draw-clustered bootstrap SE (levels share most of their draw noise, differences do not).
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    xs = np.arange(len(BUDGETS))
    a = lt[(lt.scope == "all") & (lt.budget.isin(BUDGETS))]
    for j, (k, name, scale) in enumerate([("ece", "ECE", 1.0), ("rel", "REL (x1e-3)", 1e3),
                                          ("res", "RES (x1e-3)", 1e3)]):
        top, bot = axes[0, j], axes[1, j]
        _style(top)
        _style(bot)
        bot.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
        for i, m in enumerate(models):
            r = a[a.model == m]
            if r.empty:
                continue
            r = r.set_index("budget").loc[list(BUDGETS)]
            lw, ms = _layer(i)
            top.plot(xs, r[k].to_numpy(float) * scale, color=COLORS.get(m, INK2), linewidth=lw,
                     marker=MARKERS.get(m, "o"), markersize=ms * 0.8, markeredgecolor=SURFACE, markeredgewidth=1.0,
                     label=label(m), zorder=3 + i)
            if m != base and "d_" + k in r:
                d = r["d_" + k].to_numpy(float) * 1e3
                bot.plot(xs, d, color=COLORS.get(m, INK2), linewidth=2.0, marker=MARKERS.get(m, "o"), markersize=5.5,
                         markeredgecolor=SURFACE, markeredgewidth=1.0, label=label(m), zorder=3 + i)
                if "d_" + k + "_se" in r:
                    e = r["d_" + k + "_se"].to_numpy(float) * 1e3
                    bot.fill_between(xs, d - e, d + e, color=COLORS.get(m, INK2), alpha=0.12, linewidth=0)
        for ax in (top, bot):
            ax.set_xticks(xs, [str(b) for b in BUDGETS])
        bot.set_xlabel("label budget", color=INK2, fontsize=8)
        top.set_title(name, color=INK, fontsize=10, loc="left")
        bot.set_title(f"{name.split(' ')[0]} minus {label(base)} (x1e-3)", color=INK, fontsize=10, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=INK)
    axes[1, 0].legend(fontsize=7, frameon=False, labelcolor=INK)
    _footer(fig, title + "; bands = +-1 paired draw-clustered bootstrap SE", 200)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.5)
    f = outdir / "murphy_by_budget.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    files.append(f.name)
    return files


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def _fmt(x, nd=4, scale=1.0):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * scale:.{nd}f}"


def _pm(r, k, nd, scale=1.0):
    v = r.get(k)
    se = r.get(k + "_se")
    if v is None or pd.isna(v):
        return ""
    return f"{v * scale:.{nd}f}" + (f" ± {se * scale:.{nd}f}" if se is not None and pd.notna(se) else "")


def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_summary(outdir: Path, meta: dict, checks: list[str], macro: pd.Series, budg: pd.DataFrame, lt: pd.DataFrame,
                  sens: pd.DataFrame, models: list[str], base: str, final: str, figs: list[str],
                  cells: pd.DataFrame | None = None) -> str:
    bl = list(BUDGETS) + ["alc"]
    cols = [f"B{b}" for b in BUDGETS] + ["ALC"]
    L = [f"# Calibration of {meta['preds']}", "",
         f"Generated {meta['created_utc']} by swarm/round6/cal_sims/analyze.py. Weighting: {meta['weighting']}; "
         f"{meta['bins']} equal-mass bins; draw-clustered bootstrap B = {meta['boot']} (seed {meta['seed']}). "
         f"{meta['n_draws']} draws, {meta['n_pair_draws']} pair-draws per model; ALC = budget-weighted "
         "(0.1/0.2/0.2/0.2/0.2/0.1) sum.", "", "## 1. Reproduction of the chained pair scores (bitwise)", ""]
    L += [f"- {c}" for c in checks] + [""]
    L += ["## 2. Brier", "", "Pooled over pair-draws (paired.py's per-budget table) and macro ALC (bench-equal, "
          "k-averaged; paired.py's macro line).", ""]
    rows = [[label(m)] + [f"{budg.loc[m, f'b{b}']:.4f}" for b in BUDGETS] + [f"{macro[m]:.4f}"] for m in models
            if m in budg.index]
    L += [md_table(["model"] + cols[:-1] + ["macro ALC"], rows), ""]
    if cells is not None:  # mirrors the first table of the paired.py log (mean pair ALC per benchmark x k)
        L += ["Mean pair ALC per benchmark and k (the first table of the paired.py log):", ""]
        L += [md_table(["bench", "k", "n"] + [label(m) for m in models if m in cells.columns],
                       [[b, str(k), str(int(cells.loc[(b, k), "n"]))] +
                        [f"{cells.loc[(b, k), m]:.4f}" for m in models if m in cells.columns]
                        for b, k in cells.index]), ""]
    a = lt[lt.scope == "all"]

    def block(title, key, nd, scale, note=""):
        L.extend([f"## {title}", ""] + ([note, ""] if note else []))
        rr = []
        for m in models:
            r = a[a.model == m].set_index("budget")
            if r.empty:
                continue
            rr.append([label(m)] + [_pm(r.loc[b].to_dict(), key, nd, scale) for b in bl])
        L.extend([md_table(["model"] + cols, rr), ""])

    block("3. ECE, 10 equal-mass bins (all benchmarks pooled, pair-weighted)", "ece", 4, 1.0,
          "± draw-clustered bootstrap SE (bins re-formed in every replicate).")
    if "ece_floor" in a.columns:
        L.extend(["Noise floor: the same ECE for a perfectly calibrated forecaster with the same predictions, weights "
                  "and bins (every recorded run ~ Bernoulli(pred); mean over the simulated label draws). ECE minus "
                  "this floor is the part sampling noise does not explain.", ""])
        L.extend([md_table(["model"] + cols, [[label(m)] + [_fmt(a[(a.model == m) & (a.budget == b)]["ece_floor"]
                                                                  .iloc[0], 4) for b in bl]
                                              for m in models if not a[a.model == m].empty]), ""])
    unc = a[a.budget != "alc"]["unc"]
    L += ["## 4. Murphy decomposition (binned), x1e-3", "",
          f"Brier = REL - RES + UNC + R, R = within-bin terms (WBV - 2 WBC). UNC = {unc.iloc[0] * 1e3:.3f} for every "
          f"model and budget (identical rows and weights; absolute spread {unc.max() - unc.min():.1g}). Identity "
          "verified to 1e-12 in every cell. Bias = mean prediction - observed rate.", ""]
    rr = []
    for m in models:
        r = a[a.model == m].set_index("budget")
        for b in bl:
            x = r.loc[b].to_dict()
            rr.append([label(m), f"B{b}" if b != "alc" else "ALC", _fmt(x["brier"], 2, 1e3), _pm(x, "rel", 2, 1e3),
                       _pm(x, "res", 2, 1e3), _fmt(x["resid"], 2, 1e3), _fmt(x["mean_pred"] - x["base_rate"], 2, 1e3),
                       _fmt(x.get("sd_pred"), 3) if b != "alc" else ""])
    L += [md_table(["model", "budget", "Brier", "REL", "RES", "R", "bias", "SD(p)"], rr), ""]
    L += [f"## 5. Differences vs {label(base)} (x1e-3, paired bootstrap SE)", ""]
    rr = []
    for m in models:
        if m == base:
            continue
        r = a[a.model == m].set_index("budget")
        for b in bl:
            x = r.loc[b].to_dict()
            rr.append([label(m), f"B{b}" if b != "alc" else "ALC"] +
                      [_pm({"v": x.get("d_" + k), "v_se": x.get("d_" + k + "_se")}, "v", 2, 1e3)
                       for k in ("brier", "ece", "rel", "res", "resid")])
    L += [md_table(["model", "budget", "dBrier", "dECE", "dREL", "dRES", "dR"], rr), ""]
    L += ["## 6. Per benchmark (pooled over k, pair-weighted; x1e-3 except ECE)", ""]
    for scope in [s for s in dict.fromkeys(lt.scope) if s != "all"]:
        sb = lt[lt.scope == scope]
        L += [f"### {scope} ({int(sb.n_pairs.iloc[0])} pair-draws, base rate {sb.base_rate.iloc[0]:.3f})", ""]
        rr = []
        for m in models:
            r = sb[sb.model == m].set_index("budget")
            if r.empty:
                continue
            for k, nd, sc in (("brier", 2, 1e3), ("ece", 4, 1.0), ("rel", 2, 1e3), ("res", 2, 1e3)):
                rr.append([label(m), k.upper() if k != "brier" else "Brier"] +
                          [_pm(r.loc[b].to_dict(), k, nd, sc) for b in bl])
            rr.append([label(m), "bias"] + [_fmt(r.loc[b, "mean_pred"] - r.loc[b, "base_rate"], 2, 1e3) for b in bl])
        L += [md_table(["model", "stat"] + cols, rr), ""]
    block("7. Platform-style ECE (ten fixed bins within each pair, unweighted mean over pair-draws)", "ece_platform",
          4, 1.0, "The leaderboard's ECE diagnostic. Pairs hold 44-113 evaluation items, so within-pair binning noise "
          "dominates it; it is not comparable to the pooled ECE of section 3. ALC column = the ALC-weighted per-pair "
          "ECE, the leaderboard's ECE column.")
    if "ece_platform_floor" in a.columns:
        L.extend(["Noise floor of the platform-style ECE (perfectly calibrated forecaster, same predictions; simulated):",
                  ""])
        L.extend([md_table(["model"] + cols, [[label(m)] + [_fmt(a[(a.model == m) & (a.budget == b)]
                                                                  ["ece_platform_floor"].iloc[0], 4) for b in bl]
                                              for m in models if not a[a.model == m].empty]), ""])
    if not sens.empty:
        L += ["## 8. Bin-count sensitivity (all benchmarks pooled; x1e-3 except ECE)", ""]
        rr = []
        for (m, nb), g in sens[sens.scope == "all"].groupby(["model", "bins"], sort=False):
            g = g.set_index("budget")
            rr.append([label(m), str(nb)] + [f"{g.loc[b, 'ece']:.4f} / {g.loc[b, 'rel'] * 1e3:.2f} / "
                                             f"{g.loc[b, 'res'] * 1e3:.2f}" for b in BUDGETS])
        L += ["ECE / REL / RES with other equal-mass bin counts.", "",
              md_table(["model", "bins"] + cols[:-1], rr), ""]
    L += ["## Files", ""] + [f"- {f}" for f in ["calib_long.csv", "bins.csv", "pair_scores_recomputed.csv",
                                                  "sensitivity.csv", "report_snippet.md"] + figs]
    text = "\n".join(L) + "\n"
    (outdir / "summary.md").write_text(text)
    return text


def write_snippet(outdir: Path, lt: pd.DataFrame, meta: dict, base: str, final: str) -> str:
    a = lt[lt.scope == "all"].set_index(["model", "budget"])
    if final is None or final == base or (final, "alc") not in a.index or (base, "alc") not in a.index:
        return ""
    f, b = a.loc[(final, "alc")], a.loc[(base, "alc")]
    f0, b0 = a.loc[(final, 0)], a.loc[(base, 0)]
    f31, b31 = a.loc[(final, 31)], a.loc[(base, 31)]

    def se(r, k):
        v = r.get(k + "_se")
        return f" ± {v * 1e3:.2f}" if v is not None and pd.notna(v) else ""

    txt = (
        f"**Calibration.** On the local replay ({meta['mode_note']}, {meta['n_pair_draws']} pair-draws per model, "
        f"pair-equal weights, {meta['bins']} equal-mass bins), the ECE of {label(final)} is {f0['ece']:.4f} at B0 and "
        f"{f31['ece']:.4f} at B31 against {b0['ece']:.4f} and {b31['ece']:.4f} for {label(base)} (ALC-weighted "
        f"{f['ece']:.4f} against {b['ece']:.4f}). In the binned Murphy decomposition (Brier = REL - RES + UNC + R, "
        f"UNC = {f['unc'] * 1e3:.2f} x 1e-3 for every model and budget), the ALC difference of "
        f"{f['d_brier'] * 1e3:+.2f}{se(f, 'd_brier')} x 1e-3 splits into REL {f['d_rel'] * 1e3:+.2f}{se(f, 'd_rel')}, "
        f"RES {f['d_res'] * 1e3:+.2f}{se(f, 'd_res')} (a RES increase lowers Brier) and within-bin terms "
        f"{f['d_resid'] * 1e3:+.2f}{se(f, 'd_resid')} (draw-clustered bootstrap SEs). The platform-style ECE "
        f"(ten fixed bins within each pair, ALC-weighted, mean over pairs; the leaderboard's diagnostic) is "
        f"{f['ece_platform']:.4f} for {label(final)} and {b['ece_platform']:.4f} for {label(base)}; with 44 to 113 "
        f"evaluation items per pair it is dominated by within-pair binning noise"
        + (f": a perfectly calibrated forecaster with the same predictions would score {f['ece_platform_floor']:.4f} "
           f"(pooled ECE floor {f['ece_floor']:.4f})" if pd.notna(f.get("ece_platform_floor"))
           and pd.notna(f.get("ece_floor")) else "") + ".\n")
    (outdir / "report_snippet.md").write_text(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", required=True, help="replay_preds.py output (.parquet or .csv.gz)")
    ap.add_argument("--ref-scores", default="results/combo_v1.csv",
                    help="chained paired.py pair scores to reproduce bitwise ('' = skip)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--models", default="", help="models to analyse (default: all in the file, in a fixed order)")
    ap.add_argument("--plot-models", default="base,SCP_combo,w20_SCP")
    ap.add_argument("--base", default="base")
    ap.add_argument("--final", default="w20_SCP", help="the model the report snippet calls final")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--sens-bins", default="5,20", help="extra equal-mass bin counts for the sensitivity table")
    ap.add_argument("--weighting", choices=["pair", "row"], default="pair")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--sims", type=int, default=20,
                    help="label draws for the noise floors of the ECEs (0 = off)")
    args = ap.parse_args()
    t0 = time.time()
    path = Path(args.preds) if Path(args.preds).is_absolute() else ROOT / args.preds
    outdir = Path(args.outdir) if Path(args.outdir).is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_preds(path)
    order = ["base", "SCP_combo", "w20_SCP", "smoothed_k2", "empirical_mean"]
    present = list(df["model"].cat.categories)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or \
        [m for m in order if m in present] + [m for m in present if m not in order]
    df = df[df["model"].isin(models)]
    df["model"] = df["model"].cat.remove_unused_categories()
    print(f"{len(df)} rows, models {models}, loaded in {time.time() - t0:.0f}s", flush=True)

    ps = pair_level(df, args.sims, args.seed + 1)
    ps.to_csv(outdir / "pair_scores_recomputed.csv", index=False)
    cols = ["alc"] + [f"b{b}" for b in BUDGETS]
    checks = []
    own = path.with_name(path.name.split(".")[0] + "_scores.csv")
    if own.exists():
        checks += compare(ps, own, cols + [f"e{b}" for b in BUDGETS])
    if args.ref_scores:
        ref = Path(args.ref_scores) if Path(args.ref_scores).is_absolute() else ROOT / args.ref_scores
        checks += compare(ps, ref, cols)
        checks += compare_aggregates(ps, ref)
    for c in checks:
        print(c)
    macro, budg = chained_aggregates(ps)

    clusters = ps[["bench", "k", "run"]].drop_duplicates().sort_values(["bench", "k", "run"]).reset_index(drop=True)
    cl_index = {(b, int(k), int(r)): i for i, (b, k, r) in enumerate(clusters.itertuples(index=False))}
    M = bootstrap_matrix(clusters, args.boot, args.seed) if args.boot > 0 else None
    items = item_table(df, args.weighting)
    benches = sorted(items["bench"].unique())
    scopes = ["all"] + (benches if len(benches) > 1 else [])
    sens_bins = tuple(int(x) for x in args.sens_bins.split(",") if x.strip())
    point, reps, bins, sens = calibrate(items, scopes, models, args.bins, M, cl_index, sens_bins, args.sims,
                                        args.seed + 2)
    if len(benches) == 1:  # single-benchmark file: the benchmark scope equals the pooled one
        point.update({(benches[0], m, b): v for (s, m, b), v in list(point.items())})
        reps.update({(benches[0], m, b): v for (s, m, b), v in list(reps.items())})
    lt = long_table(point, reps, ps, M, cl_index, args.base)
    lt.to_csv(outdir / "calib_long.csv", index=False)
    bins.to_csv(outdir / "bins.csv", index=False)
    sens.to_csv(outdir / "sensitivity.csv", index=False)
    plot_models = [m for m in args.plot_models.split(",") if m in models]
    rmeta = path.with_name(path.name.split(".")[0] + "_meta.json")  # replay_preds.py's record of the subsample mode
    try:
        mode = {"bench": "bench-keyed", "pair": "pair-keyed"}[json.loads(rmeta.read_text())["mode"]]
    except Exception:
        mode = "bench-keyed" if "bench" in path.name else ("pair-keyed" if "pair" in path.name else "replay")
    meta = {"preds": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "created_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "weighting": args.weighting,
            "bins": args.bins, "boot": args.boot, "seed": args.seed, "n_draws": len(clusters),
            "n_pair_draws": int(ps.groupby("model").size().iloc[0]),
            "mode_note": f"{mode} LOBO draws of {', '.join(benches)}, k = {', '.join(map(str, sorted(ps.k.unique())))}"}
    title = f"{meta['preds']}: {meta['mode_note']}; {meta['n_pair_draws']} pair-draws; pair-weighted"
    figs = [] if args.no_figs else make_figures(outdir, bins, lt, items, plot_models, title, args.base)
    final = args.final if args.final in models else \
        next((m for m in reversed(plot_models) if m != args.base), None)  # None: no snippet
    cells = ps.pivot_table(index=["bench", "k"], columns="model", values="alc", aggfunc="mean")
    cells["n"] = ps[ps.model == models[0]].groupby(["bench", "k"]).size()
    write_summary(outdir, meta, checks, macro, budg, lt, sens, models, args.base, final, figs, cells)
    snippet = write_snippet(outdir, lt, meta, args.base, final)
    if not snippet:  # never leave a stale paragraph from an earlier run next to fresh tables
        (outdir / "report_snippet.md").unlink(missing_ok=True)
    (outdir / "analyze_meta.json").write_text(json.dumps({**meta, "checks": checks, "models": models,
                                                          "seconds": round(time.time() - t0, 1),
                                                          "command": " ".join(sys.argv)}, indent=1))
    a = lt[lt.scope == "all"]
    show = a[[c for c in ("model", "budget", "brier", "ece", "ece_floor", "rel", "res", "resid", "unc",
                          "ece_platform", "ece_platform_floor") if c in a.columns]].copy()
    pd.set_option("display.width", 200)
    print("\nall benchmarks pooled (pair-weighted):\n" + show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if snippet:
        print("\n" + snippet)
    print(f"done in {time.time() - t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()
