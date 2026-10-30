"""Figures for report/paper_draft_v2.md (public data and synthetic inputs only; no formative feedback is read).
Paper numbering: fig2_adaptation.png is the paper's Figure 1 and fig1_local_curves.png is its Figure 2.

Figure 1: local LOBO learning curves (benchmark-equal macro Brier per budget) for the ladder models, from saved CSVs.
Figure 2: adaptation speed. Prediction for an unseen item of a pair alone on an unseen benchmark (unknown subject,
theta = 0, no item_features group) after k successes in n own labels on distinct items, for REF and the live
level-block arms, computed with the real HierModel; plus the exact Bayes predictive under REF's own prior (1-D grid
over the pair level, each label's item effect integrated by Gauss-Hermite), which shows the Laplace under-reaction.

Usage (from brier_rabbit/): ../.venv/bin/python swarm/round3/report_writer_sims/paper_figures.py
Writes fig1_local_curves.png, fig2_adaptation.png and paper_figures.out next to this file.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from paiec.hier import HierModel  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
REF = {"m0": -2.0, "tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5,
       "tau_theta": 0.5, "damped_newton": True}
ARMS = {
    "REF (m0 -2.0, tau_mu 1.2)": {},
    "L15 (m0 -1.5)": {"m0": -1.5},
    "W20 (tau_mu 2.0)": {"tau_mu": 2.0},
    "d06 (narrow pair prior)": {"m0": -1.693, "tau_delta": 0.6, "tau_phi": 0.3},
    "mix25 (pi .25, m0 -1.0)": {"m0": -1.0, "mix_pi": 0.25, "mix_m": -4.0, "mix_s": 0.7},
}


def macro_curve(csv: str, model: str) -> list[float]:
    df = pd.read_csv(REPO / csv)
    d = df[df.model == model]
    cells = d.groupby(["bench", "k"])[[f"b{b}" for b in BUDGETS]].mean()
    return cells.groupby(level=0).mean().mean().tolist()


def figure1(out):
    rows = [
        ("Smoothed mean (kappa=2)", "results/paired.csv", "smoothed_k2"),
        ("v1 hierarchical, tuned (m0 -0.5)", "results/paired.csv", "v1_tuned"),
        ("+ subject prior (m0 -0.5)", "results/paired_subjects.csv", "v1_subjects"),
        ("+ damped Newton, m0 -1.5", "results/paired_damped.csv", "n15D"),
        ("+ damped Newton, m0 -2.0 (REF)", "results/paired_damped.csv", "n20D"),
    ]
    x = [math.log2(1 + b) for b in BUDGETS]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, csv, model in rows:
        y = macro_curve(csv, model)
        ax.plot(x, y, marker="o", ms=3, lw=1.4, label=label)
        out.write(f"fig1 {label}: " + " ".join(f"B{b}={v:.4f}" for b, v in zip(BUDGETS, y)) + "\n")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("labels revealed per pair (log2(1+b) axis)")
    ax.set_ylabel("Brier (benchmark-equal macro)")
    ax.set_title("Local LOBO learning curves (public data)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig1_local_curves.png", dpi=200)
    plt.close(fig)


def synth_pair(k: int, n: int):
    subject = {"normalized_name": "", "provider": "", "release_date": "", "access_date": "", "harness": "",
               "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
    labeled = []
    for j in range(n):
        item = {"item_content": f"synthetic item {j}", "item_features": "", "interactors": "",
                "benchmark_id": "benchmark_000001"}
        labeled.append([[subject, item], 1 if j < k else 0])
    target = {"item_content": "synthetic target item", "item_features": "", "interactors": "",
              "benchmark_id": "benchmark_000001"}
    return [subject, target], labeled


def model_q(params: dict, k: int, n: int) -> float:
    m = HierModel({}, dict(REF, **params))  # empty prior: unknown subject, theta 0, var tau_theta^2
    inp, lab = synth_pair(k, n)
    return m.predict(inp, lab)


def exact_q(params: dict, k: int, n: int) -> float:
    """Exact posterior predictive under the same normal prior: pair level L ~ N(m0, tau_mu^2 + tau_delta^2 +
    tau_phi^2 + lam^2 tau_theta^2); each label y ~ Bern(sigmoid(L + eps)), eps ~ N(0, sigma_i^2) integrated per
    label; the target is an unseen item, so its eps is integrated too."""
    p = dict(REF, **params)
    v = p["tau_mu"] ** 2 + p["tau_delta"] ** 2 + p["tau_phi"] ** 2 + p["lam"] ** 2 * p["tau_theta"] ** 2
    grid = p["m0"] + math.sqrt(v) * np.linspace(-8, 8, 4001)
    logprior = -0.5 * ((grid - p["m0"]) ** 2) / v
    gx, gw = np.polynomial.hermite_e.hermegauss(40)
    gw = gw / gw.sum()
    pbar = (1.0 / (1.0 + np.exp(-(grid[:, None] + p["sigma_i"] * gx[None, :])))) @ gw
    pbar = np.clip(pbar, 1e-300, 1 - 1e-16)
    loglik = k * np.log(pbar) + (n - k) * np.log1p(-pbar)
    a = logprior + loglik
    w = np.exp(a - a.max())
    w /= w.sum()
    return float(np.sum(w * pbar))


def figure2(out):
    ns = [1, 3, 7, 15, 31]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=False)
    for label, params in ARMS.items():
        zeros = [model_q(params, 0, n) for n in ns]
        ones = [model_q(params, n, n) for n in ns]
        b0 = model_q(params, 0, 0)
        axes[0].plot([0] + ns, [b0] + zeros, marker="o", ms=3, lw=1.3, label=label)
        axes[1].plot([0] + ns, [b0] + ones, marker="o", ms=3, lw=1.3, label=label)
        out.write(f"fig2 {label}: B0 {b0:.3f}; 0/n " + " ".join(f"{q:.3f}" for q in zeros)
                  + "; n/n " + " ".join(f"{q:.3f}" for q in ones) + "\n")
    ez = [exact_q({}, 0, n) for n in ns]
    eo = [exact_q({}, n, n) for n in ns]
    e0 = exact_q({}, 0, 0)
    for ax, ys in ((axes[0], ez), (axes[1], eo)):
        ax.plot([0] + ns, [e0] + ys, ls="--", color="black", lw=1.2, label="exact Bayes, REF prior")
    out.write(f"fig2 exact REF prior: B0 {e0:.3f}; 0/n " + " ".join(f"{q:.3f}" for q in ez)
              + "; n/n " + " ".join(f"{q:.3f}" for q in eo) + "\n")
    for ax, title in ((axes[0], "after n failures (0/n)"), (axes[1], "after n successes (n/n)")):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0] + ns)
        ax.set_xticklabels([str(v) for v in [0] + ns])
        ax.set_xlabel("own labels n")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(success) on an unseen item")
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "fig2_adaptation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    with open(HERE / "paper_figures.out", "w") as out:
        figure1(out)
        figure2(out)
    print((HERE / "paper_figures.out").read_text())


if __name__ == "__main__":
    main()
