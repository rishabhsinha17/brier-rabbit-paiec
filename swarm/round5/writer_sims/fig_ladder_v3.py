"""Figure C1 of report/paper_draft_v3.md: paired Brier difference against the smoothed mean, by budget, with 95% CIs.

One draw set (results/combo_v1.csv: benchmark-keyed subsample, k in {1, 2}, 18 draws per benchmark and k). Every model
is scored on identical draws. Differences use the paper's convention (swarm/round5/writer_sims/paper_numbers_v3.py):
per-draw means of per-pair differences, benchmark-equal and k-averaged, draw-clustered SE; whiskers are +-1.96 SE.

Usage (from brier_rabbit/):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/writer_sims/fig_ladder_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_numbers_v3 import BUD, load, paired, summarize  # noqa: E402

SERIES = [  # (model in combo_v1, legend label); fixed categorical order, slots 1-4 of the validated palette
    ("v1_tuned", "Hierarchical, tuned (no subject prior)"),
    ("v1_subjects", "+ subject prior (m0 = -0.5)"),
    ("base", "REF (damped, m0 = -2.0)"),
    ("SCP_combo", "REF + composite S+C+P"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated: light mode, adjacent pairs, all checks pass
MARKERS = ["o", "s", "^", "D"]  # secondary encoding for print and colour-vision deficiency
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def main() -> None:
    cv = load("combo_v1")
    rows = []
    for model, label in SERIES:
        r = summarize(paired(cv, model, "smoothed_k2"))
        rows.append((label, [1e3 * r["macro"][b] for b in BUD], [1e3 * 1.96 * r["se"][b] for b in BUD]))

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(5.5, 2.7), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(BUD)))
    width = 0.16
    for i, (label, mean, ci) in enumerate(rows):
        off = (i - 1.5) * width
        x = [v + off for v in xs]
        ax.plot(x, mean, color=COLORS[i], lw=1.0, alpha=0.55, zorder=2)
        ax.errorbar(x, mean, yerr=ci, fmt=MARKERS[i], color=COLORS[i], ms=4.5, mec=SURFACE, mew=0.8,
                    elinewidth=1.0, capsize=0, label=label, zorder=3)
    ax.axhline(0.0, color=INK2, lw=0.6, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([b.upper() for b in BUD], color=INK)
    ax.set_xlabel("Label budget", color=INK)
    ax.set_ylabel("Brier minus smoothed mean (x1e-3)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, width=0.6)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.4, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out = HERE / "fig_c1_ladder.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)
    for label, mean, ci in rows:
        print(f"{label:<40s} " + " ".join(f"{m:+6.1f}[{c:4.1f}]" for m, c in zip(mean, ci)))


if __name__ == "__main__":
    main()
