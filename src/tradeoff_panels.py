"""Section 2.3 - trade-off figure: fitting cases vs precision / generalization / simplicity.

One point per row of results/iterations.csv (all model versions of the iteration log), colored
and shaped by log version; the final model (results/final_choice.json) is highlighted.
x = fitting cases [%] (alignments, full log of the version); y = ETC precision (token replay),
generalization (PM4Py, token replay) and CNC (arcs per node, lower = simpler).
Precision values are not comparable across log versions (different activity sets), so the
colors separate the versions.

Output: figures/tradeoff_panels.pdf (two-column width)
Run after final_choice.py:  python src/tradeoff_panels.py
"""
import json

import pandas as pd

from load import ROOT
from iterations import FIT_TARGET
from figures_stats import C, INK, INK2, plt

VERSIONS = [  # log, legend label, color, marker (same colors as coverage_tradeoff.pdf)
    ("A", "A (A_+O_)", C[2], "o"),
    ("B1", "B1 (+ W_ start)", C[3], "s"),
    ("B1c", "B1c (B1 without duplicates)", C[0], "D"),
    ("C", "C (all lifecycle complete)", C[1], "^"),
]
PANELS = [("precision_tbr", "precision (ETC)", (-6, 7)), ("generalization_tbr", "generalization", (-8, -12)),
          ("own_cnc", "CNC (lower = simpler)", (-6, 7))]  # column, axis label, offset of the final-model label


def main():
    tab = pd.read_csv(ROOT / "results" / "iterations.csv")
    final = json.loads((ROOT / "results" / "final_choice.json").read_text())["chosen"]["id"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharex=True)
    for ax, (col, label, offset) in zip(axes, PANELS):
        for log, name, color, marker in VERSIONS:
            d = tab[tab["log"] == log]
            ax.scatter(d["fit_traces_align_pct"], d[col], s=22, color=color, marker=marker, edgecolor="white",
                       linewidth=0.6, label=name, zorder=3)
        f = tab[tab["id"] == final].iloc[0]
        ax.scatter(f["fit_traces_align_pct"], f[col], s=90, facecolor="none", edgecolor=INK, linewidth=1.1,
                   zorder=4, label=f"final model ({final})")
        ax.annotate(final, (f["fit_traces_align_pct"], f[col]), xytext=offset, textcoords="offset points",
                    ha="right", fontsize=6.5, color=INK)
        ax.axvline(FIT_TARGET, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_ylabel(label)
        ax.set_xlabel("fitting cases [%]")
        ax.set_axisbelow(True)
    axes[0].text(FIT_TARGET + 0.8, axes[0].get_ylim()[0] + 0.01, "80 %", color=INK2, fontsize=7, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    fig.savefig(ROOT / "figures" / "tradeoff_panels.pdf")
    fig.savefig(ROOT / "figures" / "tradeoff_panels.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
