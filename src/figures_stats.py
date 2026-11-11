"""Figures for Section 2.2 (simple event log analysis).

Writes figures/variant_coverage.pdf, figures/duration_by_outcome.pdf, figures/weekly_arrivals.pdf
Run after stats.py:  python src/figures_stats.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load import load_log, ROOT, CASE, ACT, TS, LC

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# validated categorical slots (light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 3.4  # one column of the two-column template, inches


def coverage(d, collapse=False):
    tr = d.groupby(CASE)[ACT].agg(list)
    if collapse:
        tr = tr.apply(lambda s: [a for i, a in enumerate(s) if i == 0 or a != s[i - 1]])
    vc = tr.apply(tuple).value_counts()
    return np.arange(1, len(vc) + 1), (vc.cumsum() / vc.sum()).to_numpy() * 100


def fig_variant_coverage(df):
    complete = df[df[LC] == "complete"]
    ao = complete[complete[ACT].str[:2].isin(["A_", "O_"])]
    w_start = df[(df[LC] == "start") & df[ACT].str.startswith("W_")]
    b1 = pd.concat([ao, w_start]).sort_values([CASE, TS, "EventID"], kind="stable")
    b1c = b1[~b1[ACT].isin(["O_Created", "O_Accepted", "O_Refused"])].copy()
    b1c[ACT] = b1c[ACT].where(~b1c[ACT].str.startswith("O_Sent"), "O_Sent")
    # (name, data, colour, line style): B1c is derived from B1 -> same hue, dashed
    series = [
        ("raw log", coverage(df), C[0], "-"),
        ("lifecycle = complete", coverage(complete), C[1], "-"),
        ("A_ + O_ events, complete (log A)", coverage(ao), C[2], "-"),
        ("A + W_ start events (log B1)", coverage(b1), C[3], "-"),
        ("B1 without duplicates (log B1c)", coverage(b1c), C[3], (0, (3, 2))),
    ]  # collapsing direct repeats on top changes little (65 -> 61 variants for 80 %), see extra_stats
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for name, (x, y), col, ls in series:
        ax.plot(x, y, color=col, ls=ls, label=name)
        n80 = int(np.searchsorted(y, 80) + 1)
        ax.plot([n80], [80], "o", ms=4, color=col, mec="white", mew=1)
    ax.axhline(80, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.2, 82, "80 % of cases", color=INK2, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("number of most frequent variants (log scale)")
    ax.set_ylabel("cases covered [%]")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=6.5,
              columnspacing=1.0, handlelength=2.2)
    fig.savefig(FIG / "variant_coverage.pdf")
    fig.savefig(FIG / "variant_coverage.png", dpi=200)
    plt.close(fig)


def fig_duration_by_outcome(cases):
    order = [("A_Pending", "Pending"), ("A_Cancelled", "Cancelled"), ("A_Denied", "Denied")]
    bins = np.arange(0, 91, 1)
    fig, axes = plt.subplots(3, 1, figsize=(W, 2.6), sharex=True)
    for ax, (key, label), col in zip(axes, order, C):
        d = cases.loc[cases.outcome == key, "duration_s"] / 86400
        ax.hist(d, bins=bins, color=col, edgecolor="white", linewidth=0.3)
        ax.axvline(d.median(), color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.text(0.99, 0.9, f"{label}\nn = {len(d):,}\nmedian {d.median():.1f} d",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
        ax.set_ylabel("cases")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("case duration [days] (dashed: median)")
    axes[-1].set_xlim(0, 90)
    fig.savefig(FIG / "duration_by_outcome.pdf")
    fig.savefig(FIG / "duration_by_outcome.png", dpi=200)
    plt.close(fig)


def fig_weekly_arrivals(cases):
    s = cases.set_index("start").resample("W").size()
    s = s.iloc[1:-1]  # drop partial first/last week
    fig, ax = plt.subplots(figsize=(W, 1.6))
    ax.plot(s.index, s.values, color=C[0])
    ax.set_ylabel("applications / week")
    ax.set_ylim(0, s.max() * 1.1)
    fig.autofmt_xdate()
    fig.savefig(FIG / "weekly_arrivals.pdf")
    fig.savefig(FIG / "weekly_arrivals.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    df = load_log().sort_values([CASE, TS, "EventID"], kind="stable")
    cases = pd.read_parquet(ROOT / "results" / "cases.parquet")
    fig_variant_coverage(df)
    fig_duration_by_outcome(cases)
    fig_weekly_arrivals(cases)
    print("figures written to", FIG)
