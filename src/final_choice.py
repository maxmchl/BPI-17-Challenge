"""Section 2.3 - choice of the final model from the iteration log (two stages).

Stage 1 - log version by modeling purpose. The model is the base for a simulation and should show
  what the bank executes (professor, Nov 6: only W_ events are executed by the bank). Hence the
  work items have to be in the log: W_ events at lifecycle start, one per work item started by an
  employee (log B1). B1c additionally drops O_Created / O_Accepted / O_Refused, which only
  duplicate information (results/b1c_redundancy.json). Log A (It1, 93.0 % fitting, precision
  0.589) stays in the report as the A_/O_ backbone.
Stage 2 - selection rule within that log version. ETC precision is not comparable across logs with
  different activity sets (more activities = more possible continuations), so the rule is applied
  only to rows with the same log version and activity set:
    highest ETC precision (token replay, full log) among rows with >= 80 % fitting cases
    (alignments, full log); differences < 0.005 are ties, broken by lower CNC, lower share of
    silent transitions, then more fitting cases.

Outputs: results/final_choice.json, figures/coverage_tradeoff.pdf
Run after iterations.py:  python src/final_choice.py
"""
import json

import pandas as pd

from load import ROOT
from iterations import best, FIT_TARGET, PRECISION_TIE
from figures_stats import C, W, INK2, plt

RES = ROOT / "results"
FINAL_LOG = "B1c"


def c_reason(rows: dict) -> str:
    c = [r for r in rows.values() if r["log"] == "C"]
    res = "; ".join(f"{r['coverage']:g} % coverage -> {r['fit_traces_align_pct']:.1f} % fitting, precision "
                    f"{r['precision_tbr']:.3f}" for r in sorted(c, key=lambda r: r["coverage"]))
    n = c[0]["variants"] if c else "?"
    return (f"{n:,} variants; IM with variant filter: {res}. W_ complete is rare because most work "
            f"items end in ate_abort/withdraw, so lifecycle complete loses most bank work")


def choose(rows: dict) -> dict:
    cand = [r for r in rows.values() if r["log"] == FINAL_LOG and not r.get("drop_rare_w")]
    chosen = best(cand)
    per_version = {}
    for name, sel in [("A (A_+O_)", lambda r: r["log"] == "A"), ("B1", lambda r: r["log"] == "B1"),
                      ("B1c", lambda r: r["log"] == "B1c" and not r.get("drop_rare_w")),
                      ("B1c without rare W_", lambda r: r.get("drop_rare_w"))]:
        b = best([r for r in rows.values() if sel(r)])
        per_version[name] = None if b is None else {
            k: b[k] for k in ["id", "fit_traces_align_pct", "fitness_align", "precision_tbr",
                              "precision_tbr_discovery_log", "own_cnc", "own_silent_share", "own_size"]}
    return {
        "stage_1_log_version": {
            "chosen": FINAL_LOG,
            "reason": ("simulation base model and the professor's bank-vs-customer question need the "
                       "bank's work steps: W_ events at lifecycle start (one per work item started by an "
                       "employee, 0 % User_1); B1c drops O_Created, O_Accepted and O_Refused, which only "
                       "duplicate O_Create Offer, A_Pending and A_Denied (results/b1c_redundancy.json)"),
            "not_chosen": {
                "A": "no work items, so no bank lane and no resources/processing times for simulation; "
                     "kept as the A_/O_ backbone (It1)",
                "C (all activities, lifecycle complete; professor's choice)": c_reason(rows),
                "B1": "same variants as B1c, plus three activities without own information"},
        },
        "stage_2_rule": (f"within {FINAL_LOG} (same activity set): highest ETC precision (token replay, full "
                         f"log) among rows with fitting cases (alignments, full log) >= {FIT_TARGET:g} %; "
                         f"differences < {PRECISION_TIE} are ties, broken by lower CNC, lower silent share, "
                         f"then more fitting cases"),
        "candidates": [r["id"] for r in cand],
        "eligible": [r["id"] for r in cand if r["fit_traces_align_pct"] >= FIT_TARGET],
        "best_per_log_version (not comparable across versions)": per_version,
        "collapse_repeats": ("not used in the final version: direct repetitions stay in the log, so the "
                             "simulation can take repetition counts as self-loop probabilities"),
        "chosen": chosen,
    }


def plot(rows: dict):
    tab = pd.DataFrame(rows.values())
    tab = tab[tab["coverage"].notna() & ~tab["drop_rare_w"].fillna(False).astype(bool)]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    series = [("A", "log A (A_+O_)", C[2], "-"), ("B1", "log B1 (+ W_ start)", C[3], (0, (3, 2))),
              ("B1c", "log B1c (B1 without duplicates)", C[0], "-")]
    for log, label, col, ls in series:
        d = tab[tab["log"] == log].sort_values("coverage")
        ax.plot(d["fit_traces_align_pct"], d["precision_tbr"], color=col, ls=ls, marker="o", ms=4,
                mec="white", mew=1, label=label)
        if log == "B1c":  # coverage labels; identical models (same point) share one label
            pts = d.groupby([d["fit_traces_align_pct"].round(2), d["precision_tbr"].round(4)], sort=False)
            for (x, y), g in pts:
                lo = g["coverage"].max() < 72  # left part: below the B1 line, clear of log A
                mid = g["coverage"].max() == 72.5
                ax.annotate("/".join(f"{c:g}" for c in g["coverage"]) + " %", (x, y),
                            xytext=(0, -16) if lo else (5, -5) if mid else (3, 4),
                            ha="center" if lo else "left", textcoords="offset points", fontsize=6.5,
                            color=INK2)
    ax.axvline(FIT_TARGET, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(FIT_TARGET + 0.4, ax.get_ylim()[0] + 0.01, "80 % of cases", color=INK2, fontsize=7, va="bottom")
    ax.set_xlabel("fitting cases [%] (alignments, full log)")
    ax.set_ylabel("precision (ETC, full log)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.5,
              handlelength=2.2)  # point labels (B1c) = variant coverage of the discovery log -> caption
    fig.savefig(ROOT / "figures" / "coverage_tradeoff.pdf")
    fig.savefig(ROOT / "figures" / "coverage_tradeoff.png", dpi=200)
    plt.close(fig)


def main():
    rows = json.loads((RES / "iterations.json").read_text())
    out = choose(rows)
    (RES / "final_choice.json").write_text(json.dumps(out, indent=2, default=str))
    plot(rows)
    c = out["chosen"]
    print(json.dumps(out["best_per_log_version (not comparable across versions)"], indent=1))
    print(f"chosen: {c['id']} ({c['log']}, coverage {c['coverage']}): fit {c['fit_traces_align_pct']:.1f}%, "
          f"precision {c['precision_tbr']:.3f}, CNC {c['own_cnc']:.3f}")


if __name__ == "__main__":
    main()
