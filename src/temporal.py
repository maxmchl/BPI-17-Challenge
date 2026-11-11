"""Section 3.3 - temporal generalization of the final configuration (drift check).

The log covers cases started in 2016 (ending 1 Feb 2017); weekly arrivals jump from ~500 to
~700 around June 2016. A model that generalizes should replay cases from a period it was not
discovered on about as well as its own period.

(a) Temporal split by case start (first event): H1 = Jan-Jun 2016, H2 = Jul-Dec 2016.
    Discover with the final configuration on one half, evaluate on both halves
    (fitting cases and fitness from alignments, ETC precision with token replay).
    Also records whether IMf discovers the identical process tree on H1, H2 and the full log.
(b) The full-log model: share of fitting cases (alignments) per start month.

Outputs: results/temporal_split.json, results/fitting_by_month.csv, figures/fitting_by_month.pdf,
         results/final_case_alignment.parquet (per-case alignment fitness of the full-log model)
Run after iterations.py:  python src/temporal.py
"""
import json
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, TS
from preprocess import final_log
from metrics import align, timed
from pm4py.algo.evaluation.replay_fitness.variants import alignment_based as align_fitness
from figures_stats import C, W, INK2, plt

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"


def case_start(log) -> pd.Series:
    return log.groupby(CASE)[TS].min()


def score(log, net, im, fm) -> dict:
    elog = pm4py.convert_to_event_log(log)
    al, t = timed(align, elog, net, im, fm)
    ok = [a for a in al if a is not None]
    af = align_fitness.evaluate(ok)
    return {"cases": len(elog), "fit_traces_align_pct": af["percentage_of_fitting_traces"],
            "fitness_align": af["log_fitness"],
            "precision_tbr": pm4py.precision_token_based_replay(log, net, im, fm),
            "alignments_failed": len(al) - len(ok), "t_alignments": t}


def split(log, noise, start):
    halves = {"H1 (Jan-Jun)": start.index[start.dt.month <= 6],
              "H2 (Jul-Dec)": start.index[start.dt.month > 6]}
    logs = {k: log[log[CASE].isin(ids)] for k, ids in halves.items()}
    out = []
    for train, ltrain in logs.items():
        net, im, fm = pm4py.discover_petri_net_inductive(ltrain, noise_threshold=noise)
        for test, ltest in logs.items():
            r = {"discovered_on": train, "evaluated_on": test, "in_sample": train == test,
                 "places": len(net.places), "transitions": len(net.transitions), **score(ltest, net, im, fm)}
            print(f"{train} -> {test}: fit {r['fit_traces_align_pct']:.1f}%  fitness {r['fitness_align']:.3f}  "
                  f"precision {r['precision_tbr']:.3f}  ({r['cases']:,} cases)", flush=True)
            out.append(r)
    return out


def by_month(log, noise, start):
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise)
    elog = pm4py.convert_to_event_log(log)
    ids = [t.attributes["concept:name"] for t in elog]
    al = align(elog, net, im, fm)
    per_case = pd.DataFrame({"fitness": [None if a is None else a["fitness"] for a in al],
                             "cost": [None if a is None else a["cost"] for a in al]}, index=pd.Index(ids, name=CASE))
    per_case["fits"] = per_case["fitness"] == 1.0
    per_case["start"] = start.reindex(per_case.index)
    per_case.to_parquet(RES / "final_case_alignment.parquet")
    month = per_case["start"].dt.tz_convert(None).dt.to_period("M")
    tab = per_case.groupby(month).agg(cases=("fits", "size"), fitting_pct=("fits", "mean"),
                                       mean_fitness=("fitness", "mean"))
    tab["fitting_pct"] *= 100
    tab.index = tab.index.astype(str)
    tab.index.name = "start_month"
    tab.to_csv(RES / "fitting_by_month.csv")
    return tab


def plot(tab):
    x = range(len(tab))
    labels = [pd.Period(m).strftime("%b") for m in tab.index]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True, height_ratios=[2, 1])
    a1.plot(x, tab["fitting_pct"], color=C[0], marker="o", ms=4, mec="white", mew=1)
    a1.axhline(80, color=INK2, lw=0.8, ls=(0, (3, 2)))
    a1.text(len(tab) - 1, 80.5, "80 %", color=INK2, fontsize=7, ha="right", va="bottom")
    a1.set_ylabel("fitting cases [%]")
    a2.bar(x, tab["cases"] / 1000, color=C[0], width=0.7)
    a2.set_ylabel("cases [k]")
    a2.set_xticks(list(x), labels)
    a2.set_xlabel("case start month (2016)")
    a2.grid(axis="x", visible=False)
    fig.align_ylabels()
    fig.savefig(ROOT / "figures" / "fitting_by_month.pdf")
    fig.savefig(ROOT / "figures" / "fitting_by_month.png", dpi=200)
    plt.close(fig)


def main():
    log, cfg = final_log()
    noise = cfg["noise_threshold"]
    start = case_start(log)
    print(f"final configuration {cfg['id']}: log {cfg['log']}, noise {noise}", flush=True)
    halves = [log[log[CASE].isin(start.index[m])] for m in (start.dt.month <= 6, start.dt.month > 6)]
    trees = [str(pm4py.discover_process_tree_inductive(x, noise_threshold=noise)) for x in [*halves, log]]
    same = {"H1 == H2": trees[0] == trees[1], "H1 == full log": trees[0] == trees[2]}
    print("identical process trees:", same, flush=True)
    out = {"configuration": cfg["id"], "identical_process_trees": same, "split": split(log, noise, start)}
    (RES / "temporal_split.json").write_text(json.dumps(out, indent=2))
    tab = by_month(log, noise, start)
    out["fitting_by_month"] = tab.reset_index().to_dict("records")
    (RES / "temporal_split.json").write_text(json.dumps(out, indent=2))
    print(tab.round(3).to_string(), flush=True)
    plot(tab)


if __name__ == "__main__":
    main()
