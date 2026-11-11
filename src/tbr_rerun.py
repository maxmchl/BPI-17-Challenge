"""Section 2.3 - token replay with activities not in the model counted as deviations.

PM4Py's token replay skips events whose activity is not in the model by default
(`consider_activities_not_in_model_in_fitness = False`). No IMf model from noise 0.10 on contains
"O_Sent (online only)", so those traces can be "fitting" for TBR while alignments see a log move.
This script reruns token-based fitness with the parameter set to True for every model in
results/discovery_metrics.json and results/imf_sweep.json and stores both variants
(`fitness_tbr`, `fit_traces_tbr_pct` = default; `*_tbr_nim` = parameter True).

PM4Py's own implementation of the switch is buggy (see metrics.tbr_fitness), so the
intended rule is applied on top of the default replay.

Precision (ETC) and generalization are not rerun: ETC precision builds its own replay
parameters and generalization only uses transition firings, both independent of the switch
(checked once below for IMf 0.10).
`python src/tbr_rerun.py disagreement` writes results/tbr_disagreement_imf010.json: per-trace
cross-tabulation of the three fit verdicts for IMf 0.10 and the largest disagreeing variants.
Run after discovery.py and imf_sweep.py:  python src/tbr_rerun.py
"""
from collections import Counter
import json
import sys
import warnings

import pandas as pd
import pm4py
from pm4py.algo.conformance.tokenreplay.variants import token_replay as tr

from load import ROOT
from preprocess import build_log
from metrics import tbr_both, tbr_fitness, align
from discovery import write_tables
from imf_sweep import THRESHOLDS, REUSE, OUT_JSON

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"


def show(name, old, new):
    print(f"{name:>6}: fitting TBR default {new['fit_traces_tbr_pct']:.2f}% (stored "
          f"{old.get('fit_traces_tbr_pct', float('nan')):.2f}%)  -> not-in-model counted "
          f"{new['fit_traces_tbr_nim_pct']:.2f}%  | alignments {old.get('fit_traces_align_pct', float('nan')):.2f}%"
          f"  | fitness {new['fitness_tbr']:.4f} / {new['fitness_tbr_nim']:.4f}", flush=True)


def disagreement(log, net, im, fm):
    """Per trace: fit according to TBR default, TBR with the switch (own rule), and alignments."""
    elog = pm4py.convert_to_event_log(log)
    _, r0 = tbr_fitness(elog, net, im, fm, False)
    _, r1 = tbr_fitness(elog, net, im, fm, True)   # own rule
    rb = tr.apply(elog, net, im, fm, parameters={   # PM4Py's own switch (buggy, for the record)
        tr.Parameters.CONSIDER_REMAINING_IN_FITNESS: True,
        tr.Parameters.CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS: True})
    al = align(elog, net, im, fm)
    bug = sum(x["trace_is_fit"] != y["trace_is_fit"] for x, y in zip(rb, r1))
    combos, variants = Counter(), Counter()
    for tr_, a, b, c in zip(elog, r0, r1, al):
        k = (a["trace_is_fit"], b["trace_is_fit"], c["fitness"] == 1.0)
        combos[k] += 1
        if k[1] != k[2]:
            variants[(k, tuple(e["concept:name"] for e in tr_), round(b["trace_fitness"], 3),
                      b["missing_tokens"], b["remaining_tokens"], round(c["fitness"], 3))] += 1
    out = {"cases where PM4Py's switch and the intended rule disagree (bug)": int(bug),
           "combos (tbr default, tbr switch, alignments fit) -> cases":
           {str(k): v for k, v in combos.most_common()},
           "top disagreeing variants": [
               {"cases": n, "fit": k, "trace": list(tr_), "tbr_fitness": f, "missing": m,
                "remaining": r, "align_fitness": af}
               for (k, tr_, f, m, r, af), n in variants.most_common(5)]}
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    log = build_log("A")
    disc_path = RES / "discovery_metrics.json"
    disc = json.loads(disc_path.read_text())
    nets = {}
    for name in disc:
        nets[name] = pm4py.read_pnml(str(RES / "models" / f"{name}.pnml"))
        new = tbr_both(log, *nets[name])
        show(name, disc[name], new)
        disc[name].update(new)
    disc_path.write_text(json.dumps(disc, indent=2))
    write_tables(disc)

    sweep = json.loads(OUT_JSON.read_text())
    for th in THRESHOLDS:
        key = f"{th:.2f}"
        net = nets[REUSE[th]] if th in REUSE else pm4py.discover_petri_net_inductive(log, noise_threshold=th)
        new = tbr_both(log, *net)
        show(key, sweep[key], new)
        sweep[key].update(new)
        if key == "0.10":  # sanity check: generalization does not depend on the switch
            g = pm4py.algo.evaluation.generalization.algorithm.apply(
                log, *net, parameters={tr.Parameters.CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS: True})
            print(f"        generalization TBR with switch {g:.4f} vs stored {sweep[key]['generalization_tbr']:.4f}")
    OUT_JSON.write_text(json.dumps(sweep, indent=2))
    tab = pd.DataFrame(sweep).T.sort_index()
    tab.index.name = "noise_threshold"
    tab.to_csv(RES / "imf_sweep.csv")


if __name__ == "__main__":
    if sys.argv[1:] == ["disagreement"]:
        log = build_log("A")
        out = disagreement(log, *pm4py.discover_petri_net_inductive(log, noise_threshold=0.1))
        (RES / "tbr_disagreement_imf010.json").write_text(json.dumps(out, indent=2))
    else:
        main()
