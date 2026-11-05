"""Section 3.3.2 - diagnostics behind the token replay (TBR) vs alignments comparison.

1. Generalization gap (IMf): transitions that TBR never fires, with their alignment counts.
2. Trace example (IMf): variant with the largest gap between per-trace TBR fitness and
   alignment fitness, with TBR missing/remaining tokens and the alignment moves.
3. HM soundness evidence (Woflan does not terminate on these nets): workflow-net check and
   the alignment of the most frequent trace (where does the model get stuck?).

Works on one trace per variant (weights = case counts), models from results/models/*.pnml.
Outputs: results/conformance_examples.json (+ printed report)
Run after discovery.py:  python src/conformance_examples.py
"""
from collections import Counter
import json
import warnings

import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as tbr_algo
from pm4py.objects.log.obj import EventLog

from load import ROOT, CASE, ACT
from preprocess import ao_log
from metrics import align, is_workflow_net

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"


def variant_log(df):
    """EventLog with one trace per variant (most frequent first) and the case count per variant."""
    traces = df.groupby(CASE, sort=False)[ACT].agg(tuple)
    counts = traces.value_counts()
    reps = traces.reset_index().drop_duplicates(ACT).set_index(ACT)[CASE]
    ids = [reps[v] for v in counts.index]
    sub = df[df[CASE].isin(ids)]
    log = pm4py.convert_to_event_log(sub)
    order = {c: i for i, c in enumerate(ids)}
    log._list.sort(key=lambda t: order[t.attributes["concept:name"]])
    return log, counts.to_numpy(), list(counts.index)


def tbr_diag(log, net, im, fm):
    return tbr_algo.apply(log, net, im, fm)  # per trace: fitness, missing/remaining, activated


def fmt_moves(alignment):
    out = []
    for (_, _), (lab_log, lab_model) in alignment:
        if lab_log == lab_model:
            out.append(f"{lab_log}")
        elif lab_log == ">>":
            out.append("tau" if lab_model is None else f"[model: {lab_model}]")
        else:
            out.append(f"[log: {lab_log}]")
    return [m for m in out if m != "tau"]


def generalization_gap(net, tbr, al, weights):
    tbr_occ, al_occ = Counter(), Counter()
    for res, w in zip(tbr, weights):
        for t in res["activated_transitions"]:
            tbr_occ[t.name] += w
    for res, w in zip(al, weights):
        if res is None:
            continue
        for (_, name), _ in res["alignment"]:
            if name != ">>":
                al_occ[name] += w
    rows = []
    for t in net.transitions:
        if tbr_occ[t.name] == 0:
            rows.append({"transition": t.name, "label": t.label, "silent": t.label is None,
                         "tbr_firings": 0, "alignment_firings": int(al_occ[t.name]),
                         "preset": sorted(a.source.name for a in t.in_arcs),
                         "postset": sorted(a.target.name for a in t.out_arcs)})
    return rows


def trace_example(tbr, al, weights, variants, min_cases=10):
    best = None
    for i, (res_t, res_a, w) in enumerate(zip(tbr, al, weights)):
        if res_a is None or w < min_cases:
            continue
        gap = res_t["trace_fitness"] - res_a["fitness"]
        if best is None or abs(gap) > abs(best[0]):
            best = (gap, i)
    gap, i = best
    res_t, res_a = tbr[i], al[i]
    return {"variant_rank": i + 1, "cases": int(weights[i]), "trace": list(variants[i]),
            "tbr_fitness": res_t["trace_fitness"], "tbr_missing": res_t["missing_tokens"],
            "tbr_remaining": res_t["remaining_tokens"], "tbr_consumed": res_t["consumed_tokens"],
            "tbr_produced": res_t["produced_tokens"],
            "align_fitness": res_a["fitness"], "align_cost": res_a["cost"],
            "align_moves": fmt_moves(res_a["alignment"])}


def main():
    df = ao_log()
    log, weights, variants = variant_log(df)
    out = {}

    net, im, fm = pm4py.read_pnml(str(RES / "models" / "IMf.pnml"))
    tbr = tbr_diag(log, net, im, fm)
    al = align(log, net, im, fm)
    out["imf_generalization_gap"] = generalization_gap(net, tbr, al, weights)
    out["imf_trace_example"] = trace_example(tbr, al, weights, variants)
    # weighted share of cases where TBR and alignments disagree on "fits"
    dis = sum(w for t, a, w in zip(tbr, al, weights) if a and t["trace_is_fit"] != (a["fitness"] == 1.0))
    out["imf_fit_disagreement_cases"] = int(dis)

    net, im, fm = pm4py.read_pnml(str(RES / "models" / "HM.pnml"))
    out["hm_wf_net"] = is_workflow_net(net)
    top = EventLog([log[0]])
    a = align(top, net, im, fm)[0]
    out["hm_top_variant"] = {"cases": int(weights[0]), "trace": list(variants[0]),
                             "align_fitness": None if a is None else a["fitness"],
                             "align_moves": None if a is None else fmt_moves(a["alignment"])}

    (RES / "conformance_examples.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
