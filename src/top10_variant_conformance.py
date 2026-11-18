"""Section 2.3 - conformance of the 10 most frequent variants of the modelling log (B1c) with the
final BPMN (results/models/final_bpmn.bpmn, SAP Signavio remodel of It6_c75).

The BPMN is used as re-imported by signavio_reimport.py (results/models/final_signavio.pnml:
lanes and annotations stripped, PM4Py BPMN -> Petri net; same language as final.pnml).
Per variant: rank, cases, share of cases, offers (O_Create Offer count), abbreviated activity
sequence (legend below; repeats as xN), alignment fitness (PM4Py, standard costs), fits
perfectly (yes / no) and the deviating moves (log moves = event not allowed by the model,
model moves = visible activity the model requires but the trace lacks; silent moves are not
deviations).

Output: results/top10_variant_conformance.csv
Run after signavio_reimport.py:  python src/top10_variant_conformance.py
"""
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, ACT, TS
from preprocess import final_log

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"
TOP = 10
ABBR = {"A_Create Application": "A.Create", "A_Submitted": "A.Sub", "A_Concept": "A.Concept",
        "A_Accepted": "A.Acc", "A_Complete": "A.Compl", "A_Validating": "A.Valid", "A_Incomplete": "A.Incompl",
        "A_Pending": "A.Pend", "A_Denied": "A.Deny", "A_Cancelled": "A.Canc", "O_Create Offer": "O.Create",
        "O_Sent": "O.Sent", "O_Returned": "O.Ret", "O_Cancelled": "O.Canc", "W_Handle leads": "W.Leads",
        "W_Complete application": "W.ComplApp", "W_Call after offers": "W.CallOff",
        "W_Validate application": "W.Valid", "W_Call incomplete files": "W.CallInc",
        "W_Assess potential fraud": "W.Fraud", "W_Shortened completion": "W.Short",
        "W_Personal Loan collection": "W.Collect"}


def abbreviate(trace) -> str:
    out = []
    for a in trace:
        s = ABBR.get(a.strip(), a)
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return " ".join(s if k == 1 else f"{s}x{k}" for s, k in out)


def main():
    net, im, fm = pm4py.read_pnml(str(RES / "models" / "final_signavio.pnml"))
    log, cfg = final_log()
    traces = log.groupby(CASE, sort=False)[ACT].agg(tuple)
    counts = traces.value_counts()
    rows = []
    for rank, (variant, n) in enumerate(counts.head(TOP).items(), 1):
        one = pd.DataFrame({CASE: "v", ACT: list(variant),
                            TS: pd.date_range("2016-01-01", periods=len(variant), freq="min", tz="UTC")})
        a = pm4py.conformance_diagnostics_alignments(one, net, im, fm)[0]
        log_moves = [l for l, m in a["alignment"] if m == ">>"]
        model_moves = [m for l, m in a["alignment"] if l == ">>" and m is not None]
        rows.append({"rank": rank, "cases": int(n), "share_pct": round(100 * n / len(traces), 2),
                     "offers": variant.count("O_Create Offer"), "events": len(variant),
                     "sequence": abbreviate(variant), "fitness_align": round(a["fitness"], 4),
                     "fits": "yes" if a["fitness"] == 1.0 else "no",
                     "log_moves": "; ".join(log_moves), "model_moves": "; ".join(model_moves)})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "top10_variant_conformance.csv", index=False)
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(f"log {cfg['log']}: {len(traces):,} cases, {counts.size:,} variants; top {TOP} cover "
              f"{tab['share_pct'].sum():.1f} % of the cases")
        print(tab.drop(columns="sequence").to_string(index=False))
        print()
        for r in rows:
            print(f"{r['rank']:2d}  {r['sequence']}")
        print("\nlegend: " + ", ".join(f"{v} = {k}" for k, v in ABBR.items()))


if __name__ == "__main__":
    main()
