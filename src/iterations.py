"""Section 3.3 - iteration log: change the data, rediscover with IMf, evaluate (fast metrics).

Each row is one model version. No manual edits of the Petri net: every change is a change of the
event log followed by rediscovery (professor, Nov 6).

  It0   log A (A_+O_ complete), IMf 0.10                         base (numbers from imf_sweep.json)
  It1   log A, both O_Sent labels merged into "O_Sent", IMf 0.10
  It2x  log B1 (A + W_ start events), O_Sent choice from It1, IMf 0.05 / 0.10 / 0.15 / 0.20
        (the knee of the fitness/precision curve may move once W_ events are included)
  It3   best It2 + direct repetitions of the same activity collapsed

Fast metrics (metrics.evaluate(fast=True)): no alignment-based precision and no ext. cyclomatic.
Precision below is ETC precision with token replay; over the IMf sweep it matched the
alignment-based value within 0.002.

Selection rule (also written to results/final_choice.json):
  highest precision among rows with fitting cases (alignments) >= 80 %; rows within
  PRECISION_TIE of the best count as tied and the simpler one wins (lower CNC, then lower share
  of silent transitions); if the structure is identical too, the higher share of fitting cases.

Outputs: results/iterations.{json,csv}, results/models/iterations/<id>.pnml,
         results/final_choice.json
Run after imf_sweep.py:  python src/iterations.py
"""
import json
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, ACT
from preprocess import build_log
from metrics import evaluate, timed

warnings.filterwarnings("ignore", category=UserWarning)

RES = ROOT / "results"
OUT_JSON = RES / "iterations.json"
MODEL_DIR = RES / "models" / "iterations"
FIT_TARGET = 80.0      # % fitting cases (alignments), professor's baseline
PRECISION_TIE = 0.005  # precision differences below this are treated as ties
IT2_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def log_stats(log) -> dict:
    traces = log.groupby(CASE)[ACT].agg(tuple)
    return {"cases": len(traces), "events": len(log), "variants": traces.nunique(),
            "activities": log[ACT].nunique()}


def ok(row) -> bool:
    return row.get("fit_traces_align_pct", 0) >= FIT_TARGET


def best(rows: list[dict]) -> dict | None:
    """Selection rule: max precision s.t. fitting >= 80 %, ties -> simpler (CNC, silent share)."""
    cand = [r for r in rows if ok(r)]
    if not cand:
        return None
    top = max(r["precision_tbr"] for r in cand)
    tied = [r for r in cand if top - r["precision_tbr"] < PRECISION_TIE]
    return min(tied, key=lambda r: (r["own_cnc"], r["own_silent_share"], -r["fit_traces_align_pct"]))


def run_config(results, it_id, version, merge, collapse, noise, change):
    if it_id in results and "generalization_tbr" in results[it_id]:
        print(f"[{it_id}] already done", flush=True)
        return results[it_id]
    log = build_log(version, merge_osent=merge, collapse_repeats=collapse)
    meta = {"id": it_id, "log": version, "merge_osent": merge, "collapse_repeats": collapse,
            "noise_threshold": noise, "data_change": change, **log_stats(log)}
    (net, im, fm), t_disc = timed(pm4py.discover_petri_net_inductive, log, noise_threshold=noise)
    pm4py.write_pnml(net, im, fm, str(MODEL_DIR / f"{it_id}.pnml"))
    print(f"[{it_id}] {version} merge={merge} collapse={collapse} noise={noise}: "
          f"{meta['variants']:,} variants, |P|={len(net.places)} |T|={len(net.transitions)}", flush=True)

    def save(r):
        results[it_id] = {**meta, **r, "t_discovery": t_disc}
        OUT_JSON.write_text(json.dumps(results, indent=2))

    r, _, _ = evaluate(log, net, im, fm, save=save, fast=True)
    save(r)
    row = results[it_id]
    print(f"[{it_id}] fit {row['fit_traces_align_pct']:.1f}% (TBR {row['fit_traces_tbr_nim_pct']:.1f}%)  "
          f"fitness {row['fitness_align']:.3f}  precision {row['precision_tbr']:.3f}  "
          f"CNC {row['own_cnc']:.3f}  silent {row['own_silent_share']:.2f}  "
          f"alignments {row['t_alignments']:.0f}s", flush=True)
    return row


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}

    # It0: identical to the IMf 0.10 model of the sweep -> reuse its numbers
    sweep = json.loads((RES / "imf_sweep.json").read_text())["0.10"]
    log_a = build_log("A")
    pm4py.write_pnml(*pm4py.discover_petri_net_inductive(log_a, noise_threshold=0.10),
                     str(MODEL_DIR / "It0.pnml"))
    results["It0"] = {"id": "It0", "log": "A", "merge_osent": False, "collapse_repeats": False,
                      "noise_threshold": 0.10, "data_change": "base: A_+O_ complete, 98 incomplete cases removed",
                      **log_stats(log_a), **sweep}
    OUT_JSON.write_text(json.dumps(results, indent=2))

    it1 = run_config(results, "It1", "A", True, False, 0.10,
                     "merge O_Sent (mail and online) + O_Sent (online only) -> O_Sent")
    merge = best([results["It0"], it1]) is it1
    print(f"O_Sent choice: {'merge' if merge else 'keep both labels'}", flush=True)

    sent = "O_Sent merged" if merge else "O_Sent labels kept"
    it2 = [run_config(results, f"It2_{int(th * 100):02d}", "B1", merge, False, th,
                      f"add W_ start events (one per work item started); {sent}")
           for th in IT2_THRESHOLDS]
    b2 = best(it2)
    if b2 is None:  # no B1 model reaches the target: continue with the best-fitting one
        b2 = max(it2, key=lambda r: r["fit_traces_align_pct"])
    run_config(results, "It3", "B1", merge, True, b2["noise_threshold"],
               f"as {b2['id']} + collapse direct repetitions of the same activity")

    rows = list(results.values())
    tab = pd.DataFrame(rows).set_index("id")
    tab.to_csv(RES / "iterations.csv")
    final = best(rows)
    choice = {
        "rule": (f"highest ETC precision (token replay) among model versions with fitting cases "
                 f"(alignments) >= {FIT_TARGET:g} %; precision differences < {PRECISION_TIE} count as "
                 f"ties, broken by lower CNC, then lower share of silent transitions, then more fitting cases"),
        "o_sent_choice": "merge" if merge else "keep both labels",
        "eligible": [r["id"] for r in rows if ok(r)],
        "chosen": final,
    }
    (RES / "final_choice.json").write_text(json.dumps(choice, indent=2))
    cols = ["log", "merge_osent", "collapse_repeats", "noise_threshold", "variants", "activities",
            "fit_traces_align_pct", "fit_traces_tbr_nim_pct", "fitness_align", "precision_tbr",
            "generalization_align", "own_size", "own_cnc", "own_silent_share", "t_alignments"]
    print(tab[cols].to_string(), flush=True)
    print("chosen:", None if final is None else final["id"], flush=True)


if __name__ == "__main__":
    main()
