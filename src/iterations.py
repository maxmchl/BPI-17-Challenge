"""Section 2.3 - iteration log: change the data, rediscover, evaluate (fast metrics).

Each row is one model version. No manual edits of the Petri net: every change is a change of the
event log (or of the discovery log) followed by rediscovery (professor, Nov 6).

  It0   log A (A_+O_ complete), IMf 0.10                         base (numbers from imf_sweep.json)
  It1   log A, both O_Sent labels merged into "O_Sent", IMf 0.10
  It2x  log B1 (A + W_ start events), O_Sent choice from It1, IMf 0.05 / 0.10 / 0.15 / 0.20
  It3   best It2 + direct repetitions of the same activity collapsed
  It4x  log A (merged O_Sent),  IM on the most frequent variants covering x % of the cases
  It5x  log B1 (merged O_Sent), IM on the most frequent variants covering x % of the cases
  It6x  log B1c (B1 without O_Created / O_Accepted / O_Refused), IM with variant coverage x %
  It8x  log C (all activities with lifecycle complete, the professor's choice), IM, x = 60 / 70
  It7   best It6 without the rare work items (W_Assess potential fraud, W_Shortened completion,
        W_Personal Loan collection)
  x in COVERAGES. The variant filter only builds the discovery log; every metric is computed on
  the full log of the version (31,411 cases). Precision is also reported on the discovery log:
  ETC penalizes continuations the model allows but the log never shows, so it depends on the
  log it is computed on (precision_tbr_discovery_log vs precision_tbr).

Fast metrics (metrics.evaluate(fast=True)): no alignment-based precision and no ext. cyclomatic.
Precision is ETC precision with token replay; over the IMf sweep it matched the alignment-based
value within 0.002.

Outputs: results/iterations.{json,csv}, results/models/iterations/<id>.pnml
The final model is selected afterwards by final_choice.py.
Run after imf_sweep.py:  python src/iterations.py
"""
import json
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, ACT
from preprocess import build_log, log_for, discover, discovery_log
from metrics import evaluate, timed

warnings.filterwarnings("ignore", category=UserWarning)

RES = ROOT / "results"
OUT_JSON = RES / "iterations.json"
MODEL_DIR = RES / "models" / "iterations"
FIT_TARGET = 80.0      # % fitting cases (alignments), professor's baseline
PRECISION_TIE = 0.005  # precision differences below this are treated as ties
IT2_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
COVERAGES = [60, 65, 70, 72.5, 75, 80, 90]


def log_stats(log) -> dict:
    traces = log.groupby(CASE)[ACT].agg(tuple)
    return {"cases": len(traces), "events": len(log), "variants": traces.nunique(),
            "activities": log[ACT].nunique()}


def ok(row) -> bool:
    return row.get("fit_traces_align_pct", 0) >= FIT_TARGET


def best(rows: list[dict]) -> dict | None:
    """Selection rule: max precision s.t. fitting >= 80 %, ties -> simpler (CNC, silent share),
    then more fitting cases."""
    cand = [r for r in rows if ok(r)]
    if not cand:
        return None
    top = max(r["precision_tbr"] for r in cand)
    tied = [r for r in cand if top - r["precision_tbr"] < PRECISION_TIE]
    return min(tied, key=lambda r: (r["own_cnc"], r["own_silent_share"], -r["fit_traces_align_pct"]))


def run_config(results, it_id, change, log="A", merge_osent=False, collapse_repeats=False,
               noise_threshold=0.0, coverage=None, drop_rare_w=False):
    if it_id in results and "generalization_tbr" in results[it_id]:
        print(f"[{it_id}] already done", flush=True)
        return results[it_id]
    cfg = {"id": it_id, "log": log, "merge_osent": merge_osent or log == "B1c",
           "collapse_repeats": collapse_repeats, "drop_rare_w": drop_rare_w,
           "algorithm": "IM" if noise_threshold == 0 else "IMf", "noise_threshold": noise_threshold,
           "coverage": coverage, "data_change": change}
    full = log_for(cfg)
    dlog = discovery_log(full, cfg)
    meta = {**cfg, **log_stats(full),
            "variants_discovery": dlog.groupby(CASE)[ACT].agg(tuple).nunique(),
            "cases_discovery": dlog[CASE].nunique()}
    (net, im, fm), t_disc = timed(discover, full, cfg)
    pm4py.write_pnml(net, im, fm, str(MODEL_DIR / f"{it_id}.pnml"))
    print(f"[{it_id}] {log} {cfg['algorithm']} noise={noise_threshold} coverage={coverage}: "
          f"{meta['variants_discovery']:,}/{meta['variants']:,} variants, "
          f"|P|={len(net.places)} |T|={len(net.transitions)}", flush=True)

    def save(r):
        results[it_id] = {**meta, **r, "t_discovery": t_disc}
        OUT_JSON.write_text(json.dumps(results, indent=2))

    r, _, _ = evaluate(full, net, im, fm, save=save, fast=True)
    r["precision_tbr_discovery_log"] = (r["precision_tbr"] if dlog is full else
                                        pm4py.precision_token_based_replay(dlog, net, im, fm))
    save(r)
    row = results[it_id]
    print(f"[{it_id}] fit {row['fit_traces_align_pct']:.1f}% (TBR {row['fit_traces_tbr_nim_pct']:.1f}%)  "
          f"fitness {row['fitness_align']:.3f}  precision {row['precision_tbr']:.3f} "
          f"(discovery log {row['precision_tbr_discovery_log']:.3f})  CNC {row['own_cnc']:.3f}  "
          f"silent {row['own_silent_share']:.2f}  alignments {row['t_alignments']:.0f}s", flush=True)
    return row


def cov_id(prefix, c):
    return f"{prefix}_c{c:g}"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}

    # It0: identical to the IMf 0.10 model of the sweep -> reuse its numbers
    sweep = json.loads((RES / "imf_sweep.json").read_text())["0.10"]
    log_a = build_log("A")
    pm4py.write_pnml(*pm4py.discover_petri_net_inductive(log_a, noise_threshold=0.10),
                     str(MODEL_DIR / "It0.pnml"))
    results["It0"] = {**results.get("It0", {}),
                      "id": "It0", "log": "A", "merge_osent": False, "collapse_repeats": False,
                      "drop_rare_w": False, "algorithm": "IMf", "noise_threshold": 0.10, "coverage": None,
                      "data_change": "base: A_+O_ complete, 98 incomplete cases removed",
                      **log_stats(log_a), **sweep}

    it1 = run_config(results, "It1", "merge O_Sent (mail and online) + O_Sent (online only) -> O_Sent",
                     "A", merge_osent=True, noise_threshold=0.10)
    merge = best([results["It0"], it1]) is it1
    print(f"O_Sent choice: {'merge' if merge else 'keep both labels'}", flush=True)

    sent = "O_Sent merged" if merge else "O_Sent labels kept"
    it2 = [run_config(results, f"It2_{int(th * 100):02d}",
                      f"add W_ start events (one per work item started); {sent}",
                      "B1", merge_osent=merge, noise_threshold=th)
           for th in IT2_THRESHOLDS]
    b2 = best(it2) or max(it2, key=lambda r: r["fit_traces_align_pct"])
    run_config(results, "It3", f"as {b2['id']} + collapse direct repetitions of the same activity",
               "B1", merge_osent=merge, collapse_repeats=True, noise_threshold=b2["noise_threshold"])

    for prefix, log, desc in [("It4", "A", "log A (merged O_Sent)"), ("It5", "B1", "log B1 (merged O_Sent)"),
                              ("It6", "B1c", "log B1c (B1 without O_Created/O_Accepted/O_Refused)")]:
        for c in COVERAGES:
            run_config(results, cov_id(prefix, c), f"{desc}; IM on variants covering {c:g} % of cases",
                       log, merge_osent=True, coverage=c)
    it6 = [results[cov_id("It6", c)] for c in COVERAGES]
    b6 = best(it6) or max(it6, key=lambda r: r["fit_traces_align_pct"])
    run_config(results, "It7", f"as {b6['id']} without rare work items (fraud check, shortened "
               f"completion, loan collection)", "B1c", merge_osent=True, coverage=b6["coverage"],
               drop_rare_w=True)

    for c in [60, 70]:  # professor's representation (lifecycle complete for all activities), comparison
        run_config(results, cov_id("It8", c), f"log C (all activities, lifecycle complete, merged O_Sent); "
                   f"IM on variants covering {c:g} % of cases", "C", merge_osent=True, coverage=c)

    for r in results.values():  # IMf rows discover on the full log
        r.setdefault("variants_discovery", r["variants"])
        r.setdefault("cases_discovery", r["cases"])
        r.setdefault("precision_tbr_discovery_log", r["precision_tbr"])
        r.setdefault("drop_rare_w", False)
        r.setdefault("algorithm", "IMf")
        r.setdefault("coverage", None)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    tab = pd.DataFrame(list(results.values())).set_index("id")
    tab.to_csv(RES / "iterations.csv")
    cols = ["log", "algorithm", "noise_threshold", "coverage", "variants_discovery", "activities",
            "fit_traces_align_pct", "fitness_align", "precision_tbr", "precision_tbr_discovery_log",
            "own_size", "own_cnc", "own_silent_share", "t_alignments"]
    pd.set_option("display.width", 250)
    print(tab[cols].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
