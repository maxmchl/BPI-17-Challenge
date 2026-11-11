"""Section 3.3.1/3.3.2 - process discovery and quality metrics on the A_+O_ log.

Algorithms (PM4Py, parameters documented here and in the report):
  IM   Inductive Miner,            noise_threshold = 0.0
  IMf  Inductive Miner infrequent, noise_threshold = 0.2 (PM4Py default for IMf)
  HM   Heuristics Miner,           dependency_threshold = 0.5, and_threshold = 0.65,
                                    loop_two_threshold = 0.5 (PM4Py defaults)

Outputs
  results/models/<name>.pnml          discovered Petri nets
  figures/appendix/<name>.pdf         Petri net drawings for the appendix
  results/discovery_metrics.{csv,tex,json}
Run:  python src/discovery.py            (all models, full log)
      python src/discovery.py IMf HM     (subset)
"""
import json
import sys
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, ACT
from preprocess import ao_log
from metrics import evaluate, timed

warnings.filterwarnings("ignore", category=UserWarning)

MODELS = {
    "IM": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.0),
    "IMf": lambda log: pm4py.discover_petri_net_inductive(log, noise_threshold=0.2),
    "HM": lambda log: pm4py.discover_petri_net_heuristics(
        log, dependency_threshold=0.5, and_threshold=0.65, loop_two_threshold=0.5),
}
ALIGN_TIMEOUT_TRACE = 10.0  # seconds per variant; variants that time out are counted, not hidden
CYCLOMATIC_TIMEOUT = 120.0  # extended cyclomatic needs the reachability graph; '--' if it times out

RES = ROOT / "results"
MODEL_DIR = RES / "models"
FIG_DIR = ROOT / "figures" / "appendix"

# rows of the report table: (key, label)
ROWS = [
    ("fitness_tbr", "Fitness (token replay)"),
    ("fitness_align", "Fitness (alignments)"),
    ("fit_traces_tbr_pct", "Fitting traces \\% (token replay)"),
    ("fit_traces_tbr_nim_pct", "Fitting traces \\% (TBR, not-in-model counted)"),
    ("fit_traces_align_pct", "Fitting traces \\% (alignments)"),
    ("precision_tbr", "Precision (ETC, token replay)"),
    ("precision_align", "Precision (ETC, alignments)"),
    ("generalization_tbr", "Generalization (token replay)"),
    ("generalization_align", "Generalization (alignments)$^*$"),
    ("simp_arc_degree", "Simplicity (arc degree)"),
    ("simp_ext_cardoso", "Ext. Cardoso"),
    ("simp_ext_cyclomatic", "Ext. cyclomatic"),
    ("own_silent_share", "Silent transitions share$^*$"),
    ("own_size", "Size $|P|+|T|+|F|$$^*$"),
    ("own_cnc", "CNC$^*$"),
    ("own_density", "Density$^*$"),
    ("own_struct_approp", "Structural appropriateness $a_S$$^*$"),
    ("silent", "Silent transitions"),
    ("is_wf_net", "Workflow net"),
    ("t_discovery", "Runtime discovery [s]"),
    ("t_fitness_tbr", "Runtime fitness TBR [s]"),
    ("t_alignments", "Runtime alignments [s]"),
    ("t_precision_tbr", "Runtime precision TBR [s]"),
    ("t_precision_align", "Runtime precision align. [s]"),
    ("t_generalization_tbr", "Runtime generalization TBR [s]"),
    ("t_ext_cyclomatic", "Runtime ext. cyclomatic [s]"),
    ("alignments_failed", "Alignments timed out (traces)"),
]


def main(names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log = ao_log()
    n_var = log.groupby(CASE)[ACT].agg(tuple).nunique()
    print(f"log: {log[CASE].nunique():,} cases, {len(log):,} events, {n_var:,} variants", flush=True)

    out_json = RES / "discovery_metrics.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    for name in names:
        (net, im, fm), t_disc = timed(MODELS[name], log)
        pm4py.write_pnml(net, im, fm, str(MODEL_DIR / f"{name}.pnml"))
        pm4py.save_vis_petri_net(net, im, fm, str(FIG_DIR / f"{name}.pdf"), rankdir="LR")
        print(f"[{name}] discovered in {t_disc:.1f}s: |P|={len(net.places)} |T|={len(net.transitions)}", flush=True)
        def save(r):
            results[name] = {**r, "t_discovery": t_disc}
            out_json.write_text(json.dumps(results, indent=2))

        r, _, _ = evaluate(log, net, im, fm, ALIGN_TIMEOUT_TRACE, CYCLOMATIC_TIMEOUT, save)
        save(r)
        print(f"[{name}] " + ", ".join(f"{k}={v:.4g}" for k, v in r.items()), flush=True)
    write_tables(results)


def write_tables(results):
    tab = pd.DataFrame(results)
    tab = tab[[m for m in MODELS if m in tab.columns]]
    tab.to_csv(RES / "discovery_metrics.csv")
    fmt = tab.reindex([k for k, _ in ROWS]).astype(float)
    fmt.index = [label for _, label in ROWS]

    def cell(v, key):
        if pd.isna(v):
            return "--"
        if key == "is_wf_net":
            return "yes" if v else "no"
        if key in ("own_size", "silent", "alignments_failed"):
            return f"{v:.0f}"
        if key.startswith("t_"):
            return f"{v:.1f}"
        if key.endswith("_pct"):
            return f"{v:.1f}"
        return f"{v:.3f}"

    lines = [r"\begin{tabular}{l" + "r" * fmt.shape[1] + "}", r"\toprule",
             "Metric & " + " & ".join(fmt.columns) + r" \\", r"\midrule"]
    for (key, label), (_, row) in zip(ROWS, fmt.iterrows()):
        if key == "t_discovery":
            lines.append(r"\midrule")
        lines.append(label + " & " + " & ".join(cell(v, key) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RES / "discovery_metrics.tex").write_text("\n".join(lines) + "\n")
    print(tab.to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
