"""Section 3.3 - IMf noise-threshold sweep: fitness vs precision trade-off (first iteration-log rows).

For each noise threshold the IMf model is discovered on the A_+O_ log and evaluated with the
full metric set (src/metrics.py). Thresholds 0.0 (= plain IM) and 0.2 are taken from
results/discovery_metrics.json when present, since they are the identical models.

Outputs: results/imf_sweep.{csv,json}, figures/imf_sweep.pdf
Run after discovery.py:  python src/imf_sweep.py
"""
import json
import warnings

import pandas as pd
import pm4py

from load import ROOT
from preprocess import ao_log
from metrics import evaluate, timed
from figures_stats import C, W, INK2, plt  # shared figure style

warnings.filterwarnings("ignore", category=UserWarning)

THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
REUSE = {0.0: "IM", 0.2: "IMf"}  # identical models already evaluated in discovery.py
RES = ROOT / "results"
OUT_JSON = RES / "imf_sweep.json"


def run():
    log = ao_log()
    disc = json.loads((RES / "discovery_metrics.json").read_text())
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for th in THRESHOLDS:
        key = f"{th:.2f}"
        if th in REUSE and REUSE[th] in disc and "simp_ext_cyclomatic" in disc[REUSE[th]]:
            results[key] = disc[REUSE[th]]
            print(f"[{key}] reused {REUSE[th]} from discovery_metrics.json", flush=True)
            continue
        if key in results and "simp_ext_cyclomatic" in results[key]:
            print(f"[{key}] already done", flush=True)
            continue
        (net, im, fm), t_disc = timed(pm4py.discover_petri_net_inductive, log, noise_threshold=th)

        def save(r):
            results[key] = {**r, "t_discovery": t_disc}
            OUT_JSON.write_text(json.dumps(results, indent=2))

        r, _, _ = evaluate(log, net, im, fm, save=save)
        save(r)
        print(f"[{key}] fit {r['fit_traces_align_pct']:.1f}%  fitness {r['fitness_align']:.3f}  "
              f"precision {r['precision_align']:.3f}  size {r['own_size']}", flush=True)
    tab = pd.DataFrame(results).T.sort_index()
    tab.index.name = "noise_threshold"
    tab.to_csv(RES / "imf_sweep.csv")
    return tab


def plot(tab):
    x = tab["fit_traces_align_pct"].astype(float)
    y = tab["precision_align"].astype(float)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(x, y, color=C[0], marker="o", ms=4, mec="white", mew=1)
    for th, xi, yi in zip(tab.index, x, y):
        ax.annotate(f"{float(th):g}", (xi, yi), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.axvline(80, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(80.5, ax.get_ylim()[0], "80 % of cases", color=INK2, fontsize=7, va="bottom")
    ax.set_xlabel("fitting cases [%] (alignments)")
    ax.set_ylabel("precision (ETC, alignments)")
    fig.savefig(ROOT / "figures" / "imf_sweep.pdf")
    fig.savefig(ROOT / "figures" / "imf_sweep.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    tab = run()
    plot(tab)
    cols = ["fit_traces_align_pct", "fitness_align", "fit_traces_tbr_pct", "fitness_tbr",
            "precision_align", "precision_tbr", "generalization_align", "own_size", "own_silent_share"]
    print(tab[cols].astype(float).round(3).to_string())
