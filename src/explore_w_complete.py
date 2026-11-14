"""Exploration (not part of the iteration log): can a model on W_ `complete` events reach
precision ~0.8 together with >= 80 % fitting cases (a colleague reports this)?

Does NOT touch iterations.csv, final_choice or the final model. Results only in
results/explore_w_complete.csv, results/explore_w_complete_best.json and
figures/explore_w_complete_pareto.pdf. Discovered nets are kept locally in
data/explore_w_complete_models/ (not versioned).

Log variants (log C = all activities with lifecycle complete, O_Sent merged, 31,411 cases):
  C          as is
  C-dup      without the duplicates O_Created, O_Accepted, O_Refused
  C-rareW    without W_Assess potential fraud, W_Shortened completion, W_Personal Loan collection
  C-W50      only the W_ activities that reach `complete` in >= 50 % of their cases
             (results/w_stats.csv: W_Complete application 60.6 %, W_Assess potential fraud 88.5 %;
             W_Validate application misses the threshold narrowly with 49.9 %)
  C-cluster  as in Rodrigues et al. 2017: consecutive events of a case less than 2 minutes after
             the previous one and by the same resource are merged into one clustered activity,
             labelled by its activities in order of first occurrence ("A_Validating+O_Returned").
             Automatic, so 229 cluster labels (26 cover 95 % of the clusters); Rodrigues et al.
             curated 7 clusters by hand.
  each also with "+col": direct repetitions of the same activity collapsed.
Discovery: IM on the most frequent variants covering 40-80 % of the cases (2.5 % steps);
  IMf noise 0.05 / 0.10 / 0.20 on the full log; ILP miner (alpha 0.9 / 1.0) and Heuristics Miner
  (dependency_threshold 0.9 / 0.95) on the variant-filtered logs with coverage 50 / 60 / 70 / 80 %.
  HM nets are only evaluated if Woflan confirms soundness within 60 s.
Evaluation, always on all 31,411 cases of the log variant:
  screening  token replay with the corrected not-in-model rule (metrics.tbr_fitness) and ETC
             precision with token replay; 6 parallel workers, 20 min limit per model
  candidates (screening fitting >= 75 % and precision >= 0.7): alignments (fitting cases,
             fitness), generalization (token replay)
  best 3     (highest precision with >= 80 % fitting cases by alignments): alignment-based ETC
Run:  python src/explore_w_complete.py
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import signal
import warnings

import numpy as np
import pandas as pd
import pm4py

from load import load_log, ROOT, CASE, ACT, TS, RES as RESOURCE

RES = ROOT / "results"
OUT_CSV = RES / "explore_w_complete.csv"
OUT_BEST = RES / "explore_w_complete_best.json"
MODELS = ROOT / "data" / "explore_w_complete_models"
FIT, PREC = 80.0, 0.8                   # target of the colleague's claim
SCREEN_FIT, SCREEN_PREC = 75.0, 0.7     # screening thresholds for alignments
TASK_LIMIT = 1200                       # seconds per model in screening
COVERAGES_IM = [float(c) for c in np.arange(40, 80.01, 2.5)]
COVERAGES_OTHER = [50.0, 60.0, 70.0, 80.0]
BASE_LOGS = ["C", "C-dup", "C-rareW", "C-W50", "C-cluster"]
LOGS = BASE_LOGS + [f"{n}+col" for n in BASE_LOGS]
KEY = ["log", "algorithm", "param", "coverage"]

_raw, _logs = None, {}


def w50() -> list[str]:
    w = pd.read_csv(RES / "w_stats.csv", index_col=0)
    w.index = w.index.str.strip()
    share = w["cases_with_complete"] / w["cases"]
    return sorted(share.index[share >= 0.5])


def collapse(d):
    rep = (d[ACT] == d[ACT].shift()) & (d[CASE] == d[CASE].shift())
    return d[~rep].reset_index(drop=True)


def cluster(d):
    new = ((d[CASE] != d[CASE].shift()) | (d[RESOURCE] != d[RESOURCE].shift())
           | ((d[TS] - d[TS].shift()) >= pd.Timedelta(minutes=2)))
    cid = new.cumsum()
    g = d.groupby(cid, sort=False)
    return pd.DataFrame({CASE: g[CASE].first(), ACT: g[ACT].agg(lambda s: "+".join(dict.fromkeys(s))),
                         TS: g[TS].first()}).reset_index(drop=True)


def get_log(name: str) -> pd.DataFrame:
    global _raw
    if name in _logs:
        return _logs[name]
    from preprocess import build_log, B1C_DROP, RARE_W
    if _raw is None:
        _raw = load_log()
    base, col = name.removesuffix("+col"), name.endswith("+col")
    d = build_log("C", merge_osent=True, df=_raw, keep_attrs=True)[[CASE, ACT, TS, RESOURCE]]
    if base == "C-dup":
        d = d[~d[ACT].isin(B1C_DROP)]
    elif base == "C-rareW":
        d = d[~d[ACT].str.strip().isin([a.strip() for a in RARE_W])]
    elif base == "C-W50":
        d = d[~d[ACT].str.startswith("W_") | d[ACT].str.strip().isin(w50())]
    d = d.reset_index(drop=True)
    if base == "C-cluster":
        d = cluster(d)
    d = d[[CASE, ACT, TS]]
    if col:
        d = collapse(d)
    _logs[name] = d
    return d


def configs():
    for log in LOGS:
        for c in COVERAGES_IM:
            yield {"log": log, "algorithm": "IM", "param": 0.0, "coverage": c}
        for n in [0.05, 0.10, 0.20]:
            yield {"log": log, "algorithm": "IMf", "param": n, "coverage": 100.0}
        for c in COVERAGES_OTHER:
            for a in [0.9, 1.0]:
                yield {"log": log, "algorithm": "ILP", "param": a, "coverage": c}
            for t in [0.9, 0.95]:
                yield {"log": log, "algorithm": "HM", "param": t, "coverage": c}


def model_path(cfg) -> str:
    return str(MODELS / f"{cfg['log']}_{cfg['algorithm']}_{cfg['param']:g}_c{cfg['coverage']:g}.pnml")


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def discover(cfg):
    from preprocess import coverage_filter
    full = get_log(cfg["log"])
    dlog = full if cfg["coverage"] >= 100 else coverage_filter(full, cfg["coverage"])
    a, p = cfg["algorithm"], cfg["param"]
    if a in ("IM", "IMf"):
        return pm4py.discover_petri_net_inductive(dlog, noise_threshold=p), dlog
    if a == "ILP":
        return pm4py.discover_petri_net_ilp(dlog, alpha=p), dlog
    return pm4py.discover_petri_net_heuristics(dlog, dependency_threshold=p), dlog


def screen(cfg) -> dict:
    """Worker: discover, (HM: soundness), structure, token-based fitting and precision."""
    warnings.filterwarnings("ignore")
    from metrics import structural, tbr_fitness, timed
    row = dict(cfg)
    signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(TASK_LIMIT)
        full = get_log(cfg["log"])
        ((net, im, fm), dlog), row["t_discovery"] = timed(discover, cfg)
        row.update({"cases": full[CASE].nunique(), "activities": full[ACT].nunique(),
                    "variants_discovery": dlog.groupby(CASE)[ACT].agg(tuple).nunique()})
        if cfg["algorithm"] == "HM":
            signal.alarm(60)
            try:
                row["sound"] = bool(pm4py.check_soundness(net, im, fm)[0])
            except Timeout:
                row["sound"] = None
            signal.alarm(TASK_LIMIT)
            if not row["sound"]:
                row["status"] = "skipped: not a sound WF-net" if row["sound"] is False else "skipped: Woflan > 60 s"
                return row
        pm4py.write_pnml(net, im, fm, model_path(cfg))
        row.update(structural(net, im))
        row["model_activities"] = len({t.label for t in net.transitions if t.label is not None})
        (f, _), row["t_fitness_tbr"] = timed(tbr_fitness, full, net, im, fm, True)
        row["fit_traces_tbr_pct"], row["fitness_tbr"] = f["percentage_of_fitting_traces"], f["log_fitness"]
        row["precision_tbr"], row["t_precision_tbr"] = timed(pm4py.precision_token_based_replay, full, net, im, fm)
        row["status"] = "ok"
    except Timeout:
        row["status"] = f"timeout (> {TASK_LIMIT} s)"
    except Exception as e:  # noqa: BLE001 - record and continue with the other models
        row["status"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    finally:
        signal.alarm(0)
    return row


def save(rows):
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


def main():
    warnings.filterwarnings("ignore")
    from metrics import align, precision_alignments, timed
    from pm4py.algo.evaluation.replay_fitness.variants import alignment_based as align_fitness
    MODELS.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(OUT_CSV).to_dict("records") if OUT_CSV.exists() else []
    done = {tuple(r[k] for k in KEY) for r in rows}
    todo = [c for c in configs() if tuple(c[k] for k in KEY) not in done]
    todo.sort(key=lambda c: c["algorithm"] == "ILP")  # slow ILP runs last
    print(f"screening {len(todo)} models ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, fut in enumerate(as_completed([pool.submit(screen, c) for c in todo])):
            r = fut.result()
            rows.append(r)
            save(rows)
            print(f"[{i + 1}/{len(todo)}] {r['log']} {r['algorithm']} {r['param']:g} c{r['coverage']:g}: {r['status']}"
                  + (f"  fit {r['fit_traces_tbr_pct']:.1f}%  precision {r['precision_tbr']:.3f}"
                     if r["status"] == "ok" else ""), flush=True)

    tab = pd.DataFrame(rows)
    cand = tab[(tab["status"] == "ok") & (tab["fit_traces_tbr_pct"] >= SCREEN_FIT)
               & (tab["precision_tbr"] >= SCREEN_PREC)]
    print(f"alignment candidates: {len(cand)}", flush=True)
    for i in cand.index:
        if "fit_traces_align_pct" in tab and pd.notna(tab.at[i, "fit_traces_align_pct"]):
            continue
        cfg = tab.loc[i, KEY].to_dict()
        net, im, fm = pm4py.read_pnml(model_path(cfg))
        full = get_log(cfg["log"])
        al, t = timed(align, pm4py.convert_to_event_log(full), net, im, fm)
        ok = [a for a in al if a is not None]
        af = align_fitness.evaluate(ok)
        tab.loc[i, ["fit_traces_align_pct", "fitness_align", "alignments_failed", "t_alignments"]] = [
            af["percentage_of_fitting_traces"], af["log_fitness"], len(al) - len(ok), t]
        tab.loc[i, "generalization_tbr"] = pm4py.generalization_tbr(full, net, im, fm)
        save(tab.to_dict("records"))
        print(f"  {cfg}: align fit {af['percentage_of_fitting_traces']:.1f}% ({t:.0f}s)", flush=True)

    tab = pd.read_csv(OUT_CSV)
    ok80 = tab[tab.get("fit_traces_align_pct", pd.Series(dtype=float)) >= FIT] if "fit_traces_align_pct" in tab else tab.iloc[:0]
    best3 = ok80.sort_values("precision_tbr", ascending=False).head(3)
    for i in best3.index:
        if "precision_align" in tab and pd.notna(tab.at[i, "precision_align"]):
            continue
        cfg = tab.loc[i, KEY].to_dict()
        net, im, fm = pm4py.read_pnml(model_path(cfg))
        tab.loc[i, "precision_align"], tab.loc[i, "t_precision_align"] = timed(
            precision_alignments, get_log(cfg["log"]), net, im, fm)
        tab.to_csv(OUT_CSV, index=False)
        print(f"  {cfg}: precision (alignments) {tab.at[i, 'precision_align']:.3f}", flush=True)
    report(pd.read_csv(OUT_CSV))


def pareto(tab):
    d = tab[tab["status"] == "ok"].copy()
    d["fit"] = d["fit_traces_align_pct"].fillna(d["fit_traces_tbr_pct"]) if "fit_traces_align_pct" in d else d["fit_traces_tbr_pct"]
    d = d.sort_values(["fit", "precision_tbr"], ascending=False)
    front, best = [], -1.0
    for i, r in d.iterrows():
        if r["precision_tbr"] > best:
            front.append(i)
            best = r["precision_tbr"]
    return d, d.loc[front].sort_values("fit")


def report(tab):
    from figures_stats import C, W, INK2, plt
    d, front = pareto(tab)
    final_net = pm4py.read_pnml(str(RES / "models" / "final.pnml"))[0]
    final_acts = {t.label for t in final_net.transitions if t.label}
    hit = d[(d["fit"] >= FIT) & (d["precision_tbr"] >= PREC)]
    ok80 = d[d["fit"] >= FIT].sort_values("precision_tbr", ascending=False)
    cols = KEY + ["activities", "model_activities", "variants_discovery", "fit_traces_tbr_pct", "fit_traces_align_pct",
                  "fitness_align", "precision_tbr", "precision_align", "generalization_tbr", "own_size", "own_cnc",
                  "own_silent_share", "is_wf_net"]
    best_rows = []
    for _, r in ok80.head(5).iterrows():
        x = {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in cols if k in r}
        net = pm4py.read_pnml(model_path(r))[0]
        acts = {t.label for t in net.transitions if t.label}
        x["activities_of_final_model_missing"] = sorted(final_acts - acts)
        x["activities_not_in_final_model"] = sorted(acts - final_acts)
        best_rows.append(x)
    out = {"question": f"precision >= {PREC} with >= {FIT:g} % fitting cases on W_ complete logs?",
           "reachable": bool(len(hit)),
           "models_screened": int(len(tab)), "models_evaluated": int(len(d)),
           "status_counts": tab["status"].str.split(":").str[0].value_counts().to_dict(),
           "fitting_measure": "alignments where computed, otherwise token replay with the not-in-model rule",
           "best_with_80pct_fitting": best_rows,
           "pareto_front": front[KEY + ["fit", "precision_tbr"]].round(4).to_dict("records"),
           "final_model_for_comparison": "It6_c75 (log B1c): 89.4 % fitting, precision 0.700 (not comparable: other activity set)"}
    OUT_BEST.write_text(json.dumps(out, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.scatter(d["fit"], d["precision_tbr"], s=8, color="#b9b8b3", lw=0, label="screened models", zorder=2)
    ax.plot(front["fit"], front["precision_tbr"], color=C[0], marker="o", ms=4, mec="white", mew=0.8, lw=1.5,
            label="Pareto front", zorder=3)
    ax.axvline(FIT, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(PREC, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("fitting cases [%] (full log)")
    ax.set_ylabel("precision (ETC, token replay)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures" / "explore_w_complete_pareto.pdf")
    fig.savefig(ROOT / "figures" / "explore_w_complete_pareto.png", dpi=200)
    plt.close(fig)
    print(json.dumps({k: out[k] for k in ["reachable", "status_counts"]}, indent=1))
    print(pd.DataFrame(best_rows)[[c for c in cols if c in pd.DataFrame(best_rows)]].round(3).to_string() if best_rows else "no model >= 80 %")
    print(front[KEY + ["fit", "precision_tbr"]].round(3).to_string())


if __name__ == "__main__":
    main()
