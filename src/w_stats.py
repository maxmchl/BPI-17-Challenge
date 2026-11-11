"""Section 2.2 - statistics of the W_ work items (why W_ events enter the model at lifecycle start).

On the raw log (all 31,509 cases, all lifecycle transitions):
  * events per origin (A_ / O_ / W_), absolute and in %
  * per W_ activity: cases with the work item, cases with a start, cases with a complete, and how
    the work item instances end (terminal transition complete / ate_abort / withdraw, in %)
  * work items started per case (W_ start events), mean +- sd

Outputs: results/w_stats.csv, results/w_stats.tex, results/w_stats.json
Run:  python src/w_stats.py
"""
import json

import pandas as pd

from load import load_log, ROOT, CASE, ACT, LC

RES = ROOT / "results"
TERMINAL = ["complete", "ate_abort", "withdraw"]


def main():
    df = load_log()
    origin = df[ACT].str[:2].map({"A_": "A_ (application)", "O_": "O_ (offer)", "W_": "W_ (work item)"})
    orig = origin.value_counts().rename("events").to_frame()
    orig["pct"] = (orig["events"] / len(df) * 100).round(1)

    w = df[df[ACT].str.startswith("W_")]
    g = w.groupby(ACT)
    tab = pd.DataFrame({
        "cases": g[CASE].nunique(),
        "cases_with_start": w[w[LC] == "start"].groupby(ACT)[CASE].nunique(),
        "cases_with_complete": w[w[LC] == "complete"].groupby(ACT)[CASE].nunique(),
        "starts": w[w[LC] == "start"].groupby(ACT).size(),
    }).fillna(0).astype(int)
    term = w[w[LC].isin(TERMINAL)].groupby([ACT, LC]).size().unstack(fill_value=0).reindex(columns=TERMINAL, fill_value=0)
    share = term.div(term.sum(axis=1), axis=0) * 100
    tab = tab.join(share.add_prefix("ends_").add_suffix("_pct").round(1))
    tab = tab.sort_values("cases", ascending=False)
    tab.to_csv(RES / "w_stats.csv")

    per_case = (df[(df[LC] == "start") & df[ACT].str.startswith("W_")].groupby(CASE).size()
                .reindex(df[CASE].unique(), fill_value=0))
    summary = {"events_by_origin": orig.to_dict("index"),
               "work_items_started_per_case": {"mean": round(float(per_case.mean()), 2), "sd": round(float(per_case.std()), 2),
                                               "median": float(per_case.median()), "max": int(per_case.max())}}
    (RES / "w_stats.json").write_text(json.dumps(summary, indent=2))

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Work item & Cases & w/ start & w/ complete & \multicolumn{3}{c}{Instances end in [\%]} \\",
             r" & & & & complete & ate\_abort & withdraw \\", r"\midrule"]
    pct = lambda v: "--" if pd.isna(v) else f"{v:.1f}"  # no terminal event recorded
    for act, r in tab.iterrows():
        lines.append(f"{act.strip().replace('_', chr(92) + '_')} & {int(r['cases']):,} & {int(r['cases_with_start']):,} & "
                     f"{int(r['cases_with_complete']):,} & {pct(r['ends_complete_pct'])} & "
                     f"{pct(r['ends_ate_abort_pct'])} & {pct(r['ends_withdraw_pct'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RES / "w_stats.tex").write_text("\n".join(lines) + "\n")
    print(orig.to_string())
    print(tab.to_string())
    print(summary["work_items_started_per_case"])


if __name__ == "__main__":
    main()
