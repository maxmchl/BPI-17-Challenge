"""Section 3.2 - simple event log statistics for BPIC-17.

Writes results/basic_stats.csv, results/basic_stats.tex and results/extra_stats.json.
Run:  python src/stats.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from load import load_log, ROOT, CASE, ACT, TS, LC, RES

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CASE_ATTRS = ["case:concept:name", "case:LoanGoal", "case:ApplicationType", "case:RequestedAmount"]
ID_ATTRS = ["EventID", "OfferID"]  # identifiers: nominal, but not analytically categorical


def fmt_duration(sec: float) -> str:
    return f"{sec / 86400:.2f} d = {sec / 60:,.0f} min = {sec:,.0f} s"


def main():
    df = load_log().sort_values([CASE, TS, "EventID"], kind="stable")
    g = df.groupby(CASE, sort=False)

    # --- variants: sequence of activity labels (concept:name) per case
    traces = g[ACT].agg(tuple)
    variant_counts = traces.value_counts()

    # --- case length and duration
    length = g.size()
    dur = (g[TS].max() - g[TS].min()).dt.total_seconds()

    event_attrs = [c for c in df.columns if not c.startswith("case:")]
    numeric_ev = [c for c in event_attrs if pd.api.types.is_numeric_dtype(df[c])]
    ts_ev = [c for c in event_attrs if pd.api.types.is_datetime64_any_dtype(df[c])]
    nominal_ev = [c for c in event_attrs if c not in numeric_ev + ts_ev]
    categorical_ev = [c for c in nominal_ev if c not in ID_ATTRS]

    rows = [
        ("Number of cases", f"{df[CASE].nunique():,}"),
        ("Number of events", f"{len(df):,}"),
        ("Number of process variants", f"{len(variant_counts):,}"),
        ("Number of activities", f"{df[ACT].nunique()}"),
        ("Number of resources", f"{df[RES].nunique()}"),
        ("Distinct case attribute labels", f"{len(CASE_ATTRS)}"),
        ("Distinct event attribute labels", f"{len(event_attrs)}"),
        ("Categorical event attributes", f"{len(categorical_ev)} (+{len(ID_ATTRS)} ID attributes)"),
        ("Case length (events), mean ± sd", f"{length.mean():.2f} ± {length.std():.2f}"),
        ("Case duration, mean", fmt_duration(dur.mean())),
        ("Case duration, sd", fmt_duration(dur.std())),
    ]
    tab = pd.DataFrame(rows, columns=["Metric", "Value"])
    tab.to_csv(OUT / "basic_stats.csv", index=False)
    tex = tab.to_latex(index=False, escape=True, column_format="lr")
    (OUT / "basic_stats.tex").write_text(tex.replace("±", r"$\pm$"))
    print(tab.to_string(index=False))

    # ---------------- additional statistics ----------------
    extra = {}
    extra["log_period"] = [str(df[TS].min()), str(df[TS].max())]
    extra["event_attrs"] = {"numeric": numeric_ev, "timestamp": ts_ev,
                            "categorical": categorical_ev, "identifiers": ID_ATTRS}
    extra["events_per_origin"] = df["EventOrigin"].value_counts().to_dict()
    extra["lifecycle_values"] = df[LC].value_counts().to_dict()
    extra["case_attr_values"] = {c: df.drop_duplicates(CASE)[c].value_counts().head(15).to_dict()
                                 for c in ["case:LoanGoal", "case:ApplicationType"]}
    extra["requested_amount"] = df.drop_duplicates(CASE)["case:RequestedAmount"].describe().round(1).to_dict()

    # variant coverage: how many variants are needed to cover x% of cases?
    cum = variant_counts.cumsum() / variant_counts.sum()
    extra["variant_coverage"] = {f"{int(p*100)}%": int((cum < p).sum() + 1) for p in [0.25, 0.5, 0.6, 0.7, 0.8, 0.9]}
    extra["singleton_variants"] = int((variant_counts == 1).sum())
    extra["top_variant_share"] = round(float(variant_counts.iloc[0] / variant_counts.sum()), 4)

    # offers per application
    offers = df[df[ACT] == "O_Create Offer"].groupby(CASE).size().reindex(length.index, fill_value=0)
    extra["offers_per_case"] = offers.value_counts().sort_index().to_dict()

    # case outcome = which terminal A_ state the case reaches
    outcome_acts = ["A_Pending", "A_Denied", "A_Cancelled"]
    has = {a: set(df.loc[df[ACT] == a, CASE]) for a in outcome_acts}
    def outcome(c):
        hits = [a for a in outcome_acts if c in has[a]]
        return hits[0] if len(hits) == 1 else ("none" if not hits else "+".join(hits))
    out = pd.Series({c: outcome(c) for c in length.index})
    extra["outcome_counts"] = out.value_counts().to_dict()
    extra["duration_days_by_outcome"] = (dur / 86400).groupby(out).describe()[["count", "mean", "50%", "std"]].round(2).to_dict("index")

    # incomplete cases: no outcome and last event close to log end
    log_end = df[TS].max()
    last = g[TS].max()
    extra["cases_without_outcome"] = int((out == "none").sum())
    extra["cases_without_outcome_last_event_within_30d_of_log_end"] = int(((out == "none") & (last > log_end - pd.Timedelta(days=30))).sum())

    # long vs short cases (top / bottom decile by duration)
    q10, q90 = dur.quantile([0.1, 0.9])
    extra["duration_deciles_days"] = {"p10": round(q10 / 86400, 2), "p90": round(q90 / 86400, 2)}
    for name, mask in [("short_p10", dur <= q10), ("long_p90", dur >= q90)]:
        cs = dur.index[mask]
        extra[name] = {
            "n": int(mask.sum()),
            "mean_len": round(float(length[cs].mean()), 1),
            "mean_offers": round(float(offers[cs].mean()), 2),
            "outcomes": out[cs].value_counts().to_dict(),
            "top_variant": " > ".join(traces[cs].value_counts().index[0]),
        }

    (OUT / "extra_stats.json").write_text(json.dumps(extra, indent=2, default=str))
    print(json.dumps(extra, indent=2, default=str))

    # save per-case table for plotting
    pd.DataFrame({"length": length, "duration_s": dur, "offers": offers, "outcome": out,
                  "start": g[TS].min()}).to_parquet(OUT / "cases.parquet")


if __name__ == "__main__":
    main()
