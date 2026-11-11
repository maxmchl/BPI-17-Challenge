"""Event abstraction and case filtering shared by discovery, modelling and decision mining.

Abstraction: keep only `lifecycle:transition == complete` events of the application (A_) and
offer (O_) sub-processes. W_ (workflow) events are dropped for control-flow discovery; they
carry resources and processing times and are used again for simulation-related analyses.
Effect on variants (Section 2.2): 80 % of cases need 9,629 variants in the raw log, 65 here.

Filter: cases without a terminal application state (A_Pending / A_Denied / A_Cancelled) are
removed. There are 98 of them; 97 were still active in the last 30 days of the log, i.e. they
are incomplete cases cut off by the log end.

Log versions (iteration log, Section 2.3):
  A   A_ + O_ events with lifecycle complete (the abstraction above).
  B1  A + W_ events with lifecycle start: one event per work item that a bank employee actually
      started. `complete` is not usable for W_ (most work items end in ate_abort/withdraw, e.g.
      W_Call after offers completes in 341 cases), and the terminal event would add automatic
      withdraws (16,802 W_Handle leads), which are not bank work. All W_ start events are
      executed by employees (share of User_1 = 0 for every W_ activity).
  B1c B1 with merged O_Sent and without the events that only duplicate information
      (checked by redundancy_check(), results/b1c_redundancy.json):
        O_Created   always directly follows O_Create Offer, same count in every case
        O_Accepted  same count as A_Pending in every case
        O_Refused   one per open offer when the application is denied (count set by the
                    number of offers; 32 denied cases have none)
Options:
  merge_osent       map "O_Sent (mail and online)" and "O_Sent (online only)" to "O_Sent".
  collapse_repeats  keep only the first of directly repeated identical activities in a case.
  drop_rare_w       drop the rare work items W_Assess potential fraud, W_Shortened completion,
                    W_Personal Loan collection.
Variant coverage filter (coverage_filter): keep the most frequent variants that together cover
x % of the cases. Used ONLY to build the discovery log; every metric is evaluated on the full
log of the same version (31,411 cases).

Run `python src/stats.py` first (writes results/cases.parquet with the per-case outcome).
"""
import json

import pandas as pd
import pm4py

from load import load_log, ROOT, CASE, ACT, TS, LC

CASES = ROOT / "results" / "cases.parquet"


def complete_case_ids() -> pd.Index:
    cases = pd.read_parquet(CASES)
    return cases.index[cases["outcome"] != "none"]


def ao_log(df: pd.DataFrame | None = None, keep_attrs: bool = False) -> pd.DataFrame:
    """A_ + O_ complete events of complete cases, sorted by case and time.

    keep_attrs=False returns only case id, activity and timestamp (faster for conformance).
    """
    if df is None:
        df = load_log()
    mask = (df[CASE].isin(complete_case_ids())
            & (df[LC] == "complete")
            & df[ACT].str[:2].isin(["A_", "O_"]))
    out = df[mask].sort_values([CASE, TS, "EventID"], kind="stable")
    if not keep_attrs:
        out = out[[CASE, ACT, TS]]
    return out.reset_index(drop=True)


OSENT = ("O_Sent (mail and online)", "O_Sent (online only)")
B1C_DROP = ("O_Created", "O_Accepted", "O_Refused")
RARE_W = ("W_Assess potential fraud", "W_Shortened completion", "W_Personal Loan collection")


def build_log(version: str = "A", merge_osent: bool = False, collapse_repeats: bool = False,
              df: pd.DataFrame | None = None, keep_attrs: bool = False,
              drop_rare_w: bool = False) -> pd.DataFrame:
    """Event log for one iteration-log configuration (see module docstring)."""
    if df is None:
        df = load_log()
    keep = (df[LC] == "complete") & df[ACT].str[:2].isin(["A_", "O_"])
    if version in ("B1", "B1c"):
        keep |= (df[LC] == "start") & df[ACT].str.startswith("W_")
    elif version != "A":
        raise ValueError(f"unknown log version {version!r}")
    if version == "B1c":
        merge_osent = True
        keep &= ~df[ACT].isin(B1C_DROP)
    if drop_rare_w:
        keep &= ~df[ACT].isin(RARE_W)
    out = df[df[CASE].isin(complete_case_ids()) & keep].sort_values([CASE, TS, "EventID"], kind="stable")
    if not keep_attrs:
        out = out[[CASE, ACT, TS]]
    out = out.reset_index(drop=True)
    if merge_osent:
        out[ACT] = out[ACT].where(~out[ACT].isin(OSENT), "O_Sent")
    if collapse_repeats:
        rep = (out[ACT] == out[ACT].shift()) & (out[CASE] == out[CASE].shift())
        out = out[~rep].reset_index(drop=True)
    return out


def coverage_filter(log: pd.DataFrame, coverage: float) -> pd.DataFrame:
    """Keep the most frequent variants that together cover `coverage` % of the cases
    (smallest such set; ties in frequency broken by the variant itself, so it is deterministic)."""
    traces = log.groupby(CASE, sort=False)[ACT].agg(tuple)
    counts = traces.value_counts().reset_index()
    counts.columns = ["variant", "n"]
    counts["key"] = counts["variant"].map(str)
    counts = counts.sort_values(["n", "key"], ascending=[False, True]).reset_index(drop=True)
    k = int((counts["n"].cumsum() / counts["n"].sum() * 100 < coverage).sum()) + 1
    keep = set(counts["variant"].iloc[:k])
    return log[log[CASE].isin(traces.index[traces.isin(keep)])]


def discovery_log(log: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cov = cfg.get("coverage")
    return log if cov is None or pd.isna(cov) else coverage_filter(log, cov)


def discover(log: pd.DataFrame, cfg: dict, tree: bool = False):
    """Rediscover the model of an iteration-log row: (optional) variant filter, then IM/IMf
    with the row's noise threshold (IM = 0.0). Metrics are always computed on the full `log`."""
    dlog = discovery_log(log, cfg)
    f = pm4py.discover_process_tree_inductive if tree else pm4py.discover_petri_net_inductive
    return f(dlog, noise_threshold=cfg["noise_threshold"])


def log_for(cfg: dict, keep_attrs: bool = False, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full (evaluation) log of an iteration-log row."""
    return build_log(cfg["log"], merge_osent=bool(cfg["merge_osent"]),
                     collapse_repeats=bool(cfg["collapse_repeats"]), df=df, keep_attrs=keep_attrs,
                     drop_rare_w=bool(cfg.get("drop_rare_w", False)))


def final_log(keep_attrs: bool = False, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Full log of the configuration chosen in final_choice.py (results/final_choice.json) and that row."""
    cfg = json.loads((ROOT / "results" / "final_choice.json").read_text())["chosen"]
    return log_for(cfg, keep_attrs, df), cfg


def redundancy_check(df: pd.DataFrame | None = None) -> dict:
    """Evidence for dropping O_Created / O_Accepted / O_Refused in B1c."""
    b = build_log("B1", merge_osent=True, df=df)
    c = b.groupby([CASE, ACT]).size().unstack(fill_value=0)
    nxt = b.groupby(CASE)[ACT].shift(-1)
    out = {"O_Create Offer directly followed by O_Created (share)":
           float((nxt[b[ACT] == "O_Create Offer"] == "O_Created").mean())}
    for a, ref in [("O_Created", "O_Create Offer"), ("O_Accepted", "A_Pending"), ("O_Refused", "A_Denied")]:
        out[f"{a} vs {ref}"] = {"events": [int(c[a].sum()), int(c[ref].sum())],
                                "cases with different counts": int((c[a] != c[ref]).sum()),
                                "cases with one but not the other": int(((c[a] > 0) != (c[ref] > 0)).sum())}
    den = c[c["A_Denied"] > 0]
    out["O_Refused in denied cases: equals number of offers (share)"] = float(
        (den["O_Refused"] == den["O_Create Offer"]).mean())
    return out


if __name__ == "__main__":
    red = redundancy_check()
    (ROOT / "results" / "b1c_redundancy.json").write_text(json.dumps(red, indent=2))
    print(json.dumps(red, indent=1))
    for cfg in [("A", False, False), ("A", True, False), ("B1", False, False), ("B1", True, True),
                ("B1c", True, False)]:
        d = build_log(*cfg)
        traces = d.groupby(CASE)[ACT].agg(tuple)
        print(f"{cfg}: cases {len(traces):,}  events {len(d):,}  variants {traces.nunique():,}  "
              f"activities {d[ACT].nunique()}")
    d = ao_log()
    traces = d.groupby(CASE)[ACT].agg(tuple)
    print(f"cases {len(traces):,}  events {len(d):,}  variants {traces.nunique():,}")
