"""Event abstraction and case filtering shared by discovery, modelling and decision mining.

Abstraction: keep only `lifecycle:transition == complete` events of the application (A_) and
offer (O_) sub-processes. W_ (workflow) events are dropped for control-flow discovery; they
carry resources and processing times and are used again for simulation-related analyses.
Effect on variants (Section 3.2): 80 % of cases need 9,629 variants in the raw log, 65 here.

Filter: cases without a terminal application state (A_Pending / A_Denied / A_Cancelled) are
removed. There are 98 of them; 97 were still active in the last 30 days of the log, i.e. they
are incomplete cases cut off by the log end.

Log versions (iteration log, Section 3.3):
  A   A_ + O_ events with lifecycle complete (the abstraction above).
  B1  A + W_ events with lifecycle start: one event per work item that a bank employee actually
      started. `complete` is not usable for W_ (most work items end in ate_abort/withdraw, e.g.
      W_Call after offers completes in 341 cases), and the terminal event would add automatic
      withdraws (16,802 W_Handle leads), which are not bank work.
Options:
  merge_osent       map "O_Sent (mail and online)" and "O_Sent (online only)" to "O_Sent".
  collapse_repeats  keep only the first of directly repeated identical activities in a case.

Run `python src/stats.py` first (writes results/cases.parquet with the per-case outcome).
"""
import pandas as pd

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


def build_log(version: str = "A", merge_osent: bool = False, collapse_repeats: bool = False,
              df: pd.DataFrame | None = None, keep_attrs: bool = False) -> pd.DataFrame:
    """Event log for one iteration-log configuration (see module docstring)."""
    if df is None:
        df = load_log()
    keep = (df[LC] == "complete") & df[ACT].str[:2].isin(["A_", "O_"])
    if version == "B1":
        keep |= (df[LC] == "start") & df[ACT].str.startswith("W_")
    elif version != "A":
        raise ValueError(f"unknown log version {version!r}")
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


if __name__ == "__main__":
    for cfg in [("A", False, False), ("A", True, False), ("B1", False, False), ("B1", True, True)]:
        d = build_log(*cfg)
        traces = d.groupby(CASE)[ACT].agg(tuple)
        print(f"{cfg}: cases {len(traces):,}  events {len(d):,}  variants {traces.nunique():,}  "
              f"activities {d[ACT].nunique()}")
    d = ao_log()
    traces = d.groupby(CASE)[ACT].agg(tuple)
    print(f"cases {len(traces):,}  events {len(d):,}  variants {traces.nunique():,}")
