"""Event abstraction and case filtering shared by discovery, modelling and decision mining.

Abstraction: keep only `lifecycle:transition == complete` events of the application (A_) and
offer (O_) sub-processes. W_ (workflow) events are dropped for control-flow discovery; they
carry resources and processing times and are used again for simulation-related analyses.
Effect on variants (Section 3.2): 80 % of cases need 9,629 variants in the raw log, 65 here.

Filter: cases without a terminal application state (A_Pending / A_Denied / A_Cancelled) are
removed. There are 98 of them; 97 were still active in the last 30 days of the log, i.e. they
are incomplete cases cut off by the log end.

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


if __name__ == "__main__":
    d = ao_log()
    traces = d.groupby(CASE)[ACT].agg(tuple)
    print(f"cases {len(traces):,}  events {len(d):,}  variants {traces.nunique():,}")
