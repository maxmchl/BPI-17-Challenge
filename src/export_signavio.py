"""Section 2.4 - export the modelling log for SAP Signavio Process Intelligence.

Log: the full evaluation log of the final model (results/final_choice.json, It6_c75 on log B1c:
A_ + O_ complete, W_ start, merged O_Sent, without O_Created / O_Accepted / O_Refused; the
31,411 cases with an outcome).

Outputs (data/, not versioned)
  signavio_events.csv  case_id, activity, timestamp, resource; one row per event, sorted by
                       case_id and timestamp. Timestamps in UTC, ISO 8601 with milliseconds
                       (2016-01-01T09:51:15.304): 163,762 events share the second of the previous
                       event of their case, none the millisecond, so whole seconds would make the
                       event order ambiguous in Signavio.
  signavio_cases.csv   case_id, ApplicationType, LoanGoal, RequestedAmount,
                       channel (online if A_Submitted occurs, else bank),
                       outcome (A_Pending / A_Denied / A_Cancelled), n_offers (O_Create Offer events)
  signavio_events.zip  only if the event file is larger than 50 MB
Run:  python src/export_signavio.py
"""
import zipfile

import pandas as pd

from load import ROOT, CASE, ACT, TS, RES as RESOURCE
from preprocess import final_log, CASES

DATA = ROOT / "data"
EVENTS = DATA / "signavio_events.csv"
CASES_OUT = DATA / "signavio_cases.csv"
ZIP_LIMIT = 50 * 1024 ** 2


def main():
    log, cfg = final_log(keep_attrs=True)
    log = log.sort_values([CASE, TS], kind="stable")  # already in this order (build_log); kept explicit
    events = pd.DataFrame({
        "case_id": log[CASE].astype(str),
        "activity": log[ACT].astype(str).str.strip(),
        "timestamp": log[TS].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3],
        "resource": log[RESOURCE].astype(str),
    })
    events.to_csv(EVENTS, index=False)

    g = log.groupby(CASE, sort=True)
    first = g.first()
    outcome = pd.read_parquet(CASES)["outcome"]
    cases = pd.DataFrame({
        "case_id": first.index.astype(str),
        "ApplicationType": first["case:ApplicationType"].astype(str).to_numpy(),
        "LoanGoal": first["case:LoanGoal"].astype(str).to_numpy(),
        "RequestedAmount": first["case:RequestedAmount"].astype(float).to_numpy(),
        "channel": g[ACT].agg(lambda a: "online" if (a == "A_Submitted").any() else "bank").to_numpy(),
        "outcome": outcome.reindex(first.index).to_numpy(),
        "n_offers": g[ACT].agg(lambda a: int((a == "O_Create Offer").sum())).to_numpy(),
    })
    assert cases["outcome"].isin(["A_Pending", "A_Denied", "A_Cancelled"]).all()
    assert set(cases["case_id"]) == set(events["case_id"])
    cases.to_csv(CASES_OUT, index=False)

    print(f"log {cfg['log']} (model {cfg['id']}): {len(cases):,} cases, {len(events):,} events")
    for f, n in [(EVENTS, len(events)), (CASES_OUT, len(cases))]:
        print(f"{f.relative_to(ROOT)}: {n:,} rows, {f.stat().st_size / 1024 ** 2:.1f} MB")
    if EVENTS.stat().st_size > ZIP_LIMIT:
        z = EVENTS.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(EVENTS, EVENTS.name)
        print(f"{z.relative_to(ROOT)}: {z.stat().st_size / 1024 ** 2:.1f} MB (event file > 50 MB)")


if __name__ == "__main__":
    main()
