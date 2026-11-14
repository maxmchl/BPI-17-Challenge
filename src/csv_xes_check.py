"""Section 2.1 - check that the CSV version of the log equals the official XES (4TU).

The pipeline uses data/BPI_Challenge_2017.xes.gz (4TU, DOI 10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b),
cached as data/bpic17.parquet. Compared with data/"BPI Challenge 2017.csv": cases, events,
activities, lifecycle values, resources, timestamp range, and per event (joined on EventID)
activity, lifecycle, resource, timestamp and the attribute values. CSV timestamps carry no time
zone; they are compared as UTC. Also: is data/"BPI Challenge 2017.xes" byte-identical to the
decompressed .xes.gz?

Output: results/csv_xes_check.json
Run:  python src/csv_xes_check.py
"""
import gzip
import hashlib
import json

import numpy as np
import pandas as pd

from load import load_log, ROOT, CASE, ACT, TS, LC, RES as RESOURCE

DATA = ROOT / "data"
CSV = DATA / "BPI Challenge 2017.csv"
XES_PLAIN = DATA / "BPI Challenge 2017.xes"
XES_GZ = DATA / "BPI_Challenge_2017.xes.gz"


def md5(chunks) -> str:
    h = hashlib.md5()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def summary(df) -> dict:
    return {"cases": int(df[CASE].nunique()), "events": len(df), "activities": int(df[ACT].nunique()),
            "lifecycle_values": sorted(df[LC].dropna().unique().tolist()),
            "resources": int(df[RESOURCE].nunique()),
            "first_timestamp_utc": str(df[TS].min()), "last_timestamp_utc": str(df[TS].max())}


def main():
    xes = load_log()
    csv = pd.read_csv(CSV, dtype=str)
    csv[TS] = pd.to_datetime(csv[TS], format="ISO8601").dt.tz_localize("UTC")
    out = {"xes": summary(xes), "csv": summary(csv)}
    out["same_summary"] = out["xes"] == out["csv"]

    x = xes.set_index("EventID")
    c = csv.set_index("EventID")
    out["event_ids_unique"] = {"xes": bool(x.index.is_unique), "csv": bool(c.index.is_unique)}
    # join on (EventID, occurrence number): robust if an ID repeated (it does not: unique in both files)
    for d in (xes, csv):
        d["_k"] = d["EventID"] + "#" + d.groupby("EventID").cumcount().astype(str)
    x, c = xes.set_index("_k"), csv.set_index("_k")
    common = x.index.intersection(c.index)
    out["events_only_in_xes"], out["events_only_in_csv"] = int(len(x.index.difference(c.index))), int(
        len(c.index.difference(x.index)))
    diff = {}
    for col in [CASE, ACT, LC, RESOURCE, "Action", "EventOrigin", "case:LoanGoal", "case:ApplicationType",
                "OfferID", "Accepted", "Selected"]:
        a, b = x.loc[common, col].astype("string").fillna("<NA>"), c.loc[common, col].astype("string").fillna("<NA>")
        diff[col] = int((a != b).sum())
    for col in ["case:RequestedAmount", "FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "CreditScore",
                "OfferedAmount"]:
        a, b = x.loc[common, col].astype(float).to_numpy(), c.loc[common, col].astype(float).to_numpy()
        diff[col] = int((~(np.isclose(a, b) | (np.isnan(a) & np.isnan(b)))).sum())
    dt = (x.loc[common, TS] - c.loc[common, TS]).abs()
    diff[TS] = int((dt > pd.Timedelta(0)).sum())
    out["differing_values_per_column"] = diff
    out["max_timestamp_difference"] = str(dt.max())
    out["identical_events"] = out["events_only_in_xes"] == out["events_only_in_csv"] == 0 and not any(diff.values())

    if XES_PLAIN.exists():
        with gzip.open(XES_GZ, "rb") as f:
            gz = md5(iter(lambda: f.read(1 << 20), b""))
        with open(XES_PLAIN, "rb") as f:
            plain = md5(iter(lambda: f.read(1 << 20), b""))
        out["xes_plain_equals_xes_gz"] = gz == plain
    out["pipeline_source"] = ("XES from 4TU (data/BPI_Challenge_2017.xes.gz), DOI "
                              "10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b")
    (ROOT / "results" / "csv_xes_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
