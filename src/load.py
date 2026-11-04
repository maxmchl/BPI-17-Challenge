"""Load the BPIC-17 event log once from XES and cache it as Parquet.

Usage:  python src/load.py            (creates data/bpic17.parquet)
Other scripts call `load_log()` which reads the cache (seconds instead of minutes).
"""
from pathlib import Path
import time

import pandas as pd
import pm4py

ROOT = Path(__file__).resolve().parents[1]
XES = ROOT / "data" / "BPI_Challenge_2017.xes.gz"
CACHE = ROOT / "data" / "bpic17.parquet"

CASE = "case:concept:name"
ACT = "concept:name"
TS = "time:timestamp"
LC = "lifecycle:transition"
RES = "org:resource"


def build_cache() -> pd.DataFrame:
    t0 = time.time()
    df = pm4py.read_xes(str(XES))  # returns a pandas DataFrame in pm4py >= 2.x
    print(f"parsed XES in {time.time() - t0:.0f}s: {df.shape}")
    df[TS] = pd.to_datetime(df[TS], utc=True)
    # mixed-type object columns break parquet; cast them to string
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(CACHE, index=False)
    return df


def load_log() -> pd.DataFrame:
    if not CACHE.exists():
        return build_cache()
    df = pd.read_parquet(CACHE)
    df[TS] = pd.to_datetime(df[TS], utc=True)
    return df


if __name__ == "__main__":
    df = build_cache()
    print(df.dtypes)
    print(df.head())
