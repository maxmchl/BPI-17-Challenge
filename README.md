# BPI-17-Challenge — Process Mining, Conformance Checking and Decision Mining on BPIC-17

Code for Assignment I of *Business Process Prediction, Simulation, and Optimization* (TUM, 2026).

## Data

The BPI Challenge 2017 event log is not included in this repository. Download
`BPI Challenge 2017.xes.gz` from 4TU.ResearchData
(<https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b>) and save it as

```
data/BPI_Challenge_2017.xes.gz
```

## Setup

Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
python src/load.py           # parse XES once, cache as data/bpic17.parquet (~1 min)
python src/stats.py          # Section 3.2: results/basic_stats.{csv,tex}, results/extra_stats.json
python src/figures_stats.py  # Section 3.2 figures -> figures/*.pdf
```

## Structure

```
src/        analysis scripts (one per report section)
results/    computed tables and metrics (csv / tex / json)
figures/    figures used in the report (pdf)
data/       event log (not versioned)
```
