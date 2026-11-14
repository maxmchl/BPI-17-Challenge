"""Section 2.3 - guard texts for the final BPMN (Signavio remodel with three swimlanes).

For every decision point of the final net (results/decision_points.csv) one gateway entry:
where it sits (last visible activities before the place, first visible activities of each
branch), its type and kind, and the text annotation for the gateway:
  * mined decision points (D10, D6, D9): the leaf rules of the depth-3 decision tree WITHOUT
    resource/time features (results/decision_rules.json, feature set without_resource_time),
    with the class distribution on the test cases; leaves with < 2 % of the test rows omitted
  * all other decision points: branching probabilities (alignment-based, weighted by cases)
Also lists the activities per lane (results/activity_roles.csv).

Output: results/bpmn_annotations.json
Run after decision_mining.py:  python src/bpmn_annotations.py
"""
import json
import re

import pandas as pd
import pm4py

from load import ROOT
from decision_mining import DECISIONS

RES = ROOT / "results"
RULE_SET = "without_resource_time"
MIN_SUPPORT = 0.02
READING = {  # business reading of the mined rule (numbers are in the guards)
    "D10": "Limit raises and small requested amounts are entered by the bank; new credit with a larger "
           "amount mostly comes in online. Resource features would only learn 'created by User_1 = online'.",
    "D6": "W_Call after offers is one work item per offer round: the first offer always starts a call, an "
          "offer created while work items already run does not start a new one.",
    "D9": "No activity after the last O_Sent and no returned offer -> cancelled by the 30-day timer; a "
          "validated application with an incomplete-documents round mostly ends pending; denial is hard "
          "to predict from data known before the decision (CreditScore would leak the outcome).",
}


def leads_from(p) -> list[str]:
    """Last visible activities before place p (backwards through silent transitions; 'start' = source)."""
    out, seen, stack = set(), {p}, [p]
    while stack:
        q = stack.pop()
        if not q.in_arcs:
            out.add("start")
        for a in q.in_arcs:
            t = a.source
            if t.label is not None:
                out.add(t.label)
            elif t not in seen:
                seen.add(t)
                for b in t.in_arcs:
                    if b.source not in seen:
                        seen.add(b.source)
                        stack.append(b.source)
    return sorted(out)


def fmt_days(m) -> str:
    v = float(m.group(2))
    txt = (f"{v * 86400:.0f} s" if v < 1 / 1440 else f"{v * 1440:.0f} min" if v < 1 / 24
           else f"{v * 24:.1f} h" if v < 1 else f"{v:.1f} days")
    return f"{m.group(1)}{txt}"


def tidy(cond: list[str]) -> list[str]:
    """Keep the tightest bound per numeric feature, readable thresholds, no 'case:' prefixes."""
    bounds, other = {}, []
    for c in cond:
        m = re.match(r"(.+?) (<=|>) ([-\d.e+]+)( \(or missing\))?$", c)
        if not m:
            other.append(c)
            continue
        f, op, v, miss = m.group(1), m.group(2), float(m.group(3)), m.group(4) or ""
        key = (f, op)
        if key not in bounds or (op == "<=" and v < bounds[key][0]) or (op == ">" and v > bounds[key][0]):
            bounds[key] = (v, miss)
    def bound(f, op, v):  # counts (n_*) as integers: <= 1.5 -> <= 1, > 1.5 -> >= 2
        if f.startswith("n_") and v % 1 == 0.5:
            return f"{f} <= {int(v)}" if op == "<=" else f"{f} >= {int(v) + 1}"
        return f"{f} {op} {v:.4g}"
    out = other + [bound(f, op, v) + miss for (f, op), (v, miss) in bounds.items()]
    out = [re.sub(r"(days_\w+ (?:<=|>) )([-\d.e+]+)", fmt_days, c).replace("case:", "") for c in out]
    return out


def guards(leaves: list[dict]) -> list[str]:
    total = sum(l["test_rows"] for l in leaves)
    out = []
    for l in sorted(leaves, key=lambda l: -l["test_rows"]):
        if l["test_rows"] < MIN_SUPPORT * total:
            continue
        p = l["test_distribution"].get(l["then"], 0)
        out.append(f"IF {' AND '.join(tidy(l['if']))} THEN {l['then']} "
                   f"({p * 100:.1f} % of {l['test_rows']:,} test visits)")
    return out


def main():
    net, im, fm = pm4py.read_pnml(str(RES / "models" / "final.pnml"))
    places = {p.name: p for p in net.places}
    dpt = pd.read_csv(RES / "decision_points.csv")
    rules = json.loads((RES / "decision_rules.json").read_text())
    perf = pd.read_csv(RES / "decision_mining.csv")
    roles = pd.read_csv(RES / "activity_roles.csv")

    gateways = []
    for dp, g in dpt.groupby("decision_point", sort=False):
        f = g.iloc[0]
        branches = [{"transition": r.branch_label if r.branch_label != "tau" else f"silent ({r.branch_transition})",
                     "leads_to": r.branch_leads_to.split("; "), "loop_back": bool(r.branch_is_loop_back),
                     "probability": r.probability, "count": int(r["count"])} for _, r in g.iterrows()]
        entry = {"decision_point": dp, "place": f.place, "type": f.type, "kind": f.kind,
                 "rule_candidate": None if pd.isna(f.rule_candidate) else f.rule_candidate,
                 "after": leads_from(places[f.place]), "cases_reaching": int(f.cases_reaching),
                 "visits": int(f.visits), "branches": branches}
        probs = "; ".join(f"{b['transition'] if not b['transition'].startswith('silent') else 'skip -> ' + '/'.join(b['leads_to'][:3])}"
                          f" {b['probability'] * 100:.1f} %" for b in branches)
        if dp in DECISIONS:
            key = f"{dp} {RULE_SET}"
            row = perf[(perf["decision_point"] == dp) & (perf["feature_set"] == RULE_SET)].set_index("model")
            dt, base = row.loc["decision tree (depth 3, balanced)"], row.loc["majority baseline"]
            entry["guards"] = guards(rules[key]["decision tree"]["leaves"])
            entry["reading"] = READING[dp]
            entry["annotation"] = (f"{DECISIONS[dp][0]}: " + " | ".join(entry["guards"])
                                   + f" [decision tree depth 3, balanced accuracy {dt.balanced_accuracy:.2f} vs "
                                   f"baseline {base.balanced_accuracy:.2f}; branching: {probs}]")
            entry["source"] = f"results/decision_rules.json, {key}"
        else:
            entry["annotation"] = (f"{f.kind}: {probs}" + (" (no attribute rule mined; probabilities = simulation input)"
                                                          if f.kind == "business decision" else ""))
            entry["source"] = "results/decision_points.csv (alignment-based branching probabilities)"
        gateways.append(entry)

    lanes = {lane: sorted(d["activity"].str.strip()) for lane, d in roles[roles["in_model"]].groupby("lane")}
    out = {"model": "results/models/final.pnml / final.bpmn (It6_c75)",
           "how_to_use": "One XOR gateway per decision point (self-loops and loop decisions as loop-back gateways); "
                         "attach 'annotation' as a BPMN text annotation to the split gateway. Activities go into the "
                         "lanes below. If the Signavio remodel changes the structure, rerun decision_points.py and "
                         "decision_mining.py on the re-imported model.",
           "lanes": lanes, "gateways": gateways}
    (RES / "bpmn_annotations.json").write_text(json.dumps(out, indent=2))
    for g in gateways:
        print(f"{g['decision_point']:4s} after {g['after']}: {g['annotation']}")


if __name__ == "__main__":
    main()
