"""Section 3.5 - decision point inventory and branching probabilities on the final Petri net.

Decision point = place with >= 2 outgoing arcs (in a Petri net every such place is a choice:
exactly one of its output transitions consumes the token). Classification (professor, Nov 6):
  XOR            no branch leads back: real alternatives a or b (incl. skips via a silent branch)
  self-loop      one branch leads back and the loop body contains exactly one visible activity,
                 i.e. that activity can repeat indefinitely
  loop decision  one branch leads back to a block of several activities (repeat a subset,
                 e.g. the offer block)
A branch t of place p "leads back" if one of t's output places q dominates p (every path from the
source place to p passes q; back edge in the sense of natural loops). The loop body is the
natural loop of that back edge: q plus every node that reaches t without passing q.

Branch frequencies come from the optimal alignments (de Leoni & van der Aalst 2013): every sync
or model move fires a transition; a firing of t counts as "branch t taken" at each decision
place in t's preset. Counts are weighted by the number of cases per variant. Silent branches are
described by the first visible activities they lead to.

Output: results/decision_points.csv (one row per branch), results/decision_points.json
Run after final_model.py:  python src/decision_points.py
"""
from collections import Counter
import json
import warnings

import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet

from load import ROOT
from preprocess import final_log
from metrics import align
from conformance_examples import variant_log

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"


def dominators(net, im):
    g = nx.DiGraph()
    g.add_edges_from((a.source, a.target) for a in net.arcs)
    (src,) = list(im)
    idom = nx.immediate_dominators(g, src)

    def dominates(q, n):
        while True:
            if n == q:
                return True
            if idom.get(n, n) == n:
                return False
            n = idom[n]
    return dominates


def reach(starts, forward, blocked=()):
    seen, stack = set(starts), list(starts)
    while stack:
        n = stack.pop()
        for a in (n.out_arcs if forward else n.in_arcs):
            m = a.target if forward else a.source
            if m not in seen and m not in blocked:
                seen.add(m)
                stack.append(m)
    return seen


def leads_to(t) -> list[str]:
    """First visible activities reachable from t through silent transitions only ('end' = sink)."""
    if t.label is not None:
        return [t.label]
    out, seen, stack = set(), {t}, [t]
    while stack:
        n = stack.pop()
        for a in n.out_arcs:
            p = a.target
            if not p.out_arcs:
                out.add("end")
            for b in p.out_arcs:
                u = b.target
                if u in seen:
                    continue
                seen.add(u)
                if u.label is None:
                    stack.append(u)
                else:
                    out.add(u.label)
    return sorted(out)


def classify(net, im):
    dominates = dominators(net, im)
    points = {}
    for p in net.places:
        outs = [a.target for a in p.out_arcs]
        if len(outs) < 2:
            continue
        back = {t: q for t in outs for q in (a.target for a in t.out_arcs) if dominates(q, p)}
        if not back:
            kind, body = "XOR", []
        else:
            nodes = set()
            for t, q in back.items():
                nodes |= reach([t], False, blocked={q}) | {q}
            body = sorted({n.label for n in nodes if isinstance(n, PetriNet.Transition) and n.label})
            kind = "self-loop" if len(body) == 1 else "loop decision"
        points[p] = {"type": kind, "loop_body": body, "back_branches": {t.name for t in back}}
    return points


def main():
    df, cfg = final_log()
    net, im, fm = pm4py.read_pnml(str(RES / "models" / "final.pnml"))
    points = classify(net, im)
    log, weights, _ = variant_log(df)
    al = align(log, net, im, fm)
    by_name = {t.name: t for t in net.transitions}

    taken, visits_cases = Counter(), Counter()
    for res, w in zip(al, weights):
        if res is None:
            continue
        seen = set()
        for (_, tname), _ in res["alignment"]:
            if tname == ">>":
                continue
            for a in by_name[tname].in_arcs:
                if a.source in points:
                    taken[(a.source, tname)] += int(w)
                    seen.add(a.source)
        for p in seen:
            visits_cases[p] += int(w)

    rows = []
    for i, (p, info) in enumerate(sorted(points.items(), key=lambda kv: -sum(
            taken[(kv[0], a.target.name)] for a in kv[0].out_arcs))):
        total = sum(taken[(p, a.target.name)] for a in p.out_arcs)
        for a in p.out_arcs:
            t = a.target
            n = taken[(p, t.name)]
            rows.append({"decision_point": f"D{i + 1}", "place": p.name, "type": info["type"],
                         "loop_body": "; ".join(info["loop_body"]),
                         "cases_reaching": visits_cases[p], "visits": total,
                         "branch_transition": t.name, "branch_label": t.label or "tau",
                         "branch_leads_to": "; ".join(leads_to(t)),
                         "branch_is_loop_back": t.name in info["back_branches"],
                         "count": n, "probability": round(n / total, 4) if total else None})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "decision_points.csv", index=False)
    summary = tab.groupby("decision_point", sort=False).agg(
        type=("type", "first"), cases=("cases_reaching", "first"), visits=("visits", "first"),
        branches=("branch_transition", "size")).reset_index()
    (RES / "decision_points.json").write_text(json.dumps(
        {"configuration": cfg["id"], "counts_by_type": summary["type"].value_counts().to_dict(),
         "points": summary.to_dict("records")}, indent=2))
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(summary.to_string(index=False))
    print(tab[["decision_point", "type", "branch_leads_to", "branch_is_loop_back", "count", "probability"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
