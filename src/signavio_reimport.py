"""Section 2.3 - re-import the Signavio remodel of the final BPMN and re-evaluate it.

Input: results/models/final_bpmn.bpmn (SAP Signavio Process Manager export: one pool, three lanes
Customer / Bank employees / System (User_1), 14 text annotations with the gateway guards).
PM4Py's BPMN importer reads text annotations and associations as nodes and flows, so a copy
without lanes, collaboration, text annotations and associations is imported
(data/final_signavio_stripped.bpmn); the original file is not changed. The diagram plane refers to
the collaboration and is dropped with it ("diagram elements": 1); the layout is not needed for the
conversion, only the process element (tasks, gateways, events, sequence flows).

Checks
  * labels: the visible transitions of the converted net equal those of results/models/final.pnml
    (It6_c75) and all are log activities, character for character (no trailing spaces etc.)
  * behaviour: language equivalence of the two nets, independent of the log. Both reachability
    graphs are explored, silent transitions are treated as epsilon (subset construction), and a
    breadth-first search over pairs of state sets looks for the shortest trace accepted by one
    net and not the other (only states that can still reach the final marking count).
Metrics: the same evaluate() as final_model.py on the full log B1c (31,411 cases), alignments
with multiprocessing (metrics.CORES).

Outputs
  results/models/final_signavio.pnml          Petri net converted from the Signavio BPMN
  results/final_signavio_metrics.json         metrics + checks + comparison with final_metrics.json
Run after final_model.py:  python src/signavio_reimport.py
"""
import json
import warnings
import xml.etree.ElementTree as ET
from collections import deque

import pm4py
from pm4py.objects.petri_net import semantics

from load import ROOT, ACT
from preprocess import final_log
from metrics import evaluate

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"
SRC = RES / "models" / "final_bpmn.bpmn"
STRIPPED = ROOT / "data" / "final_signavio_stripped.bpmn"
PNML = RES / "models" / "final_signavio.pnml"
OUT = RES / "final_signavio_metrics.json"
NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
      "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI"}
DROP = ("laneSet", "textAnnotation", "association", "collaboration")
COMPARE = ["places", "transitions", "arcs", "silent", "own_size", "own_cnc", "own_silent_share",
           "simp_ext_cardoso", "simp_ext_cyclomatic", "is_wf_net", "alignments_failed", "fit_traces_tbr_nim_pct",
           "fit_traces_align_pct", "fitness_tbr", "fitness_align", "precision_tbr", "precision_align",
           "generalization_tbr", "generalization_align", "t_alignments", "t_precision_align"]


def strip_bpmn(src=SRC, dst=STRIPPED) -> dict:
    """Copy of a Signavio file with only the control flow (tasks, gateways, events, sequence flows)."""
    for prefix, uri in NS.items():
        ET.register_namespace("" if prefix == "bpmn" else prefix, uri)
    tree = ET.parse(src)
    root = tree.getroot()
    removed, ids = {}, set()
    for parent in root.iter():
        for child in list(parent):
            tag = child.tag.split("}")[1]
            if tag in DROP:
                ids |= {e.get("id") for e in child.iter() if e.get("id")}
                removed[tag] = removed.get(tag, 0) + 1
                parent.remove(child)
    for parent in root.iter():  # diagram shapes and edges of the removed elements
        for child in list(parent):
            if child.get("bpmnElement") in ids:
                parent.remove(child)
                removed["diagram elements"] = removed.get("diagram elements", 0) + 1
    tree.write(dst, xml_declaration=True, encoding="UTF-8")
    return removed


def explore(net, im, fm):
    """Reachability graph restricted to markings from which the final marking is reachable."""
    states, edges, queue = {im: 0}, [], deque([im])
    while queue:
        m = queue.popleft()
        for t in semantics.enabled_transitions(net, m):
            m2 = semantics.execute(t, net, m)
            if m2 not in states:
                states[m2] = len(states)
                queue.append(m2)
            edges.append((states[m], t.label, states[m2]))
    final = states.get(fm)
    alive, changed = {final}, True
    while changed:
        changed = False
        for a, _, b in edges:
            if b in alive and a not in alive:
                alive.add(a)
                changed = True
    succ = {}
    for a, lab, b in edges:
        if a in alive and b in alive:
            succ.setdefault(a, []).append((lab, b))
    return {"states": len(states), "final": final, "succ": succ, "start": states[im]}


def closure(g, s):
    out, stack = set(s), list(s)
    while stack:
        a = stack.pop()
        for lab, b in g["succ"].get(a, []):
            if lab is None and b not in out:
                out.add(b)
                stack.append(b)
    return frozenset(out)


def step(g, s, label):
    return closure(g, {b for a in s for lab, b in g["succ"].get(a, []) if lab == label})


def labels_after(g, s):
    return {lab for a in s for lab, _ in g["succ"].get(a, []) if lab is not None}


def language_diff(g1, g2):
    """Shortest trace on which the two nets disagree (accepted by one only, or a label possible in one
    only), or None if the languages are equal."""
    start = (closure(g1, {g1["start"]}), closure(g2, {g2["start"]}))
    seen, queue = {start}, deque([(start, ())])
    while queue:
        (s1, s2), trace = queue.popleft()
        acc1, acc2 = g1["final"] in s1, g2["final"] in s2
        if acc1 != acc2:
            return {"trace": list(trace), "accepted_by_final_pnml": acc1, "accepted_by_signavio": acc2}
        l1, l2 = labels_after(g1, s1), labels_after(g2, s2)
        if l1 != l2:
            return {"prefix": list(trace), "next_only_in_final_pnml": sorted(l1 - l2),
                    "next_only_in_signavio": sorted(l2 - l1)}
        for lab in sorted(l1):
            nxt = (step(g1, s1, lab), step(g2, s2, lab))
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trace + (lab,)))
    return None


def main():
    removed = strip_bpmn()
    bpmn = pm4py.read_bpmn(str(STRIPPED))
    net, im, fm = pm4py.convert_to_petri_net(bpmn)
    pm4py.write_pnml(net, im, fm, str(PNML))
    fnet, fim, ffm = pm4py.read_pnml(str(RES / "models" / "final.pnml"))

    log, cfg = final_log()
    vis = sorted(t.label for t in net.transitions if t.label is not None)
    fvis = sorted(t.label for t in fnet.transitions if t.label is not None)
    acts = set(log[ACT].unique())
    checks = {"stripped": removed,
              "visible_transitions": len(vis),
              "visible_labels_equal_final_pnml": vis == fvis,
              "labels_only_in_signavio": sorted(set(vis) - set(fvis)),
              "labels_only_in_final_pnml": sorted(set(fvis) - set(vis)),
              "labels_not_in_log": sorted(set(vis) - acts),
              "duplicate_labels": sorted({l for l in vis if vis.count(l) > 1})}
    g1, g2 = explore(fnet, fim, ffm), explore(net, im, fm)
    diff = language_diff(g1, g2)
    checks["reachable_markings"] = {"final_pnml": g1["states"], "signavio": g2["states"]}
    checks["same_language"] = diff is None
    checks["first_difference"] = diff
    print(json.dumps(checks, indent=1), flush=True)

    meta = {"model": "Signavio remodel of It6_c75 (results/models/final_bpmn.bpmn)", "log": cfg["log"],
            "cases": cfg["cases"], "events": cfg["events"], "checks": checks}

    def save(r):
        OUT.write_text(json.dumps({**meta, **r}, indent=2, default=str))

    r, _, _ = evaluate(log, net, im, fm, save=save)
    final = json.loads((RES / "final_metrics.json").read_text())
    r["comparison_with_final_metrics"] = {k: {"final_pnml": final.get(k), "signavio": r.get(k)} for k in COMPARE}
    save(r)
    for k, v in r["comparison_with_final_metrics"].items():
        print(f"{k:24s} final {v['final_pnml']!s:>22}  signavio {v['signavio']!s:>22}")


if __name__ == "__main__":
    main()
