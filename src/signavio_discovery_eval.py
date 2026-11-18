"""Section 2.4 - evaluate the BPMN that SAP Signavio generated from the top 92 variants
(Process Intelligence, Variant Explorer -> Generate BPMN) exactly like the final model.

Input: results/models/signavio_generated.bpmn (versioned copy of data/signavio_bpmn.bpmn).
Steps
  1. import: lanes/pools, text annotations and associations stripped in a copy
     (data/signavio_generated_stripped.bpmn, signavio_reimport.strip_bpmn), PM4Py BPMN -> Petri net;
     label check against the log activities. PM4Py's direct conversion is unsound (OR joins become
     AND joins, loop markers are ignored; kept in the JSON under "untranslated").
  1b. faithful translation in a second copy (data/signavio_generated_translated.bpmn); the
     original file is not changed and the discovered behaviour is kept, only BPMN constructs PM4Py
     does not support are rewritten with XOR/AND gateways (also in the JSON, field "translation"):
       T0  <incoming>/<outgoing> children and the diagram (layout) removed; PM4Py takes the flow
           endpoints from sourceRef/targetRef, the layout is not needed for the conversion
       T1  OR1 (inclusive join, inputs XOR2 = direct online path and AND1 = bank / leads path) ->
           exclusive join. XOR1 (online vs bank) and XOR2 (direct vs W_Handle leads) route every
           case into exactly one of the two inputs, so the OR join only ever waits for one token.
       T2  OR2 (inclusive join, inputs A_Concept and W_Complete application) -> parallel join.
           On the bank / leads path AND1 starts both inputs; on the direct online path only
           A_Concept runs, so the OR join must not wait for W_Complete application. The direct
           flow XOR2 -> OR1 gets a parallel split that also sends a silent token to a new
           exclusive join in front of OR2's W_Complete application input (the "silent skip").
       T3  loop markers (standardLoopCharacteristics, testBefore=false = do-while) on
           O_Create Offer, O_Sent, O_Cancelled -> XOR join before the task, XOR split after it,
           back flow from the split to the join (the task runs once, then may repeat)
  2. soundness on the full reachability graph (classical definition, van der Aalst): option to
     complete, proper completion, no dead transitions, bounded within MAX_STATES markings;
     Woflan as a cross-check with a timeout. If the net is unsound, the reasons are reported
     (deadlocks with the shortest trace into them, mapped back to the BPMN gateways; BPMN
     elements PM4Py does not translate) and no log metrics are computed.
  3. metrics on the full log B1c (31,411 cases): the same evaluate() as final_model.py
  4. structure on the BPMN graph: loop around the offer block (O_Create Offer reachable again
     after O_Sent), loop around validation (W_Validate application reachable again after
     A_Incomplete), gateways by type, task loop markers
  5. comparison table: It6_c75 (final_metrics.json), Signavio re-import of the final model
     (final_signavio_metrics.json), Signavio-generated model

Output: results/signavio_discovery_metrics.json, results/models/signavio_generated.pnml (translated net)
Run:  python src/signavio_discovery_eval.py
"""
import json
import shutil
import warnings
import xml.etree.ElementTree as ET
from collections import deque

import pandas as pd
import pm4py
from pm4py.objects.petri_net import semantics

from load import ROOT, ACT
from preprocess import final_log
from metrics import evaluate, structural, is_workflow_net, with_timeout
from signavio_reimport import strip_bpmn, NS

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"
RAW = ROOT / "data" / "signavio_bpmn.bpmn"
SRC = RES / "models" / "signavio_generated.bpmn"
STRIPPED = ROOT / "data" / "signavio_generated_stripped.bpmn"
TRANSLATED = ROOT / "data" / "signavio_generated_translated.bpmn"
OUT = RES / "signavio_discovery_metrics.json"
MAX_STATES = 200_000
WOFLAN_TIMEOUT = 300
B = "{%s}" % NS["bpmn"]
GATEWAYS = {"exclusiveGateway": "XOR", "parallelGateway": "AND", "inclusiveGateway": "OR",
            "eventBasedGateway": "event", "complexGateway": "complex"}
ROWS = ["places", "transitions", "silent", "arcs", "own_size", "own_cnc", "own_silent_share", "simp_arc_degree",
        "simp_ext_cardoso", "simp_ext_cyclomatic", "is_wf_net", "sound", "fit_traces_tbr_nim_pct", "fitness_tbr_nim",
        "fit_traces_align_pct", "fitness_align", "alignments_failed", "precision_tbr", "precision_align",
        "generalization_tbr", "generalization_align", "t_fitness_tbr_nim", "t_alignments", "t_precision_tbr",
        "t_precision_align", "t_generalization_tbr", "t_ext_cyclomatic"]


# ---------------------------------------------------------------- BPMN graph
def bpmn_graph(path):
    """Nodes (id -> (type, name)), successors and predecessors of the process in a BPMN file."""
    proc = ET.parse(path).getroot().find(B + "process")
    nodes, succ, pred = {}, {}, {}
    for e in proc:
        tag = e.tag.replace(B, "")
        if tag not in ("sequenceFlow", "extensionElements", "laneSet", "textAnnotation", "association"):
            nodes[e.get("id")] = (tag, e.get("name") or "", e.find(B + "standardLoopCharacteristics") is not None)
    flows = {}
    for f in proc.iter(B + "sequenceFlow"):
        succ.setdefault(f.get("sourceRef"), []).append(f.get("targetRef"))
        pred.setdefault(f.get("targetRef"), []).append(f.get("sourceRef"))
        flows[f.get("id")] = (f.get("sourceRef"), f.get("targetRef"))
    short, count = {}, {}
    for i, (tag, name, _) in nodes.items():  # readable names: tasks by label, gateways XOR1, AND1, OR1, ...
        if tag in GATEWAYS:
            count[tag] = count.get(tag, 0) + 1
            short[i] = f"{GATEWAYS[tag]}{count[tag]}"
        else:
            short[i] = name or tag
    for fid, (s, d) in flows.items():  # flow places of the Petri net are named after the flow id
        short[fid] = f"flow {short[s]} -> {short[d]}"
    return nodes, succ, pred, short


def reachable_again(nodes, succ, after: str, target: str) -> bool:
    """Is a task named `target` reachable from the task named `after` (along sequence flows,
    or through a loop marker on either task)?"""
    ids = {n: i for i, (_, n, _) in nodes.items() if n}
    if nodes[ids[target]][2] and target == after:
        return True
    seen, stack = set(), list(succ.get(ids[after], []))
    while stack:
        n = stack.pop()
        if n == ids[target]:
            return True
        if n not in seen:
            seen.add(n)
            stack.extend(succ.get(n, []))
    return False


def structure(path) -> dict:
    nodes, succ, pred, short = bpmn_graph(path)
    gw = {}
    for i, (tag, _, _) in nodes.items():
        if tag in GATEWAYS:
            kind = ("split" if len(succ.get(i, [])) > 1 else "join" if len(pred.get(i, [])) > 1 else "pass-through")
            gw.setdefault(GATEWAYS[tag], {}).setdefault(kind, 0)
            gw[GATEWAYS[tag]][kind] += 1
    return {"gateways": {k: {"total": sum(v.values()), **v} for k, v in gw.items()},
            "tasks": sum(t == "task" for t, _, _ in nodes.values()),
            "task_loop_markers": sorted(n for t, n, loop in nodes.values() if loop),
            "offer_loop (O_Create Offer reachable again after O_Sent)":
                reachable_again(nodes, succ, "O_Sent", "O_Create Offer"),
            "validation_loop (W_Validate application reachable again after A_Incomplete)":
                reachable_again(nodes, succ, "A_Incomplete", "W_Validate application"),
            "inclusive_gateways": [{"gateway": short[i], "direction": "join" if len(pred.get(i, [])) > 1 else "split",
                                    "incoming": [short[p] for p in pred.get(i, [])],
                                    "outgoing": [short[s] for s in succ.get(i, [])]}
                                   for i, (t, _, _) in nodes.items() if t == "inclusiveGateway"]}


# ---------------------------------------------------------------- soundness
def soundness(net, im, fm, bpmn_short: dict) -> dict:
    """Classical soundness on the reachability graph (bounded by MAX_STATES markings)."""
    index, parent, queue, edges = {im: 0}, {im: None}, deque([im]), {}
    fired, deadlocks, improper, bounded = set(), [], [], True
    while queue:
        m = queue.popleft()
        en = semantics.enabled_transitions(net, m)
        if not en and m != fm:
            deadlocks.append(m)
        if m != fm and all(m[p] >= fm[p] for p in fm):
            improper.append(m)
        for t in en:
            fired.add(t)
            m2 = semantics.execute(t, net, m)
            edges.setdefault(m, []).append(m2)
            if m2 not in index:
                if len(index) >= MAX_STATES:
                    bounded = False
                    queue.clear()
                    break
                index[m2] = len(index)
                parent[m2] = (m, t)
                queue.append(m2)
    can_finish, changed = {fm} if fm in index else set(), True
    rev = {}
    for a, bs in edges.items():
        for b in bs:
            rev.setdefault(b, []).append(a)
    stack = list(can_finish)
    while stack:
        b = stack.pop()
        for a in rev.get(b, []):
            if a not in can_finish:
                can_finish.add(a)
                stack.append(a)
    stuck = [m for m in index if m not in can_finish]

    def trace_to(m):
        out = []
        while parent[m] is not None:
            m, t = parent[m]
            if t.label is not None:
                out.append(t.label)
        return out[::-1]

    def where(m):  # marked places -> BPMN element (place names: ent_<id>, exi_<id> or the flow id)
        out = []
        for p in m:
            nid = p.name.split("_", 1)[1] if p.name.startswith(("ent_", "exi_")) else p.name
            out.append(bpmn_short.get(nid, p.name) + (" (input)" if p.name.startswith("ent_") else
                                                                     " (output)" if p.name.startswith("exi_") else ""))
        return sorted(out)

    dead = sorted(t.label or f"silent {t.name[:12]}" for t in net.transitions if t not in fired)
    first = min(deadlocks, key=lambda m: len(trace_to(m))) if deadlocks else (stuck[0] if stuck else None)
    return {"reachable_markings": len(index), "bounded_within_limit": bounded,
            "option_to_complete": bounded and not stuck, "proper_completion": not improper,
            "no_dead_transitions": not dead, "sound": bounded and not stuck and not improper and not dead,
            "markings_that_cannot_complete": len(stuck), "deadlocks": len(deadlocks),
            "dead_visible_transitions": [d for d in dead if not d.startswith("silent")],
            "dead_silent_transitions": sum(d.startswith("silent") for d in dead),
            "first_deadlock": None if first is None else {"trace": trace_to(first), "marked": where(first)}}


def woflan(net, im, fm):
    from pm4py.algo.analysis.woflan import algorithm as wf
    return wf.apply(net, im, fm, parameters={wf.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                             wf.Parameters.PRINT_DIAGNOSTICS: False})


def diagnosis(r: dict, translated: bool = False) -> list[str]:
    """Why the converted net is unsound, in BPMN terms (OR joins / loop markers only before the translation)."""
    s, out = r["structure"], []
    for g in ([] if translated else s["inclusive_gateways"]):
        if g["direction"] == "join":
            out.append(f"{g['gateway']} is an inclusive (OR) join with inputs {g['incoming']}. PM4Py translates an "
                       f"OR join like an AND join (it waits for a token on every input) and adds skip transitions "
                       f"only between an OR split and an OR join; this model has no OR split, so {g['gateway']} "
                       f"blocks every case that does not deliver a token on all of its inputs.")
    if s["task_loop_markers"] and not translated:
        out.append(f"Loop markers (standardLoopCharacteristics) on {s['task_loop_markers']} are ignored by PM4Py's "
                   f"BPMN importer: each task fires at most once per visit in the net.")
    d = r["soundness"]
    if d["first_deadlock"]:
        out.append(f"Every case deadlocks: {d['reachable_markings']} reachable markings, none reaches the final "
                   f"marking; e.g. after {d['first_deadlock']['trace']} the tokens wait in "
                   f"{d['first_deadlock']['marked']}. Dead visible transitions: {len(d['dead_visible_transitions'])} "
                   f"of {r['visible_transitions']}.")
    return out


def translate(src, dst) -> list[str]:
    """Faithful rewrite of the OR joins and loop markers (T0-T3 in the module docstring)."""
    tree = ET.parse(src)
    root = tree.getroot()
    proc = root.find(B + "process")
    nodes, succ, pred, short = bpmn_graph(src)
    by_short = {v: k for k, v in short.items() if not v.startswith("flow ")}
    flows = {f.get("id"): f for f in proc.iter(B + "sequenceFlow")}
    el = {e.get("id"): e for e in proc}
    steps, n = [], [0]

    def new_id(kind):
        n[0] += 1
        return f"tr-{kind}-{n[0]}"

    def gateway(tag, direction):
        g = ET.SubElement(proc, B + tag, {"id": new_id(tag), "gatewayDirection": direction})
        return g.get("id")

    def flow(s, d):
        ET.SubElement(proc, B + "sequenceFlow", {"id": new_id("flow"), "sourceRef": s, "targetRef": d})

    def the_flow(s, d):
        f = [f for f in flows.values() if f.get("sourceRef") == s and f.get("targetRef") == d]
        assert len(f) == 1, (short[s], short[d])
        return f[0]

    # T0
    k = 0
    for e in proc:
        for c in list(e):
            if c.tag in (B + "incoming", B + "outgoing"):
                e.remove(c)
                k += 1
    for d in root.findall("{%s}BPMNDiagram" % NS["bpmndi"]):
        root.remove(d)
    steps.append(f"T0: removed {k} <incoming>/<outgoing> references and the diagram layout (PM4Py uses "
                 f"sourceRef/targetRef; layout not needed)")

    # T1
    or1, xor2, and1 = by_short["OR1"], by_short["XOR2"], by_short["AND1"]
    assert set(pred[or1]) == {xor2, and1} and nodes[or1][0] == "inclusiveGateway"
    el[or1].tag = B + "exclusiveGateway"
    steps.append("T1: OR1 (inclusive join; inputs XOR2 = direct online path, AND1 = bank / W_Handle leads path) -> "
                 "exclusive join: XOR1 and XOR2 route every case into exactly one of the two inputs")

    # T2
    or2, concept, wca = by_short["OR2"], by_short["A_Concept"], by_short["W_Complete application"]
    assert set(pred[or2]) == {concept, wca} and nodes[or2][0] == "inclusiveGateway"
    el[or2].tag = B + "parallelGateway"
    split = gateway("parallelGateway", "Diverging")
    join = gateway("exclusiveGateway", "Converging")
    the_flow(xor2, or1).set("targetRef", split)
    flow(split, or1)
    the_flow(wca, or2).set("targetRef", join)
    flow(join, or2)
    flow(split, join)
    steps.append("T2: OR2 (inclusive join; inputs A_Concept, W_Complete application) -> parallel join; the direct "
                 "online flow XOR2 -> OR1 gets a parallel split whose second branch is a silent skip into a new "
                 "exclusive join in front of OR2's W_Complete application input (on the bank / leads path AND1 "
                 "already starts both inputs)")

    # T3
    looped = []
    for i, (tag, name, loop) in nodes.items():
        if not loop:
            continue
        task = el[i]
        task.remove(task.find(B + "standardLoopCharacteristics"))
        assert len(pred[i]) == 1 and len(succ[i]) == 1, name
        lj, ls = gateway("exclusiveGateway", "Converging"), gateway("exclusiveGateway", "Diverging")
        the_flow(pred[i][0], i).set("targetRef", lj)
        the_flow(i, succ[i][0]).set("sourceRef", ls)
        flow(lj, i)
        flow(i, ls)
        flow(ls, lj)
        looped.append(name)
    steps.append(f"T3: loop markers (standardLoopCharacteristics, testBefore=false) on {sorted(looped)} -> XOR join "
                 f"before and XOR split after each task with a back flow (runs once, may repeat)")
    tree.write(dst, xml_declaration=True, encoding="UTF-8")
    return steps


def convert(path):
    return pm4py.convert_to_petri_net(pm4py.read_bpmn(str(path)))


def label_check(net, acts) -> dict:
    labels = [t.label for t in net.transitions if t.label is not None]
    return {"visible_transitions": len(labels), "labels_not_in_log": sorted(set(labels) - acts),
            "log_activities_not_in_model": sorted(acts - set(labels)),
            "duplicate_labels": sorted({l for l in labels if labels.count(l) > 1})}


def checked_soundness(net, im, fm, short) -> dict:
    s = soundness(net, im, fm, short)
    wf = with_timeout(woflan, (net, im, fm), WOFLAN_TIMEOUT)
    s["woflan"] = "timeout" if wf is None else bool(wf)
    return s


# ---------------------------------------------------------------- main
def table(r: dict) -> pd.DataFrame:
    final = json.loads((RES / "final_metrics.json").read_text())
    reimp = json.loads((RES / "final_signavio_metrics.json").read_text())
    final["sound"] = reimp["sound"] = True  # IM nets are sound by construction; re-import has the same language
    cols = {"It6_c75 final": final, "Signavio re-import of final": reimp, "Signavio-generated (top 92)": r}
    tab = pd.DataFrame({c: [d.get(k) for k in ROWS] for c, d in cols.items()}, index=ROWS)
    return tab.map(lambda v: "n/a" if v is None else round(v, 4) if isinstance(v, float) else v)


def main():
    if RAW.exists() and (not SRC.exists() or RAW.read_bytes() != SRC.read_bytes()):
        shutil.copyfile(RAW, SRC)
    removed = strip_bpmn(SRC, STRIPPED)
    log, cfg = final_log()
    acts = set(log[ACT].unique())

    def save(x):
        OUT.write_text(json.dumps(x, indent=2, default=str))

    # PM4Py's direct conversion (kept for the report: why a translation is needed)
    _, _, _, short = bpmn_graph(STRIPPED)
    net0, im0, fm0 = convert(STRIPPED)
    u = {"stripped": removed, **structural(net0, im0), **label_check(net0, acts), "structure": structure(SRC)}
    u["soundness"] = checked_soundness(net0, im0, fm0, short)
    u["sound"] = u["soundness"]["sound"]
    u["diagnosis"] = diagnosis(u) if not u["sound"] else []
    print("untranslated: sound =", u["sound"], *u["diagnosis"], sep="\n ", flush=True)

    # faithful translation
    steps = translate(STRIPPED, TRANSLATED)
    _, _, _, short = bpmn_graph(TRANSLATED)
    net, im, fm = convert(TRANSLATED)
    pm4py.write_pnml(net, im, fm, str(RES / "models" / "signavio_generated.pnml"))
    r = {"model": "SAP Signavio Process Intelligence, Variant Explorer -> Generate BPMN (top 92 variants), "
                  "translated faithfully for PM4Py",
         "file": "results/models/signavio_generated.bpmn", "log": cfg["log"], "cases": cfg["cases"],
         "translation": steps, **structural(net, im), **label_check(net, acts), "structure": u["structure"],
         "wf_net": is_workflow_net(net), "untranslated": u}
    r["soundness"] = checked_soundness(net, im, fm, short)
    r["sound"] = r["soundness"]["sound"]
    save(r)
    print(json.dumps({k: r[k] for k in ["translation", "places", "transitions", "silent", "visible_transitions",
                                        "labels_not_in_log", "wf_net", "soundness"]}, indent=1), flush=True)

    if not r["sound"]:
        r["diagnosis"] = diagnosis(r, translated=True)
        r["metrics"] = "not computed: the translated net is not sound (see soundness, diagnosis)"
        r["comparison"] = table(r).to_dict()
        save(r)
        print("\n".join(r["diagnosis"]))
        print(table(r).to_string())
        return

    m, _, _ = evaluate(log, net, im, fm, save=lambda x: save({**r, **x}))
    r.update(m)
    r["comparison"] = table(r).to_dict()
    save(r)
    print(table(r).to_string())


if __name__ == "__main__":
    main()
