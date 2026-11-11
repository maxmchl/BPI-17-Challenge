"""Quality metrics for Petri nets: PM4Py wrappers with runtimes plus own implementations.

Own implementations
-------------------
Simplicity (structural, computed on the Petri net N = (P, T, F)):
  * size          |P| + |T| + |F|                                   (Mendling 2008; smaller = simpler)
  * CNC           |F| / (|P| + |T|), coefficient of network connectivity (Polancic & Cegnar 2017)
  * density       |F| / (2 |P| |T|), arcs relative to the maximum possible in a bipartite net
  * a_S           (|{l(t) | t in T visible}| + 2) / |P u T|, simple structural appropriateness
                  (Rozinat & van der Aalst 2008, lecture slide 29); higher = simpler. Silent
                  transitions count in the denominator but contribute no label.
  * silent share  |{t in T | t silent}| / |T|; high values indicate flower-like models whose
                  silent skips/loops let almost any order replay.
Generalization from alignments: same formula as PM4Py's token-based generalization
  (Buijs et al. 2014): 1 - mean_t( 1/sqrt(#executions of t) ), but the execution counts come
  from the model moves of the optimal alignments instead of token replay.

Token replay is run twice: with PM4Py's default (`consider_activities_not_in_model_in_fitness
= False`, events whose activity is not in the model are skipped silently) and with the
parameter set to True (`*_tbr_nim`, own implementation because of a PM4Py bug, see
tbr_fitness). The parameter only changes whether such a trace counts as
fitting; token counts, and therefore log fitness, ETC precision (own replay parameters) and
generalization (transition firings), are unaffected.
"""
from collections import Counter
from math import sqrt
import multiprocessing as mp
import time

import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
from pm4py.algo.conformance.tokenreplay.variants import token_replay as tr
from pm4py.algo.evaluation.replay_fitness.variants import token_replay as tr_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_algo
from pm4py.algo.evaluation.replay_fitness.variants import alignment_based as align_fitness
from pm4py.algo.evaluation.simplicity.variants import arc_degree, extended_cardoso, extended_cyclomatic


# ---------------------------------------------------------------- own simplicity metrics
def size(net) -> int:
    return len(net.places) + len(net.transitions) + len(net.arcs)


def cnc(net) -> float:
    return len(net.arcs) / (len(net.places) + len(net.transitions))


def density(net) -> float:
    return len(net.arcs) / (2 * len(net.places) * len(net.transitions))


def structural_appropriateness(net) -> float:
    labels = {t.label for t in net.transitions if t.label is not None}
    return (len(labels) + 2) / (len(net.places) + len(net.transitions))


# ---------------------------------------------------------------- generalization (alignments)
def generalization_from_alignments(net, aligned_traces) -> float:
    # moves are ((log_name, model_name), (log_label, model_label)), see align();
    # every move with a model part fires that transition (sync moves and model moves, incl. tau)
    occ = Counter()
    for res in aligned_traces:
        for (_, model_name), _ in res["alignment"]:
            if model_name != ">>":
                occ[model_name] += 1
    name_of = {t.name for t in net.transitions}
    inv = sum(1 / sqrt(occ[n]) if occ[n] > 0 else 1.0 for n in name_of)
    return 1 - inv / len(net.transitions)


# ---------------------------------------------------------------- alignments
CORES = 6  # worker processes for alignments (= PM4Py default cpu_count - 2 on this 8-core machine)


def precision_alignments(log, net, im, fm) -> float:
    """Alignment-based ETC precision (same as pm4py.precision_alignments, but with CORES workers)."""
    params = {"multiprocessing": True, "cores": CORES}
    return precision_algo.apply(log, net, im, fm, variant=precision_algo.Variants.ALIGN_ETCONFORMANCE,
                                parameters=params)


def align(log, net, im, fm, timeout_trace: float = 10.0):
    """Optimal alignments for all traces (PM4Py aligns each variant once).

    Moves are returned as ((log_name, model_name), (log_label, model_label)) so that silent
    transitions can be told apart (needed for generalization and decision mining).
    """
    params = {
        align_algo.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True,
        align_algo.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: timeout_trace,
        align_algo.Parameters.CORES: CORES,
    }
    return align_algo.apply_multiprocessing(log, net, im, fm, parameters=params)


def tbr_fitness(log, net, im, fm, not_in_model: bool = False) -> tuple[dict, list]:
    """Token-based fitness as in pm4py.fitness_token_based_replay, with the not-in-model switch.

    PM4Py bug (2.7.23.11, token_replay.py l. 1388/941): with
    CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS=True the dict of unknown activities is shared by
    the whole log and only checked for being non-empty, so every trace replayed after the first
    one with an unknown activity is marked unfit (IMf 0.10: 3,143 cases with 0 missing and 0
    remaining tokens). We therefore replay with the default and apply the intended rule
    ourselves: a trace fits iff TBR fits it and all its activities are labels of the model.
    Token counts and log fitness are identical in both modes.
    """
    elog = pm4py.convert_to_event_log(log) if hasattr(log, "columns") else log
    params = {tr.Parameters.CONSIDER_REMAINING_IN_FITNESS: True,
              tr.Parameters.SHOW_PROGRESS_BAR: False}
    replayed = tr.apply(elog, net, im, fm, parameters=params)
    if not_in_model:
        labels = {t.label for t in net.transitions if t.label is not None}
        replayed = [dict(r, trace_is_fit=r["trace_is_fit"] and all(e["concept:name"] in labels for e in t))
                    for r, t in zip(replayed, elog)]
    return tr_fitness.evaluate(replayed), replayed


def tbr_both(log, net, im, fm) -> dict:
    r = {}
    (f, _), r["t_fitness_tbr"] = timed(tbr_fitness, log, net, im, fm, False)
    r["fitness_tbr"], r["fit_traces_tbr_pct"] = f["log_fitness"], f["percentage_of_fitting_traces"]
    (f, _), r["t_fitness_tbr_nim"] = timed(tbr_fitness, log, net, im, fm, True)
    r["fitness_tbr_nim"], r["fit_traces_tbr_nim_pct"] = f["log_fitness"], f["percentage_of_fitting_traces"]
    return r


def timed(f, *a, **kw):
    t0 = time.perf_counter()
    out = f(*a, **kw)
    return out, time.perf_counter() - t0


def with_timeout(f, args, seconds):
    """Run f(*args) in a worker process; return None if it does not finish in time.

    Needed for PM4Py's extended cyclomatic metric, which builds the reachability graph and
    does not terminate in reasonable time on unsound / unbounded nets (e.g. the HM net).
    """
    with mp.get_context("spawn").Pool(1) as pool:
        job = pool.apply_async(f, args)
        try:
            return job.get(timeout=seconds)
        except mp.TimeoutError:
            pool.terminate()
            return None


# ---------------------------------------------------------------- structural checks
def is_workflow_net(net) -> dict:
    """WF-net check (van der Aalst): one source place, one sink place, every node on a path
    from source to sink. Cheap alternative to Woflan, whose state space explodes here."""
    sources = [p for p in net.places if not p.in_arcs]
    sinks = [p for p in net.places if not p.out_arcs]

    def reach(starts, forward):
        seen, stack = set(starts), list(starts)
        while stack:
            n = stack.pop()
            arcs = n.out_arcs if forward else n.in_arcs
            for a in arcs:
                m = a.target if forward else a.source
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    nodes = set(net.places) | set(net.transitions)
    ok_ends = len(sources) == 1 and len(sinks) == 1
    on_path = (reach(sources, True) & reach(sinks, False)) if ok_ends else set()
    return {"sources": len(sources), "sinks": len(sinks),
            "nodes_not_on_path": len(nodes - on_path) if ok_ends else None,
            "is_wf_net": ok_ends and on_path == nodes}


def structural(net, im) -> dict:
    """Simplicity and size metrics (no log needed)."""
    r = {"places": len(net.places), "transitions": len(net.transitions), "arcs": len(net.arcs),
         "silent": sum(t.label is None for t in net.transitions)}
    r["own_silent_share"] = r["silent"] / r["transitions"]
    r["own_size"] = size(net)
    r["own_cnc"] = cnc(net)
    r["own_density"] = density(net)
    r["own_struct_approp"] = structural_appropriateness(net)
    r["simp_arc_degree"] = arc_degree.apply(net)
    r["simp_ext_cardoso"] = extended_cardoso.apply(net)
    r["is_wf_net"] = is_workflow_net(net)["is_wf_net"]
    return r


def evaluate(log, net, im, fm, align_timeout_trace: float = 10.0,
             cyclomatic_timeout: float = 120.0, save=lambda r: None,
             fast: bool = False) -> tuple[dict, list, list]:
    """All metrics for one model; runtimes in seconds. `save(r)` is called after every step,
    so partial results survive if a later step hangs.

    fast=True (iterations) skips alignment-based ETC precision, which aligns every unique
    prefix and matched token-based precision within 0.002 over the whole IMf sweep, and the
    extended cyclomatic metric (reachability graph).

    Returns (metrics, per-trace alignments, case id per alignment).
    """
    r = structural(net, im)
    save(r)

    r.update(tbr_both(log, net, im, fm))
    save(r)

    # alignments are unpacked per trace, so percentages are weighted by case count
    elog = pm4py.convert_to_event_log(log) if hasattr(log, "columns") else log
    case_ids = [t.attributes["concept:name"] for t in elog]
    aligned, r["t_alignments"] = timed(align, elog, net, im, fm, align_timeout_trace)
    ok = [a for a in aligned if a is not None]
    r["alignments_failed"] = len(aligned) - len(ok)
    af = align_fitness.evaluate(ok)
    r["fitness_align"] = af["log_fitness"]
    r["fit_traces_align_pct"] = af["percentage_of_fitting_traces"]
    r["generalization_align"] = generalization_from_alignments(net, ok)
    save(r)

    r["precision_tbr"], r["t_precision_tbr"] = timed(pm4py.precision_token_based_replay, log, net, im, fm)
    r["generalization_tbr"], r["t_generalization_tbr"] = timed(pm4py.generalization_tbr, log, net, im, fm)
    save(r)
    if fast:
        return r, aligned, case_ids

    r["precision_align"], r["t_precision_align"] = timed(precision_alignments, log, net, im, fm)
    save(r)

    cyc, r["t_ext_cyclomatic"] = timed(with_timeout, extended_cyclomatic.apply, (net, im), cyclomatic_timeout)
    r["simp_ext_cyclomatic"] = float("nan") if cyc is None else cyc
    save(r)
    return r, aligned, case_ids
