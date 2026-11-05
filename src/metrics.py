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
"""
from collections import Counter
from math import sqrt
import multiprocessing as mp
import time

import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
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
CORES = 4  # worker processes for alignments (PM4Py default: cpu_count - 2); keeps the laptop cool


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
             cyclomatic_timeout: float = 120.0, save=lambda r: None) -> tuple[dict, list]:
    """All metrics for one model; runtimes in seconds. `save(r)` is called after every step,
    so partial results survive if a later step hangs."""
    r = structural(net, im)
    save(r)

    tbr, r["t_fitness_tbr"] = timed(pm4py.fitness_token_based_replay, log, net, im, fm)
    r["fitness_tbr"] = tbr["log_fitness"]
    r["fit_traces_tbr_pct"] = tbr["percentage_of_fitting_traces"]
    save(r)

    # alignments are unpacked per trace, so percentages are weighted by case count
    aligned, r["t_alignments"] = timed(align, log, net, im, fm, align_timeout_trace)
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

    r["precision_align"], r["t_precision_align"] = timed(precision_alignments, log, net, im, fm)
    save(r)

    cyc, r["t_ext_cyclomatic"] = timed(with_timeout, extended_cyclomatic.apply, (net, im), cyclomatic_timeout)
    r["simp_ext_cyclomatic"] = float("nan") if cyc is None else cyc
    save(r)
    return r, aligned
