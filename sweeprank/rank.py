"""Does the proxy order this sweep the way the objective does?

Not "is it correlated". Correlation is the wrong question for a sweep, because
a sweep is not used to predict a value -- it is used to **pick a row**. What
matters is whether every pairwise comparison comes out the same way under the
proxy as under the objective. So the unit of analysis here is the pair, and the
verdict is a pair count.

The second function is the one that keeps the answer honest. A proxy that fails
on this sweep has not been shown useless; it has been shown blind *at this
scale*. :func:`resolution` puts a number on the scale, by asking how big an
objective gap has to be before the proxy stops getting the order wrong. Without
it the conclusion collapses into "never look at utilisation", which the control
sweep in ``data/governor_control.json`` directly contradicts.
"""

from __future__ import annotations

from itertools import combinations

from .model import Sweep


def _rel_gap(a: float, b: float) -> float:
    """Relative size of an objective gap, as a fraction of the smaller value."""
    lo = min(abs(a), abs(b))
    if lo == 0:
        return float("inf")
    return abs(a - b) / lo


def pairs(sweep: Sweep, proxy: str) -> list:
    """Every pair of surviving cells, classified.

    ``verdict`` is one of ``concordant`` / ``discordant`` / ``proxy_tied`` /
    ``objective_tied``. The two tie classes are kept apart because they mean
    opposite things: a proxy tie is the proxy failing to resolve a real
    difference, and an objective tie is there being nothing to resolve.
    """
    obj = sweep.objective
    prx = sweep.metrics[proxy]
    out = []
    for a, b in combinations(sweep.cells, 2):
        # Cells at different scales are not two candidate answers to one
        # question, so they do not form a pair. See Sweep.stratum.
        if sweep.strata and sweep.stratum(a) != sweep.stratum(b):
            continue
        oa, ob = sweep.value(a, obj.name), sweep.value(b, obj.name)
        pa, pb = sweep.value(a, prx.name), sweep.value(b, prx.name)
        do, dp = obj.better(oa, ob), prx.better(pa, pb)
        if do == 0:
            verdict = "objective_tied"
        elif dp == 0:
            verdict = "proxy_tied"
        elif do == dp:
            verdict = "concordant"
        else:
            verdict = "discordant"
        out.append({
            "a": a.id, "b": b.id, "verdict": verdict,
            "objective": [oa, ob], "proxy": [pa, pb],
            "objective_rel_gap": _rel_gap(oa, ob),
            "proxy_rel_gap": _rel_gap(pa, pb),
        })
    return out


def concordance(sweep: Sweep, proxy: str) -> dict:
    """Pair counts, a tau-b, and the flat verdict :mod:`sweeprank.role` needs."""
    ps = pairs(sweep, proxy)
    counts = dict.fromkeys(
        ("concordant", "discordant", "proxy_tied", "objective_tied"), 0)
    for p in ps:
        counts[p["verdict"]] += 1
    nc, nd = counts["concordant"], counts["discordant"]
    n0 = len(sweep.cells)
    # Not n*(n-1)/2: with strata declared, only within-stratum pairs exist.
    n_pairs = len(ps)
    # tau-b: ties in either variable are excluded from both denominators.
    d1 = n_pairs - counts["objective_tied"]
    d2 = n_pairs - counts["proxy_tied"]
    tau = (nc - nd) / ((d1 * d2) ** 0.5) if d1 and d2 else None
    return {
        "sweep": sweep.sweep_id,
        "objective": sweep.objective.name,
        "proxy": proxy,
        "cells": n0,
        "pairs": n_pairs,
        "strata": list(sweep.strata),
        "counts": counts,
        "tau_b": tau,
        "concordant": nd == 0 and counts["proxy_tied"] == 0,
        "discordant_pairs": [p for p in ps if p["verdict"] == "discordant"],
        "repeats": sweep.repeats(),
    }


def resolution(sweep: Sweep, proxy: str) -> dict:
    """How big an objective gap has to be before the proxy stops mis-ordering.

    Returns ``blind_at_or_below`` (the largest objective gap on which the proxy
    got the order wrong) and ``sound_at_or_above`` (the smallest gap above that
    on which it never did). Between the two this sweep has no evidence either
    way, and that gap is reported rather than closed by interpolation.
    """
    ps = pairs(sweep, proxy)
    bad = [p["objective_rel_gap"] for p in ps if p["verdict"] == "discordant"]
    bad += [p["objective_rel_gap"] for p in ps if p["verdict"] == "proxy_tied"]
    if not bad:
        good = [p["objective_rel_gap"] for p in ps if p["verdict"] == "concordant"]
        return {
            "sweep": sweep.sweep_id, "proxy": proxy,
            "blind_at_or_below": None,
            "sound_at_or_above": min(good) if good else None,
            "unresolved_between": None,
            "note": "no pair was mis-ordered in this sweep; the smallest gap it actually "
                    "tested is the smallest gap it can vouch for",
        }
    blind = max(bad)
    above = [p["objective_rel_gap"] for p in ps
             if p["verdict"] == "concordant" and p["objective_rel_gap"] > blind]
    sound = min(above) if above else None
    return {
        "sweep": sweep.sweep_id, "proxy": proxy,
        "blind_at_or_below": blind,
        "sound_at_or_above": sound,
        "unresolved_between": [blind, sound] if sound is not None else None,
        "note": "the interval between the two is not evidence of anything; this sweep "
                "contains no pair in it",
    }


def cross_sweep_resolution(sweeps: list, proxy: str) -> dict:
    """Pool the bound over several sweeps that share a proxy.

    The point of pooling is that no single sweep brackets the proxy from both
    sides. The tuning sweep only contains small objective gaps and the control
    only contains one large one, so the bound is a joint product of the two and
    is reported as such -- with the sweep that supplied each side named, because
    a bound whose halves come from different experiments has to say so.
    """
    lows, highs = [], []
    for s in sweeps:
        if proxy not in s.metrics:
            continue
        r = resolution(s, proxy)
        if r["blind_at_or_below"] is not None:
            lows.append((r["blind_at_or_below"], s.sweep_id))
        if r["sound_at_or_above"] is not None:
            highs.append((r["sound_at_or_above"], s.sweep_id))
    if not lows:
        return {"proxy": proxy, "blind_at_or_below": None, "sound_at_or_above": None,
                "sweeps": [s.sweep_id for s in sweeps]}
    blind, blind_from = max(lows)
    above = [(v, sid) for v, sid in highs if v > blind]
    sound, sound_from = min(above) if above else (None, None)
    return {
        "proxy": proxy,
        "blind_at_or_below": blind, "blind_from": blind_from,
        "sound_at_or_above": sound, "sound_from": sound_from,
        "halves_from_different_sweeps": sound_from is not None and sound_from != blind_from,
        "sweeps": [s.sweep_id for s in sweeps],
    }


__all__ = ["concordance", "cross_sweep_resolution", "pairs", "resolution"]
