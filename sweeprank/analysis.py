"""The findings this repository actually claims, computed from ``data/``.

Everything here reads the committed JSON and nothing here needs a GPU. Each
function returns the numbers *and* the boundary of what they support, because
every one of these sweeps has ``repeats: 1`` and a single run per cell cannot
carry an error bar however it is phrased.
"""

from __future__ import annotations

from . import baseline, exclusion, goal, rank
from .model import Sweep, bundled

PROXY = "sm_util_mean_pct"


def _sweeps() -> dict:
    return {n: bundled(n) for n in ("det_tune", "det_scale", "seg_scale", "governor_control")}


def proxy_verdict() -> dict:
    """Does SM utilisation order these sweeps the way throughput does?"""
    out = {}
    for name, s in _sweeps().items():
        if PROXY not in s.metrics:
            continue
        c = rank.concordance(s, PROXY)
        out[name] = {
            "cells": c["cells"],
            "pairs": c["pairs"],
            "counts": c["counts"],
            "tau_b": c["tau_b"],
            "concordant": c["concordant"],
        }
    return out


def the_reversal() -> dict:
    """The cells that differ only in dataloader knobs.

    Batch is fixed at 24 across all four, so the comparison is clean: the only
    things that move are worker count and pinned memory.

    The claim here is narrower than the one it started as, and the narrowing
    came from running it. "The proxy ranks the sweep exactly backwards" is
    false: over four cells it is not a reversal, and over the top three it is
    not one either. The slowest cell is last under both, which is the proxy
    getting something right.

    What is true, and is what the fields below report, is the part that decides
    the tuning session:

    * the proxy ranks the objective's **winner last** of the three cells a
      person would actually be choosing between, and
    * the proxy's own **first choice is not the objective's first choice**.

    That is enough to select the wrong configuration and not enough to call the
    metric inverted, so the metric is not called inverted anywhere in this
    repository.
    """
    s = bundled("det_tune")
    ids = [c.id for c in s.cells if c.knobs.get("batch") == 24]
    rows = []
    for cid in ids:
        c = s.by_id(cid)
        rows.append({
            "id": cid,
            "num_workers": c.knobs.get("num_workers"),
            "pin_memory": c.knobs.get("pin_memory"),
            "throughput_img_s": s.value(c, "throughput_img_s"),
            "sm_util_mean_pct": s.value(c, PROXY),
            "power_mean_w": s.value(c, "power_mean_w"),
            "data_time_frac_pct": s.value(c, "data_time_frac_pct"),
        })
    by_obj = [r["id"] for r in sorted(rows, key=lambda r: -r["throughput_img_s"])]
    by_prx = [r["id"] for r in sorted(rows, key=lambda r: -r["sm_util_mean_pct"])]
    top3 = set(by_obj[:3])
    t3_obj = [x for x in by_obj if x in top3]
    t3_prx = [x for x in by_prx if x in top3]
    return {
        "rows": rows,
        "order_by_objective": by_obj,
        "order_by_proxy": by_prx,
        "exactly_reversed": by_prx == by_obj[::-1],
        "top3_by_objective": t3_obj,
        "top3_by_proxy": t3_prx,
        "top3_exactly_reversed": t3_prx == t3_obj[::-1],
        "slowest_cell_last_under_both": by_obj[-1] == by_prx[-1],
        # The two statements that are true, and that are what does the damage.
        "objective_winner": t3_obj[0],
        "objective_winner_rank_under_proxy_of_top3": t3_prx.index(t3_obj[0]) + 1,
        "objective_winner_is_last_under_proxy_of_top3": t3_prx[-1] == t3_obj[0],
        "proxy_first_choice": t3_prx[0],
        "proxy_picks_a_different_winner": t3_prx[0] != t3_obj[0],
        "objective_spread_pct": (
            max(r["throughput_img_s"] for r in rows) / min(r["throughput_img_s"] for r in rows)
            - 1.0
        ) * 100.0,
        "proxy_spread_pct": (
            max(r["sm_util_mean_pct"] for r in rows) / min(r["sm_util_mean_pct"] for r in rows)
            - 1.0
        ) * 100.0,
    }


def what_pinning_bought() -> dict:
    """Pinned memory did its job and the step did not get faster.

    The mechanism is the interesting part: dataloader wait fell by an order of
    magnitude, which is precisely what pinning is for, and the step time rose
    slightly anyway. So the proxy's 4.9-point rise is real -- it is just not
    a rise in the thing the sweep wanted.
    """
    s = bundled("det_tune")
    a, b = s.by_id("det_b24nw16"), s.by_id("det_b24nw16pin")
    def d(m):
        va, vb = s.value(a, m), s.value(b, m)
        return {"without_pin": va, "with_pin": vb, "change_pct": (vb - va) / va * 100.0}
    return {
        "same_except": "pin_memory",
        "data_time_frac_pct": d("data_time_frac_pct"),
        "step_time_s": d("step_time_s"),
        "throughput_img_s": d("throughput_img_s"),
        "sm_util_mean_pct": d("sm_util_mean_pct"),
        "power_mean_w": d("power_mean_w"),
        "reading": "the wait the knob targets fell 87%; the step it was supposed to "
                   "shorten got 0.9% longer; utilisation and power both rose",
    }


def proxy_resolution() -> dict:
    """Where the proxy stops mis-ordering, bracketed across sweeps.

    Deliberately pooled. The tuning sweep only contains small objective gaps
    and the governor control contains one large one, so neither brackets the
    proxy alone, and the report says which sweep supplied which side.
    """
    s = _sweeps()
    pooled = rank.cross_sweep_resolution(list(s.values()), PROXY)
    ctrl = bundled("governor_control")
    c = rank.concordance(ctrl, PROXY)
    lo, hi = ctrl.by_id("governor_powersave"), ctrl.by_id("governor_performance")
    pooled["control"] = {
        "sweep": "governor_control",
        "concordant": c["concordant"],
        "objective_change_pct": (
            ctrl.value(hi, "throughput_img_s") / ctrl.value(lo, "throughput_img_s") - 1.0
        ) * 100.0,
        "proxy_change_pct": (
            ctrl.value(hi, PROXY) / ctrl.value(lo, PROXY) - 1.0
        ) * 100.0,
        "reading": "over a change this size the proxy tracks the objective in direction "
                   "and to within a few points in relative size; the proxy is not useless, "
                   "it has a resolution",
    }
    # The control's gap is far above the bracket, so it does not tighten it --
    # it is there to stop the bracket being read as a verdict on the metric.
    pooled["control_is_inside_bracket"] = (
        pooled["sound_at_or_above"] is not None
        and pooled["control"]["objective_change_pct"] / 100.0 <= pooled["sound_at_or_above"]
    )
    return pooled


def placement_effect() -> dict:
    """Cross-socket beat same-socket, on both workloads, on objective and cost.

    Direction replicated on two models that share only the node and the
    harness. Magnitude has no error bar and this says so: one run per cell.
    """
    out = {"cells": {}, "repeats": 1}
    for name, same, cross in (
        ("det_scale", "det_opt_4gpu", "det_opt_4gpuX"),
        ("seg_scale", "seg_4gpu", "seg_4gpuX"),
    ):
        s = bundled(name)
        a, b = s.by_id(same), s.by_id(cross)
        obj = s.objective.name
        out["cells"][name] = {
            "same_socket": {"id": same, obj: s.value(a, obj),
                            "power_mean_w": s.value(a, "power_mean_w")},
            "cross_socket": {"id": cross, obj: s.value(b, obj),
                             "power_mean_w": s.value(b, "power_mean_w")},
            "objective_gain_pct": (s.value(b, obj) / s.value(a, obj) - 1.0) * 100.0,
            "power_change_pct": (
                s.value(b, "power_mean_w") / s.value(a, "power_mean_w") - 1.0
            ) * 100.0,
        }
    gains = [v["objective_gain_pct"] for v in out["cells"].values()]
    out["direction_agrees"] = all(g > 0 for g in gains) or all(g < 0 for g in gains)
    out["gain_range_pct"] = [min(gains), max(gains)]
    out["cheaper_too"] = all(v["power_change_pct"] < 0 for v in out["cells"].values())
    out["plausible_mechanism"] = (
        "each socket's memory channels are half-populated, so spreading four ranks "
        "across two sockets doubles the number of populated channels behind them"
    )
    out["what_the_data_cannot_do"] = (
        "one run per cell, so the 3-5% magnitude has no error bar; and the mechanism "
        "above is consistent with the numbers, not shown by them -- no channel-population "
        "arm was run, so a per-socket cache or interrupt-affinity effect is not excluded"
    )
    return out


def eight_card_attribution() -> dict:
    """What the eight-card loss is, and why this data does not explain it.

    The one diagnostic that grew with card count in the detection sweep is
    almost absent in the segmentation sweep -- which loses nearly as much. So
    the diagnostic does not generalise, and the attribution stays open. Saying
    that is the output.
    """
    out = {"workloads": {}}
    for name, one, eight in (("det_scale", "det_opt_1gpu", "det_opt_8gpu"),
                             ("seg_scale", "seg_1gpu", "seg_8gpu")):
        s = bundled(name)
        a, b = s.by_id(one), s.by_id(eight)
        obj = s.objective.name
        speedup = s.value(b, obj) / s.value(a, obj)
        out["workloads"][name] = {
            "speedup": speedup,
            "efficiency_pct": speedup / 8.0 * 100.0,
            "loss_vs_ideal_pct": (1.0 - speedup / 8.0) * 100.0,
            "step_time_growth_pct": (
                s.value(b, "step_time_s") / s.value(a, "step_time_s") - 1.0
            ) * 100.0,
            "sm_zero_frac_pct_1gpu": s.value(a, "sm_zero_frac_pct"),
            "sm_zero_frac_pct_8gpu": s.value(b, "sm_zero_frac_pct"),
            "proxy_1gpu": s.value(a, PROXY),
            "proxy_8gpu": s.value(b, PROXY),
        }
    d, g = out["workloads"]["det_scale"], out["workloads"]["seg_scale"]
    out["idle_explains_at_most_pct_of_step_growth"] = (
        d["sm_zero_frac_pct_8gpu"] / d["step_time_growth_pct"] * 100.0
    )
    out["counterexample"] = {
        "what": "the segmentation sweep loses a comparable share of ideal with almost "
                "no measured idle",
        "det_loss_pct": d["loss_vs_ideal_pct"],
        "det_idle_pct": d["sm_zero_frac_pct_8gpu"],
        "seg_loss_pct": g["loss_vs_ideal_pct"],
        "seg_idle_pct": g["sm_zero_frac_pct_8gpu"],
    }
    out["verdict"] = "unattributed"
    out["why"] = (
        "idle fraction is the only diagnostic in this data that grows with card count, "
        "it can account for at most a fifth of the detection step-time growth, and it is "
        "absent in the segmentation sweep which loses about as much; nothing here "
        "identifies the mechanism and no sentence in this repository claims one"
    )
    return out


def straggler_bound() -> dict:
    """The flagged card cannot be the eight-card story, and it reads low anyway.

    Two separate results. First, the exclusion's own justification points the
    wrong way: the card said to read high reads below the mean of the others.
    Second, even taken at face value the measured 2.4% compute gap bounds a
    synchronous-DDP straggler effect far below the observed loss.
    """
    s = bundled("det_scale")
    cfg = s.by_id("det_opt_8gpu")
    rows = exclusion.audit_config(s, cfg)
    row = rows[0]
    e = cfg.exclusions[0]
    gap = float(e.also_measured["gap_pct"])
    eff = eight_card_attribution()["workloads"]["det_scale"]
    return {
        "direction": {
            "unit": row["unit"],
            "claimed": row["claimed_direction"],
            "actual": row["actual_direction"],
            "value": row["value"],
            "mean_without_unit": row["mean_without_unit"],
            "aggregate_shift": row["aggregate_shift"],
            "contradicted": row["contradicted"],
            "deviation_rank": row["deviation_rank"],
            "units_total": row["units_total"],
            "is_the_outlier": row["is_the_outlier"],
        },
        "size": {
            "compute_gap_pct": gap,
            "loss_vs_ideal_pct": eff["loss_vs_ideal_pct"],
            "share_of_loss_explained_pct": gap / eff["loss_vs_ideal_pct"] * 100.0,
            "reading": "synchronous data-parallel steps run at the pace of the slowest "
                       "rank, so a 2.4% slower card bounds its own contribution at about "
                       "2.4% -- a tenth of the loss actually observed",
        },
        "uncorrected": exclusion.uncorrected_metrics(s, cfg),
        "reading": "the contaminant was flagged in the right place, the correction was "
                   "applied to the one column where it changed a third of a point, the "
                   "headline throughput was left alone, and once bounded the contaminant "
                   "was too small to have mattered to either",
    }


def goals() -> dict:
    out = {}
    for name, s in _sweeps().items():
        r = goal.apply_goal(s)
        if r.get("has_goal"):
            out[name] = r
    return out


def derived_columns() -> dict:
    s = _sweeps()
    return baseline.check(s["det_scale"], others=s)


def all_findings() -> dict:
    return {
        "proxy_verdict": proxy_verdict(),
        "the_reversal": the_reversal(),
        "what_pinning_bought": what_pinning_bought(),
        "proxy_resolution": proxy_resolution(),
        "placement_effect": placement_effect(),
        "eight_card_attribution": eight_card_attribution(),
        "straggler_bound": straggler_bound(),
        "goals": goals(),
        "derived_columns": derived_columns(),
    }


__all__ = [
    "Sweep", "all_findings", "derived_columns", "eight_card_attribution", "goals",
    "placement_effect", "proxy_resolution", "proxy_verdict", "straggler_bound",
    "the_reversal", "what_pinning_bought",
]
