"""An exclusion has a direction, and the direction is checkable.

Dropping one unit out of an aggregate is a legitimate move. "This card's
reading is not trustworthy" is not a legitimate *reason* for it, because it
does not say which way the untrustworthy reading pushed the number, and a
correction with no direction cannot be wrong.

The version that can be wrong is: "this card reads too high, so the fleet mean
is inflated." That is a claim about a side, and the data has a side. So this
module does three things:

1. **Direction.** Compare the excluded value with the mean of the others. If
   the reason claims the unit read high and it read low, the exclusion moved
   the number the opposite way from its own justification.
2. **Size.** Report what the exclusion changed. A correction that moves an
   aggregate by a third of a point is worth knowing about mostly because of
   where it *wasn't* applied.
3. **Rank.** Report where the excluded unit sits among all units by distance
   from the mean. "We dropped the outlier" is checkable too, and an exclusion
   that drops the fourth-most-deviant of eight units was not dropping an
   outlier.
"""

from __future__ import annotations

from .model import Config, Sweep


class ExclusionDirectionError(Exception):
    """An exclusion's stated direction is contradicted by the value it dropped."""


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def audit_config(sweep: Sweep, cfg: Config) -> list:
    out = []
    for e in cfg.exclusions:
        units = cfg.units(e.metric)
        if e.unit not in units:
            out.append({
                "sweep": sweep.sweep_id, "config": cfg.id, "unit": e.unit,
                "metric": e.metric, "problem": "no_per_unit_data",
                "message": f"unit {e.unit!r} has no per-unit value for {e.metric!r}, "
                           f"so the exclusion cannot be checked at all",
            })
            continue
        value = float(units[e.unit])
        others = [float(v) for k, v in units.items() if k != e.unit]
        mean_all = _mean(units.values())
        mean_others = _mean(others) if others else None
        actual = "high" if value > mean_others else ("low" if value < mean_others else "equal")

        devs = sorted(
            ((abs(float(v) - mean_all), k) for k, v in units.items()), reverse=True
        )
        rank = [k for _, k in devs].index(e.unit) + 1

        contradicted = e.claimed_direction in ("high", "low") and actual != e.claimed_direction
        rec = {
            "sweep": sweep.sweep_id, "config": cfg.id, "unit": e.unit, "metric": e.metric,
            "reason": e.reason,
            "claimed_direction": e.claimed_direction,
            "actual_direction": actual,
            "contradicted": contradicted,
            "value": value,
            "mean_all_units": mean_all,
            "mean_without_unit": mean_others,
            "aggregate_shift": (mean_others - mean_all) if mean_others is not None else None,
            "deviation_rank": rank,
            "units_total": len(units),
            "is_the_outlier": rank == 1,
            "reported_aggregate_without": e.reported_aggregate_without,
        }
        if e.reported_aggregate_without is not None and mean_others is not None:
            rec["reported_matches_recomputed"] = (
                abs(e.reported_aggregate_without - mean_others) <= 0.06
            )
        if contradicted:
            rec["problem"] = "direction_contradicted"
            rec["message"] = (
                f"the exclusion of unit {e.unit} is justified by a mechanism that reads "
                f"{e.claimed_direction}, but {e.metric} = {value:g} on that unit is "
                f"{actual} relative to the other {len(others)} "
                f"(mean {mean_others:g}); dropping it moved the aggregate "
                f"{rec['aggregate_shift']:+.2f}, the opposite way from the stated reason"
            )
        elif rank != 1:
            rec["problem"] = "not_the_outlier"
            rec["message"] = (
                f"unit {e.unit} is {rank} of {len(units)} by distance from the mean on "
                f"{e.metric}; whatever it was, it was not this configuration's outlier"
            )
        out.append(rec)
    return out


def audit(sweep: Sweep) -> list:
    out = []
    for cfg in sweep.configs:
        out.extend(audit_config(sweep, cfg))
    return out


def uncorrected_metrics(sweep: Sweep, cfg: Config) -> list:
    """Metrics the exclusion did *not* touch, on the cell that had one.

    This is the useful half. An exclusion is applied to a column; the objective
    is in a different column; and if the reason for the exclusion were true of
    the unit, it would be true of the whole cell. Naming the untouched columns
    is how "we corrected for it" gets pinned down to "we corrected one of
    five".
    """
    touched = {e.metric for e in cfg.exclusions}
    if not touched:
        return []
    return sorted(set(cfg.values) - touched)


def check(sweep: Sweep) -> dict:
    rows = audit(sweep)
    bad = [r for r in rows if r.get("problem") == "direction_contradicted"]
    return {
        "sweep": sweep.sweep_id,
        "exclusions": rows,
        "contradicted": bad,
        "ok": not bad,
    }


def check_directions(sweep: Sweep) -> list:
    r = check(sweep)
    if r["contradicted"]:
        raise ExclusionDirectionError("; ".join(x["message"] for x in r["contradicted"]))
    return r["exclusions"]


__all__ = [
    "ExclusionDirectionError", "audit", "audit_config", "check", "check_directions",
    "uncorrected_metrics",
]
