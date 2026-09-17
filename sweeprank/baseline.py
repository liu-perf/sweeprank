"""One column, one baseline.

A speedup column is a ratio, and a ratio has a denominator. When a table puts
two different denominators in one column, the column stops being a quantity and
the reader has no way to tell -- the cells are all bare floats with an "x"
after them.

The sweep this repository is built on did exactly that. Its speedup column
divides the four single-card rows by an untuned single-card run and the four
multi-card rows by a *tuned* single-card run. Both halves are individually
correct. Read down the column they are not comparable, and the efficiency
column derived from it prints **118.4% scaling efficiency on one GPU** without
anything going red.

So:

* :func:`check_baselines` refuses a derived column with more than one baseline.
* :func:`recompute` divides the objective values itself and compares against
  the numbers the report published, so a transcription error cannot hide behind
  a structural one.
* :func:`check_efficiency_domain` refuses a per-unit efficiency on rows where
  the unit count is 1, because dividing by one unit does not make a
  batch-size speedup into a scaling efficiency.
"""

from __future__ import annotations

from .model import Sweep


class BaselineError(Exception):
    """A derived column cannot be read as a single quantity."""


def _resolve(sweep: Sweep, key: str, others: dict) -> float:
    """Objective value of a baseline key.

    ``sweep_id:config_id`` reaches into another loaded sweep, which is how a
    cross-sweep baseline gets represented rather than silently flattened.
    """
    if ":" in key:
        sid, cid = key.split(":", 1)
        if sid not in others:
            raise BaselineError(
                f"baseline {key!r} names sweep {sid!r}, which was not supplied"
            )
        s = others[sid]
        return s.value(s.by_id(cid), s.objective.name)
    return sweep.value(sweep.by_id(key), sweep.objective.name)


def check_baselines(sweep: Sweep) -> list:
    """One finding per derived column that mixes baselines."""
    out = []
    for d in sweep.derived:
        if not d.baselines:
            continue
        if len(d.baselines) > 1:
            out.append({
                "column": d.name,
                "problem": "multiple_baselines",
                "baselines": {k: sorted(v) for k, v in d.baselines.items()},
                "message": (
                    f"column {d.name!r} divides its rows by {len(d.baselines)} different "
                    f"baselines ({', '.join(sorted(d.baselines))}); read down the column "
                    f"the cells are not comparable"
                ),
            })
    return out


def recompute(sweep: Sweep, others: dict | None = None, *, tol: float = 0.006) -> list:
    """Recompute each derived cell and compare with the published value."""
    others = dict(others or {})
    others.setdefault(sweep.sweep_id, sweep)
    rows = []
    for d in sweep.derived:
        if d.formula != "ratio_to_baseline":
            continue
        for base_key, members in d.baselines.items():
            base = _resolve(sweep, base_key, others)
            for cid in members:
                try:
                    cfg = sweep.by_id(cid)
                except Exception:
                    # A row label that has no configuration of its own: it is a
                    # transcribed alias for a cell in the baseline's sweep.
                    rows.append({"column": d.name, "row": cid, "baseline": base_key,
                                 "recomputed": None, "reported": d.reported.get(cid),
                                 "agrees": None,
                                 "note": "row has no configuration in this sweep"})
                    continue
                got = sweep.value(cfg, sweep.objective.name) / base
                rep = d.reported.get(cid)
                rows.append({
                    "column": d.name, "row": cid, "baseline": base_key,
                    "recomputed": got, "reported": rep,
                    "agrees": None if rep is None else abs(got - rep) <= tol,
                })
    return rows


def check_efficiency_domain(sweep: Sweep) -> list:
    """A per-unit efficiency printed on a one-unit row is not an efficiency."""
    out = []
    for d in sweep.derived:
        if d.formula != "speedup_over_units" or not d.units_knob:
            continue
        for cid, val in sorted(d.reported.items()):
            try:
                units = int(sweep.by_id(cid).knobs.get(d.units_knob, 0))
            except Exception:
                units = 0
            if units == 1 or (units == 0 and val > 100.0):
                out.append({
                    "column": d.name, "row": cid, "reported": val,
                    "units": units or None,
                    "problem": "efficiency_outside_domain",
                    "message": (
                        f"{d.name} = {val:g}% on row {cid!r}"
                        + (f" with {d.units_knob} = 1" if units == 1 else "")
                        + "; dividing a speedup by one unit does not make it a scaling "
                          "efficiency, and a scaling efficiency above 100% is the tell"
                    ),
                })
    return out


def check(sweep: Sweep, others: dict | None = None) -> dict:
    findings = check_baselines(sweep) + check_efficiency_domain(sweep)
    recomputed = recompute(sweep, others)
    mismatched = [r for r in recomputed if r["agrees"] is False]
    return {
        "sweep": sweep.sweep_id,
        "findings": findings,
        "recomputed": recomputed,
        "mismatched": mismatched,
        "ok": not findings and not mismatched,
    }


__all__ = [
    "BaselineError", "check", "check_baselines", "check_efficiency_domain", "recompute",
]
