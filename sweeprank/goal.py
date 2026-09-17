"""Apply a sweep's own stated goal to its own results.

Every sweep worth running starts with a sentence like "get utilisation above
70%". That sentence is a selection rule, and a selection rule can be executed.
Almost nobody executes it -- the goal is written in the header, the table is
written at the bottom, and the two are never put in the same room.

This module puts them in the same room. It applies the goal to the cells,
reports which cells it accepts and rejects, and -- the part that matters --
says whether the cell the **objective** ranks first is among the rejected. If
it is, following the stated goal would have thrown away the fastest
configuration in the sweep, and the sweep contains the proof.

It then prices the substitution: what the goal picks instead, and what that
choice costs on the objective and on every declared ``cost`` metric.
"""

from __future__ import annotations

from .model import Sweep
from .role import COST, OBJECTIVE


class GoalError(Exception):
    """The stated goal and the sweep's own results disagree about the winner."""


def apply_goal(sweep: Sweep) -> dict:
    if sweep.goal is None:
        return {"sweep": sweep.sweep_id, "has_goal": False}

    g = sweep.goal
    gm = sweep.metrics[g.metric]
    accepted, rejected = [], []
    for c in sweep.cells:
        (accepted if g.accepts(sweep.value(c, g.metric)) else rejected).append(c)

    obj_winner = sweep.winner()
    obj_name = sweep.objective.name

    # What the goal itself would pick: among the cells it accepts, the best on
    # the goal's own metric. That is what "get metric X above T" means when it
    # is used to choose.
    goal_pick = None
    if accepted:
        goal_pick = accepted[0]
        for c in accepted[1:]:
            if gm.better(sweep.value(c, g.metric), sweep.value(goal_pick, g.metric)) > 0:
                goal_pick = c

    out = {
        "sweep": sweep.sweep_id,
        "has_goal": True,
        "goal": f"{g.metric} {g.op} {g.value:g}",
        "goal_metric_role": gm.role,
        "quoted_from": g.quoted_from,
        "accepted": [c.id for c in accepted],
        "rejected": [c.id for c in rejected],
        "objective": obj_name,
        "objective_winner": obj_winner.id,
        "objective_winner_value": sweep.value(obj_winner, obj_name),
        "objective_winner_goal_value": sweep.value(obj_winner, g.metric),
        "objective_winner_rejected": obj_winner in rejected,
        "goal_pick": goal_pick.id if goal_pick else None,
        "price": None,
    }

    if goal_pick is not None and goal_pick.id != obj_winner.id:
        price = {}
        w_obj = sweep.value(obj_winner, obj_name)
        p_obj = sweep.value(goal_pick, obj_name)
        price[obj_name] = {
            "winner": w_obj, "goal_pick": p_obj,
            "change_pct": (p_obj - w_obj) / w_obj * 100.0,
        }
        for m in sweep.metrics.values():
            if m.role != COST:
                continue
            if m.name not in obj_winner.values or m.name not in goal_pick.values:
                continue
            w, p = sweep.value(obj_winner, m.name), sweep.value(goal_pick, m.name)
            price[m.name] = {
                "winner": w, "goal_pick": p,
                "change_pct": (p - w) / w * 100.0 if w else None,
            }
        out["price"] = price

    # The goal was written on a metric. Which role does that metric hold?
    if gm.role != OBJECTIVE:
        out["note"] = (
            f"the goal is written on a {gm.role}, not on the objective "
            f"({obj_name}); a {gm.role} may not select a winner unquestioned"
        )
    return out


def check_goal(sweep: Sweep) -> dict:
    """:func:`apply_goal`, but raising when the goal rejects the objective winner."""
    r = apply_goal(sweep)
    if r.get("objective_winner_rejected"):
        raise GoalError(
            f"{sweep.sweep_id}: the stated goal ({r['goal']}) rejects {r['objective_winner']}, "
            f"which is the fastest cell in the sweep "
            f"({r['objective']} = {r['objective_winner_value']:g}); it reads "
            f"{r['objective_winner_goal_value']:g} on the goal metric"
        )
    return r


__all__ = ["GoalError", "apply_goal", "check_goal"]
