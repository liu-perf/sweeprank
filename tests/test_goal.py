"""A sweep's stated goal, executed against the sweep's own results."""

from __future__ import annotations

import pytest

from sweeprank.goal import GoalError, apply_goal, check_goal
from sweeprank.model import Sweep, bundled


def _sweep(goal=None):
    raw = {
        "sweep_id": "t",
        "metrics": {
            "score": {"role": "objective", "higher_is_better": True},
            "util": {"role": "proxy", "higher_is_better": True},
            "power": {"role": "cost", "higher_is_better": False},
        },
        "configs": [
            {"id": "fast", "knobs": {}, "values": {"score": 100.0, "util": 60.0,
                                                   "power": 200.0}},
            {"id": "shiny", "knobs": {}, "values": {"score": 90.0, "util": 80.0,
                                                    "power": 250.0}},
        ],
    }
    if goal:
        raw["declared_goal"] = goal
    return Sweep.from_dict(raw)


def test_no_goal_is_reported_as_no_goal_not_as_a_pass():
    assert apply_goal(_sweep())["has_goal"] is False


def test_a_goal_that_rejects_the_objective_winner_is_an_error():
    s = _sweep({"metric": "util", "op": ">=", "value": 70})
    r = apply_goal(s)
    assert r["objective_winner"] == "fast"
    assert r["objective_winner_rejected"] is True
    assert r["goal_pick"] == "shiny"
    with pytest.raises(GoalError, match="fastest cell"):
        check_goal(s)


def test_the_substitution_is_priced_on_the_objective_and_on_every_cost():
    s = _sweep({"metric": "util", "op": ">=", "value": 70})
    price = apply_goal(s)["price"]
    assert price["score"]["change_pct"] == pytest.approx(-10.0)
    assert price["power"]["change_pct"] == pytest.approx(25.0)


def test_a_goal_that_keeps_the_winner_passes():
    s = _sweep({"metric": "util", "op": ">=", "value": 50})
    r = check_goal(s)
    assert r["objective_winner_rejected"] is False
    assert set(r["accepted"]) == {"fast", "shiny"}


def test_a_goal_written_on_a_proxy_says_so():
    r = apply_goal(_sweep({"metric": "util", "op": ">=", "value": 70}))
    assert r["goal_metric_role"] == "proxy"
    assert "not on the objective" in r["note"]


def test_a_goal_written_on_the_objective_carries_no_such_note():
    r = apply_goal(_sweep({"metric": "score", "op": ">=", "value": 50}))
    assert "note" not in r


def test_the_real_goal_rejects_the_real_winner():
    """The finding, as an assertion.

    The sweep's own header set out to get SM utilisation above 70%. Applied to
    the sweep's own table, that rule throws away the fastest configuration it
    found and replaces it with one that is slower and hotter.
    """
    s = bundled("det_tune")
    r = apply_goal(s)
    assert r["goal"] == "sm_util_mean_pct >= 70"
    assert r["goal_metric_role"] == "proxy"
    assert r["objective_winner"] == "det_b24nw16"
    assert r["objective_winner_value"] == 86.54
    assert r["objective_winner_goal_value"] == 68.7
    assert r["objective_winner_rejected"] is True
    assert r["goal_pick"] == "det_b24nw16pin"
    price = r["price"]
    assert -1.0 < price["throughput_img_s"]["change_pct"] < 0.0
    assert 7.0 < price["power_mean_w"]["change_pct"] < 8.0
    with pytest.raises(GoalError):
        check_goal(s)


def test_the_goal_quotes_its_own_source():
    """A goal transcribed with no provenance cannot be checked against anything."""
    s = bundled("det_tune")
    assert s.goal.quoted_from
    assert "70" in s.goal.quoted_from


def test_the_failed_cell_is_not_ranked_by_the_goal():
    s = bundled("det_tune")
    r = apply_goal(s)
    assert "det_b32" not in r["accepted"]
    assert "det_b32" not in r["rejected"]
