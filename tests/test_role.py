"""The device, on the things it exists to forbid."""

from __future__ import annotations

import pytest

from sweeprank.model import Sweep, SweepError
from sweeprank.role import (
    COST,
    DIAGNOSTIC,
    OBJECTIVE,
    PROXY,
    RoleError,
    UndeclaredRoleError,
    assert_may_select,
    check_role,
)

MINIMAL = {
    "sweep_id": "t",
    "metrics": {
        "score": {"role": "objective", "higher_is_better": True},
        "util": {"role": "proxy", "higher_is_better": True},
    },
    "configs": [
        {"id": "a", "knobs": {}, "values": {"score": 10.0, "util": 50.0}},
        {"id": "b", "knobs": {}, "values": {"score": 20.0, "util": 40.0}},
    ],
}


def test_a_metric_with_no_role_is_refused():
    with pytest.raises(UndeclaredRoleError):
        check_role("util", None)


def test_an_unknown_role_is_refused():
    with pytest.raises(RoleError):
        check_role("util", "vibes")


def test_the_objective_may_always_select():
    assert_may_select("score", OBJECTIVE)


def test_a_cost_may_never_select():
    with pytest.raises(RoleError, match="never select"):
        assert_may_select("power", COST)


def test_a_diagnostic_may_never_select():
    with pytest.raises(RoleError, match="never select"):
        assert_may_select("data_time", DIAGNOSTIC)


def test_a_proxy_may_not_select_without_the_concordance_check_having_run():
    """The important one.

    ``concordant`` is keyword-only and has no truthy default, so a caller who
    has not run :mod:`sweeprank.rank` cannot get past this by omission -- which
    is exactly how a proxy becomes the objective in a real report.
    """
    with pytest.raises(RoleError, match="has not"):
        assert_may_select("util", PROXY)


def test_a_discordant_proxy_may_not_select():
    with pytest.raises(RoleError, match="ranks this sweep differently"):
        assert_may_select("util", PROXY, concordant=False)


def test_a_concordant_proxy_may_select():
    assert_may_select("util", PROXY, concordant=True)


def test_winner_by_a_discordant_proxy_raises_through_the_public_api():
    """No back door: Sweep.winner routes a proxy through the same gate."""
    s = Sweep.from_dict(MINIMAL)
    assert s.winner().id == "b"  # the objective's winner
    with pytest.raises(RoleError):
        s.winner("util")  # util ranks a > b, throughput ranks b > a


def test_winner_by_a_concordant_proxy_is_allowed():
    raw = {**MINIMAL, "configs": [
        {"id": "a", "knobs": {}, "values": {"score": 10.0, "util": 40.0}},
        {"id": "b", "knobs": {}, "values": {"score": 20.0, "util": 50.0}},
    ]}
    s = Sweep.from_dict(raw)
    assert s.winner("util").id == "b"


def test_every_bundled_metric_declares_a_role():
    from sweeprank.model import bundled, bundled_names

    for name in bundled_names():
        s = bundled(name)
        assert s.metrics, name
        for m in s.metrics.values():
            assert m.role, (name, m.name)


def test_a_sweep_with_two_objectives_is_refused():
    raw = {
        "sweep_id": "t",
        "metrics": {
            "a": {"role": "objective", "higher_is_better": True},
            "b": {"role": "objective", "higher_is_better": True},
        },
        "configs": [{"id": "x", "knobs": {}, "values": {"a": 1.0, "b": 2.0}}],
    }
    with pytest.raises(SweepError, match="exactly one"):
        Sweep.from_dict(raw)


def test_a_sweep_with_no_objective_is_refused():
    raw = {
        "sweep_id": "t",
        "metrics": {"a": {"role": "proxy", "higher_is_better": True}},
        "configs": [{"id": "x", "knobs": {}, "values": {"a": 1.0}}],
    }
    with pytest.raises(SweepError, match="exactly one"):
        Sweep.from_dict(raw)
