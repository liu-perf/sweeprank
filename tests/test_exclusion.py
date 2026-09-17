"""An exclusion has a direction, and the direction is checkable."""

from __future__ import annotations

import pytest

from sweeprank import exclusion
from sweeprank.model import Sweep, bundled


def _sweep(claimed, values):
    return Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {
            "score": {"role": "objective", "higher_is_better": True},
            "util": {"role": "proxy", "higher_is_better": True},
        },
        "configs": [{
            "id": "cell", "knobs": {},
            "values": {"score": 100.0, "util": sum(values.values()) / len(values)},
            "per_unit": {"util": values},
            "exclusions": [{"unit": "x", "metric": "util", "reason": "instrument",
                            "claimed_direction": claimed}],
        }],
    })


def test_an_exclusion_claiming_high_on_a_value_that_reads_high_passes():
    s = _sweep("high", {"x": 90.0, "a": 50.0, "b": 50.0})
    rows = exclusion.audit(s)
    assert rows[0]["actual_direction"] == "high"
    assert rows[0]["contradicted"] is False
    assert exclusion.check(s)["ok"] is True


def test_an_exclusion_claiming_high_on_a_value_that_reads_low_is_contradicted():
    s = _sweep("high", {"x": 10.0, "a": 50.0, "b": 50.0})
    rows = exclusion.audit(s)
    assert rows[0]["actual_direction"] == "low"
    assert rows[0]["contradicted"] is True
    assert "the opposite way from the stated reason" in rows[0]["message"]
    with pytest.raises(exclusion.ExclusionDirectionError):
        exclusion.check_directions(s)


def test_an_exclusion_with_no_claimed_direction_is_never_contradicted_and_that_is_the_point():
    """The unfalsifiable version is accepted by the direction check by construction.

    Which is why ``claimed_direction`` is a required field on the dataclass:
    the way to stop somebody writing "unknown" forever is to make the reason
    field carry a side at the point the exclusion is recorded, not to try to
    infer one here.
    """
    s = _sweep("unknown", {"x": 10.0, "a": 50.0, "b": 50.0})
    assert exclusion.audit(s)[0]["contradicted"] is False


def test_the_shift_the_exclusion_caused_is_reported():
    s = _sweep("high", {"x": 90.0, "a": 50.0, "b": 50.0})
    r = exclusion.audit(s)[0]
    assert r["mean_all_units"] == pytest.approx(63.3333, abs=1e-3)
    assert r["mean_without_unit"] == pytest.approx(50.0)
    assert r["aggregate_shift"] == pytest.approx(-13.3333, abs=1e-3)


def test_dropping_something_that_is_not_the_outlier_is_reported():
    # Claimed low and reads low, so the direction check passes; it is simply
    # nowhere near being this cell's most deviant unit.
    s = _sweep("low", {"x": 30.0, "a": 50.0, "b": 51.0, "c": 5.0})
    r = exclusion.audit(s)[0]
    assert r["actual_direction"] == "low"
    assert r["contradicted"] is False
    assert r["is_the_outlier"] is False
    assert r["deviation_rank"] == 4
    assert r["problem"] == "not_the_outlier"


def test_an_exclusion_with_no_per_unit_data_cannot_be_checked_and_says_so():
    s = Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {"score": {"role": "objective", "higher_is_better": True}},
        "configs": [{
            "id": "cell", "knobs": {}, "values": {"score": 1.0},
            "exclusions": [{"unit": "x", "metric": "score", "reason": "r",
                            "claimed_direction": "high"}],
        }],
    })
    r = exclusion.audit(s)[0]
    assert r["problem"] == "no_per_unit_data"


def test_the_untouched_metrics_on_the_same_cell_are_named():
    s = bundled("det_scale")
    cfg = s.by_id("det_opt_8gpu")
    left = exclusion.uncorrected_metrics(s, cfg)
    assert "throughput_img_s" in left
    assert "sm_util_mean_pct" not in left
    assert len(left) == 4


def test_a_cell_with_no_exclusions_has_no_untouched_list():
    s = bundled("det_scale")
    assert exclusion.uncorrected_metrics(s, s.by_id("det_opt_4gpu")) == []


def test_the_real_exclusion_is_contradicted_by_the_value_it_dropped():
    """The finding, as an assertion.

    The card was dropped because a leftover context made it read 100% -- a
    mechanism that inflates. Its actual reading is below the mean of the other
    seven, so the exclusion moved the aggregate up, away from the direction its
    own justification implied.
    """
    s = bundled("det_scale")
    r = exclusion.audit_config(s, s.by_id("det_opt_8gpu"))[0]
    assert r["unit"] == "7"
    assert r["claimed_direction"] == "high"
    assert r["actual_direction"] == "low"
    assert r["contradicted"] is True
    assert r["value"] == 63.5
    assert r["mean_without_unit"] == pytest.approx(66.10, abs=0.01)
    assert r["aggregate_shift"] == pytest.approx(0.32, abs=0.01)
    assert r["deviation_rank"] == 4
    assert r["units_total"] == 8
    assert r["is_the_outlier"] is False


def test_the_aggregate_the_report_published_is_the_one_this_recomputes():
    """Otherwise the contradiction above could be an arithmetic disagreement."""
    s = bundled("det_scale")
    r = exclusion.audit_config(s, s.by_id("det_opt_8gpu"))[0]
    assert r["reported_aggregate_without"] == 66.1
    assert r["reported_matches_recomputed"] is True
