"""One column, one baseline -- and the ratios have to reproduce."""

from __future__ import annotations

import pytest

from sweeprank import baseline
from sweeprank.model import Sweep, bundled, bundled_names


def _all():
    return {n: bundled(n) for n in bundled_names()}


def test_a_single_baseline_column_is_accepted():
    s = Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {"score": {"role": "objective", "higher_is_better": True}},
        "derived": [{"name": "speedup", "formula": "ratio_to_baseline", "of": "score",
                     "baselines": {"a": ["b"]}, "reported": {"b": 2.0}}],
        "configs": [
            {"id": "a", "knobs": {}, "values": {"score": 10.0}},
            {"id": "b", "knobs": {}, "values": {"score": 20.0}},
        ],
    })
    assert baseline.check_baselines(s) == []
    assert baseline.check(s)["ok"] is True


def test_two_baselines_in_one_column_are_refused():
    s = Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {"score": {"role": "objective", "higher_is_better": True}},
        "derived": [{"name": "speedup", "formula": "ratio_to_baseline", "of": "score",
                     "baselines": {"a": ["b"], "b": ["c"]}}],
        "configs": [
            {"id": "a", "knobs": {}, "values": {"score": 10.0}},
            {"id": "b", "knobs": {}, "values": {"score": 20.0}},
            {"id": "c", "knobs": {}, "values": {"score": 40.0}},
        ],
    })
    f = baseline.check_baselines(s)
    assert len(f) == 1
    assert f[0]["problem"] == "multiple_baselines"
    assert "not comparable" in f[0]["message"]


def test_a_transcription_error_is_caught_separately_from_the_structural_one():
    s = Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {"score": {"role": "objective", "higher_is_better": True}},
        "derived": [{"name": "speedup", "formula": "ratio_to_baseline", "of": "score",
                     "baselines": {"a": ["b"]}, "reported": {"b": 3.0}}],
        "configs": [
            {"id": "a", "knobs": {}, "values": {"score": 10.0}},
            {"id": "b", "knobs": {}, "values": {"score": 20.0}},
        ],
    })
    r = baseline.check(s)
    assert r["findings"] == []
    assert len(r["mismatched"]) == 1
    assert r["mismatched"][0]["recomputed"] == pytest.approx(2.0)
    assert r["ok"] is False


def test_a_cross_sweep_baseline_needs_the_other_sweep_supplied():
    s = Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {"score": {"role": "objective", "higher_is_better": True}},
        "derived": [{"name": "speedup", "formula": "ratio_to_baseline", "of": "score",
                     "baselines": {"elsewhere:x": ["b"]}}],
        "configs": [{"id": "b", "knobs": {}, "values": {"score": 20.0}}],
    })
    with pytest.raises(baseline.BaselineError, match="was not supplied"):
        baseline.recompute(s)


def test_an_efficiency_on_a_one_unit_row_is_refused():
    s = Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {"score": {"role": "objective", "higher_is_better": True}},
        "derived": [{"name": "efficiency", "formula": "speedup_over_units", "of": "speedup",
                     "units_knob": "n", "reported": {"solo": 118.4, "pair": 98.9}}],
        "configs": [
            {"id": "solo", "knobs": {"n": 1}, "values": {"score": 10.0}},
            {"id": "pair", "knobs": {"n": 2}, "values": {"score": 20.0}},
        ],
    })
    f = baseline.check_efficiency_domain(s)
    assert [x["row"] for x in f] == ["solo"]
    assert "above 100%" in f[0]["message"]


def test_the_real_speedup_column_is_refused_for_having_two_baselines():
    """The finding, as an assertion."""
    s = bundled("det_scale")
    f = baseline.check_baselines(s)
    assert len(f) == 1
    assert f[0]["column"] == "speedup"
    assert set(f[0]["baselines"]) == {"det_tune:det_1gpu", "det_opt_1gpu"}


def test_the_real_efficiency_column_prints_scaling_efficiency_above_100_on_one_card():
    s = bundled("det_scale")
    f = baseline.check_efficiency_domain(s)
    rows = {x["row"]: x["reported"] for x in f}
    assert rows == {"single_bs24nw8": 117.0, "single_bs24nw32": 118.3,
                    "single_bs24nw32pin": 118.4}
    # And the baseline row itself, at exactly 100%, is not flagged: dividing the
    # baseline by itself is not the defect.
    assert "single_bs8" not in rows


def test_every_published_ratio_reproduces_from_the_objective():
    """Both halves of the column are individually correct.

    That is the whole difficulty: nothing is mis-transcribed, so no arithmetic
    check finds anything, and the column is still unreadable.
    """
    all_ = _all()
    rows = baseline.recompute(all_["det_scale"], others=all_)
    checked = [r for r in rows if r["agrees"] is not None]
    assert checked, "nothing was actually compared"
    assert all(r["agrees"] for r in checked), [r for r in checked if not r["agrees"]]


def test_the_rows_that_live_in_another_sweep_are_reported_not_silently_skipped():
    all_ = _all()
    rows = baseline.recompute(all_["det_scale"], others=all_)
    aliases = [r for r in rows if r["recomputed"] is None]
    assert {r["row"] for r in aliases} == {
        "single_bs8", "single_bs24nw8", "single_bs24nw32", "single_bs24nw32pin"}
    assert all(r["note"] for r in aliases)
