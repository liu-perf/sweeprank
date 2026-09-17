"""The loader is strict, and every rejection below is a state a real report was in."""

from __future__ import annotations

import json

import pytest

from sweeprank.model import DATA_DIR, Sweep, SweepError, bundled, bundled_names


def _base():
    return {
        "sweep_id": "t",
        "metrics": {
            "score": {"role": "objective", "higher_is_better": True},
            "util": {"role": "proxy", "higher_is_better": True},
        },
        "configs": [
            {"id": "a", "knobs": {"n": 1}, "values": {"score": 1.0, "util": 2.0}},
            {"id": "b", "knobs": {"n": 2}, "values": {"score": 3.0, "util": 4.0}},
        ],
    }


def test_a_table_with_no_metric_declaration_is_not_a_sweep():
    raw = _base()
    del raw["metrics"]
    with pytest.raises(SweepError, match="not a sweep"):
        Sweep.from_dict(raw)


def test_a_metric_that_does_not_say_which_direction_is_better_is_refused():
    raw = _base()
    del raw["metrics"]["util"]["higher_is_better"]
    with pytest.raises(SweepError, match="which direction"):
        Sweep.from_dict(raw)


def test_a_cell_with_no_values_and_no_reason_is_refused():
    raw = _base()
    raw["configs"][0]["values"] = {}
    with pytest.raises(SweepError, match="has to say why"):
        Sweep.from_dict(raw)


def test_a_cell_with_no_values_and_a_reason_is_accepted_and_excluded():
    raw = _base()
    raw["configs"].append({"id": "oom", "knobs": {"n": 3}, "values": {},
                           "failed": "CUDA out of memory"})
    s = Sweep.from_dict(raw)
    assert [c.id for c in s.failed] == ["oom"]
    assert "oom" not in [c.id for c in s.cells]
    # A failed cell is out of the sweep, not last in it.
    assert s.winner().id == "b"


def test_values_for_undeclared_metrics_are_refused():
    raw = _base()
    raw["configs"][0]["values"]["mystery"] = 1.0
    with pytest.raises(SweepError, match="undeclared metrics"):
        Sweep.from_dict(raw)


def test_duplicate_config_ids_are_refused():
    raw = _base()
    raw["configs"][1]["id"] = "a"
    with pytest.raises(SweepError, match="duplicate"):
        Sweep.from_dict(raw)


def test_a_goal_on_an_undeclared_metric_is_refused():
    raw = _base()
    raw["declared_goal"] = {"metric": "nope", "op": ">=", "value": 1}
    with pytest.raises(SweepError, match="undeclared metric"):
        Sweep.from_dict(raw)


def test_an_unknown_goal_operator_is_refused_at_use():
    raw = _base()
    raw["declared_goal"] = {"metric": "util", "op": "~=", "value": 1}
    s = Sweep.from_dict(raw)
    with pytest.raises(SweepError, match="unknown goal operator"):
        s.goal.accepts(1.0)


def test_a_stratum_knob_missing_from_a_cell_is_refused():
    """A stratum that does not partition every cell silently drops comparisons."""
    raw = _base()
    raw["strata"] = ["n"]
    del raw["configs"][1]["knobs"]["n"]
    with pytest.raises(SweepError, match="does not partition"):
        Sweep.from_dict(raw)


def test_asking_for_a_missing_value_is_an_error_not_a_zero():
    raw = _base()
    del raw["configs"][0]["values"]["util"]
    s = Sweep.from_dict(raw)
    with pytest.raises(SweepError, match="no value for metric"):
        s.value(s.by_id("a"), "util")


def test_every_bundled_sweep_loads_and_declares_its_repeat_count():
    names = bundled_names()
    assert set(names) == {"det_scale", "det_tune", "governor_control", "seg_scale"}
    for n in names:
        s = bundled(n)
        assert s.what, n
        assert s.provenance.get("repeats") is not None, (
            f"{n} does not say how many runs per cell it has, and every conclusion "
            f"drawn from it depends on that number"
        )
        assert s.cells, n


def test_every_bundled_sweep_is_valid_json_utf8_without_a_bom():
    for p in sorted(DATA_DIR.glob("*.json")):
        blob = p.read_bytes()
        assert not blob.startswith(b"\xef\xbb\xbf"), p.name
        json.loads(blob.decode("utf-8"))


def test_a_failed_cell_carries_its_reason_verbatim():
    s = bundled("det_tune")
    oom = list(s.failed)
    assert [c.id for c in oom] == ["det_b32"]
    assert "out of memory" in oom[0].failed.lower()
