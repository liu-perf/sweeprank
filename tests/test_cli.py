"""Exit codes and output shape. The CI gates below depend on both."""

from __future__ import annotations

import json

import pytest

from sweeprank.cli import main


def test_roles_lists_all_five(capsys):
    assert main(["roles"]) == 0
    out = capsys.readouterr().out
    for r in ("objective", "proxy", "cost", "diagnostic", "constraint"):
        assert r in out
    assert "may select" in out


def test_show_marks_the_objective_and_names_the_failed_cell(capsys):
    assert main(["show", "det_tune"]) == 0
    out = capsys.readouterr().out
    assert "* throughput_img_s" in out
    assert "det_b32: excluded" in out
    assert "repeats per cell: 1" in out


def test_rank_exits_nonzero_on_the_discordant_sweep_with_fail_on_error(capsys):
    assert main(["--fail-on", "error", "rank", "det_tune"]) == 1
    out = capsys.readouterr().out
    assert "[error]" in out
    assert "blind at or below" in out


def test_rank_exits_zero_on_the_control(capsys):
    assert main(["--fail-on", "error", "rank", "governor_control"]) == 0
    assert "[error]" not in capsys.readouterr().out


def test_the_single_run_warning_is_emitted_and_can_fail_the_build(capsys):
    assert main(["--fail-on", "warn", "rank", "governor_control"]) == 1
    assert "run per cell" in capsys.readouterr().out


def test_fail_on_defaults_to_none(capsys):
    assert main(["rank", "det_tune"]) == 0
    capsys.readouterr()


def test_json_output_parses(capsys):
    assert main(["--json", "rank", "det_tune"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert {"location", "status", "message"} <= set(payload[0])


def test_goal_exits_nonzero_when_the_goal_rejects_the_winner(capsys):
    assert main(["--fail-on", "error", "goal", "det_tune"]) == 1
    out = capsys.readouterr().out
    assert "fastest cell" in out
    assert "det_b24nw16pin" in out
    assert "power_mean_w +7.58%" in out


def test_goal_on_a_sweep_without_one_says_so(capsys):
    assert main(["goal", "seg_scale"]) == 0
    assert "no declared goal" in capsys.readouterr().out


def test_derived_refuses_the_two_baseline_column(capsys):
    assert main(["--fail-on", "error", "derived", "det_scale"]) == 1
    out = capsys.readouterr().out
    assert "2 different baselines" in out
    assert "118.4%" in out


def test_derived_on_a_sweep_without_derived_columns_says_so(capsys):
    assert main(["derived", "det_tune"]) == 0
    assert "no derived columns" in capsys.readouterr().out


def test_exclusions_reports_the_contradiction_and_the_untouched_columns(capsys):
    assert main(["--fail-on", "error", "exclusions", "det_scale"]) == 1
    out = capsys.readouterr().out
    assert "the opposite way from the stated reason" in out
    assert "left uncorrected" in out


def test_exclusions_on_a_sweep_without_any_says_so(capsys):
    assert main(["exclusions", "det_tune"]) == 0
    assert "no per-unit exclusions" in capsys.readouterr().out


def test_check_is_loud_and_still_exits_zero(capsys):
    """``check`` is a report, not a gate.

    Every line it prints is a defect the source report shipped with, so a
    non-zero exit would mean this repository never builds. The gates that must
    stay green are in ``tests/``.
    """
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert out.count("[error]") == 7
    assert "expected to be loud" in out


def test_claims_prints_every_section(capsys):
    assert main(["claims"]) == 0
    out = capsys.readouterr().out
    for heading in ("the reversal", "what pinned memory bought", "proxy resolution",
                    "cross-socket vs same-socket", "eight cards", "the excluded card"):
        assert heading in out
    assert "unattributed" in out


def test_claims_json_round_trips(capsys):
    assert main(["--json", "claims"]) == 0
    assert "the_reversal" in json.loads(capsys.readouterr().out)


def test_an_unknown_sweep_name_lists_the_known_ones():
    with pytest.raises(SystemExit, match="bundled sweeps are"):
        main(["rank", "not_a_sweep"])


def test_a_sweep_can_be_given_as_a_path(capsys, tmp_path):
    from sweeprank.model import DATA_DIR

    p = tmp_path / "copy.json"
    p.write_text((DATA_DIR / "det_tune.json").read_text(encoding="utf-8"),
                 encoding="utf-8")
    assert main(["--fail-on", "error", "rank", str(p)]) == 1
    capsys.readouterr()
