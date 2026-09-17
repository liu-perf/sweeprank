"""Every number this repository publishes, recomputed from ``data/``.

The convention is inherited from the sibling projects: a claim that lives only
in prose cannot fail, so it will eventually be wrong. Each assertion here
corresponds to a sentence in the README or ``docs/``, and the negative ones --
the claims this repository explicitly declines to make -- are asserted too,
because a boundary that is not tested gets quietly crossed in an edit.
"""

from __future__ import annotations

import pytest

from sweeprank import analysis


def test_the_proxy_ranks_the_winner_last_of_the_three_live_candidates():
    r = analysis.the_reversal()
    assert r["objective_winner"] == "det_b24nw16"
    assert r["objective_winner_is_last_under_proxy_of_top3"] is True
    assert r["objective_winner_rank_under_proxy_of_top3"] == 3
    assert r["proxy_first_choice"] == "det_b24nw16pin"
    assert r["proxy_picks_a_different_winner"] is True


def test_the_repository_does_not_claim_a_clean_reversal():
    """The negative claim, tested.

    "Utilisation ranks the sweep exactly backwards" is the sentence this
    started as and it is false. If a future edit makes ``exactly_reversed``
    true on this data, the data changed and the prose has to be re-read.
    """
    r = analysis.the_reversal()
    assert r["exactly_reversed"] is False
    assert r["top3_exactly_reversed"] is False
    assert r["slowest_cell_last_under_both"] is True


def test_the_spreads_that_make_this_a_small_question():
    r = analysis.the_reversal()
    assert r["objective_spread_pct"] == pytest.approx(4.53, abs=0.01)
    assert r["proxy_spread_pct"] == pytest.approx(10.34, abs=0.01)


def test_pinned_memory_did_its_job_and_the_step_got_no_faster():
    p = analysis.what_pinning_bought()
    assert p["data_time_frac_pct"]["change_pct"] == pytest.approx(-87.30, abs=0.01)
    assert p["step_time_s"]["change_pct"] == pytest.approx(0.90, abs=0.01)
    assert p["throughput_img_s"]["change_pct"] == pytest.approx(-0.87, abs=0.01)
    assert p["sm_util_mean_pct"]["change_pct"] > 7.0
    assert p["power_mean_w"]["change_pct"] == pytest.approx(7.58, abs=0.01)
    # The direction is the claim: the knob's own target improved and the
    # objective did not.
    assert p["data_time_frac_pct"]["change_pct"] < -80.0
    assert p["throughput_img_s"]["change_pct"] < 0.0


def test_the_resolution_bracket_is_bounded_on_both_sides_and_narrow():
    r = analysis.proxy_resolution()
    lo, hi = r["blind_at_or_below"], r["sound_at_or_above"]
    assert lo == pytest.approx(0.0339, abs=0.0005)
    assert hi == pytest.approx(0.0362, abs=0.0005)
    assert lo < hi
    assert r["blind_from"] == "det_tune"
    assert r["sound_from"] == "det_tune"


def test_the_control_keeps_the_bracket_from_being_read_as_a_verdict():
    c = analysis.proxy_resolution()["control"]
    assert c["concordant"] is True
    assert c["objective_change_pct"] == pytest.approx(125.0, abs=0.5)
    assert c["proxy_change_pct"] == pytest.approx(120.9, abs=0.5)
    # Direction and relative size both agree over a change this large.
    assert abs(c["objective_change_pct"] - c["proxy_change_pct"]) < 10.0
    assert analysis.proxy_resolution()["control_is_inside_bracket"] is False


def test_cross_socket_won_on_both_workloads_and_on_power_too():
    p = analysis.placement_effect()
    assert p["direction_agrees"] is True
    assert p["cheaper_too"] is True
    lo, hi = p["gain_range_pct"]
    assert 3.0 < lo < 3.3, lo
    assert 5.0 < hi < 5.3, hi
    assert p["cells"]["det_scale"]["objective_gain_pct"] == pytest.approx(5.15, abs=0.01)
    assert p["cells"]["seg_scale"]["objective_gain_pct"] == pytest.approx(3.15, abs=0.01)


def test_the_placement_result_keeps_its_own_limits_attached():
    p = analysis.placement_effect()
    assert p["repeats"] == 1
    assert "no error bar" in p["what_the_data_cannot_do"]
    assert "not shown by them" in p["what_the_data_cannot_do"]
    assert p["plausible_mechanism"]


def test_the_eight_card_loss_is_left_unattributed():
    e = analysis.eight_card_attribution()
    assert e["verdict"] == "unattributed"
    d = e["workloads"]["det_scale"]
    g = e["workloads"]["seg_scale"]
    assert d["efficiency_pct"] == pytest.approx(76.6, abs=0.1)
    assert g["efficiency_pct"] == pytest.approx(78.1, abs=0.1)
    # The counterexample that keeps the idle explanation out: comparable loss,
    # sixty times less measured idle.
    assert d["sm_zero_frac_pct_8gpu"] == 6.0
    assert g["sm_zero_frac_pct_8gpu"] == 0.1
    assert abs(d["loss_vs_ideal_pct"] - g["loss_vs_ideal_pct"]) < 2.0
    assert e["idle_explains_at_most_pct_of_step_growth"] < 25.0


def test_the_proxy_moved_the_wrong_way_across_the_scaling_sweeps_too():
    """Utilisation rose, or held, while a fifth of the cards' worth of throughput went."""
    w = analysis.eight_card_attribution()["workloads"]
    g = w["seg_scale"]
    assert g["proxy_8gpu"] > g["proxy_1gpu"]
    assert g["efficiency_pct"] < 80.0


def test_the_excluded_card_reads_the_opposite_way_from_its_own_justification():
    d = analysis.straggler_bound()["direction"]
    assert d["claimed"] == "high" and d["actual"] == "low"
    assert d["contradicted"] is True
    assert d["deviation_rank"] == 4 and d["units_total"] == 8
    assert d["is_the_outlier"] is False
    assert d["aggregate_shift"] == pytest.approx(0.32, abs=0.01)


def test_the_flagged_card_is_too_small_to_be_the_eight_card_story():
    s = analysis.straggler_bound()["size"]
    assert s["compute_gap_pct"] == 2.4
    assert s["share_of_loss_explained_pct"] == pytest.approx(10.3, abs=0.2)
    assert s["share_of_loss_explained_pct"] < 15.0


def test_the_correction_was_applied_to_one_column_of_five():
    assert len(analysis.straggler_bound()["uncorrected"]) == 4


def test_every_bundled_sweep_reports_its_own_proxy_verdict():
    v = analysis.proxy_verdict()
    assert set(v) == {"det_tune", "det_scale", "seg_scale", "governor_control"}
    assert v["det_tune"]["concordant"] is False
    assert v["governor_control"]["concordant"] is True
    # The scaling sweeps have one within-stratum pair each, and both are clean.
    assert v["det_scale"]["pairs"] == 1
    assert v["seg_scale"]["pairs"] == 1


def test_all_findings_is_serialisable_and_complete():
    import json

    a = analysis.all_findings()
    assert set(a) == {
        "proxy_verdict", "the_reversal", "what_pinning_bought", "proxy_resolution",
        "placement_effect", "eight_card_attribution", "straggler_bound", "goals",
        "derived_columns",
    }
    json.dumps(a, default=str)
