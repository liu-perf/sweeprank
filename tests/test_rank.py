"""Pairwise concordance, strata, and the resolution bracket."""

from __future__ import annotations

from sweeprank.model import Sweep, bundled
from sweeprank.rank import concordance, cross_sweep_resolution, pairs, resolution


def _sweep(rows, strata=None):
    return Sweep.from_dict({
        "sweep_id": "t",
        "metrics": {
            "score": {"role": "objective", "higher_is_better": True},
            "util": {"role": "proxy", "higher_is_better": True},
        },
        **({"strata": strata} if strata else {}),
        "configs": [
            {"id": cid, "knobs": knobs, "values": {"score": s, "util": u}}
            for cid, knobs, s, u in rows
        ],
    })


def test_a_perfectly_agreeing_proxy_is_concordant():
    s = _sweep([("a", {}, 1.0, 10.0), ("b", {}, 2.0, 20.0), ("c", {}, 3.0, 30.0)])
    c = concordance(s, "util")
    assert c["counts"] == {"concordant": 3, "discordant": 0,
                           "proxy_tied": 0, "objective_tied": 0}
    assert c["concordant"] is True
    assert c["tau_b"] == 1.0


def test_a_perfectly_inverted_proxy_is_discordant():
    s = _sweep([("a", {}, 1.0, 30.0), ("b", {}, 2.0, 20.0), ("c", {}, 3.0, 10.0)])
    c = concordance(s, "util")
    assert c["counts"]["discordant"] == 3
    assert c["concordant"] is False
    assert c["tau_b"] == -1.0


def test_a_proxy_that_cannot_tell_two_cells_apart_is_not_concordant():
    """A proxy tie on a real objective difference is a failure, not a pass.

    Kept separate from a discordant pair because they mean opposite things and
    an aggregate that merges them would let a blind proxy through.
    """
    s = _sweep([("a", {}, 1.0, 10.0), ("b", {}, 2.0, 10.0)])
    c = concordance(s, "util")
    assert c["counts"]["proxy_tied"] == 1
    assert c["counts"]["discordant"] == 0
    assert c["concordant"] is False


def test_an_objective_tie_is_not_held_against_the_proxy():
    s = _sweep([("a", {}, 1.0, 10.0), ("b", {}, 1.0, 20.0)])
    c = concordance(s, "util")
    assert c["counts"]["objective_tied"] == 1
    assert c["concordant"] is True


def test_strata_stop_cells_at_different_scales_from_forming_a_pair():
    rows = [("s1", {"n": 1}, 100.0, 70.0), ("s2", {"n": 2}, 200.0, 60.0),
            ("s2b", {"n": 2}, 210.0, 65.0)]
    without = concordance(_sweep(rows), "util")
    within = concordance(_sweep(rows, strata=["n"]), "util")
    assert without["pairs"] == 3
    assert within["pairs"] == 1
    assert within["strata"] == ["n"]
    # Only the same-scale pair survives, and it is the only one that was ever
    # a choice between two configurations.
    assert [(p["a"], p["b"]) for p in pairs(_sweep(rows, strata=["n"]), "util")] == [
        ("s2", "s2b")]


def test_the_513_percent_regression_stays_fixed():
    """A named test for a real defect in this tool's first version.

    Before strata existed, ``resolution`` on the scaling sweep compared the
    eight-card cell with the one-card cell and reported the proxy "blind at or
    below 513%". The arithmetic was right and the question was malformed: total
    throughput across eight cards is not a rival answer to total throughput
    across one.
    """
    s = bundled("det_scale")
    assert s.strata == ["ngpu"]
    r = resolution(s, "sm_util_mean_pct")
    for key in ("blind_at_or_below", "sound_at_or_above"):
        assert r[key] is None or r[key] < 1.0, (key, r[key])
    ps = pairs(s, "sm_util_mean_pct")
    assert len(ps) == 1
    assert {ps[0]["a"], ps[0]["b"]} == {"det_opt_4gpu", "det_opt_4gpuX"}


def test_resolution_reports_a_bracket_and_does_not_interpolate_across_it():
    s = _sweep([
        ("a", {}, 100.0, 10.0),
        ("b", {}, 101.0, 9.0),    # 1% apart, proxy gets it backwards
        ("c", {}, 150.0, 50.0),   # far away, proxy gets it right
    ])
    r = resolution(s, "util")
    assert 0.009 < r["blind_at_or_below"] < 0.011
    assert r["sound_at_or_above"] > r["blind_at_or_below"]
    lo, hi = r["unresolved_between"]
    assert lo == r["blind_at_or_below"] and hi == r["sound_at_or_above"]


def test_resolution_on_a_clean_proxy_reports_the_smallest_gap_it_actually_tested():
    s = _sweep([("a", {}, 100.0, 10.0), ("b", {}, 200.0, 20.0)])
    r = resolution(s, "util")
    assert r["blind_at_or_below"] is None
    assert abs(r["sound_at_or_above"] - 1.0) < 1e-9


def test_the_tuning_sweep_bracket_is_tight_and_ordered():
    r = resolution(bundled("det_tune"), "sm_util_mean_pct")
    lo, hi = r["blind_at_or_below"], r["sound_at_or_above"]
    assert lo is not None and hi is not None
    assert lo < hi, (lo, hi)
    assert 0.030 < lo < 0.040, lo
    assert 0.035 < hi < 0.040, hi


def test_pooling_never_reports_a_bracket_with_the_sides_the_wrong_way_round():
    sweeps = [bundled(n) for n in ("det_tune", "det_scale", "seg_scale",
                                   "governor_control")]
    p = cross_sweep_resolution(sweeps, "sm_util_mean_pct")
    assert p["blind_at_or_below"] < p["sound_at_or_above"]
    assert p["blind_from"] and p["sound_from"]


def test_the_control_sweep_is_concordant():
    """Without this the whole result reads as 'never look at utilisation'."""
    c = concordance(bundled("governor_control"), "sm_util_mean_pct")
    assert c["concordant"] is True
    assert c["counts"]["concordant"] == 1


def test_the_tuning_sweep_is_discordant_and_that_is_the_headline():
    c = concordance(bundled("det_tune"), "sm_util_mean_pct")
    assert c["concordant"] is False
    assert c["counts"]["discordant"] == 2
    assert c["pairs"] == 15
    assert c["tau_b"] is not None and c["tau_b"] > 0.0, (
        "tau is positive and the proxy still picks the wrong winner -- which is "
        "why this tool counts pairs instead of reporting a correlation"
    )
