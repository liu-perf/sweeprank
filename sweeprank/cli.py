"""Command line. Output format is shared with the sibling tools in this series::

    {location}: [{status}] {message}

``--json`` prints the underlying structure instead, and ``--fail-on`` decides
what makes the exit code non-zero. Subcommands taking a sweep accept either a
path to a JSON file or the bare name of one of the bundled sweeps.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import analysis, baseline, exclusion, goal, rank
from .model import DATA_DIR, Sweep, bundled, bundled_names
from .role import DESCRIPTIONS, MAY_SELECT, ROLES

ERROR, WARN, INFO = "error", "warn", "info"
_SEVERITY = {INFO: 0, WARN: 1, ERROR: 2}


def _load(spec: str) -> Sweep:
    p = pathlib.Path(spec)
    if p.exists():
        return Sweep.load(p)
    if (DATA_DIR / f"{spec}.json").exists():
        return bundled(spec)
    raise SystemExit(
        f"sweeprank: no such sweep {spec!r}; bundled sweeps are: {', '.join(bundled_names())}"
    )


def _emit(findings, args) -> int:
    if args.json:
        print(json.dumps(findings, indent=2, ensure_ascii=False, default=str))
    else:
        for f in findings:
            print(f"{f['location']}: [{f['status']}] {f['message']}")
        if not findings:
            print("no findings")
    threshold = _SEVERITY.get(args.fail_on)
    if threshold is None:
        return 0
    worst = max((_SEVERITY[f["status"]] for f in findings), default=-1)
    return 1 if worst >= threshold else 0


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


# -- subcommands -----------------------------------------------------------


def cmd_roles(args) -> int:
    for r in ROLES:
        print(f"{r:<11} {DESCRIPTIONS[r]}")
        print(f"{'':<11} may select: {MAY_SELECT[r]}")
    return 0


def cmd_show(args) -> int:
    s = _load(args.sweep)
    if args.json:
        print(json.dumps({
            "sweep": s.sweep_id, "what": s.what, "provenance": s.provenance,
            "metrics": {k: vars(v) for k, v in s.metrics.items()},
            "cells": [c.id for c in s.cells], "failed": [c.id for c in s.failed],
        }, indent=2, ensure_ascii=False))
        return 0
    print(f"{s.sweep_id}: {s.what}")
    print(f"  repeats per cell: {s.repeats()}")
    print("  metrics:")
    for m in s.metrics.values():
        mark = "*" if m.role == "objective" else " "
        arrow = "higher better" if m.higher_is_better else "lower better"
        print(f"   {mark} {m.name:<22} {m.role:<11} {arrow}"
              + (f"  [{m.unit}]" if m.unit else ""))
    print(f"  cells: {len(s.cells)} produced numbers, {len(s.failed)} failed")
    for c in s.failed:
        print(f"    {c.id}: excluded -- {c.failed}")
    return 0


def cmd_rank(args) -> int:
    s = _load(args.sweep)
    findings = []
    proxies = [args.proxy] if args.proxy else [m.name for m in s.proxies]
    if not proxies:
        print(f"{s.sweep_id}: [info] no metric declares role 'proxy'")
        return 0
    for p in proxies:
        c = rank.concordance(s, p)
        r = rank.resolution(s, p)
        counts = c["counts"]
        tau = "n/a" if c["tau_b"] is None else f"{c['tau_b']:+.3f}"
        if c["concordant"]:
            findings.append({
                "location": f"{s.sweep_id}:{p}", "status": INFO,
                "message": (f"orders all {counts['concordant']} pairs the same way as "
                            f"{c['objective']} (tau_b {tau}); may select at gaps of "
                            f"{_pct(r['sound_at_or_above'])} or more"),
                "detail": {"concordance": c, "resolution": r},
            })
        else:
            worst = max(c["discordant_pairs"], key=lambda x: x["objective_rel_gap"], default=None)
            extra = ""
            if worst:
                extra = (f"; worst is {worst['a']} vs {worst['b']}, "
                         f"{_pct(worst['objective_rel_gap'])} apart on {c['objective']}")
            findings.append({
                "location": f"{s.sweep_id}:{p}", "status": ERROR,
                "message": (f"disagrees with {c['objective']} on {counts['discordant']} of "
                            f"{c['pairs']} pairs (tau_b {tau}); blind at or below "
                            f"{_pct(r['blind_at_or_below'])}{extra}"),
                "detail": {"concordance": c, "resolution": r},
            })
        if args.resolution and not args.json:
            print(f"{s.sweep_id}:{p}: [info] resolution "
                  f"blind<={_pct(r['blind_at_or_below'])} "
                  f"sound>={_pct(r['sound_at_or_above'])}")
        if c["repeats"] < 2:
            findings.append({
                "location": f"{s.sweep_id}:{p}", "status": WARN,
                "message": (f"{c['repeats']} run per cell, so a pair whose objective gap is "
                            f"small may be a re-ordering of noise; the verdict above is "
                            f"about this sweep as recorded, not about the hardware"),
            })
    return _emit(findings, args)


def cmd_goal(args) -> int:
    s = _load(args.sweep)
    r = goal.apply_goal(s)
    if not r.get("has_goal"):
        print(f"{s.sweep_id}: [info] no declared goal")
        return 0
    findings = []
    loc = f"{s.sweep_id}:{r['goal']}"
    if r["goal_metric_role"] != "objective":
        findings.append({
            "location": loc, "status": WARN,
            "message": (f"written on a {r['goal_metric_role']}, not on the objective "
                        f"({r['objective']})"),
        })
    if r["objective_winner_rejected"]:
        price = r.get("price") or {}
        bits = []
        for k, v in price.items():
            if v.get("change_pct") is not None:
                bits.append(f"{k} {v['change_pct']:+.2f}%")
        findings.append({
            "location": loc, "status": ERROR,
            "message": (
                f"rejects {r['objective_winner']}, the fastest cell in the sweep "
                f"({r['objective']} = {r['objective_winner_value']:g}, reads "
                f"{r['objective_winner_goal_value']:g} on the goal metric); it selects "
                f"{r['goal_pick']} instead"
                + (f", which costs {', '.join(bits)}" if bits else "")
            ),
            "detail": r,
        })
    else:
        findings.append({
            "location": loc, "status": INFO,
            "message": (f"accepts {len(r['accepted'])} of {len(r['accepted']) + len(r['rejected'])} "
                        f"cells and keeps the objective winner {r['objective_winner']}"),
            "detail": r,
        })
    return _emit(findings, args)


def cmd_derived(args) -> int:
    s = _load(args.sweep)
    others = {n: bundled(n) for n in bundled_names()}
    others[s.sweep_id] = s
    r = baseline.check(s, others=others)
    findings = []
    for f in r["findings"]:
        findings.append({
            "location": f"{s.sweep_id}:{f['column']}"
                        + (f":{f['row']}" if "row" in f else ""),
            "status": ERROR, "message": f["message"], "detail": f,
        })
    for m in r["mismatched"]:
        findings.append({
            "location": f"{s.sweep_id}:{m['column']}:{m['row']}", "status": ERROR,
            "message": (f"recomputes to {m['recomputed']:.4f} against baseline "
                        f"{m['baseline']}, but the report published {m['reported']}"),
            "detail": m,
        })
    if not findings and s.derived:
        findings.append({
            "location": s.sweep_id, "status": INFO,
            "message": (f"{len(s.derived)} derived column(s); every ratio reproduces from "
                        f"the objective against a single named baseline"),
        })
    if not s.derived:
        print(f"{s.sweep_id}: [info] no derived columns")
        return 0
    return _emit(findings, args)


def cmd_exclusions(args) -> int:
    s = _load(args.sweep)
    r = exclusion.check(s)
    if not r["exclusions"]:
        print(f"{s.sweep_id}: [info] no per-unit exclusions")
        return 0
    findings = []
    for e in r["exclusions"]:
        loc = f"{s.sweep_id}:{e['config']}:unit{e['unit']}"
        problem = e.get("problem")
        if problem == "direction_contradicted":
            findings.append({"location": loc, "status": ERROR, "message": e["message"],
                             "detail": e})
        elif problem:
            findings.append({"location": loc, "status": WARN, "message": e["message"],
                             "detail": e})
        else:
            findings.append({
                "location": loc, "status": INFO,
                "message": (f"dropped from {e['metric']}; reads {e['actual_direction']} as "
                            f"claimed, aggregate shifts {e['aggregate_shift']:+.2f}"),
                "detail": e,
            })
        untouched = exclusion.uncorrected_metrics(s, s.by_id(e["config"]))
        if untouched:
            findings.append({
                "location": loc, "status": WARN,
                "message": (f"the exclusion was applied to {e['metric']} only; "
                            f"{len(untouched)} other metric(s) on the same cell were left "
                            f"uncorrected ({', '.join(untouched)})"),
            })
    return _emit(findings, args)


def cmd_check(args) -> int:
    """Every check over every bundled sweep. This is the CI gate."""
    findings = []
    loaded = {n: bundled(n) for n in bundled_names()}
    for name, s in loaded.items():
        for p in (m.name for m in s.proxies):
            c = rank.concordance(s, p)
            if not c["concordant"]:
                findings.append({
                    "location": f"{name}:{p}", "status": ERROR,
                    "message": (f"proxy disagrees with {c['objective']} on "
                                f"{c['counts']['discordant']} of {c['pairs']} pairs"),
                })
        g = goal.apply_goal(s)
        if g.get("objective_winner_rejected"):
            findings.append({
                "location": f"{name}:{g['goal']}", "status": ERROR,
                "message": f"the stated goal rejects the objective winner {g['objective_winner']}",
            })
        b = baseline.check(s, others=loaded)
        for f in b["findings"]:
            findings.append({"location": f"{name}:{f['column']}", "status": ERROR,
                             "message": f["message"]})
        for m in b["mismatched"]:
            findings.append({
                "location": f"{name}:{m['column']}:{m['row']}", "status": ERROR,
                "message": f"published {m['reported']} but recomputes to {m['recomputed']:.4f}",
            })
        for e in exclusion.check(s)["contradicted"]:
            findings.append({"location": f"{name}:{e['config']}:unit{e['unit']}",
                             "status": ERROR, "message": e["message"]})
    if args.json:
        print(json.dumps(findings, indent=2, ensure_ascii=False))
    else:
        for f in findings:
            print(f"{f['location']}: [{f['status']}] {f['message']}")
        print(f"\n{len(loaded)} sweep(s), {len(findings)} finding(s). "
              f"Every one of these is a defect the source report shipped with; "
              f"`sweeprank check` is expected to be loud.")
    return 0


def cmd_claims(args) -> int:
    """The README's numbers, recomputed. Kept as a CI gate in the sibling repos."""
    a = analysis.all_findings()
    if args.json:
        print(json.dumps(a, indent=2, ensure_ascii=False, default=str))
        return 0
    rev = a["the_reversal"]
    print("the reversal (batch fixed at 24, only dataloader knobs move)")
    print(f"  by throughput : {' > '.join(rev['order_by_objective'])}")
    print(f"  by SM util    : {' > '.join(rev['order_by_proxy'])}")
    print(f"  a clean reversal: {rev['exactly_reversed']} over four cells, "
          f"{rev['top3_exactly_reversed']} over the top three -- it is not one, "
          f"and is not claimed to be")
    print(f"  the objective's winner {rev['objective_winner']} ranks "
          f"{rev['objective_winner_rank_under_proxy_of_top3']} of 3 under the proxy "
          f"(last: {rev['objective_winner_is_last_under_proxy_of_top3']})")
    print(f"  the proxy would pick {rev['proxy_first_choice']} instead: "
          f"{rev['proxy_picks_a_different_winner']}")
    print(f"  objective spread {rev['objective_spread_pct']:.2f}%,"
          f" proxy spread {rev['proxy_spread_pct']:.2f}%")
    pin = a["what_pinning_bought"]
    print("\nwhat pinned memory bought")
    for k in ("data_time_frac_pct", "step_time_s", "throughput_img_s",
              "sm_util_mean_pct", "power_mean_w"):
        v = pin[k]
        print(f"  {k:<20} {v['without_pin']:>8} -> {v['with_pin']:>8}"
              f"   {v['change_pct']:+7.2f}%")
    res = a["proxy_resolution"]
    print("\nproxy resolution, pooled")
    print(f"  blind at or below {_pct(res['blind_at_or_below'])} "
          f"(from {res['blind_from']})")
    print(f"  sound at or above {_pct(res['sound_at_or_above'])} "
          f"(from {res['sound_from']})")
    print(f"  nothing measured in between; halves from different sweeps: "
          f"{res['halves_from_different_sweeps']}")
    ctl = res["control"]
    print(f"  control: a {ctl['objective_change_pct']:+.1f}% objective change moves the "
          f"proxy {ctl['proxy_change_pct']:+.1f}%, concordant: {ctl['concordant']}")
    print("  (the control sits above the bracket, so it does not tighten it -- it is "
          "there so the bracket is not read as a verdict on the metric)")
    pl = a["placement_effect"]
    print("\ncross-socket vs same-socket, four cards")
    for k, v in pl["cells"].items():
        print(f"  {k:<11} {v['objective_gain_pct']:+.2f}% throughput, "
              f"{v['power_change_pct']:+.2f}% power")
    print(f"  direction agrees on both workloads: {pl['direction_agrees']}, "
          f"cheaper too: {pl['cheaper_too']}, repeats: {pl['repeats']}")
    ec = a["eight_card_attribution"]
    print("\neight cards")
    for k, v in ec["workloads"].items():
        print(f"  {k:<11} {v['speedup']:.2f}x = {v['efficiency_pct']:.1f}% of ideal, "
              f"idle {v['sm_zero_frac_pct_8gpu']}%, "
              f"proxy {v['proxy_1gpu']} -> {v['proxy_8gpu']}")
    print(f"  verdict: {ec['verdict']}")
    st = a["straggler_bound"]["direction"]
    print("\nthe excluded card")
    print(f"  claimed to read {st['claimed']}, reads {st['actual']} "
          f"({st['value']} vs {st['mean_without_unit']:.2f} for the others); "
          f"contradicted: {st['contradicted']}")
    print(f"  deviation rank {st['deviation_rank']} of {st['units_total']}, "
          f"is the outlier: {st['is_the_outlier']}")
    print(f"  bounds {a['straggler_bound']['size']['share_of_loss_explained_pct']:.1f}% "
          f"of the observed eight-card loss")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sweeprank", description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the structure, not the lines")
    ap.add_argument("--fail-on", choices=["error", "warn", "none"], default="none",
                    help="exit non-zero at this severity or worse (default: none)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("roles", help="what each role may do").set_defaults(fn=cmd_roles)
    sub.add_parser("claims", help="this repository's findings, recomputed").set_defaults(
        fn=cmd_claims)
    sub.add_parser("check", help="every check over every bundled sweep").set_defaults(
        fn=cmd_check)

    for name, fn, helptext in (
        ("show", cmd_show, "the metric declaration and the cells"),
        ("goal", cmd_goal, "apply the sweep's own stated goal to its own results"),
        ("derived", cmd_derived, "one column, one baseline"),
        ("exclusions", cmd_exclusions, "check a dropped unit against its stated direction"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("sweep", help="path to a sweep JSON, or a bundled sweep name")
        p.set_defaults(fn=fn)

    p = sub.add_parser("rank", help="does the proxy order the sweep like the objective?")
    p.add_argument("sweep", help="path to a sweep JSON, or a bundled sweep name")
    p.add_argument("--proxy", help="check only this proxy")
    p.add_argument("--resolution", action="store_true", help="also print the resolution bound")
    p.set_defaults(fn=cmd_rank)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
