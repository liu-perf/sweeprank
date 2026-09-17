# Whose number picks the winner

## The state a sweep report is in

Here is a real tuning sweep, four cells, batch fixed at 24 so the only things
moving are the dataloader knobs:

| cell | workers | pin | img/s | SM util % | W | dataloader wait |
|---|---|---|---|---|---|---|
| `det_b24nw16` | 16 | no | **86.54** | 68.7 | 329.8 | 6.3% |
| `det_b24nw16pin` | 16 | yes | 85.79 | **73.6** | 354.8 | 0.8% |
| `det_b24nw32` | 32 | no | 83.70 | 70.1 | 335.1 | 6.1% |
| `det_b24` | 8 | no | 82.79 | 66.7 | 334.5 | 6.5% |

Nothing in that table says which column is the answer. Both `img/s` and
`SM util %` are floats with a header and a plausible claim to being "the
performance number". The sweep's own opening line reads:

> target SM utilisation ≥ 70%, baseline batch 8 / workers 8 / pin False →
> SM 61.4%, 70.74 img/s

So the goal is written on the second column and the winner is in the first, and
the two are never compared, because a header line and a table at the bottom of
the same file are not in the same room.

## The device

Every metric declares a role, and the roles are not symmetric:

| role | may it pick the winner? |
|---|---|
| `objective` | always — exactly one per sweep |
| `proxy` | only once shown concordant with the objective at the decision's scale |
| `cost` | never; it may veto |
| `diagnostic` | never; it explains |
| `constraint` | never; a failing cell is *out* of the sweep, not last in it |

Only the objective gets the unconditional right. Everything else either earns
it or is denied it outright — and the earning is enforced at the call site:

```python
def assert_may_select(name, role, *, concordant=None):
    ...
    if role == PROXY:
        if concordant is None:
            raise RoleError(f"{name!r} is a proxy; selecting on it requires the "
                            f"concordance check to have been run ... and it has not been")
```

`concordant` is keyword-only and its default means *"nobody checked"*, not
*"yes"*. A caller who never ran the ranking check cannot get past the gate by
omission, which is the exact failure mode in the report above: nobody decided
the proxy could select, nobody decided it couldn't, and it did.

`Sweep.winner()` routes through the same gate, so there is no back door:

```python
>>> s.winner()          # the objective
<Config det_b24nw16>
>>> s.winner("sm_util_mean_pct")
RoleError: 'sm_util_mean_pct' is a proxy and it ranks this sweep differently from
the objective; it may not select a winner here
```

### The two roles people skip

`cost` and `diagnostic` exist because without them everything that is not the
objective becomes a proxy, and then the roles stop discriminating.

Power is a **cost**: a configuration that is 1% faster and 8% hotter is a
trade, and a trade is not a winner. Costs report and veto; they never select.

Dataloader wait is a **diagnostic**: it is how you find out what a knob did,
and it is never the reason to keep the knob. Look at the pinned-memory row
above — `data_time` fell from 6.3% of the step to 0.8%, an 87% reduction, which
is precisely what `pin_memory` is for. The step time went *up* 0.9%. The
diagnostic explained the mechanism perfectly and would have selected exactly
wrong.

## Why the verdict is a pair count and not a correlation

The natural instinct is to correlate the proxy against the objective and report
an `r` or a `ρ`. That answers the wrong question.

A sweep is not used to **predict a value**. It is used to **pick a row**. So the
unit of analysis is the pair, and the verdict is a pair count:

```console
$ sweeprank rank det_tune
det_tune:sm_util_mean_pct: [error] disagrees with throughput_img_s on 2 of 15 pairs
(tau_b +0.733); blind at or below 3.39%; worst is det_b24nw16 vs det_b24nw32,
3.39% apart on throughput_img_s
```

Look at that `tau_b`: **+0.733**. Strongly positive. Any correlation-based
summary would call this metric a good stand-in — and it picks the wrong winner.
Thirteen of fifteen pairs agree, and the two that don't are the two that decide
the tuning session.

The counts also keep two things apart that an aggregate would merge:

* **discordant** — the proxy ordered a real difference backwards.
* **proxy_tied** — the proxy could not tell two genuinely different cells apart.

They mean opposite things and both are failures. A single number that averaged
them would let a completely blind proxy through with a respectable score.

## What this repository does not claim

The sentence this analysis started as was *"utilisation ranks the sweep exactly
backwards."* Running it says otherwise. Over four cells the orderings are not
reversed. Over the top three they are not either. The slowest cell is last
under both — the proxy getting something right.

What survives is narrower:

* the proxy ranks the objective's winner **3 of 3** among the live candidates;
* the proxy's own first choice is **not** the objective's first choice.

That is enough to select the wrong configuration and not enough to call the
metric inverted, so it is not called inverted anywhere here. Both negative
claims are asserted in `tests/test_claims.py`:

```python
def test_the_repository_does_not_claim_a_clean_reversal():
    r = analysis.the_reversal()
    assert r["exactly_reversed"] is False
    assert r["top3_exactly_reversed"] is False
    assert r["slowest_cell_last_under_both"] is True
```

A boundary that is not tested gets quietly crossed in an edit. This one fails
the build.
