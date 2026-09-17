# A resolution, not a verdict

The proxy picked the wrong winner. The tempting conclusion is *"never look at
SM utilisation"*, and it is wrong — so there is a control sweep in `data/`
whose only job is to make it fail.

## The control

Two cells, identical in every knob the tuning sweep touched, differing only in
the CPU frequency governor:

| | img/s | SM util % | W | dataloader wait |
|---|---|---|---|---|
| `powersave` | 31.44 | 27.8 | 165.4 | 9.7% |
| `performance` | 70.74 | 61.4 | 279.5 | 6.8% |
| | **+125.0%** | **+120.9%** | | |

Over a change that size the proxy tracks the objective in direction *and* to
within a few points of the same relative size. The metric is not broken. It has
a **resolution**, and a resolution is a number rather than a verdict.

## Putting the number on it

```python
def resolution(sweep, proxy):
    """How big an objective gap has to be before the proxy stops mis-ordering."""
```

Two figures come back:

* `blind_at_or_below` — the largest objective gap on which the proxy got the
  order wrong (proxy ties count here too; failing to resolve a real difference
  is a failure);
* `sound_at_or_above` — the smallest gap **above that** on which it never did.

On the tuning sweep:

```console
$ sweeprank rank det_tune --resolution
det_tune:sm_util_mean_pct: [info] resolution blind<=3.39% sound>=3.62%
```

Below a 3.39% throughput gap it mis-ordered. At or above 3.62% it never did.

**Between 3.39% and 3.62% this data contains no pair at all**, so `resolution`
returns the interval instead of interpolating a threshold across it:

```python
"unresolved_between": [blind, sound],
"note": "the interval between the two is not evidence of anything; this sweep "
        "contains no pair in it",
```

A single threshold would have been more usable and would have been made up.

## Where the sides come from

`cross_sweep_resolution` pools the bound across sweeps and names the sweep that
supplied each side, because a bound whose halves come from different
experiments has to say so. On this data both sides come from `det_tune`, and
the control at +125% sits far **above** the bracket rather than inside it:

```python
assert not analysis.proxy_resolution()["control_is_inside_bracket"]
```

That assertion is doing real work. If the control ever fell inside the bracket,
the bracket would be claiming the proxy is sound at a gap where the control is
the only evidence — and one pair from one sweep is not a bracket.

## Why the resolution is the whole point

The tuning sweep's total objective spread is **4.53%**. Its proxy spread is
**10.34%** — more than twice as wide. That asymmetry is the mechanism: the
proxy moves further than the objective across the same configurations, so its
ordering is dominated by whatever else it responds to.

And the pinned-memory row says what that something else is. Pinning cut the
dataloader wait 87% and raised utilisation 4.9 points while the step got 0.9%
*longer*. The copy work moved into the window the counter samples. **The
utilisation rise was real and none of it was throughput.**

So the boundary is:

> On this hardware and this workload, SM utilisation resolves the
> "did I halve the clock" question and inverts the "which dataloader config"
> question. The line between them is somewhere in 3.39%–4.53%, and this data
> does not say where.

## The malformed question that got caught on the way

The first run of `sweeprank claims` reported the proxy **"blind at or below
513%"**.

That is the eight-card cell being compared with the one-card cell. Total
throughput across eight cards is roughly eight times total throughput across
one, so the "gap" was 513% — arithmetically correct and a meaningless
comparison, because nobody chooses between one card and eight on total
throughput.

The fix is a declaration on the sweep:

```json
"strata": ["ngpu"],
"strata_note": "Total throughput across a different number of cards is not a
  choice between two configurations at the same scale, so pairwise ranking
  questions are asked only within a card count."
```

Pairs only form within a stratum. `det_scale` drops from 10 candidate pairs to
**1** — the four-card same-socket cell against the four-card cross-socket cell,
which is the only pair in that sweep that was ever a choice between two
configurations.

The loader refuses a stratum that does not partition every cell, because a
stratum with a hole in it silently drops comparisons instead of failing:

```python
raise SweepError(f"{sid}: stratum knob {k!r} is missing on {missing}; a stratum "
                 f"that does not partition every cell silently drops comparisons")
```

And the regression has a named test:

```python
def test_the_513_percent_regression_stays_fixed():
    s = bundled("det_scale")
    assert s.strata == ["ngpu"]
    r = resolution(s, "sm_util_mean_pct")
    for key in ("blind_at_or_below", "sound_at_or_above"):
        assert r[key] is None or r[key] < 1.0, (key, r[key])
```

Deleting the strata line makes that test fail. It was verified by doing exactly
that.
