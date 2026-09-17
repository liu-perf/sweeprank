# What the column cannot say

Three findings from the scaling half of the data. Two are defects in the source
report; the third is a result that this data deliberately does not close.

## One column, two denominators

The source table reports speedup and scaling efficiency for eight rows. Every
published ratio reproduces exactly:

```console
$ python -c "from sweeprank import baseline; from sweeprank.model import bundled, bundled_names; \
all_={n:bundled(n) for n in bundled_names()}; \
rows=baseline.recompute(all_['det_scale'], others=all_); \
print(all(r['agrees'] for r in rows if r['agrees'] is not None))"
True
```

Nothing is mis-transcribed. **That is the difficulty** — no arithmetic check
finds anything, and the column is still not a quantity:

```console
$ sweeprank derived det_scale
det_scale:speedup: [error] column 'speedup' divides its rows by 2 different baselines
(det_opt_1gpu, det_tune:det_1gpu); read down the column the cells are not comparable
```

The four single-card rows are divided by an **untuned** single-card run
(batch 8). The four multi-card rows are divided by a **tuned** one
(batch 24 / workers 32 / pin). Both halves are individually correct and read
down the column they are not comparable.

The efficiency column, derived by dividing speedup by the card count, makes it
visible:

| row | cards | speedup | efficiency |
|---|---|---|---|
| single, bs8 | 1 | 1.00× | 100.0% |
| single, bs24 nw8 | 1 | 1.17× | **117.0%** |
| single, bs24 nw32 | 1 | 1.18× | **118.3%** |
| single, bs24 nw32 pin | 1 | 1.18× | **118.4%** |
| 2 cards | 2 | 1.98× | 98.9% |
| 4 cards, one socket | 4 | 3.72× | 93.1% |
| 4 cards, two sockets | 4 | 3.91× | 97.9% |
| 8 cards | 8 | 6.13× | 76.6% |

`118.4%` scaling efficiency on one GPU. A reader comparing that with the
eight-card row's `76.6%` concludes that one card scales better than eight, and
the table gave them no way to know otherwise.

```python
def check_efficiency_domain(sweep):
    """A per-unit efficiency printed on a one-unit row is not an efficiency."""
```

The check refuses the three rows above 100% and **does not** flag the baseline
row at exactly 100.0%: dividing the baseline by itself is not the defect.

## An exclusion that fell on the wrong side of its own reason

The eight-card cell reports a utilisation average, and then reports a second
one with card 7 removed. The stated reason: a CUDA context left behind by an
earlier profiling run had pinned that card's `utilization.gpu` at 100% with
0 MiB allocated.

That is a mechanism that **inflates**. Here are the eight per-card readings:

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** |
|---|---|---|---|---|---|---|---|---|
| SM util % | 60.9 | 64.9 | 68.5 | 70.2 | 65.6 | 67.3 | 65.3 | **63.5** |

```console
$ sweeprank exclusions det_scale
det_scale:det_opt_8gpu:unit7: [error] the exclusion of unit 7 is justified by a
mechanism that reads high, but sm_util_mean_pct = 63.5 on that unit is low relative
to the other 7 (mean 66.1); dropping it moved the aggregate +0.32, the opposite way
from the stated reason
```

Three things, and each is checkable because the exclusion was made to declare a
**direction**:

**Direction.** 63.5 is below the mean of the other seven (66.10). Dropping it
moved the aggregate from 65.8 up to 66.1 — the opposite way from what the
stated mechanism implies. `claimed_direction` is a required field on the
`Exclusion` dataclass for exactly this reason: *"this card's reading is not
trustworthy"* is unfalsifiable, and *"this card reads too high"* is a claim
about a side, and the data has a side.

**Rank.** Card 7 is **4th of 8** by distance from the mean. Cards 0 (60.9) and
3 (70.2) are both further out. Whatever it was, it was not this cell's outlier.

**Scope.** The exclusion was applied to one column of five:

```console
det_scale:det_opt_8gpu:unit7: [warn] the exclusion was applied to sm_util_mean_pct
only; 4 other metric(s) on the same cell were left uncorrected (power_mean_w,
sm_zero_frac_pct, step_time_s, throughput_img_s)
```

If the reason were true of the card it would be true of the whole cell. The
headline throughput was left alone, and the correction was applied where it
moved a third of a point.

**And then the size bound closes it the other way.** The same run measured a
dense GEMM on that card at 218.2 TFLOPS against 223.5 on card 0 — a 2.4%
deficit. Synchronous data-parallel steps run at the pace of the slowest rank,
so a 2.4% slower card bounds its own contribution at about 2.4%. The observed
eight-card loss is **23.4%**:

```console
bounds 10.3% of the observed eight-card loss
```

So: the contaminant was flagged in the right place, corrected in the one column
where it was negligible, left in the column that mattered — and once bounded,
too small to have mattered to either. An uncorrected number and an unbounded
correction, in the same cell.

## An eight-card loss, left unattributed

| | speedup | of ideal | idle samples 1→8 | SM util 1→8 |
|---|---|---|---|---|
| detection | 6.13× | 76.6% | 0.0% → **6.0%** | 74.6 → 65.8 |
| segmentation | 6.25× | 78.1% | 0.0% → **0.1%** | 81.8 → **82.1** |

Idle fraction is the only diagnostic in this data that grows with card count,
and on detection it grows a long way — from nothing to 6% of samples, on all
eight cards (2.2% to 8.8%), which is the signature of a shared resource rather
than one bad card.

Two things stop it becoming the answer:

1. **It is too small.** Step time grew 30.5% from one card to eight. A 6.0%
   idle fraction accounts for at most a fifth of that.
2. **The other workload contradicts it.** Segmentation loses a comparable share
   of ideal — 21.9% against 23.4% — with **sixty times less** measured idle.

So the diagnostic does not generalise, nothing here identifies the mechanism,
and the output says so:

```python
out["verdict"] = "unattributed"
out["why"] = ("idle fraction is the only diagnostic in this data that grows with "
              "card count, it can account for at most a fifth of the detection "
              "step-time growth, and it is absent in the segmentation sweep which "
              "loses about as much; nothing here identifies the mechanism and no "
              "sentence in this repository claims one")
```

A CI gate keeps the verdict and the counterexample in place, so a future edit
that guesses a mechanism fails the build.

Worth one more look at that last column. **Segmentation lost a fifth of its
cards' worth of throughput while its utilisation went up.** The tuning sweep
showed the proxy mis-ordering a 3% question; this shows it moving the wrong way
across a 22% one.

## One result the data does support

Four cards on two sockets beat four cards on one, on both workloads, on
throughput and on power:

| | same socket | two sockets | throughput | power |
|---|---|---|---|---|
| detection | 311.84 img/s @ 345.4 W | 327.89 @ 329.0 W | **+5.15%** | −4.75% |
| segmentation | 38.74 img/s @ 342.7 W | 39.96 @ 325.7 W | **+3.15%** | −4.96% |

Direction replicated on two models that share only the node and the harness. A
plausible mechanism exists — each socket's memory channels are half-populated,
so spreading four ranks over two sockets doubles the populated channels behind
them.

The limits are attached to the result rather than left to the prose:

```python
out["what_the_data_cannot_do"] = (
    "one run per cell, so the 3-5% magnitude has no error bar; and the mechanism "
    "above is consistent with the numbers, not shown by them -- no channel-population "
    "arm was run, so a per-socket cache or interrupt-affinity effect is not excluded"
)
```

Every cell in every sweep here is one run. `sweeprank rank` emits that as a
`warn` on all four sweeps, `tests/test_model.py` asserts every sweep declares
its repeat count, and `--fail-on warn` makes it fail a build. It is the binding
limit on everything above.
