# sweeprank

**A tuning sweep picked a winner. A number somebody was watching picked a
different one.** This tool makes every metric in a sweep declare whether it is
allowed to choose, and then checks the stand-ins against the answer on the
sweep's own data.

Zero dependencies, pure standard library, CPU-only. The four sweeps in `data/`
are real measurements from an eight-card RTX 5090 node; re-deriving every
conclusion below is arithmetic over committed JSON and needs no GPU.

```console
$ pip install -e .
$ sweeprank goal det_tune
det_tune:sm_util_mean_pct >= 70: [warn] written on a proxy, not on the objective (throughput_img_s)
det_tune:sm_util_mean_pct >= 70: [error] rejects det_b24nw16, the fastest cell in the
sweep (throughput_img_s = 86.54, reads 68.7 on the goal metric); it selects
det_b24nw16pin instead, which costs throughput_img_s -0.87%, power_mean_w +7.58%,
mem_peak_mib -2.34%
```

That sweep's own header line set out to get SM utilisation above 70%. Applied
to the sweep's own table, that rule **throws away the fastest configuration it
found** and replaces it with one that is slower and draws 25 W more.

---

## Why a sweep report cannot defend itself

A sweep report is a table of floats. Nothing in a table says which column is
the answer and which column is a stand-in for the answer — they are both just
numbers with a header. So the goal gets written on whichever column was easiest
to instrument, the table gets written at the bottom, and the two are never put
in the same room.

`sweeprank` puts them in the same room. Every metric declares a **role**:

| role | may it pick the winner? |
|---|---|
| `objective` | always — exactly one per sweep |
| `proxy` | only after it is shown to order this sweep the way the objective does |
| `cost` | never; it may veto |
| `diagnostic` | never; it explains |
| `constraint` | never; a failing cell is *out* of the sweep, not last in it |

The asymmetry is the whole device. Only the objective gets the unconditional
right to select. A proxy has to earn it, and the check that grants it is
keyword-only with no truthy default, so a caller who never ran it cannot get
past the gate by omission:

```python
>>> from sweeprank.role import assert_may_select, PROXY
>>> assert_may_select("sm_util_mean_pct", PROXY)
RoleError: ... requires the concordance check to have been run ... and it has not been
```

---

## What came back

Five checks, run over four real sweeps. Every line `sweeprank check` prints is
a defect the source report shipped with.

### 1. The proxy ranks the winner last of the three candidates

Four cells with batch fixed at 24, so the only things moving are worker count
and pinned memory:

| cell | workers | pin | **img/s** | SM util % | W | dataloader wait |
|---|---|---|---|---|---|---|
| `det_b24nw16` | 16 | no | **86.54** | 68.7 | 329.8 | 6.3% |
| `det_b24nw16pin` | 16 | **yes** | 85.79 | **73.6** | **354.8** | **0.8%** |
| `det_b24nw32` | 32 | no | 83.70 | 70.1 | 335.1 | 6.1% |
| `det_b24` | 8 | no | 82.79 | 66.7 | 334.5 | 6.5% |

The fastest cell has the **lowest** utilisation of the three live candidates,
and the proxy's own first pick is the objective's second.

**It is not a clean reversal, and this repository does not say it is.** Over
four cells the two orderings are not reversed; over the top three they are not
either; the slowest cell is last under both. That is the proxy getting
something right. What is true is narrower and quite enough to lose a tuning
session: the proxy ranks the objective's winner **3 of 3**, and would select
something else. `tests/test_claims.py` asserts the negative claim too, so an
edit that restores the stronger sentence fails the build.

Note also what `tau_b` does here. It is **positive** and the proxy still picks
the wrong winner — which is why this tool counts pairs instead of reporting a
correlation. A sweep is not used to predict a value, it is used to pick a row.

### 2. Pinned memory did exactly what it is for, and the step got no faster

The two cells above differ in one flag:

| | without pin | with pin | |
|---|---|---|---|
| dataloader wait | 6.3% of step | 0.8% | **−87.3%** |
| step time | 0.2773 s | 0.2798 s | **+0.90%** |
| throughput | 86.54 | 85.79 | **−0.87%** |
| SM utilisation | 68.7 | 73.6 | +7.1% |
| power | 329.8 W | 354.8 W | +7.6% |

The knob's own target improved by an order of magnitude. The objective moved
the wrong way. **Utilisation rose 4.9 points and none of it was throughput.**

### 3. A resolution, not a verdict

The honest conclusion is not "never look at utilisation" — and there is a
control sweep in `data/` whose only job is to stop it becoming that. Two cells
identical in every knob the tuning sweep touched, differing only in the CPU
frequency governor:

| | img/s | SM util % |
|---|---|---|
| `powersave` | 31.44 | 27.8 |
| `performance` | 70.74 | 61.4 |
| | **+125.0%** | **+120.9%** |

Over a change that size the proxy tracks the objective in direction and to
within a few points of the same relative size. So the metric has a
**resolution**, and a resolution is a number:

```console
$ sweeprank rank det_tune --resolution
det_tune:sm_util_mean_pct: [info] resolution blind<=3.39% sound>=3.62%
```

Below a 3.39% throughput gap it got the order wrong; at or above 3.62% it never
did. **Between those two figures this data contains no pair at all**, and
`resolution()` reports the interval rather than interpolating across it.

### 4. One column, two denominators

```console
$ sweeprank derived det_scale
det_scale:speedup: [error] column 'speedup' divides its rows by 2 different baselines
(det_opt_1gpu, det_tune:det_1gpu); read down the column the cells are not comparable
det_scale:efficiency: [error] efficiency = 118.4% on row 'single_bs24nw32pin'; dividing
a speedup by one unit does not make it a scaling efficiency, and a scaling efficiency
above 100% is the tell
```

The source table's speedup column divides its four single-card rows by an
untuned single-card run and its four multi-card rows by a *tuned* one. Both
halves are individually correct — **every published ratio reproduces exactly
from the objective**, which is the difficulty: no arithmetic check finds
anything, and the column is still not a quantity. The efficiency column derived
from it prints `118.4%` scaling efficiency on one GPU, and a reader comparing
that with the eight-card row's `76.6%` concludes that one card scales better
than eight.

### 5. An exclusion that fell on the wrong side of its own reason

One card in the eight-card cell was dropped from the utilisation average: a
CUDA context left over from an earlier profiling run had pinned its
`utilization.gpu` at 100%. That is a mechanism that inflates.

```console
$ sweeprank exclusions det_scale
det_scale:det_opt_8gpu:unit7: [error] the exclusion of unit 7 is justified by a mechanism
that reads high, but sm_util_mean_pct = 63.5 on that unit is low relative to the other 7
(mean 66.1); dropping it moved the aggregate +0.32, the opposite way from the stated reason
det_scale:det_opt_8gpu:unit7: [warn] the exclusion was applied to sm_util_mean_pct only;
4 other metric(s) on the same cell were left uncorrected (power_mean_w, sm_zero_frac_pct,
step_time_s, throughput_img_s)
```

Three separate things, all checkable:

* **Direction.** The card said to read high reads *below* the mean of the other
  seven. Dropping it moved the aggregate up, +0.32 points.
* **Rank.** It is 4th of 8 by distance from the mean. It was not the outlier.
* **Scope.** The correction was applied to one column of five. The headline
  throughput was left alone — and taken at face value the flagged card's
  measured 2.4% compute deficit bounds a synchronous-DDP straggler effect at
  about **a tenth of the 23.4% loss actually observed**.

So the contaminant was flagged in the right place, corrected in the one column
where it moved a third of a point, and once bounded turned out to be too small
to have mattered to either.

---

## Two findings this data supports, and the limit on each

**Cross-socket beat same-socket, on both workloads and on power too.** Four
cards on two sockets against four on one: `+5.15%` throughput on detection,
`+3.15%` on segmentation, and `−4.8%` / `−5.0%` power. Direction replicated on
two models that share only the node and the harness.

There is a plausible mechanism — each socket's memory channels are
half-populated, so spreading four ranks over two sockets doubles the populated
channels behind them. **The data is consistent with that and does not show
it**: no channel-population arm was run, so a per-socket cache or
interrupt-affinity effect is not excluded. And every cell in every sweep here is
**one run**, so the 3–5% magnitude has no error bar. `sweeprank rank` emits that
as a `warn` on every sweep rather than leaving it to the prose.

**The eight-card loss is left unattributed, and that is the output.**

| | speedup | of ideal | idle samples | SM util 1→8 cards |
|---|---|---|---|---|
| detection | 6.13× | 76.6% | 6.0% | 74.6 → 65.8 |
| segmentation | 6.25× | 78.1% | **0.1%** | 81.8 → **82.1** |

Idle fraction is the only diagnostic here that grows with card count, it can
account for at most a fifth of the detection step-time growth, and it is
**sixty times smaller in the segmentation sweep, which loses about as much**. So
the diagnostic does not generalise and nothing in this data identifies the
mechanism. `eight_card_attribution()["verdict"]` is the string
`"unattributed"`, and a CI gate keeps it that way.

Look at the last column while you are here: segmentation lost a fifth of its
cards' worth of throughput while its utilisation went **up**.

---

## Using it on your own sweep

```console
$ sweeprank roles                      # what each role may do
$ sweeprank show my_sweep.json         # the declaration and the cells
$ sweeprank rank my_sweep.json         # does each proxy order it like the objective?
$ sweeprank goal my_sweep.json         # apply the stated goal to the results
$ sweeprank derived my_sweep.json      # one column, one baseline
$ sweeprank exclusions my_sweep.json   # did the dropped unit fall on the claimed side?
$ sweeprank check                      # all of the above over the bundled sweeps
$ sweeprank claims                     # this repository's findings, recomputed
```

`--json` prints the structure. `--fail-on error|warn` sets the exit code, which
is how these become build gates. `sweeprank check` is deliberately **loud and
exits 0**: every line it prints is a defect in the committed source data, so a
non-zero exit would mean this repository never builds. The gates that must stay
green live in `tests/`.

The loader is strict, and each refusal below is a state a real report was in:
a table with no metric declaration; a metric that does not say which direction
is better; a cell with no values and no reason; two metrics claiming
`objective`; a goal naming an undeclared metric; a stratum knob that does not
partition every cell.

### Strata

A scaling sweep measures *total* throughput, so its eight-card cell beats its
one-card cell by roughly eight times — and that gap says nothing about any
metric's ability to rank a **choice**, because nobody picks between one card and
eight on total throughput. Declaring `"strata": ["ngpu"]` confines every
pairwise question to one scale.

This was not in the first version. The first run of `sweeprank claims` reported
the proxy "blind at or below 513%", which is the eight-card cell being compared
with the one-card cell. The arithmetic was right and the question was malformed.
`tests/test_rank.py::test_the_513_percent_regression_stays_fixed` keeps it
fixed.

---

## What is in `data/`

| sweep | cells | what it is for |
|---|---|---|
| `det_tune.json` | 7 (one OOM) | single-card dataloader tuning; carries the stated goal |
| `det_scale.json` | 5 | 1/2/4/8 cards; carries the derived columns and the exclusion |
| `seg_scale.json` | 5 | the same scaling questions on a different model |
| `governor_control.json` | 2 | the control that gives the proxy a resolution instead of a verdict |

Every number was transcribed from a logged run: 240 s (detection) or 360 s
(segmentation) of steady state per cell, first 25% of steps discarded as warmup,
throughput taken from the training log rather than from a counter. Card models
and measurements are published; the machine's identity is not, and
`.gitignore` keeps other people's exported sweeps out.

**One run per cell.** Stated in every sweep's `provenance`, asserted by
`tests/test_model.py`, and emitted as a `warn` by `sweeprank rank`. It is the
binding limit on everything above and the reason `resolution()` reports a
bracket rather than a threshold.

## Tests

115, pure CPU, no dependencies beyond `pytest`. Each of the five checks was
verified by fault injection — flip the exclusion's claimed direction, mistype a
published speedup, delete a role, raise the winner's utilisation past the goal,
drop the strata declaration — and each makes a named test fail.

`tests/test_ci_runs.py` executes every `run:` block in `.github/workflows/ci.yml`
under `bash -e`. Three repositories in this series shipped a CI that was red on
every push while `pytest` and `ruff` stayed green, because the defect was in the
YAML and every test was in Python. It runs the workflow text **verbatim** via
PATH shims; the first version rewrote it with a regex, `\bsweeprank\b` matched
`sweeprank.model` inside the heredocs, and nine steps failed with a
`SyntaxError` that had nothing to do with the workflow.

## Licence

MIT.
