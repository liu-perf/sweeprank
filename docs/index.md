# sweeprank

**A tuning sweep picked a winner. A number somebody was watching picked a
different one.**

A sweep report is a table of floats. Nothing in a table says which column is
the answer and which is a stand-in for the answer, so the goal gets written on
whichever column was easiest to instrument and nothing ever objects.

`sweeprank` makes every metric declare a **role** — `objective`, `proxy`,
`cost`, `diagnostic`, `constraint` — and then checks the stand-ins against the
answer on the sweep's own data.

```console
$ sweeprank goal det_tune
det_tune:sm_util_mean_pct >= 70: [error] rejects det_b24nw16, the fastest cell in the
sweep (throughput_img_s = 86.54, reads 68.7 on the goal metric); it selects
det_b24nw16pin instead, which costs throughput_img_s -0.87%, power_mean_w +7.58%
```

That sweep's own header set out to get SM utilisation above 70%. Applied to the
sweep's own table, the rule throws away the fastest configuration it found.

## The five checks

| command | question |
|---|---|
| `sweeprank rank` | does each proxy order the sweep the way the objective does? |
| `sweeprank goal` | what does the stated goal select, and what does it cost? |
| `sweeprank derived` | does one ratio column have one denominator? |
| `sweeprank exclusions` | did the dropped unit fall on the side its reason claimed? |
| `sweeprank check` | all of the above, over every bundled sweep |

## Reading order

* **[whose-number-picks-the-winner.md](whose-number-picks-the-winner.md)** —
  the role device, why a proxy has to earn selection rights, and why the answer
  is a pair count and not a correlation.
* **[a-resolution-not-a-verdict.md](a-resolution-not-a-verdict.md)** — the
  bracket, the control sweep that stops the finding collapsing into "never look
  at utilisation", and the interval this data cannot close.
* **[what-the-column-cannot-say.md](what-the-column-cannot-say.md)** — two
  denominators in one column, an exclusion that fell the wrong way, and an
  eight-card loss left deliberately unattributed.

## Zero dependencies, no GPU

Producing the four sweeps in `data/` took an eight-card node. Re-checking every
conclusion drawn from them is arithmetic over committed JSON, so CI is pure CPU
and so is anyone reproducing the claims.

Each of the five checks was verified by fault injection, and
`tests/test_ci_runs.py` executes every `run:` block of the workflow under
`bash -e` — because three repositories in this series shipped a CI that was red
on every push while `pytest` and `ruff` stayed green.

## Sister projects

Fifteen tools, each separating two things that looked identical in somebody's
output. `sweeprank` is the sixteenth, and its device (`role`) is the
fourteenth: **"this configuration is better" against "this configuration is
better on the number I was watching."**
