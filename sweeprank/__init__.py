"""sweeprank -- a tuning sweep's winner, and the number somebody was watching.

A sweep report is a table of floats. Nothing in the table says which column is
the answer and which is a stand-in for the answer, so a goal written on a
stand-in can pick the winner and no artefact of the sweep will object.

This package makes the roles explicit (:mod:`sweeprank.role`), then checks the
stand-in against the answer on the sweep's own data (:mod:`sweeprank.rank`),
applies the sweep's own stated goal to its own results (:mod:`sweeprank.goal`),
refuses a ratio column with two denominators (:mod:`sweeprank.baseline`), and
checks that a dropped unit fell on the side its own justification claimed
(:mod:`sweeprank.exclusion`).

Zero dependencies. Everything in ``data/`` is a real measurement and every
conclusion is re-derived from it on import of :mod:`sweeprank.analysis`, so
re-checking the claims needs no GPU.
"""

from .model import Sweep, SweepError, bundled, bundled_names
from .role import ROLES, RoleError

__version__ = "0.1.0"

__all__ = ["ROLES", "RoleError", "Sweep", "SweepError", "__version__", "bundled",
           "bundled_names"]
