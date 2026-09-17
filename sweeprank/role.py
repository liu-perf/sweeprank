"""The device: every metric in a sweep declares what it is allowed to do.

Fourteenth in a series. The thirteen before it each separated two things that
looked identical in somebody's output. This one separates::

    "this configuration is better"
        from
    "this configuration is better on the number I happened to be watching"

A sweep report is a table. Nothing in a table says which column is the answer
and which column is a stand-in for the answer -- they are both just columns of
floats with a header. So the reader supplies that from memory, and the person
who wrote the sweep supplied it from memory too, and neither of them wrote it
down. A metric with no declared role can therefore be used to pick a winner
whatever it measures, and no artefact of the sweep will object.

The five roles, and what each may do:

``objective``
    The quantity the sweep exists to move. **Exactly one per sweep.** It alone
    may select a winner.

``proxy``
    A stand-in that people do in fact select on -- utilisation, occupancy,
    bandwidth counters. It may select **only after** it has been shown
    concordant with the objective at the scale of the decision being made
    (:mod:`sweeprank.rank`). Until then it may only report.

``cost``
    Must not get worse. It may report, and it may veto, and it may never
    select. Power and peak memory are costs: a configuration that is 1% faster
    and 8% hotter is a trade, and a trade is not a winner.

``diagnostic``
    Explains why the objective moved. It may never select and may never veto.
    ``data_time`` is the clean example: it is how you find out what a knob did,
    and it is never the reason to keep the knob.

``constraint``
    Hard pass/fail. A cell that fails one is out of the sweep, not last in it.

The asymmetry is deliberate. Only ``objective`` gets the unconditional right
to select; everything else has to earn it or is denied it outright. That is the
whole device, and the reason it is worth a file of its own is that the sweep
this repository is built on had its goal written on a ``proxy`` -- and the
proxy ranked the sweep backwards.
"""

from __future__ import annotations

OBJECTIVE = "objective"
PROXY = "proxy"
COST = "cost"
DIAGNOSTIC = "diagnostic"
CONSTRAINT = "constraint"

ROLES = (OBJECTIVE, PROXY, COST, DIAGNOSTIC, CONSTRAINT)

#: Which roles are allowed to name a winner, and under what condition.
MAY_SELECT = {
    OBJECTIVE: "always",
    PROXY: "only once shown concordant with the objective at the decision's scale",
    COST: "never -- may veto",
    DIAGNOSTIC: "never",
    CONSTRAINT: "never -- may exclude",
}

#: Which roles may veto a configuration that the objective ranks first.
MAY_VETO = frozenset({COST, CONSTRAINT})

DESCRIPTIONS = {
    OBJECTIVE: "the quantity the sweep exists to move; exactly one per sweep",
    PROXY: "a stand-in for the objective that people select on anyway",
    COST: "must not get worse; reports and vetoes, never selects",
    DIAGNOSTIC: "explains why the objective moved; never selects, never vetoes",
    CONSTRAINT: "hard pass/fail; a failing cell is out of the sweep, not last in it",
}


class RoleError(Exception):
    """A metric was used for something its role does not permit."""


class UndeclaredRoleError(RoleError):
    """A metric carries no role at all.

    Not a warning. A column with no declared role is exactly the state the
    original sweep was in, and it is the state in which a proxy silently
    becomes the objective.
    """


def check_role(name: str, role: object) -> str:
    if role is None:
        raise UndeclaredRoleError(
            f"metric {name!r} declares no role; one of {', '.join(ROLES)} is required"
        )
    if role not in ROLES:
        raise RoleError(f"metric {name!r} declares unknown role {role!r}; expected one of {ROLES}")
    return str(role)


def assert_may_select(name: str, role: str, *, concordant: bool | None = None) -> None:
    """Raise unless ``role`` is allowed to pick the winner of a sweep.

    ``concordant`` is the verdict from :mod:`sweeprank.rank` for a proxy. It is
    keyword-only and has no default that means "yes" -- a caller that has not
    run the ranking check cannot accidentally pass one.
    """
    if role == OBJECTIVE:
        return
    if role == PROXY:
        if concordant is None:
            raise RoleError(
                f"{name!r} is a proxy; selecting on it requires the concordance check to have "
                f"been run against the objective (sweeprank rank), and it has not been"
            )
        if not concordant:
            raise RoleError(
                f"{name!r} is a proxy and it ranks this sweep differently from the objective; "
                f"it may not select a winner here"
            )
        return
    raise RoleError(
        f"{name!r} has role {role!r} and may never select a winner ({MAY_SELECT[role]})"
    )
