"""Loading a sweep, and refusing to load one that cannot be reasoned about.

A ``Sweep`` is a list of configurations, a metric declaration and -- optionally
-- a stated goal, some derived columns and some per-unit exclusions. The loader
is strict on purpose: every failure below is a state the original sweep report
was actually in.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .role import OBJECTIVE, PROXY, RoleError, check_role

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


class SweepError(Exception):
    """The sweep as given cannot be reasoned about."""


@dataclass(frozen=True)
class Metric:
    name: str
    role: str
    higher_is_better: bool
    unit: str = ""
    note: str = ""

    def better(self, a: float, b: float) -> int:
        """1 if ``a`` is better than ``b``, -1 if worse, 0 if identical."""
        if a == b:
            return 0
        if self.higher_is_better:
            return 1 if a > b else -1
        return 1 if a < b else -1


@dataclass(frozen=True)
class Exclusion:
    """A per-unit value dropped from an aggregate, with the direction claimed.

    ``claimed_direction`` is the whole point. "This card's reading is not
    trustworthy" is not a reason to drop it; "this card reads too high, so the
    mean is inflated" is. The second one is checkable and the first one is not,
    which is why this field is required.
    """

    unit: str
    metric: str
    reason: str
    claimed_direction: str  # "high" | "low" | "unknown"
    reported_aggregate_without: float | None = None
    also_measured: dict = field(default_factory=dict)


@dataclass
class Config:
    id: str
    knobs: dict
    values: dict
    failed: str | None = None
    per_unit: dict = field(default_factory=dict)
    exclusions: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed is None

    def units(self, metric: str) -> dict:
        return dict(self.per_unit.get(metric, {}))


@dataclass
class Derived:
    """A column computed from another column -- a speedup, an efficiency.

    ``baselines`` maps a baseline identifier to the rows divided by it. More
    than one key means more than one baseline in a single column, which is the
    defect :mod:`sweeprank.baseline` exists to refuse.
    """

    name: str
    formula: str
    of: str
    baselines: dict = field(default_factory=dict)
    reported: dict = field(default_factory=dict)
    units_knob: str | None = None
    note: str = ""


@dataclass
class Goal:
    metric: str
    op: str
    value: float
    quoted_from: str = ""

    def accepts(self, x: float) -> bool:
        if self.op == ">=":
            return x >= self.value
        if self.op == ">":
            return x > self.value
        if self.op == "<=":
            return x <= self.value
        if self.op == "<":
            return x < self.value
        raise SweepError(f"unknown goal operator {self.op!r}")


@dataclass
class Sweep:
    sweep_id: str
    metrics: dict
    configs: list
    what: str = ""
    provenance: dict = field(default_factory=dict)
    goal: Goal | None = None
    derived: list = field(default_factory=list)
    strata: list = field(default_factory=list)

    def stratum(self, config: Config) -> tuple:
        """The knob values that have to match before two cells may be compared.

        A scaling sweep measures total throughput, so its eight-card cell beats
        its one-card cell by about eight times -- and that gap says nothing
        about any metric's ability to rank a *choice*, because nobody chooses
        between one card and eight on throughput alone. Declaring
        ``strata: ["ngpu"]`` restricts every pairwise question in
        :mod:`sweeprank.rank` to cells at the same scale.

        This was not in the first version. It went in because the first run of
        ``sweeprank claims`` reported a proxy "blind at or below 513%", which
        is the eight-card cell being compared with the one-card cell. The
        number was arithmetically right and the question was malformed.
        """
        return tuple(config.knobs.get(k) for k in self.strata)

    # -- accessors ---------------------------------------------------------

    @property
    def objective(self) -> Metric:
        for m in self.metrics.values():
            if m.role == OBJECTIVE:
                return m
        raise SweepError(f"{self.sweep_id}: no metric declares role {OBJECTIVE!r}")

    @property
    def proxies(self) -> list:
        return [m for m in self.metrics.values() if m.role == PROXY]

    @property
    def cells(self) -> list:
        """Configurations that produced a number. A failed cell is out, not last."""
        return [c for c in self.configs if c.ok]

    @property
    def failed(self) -> list:
        return [c for c in self.configs if not c.ok]

    def by_id(self, cid: str) -> Config:
        for c in self.configs:
            if c.id == cid:
                return c
        raise SweepError(f"{self.sweep_id}: no configuration {cid!r}")

    def value(self, config: Config, metric: str) -> float:
        if metric not in config.values:
            raise SweepError(f"{self.sweep_id}:{config.id}: no value for metric {metric!r}")
        return float(config.values[metric])

    def repeats(self) -> int:
        return int(self.provenance.get("repeats", 0))

    def winner(self, metric: str | None = None) -> Config:
        """The best cell by ``metric`` -- the objective unless told otherwise.

        Selecting on anything but the objective goes through
        :func:`sweeprank.role.assert_may_select`, so a caller cannot reach a
        proxy-selected winner without having run the concordance check.
        """
        m = self.metrics[metric] if metric else self.objective
        if m.role != OBJECTIVE:
            from .rank import concordance
            from .role import assert_may_select

            verdict = concordance(self, m.name)
            assert_may_select(m.name, m.role, concordant=verdict["concordant"])
        cells = self.cells
        if not cells:
            raise SweepError(f"{self.sweep_id}: every configuration failed; there is no winner")
        best = cells[0]
        for c in cells[1:]:
            if m.better(self.value(c, m.name), self.value(best, m.name)) > 0:
                best = c
        return best

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict, *, sweep_id: str | None = None) -> Sweep:
        sid = raw.get("sweep_id") or sweep_id or "<anonymous>"
        decl = raw.get("metrics")
        if not decl:
            raise SweepError(f"{sid}: no metric declaration; a table of floats is not a sweep")

        metrics = {}
        for name, d in decl.items():
            role = check_role(name, d.get("role"))
            if "higher_is_better" not in d:
                raise SweepError(
                    f"{sid}: metric {name!r} does not say which direction is better"
                )
            metrics[name] = Metric(
                name=name,
                role=role,
                higher_is_better=bool(d["higher_is_better"]),
                unit=d.get("unit", ""),
                note=d.get("note", ""),
            )

        n_obj = sum(1 for m in metrics.values() if m.role == OBJECTIVE)
        if n_obj != 1:
            raise SweepError(
                f"{sid}: {n_obj} metrics declare role {OBJECTIVE!r}; exactly one is required"
            )

        configs = []
        for c in raw.get("configs", []):
            if "id" not in c:
                raise SweepError(f"{sid}: a configuration has no id")
            excl = [
                Exclusion(
                    unit=str(e["unit"]),
                    metric=e["metric"],
                    reason=e.get("reason", ""),
                    claimed_direction=e.get("claimed_direction", "unknown"),
                    reported_aggregate_without=e.get("reported_aggregate_without"),
                    also_measured=e.get("also_measured", {}),
                )
                for e in c.get("exclusions", [])
            ]
            cfg = Config(
                id=c["id"],
                knobs=c.get("knobs", {}),
                values=c.get("values", {}) or {},
                failed=c.get("failed"),
                per_unit=c.get("per_unit", {}),
                exclusions=excl,
            )
            if cfg.ok:
                unknown = set(cfg.values) - set(metrics)
                if unknown:
                    raise SweepError(
                        f"{sid}:{cfg.id}: values for undeclared metrics {sorted(unknown)}"
                    )
                if not cfg.values:
                    raise SweepError(
                        f"{sid}:{cfg.id}: no values and no `failed` reason -- a cell that "
                        f"produced nothing has to say why"
                    )
            configs.append(cfg)

        if not configs:
            raise SweepError(f"{sid}: no configurations")

        seen = set()
        for c in configs:
            if c.id in seen:
                raise SweepError(f"{sid}: duplicate configuration id {c.id!r}")
            seen.add(c.id)

        goal = None
        if "declared_goal" in raw:
            g = raw["declared_goal"]
            if g["metric"] not in metrics:
                raise SweepError(f"{sid}: the goal names undeclared metric {g['metric']!r}")
            goal = Goal(
                metric=g["metric"], op=g["op"], value=float(g["value"]),
                quoted_from=g.get("quoted_from", ""),
            )

        derived = [
            Derived(
                name=d["name"], formula=d["formula"], of=d["of"],
                baselines=d.get("baselines", {}), reported=d.get("reported", {}),
                units_knob=d.get("units_knob"), note=d.get("note", ""),
            )
            for d in raw.get("derived", [])
        ]

        strata = list(raw.get("strata", []))
        for k in strata:
            missing = [c.id for c in configs if c.ok and k not in c.knobs]
            if missing:
                raise SweepError(
                    f"{sid}: stratum knob {k!r} is missing on {missing}; a stratum that "
                    f"does not partition every cell silently drops comparisons"
                )

        return cls(
            sweep_id=sid, metrics=metrics, configs=configs,
            what=raw.get("what", ""), provenance=raw.get("provenance", {}),
            goal=goal, derived=derived, strata=strata,
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Sweep:
        p = pathlib.Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), sweep_id=p.stem)


def bundled(name: str) -> Sweep:
    """Load one of the sweeps committed in ``data/``."""
    return Sweep.load(DATA_DIR / f"{name}.json")


def bundled_names() -> list:
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))


__all__ = [
    "Config", "Derived", "Exclusion", "Goal", "Metric", "RoleError", "Sweep",
    "SweepError", "bundled", "bundled_names",
]
