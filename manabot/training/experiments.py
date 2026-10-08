"""Declare experiments and resolve complete regimes without executing work.

Baseline freezes configuration bytes. Experiment combines typed component writes,
explicit cases and ordered matrix axes. Each resolved cell retains leaf-level
ownership and provenance separately from the unchanged TrainingRegime digest.
The existing study protocol and executor own cohorts, allocation and execution.
"""

from dataclasses import dataclass
from itertools import product
import json
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from manabot.arena.models import canonical_json, canonical_sha256
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.remote.job_client import prepare_experiment_job
from manabot.remote.jobs import Job
from manabot.remote.plan import DeploymentPlan, Source, compile_plan
from manabot.training.execution import validate_regime
from manabot.training.experiment_execution import ExperimentSchedule, PlannedRun
from manabot.training.models import (
    AtaraxosMoveLearning,
    Execution,
    Learning,
    Operation,
    TrainingRegime,
    TrainSelfPlay,
)

if TYPE_CHECKING:
    from manabot.remote.cohort import Cohort

ComponentName = Literal[
    "model", "environment", "learning", "resources", "pipeline", "run", "identity"
]
Origin = Literal["baseline", "preset", "override", "identity"]
Path = tuple[str, ...]
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_JSON_VALUE = TypeAdapter(JsonValue)
_ROOT_OWNERS: dict[str, ComponentName] = {
    "schema_version": "identity",
    "id": "identity",
    "world": "environment",
    "match": "environment",
    "observation": "environment",
    "agent": "model",
    "stages": "pipeline",
    "wall_seconds": "resources",
    "schedule_clock": "run",
    "recovery_max_microsteps": "run",
    "selection": "run",
}


def setting_owner(path: Path) -> ComponentName:
    """Every property has one owner; unknown schema roots fail closed."""
    if path == ("observation", "policy_history_version"):
        return "model"
    if path[0] == "stages" and len(path) > 2:
        if path[2] == "learning":
            return "learning"
        if path[2] == "execution":
            return "resources"
    return _ROOT_OWNERS[path[0]]


def _object(model: BaseModel, *, explicit: bool = False) -> dict[str, JsonValue]:
    # Pydantic's JSON boundary is narrowed once; internal writes carry JSON values.
    return _JSON_OBJECT.validate_python(
        model.model_dump(mode="json", exclude_unset=explicit)
    )


def _slug(value: str) -> None:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise ValueError(f"expected a nonempty hyphenated name: {value!r}")


@dataclass(frozen=True)
class Baseline:
    """Immutable complete snapshot; callers receive fresh validated configurations.

    Capture defaults now, rather than inheriting future default changes. A schema
    change that alters the serialized snapshot fails instead of upgrading it.
    """

    name: str
    configuration: str
    origin: Literal["baseline", "preset"] = "baseline"

    @classmethod
    def capture(cls, name: str, regime: TrainingRegime) -> "Baseline":
        _slug(name)
        validated = TrainingRegime.model_validate(regime.model_dump())
        return cls(name, canonical_json(_object(validated)).decode())

    @property
    def digest(self) -> str:
        return canonical_sha256(json.loads(self.configuration))

    def regime(self) -> TrainingRegime:
        _slug(self.name)
        regime = TrainingRegime.model_validate_json(self.configuration)
        if _object(regime) != json.loads(self.configuration):
            raise ValueError(
                "baseline snapshot is incomplete or its schema meaning changed"
            )
        return regime


@dataclass(frozen=True)
class Model:
    """Only explicitly supplied AgentSpec fields change; no implicit defaults."""

    settings: AgentSpec


@dataclass(frozen=True)
class Environment:
    world: str | None = None
    match: MatchHypers | None = None
    observation: ObservationSpaceHypers | None = None


@dataclass(frozen=True)
class LearningRule:
    """Replace the complete rule on all self-play stages, including its defaults."""

    settings: Learning | AtaraxosMoveLearning


@dataclass(frozen=True)
class Resources:
    """Patch explicit execution fields on every stage and optionally the run cap."""

    execution: Execution | None = None
    wall_seconds: float | None = None


@dataclass(frozen=True)
class Pipeline:
    """Replace the whole ordered pipeline; overlaps any other stage write."""

    stages: tuple[Operation, ...]


@dataclass(frozen=True)
class RunControl:
    schedule_clock: Literal["run_elapsed_budget", "iteration_fraction"]
    recovery_max_microsteps: int | None = None
    selection: Literal["last-complete-raw"] = "last-complete-raw"


Component = Model | Environment | LearningRule | Resources | Pipeline | RunControl


@dataclass(frozen=True)
class Case:
    name: str
    overrides: tuple[Component, ...] = ()
    label: str = ""


@dataclass(frozen=True)
class Axis:
    name: str
    choices: tuple[Case, ...]


@dataclass(frozen=True)
class SettingProvenance:
    path: Path
    component: ComponentName
    origin: Origin
    source: str
    value_json: str

    @property
    def value(self) -> JsonValue:
        return _JSON_VALUE.validate_json(self.value_json)


@dataclass(frozen=True)
class ResolvedCase:
    name: str
    label: str
    configuration: str
    baseline_name: str
    baseline_digest: str
    provenance: tuple[SettingProvenance, ...]

    @property
    def regime(self) -> TrainingRegime:
        return TrainingRegime.model_validate_json(self.configuration)

    @property
    def digest(self) -> str:
        """Existing executor identity, unaffected by authoring provenance."""
        return canonical_sha256(json.loads(self.configuration))

    def receipt(self) -> dict[str, JsonValue]:
        return {
            "schema_version": 1,
            "name": self.name,
            "label": self.label,
            "baseline": self.baseline_name,
            "baseline_digest": self.baseline_digest,
            "regime_digest": self.digest,
            "regime": json.loads(self.configuration),
            "settings": [
                {
                    "path": list(p.path),
                    "component": p.component,
                    "origin": p.origin,
                    "source": p.source,
                    "value": p.value,
                }
                for p in self.provenance
            ],
        }

    @property
    def identity(self) -> str:
        return canonical_sha256(self.receipt())


@dataclass(frozen=True)
class ResolvedExperiment:
    name: str
    cases: tuple[ResolvedCase, ...]

    @property
    def regimes(self) -> dict[str, TrainingRegime]:
        return {case.name: case.regime for case in self.cases}

    @property
    def digests(self) -> tuple[str, ...]:
        return tuple(case.digest for case in self.cases)

    def receipt(self) -> dict[str, JsonValue]:
        return {
            "schema_version": 1,
            "name": self.name,
            "cases": [case.receipt() for case in self.cases],
        }

    @property
    def identity(self) -> str:
        return canonical_sha256(self.receipt())


@dataclass(frozen=True)
class _Write:
    path: Path
    value: JsonValue
    source: str


def _fields(prefix: Path, model: BaseModel, source: str) -> list[_Write]:
    values = _object(model, explicit=True)
    # Compatibility defaults may be omitted from saved files, but an explicit
    # False remains an authored override (including switching a treatment off).
    if isinstance(model, AgentSpec):
        for field in ("compound_decisions", "recent_events"):
            if field in model.model_fields_set:
                values[field] = getattr(model, field)
    return [_Write((*prefix, key), value, source) for key, value in values.items()]


def _writes(component: Component, base: TrainingRegime, source: str) -> list[_Write]:
    match component:
        case Model(settings):
            writes = _fields(("agent",), settings, source)
            if "recent_events" in settings.model_fields_set:
                writes.append(
                    _Write(
                        ("observation", "policy_history_version"),
                        int(settings.recent_events),
                        source,
                    )
                )
            return writes
        case Environment(world, match, observation):
            writes = [] if world is None else [_Write(("world",), world, source)]
            if match is not None:
                writes.extend(_fields(("match",), match, source))
            if observation is not None:
                if "policy_history_version" in observation.model_fields_set:
                    raise ValueError(
                        "history input is selected by Model(recent_events=...), not Environment"
                    )
                writes.extend(_fields(("observation",), observation, source))
            return writes
        case LearningRule(settings):
            if not all(isinstance(stage, TrainSelfPlay) for stage in base.stages):
                raise ValueError(
                    "LearningRule requires only self-play stages; use Pipeline for other operations"
                )
            return [
                _Write(("stages", str(i), "learning"), _object(settings), source)
                for i in range(len(base.stages))
            ]
        case Resources(execution, wall_seconds):
            writes = (
                []
                if wall_seconds is None
                else [_Write(("wall_seconds",), wall_seconds, source)]
            )
            if execution is not None:
                for i in range(len(base.stages)):
                    writes.extend(
                        _fields(("stages", str(i), "execution"), execution, source)
                    )
            return writes
        case Pipeline(stages):
            return [_Write(("stages",), [_object(stage) for stage in stages], source)]
        case RunControl(clock, recovery, selection):
            return [
                _Write((key,), value, source)
                for key, value in (
                    ("schedule_clock", clock),
                    ("recovery_max_microsteps", recovery),
                    ("selection", selection),
                )
            ]
    raise TypeError(f"unsupported component: {type(component).__name__}")


def _leaves(value: JsonValue, path: Path = ()) -> list[tuple[Path, JsonValue]]:
    if isinstance(value, dict) and value:
        return [
            leaf for key in sorted(value) for leaf in _leaves(value[key], (*path, key))
        ]
    if isinstance(value, list) and value:
        return [
            leaf
            for i, item in enumerate(value)
            for leaf in _leaves(item, (*path, str(i)))
        ]
    return [(path, value)]


def _set(data: dict[str, JsonValue], write: _Write) -> None:
    cursor: JsonValue = data
    for part in write.path[:-1]:
        if isinstance(cursor, list):
            cursor = cursor[int(part)]
        elif isinstance(cursor, dict):
            cursor = cursor[part]
        else:
            raise ValueError(f"cannot write {write.path}")
    if not isinstance(cursor, dict):
        raise ValueError(f"cannot write {write.path}")
    cursor[write.path[-1]] = write.value


def _unique_cases(cases: tuple[Case, ...]) -> None:
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("cases/choices must be nonempty with unique names")
    for case in cases:
        _slug(case.name)


@dataclass(frozen=True)
class Experiment:
    """Explicit writes override a snapshot, never one another.

    Cases vary together; matrix axes cross in declaration order, last axis fastest.
    An empty cases tuple means one unnamed baseline row. The experiment name is
    an ID prefix; pass an empty prefix only to preserve existing standalone IDs.
    """

    name: str
    baseline: Baseline
    overrides: tuple[Component, ...] = ()
    cases: tuple[Case, ...] = ()
    matrix: tuple[Axis, ...] = ()
    schedule: ExperimentSchedule | None = None
    jobs: tuple[PlannedRun, ...] = ()

    def compile_jobs(self, source: "Source") -> tuple["DeploymentPlan", ...]:
        """Pure compilation; preserves cases, seed order and learning targets."""
        if self.schedule is not None and self.jobs:
            raise ValueError("choose local schedule or explicit launch allocations")
        cases = {case.name: case for case in self.resolve().cases}
        if len({(run.case, run.seed) for run in self.jobs}) != len(self.jobs):
            raise ValueError("launch case/seed bindings must be unique")
        if any(run.case not in cases for run in self.jobs):
            raise ValueError("launch references an unresolved case")
        return tuple(
            compile_plan(cases[run.case].configuration, run.spec, source, run.seed)
            for run in self.jobs
        )

    def cohort(
        self,
        source: "Source",
        cohort_id: str,
        *,
        deadline: float,
        spending_limit: float,
        prior_dollars: float,
        controller_dollars: float,
    ) -> "Cohort":
        """Freeze the existing job order for deployment by an independent service."""
        from manabot.remote.cohort import Cohort, CohortEntry

        plans = self.compile_jobs(source)
        cases = {case.name: case for case in self.resolve().cases}
        return Cohort(
            cohort_id=cohort_id,
            deadline=deadline,
            spending_limit=spending_limit,
            prior_dollars=prior_dollars,
            controller_dollars=controller_dollars,
            entries=tuple(
                CohortEntry(
                    job_id=f"{cohort_id}-{index}",
                    plan=plan,
                    monitoring=binding.monitoring,
                    checkpoint_seconds=binding.checkpoint_seconds,
                    experiment_json=json.dumps(
                        cases[binding.case].receipt(), sort_keys=True
                    ),
                )
                for index, (binding, plan) in enumerate(
                    zip(self.jobs, plans, strict=True)
                )
            ),
        )

    def prepare_job(self, index: int, source: "Source", job_id: str) -> "Job":
        """Persist one selected run through the existing disconnected job owner."""
        self.compile_jobs(source)
        binding = self.jobs[index]
        case = next(case for case in self.resolve().cases if case.name == binding.case)
        return prepare_experiment_job(
            case,
            binding.spec,
            source,
            binding.seed,
            job_id,
            monitoring=binding.monitoring,
            checkpoint_seconds=binding.checkpoint_seconds,
            destination=binding.spec.access.destination,
        )

    def resolve(self) -> ResolvedExperiment:
        if self.name:
            _slug(self.name)
        if set(TrainingRegime.model_fields) != set(_ROOT_OWNERS):
            raise ValueError(
                "TrainingRegime properties need explicit component ownership"
            )
        base = self.baseline.regime()
        if self.cases:
            _unique_cases(self.cases)
        if len({axis.name for axis in self.matrix}) != len(self.matrix):
            raise ValueError("matrix axis names must be unique")
        for axis in self.matrix:
            _slug(axis.name)
            _unique_cases(axis.choices)
        results: list[ResolvedCase] = []
        for case in self.cases or (Case(""),):
            for choices in product(*(axis.choices for axis in self.matrix)):
                parts = [
                    part
                    for part in (self.name, case.name, *(c.name for c in choices))
                    if part
                ]
                name = "-".join(parts)
                _slug(name)
                groups = [
                    ("experiment", self.overrides),
                    (f"case:{case.name}", case.overrides),
                ]
                groups.extend(
                    (f"axis:{axis.name}:{choice.name}", choice.overrides)
                    for axis, choice in zip(self.matrix, choices, strict=True)
                )
                writes = [
                    write
                    for source, components in groups
                    for component in components
                    for write in _writes(component, base, source)
                ]
                for i, write in enumerate(writes):
                    for other in writes[:i]:
                        common = min(len(write.path), len(other.path))
                        if write.path[:common] == other.path[:common]:
                            raise ValueError(
                                f"conflicting writes: {other.source} {other.path} and {write.source} {write.path}"
                            )
                data = _object(base)
                for write in writes:
                    _set(data, write)
                data["id"] = name
                resolved = validate_regime(data)
                complete = _object(resolved)
                provenance: list[SettingProvenance] = []
                effective = _object(resolved)
                agent = effective["agent"]
                assert isinstance(agent, dict)
                agent["compound_decisions"] = resolved.agent.compound_decisions
                if any(write.path == ("agent", "recent_events") for write in writes):
                    agent["recent_events"] = resolved.agent.recent_events
                    observation = effective["observation"]
                    assert isinstance(observation, dict)
                    observation["policy_history_version"] = int(
                        resolved.agent.recent_events
                    )
                for path, value in _leaves(effective):
                    origin: Origin = self.baseline.origin
                    source = self.baseline.name
                    for write in writes:
                        if path[: len(write.path)] == write.path:
                            origin, source = "override", write.source
                    if path == ("id",):
                        origin, source = "identity", "experiment-name"
                    provenance.append(
                        SettingProvenance(
                            path,
                            setting_owner(path),
                            origin,
                            source,
                            canonical_json(value).decode(),
                        )
                    )
                label = (
                    " / ".join(c.label or c.name for c in (case, *choices) if c.name)
                    or self.name
                )
                results.append(
                    ResolvedCase(
                        name,
                        label,
                        canonical_json(complete).decode(),
                        self.baseline.name,
                        self.baseline.digest,
                        tuple(provenance),
                    )
                )
        if len({case.name for case in results}) != len(results):
            raise ValueError("resolved case names collide")
        if len({case.label for case in results}) != len(results):
            raise ValueError("comparison labels must be unique")
        return ResolvedExperiment(self.name, tuple(results))
