"""Compile repository workflows with the installed lf CLI without launching work.

The CLI owns syntax and discovery. These checks cover the repository's delivery,
iteration and experiment boundaries over that compiled representation.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Literal

from pydantic import BaseModel, Field


class Route(BaseModel):
    name: str
    steps: list[Step]


class Step(BaseModel):
    key: int
    id: str | None = None
    label: str
    kind: Literal["skill", "op", "xor"]
    interactive: bool
    returns_to: int | None = None
    paths: list[Route] = Field(default_factory=list)


class Compiled(BaseModel):
    steps: list[Step]


class Graph(BaseModel):
    compiled: Compiled
    unavailable: str | None = None


class Definition(BaseModel):
    name: str


def _lf(*args: str) -> str:
    return subprocess.check_output(["lf", *args, "--json"], text=True)


def _flatten(steps: list[Step]) -> list[Step]:
    flattened: list[Step] = []
    for step in steps:
        flattened.append(step)
        for path in step.paths:
            flattened.extend(_flatten(path.steps))
    return flattened


def _one(steps: list[Step], label: str) -> Step:
    matches = [step for step in steps if step.label == label]
    assert len(matches) == 1, f"Expected one {label}, got {len(matches)}"
    return matches[0]


def _delivery(steps: list[Step], target: int, verification: str = "verify") -> None:
    boundary = [
        _one(steps, label)
        for label in ("pr-publish", "demo", "gate", "land --wait-and-fix", verification)
    ]
    positions = [steps.index(step) for step in boundary]
    assert positions == sorted(positions), "Delivery boundaries changed order"
    for review in (boundary[1], boundary[-1]):
        assert review.interactive and review.returns_to == target, review.label


def _compile_graphs(root: Path) -> dict[str, list[Step]]:
    """Let lf resolve composition, then check every discovered node/reference."""
    skills = {
        Definition.model_validate(value).name
        for value in json.loads(_lf("list", "skill"))
    }
    discovered = {
        Definition.model_validate(value).name
        for value in json.loads(_lf("graph", "list"))
    }
    graphs: dict[str, list[Step]] = {}
    for path in sorted((root / ".lf/graphs").glob("*.yaml")):
        assert path.stem in discovered, f"Undiscovered graph: {path.stem}"
        graph = Graph.model_validate_json(_lf("graph", "show", path.stem))
        assert graph.unavailable is None, graph.unavailable
        steps = _flatten(graph.compiled.steps)
        keys = {step.key for step in steps}
        assert len(keys) == len(steps), f"Duplicate node keys in {path.stem}"
        for step in steps:
            if step.kind in ("skill", "xor"):
                assert step.label in skills, f"Missing skill {step.label}"
            if step.returns_to is not None:
                assert step.returns_to in keys and step.returns_to < step.key
        graphs[path.stem] = graph.compiled.steps
        print(f"{path.stem}: compiled; {len(steps)} nodes; all skills resolve")
    return graphs


def _coverage(coverage: list[Step]) -> None:
    implementations = [step for step in coverage if step.label == "implement"]
    assert [step.id for step in implementations] == [
        "implement-working",
        "implement-scalable",
        "implement-performance",
    ]
    reviews = [
        "coverage-behavior-review",
        "coverage-decomposition-review",
        "engine-performance-review",
    ]
    for implementation, review in zip(implementations, reviews, strict=True):
        position = coverage.index(_one(coverage, review))
        assert coverage[position + 1].returns_to == implementation.key
        assert _one(coverage, review).key > implementation.key
    _delivery(coverage, coverage[0].key)
    assert _one(coverage, "pr-publish").key > _one(coverage, reviews[-1]).key


def _experiment(experiment: list[Step]) -> None:
    router = _one(experiment, "experiment-route")
    routes = {path.name: path.steps for path in router.paths}
    assert set(routes) == {"software", "ready"} and not routes["ready"]
    software = routes["software"]
    _delivery(software, _one(software, "implement").key, "experiment-software-review")
    assert not any(step.label == "verify" for step in _flatten(experiment))
    for review, target in (
        ("experiment-design-review", "experiment-design"),
        ("experiment-execution-review", "experiment-freeze"),
        ("experiment-evidence-review", "pool-validation"),
    ):
        step = _one(experiment, review)
        assert step.interactive and step.returns_to == _one(experiment, target).key
    assert (
        router.key
        < _one(experiment, "experiment-freeze").key
        < _one(experiment, "experiment-execution-review").key
        < _one(experiment, "pool-validation").key
        < _one(experiment, "experiment-analysis").key
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path.cwd() == root, "Run from the repository root"
    graphs = _compile_graphs(root)
    _coverage(graphs["coverage"])
    performance = graphs["engine-performance"]
    _delivery(performance, performance[0].key)
    _experiment(graphs["experiment"])
    print(
        "Coverage loops, single delivery, optional software and experiment review boundaries: valid"
    )


if __name__ == "__main__":
    main()
