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


def _delivery(
    steps: list[Step],
    target: int,
    verification: str = "verify",
    *,
    review: str = "demo",
) -> None:
    boundary = [
        _one(steps, label)
        for label in ("pr-publish", review, "land --wait-and-fix", verification)
    ]
    positions = [steps.index(step) for step in boundary]
    assert positions == sorted(positions), "Delivery boundaries changed order"
    for gate_review in (boundary[1], boundary[-1]):
        assert gate_review.interactive and gate_review.returns_to == target, (
            gate_review.label
        )


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


def _phase(steps: list[Step], start: int, plan: str, review: str) -> int:
    """Check one phase shaped like builtin build; return the index after it.

    The plan step cuts the work into slices. Realign decides the inner loop
    (another slice, or on), so a slice costs implement and realign only;
    compress and the phase's review run once per pass over the whole batch,
    and the review decides whether the phase goes back to implementation.
    """
    end = [step.label for step in steps].index(review, start)
    phase = steps[start : end + 1]
    head = [step.label for step in phase[:5]]
    assert head == [plan, "implement", "sync", "realign", "compress"], head
    implement = phase[1]
    assert phase[3].returns_to == implement.key, "realign decides the slice loop"
    assert phase[-1].returns_to == implement.key, f"{review} decides the phase"
    assert not any(step.interactive for step in phase), f"{plan} phase is headless"
    return end + 1


def _coverage(coverage: list[Step]) -> None:
    """Check the two-pass coverage graph: sliced build/review, then card-perf xor."""
    head = [step.label for step in coverage[:5]]
    assert head == ["card-batch", "sync", "implement", "card-review", "card-perf"], head
    sync, implement, review, perf = coverage[1:5]
    assert implement.returns_to == sync.key, "implement decides the slice loop"
    assert review.returns_to == sync.key, "card-review decides the build/review loop"
    assert perf.kind == "xor", f"Expected xor router card-perf, got {perf.kind}"
    routes = {path.name: path.steps for path in perf.paths}
    assert set(routes) == {"clean", "optimize"}, routes.keys()
    assert routes["clean"] == [], "clean path exits directly to pr-publish"
    opt = routes["optimize"]
    assert [step.label for step in opt] == ["sync", "implement", "card-perf-review"], opt
    assert opt[-1].returns_to == opt[0].key, "card-perf-review decides the optimize loop"
    assert not any(
        step.interactive for step in [*coverage[:5], *opt]
    ), "build, review and perf phases are headless"
    assert coverage[5].label == "compress"
    assert coverage[6].label == "pr-publish"
    assert coverage[7].label == "ship-review"
    _delivery(coverage, coverage[0].key, review="ship-review")


def _experiment(experiment: list[Step]) -> None:
    """Design, then run and report; only a finding ends in a person's review."""
    assert experiment[0].label == "experiment-design"
    router = _one(experiment, "experiment-tier")
    routes = {path.name: [step.label for step in path.steps] for path in router.paths}
    assert routes == {
        "exploratory": ["experiment-run", "experiment-report"],
        "finding": ["experiment-run", "experiment-report", "experiment-review"],
    }
    finding = next(path.steps for path in router.paths if path.name == "finding")
    interactive = [step for step in _flatten(experiment) if step.interactive]
    assert interactive == [finding[-1]]
    assert finding[-1].returns_to == finding[0].key


def _performance(steps: list[Step], measure: str, review: str) -> None:
    """Check the optimization loop and its single software delivery boundary."""
    position = _phase(steps, 0, measure, review)
    assert steps[position].label == "pr-publish"
    _delivery(steps, _one(steps, measure).key)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path.cwd() == root, "Run from the repository root"
    graphs = _compile_graphs(root)
    _coverage(graphs["coverage"])
    _performance(
        graphs["engine-performance"], "engine-measure", "engine-performance-review"
    )
    _performance(
        graphs["experiment-performance"],
        "experiment-measure",
        "experiment-performance-review",
    )
    _experiment(graphs["experiment"])
    print(
        "Coverage/performance loops, single delivery and the experiment review boundary: valid"
    )


if __name__ == "__main__":
    main()
