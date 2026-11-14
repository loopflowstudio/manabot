"""Manabot's baseline adapter. Discovery and training metadata stay here.

A factory pins the checkpoint plus runner/backend identity. Independent sessions
share no RNG or model cache. Optional Academy replay/advice is a separate entry
point; this module imports only the engine's public policy contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import random
from typing import TYPE_CHECKING

import managym
from managym.policy import (
    JSONValue,
    PolicyDecision,
    PolicyDescriptor,
    PolicyDetail,
    PolicyEvaluation,
    PolicyFactory,
    PolicySession,
    PolicySpec,
    PolicyView,
    ValueEstimate,
)

if TYPE_CHECKING:
    from manabot.model.catalog import CheckpointSummary


def runner_identity() -> str:
    """Bind installed implementation bytes, including encoders and backends."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return f"manabot/{version('manabot')}:{digest.hexdigest()}:torch/{version('torch')}"


def _deck_name(deck: managym.Deck) -> str:
    # Names are presentation only. Admission below always compares full decks,
    # including sideboards; an unrecognized list keeps its own content identity.
    for preset in ("two_deck", "optional_payments", "jeong_increment"):
        for key in managym.authored_deck_keys(preset):
            player = managym.authored_deck_setup(preset, key)
            if deck == managym.Deck(dict(player.decklist), dict(player.sideboard)):
                return key
    return f"custom deck {deck.identity[:10]}"


@dataclass(frozen=True)
class BaselineFactory:
    spec: PolicySpec
    summary: CheckpointSummary | None = None

    def describe(self) -> PolicyDescriptor:
        from manabot.backends import resolve
        from manabot.model.catalog import inspect_checkpoint

        kind = self.spec.config.get("kind")
        if kind in ("heuristic", "greedy"):
            return PolicyDescriptor(
                str(kind).title(),
                "manabot",
                runner_identity(),
                config=self.spec.config,
                alias=str(kind),
            )
        if self.spec.artifact is None:
            raise ValueError("Manabot policy requires an artifact.")
        summary = inspect_checkpoint(Path(self.spec.artifact), backend=resolve("torch"))
        displayed = self.summary or summary
        details = tuple(
            PolicyDetail(label, "not recorded" if value is None else str(value))
            for label, value in (
                ("Run", displayed.run),
                ("Run ID", displayed.run_id),
                ("Stage", displayed.stage),
                ("Weights", displayed.weights),
                ("Updates", displayed.updates),
                ("Transitions", displayed.transitions),
                ("World", displayed.world),
                ("Model", displayed.model),
                ("Parameters", displayed.parameters),
                ("Step", displayed.step),
                ("Source", displayed.source),
            )
        )
        if summary.setups:
            details += (
                PolicyDetail(
                    "Bound decks",
                    " vs ".join(_deck_name(deck) for deck in summary.setups),
                ),
            )
        # Search admission belongs to the shared composition, not a bare flag.
        return PolicyDescriptor(
            (self.summary or summary).label,
            "manabot",
            runner_identity(),
            {
                "actor": hashlib.sha256(
                    Path(self.spec.artifact).read_bytes()
                ).hexdigest()
            },
            self.spec.config,
            group=displayed.run or displayed.run_id,
            details=details,
            incompatibility=summary.incompatibility,
            supports_search=summary.supports_search,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        if self.spec.artifact is None:
            if self.spec.config.get("kind") not in ("heuristic", "greedy"):
                raise ValueError("Manabot policy requires an artifact.")
            return
        import json

        from manabot.backends import resolve
        from manabot.model.catalog import inspect_checkpoint

        summary = inspect_checkpoint(Path(self.spec.artifact), backend=resolve("torch"))
        from manabot.infra.config import ObservationConfig

        ObservationConfig.from_checkpoint(
            {"policy_input_version": summary.policy_input_version}
        )
        if summary.incompatibility:
            raise ValueError(
                f"{summary.label} cannot play here: {summary.incompatibility}"
            )
        trained = [
            json.dumps({"deck": deck.main, "sideboard": deck.sideboard}, sort_keys=True)
            for deck in summary.setups
        ]
        actual = [
            json.dumps(
                {"deck": dict(config.decklist), "sideboard": dict(config.sideboard)},
                sort_keys=True,
            )
            for config in setup
        ]
        if trained and not (
            sorted(trained) == sorted(actual)
            or (
                len(trained) > 2
                and len(actual) == 2
                and all(deck in trained for deck in actual)
            )
        ):
            raise ValueError(
                f"{summary.label} was trained on the recorded deck setups; selected setup differs."
            )

    def load(self, *, seat: int, seed: int) -> BaselineSession:
        return BaselineSession(self.spec, seat=seat, seed=seed)


class BaselineSession:
    def __init__(self, spec: PolicySpec, *, seat: int, seed: int) -> None:
        self.seat = seat
        self.spec = spec
        self._rng = random.Random(seed)
        self.agent = None
        self.space = None
        self.anchor = None
        if spec.artifact is not None:
            from manabot.backends.torch.checkpoint import load_checkpoint

            self.agent, self.space = load_checkpoint(spec.artifact)
        else:
            from manabot.sim.anchors import ScriptedGreedyPlayer
            from manabot.sim.heuristic import HeuristicPlayer

            self.anchor = (
                HeuristicPlayer()
                if spec.config["kind"] == "heuristic"
                else ScriptedGreedyPlayer()
            )

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        if self.anchor is not None:
            from types import SimpleNamespace
            from typing import cast

            from manabot.env.env import Env

            action = self.anchor.act(
                cast(Env, SimpleNamespace(last_raw_obs=view.observation)), {}
            )
            return PolicyDecision(view.choices[action], view.binding)
        evaluation = self.evaluate(view)
        probabilities = evaluation.probabilities
        action = (
            max(range(len(view.choices)), key=probabilities.__getitem__)
            if self.spec.config.get("deterministic", False)
            else self._rng.choices(
                range(len(view.choices)), weights=probabilities, k=1
            )[0]
        )
        return PolicyDecision(
            view.choices[action], view.binding, probabilities, evaluation.value
        )

    def evaluate(self, view: PolicyView) -> PolicyEvaluation:
        from manabot.backends.torch.bot import evaluate_observation

        if self.agent is None or self.space is None:
            raise ValueError("This policy has no baseline evaluator.")
        probabilities, value = evaluate_observation(
            self.agent, self.space, view.observation
        )
        return PolicyEvaluation(
            probabilities,
            ValueEstimate(
                value, view.seat, self.agent.value_semantic or "undeclared critic units"
            ),
        )

    def fork(self, *, seed: int) -> BaselineSession:
        # Model parameters are read-only; history-free evaluation mutates no model state.
        branch = object.__new__(BaselineSession)
        branch.seat, branch.spec = self.seat, self.spec
        branch.agent, branch.space, branch.anchor = self.agent, self.space, None
        branch._rng = random.Random(seed)
        return branch

    def close(self) -> None:
        self.agent = None
        self.space = None
        self.anchor = None


@dataclass(frozen=True)
class _UnavailablePolicy:
    descriptor: PolicyDescriptor

    def describe(self) -> PolicyDescriptor:
        return self.descriptor

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        raise ValueError(self.descriptor.incompatibility)

    def load(self, *, seat: int, seed: int) -> PolicySession:
        raise ValueError(self.descriptor.incompatibility)


class BaselineProvider:
    def policy(self, spec: PolicySpec) -> BaselineFactory:
        if set(spec.config) - {"kind", "deterministic"}:
            raise ValueError("Unknown manabot policy config fields.")
        return BaselineFactory(spec)

    def discover(self, config: dict[str, JSONValue] | None) -> list[PolicyFactory]:
        from manabot.backends import resolve
        from manabot.model.catalog import discover_checkpoints

        ambient = config is None
        if config is None:
            config = {
                "paths": [
                    p
                    for p in os.environ.get("ACADEMY_PLAY_CHECKPOINTS", "").split(
                        os.pathsep
                    )
                    if p
                ],
                "runs_dir": os.environ.get("MANABOT_RUNS_DIR")
                or str(Path.cwd() / ".runs"),
            }
        paths = config.get("paths", [])
        root = config.get("runs_dir", str(Path.cwd() / ".runs"))
        if (
            not isinstance(paths, list)
            or not all(isinstance(p, str) for p in paths)
            or not isinstance(root, str)
        ):
            raise ValueError("manabot discovery requires paths and a runs_dir string.")
        summaries, missing = discover_checkpoints(
            [Path(p).expanduser() for p in paths],
            backend=resolve("torch"),
            runs_dir=Path(root).expanduser(),
        )
        return [
            *(
                self.policy(PolicySpec("manabot", config={"kind": kind}))
                for kind in ("heuristic", "greedy")
                if ambient
            ),
            *(
                BaselineFactory(PolicySpec("manabot", str(summary.path)), summary)
                for summary in summaries
            ),
            *(
                _UnavailablePolicy(
                    PolicyDescriptor(
                        entry.label,
                        "manabot",
                        runner_identity(),
                        {"actor": entry.sha256},
                        details=(PolicyDetail("Source", "tournament"),),
                        incompatibility=entry.reason,
                    )
                )
                for entry in missing
            ),
        ]


provider = BaselineProvider()
