"""Manabot's baseline adapter. Discovery and training metadata stay here.

A factory pins the checkpoint plus runner/backend identity. Independent sessions
share no RNG or model cache. Optional Academy replay/advice is a separate entry
point; this module imports only the engine's public policy contract.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

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
    from manabot.backends.torch.policy import ActorCritic, PolicyInference
    from manabot.env.env import Env
    from manabot.env.observation import ObservationSpace
    from manabot.model.catalog import CheckpointSummary

# Scripted reference players: no artifact, selected by `config["kind"]`.
_ANCHORS = ("heuristic", "greedy")


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

    def _inspect(self) -> CheckpointSummary | None:
        """Current checkpoint facts, or None for a scripted reference player."""
        from manabot.backends import resolve
        from manabot.model.catalog import inspect_checkpoint

        if self.spec.artifact is None:
            if self.spec.config.get("kind") not in _ANCHORS:
                raise ValueError("Manabot policy requires an artifact.")
            return None
        return inspect_checkpoint(Path(self.spec.artifact), backend=resolve("torch"))

    def describe(self) -> PolicyDescriptor:
        summary = self._inspect()
        if summary is None:
            kind = str(self.spec.config["kind"])
            return PolicyDescriptor(
                kind.title(),
                "manabot",
                runner_identity(),
                config=self.spec.config,
                alias=kind,
            )
        # Discovery's summary carries the run facts; the bytes are re-read here.
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
            displayed.label,
            "manabot",
            runner_identity(),
            {"actor": hashlib.sha256(summary.path.read_bytes()).hexdigest()},
            self.spec.config,
            group=displayed.run or displayed.run_id,
            details=details,
            incompatibility=summary.incompatibility,
            supports_search=summary.supports_search,
            search_reason=summary.search_reason,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        from manabot.infra.config import ObservationConfig

        summary = self._inspect()
        if summary is None:
            return
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
            bound = " vs ".join(_deck_name(deck) for deck in summary.setups)
            raise ValueError(
                f"Checkpoint setup differs: {summary.label} was trained on {bound}; choose those decks "
                "or another policy."
            )

    def load(self, *, seat: int, seed: int) -> PolicySession:
        if self.spec.artifact is None:
            return AnchorSession(str(self.spec.config["kind"]))
        return BaselineSession(self.spec, seed=seed)


class AnchorSession:
    """A scripted reference player: one legal action, no probabilities or value."""

    def __init__(self, kind: str) -> None:
        from manabot.sim.anchors import ScriptedGreedyPlayer
        from manabot.sim.heuristic import HeuristicPlayer

        self.anchor = (
            HeuristicPlayer() if kind == "heuristic" else ScriptedGreedyPlayer()
        )

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        # The scripted players read only the raw observation from their Env.
        action = self.anchor.act(
            cast("Env", SimpleNamespace(last_raw_obs=view.observation)), {}
        )
        return PolicyDecision(view.choices[action], view.binding)

    def close(self) -> None:
        pass


class BaselineSession:
    """A loaded checkpoint; search-capable through evaluate and fork."""

    def __init__(self, spec: PolicySpec, *, seed: int) -> None:
        from manabot.backends.torch.checkpoint import load_checkpoint

        assert spec.artifact is not None
        self.spec = spec
        self._rng = random.Random(seed)
        self.agent: ActorCritic | None
        self.space: ObservationSpace | None
        self.agent, self.space = load_checkpoint(spec.artifact)
        # One inference scope per loaded policy caches the card-program table;
        # forks share it and only this loading session closes it.
        self._scope: PolicyInference | None = self.agent.inference_scope()
        self._owns_scope = True

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
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

        if self.agent is None or self.space is None or self._scope is None:
            raise ValueError("Policy session is closed.")
        probabilities, value = evaluate_observation(
            self._scope, self.space, view.observation
        )
        semantic = self.agent.value_semantic or "undeclared critic units"
        if not math.isfinite(value):
            raise ValueError("Policy returned a nonfinite value estimate.")
        if semantic == "signed_outcome":
            # A scalar head regressed on signed outcomes can overshoot its
            # targets; an expected outcome lies in [-1, 1], so project the
            # estimate there. W/D/L expectations are already inside it.
            value = min(1.0, max(-1.0, value))
        return PolicyEvaluation(
            probabilities, ValueEstimate(value, view.seat, semantic)
        )

    def fork(self, *, seed: int) -> BaselineSession:
        # Model parameters are read-only; history-free evaluation mutates no model state.
        branch = copy.copy(self)
        branch._rng = random.Random(seed)
        branch._owns_scope = False
        return branch

    def close(self) -> None:
        if self._owns_scope and self._scope is not None:
            self._scope.__exit__(None, None, None)
        self._scope = None
        self.agent = None
        self.space = None


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
            from etude.storage import default_root

            config = {
                # Explicit sources, then the policies `etude train` wrote here.
                "paths": [
                    *(
                        p
                        for p in os.environ.get("ACADEMY_PLAY_CHECKPOINTS", "").split(
                            os.pathsep
                        )
                        if p
                    ),
                    str(default_root()),
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
                for kind in _ANCHORS
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
