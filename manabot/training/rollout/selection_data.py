"""Frozen complete-game trajectories for descriptive advantage-filter analysis.

The calling recipe owns its output paths. This module records
viewer-only policy predictions before executing semantic Commands; terminal
outcomes enter only after the game ends. No gradients or policy updates occur.
Rows are ordered by game, seat, then decision for same-viewer estimators.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from manabot.env.failures import ScalarCapture
from manabot.env.match import Match
from manabot.env.observation import ACTION_TYPE_DIM, ObservationSpace
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.schema import Strict
from manabot.model.world import validate_agent_setup
from manabot.policy.protocol import BackendPolicy
import managym
from managym.decision import Command, DecisionFrame, Observation

if TYPE_CHECKING:
    from manabot.training.state.backend import TrainingBackend


class SelectionGameSpec(Strict):
    """Population and split membership frozen before observing any outcome."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    seed: int = Field(ge=0, le=2**32 - 1)
    action_seed: int = Field(ge=0, le=2**32 - 1)
    assignment: Literal[0, 1]
    split: Literal["development", "held_out"]


@dataclass(frozen=True)
class SelectionPrediction:
    action: int
    value: float
    probability: float
    entropy: float
    reference_kl: float


class SelectionPredictor(Protocol):
    """Frozen-policy numerics with an independently seeded game-local stream."""

    def predict(
        self,
        encoded: dict[str, np.ndarray],
        reference: Literal["uniform", "action_type_uniform"],
    ) -> SelectionPrediction: ...


class FrozenRecord(Strict):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class SelectionDecision(FrozenRecord):
    step: int = Field(ge=0)
    actor: Literal[0, 1]
    observation_identity: str
    action: int = Field(ge=0)
    action_type: int = Field(ge=0)
    value: float
    action_probability: float = Field(gt=0, le=1)
    entropy: float
    reference_kl: float


class SelectionGame(FrozenRecord):
    spec: SelectionGameSpec
    winner: Literal[0, 1] | None
    terminal_digest: str
    rows: tuple[SelectionDecision, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_order(self) -> "SelectionGame":
        if [r.step for r in self.rows] != list(range(len(self.rows))):
            raise ValueError("selection trajectory must retain every decision in order")
        return self


class SelectionDataset(FrozenRecord):
    schema_version: Literal[1] = 1
    run_id: str
    stage_id: str
    policy_run_id: str
    policy_stage_id: str
    policy_sha256: str
    weights: Literal["raw", "ema"]
    world_binding_sha256: str
    population_sha256: str
    # Historical datasets predate reusable collection; only new outputs are cached.
    collection_sha256: str | None = None
    games: tuple[SelectionGame, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def population_matches(self) -> "SelectionDataset":
        population = [g.spec.model_dump(mode="json") for g in self.games]
        if canonical_sha256(population) != self.population_sha256:
            raise ValueError("selection population identity mismatch")
        if len({g.spec.seed for g in self.games}) != len(self.games):
            raise ValueError("selection dataset repeats a deal")
        return self


def collect_selection_game(
    agent: BackendPolicy,
    space: ObservationSpace,
    match: Match,
    spec: SelectionGameSpec,
    path: Path,
    *,
    reference: Literal["uniform", "action_type_uniform"],
    predictor: SelectionPredictor,
    max_steps: int,
) -> SelectionGame:
    """Retain a replayable journal, including partial evidence on cap/failure.

    Each prediction consumes only the acting viewer's encoded observation.
    Both seats use the same frozen model and a game-local action RNG. Forced
    native auto-resolution is not a policy row; distance counts surfaced rows.
    """
    if agent.belief_count_buckets:
        raise ValueError(
            "selection collection requires ordinary observation-only policy"
        )
    validate_agent_setup(agent, match.to_rust())
    env = managym.Env(
        seed=spec.seed,
        skip_trivial=True,
        decision_memory=space.encoder.config.max_decision_records > 0,
        decision_memory_include_own=space.encoder.config.decision_memory_include_own,
    )
    capture = ScalarCapture(
        spec.seed,
        True,
        decision_memory=space.encoder.config.max_decision_records > 0,
        decision_memory_include_own=space.encoder.config.decision_memory_include_own,
    )
    raw, _ = capture.reset(env, match.to_rust())
    rows: list[SelectionDecision] = []
    capture.context["receipt_path"] = str(path)
    with (
        capture.guard(),
        os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as output,
    ):
        output.write(
            json.dumps(
                {
                    "seed": spec.seed,
                    "match": match.config.model_dump(mode="json"),
                    "skip_trivial": True,
                    **(
                        {"decision_memory": False}
                        if space.encoder.config.max_decision_records == 0
                        and not space.encoder.config.decision_memory_include_own
                        else {}
                    ),
                    **(
                        {"decision_memory_include_own": True}
                        if space.encoder.config.decision_memory_include_own
                        else {}
                    ),
                }
            )
            + "\n"
        )
        while not env.is_game_over():
            if len(rows) >= max_steps:
                raise RuntimeError(
                    "selection game exceeded step cap; no terminal label"
                )
            actor = env.current_agent_index()
            if actor not in (0, 1):
                raise RuntimeError("live selection game has no actor")
            frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
            observation = Observation.from_json(env.semantic_observation_json(actor))
            encoded = space.encode(raw)
            if len(frame.offers) != int(encoded["actions_valid"].sum()):
                raise ValueError("selection policy cannot encode every legal offer")
            prediction = predictor.predict(encoded, reference)
            action = prediction.action
            row = SelectionDecision(
                step=len(rows),
                actor=actor,
                observation_identity=observation.viewer_state_hash,
                action=action,
                action_type=int(encoded["actions"][action, :ACTION_TYPE_DIM].argmax()),
                value=prediction.value,
                action_probability=prediction.probability,
                entropy=prediction.entropy,
                reference_kl=prediction.reference_kl,
            )
            command = Command(
                f"selection-{spec.seed}-{row.step}",
                frame.revision,
                int(frame.offers[action]["id"]),
            )
            transition = capture.command(env, command.to_json())
            output.write(
                json.dumps(
                    {
                        "command": json.loads(command.to_json()),
                        "transition": json.loads(transition),
                        "prediction": row.model_dump(mode="json"),
                    }
                )
                + "\n"
            )
            output.flush()
            rows.append(row)
            if not env.is_game_over():
                raw = env.observation_for_player(env.current_agent_index())
        output.write(
            json.dumps(
                {"winner": env.winner_index(), "state_digest": env.state_digest()}
            )
            + "\n"
        )
    return SelectionGame(
        spec=spec,
        winner=env.winner_index(),
        terminal_digest=env.state_digest(),
        rows=tuple(rows),
    )


def replay_selection_game(path: Path) -> int:
    """Replay a retained complete journal without a policy; reject partial logs.

    Returns the number of replayed Commands. Each recorded transition and the
    terminal winner and state digest must match the native replay exactly.
    """
    with path.open() as source:
        header = json.loads(next(source))
        env = managym.Env(
            seed=header["seed"],
            skip_trivial=header["skip_trivial"],
            decision_memory=header.get("decision_memory", True),
            decision_memory_include_own=header.get(
                "decision_memory_include_own", False
            ),
        )
        env.reset(Match(MatchConfig(**header["match"])).to_rust())
        count = 0
        complete = False
        for line in source:
            row = json.loads(line)
            if complete:
                raise ValueError("data follows terminal selection receipt")
            if "command" in row:
                observed = json.loads(
                    env.execute_semantic_command_json(json.dumps(row["command"]))
                )
                if observed != row["transition"]:
                    raise ValueError("selection replay transition mismatch")
                count += 1
            else:
                if (
                    not env.is_game_over()
                    or env.winner_index() != row["winner"]
                    or env.state_digest() != row["state_digest"]
                ):
                    raise ValueError("selection replay terminal mismatch")
                complete = True
        if not complete:
            raise ValueError("selection attempt is incomplete")
    return count


def collect_selection(
    policy: Path,
    out: Path,
    population: tuple[SelectionGameSpec, ...],
    *,
    match: MatchConfig,
    reference: Literal["uniform", "action_type_uniform"] = "action_type_uniform",
    max_steps: int = 2000,
    backend: "TrainingBackend",
) -> Path:
    """Freeze a policy's complete-game diagnostics, retaining partial replay journals.

    Population seeds and both assignments in each whole-game split are checked
    before play. Existing output must bind the same policy, population and inputs.
    Retained dataset keys `stage_id` and `policy_stage_id` name the output files.
    `backend` loads the policy and makes each game's seeded predictor.
    """
    if len(population) < 4 or len({game.seed for game in population}) != len(
        population
    ):
        raise ValueError(
            "selection population requires at least four unique deal seeds"
        )
    for split in ("development", "held_out"):
        if {game.assignment for game in population if game.split == split} != {0, 1}:
            raise ValueError("each whole-game split requires both deck assignments")
    digest = file_sha256(policy)
    population_digest = canonical_sha256(
        [game.model_dump(mode="json") for game in population]
    )
    collection_digest = canonical_sha256(
        {
            "match": match.model_dump(mode="json"),
            "reference": reference,
            "max_steps": max_steps,
        }
    )
    if out.exists():
        saved = SelectionDataset.model_validate_json(out.read_bytes())
        if (
            saved.policy_sha256 != digest
            or saved.population_sha256 != population_digest
        ):
            raise ValueError("selection output belongs to another policy or population")
        if saved.collection_sha256 != collection_digest:
            raise ValueError("selection output belongs to other collection inputs")
        return out
    agent, space = backend.load_policy(str(policy))
    out.parent.mkdir(parents=True, exist_ok=True)
    journals = Path(tempfile.mkdtemp(prefix=f"{out.stem}-games-", dir=out.parent))
    base_match = Match(match)
    payload = backend.metadata(str(policy))
    weights = payload.get("bc", {}).get("weights")
    if weights not in {"raw", "ema"}:
        raise ValueError("selection checkpoint must identify raw or EMA weights")
    if file_sha256(policy) != digest:
        raise ValueError("selection policy changed while loading")
    games: list[SelectionGame] = []
    for index, spec in enumerate(population):
        journal = journals / f"{index:04d}.jsonl"
        game = collect_selection_game(
            agent,
            space,
            base_match if spec.assignment == 0 else base_match.swapped(),
            spec,
            journal,
            reference=reference,
            predictor=backend.collection.SelectionPredictor(agent, spec.action_seed),
            max_steps=max_steps,
        )
        if replay_selection_game(journal) != len(game.rows):
            raise ValueError("selection replay decision count mismatch")
        games.append(game)
    dataset = SelectionDataset(
        run_id=out.parent.name,
        stage_id=out.stem,
        policy_run_id=policy.parent.parent.name,
        policy_stage_id=policy.stem,
        policy_sha256=digest,
        weights=weights,
        world_binding_sha256=canonical_sha256(agent.world_binding),
        population_sha256=population_digest,
        collection_sha256=collection_digest,
        games=tuple(games),
    )
    atomic_json(out, dataset.model_dump(mode="json"))
    return out
