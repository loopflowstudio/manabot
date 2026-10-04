"""Frozen complete-game trajectories for descriptive advantage-filter analysis.

The regime executor owns attempts, budgets and artifacts. This module records
viewer-only policy predictions before executing semantic Commands; terminal
outcomes enter only after the game ends. No gradients or policy updates occur.
Rows are ordered by game, seat, then decision for same-viewer estimators.
"""

from collections.abc import Callable
import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator
import torch

from manabot.arena.models import canonical_sha256
from manabot.env import Match
from manabot.env.observation import ObservationSpace
from manabot.model.agent import Agent
from manabot.model.world import validate_agent_setup
from manabot.training.models import SelectionGameSpec, Strict
from manabot.training.references import reference_distribution
import managym
from managym.decision import Command, DecisionFrame, Observation


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
    games: tuple[SelectionGame, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def population_matches(self) -> "SelectionDataset":
        population = [g.spec.model_dump(mode="json") for g in self.games]
        if canonical_sha256(population) != self.population_sha256:
            raise ValueError("selection population identity mismatch")
        if len({g.spec.seed for g in self.games}) != len(self.games):
            raise ValueError("selection dataset repeats a deal")
        return self


@torch.inference_mode()
def collect_selection_game(
    agent: Agent,
    space: ObservationSpace,
    match: Match,
    spec: SelectionGameSpec,
    path: Path,
    *,
    reference: Literal["uniform", "action_type_uniform"],
    max_steps: int,
    check: Callable[[], None],
) -> SelectionGame:
    """Retain a replayable journal, including partial evidence on cap/failure.

    Each prediction consumes only the acting viewer's encoded observation.
    Both seats use the same frozen model and a game-local action RNG. Forced
    native auto-resolution is not a policy row; distance counts surfaced rows.
    """
    if agent.belief_count_buckets or agent.hypers.compound_decisions:
        raise ValueError(
            "selection collection requires ordinary observation-only policy"
        )
    validate_agent_setup(agent, match.to_rust())
    env = managym.Env(seed=spec.seed, skip_trivial=True)
    raw, _ = env.reset(match.to_rust())
    generator = torch.Generator().manual_seed(spec.action_seed)
    rows: list[SelectionDecision] = []
    with path.open("x") as output:
        output.write(
            json.dumps(
                {
                    "seed": spec.seed,
                    "match": match.hypers.model_dump(mode="json"),
                    "skip_trivial": True,
                }
            )
            + "\n"
        )
        while not env.is_game_over():
            check()
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
            tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in encoded.items()}
            logits, value = agent(tensors)
            probabilities = logits.softmax(-1)
            action = int(torch.multinomial(probabilities[0], 1, generator=generator))
            logs = probabilities.clamp_min(1e-12).log()
            ref = reference_distribution(tensors, reference)
            row = SelectionDecision(
                step=len(rows),
                actor=actor,
                observation_identity=observation.viewer_state_hash,
                action=action,
                action_type=int(tensors["actions"][0, action, :-1].argmax()),
                value=float(value.item()),
                action_probability=float(probabilities[0, action]),
                entropy=float(-(probabilities * logs).sum()),
                reference_kl=float(
                    (probabilities * (logs - ref.clamp_min(1e-12).log())).sum()
                ),
            )
            command = Command(
                f"selection-{spec.seed}-{row.step}",
                frame.revision,
                int(frame.offers[action]["id"]),
            )
            transition = env.execute_semantic_command_json(command.to_json())
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
