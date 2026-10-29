"""Collect and replay complete compound games under frozen policy weights.

Retained Commands and behavior tapes belong to collection. Parameter updates
and per-seat credit assignment live in training.learning.compound."""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import torch

from manabot.env.failures import ScalarCapture
from manabot.env.match import Match
from manabot.infra.config import MatchConfig
from manabot.model.actor_critic import ActorCritic
from manabot.sim.action.compound import CompoundDecision, sample_compound
import managym


@dataclass(frozen=True)
class EpisodeDecision:
    actor: int
    decision: CompoundDecision


@dataclass
class CompoundGame:
    decisions: list[EpisodeDecision]
    winner: int | None
    microchoices: int
    auto_resolved: int


@dataclass
class CompoundStatistics:
    attempted_games: int = 0
    failed_games: int = 0
    interrupted_microchoices: int = 0
    games: int = 0
    decisions: int = 0
    microchoices: int = 0
    factors: int = 0
    forced_factors: int = 0
    auto_resolved: int = 0
    decision_seconds: float = 0
    max_decision_seconds: float = 0
    collection_seconds: float = 0
    learning_seconds: float = 0
    optimizer_exposures: int = 0
    losses: list[float] = field(default_factory=list)
    prompt_kinds: dict[str, int] = field(default_factory=dict)

    def record_game(self, game: CompoundGame) -> None:
        """Count accepted games; native steps and decoder factors stay distinct."""
        self.games += 1
        self.decisions += len(game.decisions)
        self.microchoices += game.microchoices
        self.auto_resolved += game.auto_resolved
        for item in game.decisions:
            decision = item.decision
            kind = str(decision.offers.projection["kind"])
            self.prompt_kinds[kind] = self.prompt_kinds.get(kind, 0) + 1
            self.factors += len(decision.output.tokens)
            self.forced_factors += sum(
                int((probs > 0).sum()) == 1 for probs in decision.output.probabilities
            )
            self.decision_seconds += decision.seconds
            self.max_decision_seconds = max(self.max_decision_seconds, decision.seconds)


def collect_game(
    agent: ActorCritic,
    match: Match,
    seed: int,
    generator: torch.Generator,
    path: Path,
    *,
    max_commands: int,
    check: Callable[[], None],
    skip_trivial: bool = True,
) -> CompoundGame:
    """Retain an incomplete attempt on any exception; never label it terminal.

    Evidence contains both seats' observations in native receipts and is private
    training audit data. It must not be served as viewer-facing Study evidence.
    """
    env = managym.Env(seed=seed, skip_trivial=skip_trivial)
    capture = ScalarCapture(seed, skip_trivial)
    raw, _ = capture.reset(env, match.to_rust())
    decisions: list[EpisodeDecision] = []
    count = 0
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
                    "seed": seed,
                    "match": match.config.model_dump(mode="json"),
                    "skip_trivial": skip_trivial,
                }
            )
            + "\n"
        )
        while not env.is_game_over():
            check()
            actor = env.current_agent_index()
            if actor is None:
                raise RuntimeError("live game has no actor")
            with torch.no_grad():
                decision = sample_compound(agent, env, raw, generator=generator)
            decisions.append(EpisodeDecision(actor, decision))
            for command in decision.commands:
                check()
                if count >= max_commands:
                    raise RuntimeError(
                        "compound game command cap reached; no terminal target"
                    )
                result = capture.command(env, command.to_json())
                output.write(
                    json.dumps(
                        {
                            "command": json.loads(command.to_json()),
                            "transition": json.loads(result),
                        }
                    )
                    + "\n"
                )
                output.flush()
                count += 1
            if not env.is_game_over():
                next_actor = env.current_agent_index()
                if next_actor is None:
                    raise RuntimeError("live game has no next actor")
                raw = env.observation_for_player(next_actor)
        output.write(
            json.dumps(
                {"winner": env.winner_index(), "state_digest": env.state_digest()}
            )
            + "\n"
        )
    return CompoundGame(
        decisions,
        env.winner_index(),
        count,
        env.skip_trivial_count(),
    )


def replay_game(path: Path) -> int:
    """Replay a retained complete attempt without a policy; reject partial logs."""
    with path.open() as source:
        header = json.loads(next(source))
        env = managym.Env(seed=header["seed"], skip_trivial=header["skip_trivial"])
        env.reset(Match(MatchConfig(**header["match"])).to_rust())
        count = 0
        complete = False
        for line in source:
            row = json.loads(line)
            if complete:
                raise ValueError("data follows terminal compound receipt")
            if "command" in row:
                observed = json.loads(
                    env.execute_semantic_command_json(json.dumps(row["command"]))
                )
                if observed != row["transition"]:
                    raise ValueError("compound replay transition mismatch")
                count += 1
            else:
                if (
                    not env.is_game_over()
                    or env.winner_index() != row["winner"]
                    or env.state_digest() != row["state_digest"]
                ):
                    raise ValueError("compound replay terminal mismatch")
                complete = True
        if not complete:
            raise ValueError("compound attempt is incomplete")
    return count
