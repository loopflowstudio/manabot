"""Complete-game compound self-play and explicit decoder credit boundaries.

Collection freezes weights for whole games and retains every canonical Command
and receipt. Each seat has its own terminal-return trajectory. Grouped credit
uses one value/discount/likelihood ratio per complete submission; sequential
credit uses one per decoder factor. Neither mode groups distinct observations.
Outcome credit is a Monte Carlo terminal return; bootstrapped credit uses the
ETU-90 transition-end GAE estimator. Targets and behavior tapes are detached;
only the recomputed policy and prefix values receive optimizer gradients.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from manabot.env import Match
from manabot.infra.hypers import MatchHypers
from manabot.model.agent import Agent
from manabot.sim.compound import CompoundDecision, sample_compound
from manabot.sim.net_opponent import transition_gae
import managym

from .models import Learning
from .selection import selected_rows


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
                int((probs > 0).sum()) == 1
                for probs in decision.output.probabilities
            )
            self.decision_seconds += decision.seconds
            self.max_decision_seconds = max(
                self.max_decision_seconds, decision.seconds
            )


def collect_game(
    agent: Agent,
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
    raw, _ = env.reset(match.to_rust())
    decisions: list[EpisodeDecision] = []
    count = 0
    with path.open("x") as output:
        output.write(
            json.dumps(
                {
                    "seed": seed,
                    "match": match.hypers.model_dump(mode="json"),
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
                result = env.execute_semantic_command_json(command.to_json())
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
        env.reset(Match(MatchHypers(**header["match"])).to_rust())
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


@dataclass(frozen=True)
class Credit:
    advantages: Tensor
    returns: Tensor
    old_logs: Tensor


def episode_credit(
    game: CompoundGame,
    *,
    grouped: bool,
    estimator: Literal["outcome", "bootstrapped"],
    learning: Learning,
) -> tuple[Credit, ...]:
    """Return aligned, detached targets; discounts count the chosen credit units.

    At gamma=1 the terminal objective is identical across grouping. With gamma<1
    the time preference changes with factorization and must be called a confound.
    A terminal value never bootstraps from another seat, game, or incomplete tail.
    """
    targets: dict[int, Credit] = {}
    for actor in (0, 1):
        indexes = [
            index for index, item in enumerate(game.decisions) if item.actor == actor
        ]
        if not indexes:
            continue
        values = [
            game.decisions[index].decision.output.values[:1]
            if grouped
            else game.decisions[index].decision.output.values
            for index in indexes
        ]
        all_values = torch.cat(values).detach()
        reward = 0.0 if game.winner is None else (1.0 if game.winner == actor else -1.0)
        rewards = torch.zeros_like(all_values)
        rewards[-1] = reward
        ends = torch.zeros_like(all_values, dtype=torch.bool)
        ends[-1] = True
        policy_lambda = 1.0 if estimator == "outcome" else learning.policy_lambda
        value_lambda = 1.0 if estimator == "outcome" else learning.value_lambda
        with torch.no_grad():
            advantage, _ = transition_gae(
                rewards[:, None],
                all_values[:, None],
                ends[:, None],
                all_values.new_zeros(1),
                learning.gamma,
                policy_lambda,
            )
            _, returns = transition_gae(
                rewards[:, None],
                all_values[:, None],
                ends[:, None],
                all_values.new_zeros(1),
                learning.gamma,
                value_lambda,
            )
        offset = 0
        for index, value in zip(indexes, values, strict=True):
            logs = game.decisions[index].decision.output.log_probs
            targets[index] = Credit(
                advantage.flatten()[offset : offset + len(value)],
                returns.flatten()[offset : offset + len(value)],
                logs.sum().reshape(1) if grouped else logs,
            )
            offset += len(value)
    return tuple(targets[index] for index in range(len(game.decisions)))


@dataclass(frozen=True)
class CompoundUpdate:
    optimizer_exposures: int
    losses: list[float]


def optimize_games(
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    games: list[CompoundGame],
    learning: Learning,
    *,
    grouped: bool,
    estimator: Literal["outcome", "bootstrapped"],
    progress: float,
    generator: torch.Generator,
    check: Callable[[], None],
) -> CompoundUpdate:
    """PPO on complete submissions or conditional factors, never tape fragments.

    Conditional reverse KLs are measured at retained behavior prefixes, summed
    for a grouped row. They are sampled-prefix regularizers, not an exact joint
    reverse KL (which would require integrating over all new-policy prefixes).
    Uniform references are conditional legal uniforms, not uniform over subsets.
    """
    decisions = [item.decision for game in games for item in game.decisions]
    if not decisions:
        return CompoundUpdate(0, [])
    credits = [
        credit
        for game in games
        for credit in episode_credit(
            game, grouped=grouped, estimator=estimator, learning=learning
        )
    ]
    advantages = torch.cat([credit.advantages for credit in credits])
    chosen = selected_rows(
        advantages, learning.retained_fraction, learning.min_advantage
    )
    if not len(chosen):
        return CompoundUpdate(0, [])
    selected = torch.zeros(len(advantages), dtype=torch.bool)
    selected[chosen] = True
    normalized = (advantages - advantages[chosen].mean()) / advantages[chosen].std(
        unbiased=False
    ).clamp_min(1e-8)
    masks = selected.split([len(credit.advantages) for credit in credits])
    normalized_rows = normalized.split([len(credit.advantages) for credit in credits])
    eligible = [index for index, mask in enumerate(masks) if mask.any()]
    size = max(1, math.ceil(len(eligible) / learning.minibatches))
    for group in optimizer.param_groups:
        group["lr"] = learning.learning_rate.at(progress)
    exposures = 0
    losses: list[float] = []
    for _ in range(learning.epochs):
        order = torch.randperm(len(eligible), generator=generator).tolist()
        for start in range(0, len(order), size):
            check()
            terms: list[Tensor] = []
            for ordinal in order[start : start + size]:
                index = eligible[ordinal]
                decision, credit = decisions[index], credits[index]
                output = agent.compound(
                    decision.observation, decision.offers, tokens=decision.output.tokens
                )
                logs = output.log_prob.reshape(1) if grouped else output.log_probs
                value = output.values[:1] if grouped else output.values
                ratio = (logs - credit.old_logs).exp()
                advantage = normalized_rows[index]
                policy = torch.maximum(
                    -advantage * ratio,
                    -advantage * ratio.clamp(1 - learning.clip, 1 + learning.clip),
                )
                regularizers: list[Tensor] = []
                for probs, old in zip(
                    output.probabilities, decision.output.probabilities, strict=True
                ):
                    valid = old > 0
                    logs_valid = probs[valid].clamp_min(1e-12).log()
                    kl_old = (probs[valid] * (logs_valid - old[valid].log())).sum()
                    kl_ref = (
                        probs[valid] * (logs_valid + math.log(int(valid.sum())))
                    ).sum()
                    regularizers.append(
                        learning.collection_kl * kl_old
                        + learning.tau.at(progress) * kl_ref
                    )
                regularizer = torch.stack(regularizers)
                if grouped:
                    regularizer = regularizer.sum().reshape(1)
                loss = (
                    policy
                    + learning.value_weight * 0.5 * (value - credit.returns).square()
                    + regularizer
                )
                terms.append(loss[masks[index]])
            loss = torch.cat(terms).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite compound objective")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), learning.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            exposures += sum(term.numel() for term in terms)
            losses.append(float(loss.detach()))
    return CompoundUpdate(exposures, losses)
