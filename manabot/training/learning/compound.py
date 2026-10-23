"""Optimize compound policies from complete retained games.

Each seat owns its trajectory. Grouped credit uses one value/discount/ratio per
submission; sequential credit uses one per decoder factor. Outcome targets are
terminal Monte Carlo returns; bootstrapped targets use transition-end GAE.
Behavior tapes and targets are detached; recomputed policy/value outputs receive
gradients. Collection and exact replay live in training.rollout.compound."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.returns import transition_gae
from manabot.training.learning.selection import selected_rows
from manabot.training.regime import Learning
from manabot.training.rollout.compound import CompoundGame


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
    agent: ActorCritic,
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
