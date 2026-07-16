"""Joint policy/value supervision from search-generated self-play.

One shared encoder can learn a policy from flat-search scores, MCTS root
visits, or chosen actions, and a value from terminal outcomes or teacher root
values. Target kinds stay explicit so a score softmax is never misrepresented
as a visit distribution and teacher-value imitation is not mistaken for
outcome calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import (
    OBS_KEYS,
    ROOT_VALUE_KEY,
    SCORE_KEY,
    VISIT_COUNT_KEY,
    soft_targets_from_scores,
    split_by_game,
)

SCORE_SOFTMAX_TARGET = "score_softmax"
VISIT_DISTRIBUTION_TARGET = "visit_distribution"
CHOSEN_ACTION_TARGET = "chosen_action"
TERMINAL_OUTCOME_TARGET = "terminal_outcome"
ROOT_VALUE_TARGET = "root_value"


@dataclass(frozen=True)
class SearchSupervisedMetrics:
    policy_loss: float
    policy_kl: float
    policy_accuracy: float
    policy_accuracy_nontrivial: float
    policy_target_entropy: float
    uniform_policy_probability: float
    value_loss: float
    value_brier: float
    value_accuracy: float
    value_rows: int


@dataclass(frozen=True)
class SearchSupervisedEpochStats:
    epoch: int
    train_policy_loss: float
    train_value_loss: float
    train_total_loss: float
    validation: SearchSupervisedMetrics


def outcome_targets(
    winner: np.ndarray, seat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (usable mask, deciding-player win targets).

    Winner-less rows are excluded from value supervision.  The policy target
    remains usable because search produced it before the game outcome became
    unavailable.
    """

    winner = np.asarray(winner, dtype=np.int64)
    seat = np.asarray(seat, dtype=np.int64)
    if winner.shape != seat.shape:
        raise ValueError("winner and seat columns must have identical shapes")
    usable = winner >= 0
    targets = (winner == seat).astype(np.float32)
    return usable, targets


def value_targets_from_dataset(
    dataset: dict[str, np.ndarray], value_target_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    if value_target_kind == TERMINAL_OUTCOME_TARGET:
        return outcome_targets(dataset["winner"], dataset["seat"])
    if value_target_kind == ROOT_VALUE_TARGET:
        values = np.asarray(dataset[ROOT_VALUE_KEY], dtype=np.float32)
        usable = np.isfinite(values)
        if ((values[usable] < 0) | (values[usable] > 1)).any():
            raise ValueError("root_value targets must be in [0, 1]")
        return usable, np.nan_to_num(values, nan=0.0)
    raise ValueError(f"unsupported value target kind: {value_target_kind}")


def _validate_dataset(
    dataset: dict[str, np.ndarray],
    *,
    policy_target_kind: str,
    value_target_kind: str,
) -> None:
    required = set(OBS_KEYS) | {
        "action",
        "game_index",
        "num_valid",
        "seat",
        "winner",
    }
    if policy_target_kind == SCORE_SOFTMAX_TARGET:
        required.add(SCORE_KEY)
    elif policy_target_kind == VISIT_DISTRIBUTION_TARGET:
        required.add(VISIT_COUNT_KEY)
    elif policy_target_kind == CHOSEN_ACTION_TARGET:
        pass
    else:
        raise ValueError(f"unsupported policy target kind: {policy_target_kind}")
    if value_target_kind == ROOT_VALUE_TARGET:
        required.add(ROOT_VALUE_KEY)
    elif value_target_kind != TERMINAL_OUTCOME_TARGET:
        raise ValueError(f"unsupported value target kind: {value_target_kind}")
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"search-supervised dataset is missing {missing}")
    rows = len(dataset["action"])
    if rows == 0:
        raise ValueError("search-supervised dataset is empty")
    wrong = sorted(key for key in required if len(dataset[key]) != rows)
    if wrong:
        raise ValueError(f"search-supervised columns have inconsistent rows: {wrong}")
    valid_actions = np.asarray(dataset["actions_valid"]) > 0
    if valid_actions.ndim != 2:
        raise ValueError("actions_valid must have shape (decisions, max_actions)")
    actions = np.asarray(dataset["action"], dtype=np.int64)
    in_range = (actions >= 0) & (actions < valid_actions.shape[1])
    chosen_is_legal = np.zeros(rows, dtype=bool)
    chosen_is_legal[in_range] = valid_actions[
        np.arange(rows)[in_range], actions[in_range]
    ]
    if not chosen_is_legal.all():
        raise ValueError(
            "every teacher action must be present in the encoded legal mask"
        )
    encoded_counts = valid_actions.sum(axis=1)
    if not np.array_equal(encoded_counts, np.asarray(dataset["num_valid"])):
        raise ValueError("num_valid must exactly match the encoded legal mask")
    if policy_target_kind == SCORE_SOFTMAX_TARGET:
        scores = np.asarray(dataset[SCORE_KEY])
        if scores.shape != valid_actions.shape:
            raise ValueError("scores must have shape (decisions, max_actions)")
        if not np.isfinite(scores).all():
            raise ValueError("scores contain non-finite values")
        if (scores >= 0).sum(axis=1).min() == 0:
            raise ValueError("every decision must have at least one scored action")
        if not np.array_equal(scores >= 0, valid_actions):
            raise ValueError("scored actions must exactly match the encoded legal mask")
    elif policy_target_kind == VISIT_DISTRIBUTION_TARGET:
        visits = np.asarray(dataset[VISIT_COUNT_KEY])
        if visits.shape != valid_actions.shape:
            raise ValueError("visit_counts must have shape (decisions, max_actions)")
        if not np.isfinite(visits).all() or (visits < 0).any():
            raise ValueError("visit_counts must be finite and non-negative")
        if (visits[~valid_actions] > 0).any():
            raise ValueError("visit_counts must be zero outside the legal mask")
        if (visits.sum(axis=1) <= 0).any():
            raise ValueError("every decision must have a positive visit count")
    value_targets_from_dataset(dataset, value_target_kind)


def _batch_observations(
    dataset: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(dataset[key][indices], dtype=torch.float32, device=device)
        for key in OBS_KEYS
    }


def _policy_targets(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    target_kind: str,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    if target_kind == SCORE_SOFTMAX_TARGET:
        scores = torch.as_tensor(
            dataset[SCORE_KEY][indices], dtype=torch.float32, device=device
        )
        return soft_targets_from_scores(scores, temperature)
    if target_kind == CHOSEN_ACTION_TARGET:
        actions = torch.as_tensor(
            dataset["action"][indices], dtype=torch.long, device=device
        )
        width = int(dataset["actions_valid"].shape[1])
        return torch.nn.functional.one_hot(actions, num_classes=width).to(torch.float32)
    visits = torch.as_tensor(
        dataset[VISIT_COUNT_KEY][indices], dtype=torch.float32, device=device
    )
    valid = torch.as_tensor(
        dataset["actions_valid"][indices] > 0, dtype=torch.bool, device=device
    )
    visits = visits.masked_fill(~valid, 0.0)
    return visits / visits.sum(dim=-1, keepdim=True).clamp_min(1.0)


@torch.no_grad()
def evaluate_search_supervised(
    agent: Agent,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    policy_temperature: float,
    policy_target_kind: str = SCORE_SOFTMAX_TARGET,
    value_target_kind: str = TERMINAL_OUTCOME_TARGET,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> SearchSupervisedMetrics:
    """Evaluate the declared policy and value targets on fixed decision rows."""

    _validate_dataset(
        dataset,
        policy_target_kind=policy_target_kind,
        value_target_kind=value_target_kind,
    )
    dev = torch.device(device)
    agent.eval()
    value_usable, value_targets = value_targets_from_dataset(dataset, value_target_kind)
    policy_loss = 0.0
    policy_correct = 0
    policy_nontrivial_correct = 0
    policy_nontrivial_rows = 0
    policy_entropy = 0.0
    uniform_probability = 0.0
    value_loss = 0.0
    value_brier = 0.0
    value_correct = 0
    value_rows = 0

    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        obs = _batch_observations(dataset, batch, dev)
        logits, value_logits = agent.forward(obs)
        policy_target = _policy_targets(
            dataset, batch, policy_target_kind, policy_temperature, dev
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        row_loss = -(policy_target * log_probs).sum(dim=-1)
        policy_loss += float(row_loss.sum().item())
        policy_entropy += float(
            (-(policy_target * policy_target.clamp_min(1e-12).log()).sum(dim=-1))
            .sum()
            .item()
        )
        predictions = logits.argmax(dim=-1).cpu().numpy()
        targets = np.asarray(dataset["action"][batch], dtype=np.int64)
        hits = predictions == targets
        policy_correct += int(hits.sum())
        nontrivial = np.asarray(dataset["num_valid"][batch]) > 1
        policy_nontrivial_correct += int(hits[nontrivial].sum())
        policy_nontrivial_rows += int(nontrivial.sum())
        uniform_probability += float(
            np.sum(1.0 / np.maximum(1, np.asarray(dataset["num_valid"][batch])))
        )

        usable = value_usable[batch]
        if usable.any():
            target = torch.as_tensor(
                value_targets[batch][usable], dtype=torch.float32, device=dev
            )
            selected_logits = value_logits[torch.as_tensor(usable, device=dev)]
            value_loss += float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    selected_logits, target, reduction="sum"
                ).item()
            )
            probabilities = torch.sigmoid(selected_logits)
            value_brier += float(((probabilities - target) ** 2).sum().item())
            value_correct += int(
                ((probabilities >= 0.5) == (target >= 0.5)).sum().item()
            )
            value_rows += int(usable.sum())

    rows = max(1, len(indices))
    return SearchSupervisedMetrics(
        policy_loss=policy_loss / rows,
        policy_kl=max(0.0, (policy_loss - policy_entropy) / rows),
        policy_accuracy=policy_correct / rows,
        policy_accuracy_nontrivial=policy_nontrivial_correct
        / max(1, policy_nontrivial_rows),
        policy_target_entropy=policy_entropy / rows,
        uniform_policy_probability=uniform_probability / rows,
        value_loss=value_loss / max(1, value_rows),
        value_brier=value_brier / max(1, value_rows),
        value_accuracy=value_correct / max(1, value_rows),
        value_rows=value_rows,
    )


def train_search_supervised(
    dataset: dict[str, np.ndarray],
    *,
    policy_temperature: float = 0.05,
    policy_target_kind: str = SCORE_SOFTMAX_TARGET,
    value_target_kind: str = TERMINAL_OUTCOME_TARGET,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 512,
    val_fraction: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    agent_hypers: AgentHypers | None = None,
    initial_agent_state: dict[str, Any] | None = None,
    log: bool = False,
) -> tuple[
    Agent,
    ObservationSpace,
    SearchSupervisedMetrics,
    list[SearchSupervisedEpochStats],
]:
    """Train one Agent jointly on explicitly selected policy and value targets.

    Callers are responsible for matched experimental arms. The Teacher-0
    runner uses ``value_weight=0`` to isolate value supervision; Teacher-1
    holds root-value supervision fixed while changing the policy target.
    """

    _validate_dataset(
        dataset,
        policy_target_kind=policy_target_kind,
        value_target_kind=value_target_kind,
    )
    if policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive")
    if policy_weight < 0 or value_weight < 0 or policy_weight + value_weight <= 0:
        raise ValueError("loss weights must be non-negative and not both zero")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    torch.manual_seed(seed)
    dev = torch.device(device)
    obs_space = ObservationSpace()
    agent = Agent(obs_space, agent_hypers or AgentHypers()).to(dev)
    if initial_agent_state is not None:
        agent.load_state_dict(initial_agent_state)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    train_idx, val_idx = split_by_game(dataset, val_fraction=val_fraction, seed=seed)
    value_usable, value_targets = value_targets_from_dataset(dataset, value_target_kind)
    initial_validation = evaluate_search_supervised(
        agent,
        dataset,
        val_idx,
        policy_temperature=policy_temperature,
        policy_target_kind=policy_target_kind,
        value_target_kind=value_target_kind,
        batch_size=batch_size,
        device=dev,
    )
    rng = np.random.default_rng(seed)
    history: list[SearchSupervisedEpochStats] = []

    for epoch in range(epochs):
        agent.train()
        order = train_idx.copy()
        rng.shuffle(order)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        value_rows = 0
        total_loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            obs = _batch_observations(dataset, batch, dev)
            logits, value_logits = agent.forward(obs)
            policy_target = _policy_targets(
                dataset, batch, policy_target_kind, policy_temperature, dev
            )
            policy_loss = torch.nn.functional.cross_entropy(logits, policy_target)

            usable = value_usable[batch]
            if usable.any():
                target = torch.as_tensor(
                    value_targets[batch][usable], dtype=torch.float32, device=dev
                )
                value_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    value_logits[torch.as_tensor(usable, device=dev)], target
                )
                batch_value_rows = int(usable.sum())
            else:
                value_loss = value_logits.sum() * 0.0
                batch_value_rows = 0
            total_loss = policy_weight * policy_loss + value_weight * value_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            policy_loss_sum += float(policy_loss.item()) * len(batch)
            value_loss_sum += float(value_loss.item()) * batch_value_rows
            value_rows += batch_value_rows
            total_loss_sum += float(total_loss.item()) * len(batch)

        validation = evaluate_search_supervised(
            agent,
            dataset,
            val_idx,
            policy_temperature=policy_temperature,
            policy_target_kind=policy_target_kind,
            value_target_kind=value_target_kind,
            batch_size=batch_size,
            device=dev,
        )
        stats = SearchSupervisedEpochStats(
            epoch=epoch,
            train_policy_loss=policy_loss_sum / max(1, len(order)),
            train_value_loss=value_loss_sum / max(1, value_rows),
            train_total_loss=total_loss_sum / max(1, len(order)),
            validation=validation,
        )
        history.append(stats)
        if log:
            print(
                f"  epoch {epoch}: policy {stats.train_policy_loss:.4f}/"
                f"{validation.policy_loss:.4f} kl {validation.policy_kl:.4f} "
                f"value {stats.train_value_loss:.4f}/"
                f"{validation.value_loss:.4f} brier {validation.value_brier:.4f} "
                f"policy_acc {validation.policy_accuracy_nontrivial:.4f}",
                flush=True,
            )

    return agent, obs_space, initial_validation, history
