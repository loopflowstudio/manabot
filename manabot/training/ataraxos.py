"""Ataraxos move learning (Nature supplement S3.4, equations 5–6).

The damped update is a clipped likelihood-ratio surrogate with two reverse
KL penalties, not a novel replacement for that surrogate. Targets, behavior
likelihoods and the magnet are constants. Only current policy/critic logits
receive gradients. MTG uses same-viewer transitions and an action-type magnet;
see docs/ataraxos.md for the fidelity boundary and unresolved source details.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch, transition_gae
from manabot.training.models import AtaraxosMoveLearning
from manabot.training.references import reference_distribution


def _check_distribution(probabilities: Tensor, valid: Tensor) -> None:
    if probabilities.shape != valid.shape or probabilities.ndim < 2:
        raise ValueError("distribution and legal support shapes differ")
    if not valid.any(-1).all():
        raise ValueError("distribution requires nonempty legal support")
    if (
        not torch.isfinite(probabilities).all()
        or (probabilities[valid] <= 0).any()
        or (probabilities[~valid] != 0).any()
        or not torch.allclose(
            probabilities.sum(-1),
            torch.ones_like(probabilities.sum(-1)),
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise ValueError("distribution must normalize with positive legal support")


def damped_policy_loss(
    logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    behavior: Tensor,
    reference: Tensor,
    valid: Tensor,
    *,
    clip: float,
    collection_kl: float,
    tau: float,
) -> Tensor:
    """Equation (6), per row [B], over padded categorical offers [B,A].

    Ratios use the actual action likelihood at collection, even after earlier
    minibatches change the learner. The reverse KLs sum all legal actions.
    This is per-action reuse correction, not trajectory importance weighting.
    """
    valid = valid.bool()
    if logits.ndim != 2 or logits.shape != valid.shape:
        raise ValueError("policy logits require [B,A] legal support")
    if actions.shape != logits.shape[:1] or advantages.shape != actions.shape:
        raise ValueError("actions and advantages require [B]")
    if not 0 < clip < 1 or collection_kl < 0 or tau < 0:
        raise ValueError("invalid damping coefficients")
    _check_distribution(behavior, valid)
    _check_distribution(reference, valid)
    if not torch.isfinite(logits[valid]).all() or not torch.isfinite(advantages).all():
        raise ValueError("nonfinite policy logits or advantages")
    if ((actions < 0) | (actions >= logits.shape[-1])).any():
        raise ValueError("action index outside legal support")
    if not valid.gather(-1, actions[:, None]).all():
        raise ValueError("sampled action is not legal")
    behavior, reference, advantages = (
        behavior.detach(),
        reference.detach(),
        advantages.detach(),
    )
    # Mask log terms as well as probabilities, avoiding 0 * -inf in KL.
    logs = logits.masked_fill(~valid, -torch.inf).log_softmax(-1)
    probabilities = logs.exp()
    logs = logs.masked_fill(~valid, 0)
    behavior_logs = behavior.masked_fill(~valid, 1).log()
    reference_logs = reference.masked_fill(~valid, 1).log()
    ratio = (logs - behavior_logs).gather(-1, actions[:, None]).squeeze(-1).exp()
    surrogate = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages
    )
    return (
        surrogate
        + collection_kl * (probabilities * (logs - behavior_logs)).sum(-1)
        + tau * (probabilities * (logs - reference_logs)).sum(-1)
    )


@torch.no_grad()
def categorical_lambda_returns(
    probabilities: Tensor,
    rewards: Tensor,
    ends: Tensor,
    bootstrap: Tensor,
    trace_lambda: float,
) -> Tensor:
    """Undiscounted outcome mixtures [T,E,3], ordered loss/draw/win.

    Each row ends at the next decision of the SAME player or at terminal.
    Terminal targets are one-hot signed outcomes. A nonterminal tail uses the
    exact paused next observation; reset observations never cross end markers.
    For earlier rows mix (1-lambda)*next_prediction + lambda*next_return.
    No policy receives true hidden-state input from these outcome labels.
    """
    if probabilities.ndim != 3 or probabilities.shape[-1] != 3:
        raise ValueError("outcome probabilities require [T,E,3]")
    if (
        rewards.shape != probabilities.shape[:2]
        or ends.shape != rewards.shape
        or bootstrap.shape != probabilities.shape[1:]
        or not len(rewards)
        or not 0 <= trace_lambda <= 1
    ):
        raise ValueError("invalid outcome target shapes or lambda")
    for distribution in (probabilities, bootstrap):
        if (
            not torch.isfinite(distribution).all()
            or (distribution < 0).any()
            or not torch.allclose(
                distribution.sum(-1),
                torch.ones_like(distribution.sum(-1)),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise ValueError("outcome probabilities must normalize")
    if (
        not torch.isfinite(rewards).all()
        or not ((rewards == -1) | (rewards == 0) | (rewards == 1)).all()
    ):
        raise ValueError("outcome rewards must be signed win/loss/draw")
    ends = ends.bool()
    if (rewards[~ends] != 0).any():
        raise ValueError("nonterminal outcome reward must be zero")
    targets = torch.empty_like(probabilities)
    tail = bootstrap
    for step in reversed(range(len(rewards))):
        next_prediction = (
            bootstrap if step == len(rewards) - 1 else probabilities[step + 1]
        )
        continuation = (1 - trace_lambda) * next_prediction + trace_lambda * tail
        terminal = F.one_hot((rewards[step] + 1).long(), 3).to(probabilities.dtype)
        tail = torch.where(ends[step, :, None], terminal, continuation)
        targets[step] = tail
    return targets


@dataclass(frozen=True)
class MoveTargets:
    advantages: Tensor
    values: Tensor
    selected: Tensor


def selected_moves(advantages: Tensor, quantile: float, minimum: float) -> Tensor:
    """Supplement S3.4 inclusive magnitude filter, including quantile ties."""
    magnitude = advantages.detach().abs()
    threshold = max(float(torch.quantile(magnitude, quantile)), minimum)
    return magnitude >= threshold


@torch.no_grad()
def _targets(
    trainer: NetOpponentTrainer,
    batch: RolloutBatch,
    learning: AtaraxosMoveLearning,
) -> MoveTargets:
    device = trainer.experiment.device
    rewards = torch.as_tensor(batch.rewards, device=device)
    ends = torch.as_tensor(batch.dones, device=device)
    values = torch.as_tensor(batch.values, device=device)
    next_obs = trainer._obs_to_tensors(batch.next_obs, device)
    _, next_raw = trainer.agent.forward_distribution(next_obs)
    if trainer.agent.hypers.value_kind == "categorical_wdl":
        if batch.outcome_probabilities is None:
            raise ValueError(
                "categorical update requires collection outcome probabilities"
            )
        probabilities = torch.as_tensor(batch.outcome_probabilities, device=device)
        if not torch.allclose(
            values, probabilities[..., 2] - probabilities[..., 0], atol=1e-5
        ):
            raise ValueError(
                "saved value differs from collection outcome probabilities"
            )
        bootstrap = next_raw.softmax(-1)
        value_targets = categorical_lambda_returns(
            probabilities, rewards, ends, bootstrap, learning.value_lambda
        )
        next_value = bootstrap[:, 2] - bootstrap[:, 0]
    else:
        next_value = next_raw.squeeze(-1)
        _, value_targets = transition_gae(
            rewards, values, ends, next_value, 1.0, learning.value_lambda
        )
    advantages, _ = transition_gae(
        rewards, values, ends, next_value, 1.0, learning.policy_lambda
    )
    # Inclusive quantile keeps all ties, unlike a top-k filter with fixed count.
    selected = selected_moves(
        advantages, learning.advantage_quantile, learning.min_advantage
    )
    return MoveTargets(advantages, value_targets, selected)


def update_move_iteration(
    trainer: NetOpponentTrainer,
    batch: RolloutBatch,
    learning: AtaraxosMoveLearning,
    iteration: int,
) -> dict[str, int | float | str]:
    """One fresh, timestep-grouped epoch with frozen collection targets.

    EMA remains owned by execute_regime and advances once even on an empty
    filter. No batch shuffling, advantage normalization, replay or hidden truth.
    Scalar value is an explicit MSE ablation; categorical uses equation (5).
    """
    targets = _targets(trainer, batch, learning)
    device = trainer.experiment.device
    observations = trainer._obs_to_tensors(batch.obs, device)
    actions = torch.as_tensor(batch.actions, device=device)
    behavior = torch.as_tensor(batch.probabilities, device=device)
    old_logs = torch.as_tensor(batch.logprobs, device=device)
    valid = observations["actions_valid"] > 0
    _check_distribution(behavior, valid)
    sampled = behavior.gather(-1, actions[..., None]).squeeze(-1)
    if (sampled <= 0).any() or not torch.allclose(
        sampled.log(), old_logs, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("saved action likelihood differs from collection distribution")
    rate, tau = learning.rates(iteration)
    diagnostics: dict[str, int | float | str] = {
        "gradient": learning.gradient,
        "value_kind": trainer.agent.hypers.value_kind,
        "rows": actions.numel(),
        "retained": int(targets.selected.sum()),
        "optimizer_exposures": 0,
        "iteration": iteration,
        "learning_rate": rate,
        "tau": tau,
        "schedule_clock": "collection-update-iteration",
        "reference": learning.reference,
        "advantage_abs_mean": float(targets.advantages.abs().mean()),
        "bootstrapped_tail_fraction": float(
            (~torch.as_tensor(batch.dones[-1])).float().mean()
        ),
    }
    for group in trainer.optimizer.param_groups:
        group["lr"] = rate
    if not targets.selected.any():
        diagnostics["skipped"] = "empty advantage filter"
        return diagnostics
    # Each learner timestep is one minibatch, preserving the paper's grouping
    # in MTG learner-decision units rather than Stratego simulator plies.
    for step, selected in enumerate(targets.selected):
        if not selected.any():
            continue
        obs = {key: value[step, selected] for key, value in observations.items()}
        logits, value_logits = trainer.agent.forward_distribution(obs)
        policy_loss = damped_policy_loss(
            logits,
            actions[step, selected],
            targets.advantages[step, selected],
            behavior[step, selected],
            reference_distribution(obs, learning.reference),
            valid[step, selected],
            clip=learning.clip,
            collection_kl=learning.collection_kl,
            tau=tau,
        ).mean()
        if trainer.agent.hypers.value_kind == "categorical_wdl":
            value_loss = (
                -(targets.values[step, selected] * value_logits.log_softmax(-1))
                .sum(-1)
                .mean()
            )
        else:
            # Representation ablation, not the paper's outcome objective.
            value_loss = (
                (value_logits.squeeze(-1) - targets.values[step, selected])
                .square()
                .mean()
            )
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite Ataraxos move objective")
        trainer.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainer.agent.parameters(), learning.max_grad_norm, error_if_nonfinite=True
        )
        trainer.optimizer.step()
        diagnostics["optimizer_exposures"] = int(
            diagnostics["optimizer_exposures"]
        ) + int(selected.sum())
        diagnostics.update(
            loss=float(loss.detach()),
            value_loss=float(value_loss.detach()),
            policy_loss=float(policy_loss.detach()),
            gradient_norm=float(norm),
        )
    return diagnostics
