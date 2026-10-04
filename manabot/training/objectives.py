"""Explicit self-play estimators, regularization references and sample selection."""

import math

import torch


def reference_distribution(obs, kind):
    if kind not in {"uniform", "action_type_uniform"}:
        raise ValueError(f"unknown reference: {kind}")
    valid = obs["actions_valid"] > 0
    if not valid.any(-1).all():
        raise ValueError("reference requires at least one legal offer per row")
    if kind == "uniform":
        return valid / valid.sum(-1, keepdim=True)
    action_types = obs["actions"][..., :-1]
    types = action_types.argmax(-1)
    # Count each type once per row without an offers-by-offers matrix.
    counts = action_types.new_zeros(*types.shape[:-1], action_types.shape[-1])
    counts.scatter_add_(-1, types, valid.to(counts.dtype))
    weights = valid / counts.gather(-1, types).clamp_min(1)
    return weights / weights.sum(-1, keepdim=True)


def selected_rows(advantages, fraction, minimum):
    magnitude = advantages.abs()
    count = max(1, math.ceil(len(magnitude) * fraction))
    indices = torch.argsort(magnitude, descending=True, stable=True)[:count]
    return indices[magnitude[indices] >= minimum]


@torch.no_grad()
def update_ema(averaged, learner, rate):
    """Advance once per collection/update iteration, including filtered skips.

    Initialize with a deep copy of the learner. Parameters are averaged;
    buffers describe the current model and are copied exactly. Collection
    continues to use the learner, never this evaluation-only model.
    """
    if not 0 <= rate < 1:
        raise ValueError("EMA rate must be in [0, 1)")
    for name, dest in averaged.named_parameters():
        dest.lerp_(learner.get_parameter(name), 1 - rate)
    for name, dest in averaged.named_buffers():
        dest.copy_(learner.get_buffer(name))


def update_iteration(trainer, batch, learning, progress, rng):
    """Optimize one fresh collector batch on the existing trainer and Adam owner."""
    from manabot.sim.net_opponent import transition_gae

    dev = trainer.experiment.device
    obs = trainer._obs_to_tensors(batch.obs, dev)
    values = torch.as_tensor(batch.values, device=dev)
    rewards = torch.as_tensor(batch.rewards, device=dev)
    ends = torch.as_tensor(batch.dones, device=dev)
    with torch.no_grad():
        next_value = trainer.agent.get_value(
            trainer._obs_to_tensors(batch.next_obs, dev)
        )
        advantages, _ = transition_gae(
            rewards, values, ends, next_value, learning.gamma, learning.policy_lambda
        )
        _, returns = transition_gae(
            rewards, values, ends, next_value, learning.gamma, learning.value_lambda
        )
    obs = {k: v.flatten(0, 1) for k, v in obs.items()}
    advantages, returns = advantages.flatten(), returns.flatten()
    actions = torch.as_tensor(batch.actions, device=dev).flatten()
    old_logs = torch.as_tensor(batch.logprobs, device=dev).flatten()
    behavior = torch.as_tensor(batch.probabilities, device=dev).flatten(0, 1)
    selected = selected_rows(
        advantages, learning.retained_fraction, learning.min_advantage
    )
    diagnostics = {
        "rows": len(advantages),
        "retained": len(selected),
        "optimizer_exposures": 0,
        "advantage_abs_mean": float(advantages.abs().mean()),
        "bootstrapped_tail_fraction": float((~ends[-1]).float().mean()),
        "learning_rate": learning.learning_rate.at(progress),
        "tau": learning.tau.at(progress),
        "schedule_progress": progress,
        "schedule_clock": "elapsed-training-budget-fraction",
        "gamma": learning.gamma,
        "policy_lambda": learning.policy_lambda,
        "value_lambda": learning.value_lambda,
        "reference": learning.reference,
        "collection_kl_coefficient": learning.collection_kl,
        "retained_fraction": len(selected) / len(advantages),
        "return_mean": float(returns.mean()),
        "value_residual_abs_mean": float((returns - values.flatten()).abs().mean()),
    }
    chosen_types = obs["actions"][
        torch.arange(len(actions), device=dev), actions, :-1
    ].argmax(-1)
    diagnostics["action_types"] = torch.bincount(
        chosen_types, minlength=obs["actions"].shape[-1] - 1
    ).tolist()
    diagnostics["selected_action_types"] = torch.bincount(
        chosen_types[selected], minlength=obs["actions"].shape[-1] - 1
    ).tolist()
    if not len(selected):
        diagnostics["skipped"] = "empty advantage filter"
        return diagnostics
    advantages = (advantages - advantages[selected].mean()) / advantages[selected].std(
        unbiased=False
    ).clamp_min(1e-8)
    for group in trainer.optimizer.param_groups:
        group["lr"] = diagnostics["learning_rate"]
    reference = reference_distribution(obs, learning.reference)
    size = max(1, math.ceil(len(selected) / learning.minibatches))
    for _ in range(learning.epochs):
        order = selected[torch.as_tensor(rng.permutation(len(selected)), device=dev)]
        for indices in order.split(size):
            logits, value = trainer.agent({k: v[indices] for k, v in obs.items()})
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(actions[indices]) - old_logs[indices]).exp()
            adv = advantages[indices]
            policy = torch.maximum(
                -adv * ratio, -adv * ratio.clamp(1 - learning.clip, 1 + learning.clip)
            ).mean()
            probs = dist.probs
            logs = probs.clamp_min(1e-12).log()
            kl_ref = (
                (probs * (logs - reference[indices].clamp_min(1e-12).log()))
                .sum(-1)
                .mean()
            )
            kl_behavior = (
                (probs * (logs - behavior[indices].clamp_min(1e-12).log()))
                .sum(-1)
                .mean()
            )
            value_loss = 0.5 * (value.flatten() - returns[indices]).square().mean()
            loss = (
                policy
                + learning.value_weight * value_loss
                + diagnostics["tau"] * kl_ref
                + learning.collection_kl * kl_behavior
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite self-play objective")
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainer.agent.parameters(), learning.max_grad_norm
            )
            trainer.optimizer.step()
            diagnostics["optimizer_exposures"] += len(indices)
            diagnostics.update(
                loss=float(loss.detach()),
                entropy=float(dist.entropy().mean().detach()),
                reference_kl=float(kl_ref.detach()),
                collection_kl=float(kl_behavior.detach()),
            )
    return diagnostics
