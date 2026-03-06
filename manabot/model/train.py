"""
train.py

The primary interface for training is the Trainer class.

PPO training steps:
  - Collect trajectories (rollouts) using vectorized environments.
  - Compute advantages using GAE.
  - Perform PPO updates.
  - Save/load checkpoints.

This version uses CleanRL-style flat rollout tensors with shape (num_steps, num_envs, ...).
"""

import argparse
import datetime
import time
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import wandb

# Local imports
from manabot.env import (
    Match,
    ObservationSpace,
    Reward,
    RustVectorEnv,
)
from manabot.infra import Experiment, Hypers, TrainHypers, getLogger
from manabot.model.agent import Agent

ROLLOUT_HEALTH_KEYS = (
    "truncated_episodes",
    "action_space_truncations",
    "card_space_truncations",
    "permanent_space_truncations",
)


# -----------------------------------------------------------------------------
# Trainer Class
# -----------------------------------------------------------------------------


class Trainer:
    """
    PPO Trainer for manabot.

    Implements the training loop:
      1. Collect trajectories into flat rollout tensors.
      2. Compute advantages with standard GAE.
      3. Flatten to a deterministic batch and run PPO updates.

    Also provides checkpoint saving/loading functionality.
    """

    def __init__(
        self,
        agent: Agent,
        experiment: Experiment,
        env: RustVectorEnv,
        hypers: TrainHypers = TrainHypers(),
    ):
        self.agent = agent.to(experiment.device)
        self.experiment = experiment
        self.env = env
        self.hypers = hypers
        self.global_step = 0

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=hypers.learning_rate,
            eps=1e-5,
        )

        self.logger = getLogger(__name__)

        self.wandb = self.experiment.wandb_run
        self.rollout_health = self._new_rollout_health()
        self.rollout_health_update = self._new_rollout_health()
        self.last_explained_variance = float("nan")

        # Initialize the profiler
        self.profiler = self.experiment.profiler

        if self.wandb:
            self.wandb.summary.update(
                {
                    "max_episode_return": float("-inf"),
                    "best_win_rate": 0.0,
                    "time_to_converge": None,
                }
            )
        self.logger.info("Trainer initialized.")

    def train(self) -> None:
        # Use context manager for root timer
        with self.profiler.track("train"):
            hypers = self.hypers
            env = self.env
            device = self.experiment.device
            batch_size = hypers.num_envs * hypers.num_steps
            num_updates = hypers.total_timesteps // batch_size
            self.start_time = time.time()

            self.logger.info("Resetting environment for training.")
            next_obs, _ = env.reset()
            next_done = torch.zeros(hypers.num_envs, dtype=torch.bool, device=device)

            self.save()
            for update in range(1, num_updates + 1):
                if hypers.anneal_lr:
                    frac = 1.0 - (update - 1) / num_updates
                    lr_now = frac * hypers.learning_rate
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr_now
                    self.logger.info(f"Update {update}: Annealed LR set to {lr_now}")
                else:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(f"Update {update}: LR = {current_lr}")

                self._reset_rollout_health_update()
                (
                    obs_buf,
                    actions_buf,
                    logprobs_buf,
                    rewards_buf,
                    dones_buf,
                    values_buf,
                ) = self._init_rollout_buffers(next_obs)

                self.logger.info("Starting rollout data collection.")
                if self.wandb:
                    self.wandb.log({"rollout/step": 0}, step=self.global_step)

                if update % 10 == 0 and self.agent.hypers.attention_on:
                    self.logger.info("Verifying attention masking mechanism...")
                    attention_valid = self.verify_attention_masking(next_obs)
                    if self.wandb:
                        self.wandb.log(
                            {"verification/attention_valid": int(attention_valid)},
                            step=self.global_step,
                        )

                with self.profiler.track("rollout"):
                    with self.profiler.track("step"):
                        for step in range(hypers.num_steps):
                            for key in obs_buf:
                                obs_buf[key][step] = next_obs[key]
                            (
                                next_obs,
                                reward,
                                next_done,
                                action,
                                logprob,
                                value,
                            ) = self._rollout_step(next_obs)
                            actions_buf[step] = action
                            logprobs_buf[step] = logprob
                            rewards_buf[step] = reward
                            dones_buf[step] = next_done
                            values_buf[step] = value

                    with self.profiler.track("advantage"):
                        with torch.no_grad():
                            next_value = self.agent.get_value(next_obs)
                        advantages, returns = self._compute_gae(
                            rewards_buf,
                            values_buf,
                            dones_buf,
                            next_value,
                            next_done,
                            hypers.gamma,
                            hypers.gae_lambda,
                        )
                        obs, logprobs, actions, advantages, returns, values = (
                            self._flatten_rollout(
                                obs_buf,
                                actions_buf,
                                logprobs_buf,
                                advantages,
                                returns,
                                values_buf,
                            )
                        )
                        self.logger.info(
                            f"Flattened buffer has {logprobs.numel()} transitions."
                        )

                self._log_rollout_health(update)

                clipfracs = []
                approx_kl = 0.0
                actual_batch_size = logprobs.shape[0]
                minibatch_plan = self._build_minibatch_plan(actual_batch_size)
                if minibatch_plan is None:
                    continue
                inds, minibatch_size = minibatch_plan

                with self.profiler.track("gradient"):
                    for epoch in range(hypers.update_epochs):
                        np.random.shuffle(inds)
                        for start in range(0, actual_batch_size, minibatch_size):
                            end = start + minibatch_size
                            mb_inds = inds[start:end]
                            mb_obs = {k: v[mb_inds] for k, v in obs.items()}
                            mb_old_logprobs = logprobs[mb_inds]
                            mb_actions = actions[mb_inds]
                            mb_advantages = advantages[mb_inds]
                            mb_advantages = self._maybe_normalize_advantages(
                                mb_advantages
                            )
                            mb_returns = returns[mb_inds]
                            mb_values = values[mb_inds]

                            approx_kl, clip_fraction = self._optimize_step(
                                mb_obs,
                                mb_old_logprobs,
                                mb_actions,
                                mb_advantages,
                                mb_returns,
                                mb_values,
                                log_gradients=(update % 10 == 0),
                            )
                            clipfracs.append(clip_fraction)

                            if update % 10 == 0:
                                self._log_system_metrics()

                            if (
                                hypers.target_kl != float("inf")
                                and approx_kl > hypers.target_kl
                            ):
                                self.logger.info(
                                    f"Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f}"
                                )
                                break

                with torch.no_grad():
                    y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
                self.last_explained_variance = float(explained_var)

                sps = int(self.global_step / (time.time() - self.start_time))
                if self.wandb:
                    self.wandb.log(
                        {
                            "charts/learning_rate": self.optimizer.param_groups[0][
                                "lr"
                            ],
                            "losses/explained_variance": explained_var,
                            "charts/SPS": sps,
                        },
                        step=self.global_step,
                    )

                self.experiment.log_performance(step=self.global_step)

                time_since_start = time.time() - self.start_time
                self.logger.info(
                    f"Update {update}/{num_updates} | SPS: {sps} | Total time: {time_since_start:.2f}s"
                )

                if update % 100 == 0:
                    self.logger.info(
                        f"Saving artifact @ update: {update} step: {self.global_step}"
                    )
                    self.save()

            self.save()
            env.close()
            self.experiment.close()
            self.logger.info("Training completed.")

    def _rollout_step(
        self, next_obs: Dict[str, torch.Tensor]
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(next_obs)

        try:
            with self.profiler.track("env"):
                new_obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as e:
            self.logger.error(f"env.step() failed: {e}")
            raise

        done = terminated | truncated
        if truncated.any():
            n_truncated = int(truncated.sum().item())
            self._increment_rollout_health("truncated_episodes", n_truncated)
            self.logger.warning(
                f"Truncation in {n_truncated}/{self.hypers.num_envs} envs (no value bootstrap)"
            )
        for info_key, health_key in (
            ("action_space_truncated", "action_space_truncations"),
            ("card_space_truncated", "card_space_truncations"),
            ("permanent_space_truncated", "permanent_space_truncations"),
        ):
            count = self._count_info_events(info, info_key)
            if count > 0:
                self._increment_rollout_health(health_key, count)

        self.global_step += self.hypers.num_envs

        if not self._validate_obs(new_obs):
            raise RuntimeError("Invalid observation format detected; halting training.")

        self.logger.debug("Rollout step completed.")
        return new_obs, reward, done, action, logprob, value

    def _init_rollout_buffers(
        self, sample_obs: Dict[str, torch.Tensor]
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        num_steps = self.hypers.num_steps
        num_envs = self.hypers.num_envs
        obs_buf = {
            key: torch.zeros(
                (num_steps, *value.shape), dtype=value.dtype, device=value.device
            )
            for key, value in sample_obs.items()
        }
        actions_buf = torch.zeros(
            (num_steps, num_envs), dtype=torch.int64, device=self.experiment.device
        )
        logprobs_buf = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=self.experiment.device
        )
        rewards_buf = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=self.experiment.device
        )
        dones_buf = torch.zeros(
            (num_steps, num_envs), dtype=torch.bool, device=self.experiment.device
        )
        values_buf = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=self.experiment.device
        )
        return obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(
            rewards.shape[1], dtype=torch.float32, device=rewards.device
        )
        next_value = next_value.view(-1)

        for t in reversed(range(self.hypers.num_steps)):
            if t == self.hypers.num_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1].float()
                next_values = values[t + 1]

            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def _flatten_rollout(
        self,
        obs_buf: Dict[str, torch.Tensor],
        actions_buf: torch.Tensor,
        logprobs_buf: torch.Tensor,
        advantages_buf: torch.Tensor,
        returns_buf: torch.Tensor,
        values_buf: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flattened_obs = {
            key: value.reshape((-1, *value.shape[2:])) for key, value in obs_buf.items()
        }
        flattened_logprobs = logprobs_buf.reshape(-1)
        flattened_actions = actions_buf.reshape(-1)
        flattened_advantages = advantages_buf.reshape(-1)
        flattened_returns = returns_buf.reshape(-1)
        flattened_values = values_buf.reshape(-1)
        return (
            flattened_obs,
            flattened_logprobs,
            flattened_actions,
            flattened_advantages,
            flattened_returns,
            flattened_values,
        )

    def _reset_rollout_health_update(self) -> None:
        for key in ROLLOUT_HEALTH_KEYS:
            self.rollout_health_update[key] = 0

    def _new_rollout_health(self) -> Dict[str, int]:
        return {key: 0 for key in ROLLOUT_HEALTH_KEYS}

    def _increment_rollout_health(self, key: str, n: int) -> None:
        self.rollout_health[key] += n
        self.rollout_health_update[key] += n

    def _maybe_normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        if not self.hypers.norm_adv:
            return advantages

        advantages_centered = advantages - advantages.mean()
        std = advantages.std()
        if torch.isnan(std) or std < 1e-8:
            return advantages_centered
        return advantages_centered / (std + 1e-8)

    def _build_minibatch_plan(
        self, actual_batch_size: int
    ) -> Tuple[np.ndarray, int] | None:
        if actual_batch_size < self.hypers.num_minibatches:
            self.logger.warning(
                f"Skipping update: actual_batch_size={actual_batch_size} < num_minibatches={self.hypers.num_minibatches}"
            )
            return None
        minibatch_size = max(1, actual_batch_size // self.hypers.num_minibatches)
        return np.arange(actual_batch_size), minibatch_size

    def _count_info_events(self, info: Dict[str, Any], key: str) -> int:
        """Count truthy events in a vectorized env info dict.

        The Rust vector env returns stacked arrays directly. The legacy
        benchmark wrapper also adds Gymnasium-style autoreset masks under
        `_key`; when present we honor them to avoid counting stale values.
        """
        if key not in info:
            return 0
        events = info[key]
        autoreset_mask = info.get(f"_{key}")

        events_arr = np.asarray(events)
        if events_arr.dtype == np.object_:
            events_arr = np.array([bool(v) for v in events_arr], dtype=bool)
        else:
            events_arr = events_arr.astype(bool, copy=False)

        if autoreset_mask is not None:
            mask_arr = np.asarray(autoreset_mask).astype(bool, copy=False)
            if events_arr.shape == mask_arr.shape:
                events_arr = events_arr[mask_arr]
        return int(np.count_nonzero(events_arr))

    def _log_rollout_health(self, update: int) -> None:
        rollout_parts = [
            f"{key}={self.rollout_health_update[key]} (total={self.rollout_health[key]})"
            for key in ROLLOUT_HEALTH_KEYS
        ]
        self.logger.info(f"Update {update} rollout health | {', '.join(rollout_parts)}")
        if self.wandb:
            metrics = {}
            for key in ROLLOUT_HEALTH_KEYS:
                metrics[f"rollout/{key}"] = self.rollout_health_update[key]
                metrics[f"rollout/{key}_total"] = self.rollout_health[key]
            self.wandb.log(metrics, step=self.global_step)

    def _optimize_step(
        self,
        obs: Dict[str, torch.Tensor],
        logprobs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        log_gradients: bool = False,
    ) -> Tuple[float, float]:
        hypers = self.hypers
        _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(
            obs, actions
        )
        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - hypers.clip_coef, 1 + hypers.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        new_values = new_values.view(-1)
        if hypers.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(
                new_values - values, -hypers.clip_coef, hypers.clip_coef
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - hypers.ent_coef * entropy_loss + hypers.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.agent.parameters(), hypers.max_grad_norm)

        if log_gradients:
            self._log_gradient_norms()

        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().item()
            approx_kl = max(approx_kl, 0.0)
            clip_fraction = (
                (torch.abs(ratio - 1) > hypers.clip_coef).float().mean().item()
            )

        if self.wandb:
            self.wandb.log(
                {
                    "losses/policy_loss": pg_loss.item(),
                    "losses/value_loss": v_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl": approx_kl,
                    "losses/clip_fraction": clip_fraction,
                    "ppo/losses": {
                        "policy": pg_loss.item(),
                        "value": v_loss.item(),
                        "entropy": entropy_loss.item(),
                    },
                    "ppo/metrics": {"kl": approx_kl, "clip_fraction": clip_fraction},
                },
                step=self.global_step,
            )

        self.logger.debug(
            f"Optimize step: approx_kl={approx_kl}, clip_fraction={clip_fraction}"
        )
        return approx_kl, clip_fraction

    def verify_attention_masking(self, obs):
        """
        Verifies attention masking is working correctly by:
        1. Recording outputs with normal inputs
        2. Creating a modified observation where invalid/masked tokens have random values
        3. Ensuring outputs are identical, proving masked values don't leak through
        """
        self.logger.debug("Verifying attention masking integrity")

        # Get original object embeddings and mask
        with torch.no_grad():
            objects, is_agent, validity = self.agent._gather_object_embeddings(obs)
            key_padding_mask = validity == 0
            original_output = self.agent.attention(objects, is_agent, key_padding_mask)

        # Create a copy with random noise in the masked positions
        noisy_objects = objects.clone()
        if torch.any(key_padding_mask):
            # Add large random noise to masked positions
            noise = torch.randn_like(objects) * 10.0
            noisy_objects[key_padding_mask] = noise[key_padding_mask]

            # Get output with noisy inputs
            noisy_output = self.agent.attention(
                noisy_objects, is_agent, key_padding_mask
            )

            # Check if outputs are identical (they should be if masking works)
            diff = (original_output - noisy_output).abs().max().item()

            if diff > 1e-5:
                self.logger.error(
                    f"Attention mask leakage detected! Max difference: {diff}"
                )
                # Log additional diagnostics about which positions leaked
                leaked_positions = (
                    ((original_output - noisy_output).abs() > 1e-5).sum().item()
                )
                self.logger.error(
                    f"Number of positions with leakage: {leaked_positions}"
                )
                return False
            else:
                self.logger.debug("Attention masking verified: No leakage detected")
                return True
        else:
            self.logger.debug("No masked positions to verify")
            return True

    def _validate_obs(self, obs: dict) -> bool:
        expected_keys = set(self.env.observation_space.keys())
        if set(obs.keys()) != expected_keys:
            self.logger.error(
                f"Observation keys mismatch. Expected {expected_keys}, got {set(obs.keys())}"
            )
            return False

        for k, v in obs.items():
            expected_shape = self.env.observation_space[k].shape
            if v.shape[1:] != expected_shape:
                self.logger.error(
                    f"Observation shape mismatch for key {k}. "
                    f"Expected {expected_shape} (inside batch), got {v.shape[1:]}"
                )
                return False

        # Now log detailed statistics
        validity_stats = {
            "agent_player_valid": obs["agent_player_valid"].sum().item(),
            "opponent_player_valid": obs["opponent_player_valid"].sum().item(),
            "agent_cards_valid": obs["agent_cards_valid"].sum().item(),
            "opponent_cards_valid": obs["opponent_cards_valid"].sum().item(),
            "agent_permanents_valid": obs["agent_permanents_valid"].sum().item(),
            "opponent_permanents_valid": obs["opponent_permanents_valid"].sum().item(),
            "actions_valid": obs["actions_valid"].sum().item(),
        }

        # Log summary
        self.logger.debug("Observation validity statistics:")
        for key, count in validity_stats.items():
            self.logger.debug(f"  {key}: {count}")

        # Check for anomalies
        if (
            validity_stats["agent_player_valid"] < 1
            or validity_stats["opponent_player_valid"] < 1
        ):
            self.logger.warning("Missing valid players in observation!")

        if validity_stats["actions_valid"] < 1:
            self.logger.warning("No valid actions in observation!")

        # Log to wandb
        if self.wandb:
            wandb_stats = {f"observation/{k}": v for k, v in validity_stats.items()}
            self.wandb.log(wandb_stats, step=self.global_step)

        return True

    def _log_system_metrics(self):
        if not self.wandb:
            return

        metrics = {
            "system/memory_used": psutil.Process().memory_info().rss / (1024 * 1024),
            "system/cpu_percent": psutil.cpu_percent(),
            "system/steps_per_second": int(
                self.global_step / (time.time() - self.start_time)
            ),
        }
        if torch.cuda.is_available():
            metrics.update(
                {
                    "system/gpu_utilization": torch.cuda.utilization(),
                    "system/gpu_memory_allocated": torch.cuda.memory_allocated()
                    / (1024 * 1024),
                    "system/gpu_memory_reserved": torch.cuda.memory_reserved()
                    / (1024 * 1024),
                }
            )
        self.wandb.log(metrics, step=self.global_step)
        self.logger.debug(f"Logged system metrics: {metrics}")

    def _log_gradient_norms(self) -> None:
        """Log per-component gradient norms. Called periodically, not every minibatch."""
        layer_grad_norms = {}
        total_grad_norm = 0.0
        embedding_grad_norm = 0.0
        attention_grad_norm = 0.0
        policy_head_grad_norm = 0.0
        value_head_grad_norm = 0.0
        other_grad_norm = 0.0

        for name, param in self.agent.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().data.norm(2).item()
                layer_grad_norms[name] = grad_norm
                total_grad_norm += grad_norm**2

                if (
                    "player_embedding" in name
                    or "card_embedding" in name
                    or "perm_embedding" in name
                ):
                    embedding_grad_norm += grad_norm**2
                elif "attention" in name:
                    attention_grad_norm += grad_norm**2
                elif "policy_head" in name:
                    policy_head_grad_norm += grad_norm**2
                elif "value_head" in name:
                    value_head_grad_norm += grad_norm**2
                else:
                    other_grad_norm += grad_norm**2

        total_grad_norm = total_grad_norm**0.5
        embedding_grad_norm = embedding_grad_norm**0.5
        attention_grad_norm = attention_grad_norm**0.5
        policy_head_grad_norm = policy_head_grad_norm**0.5
        value_head_grad_norm = value_head_grad_norm**0.5
        other_grad_norm = other_grad_norm**0.5

        non_zero_norms = [norm for norm in layer_grad_norms.values() if norm > 0]
        if non_zero_norms:
            max_layer_norm = max(non_zero_norms)
            min_layer_norm = min(non_zero_norms)
            max_to_min_ratio = (
                max_layer_norm / min_layer_norm if min_layer_norm > 0 else float("inf")
            )
        else:
            max_layer_norm = 0
            min_layer_norm = 0
            max_to_min_ratio = 0

        self.logger.info(f"Total gradient norm (post-clip): {total_grad_norm:.4f}")

        if total_grad_norm > 10.0:
            self.logger.warning(
                f"Potentially exploding gradient: {total_grad_norm:.4f}"
            )
        elif total_grad_norm < 1e-4:
            self.logger.warning(
                f"Potentially vanishing gradient: {total_grad_norm:.4f}"
            )

        if max_to_min_ratio > 1000:
            self.logger.warning(
                f"Extreme gradient imbalance: max/min ratio = {max_to_min_ratio:.2f}"
            )

        if layer_grad_norms:
            top_grads = sorted(
                layer_grad_norms.items(), key=lambda x: x[1], reverse=True
            )[:5]
            self.logger.debug("Top 5 highest gradient norms:")
            for name, norm in top_grads:
                self.logger.debug(f"  {name}: {norm:.6f}")

        if self.wandb:
            self.wandb.log(
                {
                    "gradients/total_norm": total_grad_norm,
                    "gradients/embedding_norm": embedding_grad_norm,
                    "gradients/attention_norm": attention_grad_norm,
                    "gradients/policy_head_norm": policy_head_grad_norm,
                    "gradients/value_head_norm": value_head_grad_norm,
                    "gradients/other_norm": other_grad_norm,
                    "gradients/max_layer_norm": max_layer_norm,
                    "gradients/min_layer_norm": min_layer_norm,
                    "gradients/max_to_min_ratio": max_to_min_ratio,
                },
                step=self.global_step,
            )

    def save(self) -> None:
        if self.wandb is None:
            return

        name = self.experiment.exp_name

        timestamp = datetime.datetime.fromtimestamp(self.start_time).strftime(
            "%Y%m%d_%H%M%S"
        )
        version_tag = f"{timestamp}_{self.global_step}"

        # Save all relevant hyperparameters
        hypers_dict = {
            "agent_hypers": self.agent.hypers.model_dump(),
            "observation_hypers": self.env.observation_space.encoder.hypers.model_dump(),
            "train_hypers": self.hypers.model_dump(),
        }

        path = f"{name}.pt"
        torch.save(
            {
                "model_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "hypers": hypers_dict,
            },
            path,
        )

        # Create and log artifact with the version tag
        artifact = wandb.Artifact(
            name=name,
            type="model",
            description=f"Model checkpoint at step {self.global_step}",
        )

        # Add metadata for easier filtering/selection
        artifact.metadata = {
            "version": version_tag,
            "timestamp": timestamp,
            "step": self.global_step,
            "model_type": self.agent.__class__.__name__,
        }

        # You can add additional tags to make it easier to filter
        if self.agent.hypers.attention_on:
            artifact.metadata["architecture"] = "attention"

        artifact.add_file(path)

        # Log the artifact with an alias that includes the version
        self.wandb.log_artifact(
            artifact, aliases=[f"step_{self.global_step}", version_tag]
        )

        self.logger.info(f"Saved model with version tag: {version_tag}")


def build_training_components(
    hypers: Hypers,
) -> tuple[Experiment, RustVectorEnv, Agent]:
    experiment = Experiment(hypers.experiment, hypers)
    observation_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    env = RustVectorEnv(
        hypers.train.num_envs,
        match,
        observation_space,
        reward,
        device=experiment.device,
        seed=hypers.experiment.seed,
        opponent_policy=hypers.train.opponent_policy,
    )
    agent = Agent(observation_space, hypers.agent)
    return experiment, env, agent


def run_training(hypers: Hypers) -> Trainer:
    """Build all components for a training run and execute Trainer.train()."""
    experiment, env, agent = build_training_components(hypers)
    trainer = Trainer(agent, experiment, env, hypers.train)
    trainer.train()
    return trainer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manabot with preset config")
    parser.add_argument(
        "--preset",
        default="local",
        help="Training preset name (local/simple/attention)",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (repeatable key.path=value)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    from manabot.config.load import load_train_config

    hypers = load_train_config(preset=args.preset, set_overrides=args.set_values)
    run_training(hypers)


if __name__ == "__main__":
    main()
