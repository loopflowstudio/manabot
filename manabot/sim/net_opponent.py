"""PPO training with a policy-net opponent in the loop (wave/search C8, exp-11).

The Rust VectorEnv can play the opponent seat itself only for the scripted
policies ("random"/"passive"). This module adds the missing training path:
the opponent seat is played by a *network* — either a frozen checkpoint
(opponent curriculum / exploitability probe) or the live learner itself
(true self-play) — using the exp-07 batched per-seat routing
(``managym.VectorEnv`` with ``opponent_policy="none"`` +
``current_agent_indices``, as in ``manabot.sim.rollout.run_vector_games``).

Design notes, kept simple and honest:

- ``SeatRoutedCollector`` drives K env streams at the micro-step level.
  Stream ``s`` seats the learner at ``s % 2`` for its whole life, so the
  collected data is seat-balanced by construction. Each micro-step routes
  learner rows through the learner's policy/value forward (one batched
  forward) and opponent rows through the opponent controller.
- Only the learner's transitions enter the PPO buffers. A learner
  transition's reward is terminal-only and computed from ``winner_index``:
  +win_reward / +lose_reward when the game ends before the learner's next
  decision, else 0. The Rust reward stream is ignored.
- Streams pause at the exact next learner observation once their quota is
  complete. No sampled action or excess transition crosses an update boundary.
- ``NetOpponentTrainer`` reuses the stock ``Trainer`` PPO update machinery
  (flatten, minibatch, optimize, save), with transition-end GAE for this
  collector's boundary convention.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Any, Dict

import numpy as np
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.env.observation import ActionEnum
from manabot.model.agent import Agent
from manabot.model.policy_distribution import (
    NumericalError,
    legal_mask,
    policy_logs,
    require,
)
from manabot.model.train import Trainer
from manabot.sim.rollout import (
    OBS_KEYS,
    BatchedSampler,
    RandomBatchController,
    _allocate_buffers,
)
from manabot.training.self_play import SelfPlayCounter, deck_identity
import managym

OPPONENT_MODES = ("random", "frozen", "self")

# -----------------------------------------------------------------------------
# Seat-routed rollout collection
# -----------------------------------------------------------------------------


@dataclass
class _Pending:
    """A learner transition awaiting its reward (next learner decision or
    game end)."""

    obs: Dict[str, np.ndarray]
    action: int
    logprob: float
    value: float
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    outcome_probabilities: np.ndarray | None = None


@dataclass
class _Transition:
    """Finalized learner row with its collection-time outcome distribution."""

    pending: _Pending
    reward: float
    done: bool


@dataclass
class RolloutBatch:
    """One PPO update's worth of learner transitions, (num_steps, num_envs)."""

    obs: Dict[str, np.ndarray]
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_obs: Dict[str, np.ndarray]
    next_done: np.ndarray
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    # Collection-time loss/draw/win probabilities [T,E,3], absent for scalar.
    outcome_probabilities: np.ndarray | None = None


@dataclass
class CollectorStats:
    """Cumulative accounting, including the opponent fingerprint."""

    micro_steps: int = 0
    learner_transitions: int = 0
    opponent_decisions: int = 0
    games: int = 0
    learner_wins: int = 0
    self_play: SelfPlayCounter = field(default_factory=SelfPlayCounter)
    truncations: int = 0
    seconds: float = 0.0
    opponent_action_types: dict[str, int] = field(default_factory=dict)
    learner_action_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_steps": self.micro_steps,
            "learner_transitions": self.learner_transitions,
            "opponent_decisions": self.opponent_decisions,
            "games": self.games,
            "learner_wins": self.learner_wins,
            "learner_win_rate": (self.learner_wins / self.games if self.games else 0.0),
            "truncations": self.truncations,
            "seconds": self.seconds,
            "opponent_action_types": dict(self.opponent_action_types),
            "learner_action_types": dict(self.learner_action_types),
        }


@dataclass
class CollectorSnapshot:
    """Private training state. Replay is native execution, not policy inference.

    Journal rows contain [action per stream] and [active per stream]. The
    native constructor seed plus every vector step also restores auto-reset RNG.
    Only empty transition buffers at update boundaries may be snapshotted.
    """

    journal: list[tuple[list[int], list[bool]]]
    buffers: dict[str, np.ndarray]
    sampling_rng: torch.Tensor
    stats: CollectorStats


_ACTION_TYPE_NAMES = {int(member): member.name.lower() for member in ActionEnum}


class SeatRoutedCollector:
    """Collect learner-only PPO transitions with a net (or random) opponent.

    ``opponent_mode``:
        - ``"random"``  — uniform-over-valid opponent (the exp-04/06 lineage
          rebased onto this driver, seat-balanced);
        - ``"frozen"``  — ``opponent_agent`` plays the opponent seat
          (stochastic sampling, the eval protocol);
        - ``"self"``    — the live learner plays the opponent seat with
          current weights; opponent-seat transitions are discarded so the
          batch stays identical in size and shape to the other modes.
    """

    def __init__(
        self,
        observation_space: ObservationSpace,
        match: Match,
        reward: Reward,
        *,
        num_envs: int,
        seed: int = 0,
        opponent_mode: str = "random",
        opponent_agent: Agent | None = None,
        device: str = "cpu",
        recovery_max_microsteps: int | None = None,
    ) -> None:
        if opponent_mode not in OPPONENT_MODES:
            raise ValueError(
                f"opponent_mode must be one of {OPPONENT_MODES}, got {opponent_mode!r}"
            )
        if opponent_mode == "frozen" and opponent_agent is None:
            raise ValueError("opponent_mode='frozen' requires opponent_agent")

        self.observation_space = observation_space
        self.match = match
        self.stream_matches = [match] * num_envs
        self.reward = reward
        self.num_envs = num_envs
        self.opponent_mode = opponent_mode
        self.device = torch.device(device)
        self.stats = CollectorStats()
        if recovery_max_microsteps is not None and opponent_mode != "self":
            raise ValueError("recovery supports current-self opponents only")
        self._recovery_limit = recovery_max_microsteps
        self._journal: list[tuple[list[int], list[bool]]] = []

        self._env = managym.VectorEnv(
            num_envs=num_envs,
            seed=seed,
            skip_trivial=True,
            opponent_policy="none",
        )
        self._buffers = _allocate_buffers(observation_space, num_envs)
        self._env.set_buffers(self._buffers)
        self._env.reset_all_into_buffers(match.to_rust())

        self._self_play_decks = [
            (
                deck_identity(m.hero_deck, m.hero_sideboard),
                deck_identity(m.villain_deck, m.villain_sideboard),
            )
            for m in self.stream_matches
        ]

        #: Learner seat per stream: stream s seats the learner at s % 2, so
        #: half the streams have the learner on the play — seat-balanced.
        self.learner_seat = np.arange(num_envs, dtype=np.int64) % 2

        self._streams: list[list[_Transition]] = [[] for _ in range(num_envs)]
        self._pending: list[_Pending | None] = [None] * num_envs

        if opponent_mode == "random":
            self._opponent: Any = RandomBatchController(seed=seed + 1)
        elif opponent_mode == "frozen":
            self._opponent = BatchedSampler(
                opponent_agent, deterministic=False, seed=seed + 1, device=device
            )
        else:  # self-play: live learner weights, bound at collect() time
            self._opponent = None
        self._self_rng = torch.Generator(device=self.device).manual_seed(seed + 1)

        self._win_reward = float(reward.hypers.win_reward)
        self._lose_reward = float(reward.hypers.lose_reward)

    def snapshot(self) -> CollectorSnapshot:
        """Capture a completed update boundary, including native RNG replay."""
        if self._recovery_limit is None:
            raise ValueError("collector recovery is disabled")
        if any(self._streams) or any(p is not None for p in self._pending):
            raise ValueError("collector snapshot requires an update boundary")
        return CollectorSnapshot(
            deepcopy(self._journal),
            {key: value.copy() for key, value in self._buffers.items()},
            self._self_rng.get_state().clone(),
            deepcopy(self.stats),
        )

    def restore(self, state: CollectorSnapshot, check: Callable[[], None]) -> None:
        """Replay into a fresh collector; reject divergence before any learning."""
        if self.stats.micro_steps or self._journal:
            raise ValueError("restore requires a fresh collector")
        if (
            self._recovery_limit is None
            or state.stats.micro_steps > self._recovery_limit
        ):
            raise ValueError("collector replay exceeds recovery bound")
        for actions, active in state.journal:
            check()
            self._env.step_into_buffers(actions, active)
        for key, expected in state.buffers.items():
            if not np.array_equal(self._buffers[key], expected):
                raise ValueError(f"collector replay diverged: {key}")
        self._journal = deepcopy(state.journal)
        self._self_rng.set_state(state.sampling_rng)
        self.stats = deepcopy(state.stats)
        # Pre-instrumentation snapshots have no outcome counters to recover.
        if not hasattr(self.stats, "self_play"):
            self.stats.self_play = SelfPlayCounter()

    # -- opponent routing -----------------------------------------------------

    def _opponent_actions(self, agent: Agent, rows: np.ndarray) -> np.ndarray:
        if self.opponent_mode == "self":
            obs = self._slice_obs_tensors(rows)
            with torch.inference_mode():
                logits, _ = agent.forward(obs)
                probs = self._policy_logs(logits, obs).exp()
                actions = torch.multinomial(probs, 1, generator=self._self_rng).squeeze(
                    -1
                )
            return actions.cpu().numpy().astype(np.int64)
        return self._opponent.select(self._buffers, rows)

    @staticmethod
    def _policy_logs(
        logits: torch.Tensor, obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        try:
            return policy_logs(logits, legal_mask(obs["actions_valid"]))
        except NumericalError as error:
            error.tensors.update(
                {f"observation/{key}": value.detach() for key, value in obs.items()}
            )
            raise

    def _slice_obs_tensors(self, rows: np.ndarray) -> Dict[str, torch.Tensor]:
        obs = {}
        for key in OBS_KEYS:
            tensor = torch.from_numpy(self._buffers[key][rows])
            if key == "action_focus":
                # Training convention (VectorEnv): action_focus as float32.
                tensor = tensor.float()
            obs[key] = tensor.to(self.device)
        return obs

    def _count_action_types(
        self, counter: dict[str, int], rows: np.ndarray, actions: np.ndarray
    ) -> None:
        chosen = self._buffers["actions"][rows, actions, :-1]
        type_indices = chosen.argmax(axis=1)
        typed = chosen.max(axis=1) > 0
        for idx, has_type in zip(type_indices, typed):
            name = (
                _ACTION_TYPE_NAMES.get(int(idx), "unknown") if has_type else "unknown"
            )
            counter[name] = counter.get(name, 0) + 1

    # -- collection loop ------------------------------------------------------

    def collect(
        self,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        """Advance all streams until every env has ``num_steps`` finalized
        learner transitions, stopping before the bootstrap action is sampled."""

        start = time.perf_counter()
        buffers = self._buffers
        env = self._env
        num_envs = self.num_envs

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        while True:
            if check is not None:
                check()
            if (
                deadline_monotonic is not None
                and time.perf_counter() >= deadline_monotonic
            ):
                raise TimeoutError("collector exceeded training deadline")
            acting = np.asarray(env.current_agent_indices(), dtype=np.int64)
            for row in range(num_envs):
                if (
                    acting[row] == self.learner_seat[row]
                    and self._pending[row] is not None
                ):
                    self._finalize(row, reward=0.0, done=False)
            active = np.array([len(stream) < num_steps for stream in self._streams])
            if not active.any():
                break
            actions = np.zeros(num_envs, dtype=np.int64)
            learner_rows = np.flatnonzero(active & (acting == self.learner_seat))
            opp_rows = np.flatnonzero(active & (acting != self.learner_seat))

            if len(learner_rows):
                obs_t = self._slice_obs_tensors(learner_rows)
                with torch.no_grad():
                    outcomes: np.ndarray | None = None
                    if agent.hypers.value_kind == "categorical_wdl":
                        logits, outcome_logits = agent.forward_distribution(obs_t)
                        outcome_t = outcome_logits.softmax(dim=-1)
                        value_t = outcome_t[:, 2] - outcome_t[:, 0]
                        outcomes = outcome_t.cpu().numpy()
                    else:
                        logits, value_t = agent(obs_t)
                    logs = self._policy_logs(logits, obs_t)
                    require(
                        torch.isfinite(value_t),
                        "nonfinite_collection_values",
                        values=value_t,
                        valid=obs_t["actions_valid"] > 0,
                    )
                    weights = logs.exp()
                    action_t = torch.multinomial(
                        weights, 1, generator=self._self_rng
                    ).squeeze(-1)
                    logprob_t = logs.gather(-1, action_t[:, None]).squeeze(-1)
                    probabilities = weights.cpu().numpy()
                    log_probabilities = logs.cpu().numpy()
                acts = action_t.cpu().numpy().astype(np.int64)
                logprobs = logprob_t.cpu().numpy()
                values = value_t.view(-1).cpu().numpy()
                # One copied slice per key; per-row dicts view into it.
                obs_rows = {key: buffers[key][learner_rows].copy() for key in OBS_KEYS}
                for j, row in enumerate(learner_rows):
                    self._pending[row] = _Pending(
                        obs={key: obs_rows[key][j] for key in OBS_KEYS},
                        action=int(acts[j]),
                        logprob=float(logprobs[j]),
                        value=float(values[j]),
                        probabilities=probabilities[j].copy(),
                        log_probabilities=log_probabilities[j].copy(),
                        outcome_probabilities=(
                            None if outcomes is None else outcomes[j].copy()
                        ),
                    )
                actions[learner_rows] = acts
                self._count_action_types(
                    self.stats.learner_action_types, learner_rows, acts
                )

            if len(opp_rows):
                opp_acts = self._opponent_actions(agent, opp_rows)
                actions[opp_rows] = opp_acts
                self.stats.opponent_decisions += len(opp_rows)
                self._count_action_types(
                    self.stats.opponent_action_types, opp_rows, opp_acts
                )

            if self._recovery_limit is not None:
                if self.stats.micro_steps + int(active.sum()) > self._recovery_limit:
                    raise RuntimeError("collector recovery journal limit exceeded")
                self._journal.append((actions.tolist(), active.tolist()))
            env.step_into_buffers(actions.tolist(), active.tolist())
            self.stats.micro_steps += int(active.sum())

            done = (buffers["terminated"] > 0) | (buffers["truncated"] > 0)
            if done.any():
                infos = env.get_last_info()
                for row in np.flatnonzero(done):
                    winner = infos[row].get("winner_index")
                    winner = int(winner) if winner is not None else None
                    self.stats.games += 1
                    if buffers["truncated"][row]:
                        self.stats.truncations += 1
                        raise RuntimeError(
                            "collector game truncated; no terminal target available"
                        )
                    if self.opponent_mode == "self":
                        self.stats.self_play.record(
                            self._self_play_decks[int(row)], winner
                        )
                    if winner is None:
                        reward = 0.0
                    elif winner == int(self.learner_seat[row]):
                        self.stats.learner_wins += 1
                        reward = self._win_reward
                    else:
                        reward = self._lose_reward
                    if self._pending[row] is not None:
                        self._finalize(int(row), reward=reward, done=True)

        batch = self._build_batch(num_steps)
        self.stats.seconds += time.perf_counter() - start
        return batch

    def _finalize(self, row: int, *, reward: float, done: bool) -> None:
        pending = self._pending[row]
        assert pending is not None
        self._streams[row].append(_Transition(pending, reward, done))
        self._pending[row] = None
        self.stats.learner_transitions += 1

    def _build_batch(self, num_steps: int) -> RolloutBatch:
        num_envs = self.num_envs
        shapes = self.observation_space.encoder.allocate(1)
        obs = {
            key: np.zeros((num_steps, num_envs) + value.shape[1:], dtype=value.dtype)
            for key, value in shapes.items()
        }
        actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        values = np.zeros((num_steps, num_envs), dtype=np.float32)
        next_obs = {
            key: np.zeros((num_envs,) + value.shape[1:], dtype=value.dtype)
            for key, value in shapes.items()
        }
        next_done = np.zeros((num_envs,), dtype=bool)
        probabilities = np.zeros_like(obs["actions_valid"])
        log_probabilities = np.full_like(probabilities, -np.inf)
        outcome_probabilities = (
            np.zeros((num_steps, num_envs, 3), dtype=np.float32)
            if self._streams[0][0].pending.outcome_probabilities is not None
            else None
        )

        for env_index in range(num_envs):
            stream = self._streams[env_index]
            for step in range(num_steps):
                transition = stream[step]
                pending = transition.pending
                probabilities[step, env_index] = pending.probabilities
                log_probabilities[step, env_index] = pending.log_probabilities
                if outcome_probabilities is not None:
                    assert pending.outcome_probabilities is not None
                    outcome_probabilities[step, env_index] = (
                        pending.outcome_probabilities
                    )
                for key in OBS_KEYS:
                    obs[key][step, env_index] = pending.obs[key]
                actions[step, env_index] = pending.action
                logprobs[step, env_index] = pending.logprob
                values[step, env_index] = pending.value
                rewards[step, env_index] = transition.reward
                dones[step, env_index] = transition.done
            del stream[:num_steps]
            assert not stream and self._pending[env_index] is None
            for key in OBS_KEYS:
                next_obs[key][env_index] = self._buffers[key][env_index]
            next_done[env_index] = dones[num_steps - 1, env_index]

        return RolloutBatch(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            values=values,
            next_obs=next_obs,
            next_done=next_done,
            probabilities=probabilities,
            log_probabilities=log_probabilities,
            outcome_probabilities=outcome_probabilities,
        )


# -----------------------------------------------------------------------------
# Trainer: stock PPO update machinery, seat-routed collection
# -----------------------------------------------------------------------------


def transition_gae(rewards, values, ends, next_value, gamma, lam):
    """GAE for transition-end flags; terminal rewards credit their own episode."""
    advantages = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    following = next_value.reshape(-1)
    for step in reversed(range(len(rewards))):
        live = (~ends[step].bool()).to(rewards.dtype)
        delta = rewards[step] + gamma * live * following - values[step]
        tail = delta + gamma * lam * live * tail
        advantages[step] = tail
        following = values[step]
    return advantages, advantages + values


class _CollectorEnvShim:
    """Just enough env surface for Trainer.__init__ / periodic eval."""

    def __init__(
        self, observation_space: ObservationSpace, reward: Reward, match: Match
    ):
        self.observation_space = observation_space
        self.reward = reward
        self.match = match

    def close(self) -> None:
        return None


class NetOpponentTrainer(Trainer):
    """PPO trainer whose rollouts come from a SeatRoutedCollector.

    Reuses the stock Trainer's flatten, minibatch plan, optimize step,
    periodic eval (vs the scripted ``hypers.opponent_policy``, so learning
    curves stay comparable across arms) and checkpointing.
    """

    def __init__(
        self,
        agent: Agent,
        experiment,
        collector: SeatRoutedCollector,
        hypers=None,
    ):
        shim = _CollectorEnvShim(
            collector.observation_space, collector.reward, collector.match
        )
        super().__init__(agent, experiment, shim, hypers)  # type: ignore[arg-type]
        self.collector = collector

    def train(self) -> None:
        hypers = self.hypers
        device = self.experiment.device
        batch_size = hypers.num_envs * hypers.num_steps
        num_updates = hypers.total_timesteps // batch_size
        self.start_time = time.time()

        self.save()
        for update in range(1, num_updates + 1):
            if hypers.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = frac * hypers.learning_rate

            batch = self.collector.collect(self.agent, hypers.num_steps)
            obs_buf = self._obs_to_tensors(batch.obs, device)
            actions_buf = torch.as_tensor(batch.actions, device=device)
            logprobs_buf = torch.as_tensor(batch.logprobs, device=device)
            rewards_buf = torch.as_tensor(batch.rewards, device=device)
            dones_buf = torch.as_tensor(batch.dones, device=device)
            values_buf = torch.as_tensor(batch.values, device=device)
            next_obs = self._obs_to_tensors(batch.next_obs, device)

            self.global_step += batch_size

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs)
            advantages, returns = transition_gae(
                rewards_buf,
                values_buf,
                dones_buf,
                next_value,
                hypers.gamma,
                hypers.gae_lambda,
            )
            obs, logprobs, actions, advantages, returns, values = self._flatten_rollout(
                obs_buf,
                actions_buf,
                logprobs_buf,
                advantages,
                returns,
                values_buf,
            )

            minibatch_plan = self._build_minibatch_plan(logprobs.shape[0])
            if minibatch_plan is None:
                continue
            inds, minibatch_size = minibatch_plan

            approx_kl = 0.0
            for epoch in range(hypers.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, len(inds), minibatch_size):
                    mb_inds = inds[start : start + minibatch_size]
                    mb_obs = {k: v[mb_inds] for k, v in obs.items()}
                    mb_advantages = self._maybe_normalize_advantages(
                        advantages[mb_inds]
                    )
                    approx_kl, _ = self._optimize_step(
                        mb_obs,
                        logprobs[mb_inds],
                        actions[mb_inds],
                        mb_advantages,
                        returns[mb_inds],
                        values[mb_inds],
                        log_gradients=False,
                    )
                    if (
                        hypers.target_kl != float("inf")
                        and approx_kl > hypers.target_kl
                    ):
                        break

            with torch.no_grad():
                y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            self.last_explained_variance = float(explained_var)

            elapsed = time.time() - self.start_time
            sps = int(self.global_step / elapsed) if elapsed > 0 else 0
            self.logger.info(
                f"Update {update}/{num_updates} | SPS: {sps} | "
                f"rollout win_rate: {self.collector.stats.learner_wins}/"
                f"{self.collector.stats.games} | "
                f"explained_var: {explained_var:.3f}"
            )

            if hypers.eval_interval > 0 and update % hypers.eval_interval == 0:
                self._run_periodic_eval(update)

            if update % 100 == 0:
                self.save()

        self.save()
        self.experiment.close()
        self.logger.info("Training completed.")

    @staticmethod
    def _obs_to_tensors(obs: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
        tensors = {}
        for key, value in obs.items():
            tensor = torch.as_tensor(value, device=device)
            if key == "action_focus":
                tensor = tensor.float()
            tensors[key] = tensor
        return tensors
