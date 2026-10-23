"""Generate viewer-safe search targets from complete self-play games.

generate_selfplay_shard runs configured frozen players and atomically writes the
observation/target format defined in sim.data. Both seats retain their game and
seat identities. This module generates examples; it never fits model parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from managym.decision import Command, DecisionFrame
import numpy as np

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.config import (
    MatchConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.sim.data import (
    DECISION_KIND_KEY,
    LOCAL_RECEIPT_KEY,
    LOCAL_TARGET_KEY,
    NUM_LEGAL_KEY,
    OBS_KEYS,
    ROOT_VALUE_KEY,
    SCORE_KEY,
    VISIT_COUNT_KEY,
)
from manabot.sim.search.flat_mc import make_player, spec_name
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs

# -----------------------------------------------------------------------------
# Teacher self-play generation
# -----------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        import subprocess

        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def generate_selfplay_shard(
    *,
    num_games: int,
    sims: int | None = None,
    teacher_spec: dict[str, Any] | None = None,
    seed: int,
    game_offset: int = 0,
    out_path: str | Path | None = None,
    max_steps_per_game: int = 5000,
    round_index: int = 0,
    dataset_run_fingerprint: str | None = None,
    deadline_monotonic: float | None = None,
    match_config: MatchConfig | None = None,
    observation_config: ObservationConfig | None = None,
) -> dict[str, Any]:
    """Play teacher-vs-teacher self-play games, recording every decision.

    The teacher is any search player spec accepted by
    manabot.sim.search.flat_mc.make_player (flat search, policy search, or
    determinized PUCT); passing ``sims`` alone is shorthand for the exp-03
    random-rollout teacher. Every decision records the encoded observation,
    chosen action, and raw per-action values. Tree teachers additionally emit
    root visit counts and root values. Local-update teachers label trajectories
    played by their frozen base policy and retain complete update receipts.

    Returns a summary dict (games, decisions, wall/search seconds, steps).
    When ``out_path`` is given the decisions are written there as an .npz.
    """

    if teacher_spec is None:
        if sims is None:
            raise ValueError("pass either sims or teacher_spec")
        teacher_spec = {"kind": "search", "sims": sims}
    teacher_name = spec_name(teacher_spec)

    obs_space = ObservationSpace(observation_config or ObservationConfig())
    match = Match(
        match_config
        or MatchConfig(
            hero=f"{teacher_name}-a"[:32],
            villain=f"{teacher_name}-b"[:32],
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        obs_space,
        Reward(RewardConfig()),
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    players = [
        make_player(teacher_spec, seed=seed * 2 + 1)[0],
        make_player(teacher_spec, seed=seed * 2 + 2)[0],
    ]
    has_tree_targets = all(
        hasattr(player, "last_visit_counts") and hasattr(player, "last_root_value")
        for player in players
    )

    local_targets: list[np.ndarray] = []
    local_receipts: list[str] = []
    mixing_rows: list[dict[str, float]] = []
    is_local = teacher_spec["kind"] == "local_update"
    journal = Path(str(out_path) + ".receipts.jsonl") if is_local and out_path else None
    if journal is not None:
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.touch(exist_ok=False)
    max_actions = obs_space.encoder.max_actions
    obs_buffers: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    score_rows: list[np.ndarray] = []
    visit_rows: list[np.ndarray] = []
    root_values: list[float] = []
    actions: list[int] = []
    game_indices: list[int] = []
    seats: list[int] = []
    num_valids: list[int] = []
    num_legals: list[int] = []
    decision_kinds: list[int] = []
    winners_per_decision: list[list[int]] = []
    steps_per_game: list[int] = []
    winners: list[int | None] = []
    terminated_games: list[bool] = []
    truncated_games: list[bool] = []

    behavior_rng = np.random.default_rng(seed ^ 0x578ABC)
    wall_start = time.perf_counter()
    for i in range(num_games):
        game_index = game_offset + i
        env.failure_context.update(game_index=game_index, producer="distill")
        obs, _ = env.reset(seed=seed + game_index)
        for seat, player in enumerate(players):
            if hasattr(player, "start_game"):
                player.start_game(env, seat)
            if is_local:
                player.deadline = deadline_monotonic
        done = False
        terminated, truncated = False, False
        steps = 0
        info: dict[str, Any] = {}
        game_decisions: list[int] = []
        while not done and steps < max_steps_per_game:
            env.failure_context.update(
                step=steps,
                decisions=len(actions),
                deadline_monotonic=deadline_monotonic,
                output=str(out_path) if out_path else None,
            )
            if (
                deadline_monotonic is not None
                and time.perf_counter() >= deadline_monotonic
            ):
                raise TimeoutError("teacher collection exceeded training deadline")
            raw_obs = env.last_raw_obs
            acting = int(raw_obs.agent.player_index)
            num_legals.append(len(raw_obs.action_space.actions))
            decision_kinds.append(int(raw_obs.action_space.action_space_type))
            action = players[acting].act(env, obs)
            receipt = players[acting].last_receipt if is_local else None
            if is_local:
                # Frozen-policy histories make its action-likelihood posterior
                # the matched model. Search targets label, but do not generate,
                # this trajectory; arena play samples the improved distribution.
                if receipt is None:
                    raise RuntimeError("local teacher did not retain its update")
                action = players[acting].base_action(behavior_rng)
            for key in OBS_KEYS:
                obs_buffers[key].append(np.asarray(obs[key], dtype=np.float32))
            encoded_valid_count = int(np.sum(obs["actions_valid"] > 0))
            if action < 0 or action >= max_actions or obs["actions_valid"][action] <= 0:
                raise RuntimeError(
                    "teacher chose an action outside the encoded legal-action ABI"
                )
            score_row = np.full(max_actions, -1.0, dtype=np.float32)
            raw_scores = players[acting].last_scores
            if raw_scores is not None:
                if len(raw_scores) != encoded_valid_count:
                    raise RuntimeError(
                        "teacher legal surface does not fit the encoded action ABI"
                    )
                score_row[: len(raw_scores)] = raw_scores
            score_rows.append(score_row)
            if has_tree_targets:
                raw_visits = players[acting].last_visit_counts
                root_value = players[acting].last_root_value
                if raw_visits is None or root_value is None:
                    raise RuntimeError("tree teacher did not publish search targets")
                visit_row = np.zeros(max_actions, dtype=np.float32)
                if len(raw_visits) != encoded_valid_count:
                    raise RuntimeError(
                        "tree visits do not fit the encoded legal-action ABI"
                    )
                visit_row[: len(raw_visits)] = raw_visits
                visit_rows.append(visit_row)
                root_values.append(float(root_value))
            if receipt is not None:
                target = np.zeros(max_actions, dtype=np.float32)
                target[: len(receipt.target)] = receipt.target
                local_targets.append(target)
                encoded_receipt = receipt.to_json()
                local_receipts.append(encoded_receipt)
                mixing_rows.append(receipt.mixing_diagnostics())
                if journal is not None:
                    with journal.open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "game_index": game_index,
                                    "step": steps,
                                    "action": action,
                                    "receipt": json.loads(encoded_receipt),
                                }
                            )
                            + "\n"
                        )
                        output.flush()
                        os.fsync(output.fileno())
            actions.append(action)
            game_indices.append(game_index)
            seats.append(acting)
            num_valids.append(encoded_valid_count)
            game_decisions.append(len(actions) - 1)
            if is_local:
                for player in players:
                    player.prepare_step(env, acting, action)
                frame = DecisionFrame.from_json(
                    env._engine.semantic_decision_frame_json()
                )
                command = Command(
                    f"teacher-{game_index}-{steps}",
                    frame.revision,
                    int(frame.offers[action]["id"]),
                )
                obs, _, terminated, truncated, info, transition = env.step_semantic(
                    command
                )
                for player in players:
                    player.observe_step(env, acting, transition)
            else:
                obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
        winner = winner_from_info_or_obs(info, env.last_raw_obs) if done else None
        winners.append(winner)
        terminated_games.append(bool(terminated))
        truncated_games.append(bool(truncated or not done))
        winners_per_decision.append(game_decisions)
        steps_per_game.append(steps)
    wall_seconds = time.perf_counter() - wall_start

    winner_column = np.full(len(actions), -1, dtype=np.int8)
    for game_decisions, winner in zip(winners_per_decision, winners):
        if winner is not None:
            winner_column[game_decisions] = winner

    arrays: dict[str, np.ndarray] = {
        key: (
            np.stack(values)
            if values
            else np.zeros((0, *obs_space.shapes[key]), dtype=np.float32)
        )
        for key, values in obs_buffers.items()
    }
    arrays["action"] = np.asarray(actions, dtype=np.int16)
    arrays["game_index"] = np.asarray(game_indices, dtype=np.int32)
    arrays["seat"] = np.asarray(seats, dtype=np.int8)
    arrays["num_valid"] = np.asarray(num_valids, dtype=np.int16)
    arrays["winner"] = winner_column
    arrays[NUM_LEGAL_KEY] = np.asarray(num_legals, dtype=np.int16)
    arrays[DECISION_KIND_KEY] = np.asarray(decision_kinds, dtype=np.int8)
    arrays[SCORE_KEY] = (
        np.stack(score_rows)
        if score_rows
        else np.zeros((0, max_actions), dtype=np.float32)
    )
    if has_tree_targets:
        arrays[VISIT_COUNT_KEY] = (
            np.stack(visit_rows)
            if visit_rows
            else np.zeros((0, max_actions), dtype=np.float32)
        )
        arrays[ROOT_VALUE_KEY] = np.asarray(root_values, dtype=np.float32)

    if is_local:
        arrays[LOCAL_TARGET_KEY] = np.stack(local_targets)
        arrays[LOCAL_RECEIPT_KEY] = np.asarray(local_receipts)

    # Provenance tag (expert-iteration staleness accounting, exp-07): who
    # generated these labels, with what rollout policy, at which round, from
    # which code. Local teachers label frozen-policy trajectories.

    provenance = {
        "round": round_index,
        "teacher_spec": {k: v for k, v in teacher_spec.items() if k != "device"},
        "teacher_name": teacher_name,
        "rollout_policy_checkpoint": teacher_spec.get("checkpoint"),
        "generating_opponent": "frozen-policy" if is_local else teacher_name,
        "git_commit": _git_commit(),
        "seed": seed,
        "game_offset": game_offset,
        "num_games": num_games,
        "match": match.config.model_dump(),
        "observation_hypers": obs_space.encoder.config.model_dump(),
        "content_manifest": env._engine.content_pack_manifest(),
        "dataset_run_fingerprint": dataset_run_fingerprint,
        "policy_target_kind": (
            "local_soft"
            if is_local
            else "visit_distribution"
            if has_tree_targets
            else "score_softmax"
        ),
        "value_target_kind": "root_value" if has_tree_targets else "terminal_outcome",
    }
    arrays["provenance"] = np.array(json.dumps(provenance))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as destination:
                np.savez_compressed(destination, **arrays)
                destination.flush()
                os.fsync(destination.fileno())
            temporary.replace(out_path)
        finally:
            temporary.unlink(missing_ok=True)

    stats = [player.search_stats if is_local else player.stats for player in players]
    search_stats = {
        key: sum(getattr(item, key) for item in stats)
        for key in ("decisions", "seconds", "simulations", "cap_hits")
    }
    if has_tree_targets:
        search_stats.update(
            tree_nodes=players[0].stats.tree_nodes + players[1].stats.tree_nodes,
            worlds_sampled=(
                players[0].stats.worlds_sampled + players[1].stats.worlds_sampled
            ),
            max_depth_sum=(
                players[0].stats.max_depth_sum + players[1].stats.max_depth_sum
            ),
            max_depth_max=max(
                players[0].stats.max_depth_max, players[1].stats.max_depth_max
            ),
        )
    return {
        "teacher": teacher_name,
        "provenance": provenance,
        "num_games": num_games,
        "decisions": len(actions),
        "wall_seconds": wall_seconds,
        "search": search_stats,
        **(
            {
                "local_mixing": {
                    key: float(np.mean([row[key] for row in mixing_rows]))
                    for key in mixing_rows[0]
                }
            }
            if mixing_rows
            else {}
        ),
        "policy_target_kind": provenance["policy_target_kind"],
        "value_target_kind": provenance["value_target_kind"],
        "steps_per_game": steps_per_game,
        "winners": [w if w is not None else -1 for w in winners],
        "terminated": terminated_games,
        "truncated": truncated_games,
        "out_path": str(out_path) if out_path is not None else None,
    }


# -----------------------------------------------------------------------------
# Dataset loading and target preparation
# -----------------------------------------------------------------------------
