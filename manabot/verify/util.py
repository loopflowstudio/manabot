"""Shared helpers for verification ladder scripts."""

import logging
import math
from typing import Any

import numpy as np
import torch

from manabot.config.load import deep_merge
from manabot.env import Env, Match, ObservationSpace, Reward, build_opponent_policy
from manabot.env.observation import ActionEnum
from manabot.infra import Hypers

TRUNCATION_INFO_KEYS = (
    "action_space_truncated",
    "card_space_truncated",
    "permanent_space_truncated",
)

STANDARD_DECK = {
    "Mountain": 12,
    "Forest": 12,
    "Llanowar Elves": 18,
    "Grey Ogre": 18,
}
MOUNTAIN_DECK = {"Mountain": 20}


def build_hypers(**overrides) -> Hypers:
    """Build Hypers with verification defaults and nested overrides."""

    base = Hypers().model_dump()
    base["experiment"].update(
        {
            "exp_name": "verify",
            "wandb": False,
            "device": "cpu",
            "profiler_enabled": False,
            "log_level": "INFO",
        }
    )
    base["agent"]["attention_on"] = False
    base["train"]["opponent_policy"] = "passive"

    match_overrides = overrides.pop("match", None)
    merged = deep_merge(base, overrides)
    if isinstance(match_overrides, dict):
        merged["match"] = {**base["match"], **match_overrides}
    elif match_overrides is not None:
        merged["match"] = match_overrides
    return Hypers.model_validate(merged)


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Wilson score lower confidence bound for a Bernoulli rate."""

    if total <= 0:
        return 0.0
    p = wins / total
    denom = 1.0 + (z**2) / total
    center = p + (z**2) / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def _select_agent_action(agent, obs: dict[str, np.ndarray], deterministic: bool) -> int:
    device = torch.device("cpu")
    try:
        device = next(agent.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        device = torch.device("cpu")

    tensor_obs = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }
    action, _, _, _ = agent.get_action_and_value(
        tensor_obs,
        deterministic=deterministic,
    )
    return int(action.item())


def _is_attack_action(obs: dict[str, np.ndarray], action_index: int) -> bool:
    if action_index < 0 or action_index >= obs["actions"].shape[0]:
        return False
    return bool(obs["actions"][action_index, int(ActionEnum.DECLARE_ATTACKER)] > 0)


def winner_from_info_or_obs(info: dict[str, Any], raw_obs) -> int | None:
    """Return winner player index when available, else None."""

    if "winner_index" in info:
        try:
            return int(info["winner_index"])
        except (TypeError, ValueError):
            pass

    # Fallback: infer from life totals (perspective-aware).
    agent_idx = int(raw_obs.agent.player_index)
    opp_idx = int(raw_obs.opponent.player_index)

    if raw_obs.agent.life <= 0:
        return opp_idx
    if raw_obs.opponent.life <= 0:
        return agent_idx

    return None


def _pass_priority_fallback(env: Env) -> int | None:
    """Return pass-priority action index for the current raw action space, if present."""

    raw_obs = getattr(env, "last_raw_obs", None)
    action_space = getattr(raw_obs, "action_space", None)
    actions = getattr(action_space, "actions", None)
    if actions is None:
        return None

    pass_priority_type = int(ActionEnum.PRIORITY_PASS_PRIORITY)
    for idx, option in enumerate(actions):
        if int(option.action_type) == pass_priority_type:
            return idx
    return None


def step_with_fallback(env: Env, action: int, fallback_action: int = 0):
    """Step env, retrying with safer fallbacks when policy action errors."""

    try:
        return env.step(action)
    except Exception as original_error:
        candidates: list[int] = []
        if fallback_action >= 0:
            candidates.append(int(fallback_action))

        pass_idx = _pass_priority_fallback(env)
        if pass_idx is not None and pass_idx not in candidates:
            candidates.append(pass_idx)

        if 0 not in candidates:
            candidates.append(0)

        last_error = original_error
        for candidate in candidates:
            try:
                return env.step(candidate)
            except Exception as error:
                last_error = error

        raise last_error


def run_evaluation(
    agent,
    obs_space: ObservationSpace,
    match: Match,
    reward: Reward,
    *,
    num_games: int = 100,
    opponent_policy: str = "passive",
    deterministic: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Run evaluation games and return win/CI/length/action/truncation metrics."""

    env = Env(
        match,
        obs_space,
        reward,
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    opponent = build_opponent_policy(opponent_policy)

    hero_wins = 0
    game_lengths: list[int] = []
    hero_actions = 0
    hero_attack_actions = 0
    truncation_counts = {k: 0 for k in TRUNCATION_INFO_KEYS}

    was_training = bool(getattr(agent, "training", False))
    if hasattr(agent, "eval"):
        agent.eval()

    try:
        for game_index in range(num_games):
            obs, _ = env.reset(seed=seed + game_index)
            done = False
            steps = 0
            aborted = False

            while not done:
                active_player = int(env.last_raw_obs.agent.player_index)
                if active_player == 0:
                    action = _select_agent_action(
                        agent, obs, deterministic=deterministic
                    )
                    hero_actions += 1
                    if _is_attack_action(obs, action):
                        hero_attack_actions += 1
                else:
                    action = opponent(obs)

                try:
                    obs, _, terminated, truncated, info = step_with_fallback(
                        env, action
                    )
                except Exception:
                    aborted = True
                    info = {}
                    break
                steps += 1
                for key in TRUNCATION_INFO_KEYS:
                    truncation_counts[key] += int(bool(info.get(key, False)))
                done = bool(terminated or truncated)

            if aborted:
                game_lengths.append(steps)
                continue
            winner = winner_from_info_or_obs(info, env.last_raw_obs)
            if winner == 0:
                hero_wins += 1
            game_lengths.append(steps)
    finally:
        env.close()
        if hasattr(agent, "train") and was_training:
            agent.train()

    win_rate = hero_wins / num_games if num_games > 0 else 0.0
    attack_rate = hero_attack_actions / hero_actions if hero_actions > 0 else 0.0

    return {
        "num_games": float(num_games),
        "wins": float(hero_wins),
        "win_rate": win_rate,
        "win_ci_lower": wilson_lower_bound(hero_wins, num_games),
        "mean_steps": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "attack_rate": attack_rate,
        "action_space_truncations": float(truncation_counts["action_space_truncated"]),
        "card_space_truncations": float(truncation_counts["card_space_truncated"]),
        "permanent_space_truncations": float(
            truncation_counts["permanent_space_truncated"]
        ),
    }


def print_result(
    step_name: str,
    passed: bool,
    metrics: dict[str, Any],
    checks: list[tuple[str, bool, str] | tuple[str, bool, str, str]] | None = None,
) -> None:
    """Print PASS/FAIL with check details and sorted metrics."""

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {step_name}")

    if checks:
        print()
        for check in checks:
            description, ok, detail = check[0], check[1], check[2]
            explanation = check[3] if len(check) > 3 else None
            mark = "ok" if ok else "FAILED"
            print(f"  [{mark}] {description} — {detail}")
            if not ok and explanation:
                print(f"         {explanation}")
        print()

    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def suppress_truncation_logs() -> None:
    """Reduce noisy truncation warnings; scripts read truncation counters directly."""

    logging.getLogger("manabot.env.observation.encode_actions").setLevel(logging.ERROR)
    logging.getLogger("manabot.env.observation.encode_cards").setLevel(logging.ERROR)
    logging.getLogger("manabot.env.observation.encode_permanents").setLevel(
        logging.ERROR
    )
