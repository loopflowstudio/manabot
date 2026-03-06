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
from manabot.infra.metrics import MetricsDB

TRUNCATION_INFO_KEYS = (
    "action_space_truncated",
    "card_space_truncated",
    "permanent_space_truncated",
)

ACTION_TYPE_SHORT_NAMES = {
    ActionEnum.PRIORITY_PLAY_LAND: "land",
    ActionEnum.PRIORITY_CAST_SPELL: "spell",
    ActionEnum.PRIORITY_PASS_PRIORITY: "pass",
    ActionEnum.DECLARE_ATTACKER: "attack",
    ActionEnum.DECLARE_BLOCKER: "block",
    ActionEnum.CHOOSE_TARGET: "target",
}

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


def _is_action_type(obs: dict[str, np.ndarray], action_index: int, action_type: ActionEnum) -> bool:
    if action_index < 0 or action_index >= obs["actions"].shape[0]:
        return False
    return bool(obs["actions"][action_index, int(action_type)] > 0)


def _is_attack_action(obs: dict[str, np.ndarray], action_index: int) -> bool:
    return _is_action_type(obs, action_index, ActionEnum.DECLARE_ATTACKER)


def _is_land_action(obs: dict[str, np.ndarray], action_index: int) -> bool:
    return _is_action_type(obs, action_index, ActionEnum.PRIORITY_PLAY_LAND)


def _is_spell_action(obs: dict[str, np.ndarray], action_index: int) -> bool:
    return _is_action_type(obs, action_index, ActionEnum.PRIORITY_CAST_SPELL)


def _is_pass_action(obs: dict[str, np.ndarray], action_index: int) -> bool:
    return _is_action_type(obs, action_index, ActionEnum.PRIORITY_PASS_PRIORITY)


def _action_type_available(obs: dict[str, np.ndarray], action_type: ActionEnum) -> bool:
    """Check if any valid action in the observation has the given type."""
    actions = obs["actions"]
    valid = obs["actions_valid"]
    col = int(action_type)
    for i in range(actions.shape[0]):
        if valid[i] > 0 and actions[i, col] > 0:
            return True
    return False


def _attack_available(obs: dict[str, np.ndarray]) -> bool:
    return _action_type_available(obs, ActionEnum.DECLARE_ATTACKER)


def _land_available(obs: dict[str, np.ndarray]) -> bool:
    return _action_type_available(obs, ActionEnum.PRIORITY_PLAY_LAND)


def _spell_available(obs: dict[str, np.ndarray]) -> bool:
    return _action_type_available(obs, ActionEnum.PRIORITY_CAST_SPELL)


def _pass_available(obs: dict[str, np.ndarray]) -> bool:
    return _action_type_available(obs, ActionEnum.PRIORITY_PASS_PRIORITY)


def _valid_action_indices(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.flatnonzero(obs["actions_valid"] > 0)


def _choice_set_name(obs: dict[str, np.ndarray]) -> str:
    available = [
        short_name
        for action_type, short_name in ACTION_TYPE_SHORT_NAMES.items()
        if _action_type_available(obs, action_type)
    ]
    return "+".join(available) if available else "none"


def _priority_choice_name(obs: dict[str, np.ndarray]) -> str:
    parts = []
    if _land_available(obs):
        parts.append("land")
    if _spell_available(obs):
        parts.append("spell")
    if _pass_available(obs):
        parts.append("pass")
    return "+".join(parts) if parts else "none"


def _policy_action_type_probabilities(agent, obs: dict[str, np.ndarray]) -> dict[str, float] | None:
    if not hasattr(agent, "forward"):
        return None

    device = torch.device("cpu")
    try:
        device = next(agent.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        device = torch.device("cpu")

    tensor_obs = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }
    with torch.no_grad():
        logits, _ = agent.forward(tensor_obs)

    probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
    action_probs = {name: 0.0 for name in ACTION_TYPE_SHORT_NAMES.values()}
    for idx in _valid_action_indices(obs):
        for action_type, short_name in ACTION_TYPE_SHORT_NAMES.items():
            if obs["actions"][idx, int(action_type)] > 0:
                action_probs[short_name] += float(probs[idx])
    return action_probs


def _action_type_name(obs: dict[str, np.ndarray], action_index: int) -> str:
    """Get the action type name for a chosen action."""
    if action_index < 0 or action_index >= obs["actions"].shape[0]:
        return "unknown"
    action_vec = obs["actions"][action_index]
    for ae in ActionEnum:
        if ae.value < len(action_vec) and action_vec[ae.value] > 0:
            return ACTION_TYPE_SHORT_NAMES.get(ae, ae.name.lower())
    return "unknown"


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
    deterministic: bool = False,
    seed: int = 0,
    metrics_db: MetricsDB | None = None,
    model_name: str = "",
    model_step: int = 0,
) -> dict[str, float]:
    """Run evaluation games and return win/CI/length/action/truncation metrics.

    If metrics_db is provided, logs per-game and per-action data to DuckDB.
    """

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

    sim_id = None
    if metrics_db:
        sim_id = metrics_db.start_sim(
            model_name=model_name,
            model_step=model_step,
            opponent=opponent_policy,
            num_games=num_games,
        )

    hero_wins = 0
    game_lengths: list[int] = []
    hero_actions = 0
    hero_attack_actions = 0
    hero_pass_actions = 0
    hero_could_attack = 0
    hero_could_pass = 0
    hero_land_plays = 0
    hero_could_land = 0
    hero_spell_casts = 0
    hero_could_spell = 0
    single_valid_decisions = 0
    multi_valid_decisions = 0
    pass_land_decisions = 0
    pass_land_passes = 0
    pass_land_land_plays = 0
    pass_land_spell_casts = 0
    priority_choice_counts = {
        "land": 0,
        "spell": 0,
        "pass": 0,
        "land+spell": 0,
        "land+pass": 0,
        "spell+pass": 0,
        "land+spell+pass": 0,
        "none": 0,
    }
    prob_sum_count = 0
    prob_sums = {name: 0.0 for name in ACTION_TYPE_SHORT_NAMES.values()}
    land_available_prob_count = 0
    pass_land_prob_count = 0
    pass_prob_when_land_available = 0.0
    land_prob_when_land_available = 0.0
    pass_prob_when_pass_land = 0.0
    land_prob_when_pass_land = 0.0
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
            game_hero_attacks = 0
            game_hero_actions = 0
            game_could_attack = 0
            game_land_plays = 0
            game_could_land = 0
            game_spell_casts = 0
            game_could_spell = 0

            while not done:
                active_player = int(env.last_raw_obs.agent.player_index)
                if active_player == 0:
                    action = _select_agent_action(
                        agent, obs, deterministic=deterministic
                    )
                    is_attack = _is_attack_action(obs, action)
                    is_pass = _is_pass_action(obs, action)
                    is_land = _is_land_action(obs, action)
                    is_spell = _is_spell_action(obs, action)
                    has_attack = _attack_available(obs)
                    has_pass = _pass_available(obs)
                    has_land = _land_available(obs)
                    has_spell = _spell_available(obs)
                    num_valid = int((_valid_action_indices(obs)).shape[0])
                    choice_set = _choice_set_name(obs)
                    priority_choice = _priority_choice_name(obs)
                    is_trivial = num_valid <= 1
                    if is_trivial:
                        single_valid_decisions += 1
                    else:
                        multi_valid_decisions += 1
                    if priority_choice in priority_choice_counts:
                        priority_choice_counts[priority_choice] += 1
                    else:
                        priority_choice_counts[priority_choice] = 1

                    game_hero_actions += 1
                    hero_actions += 1
                    if is_attack:
                        game_hero_attacks += 1
                        hero_attack_actions += 1
                    if is_pass:
                        hero_pass_actions += 1
                    if has_attack:
                        game_could_attack += 1
                        hero_could_attack += 1
                    if has_pass:
                        hero_could_pass += 1
                    if is_land:
                        game_land_plays += 1
                        hero_land_plays += 1
                    if has_land:
                        game_could_land += 1
                        hero_could_land += 1
                    if is_spell:
                        game_spell_casts += 1
                        hero_spell_casts += 1
                    if has_spell:
                        game_could_spell += 1
                        hero_could_spell += 1

                    if has_pass and has_land:
                        pass_land_decisions += 1
                        if is_pass:
                            pass_land_passes += 1
                        if is_land:
                            pass_land_land_plays += 1
                        if is_spell:
                            pass_land_spell_casts += 1

                    action_type_probs = _policy_action_type_probabilities(agent, obs)
                    if action_type_probs is not None:
                        prob_sum_count += 1
                        for key, value in action_type_probs.items():
                            prob_sums[key] += value
                        if has_land:
                            land_available_prob_count += 1
                            pass_prob_when_land_available += action_type_probs["pass"]
                            land_prob_when_land_available += action_type_probs["land"]
                        if has_pass and has_land:
                            pass_land_prob_count += 1
                            pass_prob_when_pass_land += action_type_probs["pass"]
                            land_prob_when_pass_land += action_type_probs["land"]

                    if metrics_db and sim_id:
                        metrics_db.log_action(
                            sim_id=sim_id,
                            game_index=game_index,
                            step=steps,
                            player=0,
                            action_type=_action_type_name(obs, action),
                            choice_set=choice_set,
                            is_trivial=is_trivial,
                            is_attack=is_attack,
                            attack_available=has_attack,
                            pass_available=has_pass,
                            land_available=has_land,
                            spell_available=has_spell,
                            pass_prob=(
                                action_type_probs["pass"]
                                if action_type_probs is not None
                                else None
                            ),
                            land_prob=(
                                action_type_probs["land"]
                                if action_type_probs is not None
                                else None
                            ),
                            spell_prob=(
                                action_type_probs["spell"]
                                if action_type_probs is not None
                                else None
                            ),
                            attack_prob=(
                                action_type_probs["attack"]
                                if action_type_probs is not None
                                else None
                            ),
                            num_valid_actions=num_valid,
                        )
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
                outcome = "aborted"
            else:
                winner = winner_from_info_or_obs(info, env.last_raw_obs)
                if winner == 0:
                    hero_wins += 1
                    outcome = "hero_win"
                else:
                    outcome = "villain_win"

            game_lengths.append(steps)

            if metrics_db and sim_id:
                metrics_db.log_game(
                    sim_id=sim_id,
                    game_index=game_index,
                    outcome=outcome,
                    steps=steps,
                    hero_attacks=game_hero_attacks,
                    hero_actions=game_hero_actions,
                    could_attack=game_could_attack,
                )
    finally:
        env.close()
        if hasattr(agent, "train") and was_training:
            agent.train()

    win_rate = hero_wins / num_games if num_games > 0 else 0.0
    attack_rate = hero_attack_actions / hero_actions if hero_actions > 0 else 0.0
    passed_when_able = hero_pass_actions / hero_could_pass if hero_could_pass > 0 else 0.0
    attacked_when_able = (
        hero_attack_actions / hero_could_attack if hero_could_attack > 0 else 0.0
    )
    landed_when_able = hero_land_plays / hero_could_land if hero_could_land > 0 else 0.0
    cast_when_able = hero_spell_casts / hero_could_spell if hero_could_spell > 0 else 0.0
    pass_land_pass_rate = pass_land_passes / pass_land_decisions if pass_land_decisions > 0 else 0.0
    pass_land_land_rate = (
        pass_land_land_plays / pass_land_decisions if pass_land_decisions > 0 else 0.0
    )
    pass_land_spell_rate = (
        pass_land_spell_casts / pass_land_decisions if pass_land_decisions > 0 else 0.0
    )
    mean_pass_prob = prob_sums["pass"] / prob_sum_count if prob_sum_count > 0 else 0.0
    mean_land_prob = prob_sums["land"] / prob_sum_count if prob_sum_count > 0 else 0.0
    mean_spell_prob = prob_sums["spell"] / prob_sum_count if prob_sum_count > 0 else 0.0
    mean_pass_prob_when_land_available = (
        pass_prob_when_land_available / land_available_prob_count
        if land_available_prob_count > 0
        else 0.0
    )
    mean_land_prob_when_land_available = (
        land_prob_when_land_available / land_available_prob_count
        if land_available_prob_count > 0
        else 0.0
    )
    mean_pass_prob_when_pass_land = (
        pass_prob_when_pass_land / pass_land_prob_count
        if pass_land_prob_count > 0
        else 0.0
    )
    mean_land_prob_when_pass_land = (
        land_prob_when_pass_land / pass_land_prob_count
        if pass_land_prob_count > 0
        else 0.0
    )

    if metrics_db and sim_id:
        metrics_db.finish_sim(
            sim_id=sim_id,
            win_rate=win_rate,
            attack_rate=attack_rate,
            mean_steps=float(np.mean(game_lengths)) if game_lengths else 0.0,
        )

    return {
        "num_games": float(num_games),
        "wins": float(hero_wins),
        "win_rate": win_rate,
        "win_ci_lower": wilson_lower_bound(hero_wins, num_games),
        "mean_steps": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "attack_rate": attack_rate,
        "passed_when_able": passed_when_able,
        "could_pass": float(hero_could_pass),
        "attacked_when_able": attacked_when_able,
        "could_attack": float(hero_could_attack),
        "landed_when_able": landed_when_able,
        "could_land": float(hero_could_land),
        "land_plays": float(hero_land_plays),
        "cast_when_able": cast_when_able,
        "could_spell": float(hero_could_spell),
        "spell_casts": float(hero_spell_casts),
        "single_valid_decisions": float(single_valid_decisions),
        "multi_valid_decisions": float(multi_valid_decisions),
        "pass_land_decisions": float(pass_land_decisions),
        "pass_land_pass_rate": pass_land_pass_rate,
        "pass_land_land_rate": pass_land_land_rate,
        "pass_land_spell_rate": pass_land_spell_rate,
        "mean_pass_prob": mean_pass_prob,
        "mean_land_prob": mean_land_prob,
        "mean_spell_prob": mean_spell_prob,
        "mean_pass_prob_when_land_available": mean_pass_prob_when_land_available,
        "mean_land_prob_when_land_available": mean_land_prob_when_land_available,
        "mean_pass_prob_when_pass_land": mean_pass_prob_when_pass_land,
        "mean_land_prob_when_pass_land": mean_land_prob_when_pass_land,
        "priority_choice_land": float(priority_choice_counts["land"]),
        "priority_choice_spell": float(priority_choice_counts["spell"]),
        "priority_choice_pass": float(priority_choice_counts["pass"]),
        "priority_choice_land_spell": float(priority_choice_counts["land+spell"]),
        "priority_choice_land_pass": float(priority_choice_counts["land+pass"]),
        "priority_choice_spell_pass": float(priority_choice_counts["spell+pass"]),
        "priority_choice_land_spell_pass": float(priority_choice_counts["land+spell+pass"]),
        "priority_choice_none": float(priority_choice_counts["none"]),
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
