"""Verification step 0: environment sanity checks without training."""

import argparse

from manabot.env import (
    Env,
    Match,
    ObservationSpace,
    PassivePolicy,
    RandomPolicy,
    Reward,
)

from .util import build_hypers, print_result, suppress_truncation_logs

_TRUNCATION_INFO_KEYS = (
    "action_space_truncated",
    "card_space_truncated",
    "permanent_space_truncated",
)


def _winner(info: dict, raw_obs) -> int | None:
    if "winner" in info:
        try:
            return int(info["winner"])
        except (TypeError, ValueError):
            return None

    agent_idx = int(raw_obs.agent.player_index)
    opp_idx = int(raw_obs.opponent.player_index)
    if raw_obs.agent.life <= 0:
        return opp_idx
    if raw_obs.opponent.life <= 0:
        return agent_idx
    return None


def _run_matchup(
    hero_policy,
    villain_policy,
    *,
    num_games: int,
    seed: int,
) -> dict[str, float]:
    hypers = build_hypers()
    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    env = Env(match, obs_space, reward, seed=seed, auto_reset=False)

    hero_wins = 0
    truncation_counts = {key: 0 for key in _TRUNCATION_INFO_KEYS}

    try:
        for game in range(num_games):
            obs, _ = env.reset(seed=seed + game)
            done = False
            aborted = False
            while not done:
                active_player = int(env.last_raw_obs.agent.player_index)
                policy = hero_policy if active_player == 0 else villain_policy
                action = policy(obs)
                try:
                    obs, _, terminated, truncated, info = env.step(action)
                except Exception:
                    try:
                        obs, _, terminated, truncated, info = env.step(0)
                    except Exception:
                        aborted = True
                        info = {}
                        break
                done = bool(terminated or truncated)
                for key in _TRUNCATION_INFO_KEYS:
                    truncation_counts[key] += int(bool(info.get(key, False)))

            if aborted:
                continue
            if _winner(info, env.last_raw_obs) == 0:
                hero_wins += 1
    finally:
        env.close()

    return {
        "games": float(num_games),
        "hero_wins": float(hero_wins),
        "hero_win_rate": hero_wins / num_games,
        "action_space_truncations": float(truncation_counts["action_space_truncated"]),
        "card_space_truncations": float(truncation_counts["card_space_truncated"]),
        "permanent_space_truncations": float(
            truncation_counts["permanent_space_truncated"]
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    random_vs_random = _run_matchup(
        RandomPolicy(),
        RandomPolicy(),
        num_games=args.num_games,
        seed=args.seed,
    )
    random_vs_passive = _run_matchup(
        RandomPolicy(),
        PassivePolicy(),
        num_games=args.num_games,
        seed=args.seed + 10_000,
    )

    total_truncations = (
        random_vs_random["action_space_truncations"]
        + random_vs_random["card_space_truncations"]
        + random_vs_random["permanent_space_truncations"]
        + random_vs_passive["action_space_truncations"]
        + random_vs_passive["card_space_truncations"]
        + random_vs_passive["permanent_space_truncations"]
    )

    metrics = {
        "random_vs_random_win_rate": random_vs_random["hero_win_rate"],
        "random_vs_passive_win_rate": random_vs_passive["hero_win_rate"],
        "random_vs_random_action_truncations": random_vs_random[
            "action_space_truncations"
        ],
        "random_vs_random_card_truncations": random_vs_random["card_space_truncations"],
        "random_vs_random_permanent_truncations": random_vs_random[
            "permanent_space_truncations"
        ],
        "random_vs_passive_action_truncations": random_vs_passive[
            "action_space_truncations"
        ],
        "random_vs_passive_card_truncations": random_vs_passive[
            "card_space_truncations"
        ],
        "random_vs_passive_permanent_truncations": random_vs_passive[
            "permanent_space_truncations"
        ],
    }

    passed = (
        0.40 <= random_vs_random["hero_win_rate"] <= 0.60
        and random_vs_passive["hero_win_rate"] > 0.95
        and total_truncations == 0
    )
    print_result("step0_env_sanity", passed, metrics)


if __name__ == "__main__":
    main()
