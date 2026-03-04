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

from .util import (
    TRUNCATION_INFO_KEYS,
    build_hypers,
    print_result,
    step_with_fallback,
    suppress_truncation_logs,
    winner_from_info_or_obs,
)


def _run_matchup(
    hero_policy,
    villain_policy,
    obs_space: ObservationSpace,
    match: Match,
    reward: Reward,
    *,
    num_games: int,
    seed: int,
) -> dict[str, float]:
    env = Env(match, obs_space, reward, seed=seed, auto_reset=False)

    hero_wins = 0
    truncation_counts = {key: 0 for key in TRUNCATION_INFO_KEYS}

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
                    obs, _, terminated, truncated, info = step_with_fallback(env, action)
                except Exception:
                    aborted = True
                    info = {}
                    break
                done = bool(terminated or truncated)
                for key in TRUNCATION_INFO_KEYS:
                    truncation_counts[key] += int(bool(info.get(key, False)))

            if aborted:
                continue
            if winner_from_info_or_obs(info, env.last_raw_obs) == 0:
                hero_wins += 1
    finally:
        env.close()

    return {
        "hero_win_rate": hero_wins / num_games,
        **{key: float(value) for key, value in truncation_counts.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    hypers = build_hypers()
    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)

    random_vs_random = _run_matchup(
        RandomPolicy(),
        RandomPolicy(),
        obs_space,
        match,
        reward,
        num_games=args.num_games,
        seed=args.seed,
    )
    random_vs_passive = _run_matchup(
        RandomPolicy(),
        PassivePolicy(),
        obs_space,
        match,
        reward,
        num_games=args.num_games,
        seed=args.seed + 10_000,
    )

    total_truncations = sum(
        random_vs_random[key] + random_vs_passive[key] for key in TRUNCATION_INFO_KEYS
    )

    metrics = {
        "random_vs_random_win_rate": random_vs_random["hero_win_rate"],
        "random_vs_passive_win_rate": random_vs_passive["hero_win_rate"],
    }
    for matchup_name, matchup_metrics in (
        ("random_vs_random", random_vs_random),
        ("random_vs_passive", random_vs_passive),
    ):
        for key in TRUNCATION_INFO_KEYS:
            short_key = key.replace("_space_truncated", "_truncations")
            metrics[f"{matchup_name}_{short_key}"] = matchup_metrics[key]

    passed = (
        0.40 <= random_vs_random["hero_win_rate"] <= 0.60
        and random_vs_passive["hero_win_rate"] > 0.95
        and total_truncations == 0
    )
    print_result("step0_env_sanity", passed, metrics)


if __name__ == "__main__":
    main()
