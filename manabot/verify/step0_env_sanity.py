"""
step0_env_sanity.py
Environment sanity checks without training.

Validates that the observation space fits the deck, random play produces
decisive games, and a random agent reliably beats a passive opponent.
"""

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
    aborts = 0
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
                    obs, _, terminated, truncated, info = step_with_fallback(
                        env, action
                    )
                except Exception:
                    aborted = True
                    info = {}
                    break
                done = bool(terminated or truncated)
                for key in TRUNCATION_INFO_KEYS:
                    truncation_counts[key] += int(bool(info.get(key, False)))

            if aborted:
                aborts += 1
                continue
            if winner_from_info_or_obs(info, env.last_raw_obs) == 0:
                hero_wins += 1
    finally:
        env.close()

    return {
        "hero_win_rate": hero_wins / num_games,
        "aborts": float(aborts),
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
    total_aborts = random_vs_random["aborts"] + random_vs_passive["aborts"]

    checks = [
        (
            "random vs random: hero wins > 15%",
            random_vs_random["hero_win_rate"] > 0.15,
            f"{random_vs_random['hero_win_rate']:.1%}",
            "Sanity check that both players can win. Second-player advantage "
            "with Llanowar Elves makes the mirror asymmetric (~25/75).",
        ),
        (
            "random vs passive: hero wins > 90%",
            random_vs_passive["hero_win_rate"] > 0.90,
            f"{random_vs_passive['hero_win_rate']:.1%}",
            "Random play should reliably beat doing nothing.",
        ),
        (
            "zero observation truncations",
            total_truncations == 0,
            f"{int(total_truncations)} total",
            "Observation space limits must fit the deck. "
            "Increase max_cards/permanents/actions in ObservationSpaceHypers.",
        ),
        (
            "zero engine aborts",
            total_aborts == 0,
            f"{int(total_aborts)} total",
            "Step exceptions indicate engine bugs. Check managym logs.",
        ),
    ]

    metrics = {}
    for matchup_name, matchup_metrics in (
        ("random_vs_random", random_vs_random),
        ("random_vs_passive", random_vs_passive),
    ):
        metrics[f"{matchup_name}_win_rate"] = matchup_metrics["hero_win_rate"]
        metrics[f"{matchup_name}_aborts"] = matchup_metrics["aborts"]
        for key in TRUNCATION_INFO_KEYS:
            short_key = key.replace("_space_truncated", "_truncations")
            metrics[f"{matchup_name}_{short_key}"] = matchup_metrics[key]

    passed = all(ok for _, ok, _, _ in checks)
    print_result("step0_env_sanity", passed, metrics, checks=checks)


if __name__ == "__main__":
    main()
