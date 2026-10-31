import numpy as np

from manabot.eval.rating import bootstrap_population, fit_population


def rows() -> list[dict[str, object]]:
    result = []
    for block in range(8):
        for seat in (0, 1):
            result.append(
                {
                    "player_a": "strong-v1",
                    "player_b": "random-v1",
                    "player_a_seat": seat,
                    "score_a": 1.0 if block != 0 else 0.0,
                    "deal_block": block,
                }
            )
    return result


def test_map_fit_is_finite_under_separation_and_removes_seat_order() -> None:
    separated = [
        {
            "player_a": "strong-v1",
            "player_b": "random-v1",
            "player_a_seat": index % 2,
            "score_a": 1.0,
            "deal_block": index // 2,
        }
        for index in range(32)
    ]
    fit = fit_population(separated)
    assert fit.converged
    assert np.isfinite(list(fit.ratings.values())).all()
    assert fit.ratings["random-v1"] == 1000.0
    assert fit.ratings["strong-v1"] > 1000.0
    assert abs(fit.seat0_elo) < 1e-6


def test_bootstrap_resamples_global_deal_blocks() -> None:
    evidence = rows()
    bootstrap = bootstrap_population(evidence, replicates=20, seed=99)
    assert bootstrap["replicates"] == 20
    assert bootstrap["failures"] == 0
    assert set(bootstrap["ratings"]) == {"random-v1", "strong-v1"}
    assert "random-v1__minus__strong-v1" in bootstrap["rating_differences"]
