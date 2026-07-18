from manabot.arena.competency import competency_noninferiority
from manabot.verify.competency import SCENARIOS


def test_competency_noninferiority_bootstraps_paired_scenario_seeds() -> None:
    def runs(values: list[bool]) -> list[dict[str, object]]:
        return [
            {"run_seed": index, "correct": value} for index, value in enumerate(values)
        ]

    evidence = {"players": {"candidate": {}, "incumbent": {}}}
    for scenario in SCENARIOS:
        evidence["players"]["candidate"][scenario] = {"runs": runs([True] * 20)}
        evidence["players"]["incumbent"][scenario] = {
            "runs": runs([True] * 18 + [False] * 2)
        }
    result = competency_noninferiority(
        evidence,
        "candidate",
        "incumbent",
        margin=0.10,
        bootstrap_seed=41,
        bootstrap_replicates=200,
    )
    assert result["available"]
    assert result["passed"]
    assert all(
        block["point_difference"] == 0.1 for block in result["scenarios"].values()
    )
