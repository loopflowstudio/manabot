"""Learning decisions need paired outcome evidence, not policy drift."""

from pathlib import Path

import pytest

from manabot.recipes.baseline import BaselineConfig, settings, train
from manabot.verify.learning_regression import (
    BaselineReport,
    ChoiceScore,
    LearningEvidence,
    PairedDeal,
    PolicySnapshot,
    aggregate_reports,
    family_changes,
    learning_evidence,
    report_recipe,
)


def cohort(gain: float) -> list[PairedDeal]:
    return [
        PairedDeal(deal=i, initial=0.5, final=0.5 + gain, legs=[gain] * 4)
        for i in range(24)
    ]


def test_paired_learning_decision_and_no_update_control() -> None:
    assert (
        learning_evidence([cohort(0), cohort(0), cohort(0)]).verdict == "inconclusive"
    )
    assert (
        learning_evidence([cohort(0.125), cohort(0.25), cohort(0.125)]).verdict
        == "learning"
    )
    assert learning_evidence([cohort(-0.25)] * 3).verdict == "regression"
    assert learning_evidence([cohort(0.01)] * 3).verdict == "inconclusive"
    assert learning_evidence([cohort(0.25)[:2]]).verdict == "inconclusive"
    assert learning_evidence([]).verdict == "error"
    # Many positions or games cannot disguise disagreement between training seeds.
    assert (
        learning_evidence([cohort(0.25), cohort(0.25), cohort(-0.25)]).verdict
        == "inconclusive"
    )
    with pytest.raises(ValueError, match="same unique deal"):
        learning_evidence([cohort(0.25), cohort(0.25)[1:]])


def test_probability_drift_is_descriptive_and_corpus_must_match() -> None:
    before = PolicySnapshot(
        updates=0,
        checkpoint_sha256="a",
        maximum_invariance_error=0,
        choices=[
            ChoiceScore(case="a", family=2, log_odds=0, value=0),
            ChoiceScore(case="b", family=2, log_odds=0, value=0),
        ],
    )
    after = before.model_copy(deep=True)
    after.updates = 100
    after.choices[0].log_odds = 10
    after.choices[1].log_odds = -10
    (change,) = family_changes(before, after)
    assert change.mean_absolute_log_odds_change == 10
    assert change.std_log_odds_change == 10
    assert learning_evidence([cohort(0)]).verdict == "inconclusive"
    after.choices.reverse()
    with pytest.raises(ValueError, match="corpus changed"):
        family_changes(before, after)


def test_aggregate_rejects_missing_seeds(tmp_path: Path) -> None:
    report = BaselineReport(
        seed=1,
        regime_digest="a",
        world="w6",
        steps=256,
        software="passed",
        learning=LearningEvidence(
            verdict="learning", seeds=1, deals=24, reason="fixture"
        ),
        paired=cohort(0.25),
    )
    path = tmp_path / "seed.json"
    path.write_text(report.model_dump_json())
    evidence = aggregate_reports([path], tmp_path / "out", expected_seeds=(1, 2, 3))
    assert evidence.verdict == "error"


def test_complete_cohort_keeps_no_update_inconclusive(tmp_path: Path) -> None:
    paths = []
    for seed in (1, 2, 3):
        report = BaselineReport(
            seed=seed,
            regime_digest="a",
            source="b",
            corpus_sha256="c",
            world="w6",
            steps=256,
            software="passed",
            learning=learning_evidence([cohort(0)]),
            paired=cohort(0),
        )
        path = tmp_path / f"{seed}.json"
        path.write_text(report.model_dump_json())
        paths.append(path)
    assert (
        aggregate_reports(paths, tmp_path / "out", expected_seeds=(1, 2, 3)).verdict
        == "inconclusive"
    )
    paths[-1].unlink()
    assert (
        aggregate_reports(paths, tmp_path / "missing", expected_seeds=(1, 2, 3)).verdict
        == "error"
    )


def test_recipe_report_preserves_paired_deals_and_policy_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    config = BaselineConfig(updates=1, device="cpu")
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path))
    monkeypatch.setenv("ETUDE_EXPERIMENT", "baseline")
    monkeypatch.setenv("ETUDE_RUN", "baseline-seed-197")
    monkeypatch.setenv("ETUDE_SEED", "197")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    train(config)
    report = report_recipe(
        tmp_path,
        settings(config),
        seed=197,
        updates=1,
        deals=(1_915_100_000, 1_915_100_001),
        seconds=90,
    )
    assert report.software == "passed", report.error
    assert report.learning.verdict == "inconclusive"
    assert len(report.paired) == 2
    assert all(len(deal.legs) == 4 for deal in report.paired)
    assert [policy.updates for policy in report.policies] == [0, 1]
    assert sum(item.retained for item in report.exposure) > 0
    assert (tmp_path / "learning.json").exists()
