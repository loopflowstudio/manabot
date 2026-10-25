"""Learning decisions need paired outcome evidence, not policy drift."""

from pathlib import Path

import pytest
import torch

from manabot.env import Match, ObservationSpace
from manabot.experiment.checkpoint_evaluation import checkpoint_manifest
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.recipes.baseline import baseline
from manabot.training.records import TrainingCoordinates, TrainingRun
from manabot.verify.action_cases import decision_case
from manabot.verify.choice_corpus import Case
from manabot.verify.learning_regression import (
    BaselineReport,
    ChoiceScore,
    LearningEvidence,
    PairedDeal,
    PolicySnapshot,
    aggregate_reports,
    family_changes,
    inspect_policy,
    learning_evidence,
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


def test_frozen_policy_checks_use_ordinary_checkpoint_without_training(
    tmp_path: Path,
) -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            plan = baseline(1)
            agent = ActorCritic(ObservationSpace(), plan.regime.agent)
            path = tmp_path / "initial.pt"
            save_checkpoint(
                agent,
                ObservationSpace(),
                path,
                player_configs=Match(plan.regime.match).to_rust(),
            )
            run = TrainingRun(
                id="test",
                regime_digest=canonical_sha256(plan.regime.model_dump()),
                regime=plan.regime,
                seed=197,
                seed_streams={},
                identities=runtime_fingerprints(
                    197,
                    match_config=plan.regime.match,
                    observation_space=ObservationSpace(),
                ),
            )
            admitted = checkpoint_manifest(
                run,
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                },
                TrainingCoordinates(
                    stage_id="self-play", updates=0, training_seconds=0
                ),
                plan.evaluation,
            )
            assert admitted.expected_games == 96
            fixture = decision_case("attack")
            case = Case(
                name="attack",
                seed=0,
                swapped=False,
                trace=[],
                digest="test",
                family=2,
                groups=([fixture.candidates[0]], [fixture.candidates[1]]),
                signatures=[],
                raw_json="{}",
                obs=fixture.observation,
                successors=("a", "b"),
            )
            a = inspect_policy(path, [case], 0)
            b = inspect_policy(path, [case], 0)
            assert a == b
            assert a.maximum_invariance_error < 1e-4
            assert family_changes(a, b)[0].mean_absolute_log_odds_change == 0
    finally:
        torch.set_num_threads(previous)


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
