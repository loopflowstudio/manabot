"""The fixed pilot preserves the evidenced learner and evaluation population."""

from decimal import Decimal
import json
import random

from etude.experiment import Checkpoint, Checkpoints, ExperimentState, TrainingRun
from etude.jobs import evaluation_spec
from etude.run.records import ArtifactRef

from experiments.tla_pilot import ROOT, eval, frozen
from manabot.training.regime import DampedPolicyLearning, TrainSelfPlay


def _checkpoint(update: int) -> Checkpoint:
    return Checkpoint(
        run="fixture-training",
        experiment="tla_pilot",
        ordinal=update + 1,
        artifact=ArtifactRef(
            uri=f"file:///fixture/u{update}.pt", sha256="a" * 64, size_bytes=1
        ),
        clocks={"updates": update},
        line="fixture-training",
    )


def test_pilot_pins_independent_population_and_elemental_learning_rule() -> None:
    regime, protocol = frozen()
    assert regime.deck_source is not None
    assert len(regime.deck_source.decks) == 134
    assert len(set(regime.deck_source.entry_ids)) == 134
    assert regime.deck_source.sampling == "uniform-independent-v1"
    regime.deck_source.check()
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    reference = json.loads(
        (ROOT / "experiments/regimes/elemental-long-183-v1.json").read_bytes()
    )
    assert stage.learning == DampedPolicyLearning.model_validate(
        reference["stages"][0]["learning"]
    )
    assert (stage.streams, stage.transitions, stage.learning_seats) == (256, 64, "both")
    assert stage.execution.device == "cuda" and stage.behavior == "current-self"
    assert regime.observation.policy_input_version == 5
    assert protocol.allocation_usd == Decimal(1) < protocol.max_paid_usd == Decimal(10)
    rng = random.Random(protocol.deck_seed)
    assert [(p.first, p.second) for p in protocol.pairs] == [
        (rng.randrange(134), rng.randrange(134)) for _ in range(8)
    ]
    assert len({p.deal for p in protocol.pairs}) == 8
    assert protocol.training_seed not in {p.deal for p in protocol.pairs}


def test_pilot_evaluates_initial_milestones_and_endpoint_against_fixed_anchors() -> (
    None
):
    checkpoints = Checkpoints(_checkpoint(n) for n in (0, 25, 32))
    state = ExperimentState(
        experiment="tla_pilot",
        runs=(
            TrainingRun(
                name="fixture-training",
                experiment="tla_pilot",
                state="finished",
                checkpoints=checkpoints,
            ),
        ),
    )
    jobs = eval(state)
    assert len(jobs) == 1
    spec = evaluation_spec(jobs[0])
    matchups = spec.tournament.matchups
    assert len(matchups) == 5  # 0 vs heuristic; 25 and actual endpoint vs both.
    assert {m.name for m in matchups} == {
        f"fixture-training u{update} vs {anchor}"
        for update, anchors in (
            (0, ("heuristic",)),
            (25, ("heuristic", "initial")),
            (32, ("heuristic", "initial")),
        )
        for anchor in anchors
    }
    for matchup in matchups:
        assert matchup.games == 4 and len(matchup.conditions) == 8
        if "initial" in matchup.tags:
            assert matchup.players[1].checkpoint.key.name == "u0.pt"
    assert eval(ExperimentState(experiment="tla_pilot")) == []
