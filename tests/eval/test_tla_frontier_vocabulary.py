"""The c5fba4 fact-embedding check fixes its catalog-against-hashed criterion."""

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from etude.eval.match import Tables, evaluate
from etude.eval.records import Match
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    observe,
)
from etude.jobs import evaluation_spec
from etude.launch import client
from etude.launch.records import Status
from etude.run.records import ArtifactRef
import pytest

from experiments.tla_frontier import SMALL, frozen
from experiments.tla_frontier_analysis import Bounded, learned
from experiments.tla_frontier_vocabulary import (
    EXPERIMENT,
    FIRST_UPDATES,
    LATE_UPDATES,
    UPDATES,
    VALUE_LOSS_RATIO,
    ArmReading,
    Vocabulary,
    check_schedule,
    eval,
    read_check,
    runs,
    verdict,
    vocabulary_regime,
)
from manabot.eval.results import RateInterval
from manabot.infra.config import RuntimeConfig
from manabot.infra.runtime import RunContext, regime_tag


@pytest.fixture(autouse=True)
def launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every run the store has not ended is live at the launcher."""
    monkeypatch.setattr(
        client,
        "status",
        lambda run: Status(
            run=run,
            phase="running",
            reason=None,
            outcome=None,
            cause=None,
            until=None,
            clocks={},
            usd=Decimal(0),
            attempts=1,
            rental_ended=None,
        ),
    )


def _interval(mean: float, half: float = 0.05) -> RateInterval:
    return RateInterval(mean=mean, lower=mean - half, upper=mean + half)


def _bounded(mean: float, half: float = 0.05) -> Bounded:
    interval = _interval(mean, half)
    return Bounded(score=interval, worst=interval, best=interval)


def _arm(
    vocabulary: Vocabulary, initial: float = 0.75, late_loss: float = 0.2
) -> ArmReading:
    against = _bounded(initial)
    return ArmReading(
        vocabulary=vocabulary,
        run=vocabulary,
        state="finished",
        last_update=UPDATES,
        value_loss=[(update, late_loss) for update in LATE_UPDATES],
        late_value_loss=late_loss,
        against_initial=against,
        learned=learned(against),
    )


def test_catalog_passes_when_no_rule_fails() -> None:
    assert verdict(_arm("catalog"), _arm("hashed"), _bounded(0.5, 0.1)) == ([], True)


def test_catalog_fails_by_losing_the_head_to_head_clearly() -> None:
    failures, passed = verdict(_arm("catalog"), _arm("hashed"), _bounded(0.35))
    assert passed is False and failures[0].startswith("head-to-head")
    # An interval reaching 0.5 is no failure: "at least as well" is not "better".
    assert verdict(_arm("catalog"), _arm("hashed"), _bounded(0.45, 0.06))[1]


def test_catalog_fails_when_only_hashed_learned() -> None:
    failures, passed = verdict(
        _arm("catalog", initial=0.45), _arm("hashed"), _bounded(0.5, 0.1)
    )
    assert passed is False and failures == [
        "learning: hashed learned and catalog did not"
    ]
    # Neither learning is no evidence against catalog.
    assert verdict(
        _arm("catalog", initial=0.45), _arm("hashed", initial=0.45), _bounded(0.5)
    )[1]


def test_catalog_fails_on_a_worse_late_value_loss() -> None:
    hashed = _arm("hashed", late_loss=0.2)
    within = _arm("catalog", late_loss=0.2 * VALUE_LOSS_RATIO)
    assert verdict(within, hashed, _bounded(0.5))[1]
    failures, passed = verdict(_arm("catalog", late_loss=0.23), hashed, _bounded(0.5))
    assert passed is False and failures[0].startswith("learning curve")


def test_a_missing_input_leaves_the_check_unread() -> None:
    short = _arm("catalog").model_copy(update={"late_value_loss": None})
    assert verdict(short, _arm("hashed"), _bounded(0.5)) == ([], None)
    assert verdict(_arm("catalog"), _arm("hashed"), None) == ([], None)


def _regime(vocabulary: Vocabulary, update: int) -> str:
    """The segment's regime id: the first holds updates through FIRST_UPDATES."""
    base = f"tla-frontier-c5fba4-vocabulary-{vocabulary}"
    return base if update <= FIRST_UPDATES else f"{base}-to{UPDATES}"


def _checkpoint(vocabulary: Vocabulary, update: int, segment: str) -> Checkpoint:
    return Checkpoint(
        run=segment,
        experiment=EXPERIMENT,
        ordinal=update + 1,
        artifact=ArtifactRef(
            uri=f"file:///fixture/{vocabulary}/updates-{update}.pt",
            sha256="a" * 64,
            size_bytes=1,
        ),
        clocks={"updates": update},
        line=vocabulary,
        tags=(regime_tag(_regime(vocabulary, min(update, FIRST_UPDATES + 1))),),
    )


def _state(reached: dict[Vocabulary, int], *, ended: bool = True) -> ExperimentState:
    """Each arm's segments through `reached`; the last one ended when `ended`."""
    runs: list[TrainingRun] = []
    for vocabulary, end in reached.items():
        segments = [("first", (0, 25, 50, FIRST_UPDATES))]
        if end > FIRST_UPDATES:
            segments.append(("extension", (100, 125, UPDATES)))
        for index, (segment, updates) in enumerate(segments):
            name = f"{vocabulary}-{segment}"
            last = index == len(segments) - 1
            stored = [
                _checkpoint(vocabulary, update, name)
                for update in updates
                if update <= end
            ]
            runs.append(
                TrainingRun(
                    name=name,
                    experiment=EXPERIMENT,
                    state="finished" if ended or not last else "running",
                    checkpoints=Checkpoints(
                        # A first segment's own tag even at an extension's update.
                        checkpoint.model_copy(
                            update={
                                "tags": (
                                    regime_tag(
                                        _regime(vocabulary, 0)
                                        if segment == "first"
                                        else _regime(vocabulary, UPDATES)
                                    ),
                                )
                            }
                        )
                        for checkpoint in stored
                    ),
                )
            )
    return ExperimentState(experiment=EXPERIMENT, runs=tuple(runs))


def test_the_check_plays_three_comparisons_once_both_arms_end_at_full_length() -> None:
    (job,) = eval(_state({"catalog": UPDATES, "hashed": UPDATES}))
    benchmark = evaluation_spec(job).tournament
    assert benchmark.environment.name == "tla-frontier-c5fba4-vocabulary-v1"
    assert [m.name for m in benchmark.matchups][-3:] == [
        "catalog u150 vs catalog u0",
        "hashed u150 vs hashed u0",
        "catalog u150 vs hashed u150",
    ]
    assert all(len(m.matches()) == len(frozen()[1].pairs) for m in benchmark.matchups)


def _matchups(state: ExperimentState) -> list[str]:
    return [
        m.name for job in eval(state) for m in evaluation_spec(job).tournament.matchups
    ]


def test_the_check_waits_for_both_arms() -> None:
    early = [
        f"{a} u{n} vs {b}"
        for n in (25, 50, 100)
        for a, b in (
            ("catalog", "catalog u0"),
            ("hashed", "hashed u0"),
            ("catalog", f"hashed u{n}"),
        )
    ]
    assert (
        _matchups(_state({"catalog": UPDATES, "hashed": UPDATES}, ended=False)) == early
    )
    # An arm that ended short of full length leaves the endpoint unplayed.
    assert _matchups(_state({"catalog": UPDATES, "hashed": 125})) == early
    # A first segment that stopped early still shows its interior exports.
    assert _matchups(_state({"catalog": 64, "hashed": FIRST_UPDATES})) == early[:6]
    schedule = check_schedule(_state({"catalog": UPDATES}), frozen()[1])
    assert all(planned.matchup is None for planned in schedule)


def test_the_check_rejects_a_foreign_experiment() -> None:
    with pytest.raises(ValueError, match="its own runs"):
        eval(ExperimentState(experiment="other", runs=()))


def test_arms_differ_only_in_record_vocabulary() -> None:
    small = frozen()[0][SMALL]
    catalog, hashed = (vocabulary_regime(small, v) for v in ("catalog", "hashed"))
    assert catalog.agent.record_vocabulary == "catalog"
    assert hashed.agent.record_vocabulary == "hashed"
    exclude = {"id": True, "agent": {"record_vocabulary"}}
    assert catalog.model_dump(exclude=exclude) == hashed.model_dump(exclude=exclude)
    stage = catalog.stages[0]
    assert stage.updates == FIRST_UPDATES
    extension = vocabulary_regime(small, "catalog", extension=True)
    assert extension.stages[0].updates == UPDATES
    assert extension.model_dump(
        exclude={
            "id": True,
            "wall_seconds": True,
            "stages": {0: {"updates", "execution"}},
        }
    ) == catalog.model_dump(
        exclude={
            "id": True,
            "wall_seconds": True,
            "stages": {0: {"updates", "execution"}},
        }
    )
    assert stage.execution.workers == small.stages[0].execution.workers
    launched = runs("HEAD")
    assert len(launched) == 2 and {r.seed for r in launched} == {
        frozen()[1].training_seed
    }


def _tracked(source: Path, vocabulary: Vocabulary, loss: float) -> None:
    """A real worker's records for one arm: two segments, value loss to UPDATES."""
    for segment, updates in (
        ("first", range(0, FIRST_UPDATES + 1)),
        ("extension", range(FIRST_UPDATES, UPDATES + 1)),
    ):
        name = uuid4().hex[:12]
        regime = _regime(vocabulary, 0 if segment == "first" else UPDATES)
        with pytest.MonkeyPatch.context() as environment:
            environment.setenv("ETUDE_RUN", name)
            environment.setenv("ETUDE_TRACK_DIR", str(source))
            environment.setenv("ETUDE_JOB_NAME", regime)
            context = RunContext(
                RuntimeConfig(name=name, runs_dir=source / "local", seed=21501),
                experiment=EXPERIMENT,
                regime=regime,
            )
            for update in updates:
                context.log_update(
                    {"value_loss": loss} if update else {},
                    updates=update,
                    transitions=update * 100,
                )
                if update % 25 == 0 and (segment == "first" or update > FIRST_UPDATES):
                    path = source / f"{name}-u{update}.pt"
                    path.write_text(f"synthetic export {name} {update}")
                    context.checkpoint(path, transitions=update * 100)
            context.close("finished")


def _first_wins(legs: int) -> Callable[[Match], Tables]:
    def play(match: Match) -> Tables:
        return {
            "games": [
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": (leg + seat) % 2,
                    "score": float((leg < legs) != bool(seat)),
                    "valid": True,
                    "attempted": True,
                    "terminated": True,
                    "truncated": False,
                    "failure": None,
                    "seconds": 1.0,
                }
                for leg in range(4)
                for seat in (0, 1)
            ]
        }

    return play


@pytest.mark.parametrize(
    ("catalog_loss", "passed"), [(0.2, True), (0.3, False)], ids=["equal", "worse"]
)
def test_the_reader_applies_the_criterion_to_a_real_store(
    tmp_path: Path, catalog_loss: float, passed: bool
) -> None:
    _tracked(tmp_path, "catalog", catalog_loss)
    _tracked(tmp_path, "hashed", 0.2)
    state = observe(tmp_path, EXPERIMENT)
    benchmark = evaluation_spec(eval(state)[0]).tournament
    evaluate(
        experiment=EXPERIMENT,
        environment=benchmark.environment,
        matches=[m for matchup in benchmark.matchups for m in matchup.matches()],
        play=_first_wins(3),
        run="fixture-eval",
        directory=tmp_path,
    )
    check = read_check(state, tmp_path)
    catalog, hashed = check.arms
    assert catalog.learned is True and hashed.learned is True
    assert len(catalog.value_loss) == UPDATES
    assert check.head_to_head is not None and check.head_to_head.score is not None
    assert check.head_to_head.score.mean == 0.75
    assert check.passed is passed
    assert check.loss_ratio == pytest.approx(catalog_loss / 0.2)
