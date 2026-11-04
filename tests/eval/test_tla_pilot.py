"""The fixed pilot preserves the evidenced learner and evaluation population."""

from collections.abc import Iterator
from decimal import Decimal
import json
from pathlib import Path
import random
import subprocess
import sys
from textwrap import dedent
from typing import Literal

import duckdb
from etude.eval.block import write
from etude.eval.records import MatchRecord
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.run.declaration import declare
from etude.run.records import ArtifactRef, Clock
from etude.train.run import track
import numpy as np
import pytest

from experiments import tla_pilot_v7
from experiments.tla_pilot_analysis import analyze
from experiments.tla_pilot_v7 import (
    EXPERIMENT,
    PROTOCOL,
    ROOT,
    Protocol,
    RuntimeWorld,
    eval,
    frozen,
)
from manabot.env import observation as observation_module
from manabot.env.failures import read_failure
from manabot.env.observation import ObservationEncoder
from manabot.eval.results import RateInterval
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.training.regime import DampedPolicyLearning, TrainingRegime, TrainSelfPlay
import managym


@pytest.fixture
def historical_protocol(monkeypatch: pytest.MonkeyPatch) -> Protocol:
    """Select v5's immutable contract to exercise current-runtime rejection."""
    path = ROOT / "experiments/protocols/tla-trophy-213-v5.json"
    monkeypatch.setattr(tla_pilot_v7, "PROTOCOL", path)
    monkeypatch.setattr(
        tla_pilot_v7, "REGIME", ROOT / "experiments/regimes/tla-trophy-213-v5.json"
    )
    return Protocol.model_validate_json(path.read_bytes())


@pytest.fixture
def pilot_world(monkeypatch: pytest.MonkeyPatch) -> RuntimeWorld:
    """Declare compatible identity for fixtures that never play or train.

    Historical scheduling and controlled analysis outcomes must not depend on
    the evolving native catalog. Runtime admission is tested without this seam.
    """
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes())
    world = protocol.runtime_world
    monkeypatch.setattr(managym, "world_identity", world.model_dump)
    # Synthetic scheduling also needs the historical schema. Never use these
    # widths for encoding or playing; runtime shape rejection is tested below.
    monkeypatch.setattr(
        observation_module, "PLAYER_DIM", protocol.observation_shapes["agent_player"][1]
    )
    monkeypatch.setattr(
        observation_module, "CARD_DIM", protocol.observation_shapes["agent_cards"][1]
    )
    return world


@pytest.mark.parametrize("entry", ["frozen", "runs", "eval"])
def test_pilot_rejects_historical_runtime_world(
    entry: str, historical_protocol: Protocol
) -> None:
    assert managym.world_identity() != historical_protocol.runtime_world.model_dump()
    with pytest.raises(ValueError, match="runtime world differs"):
        if entry == "frozen":
            frozen()
        elif entry == "runs":
            tla_pilot_v7.runs("HEAD")
        else:
            eval(_state())


def test_pilot_rejects_historical_tensor_shapes(
    historical_protocol: Protocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        managym, "world_identity", historical_protocol.runtime_world.model_dump
    )
    with pytest.raises(ValueError, match="tensor capacities differ"):
        frozen()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_actions", 64),
        ("max_facts", 512),
        ("policy_input_version", 6),
        ("max_cards_per_player", 60),
    ],
)
def test_pilot_rejects_changed_input_contract(
    pilot_world: RuntimeWorld,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    # Temporary copies keep the frozen bytes intact and pass the file-integrity
    # guard so this check reaches observation admission under compatible identity.
    regime = json.loads(tla_pilot_v7.REGIME.read_bytes())
    regime["observation"][field] = value
    regime_path = tmp_path / "regime.json"
    regime_path.write_text(json.dumps(regime))
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes()).model_copy(
        update={"regime_sha256": file_sha256(regime_path)}
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(protocol.model_dump_json())
    monkeypatch.setattr(tla_pilot_v7, "REGIME", regime_path)
    monkeypatch.setattr(tla_pilot_v7, "PROTOCOL", protocol_path)
    with pytest.raises(ValueError, match="tensor capacities differ"):
        frozen()


def _checkpoint(update: int) -> Checkpoint:
    return Checkpoint(
        run="fixture-training",
        experiment=EXPERIMENT,
        ordinal=update + 1,
        artifact=ArtifactRef(
            uri=f"file:///fixture/u{update}.pt", sha256="a" * 64, size_bytes=1
        ),
        clocks={"updates": update},
        line="fixture-training",
    )


@pytest.mark.usefixtures("pilot_world")
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
    assert regime.observation.policy_input_version == 7
    assert regime.observation.max_facts == 128
    assert regime.observation.max_actions == 512
    assert regime.observation.max_cards_per_player == 128
    assert protocol.observation_shapes["facts"] == (128, 12)
    assert protocol.observation_shapes["effects"] == (128, 18)
    assert protocol.observation_shapes["effect_details"] == (512, 3)
    assert protocol.observation_shapes["effect_members"] == (2048, 2)
    assert ObservationConfig().max_facts == 512
    assert ObservationConfig().max_actions == 512
    assert protocol.allocation_usd == Decimal(1) < protocol.max_paid_usd == Decimal(10)
    rng = random.Random(protocol.deck_seed)
    assert [(p.first, p.second) for p in protocol.pairs] == [
        (rng.randrange(134), rng.randrange(134)) for _ in range(8)
    ]
    assert len({p.deal for p in protocol.pairs}) == 8
    assert protocol.training_seed not in {p.deal for p in protocol.pairs}


@pytest.mark.usefixtures("pilot_world")
def test_pilot_evaluates_initial_milestones_and_endpoint_against_fixed_anchors() -> (
    None
):
    state = _state((0, 25, 32))
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
    assert eval(ExperimentState(experiment=EXPERIMENT)) == []


@pytest.mark.usefixtures("pilot_world")
def test_pilot_cli_discovery_keeps_evaluation_callback_importable() -> None:
    experiment = read_experiment(ROOT / "experiments/tla_pilot_v7.py")
    job = experiment.eval(_state((0,)))[0]
    callback = evaluation_spec(job).play
    assert callback.module == "experiments.tla_pilot_v7"
    assert callable(callback.load(job.repo))


# Controlled outcomes exercise the analysis without running games or training.
# This schema is Etude's public statsdb projection, including both participants.
@pytest.fixture
def pilot_db(pilot_world: RuntimeWorld) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect() as db:
        db.execute("CREATE TABLE runs (run VARCHAR, environment VARCHAR)")
        db.execute(
            "CREATE TABLE matches (run VARCHAR, match VARCHAR, state VARCHAR, error VARCHAR)"
        )
        db.execute(
            "CREATE TABLE seats (run VARCHAR, match VARCHAR, seat BIGINT, source VARCHAR)"
        )
        db.execute(
            "CREATE TABLE games (run VARCHAR, match VARCHAR, seat BIGINT, game BIGINT, attempted BOOLEAN, valid BOOLEAN, score DOUBLE, terminated BOOLEAN, truncated BOOLEAN, failure VARCHAR)"
        )
        yield db


def _state(
    updates: tuple[int, ...] = (0, 25, 50, 75, 100),
    *,
    status: Literal["running", "finished", "failed"] = "finished",
) -> ExperimentState:
    return ExperimentState(
        experiment=EXPERIMENT,
        runs=(
            TrainingRun(
                name="fixture-training",
                experiment=EXPERIMENT,
                state=status,
                checkpoints=Checkpoints(_checkpoint(update) for update in updates),
            ),
        ),
    )


def _outcomes(
    db: duckdb.DuckDBPyConnection, state: ExperimentState
) -> dict[str, list[str]]:
    tournament = evaluation_spec(eval(state)[0]).tournament
    db.execute(
        "INSERT INTO runs VALUES ('fixture-eval', ?)", [tournament.environment.identity]
    )
    keys: dict[str, list[str]] = {}
    for matchup in tournament.matchups:
        keys[matchup.name] = []
        for index, match in enumerate(matchup.matches()):
            keys[matchup.name].append(match.key)
            db.execute(
                "INSERT INTO matches VALUES ('fixture-eval', ?, 'done', NULL)",
                [match.key],
            )
            # Variable initial difficulty, exactly +.25 on every pair after
            # training. Against initialization every pair scores .75.
            wins = (
                3
                if "initial" in matchup.tags
                else index % 4 + int(" u0 " not in matchup.name)
            )
            for seat in (0, 1):
                db.execute(
                    "INSERT INTO seats VALUES ('fixture-eval', ?, ?, ?)",
                    [match.key, seat, "a" * 64],
                )
                db.executemany(
                    "INSERT INTO games VALUES ('fixture-eval', ?, ?, ?, true, true, ?, true, false, NULL)",
                    [
                        [match.key, seat, leg, float((leg < wins) != bool(seat))]
                        for leg in range(4)
                    ],
                )
    return keys


def test_pilot_analysis_pairs_gain_and_checks_joint_criterion(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    report = analyze(state, pilot_db)[0]
    assert report.complete and report.success is True
    assert report.bootstrap_resamples == 2000 and report.bootstrap_seed == 21392001
    assert (
        report.counts.expected == report.counts.attempted == report.counts.valid == 288
    )
    assert report.counts.missing == report.counts.failed == 0
    assert report.counts.expected_matches == report.counts.attempted_matches == 72
    final = next(
        c for c in report.comparisons if c.endpoint and c.anchor == "heuristic"
    )
    # Independent resampling of before/after, or of individual legs, widens
    # this interval. Shared four-leg pair blocks give exactly .25 throughout.
    assert final.heuristic_gain == RateInterval(mean=0.25, lower=0.25, upper=0.25)
    assert [p.pair for p in final.pairs] == list(frozen()[1].pairs)
    assert all(p.sources == ("a" * 64, "a" * 64) for p in final.pairs)
    assert final.counts.wins == 20 and final.counts.losses == 12
    assert all(
        c.heuristic_gain is not None
        for c in report.comparisons
        if c.anchor == "heuristic"
    )


@pytest.mark.parametrize(
    "gap",
    [
        "leg",
        "pair",
        "checkpoint",
        "initial",
        "failed_match",
        "invalid",
        "unattempted",
        "truncated",
        "source",
        "environment",
        "pair_identity",
    ],
)
def test_pilot_analysis_incomplete_cohort_cannot_claim_learning(
    pilot_db: duckdb.DuckDBPyConnection, gap: str
) -> None:
    state = _state()
    keys = _outcomes(pilot_db, state)
    key = keys["fixture-training u50 vs heuristic"][0]
    if gap == "leg":
        pilot_db.execute("DELETE FROM games WHERE match = ? AND game = 0", [key])
    elif gap == "pair":
        pilot_db.execute("DELETE FROM matches WHERE match = ?", [key])
    elif gap in ("checkpoint", "initial"):
        missing = 50 if gap == "checkpoint" else 0
        state = _state(tuple(n for n in (0, 25, 50, 75, 100) if n != missing))
    elif gap == "failed_match":
        pilot_db.execute(
            "UPDATE matches SET state = 'failed', error = 'worker died' WHERE match = ?",
            [key],
        )
        pilot_db.execute("DELETE FROM games WHERE match = ?", [key])
    elif gap in ("invalid", "unattempted", "truncated"):
        pilot_db.execute(
            "UPDATE games SET valid = false, score = NULL, attempted = ?, truncated = ?, failure = 'fixture' WHERE match = ? AND game = 0",
            [gap != "unattempted", gap == "truncated", key],
        )
    elif gap == "source":
        pilot_db.execute("UPDATE seats SET source = ? WHERE match = ?", ["b" * 64, key])
    elif gap == "environment":
        pilot_db.execute("UPDATE runs SET environment = 'another-world'")
    else:
        # Same deal, different deck entry: a different pair cannot fill the hole.
        matchup = evaluation_spec(eval(state)[0]).tournament.matchups[3]
        original = matchup.matches()[0]
        replacement = original.model_copy(
            update={"conditions": {**original.conditions, "first": 0}}
        )
        assert original.key == key
        pilot_db.execute(
            "UPDATE matches SET match = ? WHERE match = ?", [replacement.key, key]
        )
    report = analyze(state, pilot_db)[0]
    assert not report.complete and report.success is None and report.issues
    assert report.counts.expected == 288  # never select the surviving games
    assert (
        report.counts.expected
        == report.counts.valid + report.counts.failed + report.counts.missing
    )
    if gap in ("leg", "unattempted"):
        assert report.counts.missing == 1 and report.counts.attempted == 287
    if gap in ("invalid", "truncated"):
        assert report.counts.failed == 1 and report.counts.attempted == 288
        assert report.counts.truncated == int(gap == "truncated")
    if gap in ("pair", "pair_identity", "failed_match"):
        assert report.counts.missing == 4
        assert report.counts.failed_matches == int(gap == "failed_match")
    if gap == "checkpoint":
        assert report.counts.missing == 64
    if gap == "initial":
        assert report.counts.missing == 160  # baseline plus all fixed-initial opponents
        assert all(c.heuristic_gain is None for c in report.comparisons)


@pytest.mark.parametrize(
    "mode", ["no_gain", "uncertain_gain", "weak_initial_opponent_score"]
)
def test_pilot_analysis_complete_negative_is_not_success(
    pilot_db: duckdb.DuckDBPyConnection, mode: str
) -> None:
    state = _state()
    keys = _outcomes(pilot_db, state)
    anchor = "initial" if mode == "weak_initial_opponent_score" else "heuristic"
    for index, key in enumerate(keys[f"fixture-training u100 vs {anchor}"]):
        wins = 2 if anchor == "initial" else index % 4
        if mode == "uncertain_gain":
            wins = 4 if index < 4 else 0  # +.125 mean; uncertainty includes zero
        pilot_db.execute(
            "UPDATE games SET score = CAST(game < ? AS DOUBLE) WHERE match = ? AND seat = 0",
            [wins, key],
        )
    report = analyze(state, pilot_db)[0]
    assert report.complete and report.success is False


@pytest.mark.parametrize("status", ["running", "failed"])
def test_pilot_analysis_retains_unknown_endpoint(
    pilot_db: duckdb.DuckDBPyConnection, status: Literal["running", "failed"]
) -> None:
    state = _state((0, 25, 50, 75), status=status)
    _outcomes(pilot_db, state)
    report = analyze(state, pilot_db)[0]
    assert report.counts.expected == 288 and report.counts.missing == 64
    assert report.success is None
    endpoint = [c for c in report.comparisons if c.endpoint]
    assert len(endpoint) == 2 and all(c.update is None for c in endpoint)


def test_pilot_analysis_early_endpoint_keeps_absent_milestones(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state((0, 25, 32))
    _outcomes(pilot_db, state)
    report = analyze(state, pilot_db)[0]
    assert report.counts.expected == 288 and report.counts.missing == 128
    assert report.success is None
    assert {c.update for c in report.comparisons if c.endpoint} == {32}


def test_pilot_analysis_rejects_duplicate_legs(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    pilot_db.execute("INSERT INTO games SELECT * FROM games LIMIT 1")
    with pytest.raises(ValueError, match="duplicate legs"):
        analyze(state, pilot_db)


def test_pilot_analysis_counts_failures_without_a_score_column(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    pilot_db.execute("UPDATE games SET valid = false, failure = 'fixture failure'")
    pilot_db.execute("ALTER TABLE games DROP COLUMN score")
    report = analyze(state, pilot_db)[0]
    assert report.counts.failed == report.counts.expected == 288
    assert report.counts.valid == report.counts.missing == 0
    assert report.success is None


@pytest.mark.parametrize("with_failure_column", [True, False])
def test_pilot_analysis_cli_reads_sealed_etude_results(
    pilot_db: duckdb.DuckDBPyConnection,
    pilot_world: RuntimeWorld,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_failure_column: bool,
) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    for key in ("ETUDE_EXPERIMENT", "ETUDE_RUN", "ETUDE_TRACK_DIR", "ETUDE_CONTINUES"):
        monkeypatch.delenv(key, raising=False)
    state = _state()
    store = tmp_path / "store"
    with track(
        experiment=EXPERIMENT,
        run="fixture-training",
        directory=store,
        clocks=(Clock(name="updates", kind="step", meaning="Learner updates"),),
        strict=True,
    ) as training:
        for checkpoint in state.runs[0].checkpoints:
            training.checkpoint(checkpoint.artifact, checkpoint.clocks)
    _outcomes(pilot_db, state)
    if not with_failure_column:
        pilot_db.execute("ALTER TABLE games DROP COLUMN failure")
    tournament = evaluation_spec(eval(state)[0]).tournament
    declared = declare(
        experiment=EXPERIMENT,
        run="fixture-eval",
        kind="evaluation",
        clocks=(),
        directory=store,
    )
    (declared.root / "environment.json").write_text(
        tournament.environment.model_dump_json()
    )
    tables = tmp_path / "games.parquet"
    pilot_db.execute("COPY games TO ? (FORMAT PARQUET)", [str(tables)])
    # Sealing fixture outcomes exercises the real statsdb projection and CLI,
    # with no worker, model inference, game or provider allocation.
    columns = dict(
        pilot_db.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'games' AND column_name NOT IN ('run', 'match') "
            "ORDER BY ordinal_position"
        ).fetchall()
    )
    records = [
        MatchRecord(
            match=match,
            state="done",
            wall_ms=1,
            columns={"games": columns},
        )
        for matchup in tournament.matchups
        for match in matchup.matches()
    ]
    write(declared.root, "f1", records, {"games": [tables]}, {}, pilot_db)
    output = tmp_path / "analysis.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # A fresh interpreter does not inherit pytest's identity fixture.
            # Run the real CLI with the same explicit compatible identity; only
            # synthetic sealed outcomes are read, with no game or training run.
            dedent("""\
                import json
                import runpy
                import sys
                from unittest.mock import patch

                from experiments.tla_pilot_v7 import PROTOCOL, Protocol

                world = json.loads(sys.argv.pop(1))
                shapes = Protocol.model_validate_json(PROTOCOL.read_bytes()).observation_shapes
                with (
                    patch("managym.world_identity", return_value=world),
                    patch("manabot.env.observation.PLAYER_DIM", shapes["agent_player"][1]),
                    patch("manabot.env.observation.CARD_DIM", shapes["agent_cards"][1]),
                ):
                    runpy.run_module(
                        "experiments.tla_pilot_analysis", run_name="__main__", alter_sys=True
                    )
                """),
            pilot_world.model_dump_json(),
            "--source",
            str(store),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())["runs"][0]
    assert report["complete"] and report["success"]
    assert report["counts"]["valid"] == report["counts"]["expected"] == 288


def test_pilot_analysis_counts_draws_without_fabricating_wins(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    pilot_db.execute("UPDATE games SET score = 0.5")
    report = analyze(state, pilot_db)[0]
    assert report.complete and report.success is False
    assert report.counts.draws == 288
    assert report.counts.wins == report.counts.losses == 0
    assert all(
        c.score == RateInterval(mean=0.5, lower=0.5, upper=0.5)
        for c in report.comparisons
    )


@pytest.mark.usefixtures("pilot_world")
def test_prospective_pilot_preserves_population_and_rejects_old_initialization() -> (
    None
):
    regime, protocol = frozen()
    predecessor = json.loads(
        (ROOT / "experiments/regimes/tla-trophy-213-v1.json").read_bytes()
    )
    assert regime.deck_source is not None
    assert regime.deck_source.model_dump(mode="json") == predecessor["deck_source"]
    old_protocol = json.loads(
        (ROOT / "experiments/protocols/tla-trophy-213-v1.json").read_bytes()
    )
    assert [pair.model_dump() for pair in protocol.pairs] == old_protocol["pairs"]
    assert protocol.corpus_sha256 == old_protocol["corpus_sha256"]
    assert protocol.milestones == tuple(old_protocol["milestones"])
    previous = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/tla-trophy-213-v5.json").read_bytes()
    )
    assert regime == previous.model_copy(
        update={
            "id": regime.id,
            "world": regime.world,
            "observation": previous.observation.model_copy(update={"max_actions": 512}),
        }
    )
    assert protocol.predecessor_protocol_sha256 == file_sha256(
        ROOT / "experiments/protocols/tla-trophy-213-v6.json"
    )
    for experiment in (
        "tla_pilot",
        "tla_pilot_v2",
        "tla_pilot_v3",
        "tla_pilot_v4",
        "tla_pilot_v5",
        "tla_pilot_v6",
    ):
        with pytest.raises(ValueError, match="fresh v7"):
            eval(ExperimentState(experiment=experiment))


@pytest.mark.usefixtures("pilot_world")
@pytest.mark.parametrize("binding", ("regime", "tensor_shapes"))
def test_pilot_rejects_changed_observation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    if binding == "regime":
        config = json.loads(tla_pilot_v7.REGIME.read_bytes())
        config["observation"]["max_cards_per_player"] += 1
        target, error = "REGIME", "regime differs"
    else:
        config = json.loads(tla_pilot_v7.PROTOCOL.read_bytes())
        config["observation_shapes"]["effect_members"][0] += 1
        target, error = "PROTOCOL", "tensor capacities differ"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(tla_pilot_v7, target, path)
    with pytest.raises(ValueError, match=error):
        frozen()


@pytest.mark.parametrize("card_capacity", (61, 128))
def test_pilot_retains_generated_cards_and_sideboard_in_all_encoders(
    card_capacity: int,
) -> None:
    evidence = read_failure(ROOT / "tests/fixtures/tla-pilot-card-capacity.jsonl")
    (game,) = evidence.games
    # Replay uses the real current world and the explicit prospective contract;
    # the retained pilot's card-allocation boundary is the regression under test.
    regime = TrainingRegime.model_validate_json(tla_pilot_v7.REGIME.read_bytes())
    # Both the exact occupied boundary and the pilot allocation retain every
    # generated token and sideboard card through scalar/vector/Python paths.
    encoder = ObservationEncoder(
        regime.observation.model_copy(update={"max_cards_per_player": card_capacity})
    )
    players = [player.native() for player in game.players]
    scalar = managym.Env(seed=game.seed, skip_trivial=evidence.skip_trivial)
    scalar.reset(players)
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        seed_stride=evidence.seed_stride,
        skip_trivial=evidence.skip_trivial,
    )
    buffers = encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector.set_buffers(buffers)
    vector.reset_all_into_buffers(players)
    for index, action in enumerate(game.actions):
        assert isinstance(action, int)
        if index == len(game.actions) - 1:
            assert scalar.semantic_decision_frame_json() == game.frame
            assert vector.semantic_decision_frames_json()[0] == game.frame
        observation, *_ = scalar.step(action)
        vector.step_into_buffers([action])
    assert len(observation.agent_cards) == 45
    assert len(observation.agent_sideboard) == 16
    reference = encoder.encode(observation)
    native = encoder.allocate()
    scalar.encode_observation_into(observation, native)
    for key, value in reference.items():
        np.testing.assert_array_equal(value, native[key], err_msg=key)
        np.testing.assert_array_equal(value, buffers[key][0], err_msg=key)
    # One fewer slot rejects rather than silently dropping an object.
    old = ObservationEncoder(
        regime.observation.model_copy(update={"max_cards_per_player": 60})
    )
    with pytest.raises(ValueError, match="agent_cards plus sideboard"):
        old.encode(observation)
    with pytest.raises(ValueError, match="agent_cards plus sideboard 61 > 60"):
        scalar.encode_observation_into(observation, old.allocate())


def test_prospective_pilot_admits_current_runtime_without_fixtures() -> None:
    _, protocol = frozen()
    assert protocol.runtime_world.model_dump() == managym.world_identity()
