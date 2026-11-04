"""Sealed synthetic results exercise reconciliation; no native games or training."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
import json
from pathlib import Path

import duckdb
from etude.eval.match import evaluate as seal_matches
from etude.eval.query import statsdb
from etude.eval.records import Code, Environment, Model
from etude.experiment import Checkpoints, ExperimentState, observe
from etude.train.read import Points, RunData, read_all
from etude.train.records import Step
from etude.train.run import track
import numpy as np
import pytest

from experiments import growing_pool, growing_pool_analysis
from experiments.growing_pool_analysis import AnalysisResult, analyze, main
from experiments.growing_pool_protocol import (
    EXPERIMENT,
    PROTOCOL,
    ROOT,
    Protocol,
    World,
)
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.eval.results import bootstrap_mean_interval
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import source_bundle_sha256
from manabot.sim.card_evidence import GameEvidence, retain_evidence
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from tests.eval.test_card_evaluation import card_fixture
from tests.eval.test_local_evaluation import _replayed_rows


@dataclass
class SealedLeague:
    protocol: Protocol
    source: Path
    state: ExperimentState
    training: list[RunData]


@pytest.fixture(scope="module")
def league(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SealedLeague]:
    root = tmp_path_factory.mktemp("sealed-league")
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes())
    # Explicit synthetic compatible world: scheduling only; never train/replay.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(World, "current", classmethod(lambda cls: protocol.world))
        patch.setenv("ETUDE_HOME", str(root / "home"))
        for name in (
            "ETUDE_RUN",
            "ETUDE_TRACK_DIR",
            "ETUDE_EXPERIMENT",
            "ETUDE_CONTINUES",
        ):
            patch.delenv(name, raising=False)
        paths = sorted((ROOT / "manabot").rglob("*.py"))
        identities = {
            "world": protocol.world.world_version,
            "engine_source_sha256": protocol.world.engine_source_sha256,
            "content_digest": protocol.world.card_digest,
            "deck_source_sha256": protocol.manifest.source().digest,
            "training_source_sha256": source_bundle_sha256(paths),
        }
        heuristic_source = canonical_sha256(
            {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
        )
        source = root / "store"
        spec = protocol.analysis
        for index, seed in enumerate(protocol.budget.seeds):
            run = track(
                experiment=EXPERIMENT,
                run=f"learner-{index}",
                seed=seed,
                clocks=[Step("updates", "completed updates")],
                directory=source,
                config={
                    "training": {
                        "regime": protocol.regime().model_dump(mode="json"),
                        "identities": identities,
                    }
                },
                strict=True,
            )
            try:
                for update in range(protocol.budget.updates + 1):
                    if update:
                        metrics = {name: 0.1 for name in spec.finite_metrics}
                        metrics.update({name: 0.0 for name in spec.zero_metrics})
                        metrics.update({name: 1.0 for name in spec.positive_metrics})
                        if update in spec.sampled_updates:
                            metrics.update(
                                {name: 0.1 for name in spec.sampled_finite_metrics}
                            )
                            metrics[spec.parameter_change] = 0.01
                        run.log(metrics, updates=update)
                    if update in (
                        0,
                        *protocol.budget.milestones,
                        protocol.budget.updates,
                    ):
                        path = root / f"seed-{seed}-u{update}.pt"
                        path.write_text(f"synthetic export {seed} {update}")
                        run.checkpoint(path, updates=update)
            finally:
                run.finish()
        state = observe(source, EXPERIMENT)
        coordinates = {
            c.artifact.sha256: (c.clocks["updates"], i)
            for i, r in enumerate(state.runs)
            for c in r.checkpoints
        }

        def registration(
            player: Model | Code, name: str, matrix: DeckMatchup, protocol: Protocol
        ) -> tuple[PlayerRegistration, Path | None]:
            common = dict(
                player_id=name,
                display_name=player.name,
                role="challenger" if name == "candidate" else "anchor",
                world=protocol.world.world_version,
                content_suite=matrix.content_suite,
                information_boundary="acting-viewer",
                observation_abi_sha256="a" * 64,
                action_abi_sha256="b" * 64,
                matchup_sha256=canonical_sha256(matrix.match().model_dump(mode="json")),
                player_seed_derivation_id="arena-pair-deal-player-v1",
                compute_class_id="policy-cpu-one-thread-one-pass",
            )
            if isinstance(player, Code):
                return PlayerRegistration(
                    **common,
                    runner_kind="code",
                    player_spec={"kind": "heuristic"},
                    source_sha256=heuristic_source,
                ), None
            return PlayerRegistration(
                **common,
                runner_kind="checkpoint",
                player_spec={
                    "kind": "checkpoint",
                    "device": "cpu",
                    "batch_size": 1,
                    "deterministic": False,
                    "allow_deck_repetition": True,
                },
                checkpoint_sha256=player.checkpoint.artifact.sha256,
                checkpoint_bytes=player.checkpoint.artifact.size_bytes,
                parameter_count=1,
                artifact_id=f"mtg245/{player.checkpoint.artifact.sha256}",
            ), Path("synthetic.pt")

        def synthetic(
            request: EvaluationRequest, *, out_dir: Path, card_evidence_dir: Path
        ) -> EvaluationResult:
            out_dir.mkdir(parents=True)
            matrix = request.authored_matchup
            assert matrix is not None
            update, seed_index = coordinates[request.candidate.checkpoint_sha256]
            rows = _replayed_rows(request)
            refs = []
            for row in rows:
                # Mean gain .0625/.125, with zero-gain blocks: lower CI is zero.
                extra_win = (
                    update == protocol.budget.updates
                    and row.deal_seed % 100 == 0
                    and row.leg < 1 + seed_index
                )
                score = float(row.leg >= 2 or extra_win)
                winner = row.leg % 2 if score else 1 - row.leg % 2
                players = (
                    ("candidate", "opponent")
                    if row.leg % 2 == 0
                    else ("opponent", "candidate")
                )
                game = card_fixture(
                    winner=winner, leg=row.leg, deal=row.deal_seed, players=players
                )
                raw = game.model_dump(mode="json", by_alias=True)
                raw["seat_decks"] = list(matrix.decks(row.leg))
                setup = matrix.match(row.leg)
                setup.hero, setup.villain = players
                raw["setup"] = setup.model_dump(mode="json")
                raw["cards"]["capture"]["content_digest"] = protocol.world.card_digest
                trace = raw["trace"]["data"]
                trace.pop("game_trace_sha256")
                trace.update(
                    seat_decks=raw["seat_decks"],
                    match_hypers=raw["setup"],
                    observation_hypers=protocol.observation.model_dump(mode="json"),
                    runtime_identities={
                        **identities,
                        "arena_source_sha256": identities["training_source_sha256"],
                        "observation_abi_sha256": "a" * 64,
                        "action_abi_sha256": "b" * 64,
                    },
                )
                trace["game_trace_sha256"] = canonical_sha256(trace)
                raw["trace_sha256"] = trace["game_trace_sha256"]
                game = GameEvidence.model_validate(raw)
                refs.append(retain_evidence(card_evidence_dir, game))
                row.score_a = score
                row.__pydantic_extra__ = {"trace_sha256": game.trace_sha256}
            (out_dir / "trace.json").write_text(
                json.dumps({"card_evidence": [ref.model_dump() for ref in refs]})
            )
            return EvaluationResult(request=request, rows=rows, status="completed")

        patch.setattr(growing_pool, "_registration", registration)
        patch.setattr(growing_pool, "evaluate", synthetic)
        patch.setenv("ETUDE_OUTPUT_DIR", str(root / "eval"))
        matches = [
            match
            for run in state.runs
            for slot in growing_pool.comparisons(run, protocol)
            if slot.matchup
            for match in slot.matchup.matches()
        ]
        seal_matches(
            experiment=EXPERIMENT,
            environment=Environment(
                name=protocol.id,
                files=protocol.digest,
                config={"status": protocol.status},
            ),
            matches=matches,
            play=growing_pool.play,
            directory=source,
            run="synthetic-evaluation",
        )
        yield SealedLeague(protocol, source, state, read_all(source))


@pytest.fixture
def db(league: SealedLeague) -> Iterator[duckdb.DuckDBPyConnection]:
    with statsdb(EXPERIMENT, league.source) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            yield connection
        finally:
            connection.execute("ROLLBACK")


def report(league: SealedLeague, db: duckdb.DuckDBPyConnection) -> AnalysisResult:
    return analyze(league.state, league.training, db, league.source, league.protocol)


def test_complete_sealed_league_uses_paired_whole_deals_per_seed(
    league: SealedLeague, db: duckdb.DuckDBPyConnection
) -> None:
    result = report(league, db)
    assert result.complete and result.modest_check is True
    assert result.counts.expected == result.counts.valid == 800
    assert result.counts.expected_matches == 200
    assert [r.seed for r in result.seeds] == list(league.protocol.budget.seeds)
    gains = []
    for run in result.seeds:
        endpoint = next(
            c
            for c in run.contrasts
            if c.panel == "anchors" and c.update == 32 and c.anchor == "heuristic"
        )
        assert endpoint.gain is not None and endpoint.gain.lower == 0
        gains.append(endpoint.gain.mean)
        blocks = np.asarray([d.score - 0.5 for d in endpoint.deals])
        indices = np.random.default_rng(result.bootstrap_seed).integers(
            0, 4, (result.bootstrap_resamples, 4)
        )
        assert endpoint.gain == bootstrap_mean_interval(blocks, indices)
    assert sorted(gains) == [0.0625, 0.125]
    assert result.training_exposure == "unavailable; acceptance unresolved"


def test_wholly_absent_seed_does_not_shrink_exam(
    league: SealedLeague, db: duckdb.DuckDBPyConnection
) -> None:
    first = league.state.runs[0]
    result = analyze(
        ExperimentState(experiment=EXPERIMENT, runs=(first,)),
        [d for d in league.training if d.name == first.name],
        db,
        league.source,
        league.protocol,
    )
    assert result.counts.expected == 800
    assert result.counts.missing == 400
    absent = next(r for r in result.seeds if r.run is None)
    assert absent.state == "missing" and all(v is None for v in absent.exports.values())
    assert absent.counts.missing_matches == 100
    assert result.modest_check is None


def test_no_runs_retains_both_declared_seeds(
    league: SealedLeague, db: duckdb.DuckDBPyConnection
) -> None:
    result = analyze(
        ExperimentState(experiment=EXPERIMENT), [], db, league.source, league.protocol
    )
    assert result.counts.missing == result.counts.expected == 800
    assert result.counts.missing_matches == 200 and not result.complete


@pytest.mark.parametrize(
    "failure", ["callback", "missing-callback", "receipt", "replay"]
)
def test_incomplete_sealed_schedule_stays_unknown(
    league: SealedLeague, db: duckdb.DuckDBPyConnection, failure: str
) -> None:
    key = db.execute(
        "SELECT match FROM matches WHERE json_extract_string(conditions, '$.panel') = 'anchors' LIMIT 1"
    ).fetchone()[0]
    if failure == "callback":
        db.execute(
            "UPDATE matches SET state = 'failed', error = 'synthetic callback failure' WHERE match = ?",
            [key],
        )
    elif failure == "missing-callback":
        db.execute("DELETE FROM matches WHERE match = ?", [key])
    elif failure == "receipt":
        db.execute("DELETE FROM evaluation_results WHERE match = ?", [key])
    else:
        db.execute("DELETE FROM card_plays WHERE match = ? AND game LIKE '%-0'", [key])
    result = report(league, db)
    assert result.counts.expected == 800 and not result.complete
    assert result.modest_check is None
    assert result.counts.valid + result.counts.failed + result.counts.missing == 800
    assert result.counts.failed_matches == int(failure == "callback")
    assert result.counts.failed == int(failure == "replay")


def test_secondary_missing_panel_does_not_redefine_primary_criterion(
    league: SealedLeague, db: duckdb.DuckDBPyConnection
) -> None:
    db.execute(
        "DELETE FROM matches WHERE json_extract_string(conditions, '$.panel') = 'cards'"
    )
    result = report(league, db)
    assert not result.complete and result.modest_check is True
    assert result.counts.missing == 160


@pytest.mark.parametrize("field,value", [("source", "wrong"), ("player", "wrong")])
def test_changed_evaluation_identity_is_rejected(
    league: SealedLeague, db: duckdb.DuckDBPyConnection, field: str, value: str
) -> None:
    db.execute(f"UPDATE seats SET {field} = ?", [value])
    with pytest.raises(ValueError, match="identity|participants"):
        report(league, db)


@pytest.mark.parametrize(
    "change",
    ["missing-leg", "failed-replay", "duplicate-leg", "checkpoint", "world", "decks"],
)
def test_evaluator_receipt_checks_each_game_and_identity(
    league: SealedLeague,
    db: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:

    digest = db.execute(
        "SELECT e.sha256 FROM evaluation_results e JOIN matches m USING (run, match) "
        "WHERE json_extract_string(m.conditions, '$.panel') = 'anchors' LIMIT 1"
    ).fetchone()[0]
    read = growing_pool_analysis._blob

    def edited(
        db: duckdb.DuckDBPyConnection,
        source: str | Path,
        sha: str,
        run: str,
        match: str,
    ) -> bytes:
        body = read(db, source, sha, run, match)
        if sha != digest:
            return body
        # Inject a controlled evaluator receipt after the independent SDK blob
        # check. This tests result reconciliation, not corrupt-byte recovery.
        raw = json.loads(body)
        if change == "missing-leg":
            raw["rows"].pop()
        elif change == "failed-replay":
            raw["rows"][0]["replay_passed"] = False
        elif change == "duplicate-leg":
            raw["rows"].append(raw["rows"][0])
        elif change == "checkpoint":
            raw["request"]["candidate"]["checkpoint_sha256"] = "0" * 64
        elif change == "world":
            raw["request"]["key"]["world"] = "w16"
        else:
            raw["request"]["authored_matchup"]["entries"].reverse()
        return json.dumps(raw).encode()

    monkeypatch.setattr(growing_pool_analysis, "_blob", edited)
    if change in ("missing-leg", "failed-replay"):
        result = report(league, db)
        assert not result.complete and result.modest_check is None
        assert result.counts.valid == 799
        assert result.counts.missing == int(change == "missing-leg")
        assert result.counts.failed == int(change == "failed-replay")
    else:
        with pytest.raises(ValueError, match="differs|duplicate"):
            report(league, db)


@pytest.mark.parametrize("change", ["missing", "nonfinite", "no-change", "rejected"])
def test_health_requires_declared_coordinates_and_actual_changes(
    league: SealedLeague,
    db: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    original = RunData.points
    spec = league.protocol.analysis
    metric = spec.zero_metrics[1] if change == "rejected" else spec.parameter_change

    def changed(data: RunData, name: str, clock: str = "wall") -> Points:
        points = original(data, name, clock)
        if name != metric:
            return points
        if change == "missing":
            return Points(x=points.x[:-1], y=points.y[:-1])
        values = points.y.copy()
        values[:] = (
            float("nan") if change == "nonfinite" else 1 if change == "rejected" else 0
        )
        return Points(x=points.x, y=values)

    monkeypatch.setattr(RunData, "points", changed)
    result = report(league, db)
    assert result.counts.valid == 800
    assert result.modest_check is (None if change == "missing" else False)
    assert result.complete is (change != "missing")
    assert all(r.health.metrics_sha256 for r in result.seeds)
    if change != "missing":
        assert all(r.health.failures for r in result.seeds)


def test_absent_export_keeps_planned_slots(
    league: SealedLeague, db: duckdb.DuckDBPyConnection
) -> None:
    runs = tuple(
        replace(
            run,
            checkpoints=Checkpoints(
                c for c in run.checkpoints if c.clocks["updates"] != 32
            ),
        )
        for run in league.state.runs
    )
    result = analyze(
        ExperimentState(experiment=EXPERIMENT, runs=runs),
        league.training,
        db,
        league.source,
        league.protocol,
    )
    assert result.counts.expected == 800 and result.counts.missing == 320
    assert all(r.exports[32] is None for r in result.seeds)
    assert result.modest_check is None


def test_analysis_cli_reports_empty_store_without_playing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    out = tmp_path / "analysis.json"
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "sys.argv", ["analysis", "--source", str(source), "--out", str(out)]
    )
    main()
    result = AnalysisResult.model_validate_json(out.read_bytes())
    assert len(result.seeds) == 2 and result.counts.missing == 800
    assert result.modest_check is None
