"""Prospective league contracts using manifests and synthetic receipts, no games."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from etude.eval.match import Result, evaluate as seal_matches
from etude.eval.query import statsdb
from etude.eval.records import Code, Environment, Match, Model
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    observe,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.launch.errors import LauncherError
from etude.launch.job_file import read_job_file
from etude.run.records import ArtifactRef
from etude.train.read import read_all
from etude.train.records import Step
from etude.train.run import track
import pytest
import torch

from experiments import (
    growing_pool as entry,
    growing_pool_analysis,
    growing_pool_runtime as growing_pool,
)
from experiments.growing_pool import BINDING
from experiments.growing_pool_exposure import decision_rows
from experiments.growing_pool_protocol import (
    ROOT,
    Manifest,
    Protocol,
    ProtocolBinding,
    World,
    audit_decks,
    read_binding,
)
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import engine_source_paths, source_bundle_sha256
from manabot.infra.source import SourceBundle
from manabot.sim.card_evidence import GameEvidence, GameTrace, retain_evidence
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.decks import UniformDecks
from manabot.training.regime import TrainSelfPlay
from tests.eval.arena_fixture import arena_fixture
from tests.eval.test_card_evaluation import card_fixture
from tests.eval.test_local_evaluation import _anchor_request, _replayed_rows

EXPERIMENT = BINDING.experiment
PROTOCOL = BINDING.protocol_path


@pytest.fixture
def protocol() -> Protocol:
    return Protocol.model_validate_json(PROTOCOL.read_bytes())


@pytest.fixture
def compatible_world(protocol: Protocol, monkeypatch: pytest.MonkeyPatch) -> Protocol:
    """Pure scheduling/compilation fixture; never encodes, replays or trains."""
    monkeypatch.setattr(World, "current", classmethod(lambda cls: protocol.world))
    return protocol


def test_reference_population_roundtrips_with_exact_ticket_mass(
    protocol: Protocol,
) -> None:
    source = protocol.manifest.source()
    restored = UniformDecks.model_validate_json(source.model_dump_json())
    assert restored.digest == source.digest
    assert source.sampling == "uniform-independent-v1"
    assert source.entry_ids is not None
    assert Counter(name.split("/")[0] for name in source.entry_ids) == {
        "anchor-ur": 2,
        "anchor-gw": 2,
        "new-ub": 1,
        "new-gr": 1,
        "mixed-ur": 1,
        "mixed-gw": 1,
    }
    assert [p.name for p in source.players()] == list(source.entry_ids)
    assert not any(name.startswith("test-") for name in source.entry_ids)
    assert restored.model_dump(mode="json") == source.model_dump(mode="json")
    changed = source.model_copy(update={"entry_ids": source.entry_ids[::-1]})
    assert changed.digest != source.digest


def test_authored_decks_meet_native_and_functional_checks(protocol: Protocol) -> None:
    protocol.manifest.source().check()
    audits = audit_decks(protocol.manifest)
    assert len(audits) == 9
    assert all(a.cards == 40 and a.lands == 17 and a.creatures >= 8 for a in audits)
    assert all(
        a.cheap_spells >= 8 and a.interaction >= 4 and a.synergy >= 4 for a in audits
    )


@pytest.mark.parametrize("zone", ["decklist", "sideboard"])
def test_card_holdout_excludes_both_training_seats_and_sideboards(
    protocol: Protocol, zone: str
) -> None:
    raw = protocol.manifest.model_dump(mode="json")
    raw["entries"][0]["deck"][zone]["Lightning Bolt"] = 1
    with pytest.raises(ValueError, match="held-out cards leak"):
        Manifest.model_validate(raw)


def test_renamed_identical_test_deck_is_not_a_holdout(protocol: Protocol) -> None:
    raw = protocol.manifest.model_dump(mode="json")
    raw["entries"][6].update(
        deck=raw["entries"][0]["deck"], parent=None, replacements=[]
    )
    with pytest.raises(ValueError, match="duplicates a training deck"):
        Manifest.model_validate(raw)


def test_combination_split_means_within_deck_not_cross_seat(protocol: Protocol) -> None:
    raw = protocol.manifest.model_dump(mode="json")
    raw["heldout_combinations"] = [["Fire Nation Cadets", "Dragonfly Swarm"]]
    with pytest.raises(ValueError, match="combination leaks"):
        Manifest.model_validate(raw)
    assert protocol.manifest.heldout_combinations == (
        ("Fire Nation Cadets", "Tiger-Dillo"),
    )


def test_heldout_deck_panel_cannot_masquerade_as_card_generalization(
    protocol: Protocol,
) -> None:
    raw = protocol.manifest.model_dump(mode="json")
    raw["panels"][2]["entries"][0] = "test-ur-card"
    with pytest.raises(ValueError, match="known training cards"):
        Manifest.model_validate(raw)


def test_changed_mixture_or_authored_construction_requires_new_manifest(
    protocol: Protocol,
) -> None:
    raw = protocol.manifest.model_dump(mode="json")
    raw["entries"][0]["tickets"] = 3
    with pytest.raises(ValueError, match="mixture"):
        Manifest.model_validate(raw)
    raw = protocol.manifest.model_dump(mode="json")
    raw["entries"][4]["replacements"][0]["copies"] = 2
    with pytest.raises(ValueError):
        Manifest.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["world", "engine", "shape", "truncate", "overspend"]
)
def test_frozen_contract_rejects_incompatible_inputs(
    protocol: Protocol, change: str
) -> None:
    raw = protocol.model_dump(mode="json")
    if change == "world":
        raw["world"]["card_digest"] = "0" * 64
    elif change == "engine":
        raw["world"]["engine_source_sha256"] = "0" * 64
    elif change == "shape":
        raw["observation"]["max_actions"] = 64
    elif change == "truncate":
        raw["observation"]["action_overflow"] = "first-legal"
    else:
        raw["budget"]["proposed_total_usd"] = 1
    with pytest.raises(ValueError):
        Protocol.model_validate(raw).admit()


def test_real_runtime_requires_exact_frozen_identity(protocol: Protocol) -> None:
    if protocol.world == World.current():
        assert BINDING.read(admit=True).digest == protocol.digest
    else:
        with pytest.raises(ValueError, match="runtime world or engine source"):
            BINDING.read(admit=True)
        with pytest.raises(ValueError, match="runtime world or engine source"):
            protocol.job(experiment=EXPERIMENT)


def test_incomplete_budget_cannot_inherit_a_spending_allowance(
    protocol: Protocol,
) -> None:
    raw = protocol.model_dump(mode="json")
    del raw["budget"]["evaluation_usd"]
    with pytest.raises(ValueError, match="budget.evaluation_usd"):
        Protocol.model_validate(raw)


def test_compilation_uses_existing_bounded_owners(compatible_world: Protocol) -> None:
    protocol = compatible_world
    job = protocol.job(experiment=EXPERIMENT)
    stage = job.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert stage.learning_seats == "both" and stage.games_per_stream == 1
    assert stage.game_errors is None and stage.behavior == "current-self"
    assert stage.execution.device == "cuda"
    assert stage.updates == 32 and stage.streams == 8
    assert job.regime.observation.action_overflow == "error"
    assert protocol.budget.training_games == 512
    assert protocol.evaluation_games == 800
    assert job.milestones == (8, 16)
    assert job.training_exposure
    assert job.model_dump(mode="json")["training_exposure"] is True
    assert "training_exposure" not in job.model_copy(
        update={"training_exposure": False}
    ).model_dump(mode="json")
    assert job.regime.deck_source == protocol.manifest.source()


def _run(tmp_path: Path, updates: tuple[int, ...]) -> TrainingRun:
    checkpoints = []
    for ordinal, update in enumerate(updates, start=1):
        path = tmp_path / f"u{update}.pt"
        path.write_text(str(update))
        checkpoints.append(
            Checkpoint(
                run="seed-2450401",
                line="seed-2450401",
                experiment=EXPERIMENT,
                ordinal=ordinal,
                artifact=ArtifactRef(uri=path.as_uri(), sha256="a" * 64, size_bytes=1),
                clocks={"updates": float(update)},
            )
        )
    return TrainingRun(
        name="seed-2450401",
        experiment=EXPERIMENT,
        state="finished",
        checkpoints=Checkpoints(checkpoints),
    )


def test_early_failure_retains_all_expected_export_slots(
    protocol: Protocol, tmp_path: Path
) -> None:
    slots = growing_pool.comparisons(
        _run(tmp_path, (0, 8)), protocol, experiment=EXPERIMENT
    )
    assert sum(s.expected_games for s in slots) == 400
    assert sum(s.expected_games for s in slots if s.matchup is not None) == 160
    assert {s.update for s in slots if s.matchup is None} == {16, 32}
    assert all(s.matchup is None for s in slots if s.anchor == "initial")


def test_etude_discovery_and_four_leg_design(
    compatible_world: Protocol, tmp_path: Path
) -> None:
    experiment = read_experiment(ROOT / "experiments/growing_pool.py")
    state = ExperimentState(
        experiment=EXPERIMENT, runs=(_run(tmp_path, (0, 8, 16, 32)),)
    )
    assert experiment.eval is not None
    jobs = experiment.eval(state)
    spec = evaluation_spec(jobs[0])
    assert spec.play.module == "experiments.growing_pool"
    matches = [
        match for matchup in spec.tournament.matchups for match in matchup.matches()
    ]
    assert sum(match.games for match in matches) == 400
    assert all(match.games == 4 for match in matches)
    matrix = growing_pool.deck_matchup(
        compatible_world, compatible_world.manifest.panels[0]
    )
    assert [matrix.decks(leg) for leg in range(4)] == [
        ("anchor-ur", "anchor-gw"),
        ("anchor-ur", "anchor-gw"),
        ("anchor-gw", "anchor-ur"),
        ("anchor-gw", "anchor-ur"),
    ]
    assert jobs[0].max_attempts == 1


def test_exposure_counts_committed_choices_by_actual_actor() -> None:
    game = card_fixture(players=("opponent", "candidate"), leg=1)
    data = dict(game.trace.data)
    data.pop("game_trace_sha256")
    data["decisions"] = [
        {
            "actor": 0,
            "player_id": "opponent",
            "action_space_kind": "Priority",
            "chosen_offer": {"action_type": "CastSpell"},
            "post_state_digest": "a" * 64,
        }
    ]
    digest = canonical_sha256(data)
    data["game_trace_sha256"] = digest
    # A synthetic receipt tests reduction only, not a claim of native replay.
    game = GameEvidence.model_validate(
        {
            **game.model_dump(by_alias=True),
            "trace": GameTrace(data=data),
            "trace_sha256": digest,
        }
    )
    rows = decision_rows([gzip.compress(game.model_dump_json(by_alias=True).encode())])
    totals = {
        row["seat"]: row["committed_choices"]
        for row in rows
        if row["decision_kind"] == "all"
    }
    assert totals == {0: 0, 1: 1}
    assert [row["game_seat"] for row in rows if row["seat"] == 1] == [0, 0]
    assert all("offered" not in row and "resolved" not in row for row in rows)


def test_paused_milestone_keeps_initialization_and_later_missing_slots(
    protocol: Protocol, tmp_path: Path
) -> None:
    slots = growing_pool.comparisons(
        _run(tmp_path, (0, 8, 8)), protocol, experiment=EXPERIMENT
    )
    assert sum(s.expected_games for s in slots if s.matchup is not None) == 160
    assert {s.update for s in slots if s.matchup is None} == {16, 32}


def test_checkpoint_population_and_initialization_role_are_admitted(
    compatible_world: Protocol, tmp_path: Path
) -> None:
    protocol = compatible_world
    path = tmp_path / "metadata.pt"
    origin = {
        "regime_digest": canonical_sha256(protocol.regime().model_dump(mode="json")),
        "weights": "initial_raw",
    }
    torch.save({"bc": origin}, path)
    growing_pool.validate_training_origin(path, protocol, initial=True)
    with pytest.raises(ValueError, match="update role"):
        growing_pool.validate_training_origin(path, protocol, initial=False)
    torch.save({"bc": {**origin, "regime_digest": "0" * 64}}, path)
    with pytest.raises(ValueError, match="different regime/population"):
        growing_pool.validate_training_origin(path, protocol, initial=True)


def test_job_refuses_a_commit_without_protocol_sources(
    compatible_world: Protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    def old_bundle(repo: Path, revision: str) -> SourceBundle:
        return SourceBundle(commit="0" * 40, files={})

    monkeypatch.setattr("manabot.experiment.job.bundle_at", old_bundle)
    with pytest.raises(ValueError, match="commit does not contain"):
        entry.runs("an-old-commit")


def test_missing_and_failed_games_stay_unknown() -> None:
    request = _anchor_request()
    rows = _replayed_rows(request)[:2]
    rows[1] = rows[1].model_copy(update={"failure": "NativePanic", "score_a": None})
    result = EvaluationResult(request=request, rows=rows, error="interrupted")
    conditions = growing_pool.Conditions(
        experiment=EXPERIMENT,
        protocol_sha256="a" * 64,
        panel="anchors",
        deal=101,
        update=8,
        anchor="heuristic",
    )
    delivered = growing_pool.outcome_rows(result, conditions)
    assert len(delivered) == 8
    assert sum(bool(row["valid"]) for row in delivered) == 2
    assert sum(bool(row["attempted"]) for row in delivered) == 4
    assert all(row["score"] is None for row in delivered if row["game"] != 0)


def test_fixed_opponent_identity_survives_manifest_revision(
    protocol: Protocol, tmp_path: Path
) -> None:
    run = _run(tmp_path, (0,))
    first = growing_pool.comparisons(run, protocol, experiment=EXPERIMENT)[0].matchup
    changed = protocol.model_copy(
        update={
            "manifest": protocol.manifest.model_copy(
                update={"id": "later-reference-league"}
            )
        }
    )
    second = growing_pool.comparisons(run, changed, experiment=EXPERIMENT)[0].matchup
    assert first is not None and second is not None
    assert first.players[1] == second.players[1]
    assert first.conditions != second.conditions


def test_callback_delivers_failure_slots_and_replay_linked_exposure(
    compatible_world: Protocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = compatible_world
    slot = growing_pool.comparisons(
        _run(tmp_path, (0,)), protocol, experiment=EXPERIMENT
    )[0]
    assert slot.matchup is not None
    match = slot.matchup.matches()[0]
    contract = arena_fixture()

    def registration(
        player: Model | Code, name: str, matchup: DeckMatchup, protocol: Protocol
    ) -> tuple[PlayerRegistration, Path | None]:
        return contract.anchors[0].model_copy(update={"player_id": name}), None

    def evaluate_fixture(
        request: EvaluationRequest, *, out_dir: Path, card_evidence_dir: Path
    ) -> EvaluationResult:
        out_dir.mkdir(parents=True)
        evidence = card_fixture(players=("candidate", "opponent"))
        reference = retain_evidence(card_evidence_dir, evidence)
        (out_dir / "trace.json").write_text(
            json.dumps({"card_evidence": [reference.model_dump()]})
        )
        return EvaluationResult(request=request, error="synthetic missing outcome")

    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "evaluation"))
    monkeypatch.setattr(growing_pool, "_registration", registration)
    monkeypatch.setattr(growing_pool, "evaluate", evaluate_fixture)
    result = entry.play(match)
    assert set(result.tables) == {
        "games",
        "decisions",
        "evaluation_results",
        "card_plays",
        "card_counts",
        "cards",
    }
    assert len(result.tables["games"]) == 8
    assert all(row["score"] is None for row in result.tables["games"])
    assert len(result.blobs) == 2
    assert {row["evidence_sha256"] for row in result.tables["decisions"]} == {
        row["evidence_sha256"] for row in result.tables["card_plays"]
    }


def test_analysis_parameters_are_prospective_and_required(protocol: Protocol) -> None:
    raw = protocol.model_dump(mode="json")
    del raw["analysis"]["bootstrap_resamples"]
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        Protocol.model_validate(raw)
    raw = protocol.model_dump(mode="json")
    raw["analysis"]["bootstrap_seed"] += 1
    assert Protocol.model_validate(raw).digest != protocol.digest


@pytest.fixture
def two_bindings(
    compatible_world: Protocol, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ProtocolBinding, ...]]:
    """Real importable entries; all outcomes are synthetic and no jobs launch."""
    names = ("league_binding_a", "league_binding_b")
    # Declaration admission requires sources inside the repository. Keep these
    # ephemeral, and remove the imported modules before deleting their files.
    with TemporaryDirectory(prefix=".binding-proof-", dir=ROOT) as directory:
        folder = Path(directory)
        monkeypatch.syspath_prepend(directory)
        bindings = []
        try:
            for index, name in enumerate(names):
                raw = compatible_world.model_dump(mode="json")
                raw["id"] = f"binding-{index}"
                raw["manifest"]["id"] = f"population-{index}"
                raw["budget"]["seeds"] = [9100 + index]
                raw["budget"]["game_seconds"] += index
                protocol = Protocol.model_validate(raw)
                path = folder / f"{name}.json"
                path.write_text(protocol.model_dump_json())
                source = (
                    BINDING.entry_point.read_text()
                    .replace(f'module="{BINDING.module}"', f'module="{name}"')
                    .replace(
                        'ROOT / "experiments/protocols/growing-pool-245-v1.json"',
                        f"ROOT / {str(path.relative_to(ROOT))!r}",
                    )
                    .replace(BINDING.protocol_sha256, protocol.digest)
                )
                (folder / f"{name}.py").write_text(source)
                bindings.append(read_binding(name))
            yield tuple(bindings)
        finally:
            for name in names:
                sys.modules.pop(name, None)


def _bound_run(binding: ProtocolBinding, tmp_path: Path) -> TrainingRun:
    run = _run(tmp_path, (0,))
    return replace(
        run,
        experiment=binding.experiment,
        checkpoints=Checkpoints(
            c.model_copy(update={"experiment": binding.experiment})
            for c in run.checkpoints
        ),
    )


def test_two_bindings_discover_declare_and_route_callbacks(
    two_bindings: tuple[ProtocolBinding, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = arena_fixture()

    def bundle(repo: Path, revision: str) -> SourceBundle:
        # Synthetic committed-source inventory, not runtime/world admission.
        paths = [
            *engine_source_paths(),
            ROOT / "uv.lock",
            *(p for b in two_bindings for p in growing_pool._source(b)),
        ]
        return SourceBundle(
            commit="0" * 40,
            files={str(p.relative_to(repo)): file_sha256(p) for p in paths},
        )

    def registration(
        player: Model | Code, name: str, matchup: DeckMatchup, protocol: Protocol
    ) -> tuple[PlayerRegistration, Path | None]:
        return contract.anchors[0].model_copy(update={"player_id": name}), None

    def synthetic(
        request: EvaluationRequest, *, out_dir: Path, card_evidence_dir: Path
    ) -> EvaluationResult:
        return EvaluationResult(request=request, error="synthetic missing outcome")

    monkeypatch.setattr("manabot.experiment.job.bundle_at", bundle)
    monkeypatch.setattr(growing_pool, "_registration", registration)
    monkeypatch.setattr(growing_pool, "evaluate", synthetic)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "evaluation"))
    matches = []
    for binding in two_bindings:
        protocol = binding.read()
        declarations = read_job_file(binding.entry_point, ["--revision", "fixture"])
        assert [r.seed for r in declarations] == list(protocol.budget.seeds)
        job = TrainingJob.model_validate_json(declarations[0].job.config or "")
        assert job.experiment == binding.experiment
        assert job.regime.id == protocol.id
        assert job.regime.deck_source == protocol.manifest.source()
        assert job.source_bundle is not None
        for path in (binding.entry_point, binding.protocol_path):
            assert job.source_bundle.files[str(path.relative_to(ROOT))] == file_sha256(
                path
            )
        experiment = read_experiment(binding.entry_point)
        assert experiment.name == binding.experiment
        state = ExperimentState(
            experiment=binding.experiment, runs=(_bound_run(binding, tmp_path),)
        )
        spec = evaluation_spec(experiment.eval(state)[0])
        assert spec.play.module == binding.module and spec.play.name == "play"
        match = spec.tournament.matchups[0].matches()[0]
        matches.append(match)
        result = spec.play.load()(match)
        retained = EvaluationResult.model_validate_json(result.blobs[0])
        assert retained.request.key.arena_version == protocol.id
        assert retained.request.game_seconds == protocol.budget.game_seconds
        assert retained.request.authored_matchup is not None
        assert (
            retained.request.authored_matchup.population_sha256
            == protocol.manifest.digest
        )
        assert {row["protocol_sha256"] for row in result.tables["games"]} == {
            protocol.digest
        }
    assert matches[0].conditions != matches[1].conditions
    with pytest.raises(ValueError, match="frozen schedule"):
        two_bindings[1].callback(matches[0])
    # Even identical protocol contents cannot cross experiment namespaces.
    crossed = matches[0].model_copy(
        update={
            "conditions": {
                **matches[0].conditions,
                "experiment": two_bindings[1].experiment,
            }
        }
    )
    with pytest.raises(ValueError, match="frozen schedule"):
        two_bindings[0].callback(crossed)


def test_binding_rejects_crossed_source_state_and_exports(
    two_bindings: tuple[ProtocolBinding, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = two_bindings
    copied = first.entry_point.with_name("unbound_copy.py")
    copied.write_text(first.entry_point.read_text())
    with pytest.raises(ValueError, match="discovered file"):
        read_experiment(copied)
    with pytest.raises(ValueError, match="discovered file"):
        read_job_file(copied, ["--revision", "fixture"])
    with pytest.raises(ValueError, match="wrong experiment"):
        read_experiment(second.entry_point).eval(
            ExperimentState(experiment=first.experiment)
        )
    with pytest.raises(ValueError, match="experiment's exports"):
        read_experiment(second.entry_point).eval(
            ExperimentState(
                experiment=second.experiment, runs=(_bound_run(first, tmp_path),)
            )
        )
    original = first.protocol_path.read_bytes()
    try:
        first.protocol_path.write_bytes(second.protocol_path.read_bytes())
        with pytest.raises(ValueError, match="protocol digest"):
            first.read()
    finally:
        first.protocol_path.write_bytes(original)

    def missing_entry(repo: Path, revision: str) -> SourceBundle:
        paths = [*growing_pool._source(first), *engine_source_paths(), ROOT / "uv.lock"]
        return SourceBundle(
            commit="0" * 40,
            files={
                str(p.relative_to(repo)): file_sha256(p)
                for p in paths
                if p != first.entry_point
            },
        )

    monkeypatch.setattr("manabot.experiment.job.bundle_at", missing_entry)
    with pytest.raises(LauncherError, match="commit does not contain"):
        read_job_file(first.entry_point, ["--revision", "missing-entry"])


def test_two_bindings_analyze_their_own_sealed_stores(
    two_bindings: tuple[ProtocolBinding, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "store"
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    for name in ("ETUDE_RUN", "ETUDE_TRACK_DIR", "ETUDE_EXPERIMENT", "ETUDE_CONTINUES"):
        monkeypatch.delenv(name, raising=False)

    def empty_callback(match: Match) -> Result:
        return Result(tables={}, blobs=[])

    for index, binding in enumerate(two_bindings, start=1):
        protocol = binding.read()
        run = track(
            experiment=binding.experiment,
            run=f"bound-learner-{index}",
            seed=protocol.budget.seeds[0],
            clocks=[Step("updates", "updates")],
            directory=source,
            strict=True,
            config={
                "training": {
                    "run_id": f"bound-learner-{index}",
                    "regime": protocol.regime().model_dump(mode="json"),
                    "identities": {
                        "world": protocol.world.world_version,
                        "engine_source_sha256": protocol.world.engine_source_sha256,
                        "content_digest": protocol.world.card_digest,
                        "deck_source_sha256": protocol.manifest.source().digest,
                        "training_source_sha256": source_bundle_sha256(
                            growing_pool.policy_source()
                        ),
                    },
                }
            },
        )
        try:
            checkpoint = tmp_path / f"initial-{index}.pt"
            checkpoint.write_text("synthetic export, never loaded")
            run.checkpoint(checkpoint, updates=0)
        finally:
            run.finish()
        state = observe(source, binding.experiment)
        slot = growing_pool.comparisons(
            state.runs[0], protocol, experiment=binding.experiment
        )[0]
        assert slot.matchup is not None
        seal_matches(
            experiment=binding.experiment,
            environment=Environment(
                name=protocol.id,
                files=protocol.digest,
                config={"status": protocol.status},
            ),
            matches=slot.matchup.matches()[:index],
            play=empty_callback,
            directory=source,
            run=f"bound-evaluation-{index}",
        )
    # Read both experiments from the same directory. Each has a distinct seed,
    # learner and callback count, so silently opening v1 or the other store fails.
    for index, binding in enumerate(two_bindings, start=1):
        out = tmp_path / f"report-{index}.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "analysis",
                "--experiment",
                binding.module,
                "--source",
                str(source),
                "--out",
                str(out),
            ],
        )
        growing_pool_analysis.main()
        report = growing_pool_analysis.AnalysisResult.model_validate_json(
            out.read_bytes()
        )
        assert report.experiment == binding.experiment
        assert report.protocol_sha256 == binding.protocol_sha256
        assert report.seeds[0].seed == binding.read().budget.seeds[0]
        assert report.seeds[0].run == f"bound-learner-{index}"
        assert report.counts.attempted_matches == index
        assert report.counts.expected == binding.read().evaluation_games
        assert not report.complete and not report.training_exposure_complete
    first, second = two_bindings
    with statsdb(first.experiment, source) as db:
        with pytest.raises(ValueError, match="selected experiment"):
            growing_pool_analysis.analyze(
                observe(source, first.experiment), read_all(source), db, source, second
            )
