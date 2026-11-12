"""Small end-to-end contracts for pools, picking, public results and stage modules."""

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

from pydantic import BaseModel
import pytest
import torch

from experiments.tla_corpus import Corpus, GameEvidence, Source, Trophy
from experiments.tla_pools import derive_pools, main as pools_main
from manabot import managym
from manabot.infra.config import ModelConfig
from manabot.infra.identity import StrictModel, file_sha256
from manabot.model.checkpoint import load_checkpoint
from manabot.training.decks import UniformDecks
from manabot.training.execution import execute_regime
from manabot.training.game_results import read_game_results
from manabot.training.pools import PoolCorpus
from manabot.training.regime import ModuleStage, TrainingRegime, TrainSelfPlay
from manabot.training.stages.contract import StageContext, StageDefinition
from manabot.training.stages.model_inventory import ModelInventory
from manabot.verify.store import VerifyStore


def test_limited_python_choices_and_malformed_prefixes() -> None:
    rules = managym.LimitedRules(pool={"Llanowar Elves": 6})
    assert not rules.choices(picks={"Llanowar Elves": 6, "Forest": 33}).can_stop
    choices = rules.choices(picks={"Llanowar Elves": 6, "Forest": 34})
    assert choices.can_stop and "Forest" in choices.cards
    assert "Llanowar Elves" not in choices.cards
    assert not rules.check(picks={"Forest": 61})  # Construction is not the cap.
    assert rules.choices(picks={"Forest": 60}).cards == []
    for picks in ({"Llanowar Elves": 7}, {"Forest": 0}, {"Ally": 1}, {"Forest": 61}):
        with pytest.raises(ValueError):
            rules.choices(picks=picks)
    for count in (True, 1.5, -1, "2"):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            managym.LimitedRules(pool={"Forest": count})


def _pool_source(event_ids: tuple[str, ...] = ("a", "b", "c")) -> Corpus:
    return Corpus(
        target=len(event_ids),
        source=Source(
            retrieved_at="2026-11-11",
            etag=None,
            last_modified=None,
            sha256="a" * 64,
            bytes=1,
        ),
        world={"version": "historical"},
        scanned_rows=21,
        exhausted=False,
        trophies=tuple(
            Trophy(
                event_id=name,
                draft_time="2026-11-11",
                games=tuple(
                    GameEvidence(
                        row=i,
                        match_number=i,
                        game_time="2026-11-11",
                        build_index=1,
                        won=True,
                    )
                    for i in range(1, 8)
                ),
                decklist=managym.Decklist(
                    f"Main\n6 Llanowar Elves\n{34 + (name == 'c')} Forest\nSideboard\n4 Llanowar Elves\n2 Island\n"
                ),
                deck=None,  # Earlier admission must not gate current-world rebuilding.
            )
            for name in event_ids
        ),
    )


def test_pool_derivation_preserves_events_references_and_frozen_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pool_source()
    path = tmp_path / "source.json"
    payload = source.model_dump(mode="json")
    payload["trophies"][0]["decklist"]["issues"] = [
        {
            "line": 2,
            "section": "main",
            "card_name": "Llanowar Elves",
            "code": "unknown",
            "message": "unknown in the historical world",
        }
    ]
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    corpus = derive_pools(path, held_out=1)
    assert derive_pools(path, held_out=1) == corpus
    assert path.read_bytes() == before
    assert len(corpus.entries) == 3
    assert len(corpus.held_out_event_ids) == 1
    assert all(entry.pool == {"Llanowar Elves": 10} for entry in corpus.entries)
    assert [sum(entry.reference.main.values()) for entry in corpus.entries] == [
        40,
        40,
        41,
    ]
    assert corpus.entries[0].reference == corpus.entries[1].reference
    assert corpus.entries[0].entry_id != corpus.entries[1].entry_id
    output = tmp_path / "pools-v1" / "corpus.json"
    monkeypatch.setattr(
        sys, "argv", ["tla_pools", str(path), str(output), "--held-out", "1"]
    )
    pools_main()
    with pytest.raises(ValueError, match="frozen"):
        pools_main()
    restored = PoolCorpus.read(output)
    assert restored.digest == corpus.digest
    assert len(restored.population("train").decks) == 2
    assert len(restored.population("held_out").decks) == 1
    damaged = corpus.model_dump(mode="json")
    damaged["held_out_event_ids"] = ["missing"]
    with pytest.raises(ValueError, match="held-out"):
        PoolCorpus.model_validate(damaged)


def test_pool_exclusions_preserve_source_split_without_backfilling(
    tmp_path: Path,
) -> None:
    source = _pool_source(("a", "b", "c", "d"))
    path = tmp_path / "source.json"
    path.write_text(source.model_dump_json())
    complete = derive_pools(path, held_out=2)
    excluded = {
        complete.held_out_event_ids[0],
        next(e.event_id for e in complete.entries if e.split == "train"),
    }
    payload = source.model_dump(mode="json")
    for trophy in payload["trophies"]:
        if trophy["event_id"] in excluded:
            trophy["decklist"] = managym.Decklist(
                "Main\n40 Forest\n1 Unknown main card\nSideboard\n1 Unknown sideboard card\n"
            ).to_dict()
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    corpus = derive_pools(path, held_out=2)
    assert path.read_bytes() == before
    assert len(corpus.entries) == 2
    assert corpus.held_out_event_ids == complete.held_out_event_ids[1:]
    assert {e.event_id: e.split for e in corpus.entries} == {
        e.event_id: e.split for e in complete.entries if e.event_id not in excluded
    }
    assert corpus.provenance.source_event_count == 4
    assert corpus.provenance.held_out_count == 2
    assert {e.event_id for e in corpus.provenance.excluded_events} == excluded
    assert all(
        e.unknown_names == ("Unknown main card", "Unknown sideboard card")
        for e in corpus.provenance.excluded_events
    )
    output = tmp_path / "pools.json"
    output.write_text(corpus.model_dump_json())
    assert PoolCorpus.read(output) == corpus
    damaged = corpus.model_dump(mode="json")
    damaged["provenance"]["excluded_events"].pop()
    with pytest.raises(ValueError, match="cover the source"):
        PoolCorpus.model_validate(damaged)


def test_pool_derivation_does_not_exclude_malformed_lists(tmp_path: Path) -> None:
    payload = _pool_source().model_dump(mode="json")
    payload["trophies"][0]["decklist"] = managym.Decklist("Main\n0 Forest\n").to_dict()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(managym.DeckImportError):
        derive_pools(path, held_out=1)


def _tiny_regime() -> TrainingRegime:
    return TrainingRegime(
        id="foundation-proof",
        world=managym.WORLD_VERSION,
        agent=ModelConfig(
            hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP
        ),
        deck_source=UniformDecks(
            decks=(managym.Deck({"Forest": 10}), managym.Deck({"Forest": 10})),
            entry_ids=("event-a", "event-b"),
            build_episodes=(None, "build-b-1"),
            sampling="uniform-independent-v1",
        ),
        schedule_clock="iteration_fraction",
        stages=[
            TrainSelfPlay(
                id="play",
                operation="train_self_play",
                trainable="none",
                streams=2,
                transitions=1,
                updates=1,
                games_per_stream=1,
            )
        ],
    )


def test_self_play_without_recovery_emits_public_game_rows(tmp_path: Path) -> None:
    regime = _tiny_regime()
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 17402, tmp_path / "run", store)
    assert run.status == "completed"
    assert run.game_error_ledgers == []
    artifact = run.stages[0].artifacts["game_results"]
    path = Path(artifact["path"])
    assert path.name == f"{file_sha256(path)}.jsonl"
    assert "/artifacts/game-results/" in str(path)
    results = read_game_results(path)
    assert len(results) == run.stages[0].games == 2
    assert {(row.stream, row.game) for row in results} == {(0, 1), (1, 1)}
    for row in results:
        assert row.on_play == 0 and row.winner in (0, 1, None)
        for seat in row.seats:
            assert seat.entry_id in ("event-a", "event-b")
            assert seat.build_episode == (
                "build-b-1" if seat.entry_id == "event-b" else None
            )
    assert not any(key in path.read_text() for key in ("actions", "seed", "matchup"))


def _named_regime(stages: list[ModuleStage]) -> TrainingRegime:
    source = _tiny_regime()
    return TrainingRegime(
        schema_version=3,
        id="named-models",
        world=source.world,
        deck_source=source.deck_source,
        models={"builder": source.agent, "pilot": source.agent.model_copy(deep=True)},
        stages=stages,
    )


def test_two_models_execute_module_and_preserve_legacy_serialization(
    tmp_path: Path,
) -> None:
    legacy = _tiny_regime().model_dump(mode="json")
    assert "models" not in legacy and "trains" not in legacy["stages"][0]
    assert TrainingRegime.model_validate(legacy).model_dump(mode="json") == legacy
    regime = _named_regime(
        [
            ModuleStage(
                id="inspect",
                module="manabot.training.stages.model_inventory",
                trains="builder",
                frozen=("pilot",),
            )
        ]
    )
    encoded = regime.model_dump(mode="json")
    assert "agent" not in encoded
    assert TrainingRegime.model_validate(encoded).model_dump(mode="json") == encoded
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 12, tmp_path / "run", store)
    assert run.status == "completed"
    output = ModelInventory.model_validate_json(
        Path(run.stages[0].artifacts["result"]["path"]).read_bytes()
    )
    assert output.trainable_parameters == output.frozen_parameters["pilot"] > 0
    for change in (
        {"trains": "missing"},
        {"frozen": ("builder",)},
        {"inputs": {"future": "later"}},
    ):
        invalid = deepcopy(encoded)
        invalid["stages"][0].update(change)
        with pytest.raises(ValueError):
            TrainingRegime.model_validate(invalid)


class WeightReceipt(StrictModel):
    value: float


def test_modules_exchange_typed_artifacts_and_cannot_mutate_frozen_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def train(context: StageContext, inputs: Mapping[str, BaseModel]) -> WeightReceipt:
        assert context.trainable is not None
        with torch.no_grad():
            next(context.trainable.parameters()).fill_(3)
            next(context.frozen["pilot"].parameters()).fill_(99)
        return WeightReceipt(value=3)

    def consume(
        context: StageContext, inputs: Mapping[str, BaseModel]
    ) -> WeightReceipt:
        receipt = inputs["weights"]
        assert isinstance(receipt, WeightReceipt)
        assert receipt.value == 3
        assert not any(p.requires_grad for p in context.frozen["builder"].parameters())
        assert next(context.frozen["builder"].parameters()).flatten()[0].item() == 3
        assert next(context.frozen["pilot"].parameters()).flatten()[0].item() != 99
        return receipt

    for name, definition in (
        ("foundation.train", StageDefinition({}, WeightReceipt, train)),
        (
            "foundation.consume",
            StageDefinition({"weights": WeightReceipt}, WeightReceipt, consume),
        ),
    ):
        module = ModuleType(name)
        module.STAGE = definition
        monkeypatch.setitem(sys.modules, name, module)
    stages = [
        ModuleStage(
            id="train", module="foundation.train", trains="builder", frozen=("pilot",)
        ),
        ModuleStage(
            id="consume",
            module="foundation.consume",
            frozen=("builder", "pilot"),
            inputs={"weights": "train"},
        ),
    ]
    regime = _named_regime(stages)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 12, tmp_path / "run", store)
    assert run.status == "completed"
    model, _ = load_checkpoint(run.stages[0].artifacts["raw"]["path"])
    assert next(model.parameters()).flatten()[0].item() == 3
    assert run.stages[1].inputs["weights"] == run.stages[0].artifacts["result"]
    with pytest.raises(ValueError, match="earlier"):
        _named_regime(list(reversed(stages)))
    incompatible = stages[0].model_copy(
        update={"module": "manabot.training.stages.model_inventory"}
    )
    with pytest.raises(ValueError, match="incompatible"):
        _named_regime([incompatible, stages[1]])


def test_completed_rows_survive_optimizer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controlled optimizer failure")

    regime = _tiny_regime()
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.trainable = "policy_value"
    monkeypatch.setattr("manabot.training.execution.update_iteration", fail)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(RuntimeError, match="controlled optimizer failure"):
            execute_regime(regime, 17402, tmp_path / "run", store)
        saved = json.loads((tmp_path / "run/run.json").read_text())
        run = store.training_run(saved["id"])
    assert run.status == "failed"
    results = read_game_results(Path(run.stages[0].artifacts["game_results"]["path"]))
    assert len(results) == 2


def test_named_self_play_keeps_its_model_and_exports_by_name(tmp_path: Path) -> None:
    source = _tiny_regime()
    play = source.stages[0]
    assert isinstance(play, TrainSelfPlay)
    play.trains = "pilot"
    regime = TrainingRegime(
        schema_version=3,
        id="named-play",
        world=source.world,
        deck_source=source.deck_source,
        models={"pilot": source.agent, "builder": source.agent.model_copy(deep=True)},
        schedule_clock="iteration_fraction",
        stages=[
            play,
            ModuleStage(
                id="inspect",
                module="manabot.training.stages.model_inventory",
                frozen=("pilot", "builder"),
            ),
        ],
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 17402, tmp_path / "run", store)
    assert run.status == "completed"
    assert run.selected_artifact is None
    assert run.selected_models == {"pilot": run.stages[0].artifacts["raw"]}
    model, _ = load_checkpoint(run.selected_models["pilot"]["path"])
    assert model.config == source.agent
    assert (
        len(read_game_results(Path(run.stages[0].artifacts["game_results"]["path"])))
        == 2
    )
