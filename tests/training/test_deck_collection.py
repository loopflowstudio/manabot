"""Complete imported decks through the ordinary native training collector.

Tiny admitted development lists bound the proof: collection/inference only,
without optimization, a data download, or a training campaign.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.deck_collection import DeckCollection, main, prepare_collection
from manabot.decklist import DeckDocument, DeckImportError, parse_decklist
from manabot.env import Match, ObservationSpace, Reward
from manabot.env.failures import CaptureScope, FailureEvidence, capture_scope
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.catalog import inspect_checkpoint
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.inference import PolicyInference
from manabot.model.world import validate_agent_setup
from manabot.training.execution import validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorPolicy,
    replay_attempt,
)
import managym

PACK = "tla-pool-v1"


def _documents() -> list[DeckDocument]:
    return [
        parse_decklist(f"10 {name}\nSideboard\n1 Forest\n", f"{name}.txt")
        for name in ("Plains", "Mountain", "Island")
    ]


@pytest.fixture
def collection() -> DeckCollection:
    return prepare_collection(_documents(), PACK)


def _agent() -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
    )


def _collector(
    collection: DeckCollection, recovery: Literal["exact", "current"] = "current"
) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(),
        Match(collection.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=2,
        deck_collection=collection,
        recovery_max_microsteps=1000 if recovery == "exact" else None,
        recovery_max_game_microsteps=1000 if recovery == "current" else None,
    )


def test_collection_preserves_sources_deduplicates_and_rejects_whole_input(
    tmp_path: Path,
) -> None:
    documents = _documents() + [
        parse_decklist(
            "# another record\r\n5 plains\r\n5 Plains\r\nSideboard\r\n1 Forest\r\n",
            "duplicate.txt",
        )
    ]
    collection = prepare_collection(documents, PACK)
    assert len(collection.decks) == 3
    assert sum(len(deck.sources) for deck in collection.decks) == 4
    assert any(
        source.source == documents[-1].source
        for deck in collection.decks
        for source in deck.sources
    )
    assert prepare_collection(list(reversed(documents)), PACK) == collection
    target = tmp_path / "collection.json"
    target.write_text(collection.model_dump_json())
    assert DeckCollection.load(target) == collection
    with pytest.raises(DeckImportError) as error:
        prepare_collection(
            documents + [parse_decklist("0 Plains\n1 Not A Card", "broken.txt")], PACK
        )
    assert "broken.txt:1" in str(error.value) and "broken.txt:2" in str(error.value)
    with pytest.raises(DeckImportError) as unsupported:
        prepare_collection(
            documents + [parse_decklist("1 Rocky Rebuke", "unsupported.txt")], PACK
        )
    assert any(
        issue.code == "unsupported"
        for doc in unsupported.value.documents.values()
        for issue in doc.issues
    )
    with pytest.raises(ValueError, match="at least two distinct"):
        prepare_collection([documents[0], documents[-1]], PACK)
    value = collection.model_dump(mode="json")
    value["decks"][0]["main"] = {"Plains": 3}
    with pytest.raises(ValueError, match="identity"):
        DeckCollection.model_validate(value)


def test_collection_cli_retains_files_and_never_publishes_a_filtered_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for document in _documents():
        path = tmp_path / document.filename
        path.write_bytes(document.source.encode("utf-8"))
        paths.append(str(path))
    target = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["deck_collection", "--pool", PACK, "--out", str(target), *paths]
    )
    main()
    assert len(DeckCollection.load(target).decks) == 3
    with pytest.raises(FileExistsError):
        main()
    target.unlink()
    Path(paths[1]).write_text("1 Unknown Card")
    with pytest.raises(DeckImportError):
        main()
    assert not target.exists()


@pytest.mark.parametrize("field", ["learning_schema_hash", "pool", "digest"])
def test_collection_rejects_content_schema_and_manifest_drift(
    collection: DeckCollection, tmp_path: Path, field: str
) -> None:
    value = collection.model_dump(mode="json")
    if field == "pool":
        value["pool"]["content_digest"] = "0" * 64
    else:
        value[field] = "0" * 64
    if field != "digest":
        value["digest"] = canonical_sha256(
            {key: item for key, item in value.items() if key != "digest"}
        )
    target = tmp_path / "changed.json"
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="mismatch"):
        DeckCollection.load(target)


@pytest.mark.parametrize("recovery", ["exact", "current"])
def test_collection_collector_resets_recovers_seats_seeds_and_exposure(
    collection: DeckCollection, recovery: Literal["exact", "current"]
) -> None:
    agent = _agent()
    first = _collector(collection, recovery)
    restored = _collector(collection, recovery)
    try:
        initial = first._env.current_game_seeds()
        batch = first.collect(agent, 1)
        assert first.stats.games == 4
        assert first._env.current_game_seeds() != initial
        exposure = first.stats.exposure
        assert len(exposure) >= 3
        assert sum(row["game_roots"] for row in exposure.values()) == 6
        assert (
            sum(row["learner_transitions"] for row in exposure.values())
            == batch.actions.size
        )
        assert all(key.split("/")[0] != key.split("/")[1] for key in exposure)
        restored.restore(first.snapshot(), lambda: None)
        left, right = first.collect(agent, 1), restored.collect(agent, 1)
        for key in left.obs:
            np.testing.assert_array_equal(left.obs[key], right.obs[key])
        np.testing.assert_array_equal(left.actions, right.actions)
        assert first._self_play_decks == restored._self_play_decks
        assert first._env.current_game_seeds() == restored._env.current_game_seeds()
        assert first.stats.exposure == restored.stats.exposure
        snapshot = first.snapshot()
        changed = prepare_collection(
            _documents() + [parse_decklist("10 Forest", "fourth.txt")], PACK
        )
        fresh = _collector(changed, recovery)
        with pytest.raises(ValueError, match="collection identity mismatch"):
            fresh.restore(snapshot, lambda: None)
        fresh.close()
    finally:
        first.close()
        restored.close()


def test_collection_regime_and_checkpoint_bind_all_decks(
    collection: DeckCollection, tmp_path: Path
) -> None:
    manifest = tmp_path / "collection.json"
    manifest.write_text(collection.model_dump_json())
    regime = TrainingRegime(
        id="collection-proof",
        world=managym.WORLD_VERSION,
        deck_collection=str(manifest),
        agent=ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
        stages=[
            TrainSelfPlay(
                id="self",
                operation="train_self_play",
                behavior="current-self",
                streams=2,
            )
        ],
    )
    admitted = validate_regime(regime)
    assert admitted.deck_collection == collection
    manifest.unlink()
    assert validate_regime(admitted).deck_collection == collection
    wrong = admitted.model_copy(deep=True)
    wrong.agent.semantic_pack = "elemental-world-v2"
    with pytest.raises(ValueError, match="complete pool"):
        validate_regime(wrong)
    target = tmp_path / "policy.pt"
    save_checkpoint(
        _agent(),
        ObservationSpace(),
        target,
        player_configs=Match(collection.match_config()).to_rust(),
        deck_collection=collection,
    )
    assert inspect_checkpoint(target).compatible
    policy, _ = load_checkpoint(str(target))
    assert len(policy.world_binding["setups"]) == 3
    roster = collection.roster()
    validate_agent_setup(policy, [roster[2], roster[0]])
    validate_agent_setup(policy, [roster[1], roster[2]])
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(
            policy, [roster[0], managym.PlayerConfig("other", {"Forest": 10}, {}, PACK)]
        )
    collector = _collector(collection)
    try:
        collector.collect(policy, 1)
        wrong_binding = deepcopy(policy.world_binding)
        wrong_binding["deck_collection"]["digest"] = "0" * 64
        policy.world_binding = wrong_binding
        with pytest.raises(ValueError, match="checkpoint deck collection"):
            collector.collect(policy, 1)
    finally:
        collector.close()


def test_collection_failure_records_actual_sampled_decks(
    collection: DeckCollection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = SeatRoutedCollector(
        ObservationSpace(),
        Match(collection.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=1,
        deck_collection=collection,
        game_errors=policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
    original = collector._sample_learning_rows
    calls = 0

    def sample(
        agent: ActorCritic,
        rows: np.ndarray,
        acting: np.ndarray,
        inference: PolicyInference,
    ) -> np.ndarray:
        nonlocal calls
        actions = original(agent, rows, acting, inference)
        calls += 1
        if calls == 2:
            actions[0] = -1
        return actions

    monkeypatch.setattr(collector, "_sample_learning_rows", sample)
    try:
        collector.collect(_agent(), 1)
        attempts = [
            GameAttempt.model_validate(json.loads(line)["attempt"])
            for line in ledger.path.read_text().splitlines()
        ]
        failed = next(row for row in attempts if row.outcome == "engine_error")
        assert replay_attempt(failed)["at_retained_failure"]
        for attempt in attempts:
            probe = managym.VectorEnv(num_envs=1, seed=attempt.seed)
            probe.set_deck_collection(collection.roster())
            probe.reset_all(Match(collection.match_config()).to_rust())
            a, b = probe.current_matchup_indices()[0]
            assert attempt.matchup == collection.match_config(a, b)
            assert attempt.collection_digest == collection.digest
        scope = CaptureScope(tmp_path / "failures")
        with capture_scope(scope), pytest.raises(Exception):
            collector._env.step_into_buffers([-1, -1])
        evidence = FailureEvidence.model_validate_json(scope.paths[-1].read_bytes())
        assert evidence.identities["deck_collection"]["digest"] == collection.digest
        for game, pair in zip(
            evidence.games, collector._env.current_matchup_indices(), strict=True
        ):
            assert [player.name for player in game.players] == [
                collection.decks[index].deck_id for index in pair
            ]
    finally:
        collector.close()
