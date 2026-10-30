"""Training decks through the ordinary native collector, and world-only checkpoints.

Tiny admitted development lists bound the proof: collection/inference only,
without optimization, a data download, or a training campaign.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env.failures import CaptureScope, FailureEvidence, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import authored_matchup_sha256, runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.catalog import checkpoint_incompatibility, inspect_checkpoint
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.inference import PolicyInference
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.training.decks import TrainingDecks, pair_match
from manabot.training.execution import _runtime_identities, validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.state.admission import _admit_setups
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorPolicy,
    replay_attempt,
)
from manabot.training.state.recovery import admit_recovery_identities
import managym
from managym import MAX_DECKLIST_BYTES, Deck, DeckImportError, Decklist

PACK = "tla-pool-v1"
ROSTER_PACK = "elemental-world-v2"


def _decklists(*names: str) -> list[Decklist]:
    return [
        Decklist(f"10 {name}\nSideboard\n1 Forest\n", f"{name}.txt")
        for name in names or ("Plains", "Mountain", "Island")
    ]


@pytest.fixture
def decks() -> TrainingDecks:
    return TrainingDecks.from_decklists(_decklists(), PACK)


def _agent() -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
    )


def _collector(
    decks: TrainingDecks, recovery: Literal["exact", "current"] = "current"
) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(),
        Match(decks.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=2,
        decks=decks,
        recovery_max_microsteps=1000 if recovery == "exact" else None,
        recovery_max_game_microsteps=1000 if recovery == "current" else None,
    )


def _regime(decks: object) -> TrainingRegime:
    return TrainingRegime(
        id="decks-proof",
        world=managym.WORLD_VERSION,
        decks=decks,
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


def test_fixed_list_is_checked_upfront_and_keeps_every_source() -> None:
    decklists = _decklists() + [
        Decklist(
            "# another record\r\n5 plains\r\n5 Plains\r\nSideboard\r\n1 Forest\r\n",
            "duplicate.txt",
        )
    ]
    decks = TrainingDecks.from_decklists(decklists, PACK)
    assert len(decks.decks) == 3 and len(decks.sources) == 4
    assert decklists[-1].source in {source.source for source in decks.sources}
    assert TrainingDecks.from_decklists(list(reversed(decklists)), PACK) == decks
    with pytest.raises(DeckImportError) as error:
        TrainingDecks.from_decklists(
            decklists + [Decklist("0 Plains\n1 Not A Card", "broken.txt")], PACK
        )
    assert "broken.txt:1" in str(error.value) and "broken.txt:2" in str(error.value)
    with pytest.raises(DeckImportError) as unsupported:
        TrainingDecks.from_decklists(
            decklists + [Decklist("1 Accumulate Wisdom", "unsupported.txt")], PACK
        )
    assert any(
        issue.code == "unsupported"
        for doc in unsupported.value.decklists.values()
        for issue in doc.issues
    )
    with pytest.raises(ValueError, match="at least two distinct"):
        TrainingDecks.from_decklists([decklists[0], decklists[-1]], PACK)
    value = decks.model_dump(mode="json")
    value["decks"].reverse()
    with pytest.raises(ValueError, match="sorted distinct"):
        TrainingDecks.model_validate(value)
    # Stored decks are re-admitted by the world itself, not trusted as members.
    stored = decks.model_copy(
        update={"decks": (*decks.decks, Deck({"Rocky Rebuke": 1}))}
    )
    with pytest.raises(ValueError, match="not admitted"):
        stored.check()


def test_regime_embeds_files_and_reports_bad_input_before_training(
    tmp_path: Path,
) -> None:
    paths = []
    for decklist in _decklists():
        path = tmp_path / decklist.filename
        path.write_bytes(("﻿" + decklist.source.replace("\n", "\r\n")).encode("utf-8"))
        paths.append(str(path))
    regime = validate_regime(_regime({"pool": PACK, "files": paths}))
    assert regime.decks is not None and len(regime.decks.decks) == 3
    for source in regime.decks.sources:
        assert source.source.encode("utf-8") == Path(source.filename).read_bytes()
    for path in paths:
        Path(path).unlink()
    assert validate_regime(regime).decks == regime.decks
    # The decks are the regime's only record of its setups.
    record = regime.model_dump(mode="json")
    assert "match" not in record and regime.match is None
    assert TrainingRegime.model_validate(record) == regime
    assert canonical_sha256(
        TrainingRegime.model_validate(record).model_dump(mode="json")
    ) == canonical_sha256(record)
    assert regime.root_match.content_pack == PACK
    authored = MatchConfig(
        hero="forest",
        villain="swamp",
        hero_deck={"Forest": 40},
        villain_deck={"Swamp": 40},
        content_pack=PACK,
    )
    with pytest.raises(ValueError, match="remove the authored match"):
        TrainingRegime.model_validate({**record, "match": authored.model_dump()})
    wrong = regime.model_copy(deep=True)
    wrong.agent.semantic_pack = "elemental-world-v2"
    with pytest.raises(ValueError, match="complete pool"):
        validate_regime(wrong)
    Path(paths[0]).write_text("10 Plains")
    Path(paths[1]).write_text("1 Unknown Card")
    with pytest.raises(ValueError, match="Mountain.txt:1: Unknown card name"):
        _regime({"pool": PACK, "files": paths[:2]})
    Path(paths[1]).write_bytes(b"#" * (MAX_DECKLIST_BYTES + 1))
    with pytest.raises(ValueError, match="64 KiB import limit"):
        _regime({"pool": PACK, "files": paths[:2]})


@pytest.mark.parametrize("recovery", ["exact", "current"])
def test_collector_resets_recovers_seats_seeds_and_exposure(
    decks: TrainingDecks, recovery: Literal["exact", "current"]
) -> None:
    agent = _agent()
    first = _collector(decks, recovery)
    restored = _collector(decks, recovery)
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
        changed = TrainingDecks.from_decklists(
            _decklists("Plains", "Mountain", "Island", "Forest"), PACK
        )
        fresh = _collector(changed, recovery)
        with pytest.raises(ValueError, match="training decks identity mismatch"):
            fresh.restore(snapshot, lambda: None)
        fresh.close()
    finally:
        first.close()
        restored.close()


def test_checkpoint_binds_world_and_plays_unseen_admitted_decks(
    decks: TrainingDecks, tmp_path: Path
) -> None:
    target = tmp_path / "policy.pt"
    save_checkpoint(_agent(), ObservationSpace(), target, content_pool=PACK)
    # Nothing about training travels with the model.
    raw = torch.load(target, map_location="cpu", weights_only=True)
    binding = raw["world_binding"]
    assert "setups" not in binding and "content_manifest" not in binding
    assert binding["content_pool"] == managym.content_pool(PACK)
    assert inspect_checkpoint(target).compatible
    policy, _ = load_checkpoint(str(target))

    unseen = Deck({"Swamp": 10, "Forest": 4}).player(PACK)
    validate_agent_setup(policy, [unseen, decks.players()[0]])
    unsupported = Deck({"Swamp": 9, "Rocky Rebuke": 1}).player(PACK)
    with pytest.raises(ValueError, match="not admitted by"):
        validate_agent_setup(policy, [unseen, unsupported])
    other_pack = managym.PlayerConfig("other", {"Swamp": 10}, {}, None)
    with pytest.raises(ValueError, match="content pool"):
        validate_agent_setup(policy, [unseen, other_pack])

    for key, change in (
        ("content_pool", {"content_digest": "0" * 64}),
        ("rules", {"decision_version": -1}),
        ("policy_input_version", None),
    ):
        changed = deepcopy(raw)
        if change is None:
            changed["world_binding"][key] = -1
        else:
            changed["world_binding"][key] = {**binding[key], **change}
        assert checkpoint_incompatibility(changed) is not None
        path = tmp_path / f"changed-{key}.pt"
        torch.save(changed, path)
        with pytest.raises(ValueError, match="binding"):
            load_checkpoint(str(path))

    # Producer admission for a deck run checks the pool, not one pair.
    regime = _regime(decks)
    _admit_setups(policy, regime)
    with pytest.raises(ValueError, match="bound to a world"):
        _admit_setups(_agent(), regime)
    other_pool = deepcopy(policy)
    other_pool.world_binding = {
        **binding,
        "content_pool": {**binding["content_pool"], "pack_key": "other-pool"},
    }
    with pytest.raises(ValueError, match="outside the checkpoint's content pool"):
        _admit_setups(other_pool, regime)
    # A setup-bound checkpoint plays only its roster, under the shared rule.
    authored = deepcopy(policy)
    authored.world_binding = checkpoint_world(
        [
            managym.authored_deck_setup(ROSTER_PACK, key)
            for key in managym.authored_deck_keys(ROSTER_PACK)[:2]
        ],
        ObservationSpace(),
    )
    with pytest.raises(ValueError, match="outside checkpoint's exact setup roster"):
        _admit_setups(authored, regime)

    # The same model collects from a later supply it never trained on.
    later = TrainingDecks.from_decklists(_decklists("Swamp", "Forest"), PACK)
    collector = _collector(later)
    try:
        collector.collect(policy, 1)
        played = {deck for pair in collector._self_play_decks for deck in pair}
        assert played <= {deck.identity for deck in later.decks}
    finally:
        collector.close()


def test_later_roots_take_new_decks_and_reject_invalid_ones_before_play(
    decks: TrainingDecks,
) -> None:
    collector = _collector(decks)
    env = collector._env
    try:
        generated = Deck({"Swamp": 10, "Rocky Rebuke": 1}).player(PACK)
        before = env.current_player_configs()
        with pytest.raises(managym.AgentError, match="Rocky Rebuke"):
            env.set_training_decks([decks.players()[0], generated], "generated")
        env.reset_seeds_into_buffers(env.current_game_seeds())
        assert [
            [player.decklist for player in pair]
            for pair in env.current_player_configs()
        ] == [[player.decklist for player in pair] for pair in before]

        replacement = [
            Deck({"Swamp": 10}).player(PACK),
            Deck({"Forest": 9, "Swamp": 1}).player(PACK),
        ]
        env.set_training_decks(replacement, "replacement")
        assert env.current_player_configs()[0][0].decklist == before[0][0].decklist
        env.reset_seeds_into_buffers(env.current_game_seeds())
        # Retained replay inputs name the decks each later root actually used.
        for game in env.games:
            assert {player.name for player in game.players} == {
                player.name for player in replacement
            }
    finally:
        collector.close()


def test_failure_records_actual_sampled_decks(
    decks: TrainingDecks, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = SeatRoutedCollector(
        ObservationSpace(),
        Match(decks.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=1,
        decks=decks,
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
            probe.set_training_decks(decks.players())
            probe.reset_all(Match(decks.match_config()).to_rust())
            assert attempt.matchup == pair_match(*probe.current_player_configs()[0])
            assert attempt.decks_digest == decks.digest
        scope = CaptureScope(tmp_path / "failures")
        with capture_scope(scope), pytest.raises(Exception):
            collector._env.step_into_buffers([-1, -1])
        evidence = FailureEvidence.model_validate_json(scope.paths[-1].read_bytes())
        assert evidence.identities["training_decks"]["digest"] == decks.digest
        for game, players in zip(
            evidence.games, collector._env.current_player_configs(), strict=True
        ):
            assert [player.name for player in game.players] == [
                player.name for player in players
            ]
    finally:
        collector.close()


def test_deck_run_identities_name_the_supply_not_a_pair(
    decks: TrainingDecks,
) -> None:
    space = ObservationSpace()
    regime = _regime(decks)
    identities = _runtime_identities(7, regime, space)
    assert "matchup" not in identities and "matchup_sha256" not in identities
    assert identities["training_decks_sha256"] == decks.digest
    assert identities["content_pool"] == PACK
    opened = runtime_fingerprints(
        7, match_config=regime.root_match, observation_space=space
    )
    for name in ("content_manifest_sha256", "observation_abi_sha256"):
        assert identities[name] == opened[name]
    with pytest.raises(ValueError, match="monitoring requires an authored matchup"):
        authored_matchup_sha256(identities, "monitoring")

    changed = _regime(TrainingDecks.from_decklists(_decklists("Plains", "Swamp"), PACK))
    with pytest.raises(ValueError, match="recovery runtime/source identity mismatch"):
        admit_recovery_identities(
            identities, _runtime_identities(7, changed, space), None
        )

    authored = TrainingRegime.model_validate(
        {
            **regime.model_dump(mode="json", exclude={"decks"}),
            "match": decks.match_config().model_dump(),
        }
    )
    recorded = _runtime_identities(7, authored, space)
    fingerprints = runtime_fingerprints(
        7, match_config=authored.match, observation_space=space
    )
    assert {k: recorded[k] for k in fingerprints} == fingerprints
    assert "training_decks_sha256" not in recorded
    assert authored_matchup_sha256(recorded, "monitoring") == recorded["matchup_sha256"]
