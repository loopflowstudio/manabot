"""Deck sources through the ordinary native collector, and world-only checkpoints.

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
from manabot.training.decks import UniformDecks, pair_match
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

WORLD = managym.WORLD_SETUP
ROSTER_PACK = "elemental-world-v2"


def _uniform(*names: str) -> UniformDecks:
    """A source an experiment builds directly: one tiny list per basic land."""
    return UniformDecks(
        decks=tuple(
            Deck({name: 10}, {"Forest": 1})
            for name in names or ("Plains", "Mountain", "Island")
        )
    )


@pytest.fixture
def decks() -> UniformDecks:
    return _uniform()


def _agent() -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=WORLD),
    )


def _collector(
    decks: UniformDecks, recovery: Literal["exact", "current"] = "current"
) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(),
        Match(decks.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=2,
        deck_source=decks,
        recovery_max_microsteps=1000 if recovery == "exact" else None,
        recovery_max_game_microsteps=1000 if recovery == "current" else None,
    )


def _regime(decks: object) -> TrainingRegime:
    return TrainingRegime(
        id="decks-proof",
        world=managym.WORLD_VERSION,
        deck_source=decks,
        agent=ModelConfig(hidden_dim=8, num_attention_heads=2, world=WORLD),
        stages=[
            TrainSelfPlay(
                id="self",
                operation="train_self_play",
                behavior="current-self",
                streams=2,
            )
        ],
    )


def test_experiment_builds_a_source_and_checks_its_decks_in_one_batch(
    tmp_path: Path,
) -> None:
    # Experiments build sources however they like, e.g. from decklist files.
    paths = []
    for name in ("Plains", "Mountain"):
        path = tmp_path / f"{name}.txt"
        path.write_bytes(f"\ufeff10 {name}\r\nSideboard\r\n1 Forest\r\n".encode())
        paths.append(path)
    imports = managym.admit_decklists(
        [(str(path), Decklist.read(path)) for path in paths]
    )
    source = UniformDecks(decks=tuple(imported.deck for imported in imports))
    assert source == _uniform("Plains", "Mountain")
    with pytest.raises(DeckImportError) as error:
        managym.admit_decklists([("broken", Decklist("0 Plains\n1 Not A Card"))])
    assert "deck.txt:1" in str(error.value) and "deck.txt:2" in str(error.value)
    (tmp_path / "big.txt").write_bytes(b"#" * (MAX_DECKLIST_BYTES + 1))
    with pytest.raises(ValueError, match="64 KiB import limit"):
        Decklist.read(tmp_path / "big.txt")

    # Order is the experiment's; a repeated deck never weights draws.
    assert _uniform("Mountain", "Plains").digest != source.digest
    with pytest.raises(ValueError, match="distinct decks"):
        UniformDecks(decks=(source.decks[0], source.decks[0]))
    # One cheap batch check gives each deck's issues; game creation admits again.
    unsupported = Deck({"Swamp": 9, "Rocky Rebuke": 1})
    checked = managym.check_decks([*source.decks, unsupported])
    assert checked[:2] == [[], []]
    assert any("Rocky Rebuke" in issue.message for issue in checked[2])
    stored = source.model_copy(update={"decks": (*source.decks, unsupported)})
    with pytest.raises(ValueError, match="not admitted"):
        stored.check()


def test_regime_records_its_source_and_rejects_bad_decks_before_training(
    decks: UniformDecks,
) -> None:
    regime = validate_regime(_regime(decks))
    # The source is the regime's only record of its setups; nothing is reread.
    record = regime.model_dump(mode="json")
    assert "match" not in record and regime.match is None
    assert TrainingRegime.model_validate(record) == regime
    assert canonical_sha256(
        TrainingRegime.model_validate(record).model_dump(mode="json")
    ) == canonical_sha256(record)
    assert regime.root_match.world == managym.WORLD_SETUP
    authored = MatchConfig(
        hero="forest",
        villain="swamp",
        hero_deck={"Forest": 40},
        villain_deck={"Swamp": 40},
    )
    with pytest.raises(ValueError, match="remove the authored match"):
        TrainingRegime.model_validate({**record, "match": authored.model_dump()})
    wrong = regime.model_copy(deep=True)
    wrong.agent.world = "elemental-world-v2"
    with pytest.raises(ValueError, match="complete world"):
        validate_regime(wrong)
    bad = _regime(
        decks.model_copy(update={"decks": (*decks.decks, Deck({"Rocky Rebuke": 1}))})
    )
    with pytest.raises(ValueError, match="Rocky Rebuke"):
        validate_regime(bad)


@pytest.mark.parametrize("recovery", ["exact", "current"])
def test_collector_resets_recovers_seats_seeds_and_exposure(
    decks: UniformDecks, recovery: Literal["exact", "current"]
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
        changed = _uniform("Plains", "Mountain", "Island", "Forest")
        fresh = _collector(changed, recovery)
        with pytest.raises(ValueError, match="deck source identity mismatch"):
            fresh.restore(snapshot, lambda: None)
        fresh.close()
    finally:
        first.close()
        restored.close()


def test_checkpoint_binds_world_and_plays_unseen_admitted_decks(
    decks: UniformDecks, tmp_path: Path
) -> None:
    target = tmp_path / "policy.pt"
    save_checkpoint(_agent(), ObservationSpace(), target, player_configs=None)
    # Nothing about training travels with the model.
    raw = torch.load(target, map_location="cpu", weights_only=True)
    binding = raw["world_binding"]
    assert "setups" not in binding and "content_manifest" not in binding
    assert set(binding) == {
        "world",
        "world_digest",
        "policy_input_version",
        "rules",
        "input_schema",
        "learning_schema_sha256",
    }
    assert binding["world"] == managym.WORLD_VERSION
    assert inspect_checkpoint(target).compatible
    policy, _ = load_checkpoint(str(target))

    unseen = Deck({"Swamp": 10, "Forest": 4}).player()
    validate_agent_setup(policy, [unseen, decks.players()[0]])
    unsupported = Deck({"Swamp": 9, "Rocky Rebuke": 1}).player()
    with pytest.raises(ValueError, match="not admitted by"):
        validate_agent_setup(policy, [unseen, unsupported])
    other_pack = managym.PlayerConfig("other", {"Swamp": 10}, {}, None)
    with pytest.raises(ValueError, match="not submitted to the world"):
        validate_agent_setup(policy, [unseen, other_pack])

    for key, change in (
        ("world_digest", "0" * 64),
        ("rules", {"decision_version": -1}),
        ("policy_input_version", -1),
    ):
        changed = deepcopy(raw)
        if not isinstance(change, dict):
            changed["world_binding"][key] = change
        else:
            changed["world_binding"][key] = {**binding[key], **change}
        assert checkpoint_incompatibility(changed) is not None
        path = tmp_path / f"changed-{key}.pt"
        torch.save(changed, path)
        with pytest.raises(ValueError, match="binding"):
            load_checkpoint(str(path))

    # Producer admission for a deck run checks the world, not one pair.
    regime = _regime(decks)
    _admit_setups(policy, regime)
    with pytest.raises(ValueError, match="bound to a world"):
        _admit_setups(_agent(), regime)
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
    later = _uniform("Swamp", "Forest")
    collector = _collector(later)
    try:
        collector.collect(policy, 1)
        played = {deck for pair in collector._self_play_decks for deck in pair}
        assert played <= {deck.identity for deck in later.decks}
    finally:
        collector.close()


def test_later_roots_take_new_decks_and_reject_invalid_ones_before_play(
    decks: UniformDecks,
) -> None:
    collector = _collector(decks)
    env = collector._env
    try:
        generated = Deck({"Swamp": 10, "Rocky Rebuke": 1}).player()
        before = env.current_player_configs()
        with pytest.raises(managym.AgentError, match="Rocky Rebuke"):
            env.set_training_decks([decks.players()[0], generated], "generated")
        env.reset_seeds_into_buffers(env.current_game_seeds())
        assert [
            [player.decklist for player in pair]
            for pair in env.current_player_configs()
        ] == [[player.decklist for player in pair] for pair in before]

        replacement = [
            Deck({"Swamp": 10}).player(),
            Deck({"Forest": 9, "Swamp": 1}).player(),
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
    decks: UniformDecks, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        deck_source=decks,
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
        assert evidence.identities["deck_source"]["digest"] == decks.digest
        for game, players in zip(
            evidence.games, collector._env.current_player_configs(), strict=True
        ):
            assert [player.name for player in game.players] == [
                player.name for player in players
            ]
    finally:
        collector.close()


def test_deck_run_identities_name_the_source_not_a_pair(
    decks: UniformDecks,
) -> None:
    space = ObservationSpace()
    regime = _regime(decks)
    identities = _runtime_identities(7, regime, space)
    assert "matchup" not in identities and "matchup_sha256" not in identities
    assert identities["deck_source_sha256"] == decks.digest
    assert "content_pool" not in identities
    opened = runtime_fingerprints(
        7, match_config=regime.root_match, observation_space=space
    )
    for name in ("content_manifest_sha256", "observation_abi_sha256"):
        assert identities[name] == opened[name]
    with pytest.raises(ValueError, match="monitoring requires an authored matchup"):
        authored_matchup_sha256(identities, "monitoring")

    changed = _regime(_uniform("Plains", "Swamp"))
    with pytest.raises(ValueError, match="recovery runtime/source identity mismatch"):
        admit_recovery_identities(
            identities, _runtime_identities(7, changed, space), None
        )

    authored = TrainingRegime.model_validate(
        {
            **regime.model_dump(mode="json", exclude={"deck_source"}),
            "match": decks.match_config().model_dump(),
        }
    )
    recorded = _runtime_identities(7, authored, space)
    fingerprints = runtime_fingerprints(
        7, match_config=authored.match, observation_space=space
    )
    assert {k: recorded[k] for k in fingerprints} == fingerprints
    assert "deck_source_sha256" not in recorded
    assert authored_matchup_sha256(recorded, "monitoring") == recorded["matchup_sha256"]
