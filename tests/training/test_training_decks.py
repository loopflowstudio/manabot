"""Deck sources through the ordinary native collector, and world-only checkpoints.

Tiny admitted development lists bound the proof: collection/inference only,
without optimization, a data download, or a training campaign.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import (
    BACKEND as TORCH_BACKEND,
    load_checkpoint,
    save_checkpoint,
)
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.failures import CaptureScope, FailureEvidence, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import authored_matchup_sha256, runtime_fingerprints
from manabot.model.catalog import checkpoint_incompatibility, inspect_checkpoint
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.sim.recording import record_games
from manabot.sim.replay import read_trace, replay_games
from manabot.sim.rollout import allocate_buffers
from manabot.training.decks import UniformDecks
from manabot.training.execution import _runtime_identities, validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.rollout.numerics import CollectionScope
from manabot.training.state.admission import admit_setups
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorPolicy,
    replay_attempt,
)
from manabot.training.state.recovery import admit_recovery_identities
import managym
from managym import MAX_DECKLIST_BYTES, Deck, DeckImportError, Decklist
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint

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
        numerics=TorchCollectionSession(device="cpu", seed=17402, precision="float32"),
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
    repaired = Deck(
        {
            "Swamp": 9,
            "Accumulate Wisdom": 1,
            "Cat-Gator": 1,
            "Lightning Strike": 1,
            "Suki, Kyoshi Warrior": 1,
            "Water Tribe Rallier": 1,
        }
    )
    admitted = UniformDecks(decks=(*source.decks, repaired))
    admitted.check()
    validate_regime(_regime(admitted))
    unsupported = Deck({"Swamp": 9, "Ally": 1})
    checked = managym.check_decks([*admitted.decks, unsupported])
    assert checked[:3] == [[], [], []]
    assert any(
        issue.code == "unsupported" and "Ally" in issue.message for issue in checked[3]
    )
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
    bad = _regime(decks.model_copy(update={"decks": (*decks.decks, Deck({"Ally": 1}))}))
    with pytest.raises(ValueError, match="Ally"):
        validate_regime(bad)


@pytest.mark.parametrize("disclosure", ["closed", "mixed"])
@pytest.mark.parametrize("sampling", ["uniform-distinct-v1", "uniform-independent-v1"])
@pytest.mark.parametrize("recovery", ["exact", "current"])
def test_collector_resets_recovers_seats_seeds_and_exposure(
    decks: UniformDecks,
    recovery: Literal["exact", "current"],
    sampling: Literal["uniform-distinct-v1", "uniform-independent-v1"],
    disclosure: Literal["closed", "mixed"],
) -> None:
    decks = decks.model_copy(update={"sampling": sampling, "decklists": disclosure})
    agent = _agent()
    first = _collector(decks, recovery)
    restored = _collector(decks, recovery)
    try:
        assert all(
            m.config.decklists in ("closed", "open") for m in first.stream_matches
        )
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
        mirrors = [key.split("/")[0] == key.split("/")[1] for key in exposure]
        assert (
            any(mirrors) if sampling == "uniform-independent-v1" else not any(mirrors)
        )
        restored.restore(first.snapshot(), lambda: None)
        left, right = first.collect(agent, 1), restored.collect(agent, 1)
        expected, actual = left.ordered_obs(), right.ordered_obs()
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        np.testing.assert_array_equal(left.actions, right.actions)
        assert first._self_play_decks == restored._self_play_decks
        assert first._env.current_game_seeds() == restored._env.current_game_seeds()
        assert first.stats.exposure == restored.stats.exposure
        assert [m.config.decklists for m in restored.stream_matches] == [
            pair.decklists for pair in restored._env.current_player_configs()
        ]
        if disclosure == "closed":
            assert not right.ordered_obs()["decklists"][..., 1, :, :].any()
        else:
            assert {
                bool(value)
                for value in right.ordered_obs()["agent_player"][..., -1].flat
            } == {False, True}
        snapshot = first.snapshot()
        changed = _uniform("Plains", "Mountain", "Island", "Forest")
        fresh = _collector(changed, recovery)
        with pytest.raises(ValueError, match="deck source identity mismatch"):
            fresh.restore(snapshot, lambda: None)
        fresh.close()
    finally:
        first.close()
        restored.close()


@pytest.mark.parametrize("workers", [2, 4])
def test_sampled_decks_agree_across_producer_counts(
    decks: UniformDecks, workers: int
) -> None:
    agent = _agent()
    collectors = [
        SeatRoutedCollector(
            ObservationSpace(),
            Match(decks.match_config()),
            Reward(RewardConfig()),
            num_envs=4,
            seed=17402,
            opponent_mode="self",
            games_per_stream=2,
            deck_source=decks,
            workers=count,
            numerics=TorchCollectionSession(
                device="cpu", seed=17402, precision="float32"
            ),
        )
        for count in (1, workers)
    ]
    one, many = collectors
    try:
        left, right = one.collect(agent, 1), many.collect(agent, 1)
        expected, actual = left.ordered_obs(), right.ordered_obs()
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        np.testing.assert_array_equal(left.actions, right.actions)
        assert one.stats.games == many.stats.games == 8
        assert one._env.current_game_seeds() == many._env.current_game_seeds()
        assert [
            [player.decklist for player in pair]
            for pair in one._env.current_player_configs()
        ] == [
            [player.decklist for player in pair]
            for pair in many._env.current_player_configs()
        ]
        assert one._self_play_decks == many._self_play_decks
        assert one.stats.exposure == many.stats.exposure
        # A rejected replacement leaves every shard's decks in place.
        before = many._env.current_player_configs()
        with pytest.raises(managym.AgentError, match="Ally"):
            many._env.set_training_decks(
                [decks.players()[0], Deck({"Swamp": 10, "Ally": 1}).player()], "bad"
            )
        many._env.reset_seeds_into_buffers(many._env.current_game_seeds())
        assert [
            [player.decklist for player in pair]
            for pair in many._env.current_player_configs()
        ] == [[player.decklist for player in pair] for pair in before]
    finally:
        one.close()
        many.close()


def test_checkpoint_binds_world_and_plays_unseen_admitted_decks(
    decks: UniformDecks, tmp_path: Path
) -> None:
    target = tmp_path / "policy.pt"
    save_checkpoint(_agent(), ObservationSpace(), target, player_configs=None)
    # Nothing about training travels with the model.
    raw = read_checkpoint(target, map_location="cpu", weights_only=True)
    binding = raw["world_binding"]
    assert "setups" not in binding and "content_manifest" not in binding
    assert set(binding) == {
        "world",
        "world_digest",
        "policy_input_version",
        "rules",
        "input_schema",
        "learning_schema_sha256",
        "naming_representation",
        "naming_branching",
        "naming_vocabulary",
    }
    assert binding["world"] == managym.WORLD_VERSION
    assert inspect_checkpoint(target, backend=TORCH_BACKEND).compatible
    policy, _ = load_checkpoint(str(target))

    unseen = Deck({"Swamp": 10, "Forest": 4, "Accumulate Wisdom": 1}).player()
    validate_agent_setup(policy, [unseen, decks.players()[0]])
    unsupported = Deck({"Swamp": 9, "Ally": 1}).player()
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
        write_checkpoint(changed, path)
        with pytest.raises(ValueError, match="binding"):
            load_checkpoint(str(path))

    # Producer admission for a deck run checks the world, not one pair.
    regime = _regime(decks)
    admit_setups(policy, regime)
    with pytest.raises(ValueError, match="bound to a world"):
        admit_setups(_agent(), regime)
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
        admit_setups(authored, regime)

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
        generated = Deck({"Swamp": 10, "Ally": 1}).player()
        before = env.current_player_configs()
        with pytest.raises(managym.AgentError, match="Ally"):
            env.set_training_decks([decks.players()[0], generated], "generated")
        env.reset_seeds_into_buffers(env.current_game_seeds())
        assert [
            [player.decklist for player in pair]
            for pair in env.current_player_configs()
        ] == [[player.decklist for player in pair] for pair in before]

        replacement = [
            Deck({"Swamp": 10, "Accumulate Wisdom": 1}).player(),
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


@pytest.mark.parametrize(
    ("sampled", "disclosure"), [(False, "closed"), (True, "closed"), (True, "mixed")]
)
def test_failure_records_actual_sampled_decks(
    decks: UniformDecks,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sampled: bool,
    disclosure: Literal["closed", "mixed"],
) -> None:
    decks = decks.model_copy(update={"decklists": disclosure})
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
        deck_source=decks if sampled else None,
        game_errors=policy,
        record_attempt=lambda row: ledger.record(row, {}),
        numerics=TorchCollectionSession(device="cpu", seed=17402, precision="float32"),
    )
    original = collector._sample_learning_rows
    calls = 0

    def sample(
        rows: np.ndarray,
        acting: np.ndarray,
        inference: CollectionScope,
    ) -> np.ndarray:
        nonlocal calls
        actions = original(rows, acting, inference)
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
            if sampled:
                probe.set_training_decks(decks.players())
            root = Match(decks.match_config()).to_rust()
            root = managym.GameSetup(root, decklists=disclosure)
            probe.reset_all(root)
            assert attempt.matchup == MatchConfig.from_players(
                *probe.current_player_configs()[0],
                decklists=probe.current_player_configs()[0].decklists,
            )
            assert attempt.decks_digest == (decks.digest if sampled else None)
        scope = CaptureScope(tmp_path / "failures")
        with capture_scope(scope), pytest.raises(Exception):
            collector._env.step_into_buffers([-1, -1])
        evidence = FailureEvidence.model_validate_json(scope.paths[-1].read_bytes())
        if sampled:
            assert evidence.identities["deck_source"]["digest"] == decks.digest
        for game, players in zip(
            evidence.games, collector._env.current_player_configs(), strict=True
        ):
            assert {p.disclosure for p in game.players} == {players.decklists}
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
    identities = _runtime_identities(7, regime, space, TORCH_BACKEND)
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
            identities, _runtime_identities(7, changed, space, TORCH_BACKEND), None
        )

    authored = TrainingRegime.model_validate(
        {
            **regime.model_dump(mode="json", exclude={"deck_source"}),
            "match": decks.match_config().model_dump(),
        }
    )
    recorded = _runtime_identities(7, authored, space, TORCH_BACKEND)
    fingerprints = runtime_fingerprints(
        7, match_config=authored.match, observation_space=space
    )
    assert {k: recorded[k] for k in fingerprints} == fingerprints
    assert "deck_source_sha256" not in recorded
    assert authored_matchup_sha256(recorded, "monitoring") == recorded["matchup_sha256"]


def test_independent_population_retains_event_ids_and_mirrors() -> None:
    deck = Deck({"Plains": 10})
    source = UniformDecks(
        decks=(deck, deck),
        sampling="uniform-independent-v1",
        entry_ids=("event-a", "event-b"),
    )
    assert [player.name for player in source.players()] == ["event-a", "event-b"]
    source.check()
    collector = _collector(source)
    try:
        batch = collector.collect(_agent(), 1)
        assert batch is not None and collector.stats.games == 4
        for pair in collector._env.current_player_configs():
            assert all(player.name in source.entry_ids for player in pair)
            assert pair[0].deck == pair[1].deck == deck
        identity = collector._env.identities["deck_source"]
        assert identity["sampling_policy"] == "uniform-independent-v1"
        assert identity["deck_ids"] == ["event-a", "event-b"]
    finally:
        collector.close()
    with pytest.raises(ValueError, match="entry IDs"):
        UniformDecks(
            decks=(deck,), sampling="uniform-independent-v1", entry_ids=("same", "same")
        )


@pytest.mark.parametrize("mode", ["closed", "mixed"])
def test_source_disclosure_preserves_open_identity_and_binds_recovery(
    mode: Literal["closed", "mixed"],
) -> None:
    source = UniformDecks(
        decks=tuple(Deck({name: 10}) for name in ("Plains", "Mountain", "Island"))
    )
    assert "decklists" not in source.model_dump(mode="json")
    assert (
        source.digest
        == "0f3c4ed44819bab893c6e693d8a2ed59dbaaedb1fd87d2ec1af604eedd544355"
    )
    configured = source.model_copy(update={"decklists": mode})
    assert configured.digest != source.digest
    assert configured.decklists == mode
    if mode == "closed":
        assert configured.match_config().decklists == "closed"
    record = _regime(configured).model_dump(mode="json")
    assert record["deck_source"]["decklists"] == mode
    assert TrainingRegime.model_validate(record).deck_source == configured
    space = ObservationSpace()
    with pytest.raises(ValueError, match="recovery runtime/source identity mismatch"):
        admit_recovery_identities(
            _runtime_identities(7, _regime(source), space, TORCH_BACKEND),
            _runtime_identities(7, _regime(configured), space, TORCH_BACKEND),
            None,
        )


def test_closed_fixed_roots_survive_auto_reset_and_restore(decks: UniformDecks) -> None:
    match = Match(decks.model_copy(update={"decklists": "closed"}).match_config())

    def collector() -> SeatRoutedCollector:
        return SeatRoutedCollector(
            ObservationSpace(),
            match,
            Reward(RewardConfig()),
            num_envs=2,
            seed=17402,
            opponent_mode="self",
            games_per_stream=2,
            recovery_max_game_microsteps=1000,
            numerics=TorchCollectionSession(
                device="cpu", seed=17402, precision="float32"
            ),
        )

    first, restored = collector(), collector()
    try:
        initial = first._env.current_game_seeds()
        first.collect(_agent(), 1)
        assert first.stats.games == 4
        assert first._env.current_game_seeds() != initial
        restored.restore(first.snapshot(), lambda: None)
        for value in (first, restored):
            assert all(m.config.decklists == "closed" for m in value.stream_matches)
            assert all(
                pair.decklists == "closed"
                for pair in value._env.current_player_configs()
            )
        batch = restored.collect(_agent(), 1)
        assert not batch.ordered_obs()["decklists"][..., 1, :, :].any()
    finally:
        first.close()
        restored.close()


def test_collector_rejects_match_disclosure_its_source_would_ignore(
    decks: UniformDecks,
) -> None:
    closed = decks.model_copy(update={"decklists": "closed"})
    with pytest.raises(ValueError, match="decklists='closed'.*decklists='open'"):
        SeatRoutedCollector(
            ObservationSpace(),
            Match(closed.match_config()),
            Reward(RewardConfig()),
            num_envs=2,
            seed=17402,
            opponent_mode="self",
            games_per_stream=2,
            deck_source=decks,
            numerics=TorchCollectionSession(
                device="cpu", seed=17402, precision="float32"
            ),
        )


@pytest.mark.parametrize("viewer", [0, 1])
def test_mixed_source_closed_streams_match_all_closed_inputs(
    decks: UniformDecks, viewer: int
) -> None:
    agent = _agent()
    space = ObservationSpace()
    mixed, closed = (
        managym.VectorEnv(num_envs=8, seed=0),
        managym.VectorEnv(num_envs=8, seed=0),
    )
    for vector, mode in ((mixed, "mixed"), (closed, "closed")):
        vector.set_training_decks(decks.players())
        root = Match(decks.match_config()).to_rust()
        root = managym.GameSetup(root, decklists=mode)
        vector.reset_all(root)
    actual = mixed.current_player_configs()
    mask = np.array([setup.decklists == "closed" for setup in actual])
    assert mask.any() and not mask.all()
    buffers = [allocate_buffers(space, 8) for _ in range(2)]
    for vector, output in zip((mixed, closed), buffers, strict=True):
        vector.write_observations_for_players([viewer] * 8, output)
    for key, value in buffers[0].items():
        np.testing.assert_array_equal(value[mask], buffers[1][key][mask], err_msg=key)
    prepared = [
        agent.prepare({key: value[mask] for key, value in output.items()})
        for output in buffers
    ]
    np.testing.assert_equal(asdict(prepared[0]), asdict(prepared[1]))


def test_mixed_training_games_export_the_resolved_disclosure(
    decks: UniformDecks, tmp_path: Path
) -> None:
    source = decks.model_copy(update={"decklists": "mixed"})
    with record_games(tmp_path / "receipts") as games:
        collector = _collector(source)
        try:
            collector.collect(_agent(), 1)
        finally:
            collector.close()
    settings: set[str] = set()
    for index, game in enumerate(games.games()):
        retained = json.loads(game.path.read_bytes())["game"]["trace"]["player_configs"]
        disclosure = retained[0].get("disclosure", "open")
        assert {seat.get("disclosure", "open") for seat in retained} == {disclosure}
        path = game.export(tmp_path / f"{index}.game.jsonl.gz")
        exported = read_trace(path)[0]
        assert exported["match_hypers"].get("decklists", "open") == disclosure
        assert replay_games([exported]).passed
        settings.add(disclosure)
    assert settings == {"open", "closed"}
