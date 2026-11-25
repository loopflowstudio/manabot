"""Authored paired deals preserve disclosure and distinct blanked checkpoint players."""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

from etude.checkpoints import CheckpointRef
from etude.eval.query import compare
from etude.eval.records import (
    Benchmark,
    CheckpointPlayer,
    Code,
    Environment,
    Match,
    Seating,
    checkpoint_player,
)
from etude.eval.tournament import evaluating
from etude.records import ArtifactRef
import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace, encode_observation
from manabot.eval import pairing
from manabot.eval.agent_loader import load
from manabot.eval.agents import blank_decision_memory, build
from manabot.eval.card_query import CardSelection, audit, card_data, card_db
from manabot.eval.evaluation import EvaluationRequest
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup
import managym


@pytest.fixture
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_checkpoint(
    directory: Path,
    seed: int,
    space: ObservationSpace,
    *,
    memory: Literal["none", "gru"] = "none",
    setup: managym.GameSetup | None = None,
) -> CheckpointRef:
    torch.manual_seed(seed)
    model = ActorCritic(
        space,
        ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=8,
            num_attention_heads=2,
            attention_on=True,
            attention_layers=1,
            policy_film=True,
            value_aggregation="masked_mean",
            decision_memory=memory,
        ),
    )
    path = directory / f"{seed}.safetensors"
    save_checkpoint(model, space, path, player_configs=setup)
    return CheckpointRef(
        experiment="fixture",
        job="pairing",
        seed=seed,
        name=f"fixture-{seed}",
        at={"updates": 0},
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )


@pytest.mark.parametrize(
    "disclosure,blank,world",
    [
        ("open", False, None),
        ("closed", False, None),
        ("open", True, None),
        ("closed", False, managym.WORLD_SETUP),
    ],
)
def test_pairing_plays_native_disclosure_and_blanked_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_threads: None,
    disclosure: Literal["open", "closed"],
    blank: bool,
    world: Literal["world"] | None,
) -> None:
    """A world matchup admits world-bound checkpoints, as trained recipes write them."""
    # Vanilla decks exercise disclosure, checkpoint identity and all four legs
    # with bounded decision histories for the recurrent test policies.
    matchup = AuthoredMatchup(
        pack_key="mtg122-complexity-v1",
        role_a="l1_swift",
        role_b="l1_heavy",
        decklists=disclosure,
        world=world,
    )
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=512,
            max_decision_reference_details=16384,
            max_decision_references=256,
            max_decision_members=8192,
        )
    )
    setup = None if world else EnvMatch(matchup.match()).to_rust()
    candidate = checkpoint_player(
        fixture_checkpoint(tmp_path, 19, space, memory="gru", setup=setup)
    )
    if blank:
        opponent = candidate.model_copy(
            update={"config": {"blank_decision_memory": True}}
        )
    else:
        opponent = checkpoint_player(
            fixture_checkpoint(tmp_path, 20, space, memory="gru", setup=setup)
        )
    assert candidate.key != opponent.key
    match = Match(players=(candidate, opponent), games=4)
    # As Etude seats them: a bare forward without config, a built player with one.
    agents = [
        build(player, {player.checkpoint.key: load(player.checkpoint)})
        if player.config
        else load(player.checkpoint)
        for player in (candidate, opponent)
    ]
    evaluate = Mock(wraps=pairing.evaluate)
    monkeypatch.setattr(pairing, "evaluate", evaluate)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    result = pairing.play_pairing(
        match,
        agents,
        matchup=matchup,
        # Deal 1 bounds each recurrent policy's history to under 300 decisions per leg.
        deal=1,
        matchup_mode="cross",
        game_seconds=60,
        arena="fixture-pairing",
    )
    request: EvaluationRequest = evaluate.call_args.args[0]
    assert request.authored_matchup is not None
    env = managym.Env(seed=123)
    env.reset(EnvMatch(request.authored_matchup.match()).to_rust())
    decklists = encode_observation(space, env.observation_for_player(0))["decklists"]
    assert decklists[0].any()
    assert bool(decklists[1].any()) == (disclosure == "open")
    assert request.opponent.player_spec.get("blank_decision_memory", False) == blank
    assert (
        evaluate.call_args.kwargs["agents"]["opponent"].blank_decision_memory == blank
    )
    if blank:
        assert request.candidate.checkpoint_sha256 == request.opponent.checkpoint_sha256
        assert request.candidate.identity_sha256 != request.opponent.identity_sha256

    rows = result.tables["plays"]
    assert "behaviour" not in result.tables
    assert len(rows) == 8
    assert {row["decklists"] for row in rows} == {disclosure}
    assert {row["game"] for row in rows} == set(range(4))
    assert all(row["valid"] for row in rows)
    for game in range(4):
        assert sum(float(row["score"]) for row in rows if row["game"] == game) == 1.0
    assert (tmp_path / "arena" / match.key).is_dir()


def test_pairing_plays_a_checkpoint_against_the_heuristic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_threads: None
) -> None:
    roles = ("ur_lessons", "gw_allies")
    first, second = (
        managym.authored_deck_setup("ur-lessons-vs-gw-allies", role) for role in roles
    )
    matchup = DeckMatchup(
        population_sha256="a" * 64,
        entries=roles,
        lists=(
            managym.Deck(first.decklist, first.sideboard),
            managym.Deck(second.decklist, second.sideboard),
        ),
    )
    assert "decklists" not in matchup.model_dump()
    closed = matchup.model_copy(update={"decklists": "closed"})
    assert closed.content_suite != matchup.content_suite
    matchup = closed
    assert matchup.match().decklists == "closed"
    ref = fixture_checkpoint(tmp_path, 19, ObservationSpace(ObservationConfig()))
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config={"max_actions": 64}
    )
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    benchmark = Benchmark(
        name="pairing",
        environment=Environment(name="fixture", files="a" * 64),
        seatings=(
            Seating(
                name="fixture", players=(checkpoint_player(ref), heuristic), games=1
            ),
        ),
    )
    track = tmp_path / "track"
    with evaluating(benchmark, experiment="pairing", directory=track) as active:
        [match] = active.matches
        result = pairing.play_pairing(
            match,
            [load(ref), build(heuristic, {})],
            matchup=matchup,
            deal=123,
            matchup_mode="cross",
            game_seconds=60,
            arena="fixture-pairing",
        )
        active.store(match, result)
    rows = result.tables["plays"]
    assert len(rows) == 8 and all(row["valid"] for row in rows)
    for game in range(4):
        assert sum(float(row["score"]) for row in rows if row["game"] == game) == 1.0

    retained = tmp_path / "arena" / match.key
    assert not (retained / "cards").exists()
    request = json.loads((retained / "evaluation.json").read_bytes())["request"]
    assert request["opponent"]["player_spec"]["max_actions"] == 64
    native = json.loads((retained / "statistics.json").read_bytes())["tables"]["play"]
    checkpoint = [native[2 * leg + leg % 2] for leg in range(4)]
    behaviour = result.tables["behaviour"]
    assert len(behaviour) == 4
    for leg, (row, expected) in enumerate(zip(behaviour, checkpoint, strict=True)):
        deck = matchup.decks(leg)[leg % 2]
        opponent_deck = matchup.decks(leg)[1 - leg % 2]
        expected = dict(expected)
        expected["chump_attacks"] = expected.pop("attackers_lost_for_nothing")
        expected["chump_blocks"] = expected.pop("blockers_lost_for_nothing")
        assert row == {
            **expected,
            "seat": 0,
            "game_seat": leg % 2,
            "game": leg,
            "deck": deck,
            "opponent_deck": opponent_deck,
            "matchup": f"{deck} v {opponent_deck}",
            "on_play": leg % 2 == 0,
            "decklists": "closed",
        }
    with card_db("pairing", track) as db:
        data = card_data(db, CardSelection(experiment="pairing"))
        assert data.games == 8 and data.cards
        assert {row["participant"] for row in result.tables["card_plays"]} == {
            "candidate",
            "opponent",
        }
        checked = audit(db, track)
        assert checked.games == len(result.blobs) == 4
        assert checked.problems == ()
        for value, native_value, per in (
            ("lands_played", "lands_played", "land_turns"),
            ("chump_blocks", "blockers_lost_for_nothing", "blockers_committed"),
        ):
            rates = compare(db, table="behaviour", value=value, per=per)
            [rate] = [r for r in rates if r.opponent == (heuristic.key,)]
            assert rate.subject == (match.players[0].key,)
            assert rate.value == pytest.approx(
                sum(row[native_value] for row in checkpoint)
                / sum(row[per] for row in checkpoint)
            )


def test_blanking_refuses_ambiguous_values() -> None:
    player = CheckpointPlayer.model_construct(
        name="fixture", config={"blank_decision_memory": "yes"}
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        blank_decision_memory(player)
