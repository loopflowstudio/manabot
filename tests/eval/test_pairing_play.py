"""Authored paired deals preserve disclosure and distinct blanked checkpoint players."""

from collections.abc import Iterator
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from etude.checkpoints import CheckpointRef
from etude.eval.query import compare, read_blobs
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
from manabot.eval.card_query import CardSelection, card_data, card_db
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.sim.agent import GamePlayer
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup
import managym


@pytest.fixture
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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
    players: list[CheckpointPlayer] = []
    for seed in (19, 20):
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
                decision_memory="gru",
                decision_memory_summaries=1,
            ),
        )
        path = tmp_path / f"{seed}.safetensors"
        save_checkpoint(
            model,
            space,
            path,
            player_configs=None if world else EnvMatch(matchup.match()).to_rust(),
        )
        ref = CheckpointRef(
            experiment="fixture",
            job="pairing",
            seed=seed,
            name=f"fixture-{seed}",
            artifact=ArtifactRef(
                uri=path.as_uri(),
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
            ),
        )
        players.append(checkpoint_player(ref))
    if blank:
        players[1] = players[0].model_copy(
            update={"config": {"blank_decision_memory": True}}
        )
        assert players[0].key != players[1].key
    # As Etude seats them: a bare forward without config, a built player with one.
    agents = [
        build(player, {player.checkpoint.key: load(player.checkpoint)})
        if player.config
        else load(player.checkpoint)
        for player in players
    ]
    original = pairing.evaluate

    def evaluate(
        request: EvaluationRequest,
        *,
        out_dir: Path,
        agents: dict[str, GamePlayer],
        card_evidence_dir: Path,
    ) -> EvaluationResult:
        assert request.authored_matchup is not None
        env = managym.Env(seed=123)
        env.reset(EnvMatch(request.authored_matchup.match()).to_rust())
        rows = encode_observation(space, env.observation_for_player(0))["decklists"]
        assert rows[0].any()
        assert bool(rows[1].any()) == (disclosure == "open")
        assert request.opponent.player_spec.get("blank_decision_memory", False) == blank
        if blank:
            assert (
                request.candidate.checkpoint_sha256
                == request.opponent.checkpoint_sha256
            )
            assert request.candidate.identity_sha256 != request.opponent.identity_sha256
        assert agents["opponent"].blank_decision_memory == blank
        return original(
            request, out_dir=out_dir, agents=agents, card_evidence_dir=card_evidence_dir
        )

    monkeypatch.setattr(pairing, "evaluate", evaluate)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "output"))
    match = Match(players=tuple(players), games=4)
    result = pairing.play_pairing(
        match,
        agents,
        matchup=matchup,
        # Every leg of deal 1 ends in under 300 decisions for each case. Untrained
        # memory policies can play much longer games (deal 123 has a 502-decision
        # leg), and memory encoding grows with the game, which made the 60 s
        # budget a race.
        deal=1,
        matchup_mode="cross",
        game_seconds=60,
        arena="fixture-pairing",
    )
    rows = result.tables["plays"]
    assert "behaviour" not in result.tables
    assert len(rows) == 8
    assert {row["decklists"] for row in rows} == {disclosure}
    assert {row["game"] for row in rows} == set(range(4))
    assert all(row["valid"] for row in rows)
    for game in range(4):
        assert sum(float(row["score"]) for row in rows if row["game"] == game) == 1.0
    assert (tmp_path / "output" / "arena" / match.key).is_dir()


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
    space = ObservationSpace(ObservationConfig())
    torch.manual_seed(19)
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
        ),
    )
    path = tmp_path / "19.safetensors"
    save_checkpoint(model, space, path, player_configs=None)
    ref = CheckpointRef(
        experiment="fixture",
        job="pairing",
        seed=19,
        name="fixture-19",
        at={"updates": 0},
        artifact=ArtifactRef(
            uri=path.as_uri(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        ),
    )
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config={"max_actions": 64}
    )
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(output))
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

    retained = output / "arena" / match.key
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
        blobs = {sha256(body).hexdigest(): body for body in result.blobs}
        assert len(result.blobs) == len(blobs) == 4
        plays = db.execute(
            "select evidence_sha256, participant from card_plays"
        ).fetchall()
        assert sorted(plays) == sorted(
            (digest, player) for digest in blobs for player in ("candidate", "opponent")
        )
        assert dict(read_blobs(db, track, blobs)) == blobs
        comparisons = compare(
            db, table="behaviour", value="lands_played", per="land_turns"
        )
        blocks = compare(
            db, table="behaviour", value="chump_blocks", per="blockers_committed"
        )
    [block_rate] = [c for c in blocks if c.opponent == (heuristic.key,)]
    assert block_rate.value == pytest.approx(
        sum(row["blockers_lost_for_nothing"] for row in checkpoint)
        / sum(row["blockers_committed"] for row in checkpoint)
    )
    [against] = [c for c in comparisons if c.opponent == (heuristic.key,)]
    assert against.subject == (match.players[0].key,)
    assert against.value == pytest.approx(
        sum(row["lands_played"] for row in checkpoint)
        / sum(row["land_turns"] for row in checkpoint)
    )


def test_blanking_refuses_ambiguous_values() -> None:
    player = CheckpointPlayer.model_construct(
        name="fixture", config={"blank_decision_memory": "yes"}
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        blank_decision_memory(player)
