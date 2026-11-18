"""Authored paired deals preserve disclosure and distinct blanked checkpoint players."""

from pathlib import Path
from typing import Literal

from etude.checkpoints import CheckpointRef
from etude.eval.records import CheckpointPlayer, Code, Match, checkpoint_player
from etude.records import ArtifactRef
import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval import pairing
from manabot.eval.agent_loader import load
from manabot.eval.agents import blank_decision_memory, build
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.sim.agent import GamePlayer
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import AuthoredMatchup
import managym


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
    disclosure: Literal["open", "closed"],
    blank: bool,
    world: Literal["world"] | None,
) -> None:
    """A world matchup admits world-bound checkpoints, as trained recipes write them."""
    torch.set_num_threads(1)
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
        request: EvaluationRequest, *, out_dir: Path, agents: dict[str, GamePlayer]
    ) -> EvaluationResult:
        assert request.authored_matchup is not None
        env = managym.Env(seed=123)
        env.reset(EnvMatch(request.authored_matchup.match()).to_rust())
        rows = space.encode(env.observation_for_player(0))["decklists"]
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
        return original(request, out_dir=out_dir, agents=agents)

    monkeypatch.setattr(pairing, "evaluate", evaluate)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "output"))
    match = Match(players=tuple(players), games=4)
    tables = pairing.play_pairing(
        match,
        agents,
        matchup=matchup,
        deal=123,
        matchup_mode="cross",
        game_seconds=60,
        arena="fixture-pairing",
    )
    rows = tables["plays"]
    assert len(rows) == 8
    assert {row["decklists"] for row in rows} == {disclosure}
    assert {row["game"] for row in rows} == set(range(4))
    assert all(row["valid"] for row in rows)
    for game in range(4):
        assert sum(float(row["score"]) for row in rows if row["game"] == game) == 1.0
    assert (tmp_path / "output" / "arena" / match.key).is_dir()


def test_pairing_plays_a_checkpoint_against_the_heuristic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    matchup = AuthoredMatchup(
        pack_key="ur-lessons-vs-gw-allies",
        role_a="ur_lessons",
        role_b="gw_allies",
        decklists="closed",
        world=managym.WORLD_SETUP,
    )
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
        artifact=ArtifactRef(
            uri=path.as_uri(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        ),
    )
    heuristic = Code(name="heuristic", source=heuristic_source_paths())
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "output"))
    match = Match(players=(checkpoint_player(ref), heuristic), games=4)
    tables = pairing.play_pairing(
        match,
        [load(ref), build(heuristic, {})],
        matchup=matchup,
        deal=123,
        matchup_mode="cross",
        game_seconds=60,
        arena="fixture-pairing",
    )
    rows = tables["plays"]
    assert len(rows) == 8 and all(row["valid"] for row in rows)
    for game in range(4):
        assert sum(float(row["score"]) for row in rows if row["game"] == game) == 1.0


def test_blanking_refuses_ambiguous_values() -> None:
    player = CheckpointPlayer.model_construct(
        name="fixture", config={"blank_decision_memory": "yes"}
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        blank_decision_memory(player)
