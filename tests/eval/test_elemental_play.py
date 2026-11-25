"""One Elemental evaluation match: failed groups stay in the denominator, rows mirror seats."""

from pathlib import Path
from typing import Literal

from etude.checkpoints import CheckpointRef
from etude.errors import RetryRequested
from etude.eval.records import Match, checkpoint_player
from etude.records import ArtifactRef
import numpy as np
import pytest
import torch

from experiments import elemental_play
from experiments.elemental_grid import GameRow, Group
from experiments.elemental_play import (
    Conditions,
    game_rows,
)
from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval.agents import build
from manabot.infra.config import ModelConfig
from manabot.infra.identity import file_sha256
from manabot.policy.prepare import Prepared
from manabot.sim.matchup import AuthoredMatchup
import managym
from tests.fixtures.evaluation_callbacks import PanicException


@pytest.mark.parametrize("game_seat", [0, 1])
def test_game_rows_mirror_each_game_for_both_players(game_seat: Literal[0, 1]) -> None:
    row = GameRow(
        model_deck="elemental_ur",
        heuristic_deck="elemental_bg",
        model_seat=game_seat,
        deal=7,
        score=1.0,
        valid=True,
        terminated=True,
        truncated=False,
        failure=None,
        decisions=40,
        model_decisions=20,
        seconds=1.5,
    )
    first, second = game_rows([row], "full")
    assert (first["deck"], first["seat"], first["game_seat"], first["score"]) == (
        "elemental_ur",
        0,
        game_seat,
        1.0,
    )
    assert (second["deck"], second["seat"], second["game_seat"], second["score"]) == (
        "elemental_bg",
        1,
        1 - game_seat,
        0.0,
    )
    assert first["unit"] == second["unit"] == "full"


@pytest.mark.parametrize("launched", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [
        managym.AgentError,
        PanicException,
        RetryRequested,
        RuntimeError,
        ValueError,
        KeyError,
    ],
)
def test_match_retry_preserves_failure_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: bool,
    error_type: type[BaseException],
) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace()
    model = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    path = tmp_path / "policy.safetensors"
    matchup = AuthoredMatchup(
        pack_key="elemental-world-v2", role_a="elemental_ur", role_b="elemental_bg"
    )
    save_checkpoint(
        model, space, path, player_configs=EnvMatch(matchup.match()).to_rust()
    )
    ref = CheckpointRef(
        address="fixture/policy",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    match = Match[Conditions](
        players=(checkpoint_player(ref), elemental_play.heuristic_code()),
        config=Conditions(
            kind="full" if launched else "subset",
            deck="elemental_ur",
            groups=(
                Group(
                    model_deck="elemental_ur",
                    heuristic_deck="elemental_bg",
                    model_seat=1,
                    deals=(7,),
                ),
            ),
        ),
    )
    monkeypatch.setattr(elemental_play, "ROOT", tmp_path)
    if launched:
        monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "launched"))
    else:
        monkeypatch.delenv("ETUDE_OUTPUT_DIR", raising=False)
    opponent = build(match.players[1], {})

    def failed_forward(rows: list[Prepared]) -> list[np.ndarray]:
        raise error_type("controlled failure")

    def forward(rows: list[Prepared]) -> list[np.ndarray]:
        return [np.zeros(row.legal.shape[1], dtype=np.float32) for row in rows]

    native = error_type in (managym.AgentError, PanicException)
    with pytest.raises(
        ValueError if native else error_type,
        match="no valid replayed games" if native else "controlled failure",
    ):
        elemental_play.play(match, [failed_forward, opponent])
    root = elemental_play.evidence_directory() / match.key
    retained = {p: p.read_bytes() for p in root.rglob("failures/*.jsonl")}
    assert len(retained) == 1
    if native:
        assert error_type.__name__ in next(iter(retained.values())).decode()
        assert list(root.rglob("rows.json"))

    result = elemental_play.play(match, [forward, opponent])
    [behaviour] = result.tables["behaviour"]
    assert (behaviour["seat"], behaviour["game_seat"], behaviour["deal"]) == (0, 1, 7)
    assert behaviour["unit"] == match.config.kind and behaviour["games"] == 1
    assert len(result.tables["games"]) == 2
    assert all(row["valid"] and row["terminated"] for row in result.tables["games"])
    assert all(row["score"] is not None for row in result.tables["games"])
    assert {p: p.read_bytes() for p in retained} == retained
    assert set(root.rglob("failures/*.jsonl")) == set(retained)
