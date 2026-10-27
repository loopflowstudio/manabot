"""MTG-181's frozen grid assigns every scheduled deck and seat through real arena legs."""

from collections import Counter
from pathlib import Path
import time
from typing import Literal

import pytest
import torch

from experiments.elemental_grid import (
    DECKS,
    GAMES_PER_CELL,
    PACK,
    SCHEDULE_SHA256,
    GameRow,
    Group,
    Request,
    _legs,
    groups,
    play_group,
    schedule,
    schedule_sha256,
    summarize,
)
from manabot.env import Match, ObservationSpace
from manabot.eval.execution import bind_execution
from manabot.eval.models import EvaluationKey
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.serving import PolicyAdmission, PolicyStore


@pytest.mark.parametrize("version", (1, 2))
def test_grid_registers_saved_policy_inputs_for_shared_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: Literal[1, 2]
) -> None:
    torch.set_num_threads(1)
    setup = AuthoredMatchup(pack_key=PACK, role_a=DECKS[0], role_b=DECKS[1])
    space = ObservationSpace(ObservationConfig(policy_input_version=version))
    model = ActorCritic(
        space, ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK)
    )
    path = tmp_path / "policy.pt"
    save_checkpoint(model, space, path, player_configs=Match(setup.match()).to_rust())
    admitted: list[int] = []

    def admit(
        *,
        key: EvaluationKey,
        player_a: PlayerRegistration,
        player_b: PlayerRegistration,
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        # Exercise the real shared-service admission without playing a game.
        # The anchor/key retain the legacy seed inputs independently of policy ABI.
        legacy = runtime_fingerprints(match_config=setup.match())
        assert player_b.observation_abi_sha256 == legacy["observation_abi_sha256"]
        assert key.anchor_cohort_sha256 == canonical_sha256(player_b.model_dump())
        execution = EvalExecution(deadline_unix=time.time() + 60)
        _, (bound,) = bind_execution(key, (player_a,), execution)
        store = PolicyStore(execution, EvalPerformance("grid-admission"))
        try:
            actual = store.admit(PolicyAdmission(bound, str(path), (setup.match(),)))
            admitted.append(actual.policy_input_version)
        finally:
            store.close()
        return [], {}, {}

    monkeypatch.setattr("manabot.eval.match.play_cell", admit)
    play_group(
        Request(
            checkpoint=path,
            sha256=file_sha256(path),
            training_seed=18301,
            group=Group(
                model_deck=DECKS[0], heuristic_deck=DECKS[1], model_seat=0, deals=(1,)
            ),
            output=tmp_path / "evidence",
        )
    )
    assert admitted == [version]


def test_schedule_is_frozen_and_complete() -> None:
    games = schedule()
    assert schedule_sha256() == SCHEDULE_SHA256
    assert len(games) == len(DECKS) ** 2 * GAMES_PER_CELL
    assert len({g.deal for g in games}) == len(games)
    cells = Counter((g.model_deck, g.heuristic_deck) for g in games)
    assert set(cells.values()) == {GAMES_PER_CELL}
    assert sum(len(g.deals) for g in groups(games)) == len(games)


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize(
    ("model_deck", "heuristic_deck"),
    [("elemental_ur", "elemental_bg"), ("elemental_wr", "elemental_wr")],
)
def test_leg_puts_model_deck_in_scheduled_seat(
    model_deck: str, heuristic_deck: str, seat: int
) -> None:
    group = Group(
        model_deck=model_deck,
        heuristic_deck=heuristic_deck,
        model_seat=seat,  # type: ignore[arg-type]
        deals=(1,),
    )
    role_a, role_b, leg = _legs(group)
    seat_decks = AuthoredMatchup(
        pack_key="elemental-world-v2", role_a=role_a, role_b=role_b
    ).decks(leg)
    # play_cell seats player A (the model) at leg % 2.
    assert leg % 2 == seat
    assert seat_decks[seat] == model_deck
    assert seat_decks[1 - seat] == heuristic_deck


def test_invalid_games_are_reported_not_scored(tmp_path: Path) -> None:
    def row(score: float | None, valid: bool) -> GameRow:
        return GameRow(
            model_deck=DECKS[0],
            heuristic_deck=DECKS[1],
            model_seat=0,
            deal=1,
            score=score,
            valid=valid,
            terminated=valid,
            truncated=False,
            failure=None if valid else "crash",
            decisions=1,
            model_decisions=1,
            seconds=1.0,
        )

    summary = summarize([row(1.0, True), row(0.5, True), row(None, False)])
    cell = summary.cells[DECKS[0]][DECKS[1]]
    assert (cell.score, cell.valid, cell.invalid) == (0.75, 2, 1)
    assert cell.low is not None and cell.high is not None
    assert cell.low < 0.75 < cell.high
    assert summary.cells[DECKS[1]][DECKS[0]].score is None
