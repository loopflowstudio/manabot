"""Signed critic units and complete, failed, partial and missing reward slots."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
import torch

from manabot.backends.torch.checkpoint import BACKEND, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import file_sha256
from manabot.sim.native_game import _InChildPlayers, execute_game
from manabot.sim.opening import signed_value
from manabot.training.draft_rewards import (
    FrozenRewardRunner,
    RewardSample,
    score_schedule,
)
from manabot.training.state.game_errors import GameErrorLedger, GameErrorPolicy
import managym
from tests.eval.test_evaluation_match import selected_players


def _check() -> None:
    pass


def _samples(mode: str = "game") -> list[RewardSample]:
    return [
        RewardSample(
            id=f"sample-{i}",
            bundle="learner-0",
            mode=mode,
            seed=10 + i // 2,
            seat=i % 2,
            match=MatchConfig(),
        )
        for i in range(4)
    ]


def test_signed_critic_units_refuse_unspecified_scalars() -> None:
    assert signed_value(0.25, "signed_outcome") == 0.25
    assert signed_value(2, "signed_outcome") == 2
    assert signed_value(0, "win_logit") == 0
    assert signed_value(1000, "win_logit") == 1
    assert signed_value(-1000, "win_logit") == -1
    for value, semantic in [
        (0, None),
        (float("nan"), "signed_outcome"),
    ]:
        with pytest.raises(ValueError):
            signed_value(value, semantic)


@pytest.mark.parametrize("opening", [-0.5, 2.0])
def test_terminal_outcomes_are_chooser_relative_and_opening_is_separate(
    tmp_path: Path,
    opening: float,
) -> None:
    def run(sample: RewardSample, journal: Path) -> dict[str, Any]:
        return {"terminated": True, "winner": sample.seat, "opening_value": -0.5}

    batch = score_schedule(_samples(), tmp_path / "game", run, check=_check)
    assert batch.rewards == {"learner-0": 1}
    assert all(
        row.status == "completed" and row.opening == -0.5 for row in batch.results
    )

    def score(sample: RewardSample, journal: Path) -> dict[str, Any]:
        return {
            "terminated": False,
            "termination_reason": "opening_scored",
            "opening_value": opening,
        }

    values = score_schedule(_samples("value"), tmp_path / "value", score, check=_check)
    assert values.rewards == {"learner-0": opening}
    receipt = json.loads((tmp_path / "value/receipt.json").read_text())
    assert receipt["completed_games"] == 0 and receipt["opening_scores"] == 4
    settled = [
        json.loads(line)["attempt"]["outcome"]
        for line in (tmp_path / "value/game-errors.jsonl").read_text().splitlines()
    ]
    assert settled == ["opening_scored"] * 4


@pytest.mark.parametrize(
    "failure",
    [
        "engine",
        "truncated",
        "action_limit",
        "deadline",
        "raised_deadline",
        "death",
        "early_terminal",
    ],
)
def test_partial_bundles_never_learn_and_dispatched_slots_settle(
    tmp_path: Path, failure: str
) -> None:
    def run(sample: RewardSample, journal: Path) -> dict[str, Any]:
        if sample.id != "sample-1":
            return {
                "terminated": False,
                "termination_reason": "opening_scored",
                "opening_value": 0.5,
            }
        if failure == "engine":
            return {
                "failure": "engine fixture",
                "error_class": "PyAgentError",
                "termination_reason": "crash",
            }
        if failure == "truncated":
            return {"failure": "truncated fixture", "termination_reason": "truncated"}
        if failure == "action_limit":
            return {"failure": "Command cap", "termination_reason": "command_cap"}
        if failure == "deadline":
            return {"failure": "deadline fixture", "termination_reason": "timeout"}
        if failure == "raised_deadline":
            raise TimeoutError("controlled runner deadline")
        if failure == "death":
            raise EOFError("controlled worker exit")
        return {"terminated": True, "winner": 0}

    batch = score_schedule(_samples("value"), tmp_path / failure, run, check=_check)
    assert batch.rewards == {"learner-0": None}
    expected = 2 if failure in ("death", "deadline", "raised_deadline") else 4
    assert len(batch.results) == expected
    assert batch.timed_out == (failure in ("deadline", "raised_deadline"))
    if batch.stopped:
        assert batch.stop_reason and "sample-1" in batch.stop_reason
    receipt = json.loads((tmp_path / failure / "receipt.json").read_text())
    assert receipt["planned"] == 4 and receipt["missing"] == 4 - expected
    events = [
        json.loads(line)
        for line in (tmp_path / failure / "events.jsonl").read_text().splitlines()
    ]
    assert (
        sum(e["event"] == "dispatch" for e in events)
        == sum(e["event"] == "settled" for e in events)
        == expected
    )


@pytest.mark.parametrize("seat", [0, 1])
def test_native_worker_scores_first_own_decision_before_acting(
    seat: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    _, registrations = selected_players()
    acted: list[int] = []
    scored: list[int] = []
    agents = []

    class Player:
        def __init__(self) -> None:
            self.agent = torch.nn.Linear(1, 1)
            agents.append(self.agent)

        def act(self, env: Any, obs: Any) -> int:
            acted.append(int(env.last_raw_obs.agent.player_index))
            return 0

        def opening_value(self, env: Any, obs: Any) -> float:
            scored.append(int(env.last_raw_obs.agent.player_index))
            return -0.25

    def build(self: object, *args: object, **kwargs: object) -> tuple[Player, None]:
        return Player(), None

    monkeypatch.setattr(_InChildPlayers, "build", build)
    game = {
        "match_id": "first-own",
        "deal_seed": 370,
        "player_seeds": {p.player_id: 7 for p in registrations},
        "seat_players": [p.player_id for p in registrations],
        "decisions": [],
        "failure": None,
        "terminated": False,
        "integrity": dict(
            private_exposures=0, root_mutations=0, illegal_actions=0, truncations=0
        ),
        "opening_score_seat": seat,
        "stop_after_opening": True,
    }
    messages: list[tuple[str, object]] = []
    execute_game(game, registrations, {}, 30, messages.append)
    assert game["failure"] is None, game
    assert game["termination_reason"] == "opening_scored"
    assert not game["terminated"]
    assert scored == [seat] and seat not in acted
    assert game["opening_value"] == -0.25
    assert all(not p.requires_grad for agent in agents for p in agent.parameters())
    assert game["opening_revision"] == len(acted)


def test_deadline_before_dispatch_preserves_all_missing_slots(tmp_path: Path) -> None:
    def deadline() -> None:
        raise TimeoutError("controlled run deadline")

    def never(sample: RewardSample, journal: Path) -> dict[str, Any]:
        raise AssertionError("must not dispatch")

    batch = score_schedule(_samples(), tmp_path / "deadline", never, check=deadline)
    assert batch.stopped and not batch.results and batch.rewards == {"learner-0": None}
    receipt = json.loads((tmp_path / "deadline/receipt.json").read_text())
    assert receipt["attempted"] == 0 and receipt["missing"] == 4
    assert batch.timed_out and receipt["timed_out"]
    assert receipt["stop_reason"] == "TimeoutError: controlled run deadline"


@pytest.mark.parametrize(
    "mode,critic,expected",
    [("value", "scalar", 0), ("game", "scalar", 0), ("value", "categorical_wdl", 0.4)],
)
def test_scalar_checkpoint_opening_scores_through_supervised_child(
    tmp_path: Path, mode: str, critic: str, expected: float
) -> None:

    torch.set_num_threads(1)
    space = ObservationSpace()
    player = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            world=managym.WORLD_SETUP,
            value_kind=critic,
        ),
    )
    for parameter in player.value_head.parameters():
        parameter.data.zero_()
    if critic == "categorical_wdl":
        with torch.no_grad():
            player.value_head[-1].bias[2] = torch.tensor(3.0).log()
    path = tmp_path / "player.pt"
    save_checkpoint(
        player,
        space,
        path,
        player_configs=None,
        extra={"value_semantic": "signed_outcome"},
    )
    runner = FrozenRewardRunner(path, BACKEND, file_sha256(path), perf_counter() + 30)
    sample = RewardSample(
        id="native-opening",
        bundle="native",
        mode=mode,
        seed=370,
        seat=1,
        match=MatchConfig(
            hero_deck={"Forest": 40},
            villain_deck={"Mountain": 40},
            world=managym.WORLD_SETUP,
        ),
    )
    batch = score_schedule([sample], tmp_path / "scores", runner, check=_check)
    assert not batch.stopped
    assert batch.rewards["native"] is not None
    assert batch.results[0].status == ("scored" if mode == "value" else "completed")
    assert Path(batch.results[0].evidence).is_file()
    trace = json.loads((tmp_path / "scores/00000.game.json").read_text())
    assert trace["opening_value"] == pytest.approx(expected)
    if mode == "value":
        assert not trace["terminated"]
        assert all(row["actor"] != 1 for row in trace["decisions"])
    else:
        assert trace["terminated"] and not trace["replay"]["error"]
        assert all(row["passed"] for row in trace["replay"]["receipts"])


def test_failure_allowance_spans_batches_and_rejects_integrity_results(
    tmp_path: Path,
) -> None:

    ledger = GameErrorLedger(
        tmp_path / "ledger.jsonl", GameErrorPolicy(window=4, warmup=4, threshold=0.25)
    )

    def run(sample: RewardSample, journal: Path) -> dict[str, Any]:
        if sample.id == ("sample-3" if journal.parent.name == "first" else "sample-0"):
            return {
                "failure": "fixture",
                "termination_reason": "crash",
                "error_class": "PyAgentError",
            }
        return {"terminated": True, "winner": sample.seat}

    first = score_schedule(
        _samples(), tmp_path / "first", run, check=_check, ledger=ledger
    )
    assert not first.stopped and ledger.state.attempted == 4
    second = score_schedule(
        _samples(), tmp_path / "second", run, check=_check, ledger=ledger
    )
    assert second.stopped and len(second.results) == 1

    def corrupt(sample: RewardSample, journal: Path) -> dict[str, Any]:
        return {
            "terminated": True,
            "winner": sample.seat,
            "integrity": {"private_exposures": 1},
        }

    third = score_schedule(_samples(), tmp_path / "integrity", corrupt, check=_check)
    assert third.stopped and third.rewards == {"learner-0": None}
