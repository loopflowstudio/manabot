"""Frontier boundary checks; fixture values are never experiment measurements."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from manabot.model.actor_critic import ActorCritic
from manabot.sim.search import local_update
from manabot.sim.search.branch import BranchSession
from manabot.sim.search.local_update import (
    LocalSearchConfig,
    LocalUpdateTeacher,
    Rollout,
)
import managym


class Branch:
    def __init__(self, terminal_after: int = 99, winner: int | None = None) -> None:
        self.steps = 0
        self.terminal_after = terminal_after
        self.winner = winner

    def is_game_over(self) -> bool:
        return self.steps >= self.terminal_after

    def current_agent_index(self) -> int:
        return self.steps % 2

    def semantic_observation_json(self, actor: int) -> str:
        return f"{actor}/{self.steps}"

    def winner_index(self) -> int | None:
        return self.winner


class Session:
    def __init__(self, truncated: bool = False) -> None:
        self.choices: list[int] = []
        self.truncated = truncated

    def apply_policy_choice(
        self, branch: Branch, *, site: str, policy_index: int
    ) -> tuple[None, float, bool, bool, dict[str, object]]:
        assert site == "child"
        self.choices.append(policy_index)
        branch.steps += 1
        return None, 0.0, branch.is_game_over(), self.truncated, {}

    def snapshot(self) -> dict[str, object]:
        return {"choices": self.choices}


def row(branch: Branch, session: Session, seed: int = 19) -> Rollout:
    # Native and branch-session seams are substituted only in these unit fixtures.
    return Rollout(
        cast(managym.Env, branch),
        cast(BranchSession, session),
        None,
        41,
        0,
        seed,
        None,
        np.asarray([0.3, 0.7]),
    )


def teacher(monkeypatch: pytest.MonkeyPatch) -> LocalUpdateTeacher:
    instance = object.__new__(LocalUpdateTeacher)
    instance.config = LocalSearchConfig(depth=3)
    instance.agent = cast(
        ActorCritic, SimpleNamespace(config=SimpleNamespace(compound_decisions=False))
    )
    monkeypatch.setattr(
        local_update.Observation,
        "from_json",
        lambda raw: SimpleNamespace(viewer_state_hash=raw),
    )
    return instance


def test_frontier_removes_terminal_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = teacher(monkeypatch)
    branches = [Branch(1, 0), Branch()]
    sessions = [Session(), Session()]
    rows = [row(b, s) for b, s in zip(branches, sessions)]
    widths: list[int] = []

    def predict(
        engines: list[managym.Env],
    ) -> list[tuple[local_update.FloatArray, float]]:
        widths.append(len(engines))
        return [(np.asarray([0.3, 0.7]), 0.25) for _ in engines]

    monkeypatch.setattr(instance, "predict_batch", predict)
    instance._run_rollouts(rows, lambda: None)
    assert widths == [1, 1, 1]
    assert sessions[0].choices == [0]
    rng = np.random.default_rng(19)
    assert sessions[1].choices == [0] + [
        int(rng.choice(2, p=[0.3, 0.7])) for _ in range(2)
    ]
    assert rows[0].receipt(0).signed_value == 1
    assert rows[1].receipt(0).signed_value == -0.25
    assert rows[1].receipt(1).signed_value == 0.25


@pytest.mark.parametrize("winner,expected", [(None, 0), (0, 1), (1, -1)])
def test_terminal_value_is_root_signed(winner: int | None, expected: int) -> None:
    assert row(Branch(0, winner), Session()).receipt(0).signed_value == expected


def test_truncation_produces_no_target(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = teacher(monkeypatch)
    with pytest.raises(RuntimeError, match="truncated"):
        instance._run_rollouts([row(Branch(), Session(truncated=True))], lambda: None)


def test_expiry_after_forward_produces_no_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = teacher(monkeypatch)
    expired = False

    def predict(
        engines: list[managym.Env],
    ) -> list[tuple[local_update.FloatArray, float]]:
        nonlocal expired
        expired = True
        return [(np.asarray([0.3, 0.7]), 0.25) for _ in engines]

    def check() -> None:
        if expired:
            raise TimeoutError("fixture deadline")

    monkeypatch.setattr(instance, "predict_batch", predict)
    with pytest.raises(TimeoutError):
        instance._run_rollouts([row(Branch(), Session())], check)
