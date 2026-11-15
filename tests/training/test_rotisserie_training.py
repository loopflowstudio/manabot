"""Draft inputs, terminal seat credit, artifacts and executor ownership."""

from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

import manabot.backends.torch as backend
from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.draft import DraftConfig, create, load, save
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import file_sha256
from manabot.policy.draft import (
    DraftInput,
    DraftPrepared,
    DraftTrajectory,
    prepare_draft,
)
from manabot.policy.prepare import Prepared
from manabot.training.draft_rewards import FrozenRewardRunner, RewardSample
from manabot.training.execution import execute_regime
from manabot.training.regime import Execution, TrainingRegime, TrainRotisserie
from manabot.training.rotisserie import collect_draft, draft_catalog, evaluate_drafter
from manabot.training.state.game_errors import GameErrorLedger, GameErrorPolicy
from manabot.training.updates.rotisserie import RunningBaseline, update_rotisserie
from manabot.verify.store import VerifyStore
import managym
from managym.format import Rotisserie


def _check() -> None:
    pass


def _batch() -> DraftPrepared[np.ndarray]:
    return DraftPrepared(
        np.arange(3),
        np.arange(18, dtype=np.float32).reshape(1, 3, 6),
        np.zeros((1, 7), np.float32),
        np.array([[True, True, False]]),
        "fixture",
    )


def test_terminal_credit_uses_only_valid_owner_and_prior_baseline() -> None:
    torch.set_num_threads(1)
    policy = create(
        DraftConfig(3, seats=2, width=8, heads=2), np.arange(3), "fixture", 7
    )
    expected = deepcopy(policy)
    frozen = deepcopy(policy).eval().requires_grad_(False)
    learner = Learner(policy, "cpu", minibatch_rows=1)
    reference = Learner(expected, "cpu", minibatch_rows=1)
    baseline = RunningBaseline(total=0.2, count=1)
    batch = _batch()
    first = DraftTrajectory(0, (batch,), (0,))
    # A failed opponent bundle would make the opposite choice. Its reward must
    # never affect the gradient or the baseline, and it cannot own seat 0's row.
    invalid = DraftTrajectory(1, (batch,), (1,))
    update_rotisserie(backend, learner, [first, invalid], [0.8, None], baseline)
    reference.set_learning_rate(1e-4)
    reference.step(-0.6 * expected.log_probability([batch], [0]), 1)
    for a, b in zip(policy.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert all(p.grad is None for p in frozen.parameters())
    assert baseline.total == 1 and baseline.count == 2
    before = deepcopy(policy.state_dict())
    assert update_rotisserie(backend, learner, [invalid], [None], baseline) is None
    for key, value in before.items():
        torch.testing.assert_close(value, policy.state_dict()[key])
    assert baseline.count == 2
    with pytest.raises(ValueError, match="torch"):
        update_rotisserie(ModuleType("unsupported"), learner, [first], [1], baseline)


def test_public_projection_preserves_counts_relative_seats_order_and_native_mask() -> (
    None
):
    catalog = draft_catalog("tla", 2, 1)
    draft = Rotisserie("tla", "public-fixture", 2, 1)
    state = draft.observe()
    item = DraftInput("draft", state, 0)
    batch = prepare_draft([item], catalog.cards, catalog.binding)
    assert isinstance(batch, Prepared)
    np.testing.assert_array_equal(
        batch.features[0, :, 0],
        np.array(item.tokens()["state"]["initial"])[catalog.cards],
    )
    np.testing.assert_array_equal(
        batch.legal[0], np.isin(catalog.cards, state["legal"])
    )
    card = state["legal"][0]
    draft.pick(state["episode_id"], 0, 0, card)
    second = draft.observe()
    projected = prepare_draft(
        [DraftInput("draft", second, 1)], catalog.cards, catalog.binding
    )
    index = int(np.flatnonzero(catalog.cards == card)[0])
    assert projected.features[0, index, 2] == 0  # chooser's pool
    assert projected.features[0, index, 3] == 1  # previous seat's public pool
    assert projected.position[0].tolist() == [0, 1, 0, 1, 2, 1, 40]
    with pytest.raises(ValueError, match="chooser"):
        prepare_draft([DraftInput("draft", second, 0)], catalog.cards, catalog.binding)
    with pytest.raises(ValueError, match="completed"):
        prepare_draft([DraftInput("build", second, 0)], catalog.cards, catalog.binding)


def test_draft_policy_reload_reproduces_transcript_and_legal_construction(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    catalog = draft_catalog("tla", 2, 1)
    policy = create(
        DraftConfig(len(catalog.names), seats=2, width=8, heads=2),
        catalog.cards,
        catalog.binding,
        3,
    )
    path = tmp_path / "policy.pt"
    save(path, policy)
    restored = load(path, binding=catalog.binding, cards=catalog.cards)
    first = collect_draft(
        tmp_path / "first",
        "fixed",
        4,
        [policy, None],
        ("learner", "uniform"),
        catalog,
        picks=1,
        check=_check,
    )
    again = collect_draft(
        tmp_path / "again",
        "fixed",
        4,
        [restored, None],
        ("learner", "uniform"),
        catalog,
        picks=1,
        check=_check,
    )
    assert first.episode == again.episode
    assert (tmp_path / "first/choices.jsonl").read_bytes() == (
        tmp_path / "again/choices.jsonl"
    ).read_bytes()
    assert len(first.trajectories) == 1 and len(first.trajectories[0].actions) == 41
    assert all(sum(d.main.values()) == 40 for d in first.episode.decks)
    with pytest.raises(ValueError, match="binding"):
        load(path, binding="changed-printings", cards=catalog.cards)
    with pytest.raises(ValueError, match="binding"):
        restored(
            DraftPrepared(
                torch.tensor(catalog.cards),
                torch.zeros(1, len(catalog.cards), 6),
                torch.zeros(1, 7),
                torch.ones(1, len(catalog.cards), dtype=torch.bool),
                "wrong",
            )
        )


def test_fixed_reference_evaluation_crosses_draft_and_game_seats(
    tmp_path: Path,
) -> None:

    torch.set_num_threads(1)
    catalog = draft_catalog("tla", 2, 1)
    candidate = create(
        DraftConfig(len(catalog.names), seats=2, width=8, heads=2),
        catalog.cards,
        catalog.binding,
        3,
    )
    initial = deepcopy(candidate).eval().requires_grad_(False)
    calls: list[RewardSample] = []

    def run(sample: RewardSample, journal: Path) -> dict[str, Any]:
        calls.append(sample)
        return {"terminated": True, "winner": sample.seat, "opening_value": -0.25}

    ledger = GameErrorLedger(tmp_path / "errors.jsonl", GameErrorPolicy())
    evaluate_drafter(
        candidate,
        initial,
        catalog,
        run,
        tmp_path / "eval",
        seed=7,
        check=_check,
        ledger=ledger,
        picks=1,
        seats=2,
    )
    assert len(calls) == 16
    assert {s.bundle for s in calls} == {
        "initial-seat0",
        "initial-seat1",
        "uniform-seat0",
        "uniform-seat1",
    }
    for start in range(0, 16, 4):
        a, b, c, d = calls[start : start + 4]
        assert (a.seat, b.seat, c.seat, d.seat) == (0, 1, 0, 1)
        assert a.seed == b.seed and c.seed == d.seed == a.seed + 1
        assert a.match.hero_deck == b.match.villain_deck
        assert a.match.villain_deck == b.match.hero_deck
    rows = json.loads((tmp_path / "eval/results.json").read_text())
    assert all(r["value"] == 1 and r["opening"] == -0.25 for r in rows)


def test_regime_executes_draft_update_with_controlled_rewards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    torch.set_num_threads(1)
    space = ObservationSpace()
    player = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    path = tmp_path / "fixture-player.pt"
    save_checkpoint(
        player,
        space,
        path,
        player_configs=None,
        extra={"value_semantic": "signed_outcome"},
    )

    def run(self: object, sample: RewardSample, journal: Path) -> dict[str, Any]:
        return {"terminated": True, "winner": sample.seat}

    monkeypatch.setattr(FrozenRewardRunner, "__call__", run)
    stage = TrainRotisserie(
        id="draft",
        player={
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        },
        reward="game",
        seats=2,
        picks_per_seat=1,
        drafts_per_update=1,
        updates=1,
        evaluate=False,
        execution=Execution(wall_seconds=60, threads=1),
    )
    regime = TrainingRegime(
        id="draft-fixture",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        stages=[stage],
        wall_seconds=60,
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        result = execute_regime(regime, 370, tmp_path / "run", store)
    (record,) = result.stages
    assert result.status == record.status == "completed"
    assert record.updates == record.optimizer_updates == 1
    assert record.optimizer_exposures == record.learner_transitions == 41
    assert set(record.artifacts) == {"draft-update-0", "draft-update-1"}
    assert result.selected_artifact is None  # Draft artifacts are not game players.
    assert result.metrics_run
