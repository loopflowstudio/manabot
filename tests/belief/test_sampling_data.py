"""Whole-game immutable evidence and real frozen-policy collection checks."""

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.belief.sampling_collection import (
    CaptureSlot,
    CaptureTransition,
    GameCaptureFailure,
    capture_game,
    collect_schedule,
    read_capture_dataset,
    retained_offer_count,
)
from manabot.belief.sampling_data import (
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
    collect_frozen_policy,
    public_sampler_input,
    read_dataset,
    save_dataset,
)
from manabot.belief.sampling_evidence import EvidenceRoot, JobAllocation, JobSettlement
from manabot.belief.sampling_runtime import EnvCaptureRuntime
from manabot.belief.sampling_types import SamplerInput, SamplerSchema
from manabot.belief.state import ViewerHistory
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.model.world import checkpoint_world
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.sim.search.flat_mc import AgentMatchupPlayer
from manabot.verify.text_observation import priority_root
import managym
from managym.decision import (
    PUBLIC_COMMITMENT_KINDS,
    SEMANTIC_DECISION_VERSION,
    Command,
    DecisionFrame,
    Observation,
    SemanticContractError,
    TransitionReceipt,
)


def _dataset() -> SamplerDataset:
    schema = SamplerSchema("test-vocabulary", ("Island", "Mountain"), 0)
    row = SamplerExample(
        SamplerInput("test-vocabulary", (2, 2), (0, 0), 2, ()),
        (1, 1),
        0,
        1,
        "observation",
    )
    splits: tuple[Split, ...] = ("train", "validation", "test")
    return SamplerDataset(
        schema,
        "policy",
        "world",
        tuple(
            SamplerGame(str(index), index, 0, split, (row,))
            for index, split in enumerate(splits)
        ),
    )


def test_dataset_round_trip_and_fail_closed_integrity(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "dataset.json"
    save_dataset(dataset, path)
    assert read_dataset(path) == dataset
    with pytest.raises(FileExistsError):
        save_dataset(dataset, path)
    payload = json.loads(path.read_text())
    payload["dataset"]["games"][0]["examples"][0]["target_hand"] = [2, 0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        read_dataset(path)
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            dataset,
            games=(
                *dataset.games,
                replace(dataset.games[0], game_id="other", split="test"),
            ),
        )


@pytest.mark.parametrize(
    ("max_actions", "action_overflow"),
    [(10, "error"), (1, "error"), (1, "first-legal")],
)
def test_real_frozen_collection_keeps_both_viewers_and_three_game_splits(
    tmp_path: Path,
    max_actions: int,
    action_overflow: Literal["error", "first-legal"],
) -> None:
    torch.set_num_threads(1)
    match_config = MatchConfig(
        hero_deck={"Mountain": 4, "Raging Goblin": 4},
        villain_deck={"Mountain": 4, "Raging Goblin": 4},
    )
    obs_config = ObservationConfig(
        max_actions=max_actions, action_overflow=action_overflow
    )
    obs_space = ObservationSpace(obs_config)
    agent_config = ModelConfig(hidden_dim=8)
    agent = ActorCritic(obs_space, agent_config)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "world_binding": checkpoint_world(Match(match_config).to_rust(), obs_space),
            "hypers": {
                "observation_hypers": obs_config.model_dump(),
                "agent_hypers": agent_config.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
        },
        checkpoint,
    )
    if max_actions == 1 and action_overflow == "error":
        with pytest.raises(
            ValueError, match="(?i)observation capacity exceeded: actions"
        ):
            collect_frozen_policy(
                checkpoint=checkpoint,
                match_config=match_config,
                games=3,
                seed=51,
                max_steps=300,
            )
        return
    dataset = collect_frozen_policy(
        checkpoint=checkpoint,
        match_config=match_config,
        games=3,
        seed=51,
        max_steps=300,
    )
    assert len(dataset.games) == 3
    assert {game.split for game in dataset.games} == {"train", "validation", "test"}
    assert {game.assignment for game in dataset.games} == {0, 1}
    assert all(
        {row.viewer for row in game.examples} == {0, 1} for game in dataset.games
    )
    assert all(
        sum(row.target_hand) == row.inputs.hand_size
        for game in dataset.games
        for row in game.examples
    )
    path = tmp_path / "real.json"
    save_dataset(dataset, path)
    assert read_dataset(path) == dataset


def test_count_sampler_rejects_retained_private_subset_before_inference() -> None:
    env = priority_root(
        ("Sokka, Bold Boomeranger", "Island", "Mountain", "Forest"),
        ("Island", "Mountain"),
    )
    decision = capture_text_decision(env)
    cast = next(action for action in decision.actions if "Cast" in action.description)
    submit_response(env, decision, cast.reference)
    for _ in range(12):
        decision = capture_text_decision(env)
        if "Stage: select cards." in decision.text:
            break
        submit_response(env, decision, decision.actions[0].reference)
    else:
        raise AssertionError("Sokka selection not reached")
    pick = next(action for action in decision.actions if "Select" in action.description)
    submit_response(env, decision, pick.reference)
    viewer = 1 - int(env.current_agent_index())
    history = ViewerHistory.from_observation(
        Observation.from_json(env.semantic_observation_json(viewer))
    )
    constraints = json.loads(env.hidden_hand_constraints_json(viewer))
    assert constraints["retained_selection"] is True
    schema = SamplerSchema("fixture", tuple(sorted(constraints["pool"])), 0)
    with pytest.raises(ValueError, match="cannot represent retained selections"):
        public_sampler_input(env, history, schema)


class _CaptureFixture:
    """Controlled semantic trajectory; no game execution or private input features."""

    retained_actions = 2

    def __init__(self, *, hand: str = "Island", failure_at: int | None = None) -> None:
        self.hand = hand
        self.failure_at = failure_at
        self.revision = 0
        self.resets = 0

    def reset(self, slot: CaptureSlot) -> None:
        self.revision = 0
        self.resets += 1

    def observation(self, viewer: int) -> Observation:
        return Observation(
            SEMANTIC_DECISION_VERSION,
            self.revision,
            viewer,
            f"viewer-{viewer}-{self.revision}",
            {},
            (),
            None,
        )

    def hidden_hand_constraints_json(self, viewer: int) -> str:
        observation = self.observation(viewer)
        return json.dumps(
            {
                "pool": {"Island": 2, "Mountain": 2},
                "known_hand": {},
                "hand_size": 1,
                "foretold": [],
                "retained_selection": self.revision == 1 and viewer == 0,
                "source_observation": {
                    "revision": observation.revision,
                    "viewer": viewer,
                    "schema_version": observation.schema_version,
                    "viewer_state_hash": observation.viewer_state_hash,
                },
            }
        )

    def hand_counts(self, player: int) -> dict[str, int]:
        return {self.hand: 1}

    def frame(self) -> DecisionFrame:
        return DecisionFrame(
            SEMANTIC_DECISION_VERSION,
            self.revision,
            self.revision % 2,
            "frame",
            ({"id": 41}, {"id": 87}, {"id": 96}),
            (),
        )

    def act(self) -> int:
        return 1

    def step(self, command: Command) -> CaptureTransition:
        if self.revision == self.failure_at:
            raise GameCaptureFailure("injected isolated failure")
        before = self.revision
        self.revision += 1
        receipt = TransitionReceipt(
            SEMANTIC_DECISION_VERSION,
            before,
            self.revision,
            command.command_id,
            {"kind": "pass_priority"},
            (),
            None,
        )
        return CaptureTransition(receipt, self.revision == 12, False)


def _capture_schema() -> SamplerSchema:
    return SamplerSchema(
        "fixture", ("Island", "Mountain"), 2 * len(PUBLIC_COMMITMENT_KINDS) * 3
    )


def test_scheduled_capture_reservoir_is_label_independent() -> None:
    slot = CaptureSlot("one", 357, 0, "train", ("a", "b"))
    results = [
        capture_game(
            _CaptureFixture(hand=hand),
            slot,
            _capture_schema(),
            moments=3,
            omit_unsupported=True,
        )
        for hand in ("Island", "Mountain")
    ]
    first, second = results
    assert first.game is not None and second.game is not None
    assert [row.inputs for row in first.game.examples] == [
        row.inputs for row in second.game.examples
    ]
    assert [row.revision for row in first.game.examples] == [
        row.revision for row in second.game.examples
    ]
    assert [row.target_hand for row in first.game.examples] != [
        row.target_hand for row in second.game.examples
    ]
    assert first.coverage.decisions == 12
    assert first.coverage.eligible_rows == (11, 12)
    assert first.coverage.unsupported_rows == (1, 0)
    assert first.coverage.omission_reasons == {"retained_selection": (1, 0)}
    assert first.coverage.selected_moments == 3
    assert first.coverage.overflow_decisions == 12
    assert first.coverage.dropped_actions == 12
    assert all(command.offer_id == 87 for command in first.commands)
    assert len(first.game.examples) <= 6


def test_scheduled_capture_failure_discards_both_viewers_and_resume_keeps_evidence(
    tmp_path: Path,
) -> None:
    slots = tuple(
        CaptureSlot(str(i), i, i % 2, split, ("a", "b"))
        for i, split in enumerate(("train", "validation", "test"))
    )
    runtime = _CaptureFixture(failure_at=3)
    collect_schedule(
        runtime,
        slots,
        _capture_schema(),
        directory=tmp_path,
        policy_identity="policy",
        world_identity="world",
        moments=2,
    )
    assert runtime.resets == 3
    for i in range(3):
        receipt = json.loads((tmp_path / f"game-{i:06d}.result.json").read_text())
        shard = json.loads((tmp_path / f"game-{i:06d}.game.json").read_text())
        assert receipt["status"] == "failed"
        assert shard["game"] is None
        assert len(shard["commands"]) == 4
    collect_schedule(
        runtime,
        slots,
        _capture_schema(),
        directory=tmp_path,
        policy_identity="policy",
        world_identity="world",
        moments=2,
    )
    assert runtime.resets == 3
    with pytest.raises(ValueError, match="bindings differ"):
        collect_schedule(
            runtime,
            slots,
            _capture_schema(),
            directory=tmp_path,
            policy_identity="different",
            world_identity="world",
            moments=2,
        )


def test_scheduled_capture_interruption_is_never_retried(tmp_path: Path) -> None:
    class Interrupted(_CaptureFixture):
        def reset(self, slot: CaptureSlot) -> None:
            raise OSError("worker unavailable")

    slot = CaptureSlot("one", 357, 0, "train", ("a", "b"))
    with pytest.raises(OSError, match="unavailable"):
        collect_schedule(
            Interrupted(),
            (slot,),
            _capture_schema(),
            directory=tmp_path / "main",
            policy_identity="policy",
            world_identity="world",
        )
    runtime = _CaptureFixture()
    EvidenceRoot(tmp_path).restore_capture(
        "main", reason="worker restored", read=_settled
    )
    collect_schedule(
        runtime,
        (slot,),
        _capture_schema(),
        directory=tmp_path / "main",
        policy_identity="policy",
        world_identity="world",
    )
    assert runtime.resets == 0
    receipt = json.loads((tmp_path / "main/game-000000.result.json").read_text())
    assert receipt["status"] == "interrupted"


def test_configured_offer_prefix_keeps_native_ids_and_rejects_holes() -> None:
    frame = _CaptureFixture().frame()
    assert retained_offer_count(frame, [True, True], overflow="first-legal") == 2
    assert frame.offers[1]["id"] == 87
    with pytest.raises(ValueError, match="every legal offer"):
        retained_offer_count(frame, [True, True], overflow="error")
    with pytest.raises(ValueError, match="configured legal prefix"):
        retained_offer_count(frame, [True, False], overflow="first-legal")


def test_scheduled_capture_complete_shards_round_trip_and_detect_corruption(
    tmp_path: Path,
) -> None:
    slots = tuple(
        CaptureSlot(str(i), i, i % 2, split, ("a", "b"))
        for i, split in enumerate(("train", "validation", "test"))
    )
    runtime = _CaptureFixture()
    collect_schedule(
        runtime,
        slots,
        _capture_schema(),
        directory=tmp_path,
        policy_identity="policy",
        world_identity="world",
        moments=3,
    )
    dataset = read_capture_dataset(tmp_path)
    assert {g.split for g in dataset.games} == {"train", "validation", "test"}
    assert {g.assignment for g in dataset.games} == {0, 1}
    assert all({r.viewer for r in g.examples} == {0, 1} for g in dataset.games)
    path = tmp_path / "dataset.json"
    save_dataset(dataset, path)
    assert read_dataset(path) == dataset
    shard = tmp_path / "game-000000.game.json"
    body = json.loads(shard.read_text())
    body["game"]["examples"][0]["target_hand"] = [0, 1]
    shard.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="digest mismatch"):
        read_capture_dataset(tmp_path)
    with pytest.raises(ValueError, match="corrupt"):
        collect_schedule(
            runtime,
            slots,
            _capture_schema(),
            directory=tmp_path,
            policy_identity="policy",
            world_identity="world",
            moments=3,
        )
    assert runtime.resets == 3


def test_scheduled_capture_malformed_constraints_are_fatal_before_labels() -> None:
    class Malformed(_CaptureFixture):
        def hidden_hand_constraints_json(self, viewer: int) -> str:
            data = json.loads(super().hidden_hand_constraints_json(viewer))
            data["pool"] = {"Island": -1}
            return json.dumps(data)

        def hand_counts(self, player: int) -> dict[str, int]:
            raise AssertionError("malformed constraints reached private labels")

    with pytest.raises(ValueError, match="malformed"):
        capture_game(
            Malformed(),
            CaptureSlot("one", 1, 0, "train", ("a", "b")),
            _capture_schema(),
            omit_unsupported=True,
        )


def test_scheduled_capture_pairs_cannot_cross_game_splits(tmp_path: Path) -> None:
    slots = (
        CaptureSlot("a", 1, 0, "train", ("a", "b")),
        CaptureSlot("b", 1, 1, "test", ("a", "b")),
    )
    runtime = _CaptureFixture()
    with pytest.raises(ValueError, match="cannot cross splits"):
        collect_schedule(
            runtime,
            slots,
            _capture_schema(),
            directory=tmp_path,
            policy_identity="policy",
            world_identity="world",
        )
    assert runtime.resets == 0


def test_capture_runtime_restores_repeated_deal_seed() -> None:
    match = Match(
        MatchConfig(
            hero_deck={"Mountain": 8, "Raging Goblin": 8},
            villain_deck={"Mountain": 8, "Raging Goblin": 8},
        )
    )
    space = ObservationSpace(ObservationConfig())
    agent = ActorCritic(space, ModelConfig(hidden_dim=8))
    env = Env(match, space, Reward(RewardConfig()), seed=357)
    runtime = EnvCaptureRuntime(env, AgentMatchupPlayer(agent), lambda slot: match)
    try:
        slot = CaptureSlot("first", 357, 0, "train", ("same", "same"))
        runtime.reset(slot)
        first = runtime.observation(0).viewer_state
        runtime.reset(replace(slot, game_id="second", assignment=1))
        assert runtime.observation(0).viewer_state == first
    finally:
        env.close()


def _settled(refs: tuple[str, ...]) -> tuple[JobSettlement, ...]:
    return tuple(
        JobSettlement(
            run=ref,
            writers_closed=True,
            usd=Decimal("0.1"),
            receipt_identity=f"receipt-{ref}",
        )
        for ref in refs
    )


def _collected_root(tmp_path: Path) -> EvidenceRoot:
    root = EvidenceRoot(tmp_path)
    slots = tuple(
        CaptureSlot(str(i), i, i % 2, split, ("a", "b"))
        for i, split in enumerate(("train", "validation", "test"))
    )
    collect_schedule(
        _CaptureFixture(),
        slots,
        _capture_schema(),
        directory=tmp_path / "main",
        policy_identity="policy",
        world_identity="world",
        moments=2,
    )
    return root


def test_evidence_root_costs_unknown_writers_and_cumulative_admission(
    tmp_path: Path,
) -> None:
    root = EvidenceRoot(tmp_path)

    def full_costs(refs: tuple[str, ...]) -> tuple[JobSettlement, ...]:
        costs = {"admit": Decimal(2), "main": Decimal(23)}
        return tuple(
            result.model_copy(update={"usd": costs[result.run]})
            for result in _settled(refs)
        )

    root.reserve(
        JobAllocation(run="admit", phase="admission", max_usd=Decimal(2)), full_costs
    )
    with pytest.raises(ValueError, match="already allocated"):
        root.reserve(
            JobAllocation(run="admit", phase="admission", max_usd=Decimal(2)),
            full_costs,
        )
    with pytest.raises(ValueError, match="USD2"):
        root.reserve(
            JobAllocation(run="admit-again", phase="admission", max_usd=Decimal("0.1")),
            full_costs,
        )
    with pytest.raises(ValueError, match="exactly once"):
        root.reconcile(lambda refs: ())
    for closed, cost in ((False, Decimal(0)), (True, None)):

        def unknown(refs: tuple[str, ...]) -> tuple[JobSettlement, ...]:
            return (
                JobSettlement(
                    run=refs[0],
                    writers_closed=closed,
                    usd=cost,
                    receipt_identity="unknown",
                ),
            )

        with pytest.raises(ValueError, match="unknown"):
            root.reserve(
                JobAllocation(run="main", phase="collection", max_usd=Decimal(1)),
                unknown,
            )
    root.reserve(
        JobAllocation(run="main", phase="collection", max_usd=Decimal(23)), full_costs
    )
    with pytest.raises(ValueError, match="USD25"):
        root.reserve(
            JobAllocation(run="fit", phase="fit", max_usd=Decimal("0.1")), full_costs
        )
    assert len(root.allocations()) == 2


def test_collection_closure_requires_no_writer_and_freezes_sources(
    tmp_path: Path,
) -> None:
    root = _collected_root(tmp_path)
    root.reserve(
        JobAllocation(run="capture", phase="collection", max_usd=Decimal(1)), _settled
    )
    with pytest.raises(FileNotFoundError):
        root.freeze_dataset()

    def live(refs: tuple[str, ...]) -> tuple[JobSettlement, ...]:
        return (
            JobSettlement(
                run=refs[0],
                writers_closed=False,
                usd=Decimal(0),
                receipt_identity="still-live",
            ),
        )

    with pytest.raises(ValueError, match="writers"):
        root.close_collection("main", reason="finished", read=live)
    assert not (tmp_path / "main/closure.json").exists()
    root.close_collection("main", reason="finished", read=_settled)
    dataset = root.freeze_dataset()
    assert root.dataset() == dataset
    assert root.freeze_dataset() == dataset
    assert [g.game_id for g in dataset.games] == ["0", "1", "2"]
    with pytest.raises(ValueError, match="closed"):
        root.reserve(
            JobAllocation(run="more", phase="collection", max_usd=Decimal(1)), _settled
        )
    (tmp_path / "main/game-000003.started.json").write_text("{}")
    with pytest.raises(ValueError, match="gained or lost"):
        root.dataset()


def test_fatal_stop_retains_prefix_and_requires_restoration(tmp_path: Path) -> None:
    class Fatal(_CaptureFixture):
        def step(self, command: Command) -> CaptureTransition:
            if self.revision == 2:
                raise ValueError("injected schema prerequisite")
            return super().step(command)

    slots = tuple(CaptureSlot(str(i), i, i % 2, "train", ("a", "b")) for i in range(2))
    args = dict(
        directory=tmp_path / "main", policy_identity="policy", world_identity="world"
    )
    with pytest.raises(ValueError, match="prerequisite"):
        collect_schedule(Fatal(), slots, _capture_schema(), **args)
    stop = json.loads((tmp_path / "main/game-000000.stop.json").read_text())
    assert stop["error_type"] == "ValueError" and len(stop["commands"]) == 3
    runtime = _CaptureFixture()
    with pytest.raises(ValueError, match="restored"):
        collect_schedule(runtime, slots, _capture_schema(), **args)
    root = EvidenceRoot(tmp_path)
    root.restore_capture("main", reason="schema restored and verified", read=_settled)
    collect_schedule(runtime, slots, _capture_schema(), **args)
    assert runtime.resets == 1
    root.close_collection("main", reason="declared stop", read=_settled)
    closure = json.loads((tmp_path / "main/closure.json").read_text())
    assert closure["statuses"] == ["interrupted", "completed"]


def test_closure_keeps_orphan_failed_zero_row_and_untouched_denominators(
    tmp_path: Path,
) -> None:
    root = _collected_root(tmp_path)
    directory = tmp_path / "main"
    # Crash after shard delivery: the missing terminal receipt excludes that game.
    (directory / "game-000002.result.json").unlink()
    (directory / "game-000001.started.json").unlink()
    (directory / "game-000001.result.json").unlink()
    (directory / "game-000001.game.json").unlink()
    root.close_collection("main", reason="worker stopped", read=_settled)
    closure = json.loads((directory / "closure.json").read_text())
    assert closure["statuses"] == ["completed", "unattempted", "interrupted"]
    with pytest.raises(ValueError, match="three nonempty"):
        root.freeze_dataset()
    assert not (tmp_path / "dataset-binding.json").exists()


def test_native_capture_failure_classification_keeps_contract_errors_fatal() -> None:
    class StepFailure:
        def step_semantic(self, command: Command) -> None:
            raise SemanticContractError("rules") from managym.AgentError(
                "no legal targets"
            )

    runtime = object.__new__(EnvCaptureRuntime)
    runtime.env = StepFailure()  # type: ignore[assignment] -- injected native boundary
    command = Command("test", 0, 1)
    with pytest.raises(GameCaptureFailure, match="no legal targets"):
        runtime.step(command)

    class Stale:
        def step_semantic(self, command: Command) -> None:
            raise SemanticContractError(
                "stale_capture: rejected"
            ) from managym.AgentError("stale_capture: rejected")

    runtime.env = Stale()  # type: ignore[assignment] -- injected native boundary
    with pytest.raises(SemanticContractError, match="stale_capture"):
        runtime.step(command)


def test_completed_zero_row_games_keep_coverage_and_both_omission_reasons(
    tmp_path: Path,
) -> None:
    class Unsupported(_CaptureFixture):
        def hidden_hand_constraints_json(self, viewer: int) -> str:
            payload = json.loads(super().hidden_hand_constraints_json(viewer))
            payload["retained_selection"] = True
            payload["foretold"] = ["native constraint"]
            return json.dumps(payload)

        def hand_counts(self, player: int) -> dict[str, int]:
            raise AssertionError("unsupported public row accessed a private label")

    root = EvidenceRoot(tmp_path)
    slots = tuple(
        CaptureSlot(str(i), i, i % 2, split, ("a", "b"))
        for i, split in enumerate(("train", "validation", "test"))
    )
    collect_schedule(
        Unsupported(),
        slots,
        _capture_schema(),
        directory=tmp_path / "main",
        policy_identity="policy",
        world_identity="world",
    )
    root.close_collection(
        "main", reason="complete games, unsupported rows", read=_settled
    )
    closure = json.loads((tmp_path / "main/closure.json").read_text())
    assert closure["statuses"] == ["completed"] * 3
    shard = json.loads((tmp_path / "main/game-000000.game.json").read_text())
    assert shard["game"] is None and shard["failure"] is None
    assert shard["coverage"]["retained_rows"] == [0, 0]
    assert shard["coverage"]["unsupported_rows"] == [12, 12]
    assert shard["coverage"]["omission_reasons"] == {
        "retained_selection": [12, 12],
        "foretold": [12, 12],
    }
    with pytest.raises(ValueError, match="three nonempty"):
        root.freeze_dataset()
