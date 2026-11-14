"""Whole-game immutable evidence and real frozen-policy collection checks."""

import base64
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from botocore.exceptions import ClientError
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.belief.sampling_collection import (
    CaptureCoverage,
    CaptureSlot,
    CaptureTransition,
    GameAccumulator,
    GameCaptureFailure,
    capture_game,
    close_capture_collection,
    collect_schedule,
    freeze_capture_dataset,
    read_capture_dataset,
    record_capture_repair,
    retained_offer_count,
)
from manabot.belief.sampling_data import (
    BehaviorIdentity,
    BehaviorSegment,
    GameKey,
    GroupSplit,
    ProducerSelection,
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
    SplitPlan,
    collect_frozen_policy,
    public_sampler_input,
    read_dataset,
    save_dataset,
)
from manabot.belief.sampling_publication import (
    create_producer,
    prepare_snapshot,
    publish_pending_snapshot,
    read_producer,
    retain_failure,
    retain_game,
)
from manabot.belief.sampling_runtime import EnvCaptureRuntime
from manabot.belief.sampling_snapshot import (
    SealedGame,
    publish_snapshot,
    seal_game,
    seal_snapshot,
    validate_snapshot_groups,
)
from manabot.belief.sampling_types import SamplerInput, SamplerSchema, _digest
from manabot.belief.state import ViewerHistory
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.artifacts import ArtifactStore
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
    close_capture_collection(tmp_path / "main", reason="worker ended")
    with pytest.raises(ValueError, match="closed"):
        collect_schedule(
            runtime,
            (slot,),
            _capture_schema(),
            directory=tmp_path / "main",
            policy_identity="policy",
            world_identity="world",
        )
    assert runtime.resets == 0
    closure = json.loads((tmp_path / "main/closure.json").read_text())
    assert closure["statuses"] == ["interrupted"]


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


def _collected_directory(
    tmp_path: Path, slots: tuple[CaptureSlot, ...] | None = None
) -> Path:
    slots = slots or tuple(
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
    return tmp_path


def test_fatal_stop_retains_prefix_through_collection_closure(tmp_path: Path) -> None:
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
    record_capture_repair(tmp_path / "main", reason="schema prerequisite repaired")
    collect_schedule(runtime, slots, _capture_schema(), **args)
    assert runtime.resets == 1
    close_capture_collection(tmp_path / "main", reason="fatal worker ended")
    closure = json.loads((tmp_path / "main/closure.json").read_text())
    assert closure["statuses"] == ["interrupted", "completed"]
    assert json.loads((tmp_path / "main/game-000000.stop.json").read_text()) == stop


def test_closure_keeps_orphan_failed_zero_row_and_untouched_denominators(
    tmp_path: Path,
) -> None:
    _collected_directory(tmp_path)
    directory = tmp_path / "main"
    # Crash after shard delivery: the missing terminal receipt excludes that game.
    (directory / "game-000002.result.json").unlink()
    (directory / "game-000001.started.json").unlink()
    (directory / "game-000001.result.json").unlink()
    (directory / "game-000001.game.json").unlink()
    close_capture_collection(directory, reason="worker stopped")
    closure = json.loads((directory / "closure.json").read_text())
    assert closure["statuses"] == ["completed", "unattempted", "interrupted"]
    with pytest.raises(ValueError, match="three nonempty"):
        freeze_capture_dataset(tmp_path)
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
    close_capture_collection(
        tmp_path / "main", reason="complete games, unsupported rows"
    )
    closure = json.loads((tmp_path / "main/closure.json").read_text())
    assert closure["statuses"] == ["completed"] * 3
    shard = json.loads((tmp_path / "main/game-000000.game.json").read_text())
    assert shard["game"]["examples"] == [] and shard["failure"] is None
    assert shard["coverage"]["retained_rows"] == [0, 0]
    assert shard["coverage"]["unsupported_rows"] == [12, 12]
    assert shard["coverage"]["omission_reasons"] == {
        "retained_selection": [12, 12],
        "foretold": [12, 12],
    }
    with pytest.raises(ValueError, match="three nonempty"):
        freeze_capture_dataset(tmp_path)


def _sealed_fixture(
    ordinal: int,
    split: Split,
    *,
    run: str = "producer",
    update: int = 1,
    rows: int = 1,
) -> SealedGame:
    """Synthetic successful closure; no game stepping or optimizer work."""
    key = GameKey(run, "original-attempt", 0, ordinal)
    dataset = _dataset()
    behavior = BehaviorIdentity(
        run, "source", "runtime", "settings", f"weights-{update}", "raw", update
    )
    game = SamplerGame(
        key.identity,
        357,
        0,
        split,
        tuple(replace(dataset.games[0].examples[0], revision=i) for i in range(rows)),
    )
    decisions = max(1, rows)
    return SealedGame(
        key,
        f"group-{ordinal}",
        dataset.schema,
        dataset.world_identity,
        (BehaviorSegment(0, decisions, (behavior, behavior)),),
        game,
        CaptureCoverage(
            decisions,
            (rows, 0),
            (decisions - rows, decisions),
            rows,
            0,
            0,
            {"retained_selection": (decisions - rows, decisions)},
            (rows, 0),
        ),
        tuple(Command(f"command-{i}", i, 1) for i in range(decisions)),
    )


def _snapshot_fixture(tmp_path: Path) -> tuple[SamplerDataset, Path]:
    shards = tuple(
        _sealed_fixture(i, split, rows=i)
        for i, split in enumerate(("train", "train", "validation", "test"))
    )
    members = tuple(seal_game(tmp_path / "games", game) for game in shards)
    plan = SplitPlan(
        "family",
        357,
        (
            *tuple(GroupSplit(game.group, game.game.split) for game in shards),
            GroupSplit("group-4", "train"),
        ),
    )
    path = tmp_path / "snapshot-a.json"
    dataset = seal_snapshot(
        tmp_path / "games",
        members,
        selection=(ProducerSelection("producer", 1, 1, ("raw",)),),
        split_plan=plan,
        watermark="update-1",
        destination=path,
    )
    return dataset, path


def test_snapshot_a_is_immutable_as_b_adds_behavior_and_zero_row_games(
    tmp_path: Path,
) -> None:
    first, path = _snapshot_fixture(tmp_path)
    before = path.read_bytes()
    assert first.policy_identity is None
    assert len(first.games) == 4 and not first.games[0].examples
    assert read_dataset(path) == first
    assert first.binding is not None
    later = _sealed_fixture(4, "train", update=2)
    member = seal_game(tmp_path / "games", later)
    plan = first.binding.split_plan
    second = seal_snapshot(
        tmp_path / "games",
        (*first.binding.games, member),
        selection=(ProducerSelection("producer", 1, 2, ("raw",)),),
        split_plan=plan,
        watermark="update-2",
        destination=tmp_path / "snapshot-b.json",
        previous=(first.binding,),
    )
    assert len(second.games) == 5
    assert second.binding is not None and len(second.binding.producers) == 2
    assert read_dataset(path).identity == first.identity
    assert path.read_bytes() == before
    # Independent runs may repeat deal seeds, without claiming dependent games.
    assert len({(game.seed, game.assignment) for game in second.games}) == 1


def test_sealed_publication_is_idempotent_and_conflicts_are_corruption(
    tmp_path: Path,
) -> None:
    game = _sealed_fixture(0, "train")
    first = seal_game(tmp_path, game)
    assert seal_game(tmp_path, game) == first
    changed = replace(
        game,
        game=replace(
            game.game,
            examples=(replace(game.game.examples[0], observation_identity="changed"),),
        ),
    )
    with pytest.raises(ValueError, match="conflicting bytes"):
        seal_game(tmp_path, changed)
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(ValueError, match="only completed"):
        replace(game, status="failed")  # type: ignore[arg-type] -- corrupt producer boundary
    with pytest.raises(ValueError, match="cover every"):
        replace(game, commands=())


def test_snapshot_selection_checks_both_seats_and_all_spanning_segments(
    tmp_path: Path,
) -> None:
    base = _sealed_fixture(0, "train", rows=2)
    first = base.segments[0].seats[0]
    later = replace(first, update=2, weights_identity="later-weights")
    builtin = BehaviorIdentity(
        "builtin", "source", "runtime", "settings", "heuristic-code", "builtin", None
    )
    spanning = replace(
        base,
        segments=(
            BehaviorSegment(0, 1, (first, builtin)),
            BehaviorSegment(1, 2, (later, builtin)),
        ),
    )
    member = seal_game(tmp_path / "games", spanning)
    plan = SplitPlan("family", 357, (GroupSplit(base.group, "train"),))
    arguments = dict(
        split_plan=plan, watermark="2", destination=tmp_path / "selected.json"
    )
    with pytest.raises(ValueError, match="no completed"):
        seal_snapshot(
            tmp_path / "games",
            (member,),
            selection=(
                ProducerSelection("producer", 1, 1, ("raw",)),
                ProducerSelection("builtin", None, None, ("builtin",)),
            ),
            **arguments,
        )
    with pytest.raises(ValueError, match="no completed"):
        seal_snapshot(
            tmp_path / "games",
            (member,),
            selection=(ProducerSelection("producer", 1, 2, ("raw",)),),
            **arguments,
        )
    selected = seal_snapshot(
        tmp_path / "games",
        (member, member),
        selection=(
            ProducerSelection("producer", 1, 2, ("raw",)),
            ProducerSelection("builtin", None, None, ("builtin",)),
        ),
        **arguments,
    )
    assert len(selected.games) == 1
    assert selected.binding is not None and len(selected.binding.producers) == 3


def test_snapshot_rejects_missing_corrupt_and_conflicting_shards(
    tmp_path: Path,
) -> None:
    dataset, path = _snapshot_fixture(tmp_path)
    assert dataset.binding is not None
    payload = json.loads(path.read_text())
    payload["shards"][1]["game"]["examples"][0]["target_hand"] = [2, 0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="byte identity"):
        read_dataset(path)
    first = dataset.binding.games[0]
    args = dict(
        selection=dataset.binding.selection,
        split_plan=dataset.binding.split_plan,
        watermark="cut",
        destination=tmp_path / "invalid.json",
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        seal_snapshot(
            tmp_path / "games", (first, replace(first, sha256="0" * 64)), **args
        )
    (tmp_path / "games" / f"{first.key.identity}.json").unlink()
    (tmp_path / "games" / f"{first.key.identity}.partial").write_text("unfinished")
    with pytest.raises(FileNotFoundError):
        seal_snapshot(tmp_path / "games", (first,), **args)
    assert not (tmp_path / "invalid.json").exists()


def test_snapshot_split_checks_span_snapshots_and_producer_runs(tmp_path: Path) -> None:
    dataset, _ = _snapshot_fixture(tmp_path)
    assert dataset.binding is not None
    binding = dataset.binding
    member = binding.games[0]
    plan = replace(
        binding.split_plan,
        overrides=tuple(
            GroupSplit(x.group, "test" if x.group == member.group else x.split)
            for x in binding.split_plan.overrides
        ),
    )
    changed = replace(
        binding,
        split_plan=plan,
        games=(replace(member, split="test"), *binding.games[1:]),
    )
    with pytest.raises(ValueError, match="crosses splits"):
        validate_snapshot_groups((binding, changed))
    # A new producer can share an explicitly declared dependent deal group.
    independent = replace(
        member, key=GameKey("another-run", "attempt", 1, 0), split="test"
    )
    changed = replace(changed, games=(independent, *changed.games[1:]))
    with pytest.raises(ValueError, match="crosses splits"):
        validate_snapshot_groups((binding, changed))


def test_v1_identity_keeps_original_serialized_meaning(tmp_path: Path) -> None:
    dataset = _dataset()
    original = asdict(dataset)
    del original["binding"]
    assert dataset.identity == _digest(original)
    path = tmp_path / "v1.json"
    save_dataset(dataset, path)
    assert "binding" not in json.loads(path.read_text())["dataset"]
    assert read_dataset(path).identity == _digest(original)


def test_snapshot_storage_failure_is_not_publication_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, path = _snapshot_fixture(tmp_path)
    objects: dict[str, bytes] = {}

    class Bucket:
        def put_object(self, **request: Any) -> dict[str, str]:
            body = request["Body"].read()
            assert (
                request["ChecksumSHA256"]
                == base64.b64encode(hashlib.sha256(body).digest()).decode()
            )
            assert request["ContentLength"] == len(body)
            objects[request["Key"]] = body
            return {}

    store = ArtifactStore()
    monkeypatch.setattr(store, "_client", lambda scheme: Bucket())
    reference = publish_snapshot(path, "s3://private/snapshots", store=store)
    assert objects[f"snapshots/sha256/{reference.sha256}"] == path.read_bytes()
    assert reference.bytes == path.stat().st_size
    assert reference.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset.binding is not None

    class Unavailable(Bucket):
        def put_object(self, **request: Any) -> dict[str, str]:
            raise OSError("storage acknowledgement unavailable")

    monkeypatch.setattr(store, "_client", lambda scheme: Unavailable())
    with pytest.raises(OSError, match="acknowledgement"):
        publish_snapshot(path, "s3://private/snapshots", store=store)
    assert read_dataset(path).identity == dataset.identity


def test_passive_accumulation_preserves_policy_rng_and_capture_results() -> None:
    runtime = _CaptureFixture()
    slot = CaptureSlot("passive", 357, 0, "train", ("a", "b"))
    scalar = capture_game(
        runtime, slot, _capture_schema(), moments=3, omit_unsupported=True
    )
    other = _CaptureFixture()
    other.reset(slot)
    accumulator = GameAccumulator(
        slot,
        _capture_schema(),
        (other.observation(0), other.observation(1)),
        moments=3,
        omit_unsupported=True,
    )
    rng = torch.random.get_rng_state().clone()
    commands: list[Command] = []
    for step in range(12):
        accumulator.capture_decision(other)
        frame = other.frame()
        action = other.act()
        accumulator.record_offer(len(frame.offers), other.retained_actions)
        command = Command(
            f"sampler-{slot.game_id}-{step}",
            frame.revision,
            int(frame.offers[action]["id"]),
        )
        commands.append(command)
        transition = other.step(command)
        accumulator.advance(
            transition.receipt,
            (other.observation(0), other.observation(1)),
            acting=frame.actor,
        )
        if transition.terminated:
            break
    assert accumulator.finish(tuple(commands), terminated=True) == scalar
    with pytest.raises(ValueError, match="successful native termination"):
        accumulator.finish(tuple(commands), terminated=False)
    assert torch.equal(torch.random.get_rng_state(), rng)


def _producer_fixture(directory: Path) -> SplitPlan:
    plan = SplitPlan(
        "family", 357, tuple(GroupSplit(f"group-{i}", "train") for i in range(4))
    )
    game = _sealed_fixture(0, "train")
    create_producer(
        directory,
        run="producer",
        schema=game.schema,
        world_identity=game.world_identity,
        split_plan=plan,
    )
    return plan


def test_producer_reopen_keeps_outputs_after_learner_rollback(tmp_path: Path) -> None:
    plan = _producer_fixture(tmp_path)
    old_learner_output_cursor = read_producer(tmp_path).watermark
    first = retain_game(tmp_path, _sealed_fixture(0, "train", rows=0))
    retain_failure(tmp_path, "a" * 64)
    retain_failure(tmp_path, "a" * 64)
    later = retain_game(tmp_path, _sealed_fixture(1, "train", update=2))
    # Unacknowledged bytes, including a complete orphan, are never discovered.
    seal_game(tmp_path / "games", _sealed_fixture(2, "train"))
    (tmp_path / "games" / "unfinished.partial").write_text("incomplete")
    reopened = create_producer(
        tmp_path,
        run="producer",
        schema=_sealed_fixture(0, "train").schema,
        world_identity=_sealed_fixture(0, "train").world_identity,
        split_plan=plan,
    )
    assert reopened.watermark != old_learner_output_cursor
    assert reopened.members == (first, later)
    assert reopened.failure_references == ("a" * 64,)
    assert retain_game(tmp_path, _sealed_fixture(1, "train", update=2)) == later
    pending = prepare_snapshot(
        tmp_path,
        selection=(ProducerSelection("producer", 1, 2, ("raw",)),),
        destination="s3://private/snapshots",
    )
    assert pending.binding.games == (first, later)
    assert pending.binding.failure_references == ("a" * 64,)
    assert read_producer(tmp_path).verified == ()
    with pytest.raises(ValueError, match="conflicting bytes"):
        retain_game(tmp_path, _sealed_fixture(1, "train", update=3))
    assert read_producer(tmp_path).members == (first, later)


def test_producer_fixed_splits_and_bindings_are_not_replaced(tmp_path: Path) -> None:
    plan = _producer_fixture(tmp_path)
    game = _sealed_fixture(0, "train")
    with pytest.raises(ValueError, match="prospective split"):
        retain_game(tmp_path, replace(game, game=replace(game.game, split="test")))
    with pytest.raises(ValueError, match="bindings differ"):
        create_producer(
            tmp_path,
            run="producer",
            schema=game.schema,
            world_identity=game.world_identity,
            split_plan=replace(plan, seed=358),
        )
    assert read_producer(tmp_path).members == ()
    with pytest.raises(FileNotFoundError):
        read_producer(tmp_path / "missing")
    (tmp_path / "producer.json").unlink()
    with pytest.raises(FileNotFoundError, match="index is missing"):
        create_producer(
            tmp_path,
            run="producer",
            schema=game.schema,
            world_identity=game.world_identity,
            split_plan=plan,
        )


def test_producer_lost_storage_reply_reconciles_same_pack_and_keeps_a_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _producer_fixture(tmp_path)
    retain_game(tmp_path, _sealed_fixture(0, "train"))
    selection = (ProducerSelection("producer", 1, 2, ("raw",)),)
    pending = prepare_snapshot(
        tmp_path, selection=selection, destination="s3://private/snapshots"
    )
    path = tmp_path / "snapshots" / f"{pending.sha256}.json"
    original = path.read_bytes()
    first_dataset = read_dataset(path)
    objects: dict[str, bytes] = {}

    class LostReplyBucket:
        def put_object(self, **request: Any) -> dict[str, str]:
            key = request["Key"]
            if key in objects:
                raise ClientError(
                    {
                        "Error": {"Code": "PreconditionFailed"},
                        "ResponseMetadata": {"HTTPStatusCode": 412},
                    },
                    "PutObject",
                )
            objects[key] = request["Body"].read()
            raise OSError("storage accepted bytes but reply was lost")

        def head_object(self, **request: Any) -> dict[str, object]:
            data = objects[request["Key"]]
            return {
                "ContentLength": len(data),
                "ChecksumSHA256": base64.b64encode(
                    hashlib.sha256(data).digest()
                ).decode(),
            }

    store = ArtifactStore()
    monkeypatch.setattr(store, "_client", lambda scheme: LostReplyBucket())
    with pytest.raises(OSError, match="reply was lost"):
        publish_pending_snapshot(tmp_path, store=store)
    assert read_producer(tmp_path).snapshots == (pending,)
    assert read_producer(tmp_path).verified == ()
    assert (
        prepare_snapshot(
            tmp_path, selection=selection, destination="s3://private/snapshots"
        )
        == pending
    )
    # Later acknowledged games cannot replace a pending cut during recovery.
    retain_game(tmp_path, _sealed_fixture(1, "train", update=2))
    with pytest.raises(ValueError, match="reconcile pending"):
        prepare_snapshot(
            tmp_path, selection=selection, destination="s3://private/snapshots"
        )
    # Storage verification can succeed while saving the local acknowledgement
    # fails too. Reopening must still expose only the same pending object.
    with monkeypatch.context() as fault:

        def lose_local_ack(source: object, target: object) -> None:
            raise OSError("local acknowledgement was lost")

        fault.setattr(os, "replace", lose_local_ack)
        with pytest.raises(OSError, match="local acknowledgement"):
            publish_pending_snapshot(tmp_path, store=store)
    assert read_producer(tmp_path).verified == ()
    assert read_producer(tmp_path).snapshots == (pending,)
    reference = publish_pending_snapshot(tmp_path, store=store)
    assert len(objects) == 1
    assert reference.sha256 == pending.sha256
    assert reference.bytes == len(original)
    assert read_producer(tmp_path).verified == (reference,)
    assert publish_pending_snapshot(tmp_path, store=store) == reference
    second = prepare_snapshot(
        tmp_path, selection=selection, destination="s3://private/snapshots"
    )
    assert len(second.binding.games) == 2
    assert second.binding.watermark != pending.binding.watermark
    assert read_producer(tmp_path).verified == (reference,)
    assert path.read_bytes() == original
    assert read_dataset(path).identity == first_dataset.identity


def test_producer_corrupt_pending_pack_cannot_be_advertised(tmp_path: Path) -> None:
    _producer_fixture(tmp_path)
    retain_game(tmp_path, _sealed_fixture(0, "train"))
    pending = prepare_snapshot(
        tmp_path,
        selection=(ProducerSelection("producer", 1, 1, ("raw",)),),
        destination="s3://private/snapshots",
    )
    (tmp_path / "snapshots" / f"{pending.sha256}.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes differ"):
        publish_pending_snapshot(tmp_path, store=ArtifactStore())
    assert read_producer(tmp_path).verified == ()


def test_producer_failed_membership_write_does_not_adopt_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _producer_fixture(tmp_path)
    game = _sealed_fixture(0, "train")
    with monkeypatch.context() as fault:

        def fail_membership_write(source: object, target: object) -> None:
            raise OSError("membership persistence failed")

        fault.setattr(os, "replace", fail_membership_write)
        with pytest.raises(OSError, match="membership persistence"):
            retain_game(tmp_path, game)
    assert (tmp_path / "games" / f"{game.key.identity}.json").exists()
    assert read_producer(tmp_path).members == ()
    with pytest.raises(ValueError, match="no completed games"):
        prepare_snapshot(
            tmp_path,
            selection=(ProducerSelection("producer", 1, 1, ("raw",)),),
            destination="s3://private/snapshots",
        )
    member = retain_game(tmp_path, game)
    assert read_producer(tmp_path).members == (member,)
