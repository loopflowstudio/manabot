"""Strict paths retain replay inputs without opting into tolerance or recovery."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from managym.decision import Command
import numpy as np
import pytest
import torch

from manabot.env import Env, Match, ObservationSpace, Reward, VectorEnv
from manabot.env.failures import (
    CaptureScope,
    capture_scope,
    read_failure,
    runtime_identities,
)
from manabot.env.replay import require_replay_runtime
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.sim import datagen, rollout
from manabot.sim.pooled_datagen import _GameStream
from manabot.training import execution
from manabot.training.regime import (
    Learning,
    SelectionGameSpec,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout import compound, selection_data
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.verify.store import VerifyStore
import managym
from tests.training.game_error_fixtures import (
    PaymentMismatchEnv,
    _bounded_fixture_replay,
    _payment_match,
)


def _model(*, compound_decisions: bool = False) -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(),
        ModelConfig(
            hidden_dim=8, attention_on=False, compound_decisions=compound_decisions
        ),
    )


def _env() -> Env:
    return Env(Match(), ObservationSpace(), Reward(RewardConfig()), seed=24)


def _bounded(path: Path) -> dict[str, Any]:
    before = path.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "manabot.training.state.game_errors",
            str(path),
            "--attempt",
            "1",
            "--seconds",
            "15",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == before
    assert path.stat().st_mode & 0o777 == 0o600
    return json.loads(result.stdout)


@pytest.mark.parametrize("field", ["world", "engine_extension_sha256", "source_sha256"])
@pytest.mark.parametrize("allow_changed_runtime", [False, True])
def test_replay_runtime_admission(field: str, allow_changed_runtime: bool) -> None:
    identities = {**runtime_identities(), field: "changed"}
    if field == "world" or not allow_changed_runtime:
        with pytest.raises(ValueError, match="differs|bytes differ"):
            require_replay_runtime(
                identities,
                allow_changed_runtime=allow_changed_runtime,
                require_source=True,
            )
    else:
        assert (
            require_replay_runtime(
                identities, allow_changed_runtime=True, require_source=True
            )
            == runtime_identities()["engine_extension_sha256"]
        )


def test_replay_requires_source_only_for_automatic_records() -> None:
    identities = runtime_identities().copy()
    del identities["source_sha256"]
    assert (
        require_replay_runtime(identities, allow_changed_runtime=False)
        == identities["engine_extension_sha256"]
    )
    with pytest.raises(KeyError, match="source_sha256"):
        require_replay_runtime(
            identities, allow_changed_runtime=False, require_source=True
        )


def test_scalar_prefix_and_reset_capture(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path, lambda: {"update": 7, "allocation_deadline": 123.0})
    with capture_scope(scope):
        env = _env()
        env.reset()
        env.step(0)
        with pytest.raises(managym.AgentError) as failure, env._capture.guard():
            env.step(-1)
    assert len(scope.paths) == 1
    evidence = read_failure(scope.paths[0])
    assert evidence.games[0].actions == [0, -1]
    assert evidence.context["update"] == 7
    assert evidence.context["allocation_deadline"] == 123.0
    assert evidence.error == str(failure.value)
    assert evidence.identities["input_shapes"]
    assert evidence.identities["content_manifest"]
    result = _bounded(scope.paths[0])["results"][0]
    assert result["at_retained_failure"] and result["matches_error"]

    with capture_scope(scope), pytest.raises(BaseException):
        env._capture.reset(env._engine, [])
    assert read_failure(scope.paths[-1]).operation == "reset"
    assert _bounded(scope.paths[-1])["results"][0]["matches_error"]


@pytest.mark.parametrize("mode", ["scalar", "vector", "search"])
@pytest.mark.parametrize("variant", ["wrong_frame", "different_error", "missing"])
def test_bounded_replay_separates_binding_and_error_agreement(
    tmp_path: Path, mode: str, variant: str
) -> None:
    scope = CaptureScope(tmp_path / "original")
    with capture_scope(scope):
        if mode == "vector":
            vector = VectorEnv(
                1, Match(), ObservationSpace(), Reward(RewardConfig()), "cpu", seed=24
            )
            vector.reset()
            with pytest.raises(managym.AgentError):
                vector.step(torch.tensor([-1]))
        else:
            env = _env()
            env.reset()
            if mode == "search":
                pool = env.rollout_pool(1, 1, 197, 100)
                buffers = rollout._allocate_buffers(ObservationSpace(), pool.num_slots)
                pool.set_buffers(buffers, pool.num_slots)
                pool.encode_active()
                with pytest.raises(managym.AgentError):
                    pool.step_active([-1] * pool.num_slots)
            else:
                with pytest.raises(managym.AgentError):
                    env.step(-1)
    original = scope.paths[0].read_bytes()
    evidence = read_failure(scope.paths[0])
    if variant == "wrong_frame":
        evidence.games[0].frame = "{}"
    elif variant == "different_error":
        evidence.error = "another rejection"
    else:
        evidence.games[0].frame = evidence.error_class = evidence.error = None
    diagnostic = tmp_path / "diagnostic.jsonl"
    diagnostic.touch(mode=0o600)
    diagnostic.write_text(evidence.model_dump_json(by_alias=True) + "\n")
    result = _bounded(diagnostic)["results"][0]
    assert result["status"] == (
        "binding_mismatch" if variant == "wrong_frame" else "failed"
    )
    assert result["matches_binding"] is (
        None if variant == "missing" else variant != "wrong_frame"
    )
    assert result["at_retained_failure"] == (variant != "wrong_frame")
    assert result["matches_error"] is (False if variant == "different_error" else None)
    assert evidence.execution_outcome == "failed"
    assert result["original_outcome"] == (
        "execution_outcome_unknown" if mode == "vector" else "failed"
    )
    assert scope.paths[0].read_bytes() == original


@pytest.mark.parametrize("policy", ["none", "random", "passive"])
def test_ordinary_vector_replays_current_game_after_autoreset(
    tmp_path: Path,
    policy: str,
) -> None:
    scope = CaptureScope(tmp_path)
    with capture_scope(scope):
        env = VectorEnv(
            2,
            Match(),
            ObservationSpace(),
            Reward(RewardConfig()),
            "cpu",
            seed=24,
            opponent_policy=policy,
        )
        env.reset()
        for _ in range(2000):
            _, _, done, truncated, _ = env.step(torch.zeros(2, dtype=torch.int64))
            if bool((done | truncated).any()):
                break
        else:
            pytest.fail("fixture failed to reach native auto-reset")
        seeds = env._rust_env.current_game_seeds()
        with pytest.raises(managym.AgentError):
            env.step(torch.tensor([-1, 0]))
    evidence = read_failure(scope.paths[0])
    assert [game.seed for game in evidence.games] == seeds
    assert evidence.seed_stride == 2
    assert all(game.frame for game in evidence.games)
    assert evidence.identities["input_buffers"]
    replay = _bounded(scope.paths[0])
    assert replay["original_outcome"] == "failed"
    result = replay["results"]
    assert result[0]["matches_error"] and result[0]["at_retained_failure"]
    assert result[1]["status"] == "advanced"  # not counted as another crash
    assert all(row["original_outcome"] == "execution_outcome_unknown" for row in result)


@pytest.mark.parametrize("workers", [1, 2])
def test_strict_collector_preserves_masked_prefixes(
    tmp_path: Path, workers: int
) -> None:
    scope = CaptureScope(tmp_path)
    with capture_scope(scope):
        collector = SeatRoutedCollector(
            ObservationSpace(),
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode="self",
            workers=workers,
        )
        try:
            collector._env.step_into_buffers([0, -999], [True, False])
            with pytest.raises(managym.AgentError):
                collector._env.step_into_buffers([-1, -999], [True, False])
        finally:
            collector.close()
    evidence = read_failure(scope.paths[0])
    assert len(evidence.games) == 1
    assert evidence.games[0].actions == [0, -1]
    assert _bounded(scope.paths[0])["results"][0]["matches_error"]
    assert collector.game_errors is None
    assert collector.game_error_state.attempted == 0


@pytest.mark.parametrize("stages", [1, 2])
def test_strict_regime_captures_before_original_error_without_recipe_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stages: int,
) -> None:
    model = _model()
    recipe = TrainingRegime(
        id="strict-capture",
        world=managym.WORLD_VERSION,
        match=Match().config,
        agent=model.config,
        wall_seconds=120,
        stages=[
            TrainSelfPlay(
                id=f"stage-{i}",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=1,
                learning=Learning(epochs=1, minibatches=1),
            )
            for i in range(stages)
        ],
    )
    frozen = recipe.model_dump_json()
    original = SeatRoutedCollector._sample_learning_rows
    creations = 0
    original_init = SeatRoutedCollector.__init__

    def initialize(self: SeatRoutedCollector, *args: Any, **kwargs: Any) -> None:
        nonlocal creations
        creations += 1
        original_init(self, *args, **kwargs)

    def actions(self: SeatRoutedCollector, *args: Any, **kwargs: Any) -> np.ndarray:
        result = original(self, *args, **kwargs)
        if creations == stages:
            result[:] = -1
        return result

    monkeypatch.setattr(SeatRoutedCollector, "__init__", initialize)
    monkeypatch.setattr(SeatRoutedCollector, "_sample_learning_rows", actions)
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        with pytest.raises(managym.AgentError) as failed:
            execution.execute_regime(recipe, 197, tmp_path / "run", store)
        run_id = store.con.execute("SELECT id FROM training_runs").fetchone()[0]
        run = store.training_run(run_id)
    paths = list((tmp_path / "run" / "failures").glob("*.jsonl"))
    assert len(paths) == 1
    evidence = read_failure(paths[0])
    assert evidence.error == str(failed.value)
    assert evidence.context["stage"] == f"stage-{stages - 1}"
    assert evidence.context["coordinates"]["updates"] == stages - 1
    assert evidence.context["coordinates"]["learner_transitions"] == sum(
        stage.learner_transitions for stage in run.stages[:-1]
    )
    assert run.stages[-1].learner_transitions == 0
    assert evidence.context["run"] == run.id
    assert any(
        item["path"] == str(paths[0])
        for item in run.stages[-1].rejected_artifacts.values()
    )
    assert run.game_errors is None and not run.game_error_ledgers
    assert run.recovery_artifact is None
    assert all(stage.status == "completed" for stage in run.stages[:-1])
    assert run.stages[-1].status == "failed"
    assert recipe.model_dump_json() == frozen
    assert _bounded(paths[0])["results"][0]["matches_error"]


@pytest.mark.parametrize("kind", ["compound", "selection"])
def test_semantic_training_retains_rejected_command_and_offer_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    scope = CaptureScope(tmp_path / "private")
    if kind == "compound":
        original = compound.sample_compound

        def sample(*args: Any, **kwargs: Any) -> Any:
            decision = original(*args, **kwargs)
            command = decision.commands[0]
            return replace(
                decision,
                commands=(Command("reject", command.expected_revision, 999999),),
            )

        monkeypatch.setattr(compound, "sample_compound", sample)
    else:

        def invalid_command(command_id: str, revision: int, offer_id: int) -> Command:
            return Command(command_id, revision, 999999)

        monkeypatch.setattr(selection_data, "Command", invalid_command)
    with capture_scope(scope), pytest.raises(managym.AgentError):
        if kind == "compound":
            compound.collect_game(
                _model(compound_decisions=True),
                Match(),
                24,
                torch.Generator().manual_seed(1),
                tmp_path / "complete.jsonl",
                max_commands=100,
                check=lambda: None,
            )
        else:
            selection_data.collect_selection_game(
                _model(),
                ObservationSpace(),
                Match(),
                SelectionGameSpec(
                    seed=24, action_seed=1, assignment=0, split="development"
                ),
                tmp_path / "complete.jsonl",
                reference="uniform",
                max_steps=100,
                check=lambda: None,
            )
    evidence = read_failure(scope.paths[0])
    assert evidence.games[0].frame
    assert json.loads(evidence.games[0].frame)["offers"]
    assert len(evidence.games[0].actions) == 1
    rows = [json.loads(line) for line in scope.paths[0].read_text().splitlines()]
    assert "pending" in rows[1] and "error" in rows[2]
    assert _bounded(scope.paths[0])["results"][0]["matches_error"]
    interrupted = tmp_path / "interrupted.jsonl"
    interrupted.touch(mode=0o600)
    interrupted.write_text("\n".join(json.dumps(row) for row in rows[:2]) + "\n")
    replay = _bounded(interrupted)
    assert replay["original_outcome"] == "execution_outcome_unknown"
    result = replay["results"][0]
    assert result["status"] == "failed"
    assert result["matches_binding"] and result["at_retained_failure"]
    assert result["matches_error"] is None
    assert result["original_outcome"] == "execution_outcome_unknown"


def test_distillation_failure_uses_scalar_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RejectingPlayer:
        last_scores: np.ndarray | None = None

        def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
            offers = json.loads(env._engine.semantic_decision_frame_json())["offers"]
            return max(
                range(len(offers)),
                key=lambda i: {"play_land": 3, "cast": 2}.get(offers[i]["verb"], 0),
            )

    monkeypatch.setattr(
        datagen, "make_player", lambda *args, **kwargs: (RejectingPlayer(), None)
    )
    scope = CaptureScope(tmp_path)
    monkeypatch.setattr(managym, "Env", PaymentMismatchEnv)
    with (
        capture_scope(scope),
        pytest.raises(managym.AgentError, match="fixture payment mismatch"),
    ):
        datagen.generate_selfplay_shard(
            num_games=1,
            teacher_spec={"kind": "random"},
            seed=24,
            match_config=_payment_match(),
        )
    assert read_failure(scope.paths[0]).context["producer"] == "distill"
    result = _bounded_fixture_replay(
        "manabot.training.state.game_errors", scope.paths[0], "--attempt", 1
    )
    assert result["results"][0]["matches_error"]
    original = scope.paths[0].read_bytes()
    repaired = _bounded_fixture_replay(
        "manabot.training.state.game_errors",
        scope.paths[0],
        "--attempt",
        1,
        repaired=True,
    )["results"][0]
    assert repaired["status"] == "advanced"
    assert repaired["matches_binding"] and repaired["at_retained_failure"]
    assert repaired["matches_error"] is None
    assert repaired["original_outcome"] == "failed"
    assert scope.paths[0].read_bytes() == original


def test_pooled_stream_and_search_slot_failures_replay(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path)
    with capture_scope(scope):
        stream = _GameStream(ObservationSpace(), Match(), 24)
        stream.start_game(11, 24)
        pool = stream.env.rollout_pool(1, 1, 197, 100)
        buffers = rollout._allocate_buffers(ObservationSpace(), pool.num_slots)
        pool.set_buffers(buffers, pool.num_slots)
        assert pool.encode_active()
        with pytest.raises(managym.AgentError):
            pool.step_active([-1] * pool.num_slots)
        with pytest.raises(managym.AgentError):
            stream.env.step(-1)
    assert read_failure(scope.paths[0]).mode == "search"
    for path in scope.paths:
        assert read_failure(path).context["game_index"] == 11
        result = _bounded(path)["results"][0]
        assert result["matches_error"] and result["at_retained_failure"]


def test_capture_write_failure_does_not_mask_native_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manabot.env import failures

    env = _env()
    env.reset()
    original = managym.AgentError("original native failure")

    class FailingNative:
        def current_agent_index(self) -> int:
            return 0

        def semantic_decision_frame_json(self) -> str:
            return "{}"

        def step(self, action: int) -> None:
            raise original

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(failures, "_append", unavailable)
    with (
        capture_scope(CaptureScope(tmp_path)),
        pytest.raises(managym.AgentError) as failed,
    ):
        env._capture.step(FailingNative(), 0)
    assert failed.value is original
    assert "disk full" in failed.value.__notes__[0]


def test_value_search_failure_keeps_sampled_pool_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manabot.sim.search import value

    class Scorer:
        agent = None

    def invalid_actions(
        buffers: dict[str, np.ndarray], rows: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return np.full(len(rows), -1, dtype=np.int64)

    monkeypatch.setattr(value, "_random_valid_actions", invalid_actions)
    scope = CaptureScope(tmp_path)
    with capture_scope(scope):
        env = _env()
        env.reset()
        scorer = value._ValueLeafScorer(Scorer(), worlds=1, depth=1)
        with pytest.raises(managym.AgentError):
            scorer.score_actions(env)
    evidence = read_failure(scope.paths[0])
    assert evidence.search and evidence.search.operations
    assert _bounded(scope.paths[0])["results"][0]["matches_error"]


def test_rollout_evaluation_propagates_without_completed_denominator(
    tmp_path: Path,
) -> None:
    class InvalidController:
        def select(
            self, buffers: dict[str, np.ndarray], rows: np.ndarray
        ) -> np.ndarray:
            return np.full(len(rows), -1, dtype=np.int64)

    scope = CaptureScope(tmp_path)
    with capture_scope(scope), pytest.raises(managym.AgentError):
        rollout.run_vector_games(
            InvalidController(), InvalidController(), num_games=2, num_streams=2
        )
    assert _bounded(scope.paths[0])["results"][0]["matches_error"]


def test_sharded_failure_names_global_stream(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path)
    with capture_scope(scope):
        collector = SeatRoutedCollector(
            ObservationSpace(),
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode="self",
            workers=2,
        )
        try:
            with pytest.raises(managym.AgentError):
                collector._env.step_into_buffers([0, -1], [False, True])
        finally:
            collector.close()
    evidence = read_failure(scope.paths[0])
    assert evidence.games[0].stream == 1
    assert evidence.games[0].seed == 1
    assert _bounded(scope.paths[0])["results"][0]["stream"] == 1


def test_command_is_durable_before_native_execution(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path)
    with capture_scope(scope):
        env = _env()
        env.reset()
        frame = json.loads(env._engine.semantic_decision_frame_json())
        command = Command("pending", frame["revision"], frame["offers"][0]["id"])
        original = managym.AgentError("native rejected pending command")

        class NativeFailure:
            def current_agent_index(self) -> int:
                return frame["actor"]

            def semantic_decision_frame_json(self) -> str:
                return env._engine.semantic_decision_frame_json()

            def execute_semantic_command_json(self, value: str) -> str:
                # Read independently while the failing native call is entered.
                evidence = read_failure(scope.paths[0])
                assert evidence.games[0].actions == [value]
                assert evidence.execution_outcome == "execution_outcome_unknown"
                assert scope.paths[0].stat().st_mode & 0o777 == 0o600
                raise original

        with pytest.raises(managym.AgentError) as failure:
            env._capture.command(NativeFailure(), command.to_json())
        assert failure.value is original
    assert read_failure(scope.paths[0]).execution_outcome == "failed"


def test_strict_uniform_capture_replays_actual_sampled_root(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path)
    match = Match(
        MatchConfig.authored("elemental-world-v2", "elemental_ug", "elemental_bg")
    )
    with capture_scope(scope):
        collector = SeatRoutedCollector(
            ObservationSpace(),
            match,
            Reward(RewardConfig()),
            num_envs=2,
            seed=17402,
            opponent_mode="self",
            matchup_curriculum="elemental-uniform",
        )
        try:
            # Recovery can enter any later sampled root; retained inputs must
            # use its actual setup and clear the previous game's action prefix.
            collector._env.step_into_buffers([0, 0])
            collector._env.reset_seeds_into_buffers([17502, 17503])
            collector._env.step_into_buffers([0, 0])
            with pytest.raises(managym.AgentError):
                collector._env.step_into_buffers([-1, 0], [True, False])
        finally:
            collector.close()
    evidence = read_failure(scope.paths[0])
    assert len(evidence.games) == 1
    assert evidence.games[0].seed == 17502
    assert evidence.games[0].actions == [0, -1]
    result = _bounded(scope.paths[0])["results"][0]
    assert result["matches_error"]
