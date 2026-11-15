"""Durable training cursors survive native/learner interruption without cleanup."""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from manabot.env.failures import (
    CapturedVector,
    CaptureScope,
    ScalarCapture,
    atomic_private_json,
    capture_scope,
    read_failure,
)
from manabot.env.observation import ObservationSpace
from manabot.env.replay import replay_failure_inputs
from manabot.training.rollout.producers import ShardedVectorEnv
import managym


def _players() -> list[managym.PlayerConfig]:
    return [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]


def _roster() -> list[managym.PlayerConfig]:
    return [
        managym.authored_deck_setup("elemental-world-v2", key)
        for key in managym.authored_deck_keys("elemental-world-v2")
    ]


def _buffers(count: int) -> dict[str, np.ndarray]:
    buffers = ObservationSpace().encoder.allocate(count)
    buffers.update(
        rewards=np.zeros(count, dtype=np.float64),
        terminated=np.zeros(count, dtype=np.uint8),
        truncated=np.zeros(count, dtype=np.uint8),
    )
    return buffers


class _ExitNative:
    """Inject a process exit at a real native boundary, without hanging a host."""

    def __init__(self, native: Any, method: str) -> None:
        self.native = native
        self.method = method

    def __getattr__(self, name: str) -> Any:
        if name == self.method:
            return lambda *args, **kwargs: os._exit(73)
        return getattr(self.native, name)


def _interrupt_child(directory: str, mode: str, phase: str) -> None:
    scope = CaptureScope(
        Path(directory) / "failures", lambda: {"update": 7}, training_game_logs=True
    )
    with capture_scope(scope):
        if mode == "scalar":
            capture = ScalarCapture(24)
            env = managym.Env(seed=24, skip_trivial=True)
            capture.reset(env, _players())
            method = {
                "preparing": "semantic_decision_frame_json",
                "executing": "step",
                "synchronizing": "content_pack_manifest",
            }[phase]
            wrapped = _ExitNative(env, method)
            if phase == "synchronizing":
                capture.reset(wrapped, _players())
            else:
                capture.step(wrapped, 0)
        else:
            vector = CapturedVector(num_envs=2, seed=24)
            vector.set_buffers(_buffers(2))
            if phase == "sampled_reset":
                vector.set_uniform_roster(_roster())
            vector.reset_all_into_buffers(_players())
            if phase in {"learner_inference", "opponent_inference", "learning"}:
                vector.step_into_buffers([0, 0])
                with scope.operation("collection"), scope.operation(phase):
                    os._exit(73)
            if phase == "executing":
                vector.step_into_buffers([0, 0])
            method = {
                "preparing": "semantic_decision_frames_json",
                "executing": "step_into_buffers",
                "synchronizing": "current_game_seeds",
                "sampled_reset": "current_player_configs",
            }[phase]
            vector.native = _ExitNative(vector.native, method)
            if phase == "sampled_reset":
                vector.reset_seeds_into_buffers([31, 32])
            else:
                vector.step_into_buffers([0, 0], [True, False])
    raise AssertionError("injected boundary was not reached")


@pytest.mark.parametrize(
    "mode,phase",
    [
        ("scalar", "preparing"),
        ("scalar", "executing"),
        ("scalar", "synchronizing"),
        ("vector", "preparing"),
        ("vector", "executing"),
        ("vector", "synchronizing"),
        ("vector", "sampled_reset"),
        ("vector", "learner_inference"),
        ("vector", "opponent_inference"),
        ("vector", "learning"),
    ],
)
def test_abrupt_exit_leaves_replayable_private_cursor(
    tmp_path: Path, mode: str, phase: str
) -> None:
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_interrupt_child'](*sys.argv[2:])"
    child = subprocess.run(
        [sys.executable, "-c", code, __file__, str(tmp_path), mode, phase],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 73, child.stderr
    directory = tmp_path / "game-journals"
    cursor = next(directory.glob("*.jsonl"))
    assert cursor.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    evidence = read_failure(cursor)
    boundary = evidence.boundary
    assert boundary is not None
    assert evidence.context == {"update": 7}
    expected = (
        "ready"
        if phase in {"learner_inference", "opponent_inference", "learning"}
        else "synchronizing"
        if phase == "sampled_reset"
        else phase
    )
    assert boundary.phase == expected
    assert boundary.returned == (expected in {"ready", "synchronizing"})
    if phase in {"learner_inference", "opponent_inference", "learning"}:
        marker = json.loads((directory / "operation.json").read_text())
        assert marker["operation"] == phase and marker["state"] == "pending"
    if phase == "sampled_reset":
        assert not boundary.setup_known
        assert boundary.proposed_seeds == [31, 32]
        assert [game.seed for game in evidence.games] == [24, 25]
    if mode == "vector" and phase == "executing":
        assert boundary.active == [True, False]
        assert evidence.games[0].actions == [0, 0]
        assert evidence.games[1].actions == [0]
    before = cursor.read_bytes()
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "manabot.training.state.game_errors",
            str(cursor),
            "--attempt",
            "1",
            "--seconds",
            "15",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert replay.returncode == 0, replay.stderr
    results = json.loads(replay.stdout)["results"]
    assert all(row["status"] == "advanced" for row in results)
    assert all(
        row["original_outcome"] == "execution_outcome_unknown" for row in results
    )
    if mode == "vector" and phase == "executing":
        assert results[0]["matches_binding"] is True
        assert results[0]["at_retained_failure"] is True
        assert results[1]["matches_binding"] is None
        assert results[1]["at_retained_failure"] is False
    assert cursor.read_bytes() == before


def test_nested_operation_restores_parent_and_records_return(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path / "failures", training_game_logs=True)
    marker = scope.journal_directory / "operation.json"
    with scope.operation("collection"):
        with scope.operation("learner_inference"):
            first = json.loads(marker.read_text())
            assert first["operation"] == "learner_inference"
        parent = json.loads(marker.read_text())
        assert parent["sequence"] > first["sequence"]
        assert parent["operation"] == "collection" and parent["state"] == "pending"
    assert json.loads(marker.read_text())["state"] == "returned"


def _interrupt_reset_child(directory: str, operation: str, first: str) -> None:
    scope = CaptureScope(Path(directory) / "failures", training_game_logs=True)
    with capture_scope(scope):
        if operation == "scalar_reset":
            capture = ScalarCapture(24)
            env = managym.Env(seed=24)
            if first != "first":
                capture.reset(env, _players())
                capture.step(env, 0)
            capture.reset(_ExitNative(env, "reset"), list(reversed(_players())))
        else:
            vector = CapturedVector(num_envs=2, seed=24)
            vector.set_buffers(_buffers(2))
            vector.set_uniform_roster(_roster())
            if first != "first":
                vector.reset_all_into_buffers(_players())
                vector.step_into_buffers([0, 0])
            method = {
                "reset_each": "reset_each_into_buffers",
                "reset_seeds": "reset_seeds_into_buffers",
                "reassign_roots": "reassign_roots_into_buffers",
                "replace_failed": "replace_failed_into_buffers",
                "reassign_preparing": "current_game_seeds",
            }[operation]
            vector.native = _ExitNative(vector.native, method)
            if operation == "reset_each":
                vector.reset_each_into_buffers([_players(), _players()])
            elif operation == "reset_seeds":
                vector.reset_seeds_into_buffers([31, 32])
            elif operation in {"reassign_roots", "reassign_preparing"}:
                vector.reassign_roots_into_buffers([list(reversed(_players())), None])
            else:
                vector.replace_failed_into_buffers([True, False])
    raise AssertionError("injected reset boundary was not reached")


@pytest.mark.parametrize(
    "operation,first",
    [
        ("scalar_reset", "prior"),
        ("scalar_reset", "first"),
        ("reset_each", "prior"),
        ("reset_each", "first"),
        ("reset_seeds", "prior"),
        ("reassign_roots", "prior"),
        ("replace_failed", "prior"),
        ("reassign_preparing", "prior"),
    ],
)
def test_interrupted_reset_replays_prior_game_without_binding_claim(
    tmp_path: Path, operation: str, first: str
) -> None:
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_interrupt_reset_child'](*sys.argv[2:])"
    child = subprocess.run(
        [sys.executable, "-c", code, __file__, str(tmp_path), operation, first],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 73, child.stderr
    cursor = next((tmp_path / "game-journals").glob("*.jsonl"))
    evidence = read_failure(cursor)
    boundary = evidence.boundary
    assert boundary is not None
    assert boundary.phase == (
        "preparing" if operation == "reassign_preparing" else "executing"
    )
    assert boundary.operation == (
        "reset"
        if operation == "scalar_reset"
        else "reassign_roots"
        if operation == "reassign_preparing"
        else operation
    )
    assert not boundary.returned
    assert evidence.execution_outcome == "execution_outcome_unknown"
    assert [game.seed for game in evidence.games] == (
        [24] if operation == "scalar_reset" else [24, 25]
    )
    assert all(
        game.actions == ([] if first == "first" else [0]) for game in evidence.games
    )
    if first == "first":
        assert all(not game.players for game in evidence.games)
    elif operation == "scalar_reset":
        assert evidence.games[0].players[0].name == "a"
        assert boundary.proposed_players[0][0].name == "b"
    else:
        assert not boundary.setup_known or operation.startswith("reassign")
        assert evidence.games[0].players[0].name in {
            player.name for player in _roster()
        }
    before = cursor.read_bytes()
    results = replay_failure_inputs(evidence)
    assert all(
        row["status"] == ("setup_unavailable" if first == "first" else "advanced")
        for row in results
    )
    assert all(row["at_retained_failure"] is False for row in results)
    assert all(
        row["matches_binding"] is None and row["matches_error"] is None
        for row in results
    )
    assert all(row["boundary_operation"] == boundary.operation for row in results)
    assert cursor.read_bytes() == before


def test_disabled_logging_has_no_writer_or_extra_native_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("disabled journal invoked writer")

    monkeypatch.setattr("manabot.env.failures.atomic_private_json", forbidden)
    scope = CaptureScope(tmp_path / "failures", context=forbidden)
    with capture_scope(scope):
        vector = CapturedVector(num_envs=1)
        vector.set_buffers(_buffers(1))
        vector.reset_all_into_buffers(_players())
        vector.step_into_buffers([0])
        vector.close()
        with scope.operation("learning"):
            pass
    assert not scope.journal_directory.exists()


@pytest.mark.parametrize("operation", ["step", "step_isolated"])
@pytest.mark.parametrize("binding", ["fresh", "missing", "inactive"])
def test_executing_replay_requires_active_fresh_step_binding(
    tmp_path: Path, operation: str, binding: str, capsys: pytest.CaptureFixture[str]
) -> None:
    scope = CaptureScope(tmp_path / "failures", training_game_logs=True)
    with capture_scope(scope):
        vector = CapturedVector(num_envs=1, seed=24)
        vector.set_buffers(_buffers(1))
        vector.reset_all_into_buffers(_players())
        frame = vector.semantic_decision_frames_json()[0]
        vector.step_into_buffers([0])
        evidence = read_failure(next(scope.journal_directory.glob("*.jsonl")))
        vector.close()
    boundary = evidence.boundary
    assert boundary is not None
    boundary.phase = "executing"
    boundary.operation = operation
    boundary.returned = False
    boundary.active = [binding != "inactive"]
    evidence.games[0].frame = None if binding == "missing" else frame
    result = replay_failure_inputs(evidence)[0]
    assert result["status"] == "advanced"
    assert result["actions"] == (1 if binding == "fresh" else 0)
    assert result["at_retained_failure"] == (binding == "fresh")
    assert result["matches_binding"] is (True if binding == "fresh" else None)
    assert evidence.games[0].actions == [0]
    # The operator reads the turn from replay; a bound attempt names it first.
    reached = result["reached"]
    assert isinstance(reached, dict) and reached["turn"] >= 1
    attempts = [
        json.loads(line)["replay_attempt"]
        for line in capsys.readouterr().err.splitlines()
        if line.startswith('{"replay_attempt"')
    ]
    assert len(attempts) == (1 if binding == "fresh" else 0)
    if attempts:
        assert attempts[0]["action_index"] == 0
        assert attempts[0]["position"]["turn"] >= 1


@pytest.mark.parametrize("seed", [24, 2**64 - 1])
@pytest.mark.parametrize("sampled", [False, True])
def test_enabled_logging_preserves_buffers_prefixes_and_reset_identity(
    tmp_path: Path, seed: int, sampled: bool
) -> None:
    snapshots: list[tuple[dict[str, np.ndarray], list[dict[str, object]]]] = []
    for enabled in (False, True):
        scope = CaptureScope(
            tmp_path / str(enabled) / "failures", training_game_logs=enabled
        )
        with capture_scope(scope):
            vector = CapturedVector(num_envs=2, seed=seed)
            buffers = _buffers(2)
            vector.set_buffers(buffers)
            if sampled:
                vector.set_uniform_roster(_roster())
            vector.reset_all_into_buffers(_players())
            vector.step_into_buffers([0, 0], [True, False])
            vector.reset_seeds_into_buffers([seed, seed])
            vector.step_into_buffers([0, 0])
            snapshots.append(
                (
                    {key: value.copy() for key, value in buffers.items()},
                    [game.model_dump() for game in vector.games],
                )
            )
            if enabled:
                retained = read_failure(next(scope.journal_directory.glob("*.jsonl")))
                assert (
                    retained.boundary is not None and retained.boundary.phase == "ready"
                )
                assert retained.boundary.setup_known
                assert all(retained.boundary.last_known_frames)
                assert retained.boundary.frame_prefix_lengths == [0, 0]
                assert all(
                    row["status"] == "advanced"
                    for row in replay_failure_inputs(retained)
                )
            vector.close()
    assert snapshots[0][1] == snapshots[1][1]
    for key in snapshots[0][0]:
        np.testing.assert_array_equal(snapshots[0][0][key], snapshots[1][0][key])


@pytest.mark.parametrize("workers", [2, 4])
def test_shards_keep_independent_paths_and_owner_coordinates(
    tmp_path: Path, workers: int
) -> None:
    point = {"update": 1}
    scope = CaptureScope(
        tmp_path / "failures", lambda: dict(point), training_game_logs=True
    )
    with capture_scope(scope):
        vector = ShardedVectorEnv(4, 24, workers)
        try:
            vector.set_buffers(_buffers(4))
            vector.reset_all_into_buffers(_players())
            point["update"] = 2
            vector.step_into_buffers([0] * 4, [True, False, True, False])
        finally:
            vector.close()
    paths = list(scope.journal_directory.glob("*.jsonl"))
    assert len(paths) == workers
    receipts = [read_failure(path) for path in paths]
    assert sorted(game.stream for row in receipts for game in row.games) == [0, 1, 2, 3]
    assert all(row.context["update"] == 2 for row in receipts)
    assert all(row.boundary is not None and row.boundary.returned for row in receipts)


def test_pending_storage_error_prevents_native_step(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path / "failures", training_game_logs=True)
    with capture_scope(scope):
        vector = CapturedVector(num_envs=1, seed=24)
        vector.set_buffers(_buffers(1))
        vector.reset_all_into_buffers(_players())
        before = vector.native.semantic_decision_frames_json()
        assert vector._cursor is not None
        vector._cursor.unlink()
        vector._cursor.mkdir()
        with pytest.raises(OSError):
            vector.step_into_buffers([0])
        assert vector.native.semantic_decision_frames_json() == before
        vector.close()


def test_atomic_replacement_retains_complete_old_bytes_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private" / "snapshot.json"
    atomic_private_json(path, {"value": "old"})

    def fail(descriptor: int) -> None:
        raise OSError("injected storage failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="storage failure"):
        atomic_private_json(path, {"value": "new"})
    assert json.loads(path.read_text()) == {"value": "old"}
    assert not list(path.parent.glob(".*.tmp"))


@pytest.mark.parametrize("recipe", ["local", "baseline", "regime"])
def test_cli_forwards_runtime_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: str
) -> None:
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from manabot.cli import app
    from manabot.recipes.local import local_smoke

    seen: list[bool] = []

    def execute(*args: object, **kwargs: object) -> SimpleNamespace:
        seen.append(kwargs["training_game_logs"] is True)
        return SimpleNamespace(
            status="completed",
            id="test",
            notebook="baseline.ipynb",
            stages=[SimpleNamespace(artifacts={"raw": {"path": "policy.pt"}})],
        )

    monkeypatch.setattr("manabot.recipes.local.run_local", execute)
    monkeypatch.setattr("manabot.recipes.baseline.run_baseline", execute)
    monkeypatch.setattr("manabot.training.execution.execute_regime", execute)
    args = ["train", "--training-game-logs", "--out", str(tmp_path / "run")]
    if recipe == "regime":
        path = tmp_path / "regime.json"
        path.write_text(local_smoke(device="cpu").model_dump_json())
        args += ["--regime", str(path)]
    else:
        args += ["--recipe", recipe]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen == [True]


def test_job_option_is_runtime_only() -> None:
    from manabot.experiment.job import TrainingJob
    from manabot.recipes.local import local_smoke

    plain = TrainingJob(regime=local_smoke(device="cpu"), experiment="fixture")
    enabled = plain.model_copy(update={"training_game_logs": True})
    assert "training_game_logs" not in plain.model_dump(mode="json")
    assert TrainingJob.model_validate_json(enabled.model_dump_json()).training_game_logs
    assert enabled.regime.model_dump_json() == plain.regime.model_dump_json()


def test_return_acknowledgment_storage_error_names_executed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import manabot.env.failures as failures

    scope = CaptureScope(tmp_path / "failures", training_game_logs=True)
    with capture_scope(scope):
        vector = CapturedVector(num_envs=1, seed=24)
        vector.set_buffers(_buffers(1))
        vector.reset_all_into_buffers(_players())
        frame = vector.native.semantic_decision_frames_json()
        write = failures.atomic_private_json

        def fail_return(path: Path, row: dict[str, object]) -> None:
            boundary = row.get("boundary")
            if isinstance(boundary, dict) and boundary["phase"] == "synchronizing":
                raise OSError("storage gone after execution")
            write(path, row)

        monkeypatch.setattr(failures, "atomic_private_json", fail_return)
        with pytest.raises(OSError) as error:
            vector.step_into_buffers([0])
        assert "Native operation returned" in " ".join(error.value.__notes__)
        assert vector.native.semantic_decision_frames_json() != frame
        vector.close()


def test_reassignment_and_failed_replacement_retain_current_inputs(
    tmp_path: Path,
) -> None:
    scope = CaptureScope(tmp_path / "failures", training_game_logs=True)
    with capture_scope(scope):
        vector = CapturedVector(num_envs=2, seed=24)
        vector.set_buffers(_buffers(2))
        vector.reset_all_into_buffers(_players())
        vector.reassign_roots_into_buffers([list(reversed(_players())), None])
        cursor = next(scope.journal_directory.glob("*.jsonl"))
        reassigned = read_failure(cursor)
        assert reassigned.games[0].players[0].name == "b"
        vector.step_isolated_into_buffers([-1, 0], [True, True])
        failed = read_failure(cursor)
        assert failed.boundary is not None and failed.boundary.returned
        assert failed.execution_outcome == "execution_outcome_unknown"
        assert replay_failure_inputs(failed)[0]["status"] == "failed"
        vector.replace_failed_into_buffers([True, False])
        replaced = read_failure(cursor)
        assert replaced.games[0].actions == []
        assert replaced.games[1].actions == [0]
        assert (
            replaced.boundary is not None
            and replaced.boundary.operation == "replace_failed"
        )
        vector.close()


def test_auto_reset_replaces_prefix_and_bounds_retained_files(tmp_path: Path) -> None:
    scope = CaptureScope(tmp_path / "failures", training_game_logs=True)
    with capture_scope(scope):
        vector = CapturedVector(num_envs=1, seed=24)
        buffers = _buffers(1)
        vector.set_buffers(buffers)
        vector.reset_all_into_buffers(_players())
        for _ in range(80):
            vector.step_into_buffers([0])
            if buffers["terminated"][0] or buffers["truncated"][0]:
                break
        else:
            pytest.fail("tiny native game did not settle")
        cursor = next(scope.journal_directory.glob("*.jsonl"))
        evidence = read_failure(cursor)
        assert evidence.games[0].seed == 25
        assert evidence.games[0].actions == []
        assert evidence.boundary is not None and evidence.boundary.phase == "ready"
        assert len(list(scope.journal_directory.glob("*.jsonl"))) == 1
        vector.close()
        assert cursor.exists()
        scope.retire_cursors()
        assert not list(scope.journal_directory.glob("*.jsonl"))


def _interrupt_execution(directory: str, phase: str) -> None:
    import torch

    from manabot.backends import resolve
    from manabot.backends.torch.collection import CollectionScope
    from manabot.recipes.local import local_smoke
    import manabot.training.execution as execution
    from manabot.training.regime import TrainSelfPlay
    from manabot.verify.store import VerifyStore

    torch.set_num_threads(1)
    regime = local_smoke(steps=1, device="cpu")
    regime.agent.hidden_dim = 8
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning_seats = "one"
    stage.batch_closure = "exact-quota"
    stage.transitions = 2

    def exit_now(*args: object, **kwargs: object) -> Any:
        os._exit(73)

    patch = pytest.MonkeyPatch()
    if phase == "learner_inference":
        patch.setattr(CollectionScope, "sample", exit_now)
    else:
        patch.setattr(execution, "update_damped_iteration", exit_now)
        patch.setattr(execution, "update_iteration", exit_now)
    with VerifyStore(Path(directory) / "training.sqlite") as store:
        execution.execute_regime(
            regime,
            24,
            Path(directory) / "run",
            store,
            backend=resolve("torch"),
            training_game_logs=True,
        )
    raise AssertionError("injected training operation not reached")


@pytest.mark.parametrize("phase", ["learner_inference", "learning"])
def test_executor_interrupt_marks_active_work_before_call(
    tmp_path: Path, phase: str
) -> None:
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_interrupt_execution'](*sys.argv[2:])"
    child = subprocess.run(
        [sys.executable, "-c", code, __file__, str(tmp_path), phase],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 73, child.stderr
    directory = tmp_path / "run/game-journals"
    marker = json.loads((directory / "operation.json").read_text())
    assert marker["operation"] == phase and marker["state"] == "pending"
    assert marker["context"]["stage"]
    assert marker["context"]["coordinates"]["updates"] == 0
    for cursor in directory.glob("*.jsonl"):
        evidence = read_failure(cursor)
        assert evidence.boundary is not None and evidence.boundary.phase == "ready"
        assert evidence.boundary.returned
    run = json.loads((tmp_path / "run/run.json").read_text())
    assert Path(run["game_journal_directory"]).resolve() == directory.resolve()
