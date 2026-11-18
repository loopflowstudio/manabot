"""Standalone recording survives abrupt native boundaries without training."""

from pathlib import Path
import subprocess
import sys

import pytest

import managym
from managym.recording import CapturedVector, ScalarCapture, read_failure, read_record
from managym.replay import replay_failure_inputs, require_replay_runtime


@pytest.mark.parametrize("contents", [None, "", "not JSON\n"])
def test_replay_cli_reports_unreadable_journal(
    tmp_path: Path, contents: str | None
) -> None:
    path = tmp_path / "game.jsonl"
    if contents is not None:
        path.write_text(contents)
    result = subprocess.run(
        [sys.executable, "-m", "managym.replay", str(path), "--seconds", "5"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert f"cannot read journal {path}" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "method,phase",
    [("semantic_decision_frame_json", "preparing"), ("step", "executing")],
)
def test_standalone_interrupted_prefix(tmp_path: Path, method: str, phase: str) -> None:
    path = tmp_path / "game.jsonl"
    code = """
import os, sys
from pathlib import Path
import managym
from managym.recording import ScalarCapture
class ExitNative:
    def __init__(self, native): self.native = native
    def __getattr__(self, name):
        if name == sys.argv[2]: return lambda *a, **k: os._exit(73)
        return getattr(self.native, name)
game = managym.Env(seed=24)
capture = ScalarCapture(24, journal_path=Path(sys.argv[1]), context={"attempt": "standalone"})
capture.reset(game, [managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")])
capture.step(game, 0)
assert not any(n.split('.')[0] in {"manabot", "torch", "etude"} for n in sys.modules)
capture.step(ExitNative(game), 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), method],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, result.stderr
    evidence = read_record(path)
    assert evidence.boundary is not None
    assert evidence.boundary.phase == phase
    assert not evidence.boundary.returned
    assert evidence.context["attempt"] == "standalone"
    assert path.stat().st_mode & 0o777 == 0o600
    require_replay_runtime(
        evidence.identities, allow_changed_runtime=False, require_source=True
    )
    replay = replay_failure_inputs(evidence, statistics=True)[0]
    assert replay["statistics"]["seed"] == 24
    assert replay["status"] == "advanced"
    assert replay["original_outcome"] == "execution_outcome_unknown"


def test_legacy_source_is_not_reinterpreted(tmp_path: Path) -> None:
    capture = ScalarCapture(7, journal_path=tmp_path / "game.jsonl")
    game = managym.Env(seed=7)
    capture.reset(game, [managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")])
    identities = dict(capture.identities)
    identities["source_sha256"] = identities.pop("engine_recording_source_sha256")
    with pytest.raises(ValueError, match="legacy source"):
        require_replay_runtime(
            identities, allow_changed_runtime=False, require_source=True
        )
    require_replay_runtime(identities, allow_changed_runtime=True, require_source=True)
    identities["world"] = "wrong-world"
    with pytest.raises(ValueError, match="world differs"):
        require_replay_runtime(
            identities, allow_changed_runtime=True, require_source=True
        )


def test_standalone_vector_records_global_streams(tmp_path: Path) -> None:
    from managym.recording import CapturedVector

    capture = CapturedVector(
        num_envs=2, seed=17, stream_offset=8, journal_path=tmp_path / "vector.jsonl"
    )
    capture.enable_statistics()
    capture.reset_all([managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")])
    capture.step([0, 0])
    evidence = read_record(tmp_path / "vector.jsonl")
    assert [game.stream for game in evidence.games] == [8, 9]
    assert [game.actions for game in evidence.games] == [[0], [0]]
    assert [row.stream for row in capture.statistics()] == [8, 9]
    assert all(
        result["status"] == "advanced" for result in replay_failure_inputs(evidence)
    )


def test_standalone_command_journal_and_statistics(tmp_path: Path) -> None:
    import json

    from managym.decision import Command

    game = managym.Env(seed=17)
    indexed = managym.Env(seed=17)
    game.enable_statistics()
    indexed.enable_statistics()
    players = [managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")]
    capture = ScalarCapture(17, journal_path=tmp_path / "command.jsonl")
    capture.reset(game, players)
    indexed.reset(players)
    frame = json.loads(game.semantic_decision_frame_json())
    command = Command("one", frame["revision"], frame["offers"][0]["id"]).to_json()
    capture.command(game, command)
    indexed.step(0)
    assert game.statistics().cards == indexed.statistics().cards
    evidence = read_record(tmp_path / "command.jsonl")
    assert evidence.boundary.returned and evidence.boundary.phase == "ready"
    assert evidence.games[0].actions == [command]
    assert replay_failure_inputs(evidence)[0]["status"] == "advanced"


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize(
    "enabled,include_own", [(False, False), (True, False), (False, True)]
)
def test_decision_memory_option_survives_journal_reset_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vector: bool,
    enabled: bool,
    include_own: bool,
) -> None:
    path = tmp_path / "memory.jsonl"
    players = [managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")]
    if vector:
        capture = CapturedVector(
            num_envs=1,
            seed=17,
            decision_memory=enabled,
            decision_memory_include_own=include_own,
            journal_path=path,
        )
        capture.reset_all(players)
        capture.step([0])
        capture.reset_all(players)
        capture.step([0])
    else:
        game = managym.Env(
            seed=17, decision_memory=enabled, decision_memory_include_own=include_own
        )
        scalar = ScalarCapture(
            17,
            decision_memory=enabled,
            decision_memory_include_own=include_own,
            journal_path=path,
        )
        scalar.reset(game, players)
        scalar.step(game, 0)
        scalar.reset(game, players)
        scalar.step(game, 0)
    evidence = read_failure(path)
    assert evidence.games[0].decision_memory_include_own is include_own
    assert evidence.games[0].decision_memory is (enabled or include_own)
    assert evidence.games[0].actions == [0]
    # Replay must configure native memory, not merely preserve the receipt bit.
    scalar_constructor, vector_constructor = managym.Env, managym.VectorEnv

    def replay_env(
        *,
        seed: int,
        skip_trivial: bool,
        decision_memory: bool,
        decision_memory_include_own: bool,
        num_envs: int = 1,
        seed_stride: int = 1,
        player_policies: tuple[str, str] = ("agent", "agent"),
    ) -> managym.Env | managym.VectorEnv:
        assert decision_memory_include_own is include_own
        assert decision_memory is (enabled or include_own)
        if vector:
            return vector_constructor(
                num_envs=num_envs,
                seed=seed,
                seed_stride=seed_stride,
                skip_trivial=skip_trivial,
                decision_memory=decision_memory,
                decision_memory_include_own=decision_memory_include_own,
                player_policies=player_policies,
            )
        return scalar_constructor(
            seed=seed,
            skip_trivial=skip_trivial,
            decision_memory=decision_memory,
            decision_memory_include_own=decision_memory_include_own,
        )

    monkeypatch.setattr(managym, "VectorEnv" if vector else "Env", replay_env)
    assert replay_failure_inputs(evidence)[0]["status"] == "advanced"
    if vector:
        capture.close()


def test_legacy_decision_capture_setting_keeps_its_replay_meaning() -> None:
    from managym.recording import GameInputs

    legacy = {"seed": 17, "players": []}
    inputs = GameInputs.model_validate(legacy)
    assert inputs.decision_memory
    assert "decision_memory" not in inputs.model_dump(mode="json")
    disabled = inputs.model_copy(update={"decision_memory": False})
    assert (
        GameInputs.model_validate(disabled.model_dump(mode="json")).decision_memory
        is False
    )
