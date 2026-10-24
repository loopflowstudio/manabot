"""Deterministic failures through the ordinary collector and update boundary."""

from copy import deepcopy
import json
import multiprocessing.connection
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal

import numpy as np
import pytest
import torch

from manabot.arena import match as arena_match
from manabot.arena.models import (
    ArenaContract,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.replay import read_game_journal, write_trace
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    RewardConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.learning.returns import transition_gae
from manabot.training.regime import Learning
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    replay_attempt,
)
import managym
from tests.training.game_error_fixtures import (
    PaymentMismatchEnv,
    _bounded_fixture_replay,
    _payment_match,
)


class ScriptedFailures:
    """Two complete five-move games; stream 1 fails after both seats acted.

    Stream 0 advances before that failure. A replacement uses disjoint visible
    markers so any leaked rows, duplicate steps or cross-game returns are visible.
    """

    def __init__(
        self,
        buffers: dict[str, np.ndarray],
        *,
        always: bool = False,
        failure_generation: int = 0,
    ) -> None:
        self.buffers = buffers
        self.positions = [0, 0]
        self.generation = [0, 0]
        self.steps = [0, 0]
        self.always = always
        self.failure_generation = failure_generation
        self.infos: list[dict[str, Any]] = [{}, {}]
        self.encode()

    def encode(self) -> None:
        for row in (0, 1):
            self.buffers["agent_player"][row, 0, 0] = (
                row * 10 + self.positions[row] + 100 * self.generation[row]
            )

    def current_agent_indices(self) -> list[int]:
        return [position % 2 for position in self.positions]

    def current_game_seeds(self) -> list[int]:
        return [row + self.generation[row] * 2 for row in (0, 1)]

    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool]
    ) -> None:
        self.buffers["terminated"].fill(0)
        self.buffers["truncated"].fill(0)
        self.infos = [{}, {}]
        for row, enabled in enumerate(active):
            if not enabled:
                continue
            self.steps[row] += 1
            if self.positions[row] == 3 and (
                self.always
                or (row == 1 and self.generation[row] == self.failure_generation)
            ):
                self.infos[row] = {
                    "game_error": "not enough producible mana",
                    "game_error_class": "AgentError",
                }
                continue
            self.positions[row] += 1
            if self.positions[row] == 5:
                self.positions[row] = 0
                self.generation[row] += 1
                self.buffers["terminated"][row] = 1
                self.infos[row] = {"winner_index": 1}
        self.encode()

    def replace_failed_into_buffers(self, failed: list[bool]) -> None:
        for row, enabled in enumerate(failed):
            if enabled:
                self.generation[row] += 1
                self.positions[row] = 0
        self.encode()

    def get_last_info(self) -> list[dict[str, Any]]:
        return self.infos


def agent() -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )


def attempt(failed: bool, seed: int = 0) -> GameAttempt:
    return GameAttempt(
        stream=0,
        seed=seed,
        seat=0,
        matchup=MatchConfig(),
        outcome="engine_error" if failed else "completed",
        discarded_samples=3 if failed else 0,
        valid_samples=4 if not failed else 0,
    )


def _tolerant_collector(
    model: ActorCritic,
    ledger: GameErrorLedger,
    *,
    seats: Literal["one", "both"] = "both",
) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats=seats,
        game_errors=ledger.policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )


def test_strict_threshold_window_and_restart(tmp_path: Path) -> None:
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "first.jsonl", policy)
    for index in range(99):
        ledger.record(attempt(index < 10, index), {})
    assert ledger.state.stop_reason is None
    assert ledger.state.error_rate > 0.1  # warmup has not settled
    assert ledger.state.replacement_samples == ledger.state.discarded_samples == 30
    assert ledger.state.replacement_debt == 0
    restored = GameErrorLedger(
        tmp_path / "second.jsonl", policy, parents=[str(ledger.path)]
    )
    restored.record(attempt(False), {})
    assert restored.state.error_rate == 0.1
    assert restored.state.stop_reason is None
    restored.record(attempt(True), {})  # oldest failure falls out
    assert restored.state.error_rate == 0.1
    for _ in range(9):
        restored.record(attempt(True), {})
    assert restored.state.stop_reason is None
    restored.record(attempt(True), {})
    assert restored.state.error_rate == 0.11
    assert restored.state.stop_reason == "game_error_rate_exceeded"
    final = GameErrorLedger(
        tmp_path / "third.jsonl", policy, parents=[str(ledger.path), str(restored.path)]
    )
    assert final.state == restored.state
    assert final.state.replacement_debt == 33
    assert (
        GameErrorState.model_validate_json(final.state.model_dump_json()) == final.state
    )
    with pytest.raises(ValueError, match="warmup"):
        GameErrorPolicy(window=10)


@pytest.mark.parametrize("seats", ["one", "both"])
def test_quarantine_refills_without_leaking_seats_or_returns(
    tmp_path: Path, seats: Literal["one", "both"]
) -> None:
    model = agent()
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = _tolerant_collector(model, ledger, seats=seats)
    scripted = ScriptedFailures(collector._buffers)
    collector._env = scripted
    batch = collector.collect(model, 1)
    markers = batch.obs["agent_player"][:, 0, 0, 0]
    assert not any(10 <= marker < 100 for marker in markers)
    assert scripted.steps == [5, 9]
    assert ledger.state.completed == 2 and ledger.state.crashed == 1
    assert ledger.state.discarded_samples == (4 if seats == "both" else 2)
    assert collector.stats.learner_transitions == batch.actions.size
    assert int(batch.dones.sum()) == (4 if seats == "both" else 2)
    _, returns = transition_gae(
        torch.from_numpy(batch.rewards),
        torch.zeros_like(torch.from_numpy(batch.values)),
        torch.from_numpy(batch.dones),
        torch.tensor([999.0]),
        1.0,
        1.0,
    )
    assert set(returns.flatten().tolist()) <= {-1, 1}
    assert not any(collector._pending) and not any(collector._streams)
    learner = Learner(model, "cpu", minibatch_rows=collector.num_envs)
    result = update_iteration(
        learner,
        batch,
        Learning(epochs=1, minibatches=1),
        0,
        np.random.default_rng(1),
        iteration=1,
    )
    assert result["rows"] == batch.actions.size
    assert learner.optimizer.state


def test_all_failures_stop_boundedly_and_count_once_per_game(tmp_path: Path) -> None:
    model = agent()
    policy = GameErrorPolicy(window=4, warmup=4)
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = _tolerant_collector(model, ledger)
    scripted = ScriptedFailures(collector._buffers, always=True)
    collector._env = scripted
    with pytest.raises(GameErrorLimit):
        collector.collect(model, 1)
    assert ledger.state.attempted == ledger.state.crashed == 4
    assert ledger.state.completed == collector.stats.games == 0
    assert scripted.steps == [8, 8]
    assert not any(collector._pending) and not any(collector._streams)
    assert collector.stats.learner_transitions == 0


@pytest.mark.parametrize("workers", [1, 2])
def test_native_partial_vector_and_replay(workers: int) -> None:
    model = agent()
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=3,
        opponent_mode="self",
        workers=workers,
    )
    try:
        _assert_native_failure_replay(collector)
    finally:
        collector.close()


def _assert_native_failure_replay(collector: SeatRoutedCollector) -> None:
    before = {k: v.copy() for k, v in collector._buffers.items()}
    collector._env.step_isolated_into_buffers([0, -1, 0], [True] * 3)
    infos = collector._env.get_last_info()
    assert "game_error" not in infos[0] and "game_error" not in infos[2]
    assert infos[1]["game_error_class"] == "AgentError"
    row = GameAttempt(
        stream=1,
        seed=1,
        seat=0,
        matchup=MatchConfig(),
        outcome="engine_error",
        actions=[-1],
        error_class=infos[1]["game_error_class"],
        error=infos[1]["game_error"],
        frame_json=infos[1]["game_error_frame"],
    )
    replay = replay_attempt(row)
    assert replay["status"] == "failed" and replay["at_retained_failure"]
    assert replay["matches_binding"] and replay["matches_error"]
    # Vector execution retains additional post-error native diagnostics. Scalar
    # replay must reproduce the rejection and frame without discarding that evidence.
    assert row.error is not None
    message, separator, diagnostic = row.error.partition("; native_failure=")
    assert replay["error"] == message
    assert separator
    native = json.loads(diagnostic)
    assert native["seed"] == row.seed
    assert native["failed_action_index"] == row.actions[-1]
    assert native["post_error_state"] is not None
    assert replay["frame"] == row.frame_json
    advanced = {k: v.copy() for k, v in collector._buffers.items()}
    collector._env.replace_failed_into_buffers([False, True, False])
    assert collector._env.current_game_seeds() == [0, 4, 2]
    for key in before:
        np.testing.assert_array_equal(
            collector._buffers[key][[0, 2]], advanced[key][[0, 2]]
        )


@pytest.mark.parametrize(
    "variant",
    [
        "matching",
        "wrong_frame",
        "different_error",
        "different_class",
        "suffix",
        "missing",
        "prefix",
    ],
)
def test_original_failure_is_immutable_and_cli_replays(
    tmp_path: Path, variant: str
) -> None:
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    row = attempt(True)
    row.actions = [-1]
    env = managym.Env(seed=row.seed, skip_trivial=True)
    env.reset(Match(row.matchup).to_rust())
    row.frame_json = env.semantic_decision_frame_json()
    with pytest.raises(managym.AgentError) as failure:
        env.step(-1)
    row.error_class = type(failure.value).__name__
    row.error = str(failure.value)
    if variant == "wrong_frame":
        row.frame_json = "{}"
    elif variant == "different_error":
        row.error = "another rejection"
    elif variant == "different_class":
        row.error_class = "PanicException"
    elif variant == "suffix":
        row.error += '; native_failure={"post_error_state":"retained"}'
    elif variant == "missing":
        row.frame_json = row.error_class = row.error = None
    elif variant == "prefix":
        row.actions.append(0)
    ledger.record(
        row,
        {
            "identities": {
                "world": managym.WORLD_VERSION,
                "engine_extension_sha256": file_sha256(Path(managym._managym.__file__)),
            }
        },
    )
    original = ledger.path.read_bytes()
    allowance = ledger.state.model_dump()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "manabot.training.state.game_errors",
            str(ledger.path),
            "--attempt",
            "1",
            "--seconds",
            "30",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    replay = json.loads(result.stdout)
    assert replay["original_outcome"] == row.outcome
    assert replay["status"] == (
        "binding_mismatch" if variant == "wrong_frame" else "failed"
    )
    assert replay["at_retained_failure"] == (variant not in {"wrong_frame", "prefix"})
    assert replay["matches_binding"] is (
        None if variant in {"missing", "prefix"} else variant != "wrong_frame"
    )
    assert replay["matches_error"] is (
        None
        if variant in {"missing", "wrong_frame"}
        else variant not in {"different_error", "different_class"}
    )
    assert ledger.path.read_bytes() == original
    assert ledger.state.model_dump() == allowance
    inherited = GameErrorLedger(
        tmp_path / "child.jsonl", policy, parents=[str(ledger.path)]
    )
    assert inherited.state.model_dump() == allowance


def test_later_crash_preserves_previous_complete_games(tmp_path: Path) -> None:
    model = agent()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", GameErrorPolicy())
    collector = _tolerant_collector(model, ledger)
    collector._env = ScriptedFailures(collector._buffers, failure_generation=1)
    batch = collector.collect(model, 7)
    markers = set(batch.obs["agent_player"][:, 0, 0, 0])
    assert markers == set(range(5)) | set(range(10, 15)) | set(range(100, 105)) | set(
        range(210, 215)
    )
    assert ledger.state.completed == 4
    assert ledger.state.crashed == 1
    assert batch.actions.size == collector.stats.learner_transitions == 20
    assert int(batch.dones.sum()) == 8


def test_deadline_retains_incomplete_games_without_regather(tmp_path: Path) -> None:
    """MTG-181 seed 18102: the deadline cut off every stream mid-game."""
    model = agent()
    policy = GameErrorPolicy(window=2, warmup=2)
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = _tolerant_collector(model, ledger)
    scripted = ScriptedFailures(collector._buffers)
    collector._env = scripted
    checks = 0

    def deadline() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise TimeoutError("original allocation deadline")

    with pytest.raises(TimeoutError, match="original allocation"):
        collector.collect(model, 1, check=deadline)
    assert ledger.state.completed == ledger.state.crashed == 0
    assert ledger.state.attempted == ledger.state.abandoned == 2
    assert ledger.state.abandoned_samples == 4 and ledger.state.discarded_samples == 0
    assert ledger.state.window == [] and ledger.state.stop_reason is None
    assert scripted.steps == [2, 2] and scripted.generation == [0, 0]
    receipts = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert all(row["attempt"]["outcome"] == "deadline" for row in receipts)
    assert not any(collector._pending) and not any(collector._streams)
    restored = GameErrorLedger(
        tmp_path / "resumed.jsonl", policy, parents=[str(ledger.path)]
    )
    assert restored.state == ledger.state


def test_deadline_after_real_crashes_counts_only_the_crashes(tmp_path: Path) -> None:
    model = agent()
    policy = GameErrorPolicy(window=4, warmup=4)
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = _tolerant_collector(model, ledger)
    scripted = ScriptedFailures(collector._buffers, always=True)
    collector._env = scripted
    checks = 0

    def deadline() -> None:
        nonlocal checks
        checks += 1
        if checks == 7:
            raise TimeoutError("collector exceeded training deadline")

    # Each stream crashes at its fourth step; the deadline lands two steps
    # into both replacement games, before the window reaches its warmup.
    with pytest.raises(TimeoutError):
        collector.collect(model, 1, check=deadline)
    assert scripted.steps == [6, 6]
    assert ledger.state.crashed == 2 and ledger.state.abandoned == 2
    assert ledger.state.window == [True, True]
    assert ledger.state.stop_reason is None
    outcomes = [
        json.loads(line)["attempt"]["outcome"]
        for line in ledger.path.read_text().splitlines()
    ]
    assert outcomes == ["engine_error", "engine_error", "deadline", "deadline"]
    # Two further real crashes complete warmup and exceed the threshold.
    ledger.record(attempt(True), {})
    ledger.record(attempt(True), {})
    assert ledger.state.stop_reason == "game_error_rate_exceeded"


def test_cancellation_settles_games_as_abandoned(tmp_path: Path) -> None:
    model = agent()
    ledger = GameErrorLedger(
        tmp_path / "attempts.jsonl", GameErrorPolicy(window=2, warmup=2)
    )
    collector = _tolerant_collector(model, ledger)
    collector._env = ScriptedFailures(collector._buffers)
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("training cancelled by launcher")

    with pytest.raises(RuntimeError, match="cancelled"):
        collector.collect(model, 1, check=cancel)
    assert ledger.state.abandoned == 2 and ledger.state.crashed == 0
    assert ledger.state.stop_reason is None
    assert all(
        json.loads(line)["attempt"]["outcome"] == "abandoned"
        for line in ledger.path.read_text().splitlines()
    )


def test_action_limit_is_a_failed_attempt_and_bounded(tmp_path: Path) -> None:
    model = agent()
    policy = GameErrorPolicy(window=2, warmup=2, max_game_actions=2)
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = _tolerant_collector(model, ledger)
    collector._env = ScriptedFailures(collector._buffers)
    with pytest.raises(GameErrorLimit):
        collector.collect(model, 1)
    assert ledger.state.completed == 0 and ledger.state.crashed == 2
    assert all(
        json.loads(line)["attempt"]["outcome"] == "action_limit"
        for line in ledger.path.read_text().splitlines()
    )


def test_truncation_is_not_a_completed_game_or_a_second_reset(tmp_path: Path) -> None:
    class TruncatedGames(ScriptedFailures):
        def step_isolated_into_buffers(
            self, actions: list[int], active: list[bool]
        ) -> None:
            super().step_isolated_into_buffers(actions, active)
            if "game_error" in self.infos[1]:
                self.infos[1] = {}
                self.buffers["truncated"][1] = 1
                # Native terminal/truncation calls already auto-reset the row.
                self.replace_failed_into_buffers([False, True])

    model = agent()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", GameErrorPolicy())
    collector = _tolerant_collector(model, ledger)
    scripted = TruncatedGames(collector._buffers)
    collector._env = scripted
    batch = collector.collect(model, 1)
    assert ledger.state.completed == collector.stats.games == 2
    assert ledger.state.crashed == collector.stats.truncations == 1
    assert scripted.generation == [1, 2]
    assert batch.actions.size == 10 and int(batch.dones.sum()) == 4


def _payment_game_worker(
    connection: multiprocessing.connection.Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
) -> None:
    managym.Env = PaymentMismatchEnv
    arena_match._game_worker(
        connection, game, registrations, checkpoint_paths, max_commands
    )


def _lost_payment_worker(
    connection: multiprocessing.connection.Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
) -> None:
    """Die or hang only after the parent acknowledges retained native inputs."""

    def send(event: tuple[str, Any]) -> None:
        connection.send(event)
        if event[0] == "pending":
            assert connection.recv() == "retained"
            if game.get("test_hang"):
                time.sleep(60)
            os._exit(7)

    arena_match._execute_game(game, registrations, checkpoint_paths, max_commands, send)


def _arena_fixture() -> tuple[dict[str, Any], list[PlayerRegistration]]:
    contract = ArenaContract.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/int-6-skill-arena-v1.json").read_text()
    )
    players = list(contract.anchors[:2])
    return {
        "match_id": "payment-failure-fixture",
        "deal_seed": 0,
        "seat_players": [p.player_id for p in players],
        "player_seeds": {p.player_id: 1 for p in players},
        "match_hypers": _payment_match().model_dump(),
        "decisions": [],
        "failure": None,
        "integrity": {
            name: 0
            for name in (
                "private_exposures",
                "root_mutations",
                "illegal_actions",
                "truncations",
            )
        },
    }, players


def test_arena_payment_failure_is_durable_and_replays_boundedly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game, players = _arena_fixture()
    journal = tmp_path / "attempt.jsonl"
    monkeypatch.setattr(arena_match, "_game_worker", _payment_game_worker)
    failed = arena_match._bounded_game(game, players, {}, 1000, 20, journal)
    assert "fixture payment mismatch" in failed["failure"]
    assert failed["winner"] is None and failed["failed_player_id"] is None
    pending = failed["pending_decision"]
    assert pending["revision"] == len(failed["decisions"])
    assert "post_state_digest" not in pending
    assert pending["chosen_offer"]["id"] == pending["command"]["offer_id"]
    assert failed["runtime_identities"]["engine_extension_sha256"]
    assert failed["failure_trace"] and failed["error_class"]
    reconstructed = read_game_journal(journal)
    failed["game_trace_sha256"] = canonical_sha256(failed)
    assert reconstructed == failed
    trace = tmp_path / "failure.jsonl.gz"
    write_trace(trace, [failed])
    original = trace.read_bytes(), journal.read_bytes()
    # Install the same deterministic payment fixture in a fresh interpreter;
    # --worker runs the CLI's worker body under this outer hard process bound.
    reproduced = _bounded_fixture_replay("manabot.arena.replay", trace, "--game", 1)
    assert reproduced["matches_binding"] and reproduced["matches_error"]
    assert reproduced["at_retained_failure"]
    assert reproduced["original_outcome"] == failed["failure"]
    for variant in ("different_error", "missing", "interrupted", "prefix"):
        diagnostic = tmp_path / f"{variant}.jsonl.gz"
        modified = deepcopy(failed)
        if variant == "different_error":
            modified["failure"] = "AgentError: another rejection"
        elif variant == "missing":
            del modified["pending_decision"]["frame_sha256"]
            modified["failure"] = "worker exited without a result"
            modified.pop("error_class", None)
        elif variant == "prefix":
            # The same native rejection now occurs before the retained boundary.
            modified["decisions"].append(deepcopy(modified["pending_decision"]))
        else:
            diagnostic = tmp_path / "interrupted.jsonl"
            diagnostic.write_text(
                "\n".join(
                    line
                    for line in journal.read_text().splitlines()
                    if json.loads(line)[0] not in {"result", "settled"}
                )
                + "\n"
            )
        if variant != "interrupted":
            modified.pop("game_trace_sha256")
            modified["game_trace_sha256"] = canonical_sha256(modified)
            write_trace(diagnostic, [modified])
        before = diagnostic.read_bytes()
        replay = _bounded_fixture_replay(
            "manabot.arena.replay", diagnostic, "--game", 1
        )
        assert replay["status"] == (
            "prefix_failed" if variant == "prefix" else "failed"
        )
        assert replay["at_retained_failure"] == (variant != "prefix")
        assert replay["matches_binding"] is (
            None if variant in {"missing", "prefix"} else True
        )
        assert replay["matches_error"] is (
            False
            if variant == "different_error"
            else True
            if variant == "prefix"
            else None
        )
        if variant == "interrupted":
            assert replay["original_outcome"] == "execution_outcome_unknown"
        assert diagnostic.read_bytes() == before
    timeout = subprocess.run(
        [
            sys.executable,
            "-m",
            "manabot.arena.replay",
            str(trace),
            "--game",
            "1",
            "--seconds",
            "0.001",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert timeout.returncode == 2, timeout.stderr
    assert json.loads(timeout.stdout)["status"] == "replay_timeout"
    # Keep the same fixture execution path and repair its payment arithmetic.
    repaired = _bounded_fixture_replay(
        "manabot.arena.replay", trace, "--game", 1, repaired=True
    )
    assert repaired["status"] == "advanced"
    assert repaired["matches_binding"] and repaired["at_retained_failure"]
    assert repaired["matches_error"] is None
    assert repaired["original_outcome"] == failed["failure"]
    assert (trace.read_bytes(), journal.read_bytes()) == original
    assert trace.stat().st_mode & 0o077 == 0
    assert journal.stat().st_mode & 0o077 == 0
    tampered = deepcopy(failed)
    tampered["pending_decision"]["chosen_offer"]["label"] = "different offer"
    tampered["game_trace_sha256"] = canonical_sha256(
        {k: v for k, v in tampered.items() if k != "game_trace_sha256"}
    )
    write_trace(tmp_path / "tampered.jsonl.gz", [tampered])
    mismatch = _bounded_fixture_replay(
        "manabot.arena.replay", tmp_path / "tampered.jsonl.gz", "--game", 1
    )
    assert mismatch["status"] == "pending_mismatch"
    assert mismatch["matches_binding"] is False
    assert mismatch["at_retained_failure"] is False
    assert mismatch["matches_error"] is None


@pytest.mark.parametrize("hang", [False, True])
def test_arena_lost_worker_retains_acknowledged_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hang: bool
) -> None:
    game, players = _arena_fixture()
    game["test_hang"] = hang
    monkeypatch.setattr(arena_match, "_game_worker", _lost_payment_worker)
    journal = tmp_path / "attempt.jsonl"
    result = arena_match._bounded_game(game, players, {}, 1000, 5, journal)
    assert result["termination_reason"] == ("timeout" if hang else "crash")
    assert result["pending_decision"]["revision"] == 0
    retained = read_game_journal(journal)
    assert retained["pending_decision"] == result["pending_decision"]
    assert not retained["terminated"] and retained["winner"] is None
    # A journal also remains useful when the parent disappears before settlement.
    unobserved = tmp_path / "unobserved.jsonl"
    lines = journal.read_text().splitlines()
    unobserved.write_text(
        "\n".join(line for line in lines if json.loads(line)[0] != "settled") + "\n"
    )
    unknown = read_game_journal(unobserved)
    assert unknown["failure"] == "execution_outcome_unknown"
    original = unobserved.read_bytes()
    replay = _bounded_fixture_replay("manabot.arena.replay", unobserved, "--game", 1)
    assert replay["status"] == "advanced"
    assert replay["original_outcome"] == "execution_outcome_unknown"
    assert replay["matches_error"] is None
    assert unobserved.read_bytes() == original
