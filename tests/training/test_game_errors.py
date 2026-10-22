"""Deterministic failures through the ordinary collector and update boundary."""

from collections.abc import Mapping
from copy import deepcopy
import inspect
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
from manabot.arena.replay import read_game_journal, replay_failure, write_trace
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra import Experiment
from manabot.infra.hypers import (
    AgentSpec,
    ExperimentHypers,
    MatchHypers,
    RewardHypers,
    TrainHypers,
)
from manabot.model.agent import Agent
from manabot.remote.plan import Allocation, JobSpec, Machine
from manabot.sim.net_opponent import (
    NetOpponentTrainer,
    SeatRoutedCollector,
    transition_gae,
)
from manabot.training import execution
from manabot.training.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    replay_attempt,
)
from manabot.training.models import (
    Learning,
    RecoveryPolicy,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.objectives import update_iteration
from manabot.verify.store import VerifyStore
import managym


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


def agent() -> Agent:
    torch.set_num_threads(1)
    return Agent(ObservationSpace(), AgentSpec(attention_on=False, hidden_dim=8))


def attempt(failed: bool, seed: int = 0) -> GameAttempt:
    return GameAttempt(
        stream=0,
        seed=seed,
        seat=0,
        matchup=MatchHypers(),
        outcome="engine_error" if failed else "completed",
        discarded_samples=3 if failed else 0,
        valid_samples=4 if not failed else 0,
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
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        learning_seats=seats,
        game_errors=policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
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
    experiment = Experiment(
        ExperimentHypers(wandb=False, device="cpu", runs_dir=tmp_path / "train")
    )
    trainer = NetOpponentTrainer(
        model, experiment, collector, TrainHypers(num_envs=2, num_steps=1)
    )
    result = update_iteration(
        trainer,
        batch,
        Learning(epochs=1, minibatches=1),
        0,
        np.random.default_rng(1),
        iteration=1,
    )
    assert result["rows"] == batch.actions.size
    assert trainer.optimizer.state
    experiment.close()


def test_all_failures_stop_boundedly_and_count_once_per_game(tmp_path: Path) -> None:
    model = agent()
    policy = GameErrorPolicy(window=4, warmup=4)
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        game_errors=policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
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
        Reward(RewardHypers()),
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
        matchup=MatchHypers(),
        outcome="engine_error",
        actions=[-1],
        error=infos[1]["game_error"],
        frame_json=infos[1]["game_error_frame"],
    )
    replay = replay_attempt(row)
    assert replay["status"] == "failed" and replay["at_retained_failure"]
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


def test_original_failure_is_immutable_and_cli_replays(tmp_path: Path) -> None:
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    row = attempt(True)
    row.actions = [-1]
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
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "manabot.training.game_errors",
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
    assert json.loads(result.stdout)["at_retained_failure"]
    assert ledger.path.read_bytes() == original


# This fixture deliberately models a payment validator disagreeing with native
# offers. It is deterministic from setup/seed/actions and shared by scalar replay
# and vector collection. It is not a reproduction or repair of MTG-175.
_NATIVE_ENV = managym.Env


def _payment_match() -> MatchHypers:
    # A narrow native fixture: every mana source is Mountain, every spell is 2R.
    return MatchHypers(
        hero_deck={"Mountain": 24, "Gray Ogre": 36},
        villain_deck={"Mountain": 24, "Gray Ogre": 36},
    )


def _broken_payment(red: int) -> bool:
    # Fault: consume every available R when satisfying the one colored symbol,
    # leaving nothing for the two generic mana. Native offers remain correct.
    return red >= 1 and red - red >= 2


def _repaired_payment(red: int) -> bool:
    # Spend just one R; the remainder can pay generic mana.
    return red >= 1 and red - 1 >= 2


class PaymentMismatchEnv:
    payment = staticmethod(_broken_payment)

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        self.seed = seed
        self.native = _NATIVE_ENV(seed=seed, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Test adapter forwards the native extension's untyped interface.
        return getattr(self.native, name)

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.native.set_seed(seed)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        frame = json.loads(self.native.semantic_decision_frame_json())
        offer = frame["offers"][action]
        if (
            self.seed % 4 == 0
            and offer.get("public_commitment", {}).get("card") == "Gray Ogre"
            and offer["verb"] == "cast"
        ):
            raw = self.native.observation_for_player(frame["actor"])
            mountains = {
                card.registry_key for card in raw.agent_cards if card.name == "Mountain"
            }
            red = sum(raw.agent.mana) + sum(
                not permanent.tapped and permanent.registry_key in mountains
                for permanent in raw.agent_permanents
            )
            assert _repaired_payment(red), (
                "fixture must select an actually payable offer"
            )
            if not self.payment(red):
                raise managym.AgentError(
                    f"fixture payment mismatch: not enough producible mana (R={red}; cost=2R)"
                )
        return self.native.step(action)


class PaymentMismatchVector:
    def __init__(self, collector: SeatRoutedCollector) -> None:
        self.native = collector._env
        self.configs = [match.to_rust() for match in collector.stream_matches]
        self.shadows = [self._new_shadow(row) for row in range(len(self.configs))]
        self.infos: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def _new_shadow(self, row: int) -> PaymentMismatchEnv:
        shadow = PaymentMismatchEnv(
            seed=self.native.current_game_seeds()[row], skip_trivial=True
        )
        shadow.reset(self.configs[row])
        return shadow

    def reset_seeds_into_buffers(self, seeds: list[int]) -> None:
        self.native.reset_seeds_into_buffers(seeds)
        self.shadows = [self._new_shadow(row) for row in range(len(self.configs))]

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        self.native.step_into_buffers(actions, active)
        for row, enabled in enumerate(active):
            if enabled:
                self.shadows[row].step(actions[row])

    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool]
    ) -> None:
        enabled = list(active)
        failures: dict[int, dict[str, Any]] = {}
        done_rows: list[int] = []
        for row, shadow in enumerate(self.shadows):
            if not active[row]:
                continue
            frame = shadow.semantic_decision_frame_json()
            try:
                _, _, done, truncated, _ = shadow.step(actions[row])
                if done or truncated:
                    done_rows.append(row)
            except managym.AgentError as error:
                enabled[row] = False
                failures[row] = {
                    "game_error": str(error),
                    "game_error_class": type(error).__name__,
                    "game_error_frame": frame,
                    "game_error_trace": "deterministic offered-cast payment fixture",
                }
        self.native.step_isolated_into_buffers(actions, enabled)
        self.infos = self.native.get_last_info()
        for row, failure in failures.items():
            self.infos[row] = failure
        for row in done_rows:
            self.shadows[row] = self._new_shadow(row)

    def get_last_info(self) -> list[dict[str, Any]]:
        return self.infos

    def replace_failed_into_buffers(self, failed: list[bool]) -> None:
        self.native.replace_failed_into_buffers(failed)
        for row, replace in enumerate(failed):
            if replace:
                self.shadows[row] = self._new_shadow(row)


class PaymentMismatchCollector(SeatRoutedCollector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = PaymentMismatchVector(self)


def _bounded_fixture_replay(
    module: str, path: Path, selector: str, index: int
) -> dict[str, Any]:
    script = """
import importlib, runpy, sys
import managym
fixture = runpy.run_path(sys.argv[1])
managym.Env = fixture['PaymentMismatchEnv']
main = importlib.import_module(sys.argv[2]).main
sys.argv = ['replay', sys.argv[3], sys.argv[4], sys.argv[5], '--worker']
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            __file__,
            module,
            str(path),
            selector,
            str(index),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_repaired_restart_preserves_post_snapshot_failures_and_costs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = agent()
    recipe = TrainingRegime(
        id="game-recovery",
        world=managym.WORLD_VERSION,
        match=_payment_match(),
        agent=model.hypers,
        schedule_clock="iteration_fraction",
        wall_seconds=120,
        recovery=RecoveryPolicy(checkpoint_updates=1),
        stages=[
            TrainSelfPlay(
                id="train",
                operation="train_self_play",
                updates=3,
                streams=2,
                transitions=1,
                game_errors=GameErrorPolicy(),
                learning=Learning(epochs=1, minibatches=1, retained_fraction=1),
            )
        ],
    )
    # A local receipt fixture exercises an unchanged absolute deadline. No
    # provider, launcher or paid execution is invoked by constructing these data.
    allocation = Allocation(
        spec=JobSpec(
            machine=Machine(
                gpu_types=("fixture",),
                vcpus=1,
                memory_gb=1,
                image="fixture@sha256:" + "0" * 64,
                hourly_ceiling=0.1,
            ),
            lifetime_hours=1,
            spending_limit=1,
        ),
        admitted_at=time.time(),
    )
    original_fingerprints = execution._runtime_identities

    def fixture_identities(
        seed: int, regime: TrainingRegime, space: ObservationSpace
    ) -> dict[str, object]:
        identities = original_fingerprints(seed, regime, space)
        return {
            **identities,
            "fixture_source_sha256": file_sha256(Path(__file__)),
            "training_source_sha256": canonical_sha256(
                {
                    "training": identities["training_source_sha256"],
                    "fixture_payment_source": inspect.getsource(
                        PaymentMismatchEnv.payment
                    ),
                }
            ),
        }

    monkeypatch.setattr(execution, "_runtime_identities", fixture_identities)
    record = GameErrorLedger.record

    def interrupt_after_second_crash(
        self: GameErrorLedger, row: GameAttempt, context: Mapping[str, object]
    ) -> GameErrorState:
        record(self, row, context)
        if self.state.crashed == 2:
            raise KeyboardInterrupt("after durable failure, before next update")
        return self.state

    with VerifyStore(tmp_path / "training.sqlite") as store:
        with monkeypatch.context() as patch:
            patch.setattr(execution, "SeatRoutedCollector", PaymentMismatchCollector)
            patch.setattr(GameErrorLedger, "record", interrupt_after_second_crash)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(
                    recipe, 197, tmp_path / "first", store, allocation=allocation
                )
        parent_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        parent = store.training_run(parent_id)
        assert 1 <= len(parent.stages[0].diagnostics) < 3
        assert parent.game_errors is not None and parent.game_errors.crashed == 2
        original = Path(parent.game_error_ledgers[0]).read_bytes()
        entries = [json.loads(line) for line in original.splitlines()]
        line_index, failed_entry = next(
            (index, entry)
            for index, entry in reversed(list(enumerate(entries, 1)))
            if entry["attempt"]["outcome"] == "engine_error"
        )
        retained = GameAttempt.model_validate(failed_entry["attempt"])
        assert retained.frame_json is not None
        offer = json.loads(retained.frame_json)["offers"][retained.actions[-1]]
        assert offer["verb"] == "cast"
        assert offer["public_commitment"]["card"] == "Gray Ogre"
        assert retained.discarded_samples > 0
        with monkeypatch.context() as patch:
            patch.setattr(managym, "Env", PaymentMismatchEnv)
            replay = replay_attempt(retained)
        assert replay["status"] == "failed" and replay["at_retained_failure"]
        assert replay["error"] == retained.error
        assert replay["frame"] == retained.frame_json
        with monkeypatch.context() as patch:
            patch.setattr(
                PaymentMismatchEnv, "payment", staticmethod(_repaired_payment)
            )
            patch.setattr(managym, "Env", PaymentMismatchEnv)
            assert replay_attempt(retained)["status"] == "advanced"
        bounded = _bounded_fixture_replay(
            "manabot.training.game_errors",
            Path(parent.game_error_ledgers[0]),
            "--attempt",
            line_index,
        )
        assert bounded["at_retained_failure"] and bounded["error"] == retained.error
        assert failed_entry["context"]["coordinates"]["updates"] == len(
            parent.stages[0].diagnostics
        )
        assert parent.recovery_artifact is not None
        frozen_snapshot = Path(parent.recovery_artifact["path"]).read_bytes()
        monkeypatch.setattr(
            PaymentMismatchEnv, "payment", staticmethod(_repaired_payment)
        )
        assert (
            fixture_identities(197, recipe, model.observation_space)[
                "training_source_sha256"
            ]
            != parent.identities["training_source_sha256"]
        )
        with pytest.raises(ValueError, match="identity mismatch"):
            execution.execute_regime(
                recipe,
                197,
                tmp_path / "unapproved",
                store,
                allocation=allocation,
                resume_from=parent.id,
            )

        with monkeypatch.context() as patch:
            patch.setattr(
                execution,
                "_runtime_identities",
                lambda *args: {**fixture_identities(*args), "world": "incompatible"},
            )
            with pytest.raises(ValueError, match="identity mismatch"):
                execution.execute_regime(
                    recipe,
                    197,
                    tmp_path / "wrong-world",
                    store,
                    allocation=allocation,
                    resume_from=parent.id,
                    source_repair_reason="A source repair cannot admit another world",
                )

        def unavailable(*args: object, **kwargs: object) -> SeatRoutedCollector:
            raise RuntimeError("repair attempt failed during collector setup")

        with monkeypatch.context() as patch:
            patch.setattr(execution, "SeatRoutedCollector", unavailable)
            with pytest.raises(RuntimeError, match="collector setup"):
                execution.execute_regime(
                    recipe,
                    197,
                    tmp_path / "setup-failed",
                    store,
                    allocation=allocation,
                    resume_from=parent.id,
                    source_repair_reason="Deterministic payment fixture repaired; unchanged world and inputs",
                )
        failed = store.training_run(
            store.con.execute(
                "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
        )
        assert failed.recovery_artifact == parent.recovery_artifact
        monkeypatch.setattr(execution, "SeatRoutedCollector", PaymentMismatchCollector)
        child = execution.execute_regime(
            recipe,
            197,
            tmp_path / "second",
            store,
            allocation=allocation,
            resume_from=failed.id,
        )
        assert child.status == "completed"
        assert child.game_errors is not None and child.game_errors.crashed == 2
        assert child.game_errors.attempted > parent.game_errors.attempted
        assert len(child.stages[0].diagnostics) == 3
        assert child.prior_seconds == failed.prior_seconds + failed.seconds
        assert (
            child.prior_watchdog_seconds
            == failed.prior_watchdog_seconds + failed.watchdog_seconds
        )
        assert child.regime.wall_seconds == parent.regime.wall_seconds
        assert child.parent_run_id == failed.id and child.source_repair_reason
        assert Path(parent.game_error_ledgers[0]).read_bytes() == original
        assert (
            child.stages[0].diagnostics[: len(parent.stages[0].diagnostics)]
            == parent.stages[0].diagnostics
        )
        assert Path(parent.recovery_artifact["path"]).read_bytes() == frozen_snapshot
        assert child.allocation_deadline == parent.allocation_deadline
        assert child.allocation_deadline == allocation.deadline
        assert child.optimizer_updates_through() == 3
        exported = TrainingRun.model_validate_json(
            (tmp_path / "second/run.json").read_text()
        )
        assert exported.status == "completed"
        assert exported.game_errors == child.game_errors
        assert exported.game_errors.attempted == exported.game_errors.completed + 2
        assert exported.game_errors.discarded_samples > 0
        assert exported.game_errors.replacement_samples > 0
        assert store.training_run(parent.id) == parent


def test_later_crash_preserves_previous_complete_games(tmp_path: Path) -> None:
    model = agent()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", GameErrorPolicy())
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        game_errors=ledger.policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
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
    model = agent()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", GameErrorPolicy())
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        game_errors=ledger.policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
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
    assert ledger.state.completed == 0 and ledger.state.crashed == 2
    assert scripted.steps == [2, 2] and scripted.generation == [0, 0]
    receipts = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert all(row["attempt"]["outcome"] == "deadline" for row in receipts)
    assert not any(collector._pending) and not any(collector._streams)


def test_action_limit_is_a_failed_attempt_and_bounded(tmp_path: Path) -> None:
    model = agent()
    policy = GameErrorPolicy(window=2, warmup=2, max_game_actions=2)
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        game_errors=policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
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
    collector = SeatRoutedCollector(
        model.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        opponent_mode="self",
        game_errors=ledger.policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
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
    assert reproduced["matches_retained_error"]
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
    with monkeypatch.context() as patch:
        patch.setattr(PaymentMismatchEnv, "payment", staticmethod(_repaired_payment))
        patch.setattr(managym, "Env", PaymentMismatchEnv)
        assert replay_failure(failed)["status"] == "advanced"
    assert (trace.read_bytes(), journal.read_bytes()) == original
    assert trace.stat().st_mode & 0o077 == 0
    assert journal.stat().st_mode & 0o077 == 0
    tampered = deepcopy(failed)
    tampered["pending_decision"]["chosen_offer"]["label"] = "different offer"
    tampered["game_trace_sha256"] = canonical_sha256(
        {k: v for k, v in tampered.items() if k != "game_trace_sha256"}
    )
    assert replay_failure(tampered)["status"] == "pending_mismatch"


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
    assert replay_failure(unknown)["status"] == "advanced"
