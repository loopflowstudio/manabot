"""The collector closes a batch at the exact quota or at complete games."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal

import numpy as np
import pytest

import manabot.backends.torch as torch_backend
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.learner import Learner
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.learning import BatchClosure, DampedPolicyLearning, Learning
from manabot.training.loops import self_play as self_play_loop
from manabot.training.loops.self_play import Counts, FreshRun, SelfPlay
from manabot.training.rollout.adaptive import AdaptiveConfig
from manabot.training.rollout.collection import (
    Collector,
    CollectorSnapshot,
    capture_collector,
    close_collector,
    collect,
    open_collector,
)
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorPolicy,
    append_attempt,
    open_ledger,
    replay_attempt,
)
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from manabot.verify.games import INTERACTIVE_DECK
from tests.fixtures.dense_collector import dense_collector
from tests.training.test_two_seat_learning import ScriptedGames, _agent

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("fields", "match", "history"),
    [
        ({"learning_seats": "one", "games_per_stream": 2}, "games_per_stream", 0),
        ({"learning_seats": "both", "games_per_stream": 2}, "games_per_stream", 0),
        ({"learning_seats": "both", "snapshots": True}, "snapshot", 0),
        (
            {
                "learning_seats": "both",
                "adaptive": AdaptiveConfig(signal="U", cells=(MatchConfig(),)),
            },
            "adaptive",
            0,
        ),
        (
            {
                "learning_seats": "both",
                "opponent_mode": "learning",
                "opponent_agent": _agent(),
            },
            "learning",
            0,
        ),
        ({"learning_seats": "both"}, "recent_events", 1),
        ({"learning_seats": "one"}, "recent_events", 1),
        (
            {
                "learning_seats": "one",
                "game_errors": GameErrorPolicy(),
                "record_attempt": lambda state, _: state,
            },
            "recover_capacity",
            0,
        ),
        (
            {
                "learning_seats": "both",
                "game_errors": GameErrorPolicy(),
                "record_attempt": lambda state, _: state,
            },
            "recover_capacity",
            0,
        ),
    ],
)
def test_collector_rejects_exact_quota_with_incompatible_settings(
    fields: dict[str, Any],
    match: str,
    history: Literal[0, 1],
) -> None:
    space = ObservationSpace(ObservationConfig(policy_history_version=history))
    kwargs = dict(fields)
    opponent_mode = kwargs.pop("opponent_mode", "self")
    with pytest.raises(ValueError, match=f"exact quota.*{match}"):
        open_collector(
            space,
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode=opponent_mode,
            batch_closure="exact-quota",
            **kwargs,
            numerics=TorchCollectionSession(device="cpu", seed=0, precision="float32"),
        )


def _one_seat(closure: BatchClosure) -> Collector:
    agent = _agent()
    collector = open_collector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="one",
        batch_closure=closure,
        numerics=TorchCollectionSession(device="cpu", seed=0, precision="float32"),
    )
    collector._env = ScriptedGames(dense_collector(collector))
    return collector


def test_exact_quota_pauses_mid_game_at_the_quota() -> None:
    agent = _agent()
    batch = collect(_one_seat("exact-quota"), agent, 2)
    assert batch.dones.shape == (2, 2)
    assert not batch.next_done.all()


def test_exact_quota_counts_trajectories_within_each_column() -> None:
    batch = collect(_one_seat("exact-quota"), _agent(), 4)
    # One three-choice game and an unfinished tail in stream 0, two whole
    # two-choice games in stream 1. Rows are interleaved in flat batch order.
    assert [list(rows) for rows in batch.trajectory_rows()] == [
        [0, 2, 4],
        [6],
        [1, 3],
        [5, 7],
    ]
    assert batch.bootstrapped_tail_fraction() == 0.25


def test_one_seat_complete_games_end_every_trajectory_at_a_result() -> None:
    agent = _agent()
    collector = _one_seat("complete-games")
    batch = collect(collector, agent, 2)
    assert batch.ordered_obs()["agent_player"].shape[1] == 1
    assert bool(batch.next_done.all())
    # Every packed trajectory closes on a terminal row, so nothing bootstraps.
    assert bool(batch.dones[-1].all())
    assert np.count_nonzero(batch.dones) >= 1
    assert collector.stats.games >= 2
    # Stream 1's shorter game leaves it idle while stream 0 finishes.
    assert collector.stats.idle_stream_steps > 0


def _interactive_match() -> Match:
    return Match(
        MatchConfig(
            hero="hero",
            villain="villain",
            hero_deck=INTERACTIVE_DECK,
            villain_deck=INTERACTIVE_DECK,
        )
    )


def _two_seat_exact_collector(
    agent: Any,
    *,
    num_envs: int = 4,
    seed: int = 11,
    workers: int = 1,
    match: Match | None = None,
    space: ObservationSpace | None = None,
    game_errors: GameErrorPolicy | None = None,
    record_attempt: Any = None,
    snapshot: CollectorSnapshot | None = None,
    collection_pipeline: bool = False,
    execution_rows: int | None = None,
) -> Collector:
    return open_collector(
        space or agent.observation_space,
        match or _interactive_match(),
        Reward(RewardConfig()),
        num_envs=num_envs,
        seed=seed,
        workers=workers,
        execution_rows=execution_rows,
        opponent_mode="self",
        learning_seats="both",
        batch_closure="exact-quota",
        collection_pipeline=collection_pipeline,
        game_errors=game_errors,
        record_attempt=record_attempt,
        snapshot=snapshot,
        numerics=TorchCollectionSession(device="cpu", seed=seed, precision="float32"),
    )


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
@pytest.mark.parametrize("workers", [1, 2])
def test_two_seat_exact_quota_zero_idle_mixed_spans_and_cross_update_finish(
    kind: Literal["scalar", "categorical_wdl"],
    workers: int,
) -> None:
    agent = _agent(kind)
    collector = _two_seat_exact_collector(agent, num_envs=4, seed=17, workers=workers)
    try:
        # Small quota cuts games mid-play; later updates finish those same seeds.
        initial_seeds = list(collector._env.current_game_seeds())
        initial_generations = list(collector._env.current_game_generations())
        first = collect(collector, agent, 8)
        assert first.actions.shape == (32, 1)
        assert collector.stats.idle_stream_steps == 0
        assert collector.stats.learner_transitions == 32
        assert collector.stats.micro_steps == 32
        assert first.trajectories is not None
        assert not any(collector._streams)
        assert all(p is None for p in collector._pending)
        for span in first.trajectories:
            terminal = bool(first.dones[span.stop - 1, 0])
            assert terminal != (span.bootstrap_index is not None)
        assert first.bootstrapped_tail_fraction() > 0

        # At least one game stays alive across the update boundary with its
        # original seed and generation and finishes in a later batch.
        cut_streams = [
            row
            for row in range(4)
            if collector._env.current_game_generations()[row]
            == initial_generations[row]
        ]
        assert cut_streams
        assert all(
            collector._env.current_game_seeds()[row] == initial_seeds[row]
            for row in cut_streams
        )

        learner = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
        if kind == "categorical_wdl":
            result = update_damped_iteration(
                torch_backend,
                learner,
                first,
                DampedPolicyLearning(
                    gradient="damped_policy", advantage_quantile=0, min_advantage=0
                ),
                1,
            )
        else:
            result = update_iteration(
                torch_backend,
                learner,
                first,
                Learning(epochs=1, minibatches=2),
                0.0,
                np.random.default_rng(1),
                iteration=1,
            )
        assert result["rows"] == 32
        assert result["bootstrapped_tail_fraction"] > 0

        # Collect until the cut games finish with real terminal rewards.
        saw_finish = False
        for _ in range(12):
            nxt = collect(collector, agent, 24)
            assert nxt.actions.shape == (96, 1)
            assert collector.stats.idle_stream_steps == 0
            assert nxt.trajectories is not None
            if any(
                collector._env.current_game_generations()[row]
                != initial_generations[row]
                for row in cut_streams
            ):
                saw_finish = True
                assert np.any(nxt.dones)
                break
        assert saw_finish
        assert collector.stats.games > 0
    finally:
        close_collector(collector)


def test_two_seat_exact_quota_cross_update_failure_receipt_replays_and_keeps_learned(
    tmp_path: Path,
) -> None:
    space = ObservationSpace(ObservationConfig(max_permanents_per_player=1))
    match = Match(MatchConfig(hero_deck={"Forest": 12}, villain_deck={"Forest": 12}))
    agent = _agent("scalar")
    agent = type(agent)(space, agent.config)
    policy = GameErrorPolicy(threshold=0.5, recover_capacity=True)
    ledger, _ = open_ledger(tmp_path / "attempts.jsonl", policy)
    collector = _two_seat_exact_collector(
        agent,
        num_envs=2,
        seed=7,
        match=match,
        space=space,
        game_errors=policy,
        record_attempt=lambda state, row: append_attempt(ledger, state, row, {}),
    )
    try:
        # First update cuts mid-game before permanent capacity overflows.
        first = collect(collector, agent, 2)
        assert first.actions.shape == (4, 1)
        assert collector.stats.idle_stream_steps == 0
        assert collector.game_error_state.crashed == 0
        # Stream-keyed learned rows count both seats combined.
        assert set(collector.stats.learned_game_rows) <= {0, 1}
        assert sum(collector.stats.learned_game_rows.values()) == 4
        first_prefixes = [list(prefix) for prefix in collector._game_actions]
        assert all(len(prefix) == 2 for prefix in first_prefixes)

        # Continue collecting until a cut game overflows capacity in a later update.
        failure: GameAttempt | None = None
        for _ in range(10):
            batch = collect(collector, agent, 4)
            assert batch.actions.shape == (8, 1)
            assert batch.trajectories is not None
            assert all(p is None for p in collector._pending)
            attempts = [
                GameAttempt.model_validate(json.loads(line)["attempt"])
                for line in ledger.path.read_text().splitlines()
            ]
            failed = [
                row
                for row in attempts
                if row.outcome == "resource_limit" and row.learned_samples > 0
            ]
            if failed:
                failure = failed[0]
                break
        assert failure is not None
        assert failure.learned_samples >= 2
        assert len(failure.actions) > failure.learned_samples
        replayed = replay_attempt(failure)
        assert replayed["status"] == "failed"
        assert replayed["at_retained_failure"] is True
        assert replayed["matches_error"] is True
        assert replayed["matches_binding"] is True
    finally:
        close_collector(collector)


@pytest.mark.parametrize("collection_pipeline", [False, True])
def test_two_seat_exact_quota_mid_game_save_restore_matches_in_fresh_process(
    tmp_path: Path,
    collection_pipeline: bool,
) -> None:
    agent = _agent("scalar")
    policy = GameErrorPolicy(threshold=0.5, recover_capacity=True)
    ledger_a, _ = open_ledger(tmp_path / "a.jsonl", policy)
    uninterrupted = _two_seat_exact_collector(
        agent,
        num_envs=2,
        seed=23,
        collection_pipeline=collection_pipeline,
        game_errors=policy,
        record_attempt=lambda state, row: append_attempt(ledger_a, state, row, {}),
    )
    try:
        collect(uninterrupted, agent, 5)
        snapshot = capture_collector(uninterrupted)
        assert snapshot.learning_seats == "both"
        assert snapshot.batch_closure == "exact-quota"
        assert snapshot.cohorts == (2 if collection_pipeline else 1)
        if collection_pipeline:
            with pytest.raises(
                ValueError, match="incompatible collector stream geometry"
            ):
                _two_seat_exact_collector(agent, num_envs=2, seed=23, snapshot=snapshot)
        assert set(snapshot.learned_game_rows) <= {0, 1}
        assert any(root.recipe.actions for root in snapshot.streams)
        rng_state = deepcopy(uninterrupted.numerics.snapshot_rng())
        expected = collect(uninterrupted, agent, 7)
        expected_digests = uninterrupted._env.state_digests()
        expected_learned = dict(uninterrupted.stats.learned_game_rows)
    finally:
        close_collector(uninterrupted)

    # Restore in a fresh Python process from serialized snapshot + weights + RNG.
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(snapshot.model_dump_json())
    weights_path = tmp_path / "agent.pt"
    rng_path = tmp_path / "rng.pt"
    import torch

    torch.save(agent.state_dict(), weights_path)
    torch.save(rng_state, rng_path)
    out_path = tmp_path / "restored.json"
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "import torch\n"
        "from manabot.backends.torch.collection import CollectionSession\n"
        "from manabot.backends.torch.policy import ActorCritic\n"
        "from manabot.env.match import Match, Reward\n"
        "from manabot.env.observation import ObservationSpace\n"
        "from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig\n"
        "from manabot.training.rollout.collection import (\n"
        "    CollectorSnapshot, close_collector, collect, open_collector\n"
        ")\n"
        "from manabot.training.state.game_errors import GameErrorPolicy, append_attempt, open_ledger\n"
        "from manabot.verify.games import INTERACTIVE_DECK\n"
        f"root = Path({str(tmp_path)!r})\n"
        "agent = ActorCritic(ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8))\n"
        "agent.load_state_dict(torch.load(root / 'agent.pt', weights_only=True))\n"
        "snap = CollectorSnapshot.model_validate_json((root / 'snapshot.json').read_text())\n"
        "policy = GameErrorPolicy(threshold=0.5, recover_capacity=True)\n"
        "ledger, _ = open_ledger(root / 'b.jsonl', policy)\n"
        "session = CollectionSession(device='cpu', seed=23, precision='float32')\n"
        "session.restore_rng(torch.load(root / 'rng.pt', weights_only=False))\n"
        "match = Match(MatchConfig(hero='hero', villain='villain', hero_deck=INTERACTIVE_DECK, villain_deck=INTERACTIVE_DECK))\n"
        "c = open_collector(\n"
        "    agent.observation_space, match, Reward(RewardConfig()),\n"
        "    num_envs=2, seed=23, opponent_mode='self', learning_seats='both',\n"
        "    batch_closure='exact-quota', game_errors=policy,\n"
        f"    collection_pipeline={collection_pipeline!r}, workers=2,\n"
        "    record_attempt=lambda s, r: append_attempt(ledger, s, r, {}),\n"
        "    snapshot=snap, numerics=session,\n"
        ")\n"
        "try:\n"
        "    batch = collect(c, agent, 7)\n"
        "    payload = {\n"
        "        'actions': batch.actions.tolist(),\n"
        "        'rewards': batch.rewards.tolist(),\n"
        "        'dones': batch.dones.tolist(),\n"
        "        'values': batch.values.tolist(),\n"
        "        'trajectories': [(s.start, s.stop, s.seat, s.bootstrap_index) for s in batch.trajectories or ()],\n"
        "        'digests': c._env.state_digests(),\n"
        "        'learned': c.stats.learned_game_rows,\n"
        "    }\n"
        "    (root / 'restored.json').write_text(json.dumps(payload))\n"
        "finally:\n"
        "    close_collector(c)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    actual = json.loads(out_path.read_text())
    assert actual["actions"] == expected.actions.tolist()
    assert actual["rewards"] == expected.rewards.tolist()
    assert actual["dones"] == expected.dones.tolist()
    np.testing.assert_allclose(actual["values"], expected.values.tolist(), atol=1e-6)
    assert actual["trajectories"] == [
        [s.start, s.stop, s.seat, s.bootstrap_index]
        for s in expected.trajectories or ()
    ]
    assert actual["digests"] == expected_digests
    assert {int(k): v for k, v in actual["learned"].items()} == expected_learned


@pytest.mark.parametrize("collection_pipeline", [False, True])
def test_self_play_step_splits_rates_and_logs_viewer_bootstrap_gap(
    tmp_path: Path,
    collection_pipeline: bool,
) -> None:
    base = gray_ogre_self_play(horizon=2).model_dump()
    settings = SelfPlay.model_validate(
        {
            **base,
            "streams": 4,
            "transitions": 6,
            "batch_closure": "exact-quota",
            "collection_pipeline": collection_pipeline,
            "learning_seats": "both",
            "learning": DampedPolicyLearning(
                gradient="damped_policy", advantage_quantile=0.0, min_advantage=0.0
            ),
        }
    )
    learner, ema = self_play_loop.create(settings, seed=5)
    counts = Counts()
    rng = np.random.default_rng(5)
    run = FreshRun(tmp_path / "out", 5)
    with self_play_loop.self_play_games(settings, run, counts) as collector:
        row1 = self_play_loop.step(
            settings, learner, ema, collector, counts, rng, out=run.out
        )
        row2 = self_play_loop.step(
            settings, learner, ema, collector, counts, rng, out=run.out
        )
    for row in (row1, row2):
        assert row["rows_per_second"] > 0
        assert row["updates_per_hour"] > 0
        assert "priority_tap_rate" in row
        assert "decisions_per_game" not in row
        assert "skipped_share" not in row
    assert row2["coordinates/idle_stream_steps"] == 0
    assert row2["coordinates/learner_transitions"] == 48
    assert row2["viewer_bootstrap_gap/count"] >= 1
    assert np.isfinite(row2["viewer_bootstrap_gap/mean_abs"])
    assert np.isfinite(row2["viewer_bootstrap_gap/mean_signed"])


def test_two_seat_pipeline_is_deterministic_across_worker_counts() -> None:
    agent = _agent("scalar")
    expected = None
    for workers in (1, 2, 4):
        collector = _two_seat_exact_collector(
            agent,
            num_envs=8,
            workers=workers,
            collection_pipeline=True,
            execution_rows=3,
        )
        try:
            batch = collect(collector, agent, 20)
            assert collector.stats.idle_stream_steps == 0
            assert batch.actions.shape == (160, 1)
            result = (
                batch.actions.tolist(),
                batch.rewards.tolist(),
                batch.logprobs.tolist(),
                batch.values.tolist(),
                batch.trajectories,
                capture_collector(collector).model_dump_json(),
            )
            if expected is None:
                expected = result
            else:
                assert result == expected
        finally:
            close_collector(collector)


def test_two_seat_exact_quota_handles_all_terminal_boundary_with_zero_cuts() -> None:
    """When every stream ends a game on step num_steps, explicit spans have zero cuts."""
    agent = _agent("scalar")
    collector = _two_seat_exact_collector(agent, num_envs=2, seed=0)
    try:
        # ScriptedGames has 5 moves on stream 0 and 3 moves on stream 1; after
        # 15 moves (LCM of 5 and 3), both streams end cleanly on step 15.
        collector._env = ScriptedGames(dense_collector(collector))
        batch = collect(collector, agent, 15)
        assert batch.actions.shape == (30, 1)
        assert collector.stats.idle_stream_steps == 0
        assert batch.trajectories is not None
        assert len(batch.next_done) == 0
        assert all(span.bootstrap_index is None for span in batch.trajectories)
        assert batch.bootstrapped_tail_fraction() == 0.0
        learner = Learner(agent, "cpu", minibatch_rows=2)
        result = update_iteration(
            torch_backend,
            learner,
            batch,
            Learning(epochs=1, minibatches=2),
            0.0,
            np.random.default_rng(0),
            iteration=1,
        )
        assert result["rows"] == 30
    finally:
        close_collector(collector)
