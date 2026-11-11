"""Plan, privacy and recovery contracts without playing games or loading a policy."""

from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import mirror_segment as segment, search_mirror as mirror
from manabot.belief.sampling import SamplerInput
from manabot.sim.search import local_update
from manabot.sim.search.sampling import PhysicalHandSampler, PreparedHands


def test_original_plan_and_extension_are_distinct() -> None:
    original = mirror.planned_units()
    assert len(original) == 8000
    assert original[0][1].key == "policy/mirror-d2-b0.1"
    assert mirror.ARMS[-1].key == "heuristic/mirror-d20-b0.03-w4"
    extension = mirror.planned_units(mirror.V2_ARMS)
    counts = Counter(arm.key for _, arm in extension)
    assert len(extension) == 3000
    assert set(counts.values()) == {500}
    assert len(counts) == 6
    assert {(a.sampler, a.worlds) for _, a in extension} == {
        (sampler, worlds) for sampler in ("oracle", "uniform") for worlds in (4, 8, 16)
    }
    assert all(
        (a.opponent, a.depth, a.beta) == ("policy", 20, 0.01) for _, a in extension
    )
    old_keys = {f"{a.key}/{g.deal}" for g, a in original}
    new_keys = {f"{a.key}/{g.deal}" for g, a in extension}
    smoke_keys = {
        f"{a.key}/{g.deal}"
        for g, a in mirror.planned_units(
            mirror.extension_arms("oracle", smoke=True)
            + mirror.extension_arms("uniform", smoke=True)
        )
    }
    assert len(new_keys) == 3000
    assert old_keys.isdisjoint(new_keys)
    assert smoke_keys.isdisjoint(new_keys | old_keys)
    for stage in ("oracle", "uniform"):
        units = mirror.planned_units(mirror.extension_arms(stage))
        assert [a.worlds for _, a in units[:3]] == [4, 8, 16]
        shards = [
            {f"{a.key}/{g.deal}" for i, (g, a) in enumerate(units) if i % 7 == w}
            for w in range(7)
        ]
        assert sum(map(len, shards)) == len(set.union(*shards)) == 1500


def test_uniform_teacher_never_constructs_oracle_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = PhysicalHandSampler()
    teacher = SimpleNamespace(_sampler_authority=sampler, hand_sampler=sampler)
    constructor = Mock(return_value=teacher)
    monkeypatch.setattr(local_update, "LocalUpdateTeacher", constructor)
    oracle = Mock(side_effect=AssertionError("oracle private-hand reader used"))
    monkeypatch.setattr(mirror, "OracleHandSampler", oracle)
    result = mirror._teacher(mirror.extension_arms("uniform")[0], Path("unused.pt"))
    assert result.hand_sampler is sampler
    assert result._sampler_authority is sampler
    oracle.assert_not_called()
    config = constructor.call_args.args[2]
    assert (config.alpha, config.beta, config.worlds, config.depth) == (
        0.02,
        0.01,
        4,
        20,
    )
    assert config.sampling == "compatible_prior"
    assert config.rollout_evidence == "full"


def test_native_segment_selects_compact_evidence_and_native_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = Mock()
    monkeypatch.setattr(local_update, "LocalUpdateTeacher", constructor)
    mirror._teacher(
        mirror.extension_arms("uniform")[0], Path("unused"), compact_native=True
    )
    assert constructor.call_args.args[2].rollout_evidence == "compact"
    assert constructor.call_args.kwargs["native_encoder"] is True


def _predecessor_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str]:
    root = tmp_path / "original"
    root.mkdir()
    (root / "attempts").mkdir()
    for worker in range(7):
        (root / f"worker-{worker}.lock").touch()
    deadline = time.time() + 60
    (root / "identity.json").write_text(
        json.dumps(
            {
                "deadline": deadline,
                "workers": 7,
                "checkpoint_sha256": mirror.CHECKPOINT_SHA256,
            }
        )
    )
    oracle = _row()
    uniform_arm = mirror.extension_arms("uniform")[0]
    uniform = replace(
        oracle,
        key=f"{uniform_arm.key}/{oracle.deal}",
        condition=uniform_arm.key,
        sampler="uniform",
    )
    interrupted_arm = mirror.extension_arms("uniform")[1]
    interrupted_key = f"{interrupted_arm.key}/{oracle.deal}"
    rows = [oracle, uniform]
    (root / "games-test-0.jsonl").write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in rows)
    )
    for key, last in [
        *((row.key, {"event": "outcome", "row": asdict(row)}) for row in rows),
        (
            interrupted_key,
            {
                "event": "interrupted",
                "error": "CampaignInterrupted: campaign stop signal 15",
            },
        ),
    ]:
        path = (
            root / "attempts" / (hashlib.sha256(key.encode()).hexdigest() + ".jsonl.gz")
        )
        with gzip.open(path, "wt") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "start",
                        "key": key,
                        "checkpoint_sha256": mirror.CHECKPOINT_SHA256,
                        "deadline": deadline,
                    }
                )
                + "\n"
            )
            stream.write(json.dumps(last) + "\n")
    status = tmp_path / "etude.json"
    status.write_text(
        json.dumps(
            {
                "status": {
                    "run": "mtg269-mirror-v2",
                    "phase": "done",
                    "outcome": "cancelled",
                },
                "attempts": [{"id": "3a2b62144b9a", "finished_at": "retained"}],
            }
        )
    )
    original_plan = segment.planned_units
    # A single oracle outcome is enough for this fixture, with real schedule keys.
    monkeypatch.setattr(
        segment,
        "planned_units",
        lambda arms: (
            original_plan(arms)[:1]
            if arms[0].sampler == "oracle"
            else original_plan(arms)
        ),
    )
    return root, status, interrupted_key


def test_segment_seal_preserves_failures_and_excludes_interrupted_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, status, interrupted = _predecessor_fixture(tmp_path, monkeypatch)
    receipt = tmp_path / "predecessor.json"
    prior = segment.seal_predecessor(root, status, receipt)
    assert len(prior.completed_keys) == 2  # Includes the scored uniform failure.
    assert list(prior.interrupted_attempts) == [interrupted]
    assert len(prior.attempt_sha256) == 3
    assert interrupted not in segment.read_predecessor(receipt, prior.deadline)
    with segment.hold_predecessor(receipt):
        with (root / "worker-6.lock").open("r") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(ValueError, match="deadline"):
        segment.read_predecessor(receipt, prior.deadline + 1)
    with (root / "games-test-0.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="bytes changed"):
        segment.read_predecessor(receipt, prior.deadline)


def test_segment_seal_rejects_live_etude_or_unsealed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, status, interrupted = _predecessor_fixture(tmp_path, monkeypatch)
    original = status.read_text()
    status.write_text(original.replace('"done"', '"running"'))
    with pytest.raises(ValueError, match="not stopped"):
        segment.seal_predecessor(root, status, tmp_path / "no.json")
    status.write_text(original)
    path = (
        root
        / "attempts"
        / (hashlib.sha256(interrupted.encode()).hexdigest() + ".jsonl.gz")
    )
    with gzip.open(path, "rt") as stream:
        start = next(stream)
    with gzip.open(path, "wt") as stream:
        stream.write(start)
    with pytest.raises(ValueError, match="unreconciled attempt"):
        segment.seal_predecessor(root, status, tmp_path / "no.json")
    assert not (tmp_path / "no.json").exists()


def test_continuation_never_replays_a_predecessor_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manabot.infra import identity

    root, status, _ = _predecessor_fixture(tmp_path, monkeypatch)
    receipt = tmp_path / "predecessor.json"
    prior = segment.seal_predecessor(root, status, receipt)
    original_plan = mirror.planned_units
    monkeypatch.setattr(
        mirror,
        "planned_units",
        lambda arms: original_plan(arms) if len(arms) > 3 else original_plan(arms)[:1],
    )
    monkeypatch.setattr(mirror, "bind_extension", Mock())
    monkeypatch.setattr(identity, "file_sha256", lambda path: mirror.CHECKPOINT_SHA256)
    teacher = Mock(side_effect=AssertionError("replayed a predecessor failure"))
    monkeypatch.setattr(mirror, "_teacher", teacher)
    # read_predecessor verifies real file digests using its independently imported helper.
    mirror.play(
        Path("unused"),
        tmp_path / "next",
        0,
        7,
        None,
        mirror.extension_arms("uniform"),
        prior.deadline,
        native_sha256="admitted",
        predecessor=receipt,
    )
    teacher.assert_not_called()


def _pending_recovery_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, Path]:
    root, status, key = _predecessor_fixture(tmp_path, monkeypatch)
    trace = root / "attempts" / f"{hashlib.sha256(key.encode()).hexdigest()}.jsonl.gz"
    with gzip.open(trace, "rt") as stream:
        start = json.loads(next(stream))
    now = time.time()
    start["started"] = now
    with gzip.open(trace, "wt") as stream:
        stream.write(json.dumps(start) + "\n")
        stream.write(
            json.dumps(
                {
                    "event": "pending",
                    "step": 0,
                    "frame": {"revision": 0},
                    "observation": {"revision": 0},
                }
            )
            + "\n"
        )
    stopped = json.loads(status.read_text())
    stopped["attempts"][0].update(
        outcome="cancelled",
        started_at=datetime.fromtimestamp(now - 10, UTC).isoformat(),
        finished_at=datetime.fromtimestamp(now + 10, UTC).isoformat(),
    )
    status.write_text(json.dumps(stopped))
    (root / "logs").mkdir()
    logs: dict[str, str] = {}
    for worker in range(7):
        name = f"logs/uniform-{worker}.log"
        (root / name).write_text("retained cancellation diagnostics\n")
        logs[name] = segment.file_sha256(root / name)
    recovery = segment.InterruptionRecovery(
        etude_receipt_sha256=segment.file_sha256(status),
        pending_attempts={key: segment.file_sha256(trace)},
        worker_logs=logs,
    )
    path = tmp_path / "recovery.json"
    path.write_text(recovery.model_dump_json())
    return root, status, key, path


def test_pending_callback_needs_exact_external_cancellation_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, status, key, recovery = _pending_recovery_fixture(tmp_path, monkeypatch)
    trace = root / "attempts" / f"{hashlib.sha256(key.encode()).hexdigest()}.jsonl.gz"
    original = trace.read_bytes()
    destination = tmp_path / "predecessor.json"
    with pytest.raises(ValueError, match="unreconciled attempt"):
        segment.seal_predecessor(root, status, destination)
    prior = segment.seal_predecessor(
        root, status, destination, interruption_recovery=recovery
    )
    assert prior.schema_name == "mirror-predecessor/v2"
    assert len(prior.completed_keys) == 2
    assert key not in segment.read_predecessor(destination, prior.deadline)
    assert trace.read_bytes() == original
    (root / "logs/uniform-0.log").write_text("changed diagnostics")
    with pytest.raises(ValueError, match="log bytes changed"):
        segment.read_predecessor(destination, prior.deadline)


@pytest.mark.parametrize(
    "invalid", ["hash", "status", "window", "failure", "truncated", "unused"]
)
def test_cancellation_recovery_rejects_missing_or_conflicting_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    root, status, key, recovery_path = _pending_recovery_fixture(tmp_path, monkeypatch)
    trace = root / "attempts" / f"{hashlib.sha256(key.encode()).hexdigest()}.jsonl.gz"
    recovery = segment.InterruptionRecovery.model_validate_json(
        recovery_path.read_text()
    )
    if invalid == "hash":
        recovery.pending_attempts[key] = "0" * 64
    elif invalid == "status":
        status.write_text(status.read_text().replace('"cancelled"', '"succeeded"'))
        recovery = recovery.model_copy(
            update={"etude_receipt_sha256": segment.file_sha256(status)}
        )
    elif invalid in ("window", "failure"):
        with gzip.open(trace, "rt") as stream:
            events = [json.loads(line) for line in stream]
        if invalid == "window":
            events[0]["started"] = 0
        else:
            events.insert(1, {"event": "failure", "error": "retained error"})
        with gzip.open(trace, "wt") as stream:
            stream.write("".join(json.dumps(event) + "\n" for event in events))
        recovery.pending_attempts[key] = segment.file_sha256(trace)
    elif invalid == "truncated":
        trace.write_bytes(trace.read_bytes()[:-5])
        recovery.pending_attempts[key] = segment.file_sha256(trace)
    else:
        recovery.pending_attempts[_row().key] = "0" * 64
    recovery_path.write_text(recovery.model_dump_json())
    with pytest.raises((ValueError, EOFError)):
        segment.seal_predecessor(
            root,
            status,
            tmp_path / "rejected.json",
            interruption_recovery=recovery_path,
        )
    assert not (tmp_path / "rejected.json").exists()


def test_uniform_samples_physical_copies_and_preserves_known_cards() -> None:
    # One known A, then one unknown copy out of one A and eight B: P(B)=8/9.
    sampler = PhysicalHandSampler()
    prepared = PreparedHands(
        sampler.identity,
        "public-history",
        ("A", "B"),
        SamplerInput("schema", (2, 8), (1, 0), 2, ()),
        "{}",
    )
    batch = sampler.sample(prepared, count=1000, seed=731, check=lambda: None)
    hands = [dict(sample.counts) for sample in batch.hands]
    assert all(hand.get("A", 0) >= 1 and sum(hand.values()) == 2 for hand in hands)
    assert 0.85 < sum(hand.get("B", 0) == 1 for hand in hands) / len(hands) < 0.93


def _row() -> mirror.MirrorRow:
    game, arm = mirror.planned_units(mirror.V2_ARMS)[0]
    return mirror.MirrorRow(
        f"{arm.key}/{game.deal}",
        arm.key,
        arm.opponent,
        arm.depth,
        arm.beta,
        game.model_deck,
        game.heuristic_deck,
        game.model_seat,
        game.deal,
        None,
        "retained native failure",
        0,
        0,
        0,
        0.0,
        "test",
        arm.worlds,
        arm.sampler,
        arm.study,
        1.0,
    )


def test_resume_keeps_failures_and_rejects_changed_or_duplicate_rows(
    tmp_path: Path,
) -> None:
    row = _row()
    path = tmp_path / "games-test.jsonl"
    path.write_text(json.dumps(asdict(row)) + "\n")
    assert mirror.reconcile_extension(tmp_path) == {row.key}
    path.write_text(json.dumps(asdict(replace(row, sampler="uniform"))) + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        mirror.reconcile_extension(tmp_path)
    path.write_text((json.dumps(asdict(row)) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        mirror.reconcile_extension(tmp_path)
    path.write_text('{"key":')
    with pytest.raises(json.JSONDecodeError):
        mirror.reconcile_extension(tmp_path)


def test_incomplete_attempt_needs_reconciliation_but_other_shards_do_not(
    tmp_path: Path,
) -> None:
    row = _row()
    directory = tmp_path / "attempts"
    directory.mkdir()
    path = directory / f"{hashlib.sha256(row.key.encode()).hexdigest()}.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"event": "start", "key": row.key}) + "\n")
    with pytest.raises(ValueError, match="unreconciled attempt"):
        mirror.reconcile_extension(tmp_path)
    assert (
        mirror.reconcile_extension(tmp_path, owned_keys={"other-worker-key"}) == set()
    )
    (tmp_path / "games-test.jsonl").write_text(json.dumps(asdict(row)) + "\n")
    with pytest.raises(ValueError, match="incomplete or conflicting"):
        mirror.reconcile_extension(tmp_path)
    with gzip.open(path, "at") as stream:
        stream.write(json.dumps({"event": "outcome", "row": asdict(row)}) + "\n")
    assert mirror.reconcile_extension(tmp_path) == {row.key}


def test_worker_never_replays_a_failed_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manabot.infra import identity

    row = _row()
    game, arm = mirror.planned_units(mirror.V2_ARMS)[0]
    monkeypatch.setattr(mirror, "planned_units", lambda arms=mirror.ARMS: [(game, arm)])
    monkeypatch.setattr(mirror, "bind_extension", Mock())
    monkeypatch.setattr(identity, "file_sha256", lambda path: mirror.CHECKPOINT_SHA256)
    monkeypatch.setattr(mirror.os, "uname", lambda: SimpleNamespace(nodename="test"))
    teacher = Mock(side_effect=AssertionError("replayed a retained failure"))
    monkeypatch.setattr(mirror, "_teacher", teacher)
    (tmp_path / "games-test-0.jsonl").write_text(json.dumps(asdict(row)) + "\n")
    mirror.play(Path("unused.pt"), tmp_path, 0, 1, None, (arm,), time.time() + 60)
    teacher.assert_not_called()


def test_uniform_stage_cannot_start_before_oracle_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manabot.infra import identity

    monkeypatch.setattr(mirror, "bind_extension", Mock())
    monkeypatch.setattr(identity, "file_sha256", lambda path: mirror.CHECKPOINT_SHA256)
    teacher = Mock(side_effect=AssertionError("started uniform too early"))
    monkeypatch.setattr(mirror, "_teacher", teacher)
    with pytest.raises(ValueError, match="oracle outcomes first"):
        mirror.play(
            Path("unused.pt"),
            tmp_path,
            0,
            1,
            None,
            mirror.extension_arms("uniform"),
            time.time() + 60,
        )
    teacher.assert_not_called()


def test_campaign_stop_is_not_an_ordinary_game_failure() -> None:
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(mirror.CampaignInterrupted):
        with mirror.campaign_deadline(time.time() + 60):
            signal.raise_signal(signal.SIGALRM)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert not issubclass(mirror.CampaignInterrupted, Exception)


def test_execution_identity_prevents_resharding_and_deadline_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manabot.infra import identity

    original_hash = identity.file_sha256

    def admitted_hash(path: Path) -> str:
        if path.suffix == ".so":
            return "7c22a1e101874ae4818ba8250326277b2fb3b0a55260edcb9da0c162080baf25"
        return original_hash(path)

    monkeypatch.setattr(identity, "file_sha256", admitted_hash)
    deadline = time.time() + 60
    mirror.bind_extension(tmp_path, 7, deadline, smoke=False)
    original = (tmp_path / "identity.json").read_bytes()
    mirror.bind_extension(tmp_path, 7, deadline, smoke=False)
    assert (tmp_path / "identity.json").read_bytes() == original
    for workers, stop in ((6, deadline), (7, deadline + 1)):
        with pytest.raises(ValueError, match="assignment or deadline changed"):
            mirror.bind_extension(tmp_path, workers, stop, smoke=False)
    with pytest.raises(ValueError, match="seven-day ceiling"):
        mirror.bind_extension(tmp_path, 7, time.time() + 169 * 3600, smoke=False)
    assert (tmp_path / "identity.json").read_bytes() == original


def test_native_identity_excludes_publication_catalog_but_locks_execution_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manabot.infra import identity
    from manabot.managym import _managym as native

    original_hash = identity.file_sha256
    changed: set[str] = set()

    def admitted_hash(path: Path) -> str:
        if path.suffix == ".so":
            return "a" * 64
        if path.name in changed:
            return "b" * 64
        return original_hash(path)

    monkeypatch.setattr(identity, "file_sha256", admitted_hash)
    monkeypatch.setattr(native, "encode_observations_into", Mock(), raising=False)
    deadline = time.time() + 60
    mirror.bind_extension(tmp_path, 7, deadline, smoke=False, native_sha256="a" * 64)
    original = (tmp_path / "identity.json").read_bytes()
    assert "experiments/data.manifest.json" not in json.loads(original)["source_files"]
    changed.add("data.manifest.json")
    mirror.bind_extension(tmp_path, 7, deadline, smoke=False, native_sha256="a" * 64)
    assert (tmp_path / "identity.json").read_bytes() == original
    changed.add("search_mirror.py")
    with pytest.raises(ValueError, match="source, plan"):
        mirror.bind_extension(
            tmp_path, 7, deadline, smoke=False, native_sha256="a" * 64
        )


def test_interruption_retains_private_attempt_without_a_game_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game, arm = mirror.planned_units(mirror.V2_ARMS)[0]
    config = SimpleNamespace(model_dump=lambda **kwargs: {})
    teacher = SimpleNamespace(
        runtime_sha256="native",
        source_sha256="source",
        config=config,
        agent=SimpleNamespace(world_binding={"world": "w13"}),
        space=SimpleNamespace(encoder=SimpleNamespace(config=config)),
    )
    monkeypatch.setattr(
        mirror, "play_game", Mock(side_effect=mirror.CampaignInterrupted("deadline"))
    )
    with pytest.raises(mirror.CampaignInterrupted):
        mirror.retained_game(
            arm, game, teacher, Path("unused.pt"), "test", tmp_path, time.time() + 60
        )
    assert not list(tmp_path.glob("games-*.jsonl"))
    (path,) = (tmp_path / "attempts").glob("*.jsonl.gz")
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    assert [event["event"] for event in events] == ["start", "interrupted"]
    assert "CampaignInterrupted: deadline" in events[-1]["error"]


@pytest.mark.parametrize("repeated", [signal.SIGTERM, signal.SIGALRM])
def test_repeated_stop_does_not_interrupt_terminal_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repeated: signal.Signals
) -> None:
    game, arm = mirror.planned_units(mirror.V2_ARMS)[0]
    config = SimpleNamespace(model_dump=lambda **kwargs: {})
    teacher = SimpleNamespace(
        runtime_sha256="native",
        source_sha256="source",
        config=config,
        agent=SimpleNamespace(world_binding={"world": "w13"}),
        space=SimpleNamespace(encoder=SimpleNamespace(config=config)),
    )
    format_error = mirror.traceback.format_exc

    def interrupted_game(*args: object, **kwargs: object) -> None:
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError("first stop signal was ignored")

    def interrupted_cleanup() -> str:
        # The retained mini log shows a second SIGTERM inside format_exc().
        signal.raise_signal(repeated)
        return format_error()

    monkeypatch.setattr(mirror, "play_game", interrupted_game)
    monkeypatch.setattr(mirror.traceback, "format_exc", interrupted_cleanup)
    with pytest.raises(mirror.CampaignInterrupted):
        with mirror.campaign_deadline(time.time() + 60):
            mirror.retained_game(
                arm,
                game,
                teacher,
                Path("unused.pt"),
                "test",
                tmp_path,
                time.time() + 60,
            )
    (path,) = (tmp_path / "attempts").glob("*.jsonl.gz")
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    assert [event["event"] for event in events] == ["start", "interrupted"]
    assert "campaign stop signal 15" in events[-1]["error"]
    assert not list(tmp_path.glob("games-*.jsonl"))


def test_compact_measurement_defaults_and_invalid_options() -> None:
    from experiments.bench_mirror_compact import arguments

    required = ["--attempt", "unused.gz", "--out", "unused-output"]
    args = arguments(required)
    assert (args.inference_threads, args.modes, args.repeats, args.skip_profiles) == (
        1,
        "paired",
        5,
        False,
    )
    for options in (
        ["--inference-threads", "0"],
        ["--inference-threads", "-2"],
        ["--inference-threads", "1.5"],
        ["--repeats", "0"],
        ["--modes", "compact"],
    ):
        with pytest.raises(SystemExit) as error:
            arguments(required + options)
        assert error.value.code == 2


@pytest.mark.parametrize("threads", [1, 4])
def test_compact_measurement_sets_threads_before_model(
    tmp_path: Path,
    threads: int,
) -> None:
    # A fresh interpreter observes startup settings at the model-loading boundary.
    # Exit there: no checkpoint, search, trajectory replay or game is executed.
    program = """
import json
import torch
from experiments import bench_mirror_compact as harness
before = torch.get_num_interop_threads()
def inspect_startup(*args: object, **kwargs: object) -> None:
    print(json.dumps({"threads": torch.get_num_threads(),
                      "interop_before": before,
                      "interop_after": torch.get_num_interop_threads()}))
    raise SystemExit(0)
harness.LocalUpdateTeacher = inspect_startup
harness.main()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--attempt",
            "unused.gz",
            "--out",
            str(tmp_path / "measurement"),
            "--inference-threads",
            str(threads),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["threads"] == threads
    assert observed["interop_before"] == observed["interop_after"]
