"""Plan, privacy and recovery contracts without playing games or loading a policy."""

from collections import Counter
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import search_mirror as mirror
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
