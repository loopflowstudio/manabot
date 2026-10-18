"""A suspended launch must never create a rental with a spent startup budget."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path

from etude.launch.executor import Refused
from etude.launch.providers.runpod.api import PodFact, ProviderUnavailable, RunPodAPI
from etude.launch.records import Session
import pytest

from experiments.runners import etude_launch_guard as guard


def _session(at: datetime) -> Session:
    return Session(
        id="test",
        pool="test",
        machine="test",
        planned_at=at,
        claimed_at=at,
        receipt=None,
        started_at=None,
        refused_at=None,
        ended_at=None,
        version=1,
    )


def _request(api: RunPodAPI) -> PodFact:
    return api.create(
        name="etude-test",
        gpu="test",
        image="test@sha256:unused",
        registry="",
        env={"PRIVATE_VALUE": "do-not-log"},
        vcpus=4,
        memory_gb=16,
        disk_gb=40,
        entrypoint=("original", "args"),
    )


@pytest.mark.parametrize("elapsed", [121, 300, 776.559974, -1])
def test_staging_delay_refuses_before_any_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elapsed: float,
) -> None:
    at = datetime(2026, 10, 17, tzinfo=UTC)
    now = [at]
    called: list[dict[str, object]] = []

    def create(api: RunPodAPI, **options: object) -> PodFact:
        called.append(options)
        raise AssertionError("No provider request is allowed")

    monkeypatch.setattr(guard, "_inhibit_idle_sleep", lambda seconds: None)
    monkeypatch.setattr(RunPodAPI, "create", create)
    evidence = tmp_path / "guard.jsonl"
    with guard.protect_launcher(
        lambda _: _session(at), evidence, seconds=3600, clock=lambda: now[0]
    ):
        now[0] += timedelta(seconds=elapsed)
        with pytest.raises(Refused):
            _request(RunPodAPI("unused"))
    assert called == []
    events = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert events[1]["phase"] == "create_refused"
    assert events[1]["claim_age_seconds"] == elapsed
    assert "do-not-log" not in evidence.read_text()
    assert RunPodAPI.create is create


def test_fresh_request_and_provider_exception_pass_through_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = datetime(2026, 10, 17, tzinfo=UTC)
    calls: list[dict[str, object]] = []
    failure = OSError("provider may have created the machine")
    fact = PodFact(
        id="pod",
        name="etude-test",
        usd_per_hour=Decimal("1.19"),
        created_at=at,
        status="RUNNING",
    )

    def create(api: RunPodAPI, **options: object) -> PodFact:
        calls.append(options)
        if len(calls) == 2:
            raise failure
        return fact

    monkeypatch.setattr(guard, "_inhibit_idle_sleep", lambda seconds: None)
    monkeypatch.setattr(RunPodAPI, "create", create)
    with guard.protect_launcher(
        lambda _: _session(at), tmp_path / "guard.jsonl", seconds=3600, clock=lambda: at
    ):
        assert _request(RunPodAPI("unused")) is fact
        with pytest.raises(OSError) as caught:
            _request(RunPodAPI("unused"))
        assert caught.value is failure
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["env"] == {"PRIVATE_VALUE": "do-not-log"}
    assert calls[0]["entrypoint"] == ("original", "args")
    assert RunPodAPI.create is create


def test_expired_guard_rejects_even_a_fresh_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [datetime(2026, 10, 17, tzinfo=UTC)]
    monkeypatch.setattr(guard, "_inhibit_idle_sleep", lambda seconds: None)
    evidence = tmp_path / "guard.jsonl"
    with guard.protect_launcher(
        lambda _: _session(now[0]), evidence, seconds=30, clock=lambda: now[0]
    ):
        now[0] += timedelta(seconds=31)
        with pytest.raises(Refused, match="launcher_lifetime_expired"):
            _request(RunPodAPI("unused"))


def test_existing_provider_receipt_prevents_duplicate_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = datetime(2026, 10, 17, tzinfo=UTC)
    current = _session(at).model_copy(update={"receipt": "existing-pod"})
    monkeypatch.setattr(guard, "_inhibit_idle_sleep", lambda seconds: None)
    with guard.protect_launcher(
        lambda _: current, tmp_path / "guard.jsonl", seconds=30, clock=lambda: at
    ):
        with pytest.raises(Refused, match="already_settled_or_created"):
            _request(RunPodAPI("unused"))


def test_creation_failure_propagates_once_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = datetime(2026, 10, 17, tzinfo=UTC)
    calls = 0
    failure = ProviderUnavailable("POST pods: no reply")

    def create(api: RunPodAPI, **options: object) -> PodFact:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(guard, "_inhibit_idle_sleep", lambda seconds: None)
    monkeypatch.setattr(RunPodAPI, "create", create)
    evidence = tmp_path / "guard.jsonl"
    with guard.protect_launcher(
        lambda _: _session(at), evidence, seconds=3600, clock=lambda: at
    ):
        with pytest.raises(ProviderUnavailable) as caught:
            _request(RunPodAPI("unused"))
        assert caught.value is failure
    assert calls == 1
    phases = [json.loads(line)["phase"] for line in evidence.read_text().splitlines()]
    assert phases == ["guard_started", "create_allowed", "guard_stopped"]
    assert RunPodAPI.create is create
