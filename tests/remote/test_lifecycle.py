"""Failure paths retain intent and settle only owned rentals, without network calls."""

from pathlib import Path

import pytest

import manabot.remote.deploy as lifecycle
from manabot.remote.plan import HardwareMix, compile_plan
from manabot.remote.provider import Pod, ProviderError
from tests.remote.test_compile import ROOT, SOURCE


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Provider:
    def __init__(self, clock: Clock, *, failure: str = "") -> None:
        self.clock = clock
        self.failure = failure
        self.pods: list[Pod] = []
        self.created = 0

    def list(self) -> list[Pod]:
        if self.failure == "inventory":
            raise ProviderError("unavailable")
        return list(self.pods)

    def get(self, pod_id: str) -> Pod | None:
        if pod_id == "pod1" and self.clock.now >= 1090:
            self.pods = [p for p in self.pods if p.id != pod_id]
        return next((p for p in self.pods if p.id == pod_id), None)

    def prices(self) -> dict[str, float]:
        return {"NVIDIA L4": 0.49}

    def create(self, payload: dict[str, object]) -> Pod:
        self.created += 1
        pod = Pod(
            id=f"pod{self.created}",
            name=str(payload["name"]),
            rate=2 if self.failure == "price" else 0.49,
            vcpus=6,
            memory_gb=48,
            gpu_count=1,
        )
        self.pods.append(pod)
        if self.failure == "ambiguous":
            raise ProviderError("create timed out")
        return pod

    def delete(self, pod_id: str) -> None:
        if self.failure == "delete":
            raise ProviderError("unavailable")
        self.pods = [p for p in self.pods if p.id != pod_id]


def test_unavailable_inventory_is_not_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(lifecycle, "time", clock)
    provider = Provider(clock, failure="inventory")
    attempt = lifecycle.Attempt(
        name="owned", purpose="training", intent_time=1000, deadline=1010
    )
    assert not lifecycle.confirm_delete(provider, attempt, 1005)
    assert attempt.deleted_time is None


def test_ambiguous_create_reconciles_exact_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(lifecycle, "time", clock)
    provider = Provider(clock)
    owned = provider.create({"name": "owned"})
    unrelated = provider.create({"name": "unrelated"})
    attempt = lifecycle.Attempt(
        name="owned",
        purpose="training",
        intent_time=1000,
        deadline=1010,
        create_ambiguous=True,
    )
    assert lifecycle.confirm_delete(provider, attempt, 1010)
    assert [p.id for p in provider.pods] == [unrelated.id]
    assert owned.id != unrelated.id


def test_delayed_unobserved_create_stays_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(lifecycle, "time", clock)
    provider = Provider(clock)
    attempt = lifecycle.Attempt(
        name="pending",
        purpose="training",
        intent_time=1000,
        deadline=1010,
        create_ambiguous=True,
    )
    assert not lifecycle.confirm_delete(provider, attempt, 1010)
    assert clock.now == 1010
    assert attempt.deleted_time is None


@pytest.mark.parametrize(
    "evidence", ["retained", "observed", "missing", "duplicate", "changed-plan"]
)
def test_cleanup_recovers_only_evidenced_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    clock = Clock()
    provider = Provider(clock)
    monkeypatch.setattr(lifecycle, "time", clock)
    monkeypatch.setattr(lifecycle, "RunPod", lambda: provider)
    monkeypatch.setattr(lifecycle.Path, "home", lambda: tmp_path)
    plan = compile_plan(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text(),
        HardwareMix.model_validate_json(
            (ROOT / "ops/mixes/runpod-small.json").read_text()
        ),
        SOURCE,
        197,
    )
    attempt = lifecycle.Attempt(
        name="owned",
        purpose="training",
        intent_time=900,
        deadline=990,
        hourly_rate=0.49 if evidence in ("retained", "changed-plan") else None,
        create_ambiguous=evidence in ("observed", "duplicate"),
    )
    if evidence in ("observed", "duplicate"):
        provider.create({"name": "owned"})
    if evidence == "duplicate":
        provider.create({"name": "owned"})
    unrelated = provider.create({"name": "unrelated"})
    path = tmp_path / "deployment.json"
    receipt = lifecycle.Receipt(
        plan_sha256=lifecycle.digest(plan.model_dump_json().encode()),
        started=900,
        deadline=990,
        attempts=[attempt],
        phase="cleanup-unconfirmed",
    )
    lifecycle.save(path, receipt)
    if evidence == "changed-plan":
        plan = plan.model_copy(update={"seed": 198})
    path.with_name("plan.json").write_text(plan.model_dump_json())
    recovered = lifecycle.cleanup(path)
    assert recovered.phase == "deleted"
    assert not recovered.complete
    assert [p.id for p in provider.pods] == [unrelated.id]
    if evidence in ("retained", "observed"):
        assert recovered.estimated_dollars == pytest.approx(
            (recovered.attempts[0].deleted_time - 900)
            * (0.49 + plan.mix.storage_hourly_allowance)
            / 3600
        )
    else:
        assert recovered.estimated_dollars is None
    clock.now += 60
    repeated = lifecycle.cleanup(path)
    assert repeated.estimated_dollars == recovered.estimated_dollars
    assert repeated.attempts[0].deleted_time == recovered.attempts[0].deleted_time


def test_client_calibration_callbacks_reject_before_rental(tmp_path: Path) -> None:
    plan = compile_plan(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text(),
        HardwareMix.model_validate_json(
            (ROOT / "ops/mixes/runpod-small.json").read_text()
        ),
        SOURCE,
        197,
    )

    def callback(transport: lifecycle.Transport) -> None:
        raise AssertionError("client callback must not run")

    out = tmp_path / ".runs/deployment"
    with pytest.raises(ValueError, match="client execution callbacks"):
        lifecycle.deploy(plan, out, tmp_path, after_training=callback)
    assert not out.exists()
