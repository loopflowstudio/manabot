"""Provider errors and subprocesses do not expose account credentials."""

from io import BytesIO
from pathlib import Path
import subprocess
from urllib.error import HTTPError

import pytest

import manabot.remote.provider as provider
from manabot.remote.transport import Transport


def test_zero_price_catalog_entries_do_not_block_quoted_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture")

    def request(
        self: provider.RunPod, method: str, url: str, payload: object = None
    ) -> object:
        return {
            "data": {
                "gpuTypes": [
                    {"id": "NVIDIA L4", "securePrice": 0.49},
                    {"id": "unquoted", "securePrice": 0},
                    {"id": "unavailable", "securePrice": None},
                ]
            }
        }

    monkeypatch.setattr(provider.RunPod, "_request", request)
    assert provider.RunPod().prices() == {"NVIDIA L4": 0.49}


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf"), 1000])
def test_invalid_prices_still_fail_admission(
    monkeypatch: pytest.MonkeyPatch, rate: float
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture")

    def request(
        self: provider.RunPod, method: str, url: str, payload: object = None
    ) -> object:
        return {"data": {"gpuTypes": [{"id": "invalid", "securePrice": rate}]}}

    monkeypatch.setattr(provider.RunPod, "_request", request)
    with pytest.raises(provider.ProviderError, match="GPU prices unavailable"):
        provider.RunPod().prices()


def test_http_rejection_does_not_include_response_or_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-account-credential"
    monkeypatch.setenv("RUNPOD_API_KEY", secret)

    def reject(*args: object, **kwargs: object) -> None:
        raise HTTPError(
            "https://rest.runpod.io/v1/pods",
            403,
            "rejected",
            {},
            BytesIO(secret.encode()),
        )

    monkeypatch.setattr(provider, "urlopen", reject)
    with pytest.raises(provider.ProviderError) as error:
        provider.RunPod().list()
    assert "403" in str(error.value)
    assert secret not in str(error.value)


def test_ssh_does_not_inherit_account_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    monkeypatch.setenv("RUNPOD_API_KEY", "private-account-credential")
    monkeypatch.setenv("DOPPLER_TOKEN", "private-doppler-credential")
    observed: dict[str, object] = {}

    def execute(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(args, 0, b"ready", b"")

    monkeypatch.setattr(subprocess, "run", execute)
    pod = provider.Pod(
        id="test",
        name="test",
        rate=0.5,
        vcpus=6,
        memory_gb=48,
        gpu_count=1,
        public_ip="127.0.0.1",
        ports={"22": 22},
    )
    result = Transport(pod, tmp_path, time.time() + 10, tmp_path / "key").shell("true")
    assert result == b"ready"
    assert "RUNPOD_API_KEY" not in observed["env"]
    assert "DOPPLER_TOKEN" not in observed["env"]


def test_pending_pod_requested_ports_are_not_ssh_mappings() -> None:
    pending: dict[str, object] = dict(
        id="pending",
        name="probe",
        gpuCount=1,
        costPerHr=0.49,
        vcpuCount=6,
        memoryInGb=62,
        ports=["22/tcp"],
    )
    assert provider.Pod.model_validate(pending).ports == {}
    pending["portMappings"] = {"22": 12345}
    assert provider.Pod.model_validate(pending).ports == {"22": 12345}
