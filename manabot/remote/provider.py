"""Bounded RunPod control plane; only explicitly selected fields leave this module.

HTTP error bodies and validation inputs may contain credentials/account data and
are never included in exceptions. Create failures are ambiguous until reconciled.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


class ProviderError(RuntimeError):
    pass


class Pod(BaseModel):
    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, allow_inf_nan=False
    )
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str
    gpu_count: int = Field(
        validation_alias=AliasChoices(
            "gpu_count", "gpuCount", AliasPath("gpu", "count")
        ),
        ge=1,
    )
    rate: float = Field(alias="costPerHr", ge=0)
    vcpus: int = Field(alias="vcpuCount", ge=1)
    memory_gb: float = Field(alias="memoryInGb", gt=0)
    public_ip: str | None = Field(default=None, alias="publicIp")
    ports: dict[str, int] = Field(default_factory=dict, alias="portMappings")

    @field_validator("ports", mode="before")
    @classmethod
    def _published_ports(cls, value: object) -> object:
        # Pending pods return requested container ports, not public mappings.
        # They are not SSH endpoints; wait for portMappings on a later GET.
        if isinstance(value, list) and all(isinstance(p, str) for p in value):
            return {}
        return value


class RunPod:
    def __init__(self) -> None:
        self._key = os.environ.get("RUNPOD_API_KEY", "")
        if not self._key:
            raise ProviderError("RUNPOD_API_KEY is unavailable; use Doppler etude/prd")

    def _request(self, method: str, url: str, payload: object = None) -> object:
        request = Request(
            url,
            method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "User-Agent": "manabot/0.1",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read(8 * 1024 * 1024)
                return json.loads(raw) if raw else None
        except HTTPError as error:
            if method in ("GET", "DELETE") and error.code == 404:
                return None
            raise ProviderError(f"RunPod {method} returned HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise ProviderError(
                f"RunPod {method} failed; result may be ambiguous"
            ) from None

    @staticmethod
    def _pod(value: object) -> Pod:
        try:
            return Pod.model_validate(value)
        except ValidationError:
            raise ProviderError("RunPod returned an unsupported pod record") from None

    def list(self) -> list[Pod]:
        value = self._request("GET", "https://rest.runpod.io/v1/pods")
        if not isinstance(value, list):
            raise ProviderError("RunPod inventory unavailable")
        return [self._pod(item) for item in value]

    def get(self, pod_id: str) -> Pod | None:
        value = self._request("GET", f"https://rest.runpod.io/v1/pods/{pod_id}")
        return None if value is None else self._pod(value)

    def create(self, payload: dict[str, object]) -> Pod:
        return self._pod(
            self._request("POST", "https://rest.runpod.io/v1/pods", payload)
        )

    def delete(self, pod_id: str) -> None:
        self._request("DELETE", f"https://rest.runpod.io/v1/pods/{pod_id}")

    def prices(self) -> dict[str, float]:
        value = self._request(
            "POST",
            "https://api.runpod.io/graphql",
            {
                "query": "query { gpuTypes { id securePrice } }",
            },
        )
        try:
            if not isinstance(value, dict) or value.get("errors"):
                raise ValueError
            rows = value["data"]["gpuTypes"]
            result = {
                str(row["id"]): float(row["securePrice"])
                for row in rows
                if row["securePrice"] is not None
            }
            if any(not 0 <= rate < 1000 for rate in result.values()):
                raise ValueError
            # RunPod includes zero-priced catalog entries alongside rentable
            # GPUs. They supply no usable quote, but must not invalidate other
            # GPUs' positive quotes or be admitted as free rentals.
            return {gpu: rate for gpu, rate in result.items() if rate > 0}
        except (KeyError, TypeError, ValueError):
            raise ProviderError("RunPod GPU prices unavailable") from None
