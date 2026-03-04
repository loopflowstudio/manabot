"""
provider.py
Provider-agnostic machine contracts for GPU sandbox and job workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class MachineSpec:
    """Machine provisioning settings loaded from ops/specs/*.yaml."""

    instance_type: str
    region: str
    disk_gb: int
    ami: str
    spot: bool
    max_spot_price: str | None = None
    key_name: str | None = None


@dataclass
class RuntimeSpec:
    """Runtime image/logging settings loaded from ops/specs/runtime.yaml."""

    image: str
    fallback_build: bool
    log_group_prefix: str = "/manabot"


@dataclass
class Machine:
    """Normalized machine identity/state across provider implementations."""

    id: str
    public_ip: str | None
    status: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class CommandResult:
    """Result envelope for remote command execution."""

    command_id: str
    status: str
    stdout: str
    stderr: str


class Provider(Protocol):
    """Provisioning contract consumed by sandbox.py and job.py."""

    def create(
        self,
        spec: MachineSpec,
        tags: dict[str, str],
        *,
        user_data: str | None = None,
        iam_instance_profile: str | None = None,
    ) -> Machine: ...

    def start(self, machine: Machine) -> Machine: ...

    def wait_until_ready(self, machine: Machine, timeout: int = 300) -> Machine: ...

    def wait_for_ssm(self, machine: Machine, timeout: int = 300) -> None: ...

    def stop(self, machine: Machine) -> None: ...

    def terminate(self, machine: Machine) -> None: ...

    def list(self, tags: dict[str, str]) -> list[Machine]: ...

    def run_command(
        self,
        machine: Machine,
        command: str,
        timeout: int = 3600,
    ) -> CommandResult: ...

    def logs(
        self,
        group: str,
        stream_prefix: str,
        limit: int = 200,
    ) -> list[str]: ...


def load_machine_spec(name: str, base_dir: Path | None = None) -> MachineSpec:
    """Load machine settings from ops/specs/<name>.yaml."""

    return _load_spec(f"{name}.yaml", MachineSpec, base_dir=base_dir)


def load_runtime_spec(base_dir: Path | None = None) -> RuntimeSpec:
    """Load runtime settings from ops/specs/runtime.yaml."""

    return _load_spec("runtime.yaml", RuntimeSpec, base_dir=base_dir)


def _load_spec(
    filename: str,
    spec_type: type[MachineSpec] | type[RuntimeSpec],
    *,
    base_dir: Path | None = None,
) -> MachineSpec | RuntimeSpec:
    specs_dir = base_dir or Path(__file__).resolve().parent / "specs"
    path = specs_dir / filename
    config = _load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid spec file: {path}")
    return spec_type(**config)


def merge_str(*tag_maps: dict[str, Any]) -> dict[str, str]:
    """Merge dictionaries, coercing values to strings and dropping None."""

    merged: dict[str, str] = {}
    for tag_map in tag_maps:
        for key, value in tag_map.items():
            if value is None:
                continue
            merged[key] = str(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return loaded
