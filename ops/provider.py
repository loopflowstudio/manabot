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

    specs_dir = base_dir or Path(__file__).resolve().parent / "specs"
    path = specs_dir / f"{name}.yaml"
    config = _load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid machine spec file: {path}")
    return MachineSpec(**config)


def load_runtime_spec(base_dir: Path | None = None) -> RuntimeSpec:
    """Load runtime settings from ops/specs/runtime.yaml."""

    specs_dir = base_dir or Path(__file__).resolve().parent / "specs"
    path = specs_dir / "runtime.yaml"
    config = _load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid runtime spec file: {path}")
    return RuntimeSpec(**config)


def merge_tags(*tag_maps: dict[str, Any]) -> dict[str, str]:
    """Merge tag dictionaries while coercing values to strings."""

    merged: dict[str, str] = {}
    for tag_map in tag_maps:
        for key, value in tag_map.items():
            if value is None:
                continue
            merged[key] = str(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML root must be mapping: {path}")
        return loaded
    except ModuleNotFoundError:
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small YAML subset parser for flat key/value spec files."""

    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", maxsplit=1)
        parsed[key.strip()] = _coerce_yaml_scalar(value.strip())
    return parsed


def _coerce_yaml_scalar(value: str) -> Any:
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False

    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]

    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)

    return value
