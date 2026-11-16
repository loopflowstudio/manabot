"""Legacy source admission and deprecated portable replay names."""

from collections.abc import Mapping
from typing import Any
import warnings

from manabot.env.failures import runtime_identities
from managym import replay


def require_replay_runtime(
    identities: Mapping[str, object],
    *,
    allow_changed_runtime: bool,
    require_source: bool = False,
) -> str:
    native = replay.require_replay_runtime(
        identities,
        allow_changed_runtime=allow_changed_runtime,
        require_source=require_source
        and "engine_recording_source_sha256" in identities,
    )
    if (
        require_source
        and "engine_recording_source_sha256" not in identities
        and not allow_changed_runtime
    ):
        if identities["source_sha256"] != runtime_identities()["source_sha256"]:
            raise ValueError(
                "source bytes differ; explicit --allow-changed-runtime required"
            )
    return native


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    warnings.warn(
        f"manabot.env.replay.{name} moved to managym.replay; removal in 0.4.0",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(replay, name)
