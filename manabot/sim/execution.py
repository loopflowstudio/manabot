"""Explicit, immutable admission envelope for allocation-scoped evaluation."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import Field

from manabot.infra.identity import StrictModel, canonical_sha256


class EvalCompute(StrictModel):
    """Stable serving contract used in player and benchmark identities.

    Scheduling and cache limits remain explicit because they can change batch
    composition or numerical results. Allocation dates and measurement switches
    belong to attempts, so equivalent windows can share completed deal evidence.
    """

    version: Literal["shared-eval-v1"] = "shared-eval-v1"
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32"] = "float32"
    max_active_games: int = Field(default=4, ge=1, le=256)
    max_batch_rows: int = Field(default=32, ge=1, le=256)
    max_delay_seconds: float = Field(default=0.005, ge=0, le=1, allow_inf_nan=False)
    resident_model_bytes: int = Field(default=2**30, gt=0)
    request_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    cache_catalog: bool = True
    resident: bool = True

    @property
    def identity(self) -> str:
        return "shared-eval-" + canonical_sha256(self.model_dump())


class EvalExecution(EvalCompute):
    """One bounded allocation; complete settings are retained in attempt receipts."""

    deadline_unix: float = Field(allow_inf_nan=False)
    instrumentation: bool = True
    diagnostic_batches: int = Field(default=0, ge=0, le=100)

    @property
    def compute(self) -> EvalCompute:
        return EvalCompute.model_validate(
            self.model_dump(include=set(EvalCompute.model_fields))
        )

    @property
    def identity(self) -> str:
        return self.compute.identity

    def remaining(self) -> float:
        return max(0.0, self.deadline_unix - time.time())
