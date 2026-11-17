"""Read-only compute identities and allocation receipts from released evidence.

V2 identities contain version, device and dtype; v1 also hashed scheduling
fields. Their recorded spellings and identities remain unchanged. No class in
this module launches, schedules or supervises evaluation.
"""

from __future__ import annotations

from typing import Literal

from manabot.infra.identity import StrictModel, canonical_sha256


class EvalCompute(StrictModel):
    """Stable serving contract used in player and benchmark identities."""

    version: Literal["shared-eval-v2"] = "shared-eval-v2"
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32"] = "float32"

    @property
    def identity(self) -> str:
        return "shared-eval-" + canonical_sha256(self.model_dump())


class RecordedExecution(EvalCompute):
    """Retained v2 allocation receipt; never an executable configuration."""

    max_active_games: int
    max_batch_rows: int
    max_delay_seconds: float
    resident_model_bytes: int
    request_seconds: float
    cache_catalog: bool
    resident: bool
    bucketed_shapes: bool = False
    live_tokens: bool = False
    deadline_unix: float
    instrumentation: bool = True
    diagnostic_batches: int = 0

    @property
    def compute(self) -> EvalCompute:
        return EvalCompute.model_validate(
            self.model_dump(include=set(EvalCompute.model_fields))
        )


# ---------------------------------------------------------------------------
# Frozen v1 records (read only)


class EvalComputeV1(StrictModel):
    """`shared-eval-v1`: identity hashed the allocation settings as well."""

    version: Literal["shared-eval-v1"]
    device: Literal["cpu", "cuda"]
    dtype: Literal["float32"]
    max_active_games: int
    max_batch_rows: int
    max_delay_seconds: float
    resident_model_bytes: int
    request_seconds: float
    cache_catalog: bool
    resident: bool

    @property
    def identity(self) -> str:
        return "shared-eval-" + canonical_sha256(self.model_dump())


class EvalExecutionV1(EvalComputeV1):
    """A retained v1 allocation; never executed by current code."""

    deadline_unix: float
    instrumentation: bool = True
    diagnostic_batches: int = 0

    @property
    def compute(self) -> EvalComputeV1:
        return EvalComputeV1.model_validate(
            self.model_dump(include=set(EvalComputeV1.model_fields))
        )
