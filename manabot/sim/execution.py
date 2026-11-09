"""Explicit, immutable admission envelope for allocation-scoped evaluation.

`EvalCompute` is the identity of served inference: how a served policy's
numbers are computed. `EvalAllocation` adds how hard one allocation drives the
service (active games, batching, residency, request limits), and
`EvalExecution` adds the allocation's deadline and measurement switches. Only
`EvalCompute` enters player, benchmark and arena identities; the rest are
attempt facts retained in receipts.

Batch composition varies with arrival order under every allocation, and
sampling is per row, so allocation settings change throughput, not the
contract: Commands are reproducible from retained traces, not by recomputing
the policy live across batch sizes. `shared-eval-v1` hashed the allocation
settings too; its frozen records stay readable through `EvalComputeV1`.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import Field

from manabot.infra.identity import StrictModel, canonical_sha256


class EvalCompute(StrictModel):
    """Stable serving contract used in player and benchmark identities."""

    version: Literal["shared-eval-v2"] = "shared-eval-v2"
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32"] = "float32"

    @property
    def identity(self) -> str:
        return "shared-eval-" + canonical_sha256(self.model_dump())


class EvalAllocation(EvalCompute):
    """The compute contract plus one allocation's scheduling and cache limits."""

    max_active_games: int = Field(default=4, ge=1, le=256)
    max_batch_rows: int = Field(default=32, ge=1, le=256)
    max_delay_seconds: float = Field(default=0.005, ge=0, le=1, allow_inf_nan=False)
    resident_model_bytes: int = Field(default=2**30, gt=0)
    request_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    cache_catalog: bool = True
    resident: bool = True

    @property
    def compute(self) -> EvalCompute:
        return EvalCompute.model_validate(
            self.model_dump(include=set(EvalCompute.model_fields))
        )

    @property
    def identity(self) -> str:
        return self.compute.identity


class EvalExecution(EvalAllocation):
    """One bounded allocation; complete settings are retained in attempt receipts."""

    deadline_unix: float = Field(allow_inf_nan=False)
    instrumentation: bool = True
    diagnostic_batches: int = Field(default=0, ge=0, le=100)

    def remaining(self) -> float:
        return max(0.0, self.deadline_unix - time.time())


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
