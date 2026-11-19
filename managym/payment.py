"""Typed, read-only payment explanations from the shared native payer.

The semantic address authorizes the query. Production is exact only after
announcement choices are known; a source plan can still leave spending open.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

import managym
from managym.experience import PooledManaView, SubjectRef


class PreviewPool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amounts: tuple[int, int, int, int, int, int]
    classes: list[PooledManaView]


class PaymentSourcePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: SubjectRef
    label: str
    output: PreviewPool


class PaymentPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["exact", "needs_choice", "not_applicable", "unavailable"]
    explanation: str
    cost: str | None
    # What objects have already paid toward the cost, in the order paid.
    contributions: list[str]
    sources: list[PaymentSourcePreview]
    pool_after_production: PreviewPool | None
    pool_after_spending: PreviewPool | None


def preview_payment(
    env: managym.Env,
    *,
    viewer: int,
    revision: int,
    fingerprint: str,
    action_index: int,
) -> PaymentPreview:
    """Reject stale/non-actor requests without advancing the native game."""
    return PaymentPreview.model_validate_json(
        env.payment_preview_json(viewer, revision, fingerprint, action_index)
    )
