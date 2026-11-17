"""Shared record validation and content-addressed artifact references."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class ArtifactReference(TypedDict):
    """Immutable local bytes admitted by content digest."""

    path: str
    sha256: str
    bytes: int


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
