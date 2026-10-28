"""Durable request reservations for sequential hosted campaigns.

SQLite commits a maximum inclusive charge before dispatch. Missing replies and
process death leave that reservation pending and prohibit further requests.
A live session may abandon a reservation whose connection was never
established: it stays counted at its full bound but no longer blocks admission.
The caller must supply a justified provider charge bound, not a cost estimate.
Amounts use integer nanodollars, rounded upward at the billing boundary.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
import sqlite3
from uuid import uuid4

from pydantic import Field, model_validator

from manabot.llm.schema import Record

_SCALE = Decimal(1_000_000_000)


def _units(value: Decimal, rounding: str = ROUND_CEILING) -> int:
    return int((value * _SCALE).to_integral_value(rounding=rounding))


class BudgetConfig(Record):
    ceiling_usd: Decimal = Field(gt=0, le=10, allow_inf_nan=False)
    request_bound_usd: Decimal = Field(gt=0, allow_inf_nan=False)
    # Reserve all non-API spending up front, including publication/shutdown.
    overhead_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    charge_multiplier: Decimal = Field(ge=1, allow_inf_nan=False)
    bound_evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def inclusive_allowance(self) -> "BudgetConfig":
        if self.overhead_usd > self.ceiling_usd:
            raise ValueError("non-API allowance exceeds the inclusive budget ceiling")
        return self


class RequestCharge(Record):
    id: str
    owner: str
    request_sha256: str
    reserved_usd: Decimal
    charged_usd: Decimal | None
    abandoned: bool = False


class Spend(Record):
    overhead_usd: Decimal
    reported_inclusive_usd: Decimal
    pending_usd: Decimal
    accounted_usd: Decimal
    requests: tuple[RequestCharge, ...]

    @property
    def cost_by_owner(self) -> dict[str, Decimal]:
        """Attribute API spending, keeping uncertain requests at their full bound."""
        costs: dict[str, Decimal] = {}
        for request in self.requests:
            cost = (
                request.reserved_usd
                if request.charged_usd is None
                else request.charged_usd
            )
            costs[request.owner] = costs.get(request.owner, Decimal(0)) + cost
        return costs


class BudgetLedger:
    def __init__(self, path: Path, config: BudgetConfig) -> None:
        self.path = path
        self.config = config
        self._bound = _units(config.request_bound_usd)
        # Charges round up; rounding the ceiling up would authorize spending
        # beyond the caller's limit when it contains fractional nanodollars.
        self._available = _units(config.ceiling_usd, ROUND_FLOOR) - _units(
            config.overhead_usd
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (body TEXT NOT NULL)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, "
                "owner TEXT NOT NULL, digest TEXT NOT NULL, reserved INTEGER NOT NULL, "
                "charged INTEGER, abandoned INTEGER NOT NULL DEFAULT 0)"
            )
            row = db.execute("SELECT body FROM config").fetchone()
            body = config.model_dump_json()
            if row is None:
                db.execute("INSERT INTO config VALUES (?)", (body,))
            elif row[0] != body:
                raise ValueError("cannot change an existing campaign budget")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _admit_request(self, db: sqlite3.Connection) -> None:
        if db.execute("SELECT 1 FROM requests WHERE charged > reserved").fetchone():
            raise RuntimeError("request charge bound was violated; campaign stopped")
        if db.execute(
            "SELECT 1 FROM requests WHERE charged IS NULL AND NOT abandoned"
        ).fetchone():
            raise RuntimeError("unsettled request reservation; campaign stopped")
        used = db.execute(
            "SELECT COALESCE(SUM(COALESCE(charged, reserved)), 0) FROM requests"
        ).fetchone()[0]
        if used + self._bound > self._available:
            raise RuntimeError("inclusive campaign budget exhausted")

    def check_admission(self) -> None:
        """Check before starting a game; reserve rechecks atomically at dispatch."""
        with self._transaction() as db:
            self._admit_request(db)

    def reserve(self, owner: str, request_sha256: str) -> str:
        with self._transaction() as db:
            self._admit_request(db)
            identifier = str(uuid4())
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?, NULL, 0)",
                (identifier, owner, request_sha256, self._bound),
            )
        return identifier

    def abandon(self, identifier: str) -> None:
        """Keep an unsent request's full bound while admitting later requests."""
        with self._transaction() as db:
            updated = db.execute(
                "UPDATE requests SET abandoned = 1 WHERE id = ? AND charged IS NULL",
                (identifier,),
            ).rowcount
        if updated != 1:
            raise ValueError("only an unsettled reservation can be abandoned")

    def settle(self, identifier: str, reported_usd: float | None) -> None:
        if reported_usd is None:
            raise RuntimeError("missing provider cost; reservation retained")
        value = Decimal(str(reported_usd))
        if not value.is_finite() or value < 0:
            raise ValueError("invalid provider cost; reservation retained")
        charge = _units(value * self.config.charge_multiplier)
        with self._transaction() as db:
            row = db.execute(
                "SELECT reserved, charged FROM requests WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown request reservation")
            if row[1] is not None and row[1] != charge:
                raise ValueError("request already settled with a different charge")
            db.execute(
                "UPDATE requests SET charged = ? WHERE id = ?", (charge, identifier)
            )
        if charge > row[0]:
            # Preserve the actual charge even when it disproves the supplied bound.
            raise RuntimeError("provider charge exceeded the justified request bound")

    def snapshot(self) -> Spend:
        with self._transaction() as db:
            requests = tuple(
                RequestCharge(
                    id=row[0],
                    owner=row[1],
                    request_sha256=row[2],
                    reserved_usd=Decimal(row[3]) / _SCALE,
                    charged_usd=None if row[4] is None else Decimal(row[4]) / _SCALE,
                    abandoned=bool(row[5]),
                )
                for row in db.execute("SELECT * FROM requests ORDER BY rowid")
            )
        reported = sum((r.charged_usd or Decimal(0) for r in requests), Decimal(0))
        pending = sum(
            (r.reserved_usd for r in requests if r.charged_usd is None), Decimal(0)
        )
        return Spend(
            overhead_usd=self.config.overhead_usd,
            reported_inclusive_usd=reported,
            pending_usd=pending,
            accounted_usd=self.config.overhead_usd + reported + pending,
            requests=requests,
        )
