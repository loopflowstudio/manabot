"""Typed read-only choice projection shared by consumers of native offers.

Rust validates the wire shape and constructive support. These immutable records
name visible relationships; object addresses route joins and are never features.
Execution still requires the bound native offer set and canonical Commands.
"""

from dataclasses import dataclass
import json
from typing import Any, Literal, Mapping, cast

from ._managym import ChoiceSupport


@dataclass(frozen=True)
class ProgramReference:
    digest: str
    ordinal: int


@dataclass(frozen=True)
class OfferDetails:
    subject: "Subject | None"
    target: "Subject | None"
    outside_candidate: int | None
    program: ProgramReference | None
    requirement: int | None
    attack: bool | None
    mana: tuple[int, ...]


@dataclass(frozen=True)
class Candidate:
    id: int
    subject: "Subject"
    label: str


@dataclass(frozen=True)
class Selection:
    role: int
    context: "ChoiceContext"
    candidates: tuple[Candidate, ...]
    minimum: int
    maximum: int


@dataclass(frozen=True)
class InteractionOffer:
    id: int
    actor: int
    verb: str
    source: "Subject | None"
    details: OfferDetails
    choices: tuple[Selection, ...]
    label: str


@dataclass(frozen=True)
class OfferProjection:
    """Native-validated independent choices, with their retained support owner."""

    schema_version: int
    factorization_version: int
    revision: int
    actor: int
    kind: str
    offers: tuple[InteractionOffer, ...]
    support: ChoiceSupport

    @classmethod
    def from_json(cls, text: str) -> "OfferProjection":
        support = ChoiceSupport(text)
        # The native parser has validated the JSON boundary. Keep untyped JSON
        # local; consumers receive named records rather than interpreting maps.
        payload: dict[str, Any] = json.loads(text)
        offers: list[InteractionOffer] = []
        for row in payload["offers"]:
            details = row["details"]
            program = details.get("program")
            choices = tuple(
                Selection(
                    choice["role"],
                    ChoiceContext.parse(choice["context"]),
                    tuple(
                        Candidate(
                            c["id"], Subject.parse(c["value"]["subject"]), c["label"]
                        )
                        for c in choice["candidates"]["initial"]
                    ),
                    choice["min"],
                    choice["max"],
                )
                for choice in row["choices"]
            )
            offers.append(
                InteractionOffer(
                    row["id"],
                    row["actor"],
                    row["verb"],
                    None if row["source"] is None else Subject.parse(row["source"]),
                    OfferDetails(
                        None
                        if details.get("subject") is None
                        else Subject.parse(details["subject"]),
                        None
                        if details.get("target") is None
                        else Subject.parse(details["target"]),
                        details.get("outside_candidate"),
                        None
                        if program is None
                        else ProgramReference(program["digest"], program["ordinal"]),
                        details.get("requirement"),
                        details.get("attack"),
                        tuple(details.get("mana") or ()),
                    ),
                    choices,
                    row["label"],
                )
            )
        return cls(
            payload["schema_version"],
            payload["factorization_version"],
            payload["revision"],
            payload["actor"],
            payload["kind"],
            tuple(offers),
            support,
        )


@dataclass(frozen=True)
class Subject:
    kind: Literal["object", "player"]
    entity: int
    incarnation: int | None

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "Subject":
        if value["kind"] == "player":
            return cls("player", int(value["id"]), None)
        return cls(
            "object", int(value["id"]["entity"]), int(value["id"]["incarnation"])
        )


@dataclass(frozen=True)
class ChoiceContext:
    kind: Literal["selection", "assignment", "target", "payment"]
    subject: Subject | None
    mana: tuple[int, ...]
    requirement: int | None
    program: ProgramReference | None
    tap_generic_units: int | None

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ChoiceContext":
        subject = value.get("subject")
        program = value.get("program")
        return cls(
            cast(
                Literal["selection", "assignment", "target", "payment"], value["kind"]
            ),
            None if subject is None else Subject.parse(subject),
            tuple(value.get("mana", ())),
            value.get("requirement"),
            None
            if program is None
            else ProgramReference(program["digest"], program["ordinal"]),
            value.get("tap_generic_units"),
        )
