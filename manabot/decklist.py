"""Source-preserving decklist imports shared by Academy and manabot.

Text parsing and name resolution live here; the native content pool owns card
eligibility and native admission owns construction policy. An admitted setup
retains both the submitted source and the exact content identity for replay.
Native imports stay lazy so configuration and historical readers remain portable.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, computed_field

if TYPE_CHECKING:
    from manabot.infra.config import MatchConfig

Section = Literal["main", "sideboard"]
MAX_SOURCE_BYTES = 65_536


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class CompiledIdentity(_Record):
    pack_key: str
    ir_hash: str
    source_hash: str


class PoolDefinition(_Record):
    card_def_id: int | None = None
    semantic_key: str | None = None
    registry_name: str
    eligibility: Literal["eligible", "unsupported"]
    reason: str | None = None


class ContentPool(_Record):
    pack_key: str
    world_version: str
    content_digest: str
    compiled_semantics: CompiledIdentity
    definitions: tuple[PoolDefinition, ...]

    @classmethod
    def load(cls, pack_key: str) -> ContentPool:
        import managym

        return cls.model_validate(managym.content_pool(pack_key))


class DeckIssue(_Record):
    line: int | None = None
    section: Section | None = None
    code: str
    message: str
    card_name: str | None = None


class DeckEntry(_Record):
    line: int
    section: Section
    count: int
    name: str
    card_def_id: int | None = None
    semantic_key: str | None = None


class DeckDocument(_Record):
    dialect: Literal["counted-lines-v1"] = "counted-lines-v1"
    source: str
    filename: str = "deck.txt"
    entries: tuple[DeckEntry, ...] = ()
    issues: tuple[DeckIssue, ...] = ()

    @computed_field
    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


class AdmittedDeck(_Record):
    document: DeckDocument
    main: dict[str, int]
    sideboard: dict[str, int]


class ImportedMatch(_Record):
    """Private setup record; never attach submitted lists to viewer frames."""

    pool: ContentPool
    ruleset: str
    hero: AdmittedDeck
    villain: AdmittedDeck

    @computed_field
    @property
    def setup_id(self) -> str:
        # Presentation (filenames, line order, comments) cannot alter game identity.
        value = {
            "pool": self.pool.model_dump(exclude={"definitions"}),
            "ruleset": self.ruleset,
            "decks": [
                {"main": deck.main, "sideboard": deck.sideboard}
                for deck in (self.hero, self.villain)
            ],
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def match_config(self) -> "MatchConfig":
        from manabot.infra.config import MatchConfig

        return MatchConfig(
            content_pack=self.pool.pack_key,
            imported_decks=self,
            hero_deck=self.hero.main,
            villain_deck=self.villain.main,
            hero_sideboard=self.hero.sideboard,
            villain_sideboard=self.villain.sideboard,
        )

    def validate_current(self) -> None:
        current = ContentPool.load(self.pool.pack_key)
        if current != self.pool:
            raise ValueError(
                "Imported deck content identity differs from the installed pool"
            )
        for deck in (self.hero, self.villain):
            admitted = admit_document(
                parse_decklist(deck.document.source, deck.document.filename),
                current,
                self.ruleset,
            )
            if isinstance(admitted, DeckDocument) or admitted != deck:
                raise ValueError(
                    "Recorded imported deck differs from its admitted source"
                )


class DeckImportError(ValueError):
    def __init__(self, documents: dict[str, DeckDocument]) -> None:
        self.documents = documents
        super().__init__(
            "; ".join(
                f"{seat} {doc.filename}:{issue.line or '-'}: {issue.message}"
                for seat, doc in documents.items()
                for issue in doc.issues
            )
        )


def _name_key(name: str) -> str:
    return " ".join(unicodedata.normalize("NFC", name).casefold().split())


def parse_decklist(source: str, filename: str = "deck.txt") -> DeckDocument:
    """Parse counts/names, Main/Deck/Sideboard headings and full-line # comments."""
    entries: list[DeckEntry] = []
    issues: list[DeckIssue] = []
    section: Section = "main"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return DeckDocument(
            source=source,
            filename=filename,
            issues=(
                DeckIssue(
                    code="size", message="Decklist exceeds the 64 KiB import limit."
                ),
            ),
        )
    for number, raw in enumerate(source.splitlines(), 1):
        line = (
            raw.strip().removeprefix("\ufeff").strip() if number == 1 else raw.strip()
        )
        if not line or line.startswith("#"):
            continue
        heading = line.casefold().removesuffix(":")
        if heading in {"main", "deck", "sideboard"}:
            section = "sideboard" if heading == "sideboard" else "main"
            continue
        match = re.fullmatch(r"([+-]?[0-9]+)\s+(.+)", line)
        if match is None:
            issues.append(
                DeckIssue(
                    line=number,
                    section=section,
                    code="syntax",
                    message="Expected COUNT Card Name, or a Main/Sideboard heading.",
                )
            )
            continue
        # Bound conversion before int() so arbitrary numeric input stays actionable.
        digits, name = match.groups()
        magnitude = digits.lstrip("+-").lstrip("0") or "0"
        if digits.startswith("-") or len(magnitude) > 9 or int(magnitude) == 0:
            issues.append(
                DeckIssue(
                    line=number,
                    section=section,
                    code="count",
                    card_name=name,
                    message="Quantity must be a positive supported integer.",
                )
            )
            continue
        entries.append(
            DeckEntry(line=number, section=section, count=int(magnitude), name=name)
        )
    return DeckDocument(
        source=source, filename=filename, entries=tuple(entries), issues=tuple(issues)
    )


def admit_document(
    document: DeckDocument, pool: ContentPool, ruleset: str
) -> AdmittedDeck | DeckDocument:
    import managym

    names: dict[str, list[PoolDefinition]] = defaultdict(list)
    reverse_faces: dict[str, str] = {}
    for definition in pool.definitions:
        names[_name_key(definition.registry_name)].append(definition)
        if " // " in definition.registry_name:
            front, reverse = definition.registry_name.split(" // ", 1)
            names[_name_key(front)].append(definition)
            reverse_faces[_name_key(reverse)] = definition.registry_name
    issues = list(document.issues)
    resolved: list[DeckEntry] = []
    sections: dict[Section, dict[str, int]] = {"main": {}, "sideboard": {}}
    for entry in document.entries:
        candidates = names.get(_name_key(entry.name), [])
        if not candidates and _name_key(entry.name) in reverse_faces:
            issues.append(
                DeckIssue(
                    line=entry.line,
                    section=entry.section,
                    code="face",
                    card_name=entry.name,
                    message=f"Reverse faces cannot be deck entries; use {reverse_faces[_name_key(entry.name)]}.",
                )
            )
            continue
        if len(candidates) != 1:
            code = "ambiguous" if candidates else "unknown"
            issues.append(
                DeckIssue(
                    line=entry.line,
                    section=entry.section,
                    code=code,
                    card_name=entry.name,
                    message=f"{code.title()} card name: {entry.name}.",
                )
            )
            continue
        definition = candidates[0]
        resolved.append(
            entry.model_copy(
                update={
                    "name": definition.registry_name,
                    "card_def_id": definition.card_def_id,
                    "semantic_key": definition.semantic_key,
                }
            )
        )
        cards = sections[entry.section]
        cards[definition.registry_name] = (
            cards.get(definition.registry_name, 0) + entry.count
        )
    # Construction and eligibility are checked by the same native entry point
    # used again at Env.reset; Python does not carry a second legality policy.
    native_issues = managym.admit_deck(
        pool.pack_key, sections["main"], sections["sideboard"], ruleset
    )
    for raw_issue in native_issues:
        issue = DeckIssue.model_validate(raw_issue)
        lines = [
            entry.line
            for entry in resolved
            if entry.section == issue.section and entry.name == issue.card_name
        ]
        issues.extend(
            issue.model_copy(update={"line": line}) for line in lines or [None]
        )
    updated = document.model_copy(
        update={"entries": tuple(resolved), "issues": tuple(issues)}
    )
    if issues:
        return updated
    return AdmittedDeck(
        document=updated, main=sections["main"], sideboard=sections["sideboard"]
    )


def import_match(
    hero: DeckDocument,
    villain: DeckDocument,
    pack_key: str,
    ruleset: str = "development-v1",
) -> ImportedMatch:
    pool = ContentPool.load(pack_key)
    first = admit_document(hero, pool, ruleset)
    second = admit_document(villain, pool, ruleset)
    errors = {
        seat: value
        for seat, value in (("hero", first), ("villain", second))
        if isinstance(value, DeckDocument)
    }
    if errors:
        raise DeckImportError(errors)
    assert isinstance(first, AdmittedDeck) and isinstance(second, AdmittedDeck)
    return ImportedMatch(pool=pool, ruleset=ruleset, hero=first, villain=second)


def _read_file(path: Path) -> DeckDocument:
    with path.open("rb") as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError(f"{path.name}: decklist exceeds the 64 KiB import limit")
    return parse_decklist(data.decode("utf-8"), path.name)


def import_files(hero: Path, villain: Path, pack_key: str) -> ImportedMatch:
    return import_match(_read_file(hero), _read_file(villain), pack_key)
