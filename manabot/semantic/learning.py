"""Viewer-safe projection of card rules for learning.

The rules engine remains authoritative and is also the only source of card
programs: ``managym.learning_catalog`` walks each pack definition (the same
``CardDefinition`` the engine runs) into typed tokens. This module binds that
catalog to one ContentPack manifest and maps only already-viewer-safe
Observation objects into it. Rows are ``CardDefId`` values.

Principal types: ``LearningSchema`` (token kinds and object roles),
``BoundSemanticPack`` (a catalog bound to one manifest) and the projection
and batch records it produces.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = ROOT / "content/semantic/current/learning_schema.json"
SCHEMA_VERSION = 9

# Explicit current packs shared by live-manifest and policy binding.
CURRENT_PACK_KEYS = frozenset(
    {
        "tla-pool-v1",
        "tla-baseline-v1",
        "ur-lessons-vs-gw-allies",
        "elemental-combat-v1",
        "elemental-sacrifice-v1",
        "elemental-temur-v1",
        "elemental-world-v2",
        "elemental-world-v1",
        "mtg122-complexity-v1",
    }
)

# Token kinds as numbered by the engine (``semantic::tokens::TokenKind``).
TOKEN_KINDS = {
    "padding": 0,
    "structure": 1,
    "field": 2,
    "variant": 3,
    "integer": 4,
    "boolean": 5,
    "text": 6,
    "definition_ref": 7,
    "null": 8,
    "program_kind": 9,
}

# Program spans as numbered by the engine (``semantic::tokens::ProgramKind``).
PROGRAM_KINDS = {
    "triggered": 1,
    "mana": 2,
    "triggered_mana": 3,
    "activated": 4,
    "spell": 5,
}


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 without ASCII escaping."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def token_value(name: str) -> int:
    """The engine's stable token value for a Rust name or printed category."""
    import managym  # Native only on this path.

    return int(managym.learning_token_value(name))


class SemanticProjectionError(ValueError):
    """Base error for invalid semantic inputs or incompatible artifacts."""


class UnknownSchemaError(SemanticProjectionError):
    """A schema version is not supported."""


class ContentPackBindingError(SemanticProjectionError):
    """The catalog cannot bind exactly to a ContentPack manifest."""


class UnadmittedDefinitionError(SemanticProjectionError):
    """A visible runtime definition is absent from the catalog."""


class ArtifactCompatibilityError(SemanticProjectionError):
    """Semantic dataset/checkpoint provenance is absent or incompatible."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticProjectionError(f"{context}: expected object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SemanticProjectionError(f"{context}: expected array")
    return value


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticProjectionError(f"{context}: expected integer")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticProjectionError(f"{context}: expected non-empty string")
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SemanticProjectionError(f"{path}: invalid JSON: {error}") from error
    return dict(_mapping(value, str(path)))


def _validate_id_map(
    value: Any, context: str, *, allow_zero: bool = False
) -> dict[str, int]:
    out = {
        _string(key, f"{context}.key"): _integer(item, f"{context}.{key}")
        for key, item in _mapping(value, context).items()
    }
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in out.values()):
        raise SemanticProjectionError(f"{context}: IDs must be >= {minimum}")
    if len(set(out.values())) != len(out):
        raise SemanticProjectionError(f"{context}: duplicate numeric IDs")
    return out


@dataclass(frozen=True)
class LearningSchema:
    raw: dict[str, Any]
    schema_hash: str
    token_kinds: dict[str, int]
    program_kinds: dict[str, int]
    object_roles: dict[str, int]
    identity_modes: dict[str, int]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCHEMA_PATH) -> "LearningSchema":
        raw = _load_json(Path(path))
        version = _integer(raw.get("schema_version"), "learning_schema.schema_version")
        if version != SCHEMA_VERSION:
            raise UnknownSchemaError(
                f"learning_schema.schema_version: expected {SCHEMA_VERSION}, got {version}"
            )
        layout = _integer(
            raw.get("array_layout_version"), "learning_schema.array_layout_version"
        )
        if layout != 1:
            raise UnknownSchemaError(
                f"learning_schema.array_layout_version: expected 1, got {layout}"
            )
        token_kinds = _validate_id_map(
            raw.get("token_kinds"), "learning_schema.token_kinds", allow_zero=True
        )
        if token_kinds != TOKEN_KINDS:
            raise SemanticProjectionError(
                "learning_schema.token_kinds disagrees with the engine's TokenKind"
            )
        program_kinds = _validate_id_map(
            raw.get("program_kinds"), "learning_schema.program_kinds"
        )
        if program_kinds != PROGRAM_KINDS:
            raise SemanticProjectionError(
                "learning_schema.program_kinds disagrees with the engine's ProgramKind"
            )
        expected_dtypes = {
            "boolean_mask": "bool",
            "index": "int32",
            "token_kind": "uint16",
            "token_value": "int32",
        }
        if raw.get("dtypes") != expected_dtypes:
            raise SemanticProjectionError(
                f"learning_schema.dtypes: expected {expected_dtypes!r}"
            )
        return cls(
            raw=raw,
            schema_hash=_hash_json(raw),
            token_kinds=token_kinds,
            program_kinds=program_kinds,
            object_roles=_validate_id_map(
                raw.get("object_roles"), "learning_schema.object_roles"
            ),
            identity_modes=_validate_id_map(
                raw.get("identity_modes"), "learning_schema.identity_modes"
            ),
        )


@dataclass(frozen=True)
class SemanticCatalog:
    token_kind: np.ndarray
    token_value: np.ndarray
    definition_offsets: np.ndarray
    program_offsets: np.ndarray
    definition_program_offsets: np.ndarray
    definition_program_rows: np.ndarray
    definition_ref_source_tokens: np.ndarray
    definition_ref_target_rows: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            int(value.nbytes)
            for value in (
                self.token_kind,
                self.token_value,
                self.definition_offsets,
                self.program_offsets,
                self.definition_program_offsets,
                self.definition_program_rows,
                self.definition_ref_source_tokens,
                self.definition_ref_target_rows,
            )
        )

    @classmethod
    def from_native(cls, raw: Mapping[str, Any]) -> "SemanticCatalog":
        """Read the arrays ``managym.learning_catalog`` returns."""

        def array(name: str, dtype: type) -> np.ndarray:
            value = np.asarray(_sequence(raw.get(name), name), dtype=dtype)
            value.setflags(write=False)
            return value

        return cls(
            token_kind=array("token_kind", np.uint16),
            token_value=array("token_value", np.int32),
            definition_offsets=array("definition_offsets", np.int32),
            program_offsets=array("program_offsets", np.int32),
            definition_program_offsets=array("definition_program_offsets", np.int32),
            definition_program_rows=array("definition_program_rows", np.int32),
            definition_ref_source_tokens=array(
                "definition_ref_source_tokens", np.int32
            ),
            definition_ref_target_rows=array("definition_ref_target_rows", np.int32),
        )

    @property
    def definitions(self) -> int:
        return len(self.definition_offsets) - 1

    @property
    def programs(self) -> int:
        return len(self.program_offsets) - 1


@dataclass(frozen=True)
class SemanticArtifactHeader:
    learning_schema_version: int
    learning_schema_hash: str
    content_pack_schema_version: int
    content_pack_hash: str
    semantic_pack_hash: str
    identity_mode: str
    binding_hash: str | None
    array_layout_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "array_layout_version": self.array_layout_version,
            "binding_hash": self.binding_hash,
            "content_pack_hash": self.content_pack_hash,
            "content_pack_schema_version": self.content_pack_schema_version,
            "identity_mode": self.identity_mode,
            "learning_schema_hash": self.learning_schema_hash,
            "learning_schema_version": self.learning_schema_version,
            "semantic_pack_hash": self.semantic_pack_hash,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "SemanticArtifactHeader":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise ArtifactCompatibilityError(
                f"invalid semantic artifact header: {error}"
            ) from error
        try:
            return cls(**dict(_mapping(raw, "semantic artifact header")))
        except TypeError as error:
            raise ArtifactCompatibilityError(
                f"invalid semantic artifact header fields: {error}"
            ) from error


@dataclass(frozen=True)
class SemanticObjectProjection:
    semantic_pack_hash: str
    identity_mode: str
    object_definition_rows: np.ndarray
    object_roles: np.ndarray
    object_slots: np.ndarray
    opaque_identity_ids: np.ndarray
    opaque_identity_valid: np.ndarray


@dataclass(frozen=True)
class SemanticObjectBatch:
    semantic_pack_hash: str
    identity_mode: str
    sample_offsets: np.ndarray
    object_definition_rows: np.ndarray
    object_roles: np.ndarray
    object_slots: np.ndarray
    opaque_identity_ids: np.ndarray
    opaque_identity_valid: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            int(value.nbytes)
            for value in (
                self.sample_offsets,
                self.object_definition_rows,
                self.object_roles,
                self.object_slots,
                self.opaque_identity_ids,
                self.opaque_identity_valid,
            )
        )

    def pad(self) -> dict[str, np.ndarray]:
        batch_size = max(0, len(self.sample_offsets) - 1)
        lengths = np.diff(self.sample_offsets)
        width = int(lengths.max()) if lengths.size else 0
        out = {
            "object_definition_rows": np.full((batch_size, width), -1, dtype=np.int32),
            "object_roles": np.zeros((batch_size, width), dtype=np.uint8),
            "object_slots": np.full((batch_size, width), -1, dtype=np.int32),
            "opaque_identity_ids": np.full((batch_size, width), -1, dtype=np.int32),
            "opaque_identity_valid": np.zeros((batch_size, width), dtype=np.bool_),
            "object_mask": np.zeros((batch_size, width), dtype=np.bool_),
        }
        for row in range(batch_size):
            start = int(self.sample_offsets[row])
            end = int(self.sample_offsets[row + 1])
            size = end - start
            if size == 0:
                continue
            out["object_definition_rows"][row, :size] = self.object_definition_rows[
                start:end
            ]
            out["object_roles"][row, :size] = self.object_roles[start:end]
            out["object_slots"][row, :size] = self.object_slots[start:end]
            out["opaque_identity_ids"][row, :size] = self.opaque_identity_ids[start:end]
            out["opaque_identity_valid"][row, :size] = self.opaque_identity_valid[
                start:end
            ]
            out["object_mask"][row, :size] = True
        return out


@dataclass(frozen=True)
class BoundSemanticPack:
    schema: LearningSchema
    pack_key: str | None
    content_pack_schema_version: int
    content_pack_hash: str
    binding_hash: str
    semantic_pack_hash: str
    definition_row_by_card_def_id: dict[int, int]
    catalog: SemanticCatalog
    # Diagnostics only: catalog rows' printed names, never tokens.
    registry_names: tuple[str, ...]

    @classmethod
    def bind(
        cls,
        manifest: Mapping[str, Any],
        *,
        schema_path: str | Path | None = None,
    ) -> "BoundSemanticPack":
        """Bind the engine's catalog for the manifest's pack (or the default pack).

        The catalog's content digest must equal the manifest's, so the tokens
        describe exactly the definitions the match runs. Manifest IDs join
        catalog rows by registry name, so rows never depend on ID allocation.
        """
        import managym  # Native only on this path; historical readers stay pure.

        manifest = _mapping(manifest, "content_pack_manifest")
        pack_key = manifest.get("pack_key")
        if pack_key is not None:
            pack_key = _string(pack_key, "content_pack_manifest.pack_key")
        schema = LearningSchema.load(schema_path or DEFAULT_SCHEMA_PATH)
        raw = _mapping(managym.learning_catalog(pack_key), "learning_catalog")
        pack_schema = _integer(
            manifest.get("schema_version"), "content_pack_manifest.schema_version"
        )
        pack_hash = _string(
            manifest.get("content_digest"), "content_pack_manifest.content_digest"
        )
        if raw.get("content_digest") != pack_hash:
            raise ContentPackBindingError("learning catalog differs from the live pack")
        catalog = SemanticCatalog.from_native(raw)

        row_by_name = {
            _string(name, "learning_catalog.registry_names[]"): row
            for row, name in enumerate(
                _sequence(raw.get("registry_names"), "learning_catalog.registry_names")
            )
        }
        rows_by_id: dict[int, int] = {}
        for index, raw_definition in enumerate(
            _sequence(manifest.get("definitions"), "content_pack_manifest.definitions")
        ):
            definition = _mapping(
                raw_definition, f"content_pack_manifest.definitions[{index}]"
            )
            card_def_id = _integer(
                definition.get("card_def_id"),
                f"content_pack_manifest.definitions[{index}].card_def_id",
            )
            registry_name = _string(
                definition.get("registry_name"),
                f"content_pack_manifest.definitions[{index}].registry_name",
            )
            if card_def_id in rows_by_id:
                raise ContentPackBindingError(f"duplicate CardDefId {card_def_id}")
            try:
                row = row_by_name[registry_name]
            except KeyError as error:
                raise ContentPackBindingError(
                    f"manifest definition {registry_name!r} does not bind"
                ) from error
            if row in rows_by_id.values():
                raise ContentPackBindingError(
                    f"duplicate ContentPack registry name {registry_name!r}"
                )
            rows_by_id[card_def_id] = row
        if len(rows_by_id) != catalog.definitions:
            raise ContentPackBindingError("catalog definitions do not all bind")

        binding_hash = _hash_json(
            [
                {"card_def_id": card_def_id, "definition_row": row}
                for card_def_id, row in sorted(rows_by_id.items())
            ]
        )
        semantic_pack_hash = _hash_json(
            {
                "content_pack_hash": pack_hash,
                "content_pack_schema_version": pack_schema,
                "learning_schema_hash": schema.schema_hash,
            }
        )
        return cls(
            schema=schema,
            pack_key=pack_key,
            content_pack_schema_version=pack_schema,
            content_pack_hash=pack_hash,
            binding_hash=binding_hash,
            semantic_pack_hash=semantic_pack_hash,
            definition_row_by_card_def_id=rows_by_id,
            catalog=catalog,
            registry_names=tuple(row_by_name),
        )

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        schema_path: str | Path | None = None,
    ) -> "BoundSemanticPack":
        try:
            manifest = env.content_pack_manifest()
        except AttributeError as error:
            raise ContentPackBindingError(
                "environment does not expose content_pack_manifest()"
            ) from error
        return cls.bind(manifest, schema_path=schema_path)

    @property
    def definition_count(self) -> int:
        return self.catalog.definitions

    def definition_row(self, card_def_id: int) -> int:
        """Return the catalog row for one runtime definition."""

        try:
            return self.definition_row_by_card_def_id[card_def_id]
        except KeyError as error:
            raise UnadmittedDefinitionError(
                f"visible CardDefId {card_def_id} is absent from the learning catalog"
            ) from error

    def program_rows(self, definition_row: int) -> tuple[int, ...]:
        """Return the program rows owned by one definition."""

        if not 0 <= definition_row < self.catalog.definitions:
            raise SemanticProjectionError(
                f"definition row {definition_row} is outside the semantic catalog"
            )
        start = int(self.catalog.definition_program_offsets[definition_row])
        stop = int(self.catalog.definition_program_offsets[definition_row + 1])
        return tuple(
            int(row) for row in self.catalog.definition_program_rows[start:stop]
        )

    def project_observation(
        self,
        observation: Any,
        *,
        identity_mode: str = "semantic_only",
    ) -> SemanticObjectProjection:
        if identity_mode not in self.schema.identity_modes:
            raise SemanticProjectionError(f"unknown identity mode {identity_mode!r}")
        definition_rows: list[int] = []
        roles: list[int] = []
        slots: list[int] = []
        identity_ids: list[int] = []

        for role_name, cards in (
            ("agent_card", observation.agent_cards),
            ("agent_sideboard", observation.agent_sideboard),
            ("opponent_card", observation.opponent_cards),
        ):
            role = self.schema.object_roles[role_name]
            for slot, card in enumerate(cards):
                card_def_id = int(card.registry_key)
                definition_rows.append(self.definition_row(card_def_id))
                roles.append(role)
                slots.append(slot)
                identity_ids.append(card_def_id)

        stack_role = self.schema.object_roles["stack_ability"]
        for slot, stack_object in enumerate(observation.stack_objects):
            if int(stack_object.kind) == 0:  # spell already appears in visible cards
                continue
            card_def_id = int(stack_object.source_card_registry_key)
            definition_rows.append(self.definition_row(card_def_id))
            roles.append(stack_role)
            slots.append(slot)
            identity_ids.append(card_def_id)

        size = len(definition_rows)
        if identity_mode == "identity":
            opaque_ids = np.asarray(identity_ids, dtype=np.int32)
            opaque_valid = np.ones(size, dtype=np.bool_)
        else:
            opaque_ids = np.full(size, -1, dtype=np.int32)
            opaque_valid = np.zeros(size, dtype=np.bool_)
        return SemanticObjectProjection(
            semantic_pack_hash=self.semantic_pack_hash,
            identity_mode=identity_mode,
            object_definition_rows=np.asarray(definition_rows, dtype=np.int32),
            object_roles=np.asarray(roles, dtype=np.uint8),
            object_slots=np.asarray(slots, dtype=np.int32),
            opaque_identity_ids=opaque_ids,
            opaque_identity_valid=opaque_valid,
        )

    def batch(
        self, projections: Sequence[SemanticObjectProjection]
    ) -> SemanticObjectBatch:
        if not projections:
            raise SemanticProjectionError(
                "cannot infer identity mode from an empty batch"
            )
        identity_mode = projections[0].identity_mode
        offsets = [0]
        for projection in projections:
            if projection.semantic_pack_hash != self.semantic_pack_hash:
                raise SemanticProjectionError(
                    "cannot batch projections from different packs"
                )
            if projection.identity_mode != identity_mode:
                raise SemanticProjectionError("cannot batch different identity modes")
            offsets.append(offsets[-1] + len(projection.object_definition_rows))

        def concatenate(field: str, dtype: np.dtype[Any]) -> np.ndarray:
            values = [getattr(projection, field) for projection in projections]
            if not values or offsets[-1] == 0:
                return np.zeros((0,), dtype=dtype)
            return np.concatenate(values).astype(dtype, copy=False)

        return SemanticObjectBatch(
            semantic_pack_hash=self.semantic_pack_hash,
            identity_mode=identity_mode,
            sample_offsets=np.asarray(offsets, dtype=np.int32),
            object_definition_rows=concatenate("object_definition_rows", np.int32),
            object_roles=concatenate("object_roles", np.uint8),
            object_slots=concatenate("object_slots", np.int32),
            opaque_identity_ids=concatenate("opaque_identity_ids", np.int32),
            opaque_identity_valid=concatenate("opaque_identity_valid", np.bool_),
        )

    def pad_catalog(self) -> dict[str, np.ndarray]:
        def pad_spans(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            lengths = np.diff(offsets)
            rows = len(lengths)
            width = int(lengths.max()) if lengths.size else 0
            kinds = np.zeros((rows, width), dtype=np.uint16)
            values = np.zeros((rows, width), dtype=np.int32)
            mask = np.zeros((rows, width), dtype=np.bool_)
            for row in range(rows):
                start = int(offsets[row])
                end = int(offsets[row + 1])
                size = end - start
                kinds[row, :size] = self.catalog.token_kind[start:end]
                values[row, :size] = self.catalog.token_value[start:end]
                mask[row, :size] = True
            return kinds, values, mask

        definition_kind, definition_value, definition_mask = pad_spans(
            self.catalog.definition_offsets
        )
        program_kind, program_value, program_mask = pad_spans(
            self.catalog.program_offsets
        )

        program_lengths = np.diff(self.catalog.definition_program_offsets)
        program_width = int(program_lengths.max()) if program_lengths.size else 0
        definition_program_rows = np.full(
            (len(program_lengths), program_width), -1, dtype=np.int32
        )
        definition_program_mask = np.zeros(
            (len(program_lengths), program_width), dtype=np.bool_
        )
        for row in range(len(program_lengths)):
            start = int(self.catalog.definition_program_offsets[row])
            end = int(self.catalog.definition_program_offsets[row + 1])
            size = end - start
            definition_program_rows[row, :size] = self.catalog.definition_program_rows[
                start:end
            ]
            definition_program_mask[row, :size] = True

        refs_by_program: list[list[tuple[int, int]]] = [
            [] for _ in range(self.catalog.programs)
        ]
        for source, target in zip(
            self.catalog.definition_ref_source_tokens,
            self.catalog.definition_ref_target_rows,
        ):
            program_row = int(
                np.searchsorted(self.catalog.program_offsets, source, side="right") - 1
            )
            if not 0 <= program_row < len(refs_by_program):
                raise SemanticProjectionError(
                    "definition reference is outside program spans"
                )
            local_source = int(source - self.catalog.program_offsets[program_row])
            refs_by_program[program_row].append((local_source, int(target)))
        ref_width = max((len(refs) for refs in refs_by_program), default=0)
        ref_source_positions = np.full(
            (len(refs_by_program), ref_width), -1, dtype=np.int32
        )
        ref_target_rows = np.full((len(refs_by_program), ref_width), -1, dtype=np.int32)
        ref_mask = np.zeros((len(refs_by_program), ref_width), dtype=np.bool_)
        for row, refs in enumerate(refs_by_program):
            for column, (source, target) in enumerate(refs):
                ref_source_positions[row, column] = source
                ref_target_rows[row, column] = target
                ref_mask[row, column] = True

        return {
            "definition_token_kind": definition_kind,
            "definition_token_value": definition_value,
            "definition_token_mask": definition_mask,
            "program_token_kind": program_kind,
            "program_token_value": program_value,
            "program_token_mask": program_mask,
            "definition_program_rows": definition_program_rows,
            "definition_program_mask": definition_program_mask,
            "definition_ref_source_positions": ref_source_positions,
            "definition_ref_target_rows": ref_target_rows,
            "definition_ref_mask": ref_mask,
        }

    def artifact_header(self, identity_mode: str) -> SemanticArtifactHeader:
        if identity_mode not in self.schema.identity_modes:
            raise SemanticProjectionError(f"unknown identity mode {identity_mode!r}")
        return SemanticArtifactHeader(
            learning_schema_version=int(self.schema.raw["schema_version"]),
            learning_schema_hash=self.schema.schema_hash,
            content_pack_schema_version=self.content_pack_schema_version,
            content_pack_hash=self.content_pack_hash,
            semantic_pack_hash=self.semantic_pack_hash,
            identity_mode=identity_mode,
            binding_hash=self.binding_hash if identity_mode == "identity" else None,
            array_layout_version=int(self.schema.raw["array_layout_version"]),
        )

    def validate_artifact_header(
        self,
        header: SemanticArtifactHeader | Mapping[str, Any] | str,
        *,
        identity_mode: str,
    ) -> SemanticArtifactHeader:
        if isinstance(header, str):
            actual = SemanticArtifactHeader.from_json(header)
        elif isinstance(header, SemanticArtifactHeader):
            actual = header
        else:
            try:
                actual = SemanticArtifactHeader(**dict(header))
            except TypeError as error:
                raise ArtifactCompatibilityError(
                    f"invalid semantic artifact header fields: {error}"
                ) from error
        expected = self.artifact_header(identity_mode)
        if actual != expected:
            mismatches = [
                field
                for field in expected.__dataclass_fields__
                if getattr(actual, field) != getattr(expected, field)
            ]
            raise ArtifactCompatibilityError(
                "semantic artifact header mismatch: " + ", ".join(mismatches)
            )
        return actual


__all__ = [
    "ArtifactCompatibilityError",
    "BoundSemanticPack",
    "ContentPackBindingError",
    "CURRENT_PACK_KEYS",
    "DEFAULT_SCHEMA_PATH",
    "LearningSchema",
    "PROGRAM_KINDS",
    "SemanticArtifactHeader",
    "SemanticCatalog",
    "SemanticObjectBatch",
    "SemanticObjectProjection",
    "SemanticProjectionError",
    "TOKEN_KINDS",
    "UnadmittedDefinitionError",
    "UnknownSchemaError",
    "token_value",
]
