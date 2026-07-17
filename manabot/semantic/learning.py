"""Versioned, viewer-safe semantic-program projection for learning.

The rules engine remains authoritative. This module binds the checked-in
semantic IR to one immutable ContentPack manifest, builds a shared typed token
graph, and maps only already-viewer-safe Observation objects into that graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .compiler import IR_SCHEMA_VERSION, Opcode, ProgramKind, canonical_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = ROOT / "content/semantic/v1/learning_schema.json"
DEFAULT_IR_PATH = ROOT / "content/semantic/v1/generated/two_deck.ir.json"


class SemanticProjectionError(ValueError):
    """Base error for invalid semantic inputs or incompatible artifacts."""


class UnknownSchemaError(SemanticProjectionError):
    """A schema version is not supported."""


class UnknownOpcodeError(SemanticProjectionError):
    """An instruction opcode is unknown or disagrees with its name."""


class ContentPackBindingError(SemanticProjectionError):
    """The checked IR cannot bind exactly to a ContentPack manifest."""


class UnadmittedDefinitionError(SemanticProjectionError):
    """A visible runtime definition is absent from the semantic IR."""


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
    structures: dict[str, int]
    fields: dict[str, int]
    program_kinds: dict[str, int]
    opcodes: dict[str, int]
    enums: dict[str, dict[str, int]]
    special_roles: dict[str, int]
    object_roles: dict[str, int]
    identity_modes: dict[str, int]
    local_role_base: int

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCHEMA_PATH) -> "LearningSchema":
        raw = _load_json(Path(path))
        version = _integer(raw.get("schema_version"), "learning_schema.schema_version")
        if version != 1:
            raise UnknownSchemaError(
                f"learning_schema.schema_version: expected 1, got {version}"
            )
        ir_version = _integer(
            raw.get("ir_schema_version"), "learning_schema.ir_schema_version"
        )
        if ir_version != IR_SCHEMA_VERSION:
            raise UnknownSchemaError(
                "learning_schema.ir_schema_version: "
                f"expected {IR_SCHEMA_VERSION}, got {ir_version}"
            )
        layout = _integer(
            raw.get("array_layout_version"),
            "learning_schema.array_layout_version",
        )
        if layout != 1:
            raise UnknownSchemaError(
                f"learning_schema.array_layout_version: expected 1, got {layout}"
            )

        token_kinds = _validate_id_map(
            raw.get("token_kinds"), "learning_schema.token_kinds", allow_zero=True
        )
        required_token_kinds = {
            "padding",
            "structure",
            "field",
            "program_kind",
            "opcode",
            "integer",
            "boolean",
            "enum",
            "role",
            "definition_ref",
            "null",
        }
        if set(token_kinds) != required_token_kinds:
            raise SemanticProjectionError(
                "learning_schema.token_kinds: expected exactly "
                f"{sorted(required_token_kinds)}"
            )
        if token_kinds["padding"] != 0:
            raise SemanticProjectionError(
                "learning_schema.token_kinds.padding: expected 0"
            )

        program_kinds = _validate_id_map(
            raw.get("program_kinds"), "learning_schema.program_kinds"
        )
        expected_program_kinds = {kind.name.lower(): int(kind) for kind in ProgramKind}
        if program_kinds != expected_program_kinds:
            raise SemanticProjectionError(
                "learning_schema.program_kinds disagrees with compiler ProgramKind"
            )

        opcodes = _validate_id_map(raw.get("opcodes"), "learning_schema.opcodes")
        expected_opcodes = {opcode.name.lower(): int(opcode) for opcode in Opcode}
        if opcodes != expected_opcodes:
            raise SemanticProjectionError(
                "learning_schema.opcodes disagrees with compiler Opcode"
            )

        enums = {
            _string(name, "learning_schema.enums.key"): _validate_id_map(
                values, f"learning_schema.enums.{name}"
            )
            for name, values in _mapping(
                raw.get("enums"), "learning_schema.enums"
            ).items()
        }
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
            structures=_validate_id_map(
                raw.get("structures"), "learning_schema.structures"
            ),
            fields=_validate_id_map(raw.get("fields"), "learning_schema.fields"),
            program_kinds=program_kinds,
            opcodes=opcodes,
            enums=enums,
            special_roles=_validate_id_map(
                raw.get("special_roles"), "learning_schema.special_roles"
            ),
            object_roles=_validate_id_map(
                raw.get("object_roles"), "learning_schema.object_roles"
            ),
            identity_modes=_validate_id_map(
                raw.get("identity_modes"), "learning_schema.identity_modes"
            ),
            local_role_base=_integer(
                raw.get("local_role_base"), "learning_schema.local_role_base"
            ),
        )


def _walk_instructions(instructions: Sequence[Any]) -> Iterable[Mapping[str, Any]]:
    for raw_instruction in instructions:
        instruction = _mapping(raw_instruction, "instruction")
        yield instruction
        for field in ("then", "otherwise", "body"):
            nested = instruction.get(field)
            if nested is not None:
                yield from _walk_instructions(_sequence(nested, f"instruction.{field}"))


@dataclass(frozen=True)
class SemanticIr:
    raw: dict[str, Any]
    ir_hash: str
    definitions: tuple[Mapping[str, Any], ...]
    programs: tuple[Mapping[str, Any], ...]

    @classmethod
    def load(
        cls,
        schema: LearningSchema,
        path: str | Path = DEFAULT_IR_PATH,
    ) -> "SemanticIr":
        raw = _load_json(Path(path))
        version = _integer(raw.get("schema_version"), "semantic_ir.schema_version")
        if version != schema.raw["ir_schema_version"]:
            raise UnknownSchemaError(
                "semantic_ir.schema_version: "
                f"expected {schema.raw['ir_schema_version']}, got {version}"
            )
        declared_hash = _string(raw.get("ir_hash"), "semantic_ir.ir_hash")
        unhashed = dict(raw)
        del unhashed["ir_hash"]
        actual_hash = _hash_json(unhashed)
        if declared_hash != actual_hash:
            raise SemanticProjectionError(
                f"semantic_ir.ir_hash: expected {actual_hash}, got {declared_hash}"
            )

        definitions = tuple(
            _mapping(item, f"semantic_ir.definitions[{index}]")
            for index, item in enumerate(
                _sequence(raw.get("definitions"), "semantic_ir.definitions")
            )
        )
        programs = tuple(
            _mapping(item, f"semantic_ir.programs[{index}]")
            for index, item in enumerate(
                _sequence(raw.get("programs"), "semantic_ir.programs")
            )
        )
        for index, definition in enumerate(definitions):
            if (
                _integer(
                    definition.get("semantic_index"),
                    f"semantic_ir.definitions[{index}].semantic_index",
                )
                != index
            ):
                raise SemanticProjectionError(
                    f"semantic_ir.definitions[{index}]: non-canonical semantic_index"
                )
            for program_index in _sequence(
                definition.get("program_indexes"),
                f"semantic_ir.definitions[{index}].program_indexes",
            ):
                program_row = _integer(program_index, "definition.program_indexes[]")
                if not 0 <= program_row < len(programs):
                    raise SemanticProjectionError(
                        f"semantic_ir.definitions[{index}]: invalid program index {program_row}"
                    )

        for index, program in enumerate(programs):
            if (
                _integer(
                    program.get("program_index"), f"program[{index}].program_index"
                )
                != index
            ):
                raise SemanticProjectionError(
                    f"semantic_ir.programs[{index}]: non-canonical program_index"
                )
            definition_index = _integer(
                program.get("definition_index"), f"program[{index}].definition_index"
            )
            if not 0 <= definition_index < len(definitions):
                raise SemanticProjectionError(
                    f"semantic_ir.programs[{index}]: invalid definition index"
                )
            kind_name = _string(program.get("kind_name"), f"program[{index}].kind_name")
            kind = _integer(program.get("kind"), f"program[{index}].kind")
            expected_kind = schema.program_kinds.get(kind_name)
            if expected_kind is None or kind != expected_kind:
                raise SemanticProjectionError(
                    f"semantic_ir.programs[{index}]: unknown or mismatched kind {kind_name!r}/{kind}"
                )
            instructions = _sequence(
                program.get("instructions"), f"program[{index}].instructions"
            )
            for instruction in _walk_instructions(instructions):
                op_name = _string(instruction.get("op_name"), "instruction.op_name")
                opcode = _integer(instruction.get("opcode"), "instruction.opcode")
                expected_opcode = schema.opcodes.get(op_name)
                if expected_opcode is None:
                    raise UnknownOpcodeError(f"unknown opcode name {op_name!r}")
                if opcode != expected_opcode:
                    raise UnknownOpcodeError(
                        f"opcode {opcode} disagrees with {op_name!r} ({expected_opcode})"
                    )
        return cls(raw, actual_hash, definitions, programs)


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


class _TokenBuilder:
    _MANA_FIELDS = {
        "generic": "mana_generic",
        "W": "mana_w",
        "U": "mana_u",
        "B": "mana_b",
        "R": "mana_r",
        "G": "mana_g",
        "C": "mana_c",
    }

    def __init__(self, schema: LearningSchema, ir: SemanticIr):
        self.schema = schema
        self.ir = ir
        self.kinds: list[int] = []
        self.values: list[int] = []
        self.references: list[tuple[int, int]] = []

    def emit(self, kind: str, value: int) -> int:
        if kind not in self.schema.token_kinds:
            raise SemanticProjectionError(f"unknown token kind {kind!r}")
        self.kinds.append(self.schema.token_kinds[kind])
        self.values.append(int(value))
        return len(self.kinds) - 1

    def structure(self, name: str) -> None:
        try:
            value = self.schema.structures[name]
        except KeyError as error:
            raise SemanticProjectionError(f"unknown structure {name!r}") from error
        self.emit("structure", value)

    def field(self, name: str) -> None:
        try:
            value = self.schema.fields[name]
        except KeyError as error:
            raise SemanticProjectionError(f"unknown semantic field {name!r}") from error
        self.emit("field", value)

    def enum(self, domain: str, value: str) -> None:
        try:
            encoded = self.schema.enums[domain][value]
        except KeyError as error:
            raise SemanticProjectionError(
                f"unknown {domain} categorical value {value!r}"
            ) from error
        self.emit("enum", encoded)

    def role(self, value: str, roles: Mapping[str, int]) -> None:
        if value in roles:
            encoded = self.schema.local_role_base + roles[value]
        else:
            try:
                encoded = self.schema.special_roles[value]
            except KeyError as error:
                raise SemanticProjectionError(
                    f"unknown semantic role {value!r}"
                ) from error
        self.emit("role", encoded)

    def mana(self, value: str, context: str) -> None:
        counts: dict[str, int] = {name: 0 for name in self._MANA_FIELDS}
        digits = ""
        for char in value:
            if char.isdigit():
                digits += char
                continue
            if digits:
                counts["generic"] += int(digits)
                digits = ""
            if char not in counts:
                raise SemanticProjectionError(
                    f"{context}: unsupported mana symbol {char!r}"
                )
            counts[char] += 1
        if digits:
            counts["generic"] += int(digits)
        self.structure("mana_begin")
        for component in ("generic", "W", "U", "B", "R", "G", "C"):
            count = counts[component]
            if count:
                self.field(self._MANA_FIELDS[component])
                self.emit("integer", count)
        self.structure("mana_end")

    def _domain_for_string(self, field: str, value: str, context: str) -> str:
        if field in {"types", "types_any"}:
            return "card_type"
        if field == "values" and value in self.schema.enums["card_type"]:
            return "card_type"
        if field == "value":
            if value in self.schema.enums["card_type"]:
                return "card_type"
            if value in self.schema.enums["subtype"]:
                return "subtype"
        fixed = {
            "colors": "color",
            "keywords": "keyword",
            "subtypes": "subtype",
            "supertypes": "supertype",
            "event": "trigger_event",
            "duration": "duration",
            "until": "duration",
            "zone": "zone",
            "controller": "controller",
            "destination": "destination",
            "remainder": "remainder",
            "remainder_order": "ordering",
        }
        if field in fixed:
            return fixed[field]
        if field == "kind":
            if context in {"selector", "predicate", "condition", "subject"}:
                return context
        raise SemanticProjectionError(
            f"{context}.{field}: no typed categorical domain for {value!r}"
        )

    def value(
        self,
        field: str,
        value: Any,
        *,
        context: str,
        roles: Mapping[str, int],
    ) -> None:
        if value is None:
            self.emit("null", 0)
            return
        if isinstance(value, bool):
            self.emit("boolean", int(value))
            return
        if isinstance(value, int):
            self.emit("integer", value)
            return
        if isinstance(value, str):
            if field in {"mana", "mana_cost", "kicker", "cost"}:
                self.mana(value, f"{context}.{field}")
            elif field in {"role", "sources", "target"}:
                self.role(value, roles)
            else:
                self.enum(self._domain_for_string(field, value, context), value)
            return
        if isinstance(value, Mapping):
            nested_context = {
                "affinity": "predicate",
                "condition": "condition",
                "if": "condition",
                "predicate": "predicate",
                "predicates": "predicate",
                "selector": "selector",
                "subject": "subject",
            }.get(field, "object")
            self.object(value, context=nested_context, roles=roles)
            return
        if isinstance(value, Sequence):
            self.list(field, value, context=context, roles=roles)
            return
        raise SemanticProjectionError(
            f"{context}.{field}: unsupported typed value {type(value).__name__}"
        )

    def list(
        self,
        field: str,
        values: Sequence[Any],
        *,
        context: str,
        roles: Mapping[str, int],
    ) -> None:
        if field in {"then", "otherwise", "body"}:
            self.structure(f"{field}_begin")
            for instruction in values:
                self.instruction(_mapping(instruction, f"{context}.{field}[]"), roles)
            self.structure(f"{field}_end")
            return
        self.structure("list_begin")
        for item in values:
            self.value(field, item, context=context, roles=roles)
        self.structure("list_end")

    def object(
        self,
        value: Mapping[str, Any],
        *,
        context: str,
        roles: Mapping[str, int],
    ) -> None:
        structure_context = (
            context
            if context
            in {
                "selector",
                "predicate",
                "condition",
                "subject",
            }
            else "object"
        )
        self.structure(f"{structure_context}_begin")
        for field in sorted(value, key=self._field_sort_key):
            if field in {"op_name", "opcode"}:
                continue
            self.field(field)
            self.value(field, value[field], context=context, roles=roles)
        self.structure(f"{structure_context}_end")

    def _field_sort_key(self, field: str) -> int:
        if field == "definition_index":
            field = "definition_ref"
        try:
            return self.schema.fields[field]
        except KeyError as error:
            raise SemanticProjectionError(
                f"unknown semantic field {field!r}"
            ) from error

    def instruction(
        self, instruction: Mapping[str, Any], roles: Mapping[str, int]
    ) -> None:
        self.structure("instruction_begin")
        op_name = _string(instruction.get("op_name"), "instruction.op_name")
        opcode = _integer(instruction.get("opcode"), "instruction.opcode")
        expected = self.schema.opcodes.get(op_name)
        if expected is None or opcode != expected:
            raise UnknownOpcodeError(
                f"unknown or mismatched opcode {op_name!r}/{opcode}"
            )
        self.emit("opcode", opcode)
        for field in sorted(
            (key for key in instruction if key not in {"op_name", "opcode"}),
            key=self._field_sort_key,
        ):
            self.field("definition_ref" if field == "definition_index" else field)
            if field == "definition_index":
                target = _integer(instruction[field], "instruction.definition_index")
                if not 0 <= target < len(self.ir.definitions):
                    raise SemanticProjectionError(
                        f"instruction.definition_index: invalid row {target}"
                    )
                source = self.emit("definition_ref", 0)
                self.references.append((source, target))
            else:
                self.value(
                    field, instruction[field], context="instruction", roles=roles
                )
        self.structure("instruction_end")

    def definition(self, definition: Mapping[str, Any]) -> None:
        self.structure("definition_begin")
        self.structure("characteristics_begin")
        characteristics = _mapping(
            definition.get("characteristics"), "definition.characteristics"
        )
        for field in sorted(characteristics, key=self._field_sort_key):
            self.field(field)
            self.value(
                field, characteristics[field], context="characteristics", roles={}
            )
        self.structure("characteristics_end")
        self.structure("programs_begin")
        self.structure("programs_end")
        self.structure("definition_end")

    def program(self, program: Mapping[str, Any]) -> None:
        raw_targets = _sequence(program.get("targets"), "program.targets")
        roles: dict[str, int] = {}
        for index, raw_target in enumerate(raw_targets):
            target = _mapping(raw_target, f"program.targets[{index}]")
            name = _string(target.get("role"), f"program.targets[{index}].role")
            if name in roles:
                raise SemanticProjectionError(f"duplicate target role {name!r}")
            roles[name] = index

        self.structure("program_begin")
        kind_name = _string(program.get("kind_name"), "program.kind_name")
        kind = _integer(program.get("kind"), "program.kind")
        if self.schema.program_kinds.get(kind_name) != kind:
            raise SemanticProjectionError(
                f"unknown or mismatched program kind {kind_name!r}"
            )
        self.emit("program_kind", kind)

        if "cost" in program:
            self.field("cost")
            self.structure("cost_begin")
            cost = _mapping(program["cost"], "program.cost")
            for field in sorted(cost, key=self._field_sort_key):
                self.field(field)
                self.value(field, cost[field], context="cost", roles=roles)
            self.structure("cost_end")

        self.field("targets")
        self.structure("targets_begin")
        for index, raw_target in enumerate(raw_targets):
            target = _mapping(raw_target, f"program.targets[{index}]")
            self.structure("target_begin")
            for field in ("role", "min", "max", "selector"):
                if field not in target:
                    raise SemanticProjectionError(
                        f"program.targets[{index}]: missing field {field!r}"
                    )
                self.field(field)
                self.value(field, target[field], context="target", roles=roles)
            unknown = set(target) - {"role", "min", "max", "selector"}
            if unknown:
                raise SemanticProjectionError(
                    f"program.targets[{index}]: unknown fields {sorted(unknown)}"
                )
            self.structure("target_end")
        self.structure("targets_end")

        if "trigger" in program:
            self.field("trigger")
            self.structure("trigger_begin")
            trigger = _mapping(program["trigger"], "program.trigger")
            for field in sorted(trigger, key=self._field_sort_key):
                self.field(field)
                self.value(field, trigger[field], context="trigger", roles=roles)
            self.structure("trigger_end")

        self.field("instructions")
        self.structure("instructions_begin")
        for instruction in _sequence(
            program.get("instructions"), "program.instructions"
        ):
            self.instruction(_mapping(instruction, "program.instructions[]"), roles)
        self.structure("instructions_end")
        self.structure("program_end")

    def build(self) -> SemanticCatalog:
        definition_offsets = [0]
        for definition in self.ir.definitions:
            self.definition(definition)
            definition_offsets.append(len(self.kinds))

        program_offsets = [len(self.kinds)]
        for program in self.ir.programs:
            self.program(program)
            program_offsets.append(len(self.kinds))

        definition_program_offsets = [0]
        definition_program_rows: list[int] = []
        for definition in self.ir.definitions:
            definition_program_rows.extend(
                _integer(item, "definition.program_indexes[]")
                for item in _sequence(
                    definition.get("program_indexes"), "definition.program_indexes"
                )
            )
            definition_program_offsets.append(len(definition_program_rows))

        arrays = SemanticCatalog(
            token_kind=np.asarray(self.kinds, dtype=np.uint16),
            token_value=np.asarray(self.values, dtype=np.int32),
            definition_offsets=np.asarray(definition_offsets, dtype=np.int32),
            program_offsets=np.asarray(program_offsets, dtype=np.int32),
            definition_program_offsets=np.asarray(
                definition_program_offsets, dtype=np.int32
            ),
            definition_program_rows=np.asarray(definition_program_rows, dtype=np.int32),
            definition_ref_source_tokens=np.asarray(
                [source for source, _ in self.references], dtype=np.int32
            ),
            definition_ref_target_rows=np.asarray(
                [target for _, target in self.references], dtype=np.int32
            ),
        )
        for value in arrays.__dict__.values():
            value.setflags(write=False)
        return arrays


@dataclass(frozen=True)
class SemanticArtifactHeader:
    learning_schema_version: int
    learning_schema_hash: str
    semantic_ir_schema_version: int
    semantic_ir_hash: str
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
            "semantic_ir_hash": self.semantic_ir_hash,
            "semantic_ir_schema_version": self.semantic_ir_schema_version,
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
    ir: SemanticIr
    content_pack_schema_version: int
    content_pack_hash: str
    binding_hash: str
    semantic_pack_hash: str
    definition_row_by_card_def_id: dict[int, int]
    catalog: SemanticCatalog

    @classmethod
    def bind(
        cls,
        manifest: Mapping[str, Any],
        *,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        ir_path: str | Path = DEFAULT_IR_PATH,
    ) -> "BoundSemanticPack":
        schema = LearningSchema.load(schema_path)
        ir = SemanticIr.load(schema, ir_path)
        manifest = _mapping(manifest, "content_pack_manifest")
        pack_schema = _integer(
            manifest.get("schema_version"), "content_pack_manifest.schema_version"
        )
        pack_hash = _string(
            manifest.get("content_digest"), "content_pack_manifest.content_digest"
        )
        manifest_definitions = _sequence(
            manifest.get("definitions"), "content_pack_manifest.definitions"
        )
        id_by_name: dict[str, int] = {}
        seen_ids: set[int] = set()
        for index, raw_definition in enumerate(manifest_definitions):
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
            if card_def_id in seen_ids:
                raise ContentPackBindingError(f"duplicate CardDefId {card_def_id}")
            if registry_name in id_by_name:
                raise ContentPackBindingError(
                    f"duplicate ContentPack registry name {registry_name!r}"
                )
            seen_ids.add(card_def_id)
            id_by_name[registry_name] = card_def_id

        rows_by_id: dict[int, int] = {}
        for row, semantic_definition in enumerate(ir.definitions):
            binding = _mapping(
                semantic_definition.get("content_pack_binding"),
                f"semantic_ir.definitions[{row}].content_pack_binding",
            )
            if binding.get("kind") != "legacy_registry_name":
                raise ContentPackBindingError(
                    f"semantic definition {row}: unsupported binding {binding.get('kind')!r}"
                )
            registry_name = _string(
                binding.get("value"),
                f"semantic_ir.definitions[{row}].content_pack_binding.value",
            )
            try:
                card_def_id = id_by_name[registry_name]
            except KeyError as error:
                raise ContentPackBindingError(
                    f"semantic definition {row} does not bind: {registry_name!r}"
                ) from error
            if card_def_id in rows_by_id:
                raise ContentPackBindingError(
                    f"CardDefId {card_def_id} is bound by multiple semantic definitions"
                )
            rows_by_id[card_def_id] = row

        binding_receipt = [
            {"card_def_id": card_def_id, "definition_row": row}
            for card_def_id, row in sorted(rows_by_id.items())
        ]
        binding_hash = _hash_json(binding_receipt)
        semantic_pack_hash = _hash_json(
            {
                "content_pack_hash": pack_hash,
                "content_pack_schema_version": pack_schema,
                "learning_schema_hash": schema.schema_hash,
                "semantic_ir_hash": ir.ir_hash,
            }
        )
        return cls(
            schema=schema,
            ir=ir,
            content_pack_schema_version=pack_schema,
            content_pack_hash=pack_hash,
            binding_hash=binding_hash,
            semantic_pack_hash=semantic_pack_hash,
            definition_row_by_card_def_id=rows_by_id,
            catalog=_TokenBuilder(schema, ir).build(),
        )

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        ir_path: str | Path = DEFAULT_IR_PATH,
    ) -> "BoundSemanticPack":
        try:
            manifest = env.content_pack_manifest()
        except AttributeError as error:
            raise ContentPackBindingError(
                "environment does not expose content_pack_manifest()"
            ) from error
        return cls.bind(manifest, schema_path=schema_path, ir_path=ir_path)

    def definition_row(self, card_def_id: int) -> int:
        """Return the admitted semantic row for one runtime definition."""

        try:
            return self.definition_row_by_card_def_id[card_def_id]
        except KeyError as error:
            raise UnadmittedDefinitionError(
                f"visible CardDefId {card_def_id} is absent from semantic IR"
            ) from error

    def program_rows(self, definition_row: int) -> tuple[int, ...]:
        """Return the typed program rows owned by one semantic definition."""

        if not 0 <= definition_row < len(self.ir.definitions):
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
            [] for _ in range(len(self.ir.programs))
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
            semantic_ir_schema_version=int(self.ir.raw["schema_version"]),
            semantic_ir_hash=self.ir.ir_hash,
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
    "DEFAULT_IR_PATH",
    "DEFAULT_SCHEMA_PATH",
    "LearningSchema",
    "SemanticArtifactHeader",
    "SemanticCatalog",
    "SemanticIr",
    "SemanticObjectBatch",
    "SemanticObjectProjection",
    "SemanticProjectionError",
    "UnadmittedDefinitionError",
    "UnknownOpcodeError",
    "UnknownSchemaError",
]
