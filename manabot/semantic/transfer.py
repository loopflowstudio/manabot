"""Preliminary typed-opcode alignment probe over semantic programs.

This is deliberately a supervised alignment probe, not a gameplay policy or
evidence of semantic compositional transfer.
It joins the checked semantic-program catalog to the experimental structured
choice head without changing the production observation or action ABI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from manabot.semantic.compiler import canonical_json
from manabot.semantic.learning import BoundSemanticPack

ARMS = (
    "card_id_legacy",
    "card_id_structured",
    "semantic_card_id_structured",
    "semantic_only_structured",
)
SEMANTIC_CONTROLS = (
    "semantic_only_opcode_masked",
    "semantic_only_token_shuffled",
)
CONTROL_OPS = frozenset({"branch", "for_each_target"})
MASKED_OPCODE_TOKEN = "control:masked_opcode"


class TransferExperimentError(ValueError):
    """The pre-registered experiment contract could not be satisfied."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_workload(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    workload = json.loads(raw)
    validate_workload(workload)
    return workload, hashlib.sha256(raw).hexdigest()


def validate_workload(workload: Mapping[str, Any]) -> None:
    if workload.get("schema_version") != 1:
        raise TransferExperimentError("workload schema_version must be 1")
    seeds = workload.get("model_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise TransferExperimentError("model_seeds must contain >=3 distinct seeds")
    holdout = workload.get("held_out_definitions")
    if not isinstance(holdout, Mapping) or set(holdout) != {
        "ur_lessons",
        "gw_allies",
    }:
        raise TransferExperimentError("holdout must contain both curated decks")
    if any(not isinstance(rows, list) or len(rows) != 2 for rows in holdout.values()):
        raise TransferExperimentError("holdout must contain two definitions per deck")
    if len({key for rows in holdout.values() for key in rows}) != 4:
        raise TransferExperimentError("held-out definitions must be distinct")
    evaluation = workload.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("seats") != [0, 1]:
        raise TransferExperimentError("evaluation seats must be exactly [0, 1]")
    if evaluation.get("confidence_method") != "wilson":
        raise TransferExperimentError(
            "only pre-registered Wilson intervals are supported"
        )
    for section, fields in {
        "training": ("epochs", "limited_retraining_steps", "hidden_dim"),
        "performance": ("warmup", "samples"),
    }.items():
        value = workload.get(section)
        if not isinstance(value, Mapping) or any(
            not isinstance(value.get(field), int) or value[field] <= 0
            for field in fields
        ):
            raise TransferExperimentError(f"{section} integer budgets must be positive")
    if workload["gates"].get("minimum_legal_choices", 0) <= 32:
        raise TransferExperimentError("minimum_legal_choices must exceed 32")


@dataclass(frozen=True)
class SemanticTransferSpec:
    """Checkpoint meaning expressed in symbolic rather than runtime IDs."""

    spec_version: int
    semantic_ir_hash: str
    content_pack_hash: str
    enum_schema_hash: str
    token_kind_symbols: tuple[str, ...]
    opcode_symbols: tuple[str, ...]
    definition_symbols: tuple[str, ...]
    token_vocabulary: tuple[str, ...]
    max_program_tokens: int
    enum_domain_rebinding: str = "unsupported_fail_closed_v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "token_kind_symbols",
            "opcode_symbols",
            "definition_symbols",
            "token_vocabulary",
        ):
            value[field] = list(value[field])
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticTransferSpec":
        raw = dict(value)
        for field in (
            "token_kind_symbols",
            "opcode_symbols",
            "definition_symbols",
            "token_vocabulary",
        ):
            raw[field] = tuple(raw[field])
        return cls(**raw)

    @property
    def digest(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class RuntimeTables:
    token_kinds: Mapping[str, int]
    opcodes: Mapping[str, int]


class SymbolicProgramBinder:
    """Rebind runtime token/opcode IDs to checkpoint-stable vocabulary slots."""

    def __init__(
        self,
        pack: BoundSemanticPack,
        spec: SemanticTransferSpec,
        *,
        runtime_tables: RuntimeTables | None = None,
    ) -> None:
        if spec.semantic_ir_hash != pack.ir.ir_hash:
            raise TransferExperimentError("checkpoint semantic IR is incompatible")
        if spec.content_pack_hash != pack.content_pack_hash:
            raise TransferExperimentError("checkpoint ContentPack is incompatible")
        if spec.enum_schema_hash != _sha256(pack.schema.enums):
            raise TransferExperimentError(
                "enum-domain schema changed; v1 flat enum tokens cannot rebind it"
            )
        tables = runtime_tables or RuntimeTables(
            token_kinds=pack.schema.token_kinds,
            opcodes=pack.schema.opcodes,
        )
        if set(tables.token_kinds) != set(spec.token_kind_symbols):
            raise TransferExperimentError("runtime token-kind symbols are incompatible")
        if set(tables.opcodes) != set(spec.opcode_symbols):
            raise TransferExperimentError("runtime opcode symbols are incompatible")
        self.pack = pack
        self.spec = spec
        self._kind_by_id = {value: key for key, value in tables.token_kinds.items()}
        self._opcode_by_id = {value: key for key, value in tables.opcodes.items()}
        self._vocabulary = {
            symbol: index for index, symbol in enumerate(spec.token_vocabulary)
        }
        self._structure_by_id = {
            value: key for key, value in pack.schema.structures.items()
        }
        self._field_by_id = {value: key for key, value in pack.schema.fields.items()}
        self._program_kind_by_id = {
            value: key for key, value in pack.schema.program_kinds.items()
        }
        self._special_role_by_id = {
            value: key for key, value in pack.schema.special_roles.items()
        }
        self._definition_ref_target = dict(
            zip(
                (int(value) for value in pack.catalog.definition_ref_source_tokens),
                (int(value) for value in pack.catalog.definition_ref_target_rows),
                strict=True,
            )
        )

    def _named(self, values: Mapping[int, str], value: int, family: str) -> str:
        try:
            return values[value]
        except KeyError as error:
            raise TransferExperimentError(
                f"unknown runtime {family} ID {value}"
            ) from error

    def token_symbol(self, absolute_index: int, kind: int, value: int) -> str:
        family = self._named(self._kind_by_id, kind, "token-kind")
        if family == "structure":
            payload = self._named(self._structure_by_id, value, family)
        elif family == "field":
            payload = self._named(self._field_by_id, value, family)
        elif family == "program_kind":
            payload = self._named(self._program_kind_by_id, value, family)
        elif family == "opcode":
            payload = self._named(self._opcode_by_id, value, family)
        elif family == "definition_ref":
            try:
                row = self._definition_ref_target[absolute_index]
                payload = str(self.pack.ir.definitions[row]["semantic_key"])
            except KeyError as error:
                raise TransferExperimentError(
                    "definition-reference token is missing its explicit edge"
                ) from error
        elif family == "role":
            if value in self._special_role_by_id:
                payload = self._special_role_by_id[value]
            elif value >= self.pack.schema.local_role_base:
                payload = f"local:{value - self.pack.schema.local_role_base}"
            else:
                raise TransferExperimentError(f"unknown runtime role ID {value}")
        elif family == "enum":
            # The v1 projection omitted the enum domain. Keep exact-schema raw
            # values and reject enum-table changes in __init__.
            payload = f"exact-schema:{value}"
        elif family in {"integer", "boolean", "null"}:
            payload = str(value)
        elif family == "padding":
            raise TransferExperimentError(
                "padding cannot occur inside a ragged program"
            )
        else:
            raise TransferExperimentError(f"unsupported token family {family!r}")
        return f"{family}:{payload}"

    def encode_program(
        self,
        program_row: int,
        *,
        token_kind: np.ndarray | None = None,
        token_value: np.ndarray | None = None,
    ) -> tuple[int, ...]:
        kinds = self.pack.catalog.token_kind if token_kind is None else token_kind
        values = self.pack.catalog.token_value if token_value is None else token_value
        start = int(self.pack.catalog.program_offsets[program_row])
        end = int(self.pack.catalog.program_offsets[program_row + 1])
        if end - start > self.spec.max_program_tokens:
            raise TransferExperimentError(
                f"program {program_row} exceeds checkpoint token budget"
            )
        encoded = []
        for index in range(start, end):
            symbol = self.token_symbol(index, int(kinds[index]), int(values[index]))
            try:
                encoded.append(self._vocabulary[symbol])
            except KeyError as error:
                raise TransferExperimentError(
                    f"runtime token {symbol!r} is absent from checkpoint vocabulary"
                ) from error
        return tuple(encoded)


def _symbolic_definition_order(pack: BoundSemanticPack) -> tuple[str, ...]:
    keys = [str(row["semantic_key"]) for row in pack.ir.definitions]
    # Intentionally opaque and unrelated to semantic/IR ordering.
    return tuple(sorted(keys, key=lambda key: hashlib.sha256(key.encode()).digest()))


def build_spec(pack: BoundSemanticPack) -> SemanticTransferSpec:
    provisional = SemanticTransferSpec(
        spec_version=1,
        semantic_ir_hash=pack.ir.ir_hash,
        content_pack_hash=pack.content_pack_hash,
        enum_schema_hash=_sha256(pack.schema.enums),
        token_kind_symbols=tuple(sorted(pack.schema.token_kinds)),
        opcode_symbols=tuple(sorted(pack.schema.opcodes)),
        definition_symbols=_symbolic_definition_order(pack),
        token_vocabulary=(),
        max_program_tokens=max(
            int(end - start)
            for start, end in zip(
                pack.catalog.program_offsets[:-1],
                pack.catalog.program_offsets[1:],
                strict=True,
            )
        ),
    )
    binder = SymbolicProgramBinder(pack, provisional)
    symbols = {
        binder.token_symbol(index, int(kind), int(value))
        for index, (kind, value) in enumerate(
            zip(pack.catalog.token_kind, pack.catalog.token_value, strict=True)
        )
        if int(kind) != pack.schema.token_kinds["padding"]
    }
    symbols.add(MASKED_OPCODE_TOKEN)
    return SemanticTransferSpec(
        spec_version=provisional.spec_version,
        semantic_ir_hash=provisional.semantic_ir_hash,
        content_pack_hash=provisional.content_pack_hash,
        enum_schema_hash=provisional.enum_schema_hash,
        token_kind_symbols=provisional.token_kind_symbols,
        opcode_symbols=provisional.opcode_symbols,
        definition_symbols=provisional.definition_symbols,
        token_vocabulary=tuple(sorted(symbols)),
        max_program_tokens=provisional.max_program_tokens,
    )


def permuted_runtime_projection(
    pack: BoundSemanticPack,
) -> tuple[RuntimeTables, np.ndarray, np.ndarray]:
    token_names = sorted(pack.schema.token_kinds)
    non_padding_ids = sorted(
        value for name, value in pack.schema.token_kinds.items() if name != "padding"
    )
    token_kinds = {"padding": 0}
    token_kinds.update(
        dict(
            zip(
                (name for name in token_names if name != "padding"),
                reversed(non_padding_ids),
            )
        )
    )
    opcode_names = sorted(pack.schema.opcodes)
    opcode_ids = sorted(pack.schema.opcodes.values())
    opcodes = dict(zip(opcode_names, reversed(opcode_ids)))

    kinds = pack.catalog.token_kind.copy()
    values = pack.catalog.token_value.copy()
    old_kind = {value: name for name, value in pack.schema.token_kinds.items()}
    old_opcode = {value: name for name, value in pack.schema.opcodes.items()}
    for index in range(len(kinds)):
        family = old_kind[int(kinds[index])]
        kinds[index] = token_kinds[family]
        if family == "opcode":
            values[index] = opcodes[old_opcode[int(values[index])]]
    return RuntimeTables(token_kinds, opcodes), kinds, values


def _walk_primary(instructions: Sequence[Mapping[str, Any]]) -> str:
    for instruction in instructions:
        name = str(instruction["op_name"])
        if name not in CONTROL_OPS:
            return name
        for field in ("then", "body", "otherwise"):
            nested = instruction.get(field)
            if nested:
                return _walk_primary(nested)
    raise TransferExperimentError("program has no selectable operation")


@dataclass(frozen=True)
class ProgramRecord:
    program_row: int
    program_key: str
    definition_key: str
    deck: str
    held_out: bool
    identity_slot: int
    token_ids: tuple[int, ...]
    target_opcode: int


@dataclass(frozen=True)
class ProbeExample:
    record: ProgramRecord
    seat: int
    replicate: int
    candidate_order: tuple[int, ...]


def build_records(
    pack: BoundSemanticPack,
    spec: SemanticTransferSpec,
    workload: Mapping[str, Any],
    *,
    semantic_control: str | None = None,
) -> tuple[ProgramRecord, ...]:
    if semantic_control not in (None, *SEMANTIC_CONTROLS):
        raise TransferExperimentError(f"unknown semantic control {semantic_control!r}")
    binder = SymbolicProgramBinder(pack, spec)
    heldout_by_key = {
        key: deck
        for deck, keys in workload["held_out_definitions"].items()
        for key in keys
    }
    missing = set(heldout_by_key)
    identity_slot = {key: index for index, key in enumerate(spec.definition_symbols)}
    opcode_slot = {key: index for index, key in enumerate(spec.opcode_symbols)}
    records = []
    for row, program in enumerate(pack.ir.programs):
        definition = pack.ir.definitions[int(program["definition_index"])]
        definition_key = str(definition["semantic_key"])
        missing.discard(definition_key)
        deck = heldout_by_key.get(definition_key, "training")
        token_ids = list(binder.encode_program(row))
        if semantic_control == "semantic_only_opcode_masked":
            masked = spec.token_vocabulary.index(MASKED_OPCODE_TOKEN)
            token_ids = [
                masked if spec.token_vocabulary[token].startswith("opcode:") else token
                for token in token_ids
            ]
        elif semantic_control == "semantic_only_token_shuffled":
            rng = np.random.default_rng(int(workload["dataset_seed"]) + row)
            token_ids = [token_ids[index] for index in rng.permutation(len(token_ids))]
        records.append(
            ProgramRecord(
                program_row=row,
                program_key=str(program["semantic_key"]),
                definition_key=definition_key,
                deck=deck,
                held_out=definition_key in heldout_by_key,
                identity_slot=identity_slot[definition_key],
                token_ids=tuple(token_ids),
                target_opcode=opcode_slot[_walk_primary(program["instructions"])],
            )
        )
    if missing:
        raise TransferExperimentError(
            f"held-out definitions are absent from IR: {sorted(missing)}"
        )
    training_ops = {row.target_opcode for row in records if not row.held_out}
    unsupported = {
        row.program_key
        for row in records
        if row.held_out and row.target_opcode not in training_ops
    }
    if unsupported:
        raise TransferExperimentError(
            f"holdout uses unseen operations: {sorted(unsupported)}"
        )
    return tuple(records)


def build_examples(
    records: Sequence[ProgramRecord],
    workload: Mapping[str, Any],
    *,
    held_out: bool,
    support: bool = False,
) -> tuple[ProbeExample, ...]:
    seed = int(workload["dataset_seed"]) + (10_000 if support else 0)
    rng = np.random.default_rng(seed)
    opcode_count = max(row.target_opcode for row in records) + 1
    candidates = np.arange(opcode_count, dtype=np.int64)
    repeats = (
        int(workload["evaluation"]["limited_retraining_examples_per_program"])
        if support
        else int(workload["evaluation"]["replicates_per_program_and_seat"])
    )
    seats = [0] if support else list(workload["evaluation"]["seats"])
    out = []
    for record in records:
        if record.held_out != held_out:
            continue
        for seat in seats:
            for replicate in range(repeats):
                out.append(
                    ProbeExample(
                        record=record,
                        seat=seat,
                        replicate=replicate,
                        candidate_order=tuple(
                            int(value) for value in rng.permutation(candidates)
                        ),
                    )
                )
    return tuple(out)


@dataclass(frozen=True)
class TensorExamples:
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    identity_slots: torch.Tensor
    candidate_orders: torch.Tensor
    targets: torch.Tensor


def masked_mean_max_program_pool(
    token_embedding: nn.Embedding,
    token_projection: nn.Module,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the landed order-invariant semantic-program pooling path.

    Keeping this operation factored but parameter-free lets diagnostic probes
    reuse the exact negative-control mechanism without changing
    ``TransferPolicy`` parameter names or checkpoint compatibility.
    """

    embedded = token_embedding(token_ids)
    mask = token_mask.unsqueeze(-1)
    mean = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    maximum = embedded.masked_fill(~mask, -torch.inf).amax(dim=1)
    return token_projection(torch.cat([mean, maximum], dim=-1))


def tensorize(examples: Sequence[ProbeExample]) -> TensorExamples:
    if not examples:
        raise TransferExperimentError("cannot tensorize an empty example set")
    width = max(len(example.record.token_ids) for example in examples)
    token_ids = torch.zeros((len(examples), width), dtype=torch.long)
    token_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for row, example in enumerate(examples):
        size = len(example.record.token_ids)
        token_ids[row, :size] = torch.tensor(example.record.token_ids)
        token_mask[row, :size] = True
    candidate_orders = torch.tensor(
        [example.candidate_order for example in examples], dtype=torch.long
    )
    return TensorExamples(
        token_ids=token_ids,
        token_mask=token_mask,
        identity_slots=torch.tensor(
            [example.record.identity_slot for example in examples], dtype=torch.long
        ),
        candidate_orders=candidate_orders,
        targets=torch.tensor(
            [example.record.target_opcode for example in examples], dtype=torch.long
        ),
    )


class TransferPolicy(nn.Module):
    def __init__(
        self,
        arm: str,
        *,
        token_count: int,
        definition_count: int,
        opcode_count: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if arm not in (*ARMS, *SEMANTIC_CONTROLS):
            raise TransferExperimentError(f"unknown arm {arm!r}")
        self.arm = arm
        self.uses_semantics = arm.startswith("semantic_")
        self.uses_identity = arm in {
            "card_id_legacy",
            "card_id_structured",
            "semantic_card_id_structured",
        }
        self.structured = arm != "card_id_legacy"
        if self.uses_semantics:
            self.token_embedding = nn.Embedding(token_count, hidden_dim)
            self.token_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
            )
        if self.uses_identity:
            self.identity_embedding = nn.Embedding(definition_count, hidden_dim)
        input_width = hidden_dim * (int(self.uses_semantics) + int(self.uses_identity))
        self.context = nn.Sequential(nn.Linear(input_width, hidden_dim), nn.Tanh())
        if self.structured:
            self.candidate_embedding = nn.Embedding(opcode_count, hidden_dim)
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.legacy_head = nn.Linear(hidden_dim, opcode_count)

    def forward(self, batch: TensorExamples) -> torch.Tensor:
        pieces = []
        if self.uses_semantics:
            pieces.append(
                masked_mean_max_program_pool(
                    self.token_embedding,
                    self.token_projection,
                    batch.token_ids,
                    batch.token_mask,
                )
            )
        if self.uses_identity:
            pieces.append(self.identity_embedding(batch.identity_slots))
        context = self.context(torch.cat(pieces, dim=-1))
        if not self.structured:
            return self.legacy_head(context)
        candidates = self.candidate_embedding(batch.candidate_orders)
        return torch.einsum("bd,bcd->bc", self.query(context), candidates) / math.sqrt(
            candidates.shape[-1]
        )


def _loss_targets(model: TransferPolicy, batch: TensorExamples) -> torch.Tensor:
    if not model.structured:
        return batch.targets
    return (
        (batch.candidate_orders == batch.targets.unsqueeze(1)).to(torch.int64).argmax(1)
    )


def _train(
    model: TransferPolicy,
    batch: TensorExamples,
    *,
    steps: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target = _loss_targets(model, batch)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(batch), target)
        loss.backward()
        optimizer.step()


def _predictions(model: TransferPolicy, batch: TensorExamples) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        predicted = model(batch).argmax(dim=1)
    if model.structured:
        predicted = batch.candidate_orders.gather(1, predicted.unsqueeze(1)).squeeze(1)
    return predicted


def _wilson(successes: int, total: int, confidence: float) -> dict[str, float | int]:
    if confidence != 0.95:
        raise TransferExperimentError(
            "only the pre-registered 95% interval is supported"
        )
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "accuracy": p,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
    }


def _evaluate(
    model: TransferPolicy,
    examples: Sequence[ProbeExample],
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch = tensorize(examples)
    predicted = _predictions(model, batch).tolist()
    rows = []
    for example, prediction in zip(examples, predicted, strict=True):
        rows.append(
            {
                "program": example.record.program_key,
                "definition": example.record.definition_key,
                "deck": example.record.deck,
                "seat": example.seat,
                "replicate": example.replicate,
                "expected": example.record.target_opcode,
                "predicted": int(prediction),
                "correct": int(prediction) == example.record.target_opcode,
            }
        )
    summary = _wilson(sum(row["correct"] for row in rows), len(rows), confidence)
    summary["by_deck"] = {
        deck: _wilson(
            sum(row["correct"] for row in rows if row["deck"] == deck),
            sum(row["deck"] == deck for row in rows),
            confidence,
        )
        for deck in sorted({row["deck"] for row in rows})
    }
    summary["by_seat"] = {
        str(seat): _wilson(
            sum(row["correct"] for row in rows if row["seat"] == seat),
            sum(row["seat"] == seat for row in rows),
            confidence,
        )
        for seat in sorted({row["seat"] for row in rows})
    }
    return summary, rows


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _performance(
    model: TransferPolicy,
    examples: Sequence[ProbeExample],
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    batch = tensorize(examples[:1])
    warmup = int(workload["performance"]["warmup"])
    samples = int(workload["performance"]["samples"])
    for _ in range(warmup):
        _predictions(model, batch)
    timings = []
    started = perf_counter_ns()
    for _ in range(samples):
        one = perf_counter_ns()
        _predictions(model, batch)
        timings.append(perf_counter_ns() - one)
    elapsed = perf_counter_ns() - started
    return {
        "latency_ns": {
            "p50": int(np.percentile(timings, 50)),
            "p95": int(np.percentile(timings, 95)),
            "samples": samples,
        },
        "examples_per_second": samples / (elapsed / 1_000_000_000),
        "peak_rss_bytes": _peak_rss_bytes(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
    }


def _checkpoint_rebind_control(
    model: TransferPolicy,
    pack: BoundSemanticPack,
    spec: SemanticTransferSpec,
    workload: Mapping[str, Any],
    heldout: Sequence[ProbeExample],
    semantic_control: str | None,
) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(
        {
            "arm": model.arm,
            "hidden_dim": int(workload["training"]["hidden_dim"]),
            "semantic_input_spec": spec.to_dict(),
            "state_dict": model.state_dict(),
        },
        buffer,
    )
    serialized_bytes = buffer.tell()
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    loaded_spec = SemanticTransferSpec.from_dict(checkpoint["semantic_input_spec"])
    reordered = BoundSemanticPack.bind(reordered_manifest(pack))
    rebound_records = build_records(
        reordered, loaded_spec, workload, semantic_control=semantic_control
    )
    rebound_examples = build_examples(rebound_records, workload, held_out=True)
    loaded = TransferPolicy(
        str(checkpoint["arm"]),
        token_count=len(loaded_spec.token_vocabulary),
        definition_count=len(loaded_spec.definition_symbols),
        opcode_count=len(loaded_spec.opcode_symbols),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    loaded.load_state_dict(checkpoint["state_dict"])
    before = _predictions(model, tensorize(heldout))
    after = _predictions(loaded, tensorize(rebound_examples))
    return {
        "serialized_bytes": serialized_bytes,
        "spec_digest_before": spec.digest,
        "spec_digest_after": loaded_spec.digest,
        "prediction_matches": int((before == after).sum()),
        "predictions": int(before.numel()),
        "match_rate": float((before == after).to(torch.float32).mean()),
    }


def reordered_manifest(pack: BoundSemanticPack) -> dict[str, Any]:
    definitions = [
        {
            "card_def_id": len(pack.ir.definitions) - index,
            "registry_name": definition["content_pack_binding"]["value"],
        }
        for index, definition in enumerate(reversed(pack.ir.definitions), 1)
    ]
    return {
        "schema_version": pack.content_pack_schema_version,
        "content_digest": pack.content_pack_hash,
        "definitions": definitions,
    }


def robustness_controls(
    pack: BoundSemanticPack,
    spec: SemanticTransferSpec,
) -> dict[str, Any]:
    regular = SymbolicProgramBinder(pack, spec)
    reordered = BoundSemanticPack.bind(reordered_manifest(pack))
    rebound = SymbolicProgramBinder(reordered, spec)
    rebind_matches = sum(
        regular.encode_program(row) == rebound.encode_program(row)
        for row in range(len(pack.ir.programs))
    )

    tables, kinds, values = permuted_runtime_projection(pack)
    permuted = SymbolicProgramBinder(pack, spec, runtime_tables=tables)
    permutation_matches = sum(
        regular.encode_program(row)
        == permuted.encode_program(row, token_kind=kinds, token_value=values)
        for row in range(len(pack.ir.programs))
    )

    unknown_values = values.copy()
    opcode_kind = tables.token_kinds["opcode"]
    opcode_index = next(
        index for index, value in enumerate(kinds) if int(value) == opcode_kind
    )
    unknown_values[opcode_index] = max(tables.opcodes.values()) + 1
    unknown_rejected = False
    try:
        program_row = int(
            np.searchsorted(pack.catalog.program_offsets, opcode_index, side="right")
            - 1
        )
        permuted.encode_program(
            program_row, token_kind=kinds, token_value=unknown_values
        )
    except TransferExperimentError:
        unknown_rejected = True

    enum_failure = False
    changed_enum_pack = BoundSemanticPack(
        **{
            **pack.__dict__,
            "schema": type(pack.schema)(
                **{
                    **pack.schema.__dict__,
                    "enums": {**pack.schema.enums, "synthetic": {"unknown": 1}},
                }
            ),
        }
    )
    try:
        SymbolicProgramBinder(changed_enum_pack, spec)
    except TransferExperimentError:
        enum_failure = True

    return {
        "programs": len(pack.ir.programs),
        "content_pack_reorder_checkpoint_rebind_matches": rebind_matches,
        "content_pack_reorder_checkpoint_rebind_rate": rebind_matches
        / len(pack.ir.programs),
        "token_kind_and_opcode_id_permutation_matches": permutation_matches,
        "token_kind_and_opcode_id_permutation_rate": permutation_matches
        / len(pack.ir.programs),
        "unknown_opcode_fail_closed": unknown_rejected,
        "enum_domain_schema_change_fail_closed": enum_failure,
        "arbitrary_enum_id_rebinding_supported": False,
    }


def _normalized_ast(value: Any) -> Any:
    """Erase literal identity/value leaves while retaining typed AST shape."""

    if isinstance(value, Mapping):
        return {
            key: _normalized_ast(item)
            for key, item in sorted(value.items())
            if key
            not in {"semantic_key", "program_index", "definition_index", "op_name"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalized_ast(item) for item in value]
    return type(value).__name__


def holdout_audit(
    pack: BoundSemanticPack,
    spec: SemanticTransferSpec,
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    records = build_records(pack, spec, workload)
    symbols = spec.token_vocabulary
    training_primitives = {
        symbols[token]
        for record in records
        if not record.held_out
        for token in record.token_ids
    }
    training_fingerprints: dict[str, list[str]] = {}
    for record in records:
        if record.held_out:
            continue
        fingerprint = _sha256(_normalized_ast(pack.ir.programs[record.program_row]))
        training_fingerprints.setdefault(fingerprint, []).append(record.program_key)

    rows = []
    for record in records:
        if not record.held_out:
            continue
        primitives = {symbols[token] for token in record.token_ids}
        fingerprint = _sha256(_normalized_ast(pack.ir.programs[record.program_row]))
        duplicates = sorted(training_fingerprints.get(fingerprint, []))
        gaps = sorted(primitives - training_primitives)
        rows.append(
            {
                "program": record.program_key,
                "definition": record.definition_key,
                "deck": record.deck,
                "normalized_ast_fingerprint": fingerprint,
                "normalized_ast_training_duplicates": duplicates,
                "normalized_ast_novel": not duplicates,
                "symbolic_primitive_gaps": gaps,
                "full_symbolic_primitive_closure": not gaps,
            }
        )
    return {
        "normalization": (
            "mapping keys and list structure retained; semantic/program/definition "
            "identity and literal scalar values erased"
        ),
        "rows": rows,
        "normalized_ast_novel_programs": sum(
            row["normalized_ast_novel"] for row in rows
        ),
        "full_symbolic_primitive_closure_programs": sum(
            row["full_symbolic_primitive_closure"] for row in rows
        ),
        "programs": len(rows),
        "all_normalized_asts_novel": all(row["normalized_ast_novel"] for row in rows),
        "full_symbolic_primitive_closure": all(
            row["full_symbolic_primitive_closure"] for row in rows
        ),
    }


def run_arm(
    pack: BoundSemanticPack,
    workload: Mapping[str, Any],
    *,
    arm: str,
    model_seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    spec = build_spec(pack)
    semantic_control = arm if arm in SEMANTIC_CONTROLS else None
    records = build_records(pack, spec, workload, semantic_control=semantic_control)
    training = build_examples(records, workload, held_out=False)
    heldout = build_examples(records, workload, held_out=True)
    support = build_examples(records, workload, held_out=True, support=True)
    model = TransferPolicy(
        arm,
        token_count=len(spec.token_vocabulary),
        definition_count=len(spec.definition_symbols),
        opcode_count=len(spec.opcode_symbols),
        hidden_dim=int(workload["training"]["hidden_dim"]),
    )
    _train(
        model,
        tensorize(training),
        steps=int(workload["training"]["epochs"]),
        learning_rate=float(workload["training"]["learning_rate"]),
    )
    confidence = float(workload["evaluation"]["confidence_level"])
    in_domain, _ = _evaluate(model, training, confidence)
    zero_shot, zero_rows = _evaluate(model, heldout, confidence)

    adapted = TransferPolicy(
        arm,
        token_count=len(spec.token_vocabulary),
        definition_count=len(spec.definition_symbols),
        opcode_count=len(spec.opcode_symbols),
        hidden_dim=int(workload["training"]["hidden_dim"]),
    )
    adapted.load_state_dict(model.state_dict())
    _train(
        adapted,
        tensorize(support),
        steps=int(workload["training"]["limited_retraining_steps"]),
        learning_rate=float(workload["training"]["learning_rate"]),
    )
    limited, limited_rows = _evaluate(adapted, heldout, confidence)
    return {
        "arm": arm,
        "model_seed": model_seed,
        "input": {
            "semantic_program": model.uses_semantics,
            "opaque_card_def_identity": model.uses_identity,
        },
        "head": "structured_ragged" if model.structured else "legacy_fixed",
        "training_examples": len(training),
        "heldout_examples": len(heldout),
        "limited_retraining_examples": len(support),
        "in_domain": in_domain,
        "zero_shot": zero_shot,
        "limited_retraining": limited,
        "zero_shot_rows": zero_rows,
        "limited_retraining_rows": limited_rows,
        "performance": _performance(model, heldout, workload),
        "checkpoint_rebind": _checkpoint_rebind_control(
            model, pack, spec, workload, heldout, semantic_control
        ),
        "semantic_input_spec": spec.to_dict(),
        "semantic_input_spec_digest": spec.digest,
    }


def aggregate_arm(rows: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    outcomes = cluster_outcomes(rows, phase)
    summary = _wilson(sum(outcomes.values()), len(outcomes), 0.95)
    summary.update(
        {
            "independent_unit": "heldout_program_x_model_seed",
            "cluster_count": len(outcomes),
            "repeated_rows_excluded_from_uncertainty": True,
        }
    )
    return summary


def cluster_outcomes(
    rows: Sequence[Mapping[str, Any]], phase: str
) -> dict[tuple[int, str], bool]:
    outcomes: dict[tuple[int, str], set[bool]] = {}
    for run in rows:
        seed = int(run["model_seed"])
        for item in run[f"{phase}_rows"]:
            key = (seed, str(item["program"]))
            outcomes.setdefault(key, set()).add(bool(item["correct"]))
    unstable = [key for key, values in outcomes.items() if len(values) != 1]
    if unstable:
        raise TransferExperimentError(
            "candidate permutation or seat changed a cluster outcome: "
            f"{sorted(unstable)}"
        )
    return {key: next(iter(values)) for key, values in outcomes.items()}


def paired_cluster_contrast(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    phase: str = "zero_shot",
    bootstrap_seed: int = 21_499,
) -> dict[str, Any]:
    left_outcomes = cluster_outcomes(left, phase)
    right_outcomes = cluster_outcomes(right, phase)
    if set(left_outcomes) != set(right_outcomes):
        raise TransferExperimentError("paired arms do not share program×seed clusters")
    differences = np.asarray(
        [
            int(left_outcomes[key]) - int(right_outcomes[key])
            for key in sorted(left_outcomes)
        ],
        dtype=np.float64,
    )
    wins = int((differences > 0).sum())
    losses = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    discordant = wins + losses
    if discordant:
        tail = sum(
            math.comb(discordant, index) for index in range(min(wins, losses) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2 * tail)
    else:
        exact_p = 1.0
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.asarray(
        [
            differences[rng.integers(0, len(differences), len(differences))].mean()
            for _ in range(10_000)
        ]
    )
    return {
        "left_minus_right_accuracy": float(differences.mean()),
        "paired_cluster_bootstrap_ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "clusters": len(differences),
        "discordant_clusters": discordant,
        "two_sided_exact_sign_p": exact_p,
        "independent_unit": "heldout_program_x_model_seed",
    }


__all__ = [
    "ARMS",
    "SEMANTIC_CONTROLS",
    "SemanticTransferSpec",
    "SymbolicProgramBinder",
    "TransferExperimentError",
    "TransferPolicy",
    "aggregate_arm",
    "build_examples",
    "build_records",
    "build_spec",
    "cluster_outcomes",
    "holdout_audit",
    "load_workload",
    "permuted_runtime_projection",
    "paired_cluster_contrast",
    "robustness_controls",
    "run_arm",
    "tensorize",
    "validate_workload",
]
