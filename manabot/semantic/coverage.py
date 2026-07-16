"""Generate checked coverage and kernel-change evidence for curated content."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
import hashlib
from pathlib import Path
import re
from typing import Any, NoReturn

from .compiler import canonical_json, compile_source, load_json, pretty_json

COVERAGE_EVIDENCE_SCHEMA_VERSION = 1
COVERAGE_ARTIFACT_SCHEMA_VERSION = 1

SUPPORT_STATES = {"sanctioned_deviation", "supported", "unsupported"}
CHANGE_CLASSIFICATIONS = {"content_only", "kernel_changing"}
RESPONSE_KINDS = {"ir_redesign", "kernel_redesign", "stop_expansion"}


class CoverageEvidenceError(ValueError):
    """A coverage evidence document or its checked artifact is invalid."""


def _fail(context: str, message: str) -> NoReturn:
    raise CoverageEvidenceError(f"{context}: {message}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(context, "must be an object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(context, "must be an array")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(context, "must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(context, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(context, f"must be at least {minimum}")
    return value


def _keys(
    value: Mapping[str, Any],
    context: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        _fail(context, f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        _fail(context, f"unknown fields: {', '.join(sorted(unknown))}")


def _string_array(value: Any, context: str, *, nonempty: bool = False) -> list[str]:
    values = [_string(item, f"{context}[]") for item in _array(value, context)]
    duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
    if duplicates:
        _fail(context, f"duplicate values: {', '.join(duplicates)}")
    if nonempty and not values:
        _fail(context, "must not be empty")
    return values


def _unique_keys(items: Sequence[Mapping[str, Any]], context: str) -> None:
    keys = [_string(item.get("key"), f"{context}[].key") for item in items]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        _fail(context, f"duplicate keys: {', '.join(duplicates)}")


def _repository_path(root: Path, value: Any, context: str) -> tuple[str, Path]:
    text = _string(value, context)
    relative = Path(text)
    if relative.is_absolute():
        _fail(context, "must be repository-relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        _fail(context, "must remain inside the repository")
    if not resolved.is_file():
        _fail(context, f"file does not exist: {text}")
    return relative.as_posix(), resolved


def _collect_predicate_atoms(predicate: Mapping[str, Any], atoms: set[str]) -> None:
    atoms.add(f"predicate:{predicate['kind']}")
    for child in predicate.get("predicates", []):
        _collect_predicate_atoms(child, atoms)


def _collect_condition_atoms(condition: Mapping[str, Any], atoms: set[str]) -> None:
    atoms.add(f"condition:{condition['kind']}")
    predicate = condition.get("predicate")
    if isinstance(predicate, dict):
        _collect_predicate_atoms(predicate, atoms)


def _collect_instruction_atoms(
    instructions: Sequence[Mapping[str, Any]], atoms: set[str]
) -> None:
    for instruction in instructions:
        atoms.add(f"opcode:{instruction['op']}")
        predicate = instruction.get("predicate")
        if isinstance(predicate, dict):
            _collect_predicate_atoms(predicate, atoms)
        condition = instruction.get("condition")
        if isinstance(condition, dict):
            _collect_condition_atoms(condition, atoms)
        for field in ("then", "otherwise", "body"):
            nested = instruction.get(field)
            if isinstance(nested, list):
                _collect_instruction_atoms(nested, atoms)


def definition_semantic_atoms(definition: Mapping[str, Any]) -> list[str]:
    """Return the typed semantic vocabulary used by one source definition."""

    atoms: set[str] = set()
    characteristics = definition["characteristics"]
    if characteristics.get("token"):
        atoms.add("characteristic:token")
    atoms.update(
        f"keyword:{keyword}" for keyword in characteristics.get("keywords", [])
    )
    for program in definition["programs"]:
        atoms.add(f"program_kind:{program['kind']}")
        cost = program.get("cost", {})
        atoms.update(f"cost_feature:{field}" for field in cost)
        if isinstance(cost.get("affinity"), dict):
            _collect_predicate_atoms(cost["affinity"], atoms)
        for target in program.get("targets", []):
            selector = target["selector"]
            atoms.add(f"selector:{selector['kind']}")
        trigger = program.get("trigger")
        if trigger:
            atoms.add(f"trigger_event:{trigger['event']}")
            subject = trigger["subject"]
            atoms.add(f"trigger_subject:{subject['kind']}")
            if isinstance(subject.get("predicate"), dict):
                _collect_predicate_atoms(subject["predicate"], atoms)
            if isinstance(trigger.get("if"), dict):
                _collect_condition_atoms(trigger["if"], atoms)
        _collect_instruction_atoms(program["ops"], atoms)
    return sorted(atoms)


_RUST_TEST = re.compile(
    r"#\[test\]\s*(?P<attributes>(?:#\[[^\]]+\]\s*)*)"
    r"fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def _rust_tests(path: Path) -> set[str]:
    tests: set[str] = set()
    for match in _RUST_TEST.finditer(path.read_text(encoding="utf-8")):
        if "#[ignore" not in match.group("attributes"):
            tests.add(match.group("name"))
    return tests


def _validate_test_catalog(
    raw_tests: Any, root: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    items = [
        _object(item, "evidence.tests[]")
        for item in _array(raw_tests, "evidence.tests")
    ]
    _unique_keys(items, "evidence.tests")
    catalog: dict[str, dict[str, Any]] = {}
    artifact_tests: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for item in items:
        context = f"evidence.tests[{item.get('key', '?')}]"
        _keys(item, context, required={"key", "runner", "path", "identifier"})
        key = _string(item["key"], f"{context}.key")
        runner = _string(item["runner"], f"{context}.runner")
        if runner != "rust":
            _fail(f"{context}.runner", "schema v1 supports only exact Rust tests")
        relative, path = _repository_path(root, item["path"], f"{context}.path")
        identifier = _string(item["identifier"], f"{context}.identifier")
        if identifier in identifiers:
            _fail(context, f"duplicate test identifier: {identifier}")
        identifiers.add(identifier)
        if path.parent.name != "rules" or path.suffix != ".rs":
            _fail(f"{context}.path", "Rust evidence must be a managym rules test file")
        module_index = path.parent / "mod.rs"
        if not module_index.is_file() or not re.search(
            rf"^\s*pub\s+mod\s+{re.escape(path.stem)}\s*;",
            module_index.read_text(encoding="utf-8"),
            re.MULTILINE,
        ):
            _fail(
                f"{context}.path",
                f"{path.stem!r} is not included by the rules test module",
            )
        harness = path.parents[1] / "rules_tests.rs"
        if not harness.is_file() or not re.search(
            r"^\s*mod\s+rules\s*;",
            harness.read_text(encoding="utf-8"),
            re.MULTILINE,
        ):
            _fail(f"{context}.path", "rules module is not included by rules_tests")
        name = identifier.rsplit("::", 1)[-1]
        expected = f"rules::{path.stem}::{name}"
        if identifier != expected:
            _fail(
                f"{context}.identifier",
                f"expected exact integration test identifier {expected!r}",
            )
        if name not in _rust_tests(path):
            _fail(f"{context}.identifier", f"test is not declared in {relative}")
        record = {
            "identifier": identifier,
            "key": key,
            "path": relative,
            "runner": runner,
        }
        catalog[key] = record
        artifact_tests.append(record)
    return catalog, sorted(artifact_tests, key=lambda item: item["key"])


def _validate_rule_families(
    raw_families: Any,
    *,
    used_atoms: set[str],
    test_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    items = [
        _object(item, "evidence.rule_families[]")
        for item in _array(raw_families, "evidence.rule_families")
    ]
    _unique_keys(items, "evidence.rule_families")
    families: dict[str, dict[str, Any]] = {}
    atom_families: dict[str, set[str]] = defaultdict(set)
    for item in items:
        context = f"evidence.rule_families[{item.get('key', '?')}]"
        _keys(
            item,
            context,
            required={"key", "label", "cr_refs", "semantic_atoms", "tests"},
        )
        key = _string(item["key"], f"{context}.key")
        atoms = _string_array(
            item["semantic_atoms"], f"{context}.semantic_atoms", nonempty=True
        )
        unknown_atoms = sorted(set(atoms) - used_atoms)
        if unknown_atoms:
            _fail(
                context,
                f"semantic atoms are not used by the pack: {', '.join(unknown_atoms)}",
            )
        tests = _string_array(item["tests"], f"{context}.tests", nonempty=True)
        unknown_tests = sorted(set(tests) - test_catalog.keys())
        if unknown_tests:
            _fail(context, f"unknown tests: {', '.join(unknown_tests)}")
        family = {
            "cr_refs": sorted(_string_array(item["cr_refs"], f"{context}.cr_refs")),
            "key": key,
            "label": _string(item["label"], f"{context}.label"),
            "semantic_atoms": sorted(atoms),
            "tests": sorted(tests),
        }
        families[key] = family
        for atom in atoms:
            atom_families[atom].add(key)
    unmapped = sorted(used_atoms - atom_families.keys())
    if unmapped:
        _fail(
            "evidence.rule_families", f"unmapped semantic atoms: {', '.join(unmapped)}"
        )
    return families, atom_families


def _validate_card_evidence(
    raw_cards: Any,
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    test_catalog: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, dict[str, Any]]:
    items = [
        _object(item, "evidence.cards[]")
        for item in _array(raw_cards, "evidence.cards")
    ]
    keys = [
        _string(item.get("semantic_key"), "evidence.cards[].semantic_key")
        for item in items
    ]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        _fail("evidence.cards", f"duplicate semantic keys: {', '.join(duplicates)}")
    missing = sorted(definitions.keys() - set(keys))
    unknown = sorted(set(keys) - definitions.keys())
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        _fail("evidence.cards", "; ".join(details))

    cards: dict[str, dict[str, Any]] = {}
    for item in items:
        semantic_key = item["semantic_key"]
        context = f"evidence.cards[{semantic_key}]"
        _keys(
            item,
            context,
            required={"semantic_key", "support", "tests"},
            optional={"decision_refs", "reason"},
        )
        support = _string(item["support"], f"{context}.support")
        if support not in SUPPORT_STATES:
            _fail(f"{context}.support", f"unknown support state {support!r}")
        tests = _string_array(item["tests"], f"{context}.tests", nonempty=True)
        unknown_tests = sorted(set(tests) - test_catalog.keys())
        if unknown_tests:
            _fail(context, f"unknown tests: {', '.join(unknown_tests)}")
        decision_refs = _string_array(
            item.get("decision_refs", []), f"{context}.decision_refs"
        )
        for index, reference in enumerate(decision_refs):
            _repository_path(root, reference, f"{context}.decision_refs[{index}]")
        reason = item.get("reason")
        if support == "sanctioned_deviation" and not decision_refs:
            _fail(context, "sanctioned_deviation requires decision_refs")
        if support == "unsupported":
            _string(reason, f"{context}.reason")
        elif reason is not None:
            _fail(context, "reason is accepted only for unsupported cards")
        cards[semantic_key] = {
            "decision_refs": sorted(decision_refs),
            "reason": reason,
            "support": support,
            "tests": sorted(tests),
        }
    return cards


def _eligible_history_keys(
    definitions: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    eligible: set[str] = set()
    for key, definition in definitions.items():
        characteristics = definition["characteristics"]
        is_basic_land = "land" in characteristics[
            "types"
        ] and "basic" in characteristics.get("supertypes", [])
        if not characteristics.get("token") and not is_basic_land:
            eligible.add(key)
    return eligible


def _validate_history(
    raw_history: Any,
    *,
    eligible_keys: set[str],
    root: Path,
) -> list[dict[str, Any]]:
    items = [
        _object(item, "evidence.admission_history[]")
        for item in _array(raw_history, "evidence.admission_history")
    ]
    semantic_keys = [
        _string(item.get("semantic_key"), "evidence.admission_history[].semantic_key")
        for item in items
    ]
    duplicates = sorted(
        key for key, count in Counter(semantic_keys).items() if count > 1
    )
    if duplicates:
        _fail("evidence.admission_history", f"duplicate cards: {', '.join(duplicates)}")
    missing = sorted(eligible_keys - set(semantic_keys))
    unknown = sorted(set(semantic_keys) - eligible_keys)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing eligible cards: {', '.join(missing)}")
        if unknown:
            details.append(f"ineligible cards: {', '.join(unknown)}")
        _fail("evidence.admission_history", "; ".join(details))

    history: list[dict[str, Any]] = []
    for item in items:
        semantic_key = item["semantic_key"]
        context = f"evidence.admission_history[{semantic_key}]"
        _keys(
            item,
            context,
            required={
                "sequence",
                "semantic_key",
                "date",
                "change_id",
                "classification",
                "evidence_refs",
            },
            optional={"kernel_surfaces"},
        )
        sequence = _integer(item["sequence"], f"{context}.sequence", minimum=1)
        date_text = _string(item["date"], f"{context}.date")
        try:
            date.fromisoformat(date_text)
        except ValueError as error:
            raise CoverageEvidenceError(
                f"{context}.date: must be an ISO-8601 calendar date"
            ) from error
        classification = _string(item["classification"], f"{context}.classification")
        if classification not in CHANGE_CLASSIFICATIONS:
            _fail(
                f"{context}.classification",
                f"unknown classification {classification!r}",
            )
        evidence_refs = _string_array(
            item["evidence_refs"], f"{context}.evidence_refs", nonempty=True
        )
        for index, reference in enumerate(evidence_refs):
            _repository_path(root, reference, f"{context}.evidence_refs[{index}]")
        kernel_surfaces = _string_array(
            item.get("kernel_surfaces", []), f"{context}.kernel_surfaces"
        )
        if classification == "kernel_changing" and not kernel_surfaces:
            _fail(context, "kernel_changing requires kernel_surfaces")
        if classification == "content_only" and kernel_surfaces:
            _fail(context, "content_only cannot name kernel_surfaces")
        history.append(
            {
                "change_id": _string(item["change_id"], f"{context}.change_id"),
                "classification": classification,
                "date": date_text,
                "evidence_refs": sorted(evidence_refs),
                "kernel_surfaces": sorted(kernel_surfaces),
                "semantic_key": semantic_key,
                "sequence": sequence,
            }
        )
    history.sort(key=lambda item: item["sequence"])
    sequences = [item["sequence"] for item in history]
    if sequences != list(range(1, len(history) + 1)):
        _fail("evidence.admission_history", "sequences must be contiguous from 1")
    return history


def _validate_response(value: Any, *, root: Path) -> dict[str, Any] | None:
    if value is None:
        return None
    response = _object(value, "evidence.policy.response")
    _keys(
        response,
        "evidence.policy.response",
        required={
            "kind",
            "decision_document",
            "rationale",
            "acknowledged_through_sequence",
        },
    )
    kind = _string(response["kind"], "evidence.policy.response.kind")
    if kind not in RESPONSE_KINDS:
        _fail("evidence.policy.response.kind", f"unknown response kind {kind!r}")
    document, _ = _repository_path(
        root,
        response["decision_document"],
        "evidence.policy.response.decision_document",
    )
    return {
        "acknowledged_through_sequence": _integer(
            response["acknowledged_through_sequence"],
            "evidence.policy.response.acknowledged_through_sequence",
            minimum=1,
        ),
        "decision_document": document,
        "kind": kind,
        "rationale": _string(
            response["rationale"], "evidence.policy.response.rationale"
        ),
    }


def _validate_policy(value: Any, *, root: Path) -> dict[str, Any]:
    policy = _object(value, "evidence.policy")
    _keys(
        policy,
        "evidence.policy",
        required={
            "window_size",
            "minimum_denominator",
            "max_kernel_change_share",
            "response",
        },
    )
    threshold = _object(
        policy["max_kernel_change_share"],
        "evidence.policy.max_kernel_change_share",
    )
    _keys(
        threshold,
        "evidence.policy.max_kernel_change_share",
        required={"numerator", "denominator"},
    )
    numerator = _integer(
        threshold["numerator"],
        "evidence.policy.max_kernel_change_share.numerator",
        minimum=0,
    )
    denominator = _integer(
        threshold["denominator"],
        "evidence.policy.max_kernel_change_share.denominator",
        minimum=1,
    )
    if numerator > denominator:
        _fail("evidence.policy.max_kernel_change_share", "must be at most 1")
    window_size = _integer(
        policy["window_size"], "evidence.policy.window_size", minimum=1
    )
    minimum = _integer(
        policy["minimum_denominator"],
        "evidence.policy.minimum_denominator",
        minimum=1,
    )
    if minimum > window_size:
        _fail("evidence.policy.minimum_denominator", "cannot exceed window_size")
    return {
        "max_kernel_change_share": {
            "denominator": denominator,
            "numerator": numerator,
        },
        "minimum_denominator": minimum,
        "response": _validate_response(policy["response"], root=root),
        "window_size": window_size,
    }


def _rolling_gate(
    history: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    window = list(history[-policy["window_size"] :])
    counts = Counter(item["classification"] for item in window)
    denominator = len(window)
    content_only = counts["content_only"]
    kernel_changing = counts["kernel_changing"]
    latest_sequence = history[-1]["sequence"] if history else None
    violations: list[str] = []

    if denominator < policy["minimum_denominator"]:
        status = "insufficient_history"
        violations.append(
            f"rolling denominator {denominator} is below minimum "
            f"{policy['minimum_denominator']}"
        )
        breached = None
    else:
        threshold = policy["max_kernel_change_share"]
        breached = (
            kernel_changing * threshold["denominator"]
            > denominator * threshold["numerator"]
        )
        if not breached:
            status = "within_threshold"
        else:
            response = policy["response"]
            if response is None:
                status = "breached_unacknowledged"
                violations.append(
                    "kernel-change threshold breached without an IR/kernel redesign response"
                )
            elif response["acknowledged_through_sequence"] != latest_sequence:
                status = "breached_unacknowledged"
                violations.append(
                    "redesign response must acknowledge newest admission sequence "
                    f"{latest_sequence}"
                )
            else:
                status = "breached_acknowledged"

    result = {
        "counts": {
            "content_only": content_only,
            "denominator": denominator,
            "kernel_changing": kernel_changing,
        },
        "gate_status": status,
        "latest_sequence": latest_sequence,
        "ratios": {
            "content_only_share": {
                "denominator": denominator,
                "numerator": content_only,
            },
            "content_only_to_kernel_changing": f"{content_only}:{kernel_changing}",
            "kernel_changing_share": {
                "denominator": denominator,
                "numerator": kernel_changing,
            },
        },
        "response": deepcopy(policy["response"]),
        "threshold_breached": breached,
        "window_sequences": [item["sequence"] for item in window],
    }
    return result, violations


def generate_coverage_artifact(
    source: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Validate source/evidence and derive the deterministic coverage artifact."""

    source = deepcopy(_object(source, "source"))
    evidence = deepcopy(_object(evidence, "evidence"))
    compile_source(source)
    _keys(
        evidence,
        "evidence",
        required={
            "schema_version",
            "pack_key",
            "tests",
            "rule_families",
            "cards",
            "admission_history",
            "policy",
        },
    )
    if evidence["schema_version"] != COVERAGE_EVIDENCE_SCHEMA_VERSION:
        _fail(
            "evidence.schema_version",
            f"expected {COVERAGE_EVIDENCE_SCHEMA_VERSION}",
        )
    if evidence["pack_key"] != source["pack_key"]:
        _fail("evidence.pack_key", f"expected {source['pack_key']!r}")

    definitions = {
        definition["key"]: definition for definition in source["definitions"]
    }
    atoms_by_card = {
        key: definition_semantic_atoms(definition)
        for key, definition in definitions.items()
    }
    used_atoms = {atom for atoms in atoms_by_card.values() for atom in atoms}
    test_catalog, artifact_tests = _validate_test_catalog(evidence["tests"], root)
    families, atom_families = _validate_rule_families(
        evidence["rule_families"],
        used_atoms=used_atoms,
        test_catalog=test_catalog,
    )
    card_evidence = _validate_card_evidence(
        evidence["cards"],
        definitions=definitions,
        test_catalog=test_catalog,
        root=root,
    )
    history = _validate_history(
        evidence["admission_history"],
        eligible_keys=_eligible_history_keys(definitions),
        root=root,
    )
    policy = _validate_policy(evidence["policy"], root=root)

    deck_keys = {
        card["definition"] for deck in source["decks"] for card in deck["cards"]
    }
    referenced_keys = {
        instruction["definition_ref"]
        for definition in source["definitions"]
        for program in definition["programs"]
        for instruction in _walk_source_instructions(program["ops"])
        if "definition_ref" in instruction
    }

    artifact_cards: list[dict[str, Any]] = []
    family_cards: dict[str, set[str]] = defaultdict(set)
    for semantic_key in sorted(definitions):
        definition = definitions[semantic_key]
        atoms = atoms_by_card[semantic_key]
        card_families = sorted(
            {family for atom in atoms for family in atom_families[atom]}
        )
        for family in card_families:
            family_cards[family].add(semantic_key)
        annotation = card_evidence[semantic_key]
        roles = []
        if semantic_key in deck_keys:
            roles.append("deck")
        if semantic_key in referenced_keys:
            roles.append("referenced")
        card = {
            "admission_roles": roles,
            "decision_refs": annotation["decision_refs"],
            "registry_name": definition["registry_name"],
            "rule_families": card_families,
            "semantic_atoms": atoms,
            "semantic_key": semantic_key,
            "support": annotation["support"],
            "tests": annotation["tests"],
        }
        if annotation["reason"] is not None:
            card["reason"] = annotation["reason"]
        artifact_cards.append(card)

    artifact_families = []
    for key in sorted(families):
        family = families[key]
        cards = sorted(family_cards[key])
        if not cards:
            _fail(f"evidence.rule_families[{key}]", "matches no admitted cards")
        artifact_families.append(
            {**family, "cards": cards, "status": "supported_tested"}
        )

    unsupported = sorted(
        card["semantic_key"]
        for card in artifact_cards
        if card["support"] == "unsupported"
    )
    sanctioned = sorted(
        card["semantic_key"]
        for card in artifact_cards
        if card["support"] == "sanctioned_deviation"
    )
    rolling_gate, gate_violations = _rolling_gate(history, policy)
    coverage_violations = (
        [f"unsupported curated cards: {', '.join(unsupported)}"] if unsupported else []
    )
    referenced_tests = {
        test for card in card_evidence.values() for test in card["tests"]
    } | {test for family in families.values() for test in family["tests"]}
    unused_tests = sorted(test_catalog.keys() - referenced_tests)
    if unused_tests:
        _fail("evidence.tests", f"unreferenced test records: {', '.join(unused_tests)}")

    artifact = {
        "admission_history": history,
        "artifact_schema_version": COVERAGE_ARTIFACT_SCHEMA_VERSION,
        "cards": artifact_cards,
        "coverage_gaps": {
            "sanctioned_deviations": sanctioned,
            "unmapped_semantic_atoms": [],
            "unsupported_cards": unsupported,
            "untested_cards": [],
        },
        "coverage_status": "complete" if not coverage_violations else "gaps",
        "evidence_hash": hashlib.sha256(
            canonical_json(evidence).encode("utf-8")
        ).hexdigest(),
        "pack_key": source["pack_key"],
        "policy": {
            key: deepcopy(value) for key, value in policy.items() if key != "response"
        },
        "rolling_kernel_change_gate": rolling_gate,
        "rule_families": artifact_families,
        "source_hash": hashlib.sha256(
            canonical_json(source).encode("utf-8")
        ).hexdigest(),
        "summary": {
            "definition_count": len(definitions),
            "deck_definition_count": len(deck_keys),
            "referenced_definition_count": len(referenced_keys),
            "rule_family_count": len(families),
            "sanctioned_deviation_count": len(sanctioned),
            "test_count": len(test_catalog),
            "unsupported_count": len(unsupported),
        },
        "tests": artifact_tests,
        "violations": coverage_violations + gate_violations,
    }
    artifact["artifact_hash"] = hashlib.sha256(
        canonical_json(artifact).encode("utf-8")
    ).hexdigest()
    return artifact


def _walk_source_instructions(
    instructions: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    walked: list[Mapping[str, Any]] = []
    for instruction in instructions:
        walked.append(instruction)
        for field in ("then", "otherwise", "body"):
            nested = instruction.get(field)
            if isinstance(nested, list):
                walked.extend(_walk_source_instructions(nested))
    return walked


def enforce_coverage_gate(artifact: Mapping[str, Any]) -> None:
    """Fail when the artifact records incomplete coverage or policy evidence."""

    violations = artifact.get("violations", [])
    if violations:
        raise CoverageEvidenceError("coverage gate failed: " + "; ".join(violations))


def generate_paths(
    source_path: Path,
    evidence_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    return generate_coverage_artifact(
        load_json(source_path),
        load_json(evidence_path),
        root=root,
    )


def write_or_check(path: Path, artifact: Mapping[str, Any], *, check: bool) -> None:
    expected = pretty_json(artifact)
    if check:
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            raise CoverageEvidenceError(
                f"{path} is stale; run `uv run scripts/generate_coverage_gaps.py`"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    base = root / "content" / "semantic" / "v1"
    return (
        root,
        base / "two_deck.source.json",
        base / "coverage.evidence.json",
        base / "generated" / "coverage-gaps.json",
    )
