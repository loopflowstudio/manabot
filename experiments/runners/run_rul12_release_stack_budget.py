#!/usr/bin/env python3
"""Measure and verify the independently bound RUL-12 workload receipt."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from etude import authored_match_parity, public_commitment_parity
from experiments.runners import run_rul9_played_workloads as rul9

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "experiments/contracts/rul-12-release-stack-budget-v1.json"
DEFAULT_OUT = ROOT / "experiments/data/rul-12-release-stack-budget-v1.json"
DEFAULT_REPORT = ROOT / "experiments/rul-12-release-stack-budget-v1.md"
EXPERIMENT_ID = "rul-12-release-stack-budget-v1"
SCHEMA_VERSION = 1

SOURCE_SINGLETONS = set(rul9.SOURCE_SINGLETONS) | {
    "etude/public_commitment_parity.py",
    "experiments/contracts/rul-12-release-stack-budget-v1.json",
    "experiments/runners/run_rul12_release_stack_budget.py",
    "manabot/belief/likelihood.py",
    "manabot/belief/tracker.py",
    "managym/possible_worlds.py",
    "scripts/verify-rul12-release-stack-budget",
}
WEBSOCKET_PACKAGES = (
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic-core",
    "anyio",
    "httpx",
    "uvicorn",
    "websockets",
)


class Rul12Error(rul9.Rul9Error):
    """RUL-12 evidence or a frozen input failed closed."""


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Rul12Error(f"cannot load RUL-12 contract: {error}") from error
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise Rul12Error("RUL-12 contract schema mismatch")
    if contract.get("experiment") != EXPERIMENT_ID:
        raise Rul12Error("RUL-12 contract experiment identity mismatch")
    frozen = json.loads(rul9.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    for key in (
        "training",
        "expected_inputs",
        "authority_fallback_counters",
        "training_fallback_counters",
        "budgets",
    ):
        if contract.get(key) != frozen[key]:
            raise Rul12Error(f"RUL-12 changed the frozen RUL-9 {key}")
    release = contract.get("release")
    if not isinstance(release, Mapping):
        raise Rul12Error("RUL-12 release cell is missing")
    for key in (
        "matchup",
        "seed",
        "surfaces",
        "warmups",
        "repetitions",
        "commands_per_game",
        "authority_receipt_sha256",
        "terminal_state_sha256",
    ):
        if release.get(key) != frozen["release"][key]:
            raise Rul12Error(f"RUL-12 changed the frozen release {key}")
    if release.get("latency_boundary") != (
        "TestClient WebSocket client send through accepted acknowledgment"
    ):
        raise Rul12Error("RUL-12 latency boundary drifted")
    if release.get("summary_cache") is not False:
        raise Rul12Error("RUL-12 cannot use a table-summary cache")
    if release.get("terminal_persistence") != ("synchronous and inside measured game"):
        raise Rul12Error("RUL-12 terminal persistence boundary drifted")
    current_proofs = contract.get("current_proofs")
    if not isinstance(current_proofs, Mapping):
        raise Rul12Error("RUL-12 current proof inputs are missing")
    if release.get("parity_receipt_sha256") != current_proofs.get(
        "authored_match_parity", {}
    ).get("file_sha256"):
        raise Rul12Error("RUL-12 authored parity identity drifted")
    if release.get("public_commitment_receipt_sha256") != current_proofs.get(
        "public_commitment_parity", {}
    ).get("file_sha256"):
        raise Rul12Error("RUL-12 public-commitment identity drifted")
    frozen_inputs = contract.get("frozen_inputs")
    if not isinstance(frozen_inputs, Mapping) or set(frozen_inputs) != {
        "rul9_measurement_origin",
        "rul9_derivation_receipt",
        "rul9_contract",
        "rul9_report",
    }:
        raise Rul12Error("RUL-12 frozen input inventory drifted")
    if any(entry.get("rerun") is not False for entry in frozen_inputs.values()):
        raise Rul12Error("RUL-12 must mark every RUL-9 input rerun=false")


def source_manifest() -> dict[str, Any]:
    relative_paths = set(SOURCE_SINGLETONS)
    for directory in rul9.SOURCE_DIRECTORIES:
        root = ROOT / directory
        if not root.is_dir():
            raise Rul12Error(f"source closure directory is missing: {directory}")
        relative_paths.update(
            path.relative_to(ROOT).as_posix()
            for path in root.rglob("*.rs")
            if path.is_file()
        )
    files = []
    for relative in sorted(relative_paths):
        path = ROOT / relative
        if not path.is_file():
            raise Rul12Error(f"source closure path is missing: {relative}")
        files.append({"path": relative, "sha256": rul9.sha256_file(path)})
    return {
        "algorithm": "relative-path-and-file-sha256-v1",
        "files": files,
        "sha256": rul9.sha256_bytes(rul9.canonical_json(files)),
    }


def runtime_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    identity = rul9.runtime_identity()
    identity["source"] = source_manifest()
    identity["binary"]["packages"] = {
        package: importlib.metadata.version(package) for package in WEBSOCKET_PACKAGES
    }
    identity["workload"].update(
        {
            "contract_sha256": rul9.sha256_bytes(rul9.canonical_json(contract)),
            "public_commitment_receipt_sha256": rul9.sha256_file(
                public_commitment_parity.RECEIPT_PATH
            ),
            "latency_boundary": contract["release"]["latency_boundary"],
            "summary_cache": contract["release"]["summary_cache"],
            "terminal_persistence": contract["release"]["terminal_persistence"],
        }
    )
    return identity


def verify_frozen_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in contract["frozen_inputs"].items():
        path = ROOT / expected["path"]
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise Rul12Error(f"frozen input {name} is unavailable: {error}") from error
        if rul9.sha256_bytes(raw) != expected["file_sha256"]:
            raise Rul12Error(f"frozen input {name} bytes changed")
        entry = deepcopy(expected)
        if "artifact_sha256" in expected:
            try:
                artifact = json.loads(raw)["artifact_sha256"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise Rul12Error(
                    f"frozen input {name} artifact identity is unavailable"
                ) from error
            if artifact != expected["artifact_sha256"]:
                raise Rul12Error(f"frozen input {name} artifact identity changed")
        observed[name] = entry
    return observed


def _checked_proof(
    contract: Mapping[str, Any], name: str
) -> tuple[dict[str, Any], str]:
    expected = contract["current_proofs"][name]
    path = ROOT / expected["path"]
    raw = path.read_bytes()
    digest = rul9.sha256_bytes(raw)
    if digest != expected["file_sha256"]:
        raise Rul12Error(f"checked {name} bytes changed")
    return json.loads(raw), digest


def _authored_projection(receipt: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "checked_receipt_sha256": digest,
        "summary": deepcopy(receipt["summary"]),
        "surfaces_sha256": rul9.sha256_bytes(rul9.canonical_json(receipt["surfaces"])),
        "stale_object_proof": deepcopy(receipt["stale_object_proof"]),
    }


def _public_projection(receipt: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "checked_receipt_sha256": digest,
        "summary": deepcopy(receipt["summary"]),
        "identity_stream_sha256": rul9.sha256_bytes(
            rul9.canonical_json(receipt["identity_stream"])
        ),
        "surface_identity_stream_sha256": {
            name: surface["identity_stream_sha256"]
            for name, surface in receipt["surfaces"].items()
        },
        "materialized_hypothesis": deepcopy(receipt["materialized_hypothesis"]),
        "atomic_negative_proof": deepcopy(receipt["atomic_negative_proof"]),
    }


def correctness_evidence(contract: Mapping[str, Any], *, rerun: bool) -> dict[str, Any]:
    authored_checked, authored_digest = _checked_proof(
        contract, "authored_match_parity"
    )
    public_checked, public_digest = _checked_proof(contract, "public_commitment_parity")
    if rerun:
        authored_match_parity.verify_receipt()
        public_commitment_parity.verify_receipt.cache_clear()
        public_commitment_parity.verify_receipt()
    evidence = {
        "authored_match_parity": _authored_projection(
            authored_checked, authored_digest
        ),
        "public_commitment_parity": _public_projection(public_checked, public_digest),
    }
    validate_correctness_evidence(evidence)
    return evidence


def validate_correctness_evidence(evidence: Mapping[str, Any]) -> None:
    authored = evidence.get("authored_match_parity", {})
    if authored.get("summary") != {
        "canonical_player_projections": 2,
        "checkpoints_per_surface": 133,
        "commands_per_surface": 132,
        "first_divergence": None,
        "ordered_transition_groups_per_surface": 132,
        "spectator_admitted": False,
        "viewer_projection_checks": 798,
    }:
        raise Rul12Error("authored-match correctness summary drifted")
    stale = authored.get("stale_object_proof", {})
    if stale.get("current_rejection") != {
        "code": "stale_object",
        "semantic_event_cursor_unchanged": True,
        "state_witness_unchanged": True,
    } or stale.get("retained_command_rejection") != {
        "code": "stale_revision",
        "semantic_event_cursor_unchanged": True,
        "state_witness_unchanged": True,
    }:
        raise Rul12Error("stale-reference proof drifted")
    public = evidence.get("public_commitment_parity", {})
    if public.get("summary") != {
        "commands": 132,
        "commitments": 62,
        "consumed_commitments": 62,
        "identity_stream_mismatches": 0,
        "negative_proof_mutations": 0,
        "rules_provider_gaps": 0,
        "tracker_records_per_viewer": 132,
        "unadmitted_commands": 70,
    }:
        raise Rul12Error("public-commitment correctness summary drifted")
    if len(set(public.get("surface_identity_stream_sha256", {}).values())) != 1:
        raise Rul12Error("public-commitment surfaces do not share one identity stream")
    materialized = public.get("materialized_hypothesis", {})
    if not all(
        materialized.get(name) is True
        for name in (
            "pre_command_decision_hidden_from_non_actor",
            "source_match_witness_unchanged",
            "source_semantic_event_cursor_unchanged",
            "source_viewer_observation_unchanged",
        )
    ):
        raise Rul12Error("materialized public-commitment privacy proof drifted")
    for proof in public.get("atomic_negative_proof", {}).values():
        if proof.get("rejection") != "RulesProviderGap" or not all(
            value is True for key, value in proof.items() if key != "rejection"
        ):
            raise Rul12Error("public-commitment atomic rejection proof drifted")


def _verify_contract_identity(
    identity: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    workload = identity.get("workload", {})
    expected = {
        "authority_receipt_sha256": contract["release"]["authority_receipt_sha256"],
        "parity_receipt_sha256": contract["release"]["parity_receipt_sha256"],
        "public_commitment_receipt_sha256": contract["release"][
            "public_commitment_receipt_sha256"
        ],
        "contract_sha256": rul9.sha256_bytes(rul9.canonical_json(contract)),
    }
    expected.update(contract["expected_inputs"])
    for name, value in expected.items():
        if workload.get(name) != value:
            raise Rul12Error(f"RUL-12 identity {name} drifted")
    if workload.get("selected_branch_driver") != contract["training"]["driver"]:
        raise Rul12Error("RUL-12 training driver identity drifted")
    if identity.get("binary", {}).get("packages", {}).keys() != set(WEBSOCKET_PACKAGES):
        raise Rul12Error("RUL-12 WebSocket package inventory drifted")


def build_receipt(contract: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    validate_contract(contract)
    frozen = verify_frozen_inputs(contract)
    identity_before = runtime_identity(contract)
    _verify_contract_identity(identity_before, contract)
    started = rul9.utc_now()
    release = rul9.run_release(contract)
    training = rul9.run_training(contract)
    correctness = correctness_evidence(contract, rerun=True)
    identity_after = runtime_identity(contract)
    if identity_after != identity_before:
        raise Rul12Error("source or binary identity changed during measurement")
    raw = {
        "release": release,
        "training": training,
        "correctness": correctness,
        "frozen_inputs": frozen,
    }
    summary = rul9.derive_summary(raw, contract)
    verdict = rul9.evaluate_verdict(summary, contract)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run": {
            "started_at": started,
            "completed_at": rul9.utc_now(),
            "contract_path": contract_path.relative_to(ROOT).as_posix(),
            "contract_file_sha256": rul9.sha256_file(contract_path),
            "contract_sha256": rul9.sha256_bytes(rul9.canonical_json(contract)),
        },
        "identity": {
            "before": identity_before,
            "after": identity_after,
            "stable": True,
        },
        "raw": raw,
        "summary": summary,
        "verdict": verdict,
    }
    receipt["artifact_sha256"] = rul9.artifact_hash(receipt)
    return receipt


def verify_receipt(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    check_current: bool,
    current_identity: Mapping[str, Any] | None = None,
    require_pass: bool = True,
) -> dict[str, Any]:
    validate_contract(contract)
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise Rul12Error("RUL-12 receipt schema mismatch")
    if receipt.get("experiment") != EXPERIMENT_ID:
        raise Rul12Error("RUL-12 receipt experiment mismatch")
    if receipt.get("artifact_sha256") != rul9.artifact_hash(receipt):
        raise Rul12Error("RUL-12 receipt artifact SHA-256 mismatch")
    if receipt.get("run", {}).get("contract_sha256") != rul9.sha256_bytes(
        rul9.canonical_json(contract)
    ):
        raise Rul12Error("RUL-12 receipt does not bind the contract")
    identities = receipt.get("identity", {})
    before = identities.get("before")
    after = identities.get("after")
    if before != after or identities.get("stable") is not True:
        raise Rul12Error("RUL-12 before/after source or binary identity drifted")
    if not isinstance(before, Mapping):
        raise Rul12Error("RUL-12 identity is missing")
    _verify_contract_identity(before, contract)
    frozen = verify_frozen_inputs(contract)
    if receipt.get("raw", {}).get("frozen_inputs") != frozen:
        raise Rul12Error("RUL-12 frozen-input receipt drifted")
    checked_correctness = correctness_evidence(contract, rerun=False)
    if receipt.get("raw", {}).get("correctness") != checked_correctness:
        raise Rul12Error("RUL-12 correctness evidence drifted")
    summary = rul9.derive_summary(receipt["raw"], contract)
    if receipt.get("summary") != summary:
        raise Rul12Error("RUL-12 summary does not rederive from raw evidence")
    rul9._require_zero_counters(summary)
    verdict = rul9.evaluate_verdict(summary, contract)
    if receipt.get("verdict") != verdict:
        raise Rul12Error("RUL-12 verdict does not recompute")
    if check_current:
        current = current_identity or runtime_identity(contract)
        if before != current:
            raise Rul12Error("RUL-12 source, binary, or workload identity is stale")
        current_correctness = correctness_evidence(contract, rerun=True)
        if current_correctness != checked_correctness:
            raise Rul12Error("RUL-12 current correctness proof drifted")
    if require_pass and verdict["overall"] != "pass":
        raise Rul12Error(f"RUL-12 product budgets missed: {verdict}")
    return verdict


def _mib(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def render_report(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    release = receipt["summary"]["release"]
    training = receipt["summary"]["training"]
    verdict = receipt["verdict"]
    identity = receipt["identity"]["before"]
    lines = [
        "# RUL-12 Release-Stack Budget Receipt",
        "",
        "## Result",
        "",
        f"Overall admission: **{verdict['overall'].upper()}**. The fixed 132-Command UR Lessons versus GW Allies trace was measured through the same TestClient client-send-to-accepted-ack boundary as RUL-9, with no summary cache and synchronous terminal replay persistence.",
        "",
        "| Surface | Command p50 | Command p95 | Steps/s | Games/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("live", "headless", "replay"):
        surface = release["surfaces"][name]
        lines.append(
            f"| {name} | {surface['command_ms']['p50']:.3f} ms | "
            f"{surface['command_ms']['p95']:.3f} ms | "
            f"{surface['steps_per_second']:.1f} | "
            f"{surface['games_per_second']:.3f} |"
        )
    live = release["surfaces"]["live"]
    lines.extend(
        [
            "",
            f"Live inner semantic Command p50/p95 was {live['inner_command_ms']['p50']:.3f}/{live['inner_command_ms']['p95']:.3f} ms. Release peak RSS was {_mib(release['peak_rss_bytes']):.1f} MiB across {release['rss_sample_count']} retained samples.",
            "",
            "## Training and capacity",
            "",
            f"The unchanged `full_clone/current_game_v1` 4x128 workload delivered {training['steps_per_second']:.3f} roots/s, {training['traversals_per_second']:.1f} traversals/s, and {training['games_per_second']:.4f} complete games/s. PUCT p95 was {training['puct_ms']['p95']:.3f} ms, Command p95 was {training['command_ms']['p95']:.3f} ms, and peak RSS was {_mib(training['peak_rss_bytes']):.1f} MiB.",
            "",
            f"The semantic catalog used {training['semantic']['catalog_active_tokens']} active tokens; maximum tokens per definition were {training['semantic']['tokens_per_definition']['max']:.0f}, maximum visible references were {training['semantic']['visible_object_references']['max']:.0f}, and every overflow, projection-failure, unadmitted-definition, native-mismatch, authority fallback, and training fallback count was zero.",
            "",
            "## Exactness and provenance",
            "",
            "Live, headless, and persisted replay retained one terminal witness and one ordered logical consequence hash. The current proof reran 798 viewer projections, spectator rejection, `stale_object` and `stale_revision` atomic rejection, 62 public commitments, a materialized revision-29 hypothesis, and zero `RulesProviderGap` without changing the checked historical RUL-11 receipt.",
            "",
            f"Artifact: `{receipt['artifact_sha256']}`. Source closure: `{identity['source']['sha256']}`. Native extension: `{identity['binary']['extension_sha256']}`. RUL-9 origin artifact `498df1eda031f1d6ea68f72f792dfd68195bb10eab884b006aa7772b586564da` was a byte-checked `rerun: false` input.",
            "",
            "WebSocket stack: "
            + ", ".join(
                f"{name} {version}"
                for name, version in identity["binary"]["packages"].items()
            )
            + ". Absolute performance remains single-host evidence.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "./scripts/verify-rul12-release-stack-budget",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _success_line(receipt: Mapping[str, Any]) -> str:
    release = receipt["summary"]["release"]
    training = receipt["summary"]["training"]
    live = release["surfaces"]["live"]
    return (
        "RUL12_RELEASE_STACK_OK "
        f"live_p50_ms={live['command_ms']['p50']:.3f} "
        f"live_p95_ms={live['command_ms']['p95']:.3f} "
        f"live_steps_s={live['steps_per_second']:.1f} "
        f"live_games_s={live['games_per_second']:.3f} "
        f"inner_p95_ms={live['inner_command_ms']['p95']:.3f} "
        f"headless_steps_s={release['surfaces']['headless']['steps_per_second']:.1f} "
        f"replay_steps_s={release['surfaces']['replay']['steps_per_second']:.1f} "
        f"training_roots_s={training['steps_per_second']:.3f} "
        f"training_traversals_s={training['traversals_per_second']:.1f} "
        f"training_games_s={training['games_per_second']:.4f} "
        f"release_peak_mib={_mib(release['peak_rss_bytes']):.1f} "
        f"training_peak_mib={_mib(training['peak_rss_bytes']):.1f} "
        f"catalog_tokens={training['semantic']['catalog_active_tokens']} "
        "fallbacks=0 overflow=0 provider_gaps=0"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = load_contract(args.contract)
        if args.verify:
            receipt = json.loads(args.out.read_text(encoding="utf-8"))
            verify_receipt(contract, receipt, check_current=True)
            print(_success_line(receipt))
            return 0
        receipt = build_receipt(contract, args.contract.resolve())
        rul9.atomic_write(
            args.out,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        rul9.atomic_write(args.report, render_report(receipt, contract).encode("utf-8"))
        verdict = verify_receipt(
            contract, receipt, check_current=True, require_pass=False
        )
        if verdict["overall"] != "pass":
            print(f"RUL-12 product budgets missed: {verdict}", file=sys.stderr)
            return 2
        print(_success_line(receipt))
        return 0
    except (OSError, json.JSONDecodeError, rul9.Rul9Error) as error:
        print(f"RUL-12 evidence failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
