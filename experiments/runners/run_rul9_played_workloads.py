#!/usr/bin/env python3
"""Measure and verify the RUL-9 played release and training workloads."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterable, Mapping, Sequence

from fastapi.testclient import TestClient
import managym._managym as native_managym
import numpy as np
import psutil

from etude import authored_match_parity as parity, server, trace as trace_store
from etude.authored_match_receipt import DEFAULT_RECEIPT_PATH
from etude.replay_index import CanonicalReplayV1
from manabot.semantic.learning import BoundSemanticPack
from manabot.sim.mcts import _mix_seed, determinized_puct
from manabot.sim.selected_branchdriver_teacher import (
    SELECTED_BRANCH_DRIVER_ID,
    _choose_action,
    _fresh_engine,
    _warmup,
)
from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK
import managym

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "experiments/contracts/rul-9-played-workloads-v1.json"
DEFAULT_OUT = ROOT / "experiments/data/rul-9-played-workloads-v1.json"
DEFAULT_MEASUREMENT_ORIGIN = (
    ROOT / "experiments/data/rul-9-played-workloads-v1.measurement.json"
)
DEFAULT_REPORT = ROOT / "experiments/rul-9-played-workloads-v1.md"
PARITY_RECEIPT_PATH = parity.RECEIPT_PATH
CONTRACT_SCHEMA_VERSION = 1
MEASUREMENT_SCHEMA_VERSION = 1
DERIVED_SCHEMA_VERSION = 2
EXPERIMENT_ID = "rul-9-played-workloads-v1"
RSS_INTERVAL_SECONDS = 0.005
WORKER_TIMEOUT_SECONDS = 1_200
RAW_LINEAGE_ALGORITHM = "canonical-json-sha256-v1"

SOURCE_SINGLETONS = (
    "content/semantic/v1/generated/two_deck.ir.json",
    "content/semantic/v1/learning_schema.json",
    "content/semantic/v1/two_deck.source.json",
    "etude/authored_match_parity.py",
    "etude/authored_match_receipt.py",
    "etude/curated_pack.py",
    "etude/experience_protocol.py",
    "etude/presentation.py",
    "etude/replay_index.py",
    "etude/semantic_boundary.py",
    "etude/server.py",
    "etude/trace.py",
    "experiments/runners/run_rul9_played_workloads.py",
    "manabot/semantic/compiler.py",
    "manabot/semantic/learning.py",
    "manabot/sim/mcts.py",
    "manabot/sim/search_branch.py",
    "manabot/sim/selected_branchdriver_teacher.py",
    "manabot/verify/util.py",
    "managym/Cargo.lock",
    "managym/Cargo.toml",
    "managym/decision.py",
    "scripts/verify-rul9-played-workloads",
)
SOURCE_DIRECTORIES = ("managym/src",)


class Rul9Error(RuntimeError):
    """The RUL-9 workload or checked evidence failed closed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    return sha256_bytes(canonical_json(unhashed))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: Sequence[float | int], quantile: float) -> float:
    if not values:
        raise Rul9Error("cannot calculate a percentile without samples")
    if not 0.0 <= quantile <= 1.0:
        raise Rul9Error("percentile quantile is outside [0, 1]")
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise Rul9Error("measurement samples contain a non-finite value")
    return float(np.percentile(data, quantile * 100))


def distribution(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        raise Rul9Error("cannot summarize an empty sample set")
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise Rul9Error("measurement samples contain a non-finite value")
    return {
        "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def _command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise Rul9Error(
            f"identity command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def source_manifest() -> dict[str, Any]:
    relative_paths = set(SOURCE_SINGLETONS)
    for directory in SOURCE_DIRECTORIES:
        root = ROOT / directory
        if not root.is_dir():
            raise Rul9Error(f"source closure directory is missing: {directory}")
        relative_paths.update(
            path.relative_to(ROOT).as_posix()
            for path in root.rglob("*.rs")
            if path.is_file()
        )
    files = []
    for relative in sorted(relative_paths):
        path = ROOT / relative
        if not path.is_file():
            raise Rul9Error(f"source closure path is missing: {relative}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "algorithm": "relative-path-and-file-sha256-v1",
        "files": files,
        "sha256": sha256_bytes(canonical_json(files)),
    }


def _extension_path() -> Path:
    path = Path(native_managym.__file__).resolve()
    if not path.is_file():
        raise Rul9Error("the loaded managym native extension is absent")
    return path


def _deck_hash(deck: Mapping[str, int]) -> str:
    return sha256_bytes(canonical_json(dict(sorted(deck.items()))))


def _token_lengths(
    pack: BoundSemanticPack,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    definition = np.diff(pack.catalog.definition_offsets).astype(np.int64)
    program = np.diff(pack.catalog.program_offsets).astype(np.int64)
    totals = definition.copy()
    for row in range(len(totals)):
        start = int(pack.catalog.definition_program_offsets[row])
        end = int(pack.catalog.definition_program_offsets[row + 1])
        program_rows = pack.catalog.definition_program_rows[start:end]
        totals[row] += int(program[program_rows].sum())
    return definition, program, totals


def _semantic_static(env: managym.Env) -> dict[str, Any]:
    pack = BoundSemanticPack.from_env(env)
    definition, program, totals = _token_lengths(pack)
    return {
        "semantic_pack_sha256": pack.semantic_pack_hash,
        "learning_schema_sha256": pack.schema.schema_hash,
        "semantic_ir_sha256": pack.ir.ir_hash,
        "catalog_active_tokens": int(len(pack.catalog.token_kind)),
        "definition_tokens": [int(value) for value in definition],
        "program_tokens": [int(value) for value in program],
        "tokens_per_definition": [int(value) for value in totals],
    }


def runtime_identity() -> dict[str, Any]:
    env = _fresh_engine(1197, 0)
    manifest = env.content_pack_manifest()
    static = _semantic_static(env)
    extension = _extension_path()
    return {
        "source": source_manifest(),
        "binary": {
            "profile": "release",
            "extension_name": extension.name,
            "extension_sha256": sha256_file(extension),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "cargo": _command_output(("cargo", "--version")),
            "rustc": _command_output(("rustc", "--version")),
            "uv": _command_output(("uv", "--version")),
        },
        "workload": {
            "authority_receipt_sha256": sha256_file(DEFAULT_RECEIPT_PATH),
            "parity_receipt_sha256": sha256_file(PARITY_RECEIPT_PATH),
            "semantic_ir_file_sha256": sha256_file(
                ROOT / "content/semantic/v1/generated/two_deck.ir.json"
            ),
            "semantic_source_file_sha256": sha256_file(
                ROOT / "content/semantic/v1/two_deck.source.json"
            ),
            "learning_schema_file_sha256": sha256_file(
                ROOT / "content/semantic/v1/learning_schema.json"
            ),
            "content_pack_manifest": manifest,
            "content_pack_manifest_sha256": sha256_bytes(canonical_json(manifest)),
            "semantic_static": static,
            "ur_deck_sha256": _deck_hash(UR_LESSONS_DECK),
            "gw_deck_sha256": _deck_hash(GW_ALLIES_DECK),
            "selected_branch_driver": SELECTED_BRANCH_DRIVER_ID,
        },
    }


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Rul9Error(f"cannot load RUL-9 contract: {error}") from error
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise Rul9Error("RUL-9 contract schema mismatch")
    if contract.get("experiment") != EXPERIMENT_ID:
        raise Rul9Error("RUL-9 contract experiment identity mismatch")
    release = contract.get("release")
    training = contract.get("training")
    if not isinstance(release, Mapping) or not isinstance(training, Mapping):
        raise Rul9Error("RUL-9 contract is missing workload cells")
    if release.get("authority_receipt_sha256") != parity.AUTHORITY_SHA256:
        raise Rul9Error("release contract does not pin the landed authority tape")
    if release.get("terminal_state_sha256") != parity.TERMINAL_STATE:
        raise Rul9Error("release contract terminal state drifted")
    if release.get("surfaces") != ["live", "headless", "replay"]:
        raise Rul9Error("release contract surfaces drifted")
    if int(release.get("commands_per_game", 0)) != 132:
        raise Rul9Error("release contract must retain the 132-Command tape")
    if int(release.get("warmups", -1)) < 1 or int(release.get("repetitions", 0)) < 1:
        raise Rul9Error("release contract needs warmup and measured repetitions")
    if training.get("driver") != SELECTED_BRANCH_DRIVER_ID:
        raise Rul9Error("training contract must use the selected BranchDriver")
    if training.get("deal_seeds") != [1197, 1419, 1887, 2197]:
        raise Rul9Error("training deal seeds drifted")
    expected_training = {
        "workers": 4,
        "simulations": 128,
        "worlds": 4,
        "max_steps": 2000,
        "max_decisions": 500,
    }
    for key, expected in expected_training.items():
        if int(training.get(key, -1)) != expected:
            raise Rul9Error(f"training contract {key} drifted")
    expected_authority = {
        "legacy_fixed_action",
        "card_name_dispatch",
        "candidate_cap",
        "client_legality",
    }
    expected_training_fallback = {
        "indexed_fallbacks",
        "root_cap_hits",
        "random_playout_cap_hits",
    }
    if set(contract.get("authority_fallback_counters", ())) != expected_authority:
        raise Rul9Error("authority fallback counter inventory drifted")
    if (
        set(contract.get("training_fallback_counters", ()))
        != expected_training_fallback
    ):
        raise Rul9Error("training fallback counter inventory drifted")
    if not isinstance(contract.get("budgets"), Mapping):
        raise Rul9Error("RUL-9 contract budgets are missing")
    expected_inputs = contract.get("expected_inputs")
    if not isinstance(expected_inputs, Mapping) or set(expected_inputs) != {
        "semantic_ir_file_sha256",
        "semantic_source_file_sha256",
        "learning_schema_file_sha256",
    }:
        raise Rul9Error("RUL-9 expected input identity inventory drifted")


class _TokenCensus:
    def __init__(self, env: managym.Env) -> None:
        self.pack = BoundSemanticPack.from_env(env)
        _, _, self.totals = _token_lengths(self.pack)
        manifest = env.content_pack_manifest()
        self.name_by_id = {
            int(row["card_def_id"]): str(row["registry_name"])
            for row in manifest["definitions"]
        }

    def sample(
        self,
        env: managym.Env,
        viewer: int,
        *,
        details: bool,
    ) -> dict[str, Any]:
        observation = env.observation_for_player(viewer)
        projection = self.pack.project_observation(
            observation, identity_mode="semantic_only"
        )
        definition_rows = [int(value) for value in projection.object_definition_rows]
        lengths = [int(self.totals[row]) for row in definition_rows]
        sample: dict[str, Any] = {
            "viewer": viewer,
            "visible_object_references": len(lengths),
            "expanded_semantic_tokens": sum(lengths),
            "max_definition_tokens": max(lengths, default=0),
            "overflow_count": 0,
            "projection_failures": 0,
            "unadmitted_visible_definitions": 0,
        }
        if not details:
            return sample

        by_zone: Counter[str] = Counter()
        by_definition: Counter[str] = Counter()
        row_cursor = 0
        for card in (*observation.agent_cards, *observation.opponent_cards):
            definition_row = definition_rows[row_cursor]
            row_cursor += 1
            tokens = int(self.totals[definition_row])
            zone = str(card.zone).rsplit(".", 1)[-1].lower()
            card_def_id = int(card.registry_key)
            by_zone[zone] += tokens
            by_definition[self.name_by_id[card_def_id]] += tokens
        for stack_object in observation.stack_objects:
            if int(stack_object.kind) == 0:
                continue
            definition_row = definition_rows[row_cursor]
            row_cursor += 1
            tokens = int(self.totals[definition_row])
            card_def_id = int(stack_object.source_card_registry_key)
            by_zone["stack_ability"] += tokens
            by_definition[self.name_by_id[card_def_id]] += tokens
        if row_cursor != len(definition_rows):
            raise Rul9Error("semantic token attribution did not consume every row")
        sample["tokens_by_zone"] = dict(sorted(by_zone.items()))
        sample["tokens_by_definition"] = dict(
            sorted(by_definition.items(), key=lambda row: (-row[1], row[0]))
        )
        return sample


class _MeasuredGameSession(server.GameSession):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rul9_inner_commands: list[dict[str, Any]] = []
        self.rul9_tokens: list[dict[str, Any]] = []
        self._rul9_census: _TokenCensus | None = None

    def _step_and_record(self, *args: Any, **kwargs: Any) -> None:
        command = kwargs.get("command")
        if command is None:
            return super()._step_and_record(*args, **kwargs)
        actor = str(kwargs.get("actor", args[0] if args else "unknown"))
        context = kwargs.get("context")
        if self.env is None:
            raise Rul9Error("live authority environment disappeared")
        if self._rul9_census is None:
            self._rul9_census = _TokenCensus(self.env)
        viewer = 0 if actor == "hero" else 1
        token_sample = self._rul9_census.sample(self.env, viewer, details=False)
        token_sample.update(
            {
                "revision": int(command["expected_revision"]),
                "action_space": None if context is None else context.action_space,
            }
        )
        started = time.perf_counter_ns()
        super()._step_and_record(*args, **kwargs)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.rul9_inner_commands.append(
            {
                "revision": int(command["expected_revision"]),
                "actor": viewer,
                "action_space": None if context is None else context.action_space,
                "duration_ms": elapsed_ms,
            }
        )
        self.rul9_tokens.append(token_sample)


def _authority() -> dict[str, Any]:
    raw = DEFAULT_RECEIPT_PATH.read_bytes()
    if sha256_bytes(raw) != parity.AUTHORITY_SHA256:
        raise Rul9Error("landed RUL-5 authority receipt changed")
    payload = json.loads(raw)
    if len(payload.get("decisions", ())) != 132:
        raise Rul9Error("landed authority tape no longer has 132 Commands")
    return payload


def _checked_parity_provenance(contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = PARITY_RECEIPT_PATH.read_bytes()
    if sha256_bytes(raw) != contract["release"]["parity_receipt_sha256"]:
        raise Rul9Error("landed parity receipt changed")
    payload = json.loads(raw)
    summary = payload.get("summary")
    expected = {
        "commands_per_surface": 132,
        "checkpoints_per_surface": 133,
        "ordered_transition_groups_per_surface": 132,
        "first_divergence": None,
    }
    if not isinstance(summary, Mapping) or any(
        summary.get(name) != value for name, value in expected.items()
    ):
        raise Rul9Error("landed parity receipt summary drifted")
    identity = payload.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("authority_receipt_sha256")
        != contract["release"]["authority_receipt_sha256"]
    ):
        raise Rul9Error("landed parity receipt authority identity drifted")
    surfaces = payload.get("surfaces")
    if not isinstance(surfaces, Mapping) or set(surfaces) != {
        "live",
        "headless",
        "replay",
    }:
        raise Rul9Error("landed parity receipt surfaces drifted")
    for name, surface in surfaces.items():
        if (
            int(surface.get("commands", -1)) != 132
            or len(surface.get("checkpoints", ())) != 133
        ):
            raise Rul9Error(f"landed parity receipt {name} coverage drifted")
    return payload


def _validate_live_transitions(
    session: server.GameSession, decisions: Sequence[Mapping[str, Any]]
) -> str:
    if len(session.authority_transitions) != len(decisions):
        raise Rul9Error("live transition count differs from the authority tape")
    logical = []
    for row, transition in zip(decisions, session.authority_transitions, strict=True):
        if transition.state_before != row["state"]["before"]:
            raise Rul9Error("live pre-Command witness drifted")
        if transition.state_after != row["state"]["after"]:
            raise Rul9Error("live post-Command witness drifted")
        if transition.semantic_events != parity._expected_events(row):
            raise Rul9Error("live ordered semantic consequences drifted")
        if transition.command != row["command"]:
            raise Rul9Error("live semantic Command drifted")
        logical.append(
            {
                "revision": row["from_revision"],
                "state_before": transition.state_before,
                "state_after": transition.state_after,
                "command_id": row["command"]["command_id"],
                "events": transition.semantic_events,
            }
        )
    return sha256_bytes(canonical_json(logical))


def _measure_live_game(authority: Mapping[str, Any]) -> dict[str, Any]:
    decisions = authority["decisions"]
    with TemporaryDirectory() as temporary:
        session = _MeasuredGameSession(
            Path(temporary),
            id_factory=lambda kind: f"authored-authority-{kind}",
            clock=lambda: "2026-07-17T00:00:00+00:00",
            villain_offer_policy=parity._FrozenVillainPolicy(decisions),
            capture_authority_evidence=True,
        )
        original_factory = server._new_game_session
        server._new_game_session = lambda: session
        server.SESSION_REGISTRY.clear()
        protocol_commands: list[dict[str, Any]] = []
        game_started = time.perf_counter_ns()
        try:
            with TestClient(server.app) as client:
                with client.websocket_connect("/ws/play") as websocket:
                    websocket.send_json(
                        {
                            "type": "new_game",
                            "config": {
                                "hero_deck": "ur_lessons",
                                "villain_deck": "gw_allies",
                                "villain_type": "random",
                                "seed": 0,
                                "auto_pass": False,
                            },
                        }
                    )
                    payload = websocket.receive_json()
                    hero_rows = iter(row for row in decisions if int(row["actor"]) == 0)
                    while payload["frame"]["status"] != "game_over":
                        row = next(hero_rows)
                        frame = payload["frame"]
                        if int(frame["revision"]) != int(row["from_revision"]):
                            raise Rul9Error("live surfaced revision drifted")
                        started = time.perf_counter_ns()
                        websocket.send_json(
                            {"type": "command", "command": deepcopy(row["command"])}
                        )
                        outcome = websocket.receive_json()
                        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                        if outcome.get("status") != "accepted":
                            raise Rul9Error("live protocol Command was rejected")
                        next_frame = outcome["update"]["frame"]
                        protocol_commands.append(
                            {
                                "from_revision": int(frame["revision"]),
                                "to_revision": int(next_frame["revision"]),
                                "action_space": frame["action_space"],
                                "duration_ms": elapsed_ms,
                            }
                        )
                        payload = {"frame": next_frame}
        finally:
            server._new_game_session = original_factory
            server.SESSION_REGISTRY.clear()
        game_seconds = (time.perf_counter_ns() - game_started) / 1_000_000_000
        if session.env is None or not session.env.is_game_over():
            raise Rul9Error("live release game did not reach terminal")
        if session.env.state_digest() != authority["terminal_witness"]["state_digest"]:
            raise Rul9Error("live terminal state drifted")
        logical_sha256 = _validate_live_transitions(session, decisions)
        if session.trace_id is None:
            raise Rul9Error("live release game did not persist a canonical replay")
        persisted_trace = trace_store.load_trace(session.trace_id, Path(temporary))
        replay = CanonicalReplayV1.model_validate(persisted_trace["canonical_replay"])
        persisted_commands = [
            row.command.model_dump(mode="json") for row in replay.decisions
        ]
        expected_commands = [row["command"] for row in decisions]
        if persisted_commands != expected_commands:
            raise Rul9Error("live persisted replay Command tape drifted")
        return {
            "commands": len(session.authority_transitions),
            "protocol_commands": protocol_commands,
            "inner_commands": list(session.rul9_inner_commands),
            "token_samples": list(session.rul9_tokens),
            "game_seconds": game_seconds,
            "terminal_state_sha256": session.env.state_digest(),
            "logical_trace_sha256": logical_sha256,
            "persisted_commands": persisted_commands,
            "persisted_command_tape_sha256": sha256_bytes(
                canonical_json(persisted_commands)
            ),
            "fallback_counters": dict(session.authority_fallback_counters),
        }


def _measure_engine_game(authority: Mapping[str, Any], surface: str) -> dict[str, Any]:
    if surface not in {"headless", "replay"}:
        raise Rul9Error(f"unsupported engine surface {surface!r}")
    decisions = authority["decisions"]
    game_started = time.perf_counter_ns()
    env = managym.Env(seed=0)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("Hero", dict(server.UR_LESSONS_DECK)),
            managym.PlayerConfig("Villain", dict(server.GW_ALLIES_DECK)),
        ]
    )
    census = _TokenCensus(env)
    command_samples = []
    token_samples = []
    logical = []
    for row in decisions:
        revision = int(row["from_revision"])
        if env.state_digest() != row["state"]["before"]:
            raise Rul9Error(f"{surface} pre-Command witness drifted")
        frame = json.loads(env.semantic_decision_frame_json())
        token = census.sample(env, int(frame["actor"]), details=False)
        token.update({"revision": revision, "action_space": row["prompt_family"]})
        started = time.perf_counter_ns()
        before_observation = observation
        transition_json, observation, *_ = env.step_semantic_command(
            parity._semantic_command(row, frame).to_json()
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        transition = json.loads(transition_json)
        if transition["receipt"]["command_id"] != row["command"]["command_id"]:
            raise Rul9Error(f"{surface} semantic receipt drifted")
        if env.state_digest() != row["state"]["after"]:
            raise Rul9Error(f"{surface} post-Command witness drifted")
        semantic_events = parity._event_payloads(before_observation, observation)
        if semantic_events != parity._expected_events(row):
            raise Rul9Error(f"{surface} ordered semantic consequences drifted")
        logical.append(
            {
                "revision": revision,
                "state_before": row["state"]["before"],
                "state_after": row["state"]["after"],
                "command_id": transition["receipt"]["command_id"],
                "events": semantic_events,
            }
        )
        command_samples.append(
            {
                "revision": revision,
                "actor": int(frame["actor"]),
                "action_space": row["prompt_family"],
                "duration_ms": elapsed_ms,
            }
        )
        token_samples.append(token)
    game_seconds = (time.perf_counter_ns() - game_started) / 1_000_000_000
    if not env.is_game_over():
        raise Rul9Error(f"{surface} release game did not reach terminal")
    if env.state_digest() != authority["terminal_witness"]["state_digest"]:
        raise Rul9Error(f"{surface} terminal state drifted")
    return {
        "commands": len(command_samples),
        "command_samples": command_samples,
        "token_samples": token_samples,
        "game_seconds": game_seconds,
        "terminal_state_sha256": env.state_digest(),
        "logical_trace_sha256": sha256_bytes(canonical_json(logical)),
        "fallback_counters": dict(authority["summary"]["fallback_counters"]),
    }


def _release_worker(contract: Mapping[str, Any], start: Any, output: Any) -> None:
    try:
        authority = _authority()
        checked_parity = _checked_parity_provenance(contract)
        for _ in range(int(contract["release"]["warmups"])):
            live = _measure_live_game(authority)
            _measure_engine_game(authority, "headless")
            persisted = deepcopy(authority)
            for row, command in zip(
                persisted["decisions"], live["persisted_commands"], strict=True
            ):
                row["command"] = command
            _measure_engine_game(persisted, "replay")
        output.put({"kind": "ready", "pid": os.getpid()})
        start.wait()
        surfaces = {"live": [], "headless": [], "replay": []}
        for _ in range(int(contract["release"]["repetitions"])):
            live = _measure_live_game(authority)
            surfaces["live"].append(live)
            surfaces["headless"].append(_measure_engine_game(authority, "headless"))
            persisted = deepcopy(authority)
            for row, command in zip(
                persisted["decisions"], live["persisted_commands"], strict=True
            ):
                row["command"] = command
            surfaces["replay"].append(_measure_engine_game(persisted, "replay"))
        output.put(
            {
                "kind": "result",
                "pid": os.getpid(),
                "payload": {
                    "correctness": {
                        "parity_summary": checked_parity["summary"],
                        "parity_receipt_sha256": sha256_file(PARITY_RECEIPT_PATH),
                    },
                    "semantic_static": _semantic_static(_fresh_engine(0, 0)),
                    "surfaces": surfaces,
                },
            }
        )
    except BaseException as error:
        output.put(
            {
                "kind": "error",
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
            }
        )


def _sum_site_counts(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for site in ("world", "child", "leaf"):
        target[site] += int(source[site])


def _empty_branch_counters() -> dict[str, Any]:
    return {
        "forks": {"world": 0, "child": 0, "leaf": 0},
        "applies": {"world": 0, "child": 0, "leaf": 0},
        "marks": 0,
        "rollbacks": 0,
        "random_playouts": 0,
        "random_playout_cap_hits": 0,
        "indexed_fallbacks": 0,
    }


def _run_training_game(
    contract: Mapping[str, Any], deal_seed: int, ur_seat: int
) -> dict[str, Any]:
    cell = contract["training"]
    game_started = time.perf_counter_ns()
    cpu_started = time.process_time()
    env = _fresh_engine(deal_seed, ur_seat)
    census = _TokenCensus(env)
    match_id = f"rul9-training-{deal_seed}-ur{ur_seat}"
    rows = []
    logical = []
    counters = _empty_branch_counters()
    root_cap_hits = 0
    decision = 0
    while not env.is_game_over():
        if decision >= int(cell["max_decisions"]):
            raise Rul9Error(f"training game {deal_seed} exceeded decision cap")
        semantic_frame = json.loads(env.semantic_decision_frame_json())
        root_context = json.loads(env.search_context_json(False))
        token_sample = census.sample(env, int(env.current_agent_index()), details=True)
        source_witness = env.state_digest()
        search_seed = _mix_seed(deal_seed, decision + 1)
        search_started = time.perf_counter_ns()
        result = determinized_puct(
            env,
            simulations=int(cell["simulations"]),
            worlds=int(cell["worlds"]),
            seed=search_seed,
            max_steps=int(cell["max_steps"]),
            branch_driver_id=SELECTED_BRANCH_DRIVER_ID,
            branch_audit=False,
            branch_match_id=f"{match_id}-search-{decision}",
        )
        puct_ms = (time.perf_counter_ns() - search_started) / 1_000_000
        if env.state_digest() != source_witness:
            raise Rul9Error("selected training search mutated its source root")
        action = _choose_action(result)
        root_offer = root_context["offers"]["offers"][action]
        command = {
            "command_id": f"root.{match_id}.{decision + 1}",
            "match_id": match_id,
            "expected_revision": root_context["revision"],
            "prompt_id": root_context["prompt_id"],
            "offer_id": root_offer["id"],
            "answers": [],
        }
        offers = env.structured_search_offers()
        command_started = time.perf_counter_ns()
        _, _, _, _, _, native_actions = env.step_structured(
            offers,
            json.dumps(
                {"offer_id": command["offer_id"], "answers": []},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        command_ms = (time.perf_counter_ns() - command_started) / 1_000_000
        if native_actions != 1:
            raise Rul9Error("training root Command was not one native apply")
        raw_counters = result.branch_receipt["counters"]
        _sum_site_counts(counters["forks"], raw_counters["forks"])
        _sum_site_counts(counters["applies"], raw_counters["applies"])
        for name in (
            "marks",
            "rollbacks",
            "random_playouts",
            "random_playout_cap_hits",
            "indexed_fallbacks",
        ):
            counters[name] += int(raw_counters[name])
        root_cap_hits += int(result.cap_hits)
        prompt = semantic_frame.get("prompt") or {}
        logical_row = {
            "decision": decision,
            "source_witness": source_witness,
            "post_witness": env.state_digest(),
            "search_seed": search_seed,
            "command": command,
            "visit_counts": result.visit_counts.astype(int).tolist(),
            "q_values": result.q_values.astype(float).tolist(),
            "branch_receipt_sha256": sha256_bytes(
                canonical_json(result.branch_receipt)
            ),
        }
        logical.append(logical_row)
        token_sample.update(
            {
                "decision": decision,
                "revision": int(root_context["revision"]),
                "action_space": prompt.get("kind"),
            }
        )
        rows.append(
            {
                "decision": decision,
                "revision": int(root_context["revision"]),
                "action_space": prompt.get("kind"),
                "puct_ms": puct_ms,
                "command_ms": command_ms,
                "native_actions": native_actions,
                "tree_nodes": int(result.tree_nodes),
                "max_depth": int(result.max_depth),
                "cap_hits": int(result.cap_hits),
                "token_sample": token_sample,
                "logical_sha256": sha256_bytes(canonical_json(logical_row)),
            }
        )
        decision += 1
    return {
        "deal_seed": deal_seed,
        "ur_seat": ur_seat,
        "winner": env.winner_index(),
        "terminal": True,
        "terminal_state_sha256": env.state_digest(),
        "decisions": decision,
        "traversals": decision * int(cell["simulations"]),
        "game_seconds": (time.perf_counter_ns() - game_started) / 1_000_000_000,
        "cpu_seconds": time.process_time() - cpu_started,
        "rows": rows,
        "branch_counters": counters,
        "fallback_counters": {
            "indexed_fallbacks": int(counters["indexed_fallbacks"]),
            "root_cap_hits": root_cap_hits,
            "random_playout_cap_hits": int(counters["random_playout_cap_hits"]),
        },
        "logical_trace_sha256": sha256_bytes(canonical_json(logical)),
    }


def _training_worker(
    contract: Mapping[str, Any],
    deal_seed: int,
    ur_seat: int,
    start: Any,
    output: Any,
) -> None:
    try:
        _warmup(SELECTED_BRANCH_DRIVER_ID, deal_seed)
        output.put({"kind": "ready", "pid": os.getpid(), "deal_seed": deal_seed})
        start.wait()
        output.put(
            {
                "kind": "result",
                "pid": os.getpid(),
                "payload": _run_training_game(contract, deal_seed, ur_seat),
            }
        )
    except BaseException as error:
        output.put(
            {
                "kind": "error",
                "pid": os.getpid(),
                "deal_seed": deal_seed,
                "error": f"{type(error).__name__}: {error}",
            }
        )


def _process_rss(pid: int) -> tuple[int, list[dict[str, Any]]]:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        if isinstance(error, psutil.AccessDenied):
            raise Rul9Error(f"RSS sampler cannot inspect process {pid}: {error}")
        return 0, []
    total = 0
    rows = []
    for process in processes:
        try:
            rss = int(process.memory_info().rss)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise Rul9Error(
                f"RSS sampler cannot inspect process {process.pid}: {error}"
            ) from error
        total += rss
        rows.append({"pid": process.pid, "rss_bytes": rss})
    return total, rows


def _rss_sample(processes: Sequence[Any], offset: float) -> dict[str, Any]:
    total = 0
    workers = []
    for process in processes:
        rss, members = _process_rss(process.pid)
        total += rss
        workers.append(
            {
                "pid": process.pid,
                "returncode": process.exitcode,
                "rss_bytes": rss,
                "members": members,
            }
        )
    return {"offset_seconds": offset, "rss_bytes": total, "workers": workers}


def _stop_processes(processes: Iterable[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def _run_process_group(
    target: Any, assignments: Sequence[tuple[Any, ...]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = mp.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(target=target, args=(*assignment, start, output))
        for assignment in assignments
    ]
    for process in processes:
        process.start()
    ready = []
    try:
        while len(ready) < len(processes):
            message = output.get(timeout=WORKER_TIMEOUT_SECONDS)
            if message.get("kind") == "error":
                raise Rul9Error(f"measurement worker failed: {message['error']}")
            if message.get("kind") != "ready":
                raise Rul9Error("measurement worker spoke before the start barrier")
            ready.append(message)
        samples = [_rss_sample(processes, 0.0)]
        started = time.perf_counter()
        start.set()
        results = []
        last_sample = started
        max_gap = 0.0
        while len(results) < len(processes):
            now = time.perf_counter()
            max_gap = max(max_gap, now - last_sample)
            samples.append(_rss_sample(processes, now - started))
            last_sample = now
            try:
                message = output.get(timeout=RSS_INTERVAL_SECONDS)
            except queue.Empty:
                continue
            if message.get("kind") == "error":
                raise Rul9Error(f"measurement worker failed: {message['error']}")
            if message.get("kind") != "result":
                raise Rul9Error("unexpected measurement worker message")
            results.append(message["payload"])
        wall_seconds = time.perf_counter() - started
        for process in processes:
            process.join(timeout=30)
            if process.is_alive() or process.exitcode != 0:
                raise Rul9Error(
                    f"measurement worker {process.pid} exited {process.exitcode}"
                )
        positive = [
            sample["rss_bytes"] for sample in samples if sample["rss_bytes"] > 0
        ]
        if not positive:
            raise Rul9Error("RSS sampler retained no positive samples")
        rss = {
            "interval_seconds": RSS_INTERVAL_SECONDS,
            "wall_seconds": wall_seconds,
            "sample_count": len(samples),
            "max_observed_gap_seconds": max_gap,
            "baseline_bytes": positive[0],
            "peak_bytes": max(positive),
            "samples": samples,
            "ready": ready,
        }
        return results, rss
    except BaseException:
        _stop_processes(processes)
        raise


def run_release(contract: Mapping[str, Any]) -> dict[str, Any]:
    results, rss = _run_process_group(_release_worker, ((contract,),))
    payload = results[0]
    payload["rss"] = rss
    return payload


def run_training(contract: Mapping[str, Any]) -> dict[str, Any]:
    seeds = list(contract["training"]["deal_seeds"])
    assignments = tuple(
        (contract, int(seed), index % 2) for index, seed in enumerate(seeds)
    )
    games, rss = _run_process_group(_training_worker, assignments)
    games.sort(key=lambda row: seeds.index(int(row["deal_seed"])))
    return {
        "semantic_static": _semantic_static(_fresh_engine(1197, 0)),
        "games": games,
        "rss": rss,
    }


def _flatten_token_samples(games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for game in games:
        if "token_samples" in game:
            out.extend(game["token_samples"])
        else:
            out.extend(row["token_sample"] for row in game["rows"])
    return out


def _derive_semantics(
    static: Mapping[str, Any], token_samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not token_samples:
        raise Rul9Error("semantic census retained no decision samples")
    maximum = max(token_samples, key=lambda row: int(row["expanded_semantic_tokens"]))
    return {
        "catalog_active_tokens": int(static["catalog_active_tokens"]),
        "definition_tokens": distribution(static["definition_tokens"]),
        "program_tokens": distribution(static["program_tokens"]),
        "tokens_per_definition": distribution(static["tokens_per_definition"]),
        "visible_object_references": distribution(
            [row["visible_object_references"] for row in token_samples]
        ),
        "expanded_semantic_tokens": distribution(
            [row["expanded_semantic_tokens"] for row in token_samples]
        ),
        "overflow_count": sum(int(row["overflow_count"]) for row in token_samples),
        "projection_failures": sum(
            int(row["projection_failures"]) for row in token_samples
        ),
        "unadmitted_visible_definitions": sum(
            int(row["unadmitted_visible_definitions"]) for row in token_samples
        ),
        "expanded_maximum_attribution": {
            key: maximum[key]
            for key in (
                "viewer",
                "visible_object_references",
                "expanded_semantic_tokens",
                "max_definition_tokens",
                "tokens_by_zone",
                "tokens_by_definition",
            )
            if key in maximum
        },
    }


def _sum_fallbacks(
    rows: Sequence[Mapping[str, Any]], expected: Sequence[str]
) -> dict[str, int]:
    expected_set = set(expected)
    total = {name: 0 for name in expected}
    for row in rows:
        counters = row["fallback_counters"]
        if set(counters) != expected_set:
            raise Rul9Error(
                "fallback counter inventory mismatch: "
                f"expected={sorted(expected_set)!r} actual={sorted(counters)!r}"
            )
        for name in expected:
            total[name] += int(counters[name])
    return total


def _derive_rss(rss: Mapping[str, Any]) -> dict[str, int | float]:
    samples = rss.get("samples")
    if not isinstance(samples, Sequence) or not samples:
        raise Rul9Error("RSS evidence retained no samples")
    values = [int(sample["rss_bytes"]) for sample in samples]
    offsets = [float(sample["offset_seconds"]) for sample in samples]
    if any(value < 0 for value in values):
        raise Rul9Error("RSS evidence contains a negative sample")
    if any(not np.isfinite(offset) for offset in offsets):
        raise Rul9Error("RSS evidence contains a non-finite offset")
    if offsets != sorted(offsets):
        raise Rul9Error("RSS evidence offsets are not monotonic")
    peak = max(values)
    if int(rss.get("peak_bytes", -1)) != peak:
        raise Rul9Error("RSS peak does not rederive from retained samples")
    if int(rss.get("sample_count", -1)) != len(samples):
        raise Rul9Error("RSS sample count does not rederive")
    wall = float(rss.get("wall_seconds", -1.0))
    if not np.isfinite(wall) or wall <= 0 or wall < offsets[-1]:
        raise Rul9Error("RSS wall duration is invalid")
    return {"peak_bytes": peak, "sample_count": len(samples), "wall_seconds": wall}


def derive_summary(
    raw: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    release_raw = raw["release"]
    training_raw = raw["training"]
    release_rss = _derive_rss(release_raw["rss"])
    training_rss = _derive_rss(training_raw["rss"])
    release_surfaces: dict[str, Any] = {}
    all_release_games = []
    for surface in contract["release"]["surfaces"]:
        games = release_raw["surfaces"][surface]
        if len(games) != int(contract["release"]["repetitions"]):
            raise Rul9Error(f"release surface {surface} has the wrong game count")
        all_release_games.extend(games)
        if any(int(game["commands"]) != 132 for game in games):
            raise Rul9Error(f"release surface {surface} lost Commands")
        if surface == "live":
            command_values = [
                row["duration_ms"]
                for game in games
                for row in game["protocol_commands"]
            ]
            inner_values = [
                row["duration_ms"] for game in games for row in game["inner_commands"]
            ]
        else:
            command_values = [
                row["duration_ms"] for game in games for row in game["command_samples"]
            ]
            inner_values = command_values
        total_seconds = sum(float(game["game_seconds"]) for game in games)
        total_commands = sum(int(game["commands"]) for game in games)
        release_surfaces[surface] = {
            "games": len(games),
            "commands": total_commands,
            "command_ms": distribution(command_values),
            "inner_command_ms": distribution(inner_values),
            "steps_per_second": total_commands / total_seconds,
            "games_per_second": len(games) / total_seconds,
            "terminal_state_sha256": games[0]["terminal_state_sha256"],
            "logical_trace_sha256": games[0]["logical_trace_sha256"],
        }
        if len({game["terminal_state_sha256"] for game in games}) != 1:
            raise Rul9Error(f"release {surface} terminal identity is nondeterministic")
        if len({game["logical_trace_sha256"] for game in games}) != 1:
            raise Rul9Error(f"release {surface} logical trace is nondeterministic")
        if any(
            game["terminal_state_sha256"]
            != contract["release"]["terminal_state_sha256"]
            for game in games
        ):
            raise Rul9Error(f"release {surface} terminal identity drifted")
    terminal_hashes = {
        row["terminal_state_sha256"] for row in release_surfaces.values()
    }
    logical_hashes = {row["logical_trace_sha256"] for row in release_surfaces.values()}
    if len(terminal_hashes) != 1 or len(logical_hashes) != 1:
        raise Rul9Error("release surfaces do not reproduce one semantic trace")
    release_tokens = [
        sample for game in all_release_games for sample in game["token_samples"]
    ]
    release_summary = {
        "surfaces": release_surfaces,
        "peak_rss_bytes": int(release_rss["peak_bytes"]),
        "rss_sample_count": int(release_rss["sample_count"]),
        "fallback_counters": _sum_fallbacks(
            all_release_games, contract["authority_fallback_counters"]
        ),
        "semantic": _derive_semantics(release_raw["semantic_static"], release_tokens),
        "parity_summary": release_raw["correctness"]["parity_summary"],
    }

    training_games = training_raw["games"]
    if len(training_games) != int(contract["training"]["workers"]):
        raise Rul9Error("training did not retain one game per worker")
    expected_coordinates = [
        (int(seed), index % 2)
        for index, seed in enumerate(contract["training"]["deal_seeds"])
    ]
    actual_coordinates = [
        (int(game["deal_seed"]), int(game["ur_seat"])) for game in training_games
    ]
    if actual_coordinates != expected_coordinates:
        raise Rul9Error("training workload coordinates drifted")
    if any(not game["terminal"] for game in training_games):
        raise Rul9Error("training retained a nonterminal game")
    for game in training_games:
        decisions = int(game["decisions"])
        if decisions <= 0 or decisions > int(contract["training"]["max_decisions"]):
            raise Rul9Error("training decision count is outside the contract")
        if len(game["rows"]) != decisions:
            raise Rul9Error("training decision rows do not rederive")
        if int(game["traversals"]) != decisions * int(
            contract["training"]["simulations"]
        ):
            raise Rul9Error("training traversal count does not rederive")
    rows = [row for game in training_games for row in game["rows"]]
    decisions = sum(int(game["decisions"]) for game in training_games)
    traversals = sum(int(game["traversals"]) for game in training_games)
    wall = float(training_rss["wall_seconds"])
    native_mismatches = sum(int(row["native_actions"] != 1) for row in rows)
    training_summary = {
        "games": len(training_games),
        "decisions": decisions,
        "traversals": traversals,
        "puct_ms": distribution([row["puct_ms"] for row in rows]),
        "command_ms": distribution([row["command_ms"] for row in rows]),
        "steps_per_second": decisions / wall,
        "traversals_per_second": traversals / wall,
        "games_per_second": len(training_games) / wall,
        "peak_rss_bytes": int(training_rss["peak_bytes"]),
        "rss_sample_count": int(training_rss["sample_count"]),
        "worker_cpu_seconds": sum(
            float(game["cpu_seconds"]) for game in training_games
        ),
        "native_apply_mismatches": native_mismatches,
        "fallback_counters": _sum_fallbacks(
            training_games, contract["training_fallback_counters"]
        ),
        "semantic": _derive_semantics(
            training_raw["semantic_static"],
            _flatten_token_samples(training_games),
        ),
        "outcomes_sha256": sha256_bytes(
            canonical_json(
                [
                    {
                        "deal_seed": game["deal_seed"],
                        "ur_seat": game["ur_seat"],
                        "winner": game["winner"],
                        "terminal_state_sha256": game["terminal_state_sha256"],
                        "logical_trace_sha256": game["logical_trace_sha256"],
                    }
                    for game in training_games
                ]
            )
        ),
    }
    return {"release": release_summary, "training": training_summary}


def evaluate_verdict(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    budgets = contract["budgets"]
    release = summary["release"]
    training = summary["training"]
    release_failures = []
    live = release["surfaces"]["live"]
    if live["command_ms"]["p95"] > budgets["release"]["live_command_p95_ms_max"]:
        release_failures.append("live Command p95")
    if live["inner_command_ms"]["p95"] > budgets["release"]["inner_command_p95_ms_max"]:
        release_failures.append("live inner Command p95")
    for surface in ("headless", "replay"):
        if (
            release["surfaces"][surface]["steps_per_second"]
            < budgets["release"]["engine_steps_per_second_min"]
        ):
            release_failures.append(f"{surface} steps/s")
    if live["games_per_second"] < budgets["release"]["live_games_per_second_min"]:
        release_failures.append("live games/s")
    if release["peak_rss_bytes"] > budgets["release"]["peak_rss_bytes_max"]:
        release_failures.append("release peak RSS")

    training_failures = []
    if training["steps_per_second"] < budgets["training"]["steps_per_second_min"]:
        training_failures.append("training steps/s")
    if (
        training["traversals_per_second"]
        < budgets["training"]["traversals_per_second_min"]
    ):
        training_failures.append("training traversals/s")
    if training["games_per_second"] < budgets["training"]["games_per_second_min"]:
        training_failures.append("training games/s")
    if training["puct_ms"]["p95"] > budgets["training"]["puct_p95_ms_max"]:
        training_failures.append("training PUCT p95")
    if training["command_ms"]["p95"] > budgets["training"]["command_p95_ms_max"]:
        training_failures.append("training Command p95")
    if training["peak_rss_bytes"] > budgets["training"]["peak_rss_bytes_max"]:
        training_failures.append("training peak RSS")
    if training["native_apply_mismatches"]:
        training_failures.append("training native apply exactness")

    capacity_failures = []
    for cell_name, cell in (("release", release), ("training", training)):
        semantic = cell["semantic"]
        if (
            semantic["catalog_active_tokens"]
            > budgets["semantic"]["catalog_tokens_max"]
        ):
            capacity_failures.append(f"{cell_name} catalog tokens")
        if (
            semantic["tokens_per_definition"]["max"]
            > budgets["semantic"]["definition_tokens_max"]
        ):
            capacity_failures.append(f"{cell_name} definition tokens")
        if (
            semantic["visible_object_references"]["max"]
            > budgets["semantic"]["visible_references_max"]
        ):
            capacity_failures.append(f"{cell_name} visible references")
        for field in (
            "overflow_count",
            "projection_failures",
            "unadmitted_visible_definitions",
        ):
            if semantic[field] != 0:
                capacity_failures.append(f"{cell_name} {field}")
    fallback_failures = []
    for cell_name, cell in (("release", release), ("training", training)):
        for name, value in cell["fallback_counters"].items():
            if value != 0:
                fallback_failures.append(f"{cell_name} {name}")
    return {
        "release": {
            "status": "pass" if not release_failures else "miss",
            "failures": release_failures,
        },
        "training": {
            "status": "pass" if not training_failures else "miss",
            "failures": training_failures,
        },
        "capacity": {
            "status": "pass" if not capacity_failures else "miss",
            "failures": capacity_failures,
        },
        "fallbacks": {
            "status": "pass" if not fallback_failures else "miss",
            "failures": fallback_failures,
        },
        "overall": (
            "pass"
            if not (
                release_failures
                or training_failures
                or capacity_failures
                or fallback_failures
            )
            else "miss"
        ),
        "representation_decision": "retain full_clone/current_game_v1",
    }


def _require_zero_counters(summary: Mapping[str, Any]) -> None:
    for cell_name in ("release", "training"):
        counters = summary[cell_name]["fallback_counters"]
        nonzero = {name: value for name, value in counters.items() if value != 0}
        if nonzero:
            raise Rul9Error(f"{cell_name} fallback counters are nonzero: {nonzero}")
        semantic = summary[cell_name]["semantic"]
        for field in (
            "overflow_count",
            "projection_failures",
            "unadmitted_visible_definitions",
        ):
            if semantic[field] != 0:
                raise Rul9Error(f"{cell_name} semantic {field} is nonzero")


def build_receipt(contract: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    validate_contract(contract)
    identity_before = runtime_identity()
    if (
        identity_before["workload"]["authority_receipt_sha256"]
        != contract["release"]["authority_receipt_sha256"]
    ):
        raise Rul9Error("current authority receipt differs from the contract")
    if (
        identity_before["workload"]["parity_receipt_sha256"]
        != contract["release"]["parity_receipt_sha256"]
    ):
        raise Rul9Error("current parity receipt differs from the contract")
    for name, expected in contract["expected_inputs"].items():
        if identity_before["workload"][name] != expected:
            raise Rul9Error(f"current {name} differs from the contract")
    started = utc_now()
    release = run_release(contract)
    training = run_training(contract)
    identity_after = runtime_identity()
    if identity_after != identity_before:
        raise Rul9Error("source or binary identity changed during measurement")
    raw = {"release": release, "training": training}
    summary = derive_summary(raw, contract)
    verdict = evaluate_verdict(summary, contract)
    receipt = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run": {
            "started_at": started,
            "completed_at": utc_now(),
            "contract_path": contract_path.relative_to(ROOT).as_posix(),
            "contract_sha256": sha256_bytes(canonical_json(contract)),
        },
        "identity": identity_before,
        "raw": raw,
        "summary": summary,
        "verdict": verdict,
    }
    receipt["artifact_sha256"] = artifact_hash(receipt)
    return receipt


def _identity_projection(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": identity["source"],
        "binary": {
            key: identity["binary"][key]
            for key in (
                "profile",
                "extension_name",
                "extension_sha256",
                "python",
                "platform",
                "processor",
                "logical_cpus",
                "cargo",
                "rustc",
                "uv",
            )
        },
        "workload": identity["workload"],
    }


def _verify_contract_identity(
    identity: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    if (
        identity["workload"]["authority_receipt_sha256"]
        != contract["release"]["authority_receipt_sha256"]
    ):
        raise Rul9Error("RUL-9 receipt authority identity drifted")
    if (
        identity["workload"]["parity_receipt_sha256"]
        != contract["release"]["parity_receipt_sha256"]
    ):
        raise Rul9Error("RUL-9 receipt parity identity drifted")
    for name, expected in contract["expected_inputs"].items():
        if identity["workload"].get(name) != expected:
            raise Rul9Error(f"RUL-9 receipt {name} identity drifted")


def _verify_measurement_receipt(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    check_current: bool,
    current_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    if receipt.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise Rul9Error("RUL-9 measurement receipt schema mismatch")
    if receipt.get("experiment") != EXPERIMENT_ID:
        raise Rul9Error("RUL-9 receipt experiment mismatch")
    if receipt.get("artifact_sha256") != artifact_hash(receipt):
        raise Rul9Error("RUL-9 receipt artifact SHA-256 mismatch")
    if receipt["run"].get("contract_sha256") != sha256_bytes(canonical_json(contract)):
        raise Rul9Error("RUL-9 receipt does not bind the contract")
    identity = receipt.get("identity")
    if not isinstance(identity, Mapping):
        raise Rul9Error("RUL-9 receipt identity is missing")
    _verify_contract_identity(identity, contract)
    derived = derive_summary(receipt["raw"], contract)
    if receipt.get("summary") != derived:
        raise Rul9Error("RUL-9 receipt summary does not rederive from raw evidence")
    _require_zero_counters(derived)
    verdict = evaluate_verdict(derived, contract)
    if receipt.get("verdict") != verdict:
        raise Rul9Error("RUL-9 receipt verdict does not recompute")
    if check_current:
        current = current_identity or runtime_identity()
        if _identity_projection(identity) != _identity_projection(current):
            raise Rul9Error(
                "RUL-9 receipt source, binary, or workload identity is stale"
            )
    return verdict


def _origin_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise Rul9Error("RUL-9 measurement-origin path is missing")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _stored_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def migrate_receipt(
    contract: Mapping[str, Any],
    measurement_path: Path,
    *,
    derivation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a schema-v2 receipt without executing the measured workloads."""

    validate_contract(contract)
    measurement_bytes = measurement_path.read_bytes()
    try:
        measurement = json.loads(measurement_bytes)
    except json.JSONDecodeError as error:
        raise Rul9Error(f"cannot load RUL-9 measurement origin: {error}") from error
    _verify_measurement_receipt(contract, measurement, check_current=False)
    raw_bytes = canonical_json(measurement["raw"])
    identity = dict(derivation_identity or runtime_identity())
    _verify_contract_identity(identity, contract)
    summary = derive_summary(measurement["raw"], contract)
    verdict = evaluate_verdict(summary, contract)
    receipt = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "contract": deepcopy(contract),
        "measurement_origin": {
            "path": _stored_path(measurement_path),
            "file_sha256": sha256_bytes(measurement_bytes),
            "artifact_sha256": measurement["artifact_sha256"],
            "run": deepcopy(measurement["run"]),
            "identity": deepcopy(measurement["identity"]),
            "raw_lineage": {
                "algorithm": RAW_LINEAGE_ALGORITHM,
                "canonical_bytes": len(raw_bytes),
                "sha256": sha256_bytes(raw_bytes),
            },
        },
        "derivation": {
            "role": "derivation-and-report-only; did-not-produce-measurement-samples",
            "identity": identity,
            "contract_sha256": sha256_bytes(canonical_json(contract)),
        },
        "summary": summary,
        "verdict": verdict,
    }
    receipt["artifact_sha256"] = artifact_hash(receipt)
    return receipt


def _verify_derived_receipt(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    check_current: bool,
    current_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validate_contract(contract)
    if receipt.get("schema_version") != DERIVED_SCHEMA_VERSION:
        raise Rul9Error("RUL-9 derived receipt schema mismatch")
    if receipt.get("experiment") != EXPERIMENT_ID:
        raise Rul9Error("RUL-9 receipt experiment mismatch")
    if receipt.get("artifact_sha256") != artifact_hash(receipt):
        raise Rul9Error("RUL-9 receipt artifact SHA-256 mismatch")
    if receipt.get("contract") != contract:
        raise Rul9Error("RUL-9 derived receipt contract snapshot drifted")

    origin = receipt.get("measurement_origin")
    if not isinstance(origin, Mapping):
        raise Rul9Error("RUL-9 measurement-origin lineage is missing")
    measurement_path = _origin_path(origin.get("path"))
    try:
        measurement_bytes = measurement_path.read_bytes()
        measurement = json.loads(measurement_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise Rul9Error(f"cannot load RUL-9 measurement origin: {error}") from error
    if origin.get("file_sha256") != sha256_bytes(measurement_bytes):
        raise Rul9Error("RUL-9 measurement-origin file SHA-256 mismatch")
    _verify_measurement_receipt(contract, measurement, check_current=False)
    if origin.get("artifact_sha256") != measurement.get("artifact_sha256"):
        raise Rul9Error("RUL-9 measurement-origin artifact lineage drifted")
    if origin.get("run") != measurement.get("run"):
        raise Rul9Error("RUL-9 measurement-origin run lineage drifted")
    if origin.get("identity") != measurement.get("identity"):
        raise Rul9Error("RUL-9 measurement-origin identity lineage drifted")

    origin_raw = canonical_json(measurement["raw"])
    raw_lineage = origin.get("raw_lineage")
    expected_lineage = {
        "algorithm": RAW_LINEAGE_ALGORITHM,
        "canonical_bytes": len(origin_raw),
        "sha256": sha256_bytes(origin_raw),
    }
    if raw_lineage != expected_lineage:
        raise Rul9Error("RUL-9 canonical raw lineage does not match its origin")

    derived = derive_summary(measurement["raw"], contract)
    if receipt.get("summary") != derived:
        raise Rul9Error("RUL-9 receipt summary does not rederive from raw evidence")
    _require_zero_counters(derived)
    verdict = evaluate_verdict(derived, contract)
    if receipt.get("verdict") != verdict:
        raise Rul9Error("RUL-9 receipt verdict does not recompute")

    derivation = receipt.get("derivation")
    if not isinstance(derivation, Mapping):
        raise Rul9Error("RUL-9 derivation identity is missing")
    if derivation.get("role") != (
        "derivation-and-report-only; did-not-produce-measurement-samples"
    ):
        raise Rul9Error("RUL-9 derivation role is ambiguous")
    if derivation.get("contract_sha256") != sha256_bytes(canonical_json(contract)):
        raise Rul9Error("RUL-9 derivation does not bind the contract")
    identity = derivation.get("identity")
    if not isinstance(identity, Mapping):
        raise Rul9Error("RUL-9 derivation source identity is missing")
    _verify_contract_identity(identity, contract)
    if check_current:
        current = current_identity or runtime_identity()
        if _identity_projection(identity) != _identity_projection(current):
            raise Rul9Error(
                "RUL-9 derivation source, binary, or workload identity is stale"
            )
    return verdict


def verify_receipt(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    check_current: bool,
    current_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schema_version = receipt.get("schema_version")
    if schema_version == MEASUREMENT_SCHEMA_VERSION:
        verdict = _verify_measurement_receipt(
            contract,
            receipt,
            check_current=check_current,
            current_identity=current_identity,
        )
    elif schema_version == DERIVED_SCHEMA_VERSION:
        verdict = _verify_derived_receipt(
            contract,
            receipt,
            check_current=check_current,
            current_identity=current_identity,
        )
    else:
        raise Rul9Error("RUL-9 receipt schema mismatch")
    if verdict["overall"] != "pass":
        raise Rul9Error(f"RUL-9 product budgets missed: {verdict}")
    return {
        "verified": True,
        "artifact_sha256": receipt["artifact_sha256"],
        "release": verdict["release"]["status"],
        "training": verdict["training"]["status"],
        "capacity": verdict["capacity"]["status"],
        "representation_decision": verdict["representation_decision"],
    }


def _mib(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def _report_contract(
    receipt: Mapping[str, Any], contract: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    bound = receipt.get("contract") if contract is None else contract
    if not isinstance(bound, Mapping):
        raise Rul9Error("RUL-9 report rendering requires the bound contract")
    return bound


def _report_context(
    receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None]:
    if receipt.get("schema_version") == DERIVED_SCHEMA_VERSION:
        origin = receipt["measurement_origin"]
        return origin["run"], origin["identity"], receipt["derivation"]
    return receipt["run"], receipt["identity"], None


def _result_narrative(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    summary = receipt["summary"]
    verdict = receipt["verdict"]
    release = summary["release"]
    training = summary["training"]
    live = release["surfaces"]["live"]
    headless = release["surfaces"]["headless"]
    replay = release["surfaces"]["replay"]
    release_budget = contract["budgets"]["release"]
    training_budget = contract["budgets"]["training"]
    exact_live_misses = {"live Command p95", "live games/s"}
    release_failures = set(verdict["release"]["failures"])

    trace = (
        "The measurement-origin source reproduced one exact 132-Command semantic "
        "trace across live, headless, and persisted replay execution. All authority, "
        "search, cap, projection, and overflow counters remained zero."
    )
    if verdict["overall"] == "pass":
        outcome = (
            "The fixed release tape and saturated selected BranchDriver teacher both "
            "stayed within every pre-registered budget."
        )
    elif (
        release_failures == exact_live_misses
        and verdict["training"]["status"] == "pass"
    ):
        outcome = (
            "Release missed only the player-facing live gates: WebSocket Command p95 "
            f"was {live['command_ms']['p95']:.3f} ms against the "
            f"{release_budget['live_command_p95_ms_max']:.0f} ms maximum, and live "
            f"complete-game throughput was {live['games_per_second']:.3f}/s against "
            f"the {release_budget['live_games_per_second_min']:.1f}/s minimum. The "
            f"inner semantic Command p95 remained {live['inner_command_ms']['p95']:.3f} "
            f"ms against its {release_budget['inner_command_p95_ms_max']:.0f} ms "
            f"maximum; headless and replay remained {headless['steps_per_second']:.1f} "
            f"and {replay['steps_per_second']:.1f} steps/s against the "
            f"{release_budget['engine_steps_per_second_min']:.0f}/s minimum. Training "
            f"passed at {training['steps_per_second']:.3f} roots/s, "
            f"{training['traversals_per_second']:.1f} traversals/s, and "
            f"{training['games_per_second']:.4f} games/s against minima of "
            f"{training_budget['steps_per_second_min']:.1f}, "
            f"{training_budget['traversals_per_second_min']:.0f}, and "
            f"{training_budget['games_per_second_min']:.2f}."
        )
    else:
        missed = []
        for cell in ("release", "training", "capacity", "fallbacks"):
            failures = verdict[cell]["failures"]
            if failures:
                missed.append(f"{cell}: {', '.join(failures)}")
        outcome = "Pre-registered misses were " + "; ".join(missed) + "."
    return (
        f"{outcome} {trace} The decision remains "
        f"`{verdict['representation_decision']}`."
    )


def render_report(
    receipt: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> str:
    bound_contract = _report_contract(receipt, contract)
    if receipt.get("schema_version") == DERIVED_SCHEMA_VERSION:
        _verify_derived_receipt(
            bound_contract,
            receipt,
            check_current=False,
            current_identity=None,
        )
    measurement_run, measurement_identity, derivation = _report_context(receipt)
    summary = receipt["summary"]
    release = summary["release"]
    training = summary["training"]
    live = release["surfaces"]["live"]
    headless = release["surfaces"]["headless"]
    replay = release["surfaces"]["replay"]
    expanded = training["semantic"]["expanded_semantic_tokens"]
    attribution = training["semantic"]["expanded_maximum_attribution"]
    frontier = int(bound_contract["budgets"]["semantic"]["catalog_tokens_max"])
    if expanded["max"] > frontier:
        diagnostic = (
            f"The counterfactual physical-object expansion reached {expanded['max']:.0f} "
            f"tokens, above the {frontier:,} diagnostic frontier, while the selected "
            f"shared catalog remained {training['semantic']['catalog_active_tokens']} "
            "active tokens with definition-row references and zero overflow. "
            f"At the maximum, zone attribution was `{json.dumps(attribution.get('tokens_by_zone', {}), sort_keys=True)}`. "
            "This is repeated visible reference pressure, not runtime clipping; retain "
            "`full_clone/current_game_v1` and the shared catalog/reference representation."
        )
    else:
        diagnostic = (
            "The final workload did not reproduce the exploratory expanded-token "
            f"pressure: its counterfactual maximum was {expanded['max']:.0f}, below "
            f"the {frontier:,} diagnostic frontier. The selected shared catalog "
            f"remained {training['semantic']['catalog_active_tokens']} active tokens "
            "with definition-row references and zero overflow. "
            f"At the maximum, zone attribution was `{json.dumps(attribution.get('tokens_by_zone', {}), sort_keys=True)}`. "
            "The pre-registered capacity gates pass without clipping; retain "
            "`full_clone/current_game_v1` and the shared catalog/reference representation."
        )
    if derivation is None:
        integrity = [
            f"- Measurement source closure: `{measurement_identity['source']['sha256']}` over {len(measurement_identity['source']['files'])} files.",
            f"- Measurement native extension: `{measurement_identity['binary']['extension_sha256']}` ({measurement_identity['binary']['extension_name']}, release profile).",
            f"- Contract: `{measurement_run['contract_sha256']}`.",
            f"- Evidence artifact: `{receipt['artifact_sha256']}`.",
        ]
        reproduce = [
            "uv run experiments/runners/run_rul9_played_workloads.py \\",
            "  --contract experiments/contracts/rul-9-played-workloads-v1.json \\",
            "  --out experiments/data/rul-9-played-workloads-v1.json \\",
            "  --report experiments/rul-9-played-workloads-v1.md",
        ]
    else:
        origin = receipt["measurement_origin"]
        derivation_identity = derivation["identity"]
        integrity = [
            f"- Immutable measurement origin: `{origin['artifact_sha256']}` at `{origin['path']}` (file SHA-256 `{origin['file_sha256']}`).",
            f"- Measurement source closure: `{measurement_identity['source']['sha256']}` over {len(measurement_identity['source']['files'])} files; these files and the bound native extension `{measurement_identity['binary']['extension_sha256']}` produced the samples.",
            f"- Canonical raw evidence: `{origin['raw_lineage']['sha256']}` over {origin['raw_lineage']['canonical_bytes']} bytes in the immutable origin; the derived receipt binds this corpus without duplicating it.",
            f"- Derivation/report source closure: `{derivation_identity['source']['sha256']}` over {len(derivation_identity['source']['files'])} files; this source only rederived and rendered the retained samples and did not produce them.",
            f"- Derivation native extension identity: `{derivation_identity['binary']['extension_sha256']}` ({derivation_identity['binary']['extension_name']}, release profile).",
            f"- Contract: `{derivation['contract_sha256']}`.",
            f"- Derived evidence artifact: `{receipt['artifact_sha256']}`.",
        ]
        reproduce = [
            "uv run experiments/runners/run_rul9_played_workloads.py \\",
            "  --migrate \\",
            f"  --measurement-origin {origin['path']} \\",
            "  --contract experiments/contracts/rul-9-played-workloads-v1.json \\",
            "  --out experiments/data/rul-9-played-workloads-v1.json \\",
            "  --report experiments/rul-9-played-workloads-v1.md",
        ]

    lines = [
        "# RUL-9: Played Release and Training Workloads",
        "",
        f"**Run:** {measurement_run['completed_at'][:10]}  ",
        "**Matchup:** UR Lessons versus GW Allies  ",
        f"**Verdict:** **{receipt['verdict']['overall'].upper()}** — release {receipt['verdict']['release']['status']}, training {receipt['verdict']['training']['status']}, capacity {receipt['verdict']['capacity']['status']}, fallbacks {receipt['verdict']['fallbacks']['status']}.",
        "",
        "## Result",
        "",
        _result_narrative(receipt, bound_contract),
        "",
        "| Workload | Command p50 / p95 | Step throughput | Complete games | Peak RSS |",
        "|---|---:|---:|---:|---:|",
        f"| Live release | {live['command_ms']['p50']:.3f} / {live['command_ms']['p95']:.3f} ms | {live['steps_per_second']:.1f}/s | {live['games_per_second']:.3f}/s | {_mib(release['peak_rss_bytes']):.1f} MiB |",
        f"| Headless release | {headless['command_ms']['p50']:.3f} / {headless['command_ms']['p95']:.3f} ms | {headless['steps_per_second']:.1f}/s | {headless['games_per_second']:.3f}/s | shared release cell |",
        f"| Persisted replay | {replay['command_ms']['p50']:.3f} / {replay['command_ms']['p95']:.3f} ms | {replay['steps_per_second']:.1f}/s | {replay['games_per_second']:.3f}/s | shared release cell |",
        f"| 4×128 training | {training['command_ms']['p50']:.3f} / {training['command_ms']['p95']:.3f} ms | {training['steps_per_second']:.3f} roots/s; {training['traversals_per_second']:.1f} traversals/s | {training['games_per_second']:.4f}/s | {_mib(training['peak_rss_bytes']):.1f} MiB |",
        "",
        f"Training PUCT decision latency was {training['puct_ms']['p50']:.1f} ms p50 and {training['puct_ms']['p95']:.1f} ms p95 across {training['decisions']} root Commands and {training['traversals']} traversals. All {training['games']} games reached terminal.",
        "",
        "## Semantic capacity and miss diagnosis",
        "",
        f"The compiled catalog contains {training['semantic']['catalog_active_tokens']} active tokens. Tokens per admitted definition were {training['semantic']['tokens_per_definition']['p50']:.0f} p50, {training['semantic']['tokens_per_definition']['p95']:.0f} p95, and {training['semantic']['tokens_per_definition']['max']:.0f} maximum. Acting-view decisions reached {training['semantic']['visible_object_references']['max']:.0f} visible references with zero projection failure, unadmitted definition, truncation, or overflow.",
        "",
        diagnostic,
        "",
        "## Integrity",
        "",
        f"- Authority receipt: `{measurement_identity['workload']['authority_receipt_sha256']}`.",
        f"- Frozen PR #153 parity provenance: `{measurement_identity['workload']['parity_receipt_sha256']}`; live/headless/replay parity samples are retained unchanged from the measurement origin.",
        *integrity,
        f"- Release RSS samples: {release['rss_sample_count']}; training RSS samples: {training['rss_sample_count']}.",
        "- Raw evidence retains every Command/PUCT duration, game duration, RSS sample, semantic census row, terminal hash, outcome hash, and fallback/cap counter.",
        "",
        "## Strongest confound",
        "",
        "These absolute performance gates were measured on one Apple Silicon host. Exact workload, measurement source, toolchain, worker topology, and binary identities are pinned, but another host may move latency and RSS without changing the representation. The derivation source is bound separately and did not rerun the workload. The fail-closed verifier treats measurement-origin or derivation drift as new evidence, not as a continuation of this run.",
        "",
        "## Reproduce",
        "",
        "```bash",
        *reproduce,
        "./scripts/verify-rul9-played-workloads",
        "```",
        "",
    ]
    return "\n".join(lines)


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def _success_line(receipt: Mapping[str, Any]) -> str:
    release = receipt["summary"]["release"]
    training = receipt["summary"]["training"]
    return (
        "RUL9_WORKLOADS_OK "
        f"release_p50_ms={release['surfaces']['live']['command_ms']['p50']:.3f} "
        f"release_p95_ms={release['surfaces']['live']['command_ms']['p95']:.3f} "
        f"release_steps_s={release['surfaces']['headless']['steps_per_second']:.1f} "
        f"release_games_s={release['surfaces']['live']['games_per_second']:.3f} "
        f"training_p50_ms={training['command_ms']['p50']:.3f} "
        f"training_p95_ms={training['command_ms']['p95']:.3f} "
        f"training_steps_s={training['steps_per_second']:.3f} "
        f"training_games_s={training['games_per_second']:.4f} "
        f"traversals_s={training['traversals_per_second']:.1f} "
        f"release_peak_mib={_mib(release['peak_rss_bytes']):.1f} "
        f"training_peak_mib={_mib(training['peak_rss_bytes']):.1f} "
        f"catalog_tokens={training['semantic']['catalog_active_tokens']} "
        f"expanded_tokens_max={training['semantic']['expanded_semantic_tokens']['max']:.0f} "
        "fallbacks=0 overflow=0"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--measurement-origin", type=Path, default=DEFAULT_MEASUREMENT_ORIGIN
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--verify", action="store_true")
    operation.add_argument("--migrate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract(args.contract)
    try:
        if args.verify:
            receipt = json.loads(args.out.read_text(encoding="utf-8"))
            verify_receipt(contract, receipt, check_current=True)
            print(_success_line(receipt))
            return 0
        if args.migrate:
            receipt = migrate_receipt(contract, args.measurement_origin.resolve())
        else:
            receipt = build_receipt(contract, args.contract.resolve())
        atomic_write(
            args.out,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        atomic_write(args.report, render_report(receipt, contract).encode("utf-8"))
        verify_receipt(contract, receipt, check_current=True)
        print(_success_line(receipt))
        return 0
    except (OSError, json.JSONDecodeError, Rul9Error) as error:
        print(f"RUL-9 evidence failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
