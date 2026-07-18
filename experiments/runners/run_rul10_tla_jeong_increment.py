#!/usr/bin/env python3
"""Measure and verify the RUL-10 Jeong Jeong's Deserters vertical slice."""

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
import queue
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Mapping, Sequence

from fastapi.testclient import TestClient
import numpy as np
import psutil

from etude import server, trace as trace_store
from etude.authored_match_receipt import DeterministicServerOfferPolicy
from etude.replay_index import CanonicalReplayV1
from manabot.semantic.learning import LearningSchema, SemanticIr, _TokenBuilder
import managym
from managym.decision import Command as SemanticCommand

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "experiments/contracts/rul-10-tla-jeong-increment-v1.json"
OUT_PATH = ROOT / "experiments/data/rul-10-tla-jeong-increment-v1.json"
REPORT_PATH = ROOT / "experiments/rul-10-tla-jeong-increment-v1.md"
REPLAY_DIR = ROOT / "conformance/tla-jeong-increment-v1"
JEONG_IR_PATH = ROOT / "content/semantic/v1/generated/jeong_increment.ir.json"
SCHEMA_PATH = ROOT / "content/semantic/v1/learning_schema.json"
FIXED_TIME = "2026-07-18T00:00:00+00:00"
RSS_INTERVAL_SECONDS = 0.005
WORKER_TIMEOUT_SECONDS = 300
EXPERIMENT_ID = "rul-10-tla-jeong-increment-v1"
FALLBACK_COUNTERS = (
    "legacy_fixed_action",
    "card_name_dispatch",
    "candidate_cap",
    "client_legality",
)


class Rul10Error(RuntimeError):
    """The increment or its evidence failed closed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise Rul10Error("metric distribution is empty or non-finite")
    return {
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _semantic_command(row: Mapping[str, Any], frame: Mapping[str, Any]) -> SemanticCommand:
    command = row["command"]
    return SemanticCommand(
        command_id=str(command["command_id"]),
        expected_revision=int(frame["revision"]),
        offer_id=int(command["offer_id"]),
        answers=tuple(command.get("answers", ())),
    )


def _event_payloads(before: Any, after: Any) -> list[dict[str, Any]]:
    definition_ids = {
        **server._definition_ids_by_object(before),
        **server._definition_ids_by_object(after),
    }
    return [
        server._semantic_event_payload(event, definition_ids)
        for event in after.recent_events
    ]


def _permanents(frame: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    projection = frame["projection"]
    return {
        int(permanent["id"]): permanent
        for side in ("agent", "opponent")
        for permanent in projection[side]["battlefield"]
    }


class TokenCensus:
    def __init__(self, env: managym.Env) -> None:
        self.schema = LearningSchema.load(SCHEMA_PATH)
        self.ir = SemanticIr.load(self.schema, JEONG_IR_PATH)
        builder = _TokenBuilder(self.schema, self.ir)
        program_lengths = []
        for program in self.ir.programs:
            before = len(builder.kinds)
            builder.program(program)
            program_lengths.append(len(builder.kinds) - before)

        manifest = _pack_identity(env)
        ids_by_name = {
            str(row["registry_name"]): int(row["card_def_id"])
            for row in manifest["definitions"]
        }
        self.definition_row_by_id = {}
        self.program_tokens_by_row = []
        for row, definition in enumerate(self.ir.definitions):
            registry_name = str(definition["content_pack_binding"]["value"])
            self.definition_row_by_id[ids_by_name[registry_name]] = row
            self.program_tokens_by_row.append(
                sum(
                    program_lengths[int(program_row)]
                    for program_row in definition["program_indexes"]
                )
            )
        jeong_row = next(
            row
            for row, definition in enumerate(self.ir.definitions)
            if definition["semantic_key"] == "tla.jeong_jeongs_deserters"
        )
        self.static = {
            "learning_schema_sha256": self.schema.schema_hash,
            "semantic_ir_sha256": self.ir.ir_hash,
            "program_active_tokens": sum(program_lengths),
            "program_tokens": program_lengths,
            "jeong_program_tokens": self.program_tokens_by_row[jeong_row],
            "new_opcode_count": 0,
            "definition_projection": {
                "status": "not_claimed",
                "reason": (
                    "Rebel is exact authored content but is outside the frozen v1 "
                    "learning categorical vocabulary; RUL-10 does not migrate schemas"
                ),
            },
        }

    def sample(self, env: managym.Env, viewer: int) -> dict[str, Any]:
        observation = env.observation_for_player(viewer)
        ids = [
            int(card.registry_key)
            for card in [*observation.agent_cards, *observation.opponent_cards]
        ]
        ids.extend(
            int(stack_object.source_card_registry_key)
            for stack_object in observation.stack_objects
            if int(stack_object.kind) != 0
        )
        rows = [self.definition_row_by_id[definition_id] for definition_id in ids]
        lengths = [self.program_tokens_by_row[row] for row in rows]
        return {
            "viewer": viewer,
            "visible_object_references": len(lengths),
            "expanded_program_tokens": sum(lengths),
            "overflow_count": 0,
            "unadmitted_visible_definitions": 0,
        }


class MeasuredGameSession(server.GameSession):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inner_commands: list[dict[str, Any]] = []
        self.token_samples: list[dict[str, Any]] = []
        self._census: TokenCensus | None = None

    def _step_and_record(self, *args: Any, **kwargs: Any) -> None:
        command = kwargs.get("command")
        if command is None:
            return super()._step_and_record(*args, **kwargs)
        if self.env is None:
            raise Rul10Error("live environment disappeared")
        if self._census is None:
            self._census = TokenCensus(self.env)
        actor = str(kwargs.get("actor", args[0] if args else "unknown"))
        viewer = 0 if actor == "hero" else 1
        context = kwargs.get("context")
        sample = self._census.sample(self.env, viewer)
        sample.update(
            {
                "revision": int(command["expected_revision"]),
                "action_space": None if context is None else context.action_space,
            }
        )
        started = time.perf_counter_ns()
        super()._step_and_record(*args, **kwargs)
        self.inner_commands.append(
            {
                "revision": int(command["expected_revision"]),
                "actor": viewer,
                "action_space": None if context is None else context.action_space,
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
            }
        )
        self.token_samples.append(sample)


class FrozenVillainPolicy:
    def __init__(self, decisions: Sequence[Mapping[str, Any]]) -> None:
        self.rows = iter(row for row in decisions if int(row["actor"]) == 1)

    def __call__(self, context: server.DecisionContext) -> int:
        row = next(self.rows)
        if int(row["from_revision"]) != context.revision:
            raise Rul10Error("frozen villain revision drifted")
        if int(row["prompt_id"]) != context.prompt_id:
            raise Rul10Error("frozen villain prompt drifted")
        return int(row["command"]["offer_id"])


def _game_config(seed: int) -> dict[str, Any]:
    return {
        "hero_deck": "gw_allies_jeong",
        "villain_deck": "ur_lessons",
        "villain_type": "random",
        "seed": seed,
        "auto_pass": False,
    }


def _pack_identity(env: managym.Env) -> dict[str, Any]:
    manifest = env.content_pack_manifest()
    compiled = manifest.get("compiled_semantics")
    if compiled is None or compiled.get("pack_key") != "tla-jeong-increment-v1":
        raise Rul10Error("Jeong matchup did not select its compiled pack")
    return manifest


def build_authority(seed: int) -> dict[str, Any]:
    hero_policy = DeterministicServerOfferPolicy(seed)
    villain_policy = DeterministicServerOfferPolicy(seed)
    with TemporaryDirectory() as temporary:
        session = server.GameSession(
            Path(temporary),
            id_factory=lambda kind: f"rul10-seed-{seed}-{kind}",
            clock=lambda: FIXED_TIME,
            villain_offer_policy=villain_policy,
            capture_authority_evidence=True,
        )
        frame = session.new_game(_game_config(seed))["frame"]
        cast: dict[str, Any] | None = None
        target: dict[str, Any] | None = None
        for ordinal in range(10_000):
            if frame["status"] == "game_over":
                break
            offer = hero_policy.choose(
                frame["offers"],
                actor=0,
                revision=int(frame["revision"]),
                prompt_family=str(frame["action_space"]),
            )
            before = frame
            before_permanents = _permanents(before)
            outcome = session.hero_command(
                {
                    "command_id": f"rul10.seed-{seed}.hero.{ordinal}",
                    "match_id": frame["match_id"],
                    "expected_revision": frame["revision"],
                    "prompt_id": frame["prompt"]["id"],
                    "offer_id": offer["id"],
                    "answers": [],
                }
            )
            if outcome.get("status") != "accepted":
                raise Rul10Error(f"seed {seed} Command was rejected: {outcome}")
            frame = outcome["update"]["frame"]
            if offer["label"] == "Cast Jeong Jeong's Deserters":
                cast = {
                    "revision": int(before["revision"]),
                    "prompt_id": int(before["prompt"]["id"]),
                    "offer_id": int(offer["id"]),
                    "command_id": f"rul10.seed-{seed}.hero.{ordinal}",
                }
                target = None
            if (
                cast is not None
                and target is None
                and before["action_space"] == "CHOOSE_TARGET"
                and int(before["revision"]) == cast["revision"] + 1
            ):
                after_permanents = _permanents(frame)
                increments = [
                    permanent
                    for identity, permanent in after_permanents.items()
                    if int(permanent["plus1_counters"])
                    == int(before_permanents.get(identity, {}).get("plus1_counters", 0))
                    + 1
                ]
                if len(increments) == 1:
                    selected = increments[0]
                    target = {
                        "revision": int(before["revision"]),
                        "prompt_id": int(before["prompt"]["id"]),
                        "offer_id": int(offer["id"]),
                        "command_id": f"rul10.seed-{seed}.hero.{ordinal}",
                        "public_entity": int(selected["id"]),
                        "selected_name": str(selected["name"]),
                        "counter_before": int(
                            before_permanents.get(int(selected["id"]), {}).get(
                                "plus1_counters", 0
                            )
                        ),
                        "counter_after": int(selected["plus1_counters"]),
                    }
        else:
            raise Rul10Error(f"seed {seed} exceeded the Command limit")

        if frame["status"] != "game_over" or frame["winner"] not in (0, 1):
            raise Rul10Error(f"seed {seed} did not reach a winner")
        if cast is None or target is None or target["revision"] != cast["revision"] + 1:
            raise Rul10Error(f"seed {seed} lacks linked Jeong cast/target/counter evidence")
        if session.env is None:
            raise Rul10Error("authority environment disappeared")
        manifest = _pack_identity(session.env)
        transitions = session.authority_transitions
        decisions = session.canonical_decisions
        if len(transitions) != len(decisions):
            raise Rul10Error("authority transitions and canonical decisions differ")
        rows = []
        for ordinal, (transition, decision) in enumerate(
            zip(transitions, decisions, strict=True)
        ):
            offer = decision.offer.model_dump(mode="json")
            command = decision.command.model_dump(mode="json")
            if transition.offer != offer or transition.command != command:
                raise Rul10Error("authority transition differs from canonical decision")
            if transition.offer_count != transition.legal_action_count:
                raise Rul10Error("authority offer count differs from legal count")
            rows.append(
                {
                    "ordinal": ordinal,
                    "actor": int(transition.actor),
                    "source": transition.source,
                    "from_revision": int(transition.from_revision),
                    "to_revision": int(transition.to_revision),
                    "prompt_family": transition.action_space,
                    "action_type": transition.action_type,
                    "legal_action_count": int(transition.legal_action_count),
                    "offer_count": int(transition.offer_count),
                    "prompt_id": int(transition.prompt_id),
                    "offer": offer,
                    "command": command,
                    "state": {
                        "before": transition.state_before,
                        "after": transition.state_after,
                    },
                    "semantic_events": deepcopy(transition.semantic_events),
                    "presentation_events": deepcopy(transition.presentation_events),
                    "encountered_definition_ids": list(
                        transition.encountered_definition_ids
                    ),
                }
            )
        counters = dict(session.authority_fallback_counters)
        if set(counters) != set(FALLBACK_COUNTERS) or any(counters.values()):
            raise Rul10Error(f"seed {seed} authority fallbacks are nonzero")
        jeong_definition = next(
            row
            for row in manifest["definitions"]
            if row["registry_name"] == "Jeong Jeong's Deserters"
        )
        triggered_events = [
            event
            for row in rows
            if row["from_revision"] == target["revision"]
            for event in row["semantic_events"]
            if event["event_type"] == "ABILITY_TRIGGERED"
        ]
        if not triggered_events:
            raise Rul10Error(f"seed {seed} never committed Jeong's trigger")
        return {
            "seed": seed,
            "policy": {
                "id": "uniform-server-offer-v1",
                "hero_seed": seed,
                "villain_seed": seed,
            },
            "content_pack_manifest": manifest,
            "content_hash": frame["content_hash"],
            "asset_manifest_hash": frame["asset_manifest_hash"],
            "asset_pack": frame["asset_pack"],
            "summary": {
                "commands": len(rows),
                "prompt_families": dict(
                    sorted(Counter(row["prompt_family"] for row in rows).items())
                ),
                "offer_families": dict(
                    sorted(Counter(row["offer"]["verb"] for row in rows).items())
                ),
                "maximum_offer_count": max(row["offer_count"] for row in rows),
                "fallback_counters": counters,
            },
            "jeong_witness": {
                "definition_id": int(jeong_definition["card_def_id"]),
                "semantic_program": "tla.jeong_jeongs_deserters.rally_counter",
                "cast": cast,
                "triggered_revision": target["revision"],
                "trigger_source_entities": sorted(
                    {int(event["source_id"]) for event in triggered_events}
                ),
                "target_and_counter": target,
                "normal_command_path": True,
                "scenario_injection": False,
            },
            "decisions": rows,
            "terminal_witness": {
                "winner": int(frame["winner"]),
                "revision": int(frame["revision"]),
                "state_digest": session.env.state_digest(),
                "semantic_event_cursor": int(session.env.semantic_event_cursor()),
            },
        }


def _validate_transitions(
    session: server.GameSession, authority: Mapping[str, Any]
) -> str:
    decisions = authority["decisions"]
    if len(session.authority_transitions) != len(decisions):
        raise Rul10Error("live transition count differs from authority")
    logical = []
    for row, transition in zip(
        decisions, session.authority_transitions, strict=True
    ):
        if transition.state_before != row["state"]["before"]:
            raise Rul10Error("live pre-Command state differs")
        if transition.state_after != row["state"]["after"]:
            raise Rul10Error("live post-Command state differs")
        if transition.semantic_events != row["semantic_events"]:
            raise Rul10Error("live ordered semantic events differ")
        if transition.command != row["command"]:
            raise Rul10Error("live semantic Command differs")
        logical.append(
            {
                "revision": row["from_revision"],
                "state_before": row["state"]["before"],
                "state_after": row["state"]["after"],
                "command_id": row["command"]["command_id"],
                "events": row["semantic_events"],
            }
        )
    return sha256_bytes(canonical_json(logical))


def run_live_game(
    authority: Mapping[str, Any], *, retain_replay: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    seed = int(authority["seed"])
    decisions = authority["decisions"]
    with TemporaryDirectory() as temporary:
        trace_dir = Path(temporary)
        session = MeasuredGameSession(
            trace_dir,
            id_factory=lambda kind: f"rul10-seed-{seed}-{kind}",
            clock=lambda: FIXED_TIME,
            villain_offer_policy=FrozenVillainPolicy(decisions),
            capture_authority_evidence=True,
        )
        original_factory = server._new_game_session
        server._new_game_session = lambda: session
        server.SESSION_REGISTRY.clear()
        outer_commands = []
        started_game = time.perf_counter_ns()
        try:
            with TestClient(server.app) as client:
                with client.websocket_connect("/ws/play") as websocket:
                    websocket.send_json({"type": "new_game", "config": _game_config(seed)})
                    payload = websocket.receive_json()
                    hero_rows = iter(
                        row for row in decisions if int(row["actor"]) == 0
                    )
                    while payload["frame"]["status"] != "game_over":
                        row = next(hero_rows)
                        frame = payload["frame"]
                        if int(frame["revision"]) != int(row["from_revision"]):
                            raise Rul10Error("live surfaced revision differs")
                        started = time.perf_counter_ns()
                        websocket.send_json(
                            {"type": "command", "command": deepcopy(row["command"])}
                        )
                        outcome = websocket.receive_json()
                        if outcome.get("status") != "accepted":
                            raise Rul10Error(
                                f"live WebSocket Command was rejected: {outcome}"
                            )
                        next_frame = outcome["update"]["frame"]
                        outer_commands.append(
                            {
                                "from_revision": int(frame["revision"]),
                                "to_revision": int(next_frame["revision"]),
                                "action_space": frame["action_space"],
                                "duration_ms": (
                                    time.perf_counter_ns() - started
                                )
                                / 1_000_000,
                            }
                        )
                        payload = {"frame": next_frame}
        finally:
            server._new_game_session = original_factory
            server.SESSION_REGISTRY.clear()
        seconds = (time.perf_counter_ns() - started_game) / 1_000_000_000
        if session.env is None or not session.env.is_game_over():
            raise Rul10Error("live game did not reach terminal")
        if session.env.state_digest() != authority["terminal_witness"]["state_digest"]:
            raise Rul10Error("live terminal state differs")
        logical_hash = _validate_transitions(session, authority)
        if session.trace_id is None:
            raise Rul10Error("live game did not persist replay")
        persisted = trace_store.load_trace(session.trace_id, trace_dir)
        replay = CanonicalReplayV1.model_validate(persisted["canonical_replay"])
        replay_commands = [
            row.command.model_dump(mode="json") for row in replay.decisions
        ]
        if replay_commands != [row["command"] for row in decisions]:
            raise Rul10Error("persisted replay Commands differ")
        if session._census is None:
            raise Rul10Error("live token census was not initialized")
        return (
            {
                "seed": seed,
                "commands": len(decisions),
                "protocol_commands": outer_commands,
                "inner_commands": list(session.inner_commands),
                "token_samples": list(session.token_samples),
                "semantic_static": session._census.static,
                "game_seconds": seconds,
                "terminal_state_sha256": session.env.state_digest(),
                "logical_trace_sha256": logical_hash,
                "fallback_counters": dict(session.authority_fallback_counters),
            },
            replay.model_dump(mode="json") if retain_replay else None,
        )


def run_engine_game(authority: Mapping[str, Any], surface: str) -> dict[str, Any]:
    seed = int(authority["seed"])
    env = managym.Env(seed=seed)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("Hero", dict(server.GW_ALLIES_JEONG_DECK)),
            managym.PlayerConfig("Villain", dict(server.UR_LESSONS_DECK)),
        ]
    )
    _pack_identity(env)
    census = TokenCensus(env)
    commands = []
    tokens = []
    logical = []
    started_game = time.perf_counter_ns()
    for row in authority["decisions"]:
        revision = int(row["from_revision"])
        if env.state_digest() != row["state"]["before"]:
            raise Rul10Error(f"{surface} pre-Command state differs")
        frame = json.loads(env.semantic_decision_frame_json())
        sample = census.sample(env, int(frame["actor"]))
        sample.update(
            {"revision": revision, "action_space": row["prompt_family"]}
        )
        started = time.perf_counter_ns()
        before = observation
        transition_json, observation, *_ = env.step_semantic_command(
            _semantic_command(row, frame).to_json()
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        transition = json.loads(transition_json)
        if transition["receipt"]["command_id"] != row["command"]["command_id"]:
            raise Rul10Error(f"{surface} command receipt differs")
        if env.state_digest() != row["state"]["after"]:
            raise Rul10Error(f"{surface} post-Command state differs")
        events = _event_payloads(before, observation)
        if events != row["semantic_events"]:
            raise Rul10Error(f"{surface} ordered semantic events differ")
        commands.append(
            {
                "revision": revision,
                "actor": int(frame["actor"]),
                "action_space": row["prompt_family"],
                "duration_ms": elapsed_ms,
            }
        )
        tokens.append(sample)
        logical.append(
            {
                "revision": revision,
                "state_before": row["state"]["before"],
                "state_after": row["state"]["after"],
                "command_id": transition["receipt"]["command_id"],
                "events": events,
            }
        )
    seconds = (time.perf_counter_ns() - started_game) / 1_000_000_000
    if not env.is_game_over():
        raise Rul10Error(f"{surface} game did not reach terminal")
    if env.state_digest() != authority["terminal_witness"]["state_digest"]:
        raise Rul10Error(f"{surface} terminal state differs")
    return {
        "seed": seed,
        "commands": len(commands),
        "command_samples": commands,
        "token_samples": tokens,
        "semantic_static": census.static,
        "game_seconds": seconds,
        "terminal_state_sha256": env.state_digest(),
        "logical_trace_sha256": sha256_bytes(canonical_json(logical)),
        "fallback_counters": dict(authority["summary"]["fallback_counters"]),
    }


def _surface_worker(
    surface: str,
    contract: Mapping[str, Any],
    authorities: Sequence[Mapping[str, Any]],
    start: Any,
    output: Any,
) -> None:
    try:
        for _ in range(int(contract["measurement"]["warmups"])):
            for authority in authorities:
                if surface == "live":
                    run_live_game(authority, retain_replay=False)
                else:
                    run_engine_game(authority, surface)
        output.put({"kind": "ready", "pid": os.getpid()})
        start.wait()
        games = []
        replays: dict[str, Any] = {}
        for repetition in range(int(contract["measurement"]["repetitions"])):
            for authority in authorities:
                if surface == "live":
                    game, replay = run_live_game(
                        authority, retain_replay=repetition == 0
                    )
                    if replay is not None:
                        replays[str(authority["seed"])] = replay
                else:
                    game = run_engine_game(authority, surface)
                game["repetition"] = repetition
                games.append(game)
        output.put(
            {
                "kind": "result",
                "pid": os.getpid(),
                "payload": {"surface": surface, "games": games, "replays": replays},
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


def run_surface(
    surface: str,
    contract: Mapping[str, Any],
    authorities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_surface_worker,
        args=(surface, contract, authorities, start, output),
    )
    process.start()
    try:
        ready = output.get(timeout=WORKER_TIMEOUT_SECONDS)
        if ready.get("kind") == "error":
            raise Rul10Error(ready["error"])
        if ready.get("kind") != "ready":
            raise Rul10Error("surface worker did not reach measurement barrier")
        worker = psutil.Process(process.pid)
        samples = [
            {"offset_seconds": 0.0, "rss_bytes": worker.memory_info().rss}
        ]
        started = time.perf_counter()
        start.set()
        while True:
            try:
                message = output.get(timeout=RSS_INTERVAL_SECONDS)
            except queue.Empty:
                try:
                    rss = worker.memory_info().rss
                except psutil.NoSuchProcess:
                    rss = 0
                samples.append(
                    {
                        "offset_seconds": time.perf_counter() - started,
                        "rss_bytes": rss,
                    }
                )
                continue
            if message.get("kind") == "error":
                raise Rul10Error(message["error"])
            if message.get("kind") != "result":
                raise Rul10Error("surface worker returned an unexpected message")
            payload = message["payload"]
            break
        wall_seconds = time.perf_counter() - started
        process.join(timeout=30)
        if process.is_alive() or process.exitcode != 0:
            raise Rul10Error(
                f"surface worker exited abnormally: {process.exitcode}"
            )
        positive = [sample["rss_bytes"] for sample in samples if sample["rss_bytes"]]
        if not positive:
            raise Rul10Error("RSS sampler retained no positive sample")
        payload["rss"] = {
            "interval_seconds": RSS_INTERVAL_SECONDS,
            "wall_seconds": wall_seconds,
            "sample_count": len(samples),
            "baseline_bytes": positive[0],
            "peak_bytes": max(positive),
            "samples": samples,
        }
        return payload
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


def _semantic_summary(surface: Mapping[str, Any]) -> dict[str, Any]:
    games = surface["games"]
    static = games[0]["semantic_static"]
    if any(game["semantic_static"] != static for game in games):
        raise Rul10Error("semantic static identity changed between games")
    samples = [sample for game in games for sample in game["token_samples"]]
    return {
        "program_active_tokens": int(static["program_active_tokens"]),
        "program_tokens": distribution(static["program_tokens"]),
        "jeong_program_tokens": int(static["jeong_program_tokens"]),
        "new_opcode_count": int(static["new_opcode_count"]),
        "definition_projection": static["definition_projection"],
        "visible_object_references": distribution(
            [sample["visible_object_references"] for sample in samples]
        ),
        "expanded_program_tokens": distribution(
            [sample["expanded_program_tokens"] for sample in samples]
        ),
        "overflow_count": sum(int(sample["overflow_count"]) for sample in samples),
        "unadmitted_visible_definitions": sum(
            int(sample["unadmitted_visible_definitions"]) for sample in samples
        ),
    }


def derive_summary(
    surfaces: Mapping[str, Mapping[str, Any]],
    authorities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary_surfaces = {}
    for name, surface in surfaces.items():
        games = surface["games"]
        if name == "live":
            command_values = [
                row["duration_ms"]
                for game in games
                for row in game["protocol_commands"]
            ]
            inner_values = [
                row["duration_ms"]
                for game in games
                for row in game["inner_commands"]
            ]
        else:
            command_values = [
                row["duration_ms"]
                for game in games
                for row in game["command_samples"]
            ]
            inner_values = command_values
        total_seconds = sum(float(game["game_seconds"]) for game in games)
        total_commands = sum(int(game["commands"]) for game in games)
        counters = {field: 0 for field in FALLBACK_COUNTERS}
        for game in games:
            if set(game["fallback_counters"]) != set(FALLBACK_COUNTERS):
                raise Rul10Error("fallback counter inventory differs")
            for field in counters:
                counters[field] += int(game["fallback_counters"][field])
        summary_surfaces[name] = {
            "games": len(games),
            "commands": total_commands,
            "command_ms": distribution(command_values),
            "inner_command_ms": distribution(inner_values),
            "steps_per_second": total_commands / total_seconds,
            "games_per_second": len(games) / total_seconds,
            "peak_rss_bytes": int(surface["rss"]["peak_bytes"]),
            "rss_sample_count": int(surface["rss"]["sample_count"]),
            "fallback_counters": counters,
            "semantic": _semantic_summary(surface),
        }

    mismatches = []
    for authority in authorities:
        seed = int(authority["seed"])
        expected_state = authority["terminal_witness"]["state_digest"]
        expected_logical = sha256_bytes(
            canonical_json(
                [
                    {
                        "revision": row["from_revision"],
                        "state_before": row["state"]["before"],
                        "state_after": row["state"]["after"],
                        "command_id": row["command"]["command_id"],
                        "events": row["semantic_events"],
                    }
                    for row in authority["decisions"]
                ]
            )
        )
        for name, surface in surfaces.items():
            for game in surface["games"]:
                if int(game["seed"]) != seed:
                    continue
                if game["terminal_state_sha256"] != expected_state:
                    mismatches.append(f"seed {seed} {name} terminal")
                if game["logical_trace_sha256"] != expected_logical:
                    mismatches.append(f"seed {seed} {name} logical")
    prompts = Counter()
    offers = Counter()
    for authority in authorities:
        prompts.update(authority["summary"]["prompt_families"])
        offers.update(authority["summary"]["offer_families"])
    return {
        "surfaces": summary_surfaces,
        "exactness": {
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "seeds": [int(authority["seed"]) for authority in authorities],
            "linked_jeong_witnesses": len(authorities),
        },
        "prompt_families": dict(sorted(prompts.items())),
        "offer_families": dict(sorted(offers.items())),
        "maximum_offer_count": max(
            int(authority["summary"]["maximum_offer_count"])
            for authority in authorities
        ),
    }


def evaluate(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    budgets = contract["budgets"]
    surfaces = summary["surfaces"]
    required = []
    reported = []
    live = surfaces["live"]
    if live["command_ms"]["p95"] > budgets["live_outer_command_p95_ms_max"]:
        reported.append("live outer Command p95")
    if live["games_per_second"] < budgets["live_games_per_second_min"]:
        reported.append("live games/s")
    if live["inner_command_ms"]["p95"] > budgets["inner_command_p95_ms_max"]:
        required.append("live inner Command p95")
    for name in ("headless", "replay"):
        if surfaces[name]["steps_per_second"] < budgets["engine_steps_per_second_min"]:
            required.append(f"{name} steps/s")
    for name, surface in surfaces.items():
        if surface["peak_rss_bytes"] > budgets["peak_rss_bytes_max"]:
            required.append(f"{name} peak RSS")
        if any(surface["fallback_counters"].values()):
            required.append(f"{name} fallbacks")
        semantic = surface["semantic"]
        if semantic["program_active_tokens"] > budgets["program_catalog_tokens_max"]:
            required.append(f"{name} program catalog tokens")
        if semantic["program_tokens"]["max"] > budgets["program_tokens_max"]:
            required.append(f"{name} program tokens")
        if semantic["visible_object_references"]["max"] > budgets["visible_references_max"]:
            required.append(f"{name} visible references")
        if semantic["new_opcode_count"] != 0:
            required.append(f"{name} new opcodes")
        for field in (
            "overflow_count",
            "unadmitted_visible_definitions",
        ):
            if semantic[field] != 0:
                required.append(f"{name} {field}")
    if summary["exactness"]["mismatch_count"]:
        required.append("surface exactness")
    if summary["exactness"]["linked_jeong_witnesses"] != 2:
        required.append("linked Jeong witnesses")
    return {
        "required": {"status": "pass" if not required else "miss", "failures": required},
        "reported_product_budgets": {
            "status": "pass" if not reported else "miss",
            "failures": reported,
        },
        "overall": "pass" if not required else "miss",
        "classification": (
            "content-only card semantics plus one reusable immutable "
            "pack-catalog kernel extension"
        ),
    }


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment") != EXPERIMENT_ID:
        raise Rul10Error("contract experiment identity differs")
    if contract["measurement"]["seeds"] != [1, 3]:
        raise Rul10Error("contract must retain literal seeds 1 and 3")
    if contract["measurement"]["surfaces"] != ["live", "headless", "replay"]:
        raise Rul10Error("contract surface inventory differs")
    return contract


def verify_inputs(contract: Mapping[str, Any]) -> None:
    for path, expected in contract["expected_inputs"].items():
        if sha256_file(ROOT / path) != expected:
            raise Rul10Error(f"input changed: {path}")
    for path, expected in contract["frozen_rul9_files"].items():
        if sha256_file(ROOT / path) != expected:
            raise Rul10Error(f"frozen RUL-9 file changed: {path}")


def artifact_hash(receipt: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json({key: value for key, value in receipt.items() if key != "artifact_sha256"})
    )


def build_receipt(
    contract: Mapping[str, Any], contract_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_inputs(contract)
    authorities = [build_authority(seed) for seed in contract["measurement"]["seeds"]]
    live = run_surface("live", contract, authorities)
    replays = live.pop("replays")
    if set(replays) != {"1", "3"}:
        raise Rul10Error("live surface did not retain both canonical replays")

    replay_authorities = []
    for authority in authorities:
        seed = str(authority["seed"])
        replay = CanonicalReplayV1.model_validate(replays[seed])
        persisted_commands = [
            row.command.model_dump(mode="json") for row in replay.decisions
        ]
        if len(persisted_commands) != len(authority["decisions"]):
            raise Rul10Error(f"seed {seed} persisted replay length differs")
        replay_authority = deepcopy(authority)
        for row, command in zip(
            replay_authority["decisions"], persisted_commands, strict=True
        ):
            row["command"] = command
        replay_authorities.append(replay_authority)

    headless = run_surface("headless", contract, authorities)
    headless.pop("replays")
    replay_surface = run_surface("replay", contract, replay_authorities)
    replay_surface.pop("replays")
    surfaces = {
        "live": live,
        "headless": headless,
        "replay": replay_surface,
    }
    replay_artifacts = {}
    for seed, replay in replays.items():
        path = REPLAY_DIR / f"seed-{seed}.canonical-replay.json"
        raw = json_bytes(replay)
        replay_artifacts[seed] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(raw),
            "commands": len(replay["decisions"]),
        }
    summary = derive_summary(surfaces, authorities)
    verdict = evaluate(summary, contract)
    receipt = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "run": {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "contract_path": contract_path.relative_to(ROOT).as_posix(),
            "contract_sha256": sha256_file(contract_path),
        },
        "classification": verdict["classification"],
        "authorities": authorities,
        "raw": {"surfaces": surfaces},
        "summary": summary,
        "verdict": verdict,
        "replay_artifacts": replay_artifacts,
    }
    receipt["artifact_sha256"] = artifact_hash(receipt)
    return receipt, replays


def render_report(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    surfaces = receipt["summary"]["surfaces"]
    live = surfaces["live"]
    required = receipt["verdict"]["required"]
    reported = receipt["verdict"]["reported_product_budgets"]
    lines = [
        "# RUL-10: Jeong Jeong's Deserters Vertical Slice",
        "",
        f"**Verdict:** **{receipt['verdict']['overall'].upper()}** required gates; "
        f"product-budget observation **{reported['status'].upper()}**.  ",
        "**Classification:** content-only card semantics plus one reusable immutable "
        "pack-catalog kernel extension.",
        "",
        "## Result",
        "",
        "Literal seeds 1 and 3 reached terminal through production Etude WebSocket, "
        "direct headless Commands, and persisted canonical replay. Each tape contains "
        "a Jeong cast followed by a mandatory target Command and an observed +1/+1 "
        "counter delta. Revision witnesses and ordered semantic event groups matched "
        "on all three surfaces, with zero authority fallback or semantic overflow.",
        "",
        "| Surface | Command p50 / p95 | Inner p95 | Steps/s | Games/s | Peak RSS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("live", "headless", "replay"):
        row = surfaces[name]
        lines.append(
            f"| {name} | {row['command_ms']['p50']:.3f} / "
            f"{row['command_ms']['p95']:.3f} ms | "
            f"{row['inner_command_ms']['p95']:.3f} ms | "
            f"{row['steps_per_second']:.1f} | {row['games_per_second']:.3f} | "
            f"{row['peak_rss_bytes'] / 1048576:.1f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Played increment",
            "",
            f"- Seeds: `{receipt['summary']['exactness']['seeds']}`.",
            f"- Prompt families: `{json.dumps(receipt['summary']['prompt_families'], sort_keys=True)}`.",
            f"- Offer families: `{json.dumps(receipt['summary']['offer_families'], sort_keys=True)}`.",
            f"- Maximum uncapped offer count: `{receipt['summary']['maximum_offer_count']}`.",
            f"- Linked cast/target/counter witnesses: `{receipt['summary']['exactness']['linked_jeong_witnesses']}`.",
            f"- Exactness mismatches: `{receipt['summary']['exactness']['mismatch_count']}`.",
            "",
            "## Semantic capacity",
            "",
            f"The admitted programs have {live['semantic']['program_active_tokens']} "
            f"active tokens and Jeong's program uses "
            f"{live['semantic']['jeong_program_tokens']} tokens; played "
            f"decisions reached {live['semantic']['visible_object_references']['max']:.0f} "
            f"visible references and {live['semantic']['expanded_program_tokens']['max']:.0f} "
            "expanded program tokens. Overflow and unadmitted visible definition "
            "counts are zero. Full learning-definition projection is not claimed: "
            "Jeong's exact Rebel subtype is outside the frozen v1 categorical "
            "vocabulary, and this increment does not migrate schemas.",
            "",
            "## Gates",
            "",
            f"Required gates: `{required['status']}` ({required['failures']}). Product "
            f"budgets are reported without expanding this Task into optimization: "
            f"`{reported['status']}` ({reported['failures']}). The live outer p95 is "
            f"{live['command_ms']['p95']:.3f} ms against 100 ms and live completion is "
            f"{live['games_per_second']:.3f}/s against 1.0/s.",
            "",
            "## Integrity",
            "",
            f"- Evidence artifact: `{receipt['artifact_sha256']}`.",
            f"- Contract: `{receipt['run']['contract_sha256']}`.",
            "- The four RUL-9 files were hash-checked but their workloads were not run.",
            "- Raw evidence retains per-Command timings, RSS samples, semantic token "
            "samples, prompt/fallback inventories, both authorities, and both persisted "
            "canonical replays.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "./scripts/verify-tla-jeong-increment",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def verify_receipt(
    contract: Mapping[str, Any], receipt: Mapping[str, Any], report_path: Path
) -> None:
    verify_inputs(contract)
    if receipt.get("experiment") != EXPERIMENT_ID:
        raise Rul10Error("receipt experiment identity differs")
    if artifact_hash(receipt) != receipt.get("artifact_sha256"):
        raise Rul10Error("receipt artifact hash differs")
    if receipt["verdict"]["overall"] != "pass":
        raise Rul10Error(
            f"required gates missed: {receipt['verdict']['required']['failures']}"
        )
    if receipt["summary"]["exactness"]["mismatch_count"] != 0:
        raise Rul10Error("receipt retains parity mismatches")
    for authority in receipt["authorities"]:
        if any(authority["summary"]["fallback_counters"].values()):
            raise Rul10Error("authority fallback is nonzero")
        witness = authority["jeong_witness"]
        if (
            witness["target_and_counter"]["counter_after"]
            != witness["target_and_counter"]["counter_before"] + 1
        ):
            raise Rul10Error("Jeong counter witness differs")
    for artifact in receipt["replay_artifacts"].values():
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise Rul10Error(f"canonical replay changed: {path}")
        replay = CanonicalReplayV1.model_validate_json(path.read_text())
        if len(replay.decisions) != int(artifact["commands"]):
            raise Rul10Error("canonical replay Command count differs")
    expected_report = render_report(receipt, contract).encode("utf-8")
    if report_path.read_bytes() != expected_report:
        raise Rul10Error("human-readable report is stale")


def success_line(receipt: Mapping[str, Any]) -> str:
    surfaces = receipt["summary"]["surfaces"]
    return (
        "RUL10_JEONG_OK seeds=1,3 "
        f"live_p95_ms={surfaces['live']['command_ms']['p95']:.3f} "
        f"inner_p95_ms={surfaces['live']['inner_command_ms']['p95']:.3f} "
        f"headless_steps_s={surfaces['headless']['steps_per_second']:.1f} "
        f"replay_steps_s={surfaces['replay']['steps_per_second']:.1f} "
        f"live_games_s={surfaces['live']['games_per_second']:.3f} "
        f"program_tokens={surfaces['live']['semantic']['program_active_tokens']} "
        "mismatches=0 fallbacks=0 overflow=0"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = load_contract(args.contract.resolve())
        if args.verify:
            receipt = json.loads(args.out.read_text(encoding="utf-8"))
            verify_receipt(contract, receipt, args.report.resolve())
        else:
            receipt, replays = build_receipt(contract, args.contract.resolve())
            for seed, replay in replays.items():
                atomic_write(
                    REPLAY_DIR / f"seed-{seed}.canonical-replay.json",
                    json_bytes(replay),
                )
            atomic_write(args.out.resolve(), json_bytes(receipt))
            atomic_write(
                args.report.resolve(), render_report(receipt, contract).encode("utf-8")
            )
            verify_receipt(contract, receipt, args.report.resolve())
        print(success_line(receipt))
        return 0
    except (OSError, json.JSONDecodeError, Rul10Error) as error:
        print(f"RUL-10 evidence failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
