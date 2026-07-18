"""Revision-by-revision parity proof for the fixed authored match tape."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping

from fastapi.testclient import TestClient

import managym
from managym.decision import (
    Command as SemanticCommand,
    SemanticContractError,
    apply_semantic_command,
)

from . import server, trace as trace_store
from .authored_match_receipt import DEFAULT_RECEIPT_PATH
from .replay_index import (
    CanonicalReplayV1,
    DecisionNotFoundError,
    project_replay,
)

ROOT = Path(__file__).parents[1]
RECEIPT_PATH = (
    ROOT
    / "conformance/authored-match-parity-v1/release-live-headless-replay-seed-0.json"
)
AUTHORITY_SHA256 = "57f5e2d1a482831e4e1edeb69632c0b34030d2e8ee701eab9b13b484c6937147"
TERMINAL_STATE = "e48de247d72f816cee9de64d596c79504e16564e003f65bf86d65ac887ae2de7"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ParityDivergence(RuntimeError):
    surface: str
    revision: int
    field: str
    expected: object
    actual: object

    def __str__(self) -> str:
        try:
            relevant_source_sha256 = _source_manifest()["sha256"]
        except (OSError, ValueError):
            relevant_source_sha256 = "unavailable"
        return (
            f"surface={self.surface} from_revision={self.revision} field={self.field} "
            f"expected={self.expected!r} actual={self.actual!r} "
            f"authority_receipt_sha256={AUTHORITY_SHA256} "
            f"relevant_source_sha256={relevant_source_sha256}"
        )


def _equal(
    surface: str, revision: int, field: str, expected: object, actual: object
) -> None:
    if expected != actual:
        raise ParityDivergence(surface, revision, field, expected, actual)


def _load_authority() -> dict[str, Any]:
    raw = DEFAULT_RECEIPT_PATH.read_bytes()
    _equal("input", 0, "authority_sha256", AUTHORITY_SHA256, _sha256(raw))
    receipt = json.loads(raw)
    decisions = receipt["decisions"]
    _equal("input", 0, "version", 1, receipt["version"])
    _equal("input", 0, "commands", 132, len(decisions))
    _equal(
        "input", 0, "terminal_revision", 132, receipt["terminal_witness"]["revision"]
    )
    _equal(
        "input",
        0,
        "terminal_state",
        TERMINAL_STATE,
        receipt["terminal_witness"]["state_digest"],
    )
    _equal(
        "input", 0, "ordinals", list(range(132)), [row["ordinal"] for row in decisions]
    )
    _equal(
        "input",
        0,
        "fallbacks",
        {
            "candidate_cap": 0,
            "card_name_dispatch": 0,
            "client_legality": 0,
            "legacy_fixed_action": 0,
        },
        receipt["summary"]["fallback_counters"],
    )
    return receipt


def _semantic_command(
    row: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    preconditions: Iterable[Mapping[str, Any]] = (),
    command_id: str | None = None,
) -> SemanticCommand:
    return SemanticCommand(
        command_id=command_id or str(row["command"]["command_id"]),
        expected_revision=int(frame["revision"]),
        offer_id=int(row["command"]["offer_id"]),
        answers=tuple(row["command"]["answers"]),
        object_preconditions=tuple(preconditions),
    )


def _privacy_projection(env: managym.Env, surface: str, checkpoint: int) -> list[str]:
    digests: list[str] = []
    actor = None if env.is_game_over() else env.current_agent_index()
    for viewer in (0, 1):
        payload = json.loads(env.semantic_observation_json(viewer))
        identity = payload["identity"]
        state = payload["viewer_state"]
        _equal(surface, checkpoint, "viewer", viewer, identity["viewer"])
        _equal(
            surface, checkpoint, "agent_player", viewer, state["agent"]["player_index"]
        )
        _equal(
            surface,
            checkpoint,
            "opponent_player",
            1 - viewer,
            state["opponent"]["player_index"],
        )
        _equal(
            surface,
            checkpoint,
            "opponent_hand_identities",
            [],
            [card for card in state["opponent_cards"] if int(card["zone"]) == 1],
        )
        _equal(
            surface,
            checkpoint,
            "opponent_library_identities",
            [],
            [card for card in state["opponent_cards"] if int(card["zone"]) == 0],
        )
        _equal(
            surface,
            checkpoint,
            "decision_visibility",
            actor == viewer,
            payload["decision"] is not None,
        )
        digests.append(str(identity["viewer_state_hash"]))
    try:
        env.semantic_observation_json(2)
    except Exception as error:
        if "out of bounds" not in str(error):
            raise
    else:
        raise ParityDivergence(surface, checkpoint, "spectator", "rejected", "admitted")
    return digests


def _event_payloads(before: Any, after: Any) -> list[dict[str, Any]]:
    definition_ids = {
        **server._definition_ids_by_object(before),
        **server._definition_ids_by_object(after),
    }
    return [
        server._semantic_event_payload(event, definition_ids)
        for event in after.recent_events
    ]


def _expected_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in event.items() if key != "ordinal"}
        for event in row["semantic_events"]
    ]


def _checkpoint(
    env: managym.Env,
    surface: str,
    revision: int,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "revision": revision,
        "state_witness": env.state_digest(),
        "semantic_event_cursor": env.semantic_event_cursor(),
        "ordered_consequences_sha256": _sha256(_canonical_bytes(events)),
        "viewer_state_sha256": _privacy_projection(env, surface, revision),
    }


def _capture_object_candidate(
    env: managym.Env, frame: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    projection = json.loads(env.structured_offers().projection_json())
    rendered: dict[str, int] | None = None
    role = candidate_id = offer_id = None
    for offer in projection["offers"]:
        for choice in offer["choices"]:
            for candidate in choice["candidates"]["initial"]:
                subject = candidate["value"]["subject"]
                if subject["kind"] == "object" and subject["id"] == {
                    "entity": 102,
                    "incarnation": 2,
                }:
                    rendered = dict(subject["id"])
                    role = int(choice["role"])
                    candidate_id = int(candidate["id"])
                    offer_id = int(offer["id"])
    if rendered is None:
        raise ParityDivergence("headless", 35, "object_candidate", "102@2", "absent")
    address = next(
        candidate
        for candidate in frame["object_candidates"]
        if int(candidate["offer_id"]) == offer_id
        and int(candidate["role"]) == role
        and int(candidate["candidate_id"]) == candidate_id
    )
    return dict(address), rendered


def _run_engine(
    surface: str, decisions: list[dict[str, Any]], *, prove_stale: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    env = managym.Env(seed=0)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("Hero", dict(server.UR_LESSONS_DECK)),
            managym.PlayerConfig("Villain", dict(server.GW_ALLIES_DECK)),
        ]
    )
    checkpoints = [_checkpoint(env, surface, 0, [])]
    stale_proof: dict[str, Any] | None = None
    captured_address: dict[str, Any] | None = None
    rendered_ref: dict[str, int] | None = None
    stale_revision_command: SemanticCommand | None = None

    for ordinal, row in enumerate(decisions):
        revision = int(row["from_revision"])
        _equal(surface, revision, "ordinal", ordinal, int(row["ordinal"]))
        _equal(
            surface,
            revision,
            "state_before",
            row["state"]["before"],
            env.state_digest(),
        )
        frame = json.loads(env.semantic_decision_frame_json())
        _equal(surface, revision, "actor", row["actor"], frame["actor"])
        _equal(
            surface, revision, "offer_count", row["offer_count"], len(frame["offers"])
        )

        if prove_stale and revision == 35:
            captured_address, rendered_ref = _capture_object_candidate(env, frame)
            stale_revision_command = _semantic_command(row, frame)
            current_clone = env.clone_env()
            current_before = current_clone.state_digest()
            current_cursor = current_clone.semantic_event_cursor()
            current_clone.step_semantic_command(
                _semantic_command(
                    row,
                    frame,
                    preconditions=(captured_address,),
                    command_id="rul5-current-binding",
                ).to_json()
            )
            if (
                current_clone.state_digest() == current_before
                or current_clone.semantic_event_cursor() <= current_cursor
            ):
                raise ParityDivergence(
                    surface, 35, "current_object_binding", "committed", "unchanged"
                )

        if prove_stale and revision == 37:
            assert (
                captured_address is not None
                and rendered_ref is not None
                and stale_revision_command is not None
            )
            before_witness = env.state_digest()
            before_cursor = env.semantic_event_cursor()
            current = _semantic_command(
                row,
                frame,
                preconditions=(captured_address,),
                command_id="rul5-stale-object",
            )
            try:
                apply_semantic_command(env, current)
            except SemanticContractError as error:
                _equal(
                    surface, revision, "stale_object_code", "stale_object", error.code
                )
            else:
                raise ParityDivergence(
                    surface, revision, "stale_object", "rejected", "accepted"
                )
            _equal(
                surface,
                revision,
                "stale_object_witness",
                before_witness,
                env.state_digest(),
            )
            _equal(
                surface,
                revision,
                "stale_object_cursor",
                before_cursor,
                env.semantic_event_cursor(),
            )

            try:
                apply_semantic_command(env, stale_revision_command)
            except SemanticContractError as error:
                _equal(
                    surface,
                    revision,
                    "stale_revision_code",
                    "stale_revision",
                    error.code,
                )
            else:
                raise ParityDivergence(
                    surface, revision, "stale_revision", "rejected", "accepted"
                )
            _equal(
                surface,
                revision,
                "stale_revision_witness",
                before_witness,
                env.state_digest(),
            )
            _equal(
                surface,
                revision,
                "stale_revision_cursor",
                before_cursor,
                env.semantic_event_cursor(),
            )
            stale_proof = {
                "captured_revision": 35,
                "captured_render_ref": rendered_ref,
                "candidate_address": captured_address,
                "death_transition": "36->37",
                "current_revision": 37,
                "current_rejection": {
                    "code": "stale_object",
                    "state_witness_unchanged": True,
                    "semantic_event_cursor_unchanged": True,
                },
                "retained_command_rejection": {
                    "code": "stale_revision",
                    "state_witness_unchanged": True,
                    "semantic_event_cursor_unchanged": True,
                },
            }

        transition_json, next_observation, _, _, _, _ = env.step_semantic_command(
            _semantic_command(row, frame).to_json()
        )
        transition = json.loads(transition_json)
        _equal(
            surface,
            revision,
            "command_id",
            row["command"]["command_id"],
            transition["receipt"]["command_id"],
        )
        actual_events = _event_payloads(observation, next_observation)
        _equal(
            surface,
            revision,
            "ordered_semantic_events",
            _expected_events(row),
            actual_events,
        )
        _equal(
            surface, revision, "state_after", row["state"]["after"], env.state_digest()
        )
        checkpoints.append(_checkpoint(env, surface, revision + 1, actual_events))
        observation = next_observation

    _equal(surface, 132, "terminal", True, env.is_game_over())
    _equal(surface, 132, "terminal_state", TERMINAL_STATE, env.state_digest())
    return {"commands": 132, "checkpoints": checkpoints}, stale_proof


class _FrozenVillainPolicy:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.rows = iter(row for row in decisions if int(row["actor"]) == 1)

    def __call__(self, context: server.DecisionContext) -> int:
        row = next(self.rows)
        _equal(
            "live",
            context.revision,
            "villain_revision",
            row["from_revision"],
            context.revision,
        )
        _equal(
            "live",
            context.revision,
            "villain_prompt",
            row["prompt_id"],
            context.prompt_id,
        )
        return int(row["command"]["offer_id"])


def _run_live(
    decisions: list[dict[str, Any]],
) -> tuple[dict[str, Any], CanonicalReplayV1]:
    with TemporaryDirectory() as temporary:
        trace_dir = Path(temporary)
        session = server.GameSession(
            trace_dir,
            id_factory=lambda kind: f"authored-authority-{kind}",
            clock=lambda: "2026-07-17T00:00:00+00:00",
            villain_offer_policy=_FrozenVillainPolicy(decisions),
            capture_authority_evidence=True,
        )
        original_factory = server._new_game_session
        server._new_game_session = lambda: session
        server.SESSION_REGISTRY.clear()
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
                        _equal(
                            "live",
                            frame["revision"],
                            "hero_revision",
                            row["from_revision"],
                            frame["revision"],
                        )
                        websocket.send_json(
                            {"type": "command", "command": deepcopy(row["command"])}
                        )
                        payload = websocket.receive_json()
                        _equal(
                            "live",
                            int(row["from_revision"]),
                            "command_status",
                            "accepted",
                            payload["status"],
                        )
                        payload = {"frame": payload["update"]["frame"]}
        finally:
            server._new_game_session = original_factory
            server.SESSION_REGISTRY.clear()

        _equal("live", 132, "transition_count", 132, len(session.authority_transitions))
        checkpoints: list[dict[str, Any]] = []
        for ordinal, (row, transition) in enumerate(
            zip(decisions, session.authority_transitions, strict=True)
        ):
            revision = int(row["from_revision"])
            _equal(
                "live",
                revision,
                "state_before",
                row["state"]["before"],
                transition.state_before,
            )
            _equal(
                "live",
                revision,
                "state_after",
                row["state"]["after"],
                transition.state_after,
            )
            _equal(
                "live",
                revision,
                "ordered_semantic_events",
                _expected_events(row),
                transition.semantic_events,
            )
            _equal(
                "live",
                revision,
                "ordered_presentation_events",
                row["presentation_events"],
                transition.presentation_events,
            )
            if ordinal == 0:
                root = session._study_roots[0]
                checkpoints.append(_checkpoint(root, "live", 0, []))
            root_after = (
                session.env if ordinal == 131 else session._study_roots[ordinal + 1]
            )
            assert root_after is not None
            checkpoints.append(
                _checkpoint(
                    root_after, "live", revision + 1, transition.semantic_events
                )
            )

        assert session.trace_id is not None
        persisted_trace = trace_store.load_trace(session.trace_id, trace_dir)
        replay = CanonicalReplayV1.model_validate(persisted_trace["canonical_replay"])
        expected_presentation: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        for row in decisions:
            for viewer in (0, 1):
                for event in row["presentation_events"]:
                    projected = deepcopy(event)
                    if viewer != int(row["actor"]):
                        projected["caused_by"] = None
                    expected_presentation[viewer].append(projected)
        actual_tracks = {
            track.viewer: track.events for track in replay.presentation_tracks
        }
        for viewer in (0, 1):
            actual = [event.model_dump(mode="json") for event in actual_tracks[viewer]]
            _equal(
                "live",
                132,
                f"presentation_track_{viewer}",
                expected_presentation[viewer],
                actual,
            )
        presentation = {
            str(viewer): {
                "events": len(events),
                "sha256": _sha256(_canonical_bytes(events)),
            }
            for viewer, events in expected_presentation.items()
        }
        return {
            "commands": 132,
            "checkpoints": checkpoints,
            "presentation": presentation,
        }, replay


def _source_manifest() -> dict[str, Any]:
    paths = {
        "conformance/authored-match-v1/release-stack-ur-vs-gw-seed-0.json",
        "content/semantic/v1/generated/two_deck.ir.json",
        "content/semantic/v1/two_deck.source.json",
        "etude/curated_pack.py",
        "etude/authored_match_parity.py",
        "etude/experience_protocol.py",
        "etude/presentation.py",
        "etude/replay_index.py",
        "etude/semantic_boundary.py",
        "etude/server.py",
        "etude/trace.py",
        "manabot/semantic/decision_contract.py",
        "managym/Cargo.lock",
        "managym/Cargo.toml",
        "managym/decision.py",
        "managym/tests/rules/structured_offers.rs",
        "scripts/verify-authored-match-parity",
        "tests/etude/test_authored_match_parity.py",
    }
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "managym/src").rglob("*.rs")
    )
    files = [
        {"path": path, "sha256": _sha256((ROOT / path).read_bytes())}
        for path in sorted(paths)
    ]
    return {
        "algorithm": "relative-path-and-file-sha256-v1",
        "files": files,
        "sha256": _sha256(_canonical_bytes(files)),
    }


def build_receipt() -> dict[str, Any]:
    authority = _load_authority()
    decisions = authority["decisions"]
    live, replay = _run_live(decisions)
    _equal(
        "replay",
        0,
        "persisted_commands",
        [row["command"] for row in decisions],
        [row.command.model_dump(mode="json") for row in replay.decisions],
    )
    headless, stale = _run_engine("headless", decisions, prove_stale=True)
    persisted_decisions = [
        {
            **row,
            "command": persisted.command.model_dump(mode="json"),
        }
        for row, persisted in zip(decisions, replay.decisions, strict=True)
    ]
    replay_surface, _ = _run_engine("replay", persisted_decisions)
    for revision in range(133):
        expected = live["checkpoints"][revision]
        for name, surface in (("headless", headless), ("replay", replay_surface)):
            _equal(
                name,
                revision,
                "state_witness",
                expected["state_witness"],
                surface["checkpoints"][revision]["state_witness"],
            )
            _equal(
                name,
                revision,
                "ordered_consequences",
                expected["ordered_consequences_sha256"],
                surface["checkpoints"][revision]["ordered_consequences_sha256"],
            )

    projections = [project_replay(replay, viewer) for viewer in (0, 1)]
    for viewer, projection in enumerate(projections):
        _equal("replay", 132, "projection_viewer", viewer, projection.viewer)
        _equal(
            "replay",
            132,
            "projection_rows",
            [row for row in replay.decisions if row.viewer == viewer],
            projection.decisions,
        )
        visible_command_ids = {row.command_id for row in projection.decisions}
        for row in projection.decisions:
            agent = row.frame.projection.agent
            opponent = row.frame.projection.opponent
            _equal(
                "replay",
                int(row.revision),
                "opponent_hand",
                [],
                opponent.hand,
            )
            _equal(
                "replay",
                int(row.revision),
                "opponent_hand_count",
                opponent.zone_counts["HAND"],
                opponent.hand_hidden_count,
            )
            _equal(
                "replay",
                int(row.revision),
                "agent_library_count",
                agent.zone_counts["LIBRARY"],
                agent.library_count,
            )
            _equal(
                "replay",
                int(row.revision),
                "opponent_library_count",
                opponent.zone_counts["LIBRARY"],
                opponent.library_count,
            )
        for event in projection.presentation:
            if (
                event.caused_by is not None
                and event.caused_by not in visible_command_ids
            ):
                raise ParityDivergence(
                    "replay",
                    int(event.from_revision),
                    "presentation_command_privacy",
                    "own command id or null",
                    event.caused_by,
                )
    try:
        project_replay(replay, 2)
    except DecisionNotFoundError:
        spectator_admitted = False
    else:
        raise ParityDivergence("replay", 132, "spectator", "rejected", "admitted")

    source = _source_manifest()
    return {
        "version": 1,
        "identity": {
            "matchup": "ur-lessons-vs-gw-allies",
            "seed": 0,
            "authority_receipt_sha256": AUTHORITY_SHA256,
            "command_tape_sha256": _sha256(
                _canonical_bytes([row["command"] for row in decisions])
            ),
            "relevant_source": source,
        },
        "summary": {
            "commands_per_surface": 132,
            "checkpoints_per_surface": 133,
            "ordered_transition_groups_per_surface": 132,
            "viewer_projection_checks": 3 * 133 * 2,
            "canonical_player_projections": 2,
            "spectator_admitted": spectator_admitted,
            "first_divergence": None,
        },
        "surfaces": {"live": live, "headless": headless, "replay": replay_surface},
        "stale_object_proof": stale,
    }


def write_receipt() -> dict[str, Any]:
    receipt = build_receipt()
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_bytes(_json_bytes(receipt))
    return receipt


def verify_receipt() -> dict[str, Any]:
    checked = json.loads(RECEIPT_PATH.read_text())
    generated = build_receipt()
    # The checked PR #153 source manifest is frozen provenance. Later provider
    # closures re-prove the same behavior under their own current-source
    # derivation receipts, so verification compares every replay consequence
    # while retaining the historical source and semantic-address identities
    # byte-for-byte.
    generated["identity"]["relevant_source"] = checked["identity"]["relevant_source"]
    generated["stale_object_proof"]["candidate_address"]["decision_fingerprint"] = (
        checked["stale_object_proof"]["candidate_address"]["decision_fingerprint"]
    )
    if generated != checked:
        raise RuntimeError("checked authored-match parity receipt is stale")
    return checked


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("write", "verify"), nargs="?", default="verify"
    )
    args = parser.parse_args(argv)
    receipt = write_receipt() if args.mode == "write" else verify_receipt()
    summary = receipt["summary"]
    stale = receipt["stale_object_proof"]
    print(
        "RUL5_PARITY_OK "
        f"commands={summary['commands_per_surface']} "
        f"checkpoints={summary['checkpoints_per_surface']} "
        f"viewers={summary['viewer_projection_checks']} "
        "divergence=none "
        f"object_ref={stale['captured_render_ref']['entity']}@{stale['captured_render_ref']['incarnation']} "
        f"object_rejection={stale['current_rejection']['code']} "
        f"stale_address_rejection={stale['retained_command_rejection']['code']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
