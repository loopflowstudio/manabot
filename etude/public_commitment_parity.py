"""Selected-match public-commitment provider-closure proof."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from fastapi.testclient import TestClient
import numpy as np

from manabot.belief.likelihood import (
    LikelihoodResult,
    RulesProviderGap,
    _matching_offer_indexes,
    public_commitment_key,
)
from manabot.belief.tracker import BeliefTracker
import managym
from managym.decision import (
    SEMANTIC_DECISION_VERSION,
    Command,
    DecisionFrame,
    SemanticTransition,
    apply_semantic_command,
)
from managym.possible_worlds import PossibleWorldSpace

from . import server, trace as trace_store
from .authored_match_receipt import DEFAULT_RECEIPT_PATH
from .replay_index import CanonicalReplayV1

ROOT = Path(__file__).parents[1]
RECEIPT_PATH = (
    ROOT / "conformance/public-commitment-parity-v1/"
    "release-live-replay-hypothesis-seed-0.json"
)
AUTHORITY_RECEIPT_SHA256 = (
    "57f5e2d1a482831e4e1edeb69632c0b34030d2e8ee701eab9b13b484c6937147"
)
RUL9_MEASUREMENT_ARTIFACT_SHA256 = (
    "498df1eda031f1d6ea68f72f792dfd68195bb10eab884b006aa7772b586564da"
)
RUL9_MEASUREMENT_FILE_SHA256 = (
    "9a3933a570772e8d3e04b59526faaf1d51b5fc0e26ba8c02e08eae36599bc951"
)
INT12_FIXTURE_SHA256 = (
    "4a3fbeaa8461e00a785e961b9819508d2c1065ae98f058cc50a3783db0945e8d"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(field: str, expected: object, actual: object) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} drifted: expected {expected!r}, got {actual!r}")


def _load_authority() -> dict[str, Any]:
    raw = DEFAULT_RECEIPT_PATH.read_bytes()
    _require("authority receipt SHA-256", AUTHORITY_RECEIPT_SHA256, _sha256(raw))
    authority = json.loads(raw)
    _require("authority commands", 132, len(authority["decisions"]))
    _require(
        "authority terminal revision", 132, authority["terminal_witness"]["revision"]
    )
    return authority


def _semantic_command(row: Mapping[str, Any], frame: DecisionFrame) -> Command:
    command = row["command"]
    return Command(
        command_id=str(command["command_id"]),
        expected_revision=frame.revision,
        offer_id=int(command["offer_id"]),
        answers=tuple(command.get("answers", ())),
    )


def _identity_row(
    ordinal: int,
    actor: int,
    transition: SemanticTransition,
) -> dict[str, Any]:
    receipt = transition.receipt
    return {
        "ordinal": ordinal,
        "actor": actor,
        "command_id": receipt.command_id,
        "before_revision": receipt.before_revision,
        "after_revision": receipt.after_revision,
        "public_commitment": (
            None
            if receipt.public_commitment is None
            else dict(receipt.public_commitment)
        ),
    }


class IdentityAuditLikelihood:
    """Neutral likelihood that audits only provider-owned grouping.

    This is a correctness instrument, not a calibrated model. Every compatible
    world receives the same likelihood after the retained actual root proves
    the observed commitment belongs to its authoritative offer family.
    """

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        root_engine: Any,
        *,
        viewer: int,
        commitment: Mapping[str, Any],
        belief: Any,
    ) -> LikelihoodResult:
        public_commitment_key(commitment)
        root_space = PossibleWorldSpace.from_engine(root_engine, viewer)
        _require("likelihood root space", belief.space.identity, root_space.identity)
        frame = DecisionFrame.from_json(root_engine.semantic_decision_frame_json())
        matching, legal_count = _matching_offer_indexes(frame, commitment)
        if not matching:
            raise RulesProviderGap(
                "retained root has no offer for the canonical public commitment"
            )
        self.calls += 1
        return LikelihoodResult(
            likelihoods=np.ones(belief.support_size, dtype=np.float64),
            legal_action_counts=np.full(
                belief.support_size, legal_count, dtype=np.int64
            ),
            matching_action_counts=np.full(
                belief.support_size, len(matching), dtype=np.int64
            ),
            seconds=0.0,
        )


def _tracker_snapshot(tracker: BeliefTracker) -> tuple[Any, ...]:
    return (
        tracker.posterior.digest,
        tracker.prior.digest,
        tracker.space.identity,
        tracker.observation,
        deepcopy(tracker.stats),
        tuple(tracker.records),
        deepcopy(tracker._pending_public_commitment),
    )


def _negative_proof(
    root: managym.Env,
    row: Mapping[str, Any],
    tracker: BeliefTracker,
    likelihood: IdentityAuditLikelihood,
) -> dict[str, Any]:
    branch = root.clone_env()
    frame = DecisionFrame.from_json(branch.semantic_decision_frame_json())
    valid = apply_semantic_command(branch, _semantic_command(row, frame))
    witness = branch.state_digest()
    cursor = branch.semantic_event_cursor()
    source_witness = root.state_digest()
    source_cursor = root.semantic_event_cursor()
    cases = (
        (
            "unsupported",
            {"kind": "declare_attacker"},
            "unsupported public commitment",
        ),
        (
            "card_mismatch",
            {"kind": "discard", "card": "Mountain"},
            "does not match canonical pool",
        ),
    )
    proof: dict[str, Any] = {}
    for name, public_commitment, message in cases:
        tracker_before = _tracker_snapshot(tracker)
        likelihood_calls = likelihood.calls
        invalid = SemanticTransition(
            receipt=replace(
                valid.receipt,
                public_commitment=public_commitment,
            ),
            observation=valid.observation,
        )
        try:
            tracker.observe(
                branch,
                acting=int(row["actor"]),
                transition=invalid,
                likelihood_root=root,
            )
        except RulesProviderGap as error:
            if message not in str(error):
                raise
        else:
            raise RuntimeError(f"{name} commitment was not rejected")
        _require(
            f"{name} tracker atomicity", tracker_before, _tracker_snapshot(tracker)
        )
        _require(f"{name} likelihood atomicity", likelihood_calls, likelihood.calls)
        _require(f"{name} branch witness", witness, branch.state_digest())
        _require(f"{name} branch cursor", cursor, branch.semantic_event_cursor())
        _require(f"{name} source witness", source_witness, root.state_digest())
        _require(f"{name} source cursor", source_cursor, root.semantic_event_cursor())
        proof[name] = {
            "rejection": "RulesProviderGap",
            "match_witness_unchanged": True,
            "semantic_event_cursor_unchanged": True,
            "posterior_digest_unchanged": True,
            "prior_digest_unchanged": True,
            "pending_identity_unchanged": True,
            "stats_unchanged": True,
            "records_unchanged": True,
            "likelihood_not_called": True,
        }
    return proof


def _fresh_env() -> managym.Env:
    env = managym.Env(seed=0)
    env.reset(
        [
            managym.PlayerConfig("Hero", dict(server.UR_LESSONS_DECK)),
            managym.PlayerConfig("Villain", dict(server.GW_ALLIES_DECK)),
        ]
    )
    return env


def _run_engine(
    decisions: list[dict[str, Any]],
    *,
    track_beliefs: bool,
) -> tuple[dict[str, Any], managym.Env | None]:
    env = _fresh_env()
    likelihood = IdentityAuditLikelihood()
    trackers = (
        {
            viewer: BeliefTracker.from_engine(
                env,
                viewer=viewer,
                likelihood=likelihood,
                epsilon=0.0,
            )
            for viewer in (0, 1)
        }
        if track_beliefs
        else {}
    )
    stream: list[dict[str, Any]] = []
    negative: dict[str, Any] | None = None
    discard_root: managym.Env | None = None

    for ordinal, row in enumerate(decisions):
        _require("decision ordinal", ordinal, int(row["ordinal"]))
        _require("state before", row["state"]["before"], env.state_digest())
        root = env.clone_env()
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        _require("decision actor", int(row["actor"]), frame.actor)
        if track_beliefs and int(row["from_revision"]) == 29:
            discard_root = root.clone_env()
            negative = _negative_proof(root, row, trackers[1], likelihood)
        transition = apply_semantic_command(env, _semantic_command(row, frame))
        stream.append(_identity_row(ordinal, int(row["actor"]), transition))
        _require("state after", row["state"]["after"], env.state_digest())
        for tracker in trackers.values():
            tracker.observe(
                env,
                acting=int(row["actor"]),
                transition=transition,
                likelihood_root=root,
            )

    _require("terminal command count", 132, len(stream))
    _require("terminal game", True, env.is_game_over())
    tracker_receipts = {
        str(viewer): {
            "records": len(tracker.records),
            "consumed_commitments": sum(
                record.public_commitment is not None for record in tracker.records
            ),
            "posterior_digest": tracker.posterior.digest,
            "prior_digest": tracker.prior.digest,
            "history_digest": tracker.replay_receipt()["history_digest"],
        }
        for viewer, tracker in trackers.items()
    }
    return (
        {
            "commands": len(stream),
            "identity_stream": stream,
            "identity_stream_sha256": _sha256(_canonical_bytes(stream)),
            "commitments": sum(row["public_commitment"] is not None for row in stream),
            "tracker_receipts": tracker_receipts,
            "likelihood_identity_audits": likelihood.calls,
            "rules_provider_gaps": 0,
            "atomic_negative_proof": negative,
        },
        discard_root,
    )


class _FrozenVillainPolicy:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.rows = iter(row for row in decisions if int(row["actor"]) == 1)

    def __call__(self, context: server.DecisionContext) -> int:
        row = next(self.rows)
        _require("live villain revision", row["from_revision"], context.revision)
        _require("live villain prompt", row["prompt_id"], context.prompt_id)
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
                        _require(
                            "live hero revision",
                            row["from_revision"],
                            payload["frame"]["revision"],
                        )
                        websocket.send_json(
                            {"type": "command", "command": deepcopy(row["command"])}
                        )
                        payload = websocket.receive_json()
                        _require("live command status", "accepted", payload["status"])
                        payload = {"frame": payload["update"]["frame"]}
        finally:
            server._new_game_session = original_factory
            server.SESSION_REGISTRY.clear()

        stream: list[dict[str, Any]] = []
        for ordinal, (row, recorded) in enumerate(
            zip(decisions, session.authority_transitions, strict=True)
        ):
            receipt = recorded.semantic_receipt
            if receipt is None:
                raise RuntimeError(f"live transition {ordinal} has no semantic receipt")
            stream.append(
                {
                    "ordinal": ordinal,
                    "actor": int(row["actor"]),
                    "command_id": receipt.command_id,
                    "before_revision": receipt.before_revision,
                    "after_revision": receipt.after_revision,
                    "public_commitment": (
                        None
                        if receipt.public_commitment is None
                        else dict(receipt.public_commitment)
                    ),
                }
            )
            _require("live state before", row["state"]["before"], recorded.state_before)
            _require("live state after", row["state"]["after"], recorded.state_after)
        assert session.trace_id is not None
        persisted = trace_store.load_trace(session.trace_id, trace_dir)
        replay = CanonicalReplayV1.model_validate(persisted["canonical_replay"])
        return {
            "commands": len(stream),
            "identity_stream": stream,
            "identity_stream_sha256": _sha256(_canonical_bytes(stream)),
            "commitments": sum(row["public_commitment"] is not None for row in stream),
        }, replay


def _materialized_discard_proof(root: managym.Env) -> dict[str, Any]:
    viewer = 1
    source_witness = root.state_digest()
    source_cursor = root.semantic_event_cursor()
    source_observation = root.semantic_observation_json(viewer)
    source_payload = json.loads(source_observation)
    _require("discard source decision privacy", None, source_payload["decision"])
    _require(
        "discard source opponent hand privacy",
        [],
        [
            card
            for card in source_payload["viewer_state"]["opponent_cards"]
            if int(card["zone"]) == 1
        ],
    )
    space = PossibleWorldSpace.from_engine(root, viewer)
    _require("discard root support", 484, space.support_size)
    world_index = next(
        world.index for world in space.worlds if dict(world.hand).get("Island", 0) > 0
    )
    branch = space.materialize(
        world_index,
        seed=907,
        refresh_opponent_commitment=True,
    )
    frame = DecisionFrame.from_json(branch.semantic_decision_frame_json())
    offer = next(
        offer
        for offer in frame.offers
        if offer.get("public_commitment") == {"kind": "discard", "card": "Island"}
    )
    transition = apply_semantic_command(
        branch,
        Command(
            command_id="rul11-materialized-discard",
            expected_revision=frame.revision,
            offer_id=int(offer["id"]),
        ),
    )
    _require(
        "materialized discard commitment",
        {"kind": "discard", "card": "Island"},
        dict(transition.receipt.public_commitment or {}),
    )
    _require("discard source witness", source_witness, root.state_digest())
    _require("discard source cursor", source_cursor, root.semantic_event_cursor())
    _require(
        "discard source Observation",
        source_observation,
        root.semantic_observation_json(viewer),
    )
    return {
        "source_revision": 29,
        "viewer": viewer,
        "support_size": space.support_size,
        "public_commitment": dict(transition.receipt.public_commitment or {}),
        "source_match_witness_unchanged": True,
        "source_semantic_event_cursor_unchanged": True,
        "source_viewer_observation_unchanged": True,
        "pre_command_decision_hidden_from_non_actor": True,
        "pre_command_opponent_hand_identities": 0,
    }


def _source_manifest() -> dict[str, Any]:
    paths = (
        "conformance/authored-match-v1/release-stack-ur-vs-gw-seed-0.json",
        "etude/public_commitment_parity.py",
        "etude/server.py",
        "experiments/contracts/int-9-exact-range-v1.json",
        "manabot/belief/likelihood.py",
        "manabot/belief/tracker.py",
        "managym/__init__.pyi",
        "managym/decision.py",
        "managym/possible_worlds.py",
        "managym/src/agent/env.rs",
        "managym/src/agent/structured_offer.rs",
        "managym/src/decision.rs",
        "managym/src/possible_worlds.rs",
        "managym/src/python/bindings.rs",
        "managym/tests/rules/stage2_cards.rs",
        "scripts/verify-public-commitment-parity",
        "tests/belief/test_player.py",
        "tests/belief/test_tracker.py",
        "tests/etude/test_public_commitment_parity.py",
        "tests/sim/test_conditional_search.py",
        "tests/sim/test_exact_range_runner.py",
    )
    files = [
        {"path": path, "sha256": _sha256((ROOT / path).read_bytes())} for path in paths
    ]
    return {
        "algorithm": "ordered-relative-path-and-file-sha256-v1",
        "files": files,
        "sha256": _sha256(_canonical_bytes(files)),
    }


def _frozen_evidence() -> dict[str, Any]:
    rul9_path = ROOT / "experiments/data/rul-9-played-workloads-v1.measurement.json"
    int12_path = ROOT / "protocol/fixtures/advice-belief-conditioned-v1.json"
    _require(
        "RUL-9 measurement bytes",
        RUL9_MEASUREMENT_FILE_SHA256,
        _sha256(rul9_path.read_bytes()),
    )
    rul9 = json.loads(rul9_path.read_text())
    _require(
        "RUL-9 measurement artifact",
        RUL9_MEASUREMENT_ARTIFACT_SHA256,
        rul9["artifact_sha256"],
    )
    _require(
        "INT-12 fixture bytes", INT12_FIXTURE_SHA256, _sha256(int12_path.read_bytes())
    )
    return {
        "rul9_measurement": {
            "path": str(rul9_path.relative_to(ROOT)),
            "artifact_sha256": RUL9_MEASUREMENT_ARTIFACT_SHA256,
            "file_sha256": RUL9_MEASUREMENT_FILE_SHA256,
            "rerun": False,
        },
        "int12_fixture": {
            "path": str(int12_path.relative_to(ROOT)),
            "file_sha256": INT12_FIXTURE_SHA256,
            "rewritten": False,
        },
    }


def build_receipt() -> dict[str, Any]:
    authority = _load_authority()
    decisions = authority["decisions"]
    live, replay = _run_live(decisions)
    _require(
        "persisted replay commands",
        [row["command"] for row in decisions],
        [row.command.model_dump(mode="json") for row in replay.decisions],
    )
    headless, discard_root = _run_engine(decisions, track_beliefs=True)
    replay_decisions = [
        {**row, "command": persisted.command.model_dump(mode="json")}
        for row, persisted in zip(decisions, replay.decisions, strict=True)
    ]
    replay_surface, _ = _run_engine(replay_decisions, track_beliefs=False)
    _require(
        "live/headless identity stream",
        live["identity_stream"],
        headless["identity_stream"],
    )
    _require(
        "live/replay identity stream",
        live["identity_stream"],
        replay_surface["identity_stream"],
    )
    _require("public commitment count", 62, headless["commitments"])
    _require(
        "revision 29 commitment",
        {"kind": "discard", "card": "Island"},
        headless["identity_stream"][29]["public_commitment"],
    )
    _require(
        "revision 40 commitment",
        {"kind": "decline_discard"},
        headless["identity_stream"][40]["public_commitment"],
    )
    for viewer in ("0", "1"):
        _require(
            f"viewer {viewer} tracker records",
            132,
            headless["tracker_receipts"][viewer]["records"],
        )
    _require(
        "consumed commitment identities",
        62,
        sum(
            tracker["consumed_commitments"]
            for tracker in headless["tracker_receipts"].values()
        ),
    )
    if discard_root is None:
        raise RuntimeError("revision-29 discard root was not retained")
    materialized = _materialized_discard_proof(discard_root)
    stream = headless.pop("identity_stream")
    replay_surface.pop("identity_stream")
    live.pop("identity_stream")
    return {
        "version": 1,
        "identity": {
            "matchup": "ur-lessons-vs-gw-allies",
            "seed": 0,
            "authority_receipt_sha256": AUTHORITY_RECEIPT_SHA256,
            "command_tape_sha256": _sha256(
                _canonical_bytes([row["command"] for row in decisions])
            ),
            "semantic_decision_version": SEMANTIC_DECISION_VERSION,
            "relevant_source": _source_manifest(),
        },
        "summary": {
            "commands": 132,
            "commitments": 62,
            "unadmitted_commands": 70,
            "tracker_records_per_viewer": 132,
            "consumed_commitments": 62,
            "rules_provider_gaps": 0,
            "identity_stream_mismatches": 0,
            "negative_proof_mutations": 0,
        },
        "identity_stream": stream,
        "surfaces": {
            "live": live,
            "headless": {
                key: value
                for key, value in headless.items()
                if key != "atomic_negative_proof"
            },
            "persisted_replay": replay_surface,
        },
        "trackers": headless["tracker_receipts"],
        "materialized_hypothesis": materialized,
        "atomic_negative_proof": headless["atomic_negative_proof"],
        "frozen_evidence": _frozen_evidence(),
    }


def write_receipt() -> dict[str, Any]:
    receipt = build_receipt()
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_bytes(_json_bytes(receipt))
    return receipt


@lru_cache(maxsize=1)
def verify_receipt() -> dict[str, Any]:
    checked = json.loads(RECEIPT_PATH.read_text())
    generated = build_receipt()
    generated["identity"]["relevant_source"] = checked["identity"]["relevant_source"]
    if generated != checked:
        raise RuntimeError("checked public-commitment receipt is stale")
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
    print(
        "RUL11_PUBLIC_COMMITMENT_OK "
        f"commands={summary['commands']} "
        f"commitments={summary['commitments']} "
        f"gaps={summary['rules_provider_gaps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
