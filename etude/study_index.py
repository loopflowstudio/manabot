"""Build a deterministic Study decision index from canonical recorded decisions.

The module is deliberately a pure consumer. It accepts exact experience
protocol objects and never reads legacy trace events, reconstructs replay
frames, or invokes rules/search code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from pydantic import ValidationError, model_validator

from etude.experience_protocol import OfferVerb, PresentationImportance, ProtocolModel
from etude.study_protocol import (
    LandmarkReason,
    RankedStudyLandmark,
    RecordedDecision,
    RecordedDecisionInput,
    StudyDecision,
    StudyDecisionIndex,
    StudyDecisionKind,
    StudyIdentity,
)

DECISION_ID_DOMAIN = b"etude.study-decision.v1\0"
REPO_ROOT = Path(__file__).parents[1]
DEFAULT_EXPECTED_INDEX = (
    REPO_ROOT / "protocol" / "fixtures" / "study-decision-index-curated.json"
)
RECEIPT_COMMAND = (
    "uv run python -m etude.study_index "
    "protocol/fixtures/recorded-match-decisions-curated.json "
    "--identity protocol/fixtures/study-index-identity-curated.json "
    "--verify --repeats 1000 "
    "--semantic-receipt experiments/data/w2-220-study-decision-index-v1.json "
    "--observations scratch/study-index-observations.json"
)

_KIND_BY_ACTION_SPACE = {
    "PRIORITY": StudyDecisionKind.PRIORITY,
    "CHOOSE_TARGET": StudyDecisionKind.TARGETING,
    "DECLARE_ATTACKER": StudyDecisionKind.ATTACK,
    "DECLARE_BLOCKER": StudyDecisionKind.BLOCK,
}
_REASON_ORDER = list(LandmarkReason)
_IMPORTANCE_WEIGHT = {
    PresentationImportance.AMBIENT: 0,
    PresentationImportance.NORMAL: 1,
    PresentationImportance.EMPHASIZED: 4,
    PresentationImportance.CRITICAL: 8,
}


class StudyIndexObservations(ProtocolModel):
    schema_version: int
    observed_at: str
    repeats: int
    p50_ms: float
    p95_ms: float

    @model_validator(mode="after")
    def validate_observation(self) -> "StudyIndexObservations":
        if self.schema_version != 1:
            raise ValueError("unsupported observation schema version")
        if self.repeats <= 0:
            raise ValueError("observation repeats must be positive")
        if self.p50_ms < 0 or self.p95_ms < self.p50_ms:
            raise ValueError("observation percentiles are not sane")
        try:
            parsed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("observation timestamp must be ISO 8601") from error
        if parsed.tzinfo is None:
            raise ValueError("observation timestamp must carry a timezone")
        return self


@dataclass(frozen=True)
class _Candidate:
    decision: StudyDecision
    family: str
    reasons: tuple[LandmarkReason, ...]
    importance: int
    deliberate_priority: int
    legal_breadth: int
    episode_size: int

    @property
    def sort_key(self) -> tuple[int, int, int, int, int, str]:
        return (
            -self.importance,
            -self.deliberate_priority,
            -self.legal_breadth,
            -self.episode_size,
            self.decision.event_cursor,
            self.decision.id,
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize semantic JSON identically on every run."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def source_replay_sha256(recorded: RecordedDecisionInput) -> str:
    return semantic_sha256(recorded.model_dump(mode="json", exclude_unset=True))


def _decision_id(
    recorded: RecordedDecision,
    viewer: int,
    source_digest: str,
) -> str:
    prompt = recorded.frame.prompt
    if prompt is None:  # Closed input validation gives a clearer boundary error.
        raise ValueError(f"decision {recorded.ordinal}: decision frame has no prompt")
    payload = {
        "event_cursor": recorded.event_cursor,
        "frame_hash": recorded.frame.frame_hash,
        "match_id": recorded.frame.match_id,
        "offer_id": recorded.offer.id,
        "played_command_id": recorded.played.command_id,
        "prompt_id": prompt.id,
        "revision": recorded.frame.revision,
        "source_replay_sha256": source_digest,
        "viewer": viewer,
    }
    return hashlib.sha256(
        DECISION_ID_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def _classify(recorded: RecordedDecision) -> StudyDecisionKind:
    return _KIND_BY_ACTION_SPACE.get(
        recorded.frame.action_space, StudyDecisionKind.OTHER
    )


def build_study_index(
    recorded: RecordedDecisionInput,
    identity: StudyIdentity,
) -> StudyDecisionIndex:
    """Index every validated decision, then rank a separate landmark subset."""

    if identity.source_replay_id != recorded.source_replay_id:
        raise ValueError("source replay id does not match recorded-decision input")
    source_digest = source_replay_sha256(recorded)
    if identity.source_replay_sha256 != source_digest:
        raise ValueError(
            "source replay digest must bind the validated viewer-safe input"
        )

    decisions: list[StudyDecision] = []
    for source in recorded.decisions:
        prompt = source.frame.prompt
        if prompt is None:
            raise ValueError(f"decision {source.ordinal}: decision frame has no prompt")
        decisions.append(
            StudyDecision(
                id=_decision_id(source, prompt.actor, source_digest),
                ordinal=source.ordinal,
                viewer=prompt.actor,
                event_cursor=source.event_cursor,
                automatic=source.automatic,
                kind=_classify(source),
                frame=source.frame,
                offer=source.offer,
                played=source.played,
            )
        )

    candidates = _build_candidates(decisions, recorded.decisions)
    landmarks = _select_landmarks(candidates)
    return StudyDecisionIndex(
        version=1,
        identity=identity,
        decisions=decisions,
        landmarks=landmarks,
    )


def _event_importance(source: RecordedDecision) -> int:
    return min(
        sum(_IMPORTANCE_WEIGHT[event.importance] for event in source.presentation),
        32,
    )


def _has_public_semantic_impact(source: RecordedDecision) -> bool:
    return any(
        event.importance
        in {PresentationImportance.EMPHASIZED, PresentationImportance.CRITICAL}
        for event in source.presentation
    )


def _has_public_stack_reference(source: RecordedDecision) -> bool:
    if source.frame.projection.agent.stack or source.frame.projection.opponent.stack:
        return True

    def contains_stack(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("kind") == "stack":
                return True
            return any(contains_stack(child) for child in value.values())
        if isinstance(value, list):
            return any(contains_stack(child) for child in value)
        return False

    for event in source.presentation:
        kind = event.kind.model_dump(mode="json")
        if kind["kind"] in {"cast", "resolved"} or contains_stack(kind):
            return True
    return False


def _ordered_reasons(
    reasons: Iterable[LandmarkReason],
) -> tuple[LandmarkReason, ...]:
    return tuple(sorted(set(reasons), key=_REASON_ORDER.index))


def _single_candidate(
    decision: StudyDecision,
    source: RecordedDecision,
) -> _Candidate | None:
    if decision.automatic or len(decision.frame.offers) <= 1:
        return None

    reasons: list[LandmarkReason] = []
    family: str
    deliberate = 0
    if decision.kind is StudyDecisionKind.PRIORITY:
        family = "priority"
        if source.offer.verb is not OfferVerb.PASS_PRIORITY:
            reasons.append(LandmarkReason.PRIORITY_COMMITMENT)
            deliberate = 1
        if _has_public_stack_reference(source):
            reasons.append(LandmarkReason.PRIORITY_RESPONSE)
        if not reasons:
            return None
    elif decision.kind is StudyDecisionKind.TARGETING:
        family = "targeting"
        reasons.append(LandmarkReason.TARGET_SELECTION)
    else:
        return None

    reasons.append(LandmarkReason.BRANCHING_CHOICE)
    importance = _event_importance(source)
    if _has_public_semantic_impact(source):
        reasons.append(LandmarkReason.PUBLIC_SEMANTIC_IMPACT)
    return _Candidate(
        decision=decision,
        family=family,
        reasons=_ordered_reasons(reasons),
        importance=importance,
        deliberate_priority=deliberate,
        legal_breadth=min(len(decision.frame.offers), 16),
        episode_size=1,
    )


def _combat_episode_key(decision: StudyDecision) -> tuple[int, int, str, str]:
    turn = decision.frame.projection.turn
    return (decision.viewer, turn.turn_number, turn.step, decision.kind.value)


def _combat_candidate(
    members: Sequence[tuple[StudyDecision, RecordedDecision]],
) -> _Candidate | None:
    representative, _ = members[0]
    if representative.automatic or len(representative.frame.offers) <= 1:
        return None
    if representative.kind is StudyDecisionKind.ATTACK:
        base_reason = LandmarkReason.ATTACK_DECLARATION
    elif representative.kind is StudyDecisionKind.BLOCK:
        base_reason = LandmarkReason.BLOCK_DECLARATION
    else:
        return None

    reasons = [base_reason, LandmarkReason.BRANCHING_CHOICE]
    importance = min(sum(_event_importance(source) for _, source in members), 32)
    if any(_has_public_semantic_impact(source) for _, source in members):
        reasons.append(LandmarkReason.PUBLIC_SEMANTIC_IMPACT)
    return _Candidate(
        decision=representative,
        family="combat",
        reasons=_ordered_reasons(reasons),
        importance=importance,
        deliberate_priority=0,
        legal_breadth=min(
            sum(len(decision.frame.offers) for decision, _ in members), 16
        ),
        episode_size=min(len(members), 16),
    )


def _build_candidates(
    decisions: Sequence[StudyDecision],
    sources: Sequence[RecordedDecision],
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    combat_episode: list[tuple[StudyDecision, RecordedDecision]] = []

    def flush_combat() -> None:
        nonlocal combat_episode
        if combat_episode:
            candidate = _combat_candidate(combat_episode)
            if candidate is not None:
                candidates.append(candidate)
            combat_episode = []

    for decision, source in zip(decisions, sources, strict=True):
        if decision.kind in {StudyDecisionKind.ATTACK, StudyDecisionKind.BLOCK}:
            if combat_episode and _combat_episode_key(
                combat_episode[-1][0]
            ) != _combat_episode_key(decision):
                flush_combat()
            combat_episode.append((decision, source))
            continue
        flush_combat()
        candidate = _single_candidate(decision, source)
        if candidate is not None:
            candidates.append(candidate)
    flush_combat()
    return candidates


def _select_landmarks(
    candidates: Sequence[_Candidate],
) -> list[RankedStudyLandmark]:
    ordered = sorted(candidates, key=lambda candidate: candidate.sort_key)
    selected: dict[str, _Candidate] = {}
    family_counts: Counter[str] = Counter()

    for family in ("priority", "targeting", "combat"):
        candidate = next(
            (candidate for candidate in ordered if candidate.family == family), None
        )
        if candidate is not None:
            selected[candidate.decision.id] = candidate
            family_counts[family] += 1

    for candidate in ordered:
        if len(selected) >= 7:
            break
        if candidate.decision.id in selected or family_counts[candidate.family] >= 3:
            continue
        selected[candidate.decision.id] = candidate
        family_counts[candidate.family] += 1

    if len(ordered) >= 3 and len(selected) < 3:
        for candidate in ordered:
            if candidate.decision.id not in selected:
                selected[candidate.decision.id] = candidate
                if len(selected) == 3:
                    break

    ranked = sorted(selected.values(), key=lambda candidate: candidate.sort_key)
    return [
        RankedStudyLandmark(
            decision_id=candidate.decision.id,
            rank=rank,
            reasons=list(candidate.reasons),
        )
        for rank, candidate in enumerate(ranked, start=1)
    ]


def _same_model_json(left: Any, right: Any) -> bool:
    return left.model_dump(mode="json", exclude_unset=True) == right.model_dump(
        mode="json", exclude_unset=True
    )


def _boundary_checks(
    recorded_json: dict[str, Any],
    identity_json: dict[str, Any],
) -> dict[str, bool]:
    private_hand = json.loads(json.dumps(recorded_json))
    secret_card = {
        "id": 99,
        "registry_key": 99,
        "name": "Secret Counterspell",
        "zone": "HAND",
        "owner_id": 0,
        "power": 0,
        "toughness": 0,
        "mana_value": 2,
        "types": {
            "is_creature": False,
            "is_land": False,
            "is_spell": True,
            "is_artifact": False,
            "is_enchantment": False,
            "is_planeswalker": False,
            "is_battle": False,
        },
    }
    if private_hand["decisions"]:
        private_hand["decisions"][0]["frame"]["projection"]["opponent"]["hand"] = [
            secret_card
        ]
    else:
        private_hand["opponent_private_hand"] = [secret_card]
    try:
        RecordedDecisionInput.model_validate(private_hand)
    except ValidationError:
        private_rejected = True
    else:
        private_rejected = False

    rng_sidecar = json.loads(json.dumps(recorded_json))
    rng_sidecar["rng_seed"] = 377
    try:
        RecordedDecisionInput.model_validate(rng_sidecar)
    except ValidationError:
        rng_rejected = True
    else:
        rng_rejected = False

    recorded = RecordedDecisionInput.model_validate(recorded_json)
    wrong_digest = StudyIdentity.model_validate(
        {**identity_json, "source_replay_sha256": "0" * 64}
    )
    try:
        build_study_index(recorded, wrong_digest)
    except ValueError:
        raw_digest_rejected = True
    else:
        raw_digest_rejected = False

    identity = StudyIdentity.model_validate(identity_json)
    first = build_study_index(recorded, identity)
    second = build_study_index(recorded, identity)
    stable = canonical_json_bytes(
        first.model_dump(mode="json", exclude_unset=True)
    ) == (canonical_json_bytes(second.model_dump(mode="json", exclude_unset=True)))
    return {
        "opponent_private_hand_rejected": private_rejected,
        "raw_authority_digest_rejected": raw_digest_rejected,
        "rng_sidecar_rejected": rng_rejected,
        "viewer_boundary_stable": stable,
    }


def build_semantic_receipt(
    recorded: RecordedDecisionInput,
    identity: StudyIdentity,
    index: StudyDecisionIndex,
    *,
    repeats: int,
    repeat_artifact_digests: int,
    boundary_checks: dict[str, bool],
) -> dict[str, Any]:
    restored = sum(
        _same_model_json(decision.frame, source.frame)
        and _same_model_json(decision.offer, source.offer)
        and _same_model_json(decision.played, source.played)
        for decision, source in zip(index.decisions, recorded.decisions, strict=True)
    )
    decision_count = len(index.decisions)
    candidates = _build_candidates(index.decisions, recorded.decisions)
    kind_counts = Counter(decision.kind.value for decision in index.decisions)
    viewer_counts = Counter(str(decision.viewer) for decision in index.decisions)
    reason_counts = Counter(
        reason.value for landmark in index.landmarks for reason in landmark.reasons
    )
    if decision_count == 0:
        ranking_status = "no_recorded_decisions"
    elif len(candidates) < 3:
        ranking_status = "insufficient_supported_landmarks"
    else:
        ranking_status = "ranked"

    index_json = index.model_dump(mode="json", exclude_unset=True)
    return {
        "artifact_sha256": semantic_sha256(index_json),
        "automatic_decisions": sum(decision.automatic for decision in index.decisions),
        "boundary_checks": boundary_checks,
        "command": RECEIPT_COMMAND,
        "completeness_ratio": 1.0
        if decision_count == recorded.decision_count
        else decision_count / recorded.decision_count,
        "content_pack": identity.content_pack.model_dump(mode="json"),
        "decision_count": decision_count,
        "decisions_by_kind": {
            kind.value: kind_counts[kind.value] for kind in StudyDecisionKind
        },
        "decisions_by_viewer": dict(sorted(viewer_counts.items())),
        "duplicate_decision_ids": decision_count
        - len({decision.id for decision in index.decisions}),
        "eligible_landmark_candidates": len(candidates),
        "engine": identity.engine.model_dump(mode="json"),
        "exact_restoration_ratio": 1.0
        if decision_count == 0
        else restored / decision_count,
        "forced_decisions": sum(
            len(decision.frame.offers) <= 1 for decision in index.decisions
        ),
        "landmark_count": len(index.landmarks),
        "landmarks": [landmark.model_dump(mode="json") for landmark in index.landmarks],
        "landmarks_by_reason": {
            reason.value: reason_counts[reason.value] for reason in LandmarkReason
        },
        "ranking_status": ranking_status,
        "repeat_artifact_digests": repeat_artifact_digests,
        "repeats": repeats,
        "schema_version": 1,
        "source_replay_id": recorded.source_replay_id,
        "source_replay_sha256": identity.source_replay_sha256,
    }


def measure_builds(
    recorded: RecordedDecisionInput,
    identity: StudyIdentity,
    repeats: int,
) -> tuple[StudyIndexObservations, int]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    build_study_index(recorded, identity)
    durations_ms: list[float] = []
    digests: set[str] = set()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        index = build_study_index(recorded, identity)
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        digests.add(semantic_sha256(index.model_dump(mode="json", exclude_unset=True)))
    durations_ms.sort()
    p95_index = min(math.ceil(0.95 * repeats) - 1, repeats - 1)
    observations = StudyIndexObservations(
        schema_version=1,
        observed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        repeats=repeats,
        p50_ms=round(statistics.median(durations_ms), 6),
        p95_ms=round(durations_ms[p95_index], 6),
    )
    return observations, len(digests)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _compare_checked(path: Path, actual: bytes, label: str) -> None:
    expected = path.read_bytes()
    if expected != actual:
        raise ValueError(f"checked {label} drifted: {path}")


def _write_checked(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recorded_decisions", type=Path)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--expected-index", type=Path, default=DEFAULT_EXPECTED_INDEX)
    parser.add_argument("--semantic-receipt", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    recorded_json = _load_json(args.recorded_decisions)
    identity_json = _load_json(args.identity)
    recorded = RecordedDecisionInput.model_validate(recorded_json)
    identity = StudyIdentity.model_validate(identity_json)
    index = build_study_index(recorded, identity)
    observations, digest_count = measure_builds(recorded, identity, args.repeats)
    boundary_checks = _boundary_checks(recorded_json, identity_json)
    if not all(boundary_checks.values()):
        raise ValueError("viewer-safe boundary checks did not all pass")
    receipt = build_semantic_receipt(
        recorded,
        identity,
        index,
        repeats=args.repeats,
        repeat_artifact_digests=digest_count,
        boundary_checks=boundary_checks,
    )
    index_json = index.model_dump(mode="json", exclude_unset=True)

    if args.write:
        _write_checked(args.expected_index, index_json)
        if args.semantic_receipt is not None:
            _write_checked(args.semantic_receipt, receipt)
    elif args.verify:
        _compare_checked(
            args.expected_index, pretty_json_bytes(index_json), "decision index"
        )
        if args.semantic_receipt is None:
            raise ValueError("--verify requires --semantic-receipt")
        _compare_checked(
            args.semantic_receipt,
            pretty_json_bytes(receipt),
            "semantic receipt",
        )

    if args.observations is not None:
        observation_json = observations.model_dump(mode="json")
        StudyIndexObservations.model_validate(observation_json)
        _write_checked(args.observations, observation_json)

    print(
        f"decisions={len(index.decisions)} landmarks={len(index.landmarks)} "
        f"completeness={receipt['completeness_ratio']:.0%} "
        f"restoration={receipt['exact_restoration_ratio']:.0%} "
        f"digest={receipt['artifact_sha256']}"
    )
    for landmark in index.landmarks:
        reasons = ",".join(reason.value for reason in landmark.reasons)
        print(f"rank={landmark.rank} decision={landmark.decision_id} reasons={reasons}")
    print(
        f"p50_ms={observations.p50_ms:.6f} p95_ms={observations.p95_ms:.6f} "
        f"observed_at={observations.observed_at}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
