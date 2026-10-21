"""Publish immutable arena outcomes through Etude's evaluation API.

The existing arena owns play, replay and paired-deal statistics. This module
reads its retained result, verifies the checkpoint bytes and recomputes cohort
admission before publishing. It never plays games, trains, or repairs evidence.
Etude episodes count attempted games; valid outcomes and failures stay distinct.
"""

import hashlib
import json
from pathlib import Path
import uuid

from etude.track.records import ArtifactRef
from etude.track.run import Run, definition

from manabot.arena.models import file_sha256
from manabot.training.monitor_evaluation import MonitorResult


def publish_saved_evaluation(tracker: Run, path: Path) -> str:
    """Publish one retained attempt; a failed attempt cannot overwrite another.

    Returns its source-bound, unique publication key. SDK finish/flush belongs to the
    caller. Republishing creates a separate publication, never more scientific
    samples; consumers deduplicate on result_sha256. The original result and checkpoint remain local authoritative bytes;
    this function does not promise remote archival or rerun canonical replay.
    """
    payload = path.read_bytes()
    result = MonitorResult.model_validate_json(payload)
    if result.status == "running":
        raise ValueError("cannot publish a still-running evaluation as settled")
    admitted = result.model_copy(deep=True)
    admitted.validate_cohort()
    for field in ("status", "win", "draw", "score"):
        if getattr(admitted, field) != getattr(result, field):
            raise ValueError(f"saved evaluation has inconsistent {field}")
    if result.expected_games != len(result.protocol.deal_seeds) * result.protocol.legs:
        raise ValueError("saved evaluation expected games differ from protocol")
    checkpoint = Path(result.artifact["path"]).resolve()
    if (
        checkpoint.stat().st_size != result.artifact["bytes"]
        or file_sha256(checkpoint) != result.artifact["sha256"]
        or result.candidate.checkpoint_sha256 != result.artifact["sha256"]
        or result.candidate.checkpoint_bytes != result.artifact["bytes"]
    ):
        raise ValueError("evaluation checkpoint bytes differ from its subject")
    subject = ArtifactRef(
        uri=checkpoint.as_uri(),
        sha256=result.artifact["sha256"],
        size_bytes=result.artifact["bytes"],
        state="verified",
    )
    meaning = {
        "schema": "manabot-arena-evaluation-v1",
        "protocol": result.protocol.model_dump(mode="json"),
        "arena_key": result.key.model_dump(mode="json"),
        "opponent": result.opponent.model_dump(mode="json"),
        "evaluation_identities": result.evaluation_identities,
        "uncertainty": "95% percentile whole-deal bootstrap conditional on checkpoint; not training-seed uncertainty",
        "episodes": "attempted games, including failed and truncated attempts",
    }
    evaluation_definition = definition(
        "manabot-arena", json.dumps(meaning, sort_keys=True, separators=(",", ":"))
    )
    attempt_digest = hashlib.sha256(payload).hexdigest()
    # A publication retry must not append duplicate episode lines to the
    # SDK's existing evaluation key. The evidence digest remains explicit so
    # later comparison readers can collapse publications of identical evidence.
    key = "arena-" + attempt_digest + "-" + uuid.uuid4().hex
    evaluation = tracker.evaluation(
        evaluation_definition, subject, expected_episodes=result.expected_games, key=key
    )
    try:
        for row in result.rows:
            # Whitelist summary facts. Do not forward arbitrary arena extras,
            # Commands, observation tensors, hidden worlds or private targets.
            evaluation.add(
                {
                    "deal_seed": row.deal_seed,
                    "leg": row.leg,
                    "valid": row.valid,
                    "score": row.score_a if row.valid else None,
                    "terminated": row.terminated,
                    "truncated": row.truncated,
                    "replay_passed": row.replay_passed,
                    "game_seconds": row.game_seconds,
                    "coordinates": result.coordinates.model_dump(mode="json"),
                    "training_seed": result.training_seed,
                    "result_sha256": attempt_digest,
                }
            )
        if result.status != "completed":
            evaluation.fail(
                result.error or "incomplete arena cohort; see retained result"
            )
        else:
            for metric in ("win", "draw", "score"):
                interval = getattr(result, metric)
                assert interval is not None
                evaluation.estimate(
                    metric,
                    interval.mean,
                    interval.lower,
                    interval.upper,
                    n=len(result.protocol.deal_seeds),
                    method="paired-whole-deal-percentile-bootstrap-95; conditional-on-checkpoint",
                )
            evaluation.settle()
    except BaseException as error:
        evaluation.fail(f"publication failed: {type(error).__name__}")
        raise
    return key
