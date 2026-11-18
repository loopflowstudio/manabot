"""Replay retained choices through naming and explicit source-payment decisions."""

from collections.abc import Iterator, Sequence
import json

from manabot.managym.decision import SEMANTIC_DECISION_VERSION
import managym


def assert_retained_frame(
    current_frame: str, retained_frame: str, *, added_payments: int = 0
) -> None:
    """Compare schema-7 fields with the exact count of inserted payment revisions."""
    current = json.loads(current_frame)
    retained = json.loads(retained_frame)
    assert retained.pop("schema_version") == 7
    assert current.pop("schema_version") == SEMANTIC_DECISION_VERSION
    assert current.pop("fingerprint") != retained.pop("fingerprint")
    assert current["revision"] == retained["revision"] + added_payments
    retained["revision"] = current["revision"]
    assert current == retained


def source_payment_tape(env: managym.Env, actions: Sequence[int]) -> Iterator[int]:
    """Replay old choices with explicit automatic payment at new source decisions.

    The caller steps the scalar env for every yielded index, and can feed the
    same expanded tape to a vector. Original receipts remain unchanged. Existing
    mana-ability choices keep their old order after removing new source taps.
    """
    for action in (*actions, None):
        while True:
            actor = env.current_agent_index()
            if actor is None:
                remaining = []
                break
            raw = env.observation_for_player(actor)
            offered = raw.action_space.actions
            has_sources = (
                raw.action_space.action_space_type == managym.ActionSpaceEnum.PAY_OR_NOT
                and any(
                    a.action_type == managym.ActionEnum.TAP_FOR_COST for a in offered
                )
            )
            remaining = [
                i
                for i, a in enumerate(offered)
                if not has_sources or a.action_type != managym.ActionEnum.TAP_FOR_COST
            ]
            if not (
                has_sources
                and len(remaining) == 1
                and offered[remaining[0]].action_type == managym.ActionEnum.PAY_COST
            ):
                break
            yield remaining[0]
        if action is not None:
            yield remaining[action]
