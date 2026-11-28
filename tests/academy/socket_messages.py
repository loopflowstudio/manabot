"""Read settled table replies in tests that do not inspect work transitions.

Lifecycle tests read the socket directly to assert early commitment and cached
recovery. Other tests ignore progress and early copies of committed replies.
"""

from typing import Any

from starlette.testclient import WebSocketTestSession


def settled_message(socket: WebSocketTestSession) -> dict[str, Any]:
    while True:
        message = socket.receive_json()
        if message["type"] == "work_status":
            continue
        running = (message.get("work") or {}).get("state") == "running"
        if running and (
            (
                message["type"] == "command_outcome"
                and message.get("status") == "accepted"
            )
            or (
                message["type"] == "observation"
                and (message.get("recovery") or {}).get("reason") == "explicit_resync"
            )
        ):
            continue
        return message


def make_command(
    frame: dict[str, Any], offer: dict[str, Any], command_id: str
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": offer["id"],
        "answers": [],
    }


def find_offer(
    frame: dict[str, Any], *, action_type: str, label: str | None = None
) -> dict[str, Any]:
    return next(
        offer
        for offer in frame["offers"]
        if offer["action_type"] == action_type
        and (label is None or offer["label"] == label)
    )
