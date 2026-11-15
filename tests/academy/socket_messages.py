"""Read settled table replies in tests that do not inspect work transitions.

Lifecycle tests read the socket directly to assert early commitment and cached
recovery. Other tests ignore progress and the early copy of an immutable receipt.
"""

from typing import Any

from starlette.testclient import WebSocketTestSession


def settled_message(socket: WebSocketTestSession) -> dict[str, Any]:
    while True:
        message = socket.receive_json()
        if message["type"] == "work_status":
            continue
        if (
            message["type"] == "command_outcome"
            and message.get("status") == "accepted"
            and (message.get("work") or {}).get("state") == "running"
        ):
            continue
        return message
