"""Isolated pinned parser runner. No credentials, HTTP client, or network access."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import socket
import sys
import types
from typing import Never


def deny_network(*args: object, **kwargs: object) -> Never:
    raise AssertionError("network disabled in Arena parser conformance")


def main() -> None:
    socket.socket = deny_network
    socket.create_connection = deny_network
    package = types.ModuleType("seventeenlands")
    api = types.ModuleType("seventeenlands.api_client")
    log = types.ModuleType("seventeenlands.logging_utils")
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    class Sink:
        def __init__(self, host: str) -> None:
            assert host == "offline.invalid"

        def submit_game_result(self, data: dict[str, object]) -> None:
            results.append(data)

        def submit_error_info(self, data: dict[str, object]) -> None:
            errors.append(data)

    api.ApiClient = Sink
    log.get_logger = logging.getLogger
    package.api_client = api
    package.logging_utils = log
    sys.modules.update(
        {
            "seventeenlands": package,
            "seventeenlands.api_client": api,
            "seventeenlands.logging_utils": log,
        }
    )
    source = Path(__file__).parents[1] / "fixtures/seventeenlands/mtga_follower.py"
    spec = importlib.util.spec_from_file_location("pinned_mtga_follower", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    follower = module.Follower(token="offline-only", host="offline.invalid")
    follower.parse_log(sys.argv[1], follow=False)
    print(json.dumps({"results": results, "errors": errors}))


if __name__ == "__main__":
    main()
