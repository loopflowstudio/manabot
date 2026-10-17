"""Startup observation preserves transport, CLI arguments and failure semantics."""

from collections.abc import Callable
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from experiments.runners import etude_worker_diagnostics as diagnostics


def test_transport_is_unchanged_and_logs_exclude_request_contents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str, bytes | None, dict[str, str]]] = []
    responses = [
        (503, b"private-error"),
        (200, b"private-bootstrap"),
        (200, b"source"),
        (204, b""),
        (200, b"later"),
    ]
    cancelled: list[bool] = []
    monkeypatch.setattr(
        diagnostics.faulthandler,
        "cancel_dump_traceback_later",
        lambda: cancelled.append(True),
    )

    def send(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        calls.append((method, url, body, headers))
        return responses[len(calls) - 1]

    observed = diagnostics.observe_send(
        "https://private.test/bootstrap?signature=secret", send
    )
    requests = [
        ("GET", "https://private.test/bootstrap?signature=secret", None, {}),
        ("GET", "https://private.test/bootstrap?signature=secret", None, {}),
        ("GET", "https://private.test/source?signature=secret", None, {}),
        (
            "POST",
            "https://private.test/write",
            b"secret-payload",
            {"Authorization": "secret-header"},
        ),
        ("GET", "https://private.test/later", None, {}),
    ]
    for request, response in zip(requests, responses, strict=True):
        assert observed(*request) == response
    assert calls == requests and cancelled == [True]
    output = capsys.readouterr().out
    assert not any(
        secret in output
        for secret in ("private", "secret", "signature", "Authorization", "https://")
    )
    events = [
        json.loads(line.removeprefix("ETUDE_WORKER_PHASE "))
        for line in output.splitlines()
    ]
    assert len([event for event in events if event["phase"] == "request_start"]) == 4
    assert [
        event["status"] for event in events if event["phase"] == "request_finished"
    ] == [503, 200, 200, 204]
    assert events[-1]["phase"] == "startup_receipt_written"


def test_transport_exception_is_not_retried_or_replaced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = OSError("https://private.test?secret=credential")
    calls = 0

    def send(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        raise failure

    observed = diagnostics.observe_send("private-bootstrap", send)
    with pytest.raises(OSError) as raised:
        observed("GET", "private-bootstrap", None, {})
    assert raised.value is failure and calls == 1
    output = capsys.readouterr().out
    assert "OSError" in output and "request_failed" in output
    assert "private" not in output and "credential" not in output


def test_worker_keeps_cli_arguments_courier_callbacks_and_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from etude.launch.worker import courier

    cli = ModuleType("etude.cli")
    argv = ["worker", "--bootstrap", "signed-private-bootstrap"]
    roots: list[Path] = []

    def cleanup() -> None:
        raise AssertionError("Observation must not invoke cleanup")

    def sleep(seconds: float) -> None:
        raise AssertionError("Observation must not alter timing")

    def send(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        assert (method, url, body, headers) == ("GET", argv[-1], None, {})
        return 200, b"unchanged-response"

    def original(
        bootstrap_url: str,
        delete_self: Callable[[], None],
        root: Path,
        transport: diagnostics.Send,
        sleeper: Callable[[float], None],
        every: float,
    ) -> None:
        assert bootstrap_url == argv[-1]
        assert delete_self is cleanup and sleeper is sleep and every == 0.7
        assert transport("GET", bootstrap_url, None, {}) == (200, b"unchanged-response")
        roots.append(root)

    def client_main(arguments: list[str]) -> int:
        assert arguments == argv
        courier.run(argv[-1], cleanup, tmp_path, send, sleep, 0.7)
        return 7

    cli.main = client_main
    monkeypatch.setitem(sys.modules, "etude.cli", cli)
    monkeypatch.setattr(courier, "run", original)
    monkeypatch.setattr(diagnostics.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(
        diagnostics.faulthandler, "dump_traceback_later", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        diagnostics.faulthandler, "cancel_dump_traceback_later", lambda: None
    )
    assert diagnostics.main(argv) == 7
    assert roots == [tmp_path] and courier.run is original
