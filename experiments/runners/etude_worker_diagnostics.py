"""Observe the Etude worker before its first durable startup receipt.

The launcher writes this stdlib entry point beside the installed client, leaving
SDK files unchanged. Delayed imports make import stalls visible. Request events
contain only timing, sizes and status; stack dumps contain no local values.
Transport, retry rules, guardian and worker deadlines remain owned by Etude.
"""

from collections.abc import Callable
import faulthandler
import hashlib
import json
from pathlib import Path
import sys
import time

type Send = Callable[[str, str, bytes | None, dict[str, str]], tuple[int, bytes]]


def phase(name: str, **fields: str | int | float) -> None:
    print(
        "ETUDE_WORKER_PHASE "
        + json.dumps({"phase": name, "unix_seconds": time.time(), **fields}),
        flush=True,
    )


def observe_send(bootstrap_url: str, send: Send) -> Send:
    """Forward requests unchanged; stop observing after startup's successful POST."""
    finished = False
    sequence = 0

    def observed(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        nonlocal finished, sequence
        if finished:
            return send(method, url, body, headers)
        sequence += 1
        kind = (
            "bootstrap"
            if url == bootstrap_url
            else "required_file"
            if method == "GET"
            else "startup_receipt"
        )
        start = time.monotonic()
        phase("request_start", request=sequence, operation=kind, method=method)
        try:
            status, response = send(method, url, body, headers)
        except Exception as error:
            phase(
                "request_failed",
                request=sequence,
                operation=kind,
                seconds=time.monotonic() - start,
                error_type=type(error).__name__,
            )
            raise
        phase(
            "request_finished",
            request=sequence,
            operation=kind,
            seconds=time.monotonic() - start,
            status=status,
            bytes=len(response),
        )
        # In this pinned SDK the first successful POST is the durable start
        # intent. Subsequent courier traffic belongs to the ordinary job logs.
        if method == "POST" and status in (200, 201, 204):
            finished = True
            faulthandler.cancel_dump_traceback_later()
            phase("startup_receipt_written")
        return status, response

    return observed


def main(argv: list[str] | None = None) -> int:
    phase(
        "entrypoint",
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        # Keep these imports inside the observed interval: an import stall is
        # one of the boundaries that the outer provider bootstrap cannot see.
        phase("cli_import_start")
        from etude.cli import main as client_main

        phase("cli_import_ready")
        from etude.launch.worker import courier

        phase("courier_import_ready")
        original = courier.run

        def run(
            bootstrap_url: str,
            delete_self: Callable[[], None],
            root: Path = courier.ROOT,
            send: Send = courier._send,
            sleep: Callable[[float], None] = time.sleep,
            every: float = 3.0,
        ) -> None:
            phase("courier_start")
            original(
                bootstrap_url,
                delete_self,
                root,
                observe_send(bootstrap_url, send),
                sleep,
                every,
            )

        courier.run = run
        try:
            return client_main(sys.argv[1:] if argv is None else argv)
        finally:
            courier.run = original
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(main())
