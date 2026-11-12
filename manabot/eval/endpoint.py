"""Allocation-owned local transport for spawned evaluation callbacks."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path
import secrets
import tempfile
import threading
from typing import Any
from uuid import uuid4

from manabot.eval.match import play_cell
from manabot.infra.files import exclusive_file_lock
from manabot.sim.execution import EvalExecution
from manabot.sim.serving_protocol import InferenceRuntime


@dataclass(frozen=True)
class InferenceEndpoint:
    """Picklable private callback endpoint; the allocation still owns all slots.

    Etude executes callbacks in spawned processes. A callback delegates the
    bounded arena panel to this allocation's supervisor, not to another model
    process. The authenticated Unix socket is local to this allocation.
    """

    address: str
    authkey: bytes
    execution: EvalExecution

    def play_cell(
        self, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:

        class ClientManager(BaseManager):
            pass

        ClientManager.register("supervisor")
        manager = ClientManager(address=self.address, authkey=self.authkey)
        manager.connect()
        return manager.supervisor().play_cell(kwargs)


class _CallbackSupervisor:
    def __init__(self, runtime: InferenceRuntime) -> None:
        self.runtime = runtime

    def play_cell(
        self, kwargs: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        directory = Path(kwargs["out_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(directory / ".execution.lock"):
            if (directory / ".sealed").exists():
                raise RuntimeError("evaluation attempt has been sealed")
            (directory / ".execution-started").touch(exist_ok=False)
            return play_cell(**kwargs, inference=self.runtime)


def _serve(server: Any) -> None:
    try:
        server.serve_forever()
    except SystemExit:
        # BaseManager's process-oriented server exits even when hosted in a thread.
        pass


@contextmanager
def serve_callbacks(runtime: InferenceRuntime) -> Iterator[InferenceEndpoint]:
    """Serve one Etude hold operation's callbacks on a private socket.

    The listener and its socket directory last until the block exits.
    """
    with tempfile.TemporaryDirectory(prefix="eval-") as directory:
        address = str(Path(directory) / "supervisor.sock")
        authkey = secrets.token_bytes(32)
        supervisor = _CallbackSupervisor(runtime)

        class ServerManager(BaseManager):
            pass

        ServerManager.register(
            "supervisor", callable=lambda: supervisor, exposed=("play_cell",)
        )
        # BaseManager's server is untyped.
        server: Any = ServerManager(address=address, authkey=authkey).get_server()
        thread = threading.Thread(target=_serve, args=(server,), daemon=True)
        thread.start()
        try:
            yield InferenceEndpoint(address, authkey, runtime.execution)
        finally:
            server.stop_event.set()
            server.listener.close()
            thread.join(timeout=1)


@contextmanager
def callback_allocation(
    execution: EvalExecution | None, directory: Path
) -> Iterator[InferenceEndpoint | None]:
    """Retain a fresh attempt and keep its service alive until callbacks finish."""
    if execution is None:
        yield None
    else:
        with InferenceRuntime(execution, directory / uuid4().hex) as runtime:
            with serve_callbacks(runtime) as endpoint:
                yield endpoint
