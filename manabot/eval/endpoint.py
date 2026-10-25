"""Allocation-owned local transport for spawned evaluation callbacks."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path
import secrets
import tempfile
import threading
from types import TracebackType
from typing import Any
from uuid import uuid4

from manabot.eval.match import play_cell
from manabot.infra.files import exclusive_file_lock
from manabot.sim.execution import EvalExecution
from manabot.sim.serving import InferenceRuntime


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


class CallbackEndpoint:
    """Own the local RPC listener for a single Etude hold operation."""

    def __init__(self, runtime: InferenceRuntime) -> None:

        self._temporary = tempfile.TemporaryDirectory(prefix="eval-")
        address = str(Path(self._temporary.name) / "supervisor.sock")
        authkey = secrets.token_bytes(32)
        supervisor = _CallbackSupervisor(runtime)

        class ServerManager(BaseManager):
            pass

        ServerManager.register(
            "supervisor", callable=lambda: supervisor, exposed=("play_cell",)
        )
        manager = ServerManager(address=address, authkey=authkey)
        self._server = manager.get_server()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.endpoint = InferenceEndpoint(address, authkey, runtime.execution)

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except SystemExit:
            # BaseManager's process-oriented server exits even when hosted in a thread.
            pass

    def __enter__(self) -> InferenceEndpoint:
        return self.endpoint

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.stop_event.set()
        self._server.listener.close()
        self._thread.join(timeout=1)
        self._temporary.cleanup()


@contextmanager
def callback_allocation(
    execution: EvalExecution | None, directory: Path
) -> Iterator[InferenceEndpoint | None]:
    """Retain a fresh attempt and keep its service alive until callbacks finish."""
    if execution is None:
        yield None
    else:
        with InferenceRuntime(execution, directory / uuid4().hex) as runtime:
            with CallbackEndpoint(runtime) as endpoint:
                yield endpoint
