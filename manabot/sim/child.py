"""Killable evaluation children launched without the parent's `__main__`.

`multiprocessing` spawn re-imports the parent's main module in every child, so
an evaluator whose main imports Torch or Etude makes every game child pay for
it. `ChildProcess` instead runs `python -m manabot.sim.child <target> <fd>`:
this module imports only the target `module:function`, receives its arguments
as the first message on a socketpair `Connection`, then calls
`function(connection, *arguments)`. Further descriptors named in `pass_fds`
keep their numbers in the child, so arguments may name them. The child starts from
`threads.child_environment()`, so its library pools are single-threaded.

Exit codes follow `multiprocessing`: negative values name the killing signal.
Torch-free.
"""

from __future__ import annotations

import importlib
from multiprocessing.connection import Connection
import socket
import subprocess
import sys

from manabot.sim.threads import child_environment


class ChildProcess:
    """One child process and the parent's end of its connection."""

    def __init__(
        self,
        target: str,
        arguments: tuple[object, ...],
        *,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        module, separator, function = target.partition(":")
        if not (module and separator and function):
            raise ValueError(f"child target must be module:function, got {target!r}")
        parent, child = socket.socketpair()
        self.connection = Connection(parent.detach())
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", __name__, target, str(child.fileno())],
                pass_fds=(child.fileno(), *pass_fds),
                env=child_environment(),
            )
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()
        self.pid = self._process.pid
        try:
            self.connection.send(arguments)
        except OSError:
            # The child exited before reading; its exit code reports why.
            pass

    @property
    def exitcode(self) -> int | None:
        return self._process.poll()

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def kill(self) -> None:
        if self.is_alive():
            self._process.kill()

    def join(self, timeout: float | None = None) -> None:
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        """Kill if still running, reap, and close the parent's connection."""
        self.kill()
        self._process.wait()
        self.connection.close()


def _main() -> None:
    target, descriptor = sys.argv[1], int(sys.argv[2])
    module, _, function = target.partition(":")
    connection = Connection(descriptor)
    arguments = connection.recv()
    getattr(importlib.import_module(module), function)(connection, *arguments)


if __name__ == "__main__":
    _main()
