"""Install an unchanged Etude client wheel under the ordinary startup deadline.

The launcher supplies a hashed source archive and its verified pure-Python wheel.
Runtime dependencies still come from the original frozen lock. Only the local
client build is replaced by an offline wheel install; the guardian stays intact.
This module is copied into the standard-library guardian archive before launch.
"""

from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


@dataclass(frozen=True)
class ClientBundle:
    source: bytes
    wheel: bytes


def read_bundle(code: bytes) -> ClientBundle:
    """Reject missing, substituted or unknown bundle members before installation."""
    with zipfile.ZipFile(io.BytesIO(code)) as archive:
        if sorted(archive.namelist()) != ["client.whl", "identity.json", "source.zip"]:
            raise ValueError("Unexpected prebuilt client bundle members")
        identity = json.loads(archive.read("identity.json"))
        source, wheel = archive.read("source.zip"), archive.read("client.whl")
    if not isinstance(identity, dict) or identity != {
        "format": 1,
        "sdk_version": "0.1.1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "wheel_sha256": hashlib.sha256(wheel).hexdigest(),
    }:
        raise ValueError("Prebuilt client identity differs")
    return ClientBundle(source, wheel)


def _phase(name: str) -> None:
    print(
        "ETUDE_BOOT_PHASE " + json.dumps({"phase": name, "unix_seconds": time.time()}),
        flush=True,
    )


def install(code: bytes) -> Path:
    """Retain a failed install; mark ready only after both bounded commands pass."""
    bundle = read_bundle(code)
    base = Path(os.environ.get("ETUDE_HOME") or Path.home() / ".etude") / "programs"
    root = base / hashlib.sha256(code).hexdigest()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "install.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "ready").exists():
            return root
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required for the transferred Etude client")
        with zipfile.ZipFile(io.BytesIO(bundle.source)) as archive:
            for name in archive.namelist():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe client source member")
            archive.extractall(root)
        wheel = root / "etude_sdk-0.1.1-py3-none-any.whl"
        wheel.write_bytes(bundle.wheel)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        }
        # Keep the original install's total 120-second allowance. No packaging
        # repair extends the guardian, startup cutoff or financial deadline.
        deadline = time.monotonic() + 120
        commands = (
            (
                "client_runtime_dependencies",
                [
                    uv,
                    "sync",
                    "--frozen",
                    "--no-dev",
                    "--no-install-project",
                    "--python",
                    sys.executable,
                ],
            ),
            (
                "client_offline_wheel",
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(root / ".venv/bin/python"),
                    "--offline",
                    "--no-index",
                    "--no-deps",
                    str(wheel),
                ],
            ),
        )
        for name, argv in commands:
            _phase(name + "_start")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, 120)
            subprocess.run(
                argv,
                cwd=root,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=remaining,
                check=True,
            )
            _phase(name + "_ready")
        (root / "ready").touch()
    return root
