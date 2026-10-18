"""Install an unchanged Etude client wheel under the ordinary startup deadline.

The launcher supplies a hashed source archive and its verified pure-Python wheel.
Runtime dependencies still come from the original frozen lock. Only the local
client build is replaced by an offline wheel install; the guardian stays intact.
This module is copied into the standard-library guardian archive before launch,
beside the SDK's own `etude.launch.logging`, so its progress and failures land
in the rental's retained bootstrap log exactly as the stock installer's do.
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

from etude.launch.logging import explain, write_log

SDK_VERSION = "0.1.2"


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
        "sdk_version": SDK_VERSION,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "wheel_sha256": hashlib.sha256(wheel).hexdigest(),
    }:
        raise ValueError("Prebuilt client identity differs")
    return ClientBundle(source, wheel)


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
        wheel = root / f"etude_sdk-{SDK_VERSION}-py3-none-any.whl"
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
            write_log("client install", f"starting {name}")
            try:
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
            except subprocess.SubprocessError as error:
                write_log("client install", explain(error))
                raise
        write_log("client install", "completed")
        (root / "ready").touch()
    return root
