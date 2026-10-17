"""Install exact locked Linux wheels before the ordinary GPU setup script.

The launcher supplies an immutable bundle URL, length and hashes in ETUDE_CONFIG.
This stdlib-only helper runs inside the worker's existing setup deadline. It
verifies the bundle, source lock and every wheel before an offline installation;
the original setup still owns project installation, native compilation and GPU
health checks. No dependency version or scientific source is rewritten.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Bundle:
    url: str
    sha256: str
    bytes: int
    manifest_sha256: str

    @classmethod
    def read(cls, value: object) -> "Bundle":
        if not isinstance(value, dict):
            raise ValueError("Missing locked dependency bundle")
        url, digest, size, manifest = (
            value.get(key) for key in ("url", "sha256", "bytes", "manifest_sha256")
        )
        if (
            not isinstance(url, str)
            or urlsplit(url).scheme != "https"
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 < size <= 6 * 1024**3
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or not isinstance(manifest, str)
            or re.fullmatch(r"[a-f0-9]{64}", manifest) is None
        ):
            raise ValueError("Invalid locked dependency bundle identity")
        return cls(url, digest, size, manifest)


def _write(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download(bundle: Bundle, target: Path, output: Path) -> None:
    """Use four bounded range readers; retain byte progress, never the signed URL."""
    chunk_size = 8 * 1024**2
    downloaded = 0
    lock = threading.Lock()
    with target.open("wb") as stream:
        stream.truncate(bundle.bytes)

    def read_range(start: int) -> None:
        nonlocal downloaded
        end = min(start + chunk_size, bundle.bytes) - 1
        request = Request(bundle.url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=30) as response:
            expected = f"bytes {start}-{end}/{bundle.bytes}"
            if (
                response.status != 206
                or response.headers.get("Content-Range") != expected
            ):
                raise ValueError(
                    "Dependency service did not honor the exact byte range"
                )
            body = response.read(end - start + 2)
        if len(body) != end - start + 1:
            raise ValueError("Truncated dependency range")
        with target.open("r+b") as stream:
            stream.seek(start)
            stream.write(body)
        with lock:
            downloaded += len(body)
            _write(
                output / "etude-result.json",
                {
                    "clocks": {
                        "dependency_download_bytes": downloaded,
                        "dependency_expected_bytes": bundle.bytes,
                        "observed_unix_seconds": time.time(),
                    }
                },
            )

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        futures = [
            pool.submit(read_range, start)
            for start in range(0, bundle.bytes, chunk_size)
        ]
        for future in as_completed(futures):
            future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    with target.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != bundle.sha256:
            raise ValueError("Dependency bundle hash differs")


def unpack_verified(
    bundle: Bundle, archive_path: Path, destination: Path, source: Path
) -> int:
    """Bind all installable bytes to both the launch receipt and current uv.lock."""
    with archive_path.open("rb") as stream:
        if (
            archive_path.stat().st_size != bundle.bytes
            or hashlib.file_digest(stream, "sha256").hexdigest() != bundle.sha256
        ):
            raise ValueError("Dependency archive identity differs")
    lock_bytes = (source / "uv.lock").read_bytes()
    locked = tomllib.loads(lock_bytes.decode())
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(
            not member.isfile() for member in members
        ):
            raise ValueError("Dependency archive requires unique regular files")
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise ValueError("Missing dependency manifest")
        with manifest_file:
            raw = manifest_file.read()
        if hashlib.sha256(raw).hexdigest() != bundle.manifest_sha256:
            raise ValueError("Dependency manifest hash differs")
        manifest = json.loads(raw)
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != 1
            or manifest.get("python") != "3.12"
            or manifest.get("platform") != "linux-x86_64"
            or hashlib.sha256(lock_bytes).hexdigest()
            not in manifest.get("allowed_lock_sha256", [])
        ):
            raise ValueError("Dependency bundle does not admit this source lock")
        packages = manifest.get("packages")
        if not isinstance(packages, list) or not packages:
            raise ValueError("Empty dependency bundle")
        expected = {"manifest.json", "requirements.txt"}
        requirements: list[str] = []
        identities: set[str] = set()
        for package in packages:
            if not isinstance(package, dict):
                raise ValueError("Malformed locked wheel")
            name, version, filename, digest, size = (
                package.get(key)
                for key in ("name", "version", "filename", "sha256", "bytes")
            )
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name) is None
                or name in identities
                or not isinstance(version, str)
                or re.fullmatch(r"[A-Za-z0-9.+!_-]+", version) is None
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".whl")
                or not isinstance(digest, str)
                or re.fullmatch(r"[a-f0-9]{64}", digest) is None
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
            ):
                raise ValueError("Malformed locked wheel")
            identities.add(name)
            candidates = [
                p
                for p in locked["package"]
                if p["name"] == name and p["version"] == version
            ]
            if len(candidates) != 1 or not any(
                wheel["hash"] == "sha256:" + digest
                and unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]) == filename
                for wheel in candidates[0].get("wheels", [])
            ):
                raise ValueError("Wheel is absent from the scientific source lock")
            expected.add("wheels/" + filename)
            requirements.append(f"{name}=={version} --hash=sha256:{digest}\n")
        if set(names) != expected:
            raise ValueError("Dependency archive contains unexpected files")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Dependency member escapes its directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            body = archive.extractfile(member)
            assert body is not None
            with body, target.open("wb") as stream:
                while chunk := body.read(1024**2):
                    stream.write(chunk)
    if (destination / "requirements.txt").read_text() != "".join(requirements):
        raise ValueError("Offline requirements differ from verified wheels")
    for package in packages:
        wheel = destination / "wheels" / package["filename"]
        with wheel.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != package["sha256"] or wheel.stat().st_size != package["bytes"]:
            raise ValueError("Locked wheel bytes differ")
    return len(packages)


def main() -> None:
    if (
        sys.version_info[:2] != (3, 12)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        raise ValueError("This setup requires CPython3.12 on Linux x86_64")
    source = Path.cwd()
    if (source / ".venv").exists():
        raise ValueError("Refusing to overwrite an existing project environment")
    config = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())
    if not isinstance(config, dict):
        raise ValueError("Expected an object in ETUDE_CONFIG")
    bundle = Bundle.read(config.get("locked_dependencies"))
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    scratch = Path(tempfile.mkdtemp(prefix="manabot-locked-wheels-"))
    archive, wheels = scratch / "bundle.tar", scratch / "verified"
    _write(
        output / "dependency-setup.json",
        {"phase": "downloading", "sha256": bundle.sha256, "bytes": bundle.bytes},
    )
    download(bundle, archive, output)
    count = unpack_verified(bundle, archive, wheels, source)
    archive.unlink()
    subprocess.run(["uv", "venv", "--python", sys.executable, ".venv"], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(source / ".venv/bin/python"),
            "--offline",
            "--no-index",
            "--find-links",
            str(wheels / "wheels"),
            "--no-deps",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "-r",
            str(wheels / "requirements.txt"),
        ],
        check=True,
    )
    _write(
        output / "dependency-setup.json",
        {
            "phase": "offline_install_completed",
            "sha256": bundle.sha256,
            "wheels": count,
        },
    )
    shutil.rmtree(scratch)
    os.execvp("bash", ["bash", "scripts/benchmarks/setup-worker-gpu.sh"])


if __name__ == "__main__":
    main()
