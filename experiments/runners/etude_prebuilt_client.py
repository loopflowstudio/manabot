"""Prepare and opt into a source-identical Etude client wheel for GPU startup.

Only this launcher's future RunPod rentals use the adapter. Frozen scientific
sources, the SDK's guardian and its runtime dependency lock remain unchanged.
An installed wheel must reproduce every SDK Python file and forwarding metadata.
"""

from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import tomllib
import zipfile

from experiments.runners import etude_client_install

_INSTALLED: str | None = None
_IMPORT = "from etude.launch.providers.runpod.pod import main\n"
_PRELUDE = (
    "from etude.launch.prebuilt_install import install as _prebuilt_install\n"
    "from etude.launch.providers.runpod import pod as _prebuilt_pod\n"
    "_prebuilt_pod.install = _prebuilt_install\n"
)


def _zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(entries.items()):
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, body)
    return output.getvalue()


def prepare(source: bytes, wheel: bytes) -> bytes:
    """Verify the built wheel against the exact transferred SDK, then bundle it."""
    with (
        zipfile.ZipFile(io.BytesIO(source)) as original,
        zipfile.ZipFile(io.BytesIO(wheel)) as built,
    ):
        project = tomllib.loads(original.read("pyproject.toml").decode())["project"]
        if project["name"] != "etude-sdk" or project["version"] != "0.1.1":
            raise ValueError("Unsupported Etude SDK version; review the adapter")
        before = {n: original.read(n) for n in original.namelist() if n.endswith(".py")}
        after = {n: built.read(n) for n in built.namelist() if n.endswith(".py")}
        if not before or before != after:
            raise ValueError("Wheel Python source differs from the transferred SDK")
        for name in ("pyproject.toml", "uv.lock"):
            if original.read(name) != built.read("etude/_package/" + name):
                raise ValueError("Wheel forwarding metadata differs")
        prefix = "etude_sdk-0.1.1.dist-info/"
        metadata = BytesParser().parsebytes(built.read(prefix + "METADATA"))
        if metadata["Name"] != "etude-sdk" or metadata["Version"] != "0.1.1":
            raise ValueError("Wheel package identity differs")
        if (
            "Tag: py3-none-any"
            not in built.read(prefix + "WHEEL").decode().splitlines()
        ):
            raise ValueError("Client wheel must be independent of platform and ABI")
    return _zip(
        {
            "source.zip": source,
            "client.whl": wheel,
            "identity.json": json.dumps(
                {
                    "format": 1,
                    "sdk_version": "0.1.1",
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "wheel_sha256": hashlib.sha256(wheel).hexdigest(),
                },
                sort_keys=True,
            ).encode(),
        }
    )


def enable(path: Path) -> str:
    """Opt in before bootstrap diagnostics; return the retained bundle digest."""
    global _INSTALLED
    from etude.launch.providers.runpod import machines

    code = path.read_bytes()
    digest = hashlib.sha256(code).hexdigest()
    if _INSTALLED is not None:
        if _INSTALLED != digest:
            raise ValueError("A different prebuilt client is already installed")
        return digest
    bundle = etude_client_install.read_bundle(code)
    original_archive = machines.program_archive
    if (
        bundle.source != original_archive()
        or prepare(bundle.source, bundle.wheel) != code
    ):
        raise ValueError("Prebuilt client differs from this launcher's exact SDK")
    if machines._BOOT.count(_IMPORT) != 1 or machines._BOOT.startswith(
        "import json as _json"
    ):
        raise ValueError("Enable the prebuilt client before bootstrap diagnostics")
    with zipfile.ZipFile(io.BytesIO(original_archive(guardian=True))) as archive:
        guardian = {name: archive.read(name) for name in archive.namelist()}
    module = "etude/launch/prebuilt_install.py"
    if module in guardian:
        raise ValueError("Guardian already defines the prebuilt installer")
    guardian[module] = Path(etude_client_install.__file__).read_bytes()
    guardian_code = _zip(guardian)

    def program_archive(*, guardian: bool = False) -> bytes:
        return guardian_code if guardian else code

    machines.program_archive = program_archive
    machines._BOOT = machines._BOOT.replace(_IMPORT, _PRELUDE + _IMPORT)
    _INSTALLED = digest
    return digest
