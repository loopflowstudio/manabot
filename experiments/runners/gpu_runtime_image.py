"""Build a GPU runtime from the existing verified Linux wheel bundle.

The image contains dependencies and native build tools, not a scientific source
checkout or checkpoint. The emitted manifest hash admits future sources with the
same lock and project metadata. Docker's digest is the deployment identity.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from experiments.runners.locked_gpu_setup import Bundle, unpack_verified


def prepare(bundle: Bundle, archive: Path, source: Path, context: Path) -> str:
    """Verify before Docker sees bytes; context includes no credentials or jobs."""
    unpack_verified(bundle, archive, context / "verified", source)
    repository = Path(__file__).resolve().parents[2]
    files = {
        "Dockerfile": repository / "deploy/training/Dockerfile",
        "prebuilt_gpu_setup.py": repository
        / "scripts/benchmarks/prebuilt_gpu_setup.py",
        "uv.lock": source / "uv.lock",
        "pyproject.toml": source / "pyproject.toml",
    }
    recipe = hashlib.sha256()
    for name, path in files.items():
        body = path.read_bytes()
        (context / name).write_bytes(body)
        recipe.update(name.encode() + b"\0" + body)
    manifest = json.loads((context / "verified/manifest.json").read_text())
    identity = {
        "format": 1,
        "allowed_lock_sha256": manifest["allowed_lock_sha256"],
        "pyproject_sha256": hashlib.sha256(
            files["pyproject.toml"].read_bytes()
        ).hexdigest(),
        "recipe_sha256": recipe.hexdigest(),
        "bundle_sha256": bundle.sha256,
        "packages": manifest["packages"],
    }
    raw = (json.dumps(identity, sort_keys=True, indent=2) + "\n").encode()
    (context / "runtime.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    bundle = Bundle.read(
        {
            **receipt,
            # Local verified bytes only; no network URL is needed or retained.
            "url": "https://local.invalid/locked-wheels.tar",
        }
    )
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="manabot-runtime-") as temporary:
        context = Path(temporary)
        identity = prepare(bundle, args.bundle, args.source, context)
        shutil.copyfile(context / "runtime.json", args.out / "runtime.json")
        (args.out / "runtime.sha256").write_text(identity + "\n")
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/amd64",
                "--push" if args.push else "--load",
                "--tag",
                args.tag,
                "--metadata-file",
                str((args.out / "image.json").resolve()),
                str(context),
            ],
            check=True,
        )
    print(
        json.dumps(
            {"runtime_sha256": identity, "image_metadata": str(args.out / "image.json")}
        )
    )


if __name__ == "__main__":
    main()
