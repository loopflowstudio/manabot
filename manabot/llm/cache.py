"""Acquire pinned model bytes before timed, offline inference and verify shards."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from huggingface_hub import snapshot_download
from pydantic import BaseModel

from manabot.llm.schema import InferenceConfig


class CachedFile(BaseModel):
    name: str
    bytes: int
    sha256: str


class CacheReceipt(BaseModel):
    model_id: str
    revision: str
    model_path: str
    tokenizer_path: str
    files: list[CachedFile]
    seconds: float
    acquisition_seconds: float | None = None
    verification_seconds: float | None = None
    offline: bool | None = None


class CacheManifest(BaseModel):
    model_id: str
    revision: str
    files: list[CachedFile]


def prepare(
    config: InferenceConfig, manifest: CacheManifest, *, offline: bool = False
) -> CacheReceipt:
    if (manifest.model_id, manifest.revision) != (
        config.model_id,
        config.model_revision,
    ):
        raise ValueError("cache manifest differs from configured model identity")
    started = time.monotonic()
    model_path = Path(
        snapshot_download(
            config.model_id,
            revision=config.model_revision,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
            local_files_only=offline,
        )
    )
    tokenizer_path = snapshot_download(
        config.tokenizer_id,
        revision=config.tokenizer_revision,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"],
        local_files_only=offline,
    )
    acquired = time.monotonic()
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text())
        shards = sorted(set(payload["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    expected = {item.name: item for item in manifest.files}
    if set(shards) != set(expected):
        raise ValueError("cache manifest does not cover the checkpoint's exact shards")
    files: list[CachedFile] = []
    for name in shards:
        path = model_path / name
        digest = sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        actual = digest.hexdigest()
        # Compare publisher metadata pinned in the manifest, never cache filenames.
        if (
            actual != expected[name].sha256
            or path.stat().st_size != expected[name].bytes
        ):
            raise ValueError(f"cached shard checksum mismatch: {name}")
        files.append(CachedFile(name=name, bytes=path.stat().st_size, sha256=actual))
    finished = time.monotonic()
    return CacheReceipt(
        model_id=config.model_id,
        revision=config.model_revision,
        model_path=str(model_path),
        tokenizer_path=tokenizer_path,
        files=files,
        seconds=finished - started,
        acquisition_seconds=acquired - started,
        verification_seconds=finished - acquired,
        offline=offline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = InferenceConfig.model_validate_json(args.config.read_text())
    manifest = CacheManifest.model_validate_json(args.manifest.read_text())
    receipt = prepare(config, manifest, offline=args.offline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(receipt.model_dump_json(indent=2) + "\n")
    print(f"Verified {sum(f.bytes for f in receipt.files):,} weight bytes")


if __name__ == "__main__":
    main()
