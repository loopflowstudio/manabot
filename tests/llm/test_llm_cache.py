"""Cache admission verifies published shard hashes before offline inference."""

from hashlib import sha256
from itertools import count
import json
from pathlib import Path

import pytest

from manabot.llm import cache
from manabot.llm.schema import InferenceConfig


def test_cache_rejects_corruption_without_trusting_cache_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = InferenceConfig.model_validate_json(
        Path("configs/llm/qwen3.5-9b.json").read_text()
    )
    payload = b"test checkpoint bytes"
    blob = tmp_path / ("f" * 64)
    blob.write_bytes(payload)
    (tmp_path / "model.safetensors").symlink_to(blob)

    def snapshot(*args: object, **kwargs: object) -> str:
        assert kwargs["local_files_only"] is True
        return str(tmp_path)

    monkeypatch.setattr(cache, "snapshot_download", snapshot)
    manifest = cache.CacheManifest(
        model_id=config.model_id,
        revision=config.model_revision,
        files=[
            cache.CachedFile(
                name="model.safetensors",
                bytes=len(payload),
                sha256=sha256(payload).hexdigest(),
            )
        ],
    )
    clock = count()
    with monkeypatch.context() as timing:
        timing.setattr(cache.time, "monotonic", lambda: float(next(clock)))
        receipt = cache.prepare(config, manifest, offline=True)
    assert receipt.files == manifest.files
    assert receipt.offline is True
    assert receipt.acquisition_seconds == receipt.verification_seconds == 1
    assert receipt.seconds == 2
    blob.write_bytes(b"bad bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cache.prepare(config, manifest, offline=True)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="exact shards"):
        cache.prepare(config, manifest, offline=True)
