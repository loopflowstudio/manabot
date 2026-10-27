"""Bounded L4 game proof using a verified prebuilt engine and a warm model cache."""

from decimal import Decimal
import json
from pathlib import Path
import subprocess

from etude.jobs import training
from etude.launch.records import Input, Job, Variable

from manabot.infra.artifacts import StoredArtifact, split_bucket_uri
from manabot.verify.llm_game import GameConfig, run


def runs(native_artifact: str, native_source: str) -> Job:
    """Receipts identify uploaded bytes and the CI commit that built the engine."""
    artifact = StoredArtifact.model_validate_json(Path(native_artifact).read_text())
    provenance = json.loads(Path(native_source).read_text())
    if (artifact.sha256, artifact.bytes) != (provenance["sha256"], provenance["bytes"]):
        raise ValueError("native artifact and build provenance disagree")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            provenance["source_commit"],
            "HEAD",
            "--",
            "managym/src",
            "managym/Cargo.toml",
            "managym/Cargo.lock",
            "managym/build.rs",
            "content",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    scheme, bucket, key = split_bucket_uri(artifact.uri)
    if (scheme, bucket) != ("r2", "manabot"):
        raise ValueError(
            "this launch recipe requires the configured manabot R2 storage"
        )
    return training(
        run,
        config=GameConfig(inference=Path("configs/llm/qwen3.5-9b-game.json")),
        repo=".",
        revision="HEAD",
        name="qwen-l4-full-game",
        max_hours=2,
        max_usd=Decimal("1.50"),
        max_attempts=1,
        setup="sh scripts/llm-game-setup.sh",
        cleanup="sh scripts/llm-game-collect.sh",
        cleanup_seconds=120,
        inputs=(
            Input(
                name="native.so",
                key=key,
                sha256=artifact.sha256,
                size_bytes=artifact.bytes,
            ),
        ),
        environment=(
            Variable(name="HF_HOME", value="/workspace/manabot-hf"),
            Variable(name="UV_CACHE_DIR", value="/workspace/manabot-uv"),
            # Etude's typed command uses uv run. Setup has already installed
            # dependencies and copied the matching engine; do not sync/build again.
            Variable(name="UV_NO_SYNC", value="1"),
            Variable(name="OMP_NUM_THREADS", value="4"),
            Variable(name="HF_HUB_DISABLE_PROGRESS_BARS", value="1"),
        ),
    )
