"""One server-configured trained opponent, pinned to its retained artifact."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Opponent:
    name: str
    producer: str
    sha256: str
    checkpoint: Path
    deterministic: bool
    content_manifest: dict
    decks: dict
    bot_id: str | None = None

    def identity(self) -> dict:
        return {
            "name": self.name,
            "producer": self.producer,
            "sha256": self.sha256,
            **({"bot_id": self.bot_id} if self.bot_id else {}),
        }

    def verify(self) -> None:
        try:
            actual = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(
                "The configured opponent checkpoint is unavailable."
            ) from exc
        if actual != self.sha256:
            raise ValueError(
                "The configured opponent checkpoint changed; restore the pinned artifact."
            )


@lru_cache(maxsize=1)
def _load(path: str) -> Opponent:
    manifest = Path(path).resolve()
    data = json.loads(manifest.read_text())
    receipt = json.loads((manifest.parent / "receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise ValueError("The configured training run did not complete.")
    recipe_bytes = (manifest.parent / "recipe.json").read_bytes()
    if hashlib.sha256(recipe_bytes).hexdigest() != data["recipe_sha256"]:
        raise ValueError("The opponent's training recipe changed.")
    recipe = json.loads(recipe_bytes)
    import managym._managym as native

    runtime_hash = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    native_hashes = [
        value
        for key, value in recipe["sources"].items()
        if key.startswith("managym/_managym") and key.endswith(".so")
    ]
    if native_hashes != [runtime_hash]:
        raise ValueError("The opponent was trained with a different rules runtime.")
    opponent = Opponent(
        name=data["name"],
        producer=data["producer"],
        sha256=data["sha256"],
        checkpoint=manifest.parent / data["checkpoint"],
        deterministic=data["inference"]["deterministic"],
        content_manifest=data["content_manifest"],
        decks=data["decks"],
        bot_id=data.get("bot_id"),
    )
    opponent.verify()
    return opponent


def configured_opponent() -> Opponent | None:
    path = os.environ.get("ETUDE_PLAY_CANDIDATE")
    try:
        return _load(path) if path else None
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(
            "The configured opponent bundle is unavailable or incomplete."
        ) from exc


def opponent_availability() -> dict:
    try:
        opponent = configured_opponent()
        if opponent is None:
            return {
                "available": False,
                "opponent": None,
                "reason": "No trained opponent configured.",
            }
        opponent.verify()
        return {"available": True, "opponent": opponent.identity(), "reason": None}
    except (OSError, ValueError, KeyError):
        return {
            "available": False,
            "opponent": None,
            "reason": "The configured trained opponent is unavailable.",
        }
