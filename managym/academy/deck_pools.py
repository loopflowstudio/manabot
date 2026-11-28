"""Server-scoped deck pools and named native deck serialization.

Pools let any caller (training loops, scripts, or external trainers) register
collections of ``managym.Deck`` objects that Academy can sample fresh matchups
from without importing any training framework or corpus format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import managym
from managym.disclosure import DecklistDisclosure

POOL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
POOL_PREFIX = "pool:"
DEFAULT_HOSTED_INDEX_URL = "https://policies.manabot.gg/index.json"
_POOLS_V2_SHA256 = "fcf307a506405682936f8310f6eb1059cba23c2ee2ff947e6b137af449004ecb"
_BASIC_COLORS = (
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NamedDeck:
    """One native deck paired with a display label for Academy."""

    name: str
    deck: managym.Deck

    def __post_init__(self) -> None:
        cleaned = self.name.strip()
        if not cleaned or len(cleaned) > 120:
            raise ValueError("Deck name must be 1 to 120 non-whitespace characters.")
        if not isinstance(self.deck, managym.Deck):
            raise ValueError("NamedDeck.deck must be a managym.Deck.")


@dataclass(frozen=True)
class DeckPool:
    """Named population of decks Academy can sample fresh matchups from."""

    id: str
    label: str
    decks: tuple[NamedDeck, ...]
    default_decklists: DecklistDisclosure = "closed"

    def __post_init__(self) -> None:
        if not POOL_ID_PATTERN.match(self.id):
            raise ValueError(
                "Deck pool id must contain only ASCII letters, digits, '.', '_' or '-'."
            )
        if not self.label.strip():
            raise ValueError("Deck pool label must be non-empty.")
        if not self.decks:
            raise ValueError(f"Deck pool {self.id!r} must contain at least one deck.")

    @property
    def key(self) -> str:
        return f"{POOL_PREFIX}{self.id}"

    def sample(self, rng: random.Random) -> NamedDeck:
        return self.decks[rng.randrange(len(self.decks))]

    def sample_pair(self, rng: random.Random) -> tuple[NamedDeck, NamedDeck]:
        """Draw two distinct entries when the pool holds at least two decks."""
        if len(self.decks) == 1:
            return self.decks[0], self.decks[0]
        first, second = rng.sample(range(len(self.decks)), 2)
        return self.decks[first], self.decks[second]


def deck_to_source(deck: managym.Deck) -> str:
    """Format a native deck into Arena-style text for native DeckImport admission."""
    lines = [f"{count} {name}" for name, count in sorted(deck.main.items())]
    if deck.sideboard:
        lines.append("Sideboard")
        lines.extend(
            f"{count} {name}" for name, count in sorted(deck.sideboard.items())
        )
    return "\n".join(lines) + "\n"


def deck_file_payload(
    deck: managym.Deck, *, name: str | None = None, filename: str | None = None
) -> dict[str, str]:
    """Build a DeckFileInput dictionary from a native managym.Deck."""
    payload: dict[str, str] = {
        "source": deck_to_source(deck),
        "filename": filename or f"{deck.identity[:10]}.txt",
    }
    if name is not None:
        payload["name"] = name
    return payload


_pools: dict[str, DeckPool] = {}


def set_deck_pools(pools: Sequence[DeckPool] | None) -> None:
    """Replace the server-scoped deck pool registry."""
    _pools.clear()
    if not pools:
        return
    for pool in pools:
        if pool.id in _pools:
            raise ValueError(f"Deck pool {pool.id!r} is already registered.")
        _pools[pool.id] = pool


def deck_pools() -> dict[str, DeckPool]:
    return dict(_pools)


def is_pool_key(value: object) -> bool:
    return isinstance(value, str) and value.startswith(POOL_PREFIX)


def resolve_deck_pool(key: str) -> DeckPool | None:
    pool_id = key.removeprefix(POOL_PREFIX) if key.startswith(POOL_PREFIX) else key
    return _pools.get(pool_id)


def deck_pool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": pool.id,
            "key": pool.key,
            "label": pool.label,
            "count": len(pool.decks),
            "default_decklists": pool.default_decklists,
        }
        for pool in _pools.values()
    ]


def _ssl_context() -> Any:
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _cache_dir() -> Path:
    base = os.environ.get("MANABOT_CACHE_DIR")
    return Path(base).expanduser() if base else Path.home() / ".cache" / "manabot"


def _hosted_pools_enabled() -> bool:
    flag = os.environ.get("ACADEMY_HOSTED_POOLS")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "off")
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def _trophy_colors(deck: managym.Deck) -> str:
    letters = "".join(
        code for land, code in _BASIC_COLORS if deck.main.get(land, 0) > 0
    )
    return letters or "Colorless"


def _pools_from_corpus_file(path: Path) -> list[DeckPool]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_split: dict[str, list[NamedDeck]] = {"held_out": [], "train": []}
    for entry in raw.get("pools") or ():
        if not isinstance(entry, dict):
            continue
        split = entry.get("split")
        if split not in by_split:
            continue
        cards = entry.get("cards") or {}
        human_build = entry.get("human_build") or {}
        if not isinstance(cards, dict) or not isinstance(human_build, dict):
            continue
        if managym.unsupported_cards([*cards, *human_build]):
            continue
        sideboard = {
            name: int(count) - int(human_build.get(name, 0))
            for name, count in cards.items()
            if int(count) - int(human_build.get(name, 0)) > 0
        }
        deck = managym.Deck({str(k): int(v) for k, v in human_build.items()}, sideboard)
        pool_id = str(entry.get("id", "trophy"))[:8]
        label = f"TLA Trophy · {_trophy_colors(deck)} · {pool_id}"
        by_split[split].append(NamedDeck(name=label, deck=deck))
    out: list[DeckPool] = []
    if by_split["held_out"]:
        out.append(
            DeckPool(
                id="held-out",
                label="TLA Trophies · Held-out",
                decks=tuple(by_split["held_out"]),
                default_decklists="closed",
            )
        )
    if by_split["train"]:
        out.append(
            DeckPool(
                id="train",
                label="TLA Trophies · Train",
                decks=tuple(by_split["train"]),
                default_decklists="closed",
            )
        )
    return out


_DEFAULT_POOLS_CACHE: list[DeckPool] | None = None


def _locate_or_fetch_trophy_corpus() -> Path | None:
    repo_candidate = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "data"
        / "mtg409"
        / "pools-v2"
        / "corpus.json"
    )
    if repo_candidate.is_file():
        return repo_candidate
    cache_root = _cache_dir() / "sha256"
    cached = cache_root / _POOLS_V2_SHA256
    if cached.is_file():
        return cached
    if not _hosted_pools_enabled():
        return None
    url = os.environ.get("ACADEMY_POLICIES_URL", DEFAULT_HOSTED_INDEX_URL)
    try:
        req = Request(url, headers={"User-Agent": "managym-academy"})
        with urlopen(req, timeout=5, context=_ssl_context()) as resp:
            index = json.loads(resp.read().decode("utf-8"))
        default_pool_id = index.get("default_deck_pool")
        pool_entry = next(
            (
                item
                for item in (index.get("deck_pools") or ())
                if isinstance(item, dict) and item.get("id") == default_pool_id
            ),
            None,
        )
        if not pool_entry:
            return None
        pool_sha = str(pool_entry["sha256"])
        pool_url = str(pool_entry["url"])
        cache_root.mkdir(parents=True, exist_ok=True)
        dest = cache_root / pool_sha
        if dest.is_file():
            return dest
        req = Request(pool_url, headers={"User-Agent": "managym-academy"})
        with urlopen(req, timeout=60, context=_ssl_context()) as resp:
            body = resp.read()
        if hashlib.sha256(body).hexdigest() != pool_sha:
            raise ValueError("Hosted deck pool SHA-256 mismatch.")
        with tempfile.NamedTemporaryFile(dir=cache_root, delete=False) as tmp:
            tmp.write(body)
            tmp_path = Path(tmp.name)
        tmp_path.replace(dest)
        return dest
    except (OSError, URLError, ValueError) as exc:
        _LOGGER.debug("Hosted trophy pool unavailable: %s", exc)
        return None


def ensure_default_deck_pools() -> list[DeckPool]:
    """Load the default 17Lands TLA trophy pools when available locally or hosted."""
    global _DEFAULT_POOLS_CACHE
    if _pools:
        return list(_pools.values())
    if _DEFAULT_POOLS_CACHE is not None:
        return list(_DEFAULT_POOLS_CACHE)
    if not _hosted_pools_enabled():
        return []
    path = _locate_or_fetch_trophy_corpus()
    if path is None:
        return []
    try:
        loaded = _pools_from_corpus_file(path)
    except (OSError, ValueError) as exc:
        _LOGGER.warning("Could not load default trophy pools from %s: %s", path, exc)
        return []
    _DEFAULT_POOLS_CACHE = loaded
    return list(loaded)
