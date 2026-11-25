"""Same-origin Scryfall art, cached by printing identity with procedural fallback.

Only explicit requests fetch art; catalog enumeration never warms the cache.
Credits are returned for requested names only, so another viewer's cached cards
cannot disclose their game. One lock serializes cold requests and atomic writes.
"""

from collections.abc import Callable
from functools import cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
import time
from typing import Literal, TypedDict
from urllib.request import Request, urlopen

import managym


class ArtCredit(TypedDict):
    artist: str
    source: str
    copyright: str


CARD_BACK = (
    "https://backs.scryfall.io/normal/0/a/0aeebaf5-8c7d-4636-9e82-8c27447861f7.jpg"
)
RETRY_SECONDS = 600
REQUEST_SPACING_SECONDS = 0.1  # Scryfall asks for 50-100 ms between requests.


def art_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@cache
def printings() -> dict[str, str]:
    """Scryfall id by name slug; back faces and unprinted tokens have none."""
    rows = managym.content_pool(managym.WORLD_SETUP)["definitions"]
    result: dict[str, str] = {}
    # Supported cards precede token rows for a shared printed name.
    for row in sorted(rows, key=lambda row: row["eligibility"] != "supported"):
        if row.get("scryfall_id"):
            result.setdefault(art_slug(row["registry_name"]), row["scryfall_id"])
    return result


def _fetch(url: str, accept: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": f"managym-academy/{version('managym')}",
            "Accept": accept,
        },
    )
    with urlopen(request, timeout=8) as response:
        return response.read()


class CardArt:
    """A directory of `<id>-<kind>.jpg` images and `<id>.json` credits."""

    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool = True,
        fetch: Callable[[str, str], bytes] = _fetch,
    ) -> None:
        self.directory = directory
        self.enabled = enabled
        self.fetch = fetch
        self.lock = Lock()
        self.last_request = 0.0
        self.failed: dict[Path, float] = {}

    def image(self, slug: str, kind: Literal["art", "full"]) -> Path | None:
        identity = printings().get(slug)
        if identity is None:
            return None
        size = "art_crop" if kind == "art" else "normal"
        return self._cached(
            self.directory / f"{identity}-{kind}.jpg",
            f"https://cards.scryfall.io/{size}/front/{identity[0]}/{identity[1]}/{identity}.jpg",
            credit=identity,
        )

    def back(self) -> Path | None:
        return self._cached(self.directory / "back-back.jpg", CARD_BACK)

    def credits(self, names: list[str]) -> dict[str, ArtCredit]:
        """Credits already cached for these names; never a cache-wide listing."""
        result: dict[str, ArtCredit] = {}
        if not self.enabled:
            return result
        for name in names:
            identity = printings().get(art_slug(name))
            path = self.directory / f"{identity}.json"
            if identity is None or not path.is_file():
                continue
            body = json.loads(path.read_text())
            if all(isinstance(body.get(key), str) for key in ArtCredit.__annotations__):
                result[name] = ArtCredit(
                    artist=body["artist"],
                    source=body["source"],
                    copyright=body["copyright"],
                )
        return result

    def _cached(self, path: Path, url: str, credit: str | None = None) -> Path | None:
        """Return the cached image, fetching it (after its credit) on a miss."""
        if not self.enabled:
            return None
        # A warm image never queues behind another card's cold network request.
        if path.is_file():
            return path
        with self.lock:
            if path.is_file():
                return path
            if self.failed.get(path, 0) > time.monotonic():
                return None
            try:
                if credit is not None:
                    self._fetch_credit(credit)
                body = self._request(url, "image/jpeg")
                if not body.startswith(b"\xff\xd8"):
                    raise ValueError("not a JPEG")
                self._write(path, body)
                return path
            except (OSError, ValueError, KeyError, TypeError):
                self.failed[path] = time.monotonic() + RETRY_SECONDS
                return None

    def _fetch_credit(self, identity: str) -> None:
        path = self.directory / f"{identity}.json"
        if path.exists():
            return
        card = json.loads(
            self._request(
                f"https://api.scryfall.com/cards/{identity}", "application/json"
            )
        )
        artist = card.get("artist") or card.get("card_faces", [{}])[0].get("artist")
        if not isinstance(artist, str):
            raise ValueError("missing artist")
        credit = ArtCredit(
            artist=artist,
            source=f"https://scryfall.com/search?q=id%3A{identity}",
            copyright="© Wizards of the Coast LLC",
        )
        self._write(path, json.dumps(credit).encode())

    def _request(self, url: str, accept: str) -> bytes:
        elapsed = time.monotonic() - self.last_request
        time.sleep(max(0, REQUEST_SPACING_SECONDS - elapsed))
        self.last_request = time.monotonic()
        return self.fetch(url, accept)

    def _write(self, path: Path, body: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(body)
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


@cache
def card_art() -> CardArt:
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return CardArt(
        Path(os.environ.get("ACADEMY_CARD_ART_DIR", str(root / "managym/card-art"))),
        enabled=os.environ.get("ACADEMY_CARD_ART", "on").lower() != "off",
    )
