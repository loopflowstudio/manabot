"""Default art uses only explicit visible-name requests, with no test network."""

import importlib.metadata
import io
import json
from pathlib import Path
from urllib.request import Request

from fastapi.testclient import TestClient
import pytest

import managym
from managym import recording
from managym.academy import card_art, server
from managym.academy.card_art import CardArt, art_slug, printings

JPEG = b"\xff\xd8test-image"
NAME = "Aang's Iceberg"


def _fetch(url: str, accept: str) -> bytes:
    if accept == "application/json":
        return json.dumps({"artist": "Test Artist"}).encode()
    return JPEG


def test_art_miss_hit_full_and_credits_are_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = CardArt(tmp_path, fetch=_fetch)
    monkeypatch.setattr(server, "card_art", lambda: art)
    slug = art_slug(NAME)
    with TestClient(server.app) as client:
        response = client.get(f"/api/card-art/art/{slug}.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == JPEG
        assert "immutable" in response.headers["cache-control"]

        def offline(url: str, accept: str) -> bytes:
            raise AssertionError("cache hit must not fetch")

        art.fetch = offline
        assert client.get(f"/api/card-art/art/{slug}.jpg").content == JPEG
        assert client.get("/api/card-art/credits").json() == {}
        credits = client.get("/api/card-art/credits", params={"names": NAME}).json()
        assert credits[NAME]["artist"] == "Test Artist"
        assert (
            client.get("/api/card-art/credits", params={"names": "Other card"}).json()
            == {}
        )
        art.fetch = _fetch
        assert client.get(f"/api/card-art/full/{slug}.jpg").content == JPEG
        assert client.get("/api/card-art/back.jpg").content == JPEG
        assert client.get("/api/card-art/art/unknown.jpg").status_code == 204
        assert (
            client.get("/api/card-art/art/aang-and-la-ocean-s-fury.jpg").status_code
            == 204
        )
        assert client.get("/api/card-art/art/zeppelin.jpg").status_code == 204


def test_failure_is_memoized_and_off_switch_never_fetches(tmp_path: Path) -> None:
    def unavailable(url: str, accept: str) -> bytes:
        raise OSError("offline")

    art = CardArt(tmp_path, fetch=unavailable)
    assert art.image(art_slug(NAME), "art") is None

    def unexpected(url: str, accept: str) -> bytes:
        raise AssertionError("must not retry during failure memo")

    art.fetch = unexpected
    assert art.image(art_slug(NAME), "art") is None
    art.enabled = False
    assert art.image(art_slug(NAME), "full") is None
    assert art.back() is None
    assert art.credits([NAME]) == {}


def test_seeded_cache_works_without_fetcher(tmp_path: Path) -> None:
    printing = printings()[art_slug(NAME)]
    (tmp_path / f"{printing}-art.jpg").write_bytes(JPEG)

    def unexpected(url: str, accept: str) -> bytes:
        raise AssertionError("seeded cache fetched")

    art = CardArt(tmp_path, fetch=unexpected)
    assert art.image(art_slug(NAME), "art").read_bytes() == JPEG


def _without_installed_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source launch: no `managym` distribution metadata on the path."""
    distribution = importlib.metadata.distribution

    def missing_distribution(name: str) -> importlib.metadata.Distribution:
        if name == "managym":
            raise importlib.metadata.PackageNotFoundError(name)
        return distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)


def test_source_launch_fetches_uncached_art(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_installed_metadata(monkeypatch)
    requests: list[Request] = []

    def scryfall(request: Request, timeout: float) -> io.BytesIO:
        requests.append(request)
        if request.get_header("Accept") == "application/json":
            return io.BytesIO(json.dumps({"artist": "Test Artist"}).encode())
        return io.BytesIO(JPEG)

    monkeypatch.setattr(card_art, "urlopen", scryfall)
    art = CardArt(tmp_path)
    monkeypatch.setattr(server, "card_art", lambda: art)
    with TestClient(server.app) as client:
        response = client.get(f"/api/card-art/full/{art_slug(NAME)}.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == JPEG
    assert [request.get_header("Accept") for request in requests] == [
        "application/json",
        "image/jpeg",
    ]
    assert {request.get_header("User-agent") for request in requests} == {
        f"managym-academy/{managym.__version__}"
    }


def test_source_launch_records_the_engine_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_installed_metadata(monkeypatch)
    assert managym.__version__ == managym._managym.__version__
    identities = recording.runtime_identities.__wrapped__()
    assert identities["managym_version"] == managym.__version__
