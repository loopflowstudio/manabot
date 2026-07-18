"""Smoke tests: the page serves, the domains converge, the attribution shows."""

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


@pytest.fixture
def client():
    return TestClient(app, base_url="http://etude.gg")


def test_homepage(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Etude Fantasia" in r.text
    assert "study of a game that improvises" in r.text
    assert "Fan Content" in r.text


def test_robots_and_sitemap(client):
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "https://etude.gg/sitemap.xml" in robots.text
    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200
    assert "<loc>https://etude.gg/</loc>" in sitemap.text


@pytest.mark.parametrize(
    "host",
    [
        "www.etude.gg",
        "etudefantasia.com",
        "www.etudefantasia.com",
        "etudefantasia.gg",
        "www.etudefantasia.gg",
    ],
)
def test_alternate_hosts_redirect(client, host):
    r = client.get("/", headers={"host": host}, follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"] == "https://etude.gg/"


def test_fly_host_serves_directly(client):
    r = client.get("/", headers={"host": "etude-website.fly.dev"})
    assert r.status_code == 200


def test_static_assets_exist():
    static = Path(__file__).parent.parent / "static"
    for asset in (
        "style.css",
        "favicon.svg",
        "board-developed.png",
        "fonts/CormorantGaramond-SemiBold.otf",
        "mana/W.svg",
    ):
        assert (static / asset).exists(), asset
