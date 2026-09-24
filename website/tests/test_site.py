"""Smoke tests: the page serves, the domains converge, the attribution shows."""

import sys
import tomllib
from pathlib import Path
from xml.etree import ElementTree

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


@pytest.fixture
def client():
    return TestClient(app, base_url="http://etude.gg")


def test_homepage_redirects_to_blog(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/blog/lol-cube-kickoff"


def test_blog_post_serves(client):
    r = client.get("/blog/lol-cube-kickoff")
    assert r.status_code == 200
    assert "Rotisserie Drafting" in r.text


@pytest.mark.parametrize("host", ["etude.gg", "etude-website.fly.dev", "localhost:5002"])
def test_healthz_plain_response(client, host):
    r = client.get("/healthz", headers={"host": host}, follow_redirects=False)
    assert r.status_code == 200
    assert r.content == b"ok"
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert "location" not in r.headers


def test_fly_healthz_probe(client):
    config = tomllib.loads((Path(__file__).parent.parent / "fly.toml").read_text())
    for check in config["http_service"]["checks"]:
        assert check["path"] == "/healthz"
        r = client.request(check["method"], check["path"], follow_redirects=False)
        assert r.status_code == 200
        assert r.content == b"ok"


def test_robots_and_sitemap(client):
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "https://etude.gg/sitemap.xml" in robots.text
    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200
    root = ElementTree.fromstring(sitemap.content)
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    assert root.tag == f"{namespace}urlset"
    assert [loc.text for loc in root.findall(f"{namespace}url/{namespace}loc")] == [
        "https://etude.gg/blog/lol-cube-kickoff"
    ]


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
    r = client.get(
        "/", headers={"host": "etude-website.fly.dev"},
        follow_redirects=False)
    assert r.status_code == 302


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
