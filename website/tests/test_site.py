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
def client() -> TestClient:
    return TestClient(app, base_url="http://manabot.gg")


def test_homepage_redirects_to_blog(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/blog/lol-cube-kickoff"


def test_blog_post_serves(client):
    r = client.get("/blog/lol-cube-kickoff")
    assert r.status_code == 200
    assert "Rotisserie Drafting" in r.text


@pytest.mark.parametrize("host", ["manabot.gg", "fantasia-website.fly.dev", "localhost:5002"])
def test_healthz_plain_response(client: TestClient, host: str) -> None:
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


def test_robots_and_sitemap(client: TestClient) -> None:
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "https://manabot.gg/sitemap.xml" in robots.text
    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200
    root = ElementTree.fromstring(sitemap.content)
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    assert root.tag == f"{namespace}urlset"
    assert [loc.text for loc in root.findall(f"{namespace}url/{namespace}loc")] == [
        "https://manabot.gg/blog/lol-cube-kickoff"
    ]


@pytest.mark.parametrize(
    "host",
    [
        "www.manabot.gg",
        "etude.gg",
        "www.etude.gg",
        "etudefantasia.com",
        "www.etudefantasia.com",
        "etudefantasia.gg",
        "www.etudefantasia.gg",
    ],
)
def test_alternate_hosts_redirect(client: TestClient, host: str) -> None:
    r = client.get("/", headers={"host": host}, follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"] == "https://manabot.gg/"


def test_fly_host_serves_directly(client: TestClient) -> None:
    r = client.get(
        "/", headers={"host": "fantasia-website.fly.dev"},
        follow_redirects=False)
    assert r.status_code == 302


@pytest.mark.parametrize("host", ["etude.gg", "www.manabot.gg", "etudefantasia.com"])
@pytest.mark.parametrize("path", [
    "/blog/lol-cube-kickoff?from=old&tag=a&tag=b",
    "/a%2Fb/space%20here?q=a%2Bb&next=https%3A%2F%2Fexample.com",
])
def test_redirect_preserves_encoded_path_and_query(
    client: TestClient, host: str, path: str
) -> None:
    response = client.get(path, headers={"host": host}, follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == f"https://manabot.gg{path}"
