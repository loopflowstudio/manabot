"""Public source admission retains exact identities through API rate limits."""

from io import BytesIO
import json
import subprocess
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from manabot.remote import deploy
from tests.remote.test_compile import SOURCE


def test_public_source_api_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def response(request: Request, *, timeout: int) -> BytesIO:
        assert "repos/loopflowstudio/fantasia/git/commits/" in request.full_url
        return BytesIO(
            json.dumps({"sha": SOURCE.commit, "tree": {"sha": SOURCE.tree}}).encode()
        )

    monkeypatch.setattr(deploy, "urlopen", response)
    deploy.verify_public_source(SOURCE)


@pytest.mark.parametrize("authenticated", [False, True])
def test_rate_limit_falls_back_without_forwarding_credentials(
    monkeypatch: pytest.MonkeyPatch, authenticated: bool
) -> None:
    def limited(request: Request, *, timeout: int) -> BytesIO:
        raise HTTPError(request.full_url, 403, "rate limit", {}, None)

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == [
            "/host/gh",
            "api",
            f"repos/loopflowstudio/fantasia/git/commits/{SOURCE.commit}",
        ]
        assert "env" not in kwargs
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"sha": SOURCE.commit, "tree": {"sha": SOURCE.tree}})
        )

    fallback: list[object] = []
    monkeypatch.setattr(deploy, "urlopen", limited)
    monkeypatch.setattr(
        deploy.shutil, "which", lambda name: "/host/gh" if authenticated else None
    )
    monkeypatch.setattr(deploy.subprocess, "run", run)
    monkeypatch.setattr(deploy, "_verify_public_source_git", fallback.append)
    deploy.verify_public_source(SOURCE)
    assert fallback == ([] if authenticated else [SOURCE])


def test_public_git_checks_lock_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str | bytes]:
        commands.append(command)
        if command[-1] == "FETCH_HEAD":
            output: str | bytes = SOURCE.commit
        elif command[-1] == "FETCH_HEAD^{tree}":
            output = SOURCE.tree
        elif command[-1] == "FETCH_HEAD:uv.lock":
            output = b"wrong lock bytes"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(deploy.subprocess, "run", run)
    with pytest.raises(ValueError, match="lock identity differs"):
        deploy._verify_public_source_git(SOURCE)
    assert any(
        "https://github.com/loopflowstudio/fantasia.git" in command
        for command in commands
    )
