"""Complete one tagged release without overwriting published artifacts.

PyPI retries require the full retained artifact inventory and exact SHA-256
identity. GitHub completion runs after publication and is independently retryable.
Only an absent version may upload; partial or different bytes require inspection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _json(request: Request) -> dict[str, Any]:
    # External JSON is narrowed at each service boundary below.
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("expected a service JSON object")
    return data


def published_files(version: str) -> dict[str, str]:
    try:
        data = _json(Request(f"https://pypi.org/pypi/manabot/{version}/json"))
    except HTTPError as error:
        if error.code == 404:
            return {}
        raise
    files: dict[str, str] = {}
    for entry in data["urls"]:
        name, digest = entry["filename"], entry["digests"]["sha256"]
        if not isinstance(name, str) or not isinstance(digest, str) or entry["yanked"]:
            raise ValueError("invalid or yanked PyPI artifact")
        if name in files:
            raise ValueError("duplicate PyPI artifact")
        files[name] = digest
    if not files:
        raise ValueError("existing PyPI version has no files; inspect before retrying")
    return files


def needs_upload(directory: Path, version: str) -> bool:
    local: dict[str, str] = {}
    for path in directory.iterdir():
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            with path.open("rb") as stream:
                local[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(local) != 5 or any(
        not name.startswith(f"manabot-{version}-")
        and name != f"manabot-{version}.tar.gz"
        for name in local
    ):
        raise ValueError("expected the validated five-artifact release inventory")
    published = published_files(version)
    if not published:
        return True
    if published != local:
        raise ValueError(
            "PyPI inventory or digests differ; inspect partial/different publication before retrying"
        )
    return False


def complete_github(repository: str, tag: str, commit: str, notes: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("expected a vX.Y.Z tag")
    if not notes.strip():
        raise ValueError("release notes are empty")
    body = (
        notes.rstrip()
        + f"\n\nInstall from [PyPI](https://pypi.org/project/manabot/{tag[1:]}/).\n"
    )
    endpoint = f"https://api.github.com/repos/{repository}/releases"
    headers = {
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        release = _json(Request(f"{endpoint}/tags/{tag}", headers=headers))
    except HTTPError as error:
        if error.code != 404:
            raise
        payload = {
            "tag_name": tag,
            "target_commitish": commit,
            "name": f"Manabot {tag[1:]}",
            "body": body,
            "draft": False,
            "prerelease": False,
        }
        release = _json(
            Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
        )
    if (
        release.get("tag_name") != tag
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or (release.get("body") or "").strip() != body.strip()
    ):
        raise ValueError(
            "GitHub Release is missing, draft, or does not match the published version and notes"
        )
    url = release.get("html_url")
    if not isinstance(url, str) or not url:
        raise ValueError("GitHub Release URL missing")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--github", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        parser.error("expected a vX.Y.Z tag")
    upload = needs_upload(args.directory, args.tag[1:])
    if args.github:
        if upload:
            raise ValueError("PyPI publication is missing; release is incomplete")
        notes = Path("docs/release-notes.md").read_text()
        print(
            complete_github(
                os.environ["GITHUB_REPOSITORY"],
                args.tag,
                os.environ["GITHUB_SHA"],
                notes,
            )
        )
    else:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"upload={str(upload).lower()}\n")


if __name__ == "__main__":
    main()
