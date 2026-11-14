"""Complete one tagged release without overwriting published artifacts.

PyPI retries require the full retained artifact inventory and exact SHA-256
identity. GitHub completion runs after publication and is independently retryable.
Only an absent version may upload; partial or different bytes require inspection.

PyPI's per-version JSON can report an uploaded version as absent for a short
while. Checks that follow an upload (`--require-engine`, `--github`) poll for
up to `--wait` seconds before treating the version as missing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
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


def published_files(project: str, version: str) -> dict[str, str]:
    try:
        data = _json(Request(f"https://pypi.org/pypi/{project}/{version}/json"))
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


def needs_upload(directory: Path, project: str, version: str) -> bool:
    local: dict[str, str] = {}
    for path in directory.iterdir():
        if path.name.startswith(f"{project}-") and (
            path.suffix == ".whl" or path.name.endswith(".tar.gz")
        ):
            with path.open("rb") as stream:
                local[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(local) != {"managym": 5, "manabot": 2}[project] or any(
        not name.startswith(f"{project}-{version}-")
        and name != f"{project}-{version}.tar.gz"
        for name in local
    ):
        raise ValueError("expected the validated package release inventory")
    published = published_files(project, version)
    if not published:
        return True
    if published != local:
        raise ValueError(
            "PyPI inventory or digests differ; inspect partial/different publication before retrying"
        )
    return False


def await_published(
    directory: Path,
    projects: tuple[str, ...],
    version: str,
    wait: float,
    *,
    interval: float = 10.0,
) -> list[str]:
    """Return the projects still absent from PyPI after up to ``wait`` seconds.

    Mismatched inventories raise immediately: only absence is a lag to wait out.
    """
    deadline = time.monotonic() + wait
    while True:
        missing = [p for p in projects if needs_upload(directory, p, version)]
        if not missing or time.monotonic() >= deadline:
            return missing
        time.sleep(interval)


def complete_github(repository: str, tag: str, commit: str, notes: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("expected a vX.Y.Z tag")
    if not notes.strip():
        raise ValueError("release notes are empty")
    body = (
        notes.rstrip()
        + f"\n\nInstall [managym](https://pypi.org/project/managym/{tag[1:]}/) "
        + f"or [manabot](https://pypi.org/project/manabot/{tag[1:]}/) from PyPI.\n"
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
    parser.add_argument("--require-engine", action="store_true")
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="seconds to wait for a just-uploaded version to appear on PyPI",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        parser.error("expected a vX.Y.Z tag")
    version = args.tag[1:]
    if args.require_engine:
        if await_published(args.directory, ("managym",), version, args.wait):
            raise ValueError("engine publication is missing; training upload must wait")
        return
    if args.github:
        if await_published(args.directory, ("managym", "manabot"), version, args.wait):
            raise ValueError("PyPI publication is missing; release is incomplete")
        root = Path(__file__).resolve().parents[2]
        notes = (root / "release" / args.tag / "NOTES.md").read_text()
        print(
            complete_github(
                os.environ["GITHUB_REPOSITORY"],
                args.tag,
                os.environ["GITHUB_SHA"],
                notes,
            )
        )
    else:
        # Inspect both before emitting either upload instruction. A failure in
        # the second project must not permit the first upload.
        uploads = {
            project: needs_upload(args.directory, project, version)
            for project in ("managym", "manabot")
        }
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for project, upload in uploads.items():
                output.write(f"{project}={str(upload).lower()}\n")


if __name__ == "__main__":
    main()
