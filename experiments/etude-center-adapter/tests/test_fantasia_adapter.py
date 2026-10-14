"""The private reader depends only on the public client and tenant HTTP authority."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from etude_client.api import Client
from etude_client.errors import EtudeError
from etude_client.models import ExperimentDetail, Principal, SourceKey, sha256
from etude_fantasia.reader import Reader
from etude_fantasia.research import import_citations
from etude_fantasia.transports import LocalSource

from etude.admission import Queue
from etude.database import create_organization


def test_adapter_cli_retains_replays_and_isolates(
    service: tuple[str, str], api_client: Client, queue: Queue, tmp_path: Path
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text(service[1])
    token_file.chmod(0o600)
    env = dict(os.environ, ETUDE_URL=service[0], ETUDE_TOKEN_FILE=str(token_file))
    env.pop("ETUDE_DATABASE_URL", None)
    command = [
        sys.executable,
        "-m",
        "etude_fantasia.cli",
        "retained",
        str(Path(__file__).resolve().parents[2] / "fantasia"),
        "capacity-recovery-v3",
    ]
    for _ in range(2):
        result = subprocess.run(command, env=env, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
    items = api_client.list().items
    assert len(items) == 1
    detail = api_client.get(items[0].experiment_id)
    assert len(detail.jobs) == 9
    key = detail.registration.source_key
    assert key is not None
    assert api_client.find_by_key(key).experiment_id == detail.experiment_id  # type: ignore[union-attr]
    for evidence in detail.evidence:
        raw = api_client.request("GET", f"/evidence/{evidence.evidence_id}")
        assert sha256(raw) == evidence.sha256
    other = Queue(queue.dsn, create_organization(queue.dsn, "other adapter tenant"))
    token = other.issue_token(
        Principal(organization_id=other.organization_id, name="other", scope="read")
    )
    with Client(service[0], token) as client:
        assert client.find_by_key(key) is None
        with pytest.raises(EtudeError):
            client.get(detail.experiment_id)
    token_file.write_text(
        queue.issue_token(
            Principal(organization_id=queue.organization_id, name="read", scope="read")
        )
    )
    result = subprocess.run(command, env=env, capture_output=True, check=False)
    assert result.returncode == 1 and result.stderr.strip() == b"fantasia_import_failed"


def test_adapter_imports_without_server() -> None:
    # A fresh interpreter rejects accidental coupling to private persistence even
    # when the development environment also happens to have the server installed.
    script = """
import sys
class NoServer:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "etude" or fullname.startswith("etude."):
            raise ImportError("server is unavailable to tenant adapter")
sys.meta_path.insert(0, NoServer())
import etude_fantasia.cli
import etude_fantasia.live_fantasia
import etude_fantasia.reader
import etude_fantasia.transports
"""
    subprocess.run([sys.executable, "-c", script], capture_output=True, check=True)


def test_local_selection_freezes_working_bytes_until_next_cut(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    reader = Reader(LocalSource(tmp_path))
    ref = reader.select(path.name)
    path.write_bytes(b"changed")
    assert reader.select(path.name) == ref
    assert reader.bytes(ref) == b"original"
    assert Reader(LocalSource(tmp_path)).select(path.name).sha256 == sha256(b"changed")


def test_citation_publication_never_reopens_sources(
    api_client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    find = api_client.find_by_key

    def expired(reader: Reader) -> None:
        raise EtudeError("timeout", 504)

    def find_after_read_deadline(key: SourceKey) -> ExperimentDetail | None:
        # Simulate HTTP latency exhausting the read budget. Prepared evidence
        # must remain publishable without consulting the source again.
        monkeypatch.setattr(Reader, "check_time", expired)
        return find(key)

    monkeypatch.setattr(api_client, "find_by_key", find_after_read_deadline)
    receipts = import_citations(api_client, Path(__file__).resolve().parents[2] / "fantasia")
    assert len(receipts) == 11
    assert sum(len(api_client.get(r.experiment_id).evidence) for r in receipts) == 23


def test_source_key_lookup_validates_private_query(api_client: Client) -> None:
    with pytest.raises(EtudeError) as error:
        api_client.request("GET", "/v1/experiment-key?namespace=secret:canary&kind=job&source_id=x")
    assert error.value.status == 422
    assert "canary" not in str(error.value)


def test_retained_cli_pins_original_revision_after_head_and_dirty_changes(
    service: tuple[str, str], api_client: Client, tmp_path: Path
) -> None:
    """Equal bytes at a new revision add provenance; explicit replay never does."""
    root = tmp_path / "source"
    root.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    path = root / "experiments/history-input.md"
    path.parent.mkdir()
    path.write_text("# History input\nRetained result.\n")
    git("add", ".")
    git("commit", "-qm", "original source")
    original = git("rev-parse", "HEAD")
    token_file = tmp_path / "token"
    token_file.write_text(service[1])
    token_file.chmod(0o600)
    env = dict(os.environ, ETUDE_URL=service[0], ETUDE_TOKEN_FILE=str(token_file))
    command = [sys.executable, "-m", "etude_fantasia.cli", "retained", str(root), "history-input"]

    def ingest(revision: str | None = None) -> None:
        args = command + (["--revision", revision] if revision else [])
        result = subprocess.run(args, env=env, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr

    ingest()
    experiment_id = api_client.list().items[0].experiment_id
    before = api_client.get(experiment_id)
    (root / "unrelated").write_text("Advance HEAD, not evidence.\n")
    git("add", ".")
    git("commit", "-qm", "unrelated change")
    newer = git("rev-parse", "HEAD")
    ingest()
    after = api_client.get(experiment_id)
    assert len(after.evidence) == 2 and len(after.facts) == 2
    assert {e.source_revision for e in after.evidence} == {original, newer}
    assert {e.sha256 for e in after.evidence} == {before.evidence[0].sha256}
    ingest(original)
    replay = api_client.get(experiment_id)
    assert replay.evidence == after.evidence and replay.facts == after.facts
    path.write_text("# History input\nDirty result.\n")
    ingest(original)
    pinned = api_client.get(experiment_id)
    assert pinned.evidence == after.evidence and pinned.facts == after.facts
    ingest()
    dirty = api_client.get(experiment_id)
    unpinned = [e for e in dirty.evidence if e.source_revision is None]
    assert len(unpinned) == 1 and unpinned[0].sha256 == sha256(path.read_bytes())
    assert path.read_text().endswith("Dirty result.\n")
