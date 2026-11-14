"""The commit of the running source, for provenance on evaluation and world records.

A release archive carries `.manabot-source.json` (`scripts/release/prepare.py`):
its commit and the digest of each source file, verified before the commit is
reported. A checkout or a launched Etude archive has no manifest and names its
commit through `etude.eval.source.source_commit`.
"""

from functools import cache
from pathlib import Path
import re
import subprocess

from etude.eval.source import source_commit as etude_source_commit
from pydantic import BaseModel, ConfigDict, field_validator

from manabot.infra.identity import file_sha256


class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    commit: str
    files: dict[str, str]

    @field_validator("commit")
    @classmethod
    def commit_identity(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("bundle requires an exact source commit")
        return value


def _verify(root: Path, bundle: SourceBundle, label: str) -> None:
    """Require every bundled file under `root` to hold its recorded bytes."""
    if not bundle.files:
        raise ValueError(f"{label} is empty")
    for name, digest in bundle.files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or file_sha256(root / relative) != digest
        ):
            raise ValueError(f"{label} differs: {name}")


@cache
def source_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / ".manabot-source.json"
    if not path.exists():
        commit = etude_source_commit([root])
        if commit is None:
            raise ValueError("source has no recorded commit")
        return commit
    bundle = SourceBundle.model_validate_json(path.read_text())
    _verify(root, bundle, "source bundle")
    return bundle.commit


@cache
def source_dirty() -> bool | None:
    """Whether tracked files differ from `source_commit`; a verified bundle never does."""
    root = Path(__file__).resolve().parents[2]
    if (root / ".manabot-source.json").exists():
        return False
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
        capture_output=True,
    )
    return bool(status.stdout) if status.returncode == 0 else None
