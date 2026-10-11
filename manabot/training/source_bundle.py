"""Verify a local immutable source bundle before a long-running worker imports data.

The bundle is an execution artifact inside the supplied Task workspace, not a Git
checkout. Its manifest binds the original commit and exact source/native bytes so
later Task commits cannot change restart admission. It is not a trust mechanism
for foreign pickle artifacts.
"""

from functools import cache
from pathlib import Path
import re
import subprocess

from pydantic import field_validator

from manabot.arena.models import file_sha256
from manabot.training.models import Strict


class SourceBundle(Strict):
    commit: str
    files: dict[str, str]

    @field_validator("commit")
    @classmethod
    def commit_identity(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("bundle requires an exact source commit")
        return value


@cache
def source_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / ".manabot-source.json"
    if not path.exists():
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    bundle = SourceBundle.model_validate_json(path.read_text())
    if not bundle.files:
        raise ValueError("source bundle is empty")
    for name, digest in bundle.files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or file_sha256(root / relative) != digest
        ):
            raise ValueError(f"source bundle differs: {name}")
    return bundle.commit
