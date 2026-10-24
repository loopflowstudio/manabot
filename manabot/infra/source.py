"""Verify a local immutable source bundle before a long-running worker imports data.

The bundle is an execution artifact inside the supplied Task workspace, not a Git
checkout. Its manifest binds the original commit and exact source/native bytes so
later Task commits cannot change restart admission. It is not a trust mechanism
for foreign pickle artifacts.
"""

from functools import cache
import json
import os
from pathlib import Path
import re
import subprocess
import sysconfig

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


def bind_etude_source() -> None:
    """Verify launcher-pinned archive bytes and bind the freshly built extension.

    The job config carries hashes computed from its exact committed source before
    submission. This is not permission to infer a commit from arbitrary workfiles.
    Native bytes are built remotely and retained as runtime evidence, not claimed
    byte-identical to another machine's build.
    """
    root = Path(__file__).resolve().parents[2]
    config = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())
    bundle = SourceBundle.model_validate(config["source_bundle"])
    if not bundle.files:
        raise ValueError("Etude source bundle is empty")
    for name, digest in bundle.files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or file_sha256(root / relative) != digest
        ):
            raise ValueError(f"Etude archive differs from submitted source: {name}")
    native = root / "managym" / ("_managym" + sysconfig.get_config_var("EXT_SUFFIX"))
    files = dict(bundle.files)
    files[str(native.relative_to(root))] = file_sha256(native)
    target = root / ".manabot-source.json"
    if target.exists():
        raise ValueError("source manifest already exists; do not overwrite admission")
    target.write_text(
        SourceBundle(commit=bundle.commit, files=files).model_dump_json(indent=2)
    )
    source_commit.cache_clear()
    assert source_commit() == bundle.commit


if __name__ == "__main__":
    bind_etude_source()
