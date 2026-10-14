"""Freeze committed source/native bytes inside this Task's evidence directory.

This is a local execution artifact, with no Git metadata or second worker. All
training imports and subsequent recovery use these checked bytes while ordinary
Task documentation and review continue in the supplied workspace.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from manabot.arena.models import file_sha256
from manabot.training.execution import atomic_json
from manabot.training.source_bundle import SourceBundle
import managym


def freeze_source(out: Path, *, environment: bool = True) -> Path:
    root = Path(__file__).resolve().parents[2]
    out = out.resolve()
    if not out.is_relative_to(root / ".runs"):
        raise ValueError("source bundles belong under this Task's .runs directory")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("checkpoint Task changes before freezing execution source")
    names = (
        subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    native = Path(managym._managym.__file__).resolve()
    names.append(str(native.relative_to(root)))
    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}
    for name in sorted(set(filter(None, names))):
        source = root / name
        if source.is_symlink() and (
            not source.resolve().is_relative_to(root) or not source.is_file()
        ):
            raise ValueError(f"source bundle symlink escapes tracked source: {name}")
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[name] = file_sha256(target)
        target.chmod(0o444)
    manifest = SourceBundle(
        commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        files=files,
    )
    atomic_json(out / ".manabot-source.json", manifest.model_dump(mode="json"))
    (out / ".manabot-source.json").chmod(0o444)
    if environment:
        build_environment(out)
    return out


def build_environment(out: Path) -> None:
    # The pinned environment belongs to the bundle too. Reviewing this Task must
    # not mutate the interpreter/dependencies needed for a later process restart.
    with (out / "environment-build.log").open("wb") as log:
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(out / ".venv")],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["uv", "sync", "--frozen", "--extra", "notebook", "--project", str(out)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def launch(bundle: Path, plan: Path, out: Path) -> int:
    """Start the existing Experiment coordinator in an isolated durable session."""
    bundle, plan, out = bundle.resolve(), plan.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("use the explicit resume command for existing evidence")
    manifest = bundle / ".manabot-source.json"
    SourceBundle.model_validate_json(manifest.read_text())
    receipt = out.with_suffix(".launch.json")
    if receipt.exists():
        raise FileExistsError("a prior launch receipt exists; retain and inspect it")
    command = [
        str(bundle / ".venv/bin/python"),
        "-m",
        "experiments.runners.sustained_baseline_supervisor",
        str(plan),
        str(out),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=bundle,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "PYTHONPATH": str(bundle),
                "PATH": str(bundle / ".venv/bin")
                + os.pathsep
                + os.environ.get("PATH", ""),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
    atomic_json(
        receipt,
        {
            "pid": process.pid,
            "started_unix": time.time(),
            "command": command,
            "cwd": str(bundle),
            "plan_sha256": file_sha256(plan),
            "source_manifest_sha256": file_sha256(manifest),
            "log": str(out.with_suffix(".log")),
        },
    )
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--launch", type=Path, help="Use this existing frozen source bundle"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    if args.launch:
        if args.plan is None:
            parser.error("--launch requires --plan")
        print(launch(args.launch, args.plan, args.out))
    elif args.environment_only:
        build_environment(args.out.resolve())
    else:
        print(freeze_source(args.out, environment=not args.source_only))


if __name__ == "__main__":
    main()
