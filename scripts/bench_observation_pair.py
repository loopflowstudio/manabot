"""Serial, alternating before/refactor/current observation measurements.

Supply an instrumented source snapshot at --before (never a managed worktree).
Both snapshots must already contain their own release extension and native
example, built with the same benchmark feature/toolchain. Fresh output only;
source manifests and binaries are checked again after the last paired run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from statistics import median
import subprocess
import sysconfig
import time
from typing import Any

EXTENSION_PATH = f"manabot/managym/_managym{sysconfig.get_config_var('EXT_SUFFIX')}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root: Path) -> dict[str, str]:
    # Include all executable/compiled sources and content, including new files.
    paths = [
        p
        for directory in ("manabot", "managym", "scripts", "content")
        for p in (root / directory).rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".pyi", ".rs", ".toml", ".lock", ".json"}
        and not {"target", "__pycache__", "node_modules"}.intersection(
            p.relative_to(root).parts
        )
    ]
    paths.extend(
        root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()
    )
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def compare(out: Path) -> dict[str, object]:
    """Fail closed on unmatched work; retain all pair ratios without a noise cutoff."""
    summary: dict[str, object] = {}
    identities = json.loads((out / "identity.json").read_text())
    arms = ("before", "refactor", "current")
    cases = (
        "priority",
        "target",
        "additional_cost",
        "large_legal",
        "large_legal_output",
    )
    for batch in (1, 64):
        vector = {
            (arm, pair): json.loads(
                (out / f"vector-{arm}-{batch}-{pair}.json").read_text()
            )
            for arm in arms
            for pair in range(3)
        }
        native = {
            (arm, pair): [
                json.loads(line)
                for line in (out / f"encode-{arm}-{pair}.jsonl")
                .read_text()
                .splitlines()
            ]
            for arm in arms
            for pair in range(3)
        }
        for (arm, _), receipt in vector.items():
            source_arm = "before" if arm == "before" else "current"
            expected = identities["binaries"][source_arm + "/" + EXTENSION_PATH]
            assert receipt["identity"]["extension_sha256"] == expected
            assert receipt["identity_after"]["extension_sha256"] == expected
        for case in cases:
            rows = {
                key: next(row for row in receipt["cases"] if row["case"] == case)
                for key, receipt in vector.items()
            }
            for arm in arms:
                for field in ("encoded_digest", "output_digest"):
                    assert (
                        len(
                            {
                                row[field]
                                for (label, _), row in rows.items()
                                if label == arm
                            }
                        )
                        == 1
                    )
            for field in (
                "input_digest",
                "command_digest",
                "legacy_encoded_digest",
                "legacy_output_digest",
                "post_state_digest",
            ):
                assert len({row[field] for row in rows.values()}) == 1, (
                    batch,
                    case,
                    field,
                )
            assert (
                len(
                    {
                        row["continuation_digest"]
                        for (arm, _), row in rows.items()
                        if arm != "before"
                    }
                )
                == 1
            )
            if case != "large_legal_output":
                encode_rows = {
                    key: [
                        row
                        for row in receipt
                        if row["case"] == case and row["batch"] == batch
                    ]
                    for key, receipt in native.items()
                }
                assert (
                    len(
                        {
                            row["input_digest"]
                            for (arm, _), receipt in encode_rows.items()
                            if arm != "before"
                            for row in receipt
                        }
                    )
                    == 1
                )
                assert (
                    len(
                        {
                            row["corpus"]
                            for receipt in encode_rows.values()
                            for row in receipt
                        }
                    )
                    == 1
                )
                assert (
                    len(
                        {
                            row["encoded_digest"]
                            for (arm, _), receipt in encode_rows.items()
                            if arm != "current"
                            for row in receipt
                        }
                    )
                    == 1
                )
            measurements = [
                (
                    "vector",
                    {key: row["median_ns_per_env"] for key, row in rows.items()},
                ),
            ]
            if case != "large_legal_output":
                measurements.append(
                    (
                        "encode",
                        {
                            key: median(row["ns_per_encode"] for row in receipt)
                            for key, receipt in encode_rows.items()
                        },
                    )
                )
            for mode, latencies in measurements:
                ratios = {
                    f"{left}_to_{right}": [
                        latencies[right, pair] / latencies[left, pair]
                        for pair in range(3)
                    ]
                    for left, right in (
                        ("before", "refactor"),
                        ("refactor", "current"),
                        ("before", "current"),
                    )
                }
                summary[f"{mode}/{batch}/{case}"] = {
                    "median_ns_per_env": {
                        arm: median(latencies[arm, pair] for pair in range(3))
                        for arm in arms
                    },
                    "paired_ratios": ratios,
                }
    return {"matched_inputs_commands_and_legacy_outputs": True, "comparisons": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    root = Path(__file__).resolve().parents[1]
    before = args.before.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifests = {"before": source_identity(before), "current": source_identity(root)}
    binaries = {
        f"{label}/{name}": sha256(checkout / name)
        for label, checkout in (("before", before), ("current", root))
        for name in (
            "managym/target/release/examples/bench_observation",
            EXTENSION_PATH,
        )
    }
    metadata = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "sources": manifests,
        "binaries": binaries,
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True),
        "uv": subprocess.check_output(["uv", "--version"], text=True),
        "platform": platform.platform(),
        "cpu_model": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "cpu_count": os.cpu_count(),
        "profile": "release",
        "feature": "benchmark (python for extension)",
        "plan": {
            "orders": [
                ["before", "refactor", "current"],
                ["current", "refactor", "before"],
                ["before", "refactor", "current"],
            ],
            "batches": [1, 64],
            "matched_steps": args.steps,
        },
        "compiler_environment": {
            k: os.environ.get(k)
            for k in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_PROFILE_RELEASE_OPT_LEVEL",
                "CARGO_PROFILE_RELEASE_LTO",
                "MACOSX_DEPLOYMENT_TARGET",
            )
        },
    }
    (out / "identity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "working.diff").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    )
    commands: list[dict[str, Any]] = []

    def run(name: str, command: list[str], checkout: Path) -> None:
        started = time.time()
        cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        load = os.getloadavg()
        with (
            (out / name).open("w") as stdout,
            (out / (name + ".stderr")).open("w") as stderr,
        ):
            result = subprocess.run(
                command,
                cwd=checkout,
                env={
                    **os.environ,
                    "PYTHONPATH": str(checkout),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        commands.append(
            {
                "file": name,
                "command": command,
                "cwd": str(checkout),
                "returncode": result.returncode,
                "started_unix": started,
                "seconds": time.time() - started,
                "cpu_seconds": after.ru_utime
                + after.ru_stime
                - cpu.ru_utime
                - cpu.ru_stime,
                "load_before": load,
                "load_after": os.getloadavg(),
            }
        )
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; retained stdout/stderr and command")
        print(name, flush=True)

    for pair, order in enumerate(metadata["plan"]["orders"]):
        for arm in order:
            checkout = before if arm == "before" else root
            encode = [
                str(checkout / "managym/target/release/examples/bench_observation")
            ]
            if arm == "refactor":
                encode.append("--legacy")
            run(f"encode-{arm}-{pair}.jsonl", encode, checkout)
            for batch in (1, 64):
                command = [
                    "uv",
                    "run",
                    "--no-project",
                    "--python",
                    str(root / ".venv/bin/python"),
                    "python",
                    str(checkout / "scripts/bench_breakdown.py"),
                    "--num-envs",
                    str(batch),
                    "--steps",
                    str(args.steps),
                    "--matched-corpus",
                    "--policy-input-version",
                    "2" if arm == "current" else "1",
                    "--json",
                ]
                run(f"vector-{arm}-{batch}-{pair}.json", command, checkout)
    (out / "comparison.json").write_text(json.dumps(compare(out), indent=2) + "\n")
    final_sources = {
        "before": source_identity(before),
        "current": source_identity(root),
    }
    if final_sources != manifests:
        raise RuntimeError("source changed during measurement; evidence is not final")
    final_binaries = {
        key: sha256(
            (before if key.startswith("before/") else root) / key.split("/", 1)[1]
        )
        for key in binaries
    }
    if final_binaries != binaries:
        raise RuntimeError("binary changed during measurement; evidence is not final")
    (out / "identity-verified.json").write_text(
        json.dumps({"sources_unchanged": True, "binaries_unchanged": True}) + "\n"
    )


if __name__ == "__main__":
    main()
