"""Collect matched layer-risk receipts sequentially on one otherwise idle runner.

The historical pre-mitigation engine receives instrumentation and current fixtures
only. Archived sources are measurement inputs, never Task worktrees. Ordinary
reference benchmarks remain separate from dependency semantics that they lack.
Run this script with uv; the dedicated workflow supplies the isolated runner.
"""

import argparse
from collections.abc import Sequence
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "fb3555ffbd36bb797908110fc549999fe8e1a381"
LEGS = ("baseline", "before", "after")
# The producer and comparator share exact case identities. File counts alone
# cannot distinguish a complete run from duplicated or mislabeled receipts.
POLICY_CASES = {
    f"input-v{version}-{state}.json": (state, version)
    for state, versions in (
        ("ordinary", (6, 7, 8)),
        ("shared", (6, 7, 8)),
        ("program", (8,)),
    )
    for version in versions
}
# JSON is validated at receipt comparisons; it has heterogeneous nested fields.
type Record = dict[str, Any]


def run(command: Sequence[str], cwd: Path = ROOT, output: Path | None = None) -> None:
    print(f"[{cwd.name}] {' '.join(command)}", flush=True)
    env = dict(
        os.environ, PYTHONPATH=str(cwd), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    if output is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    else:
        with output.open("w") as stream:
            subprocess.run(command, cwd=cwd, env=env, stdout=stream, check=True)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path: Path) -> Record:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def rows(path: Path) -> list[Record]:
    result = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not result or not all(isinstance(row, dict) for row in result):
        raise ValueError(f"Missing or malformed receipts: {path}")
    return result


def source(out: Path, leg: str) -> Path:
    return ROOT if leg == "after" else out / "sources" / leg


def ensure_revision(revision: str) -> None:
    """Fetch a missing pinned commit; never substitute a branch's current tip."""
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    )
    if probe.returncode:
        run(["git", "fetch", "--no-tags", "origin", revision], cwd=ROOT)
    git("cat-file", "-e", f"{revision}^{{commit}}")


def prepare(out: Path) -> None:
    # Never replace earlier receipts or a caller's files.
    out.mkdir(parents=True, exist_ok=False)
    reference = read(ROOT / "managym/perf/baseline.json")["reference"]["commit"]
    revisions = {
        "baseline": reference,
        "before": BEFORE,
        "after": git("rev-parse", "HEAD").decode().strip(),
    }
    (out / "candidate.patch").write_bytes(
        git("diff", "HEAD", "--", "managym", "manabot", "scripts")
    )
    harnesses = [
        ROOT / "scripts/measure_layer_risk.py",
        ROOT / "scripts/bench_layer_input.py",
        ROOT / "managym/perf/layers-attribution.patch",
        ROOT / "managym/src/benchmark_layers.rs",
        ROOT / "manabot/verify/representation.py",
    ]
    harnesses += list((ROOT / "managym/examples").glob("bench_*.rs"))
    harnesses += list((ROOT / "managym/examples/curated").glob("*.rs"))
    write(
        out / "measurement.json",
        {
            "revisions": revisions,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "initial_load": os.getloadavg(),
            "run_url": os.environ.get("RUN_URL"),
            "harness_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in harnesses
            },
            "limitations": [
                "Before receives counters/fixture access only, no cache or model fixes.",
                "Baseline cannot validate dependency semantics; compare its original TLA workloads only.",
                "Timers on instrumented layer runs overlap; allocation runs are separate.",
                "Step samples are the first 128 decisions, not a random population sample.",
                "Policy is CPU float32, B16, one thread, two warmups/eight samples, no optimizer.",
                "Peak RSS covers the fresh case process; accelerator and learning efficacy are unmeasured.",
            ],
        },
    )
    # Full branch history need not include a reference retained after a squash.
    # Acquire both exact historical commits before extracting either source tree.
    for revision in (reference, BEFORE):
        ensure_revision(revision)
    for leg in LEGS:
        (out / leg).mkdir()
        if leg == "after":
            continue
        dest = source(out, leg)
        dest.mkdir(parents=True)
        data = git(
            "archive",
            revisions[leg],
            "managym",
            "manabot",
            "content",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "academy",
            "docs/experiment-metrics.md",
            "protocol/experience-v1.schema.json",
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(dest, filter="data")
        # All legs execute the same original TLA workload and digest code.
        shutil.copytree(
            ROOT / "managym/examples/curated",
            dest / "managym/examples/curated",
            dirs_exist_ok=True,
        )
        for name in ("bench_steps.rs", "bench_search.rs"):
            shutil.copy2(
                ROOT / "managym/examples" / name, dest / "managym/examples" / name
            )
    before = source(out, "before")
    run(
        [
            "git",
            "apply",
            "--unidiff-zero",
            "--unsafe-paths",
            "--directory",
            str(before),
            str(ROOT / "managym/perf/layers-attribution.patch"),
        ]
    )
    # `before` isolates the engine, so it runs the candidate's card definitions.
    # State digests bind definitions; a card repair made after BEFORE (Honest
    # Work's static order) would otherwise differ in every compared receipt.
    for name in (
        "managym/examples/bench_layers.rs",
        "managym/src/benchmark_layers.rs",
        "manabot/verify/representation.py",
        "managym/cards/src/tla.rs",
    ):
        shutil.copy2(ROOT / name, before / name)
    with (before / "managym/Cargo.toml").open("a") as manifest:
        manifest.write(
            '\n[[example]]\nname = "bench_layers"\nrequired-features = ["benchmark"]\ntest = true\n'
        )


def native(out: Path, leg: str, payment_only: bool) -> None:
    tree = source(out, leg)
    receipt = out / leg
    cargo = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "-j",
        "2",
        "--manifest-path",
        str(tree / "managym/Cargo.toml"),
    ]
    binary = tree / "managym/target/release/examples"
    if not payment_only:
        run([*cargo, "--example", "bench_steps", "--example", "bench_search"])
        native_provenance(receipt, binary, ("bench_steps", "bench_search"))
        for name, args in (("bench_steps", ["--tail-samples"]), ("bench_search", [])):
            run([str(binary / name), *args], tree, receipt / f"{name}.jsonl")
    if leg == "baseline":
        return
    run([*cargo, "--features", "benchmark", "--example", "bench_layers"])
    native_provenance(
        receipt,
        binary,
        ("bench_layers",)
        if payment_only
        else ("bench_steps", "bench_search", "bench_layers"),
    )
    run(
        [str(binary / "bench_layers"), "--payment-only", "--allocations"],
        tree,
        receipt / "payment.jsonl",
    )
    if not payment_only:
        run([str(binary / "bench_layers")], tree, receipt / "layers.jsonl")
        run(
            [str(binary / "bench_layers"), "--allocations"],
            tree,
            receipt / "layers-allocations.jsonl",
        )


def native_provenance(receipt: Path, binary: Path, names: Sequence[str]) -> None:
    write(
        receipt / "native-provenance.json",
        {
            "load_at_recording": os.getloadavg(),
            "sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in binary.iterdir()
                if p.name in names
            },
        },
    )


def model(out: Path, leg: str, inventory: bool) -> None:
    if leg == "baseline":
        raise ValueError("The historical Engine Budget reference predates v8")
    tree = source(out, leg)
    wheels = out / leg / "wheels"
    run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "--no-sync",
            "maturin",
            "build",
            "--release",
            "-j",
            "2",
            "-i",
            str(ROOT / ".venv/bin/python"),
            "-m",
            str(tree / "managym/Cargo.toml"),
            "-o",
            str(wheels),
        ],
        tree,
    )
    (wheel,) = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (extension,) = [
            name
            for name in archive.namelist()
            if name.startswith("manabot/managym/_managym.") and name.endswith(".so")
        ]
        dest = tree / extension
        temporary = dest.with_suffix(".new")
        temporary.write_bytes(archive.read(extension))
        temporary.replace(dest)
    for name, (state, version) in POLICY_CASES.items():
        run(
            [
                "uv",
                "run",
                "--project",
                str(ROOT),
                "--no-sync",
                "python",
                str(ROOT / "scripts/bench_layer_input.py"),
                "--version",
                str(version),
                "--state",
                state,
                *(["--inventory"] if inventory else []),
            ],
            tree,
            out / leg / name,
        )


def payment_comparison(before: list[Record], after: list[Record]) -> list[Record]:
    expected = [
        (funding, sample) for funding in ("pool", "lands") for sample in range(8)
    ]
    for receipts in (before, after):
        if [(row["funding"], row["sample"]) for row in receipts] != expected:
            raise ValueError(
                "Expected eight complete payments for each of pool and lands"
            )
    results = []
    for a, b in zip(before, after, strict=True):
        if not a["counting_allocations"] or not b["counting_allocations"]:
            raise ValueError("Payment attribution requires allocation-enabled receipts")
        for key in ("funding", "sample", "before", "trace"):
            if a[key] != b[key]:
                raise ValueError(f"Payment {a['funding']}/{a['sample']} differs: {key}")
        results.append(
            {
                "funding": a["funding"],
                "sample": a["sample"],
                "semantics_equal": True,
                "before_allocations": a["allocation_count"],
                "after_allocations": b["allocation_count"],
                "before_bytes": a["allocated_bytes"],
                "after_bytes": b["allocated_bytes"],
                "before_metrics": a["metrics"],
                "after_metrics": b["metrics"],
            }
        )
    return results


def validate_samples(samples: object, count: int) -> None:
    """Require every bounded timing sample, not just a plausible summary."""
    if (
        not isinstance(samples, list)
        or len(samples) != count
        or not all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in samples
        )
    ):
        raise ValueError(f"Expected {count} finite positive timing samples")


def policy_comparison(out: Path) -> list[Record]:
    results = []
    for name, (state, version) in POLICY_CASES.items():
        a, b = (read(out / leg / name) for leg in ("before", "after"))
        for receipt in (a, b):
            if receipt["inventory_only"]:
                raise ValueError("Inventory is not a policy timing receipt")
            if (receipt["state"], receipt["version"]) != (state, version):
                raise ValueError(f"Mislabeled policy case: {name}")
            for key in (
                "observation_bytes",
                "batch_observation_bytes",
                "parameter_bytes",
                "peak_rss_bytes",
            ):
                size = receipt.get(key)
                if type(size) is not int or size <= 0:
                    raise ValueError(f"Missing or invalid policy memory: {name}/{key}")
            for operation in (
                "native_encode",
                "python_encode",
                "observation",
                "inference",
                "forward_backward",
            ):
                timing = receipt[operation]
                validate_samples(timing["samples_s"], 8)
                if timing["sample_count"] != 8 or timing["warmups"] != 2:
                    raise ValueError(f"Unmatched policy sampling: {name}/{operation}")
        for key in (
            "observation_sha256",
            "observation_config",
            "model_config",
            "batch",
            "device",
            "dtype",
            "threads",
            "interop_threads",
            "torch",
            "platform",
            "observation_bytes",
            "batch_observation_bytes",
            "live_details",
            "live_facts",
        ):
            if a[key] != b[key]:
                raise ValueError(f"Unmatched policy input: {name}/{key}")
        results.append({"case": Path(name).stem, "before": a, "after": b})
    return results


def semantic_rows(receipts: list[Record]) -> list[Record]:
    """Compare complete workload identities; timings/counters cannot buy equality."""
    keys = (
        "workload",
        "seed",
        "decision",
        "steps",
        "games",
        "capped",
        "digest",
        "score_digest_blake3",
        "decisions",
        "simulations",
        "cap_hits",
        "action_counts",
        "actions",
    )
    result = [
        {key: row[key] for key in keys if key in row}
        for row in receipts
        if "digest" in row
    ]
    if not result:
        raise ValueError("No complete workload digests")
    return result


def validate_workload(name: str, receipts: list[Record]) -> None:
    """A truncated run cannot become a successful same-runner comparison."""
    identified = semantic_rows(receipts)
    if name == "bench_search":
        if [row.get("seed") for row in identified] != [1, 2, 3]:
            raise ValueError("Expected all three original search seeds")
        for row in receipts:
            if "seed" in row and (
                not 0 < row["decisions"] <= 40
                or not row["decisions"]
                == len(row["samples_s"])
                == len(row["action_counts"])
                or len(row["score_digest_blake3"]) != 64
            ):
                raise ValueError("Incomplete search samples/counts/full digest")
    elif name == "bench_steps":
        if len(identified) != 1 or identified[0]["games"] != 512:
            raise ValueError("Expected all 512 original TLA games")
        samples = [
            row["step_prefix_samples_s"]
            for row in receipts
            if "step_prefix_samples_s" in row
        ]
        if len(samples) != 1 or len(samples[0]) != 128:
            raise ValueError("Missing bounded step prefix")
    else:
        if len(identified) != 9 or identified[0].get("games") != 16:
            raise ValueError("Expected 16 dependency games and eight search decisions")
        if [row.get("decision") for row in identified[1:]] != list(range(8)):
            raise ValueError("Incomplete dependency search")
        steps = next(
            row for row in receipts if row.get("workload") == "dependency_steps"
        )
        if len(steps["step_prefix_samples_s"]) != 128:
            raise ValueError("Missing dependency step prefix")


def compare(out: Path, payment_only: bool) -> None:
    result: Record = {
        "payment": payment_comparison(
            rows(out / "before/payment.jsonl"), rows(out / "after/payment.jsonl")
        )
    }
    if not payment_only:
        result["workloads"] = {}
        for name in ("bench_steps", "bench_search", "layers", "layers-allocations"):
            a, b = (rows(out / leg / f"{name}.jsonl") for leg in ("before", "after"))
            validate_workload(name, a)
            validate_workload(name, b)
            if semantic_rows(a) != semantic_rows(b):
                raise ValueError(f"Mitigation changed workload semantics: {name}")
            result["workloads"][name] = {
                "mitigation_semantics_equal": True,
                "before": a,
                "after": b,
            }
            if name.startswith("bench_"):
                baseline = rows(out / "baseline" / f"{name}.jsonl")
                validate_workload(name, baseline)
                result["workloads"][name].update(
                    baseline=baseline,
                    baseline_semantics_equal=semantic_rows(baseline)
                    == semantic_rows(b),
                )
        result["policy"] = policy_comparison(out)
    write(out / "comparison.json", result)
    print(
        f"Matched receipts: {out / 'comparison.json'}; Engine Budget acceptance remains separate."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "native", "model", "compare"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--leg", choices=LEGS)
    parser.add_argument("--payment-only", action="store_true")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Model parity/storage only; cannot satisfy timing comparison",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase in ("native", "model") and args.leg is None:
        parser.error("native/model require --leg")
    if args.phase == "prepare":
        prepare(out)
    elif args.phase == "native":
        native(out, args.leg, args.payment_only)
    elif args.phase == "model":
        model(out, args.leg, args.inventory)
    else:
        compare(out, args.payment_only)


if __name__ == "__main__":
    main()
