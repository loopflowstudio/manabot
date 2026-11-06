"""The budget rejects slowdowns and changed digests, including during speedups."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/engine_budget.py"
BASELINE = ROOT / "managym/perf/baseline.json"
WORKFLOW = ROOT / ".github/workflows/engine-budget.yml"

RUNNER = "GitHub-hosted ubuntu-24.04, AMD EPYC 7763 (4 vCPU), release"
BASE = {
    "tolerance": 0.10,
    "reference": {
        "commit": "abc123def456",
        "date": "2026-11-02",
        "run": "https://x",
        "runner": RUNNER,
        "reason": "test",
    },
    "bench_search": {
        "worst_s": 10.0,
        "total_s": 300.0,
        "seeds": {
            "1": {"worst_s": 8.0, "total_s": 100.0, "digest": 11},
            "2": {"worst_s": 10.0, "total_s": 100.0, "digest": 22},
            "3": {"worst_s": 9.0, "total_s": 100.0, "digest": 33},
        },
    },
    "bench_steps": {"steps_per_s": 5000.0, "steps": 99923, "games": 512, "digest": 44},
}


def _run(*args: str | Path) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.returncode, completed.stdout + completed.stderr


def _write_run(
    directory: Path,
    *,
    scale: float = 1.0,
    search_digests: tuple[int, int, int] = (11, 22, 33),
    steps_digest: int = 44,
) -> Path:
    """A run directory of both benchmarks' output; `scale` multiplies every duration."""
    directory.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "seed": seed,
                "decisions": 40,
                "worst_s": round(worst * scale, 3),
                "median_s": 2.0,
                "total_s": round(100.0 * scale, 3),
                "digest": digest,
            }
        )
        for seed, worst, digest in zip((1, 2, 3), (8.0, 10.0, 9.0), search_digests)
    ]
    lines.append(
        json.dumps({"worst_overall_s": 10.0 * scale, "total_overall_s": 300.0 * scale})
    )
    (directory / "bench_search.jsonl").write_text("\n".join(lines) + "\n")
    (directory / "bench_steps.jsonl").write_text(
        json.dumps(
            {
                "games": 512,
                "capped": 0,
                "steps": 99923,
                "elapsed_s": 20.0 * scale,
                "steps_per_s": round(5000.0 / scale, 1),
                "digest": steps_digest,
            }
        )
        + "\n"
    )
    return directory


def _write_artifact(directory: Path, *, commit: str, runner: str, scale: float) -> Path:
    """What `gh run download --name engine-budget` leaves: a run plus measurement.json."""
    _write_run(directory, scale=scale)
    (directory / "measurement.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "date": "2026-11-03",
                "run": "https://github.com/loopflowstudio/manabot/actions/runs/1",
                "runner": runner,
            }
        )
    )
    return directory


def _compare(
    tmp_path: Path,
    *,
    reference_scale: float = 1.4,
    reference_digests: tuple[int, int, int] = (11, 22, 33),
    scale: float = 1.0,
    search_digests: tuple[int, int, int] = (11, 22, 33),
    steps_digest: int = 44,
) -> tuple[int, str]:
    """A slower runner than the recording (reference at 1.4x) hosts both runs."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    artifact = _write_run(
        tmp_path / "artifact",
        scale=scale,
        search_digests=search_digests,
        steps_digest=steps_digest,
    )
    reference = _write_run(
        artifact / "reference", scale=reference_scale, search_digests=reference_digests
    )
    return _run(
        "compare",
        "--baseline",
        baseline,
        "--artifact",
        artifact,
        "--reference",
        reference,
    )


def test_within_tolerance_on_a_slower_runner_passes(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=1.4 * 1.05)
    assert code == 0, output
    assert "PASS" in output
    assert "| search worst reply | 14.000 s | 14.700 s | ok |" in output


def test_speedup_passes(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.7)
    assert code == 0, output


def test_twenty_percent_slowdown_fails(tmp_path: Path) -> None:
    """The controlled fixture: the candidate reported 20% slower than the reference."""
    code, output = _compare(tmp_path, scale=1.4 * 1.2)
    assert code == 1, output
    assert (
        "| search worst reply | 14.000 s | 16.800 s | over 15.400 s (+10%) |" in output
    )
    assert "| search total | 420.000 s | 504.000 s | over 462.000 s (+10%) |" in output
    assert (
        "| steps throughput | 3571.4/s | 2976.2/s | under 3246.7/s (-10%) |" in output
    )
    assert "FAIL: 3 check(s)" in output
    assert "record --reason" in output


def test_changed_search_digest_fails_even_when_faster(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.8, search_digests=(11, 23, 33))
    assert code == 1, output
    assert "| candidate search seed 2 digest | 22 | 23 | differs |" in output
    assert "FAIL: 1 check(s)" in output


def test_changed_steps_digest_fails_even_when_faster(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.8, steps_digest=45)
    assert code == 1, output
    assert "| candidate steps digest | 44 | 45 | differs |" in output


def test_reference_not_matching_the_file_fails_the_file(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=1.4, reference_digests=(11, 22, 34))
    assert code == 1, output
    assert (
        "| baseline file | abc123def | digests differ | digests do not describe the reference |"
        in output
    )
    assert "| reference search seed 3 digest | 33 | 34 | differs |" in output


def test_record_adopts_a_downloaded_artifact_with_its_reason(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    artifact = _write_artifact(
        tmp_path / "first", commit="0123456789abcdef", runner=RUNNER, scale=1.3
    )
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", artifact,
        "--reason", "T6 adds Sagas to the curated decks",
    )  # fmt: skip
    assert code == 0, output
    recorded = json.loads(baseline.read_text())
    assert recorded["reference"] == {
        "commit": "0123456789abcdef",
        "date": "2026-11-03",
        "run": "https://github.com/loopflowstudio/manabot/actions/runs/1",
        "runner": RUNNER,
        "reason": "T6 adds Sagas to the curated decks",
    }
    assert recorded["bench_search"]["worst_s"] == 13.0
    assert recorded["bench_search"]["total_s"] == 390.0
    assert recorded["bench_steps"]["steps_per_s"] == pytest.approx(
        5000.0 / 1.3, abs=0.1
    )
    # The tolerance is the file's own; only an edit of the file moves it.
    artifact = _write_artifact(
        tmp_path / "second",
        commit="fedcba9876543210",
        runner="GitHub-hosted ubuntu-24.04, Intel Xeon 6973P-C (4 vCPU), release",
        scale=1.0,
    )
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", artifact,
        "--reason", "indexed effect records",
    )  # fmt: skip
    assert code == 0, output
    again = json.loads(baseline.read_text())
    assert again["tolerance"] == 0.10
    assert again["reference"]["commit"] == "fedcba9876543210"
    assert again["bench_search"]["worst_s"] == 10.0


def test_record_refuses_a_directory_without_a_measurement(tmp_path: Path) -> None:
    """Hand-assembled output is not a recording: the job's measurement.json names the run."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    artifact = _write_run(tmp_path / "artifact")
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", artifact, "--reason", "x"
    )
    assert code == 1, output
    assert "not an Engine Budget artifact" in output
    assert "gh run download <run-id> --name engine-budget" in output
    assert json.loads(baseline.read_text()) == BASE


def test_committed_baseline_is_complete() -> None:
    """The committed file names a reachable reference, its runner, run and reason."""
    baseline = json.loads(BASELINE.read_text())
    assert baseline["tolerance"] == 0.10
    reference = baseline["reference"]
    assert len(reference["commit"]) == 40 and reference["run"].startswith("https://")
    assert reference["runner"] and reference["reason"]
    assert set(baseline["bench_search"]["seeds"]) == {"1", "2", "3"}
    assert baseline["bench_steps"]["games"] == 512
    assert baseline["bench_steps"]["steps_per_s"] > 0


# Execute the checked-in run bodies with Actions' shell flags. This deliberately
# follows this workflow's small YAML shape, without adding a YAML dependency or
# pretending to emulate the Actions service (checkout, upload and scheduling).
def _workflow_step(name: str) -> str:
    return (
        WORKFLOW.read_text()
        .split(f"      - name: {name}\n", 1)[1]
        .split("      - ", 1)[0]
    )


def _workflow_run(
    name: str, directory: Path, **environment: str
) -> subprocess.CompletedProcess[str]:
    step = _workflow_step(name)
    run = re.search(r"^        run: (.+)$", step, re.MULTILINE)
    assert run is not None
    script = run[1]
    if script == "|":
        body = step[run.end() + 1 :]
        lines = []
        for line in body.splitlines():
            if line and not line.startswith("          "):
                break
            lines.append(line)
        script = textwrap.dedent("\n".join(lines))
    defaults = WORKFLOW.read_text().split("    steps:", 1)[0]
    shell = re.search(r"^        shell: (\S+)$", step, re.MULTILINE) or re.search(
        r"^        shell: (\S+)$", defaults, re.MULTILINE
    )
    assert shell is None or shell[1] == "bash", (
        "Add the configured shell's Actions flags"
    )
    # An unspecified Linux shell uses bash -e; explicit bash adds pipefail.
    flags = ["--noprofile", "--norc", "-eo", "pipefail"] if shell else ["-e"]
    return subprocess.run(
        ["bash", *flags, "-c", script],
        cwd=directory,
        env={
            **os.environ,
            "GITHUB_STEP_SUMMARY": str(directory / "summary.md"),
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


def _executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("scale", "digests", "code", "message"),
    [
        (1.05, (11, 22, 33), 0, "PASS"),
        (1.2, (11, 22, 33), 1, "FAIL: 3 check(s)"),
        (0.8, (11, 23, 33), 1, "FAIL: 1 check(s)"),
    ],
)
def test_workflow_comparison_propagates_failure_through_tee(
    tmp_path: Path,
    scale: float,
    digests: tuple[int, int, int],
    code: int,
    message: str,
) -> None:
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts/engine_budget.py")
    baseline = tmp_path / "managym/perf/baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps(BASE))
    artifact = _write_run(
        tmp_path / ".cache/engine-budget", scale=scale, search_digests=digests
    )
    _write_run(artifact / "reference")

    result = _workflow_run("Compare the candidate with the reference", tmp_path)

    assert result.returncode == code, result.stdout + result.stderr
    assert message in result.stdout
    assert (tmp_path / "summary.md").read_text() == result.stdout


@pytest.mark.parametrize("side", ["candidate", "reference"])
@pytest.mark.parametrize("benchmark", ["bench_steps", "bench_search"])
def test_workflow_benchmark_failure_retains_raw_output(
    tmp_path: Path, side: str, benchmark: str
) -> None:
    fixture = _write_run(tmp_path / "fixture")
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts/engine_budget.py")
    for role, tree in (("candidate", "managym"), ("reference", "other/managym")):
        for name in ("bench_steps", "bench_search"):
            failure = role == side and name == benchmark
            _executable(
                tmp_path / tree / "target/release/examples" / name,
                f"cat {shlex.quote(str(fixture / f'{name}.jsonl'))}\nexit {17 if failure else 0}",
            )
    result = _workflow_run(
        "Run the paired benchmarks",
        tmp_path,
        CANDIDATE="managym",
        MEASURE="",
        REFERENCE_BUILD="success",
    )
    assert result.returncode == 17, result.stdout + result.stderr
    ordinal = 0 if side == "candidate" else 1
    artifact = tmp_path / ".cache/engine-budget"
    raw = artifact / "samples" / f"{ordinal}-{side}"
    assert (raw / f"{benchmark}.jsonl").read_text() == (
        fixture / f"{benchmark}.jsonl"
    ).read_text()
    evidence = json.loads((artifact / "sampling.json").read_text())
    assert evidence["samples"][-1]["exit_code"] == 17
    if benchmark == "bench_steps":
        assert not (raw / "bench_search.jsonl").exists()
    assert "if: always()" in _workflow_step("Record the measurement")
    if side == "reference":
        binaries = tmp_path / "metadata-bin"
        _executable(binaries / "git", "echo " + "a" * 40)
        _executable(binaries / "lscpu", "echo 'Model name: controlled CPU'")
        _executable(binaries / "nproc", "echo 4")
        recorded = _workflow_run(
            "Record the measurement",
            tmp_path,
            PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
            CANDIDATE="managym",
            RUN_URL="https://example.test/run",
            RUNNER_OS="Linux",
        )
        assert recorded.returncode == 0, recorded.stderr
        assert (artifact / "measurement.json").exists()


def test_paired_sampling_retains_both_orders_and_rejects_a_changed_digest(
    tmp_path: Path,
) -> None:
    candidate = _write_run(tmp_path / "candidate", scale=1.2)
    reference = _write_run(tmp_path / "reference")
    for side, source in (("candidate", candidate), ("reference", reference)):
        for name in ("bench_steps", "bench_search"):
            _executable(
                tmp_path / "bin" / side / name,
                f"cat {shlex.quote(str(source / f'{name}.jsonl'))}",
            )
    artifact = tmp_path / "artifact"
    code, output = _run(
        "sample",
        "--candidate",
        tmp_path / "bin/candidate",
        "--reference",
        tmp_path / "bin/reference",
        "--artifact",
        artifact,
    )
    assert code == 0, output
    evidence = json.loads((artifact / "sampling.json").read_text())
    assert evidence["order"] == ["candidate", "reference", "reference", "candidate"]
    assert len(evidence["samples"]) == 8
    assert all("involuntary_switches" in sample for sample in evidence["samples"])
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    code, output = _run(
        "compare",
        "--artifact",
        artifact,
        "--reference",
        artifact / "reference",
        "--baseline",
        baseline,
    )
    assert code == 1 and "FAIL: 3 check(s)" in output
    retained = (artifact / "sampling.json").read_bytes()
    code, output = _run(
        "sample",
        "--candidate",
        tmp_path / "bin/candidate",
        "--reference",
        tmp_path / "bin/reference",
        "--artifact",
        artifact,
    )
    assert code != 0 and "fresh artifact directory" in output
    assert (artifact / "sampling.json").read_bytes() == retained
    # A differing second sample cannot disappear behind averaging.
    changed = _write_run(tmp_path / "changed", steps_digest=45)
    _executable(
        tmp_path / "bin/candidate/bench_steps",
        f"if [ -f {shlex.quote(str(tmp_path / 'seen'))} ]; then cat {shlex.quote(str(changed / 'bench_steps.jsonl'))}; "
        f"else touch {shlex.quote(str(tmp_path / 'seen'))}; cat {shlex.quote(str(candidate / 'bench_steps.jsonl'))}; fi",
    )
    code, output = _run(
        "sample",
        "--candidate",
        tmp_path / "bin/candidate",
        "--reference",
        tmp_path / "bin/reference",
        "--artifact",
        tmp_path / "changed-artifact",
    )
    assert code != 0 and "disagree on workload or digest" in output


def test_order_balanced_sampling_uses_both_durations(tmp_path: Path) -> None:
    for side, scales in (("candidate", (1.2, 0.8)), ("reference", (0.9, 1.1))):
        sources = [
            _write_run(tmp_path / f"{side}-{i}", scale=scale)
            for i, scale in enumerate(scales)
        ]
        for name in ("bench_steps", "bench_search"):
            seen = shlex.quote(str(tmp_path / f"{side}-{name}-seen"))
            first, second = [
                shlex.quote(str(source / f"{name}.jsonl")) for source in sources
            ]
            _executable(
                tmp_path / "bin" / side / name,
                f"if [ -f {seen} ]; then cat {second}; else touch {seen}; cat {first}; fi",
            )
    artifact = tmp_path / "artifact"
    code, output = _run(
        "sample",
        "--candidate",
        tmp_path / "bin/candidate",
        "--reference",
        tmp_path / "bin/reference",
        "--artifact",
        artifact,
    )
    assert code == 0, output
    for directory in (artifact, artifact / "reference"):
        steps = json.loads((directory / "bench_steps.jsonl").read_text())
        # Benchmark rates are published to 0.1/s before conversion to durations.
        assert steps["steps_per_s"] == pytest.approx(5000, abs=0.1)
        seeds = [
            json.loads(line)
            for line in (directory / "bench_search.jsonl").read_text().splitlines()
        ]
        assert sum(seed["total_s"] for seed in seeds) == pytest.approx(300)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    code, output = _run(
        "compare",
        "--artifact",
        artifact,
        "--reference",
        artifact / "reference",
        "--baseline",
        baseline,
    )
    assert code == 0, output


def test_reference_build_failure_leaves_an_adoptable_candidate(tmp_path: Path) -> None:
    """Real workflow shells and record CLI; injected binaries, no Rust build or load.

    The upload action is represented by copying its declared path after failure.
    GitHub's artifact transport itself still requires PR-path verification.
    """
    fixture = _write_run(tmp_path / "fixture", scale=1.2)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts/engine_budget.py")
    for name in ("bench_steps", "bench_search"):
        _executable(
            tmp_path / "managym/target/release/examples" / name,
            f"cat {shlex.quote(str(fixture / f'{name}.jsonl'))}",
        )
    binaries = tmp_path / "bin"
    _executable(
        binaries / "cargo",
        'case "$*" in\n'
        '  *other/managym/Cargo.toml*) echo "controlled reference build failure" >&2; exit 42;;\n'
        "  *managym/Cargo.toml*) exit 0;;\n"
        "  *) exit 99;;\nesac",
    )
    commit = "a" * 40
    _executable(binaries / "git", f"echo {commit}")
    _executable(binaries / "lscpu", "echo 'Model name: controlled CPU'")
    _executable(binaries / "nproc", "echo 4")
    environment = {
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "CANDIDATE": "managym",
        "RUN_URL": "https://github.com/loopflowstudio/manabot/actions/runs/1",
        "RUNNER_OS": "Linux",
        "MEASURE": "",
        "REFERENCE_BUILD": "failure",
    }
    # Follow source order and the reference build's explicit continue-on-error.
    # Its failure is enforced only after the candidate recording is retained.
    names = re.findall(r"^      - name: (.+)$", WORKFLOW.read_text(), re.MULTILINE)
    for name in names[names.index("Build the candidate benchmarks") :]:
        if name == "Upload the measurement":
            pytest.fail("The controlled reference failure did not occur")
        result = _workflow_run(name, tmp_path, **environment)
        if result.returncode:
            if name == "Build the reference benchmarks":
                assert result.returncode == 42
                assert "continue-on-error: true" in _workflow_step(name)
                continue
            assert name == "Require the reference build", result.stderr
            assert result.returncode == 1
            break
    upload = _workflow_step("Upload the measurement")
    assert "if: always()" in upload
    assert "uses: actions/upload-artifact@v4" in upload
    path = re.search(r"^          path: (.+)$", upload, re.MULTILINE)
    assert path is not None
    artifact = tmp_path / path[1]
    assert not (artifact / "reference").exists()
    downloaded = tmp_path / "downloaded"
    shutil.copytree(artifact, downloaded)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    reason = "Controlled proof: reference no longer builds with current benchmarks"
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", downloaded, "--reason", reason
    )
    assert code == 0, output
    recorded = json.loads(baseline.read_text())
    measurement = json.loads((downloaded / "measurement.json").read_text())
    assert measurement["commit"] == commit
    assert measurement["run"] == environment["RUN_URL"]
    assert recorded["reference"] == {**measurement, "reason": reason}
    assert recorded["tolerance"] == BASE["tolerance"]
    assert recorded["bench_search"]["total_s"] == 360.0
    assert recorded["bench_steps"]["digest"] == 44
