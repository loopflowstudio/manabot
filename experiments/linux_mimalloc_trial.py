"""Exploratory Linux ABBA benchmark comparing system malloc vs mimalloc on Colab."""

from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from etude.jobs import TrainingJob
from etude.train.tracking import track
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
RUST_TOOLCHAIN = "1.98.1"
SETUP = (
    "set -eu\n"
    "{\n"
    "if command -v flock >/dev/null; then flock 9; fi\n"
    'test -x "$HOME/.cargo/bin/rustup" || curl --proto =https --tlsv1.2 -sSf '
    "https://sh.rustup.rs | sh -s -- -y --profile minimal "
    f"--default-toolchain {RUST_TOOLCHAIN}\n"
    f'export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN}\n'
    f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal\n"
    '} 9>"$HOME/.manabot-rust-setup.lock"\n'
    "uv sync --locked --python 3.13 --extra play\n"
)


class Config(BaseModel):
    bank_mib: int = 1024


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(
            result.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
            flush=True,
        )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def probe_text(cmd: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def train(config: Config) -> None:
    repo = Path(__file__).resolve().parents[1]
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    env = dict(os.environ)
    env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"
    env["RUSTUP_TOOLCHAIN"] = RUST_TOOLCHAIN
    maturin = str(Path(sys.executable).parent / "maturin")
    manifest = str(repo / "managym" / "Cargo.toml")

    with track(clocks=("phase",), config=config) as run:
        out = run.out
        out.mkdir(parents=True, exist_ok=True)

        host = {
            "uname": probe_text(["uname", "-a"], env),
            "os_release": Path("/etc/os-release").read_text()
            if Path("/etc/os-release").is_file()
            else "",
            "lscpu": probe_text(["lscpu"], env),
            "nproc": probe_text(["nproc"], env),
            "free_m": probe_text(["free", "-m"], env),
            "rustc": probe_text(["rustc", "--version"], env),
            "cargo": probe_text(["cargo", "--version"], env),
            "python": sys.version,
            "ldd": probe_text(["ldd", "--version"], env),
        }
        (out / "host.json").write_text(json.dumps(host, indent=2))
        print("=== HOST ===", flush=True)
        print(json.dumps({k: v.splitlines()[0] if v else "" for k, v in host.items()}, indent=2), flush=True)

        system_target = Path("/tmp/ecca6b-system-target")
        mimalloc_target = Path("/tmp/ecca6b-mimalloc-target")

        # 1. Build Engine Budget binaries (system vs mimalloc, no counting)
        run_cmd(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                manifest,
                "--target-dir",
                str(system_target),
                "--example",
                "bench_steps",
                "--example",
                "bench_search",
            ],
            env=env,
            cwd=repo,
        )
        run_cmd(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                manifest,
                "--target-dir",
                str(mimalloc_target),
                "--features",
                "mimalloc",
                "--example",
                "bench_steps",
                "--example",
                "bench_search",
            ],
            env=env,
            cwd=repo,
        )

        # 2. Run matched ABBA Engine Budget (mimalloc candidate vs system reference)
        budget_dir = out / "allocator-budget"
        run_cmd(
            [
                sys.executable,
                str(repo / "scripts" / "engine_budget.py"),
                "sample",
                "--candidate",
                str(mimalloc_target / "release" / "examples"),
                "--reference",
                str(system_target / "release" / "examples"),
                "--artifact",
                str(budget_dir),
            ],
            env=env,
            cwd=repo,
        )
        compare = run_cmd(
            [
                sys.executable,
                str(repo / "scripts" / "engine_budget.py"),
                "compare",
                "--artifact",
                str(budget_dir),
                "--reference",
                str(budget_dir / "reference"),
            ],
            env=env,
            cwd=repo,
        )
        (out / "budget-compare.txt").write_text(compare.stdout)

        # 3. Build and run bench_footprint (exact and cold, 10k-clone RSS) on both allocators
        run_cmd(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                manifest,
                "--target-dir",
                str(system_target),
                "--features",
                "benchmark",
                "--example",
                "bench_footprint",
            ],
            env=env,
            cwd=repo,
        )
        run_cmd(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                manifest,
                "--target-dir",
                str(mimalloc_target),
                "--features",
                "benchmark,mimalloc",
                "--example",
                "bench_footprint",
            ],
            env=env,
            cwd=repo,
        )
        footprint_outputs: dict[str, list[dict[str, object]]] = {}
        for side, target in (("system", system_target), ("mimalloc", mimalloc_target)):
            binary = str(target / "release" / "examples" / "bench_footprint")
            for mode in ("exact", "cold"):
                res = run_cmd(
                    [binary, mode, "--bank-mib", str(config.bank_mib)],
                    env=env,
                    cwd=repo,
                )
                key = f"{side}-{mode}"
                (out / f"footprint-{key}.jsonl").write_text(res.stdout)
                footprint_outputs[key] = [
                    json.loads(line) for line in res.stdout.splitlines() if line.strip()
                ]

        # 4. Build both release Python wheels and run 4-worker ShardedVectorEnv ABBA
        wheels_root = Path("/tmp/ecca6b-wheels")
        if wheels_root.exists():
            shutil.rmtree(wheels_root)
        site_dirs: dict[str, Path] = {}
        for side, target, features in (
            ("system", system_target, "python"),
            ("mimalloc", mimalloc_target, "python,mimalloc"),
        ):
            wheel_dir = wheels_root / side
            wheel_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(
                [
                    maturin,
                    "build",
                    "--release",
                    "--locked",
                    "--manifest-path",
                    manifest,
                    "--target-dir",
                    str(target),
                    "--features",
                    features,
                    "-i",
                    sys.executable,
                    "--out",
                    str(wheel_dir),
                ],
                env=env,
                cwd=repo,
            )
            wheels = list(wheel_dir.glob("managym-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"expected one wheel for {side}, got {wheels}")
            site = wheels_root / f"site-{side}"
            site.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(wheels[0]) as archive:
                archive.extractall(site)
            site_dirs[side] = site

        sharded_rows: list[dict[str, object]] = []
        for side in ("mimalloc", "system", "system", "mimalloc"):
            run_env = dict(env)
            run_env["PYTHONPATH"] = f"{site_dirs[side]}:{repo}"
            res = run_cmd(
                [
                    sys.executable,
                    str(repo / "scripts" / "bench_sharded_collection.py"),
                    "--expect-allocator",
                    side,
                ],
                env=run_env,
                cwd=repo,
            )
            row = json.loads(res.stdout.strip().splitlines()[-1])
            sharded_rows.append(row)
        (out / "sharded-abba.jsonl").write_text(
            "\n".join(json.dumps(r) for r in sharded_rows) + "\n"
        )

        # 5. Debug match_state_hash_tests with mimalloc on Linux
        run_cmd(
            [
                "cargo",
                "test",
                "--locked",
                "--manifest-path",
                manifest,
                "--target-dir",
                str(mimalloc_target),
                "--features",
                "mimalloc",
                "--test",
                "match_state_hash_tests",
            ],
            env=env,
            cwd=repo,
        )

        summary = {
            "host_uname": host["uname"],
            "nproc": host["nproc"],
            "budget_compare": compare.stdout,
            "sharded_abba": sharded_rows,
            "footprint": footprint_outputs,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("=== SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)


def training_jobs() -> list[TrainingJob[Config]]:
    return [
        TrainingJob(
            name="linux-mimalloc-trial",
            train=train,
            config=Config(),
            seeds=(2, 3),
            repo=str(ROOT),
            setup=SETUP,
            extras=("play",),
            max_hours=0.5,
            max_usd=Decimal("3.00"),
            max_attempts=1,
        )
    ]
