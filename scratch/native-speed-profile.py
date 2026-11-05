"""Sample the bounded release step/search workloads separately from timing."""

from pathlib import Path
import subprocess


def main() -> None:
    output = Path(".lf/logs/naming-speed")
    for name in ("bench_steps", "bench_search"):
        with (output / f"profile-{name}.jsonl").open("x") as log:
            process = subprocess.Popen(
                [str(output / "reference-bin" / name)], stdout=log
            )
            try:
                subprocess.run(
                    [
                        "sample",
                        str(process.pid),
                        "2",
                        "1",
                        "-file",
                        str(output / f"profile-{name}.txt"),
                    ],
                    check=True,
                    timeout=20,
                    capture_output=True,
                )
                process.wait(timeout=300)
                if process.returncode:
                    raise RuntimeError(f"{name} exited {process.returncode}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
