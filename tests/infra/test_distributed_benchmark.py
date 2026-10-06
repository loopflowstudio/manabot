"""Exercise the bounded supervisor without importing Torch or running training."""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import BinaryIO
import unittest
from unittest.mock import patch

from experiments.runners import distributed_benchmark as benchmark


class SupervisorTests(unittest.TestCase):
    def _attempt(
        self,
        mode: str,
        *,
        child: str | None = None,
        available: bool = True,
        launch_error: bool = False,
    ) -> dict[str, object]:
        real_popen = subprocess.Popen

        def launch(
            command: list[str],
            *,
            env: dict[str, str],
            stdout: BinaryIO,
            stderr: BinaryIO,
            start_new_session: bool,
        ) -> subprocess.Popen[bytes]:
            self.assertEqual(command[0:3], ["uv", "run", "--no-sync"])
            self.assertEqual(command[4], "experiments.runners.distributed_workloads")
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            if launch_error:
                raise OSError("fixture launch failure")
            return real_popen(
                [sys.executable, "-c", child or "raise SystemExit(0)"],
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=start_new_session,
            )

        with TemporaryDirectory() as directory, ExitStack() as stack:
            out = Path(directory) / "attempt"
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    [
                        "benchmark",
                        "--mode",
                        mode,
                        "--out",
                        str(out),
                        "--timeout",
                        "0.2" if child and "sleep" in child else "3",
                    ],
                )
            )
            stack.enter_context(patch.object(Path, "exists", return_value=available))
            stack.enter_context(
                patch.object(benchmark, "_read", return_value="fixture")
            )
            stack.enter_context(
                patch.object(
                    benchmark,
                    "_host",
                    return_value=benchmark.Host(
                        "fixture",
                        "fixture",
                        "fixture",
                        "unavailable",
                        1,
                        (0.0, 0.0, 0.0),
                    ),
                )
            )
            spawned = stack.enter_context(
                patch.object(benchmark.subprocess, "Popen", side_effect=launch)
            )
            try:
                benchmark.main()
            except SystemExit as error:
                self.assertNotEqual(error.code, 0)
            if mode == "inventory" or not available:
                spawned.assert_not_called()
            result = json.loads((out / "result.json").read_text())
            self.assertIsInstance(result, dict)
            self.assertTrue((out / "manifest.json").is_file())
            return result

    def test_inventory_needs_no_environment(self) -> None:
        self.assertEqual(
            self._attempt("inventory", available=False)["status"], "completed"
        )

    def test_missing_environment_does_not_launch(self) -> None:
        self.assertEqual(
            self._attempt("inference", available=False)["status"], "blocked"
        )

    def test_success(self) -> None:
        result = self._attempt("inference")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)

    def test_failed_child_retains_exit(self) -> None:
        result = self._attempt("inference", child="raise SystemExit(7)")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)

    def test_timeout_reaps_child(self) -> None:
        result = self._attempt("inference", child="import time; time.sleep(30)")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["exit_code"], -9)

    def test_launch_failure_retains_result(self) -> None:
        result = self._attempt("inference", launch_error=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "fixture launch failure")
