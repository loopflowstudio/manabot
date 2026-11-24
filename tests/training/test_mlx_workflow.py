"""Verify trained MLX exports through local CPU game children and the baseline report."""

from collections import Counter
from contextlib import suppress
import hashlib
import importlib.util
from itertools import count
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from manabot.backends.mlx.policy import MLXActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.policy.checkpoint import PolicyCheckpoint
    from manabot.verify.choice_corpus import Case


def check_policy_loads() -> None:
    """Observe each actual load, including loads inside the standalone children."""
    from manabot.backends.mlx.checkpoint import MLXBackend

    load = MLXBackend.load_policy

    def checked_load(
        backend: "MLXBackend",
        path: "str | Path | PolicyCheckpoint",
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> "tuple[MLXActorCritic, ObservationSpace]":
        policy, space = load(
            backend, path, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
        )
        assert (policy.device_type, policy.precision) == ("cpu", "float32")
        return policy, space

    MLXBackend.load_policy = checked_load


def test_linked_belief_jobs_register_portable_checkpoints_without_torch(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("requires MLX")
    (tmp_path / "sitecustomize.py").write_text(
        'import sys\nsys.modules["torch"] = None\n'
    )
    track = tmp_path / "track"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path.cwd()))),
        "ETUDE_TRACK_DIR": str(track),
        "ETUDE_HOME": str(tmp_path),
    }
    for job, config in (
        ("policy", {"backend": "mlx", "device": "mps"}),
        ("sampler", {"backend": "mlx"}),
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "etude",
                "train",
                "experiments/belief.py",
                "--experiment",
                "belief-mlx",
                "--job",
                job,
                "--config",
                json.dumps(config),
            ],
            env=env,
            check=True,
            timeout=120,
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.training.test_mlx_workflow import verify_belief_jobs; verify_belief_jobs()",
        ],
        env=env,
        check=True,
        timeout=30,
    )


def verify_belief_jobs() -> None:
    from etude.experiment import observe

    from experiments.belief import training_jobs
    from manabot.backends import resolve
    from manabot.belief.checkpoint import read_belief_sampler
    from manabot.belief.sampling_data import read_dataset

    assert "torch" in sys.modules and sys.modules["torch"] is None
    track = Path(os.environ["ETUDE_TRACK_DIR"])
    experiment = track / "belief-mlx"
    policy_run = experiment / "policy@0"
    policy = json.loads((policy_run / "checkpoints/updates-1.json").read_bytes())
    assert policy["at"]["updates"] == 1 and policy["at"]["learner_transitions"] > 0
    behavior = policy_run / policy["file"]
    digest = hashlib.sha256(behavior.read_bytes()).hexdigest()
    assert digest == policy["artifact"]["sha256"]
    backend = resolve("mlx")
    backend.load_policy(behavior)
    (start,) = training_jobs(observe(track, "belief-mlx"))[1].plain().inputs
    assert (start.name, start.run, start.sha256) == (
        "start",
        "belief-mlx/policy@0",
        digest,
    )
    sampler_run = experiment / "sampler@0"
    record = json.loads((sampler_run / "checkpoints/steps-4.json").read_bytes())
    assert record["at"]["steps"] == 4
    path = sampler_run / record["file"]
    dataset = read_dataset(sampler_run / "out/histories.json")
    assert path.suffix == ".safetensors" and dataset.policy_identity == digest
    admitted = read_belief_sampler(
        path,
        codec=backend.belief,
        expected_dataset_identity=dataset.identity,
        expected_policy_identity=digest,
        expected_schema_identity=dataset.schema.identity,
        expected_world_identity=dataset.world_identity,
        expected_checkpoint_identity=record["artifact"]["sha256"],
    )
    assert admitted.model.schema == dataset.schema
    assert admitted.training_seed == 0


@pytest.mark.parametrize("width", [64, 256])
def test_mlx_local_evaluation_and_verification(tmp_path: Path, width: int) -> None:
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("requires MLX")
    # The import block is inherited by admission, play and replay children.
    (tmp_path / "sitecustomize.py").write_text(
        'import sys\nsys.modules["torch"] = None\n'
        'if {"manabot.eval.local:_prepare", "manabot.sim.native_game:game_worker"}.intersection(sys.orig_argv):\n'
        "    from tests.training.test_mlx_workflow import check_policy_loads\n"
        "    check_policy_loads()\n"
    )
    # The report starts the CLI and game children. Terminate the entire test
    # process group if the outer deadline expires while any are still running.
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.training.test_mlx_workflow",
            str(tmp_path),
            str(width),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path.cwd()))),
        },
        start_new_session=True,
    ) as process:
        try:
            returncode = process.wait(timeout=240)
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            raise

    # The independent reference admits the very same bytes and registrations.
    if importlib.util.find_spec("torch") is not None:
        from manabot.backends import resolve
        from manabot.eval.evaluation import EvaluationRequest
        from manabot.eval.local import checkpoint_request

        request = EvaluationRequest.model_validate_json(
            (tmp_path / "cli/request.json").read_bytes()
        )
        reference = resolve("torch")
        with reference.runtime.scoped(1, "float32", "cpu"):
            assert request == checkpoint_request(
                Path(request.checkpoint_paths[request.candidate.player_id]),
                opponent="random",
                deal_seeds=request.deal_seeds,
                backend=reference,
            )


def collect_choice_prefix(
    out: Path, round_index: int, deadline: float, closest: bool = False
) -> "tuple[list[Case], Counter[str], list[dict[str, object]]]":
    """Use the first 32 native decisions per round, independent of CPU speed."""
    from manabot.verify import choice_corpus

    with patch.object(choice_corpus, "time") as clock:
        clock.perf_counter.side_effect = count()
        return choice_corpus.collect_choices(out, round_index, 32, closest=closest)


def verify_run(root: Path, width: int) -> None:
    from manabot.eval.evaluation import evaluation_status
    from manabot.recipes import baseline
    from manabot.verify import learning_regression

    config = baseline.BaselineConfig(backend="mlx", device="cpu", updates=1)
    recipe = baseline.settings(config)
    bounded = recipe.model_copy(
        update={
            "model": recipe.model.model_copy(update={"hidden_dim": width}),
            "streams": 2,
            "transitions": 8,
            "learning_seats": "one",
            "batch_closure": "exact-quota",
            "bucketed_shapes": True,
        }
    )
    out = root / "track/baseline/baseline-seed-197/out"
    os.environ.update(
        ETUDE_OUT=str(out),
        ETUDE_EXPERIMENT="baseline",
        ETUDE_RUN="baseline-seed-197",
        ETUDE_SEED="197",
        ETUDE_TRACK_DIR=str(root / "track"),
    )
    # This deal's four games against random are short (448 decisions for the
    # width-256 initial policy), so each 90-second panel fits its game budget.
    deal = baseline.BASELINE_DEALS[4]
    with (
        patch.object(baseline, "settings", return_value=bounded),
        patch.object(baseline, "BASELINE_DEALS", (deal,)),
        patch.object(learning_regression, "collect_choices", collect_choice_prefix),
    ):
        baseline.train(config)
        # The report can reuse a complete local panel for its initial policy.
        cli = root / "cli"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "manabot.eval.local",
                "--backend",
                "mlx",
                "--checkpoint",
                str(out / "policies/00000000-raw.safetensors"),
                "--opponent",
                "random",
                "--deal-seed",
                str(deal),
                "--out",
                str(cli),
                "--seconds",
                "90",
            ],
            check=True,
            timeout=100,
        )
        target = out / "evaluation"
        target.mkdir()
        (cli / "panel").rename(target / "00000000")
        check_policy_loads()
        report = baseline.evaluate(config, out, seed=197, seconds=90)
    assert report.software == "passed", report.error
    assert [point.updates for point in report.policies] == [0, 1]
    assert all(point.choices for point in report.policies)
    assert [row.deal for row in report.paired] == [deal]
    for panel in (target / "00000000", target / "00000001"):
        result = evaluation_status(panel)
        assert result.complete_deals() == (deal,)
        assert len(result.rows) == 4 and all(row.valid for row in result.rows)


if __name__ == "__main__":
    verify_run(Path(sys.argv[1]), int(sys.argv[2]))
