"""The worked example sets up once and cleans up even if the second run fails."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from manabot.remote.deploy import Attempt, Receipt
from manabot.remote.plan import DeploymentPlan
from manabot.remote.provider import Pod
from ops.examples import two_experiments as example
from tests.remote.test_compile import ROOT, SOURCE


@pytest.mark.parametrize("fail_second", [False, True])
def test_example_setup_once_and_finally_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_second: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(example, "current_source", lambda root: SOURCE)
    monkeypatch.setattr(example, "verify_public_source", lambda source: None)
    recipe = json.loads(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text()
    )
    recipe["wall_seconds"] = 300
    for stage in recipe["stages"]:
        stage["execution"]["wall_seconds"] = 150
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    mix = tmp_path / "mix.json"
    mix.write_bytes((ROOT / "ops/mixes/runpod-small.json").read_bytes())
    monkeypatch.setattr(
        "sys.argv", ["example", "--regime", "recipe.json", "--mix", str(mix)]
    )
    pods: list[Pod] = []

    def delete(pod_id: str) -> None:
        pods.clear()

    provider = SimpleNamespace(
        list=lambda: list(pods),
        get=lambda pod_id: pods[0] if pods else None,
        delete=delete,
    )
    monkeypatch.setattr(example, "RunPod", lambda: provider)

    def create(
        provider: object,
        plan: DeploymentPlan,
        receipt: Receipt,
        path: Path,
        purpose: str,
        deadline: float,
        public_key: str,
    ) -> Pod:
        pod = Pod(
            id="example",
            name="manabot-example",
            rate=0.49,
            vcpus=6,
            memory_gb=48,
            gpu_count=1,
        )
        pods.append(pod)
        receipt.attempts.append(
            Attempt(
                name=pod.name,
                purpose="training",
                intent_time=receipt.started,
                deadline=deadline,
                pod_id=pod.id,
                hourly_rate=pod.rate,
            )
        )
        return pod

    monkeypatch.setattr(example, "_create", create)
    scripts: list[str] = []
    transport = SimpleNamespace(
        shell=scripts.append, put=lambda *args: None, get=lambda *args: None
    )
    monkeypatch.setattr(example, "_ready", lambda *args: transport)
    seeds: list[int] = []

    def experiment(
        transport: object, plan: DeploymentPlan, out: Path, index: int
    ) -> example.ExperimentTiming:
        seeds.append(plan.seed + index)
        if fail_second and index == 1:
            raise RuntimeError("second experiment failed")
        return example.ExperimentTiming(
            plan.seed + index, plan.source.commit, 0, 0, 1, 1, 0.5
        )

    monkeypatch.setattr(example, "_experiment", experiment)
    if fail_second:
        with pytest.raises(RuntimeError, match="second experiment"):
            example.main()
    else:
        example.main()
    assert seeds == [197, 198]
    assert len(scripts) == 1
    assert "uv sync" in scripts[0]
    assert not pods
    out = tmp_path / ".runs/remote-acceptance/attempt-000"
    receipt = Receipt.model_validate_json((out / "deployment.json").read_text())
    assert receipt.phase == "deleted"
    assert receipt.complete is not fail_second
    assert receipt.estimated_dollars is not None
    timings = json.loads((out / "experiments.json").read_text())
    if not fail_second:
        assert timings[1]["setup_seconds"] == 0
