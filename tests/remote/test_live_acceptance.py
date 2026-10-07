"""Inventory acceptance uses observed counts and never deletes unrelated rentals."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.remote import live_acceptance as live
from tests.remote.test_compile import ROOT, SOURCE
from tests.remote.test_lifecycle import Clock, Provider


@pytest.mark.parametrize(
    ("initial", "final", "accepted"),
    [
        ((), (), True),
        ((), ("unrelated",), False),
        (("unrelated",), ("unrelated",), True),
        (("unrelated",), ("manabot-owned",), False),
    ],
)
def test_live_inventory_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial: tuple[str, ...],
    final: tuple[str, ...],
    accepted: bool,
) -> None:
    provider = Provider(Clock())
    for name in initial:
        provider.create({"name": name})
    monkeypatch.setattr(live, "RunPod", lambda: provider)
    monkeypatch.setattr(live, "current_source", lambda root: SOURCE)
    monkeypatch.chdir(tmp_path)
    recipe = json.loads((ROOT / "ops/examples/step-target.json").read_text())
    for stage in recipe["stages"]:
        stage["learning"]["ema"] = 0.9
    recipe_path = tmp_path / "regime.json"
    recipe_path.write_text(json.dumps(recipe))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acceptance",
            "--regime",
            str(recipe_path),
            "--spec",
            str(ROOT / "ops/jobs/runpod-small.json"),
        ],
    )

    def deploy(plan: object, destination: Path, root: Path) -> SimpleNamespace:
        (destination / "evidence/run").mkdir(parents=True)
        (destination / "bundle.json").write_text("{}")
        (destination / "evidence/run/run.json").write_text("{}")
        provider.pods.clear()
        for name in final:
            provider.create({"name": name})
        return SimpleNamespace(estimated_dollars=0.1)

    def resolve(evidence: Path, path: str) -> Path:
        return evidence / path

    monkeypatch.setattr(live, "deploy", deploy)
    monkeypatch.setattr(
        live,
        "Bundle",
        SimpleNamespace(
            model_validate_json=lambda text: SimpleNamespace(resolve=resolve)
        ),
    )
    monkeypatch.setattr(live, "verify_training_bundle", lambda *args: None)
    monkeypatch.setattr(
        live,
        "TrainingRun",
        SimpleNamespace(
            model_validate_json=lambda text: SimpleNamespace(
                stages=[
                    SimpleNamespace(
                        actual_device="cuda",
                        optimizer_exposures=1,
                        id="train",
                        cumulative_seconds=1.0,
                        environment_decisions=1,
                        learner_transitions=1,
                        games=1,
                    )
                ],
                selected_artifact={"path": "raw.pt"},
                updates_through=lambda stage: 1,
                seconds=1.0,
            )
        ),
    )
    monkeypatch.setattr(
        live,
        "evaluate_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(status="completed"),
    )
    if accepted:
        live.main()
    else:
        with pytest.raises(ValueError, match="live acceptance incomplete"):
            live.main()
    summary = json.loads(
        (tmp_path / ".runs/remote-acceptance/attempt-000/acceptance.json").read_text()
    )
    assert summary["initial_inventory_pods"] == len(initial)
    assert summary["inventory_pods"] == len(final)
    assert [p.name for p in provider.pods] == list(final)
