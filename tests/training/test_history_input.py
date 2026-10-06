"""History screen software fixtures: no optimization or scientific scoring."""

import json
from pathlib import Path
import signal
from typing import Any

import numpy as np
import pytest
import torch

from experiments.runners import (
    history_input as history,
    run_history_input as supervisor,
    run_training_regimes as runner,
)
from experiments.runners.history_input_analysis import history_report, paired_effect
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import PlayerRegistration, canonical_sha256, file_sha256
from manabot.env import Match, ObservationSpace
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_receipt
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import atomic_json
from manabot.training.models import StageRecord, TrainingRun


def calibration(rate: float = 2) -> history.Calibration:
    return history.Calibration(
        arms=tuple(
            history.CalibrationArm(
                recipe_id=name,
                run_path=f"/{name}/run.json",
                run_sha256="a" * 64,
                process_seconds=40 * rate,
                stage_seconds=(19 * rate, 19 * rate),
            )
            for name in history.ARMS
        ),
        seconds=80 * rate + 10,
        source_commit="a" * 40,
        runtime_identities={key: "a" * 64 for key in history.COMMON_KEYS},
        input_bindings=tuple(
            history.InputBinding(
                recipe_id=name,
                observation_abi_sha256=str(index + 1) * 64,
                input_schema_sha256=str(index + 3) * 64,
                world_binding_sha256=str(index + 5) * 64,
                policy_history_version=index,
            )
            for index, name in enumerate(history.ARMS)
        ),
        projected_disk_bytes=3 * 1024**3,
    )


def test_schedule_and_six_hour_admission() -> None:
    receipt = calibration()
    plan = supervisor.history_plan(receipt)
    assert ResolvedStudy.model_validate_json(plan.model_dump_json()) == plan
    assert receipt.admitted_updates() == 800
    assert calibration(4).admitted_updates() == 400
    with pytest.raises(ValueError, match="minimum 400"):
        calibration(5).admitted_updates()
    assert (
        sum(
            (
                history.CALIBRATION_SECONDS,
                history.TRAINING_SECONDS,
                history.EVALUATION_SECONDS,
                history.REPORT_SECONDS,
            )
        )
        == 21600
    )
    assert sum(r["wall_seconds"] for r in plan.recipes) * 3 == 14400
    schedule = [
        (history.ARMS[i], seed)
        for order, seed in zip(history.ORDER, history.SEEDS, strict=True)
        for i in order
    ]
    assert schedule == [
        ("history-off", 10631),
        ("history-on", 10631),
        ("history-on", 10632),
        ("history-off", 10632),
        ("history-off", 10633),
        ("history-on", 10633),
    ]
    assert len(schedule) * plan.protocol.checkpoint_count == 12
    assert len(schedule) * 2 * len(plan.protocol.anchor_deals) * 4 == 1200


@pytest.mark.parametrize(
    "change",
    [
        "learning",
        "capacity",
        "floor",
        "updates",
        "watchdog",
        "history",
        "binding",
        "source",
        "allocation",
        "seeds",
        "deals",
    ],
)
def test_drift_is_rejected_even_with_rehashed_recipes(change: str) -> None:
    data = supervisor.history_plan(calibration()).model_dump(mode="json")
    if change == "learning":
        data["recipes"][1]["stages"][0]["learning"]["policy_lambda"] = 0.9
    elif change == "floor":
        data["recipes"][1]["stages"][0]["learning"]["min_advantage"] = 0
    elif change == "capacity":
        data["recipes"][1]["agent"]["hidden_dim"] = 128
    elif change == "updates":
        data["recipes"][1]["stages"][0]["updates"] += 1
    elif change == "watchdog":
        data["recipes"][1]["wall_seconds"] += 1
    elif change == "history":
        data["recipes"][1]["agent"]["recent_events"] = False
    elif change == "binding":
        data["input_bindings"][1]["input_schema_sha256"] = "f" * 64
    elif change == "source":
        data["runtime_identities"]["engine_extension_sha256"] = "f" * 64
    elif change == "allocation":
        data["allocation_seconds"] = 28800
    elif change == "seeds":
        data["protocol"]["training_seeds"] = [10631, 10631, 10633]
    else:
        data["protocol"]["anchor_deals"][0] += 100
    data["protocol"]["regime_digests"] = [canonical_sha256(r) for r in data["recipes"]]
    with pytest.raises(ValueError):
        ResolvedStudy.model_validate(data)


def test_calibration_is_complete_cost_only_and_no_replacement() -> None:
    data = calibration().model_dump()
    data["arms"] = (data["arms"][0], data["arms"][0])
    with pytest.raises(ValueError, match="both arms"):
        history.Calibration.model_validate(data)
    data = calibration().model_dump()
    data["seconds"] = 1
    with pytest.raises(ValueError, match="omits process"):
        history.Calibration.model_validate(data)
    data["seconds"] = 1801
    with pytest.raises(ValueError):
        history.Calibration.model_validate(data)
    data = calibration().model_dump()
    data["strength_inspected"] = True
    with pytest.raises(ValueError):
        history.Calibration.model_validate(data)
    data = calibration().model_dump()
    data["arms"][1]["stage_seconds"] = (191, 1)
    with pytest.raises(ValueError, match="190-second"):
        history.Calibration.model_validate(data)


def test_second_arm_drift_checked_at_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = supervisor.history_plan(calibration())
    changed = plan.input_bindings[1].model_copy(
        update={"observation_abi_sha256": "f" * 64}
    )
    monkeypatch.setattr(
        supervisor,
        "runtime_bindings",
        lambda values: (plan.runtime_identities, (plan.input_bindings[0], changed)),
    )
    with pytest.raises(ValueError, match="per-arm input drift"):
        supervisor.verify_runtime(plan)


@pytest.mark.parametrize("disk", [False, True])
def test_supervisor_kills_group_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disk: bool
) -> None:
    killed: list[tuple[int, int]] = []

    class Process:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            return 0

    class Disk:
        free = 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(
        supervisor.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )
    monkeypatch.setattr(supervisor.shutil, "disk_usage", lambda path: Disk())
    with pytest.raises(
        (TimeoutError, RuntimeError), match="reserve" if disk else "deadline"
    ):
        supervisor._child(["--fixture"], tmp_path, float("inf") if disk else -1)
    assert killed == [(12345, signal.SIGKILL)]
    receipt = json.loads(next(tmp_path.glob("child-*.json")).read_text())
    assert receipt["status"] == "failed" and receipt["seconds"] >= 0


def test_remaining_projection_preserves_other_phase_reserves() -> None:
    supervisor.remaining_feasible(
        deadline=14400, now=10000, rate=2, remaining_updates=1000
    )
    with pytest.raises(RuntimeError, match="envelope"):
        supervisor.remaining_feasible(
            deadline=14400, now=12000, rate=2, remaining_updates=1000
        )


def test_paired_seed_and_deal_effect_is_not_game_bootstrap() -> None:
    values = np.zeros((2, 3, 25), dtype=np.float64)
    values[0] = 0.25
    values[1] = 0.5
    effect = paired_effect(values)
    assert effect.mean_effect == 0.25
    assert effect.seed_effects == [0.25] * 3
    assert effect.paired_seed_common_deal_95_percentile == [0.25, 0.25]
    values[1, 0] = 0
    assert paired_effect(values) == paired_effect(values)
    assert paired_effect(values).paired_seed_common_deal_95_percentile[0] < 0


def test_shared_initialization_and_parameter_receipt() -> None:
    torch.set_num_threads(1)
    for seed in history.SEEDS:
        models = []
        for recipe in history.recipes():
            torch.manual_seed(seed)
            models.append(Agent(ObservationSpace(recipe.observation), recipe.agent))
        off, on = models
        for key, tensor in off.state_dict().items():
            assert torch.equal(tensor, on.state_dict()[key])
        assert (
            architecture_receipt(on).parameters.total
            - architecture_receipt(off).parameters.total
            == 22912
        )


def test_old_protocol_serialization_has_no_history_binding() -> None:
    original = ResolvedStudy.model_validate_json(
        Path("experiments/plans/value-token-screen.json").read_text()
    )
    assert "input_bindings" not in original.model_dump(mode="json")
    with pytest.raises(ValueError):
        EvaluationProtocol.model_validate(
            {**original.protocol.model_dump(), "process_seconds": 21600}
        )


@pytest.mark.parametrize("invalid_cell", [False, True])
def test_existing_arena_receives_six_runs_two_abis_and_fixed_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_cell: bool
) -> None:

    plan = supervisor.history_plan(calibration())
    values = history.recipes()
    paths: list[Path] = []
    for order, seed in zip(history.ORDER, history.SEEDS, strict=True):
        for index in order:
            recipe = values[index]
            stages = []
            for cutoff, spec in enumerate(recipe.stages):
                path = tmp_path / f"{recipe.id}-{seed}-{cutoff}.pt"
                path.write_text(f"{recipe.id}-{seed}-{cutoff}")
                artifact = dict(
                    path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path)
                )
                stages.append(
                    StageRecord(
                        id=spec.id,
                        status="completed",
                        diagnostics=[{}] * 400,
                        learner_transitions=102400,
                        cumulative_seconds=(cutoff + 1) * 100 + index * 10,
                        optimizer_exposures=400,
                        artifacts={"raw": artifact, "ema": artifact},
                    )
                )
            run = TrainingRun(
                id=f"{recipe.id}-{seed}",
                regime=recipe,
                regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
                seed=seed,
                seed_streams={},
                identities={
                    **plan.runtime_identities,
                    "observation_abi_sha256": plan.input_bindings[
                        index
                    ].observation_abi_sha256,
                },
                status="completed",
                seconds=210,
                stages=stages,
            )
            path = tmp_path / f"{run.id}.json"
            path.write_text(run.model_dump_json())
            paths.append(path)
    monkeypatch.setattr(supervisor, "verify_runtime", lambda plan: None)

    def no_training(*args: object, **kwargs: object) -> None:
        pytest.fail("evaluation path started training")

    monkeypatch.setattr(runner, "execute_regime", no_training)
    keys: list[str] = []
    inputs: set[str] = set()

    def register(
        run: TrainingRun, stage: StageRecord, variant: str
    ) -> PlayerRegistration:
        return PlayerRegistration(
            player_id=f"{run.id}-{stage.id}",
            display_name=run.regime.id,
            role="challenger",
            runner_kind="code",
            player_spec={"kind": "random"},
            source_sha256="a" * 64,
            compute_class_id="fixture",
            information_boundary="acting-viewer",
            world=run.regime.world,
            content_suite=runner.SELECTED_SUITE,
            observation_abi_sha256=run.identities["observation_abi_sha256"],
            action_abi_sha256=run.identities["action_abi_sha256"],
            matchup_sha256=run.identities["matchup_sha256"],
            player_seed_derivation_id="arena-pair-deal-player-v1",
        )

    # Existing play_cell keyword API is heterogeneous; narrow its values here.
    def play(**kwargs: Any) -> tuple[list[dict[str, object]], None, dict[str, bool]]:
        candidate: PlayerRegistration = kwargs["player_a"]
        anchor: PlayerRegistration = kwargs["player_b"]
        assert anchor.player_id == "scripted-greedy-fixed-anchor"
        assert (
            anchor.observation_abi_sha256
            == plan.input_bindings[0].observation_abi_sha256
        )
        assert kwargs["deal_seeds"] == history.DEALS
        assert kwargs["comparison_seed_aliases"] == {
            candidate.player_id: "candidate",
            anchor.player_id: "reference",
        }
        keys.append(canonical_sha256(kwargs["key"].model_dump()))
        inputs.add(candidate.observation_abi_sha256)
        on = candidate.display_name == "history-on"
        rows = [
            dict(
                deal_seed=d,
                leg=leg,
                player_a=candidate.player_id,
                player_a_seat=leg % 2,
                score_a=0.75 if on else 0.25,
                failure=None,
                terminated=True,
                truncated=False,
                replay_passed=True,
                latency={candidate.player_id: {"count": 10, "seconds": 0.1}},
            )
            for d in history.DEALS
            for leg in range(4)
        ]
        return rows, None, {"passed": not invalid_cell}

    monkeypatch.setattr(runner, "registration", register)
    monkeypatch.setattr(runner, "play_cell", play)
    out = tmp_path / "study"
    if invalid_cell:
        with pytest.raises(RuntimeError, match="invalid arena cell"):
            runner.run_study(
                "history-input",
                out,
                plan,
                render_report=False,
                history_run_paths=tuple(paths),
            )
        assert len(keys) == 1
        history_report(out)
        assert (
            json.loads((out / "history-contrast.json").read_text())["status"]
            == "unavailable"
        )
        return
    runner.run_study(
        "history-input", out, plan, render_report=False, history_run_paths=tuple(paths)
    )
    assert len(keys) == 12 and len(set(keys)) == 1 and len(inputs) == 2
    history_report(out)
    runner.report(out)
    assert (out / "history-optimizer_exposures.png").is_file()
    result = json.loads((out / "history-contrast.json").read_text())
    assert result["checkpoints"][1]["mean_effect"] == 0.5
    cost = result["comparisons"]["training_seconds"]
    assert (cost["start_seconds"], cost["end_seconds"]) == (110, 200)
    assert {
        r["checkpoint_seconds"] for r in cost["rows"] if r["regime"] == "history-on"
    } == {110}
    assert result["disposition"] == "larger-confirmatory-allocation-merited"
    before = (out / "history-contrast.json").read_bytes()
    history_report(out)
    assert before == (out / "history-contrast.json").read_bytes()
    with pytest.raises(ValueError, match="retries require"):
        runner.run_study(
            "history-input",
            out,
            plan,
            resume=True,
            render_report=False,
            history_run_paths=tuple(paths),
        )
    data = json.loads((out / "study.json").read_text())
    data["comparisons"][0]["rows"][0]["leg"] = 1
    (out / "study.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="deal/leg"):
        history_report(out)


@pytest.mark.parametrize("fail_calibration", [False, True])
@pytest.mark.parametrize("prior_seconds", [0, 91.15643158298917])
def test_campaign_phase_accounting_and_failure_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_calibration: bool,
    prior_seconds: float,
) -> None:

    clock = [1000.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "_clean_source", lambda source: None)
    out = tmp_path / ".runs" / "history-fixture"
    previous = tmp_path / ".runs" / "previous"
    previous.mkdir(parents=True)
    trace = previous / "trace.gz"
    trace.write_bytes(b"fixture")
    atomic_json(
        previous / "study.json",
        {"comparisons": [{"trace": {"path": str(trace), "games": 100}}]},
    )
    predecessor = supervisor.Predecessor(
        path=previous,
        sha256="b" * 64,
        seconds=prior_seconds or 1,
        source_commit="c" * 40,
    )
    monkeypatch.setattr(
        supervisor, "admit_predecessor", lambda path, sha256: predecessor
    )
    selection = (
        {"predecessor_path": previous, "predecessor_sha256": "b" * 64}
        if prior_seconds
        else {}
    )
    calls: list[tuple[str, float]] = []
    receipt = calibration()

    def child(
        arguments: list[str], root: Path, deadline: float, check: object = None
    ) -> float:
        phase = arguments[0]
        assert deadline <= 1000 + history.TOTAL_SECONDS - prior_seconds
        calls.append((phase, deadline - clock[0]))
        if phase == "--preflight":
            atomic_json(
                root / "runtime-bindings.json",
                {
                    "common": receipt.runtime_identities,
                    "bindings": [
                        b.model_dump(mode="json") for b in receipt.input_bindings
                    ],
                },
            )
            clock[0] += 10
            return 10
        if phase == "--calibrate-arm":
            index = int(arguments[1])
            clock[0] += 80
            if fail_calibration and index == 1:
                raise RuntimeError("fixture calibration failed")
            recipe = history.recipes(40, calibration=True)[index]
            folder = root / "calibration" / recipe.id
            folder.mkdir(parents=True)
            artifact_path = folder / "fixture.pt"
            artifact_path.write_bytes(b"fixture")
            run = TrainingRun(
                id=f"calibration-{index}",
                regime=recipe,
                regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
                seed=history.CALIBRATION_SEED,
                seed_streams={},
                identities={
                    **receipt.runtime_identities,
                    "observation_abi_sha256": receipt.input_bindings[
                        index
                    ].observation_abi_sha256,
                },
                status="completed",
                seconds=70,
                stages=[
                    StageRecord(
                        id=s.id,
                        status="completed",
                        seconds=30,
                        learner_transitions=5120,
                        diagnostics=[{}] * 20,
                        artifacts={"raw": {}, "ema": {}},
                    )
                    for s in recipe.stages
                ],
            )
            atomic_json(folder / "run.json", run.model_dump(mode="json"))
            return 80
        clock[0] += 1000 if phase != "--report" else 10
        return 1000 if phase != "--report" else 10

    monkeypatch.setattr(supervisor, "_child", child)
    if fail_calibration:
        with pytest.raises(RuntimeError, match="fixture calibration"):
            supervisor.campaign(out, "a" * 40, **selection)
        assert [c[0] for c in calls] == [
            "--preflight",
            "--calibrate-arm",
            "--calibrate-arm",
        ]
        assert not (out / "resolved-plan.json").exists()
        status = json.loads((out / "supervisor.json").read_text())
        assert status["status"] == "failed" and status["seconds"] == 170
        assert status["cumulative_seconds"] == pytest.approx(170 + prior_seconds)
        assert calls[0][1] == pytest.approx(1800 - prior_seconds)
        assert (out / "calibration/history-off/run.json").exists()
    else:
        supervisor.campaign(out, "a" * 40, **selection)
        assert [c[0] for c in calls] == [
            "--preflight",
            "--calibrate-arm",
            "--calibrate-arm",
            *(["--train-arm"] * 6),
            "--evaluate",
            "--report",
        ]
        caps = {
            "--preflight": 1800,
            "--calibrate-arm": 400,
            "--train-arm": 2400,
            "--evaluate": 4500,
            "--report": 900,
        }
        assert all(0 < seconds <= caps[phase] for phase, seconds in calls)
        status = json.loads((out / "supervisor.json").read_text())
        assert status["status"] == "completed" and status["seconds"] == 7180
        plan = ResolvedStudy.model_validate_json(
            (out / "resolved-plan.json").read_text()
        )
        assert plan.prior_campaign_seconds == prior_seconds
        assert plan.allocation_seconds == 21600 - prior_seconds
        assert plan.protocol.process_seconds == 21600 - prior_seconds
        assert status["cumulative_seconds"] == pytest.approx(7180 + prior_seconds)
        assert calls[0][1] == pytest.approx(1800 - prior_seconds)
        assert history.Calibration.model_validate_json(
            plan.calibration_evidence
        ).seconds == pytest.approx(170 + prior_seconds)


@pytest.mark.parametrize("index", [0, 1])
def test_history_checkpoint_reload_rejects_swapped_input_metadata(
    tmp_path: Path, index: int
) -> None:
    recipe = history.recipes()[index]
    torch.manual_seed(811)
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent)
    path = tmp_path / "policy.pt"
    save_bc_checkpoint(
        agent,
        agent.observation_space,
        path,
        player_configs=Match(recipe.match).to_rust(),
    )
    loaded, space = load_checkpoint_agent(str(path))
    assert loaded.hypers == recipe.agent
    assert space.encoder.hypers == recipe.observation
    assert architecture_receipt(loaded) == architecture_receipt(agent)
    # Torch payload is the loader's untyped serialization boundary.
    payload = torch.load(path, weights_only=False)
    payload["hypers"]["observation_hypers"]["policy_history_version"] = 1 - index
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint_agent(str(path))


@pytest.fixture
def exported_history_run(tmp_path: Path) -> TrainingRun:
    """Ordinary export payloads, including Adam state, without optimization."""
    recipe = history.recipes(40, calibration=True)[0]
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent)
    record = StageRecord(id=recipe.stages[0].id, status="completed")
    for name in ("raw", "ema", "optimizer"):
        path = tmp_path / f"{name}.pt"
        if name == "optimizer":
            torch.save(torch.optim.Adam(agent.parameters()).state_dict(), path)
        else:
            save_bc_checkpoint(
                agent,
                agent.observation_space,
                path,
                player_configs=Match(recipe.match).to_rust(),
            )
        record.artifacts[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return TrainingRun(
        id="history-export-fixture",
        regime=recipe,
        regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
        seed=history.CALIBRATION_SEED,
        seed_streams={},
        identities={},
        stages=[record],
        status="completed",
    )


def test_history_reload_accepts_mixed_stage_exports(
    exported_history_run: TrainingRun,
) -> None:
    supervisor._reload(exported_history_run)


@pytest.mark.parametrize("name", ["raw", "ema", "optimizer"])
def test_history_reload_checks_every_artifact_digest(
    exported_history_run: TrainingRun, name: str
) -> None:
    artifact = exported_history_run.stages[0].artifacts[name]
    path = Path(artifact["path"])
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="bytes changed"):
        supervisor._reload(exported_history_run)


@pytest.mark.parametrize("name", ["raw", "ema"])
def test_history_reload_checks_each_policy_configuration(
    exported_history_run: TrainingRun, name: str
) -> None:
    # A valid but different history contract must fail run-level admission,
    # even with an internally consistent checkpoint and refreshed digest.
    recipe = history.recipes(40, calibration=True)[1]
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent)
    artifact = exported_history_run.stages[0].artifacts[name]
    path = Path(artifact["path"])
    save_bc_checkpoint(
        agent,
        agent.observation_space,
        path,
        player_configs=Match(recipe.match).to_rust(),
    )
    artifact.update(sha256=file_sha256(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="configuration differs"):
        supervisor._reload(exported_history_run)


def test_history_continuation_cannot_reset_allocation() -> None:
    prior = 91.15643158298917
    receipt = calibration().model_copy(
        update={"seconds": calibration().seconds + prior}
    )
    plan = supervisor.history_plan(receipt, prior)
    data = plan.model_dump(mode="json")
    data["allocation_seconds"] = history.TOTAL_SECONDS
    with pytest.raises(ValueError, match="original six-hour"):
        ResolvedStudy.model_validate(data)
