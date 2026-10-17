"""Baseline admission changes the world explicitly, not frozen evidence."""

from pathlib import Path

import pytest

from experiments.runners.corrected_baseline import MATCHUP, recipe, resolve_counts


def test_baseline_recipe_and_budget() -> None:
    plan = resolve_counts(
        slowest_update=3.7,
        panel_costs=(2, 2),
        remaining=53 * 3600,
        hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
    )
    assert plan.updates == 10000
    recipes = plan.recipes()
    assert tuple(recipes) == ("l3",)
    r = recipes["l3"]
    assert r.world == "w6"
    assert r.agent.semantic_pack == "ur-lessons-vs-gw-allies"
    assert r.agent.value_aggregation == "masked_mean"
    assert r.stages[0].matchup_curriculum == "cross-balanced"
    for update in plan.evaluation_updates:
        protocol = plan.protocol("l3", "random", update)
        assert protocol.authored_matchup == MATCHUP
        assert protocol.include_mirrors and len(protocol.deal_seeds) == 25
    assert 3 * 3 * 2 * 25 * 8 == 3600
    assert recipe(100, 1200).stages[0].updates == 100


def test_insufficient_capacity_does_not_shrink_cohort() -> None:
    with pytest.raises(ValueError):
        resolve_counts(
            slowest_update=50,
            panel_costs=(2, 2),
            remaining=36000,
            hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
        )


def test_midpoint_is_export_only() -> None:
    plan = resolve_counts(
        slowest_update=3.7,
        panel_costs=(2, 2),
        remaining=53 * 3600,
        hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
    )
    assert plan.evaluation_updates == (0, 10000)
    with pytest.raises(ValueError):
        plan.protocol("l3", "random", 5000)


def test_future_diagnostic_cadence_is_separate_from_scored_endpoint(
    tmp_path: Path,
) -> None:
    from experiments.runners.baseline_monitoring import DiagnosticMonitor

    monitor = DiagnosticMonitor(tmp_path / "run.json", tmp_path / "diagnostics")
    assert monitor.config.updates == tuple(range(1000, 10000, 1000))
    assert monitor.config.seconds == 9900
    protocols = (monitor.config.protocol, *monitor.config.additional_protocols)
    assert {p.opponent for p in protocols} == {"random", "scripted_greedy"}
    assert all(p.purpose == "supplemental-matchup-diagnostic" for p in protocols)
    assert all(len(p.deal_seeds) == 13 and p.include_mirrors for p in protocols)
    assert not monitor.thread.is_alive()


def test_retained_first_writes_into_fresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import experiments.runners.corrected_baseline as baseline

    plan = resolve_counts(
        slowest_update=3.7,
        panel_costs=(2, 2),
        remaining=53 * 3600,
        hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
    )
    producer = SimpleNamespace(
        id="producer",
        seed=12261,
        status="completed",
        regime=recipe(10000, plan.training_seconds_per_run),
        updates_through=lambda: 10000,
        model_dump=lambda mode: {"id": "producer"},
    )
    source = tmp_path / "producer/outputs/l3-12261/training/run.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    (tmp_path / "producer/outputs/initialization-parameters.json").write_text(
        '{"12261": "digest"}'
    )
    evaluated: list[Path] = []
    monkeypatch.setattr(
        baseline.TrainingRun, "model_validate_json", lambda text: producer
    )
    monkeypatch.setattr(
        baseline,
        "init",
        lambda **_: SimpleNamespace(log=lambda **_: None, finish=lambda _: None),
    )
    monkeypatch.setattr(
        baseline,
        "evaluate_fit",
        lambda plan, run, target, *_: evaluated.append(target),
    )
    monkeypatch.setenv("ETUDE_RUN", "check")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    monkeypatch.setenv("ETUDE_DEADLINE", "2099-01-01T00:00:00+00:00")

    output = tmp_path / "outputs"
    baseline.complete_retained_first(plan, source, output)

    assert (output / "l3-12261/training/run.json").exists()
    assert evaluated == [output / "l3-12261"]


def test_first_evaluated_requires_all_six_complete_cohorts(tmp_path: Path) -> None:
    import json

    from experiments.runners.corrected_baseline import _require_first_evaluated

    names = (
        "0-scripted_greedy-candidate",
        "0-random-candidate",
        "10000-scripted_greedy-candidate",
        "10000-scripted_greedy-control",
        "10000-random-candidate",
        "10000-random-control",
    )
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "monitor.json").write_text(
            json.dumps({"status": "completed", "rows": [{}] * 200})
        )
    _require_first_evaluated(tmp_path)

    (tmp_path / names[-1] / "monitor.json").write_text(
        json.dumps({"status": "incomplete", "rows": [{}] * 120})
    )
    with pytest.raises(ValueError):
        _require_first_evaluated(tmp_path)
