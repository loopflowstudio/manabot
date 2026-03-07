"""Tests for the first-light experiment harness and SQLite store."""

from __future__ import annotations

from manabot.verify.first_light import (
    recommend_next_step,
    resolve_run_config,
    run_first_light,
    write_report,
)
from manabot.verify.store import VerifyStore
from manabot.verify.util import EvaluationActionRecord, EvaluationArtifacts


def _metrics(**overrides):
    base = {
        "num_games": 20.0,
        "wins": 12.0,
        "win_rate": 0.6,
        "win_ci_lower": 0.5,
        "mean_steps": 120.0,
        "attack_rate": 0.3,
        "passed_when_able": 0.2,
        "could_pass": 40.0,
        "pass_count": 8.0,
        "attacked_when_able": 1.0,
        "could_attack": 30.0,
        "landed_when_able": 0.45,
        "could_land": 50.0,
        "land_plays": 22.0,
        "cast_when_able": 0.8,
        "could_spell": 40.0,
        "spell_casts": 32.0,
        "single_valid_decisions": 10.0,
        "multi_valid_decisions": 30.0,
        "pass_land_decisions": 20.0,
        "pass_land_pass_rate": 0.3,
        "pass_land_land_rate": 0.6,
        "pass_land_spell_rate": 0.1,
        "mean_pass_prob": 0.2,
        "mean_land_prob": 0.5,
        "mean_spell_prob": 0.2,
        "mean_attack_prob": 0.1,
        "mean_pass_prob_when_land_available": 0.25,
        "mean_land_prob_when_land_available": 0.55,
        "mean_pass_prob_when_pass_land": 0.3,
        "mean_land_prob_when_pass_land": 0.6,
        "priority_choice_land": 1.0,
        "priority_choice_spell": 2.0,
        "priority_choice_pass": 3.0,
        "priority_choice_land_spell": 4.0,
        "priority_choice_land_pass": 5.0,
        "priority_choice_spell_pass": 6.0,
        "priority_choice_land_spell_pass": 7.0,
        "priority_choice_none": 0.0,
        "action_space_truncations": 0.0,
        "card_space_truncations": 0.0,
        "permanent_space_truncations": 0.0,
    }
    base.update(overrides)
    return base


def test_verify_store_round_trip_and_report(tmp_path):
    db_path = tmp_path / "verify.sqlite"
    store = VerifyStore(db_path)
    try:
        run_id = store.create_run(
            label="demo",
            status="completed",
            mode="dev",
            seed=7,
            git_commit="abc123",
            git_branch="demo-branch",
            tree_dirty=False,
            recipe_name="first_light_shaped_v1",
            opponent_policy="random",
            num_envs=4,
            num_steps=128,
            total_timesteps=1024,
            eval_interval=256,
            eval_num_games=20,
            baseline_auto=True,
            notes=None,
            configs={
                "experiment": {"seed": 7},
                "train": {"num_envs": 4},
                "reward": {"land_play_reward": 0.03},
                "agent": {"attention_on": False},
                "match": {"hero_deck": {"Mountain": 20}},
                "observation": {"max_actions": 20},
            },
        )

        baseline = EvaluationArtifacts(
            metrics=_metrics(win_rate=0.4, win_ci_lower=0.3, could_spell=30.0),
            choice_sets={"all": {"land+pass": 5.0}, "priority": {"land+pass": 5.0}},
            actions=[
                EvaluationActionRecord(
                    game_index=0,
                    step=0,
                    player=0,
                    action_type="land",
                    choice_set="land+pass",
                    is_trivial=False,
                    num_valid_actions=2,
                    attack_available=False,
                    land_available=True,
                    spell_available=False,
                    pass_available=True,
                    pass_prob=0.3,
                    land_prob=0.6,
                    spell_prob=0.0,
                    attack_prob=0.0,
                )
            ],
        )
        final = EvaluationArtifacts(
            metrics=_metrics(),
            choice_sets={
                "all": {"land+pass": 10.0, "land+spell+pass": 2.0},
                "priority": {"land+pass": 8.0},
            },
            actions=[],
        )
        baseline_id = store.record_evaluation(
            run_id=run_id,
            kind="baseline",
            step=0,
            update_index=0,
            opponent_policy="random",
            num_games=20,
            deterministic=False,
            artifacts=baseline,
            retain_actions=True,
        )
        final_id = store.record_evaluation(
            run_id=run_id,
            kind="final",
            step=1024,
            update_index=2,
            opponent_policy="random",
            num_games=20,
            deterministic=False,
            artifacts=final,
            explained_variance=0.7,
            retain_actions=False,
        )

        markdown, written_path, summary = write_report(
            store,
            run_id=run_id,
            output_path=tmp_path / "report.md",
        )

        assert baseline_id != final_id
        assert "## Recommendation" in markdown
        assert summary["recommendation"] == "Move on to auxiliary-head experiments"
        assert written_path == tmp_path / "report.md"
        assert store.get_evaluation_choice_sets(final_id)["all"]["land+pass"] == 10.0
        assert len(store.get_evaluation_actions(baseline_id)) == 1
        assert len(store.get_reports(run_id)) == 1
    finally:
        store.close()


def test_recommend_next_step_blocks_weak_win_signal():
    recommendation, _, heuristic = recommend_next_step(
        _metrics(win_rate=0.55, landed_when_able=0.45, could_spell=40.0),
        _metrics(
            win_rate=0.52,
            win_ci_lower=0.49,
            landed_when_able=0.44,
            pass_land_pass_rate=0.35,
            pass_land_land_rate=0.45,
            mean_pass_prob_when_pass_land=0.4,
            mean_land_prob_when_pass_land=0.45,
            cast_when_able=0.8,
            could_spell=35.0,
        ),
    )

    assert (
        recommendation == "Stay in first-light: random-opponent win signal is too weak"
    )
    assert heuristic["win_signal_ok"] is False


def test_run_first_light_eval_only_records_completed_run(tmp_path):
    config = resolve_run_config(
        db=tmp_path / "verify.sqlite",
        seed=5,
        mode="dev",
        label="eval-only",
        opponent_policy="passive",
        num_envs=1,
        num_steps=8,
        total_timesteps=0,
        eval_interval=None,
        eval_num_games=1,
        baseline=False,
        report=False,
        report_path=None,
    )

    summary = run_first_light(config)

    assert summary["run_id"] > 0
    assert summary["run"]["status"] == "completed"
    assert summary["final"]["num_games"] == 1
    assert summary["baseline"] is None
