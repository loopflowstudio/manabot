"""Kickstarting with a decaying teacher loss: schedules, loss, cross-arch transfer and resume."""

import math
from pathlib import Path
from typing import Literal

from etude.train.read import read
import numpy as np
import pytest
import torch

from manabot.backends.torch import rl
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.recipes.builders import gray_ogre_match
from manabot.training.learning import DampedPolicyLearning, Kickstart, Learning
from manabot.training.loops import self_play as loop
from manabot.training.loops.self_play import (
    CLOCKS,
    Counts,
    SelfPlay,
    learner_digest,
    self_play,
)
from manabot.training.updates.objective import damped_policy_loss, minibatch_objective
from tests.fixtures.training_records import recording

SEED = 44601


def _damped_settings(
    *,
    model: ModelConfig | None = None,
    kickstart: Kickstart | None = None,
    teacher: Path | None = None,
    observation: ObservationConfig | None = None,
    learner_rows: int | None = None,
) -> SelfPlay:
    return SelfPlay(
        match=gray_ogre_match(),
        observation=observation or ObservationConfig(),
        model=model or ModelConfig(hidden_dim=16, num_attention_heads=2),
        learning=DampedPolicyLearning(
            gradient="damped_policy",
            advantage_quantile=0.5,
            min_advantage=0.0,
            kickstart=kickstart,
        ),
        streams=2,
        transitions=12,
        learner_rows=learner_rows,
        checkpoint_every=1,
        checkpoint_weights="raw",
        teacher=teacher,
    )


def _publish_teacher(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "teacher",
    seed: int = 44600,
    model: ModelConfig | None = None,
    observation: ObservationConfig | None = None,
) -> Path:
    monkeypatch.setenv("ETUDE_OUT", str(directory / f"out-{name}"))
    monkeypatch.setenv("ETUDE_SEED", str(seed))
    settings = _damped_settings(model=model, observation=observation)
    with recording(
        experiment="test",
        run=name,
        clocks=CLOCKS,
        directory=directory / f"track-{name}",
    ) as run:
        return self_play(settings, run, updates=1)


# ---------------------------------------------------------------------------
# Configuration, serialization and schedules


def test_disabled_kickstart_omits_serialized_fields_and_preserves_digest() -> None:
    default_learning = DampedPolicyLearning(gradient="damped_policy")
    assert "kickstart" not in default_learning.model_dump()

    default_settings = _damped_settings()
    assert "teacher" not in default_settings.model_dump()
    assert "teacher_sha256" not in Counts().model_dump()
    assert "teacher_sha256" not in Counts(teacher_sha256="a" * 64).coordinates()

    expected_digest = learner_digest(default_settings)
    assert learner_digest(_damped_settings()) == expected_digest


def test_kickstart_config_validation(tmp_path: Path) -> None:
    dummy = tmp_path / "teacher.safetensors"
    dummy.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="exactly one of fade_updates or fade_transitions"):
        Kickstart(policy_weight=0.2)
    with pytest.raises(ValueError, match="exactly one of fade_updates or fade_transitions"):
        Kickstart(policy_weight=0.2, fade_updates=10, fade_transitions=100)
    with pytest.raises(ValueError):
        Kickstart(policy_weight=-0.1, fade_updates=10)
    with pytest.raises(ValueError):
        Kickstart(value_weight=-0.1, fade_updates=10)

    with pytest.raises(ValueError, match="requires DampedPolicyLearning with kickstart"):
        _damped_settings(teacher=dummy)
    with pytest.raises(ValueError, match="requires DampedPolicyLearning with kickstart"):
        SelfPlay(
            match=gray_ogre_match(),
            model=ModelConfig(hidden_dim=16, num_attention_heads=2),
            learning=Learning(horizon=2, epochs=1, minibatches=1),
            streams=2,
            transitions=8,
            teacher=dummy,
        )
    with pytest.raises(ValueError, match="requires a teacher checkpoint"):
        _damped_settings(kickstart=Kickstart(policy_weight=0.2, fade_updates=10))
    # Zero weights are valid without a teacher path for control comparisons.
    zero = _damped_settings(kickstart=Kickstart(policy_weight=0.0, fade_updates=10))
    assert zero.teacher is None


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_kickstart_update_schedule_decays_to_exact_zero(
    schedule: Literal["linear", "cosine"],
) -> None:
    ks = Kickstart(
        policy_weight=0.4, value_weight=0.2, schedule=schedule, fade_updates=4
    )
    assert ks.weights(1) == (0.4, 0.2)
    mid_policy, mid_value = ks.weights(3)
    assert mid_policy == pytest.approx(0.2)
    assert mid_value == pytest.approx(0.1)
    if schedule == "linear":
        assert ks.weights(2)[0] == pytest.approx(0.3)
        assert ks.weights(4)[0] == pytest.approx(0.1)
    else:
        assert ks.weights(2)[0] == pytest.approx(0.4 * 0.5 * (1 + math.cos(math.pi / 4)))
        assert ks.weights(4)[0] == pytest.approx(
            0.4 * 0.5 * (1 + math.cos(3 * math.pi / 4))
        )
    # Exact zero once fade_updates updates have run.
    assert ks.weights(5) == (0.0, 0.0)
    assert ks.weights(500) == (0.0, 0.0)

    one_step = Kickstart(policy_weight=0.5, schedule=schedule, fade_updates=1)
    assert one_step.weights(1) == (0.5, 0.0)
    assert one_step.weights(2) == (0.0, 0.0)


def test_kickstart_transition_schedule_decays_to_exact_zero() -> None:
    ks = Kickstart(
        policy_weight=0.5, value_weight=0.1, schedule="linear", fade_transitions=100
    )
    assert ks.weights(1, learner_transitions=0) == (0.5, 0.1)
    assert ks.weights(2, learner_transitions=50) == (pytest.approx(0.25), pytest.approx(0.05))
    assert ks.weights(3, learner_transitions=100) == (0.0, 0.0)
    assert ks.weights(4, learner_transitions=250) == (0.0, 0.0)
    with pytest.raises(ValueError, match="nonnegative learner_transitions"):
        ks.weights(1, learner_transitions=None)


# ---------------------------------------------------------------------------
# Objective & gradient math


def test_damped_policy_loss_teacher_forward_kl_and_gradients() -> None:
    logits = torch.tensor([[1.5, -0.5, 0.2, 999.0]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    teacher_logits = torch.tensor([[-0.2, 2.0, 0.4, -500.0]], requires_grad=True)
    teacher_logs = rl.policy_logs(teacher_logits, valid)
    magnet = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0]])
    behavior_logs = rl.policy_logs(torch.zeros_like(logits), valid)
    args = (
        logits,
        torch.tensor([0]),
        torch.zeros(1),
        behavior_logs,
        magnet,
        valid,
    )

    loss = damped_policy_loss(
        rl,
        *args,
        clip=0.2,
        collection_kl=0.0,
        tau=0.0,
        teacher_logs=teacher_logs,
        teacher_policy_weight=0.3,
    )
    loss.sum().backward()

    t_probs = teacher_logits.detach()[valid].double().softmax(0)
    t_logs = teacher_logits.detach()[valid].double().log_softmax(0)
    s_logs = logits.detach()[valid].double().log_softmax(0)
    expected_kl = (t_probs * (t_logs - s_logs)).sum().item()
    assert loss.item() == pytest.approx(0.3 * expected_kl, rel=1e-5)

    s_probs = logits.detach().masked_fill(~valid, -torch.inf).softmax(-1)
    t_full = teacher_logits.detach().masked_fill(~valid, -torch.inf).softmax(-1)
    torch.testing.assert_close(logits.grad, 0.3 * (s_probs - t_full), atol=1e-6, rtol=1e-5)
    assert teacher_logits.grad is None

    # Zero teacher weight matches omitted teacher bit for bit.
    base = damped_policy_loss(rl, *args, clip=0.2, collection_kl=0.1, tau=0.05)
    zero = damped_policy_loss(
        rl,
        *args,
        clip=0.2,
        collection_kl=0.1,
        tau=0.05,
        teacher_logs=teacher_logs,
        teacher_policy_weight=0.0,
    )
    assert torch.equal(base, zero)

    with pytest.raises(ValueError, match="requires teacher_logs"):
        damped_policy_loss(
            rl, *args, clip=0.2, collection_kl=0.0, tau=0.0, teacher_policy_weight=0.1
        )
    with pytest.raises(ValueError, match="invalid damping coefficients"):
        damped_policy_loss(
            rl,
            *args,
            clip=0.2,
            collection_kl=0.0,
            tau=0.0,
            teacher_logs=teacher_logs,
            teacher_policy_weight=-0.1,
        )


def test_minibatch_objective_value_distillation_scalar_and_categorical() -> None:
    logits = torch.tensor([[0.5, -0.3]], requires_grad=True)
    valid = torch.tensor([[True, True]])
    behavior_logs = rl.policy_logs(torch.zeros_like(logits), valid)
    magnet = torch.tensor([[0.5, 0.5]])
    actions = torch.tensor([0])
    advantages = torch.tensor([0.2])
    actor_mask = torch.tensor([True])

    # Scalar value distillation
    values = torch.tensor([[0.1]], requires_grad=True)
    teacher_val = torch.tensor([0.7], requires_grad=True)
    _, _, val_loss = minibatch_objective(
        rl,
        logits,
        values,
        valid,
        actions=actions,
        advantages=advantages,
        value_targets=torch.tensor([0.3]),
        behavior_logs=behavior_logs,
        reference=magnet,
        actor_mask=actor_mask,
        categorical=False,
        clip=0.2,
        collection_kl=0.0,
        tau=0.0,
        teacher_value_targets=teacher_val,
        teacher_value_weight=0.5,
    )
    expected_val = (0.1 - 0.3) ** 2 + 0.5 * (0.1 - 0.7) ** 2
    assert val_loss.item() == pytest.approx(expected_val, rel=1e-5)
    val_loss.backward()
    assert teacher_val.grad is None

    # Categorical WDL value distillation
    wdl_logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    wdl_targets = torch.tensor([[0.2, 0.1, 0.7]])
    teacher_wdl = torch.tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    _, _, cat_loss = minibatch_objective(
        rl,
        logits.detach().requires_grad_(True),
        wdl_logits,
        valid,
        actions=actions,
        advantages=advantages,
        value_targets=wdl_targets,
        behavior_logs=behavior_logs,
        reference=magnet,
        actor_mask=actor_mask,
        categorical=True,
        clip=0.2,
        collection_kl=0.0,
        tau=0.0,
        teacher_value_targets=teacher_wdl,
        teacher_value_weight=0.25,
    )
    log_p = wdl_logits.detach().log_softmax(-1)
    expected_cat = -(wdl_targets * log_p).sum() + 0.25 * (-(teacher_wdl.detach() * log_p).sum())
    assert cat_loss.item() == pytest.approx(expected_cat.item(), rel=1e-5)
    cat_loss.backward()
    assert teacher_wdl.grad is None


# ---------------------------------------------------------------------------
# End-to-end self-play loop: zero-weight equivalence, cross-arch, fade, resume


def test_zero_weight_kickstart_matches_baseline_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_path = _publish_teacher(tmp_path, monkeypatch)

    def run_arm(name: str, settings: SelfPlay) -> tuple[Path, dict[str, float]]:
        monkeypatch.setenv("ETUDE_OUT", str(tmp_path / f"out-{name}"))
        monkeypatch.setenv("ETUDE_SEED", str(SEED))
        with recording(
            experiment="test",
            run=name,
            clocks=CLOCKS,
            directory=tmp_path / f"track-{name}",
        ) as run:
            final = self_play(settings, run, updates=2)
        data = read(tmp_path / f"track-{name}", "test", name)
        return final, {
            key: float(data.points(key, "updates").y[-1])
            for key in ("loss", "policy_loss", "value_loss")
        }

    baseline_path, baseline_metrics = run_arm("baseline", _damped_settings())
    zero_path, zero_metrics = run_arm(
        "zero",
        _damped_settings(
            kickstart=Kickstart(policy_weight=0.0, value_weight=0.0, fade_updates=2),
            teacher=teacher_path,
        ),
    )

    assert zero_metrics == baseline_metrics
    base_state = load_checkpoint(baseline_path)[0].state_dict()
    zero_state = load_checkpoint(zero_path)[0].state_dict()
    for name, value in base_state.items():
        assert torch.equal(zero_state[name], value), name


def test_cross_architecture_kickstart_fades_to_zero_and_skips_teacher_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_model = ModelConfig(
        hidden_dim=16,
        attention_layers=1,
        num_attention_heads=2,
        focus_film=False,
        stream_norm=False,
    )
    student_model = ModelConfig(
        hidden_dim=32,
        attention_layers=2,
        num_attention_heads=4,
        focus_film=True,
        stream_norm=True,
    )
    teacher_path = _publish_teacher(tmp_path, monkeypatch, model=teacher_model)
    teacher_before = {
        k: v.clone() for k, v in load_checkpoint(teacher_path)[0].state_dict().items()
    }

    teacher_calls_by_update: dict[int, int] = {}
    current_update = [0]
    original_teacher_policy = loop.teacher_policy

    def instrumented_teacher_policy(settings, space, backend):
        teacher = original_teacher_policy(settings, space, backend)
        original_forward = teacher.forward_distribution

        def counted_forward(obs):
            teacher_calls_by_update[current_update[0]] = (
                teacher_calls_by_update.get(current_update[0], 0) + 1
            )
            return original_forward(obs)

        teacher.forward_distribution = counted_forward  # type: ignore[method-assign]
        return teacher

    original_step = loop.step

    def tracked_step(*args, **kwargs):
        current_update[0] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(loop, "teacher_policy", instrumented_teacher_policy)
    monkeypatch.setattr(loop, "step", tracked_step)

    settings = _damped_settings(
        model=student_model,
        kickstart=Kickstart(
            policy_weight=0.5,
            value_weight=0.25,
            schedule="linear",
            fade_updates=2,
        ),
        teacher=teacher_path,
        learner_rows=4,
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out-student"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test",
        run="student",
        clocks=CLOCKS,
        directory=tmp_path / "track-student",
    ) as run:
        self_play(settings, run, updates=3)

    data = read(tmp_path / "track-student", "test", "student")
    weights = data.points("teacher_policy_weight", "updates")
    assert list(weights.x) == [1, 2, 3]
    assert list(weights.y) == pytest.approx([0.5, 0.25, 0.0])
    value_weights = data.points("teacher_value_weight", "updates")
    assert list(value_weights.y) == pytest.approx([0.25, 0.125, 0.0])

    # Teacher KL and value loss logged only on active updates (1 and 2).
    kl_points = data.points("teacher_policy_kl", "updates")
    assert list(kl_points.x) == [1, 2]
    assert all(v >= 0 for v in kl_points.y)
    val_points = data.points("teacher_value_loss", "updates")
    assert list(val_points.x) == [1, 2]
    assert all(v >= 0 for v in val_points.y)

    # Teacher forward ran during updates 1 and 2, and was completely skipped on update 3.
    assert teacher_calls_by_update.get(1, 0) > 0
    assert teacher_calls_by_update.get(2, 0) > 0
    assert teacher_calls_by_update.get(3, 0) == 0

    # Update-0 training_run table records the teacher checkpoint identity.
    (run_table,) = data.read_table("training_run", "updates")
    assert run_table.rows[0]["teacher_sha256"] == file_sha256(teacher_path)

    # Teacher file on disk is unchanged.
    teacher_after = load_checkpoint(teacher_path)[0].state_dict()
    for key, value in teacher_before.items():
        assert torch.equal(teacher_after[key], value)


def test_kickstarting_reduces_teacher_kl_relative_to_fresh_student(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_path = _publish_teacher(
        tmp_path,
        monkeypatch,
        model=ModelConfig(hidden_dim=16, num_attention_heads=2),
    )
    teacher, _ = load_checkpoint(teacher_path)
    student_model = ModelConfig(hidden_dim=24, num_attention_heads=2)

    def train_student(name: str, policy_weight: float) -> Path:
        monkeypatch.setenv("ETUDE_OUT", str(tmp_path / f"out-{name}"))
        monkeypatch.setenv("ETUDE_SEED", str(SEED))
        settings = _damped_settings(
            model=student_model,
            kickstart=Kickstart(policy_weight=policy_weight, fade_updates=4),
            teacher=teacher_path,
        )
        with recording(
            experiment="test",
            run=name,
            clocks=CLOCKS,
            directory=tmp_path / f"track-{name}",
        ) as run:
            return self_play(settings, run, updates=3)

    fresh_path = train_student("fresh", 0.0)
    ks_path = train_student("kickstarted", 4.0)

    # Measure KL(pi_teacher || pi_student) on one shared batch collected by teacher.
    settings = _damped_settings(model=student_model)
    with loop.self_play_games(
        settings, loop.FreshRun(tmp_path / "probe", SEED), Counts()
    ) as collector:
        batch = loop.collect(collector, teacher, 16)
    indexes = np.arange(batch.actions.size, dtype=np.int64)
    backend = loop.default_backend()
    t_admitted = backend.rollout.AdmittedRollout(teacher, batch)
    t_admitted.admit_rows(indexes, len(indexes))
    t_prepared, _ = t_admitted.rows(indexes, torch.device("cpu"))
    valid = rl.legal_mask(t_admitted.legal(t_prepared))
    with torch.no_grad():
        t_logits, _ = teacher.forward_distribution(t_prepared)
        t_logs = rl.policy_logs(t_logits, valid)

    def mean_kl(path: Path) -> float:
        student, _ = load_checkpoint(path)
        s_admitted = backend.rollout.AdmittedRollout(student, batch)
        s_admitted.admit_rows(indexes, len(indexes))
        s_prepared, _ = s_admitted.rows(indexes, torch.device("cpu"))
        with torch.no_grad():
            s_logits, _ = student.forward_distribution(s_prepared)
            s_logs = rl.policy_logs(s_logits, valid)
            return float(rl.forward_kl(s_logs, t_logs, valid).mean())

    assert mean_kl(ks_path) < mean_kl(fresh_path)


def test_teacher_admission_checks_observation_and_value_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scalar_teacher = _publish_teacher(
        tmp_path,
        monkeypatch,
        name="scalar-teacher",
        model=ModelConfig(hidden_dim=16, num_attention_heads=2, value_kind="scalar"),
    )
    wdl_teacher = _publish_teacher(
        tmp_path,
        monkeypatch,
        name="wdl-teacher",
        model=ModelConfig(
            hidden_dim=16, num_attention_heads=2, value_kind="categorical_wdl"
        ),
    )

    # Mismatched observation config is refused.
    bad_obs = _damped_settings(
        observation=ObservationConfig(max_actions=32),
        kickstart=Kickstart(policy_weight=0.2, fade_updates=2),
        teacher=scalar_teacher,
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out-bad-obs"))
    with recording(
        experiment="test",
        run="bad-obs",
        clocks=CLOCKS,
        directory=tmp_path / "track-bad-obs",
    ) as run:
        with pytest.raises(ValueError, match="another observation config"):
            self_play(bad_obs, run, updates=1)

    # Scalar teacher -> categorical_wdl student with value_weight > 0 is refused.
    bad_val = _damped_settings(
        model=ModelConfig(
            hidden_dim=16, num_attention_heads=2, value_kind="categorical_wdl"
        ),
        kickstart=Kickstart(policy_weight=0.2, value_weight=0.1, fade_updates=2),
        teacher=scalar_teacher,
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out-bad-val"))
    with recording(
        experiment="test",
        run="bad-val",
        clocks=CLOCKS,
        directory=tmp_path / "track-bad-val",
    ) as run:
        with pytest.raises(ValueError, match="requires a categorical_wdl teacher"):
            self_play(bad_val, run, updates=1)

    # Categorical WDL teacher -> scalar student with value_weight > 0 succeeds
    # (distills expected outcome P(win) - P(loss)).
    cross_val = _damped_settings(
        model=ModelConfig(hidden_dim=24, num_attention_heads=2, value_kind="scalar"),
        kickstart=Kickstart(policy_weight=0.2, value_weight=0.1, fade_updates=2),
        teacher=wdl_teacher,
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out-cross-val"))
    with recording(
        experiment="test",
        run="cross-val",
        clocks=CLOCKS,
        directory=tmp_path / "track-cross-val",
    ) as run:
        final = self_play(cross_val, run, updates=1)
    assert final.is_file()


def test_kickstart_resume_is_deterministic_and_refuses_teacher_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_a = _publish_teacher(tmp_path, monkeypatch, name="teacher-a", seed=44610)
    teacher_b = _publish_teacher(tmp_path, monkeypatch, name="teacher-b", seed=44611)

    settings = _damped_settings(
        model=ModelConfig(hidden_dim=24, num_attention_heads=2),
        kickstart=Kickstart(
            policy_weight=0.5, value_weight=0.2, schedule="cosine", fade_updates=2
        ),
        teacher=teacher_a,
    )

    # Uninterrupted 3-update run (active on u1, u2; faded to 0 on u3).
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out-uninterrupted"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test",
        run="uninterrupted",
        clocks=CLOCKS,
        directory=tmp_path / "track-uninterrupted",
    ) as run:
        whole_path = self_play(settings, run, updates=3)

    # Split run: u1 -> resume to u3.
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out-resumed"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test",
        run="resumed",
        clocks=CLOCKS,
        directory=tmp_path / "track-resumed",
    ) as run:
        self_play(settings, run, updates=1)

    # Refuses swapped teacher checkpoint on resume.
    with recording(
        experiment="test",
        run="resumed",
        clocks=CLOCKS,
        directory=tmp_path / "track-resumed",
        resume=True,
    ) as run:
        with pytest.raises(ValueError, match="resumed teacher checkpoint differs"):
            self_play(settings.model_copy(update={"teacher": teacher_b}), run, updates=3)

    # Refuses removing teacher on resume.
    without_teacher = _damped_settings(
        model=ModelConfig(hidden_dim=24, num_attention_heads=2)
    )
    with recording(
        experiment="test",
        run="resumed",
        clocks=CLOCKS,
        directory=tmp_path / "track-resumed",
        resume=True,
    ) as run:
        with pytest.raises(ValueError, match="requires its saved teacher checkpoint"):
            self_play(without_teacher, run, updates=3)

    # Resuming with the same teacher matches uninterrupted run bit for bit.
    with recording(
        experiment="test",
        run="resumed",
        clocks=CLOCKS,
        directory=tmp_path / "track-resumed",
        resume=True,
    ) as run:
        resumed_path = self_play(settings, run, updates=3)

    whole_state = load_checkpoint(whole_path)[0].state_dict()
    resumed_state = load_checkpoint(resumed_path)[0].state_dict()
    for name, value in whole_state.items():
        torch.testing.assert_close(resumed_state[name], value, rtol=0, atol=0)

    whole_data = read(tmp_path / "track-uninterrupted", "test", "uninterrupted")
    resumed_data = read(tmp_path / "track-resumed", "test", "resumed")
    assert list(resumed_data.points("teacher_policy_weight", "updates").y) == pytest.approx(
        list(whole_data.points("teacher_policy_weight", "updates").y)
    )
    assert list(resumed_data.points("teacher_policy_kl", "updates").y) == pytest.approx(
        list(whole_data.points("teacher_policy_kl", "updates").y)
    )
