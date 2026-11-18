"""Validated learning rules and schedules shared by recipes and updates."""

from typing import Literal

from pydantic import Field, model_validator

from manabot.infra.schema import Strict

BatchClosure = Literal["exact-quota", "complete-games"]


class Schedule(Strict):
    initial: float = Field(ge=0)
    decay: float = Field(default=0, ge=0)
    power: float = Field(default=1, ge=0)

    def at(self, progress: float) -> float:
        return (
            self.initial
            * (1 + self.decay * min(1.0, max(0.0, progress))) ** -self.power
        )


class Learning(Strict):
    """PPO settings. Schedules read `update / horizon`, clamped at 1.

    Live self-play requires the horizon, so extending a run's target leaves
    earlier rates unchanged. Retained regime records omit it and keep their bytes.
    """

    horizon: int | None = Field(
        default=None, ge=1, strict=True, exclude_if=lambda value: value is None
    )
    gradient: Literal["ppo"] = "ppo"
    gamma: float = Field(default=1, ge=0, le=1)
    policy_lambda: float = Field(default=0.95, ge=0, le=1)
    value_lambda: float = Field(default=1, ge=0, le=1)
    filter_kind: Literal["top_count", "quantile"] = "top_count"
    filter_scope: Literal["actor_critic", "actor"] = "actor_critic"
    retained_fraction: float = Field(default=0.5, gt=0, le=1)
    min_advantage: float = Field(default=0, ge=0)
    reference: Literal["uniform", "action_type_uniform"] = "uniform"
    learning_rate: Schedule = Schedule(initial=2.5e-4, decay=9)
    tau: Schedule = Schedule(initial=0.01, decay=9)
    collection_kl: float = Field(default=0.1, ge=0)
    clip: float = Field(default=0.1, gt=0)
    value_weight: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    epochs: int = Field(default=4, ge=1)
    minibatches: int = Field(default=4, ge=1)
    ema: float | None = Field(default=None, ge=0, lt=1)


class IterationSchedule(Strict):
    """Map a declared run horizon onto a reference iteration schedule.

    Both endpoints are preserved. A one-step run uses the initial rate. The
    mapping is deterministic in completed updates, never elapsed wall time;
    continuation must retain the original horizon rather than start a new one.
    """

    steps: int = Field(ge=1, strict=True)
    reference_steps: int = Field(default=10_000, ge=2, strict=True)

    def at(self, iteration: int) -> float:
        if iteration < 1:
            raise ValueError("iteration must be positive")
        if self.steps == 1:
            return 1.0
        progress = min(1.0, (iteration - 1) / (self.steps - 1))
        return 1 + progress * (self.reference_steps - 1)


class LearnerDecisionSchedule(Strict):
    """Map completed learner decisions onto the method's reference curve.

    Position is sampled before collection/update, so a complete game batch uses
    one rate. A run started from pretrained weights begins at zero; a resumed
    run retains its absolute decision count. Passing the endpoint holds the
    final rate.
    """

    transitions: int = Field(ge=1, strict=True)
    reference_steps: int = Field(default=10_000, ge=2, strict=True)

    def at(self, completed: int) -> float:
        if completed < 0:
            raise ValueError("completed learner transitions must be nonnegative")
        return 1 + min(1.0, completed / self.transitions) * (self.reference_steps - 1)


class DampedPolicyLearning(Strict):
    """Unnormalized clipped policy update with reverse-KL regularization.

    Iteration schedules, one epoch and no advantage normalization belong to
    this method, rather than inheriting the PPO control's elapsed-time recipe.
    """

    gradient: Literal["damped_policy"]
    policy_lambda: float = Field(default=0.5, ge=0, le=1)
    value_lambda: float = Field(default=0.8, ge=0, le=1)
    filter_kind: Literal["quantile", "top_count"] = "quantile"
    filter_scope: Literal["actor_critic", "actor"] = "actor_critic"
    advantage_quantile: float = Field(default=0.75, ge=0, le=1)
    min_advantage: float = Field(default=0.01, ge=0)
    reference: Literal["uniform", "action_type_uniform"] = "action_type_uniform"
    clip: float = Field(default=0.2, gt=0, lt=1)
    collection_kl: float = Field(default=0.1, ge=0)
    learning_rate_scale: float = Field(default=0.5, gt=0)
    learning_rate_power: float = Field(default=1.1, ge=0)
    learning_rate_min: float = Field(default=5e-6, gt=0)
    learning_rate_max: float = Field(default=1e-4, gt=0)
    tau_scale: float = Field(default=0.05, ge=0)
    tau_power: float = Field(default=0.3, ge=0)
    max_grad_norm: float = Field(default=0.267, gt=0)
    ema: float | None = Field(default=0.999, ge=0, lt=1)
    iteration_schedule: IterationSchedule | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    decision_schedule: LearnerDecisionSchedule | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def valid_learning_rate_bounds(self) -> "DampedPolicyLearning":
        if self.learning_rate_min > self.learning_rate_max:
            raise ValueError("learning rate minimum exceeds maximum")
        if self.iteration_schedule is not None and self.decision_schedule is not None:
            raise ValueError("choose an iteration or learner-decision schedule")
        return self

    def rates(
        self, iteration: int, *, learner_transitions: int | None = None
    ) -> tuple[float, float]:
        """Rates at the start of an update, including skipped optimizer work.

        `iteration` is the one-based collection/update iteration. A decision
        schedule reads `learner_transitions`, the decisions learned before it.
        """
        if iteration < 1:
            raise ValueError("Damped policy iteration must be positive")
        if self.decision_schedule is not None:
            if learner_transitions is None:
                raise ValueError(
                    "learner-decision schedule requires completed transitions"
                )
            position = self.decision_schedule.at(learner_transitions)
        else:
            position = (
                self.iteration_schedule.at(iteration)
                if self.iteration_schedule is not None
                else iteration
            )
        rate = self.learning_rate_scale / position**self.learning_rate_power
        return (
            min(self.learning_rate_max, max(self.learning_rate_min, rate)),
            self.tau_scale / position**self.tau_power,
        )


# float32 is exact single precision. tf32 lets CUDA matmul and cuDNN use TF32
# tensor cores; bf16 autocasts policy forwards to bfloat16 on CUDA or MPS
# (manabot.backends.torch.learner).
Precision = Literal["float32", "tf32", "bf16"]
