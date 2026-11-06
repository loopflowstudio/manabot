"""The ten-deck Elemental recipe on the current world, varied one field at a time (Task 07c490).

MTG-181/183's frozen recipe (`experiments/regimes/elemental-long-183-v1.json`) learns
from both seats of uniform random Elemental deck pairs, 256 streams, at least 64
decisions per stream per update, damped policy learning on absolute iteration
schedules. Jack Heart placed the current recipe at this ten-deck setup (2026-11-05),
so the batch-closure and lambda studies first run on Allies/Lessons (MTG-251,
`experiments/batched_baseline.py`) repeat here. The frozen file names world w13;
these variants relabel it to the current world, as MTG-181 did for w10, with new IDs.
Protocol: experiments/protocols/closure-lambda-elemental-v1.md.
"""

from experiments.elemental_long import REGIME
from manabot.training.regime import BatchClosure, TrainingRegime, TrainSelfPlay
import managym

# Rows per stream per update under one learning seat. The complete arm keeps the
# recipe's floor; the exact arm's quota is what that floor realized per stream in
# calibration once game length settled (protocol, calibration section).
CLOSURE_COMPLETE_FLOOR = 64
CLOSURE_QUOTA: int | None = 121


def _stage(regime: TrainingRegime) -> TrainSelfPlay:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return stage


def elemental_recipe(regime_id: str, updates: int) -> TrainingRegime:
    """The frozen recipe to an absolute `updates` target, on the current world."""
    regime = TrainingRegime.model_validate_json(REGIME.read_bytes())
    regime.id = regime_id
    regime.world = managym.WORLD_VERSION
    stage = _stage(regime)
    # Damped-policy rates follow the absolute iteration, so the endpoint leaves the
    # learning schedule unchanged; the launch sets the watchdog.
    stage.updates = updates
    # On the current world a game can exceed the v1 input's 64 continuation
    # relations, which the frozen recipe treats as fatal (calibration 491fa11b9fdf
    # stopped at update 9). Recovery discards that game's unfinished samples and
    # counts it against the recipe's own game-error allowance.
    if stage.game_errors is not None:
        stage.game_errors.recover_capacity = True
    return regime


def closure_comparison(
    closure: BatchClosure, updates: int, *, quota: int | None = CLOSURE_QUOTA
) -> TrainingRegime:
    """One-seat Elemental differing only in when each batch closes.

    Exact quota needs one learning seat, so both arms learn from one seat
    (seat-balanced across streams), and everything else of the recipe stays,
    capacity recovery included. Game length moves with the policy, so recorded
    counters verify the row match in each run.
    """
    transitions = CLOSURE_COMPLETE_FLOOR if closure == "complete-games" else quota
    if transitions is None:
        raise ValueError("the exact arm's quota is set by calibration")
    regime = elemental_recipe(f"elemental-closure-{closure}-v1", updates)
    stage = _stage(regime)
    stage.learning_seats = "one"
    stage.batch_closure = closure
    stage.transitions = transitions
    return TrainingRegime.model_validate(regime.model_dump())


def lambda_sweep(
    policy_lambda: float, value_lambda: float, updates: int
) -> TrainingRegime:
    """The two-seat recipe differing only in its traces."""
    regime = elemental_recipe(
        f"elemental-lambda-p{policy_lambda:g}-v{value_lambda:g}-v1", updates
    )
    stage = _stage(regime)
    stage.learning.policy_lambda = policy_lambda
    stage.learning.value_lambda = value_lambda
    return TrainingRegime.model_validate(regime.model_dump())
