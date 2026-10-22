"""Named training regimes for current experiments, composed in Python.

Each function returns a fresh `TrainingRegime` for the native world, so a caller
may edit its copy. Model defaults carry the recommended settings; a function
states only where it departs from them, and variants derive from a few
canonical regimes:

- `batched_baseline` is the GPU baseline from the size and batching study;
  `scale_ladder` is its long allocation-bounded run and
  `gpu_training_profile` its large-model profiling counterpart.
- `ataraxos_mini` is the CPU Ataraxos move-learning recipe that the early
  learning-rate screen, mirror screen and sustained baseline vary.
- `direct_self_play` is the two-checkpoint CPU PPO regime; the Ataraxos ablation
  arms, compound decisions and search distillation share its world and model.
- `belief_sampler` and its local-update variants run on the small Gray Ogre
  matchup used by the belief and local-search documentation.

`REGIMES` maps the names commands and documents use to these functions:

    uv run manabot train --regime-name batched-baseline --out .runs/batched/run
    uv run python -m experiments.recipes --name batched-baseline --out regime.json

The second writes the JSON record that `manabot train --regime` and
`manabot deploy --regime` read. A `TrainingRun` records the source commit, the
complete regime with its `id`, and `regime_digest`, the canonical JSON digest.
"""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

from manabot.arena.match import selected_match
from manabot.infra.hypers import AgentSpec, MatchHypers
from manabot.training.models import (
    AtaraxosMoveLearning,
    CollectBelief,
    CollectLocalUpdate,
    CollectSearch,
    Execution,
    IterationSchedule,
    Learning,
    LocalSearchConfig,
    Operation,
    Schedule,
    TrainBelief,
    TrainCompound,
    TrainingRegime,
    TrainSelfPlay,
    TrainSupervised,
)
import managym

PACK = "ur-lessons-vs-gw-allies"
GIB = 1024**3


def _move_learning(schedule: IterationSchedule | None = None) -> AtaraxosMoveLearning:
    if schedule is None:
        return AtaraxosMoveLearning(gradient="ataraxos_move")
    return AtaraxosMoveLearning(gradient="ataraxos_move", iteration_schedule=schedule)


# ---------------------------------------------------------------------------
# GPU baseline
# ---------------------------------------------------------------------------


def batched_baseline() -> TrainingRegime:
    """Width 64, depth 2, 64 streams on CUDA with a cross-balanced curriculum."""
    return TrainingRegime(
        id="batching-grid-small-s64",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(
            semantic_pack=PACK, value_aggregation="masked_mean", attention_layers=2
        ),
        stages=[
            TrainSelfPlay(
                id="self-play",
                operation="train_self_play",
                updates=128,
                streams=64,
                transitions=64,
                matchup_curriculum="cross-balanced",
                learning=_move_learning(
                    IterationSchedule(steps=128, reference_steps=2048)
                ),
                execution=Execution(
                    device="cuda", wall_seconds=1740, memory_bytes=48 * GIB
                ),
            )
        ],
        wall_seconds=1770,
    )


def scale_ladder() -> TrainingRegime:
    """`batched_baseline` at 256 streams, sized by a money and wall-time allocation.

    `updates` is an upper bound: the allocation pauses training at its deadline.
    Without an iteration schedule the learning-rate and tau curves follow
    absolute iterations, so a prefix of the run is the same recipe at a smaller
    budget (experiments/scale-ladder.md).
    """
    regime = batched_baseline()
    regime.id = "scale-ladder-allies-lessons-v1"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 100_000
    stage.streams = 256
    stage.learning = _move_learning()
    stage.execution = Execution(
        device="cuda", wall_seconds=400_000, memory_bytes=40 * GIB
    )
    regime.wall_seconds = 400_000
    regime.schedule_clock = "iteration_fraction"
    return TrainingRegime.model_validate(regime.model_dump())


def gpu_training_profile() -> TrainingRegime:
    """The large model (width 384, depth 8) whose CUDA pipeline costs are profiled."""
    return TrainingRegime(
        id="gpu-training-profile",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(
            semantic_pack=PACK,
            value_aggregation="value_token",
            attention_layers=8,
            attention_feedforward_dim=1536,
            hidden_dim=384,
        ),
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=512,
                streams=64,
                transitions=8,
                learning=_move_learning(),
                execution=Execution(device="cuda", wall_seconds=4500),
            )
        ],
        wall_seconds=4560,
        schedule_clock="iteration_fraction",
    )


# ---------------------------------------------------------------------------
# CPU Ataraxos move learning
# ---------------------------------------------------------------------------


def _move_stage(
    id: str,
    *,
    updates: int,
    seconds: float,
    initial: str | None = None,
    streams: int = 4,
    transitions: int = 64,
    memory_bytes: int = 32 * GIB,
) -> TrainSelfPlay:
    return TrainSelfPlay(
        id=id,
        operation="train_self_play",
        initial=initial,
        updates=updates,
        streams=streams,
        transitions=transitions,
        learning=_move_learning(),
        execution=Execution(wall_seconds=seconds, memory_bytes=memory_bytes),
    )


def ataraxos_mini() -> TrainingRegime:
    """Masked-mean width-64 depth-2 move learning: one 1,500-update CPU stage."""
    return TrainingRegime(
        id="early-learning-rate-mini-source",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(
            semantic_pack=PACK, value_aggregation="masked_mean", attention_layers=2
        ),
        stages=[
            _move_stage("self-play", updates=1500, seconds=10800, memory_bytes=15 * GIB)
        ],
        wall_seconds=10800,
    )


def mirror_screen_baseline() -> TrainingRegime:
    """`ataraxos_mini` shortened to a 20-update, 12-stream recoverable midpoint."""
    regime = ataraxos_mini()
    regime.id = "mirror-screen-baseline"
    regime.stages = [
        _move_stage(
            "midpoint",
            updates=20,
            streams=12,
            transitions=32,
            seconds=600,
            memory_bytes=15 * GIB,
        )
    ]
    regime.wall_seconds = 660
    regime.schedule_clock = "iteration_fraction"
    regime.recovery_max_microsteps = 1_000_000
    return TrainingRegime.model_validate(regime.model_dump())


def sustained_baseline_source() -> TrainingRegime:
    """`ataraxos_mini` as two cumulative 620-update checkpoints."""
    regime = ataraxos_mini()
    regime.id = "actor-critic"
    regime.stages = [
        _move_stage(
            f"policy-{index}",
            initial=f"policy-{index - 1}" if index else None,
            updates=620,
            seconds=3590,
            memory_bytes=15 * GIB,
        )
        for index in range(2)
    ]
    regime.wall_seconds = 7200
    return TrainingRegime.model_validate(regime.model_dump())


def _move_checkpoints(seconds: float) -> list[Operation]:
    return [
        _move_stage(
            f"policy-{index}",
            initial=f"policy-{index - 1}" if index else None,
            updates=1,
            seconds=seconds,
        )
        for index in range(2)
    ]


def value_model_baseline() -> TrainingRegime:
    """Default-width depth-1 move learning: two one-update checkpoints."""
    return TrainingRegime(
        id="value-historical-mean-1",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK),
        stages=_move_checkpoints(30),
        wall_seconds=60,
    )


def ataraxos_move_scalar() -> TrainingRegime:
    """Width-16 two-head scalar move learning: two one-update checkpoints."""
    return TrainingRegime(
        id="ataraxos-move-scalar",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK, hidden_dim=16, num_attention_heads=2),
        stages=_move_checkpoints(80),
        wall_seconds=180,
    )


def ataraxos_move() -> TrainingRegime:
    """`ataraxos_move_scalar` with a categorical win/draw/loss value head."""
    regime = ataraxos_move_scalar()
    regime.id = "ataraxos-move"
    regime.agent.value_kind = "categorical_wdl"
    return TrainingRegime.model_validate(regime.model_dump())


# ---------------------------------------------------------------------------
# CPU PPO self-play and its variants
# ---------------------------------------------------------------------------


def _ppo_checkpoints(learning: Learning) -> list[Operation]:
    return [
        TrainSelfPlay(
            id=f"policy-{index}",
            operation="train_self_play",
            initial=f"policy-{index - 1}" if index else None,
            transitions=64,
            learning=learning.model_copy(deep=True),
        )
        for index in range(2)
    ]


def direct_self_play() -> TrainingRegime:
    """Two two-update PPO checkpoints with every Ataraxos treatment enabled."""
    return TrainingRegime(
        id="direct-self-play",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK),
        stages=_ppo_checkpoints(Learning()),
        wall_seconds=900,
    )


AblationArm = Literal[
    "rl-control",
    "separate-estimators",
    "advantage-filtering",
    "coordinated-decay",
    "combined",
]
# Which Ataraxos treatments each ablation arm keeps: separate policy/value
# estimators (value lambda 1), advantage filtering (half the samples retained)
# and coordinated learning-rate/regularization decay.
_ABLATION_TREATMENTS: dict[AblationArm, tuple[bool, bool, bool]] = {
    "rl-control": (False, False, False),
    "separate-estimators": (True, False, False),
    "advantage-filtering": (False, True, False),
    "coordinated-decay": (False, False, True),
    "combined": (True, True, True),
}


def ataraxos_ablation(arm: AblationArm) -> TrainingRegime:
    """`direct_self_play` with only the arm's Ataraxos treatments enabled."""
    estimators, filtering, decay = _ABLATION_TREATMENTS[arm]
    learning = Learning(
        value_lambda=1.0 if estimators else 0.95,
        retained_fraction=0.5 if filtering else 1.0,
        learning_rate=Schedule(initial=2.5e-4, decay=9 if decay else 0),
        tau=Schedule(initial=0.01, decay=9 if decay else 0),
    )
    regime = direct_self_play()
    regime.id = arm
    regime.stages = _ppo_checkpoints(learning)
    return TrainingRegime.model_validate(regime.model_dump())


def compound_decisions(
    grouping: Literal["sequential", "grouped"],
    estimator: Literal["bootstrapped", "outcome"],
) -> TrainingRegime:
    """Compound-decision policy training on the `direct_self_play` setup."""
    learning = Learning(
        value_lambda=0.95,
        retained_fraction=1.0,
        learning_rate=Schedule(initial=2.5e-4),
        tau=Schedule(initial=0.01),
        epochs=1,
    )
    short = {"bootstrapped": "bootstrap", "outcome": "outcome"}[estimator]
    return TrainingRegime(
        id=f"compound-{grouping}-{short}",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK, compound_decisions=True),
        stages=[
            TrainCompound(
                id=f"policy-{index}",
                operation="train_compound",
                initial=f"policy-{index - 1}" if index else None,
                grouping=grouping,
                estimator=estimator,
                learning=learning.model_copy(deep=True),
            )
            for index in range(2)
        ],
        wall_seconds=900,
    )


def search_distillation() -> TrainingRegime:
    """Two rounds of search collection, each followed by supervised training."""
    return TrainingRegime(
        id="search-distillation",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK),
        stages=[
            CollectSearch(id="data-0", operation="collect_search", games=2),
            TrainSupervised(
                id="policy-0",
                operation="train_supervised",
                datasets=["data-0"],
                epochs=2,
            ),
            CollectSearch(id="data-1", operation="collect_search", games=2),
            TrainSupervised(
                id="policy-1",
                operation="train_supervised",
                datasets=["data-0", "data-1"],
                initial="policy-0",
                epochs=2,
            ),
        ],
        wall_seconds=900,
    )


# ---------------------------------------------------------------------------
# Belief sampling and local policy search
# ---------------------------------------------------------------------------


def _small_match() -> MatchHypers:
    return MatchHypers(
        hero_deck={"Mountain": 8, "Gray Ogre": 8},
        villain_deck={"Forest": 8, "Llanowar Elves": 8},
    )


def _belief_stages() -> list[Operation]:
    return [
        TrainSelfPlay(
            id="policy",
            operation="train_self_play",
            updates=1,
            streams=2,
            transitions=16,
            learning=Learning(epochs=1, minibatches=1),
            execution=Execution(wall_seconds=60),
        ),
        CollectBelief(
            id="histories",
            operation="collect_belief",
            policy="policy",
            games=3,
            max_steps=1000,
            execution=Execution(wall_seconds=60),
        ),
        TrainBelief(
            id="belief",
            operation="train_belief",
            dataset="histories",
            steps=4,
            batch_size=8,
            hidden_size=16,
            history_dropout=0.0,
            evaluation_samples=4,
            execution=Execution(wall_seconds=60),
        ),
    ]


def belief_sampler() -> TrainingRegime:
    """Train a frozen policy, collect its histories and fit the belief sampler."""
    return TrainingRegime(
        id="frozen-policy-belief-pilot",
        world=managym.WORLD_VERSION,
        match=_small_match(),
        agent=AgentSpec(hidden_dim=16, num_attention_heads=2),
        stages=_belief_stages(),
        wall_seconds=180,
    )


def learned_belief_local_search() -> TrainingRegime:
    """`belief_sampler` followed by belief-sampled local search and a student."""
    regime = belief_sampler()
    regime.id = "learned-belief-local-search-proof"
    regime.stages = [
        *_belief_stages(),
        CollectLocalUpdate(
            id="labels",
            operation="collect_local_update",
            policy="policy",
            games=8,
            max_steps=1000,
            search=LocalSearchConfig(depth=1, decision_seconds=5, sampling="learned"),
            sampler="belief",
            execution=Execution(wall_seconds=60),
        ),
        TrainSupervised(
            id="student",
            operation="train_supervised",
            target="local_soft",
            datasets=["labels"],
            initial="policy",
            epochs=1,
            batch_size=64,
            execution=Execution(wall_seconds=30),
        ),
    ]
    return TrainingRegime.model_validate(regime.model_dump())


def frozen_policy_local_update() -> TrainingRegime:
    """Local-update labels from a frozen policy, three targets, then a second round."""

    def student(
        id: str,
        target: Literal["local_soft", "local_argmax", "local_allocation"],
        datasets: list[str],
        initial: str,
        epochs: int,
    ) -> TrainSupervised:
        return TrainSupervised(
            id=id,
            operation="train_supervised",
            target=target,
            datasets=datasets,
            initial=initial,
            epochs=epochs,
            batch_size=64,
        )

    return TrainingRegime(
        id="frozen-policy-local-update",
        world=managym.WORLD_VERSION,
        match=_small_match(),
        agent=AgentSpec(hidden_dim=8, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=32,
                learning=Learning(value_lambda=1.0, epochs=1, minibatches=1),
            ),
            CollectLocalUpdate(
                id="labels",
                operation="collect_local_update",
                policy="policy",
                games=2,
                search=LocalSearchConfig(
                    worlds=2, full_probability=0.5, decision_seconds=5
                ),
            ),
            student("local-soft", "local_soft", ["labels"], "policy", 2),
            student("local-argmax", "local_argmax", ["labels"], "policy", 2),
            student("local-allocation", "local_allocation", ["labels"], "policy", 2),
            CollectLocalUpdate(
                id="round-two",
                operation="collect_local_update",
                policy="local-soft",
                games=2,
                search=LocalSearchConfig(
                    depth=1, full_probability=1.0, decision_seconds=5
                ),
            ),
            student(
                "cumulative", "local_soft", ["labels", "round-two"], "local-soft", 1
            ),
        ],
        wall_seconds=180,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGIMES: dict[str, Callable[[], TrainingRegime]] = {
    "batched-baseline": batched_baseline,
    "scale-ladder": scale_ladder,
    "gpu-training-profile": gpu_training_profile,
    "ataraxos-mini": ataraxos_mini,
    "mirror-screen-baseline": mirror_screen_baseline,
    "sustained-baseline-source": sustained_baseline_source,
    "value-model-baseline": value_model_baseline,
    "ataraxos-move": ataraxos_move,
    "ataraxos-move-scalar": ataraxos_move_scalar,
    "direct-self-play": direct_self_play,
    **{arm: partial(ataraxos_ablation, arm) for arm in _ABLATION_TREATMENTS},
    "compound-sequential-bootstrap": partial(
        compound_decisions, "sequential", "bootstrapped"
    ),
    "compound-grouped-bootstrap": partial(
        compound_decisions, "grouped", "bootstrapped"
    ),
    "compound-sequential-outcome": partial(compound_decisions, "sequential", "outcome"),
    "compound-grouped-outcome": partial(compound_decisions, "grouped", "outcome"),
    "search-distillation": search_distillation,
    "belief-sampler": belief_sampler,
    "learned-belief-local-search": learned_belief_local_search,
    "frozen-policy-local-update": frozen_policy_local_update,
}


def regime(name: str) -> TrainingRegime:
    """Return a fresh copy of the named regime; raise for an unknown name."""
    if name not in REGIMES:
        raise KeyError(f"unknown regime {name!r}; available: {', '.join(REGIMES)}")
    return REGIMES[name]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--name", required=True, choices=REGIMES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(regime(args.name).model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
