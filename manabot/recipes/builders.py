"""Reusable settings for current self-play recipes. No builder executes training."""

from typing import Literal

from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
)
from manabot.training.decks import UniformDecks
from manabot.training.learning import DampedPolicyLearning, IterationSchedule, Learning
from manabot.training.loops.self_play import SelfPlay
from manabot.training.state.game_errors import GameErrorPolicy
import managym

# Semantic pack of the selected Allies/Lessons matchup that current recipes train on.
PACK = "ur-lessons-vs-gw-allies"

# The damped rule's curves are written for this many updates (`IterationSchedule`).
REFERENCE_UPDATES = 10_000


def move_learning(schedule: IterationSchedule | None = None) -> DampedPolicyLearning:
    """Ataraxos move learning; a schedule pins its curves to a reference horizon."""
    return DampedPolicyLearning(gradient="damped_policy", iteration_schedule=schedule)


def fitted_horizon(wall_seconds: float, seconds_per_update: float) -> int:
    """Updates a wall-bounded run is expected to reach, capped at the reference.

    `seconds_per_update` is a measured (calibration) cost; no slowdown allowance
    is applied, because the end of the damped curve is flat and a run reaching
    only part of its horizon still ends at a low rate.
    """
    if wall_seconds <= 0 or seconds_per_update <= 0:
        raise ValueError("wall and per-update seconds must be positive")
    return max(1, min(REFERENCE_UPDATES, int(wall_seconds // seconds_per_update)))


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


def gray_ogre_match() -> MatchConfig:
    """The small Gray Ogre against Llanowar Elves matchup for belief and local search."""
    return MatchConfig(
        hero_deck={"Mountain": 8, "Gray Ogre": 8},
        villain_deck={"Forest": 8, "Llanowar Elves": 8},
    )


def sampled_self_play(
    *,
    world: str,
    agent: ModelConfig,
    observation: ObservationConfig,
    wall_seconds: float,
    seconds_per_update: float,
    deck_source: UniformDecks | None = None,
    match: MatchConfig | None = None,
    elemental_uniform: bool = False,
    workers: Literal[1, 2, 4] = 4,
) -> SelfPlay:
    """Damped self-play over a sampled roster with a fixed, fitted rate horizon.

    The launch owns the time allowance and update target. These settings retain
    the horizon calculated before launch, including on a target-only continuation.
    """
    return SelfPlay(
        world=world,
        deck_source=deck_source,
        match=match,
        observation=observation,
        model=agent,
        streams=256,
        transitions=64,
        matchup_curriculum="elemental-uniform" if elemental_uniform else "fixed",
        learning=move_learning(
            IterationSchedule(
                steps=fitted_horizon(wall_seconds, seconds_per_update),
                reference_steps=REFERENCE_UPDATES,
            )
        ),
        game_errors=GameErrorPolicy(threshold=0.25, recover_capacity=True),
        device="cuda",
        workers=workers,
        bucketed_shapes=True,
        live_tokens=True,
    )


# ---------------------------------------------------------------------------
# Self-play loop settings


def elemental_self_play() -> SelfPlay:
    """The current Elemental settings: damped-policy self-play over the ten decks.

    Both seats learn, each game draws its pair uniformly from the authored
    Elemental roster, and an engine failure ends one game within a 10% rolling
    limit. Model, observation, learning and streams are the frozen
    `elemental-long-183-v1` recipe's; the world is the installed one.
    Milestones fall every 50 updates. CPU unless the caller changes `device`.
    """
    pack = "elemental-world-v2"
    first, second = managym.authored_deck_keys(pack)[:2]
    return SelfPlay(
        match=MatchConfig.authored(pack, first, second).model_copy(
            update={"content_pack": pack}
        ),
        observation=ObservationConfig(
            max_cards_per_player=60,
            max_permanents_per_player=40,
            max_actions=64,
            max_focus_objects=2,
            max_events=32,
        ),
        model=ModelConfig(
            semantic_pack=pack,
            hidden_dim=64,
            attention_layers=2,
            num_attention_heads=4,
            value_aggregation="masked_mean",
        ),
        learning=move_learning(),
        behavior="current-self",
        streams=256,
        transitions=64,
        learning_seats="both",
        matchup_curriculum="elemental-uniform",
        game_errors=GameErrorPolicy(
            threshold=0.1, window=100, warmup=100, max_game_actions=40000
        ),
        checkpoint_every=50,
    )


def gray_ogre_self_play(*, horizon: int) -> SelfPlay:
    """A tiny PPO self-play policy on the Gray Ogre matchup, for belief pilots."""
    return SelfPlay(
        match=gray_ogre_match(),
        model=ModelConfig(hidden_dim=16, num_attention_heads=2),
        learning=Learning(horizon=horizon, epochs=1, minibatches=1),
        streams=2,
        transitions=16,
    )
