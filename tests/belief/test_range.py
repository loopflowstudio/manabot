from __future__ import annotations

import math

import numpy as np
import pytest

from manabot.belief.range import BeliefError, BeliefState
from manabot.sim.teacher1_evidence import _fresh_env
from managym.possible_worlds import PossibleWorld, PossibleWorldSpace, WorldQuery


def _space(
    identity: str,
    pool: dict[str, int],
    hand_size: int,
    rows: list[tuple[dict[str, int], int]],
) -> PossibleWorldSpace:
    return PossibleWorldSpace(
        identity=identity,
        viewer=0,
        opponent=1,
        source_revision=0,
        source_viewer_state_hash=f"hash-{identity}",
        hand_size=hand_size,
        pool=tuple(sorted(pool.items())),
        total_weight=sum(weight for _, weight in rows),
        worlds=tuple(
            PossibleWorld(index, tuple(sorted(hand.items())), weight)
            for index, (hand, weight) in enumerate(rows)
        ),
        _engine=None,
    )


def test_interactive_opening_belief_uses_canonical_10832_world_space() -> None:
    env = _fresh_env(71)
    viewer = int(env._engine.current_agent_index())
    space = PossibleWorldSpace.from_engine(env._engine, viewer)
    belief = BeliefState.compatible_prior(space)

    assert belief.support_size == 10_832
    assert math.isclose(float(belief.probabilities.sum()), 1.0)
    assert belief.probability_bytes == 10_832 * 8
    assert belief.space.allocated_bytes > belief.probability_bytes
    assert belief.allocated_bytes == (
        belief.probability_bytes + belief.space.allocated_bytes
    )
    assert 1.0 <= belief.effective_range_size <= belief.support_size

    true_receipt = space.support(WorldQuery.true())
    bolt_receipt = space.support(WorldQuery.has("Lightning Bolt"))
    assert true_receipt.space_identity == space.identity
    assert true_receipt.support_size == space.support_size
    assert true_receipt.total_weight == space.total_weight
    assert 0 < bolt_receipt.support_size < space.support_size
    assert bolt_receipt.canonical_digest != true_receipt.canonical_digest


def test_compatible_prior_uses_authoritative_exact_weights() -> None:
    space = _space(
        "small",
        {"A": 2, "B": 2},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 4), ({"B": 2}, 1)],
    )
    belief = BeliefState.compatible_prior(space)

    assert belief.probability_of_hand({"A": 2}) == pytest.approx(1 / 6)
    assert belief.probability_of_hand({"A": 1, "B": 1}) == pytest.approx(4 / 6)
    assert belief.probability_of_hand({"B": 2}) == pytest.approx(1 / 6)


def test_unknown_draw_joins_next_canonical_space() -> None:
    before = _space(
        "before-draw",
        {"A": 2, "B": 1},
        1,
        [({"A": 1}, 2), ({"B": 1}, 1)],
    )
    after = _space(
        "after-draw",
        {"A": 2, "B": 1},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 2)],
    )

    drawn = BeliefState.compatible_prior(before).draw_unknown(after)

    assert drawn.space is after
    assert drawn.probability_of_hand({"A": 2}) == pytest.approx(1 / 3)
    assert drawn.probability_of_hand({"A": 1, "B": 1}) == pytest.approx(2 / 3)


def test_known_exit_conditions_then_joins_next_space() -> None:
    before = _space(
        "before-exit",
        {"A": 2, "B": 2},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 4), ({"B": 2}, 1)],
    )
    after = _space(
        "after-exit",
        {"A": 1, "B": 2},
        1,
        [({"A": 1}, 1), ({"B": 1}, 2)],
    )

    removed = BeliefState.compatible_prior(before).remove_known("A", after)

    assert removed.probability_of_hand({"A": 1}) == pytest.approx(1 / 5)
    assert removed.probability_of_hand({"B": 1}) == pytest.approx(4 / 5)


def test_known_return_is_deterministic_over_canonical_rows() -> None:
    before = _space(
        "before-return",
        {"A": 1, "B": 2},
        1,
        [({"A": 1}, 1), ({"B": 1}, 2)],
    )
    after = _space(
        "after-return",
        {"A": 2, "B": 2},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 4), ({"B": 2}, 1)],
    )

    returned = BeliefState.compatible_prior(before).add_known("A", after)

    assert returned.probability_of_hand({"A": 2}) == pytest.approx(1 / 3)
    assert returned.probability_of_hand({"A": 1, "B": 1}) == pytest.approx(2 / 3)


def test_action_floor_keeps_canonical_rows_but_zeroes_illegal_world() -> None:
    space = _space(
        "action",
        {"A": 1, "B": 1},
        1,
        [({"A": 1}, 1), ({"B": 1}, 1)],
    )
    updated = BeliefState.compatible_prior(space).condition_likelihood(
        likelihoods=(0.0, 0.0),
        legal_action_counts=(2, 0),
        matching_action_counts=(1, 0),
        epsilon=0.1,
    )

    assert updated.support_size == 2
    assert updated.positive_support_size == 1
    assert updated.probability_of_hand({"A": 1}) == pytest.approx(1.0)
    assert updated.probability_of_hand({"B": 1}) == 0.0


def test_action_floor_preserves_grouped_offer_multiplicity() -> None:
    space = _space(
        "grouped-action",
        {"A": 1, "B": 1},
        1,
        [({"A": 1}, 1), ({"B": 1}, 1)],
    )

    updated = BeliefState.compatible_prior(space).condition_likelihood(
        likelihoods=(0.0, 0.0),
        legal_action_counts=(4, 4),
        matching_action_counts=(2, 1),
        epsilon=0.1,
    )

    assert updated.probability_of_hand({"A": 1}) == pytest.approx(2 / 3)
    assert updated.probability_of_hand({"B": 1}) == pytest.approx(1 / 3)


def test_sampling_digest_rank_and_inclusion_are_replay_stable() -> None:
    space = _space(
        "stable",
        {"A": 2, "B": 2},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 4), ({"B": 2}, 1)],
    )
    belief = BeliefState.compatible_prior(space)
    rebuilt = BeliefState.compatible_prior(space)

    assert belief.sample_indexes(20, seed=71) == belief.sample_indexes(20, seed=71)
    assert belief.digest == rebuilt.digest
    assert belief.normalization_error < 1e-12
    assert belief.rank(1) == 1
    assert belief.inclusion_probabilities() == pytest.approx({"A": 5 / 6, "B": 5 / 6})


def test_impossible_update_and_bad_normalization_fail_closed() -> None:
    space = _space("only", {"A": 1}, 1, [({"A": 1}, 1)])
    belief = BeliefState.compatible_prior(space)

    with pytest.raises(BeliefError, match="empty hidden library"):
        belief.transport(
            _space("impossible", {"A": 1}, 2, [({"A": 2}, 1)]),
            hidden_draws=1,
        )
    with pytest.raises(BeliefError, match="no finite mass"):
        belief.condition_likelihood([0.0], [0], [0], epsilon=0.1)
    with pytest.raises(BeliefError, match="not normalized"):
        BeliefState(space, "bad", np.ones(1, dtype=np.float64))
