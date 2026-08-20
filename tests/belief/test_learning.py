"""Behavioral proof for supervised exact-world belief learning."""

import numpy as np

from manabot.belief.learning import (
    BeliefTrainingExample,
    pack_belief_examples,
)
from manabot.belief.learning_demo import run_demo
from tests.belief.support import fixture_history, fixture_schema, fixture_space


def test_ragged_training_batch_uses_exact_combinatorial_p0() -> None:
    history = fixture_history()
    space = fixture_space(history)
    schema = fixture_schema(space)
    examples = tuple(
        BeliefTrainingExample(
            world_space=space,
            viewer_history=history,
            target_world=target,
            supervision_receipt=f"test-supervision-{target}",
        )
        for target in (1, 3)
    )

    packed = pack_belief_examples(examples, schema)

    support = space.support_size
    expected = np.asarray([world.weight / space.total_weight for world in space.worlds])
    assert packed.world_counts.shape == (2 * support, len(space.pool))
    assert packed.offsets.tolist() == [0, support, 2 * support]
    assert packed.world_batch.tolist() == [0] * support + [1] * support
    assert packed.target_world.tolist() == [1, 3]
    assert np.allclose(
        packed.log_p0[:support].exp().numpy(), expected, atol=1e-15, rtol=0.0
    )


def test_real_engine_fresh_model_overfits_without_truth_at_inference() -> None:
    evidence = run_demo(steps=40, seed=197)

    assert evidence["proof"] == "fresh-model-supervised-exact-world-overfit"
    assert evidence["support_size"] == 4_865
    assert evidence["training_examples"] == 1
    assert evidence["viewer_history_events"] > 0
    assert evidence["first_informative_action_by_opponent"] is True
    assert evidence["inference_inputs"] == ["possible_world_space", "viewer_history"]
    assert evidence["supervision_access"] == "authority-only-materialized-world"
    assert evidence["fresh_model_started_at_p0"] is True
    assert evidence["preinformative_exact_p0"] is True
    assert evidence["final_nll"] < 0.02
    assert evidence["initial_nll"] - evidence["final_nll"] > 9.0
    assert evidence["learned_true_world_probability"] > 0.99
    assert evidence["ordinary_agent_used_learned_belief"] is True
    assert evidence["generated_override_byte_identical"] is True
    assert evidence["viewer_hidden_swap_identical"] is True
