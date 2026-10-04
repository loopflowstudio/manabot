"""Joint normalization, physical odds, legality and learnable conditioning."""

from dataclasses import replace
import itertools

import pytest
import torch

from manabot.belief.sampling import (
    AutoregressiveBeliefSampler,
    SamplerInput,
    SamplerSchema,
    physical_deal_log_prob,
    sample_physical_deal,
)
from managym.possible_worlds import PossibleWorldSpace


def _input() -> SamplerInput:
    return SamplerInput("test-history-v1", (3, 2, 1), (1, 0, 0), 3, (0.0,))


def _model() -> AutoregressiveBeliefSampler:
    return AutoregressiveBeliefSampler(
        SamplerSchema("test-history-v1", ("A", "B", "C"), 1, 3), hidden_size=12
    )


def test_joint_normalization_and_known_card_physical_weights() -> None:
    row = _input()
    hands = torch.tensor(
        [
            hand
            for hand in itertools.product(range(4), range(3), range(2))
            if sum(hand) == 3 and hand[0] >= 1
        ]
    )
    inputs = [row] * len(hands)
    expected = physical_deal_log_prob(inputs, hands)
    model = _model()
    assert torch.allclose(model.log_prob(inputs, hands), expected, atol=1e-7)
    assert expected.exp().sum().item() == pytest.approx(1.0)
    # One known A is removed: remaining (2,2,1), draw 2. AA is 1/10.
    assert physical_deal_log_prob(
        [row], torch.tensor([[3, 0, 0]])
    ).exp().item() == pytest.approx(0.1)
    with torch.no_grad():
        model.correction.weight.normal_()
    assert model.log_prob(inputs, hands).exp().sum().item() == pytest.approx(1.0)


def test_sampling_preserves_full_joint_constraints_and_seed() -> None:
    rows = [_input()] * 400
    model = _model().eval()
    first = model.sample(rows, generator=torch.Generator().manual_seed(19))
    second = model.sample(rows, generator=torch.Generator().manual_seed(19))
    assert torch.equal(first, second)
    for counts in (
        first,
        sample_physical_deal(rows, generator=torch.Generator().manual_seed(8)),
    ):
        assert bool((counts.sum(dim=1) == 3).all())
        assert bool((counts >= torch.tensor([1, 0, 0])).all())
        assert bool((counts <= torch.tensor([3, 2, 1])).all())
        assert len(set(tuple(row) for row in counts.tolist())) > 1


def test_history_can_learn_distinct_joint_posteriors() -> None:
    torch.manual_seed(31)
    model = _model()
    rows = [replace(_input(), history_features=(float(value),)) for value in (-1, 1)]
    targets = torch.tensor([[3, 0, 0], [1, 1, 1]])
    initial = -model.log_prob(rows, targets).mean().item()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    for _ in range(70):
        optimizer.zero_grad()
        loss = -model.log_prob(rows, targets).mean()
        loss.backward()
        optimizer.step()
    assert -model.log_prob(rows, targets).mean().item() < initial * 0.1
    assert model.context.weight.grad is not None
    assert bool(torch.isfinite(model.context.weight.grad).all())
    swapped = list(reversed(rows))
    assert (
        model.log_prob(rows, targets).sum() > model.log_prob(swapped, targets).sum() + 2
    )


def test_empty_unknown_hand_and_invalid_inputs() -> None:
    row = replace(_input(), hand_size=1)
    assert torch.equal(_model().sample([row]), torch.tensor([[1, 0, 0]]))
    with pytest.raises(ValueError, match="compatible-deal"):
        _model().log_prob([_input()], torch.tensor([[0, 2, 1]]))
    with pytest.raises(ValueError, match="schema"):
        _model().sample([replace(_input(), vocabulary_identity="changed")])
    with pytest.raises(ValueError, match="no compatible deal"):
        replace(_input(), hand_size=7)
    with pytest.raises(ValueError, match="integer"):
        replace(_input(), hand_size=True)


def test_history_dropout_is_training_treatment() -> None:
    model = AutoregressiveBeliefSampler(
        _model().schema, hidden_size=12, history_dropout=1.0
    )
    with torch.no_grad():
        model.correction.weight.normal_()
    rows = [_input(), replace(_input(), history_features=(99.0,))]
    labels = torch.tensor([[3, 0, 0], [3, 0, 0]])
    training_scores = model.log_prob(rows, labels)
    assert training_scores[0].item() == training_scores[1].item()


def test_exact_world_contract_with_known_minima() -> None:
    space = PossibleWorldSpace.from_fixture(
        viewer=0,
        source_revision=0,
        source_viewer_state_hash="viewer-safe-test",
        pool={"A": 3, "B": 2, "C": 1},
        known_hand={"A": 1},
        hands=(
            ({"A": 3}, 1),
            ({"A": 2, "B": 1}, 4),
            ({"A": 2, "C": 1}, 2),
            ({"A": 1, "B": 2}, 1),
            ({"A": 1, "B": 1, "C": 1}, 2),
        ),
    )
    targets = torch.tensor(
        [[world.count(name) for name in ("A", "B", "C")] for world in space.worlds]
    )
    actual = _model().log_prob([_input()] * space.support_size, targets).exp()
    expected = torch.tensor(
        [world.weight / space.total_weight for world in space.worlds],
        dtype=torch.float64,
    )
    assert torch.allclose(actual, expected, atol=1e-12)


def test_large_support_inference_does_not_enumerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("sampler must not enumerate worlds")

    monkeypatch.setattr(PossibleWorldSpace, "from_engine", forbidden)
    schema = SamplerSchema(
        "large-vocabulary", tuple(f"card-{index}" for index in range(40)), 0, 4
    )
    row = SamplerInput(schema.vocabulary_identity, (4,) * 40, (0,) * 40, 20, ())
    model = AutoregressiveBeliefSampler(schema, hidden_size=8).eval()
    counts = model.sample([row], generator=torch.Generator().manual_seed(5))
    assert counts.shape == (1, 40)
    assert counts.sum().item() == 20
    assert torch.isfinite(model.log_prob([row], counts)).all()
