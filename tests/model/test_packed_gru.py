"""The hand-written packed GRU (MPS path) matches `nn.GRU`, run here on CPU."""

import pytest
import torch
from torch import nn

from manabot.backends.torch.packed_gru import hand_packed_gru, packed_gru

# Shrinking live counts, including several sequences finishing at one step.
SIZES = [5, 5, 4, 2, 2, 1, 1, 1]


@pytest.mark.parametrize("with_initial", [False, True])
def test_hand_packed_gru_matches_nn_gru_outputs_states_and_gradients(
    with_initial: bool,
) -> None:
    torch.manual_seed(11)
    gru = nn.GRU(6, 7, batch_first=True).double()
    data = torch.randn(sum(SIZES), 6, dtype=torch.float64, requires_grad=True)
    initial = (
        torch.randn(SIZES[0], 7, dtype=torch.float64, requires_grad=True)
        if with_initial
        else None
    )
    # Weights on every output and final state, so a dropped path changes the loss.
    out_weight = torch.randn(sum(SIZES), 7, dtype=torch.float64)
    final_weight = torch.randn(SIZES[0], 7, dtype=torch.float64)
    results = []
    for run in (packed_gru, hand_packed_gru):
        gru.zero_grad()
        data.grad = None
        if initial is not None:
            initial.grad = None
        outputs, final = run(gru, data, SIZES, initial)
        ((outputs * out_weight).sum() + (final * final_weight).sum()).backward()
        grads = [p.grad.clone() for p in gru.parameters()] + [data.grad.clone()]
        if initial is not None:
            assert initial.grad is not None
            grads.append(initial.grad.clone())
        results.append((outputs.detach(), final.detach(), grads))
    (reference_out, reference_final, reference_grads), (out, final, grads) = results
    torch.testing.assert_close(out, reference_out, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(final, reference_final, atol=1e-12, rtol=1e-10)
    assert len(grads) == len(reference_grads)
    for got, want in zip(grads, reference_grads, strict=True):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-10)
