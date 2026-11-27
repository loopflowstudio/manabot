"""Packed GRU parity, parameter-dtype contract, and backend ownership invariant."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from manabot.backends.torch.packed_gru import hand_packed_gru, packed_gru

ROOT = Path(__file__).resolve().parents[2]

# Shrinking live counts, including several sequences finishing at one step.
SIZES = [5, 5, 4, 2, 2, 1, 1, 1]


def cast_gru_output(
    dtype: torch.dtype,
) -> Callable[[nn.Module, tuple[object, ...], object], object]:
    """Forward hook that casts a packed `nn.GRU` return value without detaching."""

    def hook(module: nn.Module, args: tuple[object, ...], result: object) -> object:
        assert isinstance(result, tuple) and len(result) == 2
        output, hidden = result
        assert isinstance(output, PackedSequence) and isinstance(hidden, torch.Tensor)
        return (
            PackedSequence(
                output.data.to(dtype),
                output.batch_sizes,
                output.sorted_indices,
                output.unsorted_indices,
            ),
            hidden.to(dtype),
        )

    return hook


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_initial", [False, True])
def test_packed_gru_normalizes_reduced_kernel_outputs_with_gradients(
    dtype: torch.dtype, with_initial: bool
) -> None:
    torch.manual_seed(17)
    gru = nn.GRU(6, 7, batch_first=True)
    data = torch.randn(sum(SIZES), 6, requires_grad=True)
    initial = torch.randn(SIZES[0], 7, requires_grad=True) if with_initial else None
    hook = gru.register_forward_hook(cast_gru_output(dtype))
    try:
        outputs, final = packed_gru(gru, data, SIZES, initial)
    finally:
        hook.remove()
    assert outputs.dtype == torch.float32
    assert final.dtype == torch.float32
    (outputs.square().sum() + final.square().sum()).backward()
    for param in gru.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert data.grad.abs().sum() > 0
    if initial is not None:
        assert initial.grad is not None and torch.isfinite(initial.grad).all()
        assert initial.grad.abs().sum() > 0


@pytest.mark.parametrize("with_initial", [False, True])
def test_packed_gru_disables_autocast_and_returns_parameter_dtype(
    with_initial: bool,
) -> None:
    torch.manual_seed(19)
    gru = nn.GRU(6, 7, batch_first=True)
    data = torch.randn(sum(SIZES), 6, dtype=torch.bfloat16, requires_grad=True)
    initial = (
        torch.randn(SIZES[0], 7, dtype=torch.bfloat16, requires_grad=True)
        if with_initial
        else None
    )
    seen: list[tuple[bool, torch.dtype, torch.dtype | None]] = []

    def observe(module: nn.Module, args: tuple[object, ...]) -> None:
        packed = args[0]
        assert isinstance(packed, PackedSequence)
        h0 = args[1] if len(args) > 1 else None
        assert h0 is None or isinstance(h0, torch.Tensor)
        seen.append(
            (
                torch.is_autocast_enabled("cpu"),
                packed.data.dtype,
                None if h0 is None else h0.dtype,
            )
        )

    hook = gru.register_forward_pre_hook(observe)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs, final = packed_gru(gru, data, SIZES, initial)
    finally:
        hook.remove()
    assert seen == [(False, torch.float32, torch.float32 if with_initial else None)]
    assert outputs.dtype == torch.float32
    assert final.dtype == torch.float32
    (outputs.square().sum() + final.square().sum()).backward()
    for param in gru.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert data.grad.abs().sum() > 0
    if initial is not None:
        assert initial.grad is not None and torch.isfinite(initial.grad).all()
        assert initial.grad.abs().sum() > 0


def test_only_packed_gru_references_packed_sequence_in_torch_backend() -> None:
    torch_dir = ROOT / "manabot/backends/torch"
    offenders = sorted(
        str(path.relative_to(ROOT))
        for path in torch_dir.rglob("*.py")
        if path.name != "packed_gru.py"
        and "PackedSequence" in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"Every packed GRU in manabot/backends/torch must go through packed_gru: {offenders}"
    )
