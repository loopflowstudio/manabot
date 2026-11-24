"""One-layer GRU over packed variable-length sequences.

`packed_gru` is the entry point for every packed `nn.GRU` call in the policy
encoders. CPU and CUDA run the module's own fused kernel. On MPS `nn.GRU` over a
`PackedSequence` runs step by step, about twenty dispatched ops per step forward
and twenty-seven autograd nodes per step backward; there `hand_packed_gru`
projects every packed row once and leaves only the recurrent linear and the gate
arithmetic inside the step loop. Its backward (`_Recurrence`) recomputes the
gates for all packed rows at once and walks the steps in reverse carrying only
the previous-state gradient, then forms the input-projection, recurrent-weight
and hidden-bias gradients with one concatenation, one matmul and one sum. This
is the design of MLX's `gru_recurrence` (`manabot/backends/mlx/policy.py`).

Both paths read the same `nn.GRU` parameters, so checkpoints do not change, and
both return float32 under MPS autocast as `nn.GRU` does there.

Packed layout (PyTorch's): sequences sorted by descending length; step t holds
`sizes[t]` rows, the prefix of that order, at rows `offsets[t]:offsets[t+1]`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import PackedSequence


def packed_gru(
    gru: nn.GRU,
    data: torch.Tensor,
    sizes: list[int],
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `gru` over packed rows `data` [N,I] with live counts `sizes`.

    `initial` [sizes[0],H] is the state entering step 0 (zeros when omitted).
    Returns every step's state [N,H] in packed order and each sequence's final
    state [sizes[0],H] in sorted order.
    """
    if data.device.type == "mps":
        return hand_packed_gru(gru, data, sizes, initial)
    output, hidden = gru(
        PackedSequence(data, torch.tensor(sizes)),
        None if initial is None else initial.unsqueeze(0),
    )
    return output.data, hidden[0]


def hand_packed_gru(
    gru: nn.GRU,
    data: torch.Tensor,
    sizes: list[int],
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`packed_gru` with the input projection hoisted and a hand-written backward."""
    w_ih, w_hh, b_ih, b_hh = gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0
    with torch.autocast(data.device.type, enabled=False):
        projected = F.linear(data.to(w_ih.dtype), w_ih, b_ih)
        state = (
            projected.new_zeros(sizes[0], gru.hidden_size)
            if initial is None
            else initial.to(projected.dtype)
        )
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        total = int(offsets[-1])
        # Packed row of each row's previous state; step 0 reads the initial
        # state, appended after the N outputs. Then each sequence's last row.
        previous = [np.arange(sizes[0]) + total] + [
            offsets[step - 1] + np.arange(sizes[step]) for step in range(1, len(sizes))
        ]
        following = sizes[1:] + [0]
        final = [
            offsets[step] + np.arange(following[step], sizes[step])
            for step in range(len(sizes))
        ][::-1]
        rows = torch.from_numpy(np.concatenate(previous + final)).to(data.device)
        outputs = _Recurrence.apply(
            projected, w_hh, b_hh, state, rows[:total], sizes
        )
        return outputs, outputs[rows[total:]]


class _Recurrence(torch.autograd.Function):
    """(projected [N,3H], W_hh [3H,H], b_hh [3H], initial [B0,H]) -> states [N,H].

    Gate order is PyTorch's (reset, update, new):
    r, z = sigmoid(x_rz + (W h + b)_rz), n = tanh(x_n + r (W h + b)_n),
    h' = (1 - z) n + z h.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        projected: torch.Tensor,
        w_hh: torch.Tensor,
        b_hh: torch.Tensor,
        initial: torch.Tensor,
        previous_rows: torch.Tensor,
        sizes: list[int],
    ) -> torch.Tensor:
        width = w_hh.shape[1]
        state = initial
        outputs: list[torch.Tensor] = []
        offset = 0
        for live in sizes:
            x = projected[offset : offset + live]
            state = state[:live]
            recurrent = F.linear(state, w_hh, b_hh)
            gates = torch.sigmoid(x[:, : 2 * width] + recurrent[:, : 2 * width])
            new = torch.tanh(
                x[:, 2 * width :] + gates[:, :width] * recurrent[:, 2 * width :]
            )
            state = torch.lerp(new, state, gates[:, width:])
            outputs.append(state)
            offset += live
        result = torch.cat(outputs)
        ctx.save_for_backward(projected, w_hh, b_hh, initial, previous_rows, result)
        ctx.sizes = sizes
        return result

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, upstream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        projected, w_hh, b_hh, initial, previous_rows, result = ctx.saved_tensors
        sizes: list[int] = ctx.sizes
        width = w_hh.shape[1]
        # Recompute every row's gates at once from its previous state.
        previous = torch.cat([result, initial])[previous_rows]
        recurrent = F.linear(previous, w_hh, b_hh)
        gates = torch.sigmoid(projected[:, : 2 * width] + recurrent[:, : 2 * width])
        reset, update = gates[:, :width], gates[:, width:]
        recurrent_new = recurrent[:, 2 * width :]
        new = torch.tanh(projected[:, 2 * width :] + reset * recurrent_new)
        # d(pre-activation)/dh' per gate, so each step's pre-activation gradient
        # is the incoming state gradient times these rows.
        new_scale = (1 - update) * (1 - new * new)
        to_x = torch.cat(
            [
                new_scale * recurrent_new * reset * (1 - reset),
                (previous - new) * update * (1 - update),
                new_scale,
            ],
            dim=1,
        )
        to_recurrent = torch.cat([to_x[:, : 2 * width], new_scale * reset], dim=1)
        # Walk the steps in reverse; only the previous-state gradient is carried.
        offsets = np.concatenate(([0], np.cumsum(sizes))).tolist()
        totals: list[torch.Tensor] = []
        carried: torch.Tensor | None = None
        for step in reversed(range(len(sizes))):
            start, live = offsets[step], sizes[step]
            total = upstream[start : start + live]
            if carried is not None:
                total = total + F.pad(carried, (0, 0, 0, live - carried.shape[0]))
            pre = total.repeat(1, 3) * to_recurrent[start : start + live]
            carried = total * update[start : start + live] + pre @ w_hh
            totals.append(total)
        state_grad = torch.cat(totals[::-1]).repeat(1, 3)
        d_recurrent = state_grad * to_recurrent
        return (
            state_grad * to_x,
            d_recurrent.T @ previous,
            d_recurrent.sum(dim=0),
            carried,
            None,
            None,
        )
