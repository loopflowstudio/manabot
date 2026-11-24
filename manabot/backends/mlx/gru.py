"""Differentiable packed GRU schedules for catalogs, records and decision memory.

`PackedGRU.run` gathers and projects every token once; `gru_recurrence` owns the
steps and their hand-written gradient.
"""

from collections.abc import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _gru_cell(
    x: mx.array, previous: mx.array, recurrent_weight: mx.array, hidden_bias: mx.array
) -> mx.array:
    """One GRU step: `x` [L,3H] (input projection with bias), `previous` [L,H]."""
    hidden = previous.shape[1]
    recurrent = previous @ recurrent_weight.T + hidden_bias
    gates = mx.sigmoid(x[:, : 2 * hidden] + recurrent[:, : 2 * hidden])
    reset, update = gates[:, :hidden], gates[:, hidden:]
    new = mx.tanh(x[:, 2 * hidden :] + reset * recurrent[:, 2 * hidden :])
    return (1 - update) * new + update * previous


def _gru_cell_backward(
    x: mx.array,
    previous: mx.array,
    recurrent_weight: mx.array,
    hidden_bias: mx.array,
    upstream: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reverse one `_gru_cell` step from its inputs and dL/dh' `upstream` [L,H]
    -> (dL/dx [L,3H], dL/d(recurrent pre-activation) [L,3H], dL/dprevious [L,H]).

    The gates are recomputed rather than saved: the forward keeps only states.
    The recurrent gradient differs from dL/dx only in its new-gate third, which
    the reset gate scales.
    """
    hidden = previous.shape[1]
    recurrent = previous @ recurrent_weight.T + hidden_bias
    gates = mx.sigmoid(x[:, : 2 * hidden] + recurrent[:, : 2 * hidden])
    reset, update = gates[:, :hidden], gates[:, hidden:]
    recurrent_new = recurrent[:, 2 * hidden :]
    new = mx.tanh(x[:, 2 * hidden :] + reset * recurrent_new)
    new_pre = upstream * (1 - update) * (1 - new * new)
    reset_pre = new_pre * recurrent_new * reset * (1 - reset)
    update_pre = upstream * (previous - new) * update * (1 - update)
    dx = mx.concatenate([reset_pre, update_pre, new_pre], axis=1)
    drecurrent = mx.concatenate([reset_pre, update_pre, new_pre * reset], axis=1)
    dprevious = upstream * update + drecurrent @ recurrent_weight
    return dx, drecurrent, dprevious


# `mx.compile` fuses each step's elementwise gate arithmetic; it traces once per
# live-row count and reuses it across schedules.
_compiled_gru_cell = mx.compile(_gru_cell)
_compiled_gru_cell_backward = mx.compile(_gru_cell_backward)


def gru_recurrence(sizes: Sequence[int]) -> Callable[..., mx.array]:
    """The recurrence of a packed schedule with live counts `sizes`, as one
    differentiable function (projected [N,3H], Wh [3H,H], hidden_bias [3H],
    initial state [sizes[0],H]) -> every step's state [N,H] in packed order.

    Autograd never sees the per-step slices: MLX gives every slice output a
    whole-input zero cotangent, which makes a traced backward quadratic in the
    number of steps. The hand-written VJP runs the steps in reverse with a
    compiled cell, concatenates the input-projection gradient once and forms
    dWh and the hidden-bias gradient as one matmul and one sum over every
    packed row. The initial state receives step 0's previous-state gradient.
    """
    offsets = np.concatenate(([0], np.cumsum(sizes))).tolist()
    total = offsets[-1]
    # Packed row of each row's previous state; step 0 reads the initial state,
    # appended after the outputs.
    previous_rows = mx.array(
        np.concatenate(
            [total + np.arange(sizes[0])]
            + [
                offsets[step - 1] + np.arange(sizes[step])
                for step in range(1, len(sizes))
            ]
        ).astype(np.int32)
    )

    @mx.custom_function
    def recurrence(
        projected: mx.array,
        recurrent_weight: mx.array,
        hidden_bias: mx.array,
        initial: mx.array,
    ) -> mx.array:
        state = initial
        outputs: list[mx.array] = []
        for step, live in enumerate(sizes):
            x = projected[offsets[step] : offsets[step] + live]
            state = _compiled_gru_cell(x, state[:live], recurrent_weight, hidden_bias)
            outputs.append(state)
        return mx.concatenate(outputs)

    @recurrence.vjp
    def recurrence_vjp(
        primals: tuple[mx.array, mx.array, mx.array, mx.array],
        cotangent: mx.array,
        output: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        projected, recurrent_weight, hidden_bias, initial = primals
        previous = mx.take(
            mx.concatenate([output, initial.astype(output.dtype)]),
            previous_rows,
            axis=0,
        )
        dx: list[mx.array] = []
        drecurrent: list[mx.array] = []
        carried: mx.array | None = None
        for step in reversed(range(len(sizes))):
            start, live = offsets[step], sizes[step]
            upstream = cotangent[start : start + live]
            if carried is not None:
                pad = live - carried.shape[0]
                upstream = upstream + (
                    mx.pad(carried, [(0, pad), (0, 0)]) if pad else carried
                )
            step_dx, step_drecurrent, carried = _compiled_gru_cell_backward(
                projected[start : start + live],
                previous[start : start + live],
                recurrent_weight,
                hidden_bias,
                upstream,
            )
            dx.append(step_dx)
            drecurrent.append(step_drecurrent)
        assert carried is not None  # every schedule has a first step
        all_drecurrent = mx.concatenate(drecurrent[::-1])
        return (
            mx.concatenate(dx[::-1]),
            all_drecurrent.T @ previous,
            all_drecurrent.sum(axis=0),
            carried.astype(initial.dtype),
        )

    return recurrence


class PackedGRU(nn.Module):
    """A one-layer GRU over packed variable-length sequences (`Packed`).

    Gates are reset, update, new, as in PyTorch and `mlx.nn.GRU`:
    r, z = sigmoid(x Wx[:2H]^T + input_bias[:2H] + h Wh[:2H]^T + hidden_bias[:2H]),
    n = tanh(x Wx[2H:]^T + input_bias[2H:] + r * (h Wh[2H:]^T + hidden_bias[2H:])),
    h' = (1 - z) n + z h. An omitted initial state is zero.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        bound = hidden_size**-0.5
        self.Wx = mx.random.uniform(-bound, bound, (3 * hidden_size, input_size))
        self.Wh = mx.random.uniform(-bound, bound, (3 * hidden_size, hidden_size))
        self.input_bias = mx.zeros((3 * hidden_size,))
        self.hidden_bias = mx.zeros((3 * hidden_size,))

    def step(self, inputs: mx.array, state: mx.array) -> mx.array:
        return _compiled_gru_cell(
            inputs @ self.Wx.T + self.input_bias, state, self.Wh, self.hidden_bias
        )

    def run(
        self,
        table: mx.array,
        rows: mx.array,
        sizes: Sequence[int],
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Packed row p reads row `rows[p]` of `table` [T,D]; step t holds the
        `sizes[t]` live sequences, in sorted order, starting from `state`
        [sizes[0],H] -> (every step's state [N,H] in packed order, each
        sequence's final state [S,H] in sorted order).
        """
        projected = mx.take(table, rows, axis=0) @ self.Wx.T + self.input_bias
        if state is None:
            state = mx.zeros((sizes[0], self.hidden_size), dtype=projected.dtype)
        states = gru_recurrence(sizes)(
            projected, self.Wh, self.hidden_bias, state.astype(projected.dtype)
        )
        return states, mx.take(states, mx.array(final_rows(sizes)), axis=0)


def final_rows(sizes: Sequence[int]) -> np.ndarray:
    """Packed row of each sorted sequence's last step: sequence s is live in the
    steps whose size exceeds s, a prefix because sizes never increase."""
    counts = np.asarray(sizes)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sequences = np.arange(counts[0])
    lengths = np.searchsorted(-counts, -sequences, side="left")
    return (offsets[lengths - 1] + sequences).astype(np.int32)
