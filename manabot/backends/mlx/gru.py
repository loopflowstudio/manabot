"""Differentiable packed GRU schedules for catalogs, records and decision memory."""

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from manabot.policy.prepare import Packed


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


# `mx.compile` fuses each step's elementwise gate arithmetic; it traces once per
# live-row count, which the static catalog schedule repeats on every call.
_compiled_gru_cell = mx.compile(_gru_cell)


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
        self, table: mx.array, steps: Sequence[mx.array], state: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        """Step t reads rows `steps[t]` of `table` [T,D] (its live sequences, in
        sorted order) -> (every step's state [T,H] in packed order, each
        sequence's final state [S,H] in sorted order).

        Each step gathers its inputs from `table` rather than slicing one
        projected array: MLX materializes a whole-input zero cotangent for
        every slice or split output, which makes the backward quadratic in
        the number of steps (15.5 s against 0.39 s at 713 steps on the M2).
        """
        if state is None:
            state = mx.zeros((steps[0].shape[0], self.hidden_size), dtype=table.dtype)
        outputs: list[mx.array] = []
        finished: list[mx.array] = []
        for step, index in enumerate(steps):
            live = index.shape[0]
            state = self.step(mx.take(table, index, axis=0), state[:live])
            outputs.append(state)
            following = steps[step + 1].shape[0] if step + 1 < len(steps) else 0
            if following < live:
                finished.append(state[following:])
        return mx.concatenate(outputs), mx.concatenate(finished[::-1])


def packed_steps(rows: np.ndarray, schedule: Packed[np.ndarray]) -> list[mx.array]:
    """Split packed-order `rows` into the schedule's steps (`PackedGRU.run`)."""
    return [mx.array(part) for part in np.split(rows, np.cumsum(schedule.sizes)[:-1])]
