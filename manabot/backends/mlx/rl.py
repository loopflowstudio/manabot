"""The damped update's loss terms over MLX arrays.

Each function is the MLX form of its PyTorch counterpart in
`manabot.backends.torch.rl`, term for term, so
the parity check can compare them on the same minibatch. Rows are minibatch
samples [B]; distributions are over action slots [B,A] with `valid` marking
the legal support.
"""

import mlx.core as mx


def masked_log_softmax(logits: mx.array, valid: mx.array) -> mx.array:
    """Log-probabilities on the legal support, -inf elsewhere."""
    masked = mx.where(valid, logits, -mx.inf)
    return masked - mx.logsumexp(masked, axis=-1, keepdims=True)


def reverse_kl(logs: mx.array, target: mx.array, valid: mx.array) -> mx.array:
    """KL(current || target) per row from log-probabilities, zero on padding
    without 0 * -inf."""
    difference = mx.where(valid, logs, 0.0) - mx.where(valid, target, 0.0)
    return (mx.exp(logs) * difference).sum(-1)


def clipped_surrogate(
    logs: mx.array,
    behavior: mx.array,
    actions: mx.array,
    advantages: mx.array,
    clip: float,
) -> mx.array:
    """PPO's clipped likelihood-ratio surrogate per row, as a loss.

    `actions` [B,1] index the sampled slot. PyTorch evaluates the ratio in
    float64 so exp(log ratio) cannot overflow; MLX's GPU has no float64, so the
    ratio is float32 here. The two differ only for |log ratio| > 88, which no
    sampled action reaches in practice; the parity check would fail first.
    """
    ratio = mx.exp(
        mx.take_along_axis(logs, actions, axis=-1)[:, 0]
        - mx.take_along_axis(behavior, actions, axis=-1)[:, 0]
    )
    return -mx.minimum(
        ratio * advantages, mx.clip(ratio, 1 - clip, 1 + clip) * advantages
    )


def selected_mean(rows: mx.array, selected: mx.array) -> mx.array:
    """Mean over the selected rows (float mask [B]); zero when none is."""
    return (rows * selected).sum() / mx.maximum(selected.sum(), 1)


def squared_error(values: mx.array, targets: mx.array) -> mx.array:
    """The scalar critic's mean squared error."""
    return mx.square(values - targets).mean()
