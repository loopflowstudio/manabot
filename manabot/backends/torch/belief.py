"""Autoregressive joint hand counts without enumerating compatible worlds.

``SamplerInput`` contains only a viewer's public pool, known hand minima and
encoded public history. Actual hands enter ``log_prob`` as supervision only.
The decoder samples counts in the schema's fixed vocabulary order, retaining
card-count correlations. Its physical-deal base measure matches managym's
``PossibleWorldSpace::from_parts``: remove known cards before drawing unknown
slots. This domain does not represent library order or other hidden state.
"""

from __future__ import annotations

import io
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import torch
from torch import nn

from manabot.belief.sampling_types import SamplerInput, SamplerSchema
from manabot.sim.search.sampling_receipt import (
    HandBatch,
    PreparedHands,
    draw_hand_batch,
)


def _log_choose(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    valid = (k >= 0) & (k <= n)
    # Clamp before lgamma to avoid NaN gradients at masked infeasible counts.
    safe_k = torch.minimum(k.clamp_min(0), n)
    result = (
        torch.lgamma(n + 1) - torch.lgamma(safe_k + 1) - torch.lgamma(n - safe_k + 1)
    )
    return result.masked_fill(~valid, -torch.inf)


def _base_logits(
    available: torch.Tensor,
    later: torch.Tensor,
    remaining: torch.Tensor,
    max_count: int,
) -> torch.Tensor:
    counts = torch.arange(max_count + 1, device=available.device, dtype=torch.float64)[
        None, :
    ]
    available = available[:, None].to(torch.float64)
    later = later[:, None].to(torch.float64)
    return _log_choose(available, counts) + _log_choose(
        later, remaining[:, None] - counts
    )


def _constraints(
    inputs: Sequence[SamplerInput], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not inputs or len({len(row.pool_counts) for row in inputs}) != 1:
        raise ValueError("a nonempty batch with one vocabulary width is required")
    if len({row.vocabulary_identity for row in inputs}) != 1:
        raise ValueError("batch mixes vocabulary identities")
    pools = torch.tensor(
        [row.pool_counts for row in inputs], device=device, dtype=torch.int64
    )
    minima = torch.tensor(
        [row.known_minima for row in inputs], device=device, dtype=torch.int64
    )
    remaining = torch.tensor(
        [row.hand_size for row in inputs], device=device, dtype=torch.int64
    ) - minima.sum(dim=1)
    return pools - minima, minima, remaining


def _validate_targets(
    targets: torch.Tensor,
    residual: torch.Tensor,
    minima: torch.Tensor,
    remaining: torch.Tensor,
) -> torch.Tensor:
    if targets.shape != minima.shape or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("hand labels must be integer [batch, vocabulary] counts")
    unknown = targets.to(minima.device) - minima
    if bool(((unknown < 0) | (unknown > residual)).any()) or not torch.equal(
        unknown.sum(dim=1), remaining
    ):
        raise ValueError("hand label violates the public compatible-deal constraints")
    return unknown


def physical_deal_log_prob(
    inputs: Sequence[SamplerInput], hand_counts: torch.Tensor
) -> torch.Tensor:
    """Exact joint log probability of physical deals, not uniform count vectors."""
    residual, minima, remaining = _constraints(inputs, hand_counts.device)
    unknown = _validate_targets(hand_counts, residual, minima, remaining)
    return _log_choose(residual.double(), unknown.double()).sum(dim=1) - _log_choose(
        residual.sum(dim=1).double(), remaining.double()
    )


@torch.no_grad()
def sample_physical_deal(
    inputs: Sequence[SamplerInput], *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample the safe card-removal baseline on CPU in O(cards × max count)."""
    residual, minima, remaining = _constraints(inputs, torch.device("cpu"))
    # Empty hidden pools have one physical deal: the empty hand.
    if residual.shape[1] == 0:
        return minima
    counts: list[torch.Tensor] = []
    max_count = int(residual.max().item())
    later = residual.sum(dim=1)
    for column in range(residual.shape[1]):
        later = later - residual[:, column]
        probs = _base_logits(residual[:, column], later, remaining, max_count).softmax(
            dim=1
        )
        count = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        counts.append(count)
        remaining = remaining - count
    return torch.stack(counts, dim=1) + minima


class AutoregressiveBeliefSampler(nn.Module):
    """Learn a normalized joint posterior as corrections to physical deal odds.

    Training minimizes mean negative ``log_prob`` of private labels. Gradients
    flow through every teacher-forced conditional correction, not labels, public
    constraints or physical odds. Inference uses only preceding sampled counts.
    History dropout removes the entire history vector per training example; it
    is a treatment, not a guarantee of calibration on foreign opponents.
    """

    def __init__(
        self, schema: SamplerSchema, hidden_size: int = 64, history_dropout: float = 0.0
    ) -> None:
        super().__init__()
        if hidden_size < 1 or not 0 <= history_dropout <= 1:
            raise ValueError("invalid hidden size or history dropout")
        self.schema = schema
        self.hidden_size = hidden_size
        self.history_dropout = history_dropout
        self.context = nn.Linear(
            2 * len(schema.card_names) + 1 + schema.history_size, hidden_size
        )
        self.card = nn.Embedding(len(schema.card_names), hidden_size)
        self.recurrent = nn.GRUCell(hidden_size + 1, hidden_size)
        self.correction = nn.Linear(hidden_size, schema.max_count + 1)
        # Initial and explicit baseline inference both preserve physical odds.
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def decode(
        self,
        inputs: Sequence[SamplerInput],
        targets: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.context.weight.device
        for row in inputs:
            row.validate(self.schema)
        residual, minima, remaining = _constraints(inputs, device)
        unknown = (
            None
            if targets is None
            else _validate_targets(targets, residual, minima, remaining)
        )
        history = torch.tensor(
            [row.history_features for row in inputs], device=device, dtype=torch.float32
        )
        if self.training and self.history_dropout:
            history = history * (
                torch.rand((len(inputs), 1), device=device) >= self.history_dropout
            )
        context = torch.cat(
            (residual.float(), minima.float(), remaining[:, None].float(), history),
            dim=1,
        )
        hidden = torch.tanh(self.context(context))
        previous = torch.zeros((len(inputs), 1), device=device)
        log_prob = torch.zeros(len(inputs), dtype=torch.float64, device=device)
        counts: list[torch.Tensor] = []
        later = residual.sum(dim=1)
        for column in range(len(self.schema.card_names)):
            token = self.card.weight[column].expand(len(inputs), -1)
            hidden = self.recurrent(torch.cat((token, previous), dim=1), hidden)
            later = later - residual[:, column]
            logits = (
                _base_logits(
                    residual[:, column], later, remaining, self.schema.max_count
                )
                + self.correction(hidden).double()
            )
            conditional = logits.log_softmax(dim=1)
            count = (
                torch.multinomial(conditional.exp(), 1, generator=generator).squeeze(1)
                if unknown is None
                else unknown[:, column]
            )
            log_prob = log_prob + conditional.gather(1, count[:, None]).squeeze(1)
            counts.append(count)
            remaining = remaining - count
            previous = count[:, None].float() / self.schema.max_count
        hand = torch.stack(counts, dim=1) + minima if targets is None else targets
        return hand, log_prob

    def log_prob(
        self, inputs: Sequence[SamplerInput], hand_counts: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced joint log probabilities [B], labels [B, C]."""
        return self.decode(inputs, hand_counts, None)[1]

    @torch.no_grad()
    def sample(
        self,
        inputs: Sequence[SamplerInput],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return legal full hand counts [B, C]; call eval() to disable dropout."""
        return self.decode(inputs, None, generator)[0]

    @torch.inference_mode()
    def sample_hands(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        generator = torch.Generator().manual_seed(seed)

        def draw() -> tuple[list[int], float]:
            counts, logs = self.decode([prepared.inputs], None, generator)
            return counts[0].tolist(), float(logs[0])

        return draw_hand_batch(prepared, count, seed, check, draw)


class PhysicalDistribution:
    @torch.inference_mode()
    def sample_hands(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        inputs = [prepared.inputs]
        generator = torch.Generator().manual_seed(seed)

        def draw() -> tuple[list[int], float]:
            counts = sample_physical_deal(inputs, generator=generator)
            return counts[0].tolist(), float(physical_deal_log_prob(inputs, counts)[0])

        return draw_hand_batch(prepared, count, seed, check, draw)


_RECURRENT_NAMES = {
    "recurrent.weight_ih": "recurrent.input_weight",
    "recurrent.weight_hh": "recurrent.hidden_weight",
    "recurrent.bias_ih": "recurrent.input_bias",
    "recurrent.bias_hh": "recurrent.hidden_bias",
}


def canonical_sampler_parameters(
    state: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    return {
        _RECURRENT_NAMES.get(k, k): value.detach().cpu().numpy()
        for k, value in state.items()
    }


def sampler_parameters(model: object) -> dict[str, np.ndarray]:
    return canonical_sampler_parameters(
        cast(AutoregressiveBeliefSampler, model).state_dict()
    )


def decode_legacy_sampler(data: bytes) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("invalid legacy sampler checkpoint")
    payload["state_dict"] = canonical_sampler_parameters(payload["state_dict"])
    return payload


def restore_sampler(
    schema: SamplerSchema,
    hidden_size: int,
    history_dropout: float,
    parameters: Mapping[str, np.ndarray],
) -> AutoregressiveBeliefSampler:
    model = AutoregressiveBeliefSampler(schema, hidden_size, history_dropout)
    names = {value: key for key, value in _RECURRENT_NAMES.items()}
    model.load_state_dict(
        {names.get(k, k): torch.from_numpy(v) for k, v in parameters.items()},
        strict=True,
    )
    model.eval()
    return model


def freeze_sampler(model: object) -> None:
    cast(AutoregressiveBeliefSampler, model).requires_grad_(False)
