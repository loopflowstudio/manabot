"""Trainable ragged compound policies over one authoritative offer projection.

`CompoundDecoder` samples an offer, then includes/excludes each candidate in
canonical order. Cardinality masks give each unordered subset exactly one path.
The GRU conditions on the chosen prefix; decoding never advances the engine.
`CompoundOutput` retains differentiable conditional log probabilities and prefix
values for grouped or sequential credit. Engine IDs only route submissions;
public labels/verbs and viewer observation embeddings supply model features.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from manabot.sim.structured_policy import (
    DecodedSubmission,
    RaggedOfferBatch,
    StructuredPolicyError,
)


@dataclass(frozen=True)
class CompoundOutput:
    submission: DecodedSubmission
    tokens: tuple[int, ...]
    log_probs: Tensor  # [factors], including zero-log-probability forced factors
    entropies: (
        Tensor  # conditional entropy at each sampled prefix, not exact joint entropy
    )
    values: Tensor  # [factors], acting-seat expected terminal return
    probabilities: tuple[Tensor, ...]  # each [legal support including masked entries]

    @property
    def log_prob(self) -> Tensor:
        return self.log_probs.sum()


class CompoundDecoder(nn.Module):
    """Uncapped recurrent decoder, with no embedding indexed by physical IDs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.labels = nn.Embedding(256, hidden_dim)
        self.prefix = nn.GRUCell(hidden_dim, hidden_dim)
        self.choice = nn.Embedding(2, hidden_dim)
        self.offer_score = nn.Linear(hidden_dim, 1)
        self.include_score = nn.Linear(hidden_dim, 2)
        self.value = nn.Linear(hidden_dim, 1)
        self.position = nn.Linear(3, hidden_dim)

    def _label(self, text: str, device: torch.device) -> Tensor:
        # UTF-8 vocabulary is closed and deterministic; public text is an input,
        # never an authority for legality or a substitute for candidate identity.
        tokens = torch.tensor(list(text.encode()) or [0], device=device)
        return self.labels(tokens).mean(0)

    def forward(
        self,
        context: Tensor,
        batch: RaggedOfferBatch,
        *,
        offer_features: Tensor | None = None,
        tokens: tuple[int, ...] | None = None,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> CompoundOutput:
        """Sample or teacher-force one complete submission; reject partial tapes.

        `context` is [hidden_dim]. Gradients flow through context, all prefix
        states, and normalized conditional logits. Sampled discrete tokens are
        constants during recomputation, as required by the score-function loss.
        """
        if context.ndim != 1:
            raise ValueError("compound context must be one hidden vector")
        state = context
        selected_tokens: list[int] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        values: list[Tensor] = []
        probabilities: list[Tensor] = []

        def choose(logits: Tensor) -> int:
            if not torch.isfinite(logits).any():
                raise StructuredPolicyError("compound choice has empty support")
            distribution = Categorical(logits=logits)
            ordinal = len(selected_tokens)
            if tokens is not None:
                if ordinal >= len(tokens):
                    raise StructuredPolicyError("interrupted compound token tape")
                token = tokens[ordinal]
            elif deterministic:
                token = int(logits.argmax().item())
            else:
                token = int(
                    torch.multinomial(distribution.probs, 1, generator=generator).item()
                )
            if token < 0 or token >= len(logits) or not torch.isfinite(logits[token]):
                raise StructuredPolicyError("illegal compound token")
            selected_tokens.append(token)
            log_probs.append(
                distribution.log_prob(torch.tensor(token, device=context.device))
            )
            entropies.append(distribution.entropy())
            probabilities.append(distribution.probs)
            values.append(self.value(state).squeeze(-1))
            return token

        offer_rows = torch.stack(
            [
                self._label(f"{offer['verb']} {offer['label']}", context.device)
                for offer in batch.offers
            ]
        )
        if offer_features is not None:
            if offer_features.shape != offer_rows.shape:
                raise ValueError("offer features do not align with native offers")
            offer_rows = offer_rows + offer_features
        offer_index = choose(
            self.offer_score(torch.tanh(offer_rows + state)).squeeze(-1)
        )
        state = self.prefix(offer_rows[offer_index], state)
        answers: list[dict[str, object]] = []
        start, stop = batch.choice_offsets[offer_index : offer_index + 2]
        for row in batch.choices[start:stop]:
            selected: list[int] = []
            count = row.candidate_stop - row.candidate_start
            for ordinal, index in enumerate(
                range(row.candidate_start, row.candidate_stop)
            ):
                candidate = batch.candidates[index]
                feature = self._label(str(candidate["label"]), context.device)
                position = context.new_tensor(
                    [
                        ordinal / max(1, count),
                        len(selected) / max(1, count),
                        math.log1p(count),
                    ]
                )
                state = self.prefix(feature + self.position(position), state)
                logits = self.include_score(state)
                remaining = count - ordinal
                allowed = torch.tensor(
                    [
                        len(selected) + remaining - 1 >= row.minimum,
                        len(selected) < row.maximum,
                    ],
                    device=context.device,
                )
                take = choose(logits.masked_fill(~allowed, -torch.inf))
                if take:
                    selected.append(int(candidate["id"]))
                state = self.prefix(self.choice.weight[take], state)
            answers.append(
                {"kind": "candidates", "role": row.role, "candidates": selected}
            )
        if tokens is not None and len(tokens) != len(selected_tokens):
            raise StructuredPolicyError("compound token tape has trailing choices")
        return CompoundOutput(
            DecodedSubmission(int(batch.offers[offer_index]["id"]), tuple(answers)),
            tuple(selected_tokens),
            torch.stack(log_probs),
            torch.stack(entropies),
            torch.stack(values),
            tuple(probabilities),
        )
