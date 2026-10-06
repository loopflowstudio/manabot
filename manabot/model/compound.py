"""Trainable ragged compound policies over one authoritative offer projection.

`CompoundDecoder` samples an offer, then includes/excludes each candidate in
canonical order. Native support gives each unordered subset exactly one path.
The GRU conditions on the chosen prefix; decoding never advances the engine.
`CompoundOutput` retains differentiable conditional log probabilities and prefix
values for grouped or sequential credit. Engine IDs only route submissions;
the selected feature architecture consumes labels or typed visible object joins.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from manabot.sim.structured_policy import (
    ChoiceFeatures,
    DecodedSubmission,
    RaggedOfferBatch,
    StructuredPolicyError,
)


@dataclass(frozen=True)
class CompoundOutput:
    submission: DecodedSubmission
    tokens: tuple[int, ...]
    log_probs: Tensor  # [factors], including zero-log-probability forced factors
    values: Tensor  # [factors], acting-seat expected terminal return
    probabilities: tuple[Tensor, ...]  # each [legal support including masked entries]
    end_value: Tensor  # signed value after the complete token prefix
    projection_fingerprint: str

    @property
    def log_prob(self) -> Tensor:
        return self.log_probs.sum()


class CompoundDecoder(nn.Module):
    """Uncapped recurrent decoder, with no embedding indexed by physical IDs."""

    def __init__(self, hidden_dim: int, *, object_features: bool = False) -> None:
        super().__init__()
        self.labels = nn.Embedding(256, hidden_dim)
        self.prefix = nn.GRUCell(hidden_dim, hidden_dim)
        self.choice = nn.Embedding(2, hidden_dim)
        self.offer_score = nn.Linear(hidden_dim, 1)
        self.include_score = nn.Linear(hidden_dim, 2)
        self.value = nn.Linear(hidden_dim, 1)
        self.position = nn.Linear(3, hidden_dim)
        self.parameters_projection = (
            nn.Linear(10, hidden_dim) if object_features else None
        )

    def _features(self, features: ChoiceFeatures, objects: Tensor) -> Tensor:
        if self.parameters_projection is None:
            raise ValueError("typed features require the object decoder")
        numeric = objects.new_zeros(10)
        if features.mana:
            numeric[:7] = objects.new_tensor(features.mana) / 10
        if features.ordinal is not None:
            numeric[7] = math.log1p(features.ordinal)
            numeric[8] = 1
        if features.attack is not None:
            numeric[9] = 1 if features.attack else -1
        value = self.parameters_projection(numeric)
        for row in features.objects:
            if row < 0 or row >= objects.shape[0]:
                raise ValueError("choice object row exceeds encoded capacity")
            value = value + objects[row]
        return value

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
        objects: Tensor | None = None,
        tokens: tuple[int, ...] | None = None,
        tape: CompoundOutput | None = None,
        prefix: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> CompoundOutput:
        """Sample, teacher-force a complete tape, or complete a forced prefix.

        `context` is [hidden_dim]. Gradients flow through context, all prefix
        states, and normalized conditional logits. Sampled discrete tokens are
        constants during recomputation, as required by the score-function loss.
        """
        fingerprint = batch.fingerprint
        if tape is not None:
            if tape.projection_fingerprint != fingerprint:
                raise StructuredPolicyError(
                    "compound tape belongs to another root projection"
                )
            if tokens is not None:
                raise ValueError("provide a tape or tokens, not both")
            tokens = tape.tokens
        if tokens is not None and prefix:
            raise ValueError("provide a complete tape or prefix, not both")
        if context.ndim != 1:
            raise ValueError("compound context must be one hidden vector")
        state = context
        selected_tokens: list[int] = []
        log_probs: list[Tensor] = []
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
            elif ordinal < len(prefix):
                token = prefix[ordinal]
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
            probabilities.append(distribution.probs)
            values.append(self.value(state).squeeze(-1))
            return token

        if self.parameters_projection is not None:
            if objects is None or len(batch.offer_inputs) != len(batch.offers):
                raise ValueError("object decoder requires bound visible choice inputs")
            # Native verbs distinguish actions on the same object. Presentation
            # labels never supply semantics to this architecture.
            offer_rows = torch.stack(
                [
                    self._label(str(offer["verb"]), context.device)
                    + self._features(row, objects)
                    for offer, row in zip(batch.offers, batch.offer_inputs, strict=True)
                ]
            )
        else:
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
        for role_index in range(start, stop):
            row = batch.choices[role_index]
            if self.parameters_projection is not None:
                assert objects is not None
                state = self.prefix(
                    self._features(batch.role_inputs[role_index], objects), state
                )
            selected: list[int] = []
            count = row.candidate_stop - row.candidate_start
            for ordinal, index in enumerate(
                range(row.candidate_start, row.candidate_stop)
            ):
                candidate = batch.candidates[index]
                if self.parameters_projection is not None:
                    assert objects is not None
                    feature = self._features(batch.candidate_inputs[index], objects)
                else:
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
                allowed = torch.tensor(
                    batch.support.alternatives(selected_tokens),
                    device=context.device,
                )
                take = choose(logits.masked_fill(~allowed, -torch.inf))
                if take:
                    selected.append(int(candidate["id"]))
                state = self.prefix(self.choice.weight[take], state)
            answers.append(
                {"kind": "candidates", "role": row.role, "candidates": selected}
            )
        if len(prefix) > len(selected_tokens):
            raise StructuredPolicyError("compound prefix has trailing choices")
        if tokens is not None and len(tokens) != len(selected_tokens):
            raise StructuredPolicyError("compound token tape has trailing choices")
        batch.support.submission_json(selected_tokens)
        return CompoundOutput(
            DecodedSubmission(int(batch.offers[offer_index]["id"]), tuple(answers)),
            tuple(selected_tokens),
            torch.stack(log_probs),
            torch.stack(values),
            tuple(probabilities),
            self.value(state).squeeze(-1),
            fingerprint,
        )
