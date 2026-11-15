"""Admission and host projection of public drafting and local construction.

The catalog axis is native CardDefId. Active definitions are gathered once here;
backends receive only Prepared rows. Other seats' partial decks never enter this
input. Tokens expose the same integer state as the numerical projection.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from manabot.policy.prepare import Prepared
from managym.format import Limited

if TYPE_CHECKING:
    from managym.format import RotisserieObservation


@dataclass(frozen=True)
class DraftPrepared[A](Prepared[A]):
    cards: A  # [C] native definition ids, shared across the batch
    features: (
        A  # [B,C, 4 + seats]: initial, remaining, relative pools, own deck, legal count
    )
    position: A  # [B,7]: phase, revision, round, chooser, seats, picks/seat, build cap
    legal: A  # [B,C] exact native mask
    binding: str


@dataclass(frozen=True)
class DraftInput:
    phase: Literal["draft", "build"]
    state: "RotisserieObservation"
    chooser: int
    deck: tuple[int, ...] = ()
    build_cap: int = 40

    def tokens(self) -> dict[str, object]:
        """Public state, plus only this chooser's construction so far."""
        return {
            "phase": self.phase,
            "state": self.state,
            "chooser": self.chooser,
            "deck": list(self.deck),
            "build_cap": self.build_cap,
        }


def prepare_draft(
    inputs: list[DraftInput], cards: np.ndarray, binding: str
) -> DraftPrepared[np.ndarray]:
    """Validate integer counts and native legality before any framework forward."""
    if not inputs or cards.ndim != 1 or len(set(cards.tolist())) != len(cards):
        raise ValueError("nonempty draft batch and unique card axis required")
    features, positions, masks = [], [], []
    limited = Limited()
    for item in inputs:
        state = item.state
        initial = np.asarray(state["initial"], dtype=np.int32)
        remaining = np.asarray(state["remaining"], dtype=np.int32)
        pools = np.asarray(state["pools"], dtype=np.int32)
        seats = len(pools)
        if (
            not 0 <= item.chooser < seats
            or pools.shape != (seats, len(initial))
            or remaining.shape != initial.shape
        ):
            raise ValueError("invalid public draft axes or chooser")
        if (
            np.any(initial < 0)
            or np.any(remaining < 0)
            or np.any(pools < 0)
            or not np.array_equal(initial, remaining + pools.sum(0))
        ):
            raise ValueError("draft supply conservation failed")
        if np.any(cards < 0) or np.any(cards >= len(initial)):
            raise ValueError("draft card axis outside native catalog")
        if int(pools.sum()) != state["revision"]:
            raise ValueError("draft revision differs from committed pools")
        deck = (
            np.zeros_like(initial)
            if not item.deck
            else np.asarray(item.deck, dtype=np.int32)
        )
        if deck.shape != initial.shape or np.any(deck < 0):
            raise ValueError("invalid construction counts")
        if item.phase == "draft":
            round_index = state["revision"] // seats
            expected = state["revision"] % seats
            if round_index % 2:
                expected = seats - 1 - expected
            if (
                state["chooser"] != item.chooser
                or expected != item.chooser
                or np.any(deck)
            ):
                raise ValueError("draft chooser/order or partial deck differs")
            available = remaining
            if set(state["legal"]) != set(np.flatnonzero(available).tolist()):
                raise ValueError("draft mask differs from native supply")
        else:
            if (
                state["chooser"] is not None
                or state["revision"] != seats * state["picks_per_seat"]
            ):
                raise ValueError("construction requires a completed draft")
            available = limited.available(
                pools[item.chooser : item.chooser + 1],
                deck[None],
                max_cards=item.build_cap,
            )[0][0]
        if set(np.flatnonzero(available).tolist()) - set(cards.tolist()):
            raise ValueError("policy card axis omits legal choices")
        if not np.any(available[cards] > 0):
            raise ValueError("no legal draft/construction action")
        relative = np.roll(pools, -item.chooser, axis=0).T
        features.append(
            np.column_stack((initial, remaining, relative, deck, available))[cards]
        )
        positions.append(
            [
                int(item.phase == "build"),
                state["revision"],
                state["revision"] // seats,
                item.chooser,
                seats,
                state["picks_per_seat"],
                item.build_cap,
            ]
        )
        masks.append(available[cards] > 0)
    return DraftPrepared(
        cards.copy(),
        np.asarray(features, np.float32),
        np.asarray(positions, np.float32),
        np.asarray(masks),
        binding,
    )


@dataclass(frozen=True)
class DraftTrajectory:
    seat: int
    inputs: tuple[DraftPrepared[np.ndarray], ...]
    actions: tuple[int, ...]  # positions on Prepared.cards, never native ids
