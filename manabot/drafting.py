"""Public draft collection shared by preparation, training and evaluation.

Native Rotisserie and Limited own legality; Prepared projects their public state.
Choices are durable intents before native commit. Only completion reveals all
constructed decks. The uniform preparation uses the same collector as learning.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal, Protocol, TextIO

import numpy as np

from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.infra.source import source_commit, source_dirty
from manabot.policy.draft import (
    DraftInput,
    DraftPrepared,
    DraftTrajectory,
    prepare_draft as prepare_input,
)
import managym
from managym import _managym
from managym.format import Limited, Rotisserie


class DraftChoice(StrictModel):
    phase: Literal["draft"] = "draft"
    episode_id: str
    revision: int
    chooser: int
    card_def_id: int


class BuildChoice(StrictModel):
    phase: Literal["build"] = "build"
    episode_id: str
    revision: int
    chooser: int
    card: str


class DraftSetup(StrictModel):
    """Identity and replay inputs retained before the first choice is committed."""

    binding: str
    roles: tuple[str, ...]
    policy_identities: tuple[str, ...]
    episode_id: str
    seed: int
    set: str
    source_commit: str
    source_dirty: bool
    native_sha256: str
    world_version: str
    card_digest: str
    printing_digest: str
    seats: int
    picks_per_seat: int
    build_cap: int
    initial: tuple[int, ...]
    catalog_sha256: str
    printings_sha256: str


class DraftEpisode(DraftSetup):
    schema_version: Literal[1] = 1
    status: Literal["collected", "software_validation"] = "collected"
    remaining: tuple[int, ...]
    pools: tuple[dict[str, int], ...]
    decks: tuple[managym.Deck, ...]
    draft_picks: int
    construction_picks: int
    journal_sha256: str
    parent_cpu_seconds: float
    wall_seconds: float


def _journal(stream: TextIO, choice: DraftChoice | BuildChoice) -> None:
    stream.write(choice.model_dump_json() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


class DraftActor(Protocol):
    @property
    def identity(self) -> str: ...

    def logits(self, batch: DraftPrepared[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class CollectedDraft:
    episode: DraftEpisode
    trajectories: tuple[DraftTrajectory, ...]


@dataclass(frozen=True)
class DraftCatalog:
    cards: np.ndarray
    names: tuple[str, ...]
    binding: str


def draft_catalog(set_code: str, seats: int, picks: int) -> DraftCatalog:
    state = Rotisserie(set_code, "catalog", seats, picks).observe()
    cards = np.flatnonzero((np.asarray(state["initial"]) > 0) | Limited().basics)
    definitions = json.loads(managym.card_definitions_json())
    binding = canonical_sha256(
        {
            "schema": "rotisserie-policy-1",
            "catalog": managym.catalog_tokens(),
            "printing": state["printing_digest"],
            "world": managym.world_identity(),
            "seats": seats,
            "picks": picks,
            "build_cap": 40,
        }
    )
    return DraftCatalog(cards, tuple(d["name"] for d in definitions), binding)


def collect_draft(
    out: Path,
    episode_id: str,
    seed: int,
    actors: Sequence[DraftActor | None],
    roles: tuple[str, ...],
    catalog: DraftCatalog,
    *,
    set_code: str = "tla",
    picks: int = 45,
    check: Callable[[], None],
) -> CollectedDraft:
    """None is the uniform definition reference; all decisions precede commit."""
    if len(actors) != len(roles):
        raise ValueError("each seat requires an actor and retained role")
    started, cpu_started = perf_counter(), process_time()
    draft = Rotisserie(set_code, episode_id, len(actors), picks)
    state = draft.observe()
    out.mkdir(parents=True, exist_ok=False)
    policy_identities = tuple(
        actor.identity if actor is not None else "uniform-legal-definitions"
        for actor in actors
    )
    atomic_json(out / "catalog.json", managym.catalog_tokens())
    atomic_json(out / "printings.json", managym.format.set_printings(set_code))
    world = managym.world_identity()
    setup = DraftSetup(
        binding=catalog.binding,
        roles=roles,
        policy_identities=policy_identities,
        episode_id=episode_id,
        seed=seed,
        set=set_code,
        source_commit=source_commit(),
        source_dirty=source_dirty(),
        native_sha256=file_sha256(_managym.__file__),
        world_version=world["world_version"],
        card_digest=world["card_digest"],
        printing_digest=state["printing_digest"],
        seats=len(actors),
        picks_per_seat=picks,
        build_cap=40,
        initial=tuple(state["initial"]),
        catalog_sha256=file_sha256(out / "catalog.json"),
        printings_sha256=file_sha256(out / "printings.json"),
    )
    atomic_json(out / "setup.json", setup.model_dump(mode="json"))
    rngs = [
        np.random.default_rng(np.random.SeedSequence([seed, seat]))
        for seat in range(len(actors))
    ]
    batches: list[list[DraftPrepared[np.ndarray]]] = [[] for _ in actors]
    actions: list[list[int]] = [[] for _ in actors]

    def choose(item: DraftInput) -> int:
        check()
        batch = prepare_input([item], catalog.cards, catalog.binding)
        legal = np.flatnonzero(batch.legal[0])
        actor = actors[item.chooser]
        if actor is None:
            selected = int(rngs[item.chooser].choice(legal))
        else:
            logits = actor.logits(batch)[0, legal].astype(np.float64)
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite draft logits")
            weights = np.exp(logits - logits.max())
            selected = int(rngs[item.chooser].choice(legal, p=weights / weights.sum()))
        if roles[item.chooser] == "learner":
            batches[item.chooser].append(batch)
            actions[item.chooser].append(selected)
        return int(catalog.cards[selected])

    decks: list[managym.Deck] = []
    pools: list[dict[str, int]] = []
    limited = Limited()
    with (out / "choices.jsonl").open("x") as stream:
        while (seat := state["chooser"]) is not None:
            card = choose(DraftInput("draft", state, seat))
            _journal(
                stream,
                DraftChoice(
                    episode_id=episode_id,
                    revision=state["revision"],
                    chooser=seat,
                    card_def_id=card,
                ),
            )
            draft.pick(episode_id, state["revision"], seat, card)
            state = draft.observe()
        # Every builder sees only its own partial main deck. Others' finished
        # decks are revealed in the final receipt, never during construction.
        for seat in range(len(actors)):
            pool = draft.pool(seat)
            pools.append(pool.cards)
            build = limited.build(pool, max_cards=40)
            counts = np.zeros(len(catalog.names), np.int32)
            for revision in range(40):
                card = choose(
                    DraftInput("build", state, seat, tuple(int(n) for n in counts))
                )
                name = catalog.names[card]
                _journal(
                    stream,
                    BuildChoice(
                        episode_id=episode_id,
                        revision=revision,
                        chooser=seat,
                        card=name,
                    ),
                )
                build.select(name)
                counts[card] += 1
            if not build.can_stop or limited.issues(pool, build.deck):
                raise ValueError(f"seat {seat}: Limited admission failed")
            sideboard = {
                name: count - build.deck.get(name, 0)
                for name, count in pool.cards.items()
                if count > build.deck.get(name, 0)
            }
            decks.append(managym.Deck(build.deck, sideboard))
    if any(managym.check_decks(decks)):
        raise ValueError("generated decks failed world admission")
    episode = DraftEpisode(
        **setup.model_dump(),
        remaining=tuple(state["remaining"]),
        pools=tuple(pools),
        decks=tuple(decks),
        draft_picks=state["revision"],
        construction_picks=setup.build_cap * len(decks),
        journal_sha256=file_sha256(out / "choices.jsonl"),
        parent_cpu_seconds=process_time() - cpu_started,
        wall_seconds=perf_counter() - started,
    )
    atomic_json(out / "receipt.json", episode.model_dump(mode="json"))
    trajectories = tuple(
        DraftTrajectory(seat, tuple(batches[seat]), tuple(actions[seat]))
        for seat, role in enumerate(roles)
        if role == "learner"
    )
    return CollectedDraft(episode, trajectories)


def prepare_draft(out: Path, seed: int, *, set_code: str = "tla") -> DraftEpisode:
    """Retain an unscored eight-seat draft using the uniform legal reference.

    Output directories must be new; interrupted intents have no final receipt.
    """
    collected = collect_draft(
        out,
        f"{set_code}-prepare-{seed}",
        seed,
        [None] * 8,
        ("uniform",) * 8,
        draft_catalog(set_code, 8, 45),
        set_code=set_code,
        check=lambda: None,
    )
    episode = collected.episode.model_copy(update={"status": "software_validation"})
    atomic_json(out / "receipt.json", episode.model_dump(mode="json"))
    return episode
