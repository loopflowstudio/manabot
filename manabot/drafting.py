"""Generated public draft episodes, independent of human pool corpora.

`prepare_draft` drives native Rotisserie and Limited authority with a uniform
reference policy. Choices are journaled durably before native commit. Only the
final receipt reveals constructed decks; no partial build becomes another
seat's input. This is software preparation, never a scored training episode.
"""

import os
from pathlib import Path
import random
from time import perf_counter, process_time
from typing import Literal, TextIO

from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, file_sha256
from manabot.infra.source import source_commit, source_dirty
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

    policy: Literal["uniform_legal_definitions"] = "uniform_legal_definitions"
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


class DraftPreparation(DraftSetup):
    schema_version: Literal[1] = 1
    status: Literal["software_validation"] = "software_validation"
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


def prepare_draft(out: Path, seed: int, *, set_code: str = "tla") -> DraftPreparation:
    """Retain one eight-seat 45-pick draft and eight admitted 40-card builds.

    Requires a new output directory, so an interrupted journal cannot be erased
    or mistaken for a completed receipt. A journal row records intent, not a
    successful commit; only the final receipt certifies completion.
    """
    started, cpu_started = perf_counter(), process_time()
    episode_id = f"{set_code}-prepare-{seed}"
    draft = Rotisserie(set_code, episode_id)
    initial = draft.observe()
    # Separate streams keep construction sampling independent of draft work.
    draft_rng = random.Random(seed)
    build_rng = random.Random(f"{seed}:construction")
    limited = Limited()
    out.mkdir(parents=True, exist_ok=False)
    # Bind interrupted journals too, before any pick. Count rows use the
    # existing catalog axis, never a second draft vocabulary.
    atomic_json(out / "catalog.json", managym.catalog_tokens())
    atomic_json(out / "printings.json", managym.format.set_printings(set_code))
    world = managym.world_identity()
    setup = DraftSetup(
        episode_id=episode_id,
        seed=seed,
        set=set_code,
        source_commit=source_commit(),
        source_dirty=source_dirty(),
        native_sha256=file_sha256(_managym.__file__),
        world_version=world["world_version"],
        card_digest=world["card_digest"],
        printing_digest=initial["printing_digest"],
        seats=len(initial["pools"]),
        picks_per_seat=initial["picks_per_seat"],
        build_cap=40,
        initial=tuple(initial["initial"]),
        catalog_sha256=file_sha256(out / "catalog.json"),
        printings_sha256=file_sha256(out / "printings.json"),
    )
    atomic_json(out / "setup.json", setup.model_dump(mode="json"))
    journal_path = out / "choices.jsonl"
    decks: list[managym.Deck] = []
    pools: list[dict[str, int]] = []
    with journal_path.open("x") as stream:
        state = initial
        while (chooser := state["chooser"]) is not None:
            card = draft_rng.choice(state["legal"])
            choice = DraftChoice(
                episode_id=episode_id,
                revision=state["revision"],
                chooser=chooser,
                card_def_id=card,
            )
            _journal(stream, choice)
            draft.pick(episode_id, choice.revision, chooser, card)
            state = draft.observe()
        for seat in range(len(state["pools"])):
            pool = draft.pool(seat)
            pools.append(pool.cards)
            build = limited.build(pool, max_cards=setup.build_cap)
            for revision in range(setup.build_cap):
                # One choice per definition, not weighted by remaining copies.
                card_name = build_rng.choice(sorted(build.available))
                _journal(
                    stream,
                    BuildChoice(
                        episode_id=episode_id,
                        revision=revision,
                        chooser=seat,
                        card=card_name,
                    ),
                )
                build.select(card_name)
            issues = limited.issues(pool, build.deck)
            if not build.can_stop or issues:
                raise ValueError(
                    f"seat {seat}: native Limited admission failed: {issues}"
                )
            sideboard = {
                name: remaining
                for name, count in pool.cards.items()
                if (remaining := count - build.deck.get(name, 0)) > 0
            }
            decks.append(managym.Deck(build.deck, sideboard))
    admission = managym.check_decks(decks)
    if any(admission):
        raise ValueError(f"native world deck admission failed: {admission}")
    receipt = DraftPreparation(
        **setup.model_dump(),
        remaining=tuple(state["remaining"]),
        pools=tuple(pools),
        decks=tuple(decks),
        draft_picks=state["revision"],
        construction_picks=setup.build_cap * len(decks),
        journal_sha256=file_sha256(journal_path),
        parent_cpu_seconds=process_time() - cpu_started,
        wall_seconds=perf_counter() - started,
    )
    atomic_json(out / "receipt.json", receipt.model_dump(mode="json"))
    return receipt
