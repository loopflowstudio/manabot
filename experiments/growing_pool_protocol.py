"""Versioned growing-pool manifests, split admission and regime compilation.

The manifest owns experimental membership, never card legality. Native Deck and
UniformDecks own lists and seeded draws; TrainingRegime owns learning. Integer
tickets express a fixed deck mixture without adding a second runtime sampler.
`check` admits a frozen protocol without playing games or allocating compute.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal

from etude.eval.match import Result
from etude.eval.records import Environment, Match
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.provenance import engine_source_paths, source_bundle_sha256
from manabot.training.decks import UniformDecks
from manabot.training.regime import (
    DampedPolicyLearning,
    Execution,
    TrainingRegime,
    TrainSelfPlay,
)
import managym
from managym import Deck

ROOT = Path(__file__).resolve().parents[1]

type Group = Literal["anchor", "new", "mixed", "test"]
type Anchor = Literal["heuristic", "initial"]
type PanelKind = Literal[
    "anchor", "expanding", "heldout-deck", "heldout-card", "heldout-combination"
]


class Replacement(StrictModel):
    """A reviewed construction constraint, not a second card definition."""

    remove: str
    add: str
    copies: int = Field(ge=1, le=4)


def _substitute(parent: Deck, replacements: tuple[Replacement, ...]) -> Deck:
    """Build an authored variant; sideboard and total main-deck size stay fixed."""
    main = dict(parent.main)
    for change in replacements:
        if change.remove == change.add or main.get(change.remove, 0) < change.copies:
            raise ValueError("replacement must remove available copies")
        main[change.remove] -= change.copies
        if not main[change.remove]:
            del main[change.remove]
        main[change.add] = main.get(change.add, 0) + change.copies
    return Deck(main, dict(parent.sideboard))


class Entry(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    deck: Deck
    group: Group
    tickets: int = Field(ge=0, le=32)
    construction: str = Field(min_length=1)
    interaction_cards: tuple[str, ...] = Field(min_length=1)
    synergy_cards: tuple[str, ...] = Field(min_length=2)
    parent: str | None = None
    replacements: tuple[Replacement, ...] = ()

    @model_validator(mode="after")
    def declared(self) -> Entry:
        if (self.group == "test") != (self.tickets == 0):
            raise ValueError(
                "test entries have zero tickets; training entries need tickets"
            )
        if (self.parent is None) != (not self.replacements):
            raise ValueError("a variant needs both its parent and replacements")
        if self.group == "mixed" and self.parent is None:
            raise ValueError("mixed entries need a reproducible authored construction")
        return self


def _card_names(deck: Deck) -> set[str]:
    return set(deck.main) | set(deck.sideboard)


class _ManaCost(BaseModel):
    # Retain future native cost fields (including X/restrictions) in equality;
    # a construction check must not silently erase them from substitutions.
    model_config = ConfigDict(extra="allow")
    cost: tuple[int, ...] = Field(min_length=7, max_length=7)
    mana_value: int
    hybrid: tuple[tuple[str, str], ...] = ()


class _Types(BaseModel):
    types: tuple[str, ...]


class _ConstructionView(BaseModel):
    """Read-only projection of native definitions for construction checks."""

    name: str
    mana_cost: _ManaCost | None
    types: _Types


class DeckAudit(StrictModel):
    entry: str
    cards: int
    lands: int
    creatures: int
    cheap_spells: int
    interaction: int
    synergy: int


def audit_decks(manifest: Manifest) -> tuple[DeckAudit, ...]:
    """Check this pilot's 40-card construction template, never world eligibility.

    Interaction/synergy annotations are reviewed experimental roles, not inferred
    rules. Native definitions supply types/costs. Variants preserve the entire
    printed mana cost and types of each replaced card, plus the parent mana base.
    These cheap checks do not certify functional draws or competitive strength.
    """
    definitions = TypeAdapter(list[_ConstructionView]).validate_json(
        managym.card_definitions_json()
    )
    catalog = {card.name: card for card in definitions}
    audits = []
    for entry in manifest.entries:
        main = entry.deck.main
        rows = {name: catalog[name] for name in main}
        lands = sum(n for name, n in main.items() if "Land" in rows[name].types.types)
        creatures = sum(
            n for name, n in main.items() if "Creature" in rows[name].types.types
        )
        cheap = sum(
            n
            for name, n in main.items()
            if rows[name].mana_cost is not None and rows[name].mana_cost.mana_value <= 2
        )
        if not set(entry.interaction_cards + entry.synergy_cards) <= set(main):
            raise ValueError(
                f"{entry.id}: construction roles must occur in the main deck"
            )
        if any(
            len(set(role)) != len(role)
            for role in (entry.interaction_cards, entry.synergy_cards)
        ):
            raise ValueError(f"{entry.id}: construction roles must not duplicate cards")
        interaction = sum(main[name] for name in entry.interaction_cards)
        synergy = sum(main[name] for name in entry.synergy_cards)
        if (
            sum(main.values()) != 40
            or lands != 17
            or creatures < 8
            or cheap < 8
            or interaction < 4
            or synergy < 4
        ):
            raise ValueError(f"{entry.id}: outside v1 functional construction template")
        for name in _card_names(entry.deck):
            if (
                name not in ("Plains", "Island", "Swamp", "Mountain", "Forest")
                and main.get(name, 0) + entry.deck.sideboard.get(name, 0) > 4
            ):
                raise ValueError(f"{entry.id}: more than four copies of {name}")
        for replacement in entry.replacements:
            before, after = catalog[replacement.remove], catalog[replacement.add]
            if (
                before.mana_cost is None
                or before.mana_cost != after.mana_cost
                or before.types != after.types
            ):
                raise ValueError(
                    f"{entry.id}: variants must preserve printed mana cost and types"
                )
        audits.append(
            DeckAudit(
                entry=entry.id,
                cards=sum(main.values()),
                lands=lands,
                creatures=creatures,
                cheap_spells=cheap,
                interaction=interaction,
                synergy=synergy,
            )
        )
    return tuple(audits)


class Panel(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    kind: PanelKind
    entries: tuple[str, str]
    deals: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_deals(self) -> Panel:
        if len(set(self.deals)) != len(self.deals) or any(
            not 0 <= deal < 2**64 for deal in self.deals
        ):
            raise ValueError("deals must be unique uint64 seeds")
        return self


class Manifest(StrictModel):
    """Frozen experimental populations, including both training seats."""

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    entries: tuple[Entry, ...] = Field(min_length=1)
    heldout_cards: tuple[str, ...] = ()
    heldout_combinations: tuple[tuple[str, ...], ...] = ()
    panels: tuple[Panel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def separated(self) -> Manifest:
        by_id = {entry.id: entry for entry in self.entries}
        if len(by_id) != len(self.entries):
            raise ValueError("deck IDs must be unique")
        if len({panel.id for panel in self.panels}) != len(self.panels):
            raise ValueError("panel IDs must be unique")
        training = [entry for entry in self.entries if entry.tickets]
        totals: Counter[str] = Counter()
        for entry in training:
            totals[entry.group] += entry.tickets
        if not totals["new"] or totals != {
            "anchor": 2 * totals["new"],
            "new": totals["new"],
            "mixed": totals["new"],
        }:
            raise ValueError("v1 mixture must be 50% anchor, 25% new, 25% mixed")
        seen = set().union(*(_card_names(entry.deck) for entry in training))
        heldout = set(self.heldout_cards)
        if len(heldout) != len(self.heldout_cards) or heldout & seen:
            raise ValueError("held-out cards leak into training main or sideboard")
        identities = {entry.deck.identity for entry in training}
        for entry in self.entries:
            if entry.group == "test" and entry.deck.identity in identities:
                raise ValueError("held-out deck content duplicates a training deck")
        self._check_variants(by_id)
        for combination in self.heldout_combinations:
            names = set(combination)
            if len(names) < 2 or len(names) != len(combination) or not names <= seen:
                raise ValueError(
                    "held-out combinations need distinct known training cards"
                )
            if any(names <= _card_names(entry.deck) for entry in training):
                raise ValueError("held-out combination leaks into a training deck")
            # Opponent exposure is also experience: independent seat draws can
            # put any two training decks together. This split is explicitly
            # intra-deck composition, never absence from the whole game.
        self._check_panels(seen)
        return self

    def _check_variants(self, by_id: dict[str, Entry]) -> None:
        for entry in self.entries:
            if entry.parent is None:
                continue
            if entry.parent not in by_id or entry.parent == entry.id:
                raise ValueError("variant parent must name another manifest entry")
            parent = by_id[entry.parent]
            if parent.parent is not None:
                raise ValueError("v1 variants derive directly from an authored deck")
            if (
                _substitute(parent.deck, entry.replacements).identity
                != entry.deck.identity
            ):
                raise ValueError("variant differs from its recorded construction")

    def _check_panels(self, training_cards: set[str]) -> None:
        """Every split must have an exam that actually exercises its exclusions."""
        by_id = {entry.id: entry for entry in self.entries}
        heldout = set(self.heldout_cards)
        used_cards: set[str] = set()
        used_combinations: set[tuple[str, ...]] = set()
        for panel in self.panels:
            if any(name not in by_id for name in panel.entries):
                raise ValueError("panel names an unknown deck")
            entries = [by_id[name] for name in panel.entries]
            names = set().union(*(_card_names(entry.deck) for entry in entries))
            if panel.kind == "anchor" and any(e.group != "anchor" for e in entries):
                raise ValueError("anchor panel must retain anchor decks")
            if panel.kind == "expanding" and (
                any(not e.tickets for e in entries)
                or not any(e.group == "new" for e in entries)
            ):
                raise ValueError(
                    "expanding panel requires training decks including new cards"
                )
            if panel.kind.startswith("heldout-") and not any(
                e.group == "test" for e in entries
            ):
                raise ValueError("held-out panel requires an evaluation-only deck")
            if panel.kind == "heldout-deck" and not names <= training_cards:
                raise ValueError("held-out deck panel must use known training cards")
            if panel.kind == "heldout-card":
                if not names & heldout:
                    raise ValueError("held-out card panel contains no held-out card")
                used_cards |= names & heldout
            if panel.kind == "heldout-combination":
                found = {
                    combo
                    for combo in self.heldout_combinations
                    if any(
                        set(combo) <= _card_names(e.deck)
                        for e in entries
                        if not e.tickets
                    )
                }
                if not found:
                    raise ValueError(
                        "combination panel contains no declared combination"
                    )
                used_combinations |= found
        if used_cards != heldout or used_combinations != set(self.heldout_combinations):
            raise ValueError("every held-out card and combination needs a panel")

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    def source(self) -> UniformDecks:
        """Tickets are exchangeable population entries; mirrors remain possible."""
        tickets = [(entry, i) for entry in self.entries for i in range(entry.tickets)]
        return UniformDecks(
            decks=tuple(entry.deck for entry, _ in tickets),
            entry_ids=tuple(f"{entry.id}/ticket-{i}" for entry, i in tickets),
            sampling="uniform-independent-v1",
        )


class World(StrictModel):
    world_version: str
    card_digest: str
    engine_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def current(cls) -> World:
        return cls(
            **managym.world_identity(),
            engine_source_sha256=source_bundle_sha256(engine_source_paths()),
        )


@dataclass(frozen=True)
class EvaluationRound:
    """One checkpoint contrast, repeated across all panels and deals."""

    update: int
    anchor: Anchor


class Budget(StrictModel):
    """Require explicit bounds; the versioned JSON owns the proposed values."""

    seeds: tuple[int, ...] = Field(min_length=1)
    updates: int = Field(ge=1)
    milestones: tuple[int, ...]
    streams: int = Field(ge=2)
    games_per_stream: int = Field(ge=1)
    training_seconds: int = Field(ge=1)
    training_job_hours: float = Field(gt=0, allow_inf_nan=False)
    training_usd_per_seed: float = Field(gt=0, allow_inf_nan=False)
    evaluation_hours: float = Field(gt=0, allow_inf_nan=False)
    evaluation_usd: float = Field(gt=0, allow_inf_nan=False)
    reserve_usd: float = Field(ge=0, allow_inf_nan=False)
    game_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_commands: int = Field(ge=1)
    proposed_total_usd: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def coordinates(self) -> Budget:
        if len(set(self.seeds)) != len(self.seeds) or any(
            not 0 <= seed < 2**32 for seed in self.seeds
        ):
            raise ValueError("training seeds must be unique uint32 values")
        if tuple(sorted(set(self.milestones))) != self.milestones or any(
            not 0 < update < self.updates for update in self.milestones
        ):
            raise ValueError("milestones must be increasing interior updates")
        if (
            self.training_seconds <= 120
            or self.training_seconds + 150 >= self.training_job_hours * 3600
        ):
            raise ValueError(
                "training lifetime must cover stop and finalization reserves"
            )
        if (
            len(self.seeds) * self.training_usd_per_seed
            + self.evaluation_usd
            + self.reserve_usd
            > self.proposed_total_usd
        ):
            raise ValueError(
                "declared allocations exceed the proposed cumulative budget"
            )
        return self

    @property
    def training_games(self) -> int:
        return len(self.seeds) * self.updates * self.streams * self.games_per_stream

    @property
    def export_updates(self) -> tuple[int, ...]:
        """Planned exports, retained in analysis even when training stops early."""
        return (0, *self.milestones, self.updates)

    @property
    def evaluation_rounds(self) -> tuple[EvaluationRound, ...]:
        """Scheduling, callback admission and game budgets share these contrasts."""
        return (
            *(EvaluationRound(update, "heuristic") for update in self.export_updates),
            EvaluationRound(self.updates, "initial"),
        )


class Analysis(StrictModel):
    """Prospective estimator and sealed training-health inputs, never defaults."""

    bootstrap_seed: int = Field(ge=0)
    bootstrap_resamples: int = Field(ge=1)
    health_clock: Literal["updates"]
    parameter_change: Literal["numerical/parameter_relative_delta"]
    finite_metrics: tuple[str, ...] = Field(min_length=1)
    zero_metrics: tuple[str, ...] = Field(min_length=1)
    positive_metrics: tuple[str, ...] = Field(min_length=1)
    sampled_finite_metrics: tuple[str, ...] = Field(min_length=1)
    sampled_updates: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_metrics(self) -> Analysis:
        metrics = (
            *self.finite_metrics,
            *self.zero_metrics,
            *self.positive_metrics,
            *self.sampled_finite_metrics,
            self.parameter_change,
        )
        if len(set(metrics)) != len(metrics):
            raise ValueError("health metrics must be distinct")
        if tuple(sorted(set(self.sampled_updates))) != self.sampled_updates:
            raise ValueError("sampled health coordinates must be unique and increasing")
        return self


class Protocol(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    status: Literal["prospective-unexecuted"] = "prospective-unexecuted"
    world: World
    manifest: Manifest
    observation: ObservationConfig
    observation_shapes: dict[str, tuple[int, ...]]
    agent: ModelConfig
    learning: DampedPolicyLearning
    budget: Budget
    analysis: Analysis

    @model_validator(mode="after")
    def contract(self) -> Protocol:
        if self.observation.action_overflow != "error":
            raise ValueError("growing-pool protocol never truncates legal choices")
        if self.agent.world != "world":
            raise ValueError("growing-pool policy must read the complete world catalog")
        if any(
            not 1 <= u <= self.budget.updates for u in self.analysis.sampled_updates
        ):
            raise ValueError("sampled health coordinates outside update horizon")
        schedule = self.learning.iteration_schedule
        if schedule is None or schedule.steps != self.budget.updates:
            raise ValueError("learning schedule must bind the pilot update horizon")
        if ObservationSpace(self.observation).shapes != self.observation_shapes:
            raise ValueError("observation shapes differ from frozen protocol")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    @property
    def evaluation_games(self) -> int:
        return (
            len(self.budget.seeds)
            * len(self.budget.evaluation_rounds)
            * sum(4 * len(panel.deals) for panel in self.manifest.panels)
        )

    def environment(self) -> Environment:
        """The same sealed namespace is used by scheduling and offline reads."""
        return Environment(
            name=self.id, files=self.digest, config={"status": self.status}
        )

    def admit(self) -> None:
        if self.world != World.current():
            raise ValueError(
                "runtime world or engine source differs from frozen protocol"
            )
        # Native admission applies to all evaluation lists too, not just training.
        UniformDecks(
            decks=tuple(entry.deck for entry in self.manifest.entries),
            sampling="uniform-independent-v1",
        ).check()
        audit_decks(self.manifest)

    def regime(self) -> TrainingRegime:
        """Compile frozen settings for execution or offline identity comparison.

        Runtime admission belongs at job/callback boundaries. Retained evidence
        must remain readable when today's engine no longer admits this world.
        """
        budget = self.budget
        return TrainingRegime(
            id=self.id,
            world=self.world.world_version,
            deck_source=self.manifest.source(),
            observation=self.observation.model_copy(deep=True),
            agent=self.agent.model_copy(deep=True),
            schedule_clock="iteration_fraction",
            wall_seconds=budget.training_seconds + 30,
            stages=[
                TrainSelfPlay(
                    id="self-play",
                    operation="train_self_play",
                    updates=budget.updates,
                    streams=budget.streams,
                    games_per_stream=budget.games_per_stream,
                    transitions=64,
                    learning_seats="both",
                    learning=self.learning.model_copy(deep=True),
                    execution=Execution(
                        device="cuda",
                        threads=1,
                        workers=1,
                        wall_seconds=budget.training_seconds,
                        memory_bytes=32 * 1024**3,
                    ),
                )
            ],
        )

    def job(self, *, experiment: str) -> TrainingJob:
        self.admit()
        return TrainingJob(
            regime=self.regime(),
            experiment=experiment,
            milestones=self.budget.milestones,
            finalization_seconds=120,
            stop_reserve_seconds=120,
            validate_numerics=True,
            training_exposure=True,
        )


class ProtocolBinding(StrictModel):
    """One frozen protocol and its canonical Etude entry point.

    Etude names experiments after file stems. Deriving that name from the
    importable module keeps declarations, discovery and result stores aligned.
    Offline reads check the digest but do not admit today's runtime world.
    """

    module: str = Field(pattern=r"^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*$")
    protocol_path: Path
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def experiment(self) -> str:
        return self.module.rsplit(".", 1)[-1]

    @property
    def entry_point(self) -> Path:
        module = import_module(self.module)
        if getattr(module, "BINDING", None) != self or module.__file__ is None:
            raise ValueError("entry point differs from selected protocol binding")
        path = Path(module.__file__).resolve()
        if path.stem != self.experiment:
            raise ValueError("entry point stem differs from experiment identity")
        return path

    @property
    def callback(self) -> Callable[[Match], Result]:
        # The worker imports this symbol, so never pass a runpy wrapper or partial.
        self.entry_point
        callback = getattr(import_module(self.module), "play", None)
        if not callable(callback) or callback.__module__ != self.module:
            raise ValueError("binding requires a canonical importable play callback")
        return callback

    def check_entry(self, path: Path) -> None:
        """Reject copied or renamed discovery files still pointing at another entry."""
        if path.resolve() != self.entry_point:
            raise ValueError("discovered file differs from bound entry point")

    def read(self, *, admit: bool = False) -> Protocol:
        self.entry_point
        protocol = Protocol.model_validate_json(self.protocol_path.read_bytes())
        if protocol.digest != self.protocol_sha256:
            raise ValueError("protocol digest differs from selected binding")
        if admit:
            protocol.admit()
        return protocol


def read_binding(module: str) -> ProtocolBinding:
    """Read the authored binding, never infer a protocol from an experiment name."""
    binding = getattr(import_module(module), "BINDING", None)
    if not isinstance(binding, ProtocolBinding) or binding.module != module:
        raise ValueError("entry point does not declare its own protocol binding")
    binding.entry_point
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "regime"))
    parser.add_argument("--experiment", default="experiments.growing_pool")
    args = parser.parse_args()
    binding = read_binding(args.experiment)
    protocol = binding.read(admit=True)
    if args.command == "regime":
        print(protocol.regime().model_dump_json(indent=2))
    else:
        print(f"{binding.experiment} / {protocol.id}: {protocol.digest}")
        print(
            f"{len(protocol.manifest.entries)} decks; {len(protocol.manifest.source().decks)} sampling tickets"
        )
        print(
            f"Planned games: {protocol.budget.training_games} training + {protocol.evaluation_games} evaluation"
        )
        for audit in audit_decks(protocol.manifest):
            print(audit.model_dump_json())
        print("Protocol admitted; no games played and no compute allocated.")


if __name__ == "__main__":
    import_module("experiments.growing_pool_protocol").main()
