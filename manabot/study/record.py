"""
record.py
Play complete games and keep what each player saw, was offered and chose

A `GameRecord` is the only input the measures read, so anything a measure needs
must be captured here. Records hold the acting player's view: the opponent's
hand is never stored. Players are built from the same picklable specs the arena
uses; this module adds no rules path of its own.
"""

from __future__ import annotations

# Standard library
from dataclasses import asdict, dataclass, field
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Iterable, Iterator

# First-party imports
from manabot.infra.hypers import MatchHypers

ATTACK = "DECLARE_ATTACKER"
BLOCK = "DECLARE_BLOCKER"
PLAY_LAND = "PRIORITY_PLAY_LAND"
CAST_SPELL = "PRIORITY_CAST_SPELL"
PASS_PRIORITY = "PRIORITY_PASS_PRIORITY"


@dataclass(frozen=True)
class GameSpec:
    """One game to play. `players` and `decks` are indexed by seat."""

    game_id: str
    match: dict[str, Any]
    players: tuple[dict[str, Any], dict[str, Any]]
    labels: tuple[str, str]
    decks: tuple[str, str]
    seed: int
    max_commands: int = 10_000


@dataclass(frozen=True)
class Permanent:
    name: str
    power: int
    toughness: int
    tapped: bool
    summoning_sick: bool
    is_creature: bool
    is_land: bool
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    turn: int
    active: int
    actor: int
    phase: str
    step: str
    kind: str
    offered: tuple[str, ...]
    offer_labels: tuple[str, ...]
    chosen: str
    chosen_index: int
    label: str
    declared: bool | None
    subject: str | None
    life: tuple[int, int]
    hand: int
    mine: tuple[Permanent, ...]
    theirs: tuple[Permanent, ...]
    latency: float = 0.0

    @property
    def own_turn(self) -> bool:
        return self.actor == self.active


@dataclass
class GameRecord:
    game_id: str
    labels: tuple[str, str]
    decks: tuple[str, str]
    seed: int
    first_player: int | None = None
    winner: int | None = None
    end: str = "command_cap"
    error: str | None = None
    turns: int = 0
    seconds: float = 0.0
    decisions: list[Decision] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.end == "terminal"

    def seats(self, label: str) -> list[int]:
        return [seat for seat, name in enumerate(self.labels) if name == label]

    def by(self, seat: int) -> Iterator[Decision]:
        return (decision for decision in self.decisions if decision.actor == seat)


def build_player(spec: dict[str, Any], seed: int):
    """Build a player and its observation space from an arena player spec."""
    from manabot.arena.players import DemoSearchPlayer, ScriptedGreedyPlayer
    from manabot.sim.flat_mc import make_player

    if spec["kind"] == "demo_search":
        return DemoSearchPlayer(spec, seed), None
    if spec["kind"] == "scripted_greedy":
        return ScriptedGreedyPlayer(), None
    return make_player(dict(spec), seed=seed)


def _keywords(value: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in dir(value)
            if not name.startswith("_") and getattr(value, name) is True
        )
    )


def _battlefield(
    permanents: Iterable[Any], cards: Iterable[Any]
) -> list[tuple[Any, Any]]:
    """Pair each permanent with its card; both lists share battlefield order."""
    import managym

    on_battlefield = [
        card for card in cards if int(card.zone) == int(managym.ZoneEnum.BATTLEFIELD)
    ]
    permanents = list(permanents)
    if len(on_battlefield) != len(permanents):
        return [(permanent, None) for permanent in permanents]
    return list(zip(permanents, on_battlefield))


def _permanents(pairs: Iterable[tuple[Any, Any]]) -> tuple[Permanent, ...]:
    out = []
    for permanent, card in pairs:
        types = card.card_types if card is not None else None
        out.append(
            Permanent(
                name=str(card.name) if card is not None else "",
                power=int(permanent.power),
                toughness=int(permanent.toughness),
                tapped=bool(permanent.tapped),
                summoning_sick=bool(permanent.is_summoning_sick),
                is_creature=bool(types.is_creature or permanent.is_animated)
                if types is not None
                else int(permanent.toughness) > 0,
                is_land=bool(types.is_land) if types is not None else False,
                keywords=_keywords(permanent.keywords),
            )
        )
    return tuple(out)


def _decision(raw: Any, frame: dict[str, Any], action: int, latency: float) -> Decision:
    offers = frame["offers"]
    index = next(i for i, offer in enumerate(offers) if int(offer["id"]) == action)
    chosen = offers[index]
    engine_action = raw.action_space.actions[index]
    kind = frame["action_space"]
    pairs = _battlefield(raw.agent_permanents, raw.agent_cards)
    mine = _permanents(pairs)
    declared = None
    subject = None
    if kind in (ATTACK, BLOCK):
        declared = bool(engine_action.declared)
        names = {
            int(permanent.id): str(card.name)
            for permanent, card in pairs
            if card is not None
        }
        subject = next(
            (names[int(i)] for i in engine_action.focus if int(i) in names), None
        )
    turn = frame["projection"]["turn"]
    return Decision(
        turn=int(turn["turn_number"]),
        active=int(turn["active_player_id"]),
        actor=int(raw.agent.player_index),
        phase=str(turn["phase"]),
        step=str(turn["step"]),
        kind=kind,
        offered=tuple(sorted({offer["action_type"] for offer in offers})),
        offer_labels=tuple(offer["label"] for offer in offers),
        chosen=chosen["action_type"],
        chosen_index=index,
        label=chosen["label"],
        declared=declared,
        subject=subject,
        life=(int(raw.agent.life), int(raw.opponent.life)),
        hand=len(frame["projection"]["agent"]["hand"]),
        mine=mine,
        theirs=_permanents(_battlefield(raw.opponent_permanents, raw.opponent_cards)),
        latency=latency,
    )


def record_game(spec: GameSpec) -> GameRecord:
    """Play one game to the end. Failures are kept as records, never dropped."""
    import torch

    from etude.server import ASSET_MANIFEST_HASH, CONTENT_HASH
    from manabot.env import Env, Match, ObservationSpace, Reward
    from manabot.infra.hypers import RewardHypers
    from manabot.sim.teacher1_evidence import build_viewer_frame

    torch.set_num_threads(1)
    record = GameRecord(spec.game_id, spec.labels, spec.decks, spec.seed)
    started = time.perf_counter()
    try:
        built = [
            build_player(player, (spec.seed * 7919 + seat * 104729 + 17) % 2**63)
            for seat, player in enumerate(spec.players)
        ]
        spaces = [space for _, space in built if space is not None]
        if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
            raise ValueError("players have different observation ABIs")
        env = Env(
            Match(MatchHypers.model_validate(spec.match)),
            spaces[0] if spaces else ObservationSpace(),
            Reward(RewardHypers()),
            seed=spec.seed,
            auto_reset=False,
        )
        obs, _ = env.reset(seed=spec.seed)
        for revision in range(spec.max_commands):
            raw = env.last_raw_obs
            actor = int(raw.agent.player_index)
            frame = build_viewer_frame(
                raw,
                match_id=spec.game_id,
                revision=revision,
                content_hash=CONTENT_HASH,
                asset_manifest_hash=ASSET_MANIFEST_HASH,
            )
            began = time.perf_counter()
            action = int(built[actor][0].act(env, obs))
            decision = _decision(raw, frame, action, time.perf_counter() - began)
            if record.first_player is None:
                # Turns alternate, so the parity of any turn names who started.
                record.first_player = (
                    decision.active if decision.turn % 2 else 1 - decision.active
                )
            record.decisions.append(decision)
            record.turns = decision.turn
            obs, _, terminated, truncated, _ = env.step(action)
            if truncated:
                record.end = "truncated"
                break
            if terminated:
                record.winner = env._engine.winner_index()
                record.end = "terminal"
                break
    except Exception as exc:
        record.end = "crash"
        record.error = f"{type(exc).__name__}: {exc}"
    record.seconds = time.perf_counter() - started
    return record


def record_games(
    specs: Iterable[GameSpec], *, workers: int = 1, progress=None
) -> Iterator[GameRecord]:
    """Play games, in parallel when `workers` > 1. Yields in completion order."""
    specs = list(specs)
    if workers <= 1:
        results: Iterable[GameRecord] = map(record_game, specs)
        yield from _reported(results, len(specs), progress)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        yield from _reported(
            pool.imap_unordered(record_game, specs), len(specs), progress
        )


def _reported(results, total, progress):
    for done, record in enumerate(results, start=1):
        if progress is not None:
            progress(done, total)
        yield record


def write_games(path: Path | str, games: Iterable[GameRecord]) -> int:
    """Write one JSON object per game. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for game in games:
            handle.write(json.dumps(asdict(game)) + "\n")
            handle.flush()
            count += 1
    return count


def _permanent(data: dict[str, Any]) -> Permanent:
    return Permanent(**{**data, "keywords": tuple(data.get("keywords", ()))})


def read_games(path: Path | str) -> list[GameRecord]:
    games = []
    for line in Path(path).read_text().splitlines():
        data = json.loads(line)
        decisions = [
            Decision(
                **{
                    **decision,
                    "offered": tuple(decision["offered"]),
                    "offer_labels": tuple(decision["offer_labels"]),
                    "life": tuple(decision["life"]),
                    "mine": tuple(_permanent(p) for p in decision["mine"]),
                    "theirs": tuple(_permanent(p) for p in decision["theirs"]),
                }
            )
            for decision in data.pop("decisions")
        ]
        games.append(
            GameRecord(
                **{
                    **data,
                    "labels": tuple(data["labels"]),
                    "decks": tuple(data["decks"]),
                    "decisions": decisions,
                }
            )
        )
    return games
