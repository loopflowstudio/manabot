"""Check that every fact and shared-effect id real games emit has a catalog row.

Catalog models (`ModelConfig.record_vocabulary="catalog"`) embed fact and
effect ids through the card encoder's exact vocabulary and reject an id outside
it. That vocabulary is native `learning_record_names`, a list maintained beside
the encoders, so an encoder that starts hashing a new name without listing it
would stop training mid-run. This sweep plays seeded random games whose decks
together hold every eligible world card, encodes each decision at policy input
v9 and runs a tiny catalog policy over the batch, so a missing name fails here
with the game that produced it. `--decks elemental` instead exercises the curated
Elemental matchups, whose searches and composed effects retain execution state.

Entry points: `coverage_sweep` (library) and
`uv run python -m manabot.verify.record_coverage --games 200` (CLI).
"""

import argparse
from dataclasses import dataclass
import random
from typing import Literal

import numpy as np
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
import managym

_BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")
# Each deck: five of each basic plus `_SPELLS_PER_DECK` spells, three copies each.
_SPELLS_PER_DECK = 12
_MAX_DECISIONS = 600
_BATCH = 128


@dataclass(frozen=True)
class CoverageReport:
    games: int
    decisions: int
    fact_rows: int
    effect_rows: int
    capacity_overflows: int
    cards_played: int
    eligible_cards: int


def _eligible_spells() -> list[str]:
    pool = managym.content_pool(managym.WORLD_SETUP)["definitions"]
    return sorted(
        row["registry_name"]
        for row in pool
        if row["eligibility"] == "eligible" and row["registry_name"] not in _BASICS
    )


def _decks(games: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """Two spell lists per game, cycling a seeded shuffle of the eligible pool
    so every eligible card is dealt before any repeats."""
    rng = random.Random(seed)
    spells = _eligible_spells()
    order: list[str] = []
    out = []
    for _ in range(games):
        seats = []
        for _ in range(2):
            deck: list[str] = []
            while len(deck) < _SPELLS_PER_DECK:
                if not order:
                    order = rng.sample(spells, len(spells))
                name = order.pop()
                if name not in deck:
                    deck.append(name)
            seats.append(deck)
        out.append((seats[0], seats[1]))
    return out


def _player(name: str, spells: list[str]) -> managym.PlayerConfig:
    deck = {basic: 5 for basic in _BASICS} | {spell: 3 for spell in spells}
    return managym.PlayerConfig(name, deck)


def coverage_sweep(
    games: int, seed: int = 0, *, decks: Literal["world", "elemental"] = "world"
) -> CoverageReport:
    """Play `games` seeded random games and embed every decision through a
    catalog policy; raises `ValueError` naming the game on an unknown id."""
    torch.manual_seed(seed)
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    model = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    ).eval()
    assert model.config.record_vocabulary == "catalog"
    rng = random.Random(seed)
    decisions = facts = effects = overflows = 0
    played: set[str] = set()
    if decks == "elemental":
        keys = managym.authored_deck_keys("elemental-world-v2")
        pairs = [
            (first, second) for first in keys for second in keys if first != second
        ]
        players = [
            [
                managym.authored_deck_setup("elemental-world-v2", key)
                for key in pairs[game % len(pairs)]
            ]
            for game in range(games)
        ]
    else:
        players = [
            [_player("first", first), _player("second", second)]
            for first, second in _decks(games, seed)
        ]
    for game, seats in enumerate(players):
        first, second = [sorted(set(seat.decklist) - set(_BASICS)) for seat in seats]
        played.update(first, second)
        env = managym.Env(seed=seed * 100_003 + game)
        env.reset(seats)
        pending: list[dict[str, np.ndarray]] = []
        for _ in range(_MAX_DECISIONS):
            player = env.current_agent_index()
            if player is None:
                break
            obs = env.observation_for_player(player)
            try:
                encoded = space.encode(obs)
            except ValueError as error:
                # Training aborts a game past its capacities (`recover_capacity`).
                if "capacity exceeded" not in str(error):
                    raise
                overflows += 1
                break
            facts += int(encoded["facts"][..., -1].sum())
            effects += int((encoded["effects"][..., -1] > 0).sum())
            pending.append(encoded)
            if len(pending) == _BATCH:
                _embed(model, pending, game, first, second)
                decisions += len(pending)
                pending = []
            _, _, done, truncated, _ = env.step(
                rng.randrange(len(obs.action_space.actions))
            )
            if done or truncated:
                break
        if pending:
            _embed(model, pending, game, first, second)
            decisions += len(pending)
    return CoverageReport(
        games=games,
        decisions=decisions,
        fact_rows=facts,
        effect_rows=effects,
        capacity_overflows=overflows,
        cards_played=len(played),
        eligible_cards=len(_eligible_spells()),
    )


def _embed(
    model: ActorCritic,
    rows: list[dict[str, np.ndarray]],
    game: int,
    first: list[str],
    second: list[str],
) -> None:
    batch = {key: torch.from_numpy(np.stack([r[key] for r in rows])) for key in rows[0]}
    try:
        with torch.no_grad():
            model(batch)
    except ValueError as error:
        raise ValueError(
            f"game {game} (decks {first} / {second}) emitted an id outside the "
            f"catalog record vocabulary: {error}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decks", choices=("world", "elemental"), default="world")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = coverage_sweep(args.games, args.seed, decks=args.decks)
    print(
        f"{report.games} games, {report.decisions} decisions "
        f"({report.fact_rows} fact rows, {report.effect_rows} effect rows; "
        f"{report.capacity_overflows} games ended at a capacity limit), "
        f"{report.cards_played}/{report.eligible_cards} eligible cards dealt: "
        "every id has a catalog row"
    )


if __name__ == "__main__":
    main()
