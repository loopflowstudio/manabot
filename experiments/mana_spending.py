"""Count native spending decisions in bounded mirror games (MTG-347).

`count` writes exact decks and per-game stage/offer counts. Payment attribution
comes from the engine, never from Python mana arithmetic. This exploratory
frequency slice rents nothing and does not implement alternative decision shapes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Literal, cast

import numpy as np

from experiments.elemental_grid import DECKS, PACK
from manabot.env.env import Env
from manabot.infra.identity import file_sha256
from manabot.sim.anchors import RandomMatchupPlayer
from manabot.sim.heuristic import HeuristicPlayer
from manabot.training.decks import UniformDecks
import managym

ROOT = Path(__file__).resolve().parents[1]
TLA = ROOT / "experiments/regimes/tla-frontier-c5fba4-w64.json"
MAX_DECISIONS = 10_000
Policy = Literal["heuristic", "random"]


@dataclass(frozen=True)
class Deck:
    set: str
    name: str
    main: dict[str, int]
    sideboard: dict[str, int]

    def players(self) -> list[managym.PlayerConfig]:
        return [
            managym.PlayerConfig(
                f"seat-{seat}", self.main, self.sideboard, managym.WORLD_SETUP
            )
            for seat in range(2)
        ]


@dataclass
class _Viewer:
    last_raw_obs: managym.Observation


@dataclass
class GameCounts:
    deck_set: str
    deck: str
    policy: Policy
    seed: int
    stages: dict[str, int] = field(default_factory=dict)
    spend_offers: list[int] = field(default_factory=list)
    casts: int = 0
    mana_payments: int = 0
    mana_payments_over_tapped: int = 0
    mana_spend_decisions_over_tapped: int = 0
    terminated: bool = False
    replay_verified: bool = False
    actions: list[int] = field(default_factory=list)
    final_digest: str | None = None
    error: str | None = None


def decks() -> list[Deck]:
    elemental = [managym.authored_deck_setup(PACK, name) for name in DECKS]
    result = [
        Deck("elemental", name, config.decklist, config.sideboard)
        for name, config in zip(DECKS, elemental, strict=True)
    ]
    # Read retained population bytes as data; never load/rebind its frozen regime.
    source = UniformDecks.model_validate(json.loads(TLA.read_text())["deck_source"])
    if source.entry_ids is None:
        raise ValueError("TLA population is missing source event IDs")
    tla = [
        Deck("tla", name, deck.main, deck.sideboard)
        for name, deck in zip(source.entry_ids, source.decks, strict=True)
        if sum(deck.main.values()) == 40
    ][:10]
    if len(tla) != 10:
        raise ValueError("TLA population has fewer than ten 40-card builds")
    result += tla
    result.append(
        Deck(
            "heartbeat",
            "heartbeat-rg",
            {
                "Forest": 12,
                "Mountain": 12,
                "Heartbeat of Spring": 4,
                "Llanowar Elves": 4,
                "War Mammoth": 4,
                "Hill Giant": 4,
            },
            {},
        )
    )
    # Admit exactly the submitted lists, including sideboards, before any game.
    for deck, issues in zip(
        result,
        managym.check_decks([managym.Deck(d.main, d.sideboard) for d in result]),
        strict=True,
    ):
        if issues:
            raise ValueError(f"{deck.name}: {issues}")
    return result


def play(deck: Deck, policy: Policy, seed: int) -> GameCounts:
    row = GameCounts(deck.set, deck.name, policy, seed)
    env = managym.Env(seed=seed, skip_trivial=True)
    players = [
        HeuristicPlayer() if policy == "heuristic" else RandomMatchupPlayer(seed + seat)
        for seat in range(2)
    ]
    try:
        raw, _ = env.reset(deck.players())
        for _ in range(MAX_DECISIONS):
            if raw.game_over:
                row.terminated = True
                break
            context = json.loads(env.decision_context_json(raw.turn.agent_player_id))
            stage = context["stage"]
            if not isinstance(stage, str):
                raise ValueError("native decision context has no stage name")
            row.stages[stage] = row.stages.get(stage, 0) + 1
            if stage == "mana_spend":
                row.spend_offers.append(len(raw.action_space.actions))
            # Both code players read only this raw viewer or actions_valid.
            # Keep every native offer: policy buffer capacity must not bias a
            # frequency measurement. This narrow cast mirrors heuristic's own
            # raw-view fallback at the untyped native boundary.
            selected = players[raw.turn.agent_player_id].act(
                cast(Env, _Viewer(raw)),
                {
                    "actions_valid": np.ones(
                        len(raw.action_space.actions), dtype=np.float32
                    )
                },
            )
            if int(raw.action_space.actions[selected].action_type) == int(
                managym.ActionEnum.PRIORITY_CAST_SPELL
            ):
                row.casts += 1
            row.actions.append(selected)
            raw, _, terminated, truncated, _ = env.step(selected)
            if terminated or truncated:
                row.terminated = terminated
                if truncated:
                    row.error = "truncated"
                break
        else:
            row.error = f"decision limit ({MAX_DECISIONS})"
        if row.terminated and row.error is None:
            row.final_digest = env.state_digest()
            replay = managym.Env(seed=seed, skip_trivial=True)
            replay.reset(deck.players())
            for action in row.actions:
                replay.step(action)
            if replay.state_digest() != row.final_digest or replay.info() != env.info():
                raise ValueError("replay state or payment diagnostics disagree")
            row.replay_verified = True
    except (managym.AgentError, ValueError, RuntimeError) as error:
        row.error = f"{type(error).__name__}: {error}"
    finally:
        info = env.info()
        row.mana_payments = int(info["mana_payments"])
        row.mana_payments_over_tapped = int(info["mana_payments_over_tapped"])
        row.mana_spend_decisions_over_tapped = int(
            info["mana_spend_decisions_over_tapped"]
        )
    return row


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(rows: list[GameCounts]) -> None:
    groups: dict[tuple[str, Policy], list[GameCounts]] = defaultdict(list)
    for row in rows:
        groups[row.deck_set, row.policy].append(row)
    print(
        "decks policy games invalid decisions/game spend/game share over-tap non-over-tap-share no-spend"
    )
    for (deck_set, policy), games in groups.items():
        valid = [game for game in games if game.terminated and game.error is None]
        decisions = sum(sum(game.stages.values()) for game in valid)
        spending = sum(game.stages.get("mana_spend", 0) for game in valid)
        over = sum(game.mana_spend_decisions_over_tapped for game in valid)
        no_spend = sum("mana_spend" not in game.stages for game in valid)
        n = len(valid)
        print(
            f"{deck_set} {policy} {n} {len(games) - n} "
            f"{_ratio(decisions, n):.2f} {_ratio(spending, n):.2f} "
            f"{_ratio(spending, decisions):.2%} {_ratio(over, spending):.2%} "
            f"{_ratio(spending - over, decisions):.2%} {_ratio(no_spend, n):.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["count"])
    parser.add_argument(
        "--games", type=int, default=8, help="games per deck and policy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/data/mana_spending")
    )
    args = parser.parse_args()
    if args.games < 1 or args.seed < 0:
        parser.error("games must be positive and seed nonnegative")
    population = decks()
    args.out.mkdir(parents=True, exist_ok=True)
    # Never silently replace an earlier exploratory run's evidence.
    if any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; choose a new --out")
    metadata = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "world": managym.WORLD_VERSION,
        "native_sha256": file_sha256(Path(managym._managym.__file__)),
        "source_sha256": file_sha256(Path(__file__)),
        "tla_source_sha256": file_sha256(TLA),
        "games_per_deck": args.games,
        "seed": args.seed,
        "decks": [asdict(deck) for deck in population],
    }
    (args.out / "decks.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows: list[GameCounts] = []
    with (args.out / "games.jsonl").open("w") as output:
        for index, deck in enumerate(population):
            for policy in ("heuristic", "random"):
                for game in range(args.games):
                    row = play(deck, policy, args.seed + index * args.games + game)
                    rows.append(row)
                    output.write(json.dumps(asdict(row)) + "\n")
                    output.flush()
    summarize(rows)
    if any(row.error or not row.terminated for row in rows):
        raise SystemExit("incomplete games retained; see games.jsonl")


if __name__ == "__main__":
    main()
