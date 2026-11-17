"""Viewer-safe probe corpus and game-level readouts, without a learning backend.

Rows own compact native prefixes and public state; privileged hand-presence
labels are a separate corpus column. Reservoir sampling bounds each viewer to
16 decisions without favoring the opening. Whole games own split membership.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import MemoryStore, PrefixIdentity, PrefixObservation
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
import managym
from managym.recording import ScalarCapture

Split = Literal["fit", "validation", "test"]
CAPACITIES = (1024, 8192, 256, 16384)
_SPLIT_BUCKETS = tuple(
    sorted(range(20), key=lambda value: sha256(str(value).encode()).digest())
)


def split_game(seed: int) -> Split:
    """Fixed 70/15/15 blocks, stratified over alternating disclosure modes.

    Each aligned block of 40 seeds assigns 14/3/3 games of each disclosure.
    Adding games never moves an existing game between splits.
    """
    rank = _SPLIT_BUCKETS.index((seed // 2) % 20)
    return "fit" if rank < 14 else "validation" if rank < 17 else "test"


def probe_space() -> ObservationSpace:
    return ObservationSpace(
        ObservationConfig(
            max_decision_records=CAPACITIES[0],
            max_decision_reference_details=CAPACITIES[1],
            max_decision_references=CAPACITIES[2],
            max_decision_members=CAPACITIES[3],
        )
    )


@dataclass(frozen=True)
class ProbeRow:
    game: int
    viewer: int
    decision: int
    disclosure: Literal["open", "closed"]
    observation: PrefixObservation
    public: np.ndarray
    analytic: np.ndarray


@dataclass
class Corpus:
    rows: list[ProbeRow]
    labels: list[np.ndarray]
    names: tuple[str, ...]
    games: int = 0
    overflows: int = 0
    capped: int = 0

    def indexes(self, split: Split) -> np.ndarray:
        return np.array(
            [i for i, row in enumerate(self.rows) if split_game(row.game) == split],
            dtype=np.int64,
        )


def analytic_presence(hand: int, unseen: np.ndarray) -> np.ndarray:
    """Hand size times unseen frequency, clipped to a Bernoulli probability."""
    total = float(unseen.sum())
    return np.clip(hand * unseen / max(total, 1), 1e-6, 1 - 1e-6).astype(np.float32)


def public_state(
    obs: managym.Observation, names: tuple[str, ...], prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Only public zones and the viewer-entitled decklist enter this baseline.

    Closed games use the roster-wide prior, never the selected hidden deck.
    Opponent hand identities, including known-hand facts, are deliberately not
    used in this coarse state baseline; this limitation is reported.
    """
    registry = managym.catalog_tokens()["registry_names"]
    positions = {name: i for i, name in enumerate(names)}
    public = np.zeros(len(names), dtype=np.float32)
    for card in obs.opponent_cards:
        if card.zone in (
            managym.ZoneEnum.GRAVEYARD,
            managym.ZoneEnum.EXILE,
            managym.ZoneEnum.COMMAND,
        ):
            name = registry[card.registry_key]
            if name in positions:
                public[positions[name]] += 1
    for permanent in obs.opponent_permanents:
        if not permanent.is_token:
            name = registry[permanent.registry_key]
            if name in positions:
                public[positions[name]] += 1
    deck = np.zeros(len(names), dtype=np.float32)
    if obs.opponent.main_counts is not None:
        for definition, count in obs.opponent.main_counts.items():
            name = registry[definition]
            if name in positions:
                deck[positions[name]] += count
    unseen = np.maximum(
        (deck if obs.opponent.main_counts is not None else prior) - public, 0
    )
    hand = obs.opponent.zone_counts[int(managym.ZoneEnum.HAND)]
    library = obs.opponent.zone_counts[int(managym.ZoneEnum.LIBRARY)]
    state = np.concatenate(
        (
            np.array([hand / 10, library / 60, obs.turn.turn_number / 20], np.float32),
            public / 4,
            deck / 4,
        )
    )
    return state, analytic_presence(hand, unseen)


def collect_games(
    corpus: Corpus,
    count: int,
    out: Path,
    *,
    start: int = 0,
    max_decisions: int = 2000,
    deadline: float = float("inf"),
) -> float:
    """Collect seeded complete heuristic games; discard capped/overflow games.

    Labels are written to a separate file and never inserted in native JSON,
    observation dictionaries, prefix stores or Prepared. Errors other than
    typed native Capacity failures propagate with the failed game seed.
    """
    started = time.monotonic()
    space = probe_space()
    keys = managym.authored_deck_keys("elemental-world-v2")
    roster = [managym.authored_deck_setup("elemental-world-v2", key) for key in keys]
    if len(roster) != 10:
        raise ValueError("probe requires the Elemental ten-deck roster")
    prior = np.array(
        [
            sum(p.decklist.get(name, 0) for p in roster) / len(roster)
            for name in corpus.names
        ],
        np.float32,
    )
    store = MemoryStore(CAPACITIES)
    for number in range(start, start + count):
        seed = 48100 + number
        rng = np.random.default_rng(seed)
        disclosure: Literal["open", "closed"] = "open" if number % 2 == 0 else "closed"
        pair = rng.integers(0, len(roster), size=2)
        players = [
            managym.authored_deck_setup("elemental-world-v2", keys[int(i)])
            for i in pair
        ]
        env = managym.Env(seed=seed, decision_memory=True)
        capture = ScalarCapture(
            seed, decision_memory=True, failure_directory=out / "failures"
        )
        obs, _ = capture.reset(env, managym.GameSetup(players, decklists=disclosure))
        # The heuristic retains per-turn plans and private card references.
        # Each native seat owns its own selector, even when their code is equal.
        heuristics = (managym.Heuristic(), managym.Heuristic())
        reservoirs: list[list[tuple[ProbeRow, np.ndarray]]] = [[], []]
        seen = [0, 0]
        status = "complete"
        failure: str | None = None
        buffers = space.encoder.allocate()
        for decision in range(max_decisions):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "probe wall cap exhausted during collection; partial evidence retained"
                )
            if obs.game_over:
                break
            viewer = env.current_agent_index()
            try:
                # Admission/capacity is checked for every decision, not just
                # reservoir survivors, so long games cannot disappear silently.
                env.encode_observation_into(obs, buffers)
                dense = {k: v[None] for k, v in buffers.items()}
                prefix = store.capture(
                    dense, 0, PrefixIdentity(str(seed), viewer, "root")
                )
            except (managym.AgentError, ValueError) as error:
                if "Capacity" not in str(error) and not str(error).startswith(
                    "observation capacity exceeded:"
                ):
                    raise
                corpus.overflows += 1
                status = "capacity"
                failure = str(error)
                break
            if prefix.record_count:
                seen[viewer] += 1
                slot = int(rng.integers(seen[viewer]))
                if len(reservoirs[viewer]) < 16:
                    slot = len(reservoirs[viewer])
                if slot < 16:
                    state, analytic = public_state(obs, corpus.names, prior)
                    row = ProbeRow(
                        seed,
                        viewer,
                        decision,
                        disclosure,
                        PrefixObservation(
                            {
                                k: v.copy()
                                for k, v in dense.items()
                                if k not in MEMORY_FIELDS
                            },
                            [prefix],
                            CAPACITIES,
                        ),
                        state,
                        analytic,
                    )
                    hand = env.hand_snapshot(1 - viewer).cards
                    label = hand_presence(hand, corpus.names)
                    if slot == len(reservoirs[viewer]):
                        reservoirs[viewer].append((row, label))
                    else:
                        reservoirs[viewer][slot] = (row, label)
            obs, _, terminated, truncated, _ = capture.step(
                env, heuristics[viewer].select_action(obs)
            )
            if terminated or truncated:
                break
        else:
            corpus.capped += 1
            status = "decision-cap"
        corpus.games += 1
        if status == "complete":
            for reservoir in reservoirs:
                for row, label in sorted(reservoir, key=lambda item: item[0].decision):
                    index = len(corpus.rows)
                    corpus.rows.append(row)
                    corpus.labels.append(label)
                    # Retain exact sampled inputs in compact form; all joins
                    # remain admitted production prefixes on reload/materialize.
                    np.savez_compressed(
                        out / f"row-{index:05d}.npz", **row.observation.materialize()
                    )
        with (out / "games.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "seed": seed,
                        "split": split_game(seed),
                        "decklists": disclosure,
                        "pair": [keys[int(i)] for i in pair],
                        "status": status,
                        "failure": failure,
                        "decisions": decision + 1,
                        "eligible": seen,
                        "actions": capture.inputs.actions,
                    }
                )
                + "\n"
            )
        print(
            f"game {seed} {disclosure}: {status}; retained rows {len(corpus.rows)}",
            flush=True,
        )
    return time.monotonic() - started


def new_corpus() -> Corpus:
    roster = [
        managym.authored_deck_setup("elemental-world-v2", key)
        for key in managym.authored_deck_keys("elemental-world-v2")
    ]
    return Corpus(
        [],
        [],
        tuple(
            sorted(
                {
                    name
                    for player in roster
                    for cards in (player.decklist, player.sideboard)
                    for name in cards
                }
            )
        ),
    )


def hand_presence(hand: dict[str, int], names: tuple[str, ...]) -> np.ndarray:
    """Keep every physical hand definition in the declared diagnostic target."""
    missing = hand.keys() - set(names)
    if missing:
        raise ValueError(
            f"probe target vocabulary omits hand definitions: {sorted(missing)}"
        )
    return np.array([hand.get(name, 0) > 0 for name in names], np.float32)


def game_losses(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
    return -(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean(axis=1)


def bootstrap_mean(
    values: np.ndarray, *, seed: int = 719, repeats: int = 1000
) -> tuple[float, float, float]:
    """Resample whole-game means, giving each held-out game equal weight."""
    if not len(values):
        raise ValueError("no games for a probe readout")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(repeats, len(values)))].mean(
        axis=1
    )
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def choose_k(differences: np.ndarray, seconds: dict[int, float]) -> tuple[int, str]:
    """Differences are K4 minus K1 validation losses, paired by game."""
    mean, lo, hi = bootstrap_mean(differences)
    if hi < 0:
        return 4, f"K4-K1 validation loss {mean:.6f} [{lo:.6f}, {hi:.6f}] excludes zero"
    k = min((1, 4), key=lambda k: (seconds[k], k))
    return (
        k,
        f"validation improvement inconclusive ({mean:.6f} [{lo:.6f}, {hi:.6f}]); CPU fit time breaks tie",
    )


def save_corpus(corpus: Corpus, out: Path) -> None:
    """Persist row joins and separate privileged labels, including smoke evidence."""
    np.savez_compressed(
        out / "labels.npz",
        presence=np.stack(corpus.labels),
        names=np.array(corpus.names),
    )
    np.savez_compressed(
        out / "public.npz",
        state=np.stack([r.public for r in corpus.rows]),
        analytic=np.stack([r.analytic for r in corpus.rows]),
    )
    (out / "rows.json").write_text(
        json.dumps(
            [
                {
                    "game": r.game,
                    "viewer": r.viewer,
                    "decision": r.decision,
                    "disclosure": r.disclosure,
                    "split": split_game(r.game),
                }
                for r in corpus.rows
            ],
            indent=2,
        )
        + "\n"
    )
