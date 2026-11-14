"""Measure mana decision frequency, fixed-source outcomes and early learning.

`count` retains exact decks and per-game decisions. `shapes` replays them and
enumerates bounded counterfactuals through the native payment authority;
`summary` aggregates those rows without playing games. Etude
`runs` and `eval` compare unit spending with first-feasible automatic spending.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import json
from pathlib import Path
import subprocess
from typing import Literal, TypedDict, cast

from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
import numpy as np
from pydantic import TypeAdapter, ValidationError

from experiments.elemental_grid import DECKS, PACK
from experiments.elemental_variants import elemental_recipe
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    benchmark_jobs,
    scheduled_matchups,
)
from manabot.env.env import Env
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import file_sha256
from manabot.sim.anchors import RandomMatchupPlayer
from manabot.sim.heuristic import HeuristicPlayer
from manabot.training.decks import UniformDecks
from manabot.training.regime import TrainingRegime, TrainSelfPlay
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


def _new_evidence_dir(out: Path) -> None:
    """Never silently replace an earlier exploratory run's evidence."""
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out")


def _provenance() -> dict[str, str]:
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_sha256": file_sha256(Path(managym._managym.__file__)),
        "source_sha256": file_sha256(Path(__file__)),
    }


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


def _in_spending_mode(job: Job, arm: Literal["A", "B"]) -> Job:
    """Pin the arm's native spending mode in the command; B spends automatically."""
    return job.model_copy(
        update={
            "command": f"export MANAGYM_EXPERIMENT_AUTOSPEND={int(arm == 'B')}\n"
            + job.command,
            "max_attempts": 1,
        }
    )


_ARMS: tuple[Literal["A", "B"], ...] = ("A", "B")


def runs(revision: str, updates: int = 32, hours: float = 1.5) -> list[Run]:
    """Two seeds per arm on the current Elemental recipe; automatic pool spending only."""
    result: list[Run] = []
    for arm in _ARMS:
        regime = elemental_recipe(f"mana-spending-{arm}", updates)
        regime.wall_seconds = hours * 3600 - 1200
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        stage.execution.wall_seconds = regime.wall_seconds
        job = TrainingJob(
            regime=TrainingRegime.model_validate(regime.authored_dump()),
            experiment="mana_spending",
            finalization_seconds=180,
        ).job(repo=ROOT, revision=revision, max_hours=hours, max_usd=Decimal("3"))
        result.extend(_in_spending_mode(job, arm).runs(34701, 34702))
    return result


def eval(state: ExperimentState) -> list[Job]:
    """Replay-checked heuristic scores, each arm retaining its native spending mode."""
    jobs: list[Job] = []
    for arm in _ARMS:
        matchups = tuple(
            matchup
            for run in state.runs
            if run.final is not None and f"regime:mana-spending-{arm}" in run.final.tags
            for matchup in scheduled_matchups(run, ELEMENTAL_PAIRS)
        )
        jobs += [
            _in_spending_mode(job, arm)
            for job in benchmark_jobs(
                matchups, f"mana-spending-{arm}", ELEMENTAL_PAIRS, hours=1.5, usd=4
            )
        ]
    return jobs


@dataclass(frozen=True)
class ShapeCounts:
    """Validated projection of the native offline diagnostic."""

    spend_offers: list[int]
    cost: int
    cast: bool
    saved_obligations: int
    fixed_sources: int
    activated_sources: int
    untapped_outcomes: int
    pool_outcomes: int
    source_outcomes: int
    receipt_outcomes: int
    outcomes: int
    identity_outcomes: int
    non_redundant_outcomes: int
    non_redundant_identity_outcomes: int
    work: int
    cap_hit: bool


@dataclass(frozen=True)
class FoldedCounts:
    original: int
    expanded: int
    non_redundant_expanded: int
    identity_expanded: int
    non_redundant_identity_expanded: int
    cast_offers: int
    unresolved: int
    capped: int


@dataclass(frozen=True)
class ShapeGame:
    deck_set: str
    deck: str
    policy: Policy
    seed: int
    decisions: int
    spend_offers: list[int]
    casts: int
    payments: list[ShapeCounts]
    folded: list[FoldedCounts]
    replay_verified: bool


def shapes(source: Path, out: Path, cap: int) -> None:
    """Replay the retained decisions; enumerate counterfactuals without acting on them."""
    population = {
        (deck["set"], deck["name"]): Deck(**deck)
        for deck in json.loads((source / "decks.json").read_text())["decks"]
    }
    _new_evidence_dir(out)
    metadata = _provenance() | {
        "games_sha256": file_sha256(source / "games.jsonl"),
        "cap": cap,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (out / "shapes.jsonl").open("w") as output:
        for line in (source / "games.jsonl").read_text().splitlines():
            game = GameCounts(**json.loads(line))
            if not game.replay_verified or game.error:
                raise ValueError("shape measurement requires verified games")
            deck = population[game.deck_set, game.deck]
            env = managym.Env(seed=game.seed, skip_trivial=True)
            raw, _ = env.reset(deck.players())
            env.capture_payment_shapes(cap)
            folded: list[FoldedCounts] = []
            for action in game.actions:
                cast_actions = [
                    i
                    for i, offer in enumerate(raw.action_space.actions)
                    if int(offer.action_type)
                    == int(managym.ActionEnum.PRIORITY_CAST_SPELL)
                ]
                if cast_actions:
                    # One total per counting rule, in FoldedCounts order; a cast
                    # without exactly one immediate payment counts as one offer.
                    expanded = [len(raw.action_space.actions) - len(cast_actions)] * 4
                    unresolved = capped = 0
                    for i in cast_actions:
                        fork = env.clone_env()
                        fork.capture_payment_shapes(cap)
                        fork.step(i)
                        alternatives = [
                            ShapeCounts(**row)
                            for row in json.loads(fork.take_payment_shapes_json())
                            if row["cast"]
                        ]
                        if len(alternatives) != 1:
                            unresolved += 1
                            outcomes = (1, 1, 1, 1)
                        else:
                            payment = alternatives[0]
                            capped += payment.cap_hit
                            outcomes = (
                                payment.outcomes,
                                payment.non_redundant_outcomes,
                                payment.identity_outcomes,
                                payment.non_redundant_identity_outcomes,
                            )
                        expanded = [
                            total + max(1, n)
                            for total, n in zip(expanded, outcomes, strict=True)
                        ]
                    folded.append(
                        FoldedCounts(
                            len(raw.action_space.actions),
                            *expanded,
                            cast_offers=len(cast_actions),
                            unresolved=unresolved,
                            capped=capped,
                        )
                    )
                raw, _, _, _, _ = env.step(action)
            payments = [
                ShapeCounts(**row) for row in json.loads(env.take_payment_shapes_json())
            ]
            if env.state_digest() != game.final_digest:
                raise ValueError(
                    f"shape capture changed replay: {game.deck}/{game.seed}"
                )
            if len(payments) != game.mana_payments:
                raise ValueError("payment capture and frequency counter disagree")
            if [n for p in payments for n in p.spend_offers] != game.spend_offers:
                raise ValueError("payment capture and observed spend offers disagree")
            if any(p.outcomes == 0 and not p.cap_hit for p in payments):
                raise ValueError(
                    "an actual payment has no enumerated fixed-source outcome"
                )
            row = ShapeGame(
                deck_set=game.deck_set,
                deck=game.deck,
                policy=game.policy,
                seed=game.seed,
                decisions=sum(game.stages.values()),
                spend_offers=game.spend_offers,
                casts=game.casts,
                payments=payments,
                folded=folded,
                replay_verified=True,
            )
            output.write(json.dumps(asdict(row)) + "\n")
            output.flush()
            print(
                f"{game.deck_set} {game.policy} {game.deck} {game.seed}: {len(payments)} payments",
                flush=True,
            )


def _quantiles(values: list[int]) -> list[float] | None:
    """Median, p95 and maximum, each an observed value (higher order statistic)."""
    if not values:
        return None
    return [float(np.quantile(values, q, method="higher")) for q in (0.5, 0.95, 1)]


class _PaymentChoiceSummary(TypedDict):
    """Flat payment costs and capacity for one population and counting rule."""

    D_exact_offers: list[float] | None
    D_certified_over64: int
    D_exact_over64: int
    exact_real_payments: int
    certified_real_payments: int
    D_decisions_per_game: list[float]
    C_decisions_per_game: list[float]
    C_worst_path_exact_offers: list[float] | None


def _payment_choices(
    payments: list[ShapeCounts],
    count_outcomes: Callable[[ShapeCounts], int],
    decisions_without_spend: int,
    games: int,
) -> _PaymentChoiceSummary:
    """Count flat payment decisions; cap hits retain uncertainty, not truncation."""
    counts = [count_outcomes(p) for p in payments]
    exact_counts = [count_outcomes(p) for p in payments if not p.cap_hit]
    real_counts = [n for n in exact_counts if n > 1]
    certain = sum(n > 1 for n in counts)
    possible = sum(n > 1 or p.cap_hit for n, p in zip(counts, payments))
    return {
        "D_exact_offers": _quantiles(real_counts),
        "D_certified_over64": sum(n > 64 for n in counts),
        "D_exact_over64": sum(n > 64 for n in exact_counts),
        "exact_real_payments": len(real_counts),
        "certified_real_payments": certain,
        "D_decisions_per_game": [
            (decisions_without_spend + n) / games for n in (certain, possible)
        ],
        "C_decisions_per_game": [
            (decisions_without_spend + certain) / games,
            (
                decisions_without_spend
                + possible
                + sum(n > 2 or p.cap_hit for n, p in zip(counts, payments))
            )
            / games,
        ],
        "C_worst_path_exact_offers": _quantiles(
            [2 for _ in real_counts] + [n - 1 for n in real_counts if n > 2]
        ),
    }


class _PaymentOverlapSummary(TypedDict):
    """Potential existing-decision overlap, not proof an activation was offered."""

    certified_payments: int
    possible_payments: int
    maximum_saving_per_game: float
    merged_decisions_per_game: list[float]


def _payment_overlap(
    payments: list[ShapeCounts],
    count_outcomes: Callable[[ShapeCounts], int],
    choices: _PaymentChoiceSummary,
    games: int,
) -> _PaymentOverlapSummary:
    """Bound D's duplicate decisions using sources still present at commit.

    A source having abilities does not establish an available activation. The
    merged lower bound assumes every certified overlap already had a decision;
    the upper bound allows no overlap. Uncertain capped additions and overlaps
    cancel together, so subtract only certified overlap from D's lower bound.
    """
    certified = sum(p.activated_sources > 0 and count_outcomes(p) > 1 for p in payments)
    possible = sum(
        p.activated_sources > 0 and (count_outcomes(p) > 1 or p.cap_hit)
        for p in payments
    )
    lower, upper = choices["D_decisions_per_game"]
    return {
        "certified_payments": certified,
        "possible_payments": possible,
        "maximum_saving_per_game": possible / games,
        "merged_decisions_per_game": [lower - certified / games, upper],
    }


def summarize_shapes(source: Path, *, all_payments: bool = False) -> None:
    """Keep incomplete enumeration out of exact offer quantiles."""
    groups: dict[tuple[str, str], list[ShapeGame]] = defaultdict(list)
    adapter = TypeAdapter(ShapeGame)
    for line in (source / "shapes.jsonl").read_text().splitlines():
        try:
            game = adapter.validate_json(line)
        except ValidationError as error:
            # Rows keep the layout of the revision that wrote them; only the
            # current row layout is summarized here.
            revision = json.loads((source / "metadata.json").read_text())["revision"]
            raise SystemExit(
                f"{source}/shapes.jsonl does not match this revision's row layout "
                f"({error.error_count()} errors); it was written at {revision}. "
                "Summarize it from that revision, or use rows written by this one "
                "(published: current/non-redundant-shapes)."
            ) from None
        groups[game.deck_set, game.policy].append(game)
    summaries: list[dict[str, object]] = []
    for (deck_set, policy), games in groups.items():
        payments = [p for g in games for p in g.payments]
        casts = [p for p in payments if p.cast]
        exact = [p for p in casts if not p.cap_hit]
        real = [p for p in exact if p.outcomes > 1]
        decisions = sum(g.decisions for g in games)
        spend = [n for g in games for n in g.spend_offers]
        cast_spend = sum(len(p.spend_offers) for p in casts)
        folded = [f for g in games for f in g.folded]
        folded_exact = [f for f in folded if not f.unresolved and not f.capped]
        # Cast payments stop prompting per unit under C-E; a capped row may or
        # may not hold a real choice, which separates the lower and upper counts.
        without_cast_spend = decisions - cast_spend
        measurements: tuple[
            tuple[str, Callable[[ShapeCounts], int], Callable[[FoldedCounts], int]], ...
        ] = (
            ("capability", lambda p: p.outcomes, lambda f: f.expanded),
            ("identity", lambda p: p.identity_outcomes, lambda f: f.identity_expanded),
            (
                "non_redundant_capability",
                lambda p: p.non_redundant_outcomes,
                lambda f: f.non_redundant_expanded,
            ),
            (
                "non_redundant_identity",
                lambda p: p.non_redundant_identity_outcomes,
                lambda f: f.non_redundant_identity_expanded,
            ),
        )
        rules: dict[str, dict[str, object]] = {}
        for label, count_outcomes, count_folded in measurements:
            choices = _payment_choices(
                casts, count_outcomes, without_cast_spend, len(games)
            )
            rules[label] = {
                "D_exact_offers": choices["D_exact_offers"],
                "D_certified_over64": choices["D_certified_over64"],
                "D_exact_over64": choices["D_exact_over64"],
                "cast_cap_hits": sum(p.cap_hit for p in casts),
                "certified_real_casts": choices["certified_real_payments"],
                "D_decisions_per_game": choices["D_decisions_per_game"],
                "C_decisions_per_game": choices["C_decisions_per_game"],
                "C_worst_path_exact_offers": choices["C_worst_path_exact_offers"],
                "E_exact_offers": _quantiles([count_folded(f) for f in folded_exact]),
                "E_exact_over64": sum(count_folded(f) > 64 for f in folded_exact),
                "E_certified_over64": sum(count_folded(f) > 64 for f in folded),
            }
        capability = cast(dict[str, list[float]], rules["capability"])
        row = {
            "deck_set": deck_set,
            "policy": policy,
            "games": len(games),
            "payments": len(payments),
            "cast_payments": len(casts),
            "cast_cap_hits": sum(p.cap_hit for p in casts),
            "all_cap_hits": sum(p.cap_hit for p in payments),
            "casts_with_activated_sources": sum(p.activated_sources > 0 for p in casts),
            "casts_with_saved_obligations": sum(p.saved_obligations > 0 for p in casts),
            "exact_casts": len(exact),
            "exact_real_casts": len(real),
            "certified_real_casts": capability["certified_real_casts"],
            "exact_untapped_choices": sum(p.untapped_outcomes > 1 for p in exact),
            "exact_receipt_choices": sum(p.receipt_outcomes > 1 for p in exact),
            "exact_identity_loss": sum(p.identity_outcomes > p.outcomes for p in exact),
            "A_decisions_per_game": decisions / len(games),
            "A_offers": _quantiles(spend),
            "B_decisions_per_game": (decisions - len(spend)) / len(games),
            "C_decisions_per_game_lower": capability["C_decisions_per_game"][0],
            "C_decisions_per_game_upper": capability["C_decisions_per_game"][1],
            "D_decisions_per_game_lower": capability["D_decisions_per_game"][0],
            "D_decisions_per_game_upper": capability["D_decisions_per_game"][1],
            "C_worst_path_exact_offers": capability["C_worst_path_exact_offers"],
            "D_exact_offers": capability["D_exact_offers"],
            "D_exact_over64": capability["D_exact_over64"],
            "D_certified_over64": capability["D_certified_over64"],
            "E_decisions_per_game": without_cast_spend / len(games),
            "E_priority_windows": len(folded),
            "E_exact_windows": len(folded_exact),
            "E_unresolved_cast_offers": sum(f.unresolved for f in folded),
            "E_capped_cast_offers": sum(f.capped for f in folded),
            "E_exact_offers": _quantiles([f.expanded for f in folded_exact]),
            "E_original_offers": _quantiles([f.original for f in folded_exact]),
        } | rules
        if all_payments:
            scopes: dict[str, dict[str, object]] = {}
            for scope, selected in (
                ("noncast", [p for p in payments if not p.cast]),
                ("all", payments),
            ):
                selected_spend = sum(len(p.spend_offers) for p in selected)
                scopes[scope] = {
                    "payments": len(selected),
                    "positive_cost_payments": sum(p.cost > 0 for p in selected),
                    "cap_hits": sum(p.cap_hit for p in selected),
                    "positive_cost_cap_hits": sum(
                        p.cost > 0 and p.cap_hit for p in selected
                    ),
                    "spend_decisions": selected_spend,
                    "payments_with_spend": sum(bool(p.spend_offers) for p in selected),
                }
                for label, count_outcomes, _ in measurements:
                    choices = _payment_choices(
                        selected, count_outcomes, decisions - selected_spend, len(games)
                    )
                    scopes[scope][label] = {
                        **choices,
                        "existing_decision_overlap": _payment_overlap(
                            selected, count_outcomes, choices, len(games)
                        ),
                    }
            row["payment_scopes"] = scopes
        summaries.append(row)
        print(json.dumps(row), flush=True)
    filename = "all-payments-summary.json" if all_payments else "summary.json"
    (source / filename).write_text(json.dumps(summaries, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["count", "shapes", "summary"])
    parser.add_argument(
        "--games", type=int, default=8, help="games per deck and policy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/data/mana_spending")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("experiments/data/mana_spending")
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=16384,
        help="native enumeration work limit per payment",
    )
    parser.add_argument(
        "--all-payments",
        action="store_true",
        help="summary: add noncast/all-payment costs in all-payments-summary.json",
    )
    args = parser.parse_args()
    if args.command == "summary":
        summarize_shapes(args.source, all_payments=args.all_payments)
        return
    if args.command == "shapes":
        if args.out == args.source:
            parser.error("shapes requires a separate --out directory")
        if not 2 <= args.cap <= 65536:
            parser.error("cap must be in 2..=65536")
        shapes(args.source, args.out, args.cap)
        return
    if args.games < 1 or args.seed < 0:
        parser.error("games must be positive and seed nonnegative")
    population = decks()
    _new_evidence_dir(args.out)
    provenance = _provenance()
    metadata = {
        "revision": provenance["revision"],
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "world": managym.WORLD_VERSION,
        "native_sha256": provenance["native_sha256"],
        "source_sha256": provenance["source_sha256"],
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
