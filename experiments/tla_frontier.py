"""Task c5fba4: frontier self-play on the 17Lands TLA trophy corpus at two model sizes.

One seed per size trains under the same damped self-play rule, population,
input v9 and wall budget; the large size is the frozen w64 regime with only its
width, depth and heads replaced (see the protocol
`experiments/protocols/tla-frontier-c5fba4-v1.md`). `eval` plays, at every
shared update milestone, each size against the matched first-64 heuristic and
the larger size against the smaller at equal updates; at the endpoints it adds
each size against its own initialization and the two endpoints against each
other, which is equal wall budget only when both runs finished. A failed run's
endpoint is its last stored checkpoint, so its reach is still played and read.
Training exports more often than `eval` plays: the export cadence bounds what a
lost rental can cost, and only milestones and endpoints are evaluated.

The MTG-213 v10 runner (`experiments.tla_pilot_v10`) is executed evidence and
stays unchanged, so its evaluation callback, whose identities name that
pilot's source and artifacts, is mirrored here rather than shared.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Literal

from etude.eval.match import Row, Tables
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Inference,
    Match,
    Matchup,
    Model,
)
from etude.experiment import Checkpoint, ExperimentState, TrainingRun
from etude.jobs import evaluation
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Report, Score
import torch

from experiments import tla_frontier
from experiments.tla_pilot_v10 import HeuristicSpec, Pair, RuntimeWorld
from manabot.env.observation import ObservationSpace
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TOOLCHAIN_ENV, TrainingJob, machine_setup
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.checkpoint import load_checkpoint, local_checkpoint_file
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym

ROOT = Path(__file__).resolve().parents[1]
# `etude eval` and `etude launch` both name the experiment for this file.
EXPERIMENT = Path(__file__).stem
PROTOCOL = ROOT / "experiments/protocols/tla-frontier-c5fba4-v1.json"
REGIME = ROOT / "experiments/regimes/tla-frontier-c5fba4-w64.json"
GAMES_PER_PAIR = 4
# One single-threaded game per worker on a 32-vCPU rental, leaving room for its
# host. At cpu32's USD0.96/hour, these hours and Etude's 15-minute setup
# allowance fit the pass ceiling.
EVALUATION_WORKERS = 28
EVALUATION_HOURS = 1.75
EVALUATION_PASS_USD = Decimal("2")
EVALUATION_ATTEMPTS = 3
HEURISTIC: HeuristicSpec = {
    "kind": "heuristic",
    "max_actions": 64,
    "action_overflow": "first-legal",
}

type SizeLabel = Literal["w64", "w256"]
SMALL: SizeLabel = "w64"
LARGE: SizeLabel = "w256"
SIZES: tuple[SizeLabel, SizeLabel] = (SMALL, LARGE)


# -- Frozen contract ----------------------------------------------------------


class ModelSize(StrictModel):
    hidden_dim: int
    attention_layers: int
    num_attention_heads: int


class Protocol(StrictModel):
    runtime_world: RuntimeWorld
    corpus_sha256: str
    observation_shapes: dict[str, tuple[int, ...]]
    regime_sha256: str
    large: ModelSize
    training_seed: int
    deck_seed: int
    pairs: tuple[Pair, ...]
    milestones: tuple[int, ...]
    # Updates between exports; milestones set what is played, this what a lost
    # rental can lose.
    export_every: int
    game_seconds: float
    max_commands: int
    bootstrap_seed: int
    # Whole-Task paid ceiling; training allocations are per size, evaluation
    # is one cumulative allocation across every pass.
    max_paid_usd: Decimal
    training_usd: Decimal
    training_hours: float
    evaluation_usd: Decimal


def _regime_id(label: SizeLabel) -> str:
    return f"tla-frontier-c5fba4-{label}"


def frozen() -> tuple[dict[SizeLabel, TrainingRegime], Protocol]:
    """Both regimes and the protocol, rejecting drift from the frozen bytes.

    The hash fixes the regime's contents; `tests/eval/test_tla_frontier.py`
    checks what those bytes must declare. Only the world and the tensor shapes
    the encoder derives can still move underneath them.
    """
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes())
    if protocol.runtime_world.model_dump() != managym.world_identity():
        raise ValueError("frontier runtime world differs from the frozen protocol")
    if file_sha256(REGIME) != protocol.regime_sha256:
        raise ValueError("frontier regime differs from frozen protocol")
    small = TrainingRegime.model_validate_json(REGIME.read_bytes())
    if ObservationSpace(small.observation).shapes != protocol.observation_shapes:
        raise ValueError("frontier tensor capacities differ from frozen protocol")
    large = small.model_copy(
        update={
            "id": _regime_id(LARGE),
            "agent": small.agent.model_copy(update=protocol.large.model_dump()),
        }
    )
    return {SMALL: small, LARGE: large}, protocol


def update_cap(regime: TrainingRegime) -> int:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return stage.updates


def export_updates(regime: TrainingRegime, protocol: Protocol) -> tuple[int, ...]:
    """Every update the run exports: the cadence joined with the milestones."""
    cadence = range(protocol.export_every, update_cap(regime), protocol.export_every)
    return tuple(sorted({*cadence, *protocol.milestones}))


def runs(revision: str) -> list[Run]:
    """One capped GPU allocation per size; no automatic retry or budget expansion."""
    regimes, protocol = frozen()
    return [
        run
        for regime in regimes.values()
        for run in TrainingJob(
            regime=regime,
            experiment=EXPERIMENT,
            milestones=export_updates(regime, protocol),
            finalization_seconds=600,
            stop_reserve_seconds=600,
            validate_numerics=True,
        )
        .job(
            repo=ROOT,
            revision=revision,
            max_hours=protocol.training_hours,
            max_usd=protocol.training_usd,
        )
        .runs(protocol.training_seed)
    ]


def _source() -> list[Path]:
    return [
        *sorted((ROOT / "manabot").rglob("*.py")),
        Path(__file__),
        ROOT / "experiments/tla_pilot_v10.py",
        REGIME,
        PROTOCOL,
    ]


def environment(protocol: Protocol) -> Environment:
    return Environment(
        name="tla-frontier-c5fba4-v1",
        # The protocol names the world and corpus, and `frozen` rejects drift.
        files=canonical_sha256(protocol.model_dump(mode="json")),
        config={"purpose": "fixed-corpus-frontier-no-heldout-deck-claim"},
    )


# -- Evaluation schedule ------------------------------------------------------


@dataclass(frozen=True)
class ScheduledComparison:
    """One expected comparison; a missing checkpoint leaves an unplayable slot.

    `update` is the first player's, None for an endpoint its run has not
    reached. Head-to-head rows put the large size first, so their score is the
    large size's. `unreached` marks a milestone past an ended run's endpoint,
    which no later evidence can fill and no missing evidence explains.
    """

    tag: str
    update: int | None
    matchup: Matchup | None
    unreached: bool = False


def size_run(state: ExperimentState, label: SizeLabel) -> TrainingRun | None:
    """The size's one run, which Etude names `<regime id>-seed<seed>`."""
    prefix = f"{_regime_id(label)}-"
    found = [run for run in state.runs if run.name.startswith(prefix)]
    if len(found) > 1:
        raise ValueError(f"frontier admits one {label} run, found {len(found)}")
    return found[0] if found else None


def endpoint(run: TrainingRun | None) -> int | None:
    """An ended run's last stored update, failed or not; None while it may save more."""
    last = run.checkpoints.last if run and run.ended else None
    return int(last.clocks["updates"]) if last else None


def comparison_schedule(
    state: ExperimentState, protocol: Protocol
) -> list[ScheduledComparison]:
    """Every expected comparison, playable or not, in one order for play and analysis.

    Milestones past a wall-bounded endpoint remain expected but unplayable, so
    analysis can report how far each size reached. A milestone that is also an
    endpoint is scheduled once, as a milestone.
    """
    source = _source()
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source, config=HEURISTIC)
    conditions = tuple(pair.model_dump(mode="json") for pair in protocol.pairs)
    runs = {label: size_run(state, label) for label in SIZES}
    endpoints = {label: endpoint(run) for label, run in runs.items()}
    saved: dict[SizeLabel, dict[int, Checkpoint]] = {
        label: {int(c.clocks["updates"]): c for c in run.checkpoints} if run else {}
        for label, run in runs.items()
    }

    def model(label: SizeLabel, update: int | None) -> Model | None:
        found = saved[label].get(update) if update is not None else None
        return found.model(inference) if found else None

    schedule: list[ScheduledComparison] = []

    def expect(
        tag: str,
        label: SizeLabel,
        update: int | None,
        opponent: Model | Code | None,
        name: str,
        *,
        unreached: bool = False,
    ) -> None:
        candidate = model(label, update)
        schedule.append(
            ScheduledComparison(
                tag=tag,
                update=update,
                matchup=Matchup(
                    name=name,
                    players=(candidate, opponent),
                    conditions=conditions,
                    games=GAMES_PER_PAIR,
                    tags=(tag,),
                )
                if candidate is not None and opponent is not None
                else None,
                unreached=unreached,
            )
        )

    def past_endpoint(label: SizeLabel, update: int) -> bool:
        end = endpoints[label]
        return end is not None and update > end

    updates = (0, *protocol.milestones)
    for update in updates:
        for label in SIZES:
            expect(
                f"heuristic-{label}",
                label,
                update,
                heuristic,
                f"{label} u{update} vs heuristic",
                unreached=past_endpoint(label, update),
            )
        if update:
            expect(
                "head-to-head",
                LARGE,
                update,
                model(SMALL, update),
                f"{LARGE} u{update} vs {SMALL} u{update}",
                unreached=any(past_endpoint(label, update) for label in SIZES),
            )
    for label in SIZES:
        end = endpoints[label]
        if end not in updates:
            expect(
                f"heuristic-{label}",
                label,
                end,
                heuristic,
                f"{label} final u{end} vs heuristic",
            )
        expect(
            f"initial-{label}",
            label,
            end,
            model(label, 0),
            f"{label} final u{end} vs {label} u0",
        )
    # Equal wall budget when both finished; analysis reads no scale otherwise.
    expect(
        "endpoint-head-to-head",
        LARGE,
        endpoints[LARGE],
        model(SMALL, endpoints[SMALL]),
        f"{LARGE} final vs {SMALL} final",
    )
    return schedule


def eval(state: ExperimentState) -> list[Job]:
    """Play the available comparisons; Etude owns already-delivered matches."""
    if state.experiment != EXPERIMENT:
        raise ValueError("frontier evaluation requires this experiment's runs")
    _, protocol = frozen()
    matchups = tuple(
        comparison.matchup
        for comparison in comparison_schedule(state, protocol)
        if comparison.matchup is not None
    )
    if not matchups:
        return []
    job = evaluation(
        Benchmark(
            name="tla-frontier-c5fba4-paired-v1",
            environment=environment(protocol),
            matchups=matchups,
        ),
        # Etude discovers this file with runpy; callbacks need an importable
        # module identity even when eval itself came from <run_path>.
        play=tla_frontier.play,
        workers=EVALUATION_WORKERS,
        max_hours=EVALUATION_HOURS,
        # One pass's ceiling across its attempts; the operator keeps the
        # cumulative evaluation spend under `protocol.evaluation_usd`.
        max_usd=EVALUATION_PASS_USD,
        # Etude retries only lost or refused machines, resuming the same run and
        # skipping stored matches, so no game result can be replaced, and a lost
        # rental does not end the watch loop.
        max_attempts=EVALUATION_ATTEMPTS,
        setup=machine_setup(" --extra evaluation"),
    )
    # Etude's default command omits the extra holding its result writer (duckdb).
    command = job.command.replace(
        "uv run python",
        f"{TOOLCHAIN_ENV}\nuv run --locked --extra evaluation python",
        1,
    )
    return [job.model_copy(update={"command": command})]


# -- Evaluation callback ------------------------------------------------------


def _registration(
    player: Model | Code,
    name: Literal["candidate", "opponent"],
    matchup: DeckMatchup,
    observation: ObservationConfig,
) -> tuple[PlayerRegistration, Path | None]:
    setup = matchup.match()
    fingerprints = runtime_fingerprints(match_config=setup)
    common = dict(
        player_id=name,
        display_name=player.name,
        role="challenger" if name == "candidate" else "anchor",
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fingerprints["observation_abi_sha256"],
        action_abi_sha256=fingerprints["action_abi_sha256"],
        matchup_sha256=canonical_sha256(setup.model_dump(mode="json")),
        player_seed_derivation_id="arena-pair-deal-player-v1",
        compute_class_id="policy-cpu-one-thread-one-pass",
    )
    if isinstance(player, Code):
        if player.config != HEURISTIC:
            raise ValueError("heuristic support differs from frontier protocol")
        return PlayerRegistration(
            **common,
            runner_kind="code",
            player_spec=dict(HEURISTIC),
            source_sha256=canonical_sha256(
                {str(p.relative_to(ROOT)): file_sha256(p) for p in _source()}
            ),
        ), None
    path = local_checkpoint_file(player)
    agent, space = load_checkpoint(str(path))
    if space.encoder.config != observation:
        raise ValueError("checkpoint observation differs from frontier protocol")
    digest = file_sha256(path)
    return PlayerRegistration(
        **common,
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "device": "cpu",
            "batch_size": 1,
            "deterministic": False,
            "allow_deck_repetition": True,
        },
        checkpoint_sha256=digest,
        checkpoint_bytes=path.stat().st_size,
        parameter_count=sum(p.numel() for p in agent.parameters()),
        artifact_id=f"c5fba4/{digest}",
    ), path


@contextmanager
def _memory_journal(retained: Path) -> Iterator[Path]:
    """A pair's arena directory in memory, copied to `retained` even on failure.

    The arena syncs every record, which a rented container disk serves at about
    .1 s, and that time counts against the game limit, so a game a slow size
    could finish would fail instead. `experiments.learning_signal` measured this
    and is mirrored here because its source is hashed into retained environments.
    """
    memory = Path("/dev/shm")
    with TemporaryDirectory(dir=memory if memory.is_dir() else None) as journal:
        out_dir = Path(journal) / "deal"
        try:
            yield out_dir
        finally:
            if out_dir.is_dir():
                shutil.copytree(out_dir, retained, dirs_exist_ok=True)


def play(match: Match) -> Tables:
    """One sampled pair/deal, four policy/deck/seat legs, with failure denominators."""
    torch.set_num_threads(1)
    regimes, protocol = frozen()
    # The sizes share observation and population; only their models differ.
    regime = regimes[SMALL]
    source = regime.deck_source
    assert source is not None and source.entry_ids is not None
    pair = Pair.model_validate(match.conditions)
    if pair not in protocol.pairs or match.games != GAMES_PER_PAIR:
        raise ValueError("evaluation conditions differ from frozen schedule")
    matchup = DeckMatchup(
        population_sha256=protocol.corpus_sha256,
        entries=(source.entry_ids[pair.first], source.entry_ids[pair.second]),
        lists=(source.decks[pair.first], source.decks[pair.second]),
    )
    candidate, candidate_path = _registration(
        match.players[0], "candidate", matchup, regime.observation
    )
    opponent, opponent_path = _registration(
        match.players[1], "opponent", matchup, regime.observation
    )
    paths = {
        name: str(path)
        for name, path in (("candidate", candidate_path), ("opponent", opponent_path))
        if path is not None
    }
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="tla-frontier-c5fba4-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    retained = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    with _memory_journal(retained) as out_dir:
        result = evaluate(
            EvaluationRequest(
                key=key,
                candidate=candidate,
                opponent=opponent,
                checkpoint_paths=paths,
                deal_seeds=(pair.deal,),
                authored_matchup=matchup,
                game_seconds=protocol.game_seconds,
                max_commands=protocol.max_commands,
                bootstrap_seed=protocol.bootstrap_seed,
                purpose="tla-frontier-fixed-corpus",
            ),
            out_dir=out_dir,
        )
    by_leg = {row.leg: row for row in result.rows}
    rows: list[Row] = []
    for leg in range(GAMES_PER_PAIR):
        row = by_leg.get(leg)
        valid = row is not None and row.valid
        score = row.score_a if valid else None
        decks = matchup.decks(leg)
        for seat in (0, 1):
            game_seat = (leg + seat) % 2
            rows.append(
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": game_seat,
                    "deck": decks[game_seat],
                    "opponent_deck": decks[1 - game_seat],
                    "deal_seed": pair.deal,
                    "corpus_sha256": protocol.corpus_sha256,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": valid,
                    "attempted": row is not None,
                    "terminated": row.terminated if row else False,
                    "truncated": row.truncated if row else False,
                    "failure": row.failure if row else result.error or "missing game",
                    "seconds": row.game_seconds if row else None,
                }
            )
    return {"games": rows}


def report() -> Report:
    """Progress curves by update; `experiments.tla_frontier_analysis` owns claims."""
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(
                title="Width 64 vs fixed heuristic",
                matches=Matches(tag=f"heuristic-{SMALL}"),
            ),
            Curve(
                title="Width 256 vs fixed heuristic",
                matches=Matches(tag=f"heuristic-{LARGE}"),
            ),
            Curve(
                title="Width 256 vs width 64 at equal updates",
                matches=Matches(tag="head-to-head"),
            ),
        ),
    )
