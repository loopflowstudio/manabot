"""Do mana abilities at priority and during payments change w64 TLA learning or throughput?

Three training seeds each train with taps off and usable. Etude plays each policy
against the first-64 heuristic and compares usable against off at matched updates. Every
point uses the same 16 trophy pairs, crossing seats and decks within each deal.
Run locally with `uv run etude smoke experiments/mana_taps.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import cache
import gzip
import hashlib
import os
from pathlib import Path
import random
from statistics import median
from tempfile import mkdtemp
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord
from etude.eval.match import environment
from etude.eval.records import (
    Benchmark,
    CheckpointPlayer,
    Code,
    Match,
    Seating,
    checkpoint_player,
)
from etude.eval.result import Result, Row
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Report, Score, Series
from etude.spec import Input
from etude.train.tracking import track

from experiments import mana_taps
from experiments.tla_corpus import Corpus
from experiments.tla_frontier import HEURISTIC, learning, model
from manabot.data import load_manifest
from manabot.eval.admission import register_anchor
from manabot.eval.agents import AGENTS, game_players, registration
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import ETUDE_COMPUTE_CLASS
from manabot.training.decks import UniformDecks
from manabot.training.panels import ML
from manabot.training.state.game_errors import GameErrorPolicy
import managym
from managym.recording import GameRecord
from managym.replay import walk

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "experiments/data/mtg213/corpus-v1/corpus.json"
POPULATION = "a5062cc5130a6fdbc4531f4269c610ee18f922fe81c2639b01c8bf8d1488c5ed"
SEEDS = (21501, 21502, 21503)
OBSERVATION = ObservationConfig(
    policy_input_version=11,
    max_facts=512,
    max_cards_per_player=128,
    max_permanents_per_player=96,
    max_actions=64,
    action_overflow="first-legal",
    max_focus_objects=2,
    max_events=32,
)


def corpus_input() -> Input:
    """Etude delivers the manifest object to rentals and verifies the local smoke copy."""
    manifest = load_manifest()
    path = CORPUS.relative_to(ROOT).as_posix()
    entry = manifest.files[path]
    return Input(
        name=CORPUS.name,
        key=f"sha256/{entry.sha256}",
        sha256=entry.sha256,
        size_bytes=entry.bytes,
        uri=CORPUS.as_uri(),
    )


@cache
def corpus() -> UniformDecks:
    """Re-admit the retained text once per process, preserving event identity and order."""
    inputs = os.environ.get("ETUDE_INPUTS_DIR")
    path = Path(inputs) / CORPUS.name if inputs else CORPUS
    trophies = Corpus.model_validate_json(path.read_bytes()).trophies
    admitted = managym.admit_decklists(
        [
            (t.source_id, managym.Decklist(t.decklist.source, t.decklist.filename))
            for t in trophies
        ]
    )
    source = UniformDecks(
        decks=tuple(a.deck for a in admitted),
        entry_ids=tuple(t.source_id for t in trophies),
        sampling="uniform-independent-v1",
    )
    if source.digest != POPULATION:
        raise ValueError(
            "re-admitted TLA population differs from the 1000 training trophies"
        )
    return source


class Arm(StrictModel):
    taps: managym.ManaFloats
    device: Literal["cpu", "cuda"] = "cuda"
    updates: int = 50

    @property
    def name(self) -> str:
        return f"mana-taps-w64-{self.taps}"


def settings(arm: Arm) -> SelfPlay:
    from manabot.training.loops.self_play import SelfPlay

    local = arm.device == "cpu"
    return SelfPlay(
        deck_source=corpus(),
        observation=OBSERVATION.model_copy(update={"mana_floats": arm.taps}),
        model=model("w64"),
        learning=learning("w64"),
        behavior="current-self",
        streams=2 if local else 256,
        transitions=1 if local else 64,
        workers=1 if local else 4,
        learning_seats="both",
        batch_closure="complete-games",
        game_errors=GameErrorPolicy(
            threshold=0.25,
            window=100,
            warmup=100,
            max_game_actions=40000,
            recover_capacity=True,
        ),
        device=arm.device,
        precision="float32" if local else "bf16",
        bucketed_shapes=True,
        checkpoint_every=25,
    )


def train(config: Arm) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    hours: float = 3,
    usd_per_run: float = 2.28,
    updates: int = 50,
    max_attempts: int = 1,
) -> list[TrainingJob]:
    arms = [Arm(taps=taps, updates=updates) for taps in ("off", "usable")]
    given = corpus_input()
    return [
        TrainingJob(
            name=arm.name,
            train=train,
            config=arm,
            smoke=arm.model_copy(update={"device": "cpu", "updates": 1}),
            seeds=SEEDS,
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            inputs=(given,),
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=max_attempts,
        )
        for arm in arms
    ]


class Deals(StrictModel):
    """Each Etude match index selects one pair and its four-leg deal."""

    population: str = POPULATION
    pairs: tuple[tuple[int, int], ...]
    first_seed: int = 385000


def deals() -> Deals:
    rng = random.Random(385)
    return Deals(
        pairs=tuple((rng.randrange(1000), rng.randrange(1000)) for _ in range(16))
    )


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Compare complete seed pairs. Etude smoke supplies one seed per distinct job."""
    names = {Arm(taps=taps).name: taps for taps in ("off", "usable")}
    runs: dict[tuple[int, managym.ManaFloats], TrainingRunSummary] = {}
    for run in state.runs:
        if run.launch not in names or run.seed not in SEEDS or run.seed is None:
            continue
        key = (run.seed, names[run.launch])
        if key in runs:
            raise ValueError(f"multiple training runs for {run.launch} seed {run.seed}")
        if not run.ended:
            return []
        runs[key] = run
    if not runs or any(
        (seed, taps) not in runs for seed, _ in runs for taps in names.values()
    ):
        return []
    if not any(run.checkpoints for run in runs.values()):
        return []
    jobs = [paired_evaluation(runs, raw=False)]
    # Delivered runs identify raw exports by digest in out/. Local smoke
    # supplies only the registered EMA records.
    if any(run.out for run in runs.values()):
        jobs.append(paired_evaluation(runs, raw=True))
    return jobs


def raw_export(run: TrainingRunSummary, ema: CheckpointRecord) -> CheckpointRecord:
    """Seat the raw sibling saved at the EMA's clocks using its own stored digest."""
    update = int(ema.at["updates"])
    member = run.out.file(f"policies/{update:08d}-raw.safetensors")
    return ema.model_copy(
        update={
            "role": "raw",
            "file": member.key,
            "tags": (),
            "artifact": ArtifactRef(
                uri=member.uri, sha256=member.sha256, size_bytes=member.size_bytes
            ),
        }
    )


def paired_evaluation(
    runs: dict[tuple[int, managym.ManaFloats], TrainingRunSummary], *, raw: bool
) -> EvalJob:
    exports: list[CheckpointRecord] = []
    checkpoints: dict[tuple[int, managym.ManaFloats], dict[int, CheckpointPlayer]] = {}
    for key, run in runs.items():
        records: Sequence[CheckpointRecord] = run.checkpoints
        if raw:
            records = [raw_export(run, c) for c in records]
            exports.extend(records)
        checkpoints[key] = {
            int(c.at["updates"]): checkpoint_player(c.ref) for c in records
        }
    conditions = deals()
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config=HEURISTIC
    )
    seatings: list[Seating] = []

    def compare(
        name: str,
        tag: str,
        candidate: CheckpointPlayer,
        opponent: CheckpointPlayer | Code,
    ) -> None:
        seatings.append(
            Seating(
                name=name,
                tags=(f"raw-{tag}" if raw else tag,),
                players=(candidate, opponent),
                config=conditions,
                games=len(conditions.pairs),
            )
        )

    for seed in sorted({seed for seed, _ in runs}):
        off, usable = checkpoints[seed, "off"], checkpoints[seed, "usable"]
        for update in sorted(off.keys() & usable.keys()):
            for label, player in (("off", off[update]), ("usable", usable[update])):
                compare(
                    f"seed {seed} {label} u{update} vs heuristic",
                    "heuristic",
                    player,
                    heuristic,
                )
            compare(
                f"seed {seed} usable vs off u{update}",
                "usable-vs-off",
                usable[update],
                off[update],
            )
    return EvalJob(
        play=mana_taps.play,
        agents=AGENTS,
        checkpoints=exports,
        tournament=Benchmark(
            name="mana-taps-raw" if raw else "mana-taps-paired",
            environment=environment(
                name="mana-taps",
                files=[Path(__file__)],
                config=conditions.model_dump(mode="json"),
            ),
            seatings=tuple(seatings),
        ),
        setup=machine_setup(" --extra artifacts --extra evaluation"),
        inputs=(corpus_input(),),
        extras=("artifacts", "evaluation"),
        max_hours=1,
        max_usd=Decimal("1.50"),
        max_attempts=1,
    )


@dataclass
class FloatDecisions:
    surfaced: int = 0
    float_offered: int = 0
    float_chosen: int = 0


def spot_check(record: GameRecord, leg: int, replay_sha256: str) -> list[Row]:
    """Replay one scored leg. Count offers inside the policy's first-legal limit."""
    counts = [FloatDecisions(), FloatDecisions()]
    replay = walk(record, capture_cards=True)
    for decision in replay:
        space = decision.observation.action_space
        actions = space.actions[: OBSERVATION.max_actions]
        row = counts[int(decision.observation.agent.player_index)]
        row.surfaced += 1
        for index, action in enumerate(actions):
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY:
                row.float_offered += index + 1 < len(actions)
                row.float_chosen += decision.chosen > index
                break
    if replay.stop is None or replay.stop.status != "complete":
        raise ValueError(f"mana spot check replay failed: {replay.stop}")
    exposure = replay.cards().capture.exposure
    if exposure is None or exposure.mana is None:
        raise ValueError("native replay lacks mana expiry measurements")
    return [
        {
            "seat": (leg + game_seat) % 2,
            "game_seat": game_seat,
            "game": leg,
            "deal_seed": replay.game.seed,
            "replay_sha256": replay_sha256,
            **asdict(count),
            **mana.model_dump(),
        }
        for game_seat, (count, mana) in enumerate(
            zip(counts, exposure.mana, strict=True)
        )
    ]


def play(match: Match[Deals], agents: Sequence[object]) -> Result:
    conditions = Deals.model_validate(match.config)
    if match.index is None or conditions != deals():
        raise ValueError("expected an indexed match on the fixed trophy pairs")
    source = corpus()
    assert source.entry_ids is not None
    first, second = conditions.pairs[match.index]
    deal = conditions.first_seed + match.index
    matchup = DeckMatchup(
        population_sha256=POPULATION,
        entries=(source.entry_ids[first], source.entry_ids[second]),
        lists=(source.decks[first], source.decks[second]),
    )
    a, b = match.players
    if not isinstance(a, CheckpointPlayer):
        raise ValueError("the candidate must be a checkpoint")
    candidate = registration(
        a,
        name="candidate",
        role="challenger",
        match=matchup.match(),
        content_suite=matchup.content_suite,
    )
    paths = {"candidate": str(a.checkpoint.path)}
    if isinstance(b, CheckpointPlayer):
        opponent = registration(
            b,
            name="opponent",
            role="anchor",
            match=matchup.match(),
            content_suite=matchup.content_suite,
        )
        paths["opponent"] = str(b.checkpoint.path)
    else:
        opponent = register_anchor(
            candidate, "heuristic", player_id="opponent"
        ).model_copy(update={"player_spec": dict(HEURISTIC)})
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="mana-taps-paired",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=opponent.identity_sha256,
        evaluation_compute_envelope_id=ETUDE_COMPUTE_CLASS,
    )
    root = (
        Path(os.environ.get("ETUDE_OUTPUT_DIR") or ROOT / ".runs/mana-taps")
        / "arena"
        / match.key
    )
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(dir=root)) / "evaluation"
    result = evaluate(
        EvaluationRequest(
            key=key,
            candidate=candidate,
            opponent=opponent,
            checkpoint_paths=paths,
            deal_seeds=(deal,),
            authored_matchup=matchup,
            game_seconds=600,
            purpose="mana-taps-tla",
        ),
        out_dir=directory,
        agents=dict(
            zip(
                ("candidate", "opponent"),
                game_players(match.players, agents),
                strict=True,
            )
        ),
    )
    rows: list[Row] = []
    by_leg = {r.leg: r for r in result.rows}
    complete = deal in result.complete_deals()
    for leg in range(4):
        row = by_leg.get(leg)
        score = row.score_a if row and complete else None
        for seat in (0, 1):
            game_seat = (leg + seat) % 2
            rows.append(
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": game_seat,
                    "deck": matchup.decks(leg)[game_seat],
                    "deal_seed": deal,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": bool(row and row.valid),
                    "complete_deal": complete,
                    "failure": row.failure if row else result.error or "missing leg",
                }
            )
    # The first fixed pair supplies four matched spot checks per seating.
    # Keep each replay with its Etude match so a worker's scratch path is never required.
    spots: list[Row] = []
    blobs: list[bytes] = []
    if match.index == 0 and complete:
        for path in sorted({r.trace_path for r in result.rows}):
            with (directory / path).open("rb") as shard:
                for line in shard:
                    record = GameRecord.model_validate_json(line)
                    leg = record.context["evaluation"]["leg"]
                    if record.sha256 != (by_leg[leg].model_extra or {})["trace_sha256"]:
                        raise ValueError("spot check differs from scored replay")
                    body = gzip.compress(line, mtime=0)
                    spots.extend(
                        spot_check(record, leg, hashlib.sha256(body).hexdigest())
                    )
                    blobs.append(body)
    return Result(tables={"games": rows, "mana_spot_checks": spots}, blobs=blobs)


def report() -> Report:
    return Report(
        clock="updates",
        score=Score(table="games", value="score", reference=0.5, unit="games"),
        evals=(
            Curve(title="EMA against the heuristic", matches=Matches(tag="heuristic")),
            Curve(
                title="EMA, usable against off", matches=Matches(tag="usable-vs-off")
            ),
            Curve(
                title="Raw learner against the heuristic",
                matches=Matches(tag="raw-heuristic"),
            ),
            Curve(
                title="Raw learners, usable against off",
                matches=Matches(tag="raw-usable-vs-off"),
            ),
        ),
        perf=tuple(
            Series(title=title, keys=(key,), unit=unit)
            for title, key, unit in (
                ("Decisions per completed game", "decisions_per_game", "rows/game"),
                ("Forced decisions skipped", "skipped_share", "fraction"),
                (
                    "Priority taps per surfaced decision",
                    "priority_tap_rate",
                    "fraction",
                ),
                ("Training rows per second", "rows_per_second", "rows/s"),
                ("Updates per training hour", "updates_per_hour", "updates/h"),
                ("Games per training hour", "games_per_hour", "games/h"),
            )
        ),
        ml=ML,
    )


def random_deals() -> None:
    """Repeat the original 40 CPU deals and each post-deck-selection RNG stream."""
    rng = random.Random(385)
    decks = rng.sample(corpus().decks, 24)
    deals: list[tuple[list[managym.PlayerConfig], object]] = []
    # MTG-385's original off policy shared the RNG with deck selection.
    # Preserve its positions (32-bit draws from seed 385) as the policy changes.
    deal_offsets = (
        25,
        210,
        783,
        1191,
        1446,
        1703,
        1974,
        2699,
        3234,
        3771,
        4244,
        4611,
        4896,
        5347,
        5646,
        6055,
        6196,
        6382,
        6661,
        6866,
        7502,
        7900,
        8157,
        8434,
        8585,
        9155,
        9454,
        9778,
        9935,
        10251,
        10787,
        11169,
        12346,
        12735,
        13354,
        13906,
        14189,
        14476,
        15223,
        15529,
    )
    for offset in deal_offsets:
        rng.seed(385)
        for _ in range(offset):
            rng.getrandbits(32)
        players = [deck.player() for deck in rng.sample(decks, 2)]
        deals.append((players, rng.getstate()))
    for payment, mode in (
        ("manual", "off"),
        ("manual", "all"),
        ("manual", "usable"),
        ("auto", "off"),
        ("auto", "usable"),
    ):
        steps = skipped = offered = chosen = 0
        start = perf_counter()
        for game, (players, state) in enumerate(deals):
            rng.setstate(state)
            env = managym.Env(seed=1000 + game)
            obs, _ = env.reset(
                managym.GameSetup(
                    players, mana_floats=(mode, mode), mana_payment=(payment, payment)
                )
            )
            for _ in range(5000):
                space = obs.action_space
                actions = space.actions
                index = rng.randrange(len(actions))
                if space.action_space_type == managym.ActionSpaceEnum.PRIORITY:
                    passed = next(
                        i
                        for i, action in enumerate(actions)
                        if action.action_type
                        == managym.ActionEnum.PRIORITY_PASS_PRIORITY
                    )
                    offered += passed + 1 < len(actions)
                    chosen += index > passed
                obs, _, done, truncated, _ = env.step(index)
                steps += 1
                if truncated:
                    raise ValueError(f"truncated seed {1000 + game}")
                if done:
                    break
            else:
                raise ValueError(f"step cap at seed {1000 + game}")
            skipped += env.skip_trivial_count()
        elapsed = perf_counter() - start
        print(
            f"{payment=} {mode=} {steps=} {skipped=} {offered=} {chosen=} {elapsed=:.3f}",
            flush=True,
        )


def priority_check_timing() -> None:
    """Pair refresh times on identical board positions, including observation construction."""
    decks = [
        managym.PlayerConfig(name, {"Gray Ogre": 8, "Mountain": 32})
        for name in ("a", "b")
    ]
    for spend in (False, True):
        differences = []
        for seed in range(40):
            times = {}
            for mode in ("all", "usable") if seed % 2 else ("usable", "all"):
                env = managym.Env(seed=1000 + seed)
                env.reset(managym.GameSetup(decks, mana_floats=(mode, mode)))
                env.scenario_clear_hand(0)
                # A legal land drop keeps even the no-spend fixture at priority.
                env.scenario_force_card_in_hand(0, "Mountain")
                if spend:
                    env.scenario_force_card_in_hand(0, "Gray Ogre")
                for _ in range(3):
                    env.scenario_force_battlefield(0, "Mountain", True)
                env.scenario_refresh()
                start = perf_counter()
                obs = env.scenario_refresh()
                times[mode] = perf_counter() - start
                assert sum(
                    a.action_type == managym.ActionEnum.TAP_FOR_COST
                    for a in obs.action_space.actions
                ) == (3 if spend or mode == "all" else 0)
            differences.append((times["usable"] - times["all"]) * 1e6)
        print(
            f"{spend=} paired median usable minus everywhere {median(differences):.2f} us"
        )


if __name__ == "__main__":
    random_deals()
    priority_check_timing()
