"""Head-to-head arena between frozen trained policies on shared deals.

Every experiment so far scored a policy against scripted greedy. This runner
plays a built-in or named (`TOURNAMENTS`) tournament of checkpoints against each other
and against greedy on shared deals, so the two measures can be compared
directly. See `experiments/head-to-head.md` for the protocol and reading rule.

Principal types and entry points:

- `Tournament` declares the design (the built-in constants remain the default):
  who plays, identified by checkpoint bytes, and which pairs meet.
- `admit` verifies every checkpoint's bytes and loads it through the ordinary
  loader, which rejects a checkpoint bound to a different world, content pack
  or input schema. Admission is all-or-nothing and happens before any game.
- `run` plays the schedule through the existing arena (`play_cell`), one unit
  (pairing, deal) at a time: four games covering both deck assignments and
  both seats, or both same-deck matchups and seats in mirror mode. It owns
  the wall-time cap, the disk floor and `status.json`.
- `report` projects retained rows into per-pairing scores, per-entrant scores
  against greedy and the reference opponent, ratings, and the paired contrasts
  the protocol names. It is safe to run while games are still being played.

Data flow. Each unit is one directory under `units/` written by a single worker
process; `rows.json` appears only after its four games and their replay check
finish, so a present `rows.json` is a complete unit. Nothing is overwritten:
a rerun plays only units that have no directory yet.

Pairing. Units are scheduled deal by deal across all pairings, so every pairing
uses the same deal seeds and an interrupted run still has every pairing at
nearly the same depth. Policy sampling seeds derive from (pairing, deal,
player), so only the deal is shared between pairings, not the action noise.
"""

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, replace
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from types import FrameType
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass

from manabot.arena import players
from manabot.arena.match import (
    SELECTED_SUITE,
    MatchupMode,
    play_cell,
    selected_decks,
    selected_match,
)
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.rating import bootstrap_population, fit_population
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.teacher1_evidence import runtime_fingerprints
from manabot.training.execution import atomic_json
from manabot.training.monitor_evaluation import ArenaRow
from managym import WORLD_VERSION

NAME = "head-to-head"
GREEDY = "scripted-greedy"
MIRROR_DEAL_SEEDS = tuple(range(1_913_131_000, 1_913_131_100))
MIRROR_SMOKE_SEEDS = (1_913_131_100, 1_913_131_101)
TRAINING_CAVEAT = (
    "Every trained model except etu125-mirrors-10k was trained only on Lessons "
    "vs Allies: same-deck games are outside its training matchups. "
    "small-0 is untrained; scripted greedy is a code anchor. "
    "These are frozen-checkpoint comparisons, not training-seed uncertainty."
)

# ----------------------------------------------------------------------------
# Frozen design
# ----------------------------------------------------------------------------


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class Entrant:
    """One player in a Tournament.

    A checkpoint entrant is identified by `sha256` and `bytes`; `file` is only
    where the runner looks for those bytes under the models directory. The
    scripted anchor has no checkpoint identity.
    """

    id: str
    label: str
    origin: str
    file: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    training_seed: int | None = None
    updates: int | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", self.id):
            raise ValueError("entrant id must be a safe directory component")
        if not self.label.strip() or not self.origin.strip():
            raise ValueError("entrant label and origin must be nonempty")
        identity = (
            self.file,
            self.sha256,
            self.bytes,
            self.training_seed,
            self.updates,
        )
        if self.id == GREEDY:
            if any(value is not None for value in identity):
                raise ValueError("scripted greedy cannot carry checkpoint fields")
        else:
            if (
                self.file is None
                or self.sha256 is None
                or self.bytes is None
                or self.training_seed is None
                or self.updates is None
            ):
                raise ValueError("checkpoint entrant requires all identity fields")
            if (
                not self.file
                or Path(self.file).name != self.file
                or self.file in (".", "..")
            ):
                raise ValueError("checkpoint file must be a basename under models")
            if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
                raise ValueError("checkpoint sha256 must be 64 lowercase hex digits")
            if self.bytes <= 0 or self.training_seed < 0 or self.updates < 0:
                raise ValueError("invalid checkpoint size, seed or updates")


# The candidate frozen-checkpoint opponent: the longest-trained policy we have.
REFERENCE = "small-100k"

ENTRANTS: tuple[Entrant, ...] = (
    Entrant(GREEDY, "scripted greedy", "code anchor"),
    Entrant(
        "small-0",
        "small, untrained",
        "etu103-recovery-v3-science-0 initial weights",
        "small-0-raw.pt",
        "dc5cdbb7da7af4aacaf3b4d3f6ba36756c2ff879c38fef8c7b4df5142419e9c2",
        823445,
        10351,
        0,
    ),
    Entrant(
        "small-26k",
        "small, 26,000 updates",
        "etu103-recovery-v3-science-0 final raw",
        "small-26000-raw.pt",
        "3a3c65cbbc90231d9f96c5619b9bbc1b8ea88ac4711e1b1752fe2e99e33cb0da",
        822901,
        10351,
        26000,
    ),
    Entrant(
        "small-63k",
        "small, 63,000 updates",
        "etu126-small-63000 final raw",
        "small-63000-raw.pt",
        "4c7efdc71c4068691e4bae4d3b42e5b45a4854e98a7a0c87bee1468206ab3f2d",
        822901,
        10351,
        63000,
    ),
    Entrant(
        "small-100k",
        "small, 100,000 updates",
        "etu126-small-100000 final raw",
        "small-100000-raw.pt",
        "23e295c3873a0eb251c28ea12577073415ca6120e72727f339b9afbb7d85af82",
        822901,
        10351,
        100000,
    ),
    Entrant(
        "small-100k-ema",
        "small, 100,000 updates, averaged weights",
        "etu126-small-100000 final EMA",
        "small-100000-ema.pt",
        "060fa8286c26ea71e0f44373a0dc35d87e9e9c469f8c9f651965efc7f3e930c7",
        823605,
        10351,
        100000,
    ),
    Entrant(
        "large-15k6",
        "large, 15,600 updates",
        "etu103-recovery-v3-science-1 final raw",
        "large-15600-raw.pt",
        "b7a52466b1d613ae3daa2c243bc0dc24bb71289d039717fb1cbde9fba07638d6",
        67358871,
        10351,
        15600,
    ),
    Entrant(
        "etu118-10k",
        "ETU-118 mini baseline, 10,000 updates",
        "etu118-mini-20261007-1 seed 11851 update-10000 raw",
        "etu118-11851-10000-raw.pt",
        "2475f3e28d0097ef79c8833cc90ea81eb6808b1727b1d68765dc16f72eae0fe4",
        822681,
        11851,
        10000,
    ),
    Entrant(
        "etu125-cross-10k",
        "ETU-125 cross-deck arm, 10,000 updates",
        "etu125-10k seed 12551 cross-balanced endpoint raw",
        "etu125-12551-cross-10000-raw.pt",
        "19e548468283880fa5d7155b1675850c4c75941b74f7ccb445e7d89953e4408c",
        822337,
        12551,
        10000,
    ),
    Entrant(
        "etu125-mirrors-10k",
        "ETU-125 mirror-inclusive arm, 10,000 updates",
        "etu125-10k seed 12551 mirrors-balanced endpoint raw",
        "etu125-12551-mirrors-10000-raw.pt",
        "99fff198fe96cede65500b66795b8ef10b7809c0745103639b4fd6d9ec2568f4",
        822337,
        12551,
        10000,
    ),
)

# Entrants that meet each other in a full round robin: the small-model ladder,
# the large model and the mini baseline.
CORE: tuple[str, ...] = (
    "small-26k",
    "small-63k",
    "small-100k",
    "large-15k6",
    "etu118-10k",
)
# Meetings outside the core that answer a named question.
EXTRA: tuple[tuple[str, str], ...] = (
    ("small-100k-ema", "small-100k"),  # averaged against raw weights
    ("small-0", REFERENCE),  # untrained anchor on the head-to-head scale
    ("etu125-cross-10k", REFERENCE),
    ("etu125-mirrors-10k", REFERENCE),
    ("etu125-mirrors-10k", "etu125-cross-10k"),  # the ETU-125 arms, same seed
    ("etu118-10k", "etu125-cross-10k"),  # two 10,000-update self-play recipes
)


# Entrant pairs whose greedy-score difference is compared with their direct
# meeting. Each pair is also scheduled above.
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("small-63k", "small-26k"),
    ("small-100k", "small-63k"),
    ("small-100k", "small-26k"),
    ("small-100k", "large-15k6"),
    ("small-100k", "etu118-10k"),
    ("large-15k6", "etu118-10k"),
    ("small-100k-ema", "small-100k"),
    ("etu125-mirrors-10k", "etu125-cross-10k"),
    ("etu118-10k", "etu125-cross-10k"),
)


class Tournament(BaseModel):
    """Validated experiment design; checkpoint paths remain relative to --models.

    Greedy is the fixed code/rating anchor. Every checkpoint meets it; core
    adds a round robin and extra adds explicitly named meetings.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    entrants: tuple[Entrant, ...]
    core: tuple[str, ...]
    extra: tuple[tuple[str, str], ...]
    contrasts: tuple[tuple[str, str], ...]
    reference: str
    training_caveat: str

    def pairings(self) -> tuple[tuple[str, str], ...]:
        anchors = tuple((e.id, GREEDY) for e in self.entrants if e.id != GREEDY)
        core = tuple(
            (a, b)
            for i, a in enumerate(self.core)
            for b in self.core[i + 1 :]
            if GREEDY not in (a, b)
        )
        return (*anchors, *core, *self.extra)

    @model_validator(mode="after")
    def validate_design(self) -> "Tournament":
        ids = {entrant.id for entrant in self.entrants}
        if len(ids) != len(self.entrants):
            raise ValueError("duplicate entrant ids")
        if GREEDY not in ids or len(ids) < 2:
            raise ValueError("tournament requires scripted-greedy and a checkpoint")
        if self.reference not in ids or self.reference == GREEDY:
            raise ValueError("reference must name a checkpoint entrant")
        if len(set(self.core)) != len(self.core) or not set(self.core) <= ids:
            raise ValueError("core contains duplicate or unknown ids")
        pairs = self.pairings()
        for group in (pairs, self.contrasts):
            if any(a not in ids or b not in ids or a == b for a, b in group):
                raise ValueError("pair contains unknown ids or a self meeting")
            if len({frozenset(pair) for pair in group}) != len(group):
                raise ValueError("a pairing is scheduled twice")
        scheduled = {frozenset(pair) for pair in pairs}
        if any(
            frozenset(pair) not in scheduled or GREEDY in pair
            for pair in self.contrasts
        ):
            raise ValueError("contrasts require a scheduled checkpoint meeting")
        if not self.training_caveat.strip():
            raise ValueError("training caveat must be nonempty")
        return self


def default_tournament() -> Tournament:
    return Tournament(
        entrants=ENTRANTS,
        core=CORE,
        extra=EXTRA,
        contrasts=CONTRASTS,
        reference=REFERENCE,
        training_caveat=TRAINING_CAVEAT,
    )


def size_floor_grid() -> Tournament:
    """Greedy and three final raw grid checkpoints in a full round robin."""
    return Tournament(
        entrants=(
            Entrant(id=GREEDY, label="scripted greedy", origin="code anchor"),
            Entrant(
                id="w64-floor010",
                label="width 64, floor 0.01",
                origin="etu103-recovery-v3-science-0 final raw",
                file="small-26000-raw.pt",
                sha256="3a3c65cbbc90231d9f96c5619b9bbc1b8ea88ac4711e1b1752fe2e99e33cb0da",
                bytes=822901,
                training_seed=10351,
                updates=26000,
            ),
            Entrant(
                id="w64-floor003",
                label="width 64, floor 0.003",
                origin="size-floor-grid-20261009b-0 final raw",
                file="w64-d2-floor003-26000-raw.pt",
                sha256="b838dae96342fc97d91760a93700be043a36d56978abf6c7cb2b51c7b43d3ecc",
                bytes=822901,
                training_seed=10351,
                updates=26000,
            ),
            Entrant(
                id="w32-floor010",
                label="width 32, floor 0.01",
                origin="size-floor-grid-20261010c-0 final raw",
                file="w32-d2-floor010-26000-raw.pt",
                sha256="d405b49d66656b7df0ea73684ac129f760406a94767b5fba7368be77035b7e5d",
                bytes=277941,
                training_seed=10351,
                updates=26000,
            ),
        ),
        core=(GREEDY, "w64-floor010", "w64-floor003", "w32-floor010"),
        extra=(),
        contrasts=(
            ("w64-floor003", "w64-floor010"),
            ("w32-floor010", "w64-floor010"),
        ),
        reference="w64-floor010",
        training_caveat=(
            "These checkpoints trained on cross-deck Lessons vs Allies; mirrors "
            "are outside their training matchups. All three checkpoints use "
            "training seed 10351 and 26,000 updates, not independent training "
            "replicates. Equal updates do not imply equal samples or cost. "
            "Scripted greedy is a code anchor. Intervals describe "
            "frozen-checkpoint deal uncertainty, not training-seed uncertainty."
        ),
    )


# Tournaments `--tournament` selects by name; omitting it plays the default.
TOURNAMENTS: dict[str, Callable[[], Tournament]] = {
    "size-floor-grid": size_floor_grid,
}


def pairings() -> tuple[tuple[str, str], ...]:
    """Frozen default meetings, preserving their orientation and order."""
    return default_tournament().pairings()


# A namespace no monitoring or study cohort uses. 100 deals x 4 legs = 400
# games per pairing.
DEAL_SEEDS: tuple[int, ...] = tuple(range(1_913_260_000, 1_913_260_100))
GAME_SECONDS = 120.0
MAX_COMMANDS = 10_000
WALL_SECONDS = 6 * 3600.0
MIN_FREE_GIB = 40.0
BOOTSTRAP_REPLICATES = 2000
RATING_REPLICATES = 300
BOOTSTRAP_SEED = 1913


# ----------------------------------------------------------------------------
# Admission
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Admitted:
    """Registrations bound to this runtime, and where checkpoint bytes live."""

    key: ArenaKey
    registrations: dict[str, PlayerRegistration]
    checkpoint_paths: dict[str, str]


def admit(
    models: Path,
    entrants: Sequence[Entrant] = ENTRANTS,
    *,
    matchup_mode: MatchupMode = "cross",
) -> Admitted:
    """Verify and load every entrant; raise before any game if one is unfit.

    The ordinary loader validates the checkpoint's world, content manifest and
    input schema against this runtime. An entrant that fails is never forced
    in: the whole run stops, because the tournament is the frozen design.
    """
    match = selected_match()
    paths: dict[str, str] = {}
    parameters: dict[str, int] = {}
    identities: dict[str, str] | None = None
    for entrant in entrants:
        if entrant.file is None:
            continue
        path = (models / entrant.file).resolve()
        if path.stat().st_size != entrant.bytes or file_sha256(path) != entrant.sha256:
            raise ValueError(
                f"{entrant.id}: checkpoint bytes differ from the tournament"
            )
        agent, space = load_checkpoint_agent(str(path))
        runtime = runtime_fingerprints(
            entrant.training_seed or 0, match_hypers=match, observation_space=space
        )
        found = {
            name: str(runtime[name])
            for name in (
                "observation_abi_sha256",
                "action_abi_sha256",
                "matchup_sha256",
            )
        }
        if identities is not None and found != identities:
            raise ValueError(f"{entrant.id}: input or action layout differs")
        identities = found
        paths[entrant.id] = str(path)
        parameters[entrant.id] = sum(p.numel() for p in agent.parameters())
    if identities is None:
        raise ValueError("the tournament has no checkpoint entrant")
    common = {
        "information_boundary": "acting-viewer",
        "world": WORLD_VERSION,
        "content_suite": SELECTED_SUITE,
        "player_seed_derivation_id": "arena-pair-deal-player-v1",
        **identities,
    }
    registrations: dict[str, PlayerRegistration] = {}
    for entrant in entrants:
        if entrant.file is None:
            registrations[entrant.id] = PlayerRegistration(
                **common,
                player_id=entrant.id,
                display_name=entrant.label,
                role="anchor",
                runner_kind="code",
                player_spec={"kind": "scripted_greedy"},
                compute_class_id="scripted-greedy-cpu-v1",
                source_sha256=file_sha256(Path(players.__file__)),
            )
        else:
            registrations[entrant.id] = PlayerRegistration(
                **common,
                player_id=entrant.id,
                display_name=entrant.label,
                role="challenger",
                runner_kind="checkpoint",
                # Sampled, not argmax: the same inference the monitor uses.
                player_spec={
                    "kind": "checkpoint",
                    "deterministic": False,
                    "device": "cpu",
                    "batch_size": 1,
                    **(
                        {"allow_deck_repetition": True}
                        if matchup_mode == "mirrors"
                        else {}
                    ),
                },
                compute_class_id="policy-cpu-one-thread-one-pass",
                checkpoint_sha256=entrant.sha256,
                checkpoint_bytes=entrant.bytes,
                parameter_count=parameters[entrant.id],
                training_seed=entrant.training_seed,
                artifact_id=f"{NAME}/{entrant.sha256}",
            )
    key = ArenaKey(
        world=WORLD_VERSION,
        content_suite=SELECTED_SUITE,
        viewer_boundary="acting-viewer",
        arena_version="head-to-head-mirrors-v1"
        if matchup_mode == "mirrors"
        else "head-to-head-v1",
        rating_model_version="exploratory-bradley-terry",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(registrations[GREEDY].model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    return Admitted(key, registrations, paths)


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """Four games: one pairing on one deal, both deck orders and both seats."""

    player_a: str
    player_b: str
    deal_seed: int
    matchup_mode: MatchupMode = "cross"

    @property
    def directory(self) -> str:
        return f"{self.player_a}__{self.player_b}/deal-{self.deal_seed}"


@dataclass(frozen=True)
class Plan:
    """What a run plays. The smoke passes a reduced plan through the same path."""

    entrants: tuple[Entrant, ...]
    pairings: tuple[tuple[str, str], ...]
    deal_seeds: tuple[int, ...]
    matchup_mode: MatchupMode = "cross"
    tournament: Tournament | None = None

    def units(self) -> list[Unit]:
        # Deal-major order keeps every pairing at the same depth if the run stops.
        return [
            Unit(a, b, seed, self.matchup_mode)
            for seed in self.deal_seeds
            for a, b in self.pairings
        ]


def full_plan(
    matchup_mode: MatchupMode = "cross", tournament: Tournament | None = None
) -> Plan:
    seeds = MIRROR_DEAL_SEEDS if matchup_mode == "mirrors" else DEAL_SEEDS
    spec = tournament or default_tournament()
    return Plan(spec.entrants, spec.pairings(), seeds, matchup_mode, tournament)


def smoke_plan(
    matchup_mode: MatchupMode = "cross", tournament: Tournament | None = None
) -> Plan:
    """Two deals through checkpoint play, greedy, replay and paired contrasts.

    Mirrors compare the cross-only and mirror-trained ETU-125 policies; cross
    mode retains its small-model ladder plus the large-model timing probe.
    The mirror smoke does not estimate large-model inference cost.
    """
    if tournament is not None:
        seeds = MIRROR_SMOKE_SEEDS if matchup_mode == "mirrors" else DEAL_SEEDS[:2]
        return replace(full_plan(matchup_mode, tournament), deal_seeds=seeds)
    if matchup_mode == "mirrors":
        first, second = "etu125-cross-10k", "etu125-mirrors-10k"
        return Plan(
            tuple(e for e in ENTRANTS if e.id in (GREEDY, first, second)),
            ((first, GREEDY), (second, GREEDY), (second, first)),
            MIRROR_SMOKE_SEEDS,
            matchup_mode,
        )
    wanted = (GREEDY, "small-26k", "small-100k", "large-15k6")
    return Plan(
        tuple(e for e in ENTRANTS if e.id in wanted),
        (
            ("small-26k", GREEDY),
            ("small-100k", GREEDY),
            ("small-26k", "small-100k"),
            ("small-100k", "large-15k6"),
        ),
        DEAL_SEEDS[:2],
        matchup_mode,
    )


def _play_unit(
    unit: Unit, admitted: Admitted, out: str, game_seconds: float
) -> tuple[Unit, int, float]:
    """Worker entry: play and persist one unit; return (unit, invalid games, seconds).

    `rows.json` is written last, so its presence marks a complete unit.
    """
    import torch

    torch.set_num_threads(1)
    started = time.perf_counter()
    directory = Path(out) / "units" / unit.directory
    directory.mkdir(parents=True, exist_ok=False)
    rows, trace, replay = play_cell(
        key=admitted.key,
        player_a=admitted.registrations[unit.player_a],
        player_b=admitted.registrations[unit.player_b],
        deal_seeds=(unit.deal_seed,),
        out_dir=directory,
        checkpoint_paths={
            name: admitted.checkpoint_paths[name]
            for name in (unit.player_a, unit.player_b)
            if name in admitted.checkpoint_paths
        },
        game_seconds=game_seconds,
        max_commands=MAX_COMMANDS,
        matchup_mode=unit.matchup_mode,
    )
    atomic_json(directory / "trace.json", trace)
    atomic_json(directory / "replay.json", replay)
    atomic_json(directory / "rows.json", rows)
    invalid = sum(not ArenaRow.model_validate(row).valid for row in rows)
    return unit, invalid, time.perf_counter() - started


@dataclass
class Status:
    """Progress and outcome, republished to `status.json` after every unit."""

    state: Literal["admitting", "running", "completed", "incomplete"]
    reason: str | None
    units_total: int
    units_done: int
    units_failed: int
    invalid_games: int
    workers: int
    started_unix: float
    deadline_unix: float
    updated_unix: float
    host_load: tuple[float, float, float]
    free_gib: float


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def _read_tournament(design: dict[str, object]) -> Tournament | None:
    """Read the frozen design schema shared by reporting and continuation.

    Historical runs contain only a `roster` entrant list. Grid runs additionally
    contain `roster_spec`; keep writing those keys so existing evidence and
    continuation identities do not change when the public concept is renamed.
    """
    value = design.get("roster_spec")
    return (
        Tournament.model_validate_json(json.dumps(value)) if value is not None else None
    )


def _validate_extension(
    previous: dict[str, object], current: dict[str, object]
) -> None:
    """Compare JSON-form designs; only additive tournaments/schedules may reuse units.

    Pair orientation binds action seeds and directory names. Runtime identities,
    old registrations, deals and inference bounds remain exact.
    """
    for key in previous.keys() | current.keys():
        old, new = previous.get(key), current.get(key)
        if key == "roster":
            by_id = {row["id"]: row for row in new}
            valid = all(by_id.get(row["id"]) == row for row in old)
        elif key == "registrations":
            valid = all(new.get(name) == value for name, value in old.items())
        elif key == "pairings":
            valid = all(pair in new for pair in old)
        elif key == "roster_spec" and old is not None and new is not None:
            old_spec = _read_tournament(previous)
            new_spec = _read_tournament(current)
            valid = (
                old_spec is not None
                and new_spec is not None
                and old_spec.reference == new_spec.reference
                and old_spec.training_caveat == new_spec.training_caveat
                and all(pair in new_spec.contrasts for pair in old_spec.contrasts)
            )
        else:
            valid = old == new
        if not valid:
            raise ValueError(f"this output directory holds a different design: {key}")


def run(
    out: Path,
    models: Path,
    *,
    plan: Plan | None = None,
    workers: int = 4,
    wall_seconds: float = WALL_SECONDS,
    min_free_gib: float = MIN_FREE_GIB,
) -> Status:
    """Play every unit that has no directory yet, within the wall and disk bounds.

    Stops admitting new units at the wall cap, below the disk floor, or on
    SIGTERM/SIGINT; units already in flight finish (each is bounded by the
    per-game cap) so no directory is left half-written by a clean stop. A unit
    whose worker raised, or whose directory exists without `rows.json`, counts
    as failed and is never replayed. Returns a status that is `completed` only
    if every unit was played and every game was valid.
    """
    plan = plan or full_plan()
    design_path = out / "design.json"
    previous = json.loads(design_path.read_text()) if design_path.exists() else None
    if (
        previous is not None
        and previous.get("matchup_mode", "cross") != plan.matchup_mode
    ):
        raise ValueError("this output directory holds a different matchup mode")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    units = plan.units()
    status = Status(
        state="admitting",
        reason=None,
        units_total=len(units),
        units_done=0,
        units_failed=0,
        invalid_games=0,
        workers=workers,
        started_unix=started,
        deadline_unix=started + wall_seconds,
        updated_unix=started,
        host_load=os.getloadavg(),
        free_gib=_free_gib(out),
    )

    def publish() -> None:
        status.updated_unix = time.time()
        status.host_load = os.getloadavg()
        status.free_gib = _free_gib(out)
        atomic_json(out / "status.json", asdict(status))

    if previous is None:
        publish()
    try:
        admitted = admit(models, plan.entrants, matchup_mode=plan.matchup_mode)
    except Exception as error:
        status.state, status.reason = "incomplete", f"admission failed: {error}"
        if previous is None:
            publish()
        raise
    design = {
        "name": NAME,
        "roster": [asdict(entrant) for entrant in plan.entrants],
        "pairings": [list(pair) for pair in plan.pairings],
        "deal_seeds": list(plan.deal_seeds),
        "game_seconds": GAME_SECONDS,
        "max_commands": MAX_COMMANDS,
        "arena_key": admitted.key.model_dump(mode="json"),
        "registrations": {
            name: registration.model_dump(mode="json")
            for name, registration in admitted.registrations.items()
        },
    }
    if plan.tournament is not None:
        design["roster_spec"] = plan.tournament.model_dump(mode="json")
    if plan.matchup_mode == "mirrors":
        design["matchup_mode"] = "mirrors"
        design["training_caveat"] = (
            plan.tournament.training_caveat if plan.tournament else TRAINING_CAVEAT
        )
    if previous is not None:
        _validate_extension(previous, design)
    atomic_json(design_path, design)

    pending: list[Unit] = []
    for unit in units:
        directory = out / "units" / unit.directory
        if not directory.exists():
            pending.append(unit)
        elif (directory / "rows.json").exists():
            status.units_done += 1
            rows = json.loads((directory / "rows.json").read_text())
            status.invalid_games += sum(
                not ArenaRow.model_validate(row).valid for row in rows
            )
        else:
            status.units_failed += 1
    pending.reverse()  # pop() takes units in schedule order

    stop_reason: list[str] = []

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del frame
        stop_reason.append(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status.state = "running"
    publish()
    active: set[Future[tuple[Unit, int, float]]] = set()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        while pending or active:
            if not stop_reason:
                if time.time() >= status.deadline_unix:
                    stop_reason.append("wall-time cap reached")
                elif _free_gib(out) < min_free_gib:
                    stop_reason.append("free disk fell below the floor")
            while pending and not stop_reason and len(active) < workers:
                try:
                    active.add(
                        pool.submit(
                            _play_unit, pending[-1], admitted, str(out), GAME_SECONDS
                        )
                    )
                    pending.pop()
                except BrokenProcessPool:
                    # A worker died abruptly; the pool accepts nothing further.
                    stop_reason.append("a worker process died")
            if not active:
                break
            finished, active = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                try:
                    _, invalid, _ = future.result()
                    status.units_done += 1
                    status.invalid_games += invalid
                except Exception as error:
                    status.units_failed += 1
                    print(f"unit failed: {type(error).__name__}: {error}", flush=True)
            if finished:
                publish()
    if stop_reason:
        status.state, status.reason = "incomplete", stop_reason[0]
    elif status.units_failed or status.invalid_games:
        status.state = "incomplete"
        status.reason = "at least one unit failed or one game was invalid"
    else:
        status.state = "completed"
    publish()
    return status


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A score in [0, 1] (a draw counts half) with a 95% deal-bootstrap interval."""

    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PairingScore:
    """player_a's results in one pairing over its complete, valid deals.

    `as_lessons`/`as_allies` split by the deck player_a held (two games per
    deal each); `on_play`/`on_draw` split by player_a's seat, seat 0 being the
    engine's first player. `rejected` is why no score is given, when it is not.
    """

    player_a: str
    player_b: str
    deals: int
    games: int
    invalid_games: int
    rejected: str | None
    score: Interval | None = None
    as_lessons: Interval | None = None
    as_allies: Interval | None = None
    on_play: Interval | None = None
    on_draw: Interval | None = None
    seconds_per_game: float | None = None
    failed_units: int = 0


@dataclass(frozen=True)
class Contrast:
    """Two measures of `first` against `second` on the deals both share.

    `greedy_difference` is first's score against greedy minus second's, paired
    by deal. `direct` is first's score in their own meeting; 0.5 is parity.
    Both intervals resample whole deals.
    """

    first: str
    second: str
    deals: int
    greedy_difference: Interval
    direct: Interval


@dataclass(frozen=True)
class ScorePanel:
    pairings: tuple[PairingScore, ...]
    contrasts: tuple[Contrast, ...]
    # Elo-scaled rating relative to greedy at 0: (2.5%, median, 97.5%).
    ratings: dict[str, tuple[float, float, float]]
    rating_note: str


@dataclass(frozen=True)
class Report(ScorePanel):
    status: dict[str, object] | None
    matchup_mode: MatchupMode = "cross"
    decks: dict[str, ScorePanel] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TournamentReport(Report):
    """Declared report context; historical reports retain their exact schema."""

    reference: str
    training_caveat: str


def _interval(per_deal: np.ndarray, rng: np.random.Generator) -> Interval:
    """Percentile bootstrap over deals of the mean of one value per deal."""
    indices = rng.integers(0, len(per_deal), (BOOTSTRAP_REPLICATES, len(per_deal)))
    lower, upper = np.quantile(per_deal[indices].mean(axis=1), [0.025, 0.975])
    return Interval(float(per_deal.mean()), float(lower), float(upper))


@dataclass(frozen=True)
class _Cell:
    """Retained rows of one pairing: complete deals and their four scores.

    `scores[d, leg]` is player_a's score; `lessons` and `play` are boolean
    masks over the same shape. Only deals whose four games are all valid are
    kept; `invalid_games` counts the rest and rejects the pairing.
    """

    deal_seeds: tuple[int, ...]
    scores: np.ndarray
    lessons: np.ndarray
    play: np.ndarray
    invalid_games: int
    seconds: float
    rows: tuple[dict[str, object], ...]
    failed_units: int = 0


def _load_cell(
    out: Path,
    pairing: tuple[str, str],
    deal_seeds: Sequence[int],
    matchup_mode: MatchupMode = "cross",
) -> _Cell:
    kept: list[int] = []
    scores: list[list[float]] = []
    lessons: list[list[bool]] = []
    play: list[list[bool]] = []
    retained: list[dict[str, object]] = []
    invalid = 0
    failed_units = 0
    seconds = 0.0
    expected_decks = {
        leg: list(selected_decks(leg))
        for leg in (range(4, 8) if matchup_mode == "mirrors" else range(4))
    }
    for seed in deal_seeds:
        path = out / "units" / Unit(*pairing, seed).directory / "rows.json"
        if not path.exists():
            failed_units += int(path.parent.exists())
            continue
        raw_rows: list[dict[str, object]] = json.loads(path.read_text())
        try:
            rows = [ArenaRow.model_validate(row) for row in raw_rows]
        except ValidationError:
            invalid += max(4, len(raw_rows))
            continue
        bad = sum(not row.valid for row in rows)
        if bad or len(rows) != 4 or {row.leg for row in rows} != expected_decks.keys():
            invalid += bad or max(4, len(rows))
            continue
        ordered = sorted(zip(rows, raw_rows, strict=True), key=lambda item: item[0].leg)
        bad = sum(
            (row.player_a, row.player_b) != pairing
            or row.deal_seed != seed
            or raw.get("player_a_seat") != row.leg % 2
            or raw.get("seat_decks") != expected_decks[row.leg]
            for row, raw in ordered
        )
        if bad:
            invalid += bad
            continue
        kept.append(seed)
        scores.append([float(row.score_a or 0.0) for row, _ in ordered])
        lessons.append(
            [expected_decks[row.leg][row.leg % 2] == "ur_lessons" for row, _ in ordered]
        )
        play.append([row.leg % 2 == 0 for row, _ in ordered])
        seconds += sum(row.game_seconds for row in rows)
        # The rating bootstrap resamples `deal_block`; the arena numbers blocks
        # within one call, so rebind it to this deal's place in the cohort.
        retained.extend({**raw, "deal_block": seed} for _, raw in ordered)
    shape = (len(kept), 4)
    return _Cell(
        tuple(kept),
        np.array(scores, dtype=np.float64).reshape(shape),
        np.array(lessons, dtype=bool).reshape(shape),
        np.array(play, dtype=bool).reshape(shape),
        invalid,
        seconds,
        tuple(retained),
        failed_units,
    )


def _pairing_score(pairing: tuple[str, str], cell: _Cell) -> PairingScore:
    deals = len(cell.deal_seeds)
    base = PairingScore(
        *pairing,
        deals,
        cell.scores.size,
        cell.invalid_games,
        None,
        failed_units=cell.failed_units,
    )
    if cell.failed_units:
        return replace(base, rejected="failed or unfinished units")
    if cell.invalid_games:
        # Dropping failed games would bias the rate; give none instead.
        return replace(base, rejected="invalid games")
    if deals < 2:
        return replace(base, rejected="fewer than two deals")
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def split(mask: np.ndarray) -> Interval | None:
        if not mask.any():
            return None
        # Average within each deal before resampling; seats stay paired.
        return _interval((cell.scores * mask).sum(axis=1) / mask.sum(axis=1), rng)

    return replace(
        base,
        score=_interval(cell.scores.mean(axis=1), rng),
        as_lessons=split(cell.lessons),
        as_allies=split(~cell.lessons),
        on_play=split(cell.play),
        on_draw=split(~cell.play),
        seconds_per_game=cell.seconds / cell.scores.size,
    )


def _oriented(
    cells: dict[tuple[str, str], _Cell], first: str, second: str
) -> tuple[tuple[int, ...], np.ndarray] | None:
    """Per-deal mean score of `first` against `second`, whichever way it was played."""
    if (first, second) in cells:
        cell = cells[first, second]
        return cell.deal_seeds, cell.scores.mean(axis=1)
    if (second, first) in cells:
        cell = cells[second, first]
        return cell.deal_seeds, 1.0 - cell.scores.mean(axis=1)
    return None


def _contrast(
    cells: dict[tuple[str, str], _Cell], first: str, second: str
) -> Contrast | None:
    found = [
        _oriented(cells, first, GREEDY),
        _oriented(cells, second, GREEDY),
        _oriented(cells, first, second),
    ]
    if any(item is None for item in found):
        return None
    tables = [dict(zip(seeds, values, strict=True)) for seeds, values in found]  # type: ignore[misc]
    shared = sorted(set(tables[0]) & set(tables[1]) & set(tables[2]))
    if len(shared) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    difference = np.array([tables[0][s] - tables[1][s] for s in shared])
    direct = np.array([tables[2][s] for s in shared])
    return Contrast(
        first, second, len(shared), _interval(difference, rng), _interval(direct, rng)
    )


def _score_panel(
    cells: dict[tuple[str, str], _Cell],
    contrast_pairs: tuple[tuple[str, str], ...],
) -> ScorePanel:
    """Use the same paired-deal estimator on the full matrix or a single deck."""
    scores = tuple(_pairing_score(pair, cell) for pair, cell in cells.items())
    usable = {
        pair: cell
        for pair, cell in cells.items()
        if not cell.invalid_games
        and not cell.failed_units
        and len(cell.deal_seeds) >= 2
    }
    contrasts = tuple(
        found
        for first, second in contrast_pairs
        if (found := _contrast(usable, first, second)) is not None
    )
    rows = [row for cell in usable.values() for row in cell.rows]
    ratings: dict[str, tuple[float, float, float]] = {}
    note = "ratings unavailable: no usable pairing includes greedy"
    if any(GREEDY in pair for pair in usable):
        try:
            # The fit pins its anchor at a fixed rating; report relative to it.
            zero = fit_population(rows, anchor=GREEDY).ratings[GREEDY]
            boot = bootstrap_population(
                rows, replicates=RATING_REPLICATES, seed=BOOTSTRAP_SEED, anchor=GREEDY
            )
            ratings = {
                name: (float(low - zero), float(mid - zero), float(high - zero))
                for name, (low, mid, high) in boot["ratings"].items()
                if np.isfinite([low, mid, high]).all()
            }
            note = (
                f"{boot['replicates'] - boot['failures']} of {boot['replicates']} "
                "deal-bootstrap fits converged"
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            note = f"ratings unavailable: {error}"
    return ScorePanel(scores, contrasts, ratings, note)


def _deck_cell(cell: _Cell, deck: str) -> _Cell:
    # Mirror legs are [Lessons play, Lessons draw, Allies play, Allies draw].
    columns = slice(0, 2) if deck == "ur_lessons" else slice(2, 4)
    rows = tuple(row for row in cell.rows if row["seat_decks"] == [deck, deck])
    return replace(
        cell,
        scores=cell.scores[:, columns],
        lessons=cell.lessons[:, columns],
        play=cell.play[:, columns],
        rows=rows,
        seconds=sum(float(str(row["game_seconds"])) for row in rows),
    )


def report(out: Path) -> Report:
    """Project retained rows; never plays a game or loads a model."""
    design = json.loads((out / "design.json").read_text())
    mode = design.get("matchup_mode", "cross")
    if mode not in ("cross", "mirrors"):
        raise ValueError("unknown head-to-head matchup mode")
    scheduled = [(str(a), str(b)) for a, b in design["pairings"]]
    deal_seeds = [int(seed) for seed in design["deal_seeds"]]
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    cells = {pair: _load_cell(out, pair, deal_seeds, mode) for pair in scheduled}
    spec = _read_tournament(design)
    contrasts = spec.contrasts if spec else CONTRASTS
    panel = _score_panel(cells, contrasts)
    decks = (
        {
            deck: _score_panel(
                {pair: _deck_cell(cell, deck) for pair, cell in cells.items()},
                contrasts,
            )
            for deck in ("ur_lessons", "gw_allies")
        }
        if mode == "mirrors"
        else {}
    )
    # Select the schema before construction; legacy JSON has no tournament metadata.
    report_type = (
        partial(
            TournamentReport,
            reference=spec.reference,
            training_caveat=spec.training_caveat,
        )
        if spec is not None
        else Report
    )
    return report_type(
        panel.pairings,
        panel.contrasts,
        panel.ratings,
        panel.rating_note,
        status,
        mode,
        decks,
    )


def _points(interval: Interval | None, *, signed: bool = False) -> str:
    if interval is None:
        return "-"
    form = "{:+.1f} [{:+.1f}, {:+.1f}]" if signed else "{:.1f} [{:.1f}, {:.1f}]"
    return form.format(100 * interval.mean, 100 * interval.lower, 100 * interval.upper)


def render(found: Report) -> str:
    lines: list[str] = []
    if found.status is not None:
        s = found.status
        lines.append(
            f"status: {s['state']}"
            + (f" ({s['reason']})" if s["reason"] else "")
            + f"; units {s['units_done']}/{s['units_total']}, failed {s['units_failed']},"
            f" invalid games {s['invalid_games']}"
        )
    if isinstance(found, TournamentReport):
        lines += ["", found.training_caveat, f"Reference entrant: {found.reference}"]
    if found.matchup_mode == "mirrors":
        if not isinstance(found, TournamentReport):
            lines += ["", TRAINING_CAVEAT]
        for deck, panel in found.decks.items():
            label = "Lessons" if deck == "ur_lessons" else "Allies"
            lines += ["", f"## {label} vs {label}", _render_panel(panel)]
    else:
        lines.append(_render_panel(found))
    return "\n".join(lines)


def _render_panel(found: ScorePanel) -> str:
    lines: list[str] = []
    lines += [
        "",
        "Pairings. Score is player A's, in points; a draw counts half."
        " Intervals are 95%, resampling whole deals.",
        "A | B | valid games | invalid games | failed units | score | A as Lessons | A as Allies | A on play | A on draw | s/game",
    ]
    for p in found.pairings:
        lines.append(
            " | ".join(
                [
                    p.player_a,
                    p.player_b,
                    str(p.games),
                    str(p.invalid_games),
                    str(p.failed_units),
                    _points(p.score) if p.rejected is None else f"none: {p.rejected}",
                    _points(p.as_lessons),
                    _points(p.as_allies),
                    _points(p.on_play),
                    _points(p.on_draw),
                    "-" if p.seconds_per_game is None else f"{p.seconds_per_game:.2f}",
                ]
            )
        )
    lines += [
        "",
        "Contrasts on shared deals. 'vs greedy' is first's greedy score minus"
        " second's; 'direct' is first's score against second minus 50.",
        "first | second | deals | vs greedy | direct",
    ]
    for c in found.contrasts:
        direct = Interval(
            c.direct.mean - 0.5, c.direct.lower - 0.5, c.direct.upper - 0.5
        )
        lines.append(
            f"{c.first} | {c.second} | {c.deals} | "
            f"{_points(c.greedy_difference, signed=True)} | {_points(direct, signed=True)}"
        )
    lines += [
        "",
        f"Ratings, Elo scale, greedy = 0 ({found.rating_note}).",
        "entrant | rating",
    ]
    for name, (low, mid, high) in sorted(
        found.ratings.items(), key=lambda item: -item[1][1]
    ):
        lines.append(f"{name} | {mid:+.0f} [{low:+.0f}, {high:+.0f}]")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("out", type=Path)
        command.add_argument("--models", type=Path, required=True)
        command.add_argument("--tournament", choices=TOURNAMENTS)
        command.add_argument(
            "--matchups", choices=("cross", "mirrors"), default="cross"
        )
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--wall-seconds", type=float, default=WALL_SECONDS)
        command.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    commands.add_parser("report").add_argument("out", type=Path)
    commands.add_parser("tournament")
    args = parser.parse_args()
    if args.command == "tournament":
        for entrant in ENTRANTS:
            print(
                f"{entrant.id} | {entrant.label} | {entrant.origin} | {entrant.sha256}"
            )
        for a, b in pairings():
            print(f"{a} vs {b}")
        return
    if args.command == "report":
        found = report(args.out)
        atomic_json(args.out / "report.json", asdict(found))
        print(render(found))
        return
    tournament = TOURNAMENTS[args.tournament]() if args.tournament else None
    status = run(
        args.out,
        args.models,
        plan=smoke_plan(args.matchups, tournament)
        if args.command == "smoke"
        else full_plan(args.matchups, tournament),
        workers=args.workers,
        wall_seconds=args.wall_seconds,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(asdict(status), indent=2))
    if status.state != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
