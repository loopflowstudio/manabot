"""Public draft collection and bounded terminal-reward training orchestration.

Native Rotisserie/Limited own choices. The regime executor owns this operation's
clock, persistence and artifact registration; the injected backend owns the
network and Learner. Reward supervision is shared with ordinary game execution.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter, process_time
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

import numpy as np

from manabot.drafting import BuildChoice, DraftChoice, _journal
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.policy.draft import (
    DraftInput,
    DraftPrepared,
    DraftTrajectory,
    prepare_draft,
)
from manabot.training.draft_rewards import (
    FrozenRewardRunner,
    RewardSample,
    score_schedule,
)
from manabot.training.state.game_errors import GameErrorLedger
from manabot.training.updates.rotisserie import RunningBaseline, update_rotisserie
import managym
from managym.format import Limited, Rotisserie

if TYPE_CHECKING:
    from manabot.training.regime import TrainRotisserie


class DraftUpdate(TypedDict):
    update: int
    loss: float | None
    valid_trajectories: int
    learner_decisions: int
    environment_decisions: int
    planned_rewards: int
    attempted_rewards: int
    completed_games: int
    opening_scores: int
    failed_rewards: int
    worker_cpu_seconds: float
    unknown_worker_cpu_samples: int
    gradient_decisions: int
    baseline: float
    collection_seconds: float
    learning_seconds: float
    parent_cpu_seconds: float
    wall_seconds: float


class DraftEpisodePlan(StrictModel):
    id: str
    update: int
    seed: int
    reward_seed: int
    roles: tuple[str, ...]


def _schedule(stage: "TrainRotisserie", seed: int) -> list[DraftEpisodePlan]:
    plans = []
    for update in range(stage.updates):
        for episode in range(stage.drafts_per_update):
            ordinal = update * stage.drafts_per_update + episode
            live = {(ordinal + 2 * i) % stage.seats for i in range(stage.seats // 2)}
            frozen = [seat for seat in range(stage.seats) if seat not in live]
            roles = tuple(
                "learner"
                if seat in live
                else ("initial" if frozen.index(seat) % 2 == 0 else f"history-{update}")
                for seat in range(stage.seats)
            )
            plans.append(
                DraftEpisodePlan(
                    id=f"update{update + 1}-draft{episode}",
                    update=update + 1,
                    seed=seed + 10000 + ordinal,
                    reward_seed=seed + 20000 + ordinal * 100,
                    roles=roles,
                )
            )
    return plans


class DraftActor(Protocol):
    @property
    def identity(self) -> str: ...

    def logits(self, batch: DraftPrepared[np.ndarray]) -> np.ndarray: ...


class DraftEpisode(StrictModel):
    episode_id: str
    binding: str
    seed: int
    roles: tuple[str, ...]
    policy_identities: tuple[str, ...]
    decks: tuple[managym.Deck, ...]
    remaining: tuple[int, ...]
    draft_decisions: int
    build_decisions: int


@dataclass(frozen=True)
class CollectedDraft:
    episode: DraftEpisode
    trajectories: tuple[DraftTrajectory, ...]


@dataclass(frozen=True)
class DraftCatalog:
    cards: np.ndarray
    names: tuple[str, ...]
    binding: str


def draft_catalog(set_code: str, seats: int, picks: int) -> DraftCatalog:
    state = Rotisserie(set_code, "catalog", seats, picks).observe()
    cards = np.flatnonzero((np.asarray(state["initial"]) > 0) | Limited().basics)
    definitions = json.loads(managym.card_definitions_json())
    binding = canonical_sha256(
        {
            "schema": "rotisserie-policy-1",
            "catalog": managym.catalog_tokens(),
            "printing": state["printing_digest"],
            "world": managym.world_identity(),
            "seats": seats,
            "picks": picks,
            "build_cap": 40,
        }
    )
    return DraftCatalog(cards, tuple(d["name"] for d in definitions), binding)


def collect_draft(
    out: Path,
    episode_id: str,
    seed: int,
    actors: Sequence[DraftActor | None],
    roles: tuple[str, ...],
    catalog: DraftCatalog,
    *,
    set_code: str = "tla",
    picks: int = 45,
    check: Callable[[], None],
) -> CollectedDraft:
    """None is the uniform definition reference; all decisions precede commit."""
    if len(actors) != len(roles):
        raise ValueError("each seat requires an actor and retained role")
    draft = Rotisserie(set_code, episode_id, len(actors), picks)
    state = draft.observe()
    out.mkdir(parents=True, exist_ok=False)
    policy_identities = tuple(
        actor.identity if actor is not None else "uniform-legal-definitions"
        for actor in actors
    )
    atomic_json(
        out / "setup.json",
        {
            "episode_id": episode_id,
            "seed": seed,
            "roles": roles,
            "policy_identities": policy_identities,
            "state": state,
            "binding": catalog.binding,
        },
    )
    rngs = [
        np.random.default_rng(np.random.SeedSequence([seed, seat]))
        for seat in range(len(actors))
    ]
    batches: list[list[DraftPrepared[np.ndarray]]] = [[] for _ in actors]
    actions: list[list[int]] = [[] for _ in actors]

    def choose(item: DraftInput) -> int:
        check()
        batch = prepare_draft([item], catalog.cards, catalog.binding)
        legal = np.flatnonzero(batch.legal[0])
        actor = actors[item.chooser]
        if actor is None:
            selected = int(rngs[item.chooser].choice(legal))
        else:
            logits = actor.logits(batch)[0, legal].astype(np.float64)
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite draft logits")
            weights = np.exp(logits - logits.max())
            selected = int(rngs[item.chooser].choice(legal, p=weights / weights.sum()))
        if roles[item.chooser] == "learner":
            batches[item.chooser].append(batch)
            actions[item.chooser].append(selected)
        return int(catalog.cards[selected])

    decks = []
    limited = Limited()
    with (out / "choices.jsonl").open("x") as stream:
        while (seat := state["chooser"]) is not None:
            card = choose(DraftInput("draft", state, seat))
            _journal(
                stream,
                DraftChoice(
                    episode_id=episode_id,
                    revision=state["revision"],
                    chooser=seat,
                    card_def_id=card,
                ),
            )
            draft.pick(episode_id, state["revision"], seat, card)
            state = draft.observe()
        # Every builder sees only its own partial main deck. Others' finished
        # decks are revealed in the final receipt, never during construction.
        for seat in range(len(actors)):
            pool = draft.pool(seat)
            build = limited.build(pool, max_cards=40)
            counts = np.zeros(len(catalog.names), np.int32)
            for revision in range(40):
                card = choose(
                    DraftInput("build", state, seat, tuple(int(n) for n in counts))
                )
                name = catalog.names[card]
                _journal(
                    stream,
                    BuildChoice(
                        episode_id=episode_id,
                        revision=revision,
                        chooser=seat,
                        card=name,
                    ),
                )
                build.select(name)
                counts[card] += 1
            if not build.can_stop or limited.issues(pool, build.deck):
                raise ValueError(f"seat {seat}: Limited admission failed")
            sideboard = {
                name: count - build.deck.get(name, 0)
                for name, count in pool.cards.items()
                if count > build.deck.get(name, 0)
            }
            decks.append(managym.Deck(build.deck, sideboard))
    if any(managym.check_decks(decks)):
        raise ValueError("generated decks failed world admission")
    episode = DraftEpisode(
        episode_id=episode_id,
        binding=catalog.binding,
        seed=seed,
        roles=roles,
        policy_identities=policy_identities,
        decks=tuple(decks),
        remaining=tuple(state["remaining"]),
        draft_decisions=state["revision"],
        build_decisions=40 * len(actors),
    )
    atomic_json(out / "receipt.json", episode.model_dump(mode="json"))
    trajectories = tuple(
        DraftTrajectory(seat, tuple(batches[seat]), tuple(actions[seat]))
        for seat, role in enumerate(roles)
        if role == "learner"
    )
    return CollectedDraft(episode, trajectories)


def reward_samples(
    episode: DraftEpisode,
    learner: int,
    opponent: int,
    *,
    mode: Literal["game", "value"],
    seed: int,
    bundle: str,
) -> list[RewardSample]:
    """Two deals crossed with both game seats. Deal identity is pair-local."""
    hero, villain = episode.decks[learner], episode.decks[opponent]
    rows = []
    for deal in range(2):
        for seat in (0, 1):
            first, second = (hero, villain) if seat == 0 else (villain, hero)
            rows.append(
                RewardSample(
                    id=f"{bundle}-deal{deal}-seat{seat}",
                    bundle=bundle,
                    mode=mode,
                    seed=seed + deal,
                    seat=seat,
                    match=MatchConfig(
                        hero_deck=first.main,
                        villain_deck=second.main,
                        hero_sideboard=first.sideboard,
                        villain_sideboard=second.sideboard,
                        world=managym.WORLD_SETUP,
                    ),
                )
            )
    return rows


def evaluate_drafter(
    candidate: DraftActor,
    initial: DraftActor,
    catalog: DraftCatalog,
    runner: Callable[[RewardSample, Path], dict[str, Any]],
    out: Path,
    *,
    seed: int,
    check: Callable[[], None],
    ledger: GameErrorLedger,
    set_code: str = "tla",
    picks: int = 45,
    seats: int = 8,
) -> None:
    """No live self-play aggregate: each candidate faces earlier fixed tables."""
    out.mkdir(parents=True, exist_ok=False)
    expected = [
        {
            "reference": ref,
            "draft_seat": seat,
            "opponent": (seat + seats // 2) % seats,
            "seed": seed + seat,
        }
        for ref in ("initial", "uniform")
        for seat in range(seats)
    ]
    atomic_json(out / "tables.json", expected)
    rows = []
    for cell in expected:
        seat = int(cell["draft_seat"])
        identity = f"{cell['reference']}-seat{seat}"
        actors = [
            initial if cell["reference"] == "initial" else None for _ in range(seats)
        ]
        actors[seat] = candidate
        roles = tuple(
            "candidate" if s == seat else str(cell["reference"]) for s in range(seats)
        )
        episode = collect_draft(
            out / identity,
            identity,
            seed + seat,
            actors,
            roles,
            catalog,
            set_code=set_code,
            picks=picks,
            check=check,
        ).episode
        samples = reward_samples(
            episode,
            seat,
            int(cell["opponent"]),
            mode="game",
            seed=seed + 1000 + seat * 2,
            bundle=identity,
        )
        batch = score_schedule(
            samples, out / f"{identity}-games", runner, check=check, ledger=ledger
        )
        rows.extend(
            {
                "reference": cell["reference"],
                "draft_seat": seat,
                **r.model_dump(mode="json"),
            }
            for r in batch.results
        )
        atomic_json(out / "results.json", rows)
        summaries = []
        for reference in ("initial", "uniform"):
            selected = [row for row in rows if row["reference"] == reference]
            completed = [row for row in selected if row["status"] == "completed"]
            openings = [
                row["opening"] for row in completed if row["opening"] is not None
            ]
            summaries.append(
                {
                    "reference": reference,
                    "planned": seats * 4,
                    "attempted": len(selected),
                    "completed": len(completed),
                    "failed": len(selected) - len(completed),
                    "missing": seats * 4 - len(selected),
                    "wins": sum(row["value"] == 1 for row in completed),
                    "draws": sum(row["value"] == 0 for row in completed),
                    "losses": sum(row["value"] == -1 for row in completed),
                    "win_rate": sum(row["value"] == 1 for row in completed)
                    / len(completed)
                    if completed
                    else None,
                    "opening_count": len(openings),
                    "opening_mean": sum(openings) / len(openings) if openings else None,
                    "opening_min": min(openings) if openings else None,
                    "opening_max": max(openings) if openings else None,
                    "mean_absolute_value_outcome_gap": sum(
                        abs(row["opening"] - row["value"])
                        for row in completed
                        if row["opening"] is not None
                    )
                    / len(openings)
                    if openings
                    else None,
                }
            )
        atomic_json(out / "summary.json", summaries)
        if batch.stopped:
            raise RuntimeError(
                "fixed-reference evaluation stopped; unreached cells remain missing"
            )


def train_rotisserie(
    stage: "TrainRotisserie",
    *,
    backend: ModuleType,
    seed: int,
    out: Path,
    check: Callable[[], None],
    remaining: Callable[[], float],
    progress: Callable[[DraftUpdate], None],
    artifact: Callable[[str, Path], None],
    ledger: GameErrorLedger,
) -> None:
    """Called only by execute_regime after stage/resource admission."""
    if backend.__name__ != "manabot.backends.torch":
        raise ValueError("Rotisserie currently requires the torch backend")
    started, cpu_started = perf_counter(), process_time()
    out.mkdir(parents=True, exist_ok=False)
    # Admission precedes policy collection. It does not establish competence.
    player, _ = backend.checkpoint.load_checkpoint(stage.player["path"])
    player.eval().requires_grad_(False)
    if "world_digest" not in (player.world_binding or {}):
        raise ValueError(
            "Rotisserie requires a world-bound frozen player admitting generated decks"
        )
    if player.value_semantic not in ("signed_outcome", "win_logit"):
        raise ValueError("frozen player critic semantic is unspecified")
    catalog = draft_catalog(stage.set, stage.seats, stage.picks_per_seat)
    atomic_json(out / "catalog.json", managym.catalog_tokens())
    atomic_json(out / "printings.json", managym.format.set_printings(stage.set))
    policy = backend.draft.create(
        backend.draft.DraftConfig(len(catalog.names), seats=stage.seats),
        catalog.cards,
        catalog.binding,
        seed,
    )
    learner = backend.learner.Learner(policy, "cpu", minibatch_rows=8)
    initial = deepcopy(policy).eval().requires_grad_(False)
    history = initial
    atomic_json(
        out / "initialization.json",
        {
            "seed": seed,
            "parameter_digest": initial.identity,
            "binding": catalog.binding,
        },
    )
    baseline = RunningBaseline()
    runner = FrozenRewardRunner(
        Path(stage.player["path"]),
        backend.checkpoint.BACKEND,
        stage.player["sha256"],
        perf_counter() + remaining(),
    )
    plans = _schedule(stage, seed)
    atomic_json(out / "schedule.json", [plan.model_dump(mode="json") for plan in plans])
    for update in range(stage.updates + 1):
        target = out / f"update-{update:04d}.pt"
        backend.draft.save(target, policy)
        backend.draft.load(target, binding=catalog.binding, cards=catalog.cards)
        artifact(f"draft-update-{update}", target)
        if stage.evaluate and (update > 0 or stage.evaluate_initial):
            evaluate_drafter(
                policy,
                initial,
                catalog,
                runner,
                out / f"eval-{update:04d}",
                seed=seed + 30000,
                check=check,
                ledger=ledger,
                set_code=stage.set,
                picks=stage.picks_per_seat,
                seats=stage.seats,
            )
            artifact(
                f"evaluation-{update}", out / f"eval-{update:04d}" / "results.json"
            )
        if update == stage.updates:
            break
        collection_started = perf_counter()
        trajectories, rewards = [], []
        environment_decisions = 0
        reward_results = []
        for plan in (plan for plan in plans if plan.update == update + 1):
            roles = plan.roles
            frozen = [seat for seat, role in enumerate(roles) if role != "learner"]
            actors = [
                policy
                if role == "learner"
                else (initial if role == "initial" else history)
                for role in roles
            ]
            identity = plan.id
            collected = collect_draft(
                out / identity,
                identity,
                plan.seed,
                actors,
                roles,
                catalog,
                set_code=stage.set,
                picks=stage.picks_per_seat,
                check=check,
            )
            samples = []
            for index, trajectory in enumerate(collected.trajectories):
                bundle = f"{identity}-learner{trajectory.seat}"
                samples.extend(
                    reward_samples(
                        collected.episode,
                        trajectory.seat,
                        frozen[index],
                        mode=stage.reward,
                        seed=plan.reward_seed + index * 2,
                        bundle=bundle,
                    )
                )
            scored = score_schedule(
                samples, out / f"{identity}-rewards", runner, check=check, ledger=ledger
            )
            environment_decisions += (
                collected.episode.draft_decisions + collected.episode.build_decisions
            )
            reward_results.extend(scored.results)
            if scored.stopped:
                raise RuntimeError(
                    "reward supervision stopped; retain missing slots and do not optimize"
                )
            trajectories.extend(collected.trajectories)
            rewards.extend(
                scored.rewards[f"{identity}-learner{t.seat}"]
                for t in collected.trajectories
            )
        collection_seconds = perf_counter() - collection_started
        learning_started = perf_counter()
        loss = update_rotisserie(backend, learner, trajectories, rewards, baseline)
        if loss is not None:
            history = deepcopy(policy).eval().requires_grad_(False)
        row: DraftUpdate = {
            "update": update + 1,
            "loss": loss,
            "valid_trajectories": sum(r is not None for r in rewards),
            "learner_decisions": sum(len(t.actions) for t in trajectories),
            "environment_decisions": environment_decisions,
            "planned_rewards": stage.drafts_per_update * (stage.seats // 2) * 4,
            "attempted_rewards": len(reward_results),
            "completed_games": sum(r.status == "completed" for r in reward_results),
            "opening_scores": sum(r.status == "scored" for r in reward_results),
            "failed_rewards": sum(r.value is None for r in reward_results),
            "worker_cpu_seconds": sum(
                r.worker_cpu_seconds
                for r in reward_results
                if r.worker_cpu_seconds is not None
            ),
            "unknown_worker_cpu_samples": sum(
                r.worker_cpu_seconds is None for r in reward_results
            ),
            "gradient_decisions": sum(
                len(t.actions)
                for t, r in zip(trajectories, rewards, strict=True)
                if r is not None
            ),
            "baseline": baseline.value,
            "collection_seconds": collection_seconds,
            "learning_seconds": perf_counter() - learning_started,
            "parent_cpu_seconds": process_time() - cpu_started,
            "wall_seconds": perf_counter() - started,
        }
        atomic_json(out / f"update-{update + 1:04d}.json", row)
        progress(row)


def validate_software(backend: ModuleType, out: Path, seed: int) -> dict[str, object]:
    """One controlled update; neither synthetic returns nor decks are evidence."""
    if backend.__name__ != "manabot.backends.torch":
        raise ValueError("Rotisserie currently requires the torch backend")
    out.mkdir(parents=True, exist_ok=False)
    catalog = draft_catalog("tla", 2, 1)
    policy = backend.draft.create(
        backend.draft.DraftConfig(len(catalog.names), seats=2),
        catalog.cards,
        catalog.binding,
        seed,
    )
    learner = backend.learner.Learner(policy, "cpu", minibatch_rows=1)
    frozen = deepcopy(policy).eval().requires_grad_(False)
    collected = collect_draft(
        out / "draft",
        "software-validation",
        seed,
        [policy, frozen],
        ("learner", "initial"),
        catalog,
        picks=1,
        check=lambda: None,
    )
    loss = update_rotisserie(
        backend, learner, list(collected.trajectories), [0.5], RunningBaseline()
    )
    path = out / "draft-policy.pt"
    backend.draft.save(path, policy)
    restored = backend.draft.load(path, binding=catalog.binding, cards=catalog.cards)
    repeated = collect_draft(
        out / "reload",
        "software-reload",
        seed,
        [restored, frozen],
        ("learner", "initial"),
        catalog,
        picks=1,
        check=lambda: None,
    )
    result = {
        "status": "software_validation",
        "updates": 1,
        "synthetic_return": 0.5,
        "loss": loss,
        "learner_decisions": 41,
        "legal_decks": len(repeated.episode.decks),
        "reward_games": 0,
        "strength_result": False,
    }
    atomic_json(out / "receipt.json", result)
    return result
