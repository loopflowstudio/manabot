"""MTG-183's frozen windows, rated checkpoints and pairing graph."""

from collections import Counter
import json
from pathlib import Path

from etude.run.storage import DirectoryStorage
import pytest

from experiments.elemental_grid import DECKS, schedule
from experiments.elemental_long import (
    CADENCE,
    H2H_GAMES_PER_CELL,
    H2H_SCHEDULE_SHA256,
    HEURISTIC_FULL,
    REFERENCES,
    REGIME,
    SEED,
    STAGE,
    h2h_schedule,
    h2h_schedule_sha256,
    heuristic_games,
    pair_design,
    rated_updates,
    segment_job,
    stage_parent,
)
from manabot.experiment.job import LearnerContinuation, StagedExport
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.training.execution import execute_regime
from manabot.training.regime import (
    DampedPolicyLearning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym


def _export(name: str) -> StagedExport:
    return StagedExport(key=f"mtg183/w2/{name}", sha256="0" * 64, bytes=1)


def test_segment_continues_the_frozen_recipe_on_the_cadence() -> None:
    continuation = LearnerContinuation(
        from_update=2551,
        source_stage=STAGE,
        **{role: _export(role) for role in ("source_run", "raw", "ema", "optimizer")},
    )
    spec = segment_job(continuation, 4000, hours=24)
    frozen = TrainingRegime.model_validate_json(REGIME.read_bytes()).stages[0]
    stage = spec.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay) and isinstance(frozen, TrainSelfPlay)
    assert stage.learning == frozen.learning and stage.streams == frozen.streams
    assert spec.milestones[0] == 2600 and spec.milestones[-1] == 3950
    assert all(m % CADENCE == 0 for m in spec.milestones)
    job = spec.job(repo=REGIME.parent, revision="fixture", max_hours=24)
    assert job.until == "updates=4000" and len(job.inputs) == 4


def _window(tmp_path: Path) -> Path:
    """A tiny run shaped like a window: MTG-183's seed and stage name."""
    regime = TrainingRegime(
        id="window-fixture",
        world=managym.WORLD_VERSION,
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=ModelConfig(
            hidden_dim=8, num_attention_heads=2, value_kind="categorical_wdl"
        ),
        schedule_clock="iteration_fraction",
        stages=[
            TrainSelfPlay(
                id=STAGE,
                operation="train_self_play",
                updates=2,
                streams=2,
                transitions=16,
                learning=DampedPolicyLearning(
                    gradient="damped_policy",
                    ema=0.9,
                    min_advantage=0,
                    advantage_quantile=0,
                ),
            )
        ],
    )
    with VerifyStore(tmp_path / "window.sqlite") as store:
        execute_regime(regime, SEED, tmp_path / "window", store)
    return tmp_path / "window"


def test_staging_binds_published_exports_at_the_parent_update(
    tmp_path: Path,
) -> None:
    parent = _window(tmp_path)
    storage = DirectoryStorage(tmp_path / "bucket")
    continuation = stage_parent(parent, storage, "mtg183/w2")
    assert continuation.from_update == 2
    assert continuation.raw.key == f"mtg183/w2/{STAGE}-raw.pt"
    assert (
        storage.get(continuation.ema.key) == (parent / f"{STAGE}-ema.pt").read_bytes()
    )
    # Restaging is idempotent; other bytes under a key are refused.
    assert stage_parent(parent, storage, "mtg183/w2") == continuation
    storage.put(continuation.raw.key, b"other")
    with pytest.raises(ValueError, match="other bytes"):
        stage_parent(parent, storage, "mtg183/w2")
    (parent / f"{STAGE}-optimizer.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="published optimizer"):
        stage_parent(parent, storage, "mtg183/w3")


def test_staging_counts_a_historical_parent_by_its_inline_update_rows(
    tmp_path: Path,
) -> None:
    # w1/w2 were written before StageRecord.updates; their exports keep one
    # diagnostics row per update and are admitted by digest, never rewritten.
    parent = _window(tmp_path)
    record = json.loads((parent / "run.json").read_bytes())
    stage = record["stages"][0]
    stage["diagnostics"] = [{"update": i} for i in range(stage.pop("updates"))]
    (parent / "run.json").write_text(json.dumps(record))
    continuation = stage_parent(parent, DirectoryStorage(tmp_path / "bucket"), "w")
    assert continuation.from_update == 2


def test_h2h_schedule_is_frozen_and_disjoint_from_the_heuristic_grid() -> None:
    games = h2h_schedule()
    assert h2h_schedule_sha256() == H2H_SCHEDULE_SHA256
    cells = Counter((g.newer_deck, g.older_deck) for g in games)
    assert len(cells) == len(DECKS) ** 2
    assert set(cells.values()) == {H2H_GAMES_PER_CELL}
    assert not {g.deal for g in games} & {g.deal for g in schedule()}


def test_rated_updates_follow_the_declared_spacing() -> None:
    rated = rated_updates(3000)
    assert rated[:9] == tuple(range(0, 401, 50))
    assert 1000 in rated and 1100 not in rated and 2200 not in rated
    assert rated[-1] == 2800
    assert rated_updates(3000, 3013)[-1] == 3013


def test_heuristic_grid_is_full_at_anchors_and_endpoint() -> None:
    for updates in (*HEURISTIC_FULL, 3013):
        grid, games = heuristic_games(updates, 3013)
        assert (grid, len(games)) == ("full", 5000)
    grid, games = heuristic_games(150, 3013)
    full = {g.deal for g in schedule()}
    assert grid == "subset" and len(games) == 1000 and {g.deal for g in games} <= full


def _connected(nodes: tuple[int, ...], edges: list[tuple[int, int]]) -> bool:
    seen, frontier = {nodes[0]}, [nodes[0]]
    while frontier:
        node = frontier.pop()
        for a, b in edges:
            for x, y in ((a, b), (b, a)):
                if x == node and y not in seen:
                    seen.add(y)
                    frontier.append(y)
    return seen == set(nodes)


def test_pair_design_is_connected_with_disjoint_held_out_pairs() -> None:
    rated = rated_updates(8600, 8613)
    pairs = pair_design(rated, 8613)
    keys = [(p.newer, p.older) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(p.newer > p.older for p in pairs)
    fit = [(p.newer, p.older) for p in pairs if p.role != "held-out"]
    assert _connected(rated, fit)
    roles = Counter(p.role for p in pairs)
    assert roles["consecutive"] == len(rated) - 1
    assert roles["held-out"] > 0
    assert {(8613, older) for older in rated[:-1]} <= set(fit)
    for reference in REFERENCES:
        assert all((newer, reference) in fit for newer in rated if newer > reference)
