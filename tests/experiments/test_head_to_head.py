"""Report arithmetic for the head-to-head runner, over synthetic retained rows."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

from experiments.runners import head_to_head
from experiments.runners.head_to_head import GREEDY, Unit
from manabot.arena.models import ArenaKey, canonical_sha256
from manabot.training.execution import atomic_json

KEY = ArenaKey(
    world="w4",
    content_suite="suite",
    viewer_boundary="acting-viewer",
    arena_version="test",
    rating_model_version="test",
    rating_prior_sha256=canonical_sha256({}),
    anchor_cohort_sha256=canonical_sha256({}),
    evaluation_compute_envelope_id="test",
)


def _write_unit(
    out: Path,
    pairing: tuple[str, str],
    seed: int,
    scores: tuple[float, ...],
    *,
    mirrors: bool = False,
) -> None:
    """Persist four rows laid out as the arena does: legs alternate player_a's seat."""
    rows = []
    for leg, score in enumerate(scores, start=4 if mirrors else 0):
        decks = ["ur_lessons", "gw_allies"] if leg < 2 else ["gw_allies", "ur_lessons"]
        if mirrors:
            decks = ["ur_lessons"] * 2 if leg < 6 else ["gw_allies"] * 2
        rows.append(
            {
                "arena_key": KEY.model_dump(),
                "deal_seed": seed,
                "deal_block": 0,
                "leg": leg,
                "player_a": pairing[0],
                "player_b": pairing[1],
                "player_a_registration_sha256": "a",
                "player_b_registration_sha256": "b",
                "player_a_seat": leg % 2,
                "seat_decks": decks,
                "score_a": score,
                "failure": None,
                "terminated": True,
                "truncated": False,
                "replay_passed": True,
                "trace_path": "traces/x",
                "game_seconds": 1.0,
                "integrity": {"illegal_actions": 0},
            }
        )
    directory = out / "units" / Unit(*pairing, seed).directory
    directory.mkdir(parents=True)
    atomic_json(directory / "rows.json", rows)


def test_report_splits_and_orients_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "strong" is scheduled as player_b of its meeting with "weak", so the
    # contrast must flip that cell to read it from strong's side.
    pairings = [("strong", GREEDY), ("weak", GREEDY), ("weak", "strong")]
    seeds = [1, 2, 3]
    atomic_json(tmp_path / "design.json", {"pairings": pairings, "deal_seeds": seeds})
    for seed in seeds:
        # Legs: Lessons on play, Allies on draw, Allies on play, Lessons on draw.
        _write_unit(tmp_path, ("strong", GREEDY), seed, (0.0, 1.0, 1.0, 1.0))
        _write_unit(tmp_path, ("weak", GREEDY), seed, (0.0, 1.0, 1.0, 0.0))
        _write_unit(tmp_path, ("weak", "strong"), seed, (0.0, 0.0, 1.0, 0.0))
    monkeypatch.setattr(head_to_head, "CONTRASTS", (("strong", "weak"),))
    found = head_to_head.report(tmp_path)

    strong = found.pairings[0]
    assert strong.games == 12 and strong.rejected is None
    assert strong.score is not None and strong.score.mean == 0.75
    assert strong.as_lessons is not None and strong.as_lessons.mean == 0.5
    assert strong.as_allies is not None and strong.as_allies.mean == 1.0
    assert strong.on_play is not None and strong.on_play.mean == 0.5
    assert strong.on_draw is not None and strong.on_draw.mean == 1.0

    (contrast,) = found.contrasts
    assert contrast.deals == 3
    assert contrast.greedy_difference.mean == 0.25
    assert contrast.direct.mean == 0.75
    assert found.ratings["strong"][1] > found.ratings["weak"][1]
    assert abs(found.ratings[GREEDY][1]) < 1e-6


def test_invalid_game_rejects_its_pairing(tmp_path: Path) -> None:
    atomic_json(
        tmp_path / "design.json",
        {"pairings": [("weak", GREEDY)], "deal_seeds": [1, 2, 3]},
    )
    for seed in (1, 2, 3):
        _write_unit(tmp_path, ("weak", GREEDY), seed, (0.0, 1.0, 1.0, 0.0))
    path = tmp_path / "units" / Unit("weak", GREEDY, 3).directory / "rows.json"
    path.write_text(
        path.read_text().replace('"replay_passed": true', '"replay_passed": false')
    )

    (score,) = head_to_head.report(tmp_path).pairings
    assert score.rejected == "invalid games"
    assert score.score is None and score.invalid_games == 4


def test_schedule_covers_every_contrast() -> None:
    scheduled = {frozenset(pair) for pair in head_to_head.pairings()}
    for first, second in head_to_head.CONTRASTS:
        assert frozenset((first, second)) in scheduled
        assert frozenset((first, GREEDY)) in scheduled
        assert frozenset((second, GREEDY)) in scheduled


def test_mirror_plan_is_shared_disjoint_and_keeps_anchor() -> None:
    plan = head_to_head.full_plan("mirrors")
    assert plan.entrants == head_to_head.ENTRANTS
    assert plan.pairings == head_to_head.pairings()
    assert not set(plan.deal_seeds) & set(head_to_head.DEAL_SEEDS)
    assert not set(plan.deal_seeds) & set(head_to_head.smoke_plan("mirrors").deal_seeds)
    for pair in plan.pairings:
        assert [
            u.deal_seed for u in plan.units() if (u.player_a, u.player_b) == pair
        ] == list(plan.deal_seeds)
    assert all(u.matchup_mode == "mirrors" for u in plan.units())


def test_mirror_deck_seat_contrasts_and_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = [("strong", GREEDY), ("weak", GREEDY), ("weak", "strong")]
    seeds = [1, 2, 3, 4]
    atomic_json(
        tmp_path / "design.json",
        {
            "pairings": pairs,
            "deal_seeds": seeds,
            "matchup_mode": "mirrors",
        },
    )
    for seed in seeds:
        _write_unit(tmp_path, pairs[0], seed, (1.0, 1.0, 0.0, 0.0), mirrors=True)
        _write_unit(tmp_path, pairs[1], seed, (0.0, 0.0, 1.0, 1.0), mirrors=True)
        # Strong wins Lessons, loses Allies. Aggregate hides this exactly.
        _write_unit(tmp_path, pairs[2], seed, (0.0, 0.0, 1.0, 1.0), mirrors=True)
    monkeypatch.setattr(head_to_head, "CONTRASTS", (("strong", "weak"),))
    found = head_to_head.report(tmp_path)
    for deck, expected in (("ur_lessons", 1.0), ("gw_allies", 0.0)):
        panel = found.decks[deck]
        score = panel.pairings[0]
        assert score.games == 8 and score.invalid_games == 0
        assert score.score.mean == expected
        assert score.on_play.mean == expected
        assert score.on_draw.mean == expected
        assert panel.contrasts[0].direct.mean == expected
        assert panel.contrasts[0].greedy_difference.mean == 2 * expected - 1
        assert (panel.ratings["strong"][1] > panel.ratings["weak"][1]) == bool(expected)
        assert abs(panel.ratings[GREEDY][1]) < 1e-6
    text = head_to_head.render(found)
    assert "Lessons vs Lessons" in text and "Allies vs Allies" in text
    assert head_to_head.TRAINING_CAVEAT in text


def test_mirror_intervals_resample_deals_not_games(tmp_path: Path) -> None:
    pair = ("strong", GREEDY)
    seeds = list(range(12))
    atomic_json(
        tmp_path / "design.json",
        {
            "pairings": [pair],
            "deal_seeds": seeds,
            "matchup_mode": "mirrors",
        },
    )
    for seed in seeds:
        # Every deal averages exactly half; independent-game resampling would
        # fabricate uncertainty even though these paired observations agree.
        _write_unit(tmp_path, pair, seed, (1.0, 0.0, 1.0, 0.0), mirrors=True)
    panel = head_to_head.report(tmp_path).decks["ur_lessons"]
    score = panel.pairings[0]
    assert score.score == head_to_head.Interval(0.5, 0.5, 0.5)
    assert score.on_play == head_to_head.Interval(1.0, 1.0, 1.0)
    assert score.on_draw == head_to_head.Interval(0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "corruption", ["duplicate", "deck", "seat", "seed", "pair", "replay", "unfinished"]
)
def test_mirror_corruption_is_not_dropped(tmp_path: Path, corruption: str) -> None:
    pair = ("weak", GREEDY)
    atomic_json(
        tmp_path / "design.json",
        {
            "pairings": [pair],
            "deal_seeds": [1, 2, 3],
            "matchup_mode": "mirrors",
        },
    )
    for seed in (1, 2, 3):
        _write_unit(tmp_path, pair, seed, (0.0, 1.0, 1.0, 0.0), mirrors=True)
    path = tmp_path / "units" / Unit(*pair, 3).directory / "rows.json"
    rows = json.loads(path.read_text())
    if corruption == "unfinished":
        path.unlink()
    else:
        name, value = {
            "duplicate": ("leg", 5),
            "deck": ("seat_decks", ["gw_allies", "ur_lessons"]),
            "seat": ("player_a_seat", 1),
            "seed": ("deal_seed", 99),
            "pair": ("player_a", "other"),
            "replay": ("replay_passed", False),
        }[corruption]
        rows[0][name] = value
        atomic_json(path, rows)
    found = head_to_head.report(tmp_path)
    for panel in found.decks.values():
        assert panel.pairings[0].rejected
        assert panel.pairings[0].score is None
        assert panel.pairings[0].invalid_games or panel.pairings[0].failed_units
        assert not panel.ratings and not panel.contrasts


def test_mirror_mode_cannot_overwrite_cross_run(tmp_path: Path) -> None:
    atomic_json(tmp_path / "design.json", {"matchup_mode": "cross"})
    atomic_json(tmp_path / "status.json", {"state": "running"})
    before = (tmp_path / "status.json").read_bytes()
    with pytest.raises(ValueError, match="different matchup mode"):
        head_to_head.run(
            tmp_path,
            tmp_path / "missing-models",
            plan=head_to_head.smoke_plan("mirrors"),
        )
    assert (tmp_path / "status.json").read_bytes() == before


<<<<<<< HEAD
def _grid_tournament() -> head_to_head.Tournament:
    return head_to_head.Tournament.model_validate_json(
        Path("experiments/tournaments/size-floor-grid.json").read_text()
    )


=======
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
@pytest.mark.parametrize(
    "corruption",
    [
        "unknown",
        "duplicate",
        "missing-reference",
        "unknown-reference",
        "missing-greedy",
        "duplicate-id",
        "extra-field",
        "incomplete",
        "bool-size",
        "unsafe-id",
        "bad-hash",
        "unscheduled-contrast",
    ],
)
def test_tournament_rejects_invalid_design(corruption: str) -> None:
<<<<<<< HEAD
    data = _grid_tournament().model_dump(mode="json")
=======
    data = head_to_head.size_floor_grid().model_dump(mode="json")
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    if corruption == "unknown":
        data["core"].append("unknown")
    elif corruption == "duplicate":
        data["extra"].append(["w64-floor003", "w64-floor010"])
    elif corruption == "missing-reference":
        del data["reference"]
    elif corruption == "unknown-reference":
        data["reference"] = "unknown"
    elif corruption == "missing-greedy":
        data["entrants"].pop(0)
    elif corruption == "duplicate-id":
        data["entrants"].append(data["entrants"][1])
    elif corruption == "extra-field":
        data["entrants"][1]["typo"] = 1
    elif corruption == "incomplete":
        del data["entrants"][1]["sha256"]
    elif corruption == "bool-size":
        data["entrants"][1]["bytes"] = True
    elif corruption == "unsafe-id":
        data["entrants"][1]["id"] = "../escape"
    elif corruption == "bad-hash":
        data["entrants"][1]["sha256"] = "x" * 64
    else:
        data["core"] = []
    with pytest.raises(ValueError):
        head_to_head.Tournament.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("mode", ["cross", "mirrors"])
def test_custom_smoke_and_saved_report(
    tmp_path: Path,
    mode: head_to_head.MatchupMode,
) -> None:
<<<<<<< HEAD
    spec = _grid_tournament()
=======
    spec = head_to_head.size_floor_grid()
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    plan = head_to_head.smoke_plan(mode, spec)
    assert len(plan.pairings) == 6
    assert len(plan.units()) == 12
    assert plan.entrants == spec.entrants
    atomic_json(
        tmp_path / "design.json",
        {
            "roster": [asdict(e) for e in spec.entrants],
            "roster_spec": spec.model_dump(mode="json"),
            "pairings": plan.pairings,
            "deal_seeds": plan.deal_seeds,
            "matchup_mode": mode,
        },
    )
    for unit in plan.units():
        _write_unit(
            tmp_path,
            (unit.player_a, unit.player_b),
            unit.deal_seed,
            (1.0, 0.0, 1.0, 0.0),
            mirrors=mode == "mirrors",
        )
    found = head_to_head.report(tmp_path)
    # Pre-rename runner dd1ff797 on these exact synthetic grid rows.
    expected = {
        "cross": (
            "c43aff2014fb3464f92c8774d7aa2d4037d8b48e370d7eda95456e467edbda32",
            "9570ab6fca16a959cefe8ca115a51a7c63e0b811accfaf537e02d2e6ce9cdb71",
        ),
        "mirrors": (
            "b3c974dcfb5abdb7d5419d6374f468f9c5a809e1deae1079e6b87de4bbfd823d",
            "11b343f876c2a0e4e265077a1ba536f2e0223317f49d06072bf3f510aebba252",
        ),
    }
    assert canonical_sha256(asdict(found)) == expected[mode][0]
    assert canonical_sha256(head_to_head.render(found)) == expected[mode][1]
    assert isinstance(found, head_to_head.TournamentReport)
    assert found.reference == spec.reference
    assert len(found.contrasts) == 2
    assert spec.training_caveat in head_to_head.render(found)
    if mode == "mirrors":
        assert all(len(panel.contrasts) == 2 for panel in found.decks.values())


@pytest.mark.parametrize("mode", ["cross", "mirrors"])
def test_extension_reuses_finished_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: head_to_head.MatchupMode,
) -> None:

    # Only execution/admission are fixtures: scheduling, persistence, continuation
    # and reporting run through the ordinary entry point, without real games.
    def admit(
        models: Path,
        entrants: Sequence[head_to_head.Entrant],
        *,
        matchup_mode: head_to_head.MatchupMode,
    ) -> head_to_head.Admitted:
        return head_to_head.Admitted(KEY, {}, {})

    def pool(workers: int, *, mp_context: object) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(workers)

    played: list[Unit] = []

    def play(
        unit: Unit,
        admitted: head_to_head.Admitted,
        out: str,
        seconds: float,
    ) -> tuple[Unit, int, float]:
        _write_unit(
            Path(out),
            (unit.player_a, unit.player_b),
            unit.deal_seed,
            (1.0, 0.0, 1.0, 0.0),
            mirrors=unit.matchup_mode == "mirrors",
        )
        played.append(unit)
        return unit, 0, 0.0

    monkeypatch.setattr(head_to_head, "admit", admit)
    monkeypatch.setattr(head_to_head, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(head_to_head, "_play_unit", play)
<<<<<<< HEAD
    full = _grid_tournament()
=======
    full = head_to_head.size_floor_grid()
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    initial = head_to_head.Tournament(
        entrants=full.entrants[:3],
        core=full.core[:3],
        extra=(),
        contrasts=full.contrasts[:1],
        reference=full.reference,
        training_caveat=full.training_caveat,
    )
    first = head_to_head.smoke_plan(mode, initial)
    second = head_to_head.smoke_plan(mode, full)
    assert (
        head_to_head.run(tmp_path, tmp_path, plan=first, min_free_gib=0).state
        == "completed"
    )
    original = {p: p.read_bytes() for p in tmp_path.glob("units/**/rows.json")}
    assert len(played) == 6
    assert (
        head_to_head.run(tmp_path, tmp_path, plan=second, min_free_gib=0).state
        == "completed"
    )
    assert len(played) == 12
    assert all(p.read_bytes() == content for p, content in original.items())
    assert head_to_head.report(tmp_path).reference == full.reference
    design_before = (tmp_path / "design.json").read_bytes()
    status_before = (tmp_path / "status.json").read_bytes()
    for bad in (
        replace(second, deal_seeds=(999,)),
        replace(
            second,
            entrants=(
                full.entrants[0],
                replace(full.entrants[1], sha256="0" * 64),
                *full.entrants[2:],
            ),
        ),
        first,
        replace(second, pairings=tuple((b, a) for a, b in second.pairings)),
    ):
        with pytest.raises(ValueError, match="different design"):
            head_to_head.run(tmp_path, tmp_path, plan=bad, min_free_gib=0)
    assert len(played) == 12
    assert (tmp_path / "design.json").read_bytes() == design_before
    assert (tmp_path / "status.json").read_bytes() == status_before
    # A previously started but unfinished unit is not silently retried.
    next(iter(original)).unlink()
    status = head_to_head.run(tmp_path, tmp_path, plan=second, min_free_gib=0)
    assert status.state == "incomplete" and status.units_failed == 1
    assert len(played) == 12


@pytest.mark.parametrize(
    ("mode", "plan_hash", "report_hash", "render_hash"),
    [
        (
            "cross",
            "fc061d88f428b4675ae7021227b8127b165db0cb33ae395e5f4a376a27dbdc3c",
            "4d5e174f7490908fa6df555c16f146213939d26a1845c7bf40229af35f189bca",
            "d671aa30e06c61be698d5438b8720c4d0282a63c934334d74bde4c7c33f26ac5",
        ),
        (
            "mirrors",
            "42c1f752e012e75a4a7c1135c63c63d01a6c5a75b5a96f68fd32ac6ae6b22324",
            "35ad409adebef3a004b1dfc1e7d9ee984582c7c709d5aa92107ae932af24f52a",
            "5fb60579436693141aeb3d846f1d79dcbc4615636cf2219ef04ec977a6f7115a",
        ),
    ],
)
def test_default_design_and_historical_report_stability(
    tmp_path: Path,
    mode: head_to_head.MatchupMode,
    plan_hash: str,
    report_hash: str,
    render_hash: str,
) -> None:
    # Goldens produced by c9c28153's runner on these same synthetic rows.
    # No completed scientific cohort is reconstructed or relabeled here.
    plan_data = asdict(head_to_head.full_plan(mode))
    assert plan_data.pop("tournament") is None
    plan_data["roster"] = plan_data.pop("entrants")
    assert canonical_sha256(plan_data) == plan_hash
    assert canonical_sha256([asdict(e) for e in head_to_head.ENTRANTS]) == (
        "9b404f7d21bf326607eb38f2a2940847548ede82d49fe86275d69e000cf07328"
    )
    plan = head_to_head.smoke_plan(mode)
    design: dict[str, object] = {
        "pairings": plan.pairings,
        "deal_seeds": plan.deal_seeds,
    }
    if mode == "mirrors":
        design.update(matchup_mode=mode, training_caveat=head_to_head.TRAINING_CAVEAT)
    atomic_json(tmp_path / "design.json", design)
    for unit in plan.units():
        _write_unit(
            tmp_path,
            (unit.player_a, unit.player_b),
            unit.deal_seed,
            (1.0, 0.0, 1.0, 0.0),
            mirrors=mode == "mirrors",
        )
    found = head_to_head.report(tmp_path)
    assert canonical_sha256(asdict(found)) == report_hash
    assert canonical_sha256(head_to_head.render(found)) == render_hash


@pytest.mark.parametrize("command", ["smoke", "run"])
def test_tournament_cli(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(
        out: Path,
        models: Path,
        *,
        plan: head_to_head.Plan,
        workers: int,
        wall_seconds: float,
        min_free_gib: float,
    ) -> None:
<<<<<<< HEAD
        assert plan.tournament == _grid_tournament()
=======
        assert plan.tournament == head_to_head.size_floor_grid()
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
        raise RuntimeError("execution boundary")

    monkeypatch.setattr(head_to_head, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "head_to_head",
            command,
            str(tmp_path),
            "--models",
            str(tmp_path),
            "--tournament",
<<<<<<< HEAD
            "experiments/tournaments/size-floor-grid.json",
=======
            "size-floor-grid",
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
        ],
    )
    with pytest.raises(RuntimeError, match="execution boundary"):
        head_to_head.main()


def test_tournament_subcommand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["head_to_head", "tournament"])
    head_to_head.main()
    assert head_to_head.ENTRANTS[0].id in capsys.readouterr().out
