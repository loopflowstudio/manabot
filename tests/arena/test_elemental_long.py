"""MTG-183's frozen windows, rated checkpoints and pairing graph."""

from collections import Counter

from experiments.elemental_grid import DECKS, schedule
from experiments.elemental_long import (
    CAP_DOLLARS,
    H2H_GAMES_PER_CELL,
    H2H_SCHEDULE_SHA256,
    HEURISTIC_FULL,
    REFERENCES,
    WINDOWS,
    h2h_schedule,
    h2h_schedule_sha256,
    heuristic_games,
    pair_design,
    rated_updates,
)
from manabot.remote.plan import MAX_JOB_SECONDS


def test_windows_fit_the_cap_and_the_credential_limit() -> None:
    assert sum(spec.spending_limit for _, spec in WINDOWS) <= CAP_DOLLARS
    assert all(spec.lifetime_seconds <= MAX_JOB_SECONDS - 60 for _, spec in WINDOWS)
    assert all(spec.projected_dollars < spec.spending_limit for _, spec in WINDOWS)


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
