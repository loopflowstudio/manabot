"""Evaluation children stay light and within their declared thread budgets."""

import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from manabot.sim.child import ChildProcess
from manabot.sim.threads import (
    GAME_BUDGET,
    POOL_VARIABLES,
    REPLAY_BUDGET,
    ProcessCensus,
    ThreadBudget,
    ThreadBudgetExceeded,
    child_environment,
    effective_cpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The parent's main module imports Torch, as an evaluator's does. Spawned
# children re-imported it as __mp_main__; launched children must not.
_PARENT_MAIN = textwrap.dedent(
    """
    import json
    from pathlib import Path
    import sys
    import time

    import torch  # noqa: F401

    from manabot.eval.match import play_cell
    from manabot.sim.execution import EvalExecution
    from manabot.sim.replay import read_trace
    from manabot.sim.supervision import replay_retained
    from tests.eval.test_batched_eval import fixture_policy

    out = Path(sys.argv[1])
    key, player, anchor, path = fixture_policy(out)
    _, other, _, other_path = fixture_policy(out, 84)
    search = anchor.model_copy(
        update={"player_id": "search-1", "player_spec": {"kind": "search", "sims": 1}}
    )
    served = EvalExecution(deadline_unix=time.time() + 120, max_active_games=2)
    report = {}
    # Checkpoints are always served. A code anchor plays Torch-free in-child;
    # a search player is the one in-child Torch consumer.
    for mode, opponent in (("served", other), ("anchor", anchor), ("search", search)):
        _, trace, replay = play_cell(
            key=key,
            player_a=player,
            player_b=opponent,
            deal_seeds=(831,),
            out_dir=out / mode,
            checkpoint_paths={player.player_id: str(path), other.player_id: str(other_path)},
            execution=served,
            max_commands=4,
        )
        games = read_trace(Path(trace["path"]))
        report[mode] = {
            "replay_passed": replay["passed"],
            # Four Commands bound the fixture; a cap is the only expected stop.
            "failures": [
                g["failure"]
                for g in games
                if g["failure"] not in (None, "game exceeded its Command cap")
            ],
            "children": [g["worker_performance"]["counters"] for g in games],
        }
    outcome = replay_retained(games, 60)
    report["replay"] = {"error": outcome.error, "process": outcome.process}
    print(json.dumps(report))
    """
)


@pytest.fixture(scope="module")
def child_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("children")
    script = directory / "evaluator.py"
    script.write_text(_PARENT_MAIN)
    completed = subprocess.run(
        [sys.executable, str(script), str(directory)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return json.loads(completed.stdout.splitlines()[-1])


@pytest.mark.parametrize("mode", ["served", "anchor"])
def test_served_and_code_anchor_game_children_never_import_torch(
    child_report: dict, mode: str
) -> None:
    report = child_report[mode]
    assert report["replay_passed"] and report["failures"] == []

    for counters in report["children"]:
        assert counters["torch_loaded/game"] == 0
        assert 1 <= counters["threads/game"] <= GAME_BUDGET.threads
        assert counters["threads_budget/game"] == GAME_BUDGET.threads
        assert counters["rss_bytes/game"] > 0


def test_replay_child_never_imports_torch(child_report: dict) -> None:
    replay = child_report["replay"]
    assert replay["error"] is None
    assert replay["process"]["torch_loaded/replay"] == 0
    assert replay["process"]["threads/replay"] <= REPLAY_BUDGET.threads


def test_search_game_child_stays_within_budget(child_report: dict) -> None:
    search = child_report["search"]
    assert search["replay_passed"] and search["failures"] == []
    for counters in search["children"]:
        assert counters["torch_loaded/game"] == 1
        assert counters["threads/game"] <= GAME_BUDGET.threads


def _report_environment(connection: Connection, names: tuple[str, ...]) -> None:
    connection.send({name: os.environ.get(name) for name in names})
    connection.close()


def test_child_starts_with_single_threaded_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = ChildProcess(
        "tests.sim.test_child_surface:_report_environment", (POOL_VARIABLES,)
    )
    try:
        assert child.connection.recv() == dict.fromkeys(POOL_VARIABLES, "1")
        child.join(60)
    finally:
        child.close()
    assert child.exitcode == 0
    missing = ChildProcess("manabot.sim.child_absent:main", ())
    missing.join(60)
    assert missing.exitcode not in (None, 0)
    missing.close()
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("HOME", "/h")
    assert child_environment() == {
        **os.environ,
        **dict.fromkeys(POOL_VARIABLES, "1"),
    }
    assert child_environment()["HOME"] == "/h"


def test_effective_cpus_honors_affinity_and_cgroup_quota(tmp_path: Path) -> None:
    logical = os.cpu_count() or 1
    unlimited = tmp_path / "unlimited"
    unlimited.mkdir()
    (unlimited / "cpu.max").write_text("max 100000\n")
    assert effective_cpus(affinity=set(range(logical)), cgroup_root=unlimited) == (
        logical
    )
    assert effective_cpus(affinity={0}, cgroup_root=unlimited) == 1
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("150000 100000\n")
    assert effective_cpus(affinity=set(range(logical)), cgroup_root=v2) == min(
        logical, 2
    )
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("100000\n")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert effective_cpus(affinity=set(range(logical)), cgroup_root=v1) == 1
    assert effective_cpus(affinity=set(range(logical)), cgroup_root=tmp_path) == (
        logical
    )


def test_census_records_maxima_and_rejects_excess() -> None:
    counters: dict[str, int] = {}
    census = ProcessCensus(3, {"python": 1, "worker": 2}, 10, False)
    budget = ThreadBudget("probe", 2)
    census.record(counters, budget)
    ProcessCensus(1, {}, 20, True).record(counters, budget)
    assert counters == {
        "threads/probe": 3,
        "threads_budget/probe": 2,
        "rss_bytes/probe": 20,
        "torch_loaded/probe": 1,
        "thread_names/probe/python": 1,
        "thread_names/probe/worker": 2,
    }
    with pytest.raises(ThreadBudgetExceeded, match="3 threads, budget 2"):
        census.enforce(budget)
