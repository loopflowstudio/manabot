"""Local evidence uses the shared arena without training or cloud execution."""

from collections.abc import Iterator
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import torch

from manabot.backends.torch.checkpoint import BACKEND as TORCH_BACKEND
from manabot.eval.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    continue_evaluation,
    evaluate,
    evaluation_status,
)
from manabot.eval.local import checkpoint_request, main as local_main
from manabot.eval.results import EvaluationRow
from manabot.infra.files import atomic_json, exclusive_file_lock
from manabot.infra.identity import canonical_sha256
from manabot.sim.match import play_game
from manabot.sim.replay import read_trace, write_trace
from tests.eval.arena_fixture import arena_fixture
from tests.eval.test_evaluation_policy_inputs import fixture_policy


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _request(
    tmp_path: Path, *, commands: int = 1, second: bool = False
) -> EvaluationRequest:
    _, _, _, path = fixture_policy(tmp_path)
    if second:
        _, _, _, opponent = fixture_policy(tmp_path, 84)
    else:
        opponent = "heuristic"
    return checkpoint_request(
        path,
        opponent=opponent,
        max_commands=commands,
        deal_seeds=(101,),
        game_seconds=240,
        backend=TORCH_BACKEND,
    )


def _anchor_request(seeds: tuple[int, ...] = (101,)) -> EvaluationRequest:
    contract = arena_fixture()
    return EvaluationRequest(
        key=contract.key,
        candidate=contract.anchors[0],
        opponent=contract.anchors[1],
        deal_seeds=seeds,
    )


def _replayed_rows(request: EvaluationRequest) -> list[EvaluationRow]:
    return [
        EvaluationRow(
            arena_key=request.key,
            deal_seed=seed,
            leg=leg,
            player_a=request.candidate.player_id,
            player_b=request.opponent.player_id,
            player_a_registration_sha256=request.candidate.identity_sha256,
            player_b_registration_sha256=request.opponent.identity_sha256,
            score_a=float(seed % 2),
            failure=None,
            terminated=True,
            truncated=False,
            replay_passed=True,
            trace_path="traces/retained.commands.jsonl.gz",
            game_seconds=0.01,
            integrity={},
        )
        for seed in request.deal_seeds
        for leg in request.legs
    ]


@pytest.mark.parametrize("failed", [False, True])
def test_evaluate_rejects_repeat_without_changing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    request = _anchor_request()
    out = tmp_path / "attempt"
    calls = 0

    def play_cell(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        nonlocal calls
        calls += 1
        (out / "journal.jsonl").write_text("retained pending Command\n")
        if failed:
            raise TimeoutError("retained execution failure")
        return [row.model_dump(mode="json") for row in _replayed_rows(request)], {}, {}

    monkeypatch.setattr("manabot.eval.evaluation.play_cell", play_cell)
    result = evaluate(request, out_dir=out)
    assert result.status == ("incomplete" if failed else "completed")
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*")}
    for _ in range(2):
        with pytest.raises(FileExistsError):
            evaluate(request, out_dir=out)
    assert {
        path.relative_to(out): path.read_bytes() for path in out.rglob("*")
    } == before
    assert calls == 1
    empty = tmp_path / "already-claimed"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        evaluate(request, out_dir=empty)
    assert not list(empty.iterdir())


@pytest.mark.parametrize("seeds", [(101,), (101, 102)])
def test_cli_describes_deal_uncertainty_without_changing_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seeds: tuple[int, ...],
) -> None:
    request = _anchor_request(seeds)
    result = EvaluationResult(request=request, rows=_replayed_rows(request))
    result.validate_cohort()
    assert result.score is not None
    assert result.score.model_dump() == (
        {"mean": 1, "lower": 1, "upper": 1}
        if len(seeds) == 1
        else {"mean": 0.5, "lower": 0, "upper": 1}
    )

    def evaluate_local(
        request: EvaluationRequest, *, deadline: float, out_dir: Path
    ) -> EvaluationResult:
        return result

    monkeypatch.setattr("manabot.eval.local.evaluate_local", evaluate_local)
    source = tmp_path / "request.json"
    source.write_text(request.model_dump_json())
    assert local_main(["--request", str(source), "--out", str(tmp_path / "cli")]) == 0
    output = capsys.readouterr().out
    if len(seeds) == 1:
        assert "score: 1.000 (1 deal block; uncertainty unavailable" in output
        assert "95%" not in output
    else:
        assert "2 deal blocks; paired-deal 95% bootstrap interval 0.000–1.000" in output
    assert request.purpose in output


def test_cli_completes_without_training_cloud_or_callback(tmp_path: Path) -> None:
    _, _, _, path = fixture_policy(tmp_path)
    # Spawn imports this module-level guard too, so admission/serving/game/replay
    # children cannot silently take a managed route hidden from parent patches.
    guard = tmp_path / "offline.py"
    guard.write_text("""import importlib.abc
import socket
import sys

class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # etude.eval.source only reads the checkout's revision; every module that
        # declares, plays or stores an Etude evaluation stays forbidden.
        if fullname.startswith(("manabot.training", "manabot.experiment", "manabot.eval.endpoint", "etude.eval.", "etude.launch", "boto3", "botocore")) and fullname != "etude.eval.source":
            raise AssertionError("forbidden managed import: " + fullname)
sys.meta_path.insert(0, Deny())

def forbidden(*args, **kwargs):
    raise AssertionError("forbidden cloud/credential/callback entry point")

socket.create_connection = forbidden
from manabot.backends.torch import checkpoint
from manabot.infra import artifacts
artifacts.storage_from_uri = forbidden
checkpoint.materialize_artifact = forbidden
from manabot.eval.local import main
if __name__ == "__main__":
    raise SystemExit(main())
""")
    out = tmp_path / "offline"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(guard),
            "--checkpoint",
            str(path),
            "--opponent",
            "heuristic",
            "--deal-seed",
            "101",
            "--out",
            str(out),
            "--seconds",
            "360",
            "--game-seconds",
            "240",
        ],
        capture_output=True,
        text=True,
        timeout=390,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = EvaluationResult.model_validate_json(
        (out / "panel/evaluation.json").read_bytes()
    )
    assert result.status == "completed"
    assert len(result.rows) == 4 and all(row.valid for row in result.rows)
    assert result.score is not None
    assert result.request.candidate.training_seed is None
    assert result.request.purpose == "local-check-not-admission"
    assert {row.leg for row in result.rows} == {0, 1, 2, 3}
    trace = read_trace(out / "panel" / result.rows[0].trace_path)
    assert all(game["decisions"] for game in trace)
    assert all(decision["command"] for game in trace for decision in game["decisions"])
    evidence = {p: p.read_bytes() for p in (out / "panel").rglob("*") if p.is_file()}
    for operation in ("--status", "--continue"):
        inspected = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(guard),
                operation,
                str(out / "panel"),
                "--seconds",
                "30",
            ],
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert inspected.returncode == 0, inspected.stdout + inspected.stderr
        assert "1/1 complete deals" in inspected.stdout
    assert all(p.read_bytes() == content for p, content in evidence.items())
    assert not (out / "panel/attempts").exists()


def test_cli_admission_deadline_is_retained(tmp_path: Path) -> None:
    out = tmp_path / "prep-timeout"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "manabot.eval.local",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--seconds",
            "0.001",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    receipt = json.loads((out / "admission-failure.json").read_bytes())
    assert receipt["status"] == "incomplete"
    assert "deadline" in receipt["error"]
    assert not (out / "panel").exists()


def test_cli_saved_request_retains_deadline_result(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source = tmp_path / "request.json"
    source.write_text(request.model_dump_json())
    out = tmp_path / "saved-request"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "manabot.eval.local",
            "--request",
            str(source),
            "--seconds",
            "0.01",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    result = EvaluationResult.model_validate_json(
        (out / "panel/evaluation.json").read_bytes()
    )
    assert result.status == "incomplete" and result.score is None
    assert result.request.deal_seeds == request.deal_seeds
    assert (
        result.request.candidate.checkpoint_sha256
        == request.candidate.checkpoint_sha256
    )


def _publish_blocks(directory: Path, rows: list[EvaluationRow]) -> None:
    """Synthetic finalized evidence for lifecycle tests; no game strength claim."""

    (directory / "blocks").mkdir(exist_ok=True)
    for seed in {row.deal_seed for row in rows}:
        block = sorted(
            (row for row in rows if row.deal_seed == seed), key=lambda r: r.leg
        )
        relative = f"traces/{seed}.commands.jsonl.gz"
        trace = write_trace(
            directory / relative, [{"deal_seed": seed, "fixture": True}]
        )
        for row in block:
            row.trace_path = relative
            row.trace_shard_sha256 = trace["sha256"]
        atomic_json(
            directory / "blocks" / f"{seed}.json",
            {
                "rows_sha256": canonical_sha256(
                    [row.model_dump(mode="json") for row in block]
                ),
            },
        )
    atomic_json(directory / "rows.json", [row.model_dump(mode="json") for row in rows])


def test_continuation_finishes_missing_sixty_without_replacing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    request = _anchor_request(tuple(range(100)))
    bound = request
    cohorts: list[tuple[int, ...]] = []

    def runner(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        seeds, directory = kwargs["deal_seeds"], kwargs["out_dir"]
        assert isinstance(seeds, tuple) and isinstance(directory, Path)
        cohorts.append(seeds)
        selected = bound.model_copy(update={"deal_seeds": seeds})
        rows = _replayed_rows(selected)
        if len(cohorts) == 1:
            # Forty complete deals plus a partial 41st. Publish before failure,
            # just as incremental replay does if later execution stops.
            _publish_blocks(directory, rows[:81])
            raise TimeoutError("interrupted after forty complete blocks")
        _publish_blocks(directory, rows)
        return [row.model_dump(mode="json") for row in rows], {}, {"passed": True}

    monkeypatch.setattr("manabot.eval.evaluation.play_cell", runner)
    out = tmp_path / "evaluation"
    first = evaluate(request, out_dir=out)
    assert first.status == "incomplete" and first.error
    status = evaluation_status(out)
    assert status.complete_deals() == tuple(range(40))
    assert status.score is None
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    second = continue_evaluation(out)
    assert second.status == "completed"
    assert second.evaluation_id == first.evaluation_id
    assert len(second.rows) == 200
    assert second.score is not None and second.score.mean == 0.5
    assert cohorts == [tuple(range(100)), tuple(range(40, 100))]
    assert all((out / name).read_bytes() == data for name, data in before.items())
    assert all((out / row.trace_path).is_file() for row in second.rows)
    assert continue_evaluation(out) == second
    assert len(cohorts) == 2
    # A full repeat has another logical identity and executes the full cohort.
    fresh = evaluate(request, out_dir=tmp_path / "repeat")
    assert fresh.evaluation_id != first.evaluation_id
    assert cohorts[-1] == tuple(range(100))


def test_continuation_rejects_contract_drift_corruption_and_competing_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    request = _anchor_request((1, 2))
    bound = request

    def runner(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        directory = kwargs["out_dir"]
        assert isinstance(directory, Path)
        rows = _replayed_rows(bound)[:2]
        _publish_blocks(directory, rows)
        return [row.model_dump(mode="json") for row in rows], {}, {}

    monkeypatch.setattr("manabot.eval.evaluation.play_cell", runner)
    out = tmp_path / "evaluation"
    evaluate(request, out_dir=out)
    with pytest.raises(ValueError, match="frozen evaluation"):
        continue_evaluation(
            out,
            request=request.model_copy(update={"deal_seeds": (1, 3)}),
        )
    with exclusive_file_lock(out / ".evaluation.lock"):
        with pytest.raises(RuntimeError, match="active attempt"):
            continue_evaluation(out)
    # A managed supervisor can outlive its callback process. Its execution
    # claim prevents another attempt from racing still-publishing evidence.
    with exclusive_file_lock(out / ".execution.lock"):
        with pytest.raises(RuntimeError, match="active attempt"):
            continue_evaluation(out)
    assert not (out / "attempts").exists()
    rows = json.loads((out / "rows.json").read_bytes())
    rows[0]["score_a"] = 0.0
    atomic_json(out / "rows.json", rows)
    with pytest.raises(ValueError, match="rows digest"):
        evaluation_status(out)
    rows[0]["score_a"] = 1.0
    atomic_json(out / "rows.json", rows)
    (out / rows[0]["trace_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="trace digest"):
        continue_evaluation(out)


def test_status_recovers_published_blocks_after_coordinator_death_and_rejects_duplicates(
    tmp_path: Path,
) -> None:

    request = _anchor_request((1, 2))
    original = EvaluationResult(
        request=request, evaluation_id="retained", status="running"
    )
    out = tmp_path / "evaluation"
    out.mkdir()
    atomic_json(out / "evaluation.json", original.model_dump(mode="json"))
    _publish_blocks(out, _replayed_rows(request)[:2])
    assert evaluation_status(out).complete_deals() == (1,)
    duplicate = out / "attempts/000001"
    duplicate.mkdir(parents=True)
    atomic_json(duplicate / "evaluation.json", original.model_dump(mode="json"))
    _publish_blocks(duplicate, _replayed_rows(request)[:2])
    with pytest.raises(ValueError, match="duplicate completed"):
        evaluation_status(out)


def test_incremental_replay_retains_first_block_when_later_game_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # Real bounded native games and replay, with a coordinator failure injected
    # after the first complete deal. No checkpoint or scientific evaluation.
    request = _anchor_request((777, 778))
    calls = 0

    def interrupted(game: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("coordinator interrupted")
        return play_game(game, *args, **kwargs)

    monkeypatch.setattr("manabot.eval.match.play_game", interrupted)
    out = tmp_path / "evaluation"
    first = evaluate(request, out_dir=out)
    assert first.error and "coordinator interrupted" in first.error
    assert evaluation_status(out).complete_deals() == (777,)
    cell = "__".join(sorted((request.candidate.player_id, request.opponent.player_id)))
    aggregate = out / "traces" / f"{cell}.commands.jsonl.gz"
    assert len(read_trace(aggregate)) == 2  # diagnostics also survive interruption
    original_trace = aggregate.read_bytes()
    completed = continue_evaluation(out)
    assert completed.status == "completed"
    assert calls == 5  # two retained games, injected failure, two missing games
    assert completed.complete_deals() == (777, 778)
    assert all(read_trace(out / row.trace_path) for row in completed.rows)
    assert aggregate.read_bytes() == original_trace
