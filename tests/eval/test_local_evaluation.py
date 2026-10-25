"""Local evidence uses the shared arena without training or cloud execution."""

from collections.abc import Iterator
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import pytest
import torch

from manabot.eval.endpoint import CallbackEndpoint
from manabot.eval.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    evaluate,
    evaluate_local,
)
from manabot.eval.local import checkpoint_request, main as local_main
from manabot.eval.models import EvaluationContract
from manabot.eval.results import EvaluationRow
from manabot.eval.serving_benchmark import Panel, _panel_worker, _select_profile_roots
from manabot.sim.execution import EvalExecution
from manabot.sim.replay import read_trace
from manabot.sim.serving import InferenceRuntime
from tests.eval.test_batched_eval import fixture_policy


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
    )


def _anchor_request(seeds: tuple[int, ...] = (101,)) -> EvaluationRequest:
    contract = EvaluationContract.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/int-6-skill-arena-v1.json").read_bytes()
    )
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
        request: EvaluationRequest, *, limits: EvalExecution, out_dir: Path
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


def test_calibration_selects_declared_roots_in_deal_leg_order() -> None:
    games: list[dict[str, object]] = [
        {
            "match_id": identity,
            "deal_seed": seed,
            "leg": leg,
            "decisions": [{"revision": 0}, {"revision": 1}],
        }
        for seed, leg, identity in ((2, 0, "later"), (1, 1, "leg"), (1, 0, "first"))
    ]
    roots = _select_profile_roots(games, measured=3)
    assert [(root.game["match_id"], root.revision) for root in roots] == [
        ("first", 0),
        ("first", 1),
        ("leg", 0),
    ]
    with pytest.raises(ValueError, match="6 roots; 7 are required"):
        _select_profile_roots(games, measured=7)


def test_calibration_rejects_partial_replayed_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _anchor_request()

    def play_cell(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        rows = _replayed_rows(request)[:-1]
        return [row.model_dump(mode="json") for row in rows], {}, {"passed": True}

    monkeypatch.setattr("manabot.eval.evaluation.play_cell", play_cell)
    receiver, sender = mp.Pipe(duplex=False)
    out = tmp_path / "calibration"
    try:
        _panel_worker(
            sender,
            Panel(name="partial", request=request),
            EvalExecution(deadline_unix=time.time() + 30),
            "cold_scalar",
            out,
        )
        receipt = receiver.recv()
    finally:
        receiver.close()
    assert not receipt["passed"]
    assert receipt["games"] == 1
    result = EvaluationResult.model_validate_json(
        (out / "evaluation.json").read_bytes()
    )
    assert result.status == "incomplete" and result.score is None
    assert all(row.valid for row in result.rows)


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
        if fullname.startswith(("manabot.training", "manabot.experiment", "manabot.eval.endpoint", "etude.eval", "etude.launch", "boto3", "botocore")):
            raise AssertionError("forbidden managed import: " + fullname)
sys.meta_path.insert(0, Deny())

def forbidden(*args, **kwargs):
    raise AssertionError("forbidden cloud/credential/callback entry point")

socket.create_connection = forbidden
from manabot.model import checkpoint
checkpoint.storage_from_uri = forbidden
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
            "--workers",
            "4",
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
    serving = json.loads((out / "panel/allocation/serving.json").read_bytes())
    assert serving["counters"]["model_loads"] == 1
    assert serving["counters"]["catalog_builds"] == 1
    assert (out / "panel/performance.json").exists()
    assert (out / "panel/report-run.json").exists()


def test_local_and_managed_share_admission_seeds_and_incomplete_results(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    limits = EvalExecution(
        deadline_unix=time.time() + 120, max_active_games=2, max_batch_rows=1
    )
    local = evaluate_local(request, limits=limits, out_dir=tmp_path / "local")
    with InferenceRuntime(limits, tmp_path / "managed-allocation") as runtime:
        with CallbackEndpoint(runtime) as endpoint:
            managed = evaluate(
                request, out_dir=tmp_path / "managed", inference=endpoint
            )
    assert local.request == managed.request
    assert local.status == managed.status == "incomplete"
    assert local.score is managed.score is None
    assert len(local.rows) == len(managed.rows) == 4
    for left, right in zip(local.rows, managed.rows, strict=True):
        assert left.failure == right.failure
        assert left.replay_passed and right.replay_passed
        assert left.model_extra["player_a_seed"] == right.model_extra["player_a_seed"]
        assert left.model_extra["player_b_seed"] == right.model_extra["player_b_seed"]
    left_trace = read_trace(tmp_path / "local" / local.rows[0].trace_path)
    right_trace = read_trace(tmp_path / "managed" / managed.rows[0].trace_path)
    assert [[d["command"] for d in g["decisions"]] for g in left_trace] == [
        [d["command"] for d in g["decisions"]] for g in right_trace
    ]
    assert (
        EvaluationResult.model_validate_json(
            (tmp_path / "local/evaluation.json").read_bytes()
        )
        == local
    )


def test_checkpoint_pair_uses_two_resident_policies(tmp_path: Path) -> None:
    request = _request(tmp_path, second=True, commands=2)
    result = evaluate_local(
        request,
        limits=EvalExecution(deadline_unix=time.time() + 120),
        out_dir=tmp_path / "pair",
    )
    assert result.error is None
    assert result.status == "incomplete" and len(result.rows) == 4
    assert all(row.replay_passed for row in result.rows)
    counters = json.loads((tmp_path / "pair/allocation/serving.json").read_bytes())[
        "counters"
    ]
    assert counters["model_loads"] == 2
    assert counters["catalog_builds"] == 2


@pytest.mark.parametrize("seconds", [-1.0, 0.1])
def test_deadline_retains_incomplete_receipt_and_reaps_children(
    tmp_path: Path, seconds: float
) -> None:
    request = _request(tmp_path)
    before = {child.pid for child in mp.active_children()}
    started = time.monotonic()
    result = evaluate_local(
        request,
        limits=EvalExecution(deadline_unix=time.time() + seconds),
        out_dir=tmp_path / "timeout",
    )
    assert time.monotonic() - started < 15
    assert {child.pid for child in mp.active_children()} == before
    assert result.status == "incomplete" and result.score is None
    assert result.error is not None
    assert json.loads((tmp_path / "timeout/unattempted.json").read_bytes())
    assert (tmp_path / "timeout/performance.json").exists()
    assert (
        EvaluationResult.model_validate_json(
            (tmp_path / "timeout/evaluation.json").read_bytes()
        )
        == result
    )


def test_drift_and_nonlocal_compute_fail_without_fallback(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="CPU"):
        evaluate_local(
            request,
            limits=EvalExecution(device="cuda", deadline_unix=time.time() + 30),
            out_dir=tmp_path / "cuda",
        )
    assert not (tmp_path / "cuda").exists()
    Path(request.checkpoint_paths[request.candidate.player_id]).write_bytes(b"changed")
    result = evaluate_local(
        request,
        limits=EvalExecution(deadline_unix=time.time() + 30),
        out_dir=tmp_path / "drift",
    )
    assert result.status == "incomplete"
    assert "byte identity drift" in result.error
    assert not result.rows
    with pytest.raises(FileExistsError):
        evaluate_local(
            request,
            limits=EvalExecution(deadline_unix=time.time() + 30),
            out_dir=tmp_path / "drift",
        )


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
