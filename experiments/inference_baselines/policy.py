"""Policy baseline execution through shared admission, play and retained roots."""

from __future__ import annotations

from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from pydantic import JsonValue
import torch

from manabot.env import ObservationSpace
from manabot.eval.evaluation import EvaluationRequest
from manabot.eval.match import play_cell
from manabot.infra.config import MatchConfig
from manabot.sim.execution import EvalCompute, EvalExecution
from manabot.sim.performance import EvalPerformance
from manabot.sim.replay import read_trace, replay_prefix
from manabot.sim.serving import (
    DecisionRequest,
    PolicyAdmission,
    PolicyKey,
    PolicyStore,
    _infer,
)

if TYPE_CHECKING:
    from experiments.inference_baseline import Cell, PolicyWorkload, Protocol


def execute(
    protocol: Protocol,
    workload: PolicyWorkload,
    cell: Cell,
    out: Path,
    replay: bool,
    deadline: float,
) -> dict[str, JsonValue]:
    from experiments.inference_baseline import _write

    preparation_start = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    request = EvaluationRequest.model_validate_json(workload.request.read_bytes())
    if request.game_seconds > 120 or request.max_commands > 10_000:
        raise ValueError("policy game bounds exceed the baseline ceiling")
    if len(request.deal_seeds) > 4 or not set(workload.legs) <= set(request.legs):
        raise ValueError("invalid bounded deal/leg selection")
    compute = (
        EvalCompute.model_validate_json(cell.execution.read_bytes())
        if cell.execution
        else EvalCompute(
            max_active_games=1,
            max_batch_rows=1,
            max_delay_seconds=0,
            resident=False,
            cache_catalog=False,
        )
    )
    if (
        compute.max_active_games > 4
        or compute.max_batch_rows > 4
        or compute.max_delay_seconds != 0
    ):
        raise ValueError(
            "baseline allows at most four workers/rows and no added batch delay"
        )
    execution = EvalExecution(
        **compute.model_dump(),
        deadline_unix=deadline,
        instrumentation=cell.instrumentation,
        diagnostic_batches=1 if cell.mode == "diagnostic" else 0,
    )
    _write(
        out / "preparation.json",
        {
            "seconds": time.monotonic() - preparation_start,
            "boundary": "request/compute parsing; checkpoint byte verification and load belong to admission",
            "device": compute.device,
            "torch_threads": torch.get_num_threads(),
            "tf32": False,
            "execution": execution.model_dump(mode="json"),
        },
    )
    _write(out / "request.json", request)
    if replay:
        return _retained(protocol, workload, request, execution, out)
    rows, trace, replay_result = play_cell(
        key=request.key,
        player_a=request.candidate,
        player_b=request.opponent,
        deal_seeds=request.deal_seeds,
        out_dir=out,
        checkpoint_paths=request.checkpoint_paths,
        game_seconds=request.game_seconds,
        max_commands=request.max_commands,
        matchup_mode=request.matchup_mode,
        authored_matchup=request.authored_matchup,
        execution=execution if cell.execution else None,
        legs=workload.legs,
        keep_game_logs=True,
    )
    for name, value in (("trace", trace), ("replay", replay_result)):
        _write(out / f"{name}.json", value)
    completed = sum(
        bool(row["terminated"] and row["replay_passed"] and not row["failure"])
        for row in rows
    )
    expected = len(request.deal_seeds) * len(set(workload.legs))
    return {
        "status": "completed" if completed == expected else "failed",
        "reason": f"{completed}/{expected} completed replay-verified games",
        "completed_games": completed,
        "committed_decisions": sum(row["decisions"] for row in rows),
    }


def _retained(
    protocol: Protocol,
    workload: PolicyWorkload,
    request: EvaluationRequest,
    execution: EvalExecution,
    out: Path,
) -> dict[str, JsonValue]:
    from experiments.inference_baseline import _write

    if workload.trace is None:
        raise ValueError("retained-input execution requires a pinned trace")
    games = read_trace(workload.trace)
    player = request.candidate
    if player.runner_kind != "checkpoint":
        raise ValueError("retained policy inputs require a checkpoint candidate")
    roots = [
        (game, decision)
        for game in sorted(
            games, key=lambda g: (g["deal_seed"], g["leg"], g["match_id"])
        )
        for decision in game["decisions"]
        if decision["player_id"] == player.player_id
    ][: workload.roots]
    if len(roots) != workload.roots:
        raise ValueError(
            "insufficient retained candidate roots; no synthetic enlargement"
        )
    perf = EvalPerformance("retained-inputs", execution.instrumentation)
    store = PolicyStore(execution, perf)
    request = request.bind(execution)
    player = request.candidate
    references: list[np.ndarray] = []
    action_references: list[list[int]] = []
    try:
        observation_config = store.admit(
            PolicyAdmission(
                player,
                request.checkpoint_paths[player.player_id],
                tuple(MatchConfig.model_validate(g["match_hypers"]) for g, _ in roots),
            )
        )
        observations, seeds = [], []
        for game, decision in roots:
            env, observation = replay_prefix(
                game, decision["revision"], ObservationSpace(observation_config)
            )
            try:
                observations.append(observation)
                seeds.append(decision["policy_rng_seed"])
            finally:
                env.close()
        np.savez(
            out / "inputs.npz",
            **{
                key: np.stack([obs[key] for obs in observations])
                for key in observations[0]
            },
            seeds=np.asarray(seeds, dtype=np.uint64),
        )
        _write(
            out / "roots.json",
            [
                {
                    "game": g["match_id"],
                    "revision": d["revision"],
                    "seed": d["policy_rng_seed"],
                }
                for g, d in roots
            ],
        )
        for pass_index in range(protocol.recipe.warmup_passes + 1):
            phase = (
                "warmup" if pass_index < protocol.recipe.warmup_passes else "measured"
            )
            phase_start = time.monotonic()
            for batch_index, start in enumerate(
                range(0, len(roots), execution.max_batch_rows)
            ):
                stop = min(start + execution.max_batch_rows, len(roots))
                requests = [
                    DecisionRequest(
                        str(i),
                        str(roots[i][0]["match_id"]),
                        0,
                        int(roots[i][1]["revision"]),
                        PolicyKey.from_registration(player, execution),
                        i,
                        seeds[i],
                        observations[i],
                        execution.deadline_unix,
                        time.monotonic(),
                    )
                    for i in range(start, stop)
                ]

                def witness(
                    logits: torch.Tensor, masks: torch.Tensor, actions: list[int]
                ) -> None:
                    values = logits.numpy()
                    np.savez(
                        out / f"pass-{pass_index}-batch-{batch_index}.npz",
                        logits=values,
                        masks=masks.numpy(),
                        actions=np.asarray(actions),
                        seeds=np.asarray(seeds[start:stop], dtype=np.uint64),
                    )
                    if pass_index == 0:
                        references.append(values.copy())
                        action_references.append(actions)
                    elif (
                        not np.array_equal(values, references[batch_index])
                        or actions != action_references[batch_index]
                    ):
                        raise ValueError(
                            "same-device retained-input outputs changed between passes"
                        )

                _infer(store, requests, f"{pass_index}-{batch_index}", witness=witness)
            _write(
                out / f"pass-{pass_index}.json",
                {
                    "population": phase,
                    "requests": len(roots),
                    "seconds": time.monotonic() - phase_start,
                },
            )
    finally:
        store.close()
        perf.write(out / "input-performance.json")
    return {
        "status": "completed",
        "reason": "retained inputs executed; not game throughput",
        "roots": len(roots),
        "warmup_passes": protocol.recipe.warmup_passes,
        "measured_passes": 1,
    }
