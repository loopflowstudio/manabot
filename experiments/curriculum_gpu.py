"""Play a curriculum grid through the existing bounded shared GPU service.

Etude still owns matches, checkpoint delivery, table publication and rentals.
Each match owns one resident inference allocation; concurrent groups share its
four-game limit. Conditions retain the compute contract so GPU evidence cannot
collide with historical scalar CPU matches. Decks, deals, native seats, policy
seeds and replay validation come from the original Elemental grid.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from etude.eval.match import Tables
from etude.eval.records import Benchmark, Inference, Match

from experiments.elemental_grid import GameRow, Group, Request, play_group
from experiments.elemental_long_eval import (
    INFERENCE_CODE,
    ROOT,
    SEED,
    Conditions,
    _local,
    evidence_directory,
    game_rows,
)
from manabot.infra.files import atomic_json
from manabot.sim.execution import EvalCompute, EvalExecution
from manabot.sim.serving import InferenceRuntime

COMPUTE = EvalCompute(
    device="cuda", max_active_games=4, max_batch_rows=4, max_delay_seconds=0.005
)


class GPUConditions(Conditions):
    """The original schedule plus its explicit serving identity."""

    compute: EvalCompute


def benchmark(original: Benchmark) -> Benchmark:
    """Preserve the authored population and anchor; separate GPU match identity."""
    matchups = []
    for matchup in original.matchups:
        candidate, anchor = matchup.players
        if not isinstance(candidate, Inference):
            raise ValueError("GPU curriculum benchmark requires one model placeholder")
        inference = Inference(
            source=[*INFERENCE_CODE, ROOT / "manabot/sim", Path(__file__)],
            config=candidate.config,
        )
        conditions = tuple(
            GPUConditions(
                **Conditions.model_validate(row).model_dump(), compute=COMPUTE
            ).model_dump(mode="json")
            for row in matchup.conditions
        )
        matchups.append(
            matchup.model_copy(
                update={
                    "players": (inference, anchor),
                    "conditions": conditions,
                }
            )
        )
    return original.model_copy(
        update={"name": original.name + "-gpu", "matchups": tuple(matchups)}
    )


def play(match: Match) -> Tables:
    """Run one checkpoint's row with bounded CUDA concurrency and native replay."""
    import torch

    torch.set_num_threads(1)
    conditions = GPUConditions.model_validate(match.conditions)
    if conditions.kind == "pair" or conditions.compute != COMPUTE:
        raise ValueError("expected the declared heuristic GPU grid")
    if any(group.model_deck != conditions.deck for group in conditions.groups):
        raise ValueError("a group plays another deck than its match")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requires a GPU; no CPU fallback")
    checkpoint = _local(match.players[0])
    root = evidence_directory() / match.key / f"attempt-{time.time_ns()}"
    execution = EvalExecution(
        **conditions.compute.model_dump(), deadline_unix=time.time() + 3600
    )
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "execution.json", execution.model_dump(mode="json"))
    with InferenceRuntime(execution, root / "allocation") as inference:

        def run(group: Group) -> list[GameRow]:
            output: Path = root / f"{group.heuristic_deck}-seat-{group.model_seat}"
            request = Request(
                checkpoint=checkpoint.path,
                sha256=checkpoint.sha256,
                training_seed=SEED,
                group=group,
                output=output,
            )
            return play_group(request, inference=inference)

        # Etude uses one callback process. This executor feeds the allocation;
        # InferenceRuntime enforces the shared game bound across all groups.
        with ThreadPoolExecutor(max_workers=COMPUTE.max_active_games) as pool:
            played = list(pool.map(run, conditions.groups))
    return {
        "games": game_rows([row for group in played for row in group], conditions.kind)
    }
