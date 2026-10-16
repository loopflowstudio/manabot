"""Diagnostic witnesses around the real MOVE learner, never timing evidence.

The context patches existing boundaries, runs each operation exactly once, and
restores them on exit. Tensor hashing synchronizes and transfers device data.
JSONL survives partial failures; at most one large-movement snapshot is retained.
No loss, optimizer, RNG or production health policy is replaced.
"""

from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import torch
from torch import Tensor

from experiments.runners.inference_preparation import state_digest
from manabot.model.agent import Agent
from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch
from manabot.training import ataraxos, objectives
from manabot.training.health import OptimizerHealth
from manabot.training.models import AtaraxosMoveLearning
from manabot.training.selection import UpdateDiagnostics


def _rng() -> dict[str, str]:
    result = {"torch_cpu": state_digest(torch.get_rng_state())}
    if torch.cuda.is_available():
        result["torch_cuda"] = state_digest(torch.cuda.get_rng_state_all())
    return result


def _parameters(model: torch.nn.Module) -> dict[str, Tensor]:
    # CPU clone is needed too: a CPU tensor's .cpu() retains shared storage.
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


@contextmanager
def learner_witnesses(root: Path) -> Iterator[None]:
    """Record ordered forward/target/Adam witnesses during actual learner updates."""
    original_update = objectives.update_move_iteration
    original_targets = ataraxos._targets
    original_forward = Agent.forward_distribution
    original_step = OptimizerHealth.step
    iteration = 0
    optimizer_step = 0
    active = False
    phase = "minibatch"
    captured = False

    with (root / "learner-witnesses.jsonl").open("w") as stream:

        def emit(kind: str, **values: object) -> None:
            stream.write(
                json.dumps(
                    {
                        "kind": kind,
                        "iteration": iteration,
                        "step": optimizer_step,
                        **values,
                    }
                )
                + "\n"
            )
            stream.flush()

        @wraps(original_forward)
        def forward(agent: Agent, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
            if not active:
                return original_forward(agent, obs)
            emit("forward_before", phase=phase, inputs=state_digest(obs), rng=_rng())
            outputs = original_forward(agent, obs)
            emit(
                "forward_after",
                phase=phase,
                policy=state_digest(outputs[0]),
                value=state_digest(outputs[1]),
                rng=_rng(),
            )
            return outputs

        @wraps(original_targets)
        def targets(
            trainer: NetOpponentTrainer,
            batch: RolloutBatch,
            learning: AtaraxosMoveLearning,
            bootstrap_agent: Agent | None = None,
        ) -> ataraxos.MoveTargets:
            nonlocal phase
            phase = "targets"
            try:
                value = original_targets(trainer, batch, learning, bootstrap_agent)
                emit(
                    "targets",
                    advantages=state_digest(value.advantages),
                    values=state_digest(value.values),
                    selected=state_digest(value.selected),
                )
                return value
            finally:
                phase = "minibatch"

        @wraps(original_step)
        def step(observer: OptimizerHealth, loss: Tensor, max_norm: float) -> Tensor:
            nonlocal optimizer_step, captured
            optimizer_step += 1
            before = _parameters(observer.model)
            emit(
                "backward_before",
                loss=state_digest(loss),
                parameters=state_digest(before),
                buffers=state_digest(dict(observer.model.named_buffers())),
                rng=_rng(),
            )

            # PyTorch's hook signature includes an opaque closure argument list;
            # it is not inspected or changed by this observer.
            def before_adam(
                optimizer: torch.optim.Optimizer, args: object, kwargs: object
            ) -> None:
                emit(
                    "adam_before",
                    gradients=state_digest(
                        {name: p.grad for name, p in observer.model.named_parameters()}
                    ),
                    optimizer=state_digest(optimizer.state_dict()),
                    rng=_rng(),
                )

            handle = observer.optimizer.register_step_pre_hook(before_adam)
            try:
                result = original_step(observer, loss, max_norm)
            finally:
                handle.remove()
            after = _parameters(observer.model)
            movement = {
                name: float((after[name].double() - value.double()).abs().max())
                for name, value in before.items()
                if value.numel()
            }
            largest = max(movement, key=movement.__getitem__)
            emit(
                "adam_after",
                parameters=state_digest(after),
                optimizer=state_digest(observer.optimizer.state_dict()),
                rng=_rng(),
                max_movement=movement[largest],
                max_movement_parameter=largest,
                max_parameter_abs=max(
                    float(p.abs().max()) for p in after.values() if p.numel()
                ),
            )
            # A diagnostic trigger, not an acceptance tolerance or production guard.
            if movement[largest] > 0.01 and not captured:
                captured = True
                torch.save(
                    {
                        "before": before,
                        "after": after,
                        "optimizer": observer.optimizer.state_dict(),
                        "iteration": iteration,
                        "step": optimizer_step,
                    },
                    root / "large-movement.pt",
                )
                emit("large_movement_saved", parameter=largest)
            return result

        @wraps(original_update)
        def update(
            trainer: NetOpponentTrainer,
            batch: RolloutBatch,
            learning: AtaraxosMoveLearning,
            update_iteration: int,
            *,
            bootstrap_agent: Agent | None = None,
        ) -> UpdateDiagnostics:
            nonlocal iteration, active
            iteration, active = update_iteration, True
            emit(
                "update_before",
                parameters=state_digest(trainer.agent.state_dict()),
                optimizer=state_digest(trainer.optimizer.state_dict()),
                rng=_rng(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            )
            try:
                result = original_update(
                    trainer,
                    batch,
                    learning,
                    update_iteration,
                    bootstrap_agent=bootstrap_agent,
                )
                emit("update_after", diagnostics=state_digest(result), rng=_rng())
                return result
            except BaseException as error:
                emit("error", error_type=type(error).__name__, message=str(error))
                raise
            finally:
                active = False

        with (
            patch.object(objectives, "update_move_iteration", update),
            patch.object(ataraxos, "_targets", targets),
            patch.object(Agent, "forward_distribution", forward),
            patch.object(OptimizerHealth, "step", step),
        ):
            yield
