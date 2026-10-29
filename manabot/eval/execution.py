"""Bind comparison identities to an explicit simulation compute envelope."""

from __future__ import annotations

from typing import TYPE_CHECKING

from manabot.eval.models import EvaluationKey
from manabot.sim.execution import EvalExecution
from manabot.sim.registration import PlayerRegistration
from manabot.sim.serving_protocol import InferenceRuntime

if TYPE_CHECKING:
    from manabot.eval.endpoint import InferenceEndpoint


def is_served(player: PlayerRegistration) -> bool:
    """Flat checkpoint players play through shared inference when one is declared."""
    return (
        player.runner_kind == "checkpoint"
        and player.player_spec.get("kind") == "checkpoint"
    )


def resolve_execution(
    execution: EvalExecution | None,
    inference: InferenceRuntime | InferenceEndpoint | None,
) -> EvalExecution | None:
    """The declared allocation; a running service fixes it.

    Without a declaration, checkpoint players play scalar in their game child.
    """
    if inference is not None:
        if execution is not None and execution != inference.execution:
            raise ValueError("conflicting execution declaration")
        return inference.execution
    return execution


def bind_execution(
    key: EvaluationKey,
    players: tuple[PlayerRegistration, ...],
    execution: EvalExecution,
) -> tuple[EvaluationKey, tuple[PlayerRegistration, ...]]:
    """Derive compute-bound copies without modifying frozen inputs.

    Flat checkpoint players are bound to the shared service. Code and search
    players keep their registrations and play in their game child.
    """
    compute = execution.compute.model_dump()
    bound = []
    for player in players:
        spec = dict(player.player_spec)
        if is_served(player):
            existing = spec.get("execution")
            if existing is not None and existing != compute:
                raise ValueError("conflicting execution declaration")
            # Batch size is an allocation setting, not part of the player.
            spec.pop("batch_size", None)
            spec.update(device=execution.device, execution=compute)
            player = PlayerRegistration.model_validate(
                {
                    **player.model_dump(),
                    "player_spec": spec,
                    "compute_class_id": execution.identity,
                }
            )
        bound.append(player)
    if key.evaluation_compute_envelope_id.startswith("shared-eval-") and (
        key.evaluation_compute_envelope_id != execution.identity
    ):
        raise ValueError("conflicting arena execution envelope")
    return key.model_copy(
        update={
            "evaluation_compute_envelope_id": execution.identity,
            "seed_compute_envelope_id": key.seed_compute_envelope_id
            or key.evaluation_compute_envelope_id,
        }
    ), tuple(bound)
