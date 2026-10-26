"""Bind comparison identities to an explicit simulation compute envelope."""

from manabot.eval.models import EvaluationKey
from manabot.sim.execution import EvalExecution
from manabot.sim.registration import PlayerRegistration


def bind_execution(
    key: EvaluationKey,
    players: tuple[PlayerRegistration, ...],
    execution: EvalExecution,
) -> tuple[EvaluationKey, tuple[PlayerRegistration, ...]]:
    """Derive compute-bound copies without modifying frozen inputs."""
    compute = execution.compute.model_dump()
    bound = []
    for player in players:
        spec = dict(player.player_spec)
        if player.runner_kind == "checkpoint":
            if spec.get("kind") != "checkpoint":
                raise ValueError("shared inference requires ordinary flat checkpoints")
            existing = spec.get("execution")
            if existing is not None and existing != compute:
                raise ValueError("conflicting execution declaration")
            spec.update(
                device=execution.device,
                batch_size=execution.max_batch_rows,
                execution=compute,
            )
            player = PlayerRegistration.model_validate(
                {
                    **player.model_dump(),
                    "player_spec": spec,
                    "compute_class_id": execution.identity,
                }
            )
        elif spec.get("kind") not in {"random", "scripted_greedy", "heuristic"}:
            raise ValueError("shared evaluation supports stateless anchors only")
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
