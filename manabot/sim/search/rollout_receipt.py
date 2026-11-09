"""Durable evidence from one policy rollout, independent of its execution state.

The frozen fields form the serialized local-update receipt contract, including
legacy compound-policy fields retained for historical readers.
"""

from dataclasses import dataclass

from manabot.sim.search.sampling import HandSample


@dataclass(frozen=True)
class RolloutReceipt:
    world_index: int | None
    world_seed: int
    action_index: int
    rollout_seed: int
    signed_value: float
    terminal: bool
    actor_observation_hashes: tuple[str, ...]
    branch_audit_json: str
    sampled_hand: HandSample | None = None
    # Always zero for current searches; nonzero only in historical compound
    # receipts.
    decoder_factors: int = 0
    canonical_commands: int = 0
