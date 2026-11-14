"""Execution state for one independently seeded policy rollout.

`Rollout` owns a forked branch, its Commands, policy RNG and final receipt.
Full evidence records the actor's viewer hash before each Command and an
audited native session's witnesses; compact evidence records neither and
keeps only the Command count, which the receipt reconciles with the session's
native counters.
The caller schedules plies and supplies actor-view predictions between advances;
this module does not own inference, world sampling or the local update rule.
"""

from dataclasses import dataclass, field
import json

import numpy as np
from numpy.typing import NDArray

from manabot.sim.search.branch import BranchSession
from manabot.sim.search.rollout_receipt import RolloutReceipt
from manabot.sim.search.sampling_receipt import HandSample
import managym


@dataclass
class Rollout:
    """One independently seeded branch, retained in world/action receipt order."""

    branch: managym.Env
    session: BranchSession
    world_index: int | None
    world_seed: int
    action_index: int
    rollout_seed: int
    sampled_hand: HandSample | None
    probabilities: NDArray[np.float64]
    record_hashes: bool = True
    policy_rng: np.random.Generator = field(init=False)
    hashes: list[str] = field(default_factory=list)
    canonical_commands: int = 0
    value: float = 0.0

    def __post_init__(self) -> None:
        self.policy_rng = np.random.default_rng(self.rollout_seed)

    def advance(self, ply: int) -> None:
        """Apply one Command; ply zero uses the assigned root action.

        Later plies sample the latest actor-view probabilities with this branch's
        independent RNG. Hashes, when recorded, describe the view before each
        Command. Truncation fails the search rather than manufacturing a
        terminal result.
        """
        if self.record_hashes:
            actor = int(self.branch.current_agent_index())
            self.hashes.append(self.branch.semantic_viewer_state_hash(actor))
        choice = (
            self.action_index
            if ply == 0
            else int(
                self.policy_rng.choice(len(self.probabilities), p=self.probabilities)
            )
        )
        _, _, _, truncated, _ = self.session.apply_policy_choice(
            self.branch, site="child", policy_index=choice
        )
        self.canonical_commands += 1
        if truncated:
            raise RuntimeError(
                "local rollout truncated before an authoritative outcome"
            )

    def receipt(self, viewer: int) -> RolloutReceipt:
        terminal = bool(self.branch.is_game_over())
        value = self.value
        if terminal:
            winner = self.branch.winner_index()
            value = 0.0 if winner is None else (1.0 if winner == viewer else -1.0)
        elif int(self.branch.current_agent_index()) != viewer:
            value = -value
        return RolloutReceipt(
            canonical_commands=self.canonical_commands,
            world_index=self.world_index,
            sampled_hand=self.sampled_hand,
            world_seed=self.world_seed,
            action_index=self.action_index,
            rollout_seed=self.rollout_seed,
            signed_value=value,
            terminal=terminal,
            actor_observation_hashes=tuple(self.hashes),
            branch_audit_json=json.dumps(self.session.snapshot(), sort_keys=True),
        )
