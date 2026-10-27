"""Exact branch backends for visit-based search.

The selected backend is the RUL-1 full-clone driver with one structured
Command mutation seam. The legacy clone backend remains an explicit
differential oracle and is never a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, Protocol

import managym

SELECTED_BRANCH_DRIVER_ID = "full_clone/current_game_v1"
REFERENCE_BRANCH_DRIVER_ID = "legacy_env_clone/reference_v1"
BranchSite = Literal["world", "child", "leaf"]


class BranchSession(Protocol):
    driver_id: str

    def fork_exact(self, source: Any, site: BranchSite) -> Any: ...

    def determinize(self, branch: Any, *, seed: int, perspective: int) -> None: ...

    def reseed_rollout(self, branch: Any, *, seed: int) -> None: ...

    def sample_policy_index(self, branch: Any) -> int: ...

    def apply_policy_choice(
        self, branch: Any, *, site: BranchSite, policy_index: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...

    def record_leaf_playout(self, *, hit_cap: bool) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


class SearchBranchBackend(Protocol):
    driver_id: str

    def open_session(self, *, match_id: str, audit: bool) -> BranchSession: ...


@dataclass(frozen=True)
class SelectedFullCloneBackend:
    """Production backend statically bound to the retained Rust driver."""

    driver_id: str = SELECTED_BRANCH_DRIVER_ID

    def open_session(self, *, match_id: str, audit: bool) -> BranchSession:
        return _SelectedSession(match_id=match_id, audit=audit)


@dataclass(frozen=True)
class LegacyCloneReferenceBackend:
    """The pre-RUL-2 clone/index path, available only as a named oracle."""

    driver_id: str = REFERENCE_BRANCH_DRIVER_ID

    def open_session(self, *, match_id: str, audit: bool) -> BranchSession:
        return _ReferenceSession(match_id=match_id, audit=audit)


def branch_backend(driver_id: str) -> SearchBranchBackend:
    if driver_id == SELECTED_BRANCH_DRIVER_ID:
        return SelectedFullCloneBackend()
    if driver_id == REFERENCE_BRANCH_DRIVER_ID:
        return LegacyCloneReferenceBackend()
    raise ValueError(f"unknown search branch driver {driver_id!r}")


def structured_random_playout(
    session: BranchSession,
    source: Any,
    *,
    seed: int,
    max_steps: int,
) -> tuple[int | None, bool]:
    """Run the seeded leaf policy through the session's one mutation seam."""

    rollout = session.fork_exact(source, "leaf")
    session.reseed_rollout(rollout, seed=seed)
    steps = 0
    while not rollout.is_game_over() and steps < max_steps:
        policy_index = session.sample_policy_index(rollout)
        session.apply_policy_choice(rollout, site="leaf", policy_index=policy_index)
        steps += 1
    hit_cap = steps >= max_steps and not rollout.is_game_over()
    session.record_leaf_playout(hit_cap=hit_cap)
    return rollout.winner_index(), hit_cap


@dataclass
class _SelectedSession:
    match_id: str
    audit: bool
    runtime: Any = field(init=False)
    driver_id: str = SELECTED_BRANCH_DRIVER_ID

    def __post_init__(self) -> None:
        self.runtime = managym.SelectedBranchRuntime(self.match_id, self.audit)
        if self.runtime.driver_id != self.driver_id:
            raise RuntimeError("native selected driver identity mismatch")

    def fork_exact(self, source: Any, site: BranchSite) -> Any:
        return self.runtime.fork_exact(source, site)

    def determinize(self, branch: Any, *, seed: int, perspective: int) -> None:
        self.runtime.determinize(branch, seed, perspective)

    def reseed_rollout(self, branch: Any, *, seed: int) -> None:
        self.runtime.reseed_rollout(branch, seed)

    def sample_policy_index(self, branch: Any) -> int:
        return int(self.runtime.sample_policy_index(branch))

    def apply_policy_choice(
        self, branch: Any, *, site: BranchSite, policy_index: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info, _receipt = (
            self.runtime.apply_policy_choice(branch, site, policy_index)
        )
        return observation, reward, terminated, truncated, info

    def record_leaf_playout(self, *, hit_cap: bool) -> None:
        self.runtime.record_leaf_playout(hit_cap)

    def snapshot(self) -> dict[str, Any]:
        return json.loads(self.runtime.snapshot_json())


@dataclass
class _ReferenceSession:
    match_id: str
    audit: bool
    driver_id: str = REFERENCE_BRANCH_DRIVER_ID
    command_sequence: int = 0
    apply_sequence: int = 0
    forks: dict[BranchSite, int] = field(
        default_factory=lambda: {"world": 0, "child": 0, "leaf": 0}
    )
    applies: dict[BranchSite, int] = field(
        default_factory=lambda: {"world": 0, "child": 0, "leaf": 0}
    )
    tapes: dict[BranchSite, list[dict[str, Any]]] = field(
        default_factory=lambda: {"world": [], "child": [], "leaf": []}
    )
    leaf_playouts: int = 0
    leaf_cap_hits: int = 0

    def fork_exact(self, source: Any, site: BranchSite) -> Any:
        self.forks[site] += 1
        return source.clone_env()

    def determinize(self, branch: Any, *, seed: int, perspective: int) -> None:
        branch.determinize(seed=seed, perspective=perspective)

    def reseed_rollout(self, branch: Any, *, seed: int) -> None:
        branch.reseed_rollout(seed)

    def sample_policy_index(self, branch: Any) -> int:
        return int(branch.random_action_index())

    def apply_policy_choice(
        self, branch: Any, *, site: BranchSite, policy_index: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        context = json.loads(branch.search_context_json(self.audit))
        offers = context["offers"]["offers"]
        if policy_index < 0 or policy_index >= len(offers):
            raise RuntimeError(
                f"policy lookup key {policy_index} is out of range for {len(offers)} offers"
            )
        offer = offers[policy_index]
        self.command_sequence += 1
        command = {
            "command_id": f"search.{self.match_id}.{self.command_sequence}",
            "match_id": self.match_id,
            "expected_revision": context["revision"],
            "prompt_id": context["prompt_id"],
            "offer_id": offer["id"],
            "answers": [],
        }
        observation, reward, terminated, truncated, info = branch.step(policy_index)
        self.apply_sequence += 1
        self.applies[site] += 1
        record = {
            "site": site,
            "policy_index": policy_index,
            "offer_id": offer["id"],
            "command": command,
            "source": {
                "prompt_id": context["prompt_id"],
                "expected_revision": context["revision"],
                "authority_hash": (
                    context["witness"]["authority_hash"] if self.audit else None
                ),
                "legal_surface_hash": (
                    context["witness"]["legal_surface_hash"] if self.audit else None
                ),
            },
            "native_receipt": {
                "driver_id": self.driver_id,
                "apply_sequence": self.apply_sequence,
                "site_sequence": self.applies[site],
                "accepted_apply_counter": sum(self.applies.values()),
                "native_apply_count": 1,
                "terminal": bool(terminated or truncated),
            },
            "post_apply_witness": (
                json.loads(branch.search_witness_json()) if self.audit else None
            ),
        }
        if self.audit:
            self.tapes[site].append(record)
        return observation, reward, terminated, truncated, info

    def record_leaf_playout(self, *, hit_cap: bool) -> None:
        self.leaf_playouts += 1
        self.leaf_cap_hits += int(hit_cap)

    def snapshot(self) -> dict[str, Any]:
        tape_lengths = {site: len(tape) for site, tape in self.tapes.items()}
        reconciled = not self.audit or all(
            self.applies[site] == tape_lengths[site]
            for site in ("world", "child", "leaf")
        )
        return {
            "driver_id": self.driver_id,
            "match_id": self.match_id,
            "audit": self.audit,
            "counters": {
                "forks": self.forks,
                "applies": self.applies,
                "marks": 0,
                "rollbacks": 0,
                "random_playouts": self.leaf_playouts,
                "random_playout_cap_hits": self.leaf_cap_hits,
                "indexed_fallbacks": 0,
            },
            "tape_lengths": tape_lengths,
            "tapes": self.tapes,
            "reconciliation": {
                "per_site_and_total": reconciled,
                "zero_unmeasured_fallback": True,
            },
        }
