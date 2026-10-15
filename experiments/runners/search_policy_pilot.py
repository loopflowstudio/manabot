"""Compare two completed calibration policies using the existing paired arena.

VerifyStore owns training provenance. This driver admits final raw bytes and
freezes player registrations before calling play_cell, which owns bounded game
execution, Command traces, and replay. These eight development games are not a
strength estimate or a checkpoint-selection procedure.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import signal
import time
from types import FrameType

from manabot.arena.match import SELECTED_SUITE, play_cell
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.teacher1_evidence import runtime_fingerprints
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore


@dataclass(frozen=True)
class AdmittedPolicy:
    registration: PlayerRegistration
    checkpoint: Path


def _admit(run: TrainingRun, family: str) -> AdmittedPolicy:
    artifact = run.selected_artifact
    if run.status != "completed" or artifact is None:
        raise ValueError(f"{family} has no completed selected artifact")
    path = Path(artifact["path"])
    if (
        path.stat().st_size != artifact["bytes"]
        or file_sha256(path) != artifact["sha256"]
    ):
        raise ValueError(f"{family} checkpoint bytes changed")
    agent, space = load_checkpoint_agent(str(path))
    if agent.hypers.compound_decisions != (family == "compound"):
        raise ValueError(f"{family} checkpoint has the wrong policy family")
    runtime = runtime_fingerprints(
        run.seed, match_hypers=run.regime.match, observation_space=space
    )
    for name in (
        "world",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
        "engine_source_sha256",
        "content_manifest_sha256",
    ):
        if runtime[name] != run.identities.get(name):
            raise ValueError(f"{family} runtime drift: {name}")
    return AdmittedPolicy(
        PlayerRegistration(
            player_id=f"pilot-{family}",
            display_name=f"Calibration {family}",
            role="challenger",
            runner_kind="checkpoint",
            player_spec={
                "kind": "checkpoint",
                "deterministic": False,
                "device": "cpu",
                "batch_size": 1,
            },
            compute_class_id="policy-cpu-one-thread-one-pass",
            information_boundary="acting-viewer",
            world=run.regime.world,
            content_suite=SELECTED_SUITE,
            observation_abi_sha256=str(runtime["observation_abi_sha256"]),
            action_abi_sha256=str(runtime["action_abi_sha256"]),
            matchup_sha256=str(runtime["matchup_sha256"]),
            checkpoint_sha256=artifact["sha256"],
            checkpoint_bytes=artifact["bytes"],
            parameter_count=sum(p.numel() for p in agent.parameters()),
            training_seed=run.seed,
            artifact_id=f"search-policy-pilot/{run.id}/{artifact['sha256']}",
            player_seed_derivation_id="arena-pair-deal-player-v1",
        ),
        path,
    )


def _deadline(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("pilot evaluation exceeded 600 seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flat-run", required=True)
    parser.add_argument("--compound-run", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(600)
    with VerifyStore(args.root / "training.sqlite") as store:
        flat = _admit(store.training_run(args.flat_run), "flat")
        compound = _admit(store.training_run(args.compound_run), "compound")
    key = ArenaKey(
        world=compound.registration.world,
        content_suite=SELECTED_SUITE,
        viewer_boundary="acting-viewer",
        arena_version="search-policy-calibration-20261015",
        rating_model_version="unrated-development",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(flat.registration.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    out = args.root / "arena"
    out.mkdir(exist_ok=False)
    atomic_json(
        out / "protocol.json",
        {
            "key": key.model_dump(),
            "players": [p.registration.model_dump() for p in (compound, flat)],
            "deal_seeds": [1910141001, 1910141002],
            "game_seconds": 60,
            "max_commands": 4000,
            "driver_sha256": file_sha256(Path(__file__)),
        },
    )
    rows, trace, replay = play_cell(
        key=key,
        player_a=compound.registration,
        player_b=flat.registration,
        deal_seeds=(1910141001, 1910141002),
        out_dir=out,
        checkpoint_paths={
            p.registration.player_id: str(p.checkpoint) for p in (compound, flat)
        },
        game_seconds=60,
        max_commands=4000,
    )
    atomic_json(out / "rows.json", rows)
    atomic_json(out / "replay.json", replay)
    atomic_json(out / "trace-receipt.json", trace)
    passed = (
        len(rows) == 8
        and bool(replay["passed"])
        and all(
            row["terminated"]
            and row["failure"] is None
            and not row["truncated"]
            and not any(row["integrity"].values())
            for row in rows
        )
    )
    atomic_json(
        out / "result.json",
        {
            "passed": passed,
            "seconds": time.perf_counter() - started,
            "games": len(rows),
            "compound_scores": [r["score_a"] for r in rows],
            "purpose": "development-integration-not-strength",
        },
    )
    signal.alarm(0)
    if not passed:
        raise RuntimeError("pilot arena failed its completion/replay gate")


if __name__ == "__main__":
    main()
