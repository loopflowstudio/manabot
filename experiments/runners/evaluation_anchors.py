"""MTG-153: admit a retained cohort, smoke each pairing, then run once.

Prepare freezes checkpoint bytes, original held-out families, historical greedy
rows, registrations and source identity. Run refuses changed inputs and retains
all attempts; a failed game stops the cohort. Report never treats a prefix as
strength evidence. Targeting is reconstructed from exact Commands with the Study
recorder, without sampling policies or starting additional games.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
from typing import Any, Literal

import numpy as np
from pydantic import Field, TypeAdapter, model_validator
import torch

from academy.server import ASSET_MANIFEST_HASH, CONTENT_HASH
from experiments.runners.greedy_targeting import audit
from experiments.runners.heuristic_check import _admit, _checkpoint
from manabot.arena.match import play_cell, selected_decks
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.arena.replay import read_trace, replay_environment
from manabot.sim.teacher1_evidence import build_viewer_frame, source_bundle_sha256
from manabot.study.record import GameRecord, _decision, write_games
from manabot.training.execution import atomic_json
from manabot.training.models import Strict, TrainingRun
from manabot.training.monitor_evaluation import (
    ArenaRow,
    FrozenCheckpointOpponent,
    MonitorProtocol,
    MonitorResult,
    RateInterval,
    _manifest,
)
from managym import _managym

CELLS = ("LvA", "AvL", "LvL", "AvA")
DECKS = {"ur_lessons": "L", "gw_allies": "A"}


class Member(Strict):
    name: str
    group: Literal["init", "10k", "25k"]
    producer: Path
    point: Literal["initial", "endpoint"]
    baseline: Path
    baseline_sha256: str
    baseline_rows: list[ArenaRow]
    manifest: MonitorResult


class Pairing(Strict):
    name: str
    member: str
    anchor: Literal[
        "heuristic",
        "checkpoint",
        "greedy-v-heuristic",
        "greedy-v-checkpoint",
        "heuristic-v-checkpoint",
    ]
    manifest: MonitorResult


class Cohort(Strict):
    schema_version: Literal[1] = 1
    source_sha256: str
    native_sha256: str
    analysis_sha256: str
    frozen_member: Literal["mini-12261"] = "mini-12261"
    members: list[Member]
    pairings: list[Pairing]
    wall_seconds: int = Field(default=8 * 3600, gt=0, le=8 * 3600)
    cost_dollars: Literal[0] = 0

    @model_validator(mode="after")
    def complete_roster(self) -> Cohort:
        expected = {
            "init",
            "mini-12261",
            "mini-12262",
            "mini-12263",
            "gpu-12261",
            "gpu-12262",
        }
        if len(self.members) != 6 or {m.name for m in self.members} != expected:
            raise ValueError("requires the complete six-checkpoint cohort")
        pairings = {
            f"{name}-vs-{anchor}"
            for name in expected
            for anchor in ("heuristic", "checkpoint")
        }
        pairings |= {
            "greedy-v-heuristic",
            "greedy-v-checkpoint",
            "heuristic-v-checkpoint",
        }
        if len(self.pairings) != 15 or {p.name for p in self.pairings} != pairings:
            raise ValueError("requires all fifteen pairings")
        members = {member.name: member for member in self.members}
        frozen = members[self.frozen_member]
        for pairing in self.pairings:
            member = members[pairing.member]
            if (
                pairing.manifest.protocol.deal_seeds
                != member.manifest.protocol.deal_seeds
            ):
                raise ValueError("pairing changed its historical deal family")
            opponent = pairing.manifest.protocol.checkpoint_opponent
            if opponent is not None and opponent.artifact != frozen.manifest.artifact:
                raise ValueError("pairing changed the frozen checkpoint")
        return self


def source_digest() -> str:
    paths = [
        Path(__file__),
        Path("experiments/runners/heuristic_check.py"),
        Path("experiments/runners/greedy_targeting.py"),
        *Path("manabot").rglob("*.py"),
        *Path("academy").rglob("*.py"),
    ]
    return source_bundle_sha256(sorted(path.resolve() for path in paths))


def prepare(root: Path, analysis_path: Path, destination: Path) -> Cohort:
    """Read-only admission of retained producers; output is never overwritten."""
    destination.mkdir(parents=True, exist_ok=False)
    analysis = json.loads(analysis_path.read_text())
    runs = [
        ("init", "init", root / "mini-final/seed-12261", "initial", "mini", 12261, 0),
        (
            "mini-12261",
            "10k",
            root / "mini-final/seed-12261",
            "endpoint",
            "mini",
            12261,
            10000,
        ),
        (
            "mini-12262",
            "10k",
            root / "mini-final/remaining/l3-12262",
            "endpoint",
            "mini",
            12262,
            10000,
        ),
        (
            "mini-12263",
            "10k",
            root / "mini-final/remaining/l3-12263",
            "endpoint",
            "mini",
            12263,
            10000,
        ),
        (
            "gpu-12261",
            "25k",
            root / "gpu/retained/14353d64c8d3/outputs",
            "endpoint",
            "gpu",
            12261,
            25000,
        ),
        (
            "gpu-12262",
            "25k",
            root / "gpu/retained/cb14c42a3338/outputs",
            "endpoint",
            "gpu",
            12262,
            25000,
        ),
    ]
    members: list[Member] = []
    producers: dict[str, TrainingRun] = {}
    for name, group, outputs, point, host, seed, updates in runs:
        original = TrainingRun.model_validate_json(
            (outputs / "training/run.json").read_text()
        )
        if original.seed != seed or original.status != "completed":
            raise ValueError(f"{name}: invalid producer")
        producer = _admit(original, destination / f"{name}-admission.json")
        checkpoint = _checkpoint(producer, root, point)
        if checkpoint.coordinates.updates != updates:
            raise ValueError(f"{name}: wrong checkpoint horizon")
        expected = analysis[host]["counts"][str(seed)][
            f"{updates}-scripted_greedy-candidate"
        ]
        if checkpoint.artifact["sha256"] != expected["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint differs from MTG-137")
        candidates = root.rglob(f"{updates}-scripted_greedy-candidate/monitor.json")
        baseline = next(
            (p for p in candidates if file_sha256(p) == expected["monitor_sha256"]),
            None,
        )
        if baseline is None:
            raise ValueError(f"{name}: missing original greedy rows")
        raw = json.loads(baseline.read_text())
        rows = [ArenaRow.model_validate(r) for r in raw["rows"]]
        seeds = tuple(raw["protocol"]["deal_seeds"])
        if len(seeds) != 25 or len(rows) != 200 or not all(r.valid for r in rows):
            raise ValueError(f"{name}: incomplete historical evaluation")
        protocol = MonitorProtocol(
            purpose="supplemental-matchup-diagnostic",
            include_mirrors=True,
            deal_seeds=seeds,
            opponent="heuristic",
            bootstrap_seed=12290,
            bootstrap_replicates=10000,
        )
        manifest = _manifest(
            producer, checkpoint.artifact, checkpoint.coordinates, protocol
        )
        members.append(
            Member(
                name=name,
                group=group,
                producer=outputs / "training/run.json",
                point=point,
                baseline=baseline,
                baseline_sha256=file_sha256(baseline),
                baseline_rows=rows,
                manifest=manifest,
            )
        )
        producers[name] = producer
    frozen = next(member for member in members if member.name == "mini-12261")
    pairings: list[Pairing] = []
    for member in members:
        for anchor in ("heuristic", "checkpoint"):
            manifest = member.manifest
            if anchor == "checkpoint":
                protocol = manifest.protocol.model_copy(
                    update={
                        "opponent": anchor,
                        "checkpoint_opponent": FrozenCheckpointOpponent(
                            artifact=frozen.manifest.artifact,
                            training_seed=frozen.manifest.training_seed,
                        ),
                    }
                )
                manifest = _manifest(
                    producers[member.name],
                    manifest.artifact,
                    manifest.coordinates,
                    protocol,
                )
            pairings.append(
                Pairing(
                    name=f"{member.name}-vs-{anchor}",
                    member=member.name,
                    anchor=anchor,
                    manifest=manifest,
                )
            )
    # Use the same admitted interfaces for code/code and code/checkpoint games.
    greedy = _manifest(
        producers["mini-12261"],
        frozen.manifest.artifact,
        frozen.manifest.coordinates,
        frozen.manifest.protocol.model_copy(update={"opponent": "scripted_greedy"}),
    ).opponent
    heuristic = frozen.manifest.opponent
    checkpoint_pairing = next(
        pairing.manifest
        for pairing in pairings
        if pairing.member == frozen.name and pairing.anchor == "checkpoint"
    )
    checkpoint = checkpoint_pairing.opponent
    for name, a, b, base in (
        ("greedy-v-heuristic", greedy, heuristic, frozen.manifest),
        ("greedy-v-checkpoint", greedy, checkpoint, checkpoint_pairing),
        ("heuristic-v-checkpoint", heuristic, checkpoint, checkpoint_pairing),
    ):
        manifest = base.model_copy(update={"candidate": a, "opponent": b})
        pairings.append(
            Pairing(name=name, member="mini-12261", anchor=name, manifest=manifest)
        )
    cohort = Cohort(
        source_sha256=source_digest(),
        native_sha256=file_sha256(_managym.__file__),
        analysis_sha256=file_sha256(analysis_path),
        members=members,
        pairings=pairings,
    )
    atomic_json(destination / "cohort.json", cohort.model_dump(mode="json"))
    return cohort


def _study_game(game: dict[str, Any]) -> GameRecord:
    """Project an already replay-validated trace through the MTG-152 recorder."""
    record = GameRecord(
        game["match_id"],
        tuple(game["seat_players"]),
        tuple(game["seat_decks"]),
        game["deal_seed"],
    )
    env, _ = replay_environment(game)
    try:
        for expected in game["decisions"]:
            frame = build_viewer_frame(
                env.last_raw_obs,
                match_id=game["match_id"],
                revision=expected["revision"],
                content_hash=CONTENT_HASH,
                asset_manifest_hash=ASSET_MANIFEST_HASH,
            )
            if canonical_sha256(frame) != expected["frame_sha256"]:
                raise ValueError("targeting reconstruction frame mismatch")
            action = int(expected["command"]["offer_id"])
            record.decisions.append(
                _decision(env.last_raw_obs, frame, action, expected["latency_seconds"])
            )
            env.step(action)
        record.winner = game["winner"]
        record.end = "terminal" if game["terminated"] else game["termination_reason"]
        record.error = game["failure"]
        return record
    finally:
        env.close()


def execute(cohort_path: Path, destination: Path, *, smoke: bool) -> None:
    """One fresh attempt only; all limits include admission and targeting replay."""
    cohort = Cohort.model_validate_json(cohort_path.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    if not smoke:
        # Claim before any work; a different output directory cannot relaunch it.
        with (cohort_path.parent / "scientific-attempt.json").open("x") as handle:
            json.dump(
                {"out": str(destination.resolve()), "started_unix": time.time()}, handle
            )
    started = time.monotonic()
    state: dict[str, Any] = {
        "status": "running",
        "smoke": smoke,
        "cohort_sha256": file_sha256(cohort_path),
        "completed_games": 0,
    }
    atomic_json(destination / "status.json", state)
    try:
        if (
            cohort.source_sha256 != source_digest()
            or cohort.native_sha256 != file_sha256(_managym.__file__)
        ):
            raise ValueError("instrument source changed after prepare")
        # Admit every participant before any game, including all frozen weights.
        for member in cohort.members:
            artifact = member.manifest.artifact
            if (
                file_sha256(artifact["path"]) != artifact["sha256"]
                or Path(artifact["path"]).stat().st_size != artifact["bytes"]
            ):
                raise ValueError(f"{member.name}: changed checkpoint")
            if file_sha256(member.baseline) != member.baseline_sha256:
                raise ValueError(f"{member.name}: changed baseline")
        for pairing in cohort.pairings:
            result = pairing.manifest
            paths = {result.candidate.player_id: result.artifact["path"]}
            if result.protocol.checkpoint_opponent is not None:
                paths[result.opponent.player_id] = (
                    result.protocol.checkpoint_opponent.artifact["path"]
                )
            for index, seed in enumerate(
                result.protocol.deal_seeds[:1] if smoke else result.protocol.deal_seeds
            ):
                for leg in (0,) if smoke else range(8):
                    remaining = cohort.wall_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("8-hour cohort cap exhausted")
                    directory = destination / pairing.name / f"{index:02d}-{leg}"
                    directory.mkdir(parents=True)
                    rows, trace, replay = play_cell(
                        key=result.key,
                        player_a=result.candidate,
                        player_b=result.opponent,
                        deal_seeds=(seed,),
                        out_dir=directory,
                        checkpoint_paths=paths,
                        matchup_mode="all",
                        selected_legs=(leg,),
                        game_seconds=min(remaining, result.protocol.game_seconds),
                        max_commands=result.protocol.max_commands,
                    )
                    atomic_json(directory / "rows.json", rows)
                    atomic_json(directory / "replay.json", replay)
                    if not all(ArenaRow.model_validate(r).valid for r in rows):
                        raise RuntimeError(f"{pairing.name}: failed game; no relaunch")
                    games = [
                        _study_game(g)
                        for g in read_trace(directory / trace["artifact_path"])
                    ]
                    record_path = directory / "study.jsonl.gz"
                    write_games(record_path, games)
                    atomic_json(
                        directory / "targeting.json",
                        {
                            label: asdict(audit(record_path, label))
                            for label in games[0].labels
                        },
                    )
                    state["completed_games"] += 1
                    atomic_json(destination / "status.json", state)
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["seconds"] = time.monotonic() - started
        atomic_json(destination / "status.json", state)


def _execute_process(cohort: Path, destination: Path, smoke: bool) -> None:
    # Isolate the worker and its native game children for a hard wall-clock cap.
    os.setsid()
    torch.set_num_threads(1)
    execute(cohort, destination, smoke=smoke)


def supervise(cohort_path: Path, destination: Path, *, smoke: bool) -> None:
    cohort = Cohort.model_validate_json(cohort_path.read_text())
    if destination.exists():
        raise FileExistsError(destination)
    process = mp.get_context("spawn").Process(
        target=_execute_process, args=(cohort_path, destination, smoke)
    )
    deadline = time.monotonic() + cohort.wall_seconds
    process.start()
    try:
        while process.is_alive() and time.monotonic() < deadline:
            process.join(timeout=min(1, max(0, deadline - time.monotonic())))
        if process.is_alive():
            os.killpg(process.pid, signal.SIGKILL)
            process.join()
            status_path = destination / "status.json"
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(
                status_path,
                {
                    **state,
                    "status": "failed",
                    "error": "8-hour cohort wall-clock cap exhausted; no relaunch",
                },
            )
            raise TimeoutError("cohort wall-clock cap exhausted")
        if process.exitcode != 0:
            raise RuntimeError(
                "cohort failed; retained status.json and traces; no relaunch"
            )
    finally:
        if process.is_alive():
            os.killpg(process.pid, signal.SIGKILL)
            process.join()


def cell_scores(rows: list[ArenaRow], seeds: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Validate the deal/seat layout before computing per-deal cell means."""
    found: dict[tuple[int, int], float] = {}
    for row in rows:
        extra = row.model_extra or {}
        coordinate = (row.deal_seed, row.leg)
        if not row.valid or coordinate in found or row.deal_seed not in seeds:
            raise ValueError("invalid, duplicate or foreign row")
        if (
            extra.get("seat_decks") != list(selected_decks(row.leg))
            or extra.get("player_a_seat") != row.leg % 2
        ):
            raise ValueError("row differs from authored seat/deck layout")
        assert row.score_a is not None
        found[coordinate] = row.score_a
    if set(found) != {(s, leg) for s in seeds for leg in range(8)}:
        raise ValueError("incomplete pairing; no strength over survivors")
    cell_legs: dict[str, list[int]] = {cell: [] for cell in CELLS}
    for leg in range(8):
        decks = selected_decks(leg)
        cell = f"{DECKS[decks[leg % 2]]}v{DECKS[decks[1 - leg % 2]]}"
        cell_legs[cell].append(leg)
    return {
        cell: np.array([[found[seed, leg] for leg in legs] for seed in seeds]).mean(
            axis=1
        )
        for cell, legs in cell_legs.items()
    }


def interval(values: np.ndarray) -> RateInterval:
    rng = np.random.default_rng(12290)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return RateInterval(mean=float(values.mean()), lower=float(low), upper=float(high))


def report(cohort_path: Path, destination: Path) -> dict[str, Any]:
    cohort = Cohort.model_validate_json(cohort_path.read_text())
    state = json.loads((destination / "status.json").read_text())
    if state["cohort_sha256"] != file_sha256(cohort_path):
        raise ValueError("attempt belongs to a different cohort")
    if state["status"] != "completed" or state["smoke"]:
        return {
            "status": state,
            "strength": None,
            "reason": "Only a complete scientific cohort supports strength analysis.",
        }
    panels: dict[str, dict[str, RateInterval]] = {}
    deltas: dict[str, dict[str, RateInterval]] = {}
    targeting: dict[str, Any] = {}
    members = {member.name: member for member in cohort.members}
    for pairing in cohort.pairings:
        row_paths = sorted((destination / pairing.name).glob("*/rows.json"))
        if any(not (path.parent / "targeting.json").is_file() for path in row_paths):
            raise ValueError("missing targeting evidence for a retained game")
        rows = [
            ArenaRow.model_validate(row)
            for path in row_paths
            for row in json.loads(path.read_text())
        ]
        for row in rows:
            if (
                row.player_a_registration_sha256
                != pairing.manifest.candidate.identity_sha256
                or row.player_b_registration_sha256
                != pairing.manifest.opponent.identity_sha256
                or row.arena_key != pairing.manifest.key
            ):
                raise ValueError("row registration differs from frozen pairing")
        scores = cell_scores(rows, pairing.manifest.protocol.deal_seeds)
        panels[pairing.name] = {cell: interval(scores[cell]) for cell in CELLS}
        if pairing.anchor in ("heuristic", "checkpoint"):
            baseline = cell_scores(
                members[pairing.member].baseline_rows,
                pairing.manifest.protocol.deal_seeds,
            )
            deltas[pairing.name] = {
                cell: interval(scores[cell] - baseline[cell]) for cell in CELLS
            }
        # Pool the recorder's descriptive counts, retaining cell, seat and source.
        counts: dict[str, int] = {}
        for path in sorted((destination / pairing.name).glob("*/targeting.json")):
            for label, audit_data in json.loads(path.read_text()).items():
                for cell in audit_data["cells"]:
                    for count in cell["counts"]:
                        key = json.dumps(
                            {
                                "label": label,
                                "cell": cell["cell"],
                                **{
                                    k: v
                                    for k, v in count.items()
                                    if k not in ("count", "deck", "opponent_deck")
                                },
                            },
                            sort_keys=True,
                        )
                        counts[key] = counts.get(key, 0) + count["count"]
        targeting[pairing.name] = [
            {"count": count, **json.loads(key)} for key, count in sorted(counts.items())
        ]
        # Every prompt contributes once to its label/seat/cell denominator.
        totals: Counter[tuple[str, str, int]] = Counter()
        for entry in targeting[pairing.name]:
            totals[entry["label"], entry["cell"], entry["seat"]] += entry["count"]
        for entry in targeting[pairing.name]:
            entry["fraction_of_target_prompts_in_seat_cell"] = (
                entry["count"] / totals[entry["label"], entry["cell"], entry["seat"]]
            )
    summaries: dict[str, Any] = {}
    for anchor in ("heuristic", "checkpoint"):
        summaries[anchor] = {}
        for cell in CELLS:
            groups = {
                group: [
                    panels[f"{m.name}-vs-{anchor}"][cell].mean
                    for m in cohort.members
                    if m.group == group
                ]
                for group in ("init", "10k", "25k")
            }
            means = {group: float(np.mean(values)) for group, values in groups.items()}
            summaries[anchor][cell] = {
                "group_means": means,
                "orders_group_means": means["init"] < means["10k"] < means["25k"],
                "seed_ranges": {
                    group: [min(values), max(values)]
                    for group, values in groups.items()
                },
                "seed_spread": {
                    group: max(values) - min(values) for group, values in groups.items()
                },
                "headroom_per_checkpoint": {
                    m.name: 1 - panels[f"{m.name}-vs-{anchor}"][cell].mean
                    for m in cohort.members
                },
            }
    ordering_counts = {
        anchor: sum(summaries[anchor][cell]["orders_group_means"] for cell in CELLS)
        for anchor in summaries
    }
    preferred = max(
        ordering_counts,
        key=lambda anchor: (ordering_counts[anchor], anchor == "heuristic"),
    )
    return {
        "status": state,
        "recommendation": {
            "future_studies": "Retain both measured anchors and historical greedy continuity; two opponents help expose policy-specific strengths.",
            "daily_regression": preferred,
            "basis": "Most cells with init < 10k < 25k group means; ties choose the cheaper heuristic. Provisional, not independent validation.",
            "ordering_cells": ordering_counts,
        },
        "scores": panels,
        "paired_delta_from_greedy": deltas,
        "anchor_summary": summaries,
        "targeting": targeting,
        "limitations": "Whole-deal intervals describe checkpoint evaluation noise. Three Mini and two GPU seeds do not establish a method-level ordering. Own-side targeting is descriptive, not an error rate. Group mean ordering is not statistical significance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--retained", type=Path, required=True)
    prep.add_argument("--analysis", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    for name in ("smoke", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--cohort", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        try:
            prepare(args.retained, args.analysis, args.out)
        except Exception as error:
            if args.out.is_dir() and not (args.out / "cohort.json").exists():
                atomic_json(
                    args.out / "admission-failure.json",
                    {"error": f"{type(error).__name__}: {error}"},
                )
            raise
    elif args.command == "report":
        found = report(args.cohort, args.out)
        # Serialize interval models alongside the report dictionaries.
        print(TypeAdapter(dict[str, Any]).dump_json(found, indent=2).decode())
    else:
        supervise(args.cohort, args.out, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
